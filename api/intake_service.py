#!/usr/bin/env python3
"""Seeking Alpha Legal — Intake backend.

A small FastAPI service that turns the Intake Console's "live mode" on end to end.

POST /extract
    Accepts EITHER application/json {"transcript": "..."} OR multipart/form-data
    with a file (audio, pdf, txt, chat). Pipeline:
      1. Resolve text: audio -> ASR transcription; pdf/chat/txt -> parse.
      2. Redact PII deterministically BEFORE the text leaves the process.
      3. Run TRANSCRIPT_PROMPT through Claude to get the structured record.
      4. Return the firm schema the console renders (case_metadata, legal_facts,
         form_fields, action_items, privacy_notice, review_needed).

GET /health   liveness + which providers are configured.

Run:
    pip install -r requirements-backend.txt
    export ANTHROPIC_API_KEY=...        # extraction (else heuristic fallback)
    export OPENAI_API_KEY=...           # ASR via Whisper (optional)
    uvicorn intake_service:app --reload --port 8000

Then in the Intake Console set the backend endpoint to:
    http://localhost:8000/extract
"""
import json
import os
import re
import tempfile

from fastapi import FastAPI, Request, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Reuse the engine you already have, so the contract is identical everywhere.
from intake_agent import TRANSCRIPT_PROMPT, redact_pii, extract_text

MODEL = os.environ.get("INTAKE_MODEL", "claude-opus-4-8")
AUDIO_EXT = (".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac")

app = FastAPI(title="Seeking Alpha Legal — Intake", version="1.0")

# The console is served from another origin (the static site), so allow CORS.
# Lock ALLOW_ORIGINS down to your site domain in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("ALLOW_ORIGINS", "*").split(","),
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)


# ----------------------------------------------------------------------------
# ASR: transcribe audio to text (pluggable provider)
# ----------------------------------------------------------------------------
def asr_transcribe(data: bytes, filename: str) -> str:
    provider = os.environ.get("ASR_PROVIDER", "openai").lower()
    suffix = os.path.splitext(filename)[1] or ".mp3"
    if provider == "openai" and os.environ.get("OPENAI_API_KEY"):
        from openai import OpenAI
        client = OpenAI()
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
            tmp.write(data); tmp.flush(); tmp.seek(0)
            with open(tmp.name, "rb") as fh:
                tr = client.audio.transcriptions.create(model="whisper-1", file=fh)
        return tr.text
    if provider == "deepgram" and os.environ.get("DEEPGRAM_API_KEY"):
        import requests
        r = requests.post(
            "https://api.deepgram.com/v1/listen?smart_format=true&punctuate=true",
            headers={"Authorization": f"Token {os.environ['DEEPGRAM_API_KEY']}",
                     "Content-Type": "application/octet-stream"},
            data=data, timeout=300)
        r.raise_for_status()
        return r.json()["results"]["channels"][0]["alternatives"][0]["transcript"]
    raise HTTPException(status_code=501,
                        detail="No ASR provider configured. Set ASR_PROVIDER and the matching API key.")


# ----------------------------------------------------------------------------
# Extraction: Claude (production) or a transcript-shaped heuristic (fallback)
# ----------------------------------------------------------------------------
def heuristic_transcript(text: str) -> dict:
    """Returns the firm schema without an API key, so the service is testable
    and degrades gracefully. Mirrors the console's in-browser extractor."""
    facts = []
    for m in re.finditer(r"\[(\d{1,2}:\d{2})\]\s*([^\n]*)", text):
        seg = re.sub(r"^(Attorney|Client)[^:]*:\s*", "", m.group(2)).strip()
        if len(seg) > 25 and not re.match(r"(?i)(good morning|a contact|registered address|could you)", seg):
            facts.append({"fact": seg, "source_timestamp": m.group(1)})
    money = re.findall(r"(?:US\$|S\$|USD|SGD)\s?[\d,]+(?:\.\d+)?(?:\s?(?:million|m|bn))?", text)
    dates = re.findall(r"\b\d{1,2}\s+[A-Z][a-z]+\s+20\d{2}\b", text)
    ff = {}
    if money:
        ff["amount_1"] = money[0]
    if len(money) > 1:
        ff["amount_2"] = money[1]
    if dates:
        ff["key_date"] = dates[0]
    cm = re.search(r"Client\s*\(([^)]+)\)", text)
    return {
        "case_metadata": {"client_name": cm.group(1).strip() if cm else None,
                          "matter_type": None,
                          "jurisdiction": "SICC (mentioned)" if re.search(r"SICC|Singapore International Commercial", text) else None,
                          "date_of_meeting": None},
        "legal_facts": facts, "form_fields": ff, "action_items": [],
        "privacy_notice": "", "review_needed": [
            {"field": "case_metadata.date_of_meeting", "reason": "Meeting date not detected; confirm manually."}],
    }


def model_extract(text: str):
    """Run TRANSCRIPT_PROMPT through Claude. Falls back to heuristic if no key."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return heuristic_transcript(text), "heuristic"
    import anthropic
    client = anthropic.Anthropic()
    msg = client.messages.create(
        model=MODEL, max_tokens=4000,
        system=TRANSCRIPT_PROMPT,
        messages=[{"role": "user", "content": text[:120000]}])
    out = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text").strip()
    out = re.sub(r"^```(?:json)?|```$", "", out, flags=re.M).strip()
    try:
        return json.loads(out), "anthropic"
    except json.JSONDecodeError:
        raise HTTPException(status_code=502, detail="Model did not return valid JSON.")


# ----------------------------------------------------------------------------
# Core pipeline
# ----------------------------------------------------------------------------
def process(text: str) -> dict:
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="Empty transcript.")
    # Redact sensitive identifiers before the text reaches the model.
    redacted, categories = redact_pii(text)
    record, engine = model_extract(redacted)
    if categories:
        record["privacy_notice"] = ("Redacted PII categories identified: "
                                    + ", ".join(categories) + ".")
    elif not record.get("privacy_notice"):
        record["privacy_notice"] = "No structured PII detected."
    record["redacted_categories"] = categories
    record["_engine"] = engine
    return record


def text_from_upload(up: UploadFile, data: bytes) -> str:
    name = (up.filename or "").lower()
    ext = os.path.splitext(name)[1]
    if ext in AUDIO_EXT:
        return asr_transcribe(data, up.filename)
    # Reuse the engine's extractor for pdf / txt / chat by writing a temp file.
    import pathlib
    with tempfile.NamedTemporaryFile(suffix=ext or ".txt", delete=False) as tmp:
        tmp.write(data); tmp_path = tmp.name
    try:
        rec = extract_text(pathlib.Path(tmp_path))
        if not rec["text"]:
            raise HTTPException(status_code=415,
                                detail=f"Could not extract text ({rec.get('note', 'unsupported')}).")
        return rec["text"]
    finally:
        os.unlink(tmp_path)


# ----------------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok",
            "extraction": "anthropic" if os.environ.get("ANTHROPIC_API_KEY") else "heuristic-fallback",
            "asr": os.environ.get("ASR_PROVIDER", "openai") if (os.environ.get("OPENAI_API_KEY") or os.environ.get("DEEPGRAM_API_KEY")) else "not-configured",
            "model": MODEL}


@app.post("/extract")
async def extract(request: Request):
    ct = request.headers.get("content-type", "")
    if ct.startswith("application/json"):
        body = await request.json()
        text = body.get("transcript", "")
        return JSONResponse(process(text))
    if "multipart/form-data" in ct:
        form = await request.form()
        up = form.get("file")
        if up is None:
            raise HTTPException(status_code=400, detail="No file field provided.")
        data = await up.read()
        text = text_from_upload(up, data)
        return JSONResponse(process(text))
    raise HTTPException(status_code=415, detail="Send application/json {transcript} or multipart file.")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("intake_service:app", host="0.0.0.0",
                port=int(os.environ.get("PORT", 8000)), reload=False)
