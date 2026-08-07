#!/usr/bin/env python3
"""Seeking Alpha Legal — Matter Intake pipeline.

Ingest mixed evidence files (audio, chat, PDF, txt), extract a structured
"matter record", auto-map that record onto the SICC forms, and persist
everything to a database for later reuse.

Pipeline:  ingest -> extract -> map -> store

What is real and runnable here:
  - text extraction from PDF, txt and chat exports
  - the matter-record schema
  - the deterministic matter-to-form field mapping
  - SQLite storage (intake.db) + JSON export

What needs production wiring (clearly marked):
  - the extract step is an LLM call (prompt below). Set ANTHROPIC_API_KEY and
    pass --llm to run it for real; otherwise a heuristic fallback runs.
  - audio is transcribed by an ASR service (Whisper / Deepgram) before extract.

Usage:
    python3 intake_agent.py ingest --uploads ./intake_uploads --matter "Name"
    python3 intake_agent.py demo            # build the worked demo matter
"""
import argparse
import hashlib
import json
import os
import re
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

# ----------------------------------------------------------------------------
# Matter-record schema (the database format for reuse). Every extraction
# conforms to this, so downstream forms, analytics and exports are stable.
# ----------------------------------------------------------------------------
MATTER_SCHEMA = {
    "matter": "name, internal ref, status, one-line summary",
    "parties": "claimants[] and defendants[]: name, type, place_of_business, country",
    "jurisdiction": "written_agreement (bool), clause, international_factors[], commercial_basis",
    "offshore": "is_offshore (bool), reasons[]",
    "causes_of_action": "[{party, cause, summary}]",
    "material_facts": "[{n, fact, date, source}]",
    "harm": "narrative of loss/harm",
    "relief": "[{type, description, quantum, currency}]",
    "key_dates": "[{date, event, source}]",
    "witnesses": "[{name, role, party, topics[]}]",
    "issues": "[{party, cause, issue, sub_issues[], status: agreed|non-agreed|common-ground}]",
    "counsel": "[{name, firm, side}]",
    "documents": "[{title, kind, source_file, relevance}]",
}

# Production extraction prompt. Send the concatenated source text as the user
# message; parse the JSON reply against MATTER_SCHEMA.
EXTRACTION_PROMPT = """You are a litigation intake analyst for the Singapore International Commercial Court.
From the supplied evidence (transcribed audio, chat logs, pleadings, correspondence),
extract a single JSON object matching this schema exactly, inferring nothing that is
not supported by the text and tagging each fact with the source it came from:

{schema}

Rules:
- Use only what the sources support. Mark anything uncertain with "confidence":"low".
- Frame each issue as a "Whether..." question and group by party and cause of action.
- Quantify relief where a figure appears, with its currency.
- Output JSON only, no commentary.""".format(schema=json.dumps(MATTER_SCHEMA, indent=2))


# Transcript extraction contract (attorney-client meeting -> firm schema).
# This is the production system prompt for the analyze step on transcripts.
TRANSCRIPT_PROMPT = """Role: You are a specialized Legal AI Assistant for a law firm document automation system. Your goal is to process meeting transcripts into structured data that maps directly to legal forms.
Input: a transcript of a meeting between an attorney and a client.

Instructions:
1. Analyze the transcript to identify key legal facts, entities, and action items.
2. Identify and categorize sensitive PII (NRIC or passport numbers, addresses, bank details). Do not reproduce the PII values; list the categories found.
3. Output a JSON object with this exact schema: case_metadata {client_name, matter_type, jurisdiction, date_of_meeting}; legal_facts [{fact, source_timestamp "mm:ss"}]; form_fields {field_name: value}; action_items [{task, assigned_to, deadline}]; privacy_notice (list of redacted PII categories); review_needed [{field, reason, value}].
4. If a fact is mentioned ambiguously, do not guess: set the value to null and flag it under review_needed.
5. Maintain professional, neutral, precise language. Do not offer legal advice; only transcribe and structure.

Constraints:
- Do not hallucinate information not present in the transcript.
- Provide a source_timestamp for every legal fact so the lawyer can verify against the audio.
- If the transcript is unclear, prefer omission (safety) over guessing (accuracy).

Mapping example:
[Transcript]: "The client, John Doe, mentioned his company address is 123 Business Rd."
[Output]: {"client_name": "John Doe", "company_address": "123 Business Rd"}
Output JSON only."""


# ----------------------------------------------------------------------------
# Ingestion: extract text from each supported file type
# ----------------------------------------------------------------------------
def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:16]


PII_PATTERNS = [
    ("NRIC", r"\b[STFGM]\d{7}[A-Z]\b"),
    ("passport number", r"\b[A-Z]{1,2}\d{7}\b"),
    ("telephone number", r"(?:\+?65[\s-]?)?\b[89]\d{3}[\s-]?\d{4}\b"),
    ("email address", r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    ("bank account number", r"\b\d{3}-\d{6}-\d\b"),
    ("address", r"(?:Singapore\s\d{6})|(?:#\d{2}-\d{2,3})"),
]


def redact_pii(text: str):
    """Detect and mask PII. Returns (redacted_text, sorted categories found).
    Deterministic and runnable; complements the model's own PII flags."""
    found = []
    red = text
    for label, pat in PII_PATTERNS:
        if re.search(pat, red):
            found.append(label)
            red = re.sub(pat, f"[REDACTED: {label}]", red)
    # de-dupe while preserving a stable order
    seen, cats = set(), []
    for c in found:
        if c not in seen:
            seen.add(c); cats.append(c)
    return red, cats


def extract_text(path: Path) -> dict:
    """Return {kind, text, note}. Real for pdf/txt/chat; audio is a stub that
    marks where ASR transcription plugs in."""
    ext = path.suffix.lower()
    raw = path.read_bytes()
    rec = {"kind": "other", "text": "", "note": "", "sha": sha256(raw)}
    if ext in (".txt", ".md"):
        rec["kind"] = "text"
        rec["text"] = raw.decode("utf-8", errors="replace")
    elif ext in (".json", ".chat"):
        rec["kind"] = "chat"
        try:
            data = json.loads(raw.decode("utf-8", errors="replace"))
            msgs = data if isinstance(data, list) else data.get("messages", [])
            rec["text"] = "\n".join(f"{m.get('from', m.get('sender', '?'))}: {m.get('text', m.get('message', ''))}"
                                    for m in msgs)
        except Exception:
            rec["text"] = raw.decode("utf-8", errors="replace")
    elif ext == ".pdf":
        rec["kind"] = "pdf"
        try:
            import pypdf
            reader = pypdf.PdfReader(str(path))
            rec["text"] = "\n".join((pg.extract_text() or "") for pg in reader.pages)
        except Exception as ex:  # noqa: BLE001
            rec["note"] = f"PDF extract failed: {ex}"
    elif ext in (".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac"):
        rec["kind"] = "audio"
        rec["note"] = ("ASR transcription required (Whisper / Deepgram). "
                       "Transcribe to text, then feed into extract.")
    else:
        rec["note"] = f"Unsupported type {ext}"
    return rec


def ingest(uploads: Path) -> list:
    sources = []
    for p in sorted(uploads.glob("**/*")):
        if p.is_file():
            r = extract_text(p)
            r["filename"] = p.name
            r["chars"] = len(r["text"])
            sources.append(r)
            print(f"  ingested {r['kind']:5} {p.name}  ({r['chars']} chars)"
                  + (f"  [{r['note']}]" if r["note"] else ""))
    return sources


# ----------------------------------------------------------------------------
# Extraction (text -> matter record). LLM in production; heuristic fallback.
# ----------------------------------------------------------------------------
def llm_extract(text: str):
    """Call Claude to extract the matter record. Requires ANTHROPIC_API_KEY."""
    import anthropic  # noqa: F401
    client = anthropic.Anthropic()
    msg = client.messages.create(
        model="claude-opus-4-8", max_tokens=4000,
        system=EXTRACTION_PROMPT,
        messages=[{"role": "user", "content": text[:120000]}])
    out = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
    out = re.sub(r"^```json|```$", "", out.strip(), flags=re.M).strip()
    return json.loads(out)


def heuristic_extract(text: str) -> dict:
    """Cheap baseline so the pipeline produces something without an API key."""
    money = re.findall(r"(US\$|S\$|USD|SGD|\u20ac|\u00a3)\s?([\d,]+(?:\.\d+)?)\s*(million|m|bn)?", text)
    relief = []
    for ccy, amt, scale in money[:5]:
        relief.append({"type": "monetary", "description": "sum claimed",
                       "quantum": amt + (" " + scale if scale else ""), "currency": ccy})
    dates = re.findall(r"\b(\d{1,2}\s+\w+\s+20\d{2}|20\d{2}-\d{2}-\d{2})\b", text)
    return {"matter": {"summary": "heuristic extraction (no LLM key set)"},
            "relief": relief, "key_dates": [{"date": d, "event": ""} for d in dates[:8]],
            "_note": "Heuristic fallback. Set ANTHROPIC_API_KEY and pass --llm for full extraction."}


# ----------------------------------------------------------------------------
# Mapping: matter record -> SICC form fill packets
# ----------------------------------------------------------------------------
def _names(parties):
    return "; ".join(p.get("name", "") for p in parties or [])


def map_to_forms(rec: dict, forms_index: list) -> list:
    """Deterministically populate the relevant SICC forms from the matter record.
    Returns a list of fill packets with field values and a completeness score."""
    url = {f["name"]: f["url"] for f in (forms_index or [])}
    cl = rec.get("parties", {}).get("claimants", []) if isinstance(rec.get("parties"), dict) else []
    df = rec.get("parties", {}).get("defendants", []) if isinstance(rec.get("parties"), dict) else []
    rel = rec.get("relief", [])
    rel_txt = "; ".join(f"{r.get('description', r.get('type', ''))}"
                        f"{' (' + str(r.get('quantum')) + ' ' + str(r.get('currency', '')) + ')' if r.get('quantum') else ''}"
                        for r in rel)
    facts = rec.get("material_facts", [])
    facts_txt = " ".join(f"{f.get('n', '')}. {f.get('fact', '')}".strip() for f in facts)
    coa = rec.get("causes_of_action", [])
    coa_txt = "; ".join(f"{c.get('party', '')}: {c.get('cause', '')}" for c in coa)
    juris = rec.get("jurisdiction", {})
    offshore = rec.get("offshore", {})
    issues = rec.get("issues", [])
    counsel = rec.get("counsel", [])

    def packet(form_name, fields):
        filled = sum(1 for _, v in fields if v)
        return {"form": form_name, "form_url": url.get(form_name),
                "fields": [{"label": k, "value": v} for k, v in fields],
                "completeness": round(100 * filled / max(1, len(fields)))}

    packets = [
        packet("Form of Originating Application", [
            ("Claimant(s)", _names(cl)), ("Defendant(s)", _names(df)),
            ("Relief sought", rel_txt),
            ("Written jurisdiction agreement", juris.get("clause", "Yes" if juris.get("written_agreement") else "")),
        ]),
        packet("Form of Claimant's Statement", [
            ("Material facts giving rise to the claim", facts_txt),
            ("Harm suffered", rec.get("harm", "")),
            ("Cause(s) of action", coa_txt),
            ("Relief sought, with initial quantification", rel_txt),
        ]),
        packet("Form of offshore case declaration", [
            ("Is the action an offshore case", "Yes" if offshore.get("is_offshore") else ""),
            ("Basis / reasons", "; ".join(offshore.get("reasons", []) or [])),
        ]),
        packet("Form of list of issues", [
            ("Agreed issues", "; ".join(i.get("issue", "") for i in issues if i.get("status") == "agreed")),
            ("Non-agreed issues", "; ".join(i.get("issue", "") for i in issues if i.get("status") == "non-agreed")),
            ("Common ground", "; ".join(i.get("issue", "") for i in issues if i.get("status") == "common-ground")),
        ]),
        packet("Form for notice of appointment of counsel", [
            ("Counsel", "; ".join(f"{c.get('name', '')} ({c.get('firm', '')})" for c in counsel)),
        ]),
    ]
    return packets


# ----------------------------------------------------------------------------
# Storage (SQLite) + JSON export
# ----------------------------------------------------------------------------
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS matters(
  id TEXT PRIMARY KEY, name TEXT, status TEXT, summary TEXT,
  record_json TEXT, created_at TEXT);
CREATE TABLE IF NOT EXISTS sources(
  id INTEGER PRIMARY KEY AUTOINCREMENT, matter_id TEXT, filename TEXT,
  kind TEXT, sha256 TEXT, chars INTEGER, note TEXT, ingested_at TEXT);
CREATE TABLE IF NOT EXISTS form_fills(
  id INTEGER PRIMARY KEY AUTOINCREMENT, matter_id TEXT, form TEXT,
  form_url TEXT, completeness INTEGER, fields_json TEXT);
CREATE TABLE IF NOT EXISTS transcripts(
  matter_id TEXT PRIMARY KEY, filename TEXT, redacted_categories TEXT,
  extraction_json TEXT, created_at TEXT);
"""


def store(db: Path, matter_id: str, name: str, rec: dict, sources: list, fills: list, transcript: dict = None):
    con = sqlite3.connect(db)
    con.executescript(SCHEMA_SQL)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    summary = (rec.get("matter", {}) or {}).get("summary", "")
    con.execute("INSERT OR REPLACE INTO matters VALUES (?,?,?,?,?,?)",
                (matter_id, name, "intake", summary, json.dumps(rec, ensure_ascii=False), now))
    con.execute("DELETE FROM sources WHERE matter_id=?", (matter_id,))
    for s in sources:
        con.execute("INSERT INTO sources(matter_id,filename,kind,sha256,chars,note,ingested_at) VALUES (?,?,?,?,?,?,?)",
                    (matter_id, s["filename"], s["kind"], s["sha"], s["chars"], s.get("note", ""), now))
    con.execute("DELETE FROM form_fills WHERE matter_id=?", (matter_id,))
    for f in fills:
        con.execute("INSERT INTO form_fills(matter_id,form,form_url,completeness,fields_json) VALUES (?,?,?,?,?)",
                    (matter_id, f["form"], f.get("form_url"), f["completeness"], json.dumps(f["fields"], ensure_ascii=False)))
    if transcript:
        con.execute("INSERT OR REPLACE INTO transcripts VALUES (?,?,?,?,?)",
                    (matter_id, transcript.get("filename", ""),
                     json.dumps(transcript.get("redacted_categories", []), ensure_ascii=False),
                     json.dumps(transcript.get("extraction", {}), ensure_ascii=False), now))
    con.commit(); con.close()


def export_json(db: Path, out: Path):
    con = sqlite3.connect(db); con.row_factory = sqlite3.Row
    matters = []
    for m in con.execute("SELECT * FROM matters"):
        mid = m["id"]
        matters.append({
            "id": mid, "name": m["name"], "status": m["status"], "summary": m["summary"],
            "created_at": m["created_at"], "record": json.loads(m["record_json"]),
            "sources": [dict(r) for r in con.execute("SELECT filename,kind,sha256,chars,note FROM sources WHERE matter_id=?", (mid,))],
            "form_fills": [{**dict(r), "fields": json.loads(r["fields_json"])}
                           for r in con.execute("SELECT form,form_url,completeness,fields_json FROM form_fills WHERE matter_id=?", (mid,))],
        })
        trow = con.execute("SELECT filename,redacted_categories,extraction_json FROM transcripts WHERE matter_id=?", (mid,)).fetchone()
        if trow:
            matters[-1]["transcript"] = {"filename": trow["filename"],
                                          "redacted_categories": json.loads(trow["redacted_categories"]),
                                          "extraction": json.loads(trow["extraction_json"])}
    con.close()
    out.write_text(json.dumps({"matters": matters}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Exported {len(matters)} matter(s) -> {out}")


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def cmd_ingest(args):
    forms_index = json.loads(Path("forms.json").read_text()) if Path("forms.json").exists() else []
    uploads = Path(args.uploads)
    print(f"Ingesting {uploads} ...")
    sources = ingest(uploads)
    text = "\n\n".join(f"[{s['filename']}]\n{s['text']}" for s in sources if s["text"])
    if args.llm and os.environ.get("ANTHROPIC_API_KEY"):
        print("Extracting via Claude ...")
        rec = llm_extract(text)
    else:
        print("Extracting via heuristic fallback (no --llm / no key) ...")
        rec = heuristic_extract(text)
    fills = map_to_forms(rec, forms_index)
    mid = "m_" + sha256(text.encode())[:8]
    store(Path(args.db), mid, args.matter or mid, rec, sources, fills)
    export_json(Path(args.db), Path(args.out))
    print(f"Stored matter {mid}. Form fills: "
          + ", ".join(f"{f['form'].replace('Form of ', '').replace('Form for ', '')[:24]} {f['completeness']}%" for f in fills))


def cmd_demo(args):
    """Build a realistic worked matter from the bundled sample sources so the
    Intake view has something to show. The matter record here is the kind of
    output the LLM extract step produces."""
    forms_index = json.loads(Path("forms.json").read_text()) if Path("forms.json").exists() else []
    up = Path("intake_uploads")
    sources = ingest(up) if up.exists() else []
    rec = json.loads(Path("intake_demo_record.json").read_text(encoding="utf-8"))
    fills = map_to_forms(rec, forms_index)
    transcript = None
    tpath = Path("intake_transcript_extraction.json")
    tfile = Path("intake_uploads/intake_call_transcript.txt")
    if tpath.exists():
        extraction = json.loads(tpath.read_text(encoding="utf-8"))
        cats = []
        if tfile.exists():
            _, cats = redact_pii(tfile.read_text(encoding="utf-8"))
        transcript = {"filename": tfile.name if tfile.exists() else "transcript.txt",
                      "redacted_categories": cats, "extraction": extraction}
    store(Path(args.db), "m_demo01", rec["matter"]["name"], rec, sources, fills, transcript)
    export_json(Path(args.db), Path(args.out))
    print("Demo matter stored. Form fills: "
          + ", ".join(f"{f['form'].replace('Form of ', '').replace('Form for ', '')[:22]} {f['completeness']}%" for f in fills))


def main():
    ap = argparse.ArgumentParser(description="Seeking Alpha Legal matter intake")
    sub = ap.add_subparsers(required=True)
    a = sub.add_parser("ingest"); a.add_argument("--uploads", required=True)
    a.add_argument("--matter", default=""); a.add_argument("--db", default="intake.db")
    a.add_argument("--out", default="intake_demo.json"); a.add_argument("--llm", action="store_true")
    a.set_defaults(func=cmd_ingest)
    d = sub.add_parser("demo"); d.add_argument("--db", default="intake.db")
    d.add_argument("--out", default="intake_demo.json"); d.set_defaults(func=cmd_demo)
    args = ap.parse_args(); args.func(args)


if __name__ == "__main__":
    main()
