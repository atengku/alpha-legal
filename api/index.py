"""Vercel entry for Seeking Alpha Legal intake API - hardened + path fix."""
import os, sys, traceback
sys.path.insert(0, os.path.dirname(__file__))
from fastapi import FastAPI
_inner = None
_err = None
try:
    from intake_service import app as _inner_app
    _inner = _inner_app
except Exception:
    _err = traceback.format_exc()
app = FastAPI()
@app.get("/api/health")
def health():
    return {"status":"ok","layer":"entry","backend_loaded":_inner is not None,"import_error":_err}
@app.get("/api/_debug")
def debug():
    return {"backend_loaded":_inner is not None,"import_error":_err}
if _inner is not None:
    app.mount("/api", _inner)
