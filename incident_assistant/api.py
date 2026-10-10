"""HTTP API.

  POST /triage            {"text": "..."} -> TriageResult JSON
  POST /triage?debug=true -> {"result": TriageResult, "trace": {...}}
  GET  /schema            -> JSON schema of TriageResult
  GET  /health
Run: uvicorn incident_assistant.api:app --port 8000
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from .agent import MAX_INPUT_CHARS, IncidentAgent, TriageError
from .config import Settings
from .llm import LLMError
from .schema import TriageResult

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper(), format="%(levelname)s %(name)s: %(message)s")

app = FastAPI(title="AI Incident Assistant", version="0.1.0")


class TriageRequest(BaseModel):
    text: str = Field(min_length=10, max_length=MAX_INPUT_CHARS, description="Free-text incident description")


@lru_cache(maxsize=1)
def get_agent() -> IncidentAgent:
    return IncidentAgent.from_settings(Settings.from_env())


@app.get("/health")
def health(agent: IncidentAgent = Depends(get_agent)) -> dict:
    return {"status": "ok", "provider": agent.llm.name}


@app.get("/schema")
def schema() -> dict:
    return TriageResult.model_json_schema()


@app.post("/triage")
def triage(req: TriageRequest, debug: bool = False, agent: IncidentAgent = Depends(get_agent)) -> dict:
    try:
        outcome = agent.run(req.text)
    except TriageError as exc:
        raise HTTPException(status_code=502, detail={"error": str(exc), "trace": exc.trace}) from exc
    except LLMError as exc:
        raise HTTPException(status_code=503, detail={"error": str(exc)}) from exc
    result = outcome.result.model_dump()
    return {"result": result, "trace": outcome.trace} if debug else result
