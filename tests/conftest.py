import json

import pytest

from incident_assistant.config import DEFAULT_KNOWLEDGE_DIR
from incident_assistant.knowledge import KnowledgeBase


class ScriptedLLM:
    """Test double: returns pre-scripted answers per stage and records every call."""

    name = "scripted"

    def __init__(self, parse=None, triage=None):
        self.scripts = {"parse": list(parse or []), "triage": list(triage or [])}
        self.calls = []

    def complete(self, stage, system, messages):
        self.calls.append({"stage": stage, "messages": [dict(m) for m in messages]})
        script = self.scripts[stage]
        if not script:
            raise AssertionError(f"unexpected extra call for stage {stage}")
        answer = script.pop(0)
        return answer if isinstance(answer, str) else json.dumps(answer)


VALID_PARSE = {
    "symptoms": ["card payments fail"],
    "services": ["payment-service"],
    "error_signals": ["timeouts calling PayGate"],
    "start_time": "12:05 UTC",
    "user_impact": "Customers paying by card",
}

VALID_TRIAGE = {
    "category": "External payment provider issue",
    "summary": "PayGate does not respond in time, so card payments fail for many customers.",
    "severity": "high",
    "hypotheses": [
        {
            "title": "PayGate outage",
            "reasoning": "Timeouts only on PayGate calls.",
            "next_steps": ["Check PayGate status page.", "Compare PayGate error rate with other providers."],
        }
    ],
}

INCIDENT = (
    "Customers complain that card payments often fail. payment-service logs show many timeouts "
    "when calling PayGate, starting from 12:05 UTC. Other services look normal."
)


@pytest.fixture
def kb():
    return KnowledgeBase.load(DEFAULT_KNOWLEDGE_DIR)
