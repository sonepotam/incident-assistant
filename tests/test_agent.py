import copy

import pytest

from incident_assistant.agent import IncidentAgent, TriageError
from incident_assistant.llm import LLMError
from incident_assistant.mock_llm import MockLLM

from .conftest import INCIDENT, VALID_PARSE, VALID_TRIAGE, ScriptedLLM


def triage_attempts(outcome):
    return next(s for s in outcome.trace["stages"] if s["stage"] == "triage")["attempts"]


def test_happy_path_single_call_per_stage(kb):
    llm = ScriptedLLM(parse=[VALID_PARSE], triage=[VALID_TRIAGE])
    outcome = IncidentAgent(llm, kb).run(INCIDENT)
    assert outcome.result.category == "External payment provider issue"
    assert [c["stage"] for c in llm.calls] == ["parse", "triage"]
    # stage 2 output (past incidents, system context) reaches the generation prompt
    triage_prompt = llm.calls[1]["messages"][0]["content"]
    assert "INC-101" in triage_prompt and "payment-service" in triage_prompt


def test_invalid_json_triggers_regeneration_with_error_feedback(kb):
    llm = ScriptedLLM(parse=[VALID_PARSE], triage=["Sure! The category is provider issue.", VALID_TRIAGE])
    outcome = IncidentAgent(llm, kb).run(INCIDENT)
    attempts = triage_attempts(outcome)
    assert [a["ok"] for a in attempts] == [False, True]
    repair_msgs = llm.calls[-1]["messages"]
    assert [m["role"] for m in repair_msgs] == ["user", "assistant", "user"]
    assert "no JSON object found" in repair_msgs[-1]["content"]


def test_wrong_format_triggers_regeneration(kb):
    bad = {"category": "Payment stuff", "summary": "short", "severity": "urgent", "hypotheses": []}
    llm = ScriptedLLM(parse=[VALID_PARSE], triage=[bad, VALID_TRIAGE])
    outcome = IncidentAgent(llm, kb).run(INCIDENT)
    errors = triage_attempts(outcome)[0]["errors"]
    joined = " ".join(errors)
    assert "category" in joined and "severity" in joined and "hypotheses" in joined
    assert outcome.result.severity == "high"


def test_wrong_language_triggers_regeneration(kb):
    russian = copy.deepcopy(VALID_TRIAGE)
    russian["summary"] = "Платёжный провайдер PayGate не отвечает, оплаты картой не проходят."
    russian["hypotheses"][0]["reasoning"] = "Таймауты наблюдаются только при обращении к PayGate."
    llm = ScriptedLLM(parse=[VALID_PARSE], triage=[russian, VALID_TRIAGE])
    outcome = IncidentAgent(llm, kb).run(INCIDENT)
    assert "English" in triage_attempts(outcome)[0]["errors"][0]
    assert "English" in llm.calls[-1]["messages"][-1]["content"]


def test_harmless_deviation_is_autocorrected_without_extra_call(kb):
    sloppy = copy.deepcopy(VALID_TRIAGE)
    sloppy["severity"] = "High"
    llm = ScriptedLLM(parse=[VALID_PARSE], triage=["```json\n" + __import__("json").dumps(sloppy) + "\n```"])
    outcome = IncidentAgent(llm, kb).run(INCIDENT)
    attempt = triage_attempts(outcome)[0]
    assert attempt["ok"] and attempt["corrections"] == ["severity 'High' -> 'high'"]
    assert len(llm.calls) == 2


def test_gives_up_after_max_attempts(kb):
    llm = ScriptedLLM(parse=[VALID_PARSE], triage=["nope", "still nope", "{broken"])
    with pytest.raises(TriageError) as exc_info:
        IncidentAgent(llm, kb, max_attempts=3).run(INCIDENT)
    stage = exc_info.value.trace["stages"][-1]
    assert len(stage["attempts"]) == 3 and not any(a["ok"] for a in stage["attempts"])


def test_parse_stage_falls_back_to_rules(kb):
    llm = ScriptedLLM(parse=["garbage", "garbage", "garbage"], triage=[VALID_TRIAGE])
    outcome = IncidentAgent(llm, kb).run(INCIDENT)
    parse_stage = outcome.trace["stages"][0]
    assert parse_stage["source"] == "rules_fallback"
    assert "payment-service" in outcome.trace["stages"][1]["services"]


def test_parse_stage_drops_hallucinated_services_and_adds_mentioned_ones(kb):
    parse = dict(VALID_PARSE, services=["Payment Service", "fraud-service"])
    text = INCIDENT + " api-gateway is fine."
    llm = ScriptedLLM(parse=[parse], triage=[VALID_TRIAGE])
    outcome = IncidentAgent(llm, kb).run(text)
    services = outcome.trace["stages"][1]["services"]
    assert "fraud-service" not in services
    assert services[:2] == ["payment-service", "api-gateway"]


def test_provider_failure_becomes_triage_error(kb):
    class Down:
        name = "down"

        def complete(self, stage, system, messages):
            raise LLMError("connection refused")

    with pytest.raises(TriageError):
        IncidentAgent(Down(), kb).run(INCIDENT)


def test_empty_input_rejected(kb):
    with pytest.raises(ValueError):
        IncidentAgent(ScriptedLLM(), kb).run("   ")


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Card payments fail, payment-service shows timeouts calling PayGate.", "External payment provider issue"),
        ("/payments/create is slow, DB CPU high with long-running queries from reporting-service.",
         "DB degradation caused by reporting"),
        ("Users do not receive top-up e-mails, SMTP connection errors in notification-service.",
         "Notification delivery issue"),
        ("Users cannot log in, auth-service returns 401, invalid token signatures.", "User authentication errors"),
        ("Merchants received duplicate invoices after a deploy.", "Other"),
    ],
)
def test_end_to_end_with_offline_mock(kb, text, expected):
    assert IncidentAgent(MockLLM(), kb).run(text).result.category == expected
