from fastapi.testclient import TestClient

from incident_assistant import cli
from incident_assistant.agent import IncidentAgent
from incident_assistant.api import _build_agent, app, get_agent
from incident_assistant.llm import LLMError
from incident_assistant.mock_llm import MockLLM

from .conftest import INCIDENT, VALID_PARSE, ScriptedLLM


def test_retrieval_finds_matching_past_incident(kb):
    top = kb.similar_incidents(
        "API latency up, DB dashboards show high CPU, long-running queries from reporting-service",
        ["reporting-service"],
    )
    assert top[0].incident.id == "INC-102"


def test_retrieval_drops_weak_noise_match(kb):
    # INC-101 shares only the token "utc" with this report and must not reach the prompt
    text = (
        "Since 09:40 UTC /payments/create takes 4-8 seconds and api-gateway returns 504 for a part of requests. "
        "The month-end export job in reporting-service was started at 09:35. Payment DB CPU is at 95%."
    )
    ids = [s.incident.id for s in kb.similar_incidents(text, ["api-gateway", "reporting-service"])]
    assert ids == ["INC-102"]


def test_retrieval_returns_nothing_for_unrelated_incident(kb):
    text = "Several merchants report they received two identical invoices. No errors in logs, CPU is normal."
    assert kb.similar_incidents(text, []) == []


def test_retrieval_auth(kb):
    assert kb.similar_incidents("401 errors, invalid token signatures on login", [])[0].incident.id == "INC-104"


def test_services_context_falls_back_to_whole_system(kb):
    assert all(name in kb.services_context([]) for name in kb.service_names)


def test_api_triage_and_debug(kb):
    app.dependency_overrides[get_agent] = lambda: IncidentAgent(MockLLM(), kb)
    try:
        client = TestClient(app)
        resp = client.post("/triage", json={"text": INCIDENT})
        assert resp.status_code == 200
        assert resp.json()["category"] == "External payment provider issue"

        resp = client.post("/triage?debug=true", json={"text": INCIDENT})
        assert [s["stage"] for s in resp.json()["trace"]["stages"]] == ["parse", "retrieve", "triage"]

        assert client.post("/triage", json={"text": "short"}).status_code == 422
    finally:
        app.dependency_overrides.clear()


def test_api_returns_502_when_model_output_never_valid(kb):
    llm = ScriptedLLM(parse=[VALID_PARSE], triage=["bad"] * 3)
    app.dependency_overrides[get_agent] = lambda: IncidentAgent(llm, kb)
    try:
        resp = TestClient(app).post("/triage", json={"text": INCIDENT})
        assert resp.status_code == 502
        assert len(resp.json()["detail"]["trace"]["stages"][-1]["attempts"]) == 3
    finally:
        app.dependency_overrides.clear()


def test_api_returns_503_when_provider_is_down(kb):
    class Down:
        name = "down"

        def complete(self, stage, system, messages):
            raise LLMError("connection refused")

    app.dependency_overrides[get_agent] = lambda: IncidentAgent(Down(), kb)
    try:
        assert TestClient(app).post("/triage", json={"text": INCIDENT}).status_code == 503
    finally:
        app.dependency_overrides.clear()


def test_api_returns_503_when_api_key_missing(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    monkeypatch.setattr("incident_assistant.config.load_dotenv", lambda: None)  # ignore a local .env
    _build_agent.cache_clear()
    try:
        resp = TestClient(app).post("/triage", json={"text": INCIDENT})
        assert resp.status_code == 503 and "ANTHROPIC_API_KEY" in resp.json()["detail"]["error"]
    finally:
        _build_agent.cache_clear()


def test_cli_exit_codes(monkeypatch, capsys):
    monkeypatch.setattr("incident_assistant.config.load_dotenv", lambda: None)
    assert cli.main(["--provider", "mock", "   "]) == 1                          # bad input
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    assert cli.main(["--provider", "anthropic", INCIDENT]) == 1                  # config: no key
    assert cli.main(["--provider", "mock", INCIDENT]) == 0


def test_schema_endpoint():
    schema = TestClient(app).get("/schema").json()
    assert set(schema["required"]) == {"category", "summary", "severity", "hypotheses"}
