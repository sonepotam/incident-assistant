import pytest

from incident_assistant.agent import normalize_triage
from incident_assistant.structured import JSONExtractionError, extract_json, non_latin_ratio


def test_extract_plain_json():
    assert extract_json('{"a": 1}') == {"a": 1}


def test_extract_from_markdown_fence_and_prose():
    text = 'Sure! Here is the triage:\n```json\n{"a": [1, 2]}\n```\nHope it helps.'
    assert extract_json(text) == {"a": [1, 2]}


def test_extract_ignores_trailing_text_after_object():
    assert extract_json('{"a": 1} and some explanation') == {"a": 1}


def test_extract_repairs_trailing_commas():
    assert extract_json('{"a": [1, 2,], "b": 3,}') == {"a": [1, 2], "b": 3}


@pytest.mark.parametrize("bad", ["", "I cannot help with that.", '{"a": 1', "{'a': 1}"])
def test_extract_rejects_garbage(bad):
    with pytest.raises(JSONExtractionError):
        extract_json(bad)


def test_non_latin_ratio():
    assert non_latin_ratio("PayGate timeouts") == 0
    assert non_latin_ratio("Таймауты PayGate") > 0.5


@pytest.mark.parametrize(
    "text, expected",
    [
        ("No errors in logs, billing-service CPU and latency are normal.", False),
        ("Sharp increase in response time for /payments/create.", True),
        ("/payments/create takes 4-8 seconds.", True),
        ("Checkout slowed down after the deploy.", True),
    ],
)
def test_latency_signal_ignores_negated_mentions(text, expected):
    from incident_assistant.rules import rule_based_parse

    assert ("high latency" in rule_based_parse(text).error_signals) is expected


@pytest.mark.parametrize(
    "text, signal",
    [
        ("Payment DB CPU is at 95%.", "high DB CPU"),
        ("DB dashboards show high CPU.", "high DB CPU"),
        ('Logs: "token signature verification failed".', "invalid token signature"),
        ("Logs show messages about invalid token signatures.", "invalid token signature"),
        ("В логах payment-service массовые таймауты при обращении к PayGate.", "timeouts"),
        ("В логах ошибки соединения с SMTP-провайдером.", "connection errors"),
        ("Высокая нагрузка на CPU, много долгих запросов.", "high DB CPU"),
        ("Высокая нагрузка на CPU, много долгих запросов.", "long-running queries"),
        ("Неверная подпись токена при входе.", "invalid token signature"),
    ],
)
def test_error_signals_english_and_russian(text, signal):
    from incident_assistant.rules import rule_based_parse

    assert signal in rule_based_parse(text).error_signals


def test_cpu_signal_ignores_normal_cpu():
    from incident_assistant.rules import rule_based_parse

    assert "high DB CPU" not in rule_based_parse("billing-service CPU and latency are normal.").error_signals


def test_normalize_fixes_harmless_deviations():
    data = {
        "result": {
            "category": "external payment provider issues",
            "summary": "x" * 20,
            "severity": "Critical",
            "hypotheses": [{"title": "t", "reasoning": "r", "nextSteps": ["a", "b", "c", "d"]}] * 4,
        }
    }
    fixed, fixes = normalize_triage(data)
    assert fixed["category"] == "External payment provider issue"
    assert fixed["severity"] == "high"
    assert len(fixed["hypotheses"]) == 3
    assert fixed["hypotheses"][0]["next_steps"] == ["a", "b", "c"]
    assert any("unwrapped" in f for f in fixes)


def test_normalize_does_not_guess_unrelated_category():
    fixed, _ = normalize_triage({"category": "Cosmic rays"})
    assert fixed["category"] == "Cosmic rays"  # left for schema validation -> regeneration