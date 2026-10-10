"""Output contracts (pydantic models = JSON schema) and the fixed category / severity vocabularies."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

# Closed set of categories. A closed set keeps the output machine-usable (routing, dashboards)
# and lets us validate it; "Other" is the explicit escape hatch instead of an invented label.
CATEGORIES: dict[str, str] = {
    "External payment provider issue": (
        "payment-service fails when calling an external payment provider (e.g. PayGate): "
        "timeouts, 5xx, invalid credentials."
    ),
    "DB degradation caused by reporting": (
        "PostgreSQL overload (high CPU, long-running queries, locks) caused by reporting-service "
        "analytical queries, slowing down other services."
    ),
    "Database issue": (
        "Other PostgreSQL problems (connections, replication, disk, locks, slow queries) "
        "not caused by reporting-service."
    ),
    "Notification delivery issue": (
        "notification-service fails to deliver e-mail/SMS, typically due to external SMTP/SMS providers."
    ),
    "User authentication errors": (
        "auth-service rejects logins or tokens: 401s, invalid token signatures, token issuance failures."
    ),
    "Other": "Does not match any category above (e.g. billing logic bugs, gateway misrouting, deploy regressions).",
}

SEVERITY_SCALE: dict[str, str] = {
    "high": (
        "A core money or access flow (payments, balance, login) is broken or badly degraded "
        "for many customers (including a step required to complete it, e.g. one-time codes, failing for a "
        "significant share of users); OR money data may be wrong (double charges, duplicate invoices, "
        "incorrect balances, lost transactions) - this is high regardless of how many customers are affected."
    ),
    "medium": (
        "Partial degradation or a non-core flow is broken with visible customer impact "
        "(e.g. informational notifications such as top-up confirmation e-mails not delivered, "
        "some users affected, workaround exists), and money data is not at risk."
    ),
    "low": "Minor or internal impact only, no or negligible customer impact.",
}

Severity = Literal["low", "medium", "high"]


class _Strict(BaseModel):
    # Unknown extra keys from the model are ignored; whitespace is stripped.
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)


class Hypothesis(_Strict):
    title: str = Field(min_length=3)
    reasoning: str = Field(min_length=3)
    next_steps: list[str] = Field(min_length=2, max_length=3)

    @field_validator("next_steps")
    @classmethod
    def _non_empty_steps(cls, steps: list[str]) -> list[str]:
        cleaned = [s.strip() for s in steps]
        if any(not s for s in cleaned):
            raise ValueError("next_steps must not contain empty strings")
        return cleaned


class TriageResult(_Strict):
    category: str
    summary: str = Field(min_length=10)
    severity: Severity
    hypotheses: list[Hypothesis] = Field(min_length=1, max_length=3)

    @field_validator("category")
    @classmethod
    def _known_category(cls, value: str) -> str:
        if value not in CATEGORIES:
            raise ValueError(f"category must be one of {list(CATEGORIES)}, got {value!r}")
        return value


class ParsedIncident(_Strict):
    """Stage-1 output: structured signals extracted from free-text incident description."""

    symptoms: list[str] = Field(default_factory=list)
    services: list[str] = Field(default_factory=list)
    error_signals: list[str] = Field(default_factory=list)
    start_time: str | None = None
    user_impact: str | None = None
