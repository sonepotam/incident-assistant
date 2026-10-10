"""Deterministic signal extraction.

Used (1) to ground the LLM parse stage (services found by regex are always kept) and
(2) as a fallback parser when the LLM parse stage fails completely.
"""

from __future__ import annotations

import re

from .schema import ParsedIncident

SERVICE_PATTERNS: dict[str, str] = {
    "api-gateway": r"\bapi[- ]?gateway\b|\bgateway\b",
    "auth-service": r"\bauth[- ]service\b|\bauth\b",
    "payment-service": r"\bpayments?[- ]service\b",
    "billing-service": r"\bbilling[- ]service\b|\bbilling\b",
    "notification-service": r"\bnotifications?[- ]service\b",
    "reporting-service": r"\breporting[- ]service\b|\breporting\b",
}

# Patterns run on lower-cased text. Each pattern has an English and a Russian part.
# Only positive statements are matched: "latency is normal" must not produce a latency signal.
ERROR_PATTERNS: list[tuple[str, str]] = [
    ("timeouts", r"time[- ]?outs?\b|тайм-?аут\w*"),
    ("504 Gateway Timeout", r"\b504\b"),
    ("5xx errors", r"\b5xx\b|\b50[0-3]\b"),
    ("401 Unauthorized", r"\b401\b"),
    ("invalid token signature", (
        r"invalid (?:token )?signatures?|signature verification failed|signature (?:is )?invalid"
        r"|(?:неверн|недействительн)\w* подпис\w*|ошибк\w* проверки подписи"
    )),
    ("invalid credentials", r"invalid credentials|неверн\w* (?:учетн|учётн)\w* данн\w*"),
    ("connection errors", (
        r"connection (?:errors?|refused|reset|failures?)"
        r"|ошибк\w* (?:подключения|соединения)|не уда[её]тся подключиться"
    )),
    ("high DB CPU", (
        r"high cpu|cpu (?:is )?(?:high|spik)|cpu\D{0,15}(?:[89]\d|100) ?%"
        r"|высок\w* (?:нагрузк\w* (?:на )?)?(?:cpu|цп|процессор)"
    )),
    ("long-running queries", r"long[- ]running quer|долг\w* запрос\w*"),
    ("high latency", (
        r"(?:high|increased|growing) latency|latency (?:spike|increase|grew|went up|is (?:high|up))"
        r"|response time (?:grew|increase|is (?:high|up))|increase in response time"
        r"|\bslow(?:s|ed)? down\b|\bis slow\b|\btakes? (?:up to )?\d+(?:-\d+)? ?(?:s|sec|seconds)\b"
        r"|время ответа (?:вырос|увелич)\w*|медленн\w*|тормоз\w*"
    )),
]

_TIME = re.compile(r"\b(\d{1,2}:\d{2})\s*(UTC|GMT|MSK)?\b", re.IGNORECASE)


def find_services(text: str) -> list[str]:
    lower = text.lower()
    return [name for name, pattern in SERVICE_PATTERNS.items() if re.search(pattern, lower)]


def canonical_service(name: str) -> str | None:
    """Map a free-form service name from the LLM ('Payment Service') to a known one."""
    lower = name.strip().lower()
    if lower in SERVICE_PATTERNS:
        return lower
    found = find_services(lower)
    return found[0] if len(found) == 1 else None


def rule_based_parse(text: str) -> ParsedIncident:
    lower = text.lower()
    errors = [label for label, pattern in ERROR_PATTERNS if re.search(pattern, lower)]
    time_match = _TIME.search(text)
    start_time = " ".join(p for p in time_match.groups() if p) if time_match else None
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]
    return ParsedIncident(
        symptoms=sentences[:5],
        services=find_services(text),
        error_signals=errors,
        start_time=start_time,
        user_impact=None,
    )
