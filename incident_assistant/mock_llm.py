"""Offline keyword-based stand-in for an LLM (LLM_PROVIDER=mock).

It lets you run the whole pipeline (CLI, API, validation, tracing) without an API key.
It is NOT the triage logic of the product: quality of real triage comes from the real model.
"""

from __future__ import annotations

import json
import re

from .llm import Message
from .rules import rule_based_parse

_INCIDENT = re.compile(r"<incident>\s*(.*?)\s*</incident>", re.DOTALL)

_TEMPLATES: list[tuple[tuple[str, ...], dict]] = [
    (
        ("paygate", "provider", "card"),
        {
            "category": "External payment provider issue",
            "severity": "high",
            "summary": "payment-service calls to an external payment provider are failing, so card payments do not go through. Customers paying by card are affected.",
            "hypotheses": [
                {
                    "title": "Outage or degradation on the payment provider side",
                    "reasoning": "Errors are concentrated on calls from payment-service to the external provider.",
                    "next_steps": [
                        "Check the provider status page and recent provider notifications.",
                        "In ELK, break down payment-service provider-call errors by provider and error type (timeout, 5xx, auth).",
                        "Compare latency and error rate of this provider with other providers; consider shifting traffic.",
                    ],
                },
                {
                    "title": "Network or credential problem between payment-service and the provider",
                    "reasoning": "Timeouts and invalid-credential errors can come from our side (egress, DNS, rotated keys).",
                    "next_steps": [
                        "Check recent deploys/config changes of payment-service (provider endpoints, credentials).",
                        "Test connectivity to the provider endpoint from a payment-service pod.",
                    ],
                },
            ],
        },
    ),
    (
        ("reporting", "long-running"),
        {
            "category": "DB degradation caused by reporting",
            "severity": "high",
            "summary": "Database load from reporting-service queries slows down payment processing; some requests time out at api-gateway. Customers creating payments are affected.",
            "hypotheses": [
                {
                    "title": "Heavy analytical queries from reporting-service saturate PostgreSQL",
                    "reasoning": "High DB CPU coincides with long-running queries from reporting-service and slow payment endpoints.",
                    "next_steps": [
                        "List active long-running queries in pg_stat_activity and identify those from reporting-service.",
                        "Correlate the start of the latency spike with scheduled report/export jobs.",
                        "If confirmed, cancel the heaviest reporting queries and pause report jobs.",
                    ],
                },
            ],
        },
    ),
    (
        ("e-mail", "email", "smtp", "sms", "notification"),
        {
            "category": "Notification delivery issue",
            "severity": "medium",
            "summary": "Notifications are not delivered to users while the underlying operations succeed. Users waiting for confirmations are affected.",
            "hypotheses": [
                {
                    "title": "External SMTP/SMS provider is degraded",
                    "reasoning": "notification-service logs show connection errors to the external provider.",
                    "next_steps": [
                        "Check the SMTP/SMS provider status page.",
                        "In ELK, chart notification-service send errors over time by provider and error type.",
                        "Check the notification queue/backlog to estimate how many messages are pending.",
                    ],
                },
            ],
        },
    ),
    (
        ("401", "login", "log in", "token", "auth"),
        {
            "category": "User authentication errors",
            "severity": "high",
            "summary": "auth-service rejects user logins or tokens. Users trying to log in are affected.",
            "hypotheses": [
                {
                    "title": "Signing key mismatch after key rotation or deploy",
                    "reasoning": "Invalid token signature errors usually mean issuer and verifier use different keys.",
                    "next_steps": [
                        "Check recent auth-service deploys and signing-key/config changes.",
                        "In ELK, break down 401 errors by client type and app version.",
                    ],
                },
            ],
        },
    ),
]

_OTHER = {
    "category": "Other",
    "severity": "medium",
    "summary": "The incident does not match known patterns; manual investigation is required. Impact is unclear from the report.",
    "hypotheses": [
        {
            "title": "Recent change in an affected service",
            "reasoning": "No known incident pattern matches the report.",
            "next_steps": [
                "Check recent deploys and config changes of the mentioned services.",
                "Review error logs of the mentioned services in ELK around the reported time.",
            ],
        }
    ],
}


class MockLLM:
    name = "mock"

    def complete(self, stage: str, system: str, messages: list[Message]) -> str:
        match = _INCIDENT.search(messages[0]["content"])
        text = match.group(1) if match else messages[0]["content"]
        if stage == "parse":
            return rule_based_parse(text).model_dump_json()
        lower = text.lower()
        best, best_hits = _OTHER, 0
        for keywords, template in _TEMPLATES:
            hits = sum(1 for k in keywords if k in lower)
            if hits > best_hits:
                best, best_hits = template, hits
        return json.dumps(best)
