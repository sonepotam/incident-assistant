"""Orchestrator: parse -> retrieve context -> generate structured triage (with validation/repair)."""

from __future__ import annotations

import difflib
import logging
import time
from dataclasses import dataclass, field

from . import prompts
from .config import Settings
from .knowledge import KnowledgeBase
from .llm import LLMClient, LLMError, build_client
from .rules import canonical_service, find_services, rule_based_parse
from .schema import CATEGORIES, ParsedIncident, TriageResult
from .structured import StructuredOutputError, call_structured, non_latin_ratio

log = logging.getLogger(__name__)

MAX_INPUT_CHARS = 8000
NON_LATIN_THRESHOLD = 0.15

_SEVERITY_ALIASES = {
    "critical": "high", "blocker": "high", "major": "high", "sev1": "high", "p1": "high",
    "moderate": "medium", "med": "medium", "sev2": "medium", "p2": "medium",
    "minor": "low", "sev3": "low", "p3": "low",
}
_NEXT_STEPS_ALIASES = ("nextSteps", "next-steps", "steps", "actions")


class TriageError(Exception):
    def __init__(self, message: str, trace: dict):
        super().__init__(message)
        self.trace = trace


@dataclass
class TriageOutcome:
    result: TriageResult
    trace: dict = field(default_factory=dict)


# ---------- auto-correction (deterministic, no LLM call) ----------

def match_category(value: str) -> str | None:
    by_lower = {c.lower(): c for c in CATEGORIES}
    key = value.strip().lower()
    if key in by_lower:
        return by_lower[key]
    close = difflib.get_close_matches(key, list(by_lower), n=1, cutoff=0.8)
    return by_lower[close[0]] if close else None


def normalize_triage(data: dict) -> tuple[dict, list[str]]:
    fixes: list[str] = []
    if "category" not in data and len(data) == 1 and isinstance(next(iter(data.values())), dict):
        data = next(iter(data.values()))
        fixes.append("unwrapped result from a single top-level key")
    data = dict(data)

    sev = data.get("severity")
    if isinstance(sev, str):
        norm = sev.strip().lower()
        norm = _SEVERITY_ALIASES.get(norm, norm)
        if norm != sev:
            fixes.append(f"severity {sev!r} -> {norm!r}")
            data["severity"] = norm

    cat = data.get("category")
    if isinstance(cat, str) and cat not in CATEGORIES:
        matched = match_category(cat)
        if matched:
            fixes.append(f"category {cat!r} -> {matched!r}")
            data["category"] = matched

    hyps = data.get("hypotheses")
    if isinstance(hyps, list):
        if len(hyps) > 3:
            fixes.append(f"trimmed hypotheses {len(hyps)} -> 3")
            hyps = hyps[:3]
        fixed_hyps = []
        for i, hyp in enumerate(hyps):
            if isinstance(hyp, dict):
                hyp = dict(hyp)
                if "next_steps" not in hyp:
                    for alias in _NEXT_STEPS_ALIASES:
                        if alias in hyp:
                            hyp["next_steps"] = hyp.pop(alias)
                            fixes.append(f"hypotheses.{i}: renamed {alias!r} -> 'next_steps'")
                            break
                steps = hyp.get("next_steps")
                if isinstance(steps, list) and len(steps) > 3:
                    fixes.append(f"hypotheses.{i}: trimmed next_steps {len(steps)} -> 3")
                    hyp["next_steps"] = steps[:3]
            fixed_hyps.append(hyp)
        data["hypotheses"] = fixed_hyps
    return data, fixes


def check_english(result: TriageResult) -> list[str]:
    text = " ".join(
        [result.summary]
        + [f"{h.title} {h.reasoning} {' '.join(h.next_steps)}" for h in result.hypotheses]
    )
    ratio = non_latin_ratio(text)
    if ratio > NON_LATIN_THRESHOLD:
        return [f"text must be in English, but {ratio:.0%} of the letters are non-Latin"]
    return []


def make_parsed_normalizer(known: list[str]):
    def normalize(data: dict) -> tuple[dict, list[str]]:
        data, fixes = dict(data), []
        services = data.get("services")
        if isinstance(services, list):
            mapped = []
            for name in services:
                canon = canonical_service(name) if isinstance(name, str) else None
                if canon in known and canon not in mapped:
                    mapped.append(canon)
                else:
                    fixes.append(f"dropped unknown service {name!r}")
            data["services"] = mapped
        for key in ("symptoms", "error_signals"):
            if isinstance(data.get(key), str):
                data[key] = [data[key]]
                fixes.append(f"{key}: wrapped string into list")
        return data, fixes

    return normalize


# ---------- agent ----------

class IncidentAgent:
    def __init__(self, llm: LLMClient, kb: KnowledgeBase, max_attempts: int = 3):
        self.llm = llm
        self.kb = kb
        self.max_attempts = max_attempts

    @classmethod
    def from_settings(cls, settings: Settings) -> "IncidentAgent":
        return cls(build_client(settings), KnowledgeBase.load(settings.knowledge_dir), settings.max_attempts)

    def run(self, incident_text: str) -> TriageOutcome:
        text = incident_text.strip()
        if not text:
            raise ValueError("incident text is empty")
        if len(text) > MAX_INPUT_CHARS:
            text = text[:MAX_INPUT_CHARS]

        started = time.monotonic()
        trace: dict = {"provider": self.llm.name, "stages": []}

        # Stage 1: parse free text into signals (LLM, grounded by regex; rule-based fallback)
        parsed, parse_trace = self._parse(text)
        trace["stages"].append(parse_trace)

        # Stage 2: gather auxiliary data (deterministic tools, no LLM)
        similar = self.kb.similar_incidents(text, parsed.services)
        context_services = list(dict.fromkeys([*parsed.services, *(s for x in similar for s in x.incident.services)]))
        system_context = self.kb.services_context(context_services)
        trace["stages"].append({
            "stage": "retrieve",
            "services": context_services,
            "similar_incidents": [{"id": s.incident.id, "score": s.score} for s in similar],
        })

        # Stage 3: generate structured triage with validation and repair
        try:
            result, attempts = call_structured(
                self.llm,
                stage="triage",
                system=prompts.TRIAGE_SYSTEM,
                user=prompts.triage_user(text, parsed, system_context, similar),
                model=TriageResult,
                max_attempts=self.max_attempts,
                normalize=normalize_triage,
                checks=(check_english,),
            )
        except StructuredOutputError as exc:
            trace["stages"].append({"stage": "triage", "attempts": [a.as_dict() for a in exc.attempts]})
            raise TriageError(str(exc), trace) from exc
        except LLMError as exc:
            trace["stages"].append({"stage": "triage", "error": str(exc)})
            raise TriageError(str(exc), trace) from exc

        trace["stages"].append({"stage": "triage", "attempts": [a.as_dict() for a in attempts]})
        trace["duration_ms"] = int((time.monotonic() - started) * 1000)
        return TriageOutcome(result, trace)

    def _parse(self, text: str) -> tuple[ParsedIncident, dict]:
        stage: dict = {"stage": "parse"}
        try:
            parsed, attempts = call_structured(
                self.llm,
                stage="parse",
                system=prompts.PARSE_SYSTEM,
                user=prompts.parse_user(text, self.kb.service_names),
                model=ParsedIncident,
                max_attempts=self.max_attempts,
                normalize=make_parsed_normalizer(self.kb.service_names),
            )
            stage.update(source="llm", attempts=[a.as_dict() for a in attempts])
        except (StructuredOutputError, LLMError) as exc:
            # Parsing is an enrichment step: degrade to regex extraction instead of failing the request.
            log.warning("parse stage failed, using rule-based fallback: %s", exc)
            parsed = rule_based_parse(text)
            stage.update(source="rules_fallback", error=str(exc))
            if isinstance(exc, StructuredOutputError):
                stage["attempts"] = [a.as_dict() for a in exc.attempts]
            return parsed, stage

        # Ground the LLM output: services explicitly named in the text are always kept.
        regex_services = find_services(text)
        merged = list(dict.fromkeys([*parsed.services, *regex_services]))
        if merged != parsed.services:
            stage["services_added_by_rules"] = [s for s in merged if s not in parsed.services]
            parsed = parsed.model_copy(update={"services": merged})
        return parsed, stage
