"""Getting a validated object out of an LLM: extract -> auto-correct -> validate -> repair loop.

Failure classes handled:
  * not JSON at all / JSON wrapped in prose or markdown fences / trailing commas
  * JSON of the wrong shape (missing fields, wrong types, values outside enums, too many items)
  * text in an unexpected language
Recovery strategy:
  1. cheap deterministic auto-correction of known harmless deviations (no extra LLM call);
  2. otherwise, re-ask the model with the previous answer and the exact list of validation errors
     (a "refined prompt"), up to `max_attempts` in total;
  3. if still invalid, raise StructuredOutputError with the full attempt history.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from typing import Callable, TypeVar

from pydantic import BaseModel, ValidationError

from .llm import LLMClient, Message

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)
Normalizer = Callable[[dict], tuple[dict, list[str]]]
Check = Callable[[BaseModel], list[str]]


class JSONExtractionError(ValueError):
    pass


class OutputRejected(ValueError):
    def __init__(self, errors: list[str]):
        super().__init__("; ".join(errors))
        self.errors = errors


@dataclass
class Attempt:
    number: int
    ok: bool
    errors: list[str] = field(default_factory=list)
    corrections: list[str] = field(default_factory=list)
    raw_preview: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


class StructuredOutputError(Exception):
    def __init__(self, message: str, attempts: list[Attempt]):
        super().__init__(message)
        self.attempts = attempts


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_TRAILING_COMMA = re.compile(r",\s*([}\]])")


def extract_json(text: str) -> object:
    """Find the JSON object in a model answer, tolerating fences, surrounding prose and trailing commas."""
    if not text or not text.strip():
        raise JSONExtractionError("empty response")
    candidate = text.strip()
    fenced = _FENCE.search(candidate)
    if fenced:
        candidate = fenced.group(1).strip()
    start = candidate.find("{")
    if start == -1:
        raise JSONExtractionError("no JSON object found in the response")
    candidate = candidate[start:]
    decoder = json.JSONDecoder()
    for attempt in (candidate, _TRAILING_COMMA.sub(r"\1", candidate)):
        try:
            obj, _ = decoder.raw_decode(attempt)
            return obj
        except json.JSONDecodeError as exc:
            last_error = exc
    raise JSONExtractionError(f"invalid JSON: {last_error.msg} at line {last_error.lineno} col {last_error.colno}")


def non_latin_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    non_latin = sum(1 for c in letters if not ("a" <= c.lower() <= "z"))
    return non_latin / len(letters)


def _format_validation_error(exc: ValidationError) -> list[str]:
    errors = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err["loc"]) or "<root>"
        errors.append(f"{loc}: {err['msg']}")
    return errors


def repair_instruction(errors: list[str]) -> str:
    bullet_list = "\n".join(f"- {e}" for e in errors)
    return (
        "Your previous response could not be accepted because of these problems:\n"
        f"{bullet_list}\n\n"
        "Return the corrected answer as a single JSON object that strictly follows the required schema. "
        "All text values must be in English. Output only the JSON object: no markdown fences, no comments, "
        "no text before or after it."
    )


def call_structured(
    llm: LLMClient,
    *,
    stage: str,
    system: str,
    user: str,
    model: type[T],
    max_attempts: int,
    normalize: Normalizer | None = None,
    checks: tuple[Check, ...] = (),
) -> tuple[T, list[Attempt]]:
    base: list[Message] = [{"role": "user", "content": user}]
    messages = base
    attempts: list[Attempt] = []

    for number in range(1, max_attempts + 1):
        raw = llm.complete(stage, system, messages)
        corrections: list[str] = []
        try:
            data = extract_json(raw)
            if not isinstance(data, dict):
                raise OutputRejected(["the top-level JSON value must be an object"])
            if normalize:
                data, corrections = normalize(data)
            obj = model.model_validate(data)
            problems = [p for check in checks for p in check(obj)]
            if problems:
                raise OutputRejected(problems)
            attempts.append(Attempt(number, True, [], corrections, raw[:300]))
            if corrections:
                log.info("stage=%s auto-corrected: %s", stage, corrections)
            return obj, attempts
        except JSONExtractionError as exc:
            errors = [str(exc)]
        except ValidationError as exc:
            errors = _format_validation_error(exc)
        except OutputRejected as exc:
            errors = exc.errors

        attempts.append(Attempt(number, False, errors, corrections, raw[:300]))
        log.warning("stage=%s attempt %d/%d rejected: %s", stage, number, max_attempts, errors)
        # Keep only the latest failed answer in context: short prompt, focused correction.
        messages = [
            *base,
            {"role": "assistant", "content": raw.strip() or "(empty response)"},
            {"role": "user", "content": repair_instruction(errors)},
        ]

    raise StructuredOutputError(f"stage {stage!r}: no valid output after {max_attempts} attempts", attempts)
