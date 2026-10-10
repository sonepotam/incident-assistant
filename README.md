# AI Incident Assistant

A small triage agent for on-call engineers. Given a free-text incident description it returns a
machine-readable triage: category, summary, severity and up to 3 root-cause hypotheses with concrete
next steps. The LLM is used as a controlled component inside a three-stage pipeline, with schema
validation, auto-correction and regeneration on bad output.

Interfaces: CLI and HTTP API (FastAPI). Providers: Anthropic, OpenAI (or any OpenAI-compatible
endpoint, e.g. a local Ollama model), plus an offline `mock` provider for running without keys.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                     # put ANTHROPIC_API_KEY or OPENAI_API_KEY there

# CLI
python -m incident_assistant -f examples/01_paygate_timeouts.txt
python -m incident_assistant "auth-service returns 401, invalid token signatures" --trace
cat examples/02_reporting_db_load.txt | python -m incident_assistant

# HTTP API
uvicorn incident_assistant.api:app --port 8000
curl -s localhost:8000/triage -H 'Content-Type: application/json' \
     -d '{"text": "Customers cannot pay by card, payment-service shows PayGate timeouts since 12:05 UTC"}'
curl -s 'localhost:8000/triage?debug=true' ...     # result + stage trace (attempts, retrieval)
curl -s localhost:8000/schema                      # JSON schema of the result

# Tests (no API key needed)
pytest -q
```

Without any key `LLM_PROVIDER=auto` falls back to the offline keyword-based `mock`, so the whole
pipeline can be exercised; real triage quality obviously requires a real model.
All configuration is in environment variables, see `.env.example`.

Output example: a real answer from `claude-sonnet-5-5` for `examples/04_auth_after_deploy.txt`
(the first of 3 hypotheses shown; all six full answers are in [`results/`](results/)):

```json
{
  "category": "User authentication errors",
  "summary": "Since the auth-service release at 14:10 UTC, tokens are failing signature verification, and the api-gateway 401 rate rose from 0.2% to 30%. Web and mobile users are being logged out and cannot log back in, so a core access flow is broken for many customers.",
  "severity": "high",
  "hypotheses": [
    {
      "title": "Signing key or key configuration changed in the auth-service release",
      "reasoning": "The failures started right after the 14:10 UTC release, and the logs say 'token signature verification failed'. This points to tokens being signed with a different key, algorithm or key ID than the one used to verify them. That would invalidate existing sessions (the logouts) and could also break newly issued tokens (the failed re-logins).",
      "next_steps": [
        "Compare the auth-service signing key, key ID and algorithm configuration between the previous and the new release, and check the deploy diff for secret or config changes.",
        "Query ELK for auth-service logs from 14:10 UTC onward and check whether newly issued tokens fail verification immediately, or only tokens issued before the release.",
        "Decode the header of a failing token (without logging its contents) and compare its kid/alg with the keys currently published or configured for verification."
      ]
    }
  ]
}
```

CLI exit codes: `0` ok, `1` bad input/config/provider error, `2` the model did not produce a valid
answer within the retry budget (the stage trace is printed to stderr). The API maps these to
`422`, `503` and `502`.

## How the agent works

```
incident text
   │
   ▼
1. PARSE (LLM)        -> ParsedIncident {symptoms, services, error_signals, start_time, user_impact}
   │                     validated; unknown service names dropped; services literally mentioned in
   │                     the text are always added (regex grounding); if the LLM stage fails
   │                     completely -> rule-based parser (degraded, but the request continues)
   ▼
2. RETRIEVE (tools)   -> similar past incidents (IDF-weighted token overlap + shared services)
   │                     + description of the relevant services and their dependencies/risks
   ▼
3. TRIAGE (LLM)       -> TriageResult, through the validation / recovery loop below
```

Stages are explicit functions in `agent.py` with their own prompts (`prompts.py`) and their own
output models (`schema.py`). Stage 2 is deterministic: the LLM never "remembers" the system or past
incidents, it receives them as data. Categories and severity are closed vocabularies defined in
code (the category list includes an explicit `"Other"` so the model is not pushed to force a match),
and the severity scale is defined in the prompt so it is consistent between runs.
Every run produces a trace (per-stage attempts, errors, auto-corrections, retrieved incidents).

**Handling model errors** (`structured.py`), applied to every LLM stage:

1. *Extraction*: JSON is extracted from markdown fences / surrounding prose, trailing commas are tolerated.
2. *Auto-correction* without an extra LLM call for harmless deviations: `"High"`/`"critical"` →
   `"high"`, near-miss category names (`"external payment provider issues"`) → canonical name,
   `nextSteps` → `next_steps`, >3 hypotheses or steps → trimmed, result wrapped in an extra key → unwrapped.
3. *Validation*: pydantic schema (types, enums, 1–3 hypotheses, 2–3 steps each, non-empty strings)
   plus a language check (output must be English; >15% non-Latin letters is rejected).
4. *Regeneration with a refined prompt*: on failure the model gets its previous answer plus the
   exact list of validation errors and is asked to return corrected JSON only. Up to
   `AGENT_MAX_ATTEMPTS` (default 3) attempts per stage.
5. *Give-up policy*: parse stage → rule-based fallback; triage stage → explicit error with the full
   trace (no fabricated "best guess" answer). Transport errors are retried by the SDKs (2 retries).

Prompt-injection hygiene: the incident text is wrapped in `<incident>` tags and declared as data.

## Project layout

```
incident_assistant/
  agent.py        orchestrator, auto-correction, language check
  structured.py   JSON extraction + validate/repair loop (provider-agnostic)
  prompts.py      stage prompts
  schema.py       output models, categories, severity scale
  knowledge.py    system description + past incidents, retrieval
  rules.py        regex signal extraction (grounding + fallback parser)
  llm.py          Anthropic / OpenAI adapters behind one small protocol
  mock_llm.py     offline provider
  cli.py, api.py  interfaces
  data/           system.json, incidents.json (from the assignment)
examples/         test incidents
results/          real model answers for examples/
tests/            pytest suite, no network
```

## Testing

**Automated (`pytest`, no network).** A scripted fake LLM returns pre-defined answers per stage, so
each failure mode is reproduced deterministically:

| Test | Model answer | Expected behaviour |
|---|---|---|
| invalid JSON | `"Sure! The category is provider issue."` | attempt 1 rejected, repair prompt contains the error, attempt 2 accepted |
| wrong format | bad category, `severity: "urgent"`, empty hypotheses | all three errors reported back to the model, regenerated |
| wrong language | summary/reasoning in Russian | rejected by language check, regenerated in English |
| harmless deviation | fenced JSON with `"High"` | auto-corrected, **no** extra LLM call |
| never valid | 3 bad answers | `TriageError` with 3 attempts in trace; API → 502 |
| parse stage broken | garbage for parse | rule-based fallback, pipeline still completes |
| hallucinated service | `"fraud-service"` | dropped; services mentioned in text added |
| provider down | `LLMError` | clean error, no crash |

Plus retrieval tests (INC-102 / INC-104 found for matching descriptions), API tests, and an
end-to-end run through the offline mock.

**Manual, with a real model** (`examples/`, model `claude-sonnet-5-5`, answers saved in [`results/`](results/)).
The "Expected" column was written before the first run and is left unchanged.

| Input | Expected | Got |
|---|---|---|
| `01` PayGate timeouts since 12:05 UTC, other services normal | External payment provider issue, high; PayGate-side degradation first | ✅ as expected; hypotheses: PayGate outage → egress/DNS/TLS path → payment-service DB or HTTP pool (marked "less likely") |
| `02` slow `/payments/create`, 504 at gateway, month-end export at 09:35, payment DB CPU 95% | DB degradation caused by reporting, high; `pg_stat_activity`, correlate with export job | ✅ as expected; also uses the 09:35 → 09:40 timing and checks the export hits the payment instance, not a replica |
| `03` SMS one-time codes not delivered, SMS provider 503 for ~40%, e-mail fine | Notification delivery issue, medium–high (blocks payment confirmation) | ⚠️ category ✅, severity **medium**: inside the expected range, but chosen for the wrong reason, see "What the evaluation changed"; hypotheses: provider outage → throttling/quota → own config change |
| `04` 401 rate 0.2% → 30% after auth-service release, "token signature verification failed" | User authentication errors, high; signing key/config mismatch from the release | ✅ as expected; also explains the partial 30% by an inconsistent rollout and checks the gateway JWKS cache |
| `05` duplicate invoices after billing-service deploy, no errors | **Other** (must not be forced into a known category), medium–high (financial impact) | ✅ Other, high (first run before the scale fix: medium); hypotheses: broken idempotency → job running on two instances → retry/re-processing |
| `06` same as 01, written in Russian | same as 01, answer in English | ✅ same category/severity, all text in English |

What I checked on each answer: category from the closed list, severity against the scale, hypotheses
grounded in facts present in the report (no invented metric values or log lines), next steps that
name concrete services and data sources of this system, and that hypotheses are distinguishable.

**What the evaluation changed.** Two problems were found by reading real answers and fixed in the prompt:

- *Severity scale* (`schema.py`). On `05` the model first chose `medium` while writing "risk of
  incorrect balances" in its own summary: the old scale tied `high` to "many customers". Money-data
  integrity is now `high` regardless of how many customers are affected (3 of 3 re-runs → `high`).
  On `03` the scale itself pointed to `medium`: its `medium` example said "confirmations not
  delivered", which does not distinguish an informational e-mail from a one-time code that blocks a
  payment. The scale now separates the two; `03` was not re-run after this change.
- *Distinct hypotheses* (`prompts.py`). Runs of `05` produced two hypotheses that were the same cause
  in different words and repeated the same check. The prompt now requires different root causes,
  each with a step that rules it in or out against the others.

Caveat: both fixes were made while looking at these six examples, so they are tuned on the same
data they are evaluated on. Their wording is general (money-data integrity, blocking steps), but a
proper check needs held-out incidents.

**Run-to-run variance.** The Anthropic SDK 1.x `messages.create` has no `temperature` parameter, so
answers to the same input differ. Over 3 runs of `05`: category and severity identical in 3/3, the
first two hypotheses the same in 3/3, the third hypothesis varied (the useful "one invoice sent twice
vs two invoices created" alternative appeared in 1 of 3). The machine-consumed fields are stable;
hypothesis wording and ordering of the weakest one are not.

**Known limitations seen in the answers.**
- The model assumes plausible schema and metric names it was not given (`invoices` table,
  `created_at`, `(merchant, period)`, HTTP client pool metrics). They are phrased as things to check,
  not as facts, but they are not grounded in the input. The fix is to pass the real schema / dashboard
  catalogue as context, not to forbid concrete steps.
- The third hypothesis is often a weak, explicitly "less likely" one (e.g. payment DB for `01`). I kept
  this: it is ranked honestly and cheap to rule out.

## Trade-offs

Simplified to fit ~3 hours:
- Retrieval is lexical over 4 incidents; no embeddings or vector store (the interface allows swapping).
- The "agent" is a fixed pipeline, not a planner choosing tools: predictable, cheap, easy to test.
  Triage of an incident report doesn't need open-ended tool selection at this stage.
- Language detection is a script-based heuristic (non-Latin letter ratio), not a language-ID model.
- No real integrations (ELK, metrics, provider status pages); next steps tell the engineer what to check.
- Synchronous, no caching, no auth on the API.
- No evaluation set with scored model outputs; behaviour with a real model is checked manually on
  6 examples (single runs, plus 3 runs of `05`).

With more time:
- Provider-native structured output (tool use / JSON schema mode) as the first line of defence,
  keeping the current validation as the safety net.
- Tool-using agent stage: query ELK / Prometheus / status pages for the hypotheses and attach evidence.
- Evaluation harness: a labelled set of incidents, metrics for category accuracy and severity agreement,
  run on every prompt change; LLM-as-judge for hypothesis quality.
- Embedding retrieval over a real incident history with postmortems and resolutions.
- Observability: token/latency/retry metrics per stage, structured logs, request IDs.
- Calibrated severity using business signals (affected payment volume, error rate), not text only.
