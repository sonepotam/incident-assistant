"""Auxiliary data: system description and past incidents, plus simple lexical retrieval.

Retrieval is deliberately deterministic and dependency-free (IDF-weighted token overlap + service
overlap). With 4 past incidents embeddings would be overkill; the interface stays the same if the
corpus grows and is swapped for a vector store.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

_STOPWORDS = {
    "the", "and", "for", "with", "from", "that", "this", "are", "was", "were", "have", "has", "not",
    "but", "some", "many", "other", "our", "they", "their", "see", "seen", "see", "into", "when",
    "who", "what", "about", "around", "since", "after", "before", "users", "customers", "logs",
    "service", "services", "issue", "started",
}


def tokenize(text: str) -> set[str]:
    tokens = set()
    for tok in re.findall(r"[a-z0-9]+(?:[-/][a-z0-9]+)*", text.lower()):
        if len(tok) < 3 or tok in _STOPWORDS:
            continue
        if len(tok) > 4 and tok.endswith("s") and not tok.endswith("ss"):
            tok = tok[:-1]  # crude plural folding: timeouts -> timeout, errors -> error
        tokens.add(tok)
    return tokens


@dataclass(frozen=True)
class PastIncident:
    id: str
    category: str
    services: tuple[str, ...]
    text: str


@dataclass(frozen=True)
class ScoredIncident:
    incident: PastIncident
    score: float


class KnowledgeBase:
    def __init__(self, system: dict, incidents: list[PastIncident]):
        self.system = system
        self.incidents = incidents
        self._services = {s["name"]: s for s in system["services"]}
        self._incident_tokens = {inc.id: tokenize(inc.text) for inc in incidents}
        n = len(incidents)
        df: dict[str, int] = {}
        for toks in self._incident_tokens.values():
            for t in toks:
                df[t] = df.get(t, 0) + 1
        self._idf = {t: math.log(1 + n / c) for t, c in df.items()}

    @classmethod
    def load(cls, directory: Path) -> "KnowledgeBase":
        system = json.loads((directory / "system.json").read_text(encoding="utf-8"))
        raw = json.loads((directory / "incidents.json").read_text(encoding="utf-8"))
        incidents = [
            PastIncident(id=i["id"], category=i["category"], services=tuple(i["services"]), text=i["text"])
            for i in raw
        ]
        return cls(system, incidents)

    @property
    def service_names(self) -> list[str]:
        return list(self._services)

    '''
    def similar_incidents(self, text: str, services: list[str], k: int = 2) -> list[ScoredIncident]:
        query = tokenize(text)
        scored = []
        for inc in self.incidents:
            lexical = sum(self._idf.get(t, 0.0) for t in query & self._incident_tokens[inc.id])
            service_bonus = 0.5 * len(set(services) & set(inc.services))
            score = lexical + service_bonus
            if score > 0:
                scored.append(ScoredIncident(inc, round(score, 2)))
        scored.sort(key=lambda s: s.score, reverse=True)
        return scored[:k]
    '''

    def similar_incidents(
        self,
        text: str,
        services: list[str],
        k: int = 2,
        min_score: float = 2.0,
        min_ratio: float = 0.3,
    ) -> list[ScoredIncident]:
        """Top-k past incidents, with noise filtered out.

        min_score: absolute floor; one shared token (max IDF here ~1.61) is not evidence of similarity.
        min_ratio: matches scoring below this share of the best match are dropped.
        """
        query = tokenize(text)
        scored = []
        for inc in self.incidents:
            lexical = sum(self._idf.get(t, 0.0) for t in query & self._incident_tokens[inc.id])
            service_bonus = 0.5 * len(set(services) & set(inc.services))
            score = lexical + service_bonus
            if score >= min_score:
                scored.append(ScoredIncident(inc, round(score, 2)))
        if not scored:
            return []
        scored.sort(key=lambda s: s.score, reverse=True)
        cutoff = scored[0].score * min_ratio
        return [s for s in scored[:k] if s.score >= cutoff]

    def services_context(self, names: list[str]) -> str:
        """Description of the given services; whole system if none are known."""
        selected = [self._services[n] for n in names if n in self._services] or list(self._services.values())
        lines = [f"Platform: {self.system['platform']}", "General notes:"]
        lines += [f"- {note}" for note in self.system["general_notes"]]
        lines.append("Relevant services:")
        for svc in selected:
            lines.append(f"- {svc['name']}: {svc['description']}")
            if svc["dependencies"]:
                lines.append(f"  depends on: {', '.join(svc['dependencies'])}")
            for risk in svc["known_risks"]:
                lines.append(f"  known risk: {risk}")
        return "\n".join(lines)
