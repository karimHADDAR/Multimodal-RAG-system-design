"""Small local retrieval baselines and hybrid evidence fusion."""

import hashlib
import math
import re
from collections import defaultdict

from .models import EvidenceUnit, RankedEvidence
from .routing import analyze_query, retrieval_budget


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _vector(text: str, dimensions: int = 256) -> list[float]:
    vector = [0.0] * dimensions
    for token in _tokens(text):
        index = int(hashlib.sha256(token.encode()).hexdigest(), 16) % dimensions
        vector[index] += 1.0
    length = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / length for value in vector]


def _cosine(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


class AdaptiveEvidenceRetriever:
    """Routes questions and fuses text and visual-metadata candidate rankings.

    The visual channel intentionally uses visual labels/captions in this lightweight
    baseline. Replace `_image_query` with CLIP or SigLIP embeddings for production.
    """

    def __init__(self, evidence: list[EvidenceUnit]) -> None:
        self.evidence = evidence
        self.text_vectors = {unit.id: _vector(unit.text) for unit in evidence}

    def _search_text(self, question: str, limit: int) -> list[RankedEvidence]:
        query = _vector(question)
        ranked = [RankedEvidence(unit, _cosine(query, self.text_vectors[unit.id]), "text")
                  for unit in self.evidence if unit.text]
        return sorted(ranked, key=lambda item: item.score, reverse=True)[:limit]

    def _search_visual(self, question: str, limit: int) -> list[RankedEvidence]:
        query = _vector(question)
        visual = [unit for unit in self.evidence if unit.type != "text" or unit.image_path]
        ranked = [RankedEvidence(unit, _cosine(query, _vector(
            f"{unit.type} {unit.metadata.get('caption', '')} {unit.text}")), "image")
            for unit in visual]
        return sorted(ranked, key=lambda item: item.score, reverse=True)[:limit]

    def retrieve(self, question: str) -> tuple[str, list[RankedEvidence]]:
        route = analyze_query(question)
        text_limit, visual_limit = retrieval_budget(route)
        text_results = self._search_text(question, text_limit)
        image_results = self._search_visual(question, visual_limit)
        fused: dict[str, float] = defaultdict(float)
        sources: dict[str, set[str]] = defaultdict(set)
        units = {unit.id: unit for unit in self.evidence}
        for results, source in ((text_results, "text"), (image_results, "image")):
            for rank, item in enumerate(results, start=1):
                fused[item.evidence.id] += 1 / (60 + rank)
                sources[item.evidence.id].add(source)
        ranked = [RankedEvidence(
            units[unit_id], score,
            "hybrid" if len(sources[unit_id]) == 2 else next(iter(sources[unit_id])),
        ) for unit_id, score in fused.items()]
        return route, sorted(ranked, key=lambda item: item.score, reverse=True)