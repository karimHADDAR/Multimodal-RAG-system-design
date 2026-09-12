"""Small local retrieval baselines and hybrid evidence fusion."""

from __future__ import annotations

import hashlib
import math
import re
from collections import defaultdict
from typing import Any, Literal

from .models import EvidenceUnit, RankedEvidence
from .routing import analyze_query, retrieval_budget

Strategy = Literal["text", "visual", "hybrid", "adaptive"]


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


def _embedding_similarity(query: Any, candidate: Any) -> float:
    """Compute similarity for either Torch tensors or lightweight test vectors."""
    if hasattr(query, "matmul"):
        return float(query.matmul(candidate))
    return sum(left * right for left, right in zip(query, candidate))


class AdaptiveEvidenceRetriever:
    """Routes questions and fuses text and visual-metadata candidate rankings.

    The visual channel intentionally uses visual labels/captions in this lightweight
    baseline. Replace `_image_query` with CLIP or SigLIP embeddings for production.
    """

    def __init__(
        self,
        evidence: list[EvidenceUnit],
        image_encoder: Any | None = None,
        text_encoder: Any | None = None,
    ) -> None:
        self.evidence = evidence
        self.text_encoder = text_encoder
        self.text_vectors = {
            unit.id: text_encoder.encode(unit.text) if text_encoder is not None else _vector(unit.text)
            for unit in evidence
        }
        self.image_encoder = image_encoder
        self.image_vectors = {
            unit.id: image_encoder.encode_image(unit.image)
            for unit in evidence if image_encoder is not None and unit.image is not None
        }

    def _search_text(self, question: str, limit: int) -> list[RankedEvidence]:
        query = self.text_encoder.encode(question) if self.text_encoder is not None else _vector(question)
        ranked = [RankedEvidence(
            unit, _embedding_similarity(query, self.text_vectors[unit.id]), "text"
        )
                  for unit in self.evidence if unit.text]
        return sorted(ranked, key=lambda item: item.score, reverse=True)[:limit]

    def _search_visual(self, question: str, limit: int) -> list[RankedEvidence]:
        if self.image_encoder is not None and self.image_vectors:
            query = self.image_encoder.encode_text(question)
            ranked = [RankedEvidence(
                unit, _embedding_similarity(query, self.image_vectors[unit.id]), "image"
            ) for unit in self.evidence if unit.id in self.image_vectors]
            return sorted(ranked, key=lambda item: item.score, reverse=True)[:limit]

        query = _vector(question)
        visual = [unit for unit in self.evidence if unit.type != "text" or unit.image_path]
        ranked = [RankedEvidence(unit, _cosine(query, _vector(
            f"{unit.type} {unit.metadata.get('caption', '')} {unit.text}")), "image")
            for unit in visual]
        return sorted(ranked, key=lambda item: item.score, reverse=True)[:limit]

    def retrieve(
        self, question: str, strategy: Strategy = "adaptive"
    ) -> tuple[str, list[RankedEvidence]]:
        """Retrieve evidence with a fixed baseline or adaptive routing."""
        route = analyze_query(question)
        if strategy == "text":
            return route, self._search_text(question, 5)
        if strategy == "visual":
            return route, self._search_visual(question, 5)

        text_limit, visual_limit = (3, 3) if strategy == "hybrid" else retrieval_budget(route)
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