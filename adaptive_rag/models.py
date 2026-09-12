"""Shared, serializable data models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


EvidenceType = Literal["text", "table", "figure", "caption", "visual_region"]
Route = Literal["text", "visual", "multimodal"]


@dataclass(frozen=True)
class BoundingBox:
    left: int
    top: int
    width: int
    height: int


@dataclass
class EvidenceUnit:
    id: str
    document_id: str
    page: int
    type: EvidenceType
    text: str = ""
    bbox: BoundingBox | None = None
    image_path: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass
class RankedEvidence:
    evidence: EvidenceUnit
    score: float
    source: Literal["text", "image", "hybrid"]