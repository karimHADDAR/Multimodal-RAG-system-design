"""Pretrained semantic text encoder for document retrieval."""

from __future__ import annotations

from typing import Any


class SentenceTransformerEncoder:
    """Encode questions and OCR text with a compact pretrained MiniLM model."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Install sentence-transformers to use semantic text retrieval."
            ) from exc
        self.model = SentenceTransformer(model_name)

    def encode(self, text: str) -> Any:
        return self.model.encode(text, normalize_embeddings=True)