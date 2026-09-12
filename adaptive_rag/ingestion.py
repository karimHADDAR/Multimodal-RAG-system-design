"""Build evidence units from DocVQA OCR annotations."""

from collections.abc import Iterable

from .models import BoundingBox, EvidenceUnit


def _question_text(example: dict) -> str:
    query = example.get("query", example.get("question", ""))
    return str(query.get("en", "")) if isinstance(query, dict) else str(query)


def evidence_from_docvqa(examples: Iterable[dict]) -> list[EvidenceUnit]:
    """Create OCR and visual evidence units for each annotated document image.

    The visual unit retains the original PIL document image. Its ranking is still
    OCR-grounded until a real image embedding model is introduced.
    """
    units: list[EvidenceUnit] = []
    for index, example in enumerate(examples):
        words = example.get("words") or example.get("ocr_tokens") or []
        text = " ".join(words) if isinstance(words, list) else str(words)
        document_id = str(example.get("id", example.get("questionId", index)))
        image = example.get("image")
        width, height = image.size if image is not None else (1000, 1000)
        metadata = {"question": _question_text(example)}
        units.append(EvidenceUnit(
            id=f"{document_id}:ocr",
            document_id=document_id,
            page=1,
            type="text",
            text=text,
            bbox=BoundingBox(0, 0, width, height),
            metadata=metadata,
        ))
        units.append(EvidenceUnit(
            id=f"{document_id}:document",
            document_id=document_id,
            page=1,
            type="visual_region",
            text=text,
            bbox=BoundingBox(0, 0, width, height),
            image=image,
            metadata=metadata,
        ))
    return units


def load_docvqa_split(split: str = "train", limit: int = 100) -> list[EvidenceUnit]:
    """Load a limited DocVQA split only when the optional datasets extra is installed."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install the 'dataset' extra to download DocVQA.") from exc

    dataset = load_dataset("nielsr/docvqa_1200_examples", split=f"{split}[:{limit}]")
    return evidence_from_docvqa(dataset)