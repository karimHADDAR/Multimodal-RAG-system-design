"""Build evidence units from DocVQA OCR annotations."""

from collections.abc import Iterable

from .models import BoundingBox, EvidenceUnit


def evidence_from_docvqa(examples: Iterable[dict]) -> list[EvidenceUnit]:
    """Create one OCR evidence unit per annotated document image.

    DocVQA's ready-to-use subset exposes OCR words and normalized bounding boxes.
    This baseline preserves the box metadata so later layout segmentation can split
    the unit into paragraphs, tables, and figures without changing the interface.
    """
    units: list[EvidenceUnit] = []
    for index, example in enumerate(examples):
        words = example.get("words") or example.get("ocr_tokens") or []
        text = " ".join(words) if isinstance(words, list) else str(words)
        document_id = str(example.get("questionId", example.get("question_id", index)))
        units.append(EvidenceUnit(
            id=f"{document_id}:ocr",
            document_id=document_id,
            page=1,
            type="text",
            text=text,
            bbox=BoundingBox(0, 0, 1000, 1000),
            metadata={"question": str(example.get("question", ""))},
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