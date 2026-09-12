"""Vision-language answer generation grounded in retrieved evidence."""

from __future__ import annotations

from typing import Any

from .models import RankedEvidence


def evidence_text(evidence: list[RankedEvidence], limit: int = 3) -> str:
    """Format retrieved OCR evidence with IDs for traceable model context."""
    parts = []
    for item in evidence[:limit]:
        text = item.evidence.text.strip()
        if text:
            parts.append(f"[{item.evidence.id}] {text[:1500]}")
    return "\n\n".join(parts)


class SmolVLMGenerator:
    """Generate a short answer from retrieved document image and OCR evidence."""

    def __init__(self, model_name: str = "HuggingFaceTB/SmolVLM-256M-Instruct") -> None:
        try:
            import torch
            from transformers import AutoModelForVision2Seq, AutoProcessor
        except ImportError as exc:
            raise RuntimeError("Install torch and transformers to use VLM generation.") from exc

        self.torch = torch
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForVision2Seq.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def answer(self, question: str, evidence: list[RankedEvidence]) -> str:
        image = next((item.evidence.image for item in evidence if item.evidence.image is not None), None)
        if image is None:
            raise ValueError("Generation requires a retrieved document image.")
        if hasattr(image, "convert"):
            image = image.convert("RGB")

        prompt = (
            "Answer the question using only the document image and OCR evidence. "
            "Return only a concise answer. If unsupported, say 'Not found'.\n\n"
            f"Question: {question}\n\nOCR evidence:\n{evidence_text(evidence)}"
        )
        messages = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": prompt},
        ]}]
        rendered_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=rendered_prompt, images=[image], return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            generated = self.model.generate(**inputs, max_new_tokens=40, do_sample=False)
        new_tokens = generated[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()