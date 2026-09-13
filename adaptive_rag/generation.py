"""Vision-language answer generation grounded in retrieved evidence."""

from __future__ import annotations

from typing import Any

from .models import RankedEvidence


def top_document_evidence(evidence: list[RankedEvidence]) -> list[RankedEvidence]:
    """Keep only evidence from the highest-ranked retrieved document."""
    if not evidence:
        return []
    document_id = evidence[0].evidence.document_id
    return [item for item in evidence if item.evidence.document_id == document_id]


def evidence_text(evidence: list[RankedEvidence], limit: int = 3) -> str:
    """Format retrieved OCR evidence with IDs for traceable model context."""
    parts = []
    for item in evidence[:limit]:
        text = item.evidence.text.strip()
        if text:
            parts.append(f"[{item.evidence.id}] {text[:1500]}")
    return "\n\n".join(parts)


def answer_prompt(question: str, evidence: list[RankedEvidence]) -> str:
    """Build a short-answer prompt aligned with DocVQA answer annotations."""
    return (
        "You are answering a document question. Use only the supplied document image and OCR. "
        "Return the exact shortest answer phrase, not an explanation, full sentence, quote, "
        "or evidence ID. Do not copy unrelated text. If the answer cannot be found, return: Not found.\n\n"
        f"Question: {question}\n\nOCR evidence:\n{evidence_text(evidence)}\n\nAnswer:"
    )


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
        document_evidence = top_document_evidence(evidence)
        image = next((item.evidence.image for item in document_evidence if item.evidence.image is not None), None)
        if image is None:
            raise ValueError("Generation requires a retrieved document image.")
        if hasattr(image, "convert"):
            image = image.convert("RGB")

        prompt = answer_prompt(question, document_evidence)
        messages = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": prompt},
        ]}]
        rendered_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=rendered_prompt, images=[image], return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            generated = self.model.generate(**inputs, max_new_tokens=16, do_sample=False)
        new_tokens = generated[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()


class QwenMLXGenerator:
    """Run the 4-bit Qwen2.5-VL 3B model through MLX on Apple Silicon."""

    def __init__(
        self, model_name: str = "mlx-community/Qwen2.5-VL-3B-Instruct-4bit"
    ) -> None:
        try:
            from transformers import Qwen2Tokenizer
            from mlx_vlm import generate, load
            from mlx_vlm.prompt_utils import apply_chat_template
            from mlx_vlm.utils import load_config
        except ImportError as exc:
            raise RuntimeError("Install mlx-vlm to use Qwen MLX generation.") from exc

        self.generate = generate
        self.apply_chat_template = apply_chat_template
        if not hasattr(Qwen2Tokenizer, "vocab"):
            Qwen2Tokenizer.vocab = property(lambda tokenizer: tokenizer.get_vocab())
        self.model, self.processor = load(model_name, use_fast=False)
        self.config = load_config(model_name)

    def answer(self, question: str, evidence: list[RankedEvidence]) -> str:
        document_evidence = top_document_evidence(evidence)
        image = next((item.evidence.image for item in document_evidence if item.evidence.image is not None), None)
        if image is None:
            raise ValueError("Generation requires a retrieved document image.")
        if hasattr(image, "convert"):
            image = image.convert("RGB")
        if hasattr(image, "thumbnail"):
            image.thumbnail((512, 512))

        prompt = self.apply_chat_template(
            self.processor, self.config, answer_prompt(question, document_evidence), num_images=1
        )
        return self.generate(
            self.model, self.processor, prompt, [image], max_tokens=16,
            temperature=0.0, verbose=False,
        ).strip()