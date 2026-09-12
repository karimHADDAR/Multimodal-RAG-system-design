"""Pretrained CLIP encoder for image-pixel retrieval."""

from __future__ import annotations

from typing import Any


class CLIPImageEncoder:
    """Encode text questions and PIL images into the same CLIP vector space."""

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32") -> None:
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as exc:
            raise RuntimeError(
                "Install torch and transformers to use CLIP image retrieval."
            ) from exc

        self.torch = torch
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def encode_text(self, text: str) -> Any:
        inputs = self.processor(text=[text], return_tensors="pt", padding=True).to(self.device)
        with self.torch.no_grad():
            features = self.model.get_text_features(**inputs)
        return self._normalize(features)[0]

    def encode_image(self, image: Any) -> Any:
        inputs = self.processor(images=[image], return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            features = self.model.get_image_features(**inputs)
        return self._normalize(features)[0]

    def _normalize(self, features: Any) -> Any:
        return features / features.norm(dim=-1, keepdim=True)