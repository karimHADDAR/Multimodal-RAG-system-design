"""Cheap, inspectable routing for evidence-aware retrieval."""

import re

from .models import Route

VISUAL_TERMS = {
    "chart", "graph", "figure", "image", "picture", "logo", "color",
    "largest", "smallest", "highest", "lowest", "shown", "visual",
}
TEXT_TERMS = {"who", "signed", "said", "paragraph", "states", "mention", "name"}


def analyze_query(question: str) -> Route:
    """Classify a question using explicit cues, suitable as an MVP baseline."""
    terms = set(re.findall(r"[a-z]+", question.lower()))
    visual_score = len(terms & VISUAL_TERMS)
    text_score = len(terms & TEXT_TERMS)
    if visual_score and text_score:
        return "multimodal"
    if visual_score:
        return "visual"
    if text_score:
        return "text"
    return "multimodal"


def retrieval_budget(route: Route) -> tuple[int, int]:
    """Return (text candidates, image candidates) for the selected route."""
    return {"text": (5, 1), "visual": (1, 5), "multimodal": (3, 3)}[route]