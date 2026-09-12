"""DocVQA-compatible answer quality metrics."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable


def normalize_answer(answer: str) -> str:
    """Apply the case and whitespace normalization used before ANLS scoring."""
    text = unicodedata.normalize("NFKD", answer).lower()
    return " ".join(re.findall(r"\w+", text))


def levenshtein_distance(left: str, right: str) -> int:
    """Return character edit distance using memory proportional to one string."""
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for left_index, left_character in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_character in enumerate(right, start=1):
            substitution_cost = 0 if left_character == right_character else 1
            current.append(min(
                current[-1] + 1,
                previous[right_index] + 1,
                previous[right_index - 1] + substitution_cost,
            ))
        previous = current
    return previous[-1]


def anls(prediction: str, answers: Iterable[str], threshold: float = 0.5) -> float:
    """Return maximum ANLS against accepted answers, as used by DocVQA."""
    normalized_prediction = normalize_answer(prediction)
    similarities = []
    for answer in answers:
        normalized_answer = normalize_answer(answer)
        largest_length = max(len(normalized_prediction), len(normalized_answer), 1)
        similarity = 1 - levenshtein_distance(normalized_prediction, normalized_answer) / largest_length
        similarities.append(similarity if similarity >= threshold else 0.0)
    return max(similarities, default=0.0)