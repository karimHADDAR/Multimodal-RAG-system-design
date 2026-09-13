"""Evaluate the baseline retriever on a fixed DocVQA corpus.

Run: python -m evaluation.evaluate_retrieval --split train --limit 50
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

from datasets import load_dataset

from adaptive_rag.image_encoder import CLIPImageEncoder
from adaptive_rag.ingestion import evidence_from_docvqa
from adaptive_rag.retrieval import AdaptiveEvidenceRetriever, Strategy
from adaptive_rag.text_encoder import SentenceTransformerEncoder
from adaptive_rag.generation import QwenMLXGenerator, SmolVLMGenerator
from evaluation.qa_metrics import anls

SPLITS = {
    "train": "train[:800]",
    "validation": "train[800:1000]",
    "test": "test",
}


def question_text(example: dict) -> str:
    """Return the English question from the dataset's multilingual query field."""
    query = example["query"]
    return query.get("en", "") if isinstance(query, dict) else str(query)


def document_id(example: dict, index: int) -> str:
    """Match the document identifiers emitted by the current ingestion layer."""
    return str(example.get("id", example.get("questionId", example.get("question_id", index))))


def create_generator(name: str) -> object:
    """Create the selected locally runnable vision-language generator."""
    if name == "smolvlm":
        return SmolVLMGenerator()
    if name == "qwen-mlx":
        return QwenMLXGenerator()
    raise ValueError(f"Unsupported answer generator: {name}")


def evaluate(
    examples: list[dict],
    strategy: Strategy = "adaptive",
    image_encoder: object | None = None,
    text_encoder: object | None = None,
    generator: object | None = None,
    prediction_records: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    """Return retrieval metrics over questions whose source document is known."""
    retriever = AdaptiveEvidenceRetriever(
        evidence_from_docvqa(examples), image_encoder, text_encoder
    )
    hit_counts = {1: 0, 3: 0, 5: 0}
    reciprocal_rank_total = 0.0
    answer_score_total = 0.0
    routes: Counter[str] = Counter()
    started_at = time.perf_counter()

    for index, example in enumerate(examples):
        route, results = retriever.retrieve(question_text(example), strategy)
        routes[route] += 1
        target_document_id = document_id(example, index)
        rank = next(
            (position for position, result in enumerate(results, start=1)
             if result.evidence.document_id == target_document_id),
            None,
        )
        if rank is not None:
            reciprocal_rank_total += 1 / rank
            for cutoff in hit_counts:
                if rank <= cutoff:
                    hit_counts[cutoff] += 1
        prediction = None
        answer_score = None
        if generator is not None:
            prediction = generator.answer(question_text(example), results)
            answer_score = anls(prediction, example["answers"])
            answer_score_total += answer_score
        if prediction_records is not None:
            prediction_records.append({
                "example_id": document_id(example, index),
                "question": question_text(example),
                "accepted_answers": example["answers"],
                "route": route,
                "target_rank": rank,
                "retrieved_evidence_ids": [item.evidence.id for item in results],
                "retrieved_document_ids": list(dict.fromkeys(
                    item.evidence.document_id for item in results
                )),
                "prediction": prediction,
                "anls": answer_score,
            })

    count = len(examples)
    elapsed_seconds = time.perf_counter() - started_at
    return {
        "strategy": strategy,
        "examples": count,
        "recall_at": {str(cutoff): round(hits / count, 4) if count else 0.0
                      for cutoff, hits in hit_counts.items()},
        "mrr": round(reciprocal_rank_total / count, 4) if count else 0.0,
        "anls": round(answer_score_total / count, 4) if generator is not None and count else None,
        "route_counts": dict(routes),
        "average_retrieval_ms": round((elapsed_seconds / count) * 1000, 2) if count else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate baseline DocVQA retrieval.")
    parser.add_argument("--split", choices=tuple(SPLITS), default="validation")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument(
        "--strategy", choices=("text", "visual", "hybrid", "adaptive", "all"), default="all"
    )
    parser.add_argument("--image-encoder", choices=("baseline", "clip"), default="baseline")
    parser.add_argument("--text-encoder", choices=("baseline", "minilm"), default="baseline")
    parser.add_argument("--generate-answers", action="store_true")
    parser.add_argument(
        "--generator", choices=("smolvlm", "qwen-mlx"), default="smolvlm",
        help="Vision-language model used when --generate-answers is enabled.",
    )
    parser.add_argument(
        "--predictions-out",
        type=Path,
        help="Write one prediction and evidence record per example as JSON Lines.",
    )
    arguments = parser.parse_args()
    if arguments.limit < 1:
        parser.error("--limit must be at least 1")
    if arguments.generate_answers and arguments.strategy == "all":
        parser.error("Choose one --strategy when using --generate-answers.")

    split = SPLITS[arguments.split]
    dataset = load_dataset("nielsr/docvqa_1200_examples", split=split)
    examples = [dict(example) for example in dataset.select(range(min(arguments.limit, len(dataset))))]
    strategies = ("text", "visual", "hybrid", "adaptive") if arguments.strategy == "all" else (arguments.strategy,)
    image_encoder = CLIPImageEncoder() if arguments.image_encoder == "clip" else None
    text_encoder = SentenceTransformerEncoder() if arguments.text_encoder == "minilm" else None
    generator = create_generator(arguments.generator) if arguments.generate_answers else None
    prediction_records: list[dict[str, object]] | None = [] if arguments.predictions_out else None
    results = {
        strategy: evaluate(
            examples, strategy, image_encoder, text_encoder, generator, prediction_records
        )
        for strategy in strategies
    }
    if arguments.predictions_out:
        arguments.predictions_out.parent.mkdir(parents=True, exist_ok=True)
        with arguments.predictions_out.open("w", encoding="utf-8") as output_file:
            for record in prediction_records or []:
                output_file.write(json.dumps(record) + "\n")
    print(json.dumps({
        "split": arguments.split,
        "image_encoder": arguments.image_encoder,
        "text_encoder": arguments.text_encoder,
        "answer_generator": arguments.generator if arguments.generate_answers else None,
        "predictions_file": str(arguments.predictions_out) if arguments.predictions_out else None,
        "results": results,
    }, indent=2))


if __name__ == "__main__":
    main()