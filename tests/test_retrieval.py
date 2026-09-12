from adaptive_rag.ingestion import evidence_from_docvqa
from adaptive_rag.generation import evidence_text
from adaptive_rag.retrieval import AdaptiveEvidenceRetriever
from adaptive_rag.routing import analyze_query, retrieval_budget
from adaptive_rag.sample_data import sample_evidence
from evaluation.qa_metrics import anls, normalize_answer


class FakeImageEncoder:
    def encode_text(self, text: str) -> list[float]:
        return [1.0, 0.0] if "chart" in text else [0.0, 1.0]

    def encode_image(self, image: object) -> list[float]:
        return image


class FakeTextEncoder:
    def encode(self, text: str) -> list[float]:
        return [1.0, 0.0] if "Aurora" in text else [0.0, 1.0]


def test_visual_question_uses_visual_route() -> None:
    assert analyze_query("What is the highest value shown in the chart?") == "visual"
    assert retrieval_budget("visual") == (1, 5)


def test_hybrid_retrieval_returns_evidence() -> None:
    route, results = AdaptiveEvidenceRetriever(sample_evidence()).retrieve(
        "Which product had the highest revenue growth according to the chart?"
    )
    assert route == "visual"
    assert results
    assert results[0].evidence.document_id == "report-2023"


def test_fixed_text_strategy_returns_text_evidence() -> None:
    _, results = AdaptiveEvidenceRetriever(sample_evidence()).retrieve(
        "What revenue did Aurora have?", strategy="text"
    )
    assert results
    assert all(result.source == "text" for result in results)


def test_docvqa_ingestion_creates_text_and_image_evidence() -> None:
    image = type("Image", (), {"size": (640, 480)})()
    units = evidence_from_docvqa([{
        "id": "doc-1",
        "image": image,
        "query": {"en": "What is shown?"},
        "words": ["Annual", "report"],
    }])
    assert [unit.type for unit in units] == ["text", "visual_region"]
    assert units[1].image is image
    assert units[1].bbox is not None and units[1].bbox.width == 640


def test_visual_search_uses_image_encoder_vectors() -> None:
    evidence = sample_evidence()
    evidence[1].image = [0.0, 1.0]
    evidence[2].image = [1.0, 0.0]
    _, results = AdaptiveEvidenceRetriever(evidence, FakeImageEncoder()).retrieve(
        "What does the chart show?", strategy="visual"
    )
    assert results[0].evidence.id == "report-2023:figure-2"


def test_text_search_uses_semantic_encoder_vectors() -> None:
    evidence = sample_evidence()
    _, results = AdaptiveEvidenceRetriever(evidence, text_encoder=FakeTextEncoder()).retrieve(
        "Aurora growth", strategy="text"
    )
    assert results[0].evidence.id == "report-2023:paragraph"


def test_evidence_text_preserves_evidence_identifiers() -> None:
    _, results = AdaptiveEvidenceRetriever(sample_evidence()).retrieve("Aurora revenue")
    assert "[report-2023:" in evidence_text(results)


def test_anls_uses_best_accepted_answer_and_threshold() -> None:
    assert normalize_answer("Revenue: $15.8M!") == "revenue 15 8m"
    assert anls("15.8m", ["12.4m", "15.8M"]) == 1.0
    assert anls("unrelated", ["15.8m"]) == 0.0