from adaptive_rag.retrieval import AdaptiveEvidenceRetriever
from adaptive_rag.routing import analyze_query, retrieval_budget
from adaptive_rag.sample_data import sample_evidence


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