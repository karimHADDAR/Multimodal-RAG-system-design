"""FastAPI interface for evidence retrieval."""

from fastapi import FastAPI
from pydantic import BaseModel, Field

from adaptive_rag.retrieval import AdaptiveEvidenceRetriever
from adaptive_rag.sample_data import sample_evidence

app = FastAPI(title="Adaptive Evidence RAG", version="0.1.0")
retriever = AdaptiveEvidenceRetriever(sample_evidence())


class QuestionRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ask")
def ask(request: QuestionRequest) -> dict:
    route, results = retriever.retrieve(request.question)
    evidence = [{
        "id": result.evidence.id,
        "page": result.evidence.page,
        "type": result.evidence.type,
        "text": result.evidence.text,
        "score": round(result.score, 4),
        "source": result.source,
        "metadata": result.evidence.metadata,
    } for result in results]
    return {"question": request.question, "route": route, "evidence": evidence}