"""Interactive retrieval inspection UI."""

import streamlit as st

from adaptive_rag.retrieval import AdaptiveEvidenceRetriever
from adaptive_rag.sample_data import sample_evidence

st.set_page_config(page_title="Adaptive Evidence RAG", layout="wide")
st.title("Adaptive Evidence RAG")
question = st.text_input("Question", "Which product had the highest revenue growth?")
if question:
    route, evidence = AdaptiveEvidenceRetriever(sample_evidence()).retrieve(question)
    st.caption(f"Evidence route: {route}")
    for rank, item in enumerate(evidence, start=1):
        st.subheader(f"{rank}. Page {item.evidence.page} - {item.evidence.type}")
        st.write(item.evidence.text or item.evidence.metadata.get("caption", "Visual region"))
        st.caption(f"{item.source} score: {item.score:.4f}")