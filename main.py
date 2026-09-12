"""A simple DocVQA evidence-retrieval demo."""

import re

import streamlit as st
from datasets import load_dataset


VISUAL_WORDS = {"chart", "graph", "figure", "image", "photo", "logo", "color", "shown"}
TEXT_WORDS = {"who", "signed", "said", "name", "date", "address", "letter"}


def words(text: str) -> set[str]:
	"""Turn text into lowercase words for the simple retrieval baseline."""
	return set(re.findall(r"[a-z0-9]+", text.lower()))


def route_question(question: str) -> str:
	"""Choose the type of evidence the question is most likely to need."""
	question_words = words(question)
	if question_words & VISUAL_WORDS and question_words & TEXT_WORDS:
		return "multimodal"
	if question_words & VISUAL_WORDS:
		return "visual"
	if question_words & TEXT_WORDS:
		return "text"
	return "multimodal"


def english_question(example: dict) -> str:
	query = example["query"]
	return query.get("en", "") if isinstance(query, dict) else str(query)


def ocr_text(example: dict) -> str:
	return " ".join(example["words"])


@st.cache_data(show_spinner="Downloading DocVQA examples from Hugging Face...")
def load_examples(limit: int) -> list[dict]:
	"""Load a small, real subset of the Hugging Face DocVQA test data."""
	dataset = load_dataset("nielsr/docvqa_1200_examples", split=f"test[:{limit}]")
	return [dict(row) for row in dataset]


def retrieve(question: str, examples: list[dict], limit: int = 3) -> list[tuple[float, dict]]:
	"""Rank documents by shared words between the question and their OCR text."""
	query_words = words(question)
	scored = []
	for example in examples:
		document_words = words(ocr_text(example))
		score = len(query_words & document_words) / max(len(query_words), 1)
		scored.append((score, example))
	return sorted(scored, key=lambda item: item[0], reverse=True)[:limit]


st.set_page_config(page_title="Adaptive Evidence RAG", layout="wide")
st.title("Adaptive Evidence RAG")
st.caption("Real DocVQA documents, OCR evidence, and adaptive retrieval routing")

try:
	examples = load_examples(50)
except Exception as error:
	st.error(f"DocVQA could not be loaded: {error}")
	st.stop()

selected_index = st.selectbox(
	"Choose a benchmark question",
	range(len(examples)),
	format_func=lambda index: english_question(examples[index]),
)
selected = examples[selected_index]
question = st.text_input("Question", english_question(selected))
route = route_question(question)
results = retrieve(question, examples)

document_column, evidence_column = st.columns(2)
with document_column:
	st.subheader("Selected Document")
	st.image(selected["image"], use_container_width=True)
	st.write(f"Reference answer: **{' | '.join(selected['answers'])}**")

with evidence_column:
	st.subheader("Retrieved Evidence")
	st.write(f"Evidence route: **{route}**")
	for rank, (score, example) in enumerate(results, start=1):
		with st.expander(f"{rank}. Document {example['id']} (score {score:.2f})", expanded=rank == 1):
			st.write(f"Question: {english_question(example)}")
			st.write(f"OCR evidence: {ocr_text(example)[:1200]}")
			st.image(example["image"], use_container_width=True)

st.info("This is a retrieval baseline. The reference answer is from DocVQA; no answer-generation model is trained or connected yet.")
