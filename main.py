"""A simple DocVQA evidence-retrieval demo."""

import re

import streamlit as st
from datasets import load_dataset

from adaptive_rag.generation import SmolVLMGenerator
from adaptive_rag.ingestion import evidence_from_docvqa
from adaptive_rag.retrieval import AdaptiveEvidenceRetriever


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


@st.cache_resource(show_spinner="Loading SmolVLM for answer generation...")
def load_generator() -> SmolVLMGenerator:
	return SmolVLMGenerator()


st.set_page_config(page_title="Adaptive Evidence RAG", layout="wide")
st.title("Adaptive Evidence RAG")
st.caption("Real DocVQA documents, OCR evidence, and adaptive retrieval routing")

try:
	examples = load_examples(50)
except Exception as error:
	st.error(f"DocVQA could not be loaded: {error}")
	st.stop()

if "selected_index" not in st.session_state:
	st.session_state.selected_index = 0

if "question" not in st.session_state:
	st.session_state.question = english_question(examples[st.session_state.selected_index])


def select_question(index: int) -> None:
	"""Synchronize the editable question with a selected benchmark item."""
	st.session_state.selected_index = index
	st.session_state.question_picker = index
	st.session_state.question = english_question(examples[index])


controls_column, navigation_column = st.columns([4, 1])
with controls_column:
	selected_index = st.selectbox(
		"Choose a benchmark question",
		range(len(examples)),
		key="question_picker",
		index=st.session_state.selected_index,
		format_func=lambda index: f"{index + 1}. {english_question(examples[index])}",
		on_change=lambda: select_question(st.session_state.question_picker),
	)

with navigation_column:
	st.write("Browse")
	previous, next_question = st.columns(2)
	with previous:
		st.button(
			"Previous",
			use_container_width=True,
			disabled=selected_index == 0,
			on_click=select_question,
			args=(selected_index - 1,),
		)
	with next_question:
		st.button(
			"Next",
			use_container_width=True,
			disabled=selected_index == len(examples) - 1,
			on_click=select_question,
			args=(selected_index + 1,),
		)

selected = examples[selected_index]
question = st.text_area(
	"Question",
	key="question",
	placeholder="Choose a benchmark question above or enter a question about the selected document.",
	height=80,
)
retriever = AdaptiveEvidenceRetriever(evidence_from_docvqa(examples))
route, results = retriever.retrieve(question)

document_column, evidence_column = st.columns(2)
with document_column:
	st.subheader("Selected Document")
	st.image(selected["image"], use_container_width=True)
	st.write(f"Reference answer: **{' | '.join(selected['answers'])}**")

with evidence_column:
	st.subheader("Retrieved Evidence")
	st.write(f"Evidence route: **{route}**")
	for rank, item in enumerate(results[:6], start=1):
		unit = item.evidence
		with st.expander(f"{rank}. {unit.id} | {unit.type} | {item.source}", expanded=rank == 1):
			st.write(unit.text[:1200] or "Document image evidence")
			if unit.image is not None:
				st.image(unit.image, use_container_width=True)
			st.caption(f"Score: {item.score:.4f}")

if st.button("Generate answer with SmolVLM", type="primary"):
	try:
		answer = load_generator().answer(question, results)
		st.subheader("Generated Answer")
		st.write(answer)
		st.caption("Evidence: " + ", ".join(item.evidence.id for item in results[:3]))
	except Exception as error:
		st.error(f"Answer generation failed: {error}")

st.info("The reference answer is from DocVQA. Generated answers are model predictions and should be evaluated against it.")
