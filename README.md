# Adaptive Evidence RAG

A runnable MVP of evidence-aware multimodal document retrieval for DocVQA-style
questions. It routes a question to text, visual, or hybrid retrieval, retrieves
evidence units, and returns the ranked evidence with provenance.

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python main.py
```

Open `http://127.0.0.1:8000/docs` and send a request to `POST /ask`:

```json
{"question": "Which product had the highest revenue growth according to the chart?"}
```

For the evidence-inspection UI:

```bash
streamlit run demo/app.py
```

## Dataset

`adaptive_rag.ingestion.load_docvqa_split()` downloads a limited split from
`nielsr/docvqa_1200_examples` using Hugging Face Datasets. The starter retrieval
engine uses deterministic hashed vectors to remain runnable without model downloads.
Replace its text and visual vector methods with Sentence Transformers and CLIP/SigLIP
embeddings as the next experiment.

## Current Experimental Contract

The API exposes `route`, ranked evidence IDs, evidence types, pages, scores, and
metadata. This supports direct comparison of text-only, image-only, and hybrid
strategies before adding a VLM generator and answer metrics.