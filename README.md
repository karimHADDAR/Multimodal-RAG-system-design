# Adaptive Evidence RAG

A simple document-retrieval project using the Hugging Face DocVQA dataset.

The app loads real document images, questions, ground-truth answers, and OCR
words. It classifies a question as text, visual, or multimodal, then retrieves
the documents whose OCR evidence is most relevant.

## Start the App

```bash
source .venv/bin/activate
python main.py
```

Open the address printed in the terminal, normally `http://localhost:8501`.
The first startup downloads and caches a small DocVQA subset. Later starts use
the local cache.

## What You See

1. Choose a real DocVQA benchmark question.
2. View its document image and reference answer.
3. Inspect the selected evidence route.
4. Inspect the top retrieved documents with OCR text and images.

## Evaluate the Baseline

Compare all four retrieval strategies on the held-out validation partition:

```bash
python -m evaluation.evaluate_retrieval --split validation --limit 50 --strategy all
```

The command reports `Recall@1`, `Recall@3`, `Recall@5`, mean reciprocal rank,
routing counts, and average retrieval time for `text`, `visual`, fixed `hybrid`,
and `adaptive` retrieval. The splits are `train[:800]` for development,
`train[800:1000]` for validation, and the original `test` split for final results.
Do not tune settings on the test split; reserve it for the final comparison:

```bash
python -m evaluation.evaluate_retrieval --split test --limit 200
```

To evaluate actual image pixels with the pretrained CLIP model, use:

```bash
python -m evaluation.evaluate_retrieval --split validation --limit 50 --strategy all --image-encoder clip
```

The first CLIP run downloads `openai/clip-vit-base-patch32` from Hugging Face.

To evaluate semantic text retrieval with pretrained MiniLM embeddings, use:

```bash
python -m evaluation.evaluate_retrieval --split validation --limit 50 --strategy all --text-encoder minilm
```

For the full pretrained multimodal comparison, enable both encoders:

```bash
python -m evaluation.evaluate_retrieval --split validation --limit 50 --strategy all --text-encoder minilm --image-encoder clip
```

## Honest Limitation

The demo can generate an answer with the pretrained
`HuggingFaceTB/SmolVLM-256M-Instruct` vision-language model. Click **Generate
answer with SmolVLM** after retrieval. On the first click, the model downloads
from Hugging Face; it receives the question, a retrieved document image, and
retrieved OCR evidence. The app displays the evidence IDs beside the prediction.

Generated answers are not guaranteed correct. Compare every prediction with the
DocVQA reference answer and retrieved evidence. To calculate DocVQA ANLS for
generated answers, select one retrieval strategy and keep the example count
small while running locally:

```bash
python -m evaluation.evaluate_retrieval --split validation --limit 10 \
	--strategy adaptive --text-encoder minilm --image-encoder clip --generate-answers
```

ANLS compares a prediction with all accepted DocVQA answers using normalized
Levenshtein similarity. Scores below $0.5$ are counted as $0$.
