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

## Honest Limitation

This version is a retrieval baseline, not a trained AI model. It uses shared
OCR words to rank documents and simple keyword rules to choose the route. The
next step is replacing these baselines with pretrained text embeddings, image
embeddings, and a vision-language model that generates a cited answer.
