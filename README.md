# CP5403 Assessment 2 - RAG Paper Question Answering

A small, reproducible Retrieval-Augmented Generation (RAG) prototype for answering questions about landmark LLM papers.

## What this project compares

- **no-RAG:** Qwen answers from parametric knowledge only.
- **with-RAG:** the same Qwen model receives Top-3 retrieved paper records as context.

The goal is to test when retrieval improves grounding, not to claim that RAG is always better. RAG can only help when relevant evidence exists in the corpus and is retrieved successfully.

## Implementation

| Component | Choice |
| --- | --- |
| Generator | Qwen/Qwen2.5-1.5B-Instruct |
| Corpus | 40 LLM-paper records |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Retriever | FAISS IndexFlatIP, normalized dense vectors, Top-3 |
| Runtime | Google Colab T4 GPU |

## Repository structure

- data: 40-paper corpus
- notebooks: runnable no-RAG, with-RAG, and cloud smoke-test notebooks
- outputs: verified evaluation evidence
- report: English report draft and Chinese verification version
- docs: data provenance, AI-use declaration, submission checklist

## Evidence and limitations

- With-RAG gave a more specific LoRA explanation in a paired test, but added generation time.
- A correct Transformer answer can occur even when retrieval misses the expected paper; correct output is not automatically proof of grounding.
- DPR is absent from the corpus, so retrieved related papers cannot supply DPR-specific evidence.
- The planned Goldbach-conjecture test examines whether the system states that the current corpus is insufficient. The current version has no enforced abstention rule, so it cannot guarantee an I do not know response.

Run notebooks/with_rag_v6.ipynb in Colab from top to bottom. Place data/papers.json in the configured path before running the data-loading cell.

## Academic integrity

This is an individual course project. Generative-AI assistance is declared in docs/AI_USE_DECLARATION.md; all model outputs and factual claims require checking against the supplied data and cited sources.
