# Data provenance and curation note

## Dataset used by the prototype

The prototype uses `data/papers.json`, a locally curated 40-record corpus of influential LLM-related research papers. Each record stores four fields: `title`, `authors`, `year`, and `abstract`.

The dataset was selected to cover foundational Transformer work, pretrained language models, parameter-efficient fine-tuning, instruction tuning and alignment, RAG, chain-of-thought prompting, and recent open-weight model reports. Examples include *Attention Is All You Need*, BERT, LoRA, QLoRA, Retrieval-Augmented Generation, Self-RAG, LLaMA, and DeepSeek-R1.

## Scope and limitation

This is a small educational corpus, not a comprehensive literature database. It is appropriate for transparent Colab demonstration because each record can be inspected manually, but it cannot answer paper-specific questions about absent documents. The DPR hallucination test deliberately demonstrates this boundary.

## Final-submission action

Before submission, add a source appendix or a `sources.csv` mapping every paper title to its authoritative landing page, such as ACL Anthology, arXiv, NeurIPS proceedings, or the publisher's official technical-report page. This will make abstract provenance independently auditable.

