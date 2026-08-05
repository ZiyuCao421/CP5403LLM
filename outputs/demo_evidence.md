# LLMAssessment2 — Colab Demo Evidence

Run date: 2026-07-23  
Environment: Google Colab, T4 GPU  
Scope: one-question end-to-end smoke test, not the full 10×2 A/B evaluation

## What actually ran

The smoke harness executed the real setup and function cells from `with_rag_v6.ipynb`:

1. Loaded `Qwen/Qwen2.5-1.5B-Instruct` on the T4.
2. Loaded the 40-paper `papers.json` corpus.
3. Encoded the corpus with `sentence-transformers/all-MiniLM-L6-v2` on CPU.
4. Built a normalized inner-product FAISS index.
5. Retrieved Top-3 papers.
6. Generated one answer with an 80-token cap.
7. Evaluated retrieval and answer content separately.

## Observed result

```text
CLOUD_SMOKE_PASS
model: Qwen/Qwen2.5-1.5B-Instruct
model VRAM after load: 2944 MB
question: Who introduced the Transformer architecture and in which year was the paper published?
retrieved_titles:
  1. Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer
  2. LoRA: Low-Rank Adaptation of Large Language Models
  3. The Llama 3 Herd of Models
retrieval_hit: 0.0
keyword_recall: 0.667
latency_seconds: 8.7
answer: The Transformer architecture was introduced by Google researchers in 2017.
        The paper "Attention Is All You Need" by Vaswani et al., published in 2017,
        introduced the Transformer model, which became a foundational component in
        modern deep learning architectures, particularly in natural language processing tasks.
```

## Correct interpretation

- The end-to-end technical path works: model loading, corpus loading, embedding, FAISS retrieval, context construction, generation, and evaluation all completed.
- Retrieval failed for this question because the expected paper, `Attention Is All You Need`, was not in the Top-3. Therefore `retrieval_hit = 0.0` is correct.
- The generated answer was still substantially correct. The likely explanation is the base model's parametric knowledge, not successful retrieval.
- This run does **not** prove that RAG is better than no-RAG. A fair conclusion requires the planned paired 10-question runs.
- `citation_marker_present` detects only a year/author-shaped marker. It is not citation-correctness verification.

## Demo value of the failure

This is a useful example of why a RAG system must evaluate retrieval and generation separately. A correct-looking answer can hide a failed retriever. The next retrieval improvements are:

1. Include title, authors, and year—not only title and abstract—in indexed text.
2. Add a lexical BM25 path and combine it with dense retrieval.
3. Rewrite entity-heavy questions before retrieval.
4. Compare Top-3 and Top-5 using retrieval hit rate before changing generation.

## Completion boundary

Completed: static audit, fair shared code, validated corpus, cloud end-to-end RAG smoke test.  
Not completed: full 10-question no-RAG run, full 10-question RAG run, aggregate A/B comparison.

## 2026-07-29 paired tests and hybrid retrieval update

Notebook used for the live work: `cloud_smoke_test.ipynb`.  The source notebooks remain
`with_rag_v6.ipynb` and `no_rag_v6.ipynb`; `papers.json` remains the 40-paper corpus.

### Same-question RAG / no-RAG check

Question: `Who introduced the Transformer architecture, and in what year?`

| Condition | Retrieved titles | keyword recall | Generation latency |
| --- | --- | ---: | ---: |
| no-RAG | none | 1.0 | 4.03 s |
| dense RAG | T5, LoRA, BERT | 1.0 | 1.79 s |

Both answers named Vaswani et al. and 2017, while the dense RAG retriever still did not
retrieve `Attention Is All You Need` (`retrieval_hit = 0.0`).  This remains evidence of
parametric model knowledge, not evidence that RAG improved correctness.  The two reported
latencies cover generation only and are not full end-to-end latency measurements.

### Hybrid retrieval diagnostic

A diagnostic hybrid reranker combined normalized dense scores with lexical overlap from
title, authors, and abstract.  At dense weight 0.5 it promoted LoRA and exact-title
Attention queries to Top-1.  In the five retrieval-only questions below, both dense and
hybrid achieved 4/5 expected-title Top-3 hits; hybrid improved the Top-1 result for the
Attention and LoRA queries.

| Query type | Dense Top-3 hit | Hybrid Top-3 hit | Important observation |
| --- | ---: | ---: | --- |
| Exact Attention title | yes | yes | hybrid moved target to Top-1 |
| LoRA | yes | yes | hybrid moved target to Top-1 |
| QLoRA | yes | yes | target already Top-1 |
| BERT | yes | yes | RoBERTa remained Top-1 |
| Generic Transformer origin | no | no | unresolved retrieval failure |

The hybrid approach is therefore a limited improvement for explicit paper/entity queries,
not a general fix.  The obsolete manual-upload setup cell was removed from
`cloud_smoke_test.ipynb` so that its old missing-file error is no longer part of the demo.

### Manual grounding / citation spot check

The paired LoRA RAG answer said that LoRA freezes pretrained weights and injects trainable
rank-decomposition matrices into Transformer layers.  That mechanism is supported directly
by the abstract of `LoRA: Low-Rank Adaptation of Large Language Models` (Hu et al., 2021),
which was retrieved at Top-1.  The answer did **not** explicitly cite the paper title or
year, so the factual grounding check passes for this one example but the citation-format
check fails.  This is a one-example spot check, not a corpus-level citation-correctness score.
