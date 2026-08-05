# Hybrid Retrieval — Fixed Five-Question Diagnostic

Run this diagnostic in `cloud_smoke_test.ipynb` after loading `papers.json`, the
all-MiniLM-L6-v2 embedder, and the FAISS index.  Use the reusable logic in
`hybrid_retrieval_extension.py` with `dense_weight=0.5` and `k=3`.

| Question | Expected title | Purpose |
| --- | --- | --- |
| Who wrote Attention Is All You Need? | Attention Is All You Need | Exact paper-title evidence |
| What is LoRA? | LoRA: Low-Rank Adaptation of Large Language Models | Short acronym / title evidence |
| What is QLoRA? | QLoRA: Efficient Finetuning of Quantized LLMs | Related acronym discrimination |
| What is BERT? | BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding | Dense-neighbour confusion check |
| Who introduced the Transformer architecture? | Attention Is All You Need | Known hard, generic-query failure case |

## How to report it

Record both dense and hybrid `Title Hit@3`, plus the Top-1 title.  Do not remove the
fifth question merely because it fails: it is the counterexample that demonstrates the
limit of lexical and dense matching for a generic entity question.

The current live result in `cloud_smoke_test.ipynb` was 4/5 Hit@3 for both approaches.
Hybrid nevertheless moved the Attention and LoRA queries to Top-1.  This is an ordering
improvement for explicit-entity questions, not proof of a globally better retriever.
