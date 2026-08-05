# Evaluating a Small Retrieval-Augmented LLM for Paper Question Answering

**CP5403 Large Language Models — Assessment 2 draft**  
**Student:** [Your name / student number]  
**Project repository:** `https://github.com/ZiyuCao421/CP5403LLM`  
**Word count:** verify in the final Word template; references, tables, and appendices must be treated according to the course instructions.

## Abstract

Large language models (LLMs) can answer questions fluently even when their answers are not supported by the information available to a user. This project implements and evaluates a compact retrieval-augmented generation (RAG) prototype for questions about landmark LLM papers. The system uses Qwen2.5-1.5B-Instruct as the generator, a local corpus of 40 paper records, all-MiniLM-L6-v2 sentence embeddings, and a FAISS inner-product index. For every question, the same generator is run twice: first without retrieved context (no-RAG), and then with the top three retrieved papers appended to the prompt (with-RAG). The evaluation combines retrieval diagnostics, paired qualitative tests, latency observations, and a deliberately out-of-corpus hallucination test. On an in-corpus LoRA question, the no-RAG response inaccurately described LoRA as reducing the rank of the original weight matrix, while the RAG response correctly described freezing pretrained weights and adding trainable low-rank updates. However, retrieval did not always return the expected paper, and an out-of-corpus Dense Passage Retrieval (DPR) question showed that RAG does not automatically prevent unsupported answers when the relevant document is missing. The project therefore concludes that RAG is best viewed as a grounding mechanism conditional on corpus coverage and retrieval quality, rather than as a universal solution to hallucination.

## 1. Introduction

LLMs are useful because they can transform a natural-language question into a coherent answer. Their fluency can also conceal an important limitation: a plausible answer may be unsupported, incomplete, or factually wrong. This issue is particularly visible in academic question answering, where the title, authorship, date, and technical contribution of a paper must be distinguished precisely. A model may know a common fact from pretraining, but it may also mix details from neighbouring papers or provide an answer without any inspectable evidence.

Retrieval-augmented generation addresses this problem by retrieving external text at inference time and supplying it as context to a generator. In the original RAG formulation, a parametric model is combined with a non-parametric knowledge source (Lewis et al., 2020). This division is attractive for paper question answering because a corpus can be updated without retraining the language model. It also creates an auditable intermediate step: the retrieved documents can be inspected before interpreting the generated answer.

The aim of this project was not to claim that RAG is always more accurate. Instead, it was to build a small, runnable system and investigate three questions:

1. Can a 1.5B-parameter instruction-tuned model answer paper questions more accurately when it receives retrieved evidence?
2. Does retrieval quality determine whether an apparently good RAG answer is genuinely grounded?
3. What happens when a question concerns a paper that is not present in the RAG corpus?

The final question is essential. A RAG system cannot retrieve knowledge that it does not contain. Therefore, an out-of-corpus test is a more meaningful hallucination diagnostic than showing only successful in-corpus examples.

## 2. Target Problem and Related Concepts

### 2.1 Target problem

The target problem is paper-specific question answering over a deliberately bounded collection of LLM research papers. A useful answer should distinguish a paper's central contribution from superficially related concepts, and should make it possible to inspect the evidence used. This is suitable for LLM techniques because the input is natural language, the knowledge source is unstructured paper metadata and abstracts, and answer quality depends on semantic retrieval as well as generation. The problem is intentionally narrower than open-domain question answering: the aim is to make the evidence path observable and reproducible during a live classroom demonstration.

### 2.2 Parametric knowledge and hallucination

The no-RAG condition relies solely on knowledge represented in Qwen2.5-1.5B-Instruct's parameters. Such knowledge can be useful: for example, a widely discussed paper such as *Attention Is All You Need* may be recalled correctly. However, it is not attached to a document retrieved at answer time. A correct answer in this condition demonstrates model knowledge, not evidence-grounding.

Hallucination is used here in a practical sense: the model states a specific claim that is unsupported by the available evidence and is false, unverifiable, or overly certain. The project does not use fluency or answer length as a proxy for truth. Instead, it separately examines (a) whether the expected paper was retrieved, (b) whether the answer contains expected facts, and (c) whether a response remains appropriately uncertain when the corpus does not contain the requested paper.

### 2.3 Retrieval-augmented generation

The implemented RAG architecture is:

`Question → embedding → FAISS Top-3 retrieval → paper context → Qwen generation → answer`

The retriever encodes both paper text and the query into vectors. Similarity is computed using normalized inner product, which corresponds to cosine similarity for normalized vectors. FAISS is used for efficient vector search. The top three retrieved paper records are converted into prompt context for the same generator used in the no-RAG condition. Holding the generator constant makes the paired comparison fair: the intended experimental difference is access to retrieved context rather than a different LLM.

Dense retrieval is useful when the query and document express similar ideas with different wording. The sentence-embedding approach follows the general idea of semantically comparable sentence representations described by Reimers and Gurevych (2019), while FAISS provides dense-vector similarity search primitives (Douze et al., 2024). However, dense retrieval can confuse closely related papers and is often weak for short acronyms, exact titles, or generic entity questions. To investigate this limitation, the project also conducted a retrieval-only diagnostic that combined dense scores with lexical overlap from title, author, and abstract fields. This is a small hybrid-retrieval extension rather than a claim of a production-ready ranking system.

## 3. System Design and Implementation

### 3.1 Choice of data sources

The knowledge base contains 40 JSON records describing influential LLM-related papers. Each record includes a title, authors, year, and abstract. The collection includes papers such as *Attention Is All You Need*, BERT, LoRA, QLoRA, RAG, Self-RAG, LLaMA, and DeepSeek-R1. The corpus is deliberately small. This makes it easy to inspect during a demonstration, but it also means that corpus coverage is a central limitation rather than an implementation detail. The curation scope and remaining provenance limitation are documented in `docs/DATA_PROVENANCE.md`: the final submission should include a title-to-authoritative-source mapping for all 40 records, rather than treating the JSON file itself as an original scholarly source.

### 3.2 Models and retrieval stack

The generator is `Qwen/Qwen2.5-1.5B-Instruct` (Qwen Team, 2025). It was selected because it is small enough to load and demonstrate on a Google Colab T4 GPU while still producing readable explanatory answers. The embedding model is `sentence-transformers/all-MiniLM-L6-v2`, run on CPU. Paper representations are indexed in a FAISS `IndexFlatIP` index after L2 normalization. The system uses Top-*k* = 3 retrieval.

The implementation was tested in Google Colab using a T4 GPU. A recorded smoke test loaded the Qwen model using approximately 2,944 MB of VRAM after loading, embedded the 40-paper corpus, built the FAISS index, retrieved top-three records, constructed a context, generated an answer, and computed basic diagnostic metrics. This is important evidence that the pipeline is executable end to end rather than only a design diagram.

### 3.3 Fair A/B procedure

The final notebook includes an interactive A/B test cell. The user changes only `QUESTION`; the cell then executes two generations:

1. **No-RAG:** `generate(question, context="")`.
2. **With-RAG:** retrieve top-three papers, concatenate their title/year/abstract as context, then run the same generator.

Generation uses deterministic decoding (`do_sample=False`) and the same token cap in both conditions. The recorded time is generation time, not full end-to-end latency; embedding and retrieval overhead should be included in a future production evaluation.

### 3.4 Evaluation measures

The project uses several modest but interpretable measures:

- **Title Hit@3 / retrieval hit:** whether the expected paper is among the top three retrieved records.
- **Keyword recall:** proportion of expected task-specific keywords appearing in the generated answer.
- **Manual grounding check:** whether the stated mechanism is actually supported by the retrieved paper text.
- **Generation latency:** seconds spent in text generation.
- **Out-of-corpus behaviour:** whether the system retrieves irrelevant substitutes and whether its answer communicates uncertainty.

These metrics are intentionally separated. A generated answer can be correct even when retrieval fails because the base model already knows the fact. Conversely, a retrieved title can be correct while the generator misreads or ignores its context. Treating these as one score would hide the reason for a success or failure.

## 4. Results

### 4.1 End-to-end smoke test and retrieval diagnostic

The initial cloud smoke test used the question: “Who introduced the Transformer architecture and in which year was the paper published?” The final answer correctly named Vaswani et al. and 2017. However, the retriever returned *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer*, LoRA, and Llama 3 rather than *Attention Is All You Need*. Retrieval Hit@3 was therefore 0.0. The correct generation must be interpreted primarily as parametric knowledge, not evidence supplied by RAG.

This result is valuable rather than embarrassing. It demonstrates why RAG evaluation must inspect both retrieval and generation. Without the retrieval diagnostic, the answer could be presented misleadingly as a RAG success.

A five-question retrieval-only test compared dense retrieval with a simple hybrid reranker. Both approaches achieved 4/5 expected-title Hit@3. Hybrid retrieval moved the target paper to Top-1 for explicit Attention and LoRA queries, but it did not solve the generic “Who introduced the Transformer architecture?” query. The finding is limited: lexical information can improve ordering for entity-heavy questions, but it is not a general solution.

### 4.2 In-corpus LoRA A/B test

The final interactive cell was run with: “What is the main idea of LoRA?” The output is summarised below.

| Condition | Generation time | Observed answer quality |
|---|---:|---|
| no-RAG | 3.49 s | Explained LoRA as reducing the rank of a neural-network weight matrix. This is an oversimplification and risks implying that the pretrained matrix itself is replaced or reduced. |
| with-RAG | 4.94 s | Explained that pretrained weights are frozen and trainable low-rank updates are injected into Transformer layers, reducing the number of trainable parameters. |

The with-RAG answer is closer to the central mechanism in Hu et al. (2021): LoRA freezes pretrained weights and adds trainable rank-decomposition matrices. The RAG answer is not merely longer; it identifies the important distinction between the original frozen parameters and the new low-rank adaptation. The cost was approximately 1.45 seconds more generation time in this particular run.

There is still an important limitation. The top-three retrieval list was *The Llama 3 Herd of Models*, *Alpaca*, and *LoRA*, so the target appeared only at rank three. The generated answer nevertheless used the relevant LoRA evidence well enough. This is a positive qualitative example, but it should not be generalized to all questions.

### 4.3 Out-of-corpus hallucination test

To test the boundary of the system, I asked about *Dense Passage Retrieval for Open-Domain Question Answering* (DPR), a 2020 paper by Karpukhin et al. This paper was checked against `papers.json` and is not included in the 40-record corpus. The question asked for DPR's retrieval mechanism and negative-sampling strategy.

| Condition | Generation time | Retrieved evidence / response behaviour |
|---|---:|---|
| no-RAG | 4.71 s | Correctly recognised the broad idea of dense retrieval, but described negative sampling only generically and did not state the paper's precise training choice. |
| with-RAG | 5.20 s | Retrieved REALM, Atlas, and the original RAG paper—related but not DPR. It gave a generic dense-retrieval explanation and stated that the negative-sampling strategy was not explicitly detailed in the available abstract. |

The reference answer is that DPR trains a dual encoder for dense passage retrieval and uses negative passages, including BM25-derived hard negatives in the reported training setup; in-batch negatives are also part of the contrastive formulation. The DPR paper describes learning dense question and passage representations using a simple dual-encoder framework (Karpukhin et al., 2020).

Neither system variant gave this complete answer. The no-RAG response relied on general model knowledge and risked presenting an imprecise description as a paper-specific fact. The RAG system retrieved semantically related documents because its corpus lacked DPR. Its statement that the detail was not present in the evidence is more cautious, but it should have made the stronger point that DPR itself was absent from the corpus. This experiment therefore does not show that RAG caused a hallucination; it shows that unguarded RAG does not prevent one when coverage is missing.

## 5. Discussion

The central result is conditional rather than absolute: RAG improved factual specificity for the LoRA question when relevant evidence was available, but it did not solve knowledge gaps. This is exactly the behaviour expected from a retrieval-grounding architecture. Retrieval augments a model with a selected corpus; it does not turn a finite local corpus into the open web.

The LoRA example is useful for presentation because the difference is easy to explain in plain language. The no-RAG model produced a plausible general account of low-rank ideas. The RAG answer connected the concept to the actual implementation: keep the large pretrained model fixed and learn small additional matrices. This is a meaningful improvement for a user who asks what the paper contributed.

The DPR example is equally important because it prevents an overclaim. The system retrieved three papers about retrieval-augmented language models, but none was DPR. Similarity search found topical neighbours, not the required source. Feeding those neighbours into the generator cannot establish the specifics of DPR's negative-sampling method. In a demonstration, this failure is evidence of learning: the user should distinguish a *retrieval failure* from a *generation failure* and should explain why the answer cannot be treated as grounded.

The most direct improvement is an abstention rule. Before generating a RAG answer, the system should inspect retrieval similarity scores and optionally title/entity overlap. If the best score is below a validation-set threshold, or if the query contains a paper title not found in the corpus, the assistant should say: “The current knowledge base does not contain sufficient evidence to answer this paper-specific question.” It could then offer a general answer clearly labelled as parametric knowledge. This would make the difference between RAG evidence and LLM prior knowledge visible to the user.

Other improvements include indexing titles, authors, years, and abstracts together; adding BM25 or a learned reranker; increasing *k* only after measuring precision; and displaying the retrieved source titles next to every answer. A larger corpus is not automatically better, because irrelevant context may distract the generator. Corpus curation and retrieval evaluation are therefore as important as model selection.

### Reflection and limitations

The project changed my understanding of RAG. Initially, it was tempting to describe RAG as a technique that “reduces hallucinations.” The implementation shows a more precise statement: RAG can reduce unsupported claims only when the retriever returns relevant and sufficient evidence, the context fits within the prompt, and the generator follows it. The Transformer smoke test showed a correct answer despite retrieval failure; the DPR test showed related retrieval despite missing evidence. Both cases would be missed by evaluating only answer fluency.

There are also methodological limits. The corpus has only 40 papers, the generated evaluation set is small, and several measures are lexical proxies rather than human judgements of truth. Latencies are generation-only measurements. The current prompt does not force formal citations, and the system does not yet calculate a calibrated retrieval threshold. These limitations should be reported openly rather than hidden, because they identify concrete next steps.

### Week 9 draft feedback and the resulting evaluation requirement

The Week 9 draft discussion raised two precise questions: whether a prompt about a paper outside the 40-paper corpus can trigger an unsupported answer, and whether the system will appropriately abstain when asked an unresolved question such as Goldbach's conjecture. The DPR experiment addresses the first question with an actual out-of-corpus paper. It shows that related retrieval is not equivalent to source coverage. The second question is retained as a pre-specified next test, not as a completed result: the current system has no calibrated similarity threshold or enforced abstention prompt, so it cannot guarantee that it will say “I do not know.” This feedback changed the evaluation criterion from merely comparing fluent answers to testing corpus coverage, retrieval evidence, and uncertainty behaviour separately. The presentation explains this learning process in detail; this report records the resulting system requirement and its current implementation boundary.

### Presentation peer-feedback reflection

During the Week 9 draft discussion, the teacher asked whether an out-of-corpus prompt could lead to hallucination and whether the model would say “I do not know” for an unsolved problem such as Goldbach's conjecture. I responded by adding the DPR out-of-corpus diagnostic, separating retrieval evidence from answer fluency, and specifying a Goldbach abstention test as the next evaluation. This improves the presentation because it replaces a generic claim that “RAG reduces hallucination” with a concrete explanation of corpus coverage, retrieval failure, and the missing abstention guardrail.

### Reproducibility and evaluation choices

The design intentionally favoured reproducibility over maximum scale. A Colab T4 can load the 1.5B generator and execute the end-to-end pipeline in a classroom demonstration without specialist infrastructure. This is an appropriate engineering trade-off for an assessment prototype: the model is sufficiently capable to demonstrate grounding, but small enough that the student can rerun the experiment, inspect the prompts, and explain the retrieval path. Larger models or external commercial APIs could improve answer quality, but would make cost, reproducibility, and access control harder to explain.

The evaluation also avoids a misleading single “accuracy” number. For example, exact keyword matching can undervalue a correct paraphrase, whereas an eloquent answer can contain the expected words while misattributing the contribution to the wrong paper. Retrieval Hit@3 is therefore reported alongside answer-level evidence rather than treated as the final outcome. The manual LoRA check is particularly useful because it tests the causal claim that RAG supplied relevant evidence: the retrieved LoRA abstract supports the freezing-plus-low-rank-update explanation. In contrast, the generic Transformer question demonstrates that a correct answer may arise despite a retrieval miss. These examples make the numerical metrics interpretable.

For a stronger final evaluation, I would pre-register a balanced question set before observing outputs, including exact-title questions, acronym questions, author/year questions, contribution questions, and out-of-corpus questions. Two independent human markers could judge factual correctness, faithfulness to the retrieved documents, and appropriate uncertainty. Retrieval would be evaluated at several values of *k* and with a validation set for selecting a confidence threshold. These changes would improve scientific rigor, but are outside the practical scope of the current 40-document demonstrator.

### Ethical and practical considerations

Although this project is educational, the same risks matter in professional knowledge systems. A fluent answer to a literature question may influence a student's understanding, a research decision, or a citation in a written report. Showing retrieved titles is therefore not only a technical convenience; it gives users a simple way to inspect what evidence influenced the answer. It is also important not to describe a generated answer as a source. A generated response is an interpretation produced by the model, whereas the paper record is the evidence that should be checked.

The prototype does not collect personal data and uses public paper metadata and abstracts. Nevertheless, a deployed version would need governance for document provenance, copyright permissions, corpus updates, and access control. It would also need a clear user interface that distinguishes “answer supported by retrieved documents” from “general model answer without supporting documents.” The out-of-corpus DPR test motivates this distinction directly. A responsible system should prefer a transparent limitation statement over a confident but unsupported answer. This principle is relevant beyond RAG: reliability requires matching the strength of a claim to the strength of the available evidence.

## 6. Conclusion

This project built a runnable paper-question-answering RAG prototype using Qwen2.5-1.5B-Instruct, MiniLM embeddings, FAISS, and a 40-paper local corpus. A paired design enabled the same model to answer each question with and without retrieval. The LoRA test showed a clear qualitative grounding benefit: RAG produced the important frozen-weight plus low-rank-update explanation, while no-RAG provided a less accurate simplification. However, retrieval diagnostics and the DPR out-of-corpus test show that this benefit is conditional. RAG cannot retrieve absent papers, and semantically related documents can create an illusion of evidence. The key learning outcome is therefore not “RAG is always better,” but that trustworthy LLM applications require separate evaluation of corpus coverage, retrieval, generation, and uncertainty behaviour.

## AI-use declaration (adapt to the course policy)

MiniMax and Codex were used for brainstorming, drafting, language editing, bilingual checking, and presentation planning. The student reviewed the system design, Colab execution, data inspection, result validation, citations, and final factual claims. All reported runs and limitations should be retained only if the student can explain and reproduce them during the presentation.

## References

Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2021). *LoRA: Low-rank adaptation of large language models*. arXiv. https://arxiv.org/abs/2106.09685

Karpukhin, V., Oguz, B., Min, S., Lewis, P., Wu, L., Edunov, S., Chen, D., & Yih, W.-t. (2020). Dense passage retrieval for open-domain question answering. In *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing* (pp. 6769–6781). Association for Computational Linguistics. https://doi.org/10.18653/v1/2020.emnlp-main.550

Douze, M., Guzhva, A., Deng, C., Johnson, J., Szilvasy, G., Mazaré, P.-E., Lomeli, M., Hosseini, L., & Jégou, H. (2024). *The Faiss library*. arXiv. https://arxiv.org/abs/2401.08281

Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Küttler, H., Lewis, M., Yih, W.-t., Rocktäschel, T., Riedel, S., & Kiela, D. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. *Advances in Neural Information Processing Systems, 33*, 9459–9474.

Qwen Team. (2025). *Qwen2.5 technical report*. arXiv. https://arxiv.org/abs/2412.15115

Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-networks. In *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing* (pp. 3982–3992). Association for Computational Linguistics. https://doi.org/10.18653/v1/D19-1410

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention is all you need. *Advances in Neural Information Processing Systems, 30*.
