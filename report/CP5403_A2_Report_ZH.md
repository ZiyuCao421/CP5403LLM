# 小型检索增强语言模型的论文问答评测（中文核对版）

**课程：** CP5403 Large Language Models - Assessment 2  
**学生：** [填写姓名与学号]  
**GitHub 项目地址：** `https://github.com/[your-username]/[your-repository-name]`  
**用途：** 本文件用于核对英文报告 `CP5403_A2_Report_EN.md` 的事实与逻辑；正式提交应使用英文版并按最终 Word 模板检查词数和 APA 格式。

## 摘要

大语言模型能够生成流畅的回答，但流畅性并不保证内容受到证据支持。该项目实现并评测了一个面向重要 LLM 论文问答的小型检索增强生成（RAG）系统。系统使用 Qwen2.5-1.5B-Instruct 作为生成模型、40 篇论文组成的本地知识库、all-MiniLM-L6-v2 句向量模型，以及 FAISS 内积索引。对每一个问题，系统先在没有检索上下文的条件下生成 no-RAG 回答，再将检索到的前三篇论文加入 prompt 后生成 with-RAG 回答。评测包括检索命中诊断、成对回答比较、生成时间和语料库外幻觉测试。

真实运行显示：对于 LoRA 问题，no-RAG 将 LoRA 过度概括为“降低权重矩阵的秩”，而 with-RAG 正确指出 LoRA 冻结预训练权重、再加入可训练的低秩更新。另一方面，当提问的 DPR 论文不在 40 篇语料中时，RAG 检索到的是 REALM、Atlas 和原始 RAG 等相关但错误的论文，无法给出 DPR 的精确训练细节。因此，本项目的结论不是“RAG 必然优于 no-RAG”，而是：RAG 只有在知识库包含正确证据且检索成功时，才可能提高回答的 grounding 和可追溯性。

## 1. Introduction

本项目关注学术论文问答中的一个实际问题：模型可能给出看似可信、但没有可核查来源的回答。论文类问题通常要求精确区分题目、作者、年份和技术贡献，因此很适合检验 LLM 的事实可靠性。

RAG 的基本思想是在推理阶段为生成模型补充外部文本证据。原始 RAG 工作将参数化生成模型与非参数化知识来源结合（Lewis et al., 2020）。它的优势是可以更新语料库而不必重新训练 LLM，也可以显式检查模型在回答前检索了哪些文档。

本项目的研究问题是：

1. 当接收到检索到的论文证据时，1.5B 参数的 instruction-tuned 模型是否能更准确回答论文问题？
2. 检索质量是否决定一个看起来正确的 RAG 回答是否真的受到证据支持？
3. 当目标论文不存在于知识库中时，RAG 系统会如何表现？

## 2. Target Problem and Related Concepts

### 2.1 Target problem

目标任务是一个受限语料库内的论文特定问答系统。一个高质量回答不仅要说明概念，还应能区分相近论文，并让使用者检查回答所依据的文档。该任务适合 LLM，因为输入是自然语言，知识来源是论文元数据与摘要，最终效果同时依赖语义检索与文本生成。任务范围被有意限制为 40 篇论文，以保证课堂演示时能够检查检索路径、prompt 和输出。

### 2.2 Parametric knowledge and hallucination

no-RAG 条件依赖 Qwen 参数中已经学习到的知识。对于 Transformer 等广为人知的内容，模型可能不依赖检索也能正确回答；但这种回答没有在生成时检索到的可核查来源。因而“回答正确”不等于“RAG 成功”。

在本项目中，幻觉是指模型做出没有被现有证据支持、错误、不可验证或过度确定的具体主张。项目不以回答长度或流畅性判断真实性，而是分别检查：预期论文是否被检索到、答案是否包含关键事实、以及当知识库没有该论文时模型是否恰当地表达不确定性。

### 2.3 RAG mechanism

系统流程为：

`Question -> embedding -> FAISS Top-3 retrieval -> paper context -> Qwen generation -> answer`

问题与每篇论文都被编码为向量；经 L2 归一化后，内积相当于 cosine similarity。FAISS 用于密集向量相似性搜索。检索到的三篇论文记录被转化为上下文，输入到与 no-RAG 条件完全相同的生成模型中。这样，成对实验中唯一预期改变的变量是“是否获得检索证据”。句向量检索的思想与 Sentence-BERT（Reimers & Gurevych, 2019）一致，FAISS 提供了向量检索基础设施（Douze et al., 2024）。

## 3. System Design and Implementation

### 3.1 Choice of data sources

知识库包含 40 条 JSON 论文记录。每条记录有 `title`、`authors`、`year` 和 `abstract` 字段，覆盖 Transformer、BERT、LoRA、QLoRA、RAG、Self-RAG、LLaMA 和 DeepSeek-R1 等主题。小语料库的优点是透明、可演示和可人工检查；缺点是语料覆盖范围成为系统的明确边界。

**正式提交前必须补充：** 为 40 篇论文加入来源附录或 `sources.csv`，将每个标题链接到 ACL Anthology、arXiv、NeurIPS proceedings 或技术报告官方页，使摘要来源可审计。

### 3.2 Models and retrieval stack

生成模型为 `Qwen/Qwen2.5-1.5B-Instruct`（Qwen Team, 2025）。选择该模型的原因是：它能在 Google Colab T4 GPU 上可运行、可复现，同时仍能够生成可讲解的论文回答。embedding 模型为 `sentence-transformers/all-MiniLM-L6-v2`，在 CPU 上运行。论文向量被加入 FAISS `IndexFlatIP`；Top-k 固定为 3。

系统已在 Google Colab T4 上完成端到端 smoke test：加载 Qwen、加载 40 篇语料、生成 embedding、建立 FAISS index、检索 Top-3、构造 context、生成答案并计算诊断指标。模型加载后记录的显存约为 2,944 MB。

### 3.3 Fair A/B procedure and metrics

Cell 14 是交互式 A/B 测试窗口。使用者只修改 `QUESTION`，Cell 会连续生成：

1. **no-RAG：** `generate(question, context="")`；
2. **with-RAG：** 检索 Top-3 论文，拼接 title/year/abstract，再用相同模型生成。

两种条件均使用 deterministic decoding（`do_sample=False`）和相同 token cap。当前的时间为 generation time，不是完整的 embedding + retrieval + generation end-to-end latency；这必须在报告中透明说明。

评测指标包括 Title Hit@3、keyword recall、manual grounding check、generation latency 与 out-of-corpus behaviour。它们必须分开解释：检索命中不保证模型会正确使用上下文；反过来，模型生成正确答案也可能只来自参数记忆。

## 4. Results

### 4.1 Smoke test and retrieval diagnostic

Transformer 起源问题的 smoke test 最终回答正确指出 Vaswani et al. 和 2017，但 Top-3 检索到的是 T5、LoRA 和 Llama 3，而不是 *Attention Is All You Need*，因此 retrieval Hit@3 = 0.0。正确答案应被解释为模型参数知识，而不是 RAG grounding。这一反例说明 RAG 评测必须同时报告 retrieval 和 generation。

固定的五题检索诊断中，dense 与 hybrid retrieval 的预期标题 Hit@3 均为 4/5。Hybrid 改善了明确 Attention 和 LoRA 查询的 Top-1 排序，但没有解决“谁提出了 Transformer？”这一泛化问题。因此，hybrid 的结果只是实体/标题查询的有限排序改善，而非通用检索器胜利。

### 4.2 LoRA in-corpus A/B test

问题：*What is the main idea of LoRA?*

| 条件 | 生成时间 | 真实输出的主要含义 |
|---|---:|---|
| no-RAG | 3.49 s | 将 LoRA 概括为降低神经网络权重矩阵的秩，容易误导为直接改变预训练矩阵。 |
| with-RAG | 4.94 s | 说明冻结预训练权重，并在 Transformer 层中加入可训练的低秩更新，从而减少可训练参数。 |

with-RAG 回答更符合 Hu et al.（2021）对 LoRA 的机制描述：原权重被冻结，学习的是低秩增量矩阵。代价是本次运行多了约 1.45 秒生成时间。需注意，LoRA 在该次 Top-3 中排第 3，因此这是一项有价值的定性成功案例，而不是对所有问题的普遍结论。

### 4.3 DPR out-of-corpus hallucination test

测试问题针对 *Dense Passage Retrieval for Open-Domain Question Answering*（Karpukhin et al., 2020）。该论文已核对不在 `papers.json` 的 40 篇语料中。

| 条件 | 生成时间 | 真实表现 |
|---|---:|---|
| no-RAG | 4.71 s | 知道 dense retrieval 的宽泛概念，但对 negative sampling 的描述不完整。 |
| with-RAG | 5.20 s | 检索 REALM、Atlas 和原始 RAG，不是 DPR；无法给出 DPR 特有的负样本训练细节。 |

DPR 的正确机制是使用 dual encoder 进行 dense passage retrieval，并在对比式训练中使用 negative passages，包括 BM25-derived hard negatives；in-batch negatives 也是其训练形式的一部分。两种模式都没有完整给出这些细节。该实验并不证明“RAG 会导致幻觉”，而是证明：当语料库中没有目标论文时，没有检索阈值与拒答机制的 RAG 不能阻止不充分或无依据的回答。

## 5. Discussion, Limitations and Reflection

核心结论是有条件的：RAG 在 LoRA 问题中提供了更具体、更贴近论文的解释，但它没有解决知识库外的 DPR 问题。RAG 扩展的是一个被选定的本地语料库，不是整个互联网。

最直接的改进是加入 abstention rule：当最高检索相似度低于验证集确定的阈值，或查询中出现不在库内的论文标题时，系统应明确输出“当前知识库没有足够证据回答该论文特定问题”。系统可以额外给出一个清晰标记为 parametric knowledge 的一般性回答，但不能将其伪装成来源支持的结论。

其他改进包括：将 title、authors、year、abstract 一同索引；加入 BM25、hybrid retrieval 或 reranker；展示每次检索的标题和明确引用；扩大预先定义的测试集；以及使用独立人工评分判断事实正确性、faithfulness 和适当的不确定性。

当前局限包括：只有 40 篇语料、测试集规模较小、keyword matching 只是代理指标、时间不包含完整端到端检索成本、prompt 不强制正式引用、且没有经过校准的 retrieval threshold。

### Presentation peer-feedback reflection - final field

以下内容必须在 Week 9/10 收到真实反馈后填写，不能由 AI 编造：

> **[待本人填写]** 在 Week [9/10] 的同伴反馈中，我收到的具体反馈是：[反馈内容]。我修改了：[某页 slide、图表、技术解释、时间控制或评测说明]。修改原因是：[原因]。这一修改通过：[如何使听众更容易理解、提高技术准确性或改善时间控制] 改善了项目展示。

## 6. Conclusion

本项目实现了一个可在 Colab 上运行的论文问答 RAG 原型。成对 A/B 设计显示：在相关证据被检索到时，with-RAG 可以使 LoRA 回答更加准确和具体；但 DPR 测试与 Transformer retrieval miss 同时说明，可信 LLM 应用必须区分 corpus coverage、retrieval quality、generation quality 和 uncertainty handling。最终结论是：**RAG 是条件性的 grounding mechanism，而不是自动消除 hallucination 的通用解决方案。**

## GenAI use declaration - draft

MiniMax 和 Codex 被用于 brainstorming、代码草稿、语言编辑、双语核对和 presentation planning。学生负责检查最终 notebook、Colab 输出、数值结果、解释、引用和提交内容。任何生成文本在成为实验结论前，均应根据本地语料、运行输出或原始论文来源进行核对。

## References

Douze, M., Guzhva, A., Deng, C., Johnson, J., Szilvasy, G., Mazaré, P.-E., Lomeli, M., Hosseini, L., & Jégou, H. (2024). *The Faiss library*. arXiv. https://arxiv.org/abs/2401.08281

Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2021). *LoRA: Low-rank adaptation of large language models*. arXiv. https://arxiv.org/abs/2106.09685

Karpukhin, V., Oguz, B., Min, S., Lewis, P., Wu, L., Edunov, S., Chen, D., & Yih, W.-t. (2020). Dense passage retrieval for open-domain question answering. In *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing* (pp. 6769-6781). Association for Computational Linguistics. https://doi.org/10.18653/v1/2020.emnlp-main.550

Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Küttler, H., Lewis, M., Yih, W.-t., Rocktäschel, T., Riedel, S., & Kiela, D. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. *Advances in Neural Information Processing Systems, 33*, 9459-9474.

Qwen Team. (2025). *Qwen2.5 technical report*. arXiv. https://arxiv.org/abs/2412.15115

Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-networks. In *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing* (pp. 3982-3992). Association for Computational Linguistics. https://doi.org/10.18653/v1/D19-1410

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention is all you need. *Advances in Neural Information Processing Systems, 30*.
