# 课程资料 RAG 问答系统

基于本地 Qdrant、BGE 中文向量模型、Ollama、FastAPI 和 Gradio 的课程资料问答项目。系统展示检索候选、句子级引用与原文片段，便于追溯回答来源和人工核对。

重点是完成“资料处理 → 检索 → 生成 → 证据约束 → 回归评测”的工程闭环，而不是声称回答一定正确。

## 已实现的功能与代码入口

| 能力 | 实现与边界 |
|---|---|
| Markdown 资料处理 | `app/loader.py`、`app/splitter.py` 按章节和字符窗口切块，保留课程、文件、章节等元数据 |
| 本地向量检索 | `app/retriever.py` 使用 BGE 与 Qdrant，支持课程过滤，扩大 Dense 候选后结合标题、正文词法匹配排序 |
| 显式线索恢复 | 问题中有带引号线索时，从同课程索引补入匹配片段，减少“文件已召回但答案段落遗漏” |
| 生成与引用 | `app/generator.py` 调用 Ollama，`app/citations.py` 将引用编号映射到原文片段 |
| 句子级证据约束 | `app/evidence.py` 用向量语义相似度匹配结论与资料，补充引用、移除低分表述，覆盖不足时拒答 |
| 澄清与拒答 | `app/guardrails.py` 处理部分缺少上下文、敏感或缺乏依据的问题；不是通用安全保证 |
| 窄范围摘录降级 | 对有明确引号线索的公式、关系或复杂度问题，在模型拒答时尝试摘录命中原文，并继续做证据校验 |
| 服务接口 | FastAPI 问答、存活与就绪检查，Gradio 网页；`app/service.py` 管理共享管线和并发访问生命周期 |
| 检索实验与评测 | `app/hybrid.py` 实现 Dense + BM25 的 RRF 融合，`app/reranker.py` 实现 CrossEncoder 重排，评测脚本支持对照 |
| 模型答案人工复核 | `scripts/review_answers.py` 展示答案、引用和原文，记录四项人工评分、问题分类、来源哈希与修订历史；不自动替人打分 |

当前 API 问答管线使用 Dense 检索及上述词法/显式线索优化；Hybrid 和 CrossEncoder 是可运行的实验策略，尚未接入默认问答管线。

当前只读取 Markdown。没有 PDF 解析、OCR 或多模态输入；元数据中的 `page=1` 是占位值，不是真实 PDF 页码。

## 数据范围

仓库包含 7 门课程、30 份 Markdown 演示讲义；使用默认 `chunk_size=500`、`chunk_overlap=80` 时，已有处理清单记录 557 个章节单元、612 个文本块。

评测报告将这些资料标注为“合成讲义、小规模自建测试”，不能据此代表真实企业知识库效果。`INDEX.md` 不参与切块。

原始资料在 [data/raw](data/raw)。`data/processed/` 和 `index/` 被 Git 忽略，克隆仓库后需要在本机生成文本块和索引，不能直接假设已有 612 个向量点。

## 快速开始

本地验证环境：Python 3.12.4。以下命令在项目根目录执行，依赖见 [requirements.txt](requirements.txt)。

### 1. 创建环境

Windows PowerShell：

```powershell
git clone https://github.com/duanheming-afk/rag-course-qa.git
cd rag-course-qa
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
```

已有 `.env` 时不要覆盖它。默认 `EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5`、`LLM_MODEL=qwen2.5:3b`，Ollama 客户端地址为 `http://127.0.0.1:11434`。

Linux / macOS：

```bash
git clone https://github.com/duanheming-afk/rag-course-qa.git
cd rag-course-qa
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

以下 Windows 命令显式使用虚拟环境解释器；Linux / macOS 激活环境后将 `.\.venv\Scripts\python.exe` 换为 `python`。

### 2. 准备模型

先安装 Ollama 并下载模型：

```powershell
ollama pull qwen2.5:3b
ollama list
```

如 Ollama 尚未运行，在独立终端执行 `ollama serve`；已有后台服务时无需重复启动。构建索引时会加载 BGE 模型，首次使用需下载模型文件。

### 3. 处理资料、构建索引

```powershell
.\.venv\Scripts\python.exe scripts/ingest.py
.\.venv\Scripts\python.exe scripts/build_index.py
```

这些命令使用默认路径、集合名和向量模型。自定义 `.env` 后，需通过 `build_index.py` 的 `--chunks`、`--index`、`--collection`、`--model` 参数使用相同配置，不能混用不同模型或维度的索引。

Qdrant 使用本地存储模式；构建索引或运行独立评测前，应停止正在占用同一索引的 API/CLI 进程。

### 4. 启动服务

```powershell
.\.venv\Scripts\python.exe scripts/run_api.py
```

| 地址 / 接口 | 用途 |
|---|---|
| `http://127.0.0.1:8000/ui/` | Gradio 问答界面 |
| `http://127.0.0.1:8000/docs` | Swagger API 文档 |
| `GET /live` | 进程存活检查 |
| `GET /health` | 索引和 Ollama 就绪检查；依赖未就绪时返回 503 |
| `GET /courses` | 可用课程列表 |
| `POST /ask` | 问答接口 |

示例请求，在另一个 PowerShell 窗口执行：

```powershell
$questionRequest = @{
    question = '什么是导数的定义？'
    course = '高等数学'
    top_k = 8
} | ConvertTo-Json
Invoke-RestMethod http://127.0.0.1:8000/ask -Method Post -ContentType 'application/json; charset=utf-8' -Body $questionRequest
```

`/health` 返回 `ready: true` 只说明依赖就绪，不代表回答正确。使用时请核对检索原文、引用与被移除表述的警告。

也可在停止 API 后直接提问：

```powershell
.\.venv\Scripts\python.exe scripts/ask.py '什么是反向传播算法？' --course '机器学习'
```

## 证据约束如何工作

系统把生成回答拆成结论，将每条结论与检索片段的正文作向量语义匹配，在保留结论后添加 `[资料N]`，并删除低于 `EVIDENCE_MIN_SCORE` 的表述。章节标题、Markdown 标题和合成讲义测试说明不作为匹配证据；只有这些内容的片段不参与引用候选。每条结论只挂接正文相似度最高的一个片段，不因分数接近顺带挂接其他引用。过滤后仍保留原始检索编号，避免引用错位。保留结论占原始结论的比例低于 `EVIDENCE_MIN_COVERAGE` 时，拒绝输出回答。

正文阈值之外保留原始整块相似度阈值作为回归保护；任一检查不通过就过滤。这样更换引用匹配输入不会重新放行旧规则下的低分句子，但仍不能证明通过的句子正确。编码前去重重复文本，降低额外匹配开销。

默认相似度阈值为 0.52、覆盖率阈值为 0.50。这里的“证据覆盖率”是语义匹配代理指标，不是自然语言蕴含证明，也不是答案准确率。相似但错误的结论仍可能通过，删除结论也可能损失完整性，需要人工复核。一句中涉及多个不同来源的复杂推理还需要更细的结论拆分；本步骤不声称解决公式/逻辑错误。旧报告对应旧实现，不能作为修改后效果的证明。

## 测试与评测命令

单元测试不要求加载真实 Ollama 或 Qdrant 索引：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

2026-10-09 本地核验：62 个自动化测试全部通过（原有 24 项、人工评分流程 25 项、正文证据与引用回归 13 项）。覆盖引用映射、词法与显式线索排序、澄清/拒答、证据筛选、窄范围公式摘录、评测队列导出、模型空回答/截断回答处理，以及评分校验、原子保存、来源哈希、修订历史、未评分/N/A 的分母处理、标题/说明排除、原始引用编号、最优片段选择和旧低分句子不被重新放行。新增接口测试使用替身检索与生成器检查 `/ask` 的引用编号和片段序列化；不等于真实端到端准确率验证。

停止占用索引的服务后，运行检索对照：

```powershell
.\.venv\Scripts\python.exe scripts/evaluate.py --skip-rerank
```

去掉 `--skip-rerank` 可加入 CrossEncoder 策略；首次运行需下载重排模型。运行已有 136 条回归集：

```powershell
.\.venv\Scripts\python.exe scripts/evaluate_answers.py --questions data/eval/holdout_approved.jsonl --top-k 8 --output-dir reports/holdout_runs
```

每次评测会新建带时间戳的报告目录，保留配置、数据哈希、逐题结果与人工复核模板。数值可能随模型版本、硬件和配置变化。

## 已有报告：指标与限制

### 136 条答案回归

来源：[2026-09-21 报告](reports/holdout_runs/20260921T061907988848Z/summary.md)、[逐题结果与配置](reports/holdout_runs/20260921T061907988848Z/answers.json)。使用 `qwen2.5:3b`、Dense、`top_k=8`、`max_tokens=2048`、`temperature=0`、`seed=42`。

该自建集含 112 条可回答题、12 条歧义题、12 条无依据题，用于工程回归，不作为严格独立的泛化能力证明。

| 指标 | 已记录结果 | 解释 |
|---|---:|---|
| 请求成功率 | 100% | 完成处理，不等于答案正确 |
| 关键词召回代理分 | 45.83% | 参考关键词命中程度，不是语义准确率；也提示需要继续检查答案完整性 |
| 目标文件召回率 / 引用率 | 100% / 100% | 文件层面覆盖，不保证命中正确段落或引用真正支持结论 |
| 平均语义证据覆盖率 | 98.33% | 原始生成结论中通过相似度筛选的平均比例 |
| 全句语义证据率 | 95.54% | 原始结论全部通过相似度筛选的答案比例 |
| 无依据题拒答率 | 100%（12 题） | 仅限本次无依据样本 |
| 歧义题澄清率 | 100%（12 题） | 仅限本次歧义样本 |
| 可回答题误拒答率 | 0% | 不能据此判断答案完整性 |
| 无效 / 缺失引用编号样本数 | 0 / 0 | 引用格式检查，不是引用忠实性检查 |
| P50 / P95 单题耗时 | 1542.36 / 2743.21 ms | 逐题检索与生成耗时；首题可能包含冷启动，不是并发压测结果 |
| 人工答案正确性 / 证据支持评分 | 未完成 | 报告字段仍为 `null` |

题目与参考答案的审核确认，不等于对模型生成答案完成了正确性评分。待填模板见 [manual_review.json](reports/holdout_runs/20260921T061907988848Z/manual_review.json)。

### 30 条文件级检索对照

来源：[检索报告](reports/retrieval_runs_after_anchor/20260921T033410319879Z/summary.md)、[逐题结果](reports/retrieval_runs_after_anchor/20260921T033410319879Z/retrieval.json)。`top_k=5`、`candidate_k=20`，计时不含模型加载。

| 策略 | Top-1 | Top-5 | MRR | 平均耗时 / P95 |
|---|---:|---:|---:|---:|
| Dense | 100% | 100% | 1.0000 | 8.86 / 11.09 ms |
| Hybrid_RRF | 100% | 100% | 1.0000 | 9.98 / 12.01 ms |

本组样本上没有观察到 Hybrid 的文件命中优势。它是合成讲义上的小样本文件级实验，不能说明段落检索或最终答案已达到 100% 准确率。旧报告计时口径不同，不直接作性能比较。

## 人工评测流程与下一步

新建候选队列会写入评测数据文件；已有审核结果时先备份，不要为了查看报告重复生成队列。

```powershell
.\.venv\Scripts\python.exe scripts/build_holdout_review_queue.py
.\.venv\Scripts\python.exe scripts/pre_review_holdout.py --count 20
.\.venv\Scripts\python.exe scripts/review_holdout.py
.\.venv\Scripts\python.exe scripts/export_approved_holdout.py --check
.\.venv\Scripts\python.exe scripts/export_approved_holdout.py --min-approved 100
```

候选队列由 AI 起草；按 [评测标注规范](docs/EVALUATION_PROTOCOL.md) 阅读原文、审核题目和参考答案后才可导出。要做独立评测，应锁定未用于调参的新题集，并在生成后另行填写答案正确性、完整性及证据支持评分。

### 对已有模型答案评分（不同于审核题目）

```powershell
.\.venv\Scripts\python.exe scripts/review_answers.py
```

打开 `http://127.0.0.1:7862/`，逐题核对最终答案、真实引用片段和原文，记录正确性、完整性、证据支持、引用支持与问题分类。工具默认未评分，不生成虚假的人工准确率；详情见 [模型答案人工评分说明](docs/ANSWER_REVIEW.md)。

风险排序只用于决定先审核哪些题。已有样本存在将章节标题作为关键词的情况，字面 0 分不等于答案错误，需人工核对标注。

评分和汇总保存在被 Git 忽略的 `reports/manual_review_local/<运行ID>/`，原始报告和题目标签不被覆盖。只有完整评分后才可报告整个评测集的人工效果；部分复核必须注明数量与选择偏差。

下一步优先完成已有 136 条模型答案的人工评分与错误分类，再以真实资料、新题集对比 Dense、Hybrid、Rerank。PDF/OCR、多模态和生产级并发部署属于后续功能，不列为当前已完成能力。
