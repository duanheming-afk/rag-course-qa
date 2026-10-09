# 课程资料 RAG 问答系统

这是一个基于本地 Qdrant、BGE 中文向量模型和 Ollama 的课程资料问答项目。回答会展示检索候选、句子级证据引用和引用片段，便于人工核对。

## 1. 配置环境

```powershell
.\.venv\Scripts\Activate.ps1
Copy-Item .env.example .env
```

根据本机模型修改 `.env`。当前默认模型是 `qwen2.5:3b`，Ollama 地址建议使用 `http://127.0.0.1:11434`。

启动 Ollama 后确认模型存在：

```powershell
ollama serve
ollama list
```

## 2. 数据和索引

原始 Markdown 放在 `data/raw`。如需重新处理数据和构建索引：

```powershell
python scripts/ingest.py
python scripts/build_index.py
```

构建索引使用的向量模型、集合名和路径必须与 `.env` 保持一致。当前索引已有 612 个向量点。

## 3. 启动服务

```powershell
python scripts/run_api.py
```

打开 `http://127.0.0.1:8000/ui/` 使用网页界面，或访问：

```text
GET  /live       进程存活检查
GET  /health     索引和 Ollama 就绪检查
GET  /courses    可用课程列表
POST /ask        问答接口
```

示例请求：

```powershell
Invoke-RestMethod http://127.0.0.1:8000/ask -Method Post -ContentType 'application/json' -Body (@{
  question = '什么是导数的定义？'
  course = '高等数学'
  top_k = 8
} | ConvertTo-Json)
```

`/health` 返回 `ready: true` 后才表示依赖就绪；实际回答仍需结合返回的检索片段和引用进行核对。

每个回答会经过句子级证据校验：系统将每个结论与检索片段做语义匹配、在结论末尾补上对应 `[资料N]`，并移除未达到 `EVIDENCE_MIN_SCORE` 的表述。若保留结论的比例低于 `EVIDENCE_MIN_COVERAGE`，系统会拒绝该回答。它提供的是可审计的相关性约束，不是替代人工审核的逻辑蕴含证明。

对于问题中明确给出的带引号线索（例如“例 3”“贝叶斯公式”“过采样”），检索器会在同课程索引中恢复精确匹配的片段，再与 Dense 候选合并排序。该策略只处理显式线索，不改变普通问题的 Dense 检索路径，用于避免“目标文件已召回但具体答案片段未进入上下文”的情况。

## 4. 运行评测

先跑单元测试：

```powershell
python -m unittest discover -s tests -v
```

检索评测：

```powershell
python scripts/evaluate.py --skip-rerank
```

答案评测：

```powershell
python scripts/evaluate_answers.py --include-challenges
```

每次评测会写入带时间戳的目录，包含 `answers.json`、`summary.md` 和人工复核模板。关键词命中率、文件召回率和引用编号有效率只是自动代理指标，不能替代人工判断答案是否正确、完整、忠实于资料。

最近一次 136 条人工确认集回归（`qwen2.5:3b`）中，目标文件引用率为 100%、平均证据覆盖率为 98.33%、全句证据率为 95.54%，歧义题澄清率和无依据题拒答率均为 100%；检索基准 Dense Top-1/Top-5 均为 100%。这些指标只用于工程回归，不能替代人工复核。

## 5. 建立独立人工评测集

先从课程资料创建待复核队列；它是 AI 起草的标注候选，不是人工标签：

```powershell
python scripts/build_holdout_review_queue.py
```

默认会生成 120 条可回答候选题和 24 条歧义/无依据候选题。按照 [评测标注规范](docs/EVALUATION_PROTOCOL.md) 完成人工复核后，再导出至少 100 条审核通过且不与现有题目重复的独立评测集：

```powershell
python scripts/pre_review_holdout.py --count 20
python scripts/review_holdout.py
python scripts/export_approved_holdout.py --check
python scripts/export_approved_holdout.py --min-approved 100
python scripts/evaluate_answers.py --questions data/eval/holdout_approved.jsonl --top-k 8
```

## 6. 直接提问

```powershell
python scripts/ask.py '什么是反向传播算法？' --course '机器学习'
```

如果模型服务未启动、模型名称不匹配、索引被另一个进程占用或索引维度不一致，服务会返回可识别的错误码，而不是静默返回错误答案。
