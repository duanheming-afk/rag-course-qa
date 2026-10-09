# 独立评测集标注规范

`data/eval/holdout_review_queue.jsonl` 是从课程资料自动抽取的候选题，不是人工标注集，也不能直接用于汇报模型效果。

## 标注步骤

1. 每位标注者只查看题目、候选证据和原始资料，不查看模型回答。
2. 在 `annotation` 填写人工确认的 `reference_answer`、`gold_source_files`、`required_keywords`、标注者和日期。
3. 将 `status` 改为 `approved`；证据不足、题目含糊或题目质量差时标为 `rejected` 或 `needs_rewrite`。
4. 对 `answer` 类题目，至少标一个目标文件和一个必须出现的关键词或同义词组；对 `clarify`、`refuse` 类题目，参考答案应写清预期行为。
5. 重要样本建议由两人独立标注。结论冲突时记录在 `notes`，由第三人裁决。

可以运行 `python scripts/review_holdout.py` 打开本地复核界面；保存时会原子更新队列文件。

首次使用时可先运行 `python scripts/pre_review_holdout.py --count 20`。它只检查题目与候选证据的结构完整性，并输出人工核对清单；不会写入人工参考答案或将状态改为 `approved`。

## 导出规则

```powershell
python scripts/export_approved_holdout.py --check
python scripts/export_approved_holdout.py --min-approved 100
```

导出脚本会拒绝少于 100 条、缺少人工参考答案、可回答题缺少关键词/目标文件、或与既有评测问题重复的数据。通过后生成的 `data/eval/holdout_approved.jsonl` 才可以作为 `scripts/evaluate_answers.py --questions ...` 的输入。

## 报告要求

报告必须同时列出：样本总量、每类题目分布、人工复核人数、拒答/澄清通过率、检索命中率、句子级证据覆盖率，以及人工正确性与证据支持评分。自动关键词指标和向量相似度不能单独作为“正确率”。
