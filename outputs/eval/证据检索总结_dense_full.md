# 证据级检索评测总结（检索模式: dense）

**匹配阈值:** 3-gram 包含度=0.75, ROUGE-L=0.65, 部分匹配率=0.85

## 方法说明

**Evidence Unit（证据单元）**：将原始 evidence 文本按编号条款（如 4.1 / 4.2 / 1) / 2) 等）拆分为独立的事实单元。每条 unit 是一个不可再分的证据陈述，更精细地衡量检索结果对标准答案的覆盖程度。

**匹配流程**：

1. 将 top-K 个检索 chunk 的文本拼接为检索上下文
2. 对每个 evidence unit，先做归一化精确子串匹配（全角→半角、去空白标点）
3. 精确匹配失败时，使用模糊匹配（3-gram 包含度 + ROUGE-L + 部分匹配率 三选一命中）

**核心指标**：

| 指标 | 定义 |
|------|------|
| `evidence_unit_total` | 该 query 的 evidence unit 总数 |
| `evidence_unit_hit@K` | top-K context 中命中的 unit 数量 |
| `evidence_unit_recall@K` | `hit@K / total`，命中 unit 比例 |
| `evidence_group_any_hit@K` | 至少 1 个 unit 命中（宽松） |
| `evidence_group_full_hit@K` | 全部 unit 命中（严格，= 本报告的 Hit@K） |
| `MRR` | 首次实现 `group_full_hit` 的最小 K 的倒数 |

> 下表中「精确 (Quote)」列基于原始 evidence quote 级别计算，「Evidence Unit」列基于上述 `group_full_hit` 定义。

## 总体指标

| 指标 | 精确 (Quote) | Evidence Unit |
|------|----------|----------|
| Hit@3 | 57.69% | 73.08% |
| Hit@5 | 65.38% | 84.62% |
| MRR | 0.5237 | 0.6724 |
| Recall@3 | 53.21% | 77.97% |
| Recall@5 | 60.90% | 92.31% |

## Evidence Unit 指标

| 指标 | 值 |
|------|----|
| 平均 Unit 数 | 3.3 |
| Unit Recall@3 | 77.97% |
| Unit Recall@5 | 92.31% |
| 平均命中 Unit@3 | 1.6 |
| 平均命中 Unit@5 | 2.4 |
| Group Any Hit@3 | 84.62% |
| Group Any Hit@5 | 96.15% |
| Group Full Hit@3 | 73.08% |
| Group Full Hit@5 | 84.62% |

## 按意图分组

| 意图 | 样本数 | 精确 Hit@5 | Evidence Unit Hit@5 | 精确 MRR | Evidence Unit MRR | Evidence Unit Recall@5 |
|------|--------|------------|------------|----------|----------|---------------|
| 人事与员工事务 | 8 | 87.50% | 87.50% | 0.7500 | 0.7708 | 93.48% |
| 账号与统一门户 | 6 | 66.67% | 100.00% | 0.6944 | 0.9167 | 100.00% |
| 内网访问 | 5 | 40.00% | 40.00% | 0.2400 | 0.1133 | 70.46% |
| 办公平台 | 7 | 57.14% | 100.00% | 0.3214 | 0.7500 | 100.00% |

## 按难度分组

| 难度 | 样本数 | 精确 Hit@5 | Evidence Unit Hit@5 | 精确 MRR | Evidence Unit MRR |
|------|--------|------------|------------|----------|----------|
| hard | 4 | 75.00% | 75.00% | 0.6250 | 0.4250 |
| medium | 12 | 41.67% | 83.33% | 0.3014 | 0.6764 |
| simple | 10 | 90.00% | 90.00% | 0.7500 | 0.7667 |

## 按评测关注点分组

| 关注点 | 样本数 | 精确 Hit@5 | Evidence Unit Hit@5 | 精确 MRR | Evidence Unit MRR |
|--------|--------|------------|------------|----------|----------|
| retrieval | 26 | 65.38% | 84.62% | 0.5237 | 0.6724 |
