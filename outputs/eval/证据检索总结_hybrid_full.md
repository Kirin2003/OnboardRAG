# 证据级检索评测总结（检索模式: hybrid）

**匹配阈值:** 3-gram 包含度=0.75, ROUGE-L=0.65, 部分匹配率=0.85

## 方法说明

**Evidence Unit（证据单元）**：将原始 evidence 文本按编号条款（如 4.1 / 4.2 / 1) / 2) 等）拆分为独立的事实单元。每条 unit 是一个不可再分的证据陈述，更精细地衡量检索结果对标准答案的覆盖程度。

**匹配流程**：

1. 将 top-K 个检索 chunk 的文本拼接为检索上下文
2. 对每个 evidence unit，做归一化精确子串匹配（全角→半角、去空白标点）
3. 「精确 (Unit)」列仅使用步骤 2 的精确匹配结果
4. 「Evidence Unit」列在精确匹配失败时，补充模糊匹配（3-gram 包含度 + ROUGE-L + 部分匹配率 三选一命中）

**核心指标**：

| 指标 | 定义 |
|------|------|
| `evidence_unit_total` | 该 query 的 evidence unit 总数 |
| `evidence_unit_hit@K` | top-K context 中命中的 unit 数量 |
| `evidence_unit_recall@K` | `hit@K / total`，命中 unit 比例 |
| `evidence_group_any_hit@K` | 至少 1 个 unit 命中（宽松） |
| `evidence_group_full_hit@K` | 全部 unit 命中（严格，= 本报告的 Hit@K） |
| `MRR` | 首次实现 `group_full_hit` 的最小 K 的倒数 |

> 下表中「精确 (Unit)」列基于 evidence unit 级别的纯精确匹配（归一化子串匹配），
> 「Evidence Unit」列在精确匹配基础上，对未命中的 unit 使用模糊匹配进行补充判断。
> 两列均以 `group_full_hit`（全部 unit 命中）作为 Hit@K 标准。

## 总体指标

| 指标 | 精确 (Unit) | Evidence Unit |
|------|----------|----------|
| Hit@3 | 61.54% | 80.77% |
| Hit@5 | 61.54% | 80.77% |
| MRR | 0.5440 | 0.7209 |
| Recall@3 | 67.96% | 87.62% |
| Recall@5 | 69.54% | 89.50% |

## Evidence Unit 指标

| 指标 | 值 |
|------|----|
| 平均 Unit 数 | 3.3 |
| Unit Recall@3 | 87.62% |
| Unit Recall@5 | 89.50% |
| 平均命中 Unit@3 | 2.4 |
| 平均命中 Unit@5 | 2.5 |
| Group Any Hit@3 | 96.15% |
| Group Any Hit@5 | 96.15% |
| Group Full Hit@3 | 80.77% |
| Group Full Hit@5 | 80.77% |

## 按意图分组

| 意图 | 样本数 | 精确 Hit@5 | Evidence Unit Hit@5 | 精确 MRR | Evidence Unit MRR | Evidence Unit Recall@5 |
|------|--------|------------|------------|----------|----------|---------------|
| 人事与员工事务 | 8 | 100.00% | 100.00% | 0.8333 | 0.8333 | 100.00% |
| 账号与统一门户 | 6 | 66.67% | 83.33% | 0.6905 | 0.8571 | 83.33% |
| 内网访问 | 5 | 20.00% | 20.00% | 0.2000 | 0.2200 | 65.38% |
| 办公平台 | 7 | 42.86% | 100.00% | 0.3333 | 0.8333 | 100.00% |

## 按难度分组

| 难度 | 样本数 | 精确 Hit@5 | Evidence Unit Hit@5 | 精确 MRR | Evidence Unit MRR |
|------|--------|------------|------------|----------|----------|
| hard | 4 | 25.00% | 50.00% | 0.2500 | 0.3750 |
| medium | 12 | 58.33% | 91.67% | 0.4167 | 0.7583 |
| simple | 10 | 80.00% | 80.00% | 0.8143 | 0.8143 |

## 按评测关注点分组

| 关注点 | 样本数 | 精确 Hit@5 | Evidence Unit Hit@5 | 精确 MRR | Evidence Unit MRR |
|--------|--------|------------|------------|----------|----------|
| retrieval | 26 | 61.54% | 80.77% | 0.5440 | 0.7209 |
