# OnboardRAG 检索评测报告

**匹配方法:** exact

**阈值:** 3-gram 包含度=0.75, ROUGE-L=0.65, 部分匹配率=0.85

## 总体指标

### Query-level 指标

| 方法 | Hit@3 | Hit@5 | Hit@10 | Hit@30 | MRR |
|------|------|------|------|------|------|
| BM25 | 53.85% | 53.85% | 57.69% | 61.54% | 0.4357 |
| Dense | 50.00% | 57.69% | 65.38% | 65.38% | 0.4724 |
| RRF Hybrid | 53.85% | 57.69% | 61.54% | 65.38% | 0.5110 |

### Evidence Unit Recall

| 方法 | Recall@3 | Recall@5 | Recall@10 | Recall@30 |
|------|------|------|------|------|
| BM25 | 61.32% | 63.27% | 67.63% | 72.22% |
| Dense | 56.52% | 68.32% | 75.46% | 75.71% |
| RRF Hybrid | 59.78% | 67.37% | 72.22% | 77.25% |

### Evidence Group Full Hit

| 方法 | Hit@3 | Hit@5 | Hit@10 | Hit@30 |
|------|------|------|------|------|
| BM25 | 53.85% | 53.85% | 57.69% | 61.54% |
| Dense | 50.00% | 57.69% | 65.38% | 65.38% |
| RRF Hybrid | 53.85% | 57.69% | 61.54% | 65.38% |

## Candidate-stage 指标

| 指标 | 值 |
|------|----|
| BM25 Candidate Recall@30 | 72.22% |
| Dense Candidate Recall@30 | 75.71% |
| Union Candidate Recall@30 | 77.25% |
| RRF Recall@10 | 72.22% |
| Rerank Recall@10 | 0.00% |

## Overlap 分析

**基于 Evidence Group Full Hit@10**  (有效样本数: 26)

| 分类 | 数量 | 占比 |
|------|------|------|
| 两者都命中 (BM25 ✓, Dense ✓) | 15 | 57.7% |
| BM25 独有 (BM25 ✓, Dense ✗) | 0 | 0.0% |
| Dense 独有 (Dense ✓, BM25 ✗) | 2 | 7.7% |
| 两者都未命中 | 9 | 34.6% |
| 🔵 RRF Hybrid Gain (增量) | 0 | 0.0% |
| 🔴 RRF Hybrid Loss (损失) | 1 | 3.8% |
| 🟢 Reranker Gain (增量) | N/A | (未运行) |
| 🟠 Reranker Loss (损失) | N/A | (未运行) |

## Error 分析

| 错误类型 | 数量 | 说明 |
|------|------|------|
| union_miss | 0 | union_candidate@30 未命中：一阶段召回失败 |
| rerank_fail | 0 | union_candidate@30 命中但 rerank top10 未命中 |
| rerank_negative | 0 | RRF 命中但 rerank 未命中：reranker 负优化 |
| rerank_positive | 0 | RRF 未命中但 rerank 命中：reranker 正优化 |

## 按意图分组 (Evidence Group Full Hit@10)

| intent | 样本数 | BM25 Hit@10 | Dense Hit@10 | RRF Hybrid Hit@10 |
|------|------|------|------|------|
| 人事与员工事务 | 8 | 75.00% | 100.00% | 87.50% |
| 内网访问 | 5 | 20.00% | 20.00% | 20.00% |
| 办公平台 | 7 | 42.86% | 42.86% | 42.86% |
| 账号与统一门户 | 6 | 83.33% | 83.33% | 83.33% |

## 按难度分组 (Evidence Group Full Hit@10)

| difficulty | 样本数 | BM25 Hit@10 | Dense Hit@10 | RRF Hybrid Hit@10 |
|------|------|------|------|------|
| hard | 4 | 25.00% | 25.00% | 25.00% |
| medium | 12 | 50.00% | 58.33% | 58.33% |
| simple | 10 | 80.00% | 90.00% | 80.00% |
