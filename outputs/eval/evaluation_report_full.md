# OnboardRAG 检索评测报告

**匹配方法:** fuzzy

**阈值:** 3-gram 包含度=0.75, ROUGE-L=0.65, 部分匹配率=0.85

## 总体指标

### Query-level 指标

| 方法 | Hit@3 | Hit@5 | Hit@10 | Hit@30 | MRR |
|------|------|------|------|------|------|
| BM25 | 76.92% | 76.92% | 80.77% | 92.31% | 0.6079 |
| Dense | 73.08% | 84.62% | 92.31% | 92.31% | 0.6724 |
| RRF Hybrid | 73.08% | 76.92% | 92.31% | 100.00% | 0.7021 |
| Hybrid + Reranker | 80.77% | 84.62% | 92.31% | 100.00% | 0.7402 |

### Evidence Unit Recall

| 方法 | Recall@3 | Recall@5 | Recall@10 | Recall@30 |
|------|------|------|------|------|
| BM25 | 81.74% | 84.69% | 89.60% | 94.97% |
| Dense | 77.97% | 92.31% | 95.38% | 95.64% |
| RRF Hybrid | 77.68% | 86.52% | 94.97% | 100.00% |
| Hybrid + Reranker | 88.06% | 95.89% | 98.21% | 100.00% |

### Evidence Group Full Hit

| 方法 | Hit@3 | Hit@5 | Hit@10 | Hit@30 |
|------|------|------|------|------|
| BM25 | 76.92% | 76.92% | 80.77% | 92.31% |
| Dense | 73.08% | 84.62% | 92.31% | 92.31% |
| RRF Hybrid | 73.08% | 76.92% | 92.31% | 100.00% |
| Hybrid + Reranker | 80.77% | 84.62% | 92.31% | 100.00% |

## Candidate-stage 指标

| 指标 | 值 |
|------|----|
| BM25 Candidate Recall@30 | 94.97% |
| Dense Candidate Recall@30 | 95.64% |
| Union Candidate Recall@30 | 100.00% |
| RRF Recall@10 | 94.97% |
| Rerank Recall@10 | 98.21% |

## Overlap 分析

**基于 Evidence Group Full Hit@10**  (有效样本数: 26)

| 分类 | 数量 | 占比 |
|------|------|------|
| 两者都命中 (BM25 ✓, Dense ✓) | 20 | 76.9% |
| BM25 独有 (BM25 ✓, Dense ✗) | 1 | 3.8% |
| Dense 独有 (Dense ✓, BM25 ✗) | 4 | 15.4% |
| 两者都未命中 | 1 | 3.8% |
| 🔵 RRF Hybrid Gain (增量) | 1 | 3.8% |
| 🔴 RRF Hybrid Loss (损失) | 2 | 7.7% |
| 🟢 Reranker Gain (增量) | 2 | 7.7% |
| 🟠 Reranker Loss (损失) | 2 | 7.7% |

## Error 分析

| 错误类型 | 数量 | 说明 |
|------|------|------|
| union_miss | 0 | union_candidate@30 未命中：一阶段召回失败 |
| rerank_fail | 0 | union_candidate@30 命中但 rerank top10 未命中 |
| rerank_negative | 2 | RRF 命中但 rerank 未命中：reranker 负优化 |
| rerank_positive | 2 | RRF 未命中但 rerank 命中：reranker 正优化 |

## 按意图分组 (Evidence Group Full Hit@10)

| intent | 样本数 | BM25 Hit@10 | Dense Hit@10 | RRF Hybrid Hit@10 | Hybrid + Reranker Hit@10 |
|------|------|------|------|------|------|
| 人事与员工事务 | 8 | 75.00% | 100.00% | 87.50% | 100.00% |
| 内网访问 | 5 | 40.00% | 60.00% | 80.00% | 60.00% |
| 办公平台 | 7 | 100.00% | 100.00% | 100.00% | 100.00% |
| 账号与统一门户 | 6 | 100.00% | 100.00% | 100.00% | 100.00% |

## 按难度分组 (Evidence Group Full Hit@10)

| difficulty | 样本数 | BM25 Hit@10 | Dense Hit@10 | RRF Hybrid Hit@10 | Hybrid + Reranker Hit@10 |
|------|------|------|------|------|------|
| hard | 4 | 50.00% | 75.00% | 100.00% | 75.00% |
| medium | 12 | 91.67% | 91.67% | 100.00% | 91.67% |
| simple | 10 | 80.00% | 100.00% | 80.00% | 100.00% |
