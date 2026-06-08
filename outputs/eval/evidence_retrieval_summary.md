# Evidence-Level Retrieval Evaluation Summary

**Thresholds:** containment=0.75, rouge_l=0.65, partial_ratio=0.85

## Overall Metrics

| Metric | Exact | Fuzzy |
|--------|-------|-------|
| Hit@3 | 57.69% | 80.77% |
| Hit@5 | 61.54% | 84.62% |
| MRR | 0.5224 | 0.7724 |
| Recall@3 | 55.77% | 80.77% |
| Recall@5 | 59.62% | 84.62% |

## By Intent

| Intent | Count | Exact Hit@5 | Fuzzy Hit@5 | Exact MRR | Fuzzy MRR | Fuzzy Recall@5 |
|--------|-------|-------------|-------------|-----------|-----------|----------------|
| 人事与员工事务 | 8 | 75.00% | 75.00% | 0.7500 | 0.7500 | 75.00% |
| 账号与统一门户 | 6 | 83.33% | 100.00% | 0.6389 | 0.8056 | 100.00% |
| 内网访问 | 5 | 20.00% | 60.00% | 0.2000 | 0.6000 | 60.00% |
| 办公平台 | 7 | 57.14% | 100.00% | 0.3929 | 0.8929 | 100.00% |

## By Difficulty

| Difficulty | Count | Exact Hit@5 | Fuzzy Hit@5 | Exact MRR | Fuzzy MRR |
|------------|-------|-------------|-------------|-----------|----------|
| hard | 3 | 66.67% | 100.00% | 0.5000 | 1.0000 |
| medium | 12 | 50.00% | 91.67% | 0.3958 | 0.8125 |
| simple | 11 | 72.73% | 72.73% | 0.6667 | 0.6667 |

## By Eval Focus

| Focus | Count | Exact Hit@5 | Fuzzy Hit@5 | Exact MRR | Fuzzy MRR |
|-------|-------|-------------|-------------|-----------|----------|
| retrieval | 26 | 61.54% | 84.62% | 0.5224 | 0.7724 |
