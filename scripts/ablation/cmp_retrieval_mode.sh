#!/bin/bash
# ============================================================
# 消融实验：检索模式对比
# 比较 hybrid（混合检索）/ bm25（仅关键词）/ dense（仅向量）
#
# 用法:
#   bash scripts/ablation/cmp_retrieval_mode.sh
#
# 输出:
#   outputs/ablation/retrieval_mode/hybrid/
#   outputs/ablation/retrieval_mode/bm25/
#   outputs/ablation/retrieval_mode/dense/
# ============================================================
set -euo pipefail

cd "$(dirname "$0")/../.."

echo "========================================"
echo "  消融实验：检索模式对比"
echo "  hybrid vs bm25 vs dense"
echo "========================================"

for mode in hybrid bm25 dense; do
  echo ""
  echo "── 检索模式: $mode ──"
  python scripts/run_eval.py \
    --mode all \
    --retrieval-mode "$mode" \
    --output-dir "outputs/ablation/retrieval_mode/$mode"
done

echo ""
echo "========================================"
echo "  完成！结果在 outputs/ablation/retrieval_mode/"
echo "========================================"
