#!/bin/bash
# ============================================================
# 消融实验：Chunk 切分方法对比
# 比较 section_aware（自有策略）/ recursive（LangChain）
#
# 用法:
#   bash scripts/ablation/cmp_chunk_method.sh
#
# 输出:
#   outputs/ablation/chunk_method/section_aware/
#   outputs/ablation/chunk_method/recursive/
#
# 注意: 修改 CHUNK_METHOD 后需要重新 build_chunks + build_index
#       如果已经分别构建过索引，可以跳过重建步骤
# ============================================================
set -euo pipefail

cd "$(dirname "$0")/../.."

echo "========================================"
echo "  消融实验：Chunk 方法对比"
echo "  section_aware vs recursive"
echo "========================================"

for method in section_aware recursive; do
  echo ""
  echo "── Chunk 方法: $method ──"

  # 通过环境变量覆盖 CHUNK_METHOD
  CHUNK_METHOD="$method" python scripts/run_eval.py \
    --mode all \
    --output-dir "outputs/ablation/chunk_method/$method"
done

echo ""
echo "========================================"
echo "  完成！结果在 outputs/ablation/chunk_method/"
echo "========================================"
