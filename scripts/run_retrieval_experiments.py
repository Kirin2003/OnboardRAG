#!/usr/bin/env python3
"""
run_retrieval_experiments.py — 阶段一：检索实验

对评测集中的每个 query，分别运行并保存以下 5 种方法的结果：
  1. BM25-only          — BM25 top-30 raw results
  2. Dense-only         — Dense top-30 raw results
  3. RRF Hybrid         — Dense + BM25 → RRF fusion → top-30
  4. Union Candidate    — Dense + BM25 去重合并 → candidate pool
  5. Hybrid + Reranker  — Union candidate → reranker 精排 → top-30

结果保存为 JSONL，每条包含 query_id、method、config、results（含 chunk text）。

用法:
    # 默认参数
    python scripts/run_retrieval_experiments.py

    # 指定输出目录和评测集
    python scripts/run_retrieval_experiments.py \\
        --output-dir outputs/retrieval_results/ \\
        --evalset data/eval/eval_queries_v2.jsonl

    # 跳过 reranker（仅跑 bm25/dense/rrf/union）
    python scripts/run_retrieval_experiments.py --no-reranker
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.retriever import Retriever
from src.config import (
    EVAL_BM25_TOP_K,
    EVAL_DENSE_TOP_K,
    EVAL_RRF_K,
    EVAL_RERANK_TOP_K,
    ENABLE_RERANKER,
    RERANKER_MODEL,
    EMBEDDING_MODEL_NAME,
    MILVUS_COLLECTION_NAME,
)


def _build_result_entry(
    chunk: dict,
    rank: int,
    method: str,
    score: float,
) -> dict:
    """将检索返回的 chunk 转为统一的 result entry 格式。

    不同 method 在对应 score 字段填值，其余 score 字段为 null。
    """
    entry = {
        "rank": rank,
        "doc_id": chunk.get("chunk_id", ""),
        "chunk_id": chunk.get("chunk_id", ""),
        "document_name": chunk.get("source_file", ""),
        "section": chunk.get("doc_title", ""),
        "text": chunk.get("body_text") or chunk.get("text", ""),
        "score": score,
        "bm25_score": None,
        "dense_score": None,
        "rrf_score": None,
        "rerank_score": None,
    }

    # 根据 method 设置对应 score 字段
    if method == "bm25":
        entry["bm25_score"] = score
    elif method == "dense":
        entry["dense_score"] = score
    elif method in ("rrf_hybrid", "union_candidate"):
        entry["rrf_score"] = score
    elif method == "hybrid_rerank":
        entry["rerank_score"] = score
        # 同时保留 rrf_score（如果有）
        if chunk.get("_rrf_score") is not None:
            entry["rrf_score"] = chunk["_rrf_score"]

    return entry


def _chunk_id_key(chunk: dict) -> str:
    """获取 chunk 的唯一标识，用于去重。"""
    return chunk.get("chunk_id", "")


def _dedup_chunks(chunks: list[dict]) -> list[dict]:
    """按 chunk_id 去重，保留首次出现的 chunk。"""
    seen = set()
    result = []
    for c in chunks:
        cid = _chunk_id_key(c)
        if cid and cid not in seen:
            seen.add(cid)
            result.append(c)
    return result


def _union_candidate_pool(
    dense_chunks: list[dict],
    bm25_chunks: list[dict],
) -> list[dict]:
    """构建 union candidate pool：合并 dense + bm25，按 chunk_id 去重。

    保留原始来源分数，按 chunk_id 去重时优先保留 dense 侧（先加入）。
    """
    # 先加 dense，再加 bm25（dense 优先）
    merged = list(dense_chunks)
    dense_ids = {_chunk_id_key(c) for c in dense_chunks}
    for c in bm25_chunks:
        cid = _chunk_id_key(c)
        if cid and cid not in dense_ids:
            merged.append(c)
            dense_ids.add(cid)

    return merged


def run_experiments(
    entries: list[dict],
    output_path: Path,
    skip_reranker: bool = False,
) -> Path:
    """运行检索实验，将结果写入 JSONL 文件。

    Args:
        entries: 评测样本列表
        output_path: JSONL 输出路径
        skip_reranker: 是否跳过 reranker 步骤

    Returns:
        输出文件路径
    """
    retriever = Retriever()
    timestamp = datetime.now(timezone.utc).isoformat()

    config_base = {
        "bm25_top_k": EVAL_BM25_TOP_K,
        "dense_top_k": EVAL_DENSE_TOP_K,
        "rrf_k": EVAL_RRF_K,
        "reranker_enabled": ENABLE_RERANKER and not skip_reranker,
        "reranker_model": RERANKER_MODEL if (ENABLE_RERANKER and not skip_reranker) else None,
        "embedding_model": EMBEDDING_MODEL_NAME,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    total_queries = len(entries)
    written = 0

    with open(output_path, "w", encoding="utf-8") as f:
        for i, entry in enumerate(entries):
            query_id = entry["id"]
            query = entry["query"]
            answerable = entry.get("answerable", True)

            print(f"\n[{i+1}/{total_queries}] {query_id}: {query[:60]}...")

            if not answerable:
                # 不可回答的 query 写入占位行
                for method in ["bm25", "dense", "rrf_hybrid", "union_candidate", "hybrid_rerank"]:
                    if skip_reranker and method == "hybrid_rerank":
                        continue
                    row = {
                        "query_id": query_id,
                        "query": query,
                        "method": method,
                        "config": config_base,
                        "timestamp": timestamp,
                        "evalset_file": "eval_queries_v2.jsonl",
                        "index_collection": MILVUS_COLLECTION_NAME,
                        "answerable": False,
                        "results": [],
                    }
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    written += 1
                continue

            # ── 1. BM25-only ──
            print("  → BM25...", end=" ", flush=True)
            bm25_chunks = retriever._bm25_search(query, top_k=EVAL_BM25_TOP_K)
            bm25_results = [
                _build_result_entry(c, rank, "bm25", c.get("score", 0))
                for rank, c in enumerate(bm25_chunks, start=1)
            ]
            print(f"{len(bm25_results)} chunks")

            # ── 2. Dense-only ──
            print("  → Dense...", end=" ", flush=True)
            dense_chunks = retriever._dense_search(query, top_k=EVAL_DENSE_TOP_K)
            dense_results = [
                _build_result_entry(c, rank, "dense", c.get("score", 0))
                for rank, c in enumerate(dense_chunks, start=1)
            ]
            print(f"{len(dense_results)} chunks")

            # ── 3. RRF Hybrid ──
            print("  → RRF Hybrid...", end=" ", flush=True)
            rrf_merged = retriever._rrf_merge(dense_chunks, bm25_chunks, k=EVAL_RRF_K)
            rrf_top30 = rrf_merged[:EVAL_BM25_TOP_K]  # top 30
            rrf_results = [
                _build_result_entry(c, rank, "rrf_hybrid", c.get("score", 0))
                for rank, c in enumerate(rrf_top30, start=1)
            ]
            print(f"{len(rrf_results)} chunks")

            # ── 4. Union Candidate Pool ──
            print("  → Union Candidate...", end=" ", flush=True)
            union_chunks = _union_candidate_pool(dense_chunks, bm25_chunks)
            union_results = [
                _build_result_entry(c, rank, "union_candidate", c.get("score", 0))
                for rank, c in enumerate(union_chunks, start=1)
            ]
            print(f"{len(union_results)} chunks")

            # ── 5. Hybrid + Reranker ──
            rerank_results = []
            if not skip_reranker:
                print("  → Reranker...", end=" ", flush=True)
                reranked = retriever._reranker.rerank(
                    query, union_chunks, top_k=EVAL_RERANK_TOP_K
                )
                rerank_results = [
                    _build_result_entry(c, rank, "hybrid_rerank", c.get("rerank_score", 0))
                    for rank, c in enumerate(reranked, start=1)
                ]
                print(f"{len(rerank_results)} chunks")
            else:
                print("  → Reranker (skipped)")

            # ── 写入 JSONL ──
            methods = [
                ("bm25", bm25_results),
                ("dense", dense_results),
                ("rrf_hybrid", rrf_results),
                ("union_candidate", union_results),
            ]
            if not skip_reranker:
                methods.append(("hybrid_rerank", rerank_results))

            for method, results in methods:
                row = {
                    "query_id": query_id,
                    "query": query,
                    "method": method,
                    "config": config_base,
                    "timestamp": timestamp,
                    "evalset_file": "eval_queries_v2.jsonl",
                    "index_collection": MILVUS_COLLECTION_NAME,
                    "answerable": True,
                    "results": results,
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1

    print(f"\n{'='*60}")
    print(f"  检索实验完成")
    print(f"  总 query 数: {total_queries}")
    print(f"  写入行数: {written}")
    print(f"  输出文件: {output_path}")
    print(f"{'='*60}")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="OnboardRAG 检索实验 — 阶段一：批量运行所有检索方法并保存结果"
    )
    parser.add_argument(
        "--evalset",
        type=str,
        default="data/eval/eval_queries_v2.jsonl",
        help="评测集 JSONL 文件路径（默认: data/eval/eval_queries_v2.jsonl）",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/retrieval_results",
        help="输出目录（默认: outputs/retrieval_results/）",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="输出文件名（不含路径）。默认使用时间戳命名: retrieval_results_YYYYMMDD_HHMMSS.jsonl",
    )
    parser.add_argument(
        "--no-reranker",
        action="store_true",
        help="跳过 reranker 步骤（仅跑 bm25 / dense / rrf / union）",
    )
    args = parser.parse_args()

    # 加载评测集
    evalset_path = PROJECT_ROOT / args.evalset
    if not evalset_path.exists():
        print(f"错误: 评测文件不存在: {evalset_path}")
        return 1

    with open(evalset_path, "r", encoding="utf-8") as f:
        entries = [json.loads(line) for line in f if line.strip()]

    print(f"加载了 {len(entries)} 条评测样本")
    print(f"Reranker: {'启用' if ENABLE_RERANKER and not args.no_reranker else '禁用'}")
    print(f"输出目录: {args.output_dir}")

    # 确定输出文件名
    if args.output_name:
        output_name = args.output_name
    else:
        from datetime import datetime as dt
        ts = dt.now().strftime("%Y%m%d_%H%M%S")
        rerank_tag = "no_rerank" if args.no_reranker else "full"
        output_name = f"retrieval_results_{rerank_tag}_{ts}.jsonl"

    output_path = PROJECT_ROOT / args.output_dir / output_name

    run_experiments(entries, output_path, skip_reranker=args.no_reranker)

    return 0


if __name__ == "__main__":
    sys.exit(main())
