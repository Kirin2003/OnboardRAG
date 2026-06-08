#!/usr/bin/env python3
"""
debug_dense.py — Dense 检索调试工具

逐步诊断 dense 检索管线中的每个环节，用于排查 dense 检索结果为 0% 的问题。

用法:
    python scripts/debug_dense.py
    python scripts/debug_dense.py --query "VPN 连不上怎么办"
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def step_separator(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


# ═══════════════════════════════════════════════════════════════
# Step 1: Milvus Collection 状态
# ═══════════════════════════════════════════════════════════════

def check_milvus() -> None:
    from src.milvus_store import get_client, get_collection_stats, MILVUS_COLLECTION_NAME

    step_separator("Step 1: Milvus Collection 状态")

    client = get_client()
    has = client.has_collection(MILVUS_COLLECTION_NAME)
    print(f"  Collection '{MILVUS_COLLECTION_NAME}' 存在: {has}")

    stats = get_collection_stats(client)
    print(f"  记录总数: {stats.get('total', '?')}")
    print(f"  完整 stats: {stats}")

    if not has or stats.get("total", 0) == 0:
        print("  ⚠️  WARNING: Collection 为空或不存在！请先运行 build_index.py")


# ═══════════════════════════════════════════════════════════════
# Step 2: Embedding 测试
# ═══════════════════════════════════════════════════════════════

def check_embedding(query: str) -> list[float]:
    from src.embedder import Embedder

    step_separator("Step 2: Query Embedding")

    embedder = Embedder()
    print(f"  模型: {embedder.model_name}")
    print(f"  维度: {embedder.dim}")
    print(f"  Query: {query}")

    vec = embedder.encode([query], show_progress=False)[0]
    print(f"  向量维度: {len(vec)}")
    print(f"  向量前 5 个分量: {[round(v, 6) for v in vec[:5]]}")
    print(f"  向量范围: [{min(vec):.4f}, {max(vec):.4f}]")

    # 检查是否有 NaN 或全零
    if all(v == 0.0 for v in vec):
        print("  ⚠️  WARNING: 向量全为零！")
    if any(v != v for v in vec):  # NaN check
        print("  ⚠️  WARNING: 向量包含 NaN！")

    return vec


# ═══════════════════════════════════════════════════════════════
# Step 3: Dense 检索
# ═══════════════════════════════════════════════════════════════

def check_dense_search(query: str) -> list[dict]:
    from src.retriever import Retriever

    step_separator("Step 3: Dense 检索结果")

    retriever = Retriever()
    chunks = retriever.retrieve(query, top_k=5, mode="dense")

    print(f"  返回 chunk 数: {len(chunks)}")
    if not chunks:
        print("  ⚠️  WARNING: Dense 检索返回 0 条结果！")
        return []

    for i, c in enumerate(chunks):
        print(f"\n  ── Chunk {i+1} ──")
        print(f"    chunk_id:     {c.get('chunk_id', 'MISSING')}")
        print(f"    source_file:  {c.get('source_file', 'MISSING')}")
        print(f"    doc_title:    {c.get('doc_title', 'MISSING')[:50]}")
        print(f"    category:     {c.get('category', 'MISSING')}")
        print(f"    score:        {c.get('score', 'MISSING')}")
        print(f"    page:         {c.get('page_start')}-{c.get('page_end')}")
        print(f"    body_text:    {'✓ 有内容' if c.get('body_text') else '✗ 空/不存在'} "
              f"(len={len(c.get('body_text', ''))})")
        print(f"    text:         {'✓ 有内容' if c.get('text') else '✗ 空/不存在'} "
              f"(len={len(c.get('text', ''))})")

        # 打印 text 前 100 字符
        txt = c.get("text", "")
        if txt:
            print(f"    text 预览:    {txt[:100]}...")

    return chunks


# ═══════════════════════════════════════════════════════════════
# Step 4: 关键字段对比（_dense_search vs _bm25_search）
# ═══════════════════════════════════════════════════════════════

def check_field_consistency(query: str) -> None:
    from src.retriever import Retriever

    step_separator("Step 4: Dense vs BM25 字段对比")

    retriever = Retriever()
    dense_chunks = retriever.retrieve(query, top_k=3, mode="dense")
    bm25_chunks = retriever.retrieve(query, top_k=3, mode="bm25")

    fields_to_check = ["body_text", "text", "chunk_id", "source_file", "score"]
    print(f"  {'Field':<16} {'Dense':<15} {'BM25':<15}")
    print(f"  {'-' * 16} {'-' * 15} {'-' * 15}")

    for field in fields_to_check:
        if dense_chunks and bm25_chunks:
            d_val = dense_chunks[0].get(field, "N/A")
            b_val = bm25_chunks[0].get(field, "N/A")

            if isinstance(d_val, str):
                d_display = f"len={len(d_val)}" if d_val else "空字符串"
            elif isinstance(d_val, float):
                d_display = f"{d_val:.4f}"
            else:
                d_display = str(d_val)[:15]

            if isinstance(b_val, str):
                b_display = f"len={len(b_val)}" if b_val else "空字符串"
            elif isinstance(b_val, float):
                b_display = f"{b_val:.4f}"
            else:
                b_display = str(b_val)[:15]

            status = "⚠️ " if d_display != b_display and field == "body_text" else ""
            print(f"  {status}{field:<16} {d_display:<15} {b_display:<15}")

    # 检查 fallback 行为
    if dense_chunks:
        dc = dense_chunks[0]
        via_get = dc.get("body_text", dc.get("text", ""))
        via_or = dc.get("body_text") or dc.get("text", "")
        print(f"\n  body_text fallback 测试（Dense 第一个 chunk）:")
        print(f"    .get('body_text', .get('text')) → len={len(via_get)}")
        print(f"    .get('body_text') or .get('text') → len={len(via_or)}")


# ═══════════════════════════════════════════════════════════════
# Step 5: Evidence 匹配测试
# ═══════════════════════════════════════════════════════════════

def check_evidence_match(chunks: list[dict], query: str) -> None:
    from src.eval_metrics import normalize_text, exact_evidence_match, fuzzy_evidence_match

    step_separator("Step 5: Evidence 匹配测试")

    # 从评测集中找一个跟 query 相关的 evidence
    eval_path = PROJECT_ROOT / "data/eval/eval_queries_v2.jsonl"
    if not eval_path.exists():
        print("  评测文件不存在，跳过")
        return

    with open(eval_path, "r", encoding="utf-8") as f:
        entries = [json.loads(line) for line in f if line.strip()]

    # 找一个包含关键词的 answerable 样本
    test_entry = None
    for entry in entries:
        if entry.get("answerable") and entry.get("evidence"):
            test_entry = entry
            break

    if not test_entry:
        print("  没有可用的评测样本，跳过")
        return

    eid = test_entry["id"]
    evidence_list = test_entry.get("evidence", [])
    print(f"  测试样本: {eid}")
    print(f"  Query: {test_entry['query'][:80]}")
    print(f"  Evidence 条数: {len(evidence_list)}")

    for i, ev in enumerate(evidence_list[:3]):
        quote = ev.get("quote", "")
        print(f"\n  Evidence [{i}]: {quote[:100]}...")

        # 对每个 chunk 测试匹配
        for rank, chunk in enumerate(chunks[:5], start=1):
            chunk_text = chunk.get("body_text") or chunk.get("text", "")
            exact = exact_evidence_match(quote, chunk_text)
            fuzzy, scores = fuzzy_evidence_match(quote, chunk_text)

            status = "✓ match" if exact or fuzzy else "✗ no match"
            print(f"    chunk {rank} ({chunk.get('source_file', '?')}): {status}  "
                  f"containment={scores.get('containment', 0):.3f}  "
                  f"rouge_l={scores.get('rouge_l', 0):.3f}  "
                  f"partial_ratio={scores.get('partial_ratio', 0):.3f}")


# ═══════════════════════════════════════════════════════════════
# Step 6: Dense vs BM25 排序差异
# ═══════════════════════════════════════════════════════════════

def check_rank_comparison(query: str) -> None:
    from src.retriever import Retriever

    step_separator("Step 6: Dense vs BM25 排序对比")

    retriever = Retriever()
    dense = retriever.retrieve(query, top_k=5, mode="dense")
    bm25 = retriever.retrieve(query, top_k=5, mode="bm25")

    print(f"  {'Rank':<6} {'Dense chunk_id':<30} {'BM25 chunk_id':<30}")
    print(f"  {'-' * 6} {'-' * 30} {'-' * 30}")
    for i in range(5):
        d_id = dense[i]["chunk_id"] if i < len(dense) else "—"
        b_id = bm25[i]["chunk_id"] if i < len(bm25) else "—"
        same = "← 相同" if d_id == b_id and d_id != "—" else ""
        print(f"  {i+1:<6} {d_id:<30} {b_id:<30} {same}")


# ═══════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Dense 检索调试工具")
    parser.add_argument(
        "--query", "-q",
        type=str,
        default="VPN 连不上怎么办",
        help="测试查询文本（默认: VPN 连不上怎么办）",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  Dense 检索调试工具")
    print(f"  Query: {args.query}")
    print("=" * 60)

    check_milvus()
    check_embedding(args.query)
    chunks = check_dense_search(args.query)
    check_field_consistency(args.query)
    if chunks:
        check_evidence_match(chunks, args.query)
    check_rank_comparison(args.query)

    print("\n" + "=" * 60)
    print("  调试完成")
    print("=" * 60)


if __name__ == "__main__":
    sys.exit(main())
