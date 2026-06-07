#!/usr/bin/env python3
"""
build_index.py — OnboardRAG 离线数据入库主脚本。

运行流程:
    PDF 文档 → 文本提取 → 文本清洗 → chunk 切分 → embedding 向量化 → 写入 Milvus

用法:
    # 处理所有 PDF，全链路入库
    python scripts/build_index.py

    # 只处理指定的一个 PDF
    python scripts/build_index.py --pdf "员工手册.pdf"

    # 跳过 embedding 和入库，只测试 PDF→chunk 前三步
    python scripts/build_index.py --pdf "OA系统使用手册.pdf" --skip-embed
"""

import argparse
import json
import sys
from pathlib import Path

# 将项目根目录加入 sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import RAW_PDFS_DIR, PROCESSED_DIR
from src.pdf_loader import load_pdfs
from src.cleaner import clean_pages
from src.chunker import chunk_pages


def _pages_to_dicts(pages) -> list[dict]:
    """将 PDFPage 对象列表转为字典列表（供 cleaner 使用）。"""
    return [p.to_dict() for p in pages]


def save_chunks_jsonl(chunks: list[dict], path: str | Path) -> None:
    """将 chunk 列表保存为 JSONL 文件。"""
    with open(path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")


def run_extract(pdf_files: list[str] | None = None) -> list[dict]:
    """Step 1-2: PDF 提取 + 文本清洗，返回 page dicts。"""
    print("\n[1/5] PDF 文本提取...")
    pages = load_pdfs(RAW_PDFS_DIR, filenames=pdf_files)
    if not pages:
        print("  没有找到匹配的 PDF 文件！")
        return []
    sources = sorted(set(p.source_file for p in pages))
    print(f"  读取了 {len(pages)} 页（来自 {len(sources)} 个 PDF: {', '.join(sources)}）")

    print("\n[2/5] 文本清洗...")
    page_dicts = _pages_to_dicts(pages)
    page_dicts = clean_pages(page_dicts)
    print(f"  清洗后剩余 {len(page_dicts)} 页（非空）")
    return page_dicts


def run_chunk(page_dicts: list[dict]) -> list[dict]:
    """Step 3: chunk 切分。"""
    print("\n[3/5] Chunk 切分...")
    chunks = chunk_pages(page_dicts)
    print(f"  切分为 {len(chunks)} 个 chunk")

    sizes = [len(c["text"]) for c in chunks]
    if sizes:
        print(f"  chunk 大小: 最小={min(sizes)}字, 最大={max(sizes)}字, "
              f"平均={sum(sizes)//len(sizes)}字")

    chunks_path = PROCESSED_DIR / "chunks.jsonl"
    save_chunks_jsonl(chunks, chunks_path)
    print(f"  已保存到 {chunks_path}")
    return chunks


def run_embed(chunks: list[dict]) -> list[list[float]]:
    """Step 4: embedding 向量化。"""
    from src.embedder import Embedder

    print("\n[4/5] Embedding 向量化...")
    embedder = Embedder()
    print(f"  模型: {embedder.model_name}, 维度: {embedder.dim}")
    vectors = embedder.encode_chunks(chunks)
    print(f"  生成了 {len(vectors)} 个向量")
    return vectors


def run_milvus(chunks: list[dict], vectors: list[list[float]]) -> int:
    """Step 5: 写入 Milvus。"""
    from src.milvus_store import (
        get_client,
        create_collection,
        insert_chunks,
        get_collection_stats,
    )

    print("\n[5/5] 写入 Milvus...")
    client = get_client()
    create_collection(client, drop_if_exists=True)

    count = insert_chunks(chunks, vectors, client)
    print(f"  入库 {count} 条记录")

    stats = get_collection_stats(client)
    print(f"\n{'=' * 60}")
    print(f"入库完成！Collection: {stats}")
    return count


def main():
    parser = argparse.ArgumentParser(
        description="OnboardRAG — 离线数据入库 Pipeline",
    )
    parser.add_argument(
        "--pdf", "-p",
        type=str,
        nargs="*",
        default=None,
        help="指定要处理的 PDF 文件名（可多个）。不指定则处理全部。"
             ' 例如: --pdf "员工手册.pdf" "OA系统使用手册.pdf"',
    )
    parser.add_argument(
        "--skip-embed",
        action="store_true",
        help="跳过 embedding 和 Milvus 入库，只生成 chunks.jsonl",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="重建 Milvus collection（删除已有数据）",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("OnboardRAG — 离线数据入库 Pipeline")
    print("=" * 60)
    if args.pdf:
        print(f"  指定 PDF: {args.pdf}")
    if args.skip_embed:
        print("  模式: 仅 chunk（跳过 embedding 和入库）")

    # ── Step 1-2: PDF 提取 + 清洗 ──────────────────
    page_dicts = run_extract(args.pdf)
    if not page_dicts:
        return 1

    # ── Step 3: Chunk 切分 ─────────────────────────
    chunks = run_chunk(page_dicts)
    if not chunks:
        print("  没有生成任何 chunk！")
        return 1

    # ── 如果 skip-embed，到此结束 ──────────────────
    if args.skip_embed:
        print(f"\n{'=' * 60}")
        print(f"chunk 完成！已保存 {len(chunks)} 条到 {PROCESSED_DIR / 'chunks.jsonl'}")
        return 0

    # ── Step 4: Embedding ──────────────────────────
    vectors = run_embed(chunks)

    # ── Step 5: Milvus 入库 ────────────────────────
    run_milvus(chunks, vectors)

    return 0


if __name__ == "__main__":
    sys.exit(main())
