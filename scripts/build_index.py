#!/usr/bin/env python3
"""
build_index.py — OnboardRAG 离线数据入库主脚本。

运行流程:
    PDF 文档 → 文本提取 → 文本清洗 → chunk 切分 → embedding 向量化 → 写入 Milvus
"""

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
from src.embedder import Embedder
from src.milvus_store import get_client, create_collection, insert_chunks, get_collection_stats


def _pages_to_dicts(pages) -> list[dict]:
    """将 PDFPage 对象列表转为字典列表（供 cleaner 使用）。"""
    return [p.to_dict() for p in pages]


def save_chunks_jsonl(chunks: list[dict], path: str | Path) -> None:
    """将 chunk 列表保存为 JSONL 文件，方便调试和后续复用。"""
    with open(path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")


def main():
    print("=" * 60)
    print("OnboardRAG — 离线数据入库 Pipeline")
    print("=" * 60)

    # ── Step 1: PDF 文本提取 ──────────────────────────
    print("\n[1/5] PDF 文本提取...")
    pages = load_pdfs(RAW_PDFS_DIR)
    print(f"  读取了 {len(pages)} 页（来自 {len(set(p.source_file for p in pages))} 个 PDF）")

    # ── Step 2: 文本清洗 ───────────────────────────────
    print("\n[2/5] 文本清洗...")
    page_dicts = _pages_to_dicts(pages)
    page_dicts = clean_pages(page_dicts)
    print(f"  清洗后剩余 {len(page_dicts)} 页（非空）")

    # ── Step 3: Chunk 切分 ─────────────────────────────
    print("\n[3/5] Chunk 切分...")
    chunks = chunk_pages(page_dicts)
    print(f"  切分为 {len(chunks)} 个 chunk")

    # 打印 chunk 大小分布
    sizes = [len(c["text"]) for c in chunks]
    print(f"  chunk 大小: 最小={min(sizes)}字, 最大={max(sizes)}字, "
          f"平均={sum(sizes)//len(sizes)}字")

    # 保存 chunks.jsonl
    chunks_path = PROCESSED_DIR / "chunks.jsonl"
    save_chunks_jsonl(chunks, chunks_path)
    print(f"  已保存到 {chunks_path}")

    # ── Step 4: Embedding 向量化 ───────────────────────
    print("\n[4/5] Embedding 向量化...")
    embedder = Embedder()
    print(f"  模型: {embedder.model_name}, 维度: {embedder.dim}, "
          f"设备: {embedder.device}")
    vectors = embedder.encode_chunks(chunks)
    print(f"  生成了 {len(vectors)} 个向量")

    # ── Step 5: 写入 Milvus ────────────────────────────
    print("\n[5/5] 写入 Milvus...")
    client = get_client()
    create_collection(client, drop_if_exists=True)

    count = insert_chunks(chunks, vectors, client)
    print(f"  入库 {count} 条记录")

    # ── 统计信息 ───────────────────────────────────────
    stats = get_collection_stats(client)
    print(f"\n{'=' * 60}")
    print(f"入库完成！Collection: {stats}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
