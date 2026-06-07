#!/usr/bin/env python3
"""
build_index.py — 从 PROCESSED_DIR 读取所有 chunk 文件，embedding 向量化后写入 Milvus。

运行流程:
    读取 PROCESSED_DIR/*_chunks.jsonl → embedding 向量化 → 写入 Milvus

用法:
    # 全量入库（需先运行 build_chunks.py）
    python scripts/build_index.py

    # 重建 collection
    python scripts/build_index.py --rebuild
"""

import argparse
import json
import sys
from pathlib import Path

# 将项目根目录加入 sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PROCESSED_DIR


def load_chunks_jsonl(path: str | Path) -> list[dict]:
    """从 JSONL 文件读取 chunk 列表。"""
    chunks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def load_all_chunks() -> list[dict]:
    """从 PROCESSED_DIR 读取所有 *_chunks.jsonl 文件，合并返回。"""
    chunk_files = sorted(PROCESSED_DIR.glob("*_chunks.jsonl"))
    if not chunk_files:
        raise FileNotFoundError(
            f"PROCESSED_DIR 下没有找到 *_chunks.jsonl 文件\n"
            f"  请先运行: python scripts/build_chunks.py"
        )

    all_chunks = []
    for path in chunk_files:
        chunks = load_chunks_jsonl(path)
        print(f"  {path.name}  — {len(chunks)} chunks")
        all_chunks.extend(chunks)

    return all_chunks


def run_embed() -> tuple[list[dict], list[list[float]]]:
    """Step 1: 从 PROCESSED_DIR 读取所有 chunk，embedding 向量化。

    Returns:
        (chunks, vectors) 元组
    """
    from src.embedder import Embedder

    print("\n[1/2] Embedding 向量化...")
    chunks = load_all_chunks()
    print(f"  共读取 {len(chunks)} 个 chunk")

    embedder = Embedder()
    print(f"  模型: {embedder.model_name}, 维度: {embedder.dim}")
    vectors = embedder.encode_chunks(chunks)
    print(f"  生成了 {len(vectors)} 个向量")
    return chunks, vectors


def run_milvus(chunks: list[dict], vectors: list[list[float]], drop_if_exists: bool = False) -> int:
    """Step 2: 写入 Milvus。"""
    from src.milvus_store import (
        get_client,
        create_collection,
        insert_chunks,
        get_collection_stats,
    )

    print("\n[2/2] 写入 Milvus...")
    client = get_client()
    create_collection(client, drop_if_exists=drop_if_exists)

    count = insert_chunks(chunks, vectors, client)
    print(f"  入库 {count} 条记录")

    stats = get_collection_stats(client)
    print(f"\n{'=' * 60}")
    print(f"入库完成！Collection: {stats}")
    return count


def main():
    parser = argparse.ArgumentParser(
        description="OnboardRAG — Embedding + Milvus 入库",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="重建 Milvus collection（删除已有数据）",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("OnboardRAG — Embedding + Milvus 入库")
    print("=" * 60)

    chunks, vectors = run_embed()
    run_milvus(chunks, vectors, drop_if_exists=args.rebuild)

    return 0


if __name__ == "__main__":
    sys.exit(main())
