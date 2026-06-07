#!/usr/bin/env python3
"""
build_chunks.py — PDF 文档 → chunk 切分，结果保存到 PROCESSED_DIR。

每个 PDF 独立保存为一个 chunk 文件，文件名格式: {pdf_stem}_chunks.jsonl

运行流程:
    PDF 文档 → 文本提取 → 文本清洗 → chunk 切分 → 按 PDF 分别保存

用法:
    # 处理所有 PDF
    python scripts/build_chunks.py

    # 只处理指定的 PDF
    python scripts/build_chunks.py --pdf "员工手册.pdf" "OA系统使用手册.pdf"

    # 查看已生成的 chunk 文件
    python scripts/build_chunks.py --list
"""

import argparse
import json
import sys
from collections import defaultdict
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


def load_chunks_jsonl(path: str | Path) -> list[dict]:
    """从 JSONL 文件读取 chunk 列表。"""
    chunks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def chunk_filename(pdf_filename: str) -> str:
    """根据 PDF 文件名生成对应的 chunk 文件名。

    "员工手册.pdf" → "员工手册_chunks.jsonl"
    """
    stem = Path(pdf_filename).stem
    return f"{stem}_chunks.jsonl"


def list_chunk_files() -> list[Path]:
    """列出 PROCESSED_DIR 下所有 chunk 文件。"""
    return sorted(PROCESSED_DIR.glob("*_chunks.jsonl"))


def main():
    parser = argparse.ArgumentParser(
        description="OnboardRAG — PDF → Chunk 切分",
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
        "--list", "-l",
        action="store_true",
        help="列出 PROCESSED_DIR 下已有的 chunk 文件",
    )

    args = parser.parse_args()

    # ── --list 模式 ──────────────────────────────
    if args.list:
        files = list_chunk_files()
        if not files:
            print("PROCESSED_DIR 下暂无 chunk 文件")
        else:
            print(f"PROCESSED_DIR 下的 chunk 文件 ({len(files)} 个):")
            for f in files:
                chunks = load_chunks_jsonl(f)
                print(f"  {f.name}  — {len(chunks)} chunks")
        return 0

    print("=" * 60)
    print("OnboardRAG — PDF → Chunk 切分")
    print("=" * 60)
    if args.pdf:
        print(f"  指定 PDF: {args.pdf}")

    # ── Step 1: PDF 文本提取 ──────────────────────
    print("\n[1/3] PDF 文本提取...")
    pages = load_pdfs(RAW_PDFS_DIR, filenames=args.pdf)
    if not pages:
        print("  没有找到匹配的 PDF 文件！")
        return 1
    sources = sorted(set(p.source_file for p in pages))
    print(f"  读取了 {len(pages)} 页（来自 {len(sources)} 个 PDF: {', '.join(sources)}）")

    # ── Step 2: 文本清洗 ──────────────────────────
    print("\n[2/3] 文本清洗...")
    page_dicts = _pages_to_dicts(pages)
    page_dicts = clean_pages(page_dicts)
    print(f"  清洗后剩余 {len(page_dicts)} 页（非空）")

    if not page_dicts:
        print("  清洗后没有有效内容！")
        return 1

    # ── Step 3: Chunk 切分 ────────────────────────
    print("\n[3/3] Chunk 切分...")
    chunks = chunk_pages(page_dicts)
    print(f"  切分为 {len(chunks)} 个 chunk")

    sizes = [len(c["text"]) for c in chunks]
    if sizes:
        print(f"  chunk 大小: 最小={min(sizes)}字, 最大={max(sizes)}字, "
              f"平均={sum(sizes)//len(sizes)}字")

    if not chunks:
        print("  没有生成任何 chunk！")
        return 1

    # ── 按 source_file 分组，分别保存 ──────────────
    grouped: dict[str, list[dict]] = defaultdict(list)
    for c in chunks:
        grouped[c["source_file"]].append(c)

    print(f"\n{'=' * 60}")
    print("保存 chunk 文件:")
    for pdf_name, pdf_chunks in grouped.items():
        path = PROCESSED_DIR / chunk_filename(pdf_name)
        save_chunks_jsonl(pdf_chunks, path)
        print(f"  {path.name}  — {len(pdf_chunks)} chunks")

    print(f"\n共 {len(chunks)} 条 chunk，分布到 {len(grouped)} 个文件")
    return 0


if __name__ == "__main__":
    sys.exit(main())
