#!/usr/bin/env python3
"""
generate_rag_outputs.py — 批量生成 RAG 输出，供 evaluate_generation.py 评估使用。

读取评测数据集，对每条 query 执行检索 + LLM 生成，保存为 JSONL。

用法:
    # 全量生成
    python scripts/generate_rag_outputs.py

    # 限制条数（调试用）
    python scripts/generate_rag_outputs.py --limit 5

    # 指定输出路径
    python scripts/generate_rag_outputs.py --output outputs/eval/rag_outputs.jsonl

    # 只生成指定 ID 的 case（逗号分隔）
    python scripts/generate_rag_outputs.py --ids eval_0001,eval_0005,eval_0010 \\
        --output outputs/eval/rag_outputs_hard_cases.jsonl

    # 从文件读取要生成的 ID 列表
    python scripts/generate_rag_outputs.py --id-file data/eval/hard_cases.txt \\
        --output outputs/eval/rag_outputs_hard_cases.jsonl
"""

import argparse
import json
import sys
import time
from pathlib import Path

# 将项目根目录加入 sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.retriever import Retriever
from src.generator import Generator
from src.config import RETRIEVAL_TOP_K


def main():
    parser = argparse.ArgumentParser(
        description="OnboardRAG — 批量生成 RAG 输出",
    )
    parser.add_argument(
        "--eval-dataset",
        type=str,
        default="data/eval/eval_queries_v2.jsonl",
        help="评测数据集 JSONL 文件路径（默认: data/eval/eval_queries_v2.jsonl）",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="outputs/eval/rag_outputs.jsonl",
        help="输出 JSONL 文件路径（默认: outputs/eval/rag_outputs.jsonl）",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="限制生成样本数量（调试用，默认全部）",
    )
    parser.add_argument(
        "--retrieval-mode",
        type=str,
        choices=["hybrid", "dense", "bm25"],
        default="hybrid",
        help="检索模式（默认: hybrid）",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=RETRIEVAL_TOP_K,
        help=f"检索返回的 chunk 数量（默认: {RETRIEVAL_TOP_K}）",
    )
    parser.add_argument(
        "--ids",
        type=str,
        default=None,
        help="只生成指定 ID 的 case，多个 ID 用逗号分隔（如: eval_0001,eval_0005）",
    )
    parser.add_argument(
        "--id-file",
        type=str,
        default=None,
        help="从文件读取要生成的 ID 列表，每行一个 ID",
    )
    args = parser.parse_args()

    # ── 加载评测数据集 ──
    eval_path = Path(args.eval_dataset)
    if not eval_path.exists():
        print(f"错误: 评测数据集文件不存在: {eval_path}")
        return 1

    with open(eval_path, "r", encoding="utf-8") as f:
        eval_entries = [json.loads(line) for line in f if line.strip()]

    print(f"加载评测数据集: {len(eval_entries)} 条样本")

    # ── 按 ID 过滤 ──
    if args.ids or args.id_file:
        target_ids = set()
        if args.ids:
            target_ids.update(s.strip() for s in args.ids.split(",") if s.strip())
        if args.id_file:
            id_file_path = Path(args.id_file)
            if not id_file_path.exists():
                print(f"错误: ID 文件不存在: {id_file_path}")
                return 1
            with open(id_file_path, "r", encoding="utf-8") as f:
                target_ids.update(line.strip() for line in f if line.strip())
        eval_entries = [e for e in eval_entries if e.get("id") in target_ids]
        missing = target_ids - {e.get("id") for e in eval_entries}
        if missing:
            print(f"警告: 以下 ID 在数据集中未找到: {', '.join(sorted(missing))}")
        print(f"按 ID 过滤后: {len(eval_entries)} 条样本")

    if args.limit:
        eval_entries = eval_entries[:args.limit]
        print(f"限制条数: {args.limit}")

    # ── 初始化检索器 + 生成器 ──
    print("初始化 Retriever ...")
    retriever = Retriever()

    print("初始化 Generator ...")
    generator = Generator()

    # ── 逐条生成 ──
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f_out:
        for i, entry in enumerate(eval_entries, 1):
            eid = entry.get("id", f"unknown_{i}")
            query = entry["query"]
            print(f"[{i:03d}/{len(eval_entries)}] {eid} — {query[:60]}...", end=" ", flush=True)

            # 检索
            chunks = retriever.retrieve(
                query,
                top_k=args.top_k,
                mode=args.retrieval_mode,
            )

            # 生成答案
            answer, sources = generator.generate(query, chunks)

            # 构建 citations
            citations = []
            for s in sources:
                citations.append({
                    "chunk_id": s.get("chunk_id", ""),
                    "doc_title": s.get("doc_title", ""),
                    "source_file": s.get("source_file", ""),
                    "page_start": s.get("page_start", 0),
                    "page_end": s.get("page_end", 0),
                })

            # 保存
            record = {
                "id": eid,
                "query": query,
                "generated_answer": answer,
                "expected_answer": entry.get("reference_answer", ""),
                "retrieved_chunks": chunks,
                "citations": citations,
            }
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            f_out.flush()

            print(f"✓ ({len(chunks)} chunks, {len(answer)} chars)")

            # 每 5 条休息一下，避免 API 限流
            if i % 5 == 0 and i < len(eval_entries):
                time.sleep(1.0)

    print(f"\n完成! 共 {len(eval_entries)} 条样本 → {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
