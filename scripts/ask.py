#!/usr/bin/env python3
"""
ask.py — OnboardRAG CLI 问答脚本。

用法:
    python scripts/ask.py "入职第一天需要做什么？"
    python scripts/ask.py "VPN 连上了但是打不开内网怎么办？"
    python scripts/ask.py --top-k 5 "请假流程是什么？"
    python scripts/ask.py --show-chunks "怎么重置OA密码？"
"""

import argparse
import json
import sys
from pathlib import Path

# 将项目根目录加入 sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.query_rewriter import QueryRewriter
from src.retriever import Retriever
from src.reranker import Reranker
from src.generator import Generator
from src.config import RETRIEVAL_TOP_K


def run(query: str, top_k: int = RETRIEVAL_TOP_K, show_chunks: bool = False) -> dict:
    """执行完整的 RAG 问答链路。

    Args:
        query: 用户问题
        top_k: 检索返回的 chunk 数量
        show_chunks: 是否展示检索到的原始 chunks
    """
    print("=" * 60)
    print(f"  📝 问题: {query}")
    print("=" * 60)

    # Step 1: Query Rewrite
    print("\n[1/4] 查询改写...")
    rewriter = QueryRewriter()
    rewritten = rewriter.rewrite(query)
    if rewritten != query:
        print(f"  改写后: {rewritten}")

    # Step 2: 混合检索
    print("[2/4] 混合检索 (Dense + BM25 → RRF)...")
    retriever = Retriever()
    chunks = retriever.retrieve(rewritten, top_k=top_k)
    print(f"  检索到 {len(chunks)} 个 chunk")

    if show_chunks and chunks:
        print(f"\n  ── 检索到的 chunks ──")
        for i, c in enumerate(chunks, 1):
            text_preview = c.get("body_text", c.get("text", ""))[:120]
            print(f"  [{i}] {c['doc_title']} "
                  f"第{c.get('page_start',0)}-{c.get('page_end',0)}页 "
                  f"score={c.get('score',0):.4f}")
            print(f"      {text_preview}...")

    # Step 3: Reranker
    print("[3/4] 重排序...")
    reranker = Reranker()
    chunks = reranker.rerank(rewritten, chunks, top_k=top_k)
    print(f"  重排后保留 {len(chunks)} 个 chunk")

    # Step 4: LLM 生成
    print("[4/4] LLM 生成答案...")
    generator = Generator()
    answer, sources = generator.generate(rewritten, chunks[:top_k])

    # ── 输出 ──
    print("\n" + "=" * 60)
    print("  🤖 答案")
    print("=" * 60)
    print(answer)

    if sources:
        print("\n" + "=" * 60)
        print("  📚 参考来源")
        print("=" * 60)
        for i, src in enumerate(sources, 1):
            if src.get("page_start") == src.get("page_end"):
                page_info = f"第{src.get('page_start', 0)}页"
            else:
                page_info = (f"第{src.get('page_start', 0)}-"
                             f"{src.get('page_end', 0)}页")
            print(f"  [{i}] {src['doc_title']} — {src['source_file']} "
                  f"({page_info})")
            print(f"      chunk_id: {src['chunk_id']}")
            print(f"      内容摘要: {src.get('text', '')[:100]}...")

    return {"query": query, "answer": answer, "sources": sources}


def main():
    parser = argparse.ArgumentParser(
        description="OnboardRAG — CLI 问答（阶段二）",
    )
    parser.add_argument(
        "query",
        type=str,
        nargs="?",
        default=None,
        help="要提问的问题（用引号包裹）",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=RETRIEVAL_TOP_K,
        help=f"检索返回的 chunk 数量（默认: {RETRIEVAL_TOP_K}）",
    )
    parser.add_argument(
        "--show-chunks",
        action="store_true",
        help="展示检索到的原始 chunks",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="以 JSON 格式输出结果",
    )

    args = parser.parse_args()

    if args.query is None:
        parser.print_help()
        print("\n示例: python scripts/ask.py \"入职第一天需要做什么？\"")
        return 1

    result = run(args.query, top_k=args.top_k, show_chunks=args.show_chunks)

    if args.json_output:
        print("\n" + json.dumps(result, ensure_ascii=False, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
