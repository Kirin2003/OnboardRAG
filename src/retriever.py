"""
检索模块，支持三种检索模式：

1. hybrid（默认）—— Dense 向量（Milvus）+ BM25 关键词 → RRF 合并
2. dense —— 仅向量检索（embedding）
3. bm25  —— 仅关键词检索（BM25）

Usage:
    from src.retriever import Retriever
    retriever = Retriever()

    # 混合检索
    chunks = retriever.retrieve("VPN 连不上怎么办？", top_k=10, mode="hybrid")

    # 仅关键词（消融实验）
    chunks = retriever.retrieve("VPN 连不上怎么办？", top_k=10, mode="bm25")

    # 仅向量（消融实验）
    chunks = retriever.retrieve("VPN 连不上怎么办？", top_k=10, mode="dense")
"""

import json
import math
from collections import defaultdict
from pathlib import Path

import jieba
from rank_bm25 import BM25Okapi

from src.config import (
    PROCESSED_DIR,
    MILVUS_DB_PATH,
    MILVUS_COLLECTION_NAME,
    DENSE_TOP_K,
    BM25_TOP_K,
    RRF_K,
)
from src.embedder import Embedder
from src.milvus_store import get_client


class Retriever:
    """检索器，支持三种模式：
    - "hybrid"（默认）：Dense + BM25 → RRF 混合检索
    - "dense"：仅向量（embedding）检索
    - "bm25"：仅关键词（BM25）检索

    初始化时加载 BM25 索引和 embedding 模型，
    后续调用 retrieve() 进行检索。
    """

    def __init__(self):
        # 延迟初始化，避免未装入库时构建 BM25 失败
        self._embedder: Embedder | None = None
        self._bm25: BM25Okapi | None = None
        self._bm25_chunks: list[dict] = []
        self._bm25_tokenized: list[list[str]] = []

    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = Embedder()
        return self._embedder

    def _load_bm25(self) -> None:
        """从 chunks.jsonl 加载 chunk 并建立 BM25 索引。"""
        if self._bm25 is not None:
            return  # 已加载

        chunk_files = sorted(PROCESSED_DIR.glob("*_chunks.jsonl"))
        if not chunk_files:
            print("  [BM25] 未找到 chunks.jsonl 文件，BM25 检索将返回空结果")
            self._bm25 = BM25Okapi([])
            return

        all_chunks = []
        for path in chunk_files:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        all_chunks.append(json.loads(line))

        print(f"  [BM25] 从 {len(chunk_files)} 个文件加载了 {len(all_chunks)} 个 chunk")

        # 对 chunk 文本进行 jieba 分词
        tokenized_corpus = []
        for chunk in all_chunks:
            # 用 body_text（更干净的文本）做 BM25，fallback 到 text
            text = chunk.get("body_text", chunk.get("text", ""))
            tokens = list(jieba.cut(text))
            tokenized_corpus.append(tokens)

        self._bm25_chunks = all_chunks
        self._bm25_tokenized = tokenized_corpus
        self._bm25 = BM25Okapi(tokenized_corpus)

    def _dense_search(self, query: str, top_k: int) -> list[dict]:
        """Dense 向量检索：对 query 编码，从 Milvus 搜索相似向量。

        Returns:
            [{chunk_id, text, doc_title, source_file, category,
              page_start, page_end, score}, ...]
        """
        client = get_client()
        if not client.has_collection(MILVUS_COLLECTION_NAME):
            print("  [Dense] Milvus collection 不存在，请先运行 build_index.py")
            return []

        # 加载 collection 到内存（搜索前必须 load）
        client.load_collection(MILVUS_COLLECTION_NAME)

        # 编码 query
        query_vec = self.embedder.encode([query], show_progress=False)[0]

        # Milvus 搜索，指定输出字段
        results = client.search(
            collection_name=MILVUS_COLLECTION_NAME,
            data=[query_vec],
            limit=top_k,
            output_fields=[
                "chunk_id", "text", "doc_title", "source_file",
                "category", "page_start", "page_end",
            ],
        )

        chunks = []
        for hit in results[0]:
            entity = hit.get("entity", hit)
            chunks.append({
                "chunk_id": entity.get("chunk_id", ""),
                "text": entity.get("text", ""),
                "doc_title": entity.get("doc_title", ""),
                "source_file": entity.get("source_file", ""),
                "category": entity.get("category", ""),
                "page_start": entity.get("page_start", 0),
                "page_end": entity.get("page_end", 0),
                "score": hit.get("distance", hit.get("score", 0)),
                "source": "dense",
            })

        return chunks

    def _bm25_search(self, query: str, top_k: int) -> list[dict]:
        """BM25 关键词检索。

        Returns:
            [{chunk_id, body_text, text, doc_title, source_file, category,
              page_start, page_end, score}, ...]
        """
        self._load_bm25()
        if not self._bm25_chunks or not self._bm25:
            return []

        # 对 query 分词
        query_tokens = list(jieba.cut(query))
        scores = self._bm25.get_scores(query_tokens)

        # 取 top_k
        indexed_scores = list(enumerate(scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        top_indices = indexed_scores[:top_k]

        # 归一化 BM25 分数到 [0, 1]
        max_score = top_indices[0][1] if top_indices else 1.0
        min_score = top_indices[-1][1] if top_indices else 0.0
        score_range = max_score - min_score or 1.0

        chunks = []
        for idx, raw_score in top_indices:
            c = self._bm25_chunks[idx]
            norm_score = (raw_score - min_score) / score_range
            chunks.append({
                "chunk_id": c.get("chunk_id", ""),
                "body_text": c.get("body_text", c.get("text", "")),
                "text": c.get("text", ""),
                "doc_title": c.get("doc_title", ""),
                "source_file": c.get("source_file", ""),
                "category": c.get("category", ""),
                "page_start": c.get("page_start", 0),
                "page_end": c.get("page_end", 0),
                "score": norm_score,
                "source": "bm25",
            })

        return chunks

    def _rrf_merge(
        self,
        dense_chunks: list[dict],
        bm25_chunks: list[dict],
        k: int = 60,
    ) -> list[dict]:
        """Reciprocal Rank Fusion 合并两路检索结果。

        Args:
            dense_chunks: dense 检索结果
            bm25_chunks: BM25 检索结果
            k: RRF 参数（默认 60）

        Returns:
            按 RRF 分数降序排列的 chunk 列表（去重，合并字段）
        """
        rrf_scores: dict[str, float] = defaultdict(float)
        chunk_map: dict[str, dict] = {}

        # Dense 结果
        for rank, chunk in enumerate(dense_chunks, start=1):
            cid = chunk["chunk_id"]
            rrf_scores[cid] += 1.0 / (k + rank)
            if cid not in chunk_map:
                chunk_map[cid] = dict(chunk)
                chunk_map[cid]["_rrf_dense_rank"] = rank
            else:
                # 合并 body_text（如果 BM25 有）
                if chunk.get("body_text") and not chunk_map[cid].get("body_text"):
                    chunk_map[cid]["body_text"] = chunk["body_text"]

        # BM25 结果
        for rank, chunk in enumerate(bm25_chunks, start=1):
            cid = chunk["chunk_id"]
            rrf_scores[cid] += 1.0 / (k + rank)
            if cid not in chunk_map:
                chunk_map[cid] = dict(chunk)
                chunk_map[cid]["_rrf_bm25_rank"] = rank
            else:
                # BM25 有更干净的 body_text，优先使用
                if chunk.get("body_text"):
                    chunk_map[cid]["body_text"] = chunk["body_text"]

        # 按 RRF 分数排序
        sorted_ids = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)

        result = []
        for cid in sorted_ids:
            chunk = chunk_map[cid]
            chunk["score"] = round(rrf_scores[cid], 6)
            chunk["_rrf_score"] = chunk["score"]
            # 清理内部字段
            chunk.pop("source", None)
            result.append(chunk)

        return result

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        dense_top_k: int | None = None,
        bm25_top_k: int | None = None,
        mode: str = "hybrid",
    ) -> list[dict]:
        """执行检索，支持三种模式。

        Args:
            query: 用户查询文本
            top_k: 返回的最终 chunk 数量
            dense_top_k: dense 检索候选数（仅 hybrid / dense 模式使用）
            bm25_top_k: BM25 检索候选数（仅 hybrid / bm25 模式使用）
            mode: 检索模式 —— "hybrid"（混合检索，默认）、
                  "dense"（仅向量检索）、"bm25"（仅关键词检索）

        Returns:
            chunk 列表，字段包括：
            chunk_id, text, body_text, doc_title, source_file,
            category, page_start, page_end, score
        """
        from src.config import RETRIEVAL_TOP_K

        if mode not in ("hybrid", "dense", "bm25"):
            raise ValueError(f"不支持的检索模式: {mode!r}，可选值: hybrid, dense, bm25")

        final_top_k = top_k if top_k is not None else RETRIEVAL_TOP_K

        # ── 仅 BM25 检索 ──
        if mode == "bm25":
            b_top_k = bm25_top_k if bm25_top_k is not None else BM25_TOP_K
            return self._bm25_search(query, top_k=max(b_top_k, final_top_k))[:final_top_k]

        # ── 仅 Dense 检索 ──
        if mode == "dense":
            d_top_k = dense_top_k if dense_top_k is not None else DENSE_TOP_K
            return self._dense_search(query, top_k=max(d_top_k, final_top_k))[:final_top_k]

        # ── 混合检索（Dense + BM25 → RRF）──
        d_top_k = dense_top_k if dense_top_k is not None else DENSE_TOP_K
        b_top_k = bm25_top_k if bm25_top_k is not None else BM25_TOP_K

        dense_chunks = self._dense_search(query, top_k=d_top_k)
        bm25_chunks = self._bm25_search(query, top_k=b_top_k)

        merged = self._rrf_merge(dense_chunks, bm25_chunks, k=RRF_K)
        return merged[:final_top_k]
