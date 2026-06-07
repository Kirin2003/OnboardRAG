"""
重排序模块（可插拔接口）。

MVP 阶段默认关闭（ENABLE_RERANKER=false），直接返回原排序。
开启后可使用 BAAI/bge-reranker-base 等模型对候选 chunks 重排序。

Usage:
    from src.reranker import Reranker
    reranker = Reranker()
    reranked = reranker.rerank(query, chunks, top_k=5)
"""

from src.config import ENABLE_RERANKER, RERANKER_MODEL


class Reranker:
    """重排序器。

    当 ENABLE_RERANKER=false 时，直接返回原排序结果（pass-through）。
    当 ENABLE_RERANKER=true 时，加载 CrossEncoder 模型进行重排序。
    """

    def __init__(self):
        self._enabled = ENABLE_RERANKER
        self._model = None
        if self._enabled:
            self._load_model()

    def _load_model(self):
        """延迟加载 reranker 模型。"""
        try:
            # 使用 FlagEmbedding 的 bge-reranker（如果可用）
            # 也可用 sentence-transformers 的 CrossEncoder
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(RERANKER_MODEL)
            print(f"  [Reranker] 已加载模型: {RERANKER_MODEL}")
        except ImportError:
            print("  [Reranker] sentence-transformers 未安装，"
                  "reranker 将降级为 pass-through")
            self._enabled = False
        except Exception as e:
            print(f"  [Reranker] 加载模型失败: {e}，降级为 pass-through")
            self._enabled = False

    def rerank(
        self,
        query: str,
        chunks: list[dict],
        top_k: int = 10,
    ) -> list[dict]:
        """对候选 chunks 进行重排序。

        Args:
            query: 用户查询文本
            chunks: 候选 chunk 列表（每个包含 text/body_text 字段）
            top_k: 返回的 chunk 数量

        Returns:
            重排序后的 chunk 列表，每个新增 'rerank_score' 字段
        """
        if not self._enabled or self._model is None or len(chunks) <= 1:
            return chunks[:top_k]

        # 使用 body_text 做 rerank（更干净的文本），fallback 到 text
        texts = [c.get("body_text", c.get("text", "")) for c in chunks]
        pairs = [(query, t) for t in texts]

        try:
            scores = self._model.predict(pairs, show_progress_bar=False)
        except Exception as e:
            print(f"  [Reranker] 预测失败: {e}，使用原排序")
            return chunks[:top_k]

        # 附加分数并重排
        for chunk, score in zip(chunks, scores):
            chunk["rerank_score"] = float(score)

        sorted_chunks = sorted(
            chunks,
            key=lambda c: c.get("rerank_score", 0),
            reverse=True,
        )
        return sorted_chunks[:top_k]
