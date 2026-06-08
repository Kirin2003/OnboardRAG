"""
重排序模块（硅基流动 SiliconFlow Rerank API）。

MVP 阶段默认关闭（ENABLE_RERANKER=false），直接返回原排序。
开启后使用硅基流动 BAAI/bge-reranker-v2-m3 等模型对候选 chunks 重排序。

Usage:
    from src.reranker import Reranker
    reranker = Reranker()
    reranked = reranker.rerank(query, chunks, top_k=5)
"""

import httpx

from src.config import (
    ENABLE_RERANKER,
    RERANKER_MODEL,
    SILICONFLOW_API_KEY,
    SILICONFLOW_BASE_URL,
)


class Reranker:
    """重排序器（通过硅基流动 Rerank API）。

    当 ENABLE_RERANKER=false 时，直接返回原排序结果（pass-through）。
    当 ENABLE_RERANKER=true 时，调用硅基流动 /v1/rerank 接口进行重排序。
    """

    def __init__(
        self,
        model_name: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        self._enabled = ENABLE_RERANKER
        self._model_name = model_name or RERANKER_MODEL
        self._api_key = api_key or SILICONFLOW_API_KEY
        self._base_url = (base_url or SILICONFLOW_BASE_URL).rstrip("/")

        if self._enabled and not self._api_key:
            print(
                "  [Reranker] SILICONFLOW_API_KEY 未设置，"
                "reranker 将降级为 pass-through"
            )
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
        if not self._enabled or len(chunks) <= 1:
            return chunks[:top_k]

        # 使用 body_text 做 rerank（更干净的文本），fallback 到 text
        documents = [c.get("body_text", c.get("text", "")) for c in chunks]

        try:
            with httpx.Client(timeout=30) as client:
                response = client.post(
                    f"{self._base_url}/rerank",
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self._model_name,
                        "query": query,
                        "documents": documents,
                        "top_n": min(top_k, len(documents)),
                    },
                )
                response.raise_for_status()
                result = response.json()
        except httpx.HTTPStatusError as e:
            print(f"  [Reranker] API 请求失败 ({e.response.status_code}): {e}，"
                  "使用原排序")
            return chunks[:top_k]
        except httpx.RequestError as e:
            print(f"  [Reranker] 网络请求失败: {e}，使用原排序")
            return chunks[:top_k]
        except Exception as e:
            print(f"  [Reranker] 未知错误: {e}，使用原排序")
            return chunks[:top_k]

        # 附加分数并重排
        for item in result.get("results", []):
            idx = item["index"]
            chunks[idx]["rerank_score"] = float(item["relevance_score"])

        sorted_chunks = sorted(
            chunks,
            key=lambda c: c.get("rerank_score", 0),
            reverse=True,
        )
        return sorted_chunks[:top_k]
