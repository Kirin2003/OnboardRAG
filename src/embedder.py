"""
Embedding 向量化模块。

使用硅基流动 SiliconFlow API 生成 dense embedding。
模型: BAAI/bge-large-zh-v1.5，输出 1024 维。
"""

from openai import OpenAI

from src.config import (
    EMBEDDING_API_KEY,
    EMBEDDING_API_BASE_URL,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_DIM,
    EMBEDDING_BATCH_SIZE,
)


class Embedder:
    """文本向量化器（通过硅基流动 API）。

    Usage:
        embedder = Embedder()
        vectors = embedder.encode(texts)
    """

    def __init__(
        self,
        model_name: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        self.model_name = model_name or EMBEDDING_MODEL_NAME
        api_key = api_key or EMBEDDING_API_KEY
        base_url = base_url or EMBEDDING_API_BASE_URL

        if not api_key:
            raise ValueError(
                "SILICONFLOW_API_KEY 未设置。请在 .env 文件中配置或设置环境变量。"
            )

        self._client = OpenAI(api_key=api_key, base_url=base_url)

    @property
    def dim(self) -> int:
        """返回 embedding 向量维度。"""
        return EMBEDDING_DIM

    def encode(
        self,
        texts: list[str],
        batch_size: int | None = None,
        show_progress: bool = True,
    ) -> list[list[float]]:
        """将文本列表编码为 embedding 向量。

        Args:
            texts: 文本列表
            batch_size: 批处理大小（每批发送多少条文本）
            show_progress: 是否显示进度条

        Returns:
            embedding 向量列表（每个是 float 列表），顺序与 texts 一致
        """
        batch_size = batch_size or EMBEDDING_BATCH_SIZE

        all_embeddings: list[list[float]] = []

        total = len(texts)
        for i in range(0, total, batch_size):
            batch = texts[i:i + batch_size]

            if show_progress:
                print(f"  embedding: {min(i + batch_size, total)}/{total}")

            response = self._client.embeddings.create(
                model=self.model_name,
                input=batch,
            )

            # 按 index 排序后取 embedding，确保顺序
            sorted_data = sorted(response.data, key=lambda d: d.index)
            all_embeddings.extend([d.embedding for d in sorted_data])

        return all_embeddings

    def encode_chunks(
        self,
        chunks: list[dict],
        batch_size: int | None = None,
        show_progress: bool = True,
    ) -> list[list[float]]:
        """对 chunk 列表进行向量化。

        Args:
            chunks: chunk 字典列表，每个包含 'text' 键
            batch_size: 批处理大小
            show_progress: 是否显示进度条

        Returns:
            embedding 向量列表，顺序与 chunks 一致
        """
        texts = [c["text"] for c in chunks]
        return self.encode(texts, batch_size=batch_size, show_progress=show_progress)
