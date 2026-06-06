"""
Embedding 向量化模块。

使用 BAAI/bge-large-zh-v1.5 模型生成 dense embedding。
"""

from sentence_transformers import SentenceTransformer

from src.config import (
    EMBEDDING_MODEL_NAME,
    EMBEDDING_DIM,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_DEVICE,
)


class Embedder:
    """文本向量化器。

    Usage:
        embedder = Embedder()
        vectors = embedder.encode(texts)
    """

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
    ):
        self.model_name = model_name or EMBEDDING_MODEL_NAME
        self.device = device or EMBEDDING_DEVICE

        self._model = SentenceTransformer(
            self.model_name,
            device=self.device,
            trust_remote_code=True,
        )

    @property
    def dim(self) -> int:
        return self._model.get_sentence_embedding_dimension()

    def encode(
        self,
        texts: list[str],
        batch_size: int | None = None,
        show_progress: bool = True,
    ) -> list[list[float]]:
        """将文本列表编码为 embedding 向量。

        Args:
            texts: 文本列表
            batch_size: 批处理大小
            show_progress: 是否显示进度条

        Returns:
            embedding 向量列表（每个是 float 列表）
        """
        batch_size = batch_size or EMBEDDING_BATCH_SIZE

        # sentence-transformers 的 encode 已内置 normalize
        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,  # bge 推荐 normalize，方便余弦相似度
        )

        return embeddings.tolist()

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
