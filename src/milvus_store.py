"""
Milvus Lite 向量数据库存储模块。

使用 pymilvus 自带的 Milvus Lite（嵌入式文件数据库），
无需 Docker，零配置。
"""

from pymilvus import (
    MilvusClient,
    DataType,
    FieldSchema,
    CollectionSchema,
)

from src.config import (
    MILVUS_DB_PATH,
    MILVUS_COLLECTION_NAME,
    EMBEDDING_DIM,
)


def _build_schema() -> CollectionSchema:
    """构建 collection 的字段 schema。"""
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=8192),
        FieldSchema(name="doc_title", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="source_file", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=128),
        FieldSchema(name="page_start", dtype=DataType.INT32),
        FieldSchema(name="page_end", dtype=DataType.INT32),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM),
    ]
    return CollectionSchema(fields, description="OnboardRAG chunks")


# 向量索引参数
INDEX_PARAMS = {
    "field_name": "vector",
    "index_type": "IVF_FLAT",
    "metric_type": "COSINE",
    "params": {"nlist": 128},
}

# 标量字段索引（加速按 category/source_file 过滤）
SCALAR_FIELDS = ["category", "source_file"]


def get_client(db_path: str | None = None) -> MilvusClient:
    """创建并返回 Milvus Lite 客户端。

    Args:
        db_path: 数据库文件路径，默认使用 config 中的路径

    Returns:
        MilvusClient 实例
    """
    path = db_path or MILVUS_DB_PATH
    return MilvusClient(str(path))


def create_collection(
    client: MilvusClient | None = None,
    drop_if_exists: bool = False,
) -> None:
    """创建 collection（如果不存在）。

    Args:
        client: MilvusClient，为 None 时自动创建
        drop_if_exists: 是否先删除已有的同名 collection
    """
    if client is None:
        client = get_client()

    has_collection = client.has_collection(MILVUS_COLLECTION_NAME)

    if has_collection and drop_if_exists:
        client.drop_collection(MILVUS_COLLECTION_NAME)
        has_collection = False

    if has_collection:
        return

    schema = _build_schema()
    client.create_collection(
        collection_name=MILVUS_COLLECTION_NAME,
        schema=schema,
        index_params=INDEX_PARAMS,
    )

    # 创建标量索引
    for field in SCALAR_FIELDS:
        try:
            client.create_index(
                collection_name=MILVUS_COLLECTION_NAME,
                field_name=field,
                index_type="INVERTED",
            )
        except Exception:
            pass  # 某些 Milvus Lite 版本可能不支持标量索引，忽略


def insert_chunks(
    chunks: list[dict],
    vectors: list[list[float]],
    client: MilvusClient | None = None,
) -> int:
    """将 chunk 和对应的向量批量写入 Milvus。

    Args:
        chunks: chunk 字典列表
        vectors: embedding 向量列表（与 chunks 顺序对应）
        client: MilvusClient，为 None 时自动创建

    Returns:
        插入的记录数
    """
    if client is None:
        client = get_client()

    if len(chunks) != len(vectors):
        raise ValueError(
            f"chunks 和 vectors 长度不一致: {len(chunks)} vs {len(vectors)}"
        )

    if not chunks:
        return 0

    data = []
    for chunk, vector in zip(chunks, vectors):
        data.append({
            "chunk_id": chunk["chunk_id"],
            "text": chunk["text"],
            "doc_title": chunk["doc_title"],
            "source_file": chunk["source_file"],
            "category": chunk["category"],
            "page_start": chunk["page_start"],
            "page_end": chunk["page_end"],
            "vector": vector,
        })

    result = client.insert(
        collection_name=MILVUS_COLLECTION_NAME,
        data=data,
    )
    return result["insert_count"]


def get_collection_stats(client: MilvusClient | None = None) -> dict:
    """获取 collection 统计信息。"""
    if client is None:
        client = get_client()

    if not client.has_collection(MILVUS_COLLECTION_NAME):
        return {"exists": False, "total": 0}

    stats = client.get_collection_stats(MILVUS_COLLECTION_NAME)
    return {"exists": True, "total": stats.get("row_count", 0)}
