"""
OnboardRAG 全局配置。

通过环境变量和 .env 文件覆盖默认值。
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# ── 路径配置 ────────────────────────────────────────────

DATA_DIR = Path(os.getenv("ONBOARDRAG_DATA_DIR", PROJECT_ROOT / "data"))
RAW_PDFS_DIR = DATA_DIR / "raw_pdfs"
PROCESSED_DIR = DATA_DIR / "processed"

# 确保目录存在
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ── Milvus Lite 配置 ─────────────────────────────────────

MILVUS_DB_PATH = os.getenv(
    "MILVUS_DB_PATH",
    str(DATA_DIR / "onboard_rag.db"),
)
MILVUS_COLLECTION_NAME = "onboard_chunks"

# ── Embedding API 配置（硅基流动 SiliconFlow）────────────

EMBEDDING_API_KEY = os.getenv("SILICONFLOW_API_KEY", "")
EMBEDDING_API_BASE_URL = os.getenv(
    "SILICONFLOW_BASE_URL",
    "https://api.siliconflow.cn/v1",
)
EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL",
    "BAAI/bge-large-zh-v1.5",
)
EMBEDDING_DIM = 1024  # bge-large-zh-v1.5 输出 1024 维
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))

# ── Chunk 参数 ──────────────────────────────────────────

CHUNK_MIN_SIZE = int(os.getenv("CHUNK_MIN_SIZE", "300"))
CHUNK_MAX_SIZE = int(os.getenv("CHUNK_MAX_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "80"))

# 切分方法: "section_aware"（自有 section-aware 策略）或 "recursive"（LangChain RecursiveCharacterTextSplitter）
CHUNK_METHOD = os.getenv("CHUNK_METHOD", "section_aware")

# ── PDF 文件名 → (doc_title, category) 映射 ──────────────

PDF_MAPPING = {
    "员工手册.pdf": {
        "doc_title": "员工手册",
        "category": "employee_handbook",
    },
    "OA系统使用手册.pdf": {
        "doc_title": "OA系统使用手册",
        "category": "oa_manual",
    },
    "企业内部移动办公.pdf": {
        "doc_title": "企业内部移动办公使用手册",
        "category": "mobile_office_manual",
    },
    "VPN使用.pdf": {
        "doc_title": "VPN使用手册",
        "category": "vpn_manual",
    },
}


def get_pdf_config(filename: str) -> dict:
    """根据 PDF 文件名获取预设的 doc_title 和 category。

    如果文件名不在映射表中，则从文件名自动推断。
    """
    if filename in PDF_MAPPING:
        return PDF_MAPPING[filename]

    # fallback: 去掉 .pdf 后缀作为 title，category 用拼音缩写
    stem = filename.replace(".pdf", "").replace(".PDF", "")
    return {"doc_title": stem, "category": stem[:20]}
