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
# 所有相对路径都基于 PROJECT_ROOT 解析，避免从不同 CWD 运行时出错

def _resolve_path(raw: str | None, default: Path) -> Path:
    """将可能为相对路径的字符串解析为基于 PROJECT_ROOT 的绝对路径。"""
    if raw is None or raw == "":
        return default
    p = Path(raw)
    if p.is_absolute():
        return p
    return (PROJECT_ROOT / p).resolve()


DATA_DIR = _resolve_path(
    os.getenv("ONBOARDRAG_DATA_DIR"), PROJECT_ROOT / "data"
)
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# 确保目录存在
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ── Milvus Lite 配置 ─────────────────────────────────────

MILVUS_DB_PATH = str(
    _resolve_path(os.getenv("MILVUS_DB_PATH"), DATA_DIR / "onboard_rag.db")
)
MILVUS_COLLECTION_NAME = "onboard_chunks"

# ── Embedding API 配置（硅基流动 SiliconFlow）────────────

SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY", "")
SILICONFLOW_BASE_URL = os.getenv(
    "SILICONFLOW_BASE_URL",
    "https://api.siliconflow.cn/v1",
)

# Embedding 用的别名（复用硅基流动 API）
EMBEDDING_API_KEY = SILICONFLOW_API_KEY
EMBEDDING_API_BASE_URL = SILICONFLOW_BASE_URL
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

# ── 阶段二：LLM 配置 ────────────────────────────────────

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", os.getenv("OPENAI_API_KEY", ""))
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3.6-flash-2026-04-16")

# ── 阶段二：检索参数 ────────────────────────────────────

# 检索模式: "hybrid"（混合检索）、"dense"（仅向量）、"bm25"（仅关键词）
RETRIEVAL_MODE = os.getenv("RETRIEVAL_MODE", "hybrid")

# 检索返回的候选 chunk 数量（RRF 合并后 / 单路检索后）
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "10"))

# Dense 检索返回的候选数（通常设大一些供 RRF 合并）
DENSE_TOP_K = int(os.getenv("DENSE_TOP_K", "20"))

# BM25 检索返回的候选数
BM25_TOP_K = int(os.getenv("BM25_TOP_K", "20"))

# RRF 参数：k 值（默认 60）
RRF_K = int(os.getenv("RRF_K", "60"))

# ── 阶段二：Reranker 配置（硅基流动）──────────────────

ENABLE_RERANKER = os.getenv("ENABLE_RERANKER", "false").lower() == "true"
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")

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
    "钉钉考勤.pdf": {
        "doc_title": "钉钉考勤使用说明",
        "category": "dingtalk_manual",
    },
}

# ── Markdown 文档映射 ───────────────────────────────────

MD_MAPPING = {
    "统一门户账号与安全.md": {
        "doc_title": "统一门户账号与安全",
        "category": "account_portal",
    },
    "vpn使用和故障排查.md": {
        "doc_title": "VPN使用与故障排查",
        "category": "vpn_manual",
    },
}

# MarkdownHeaderTextSplitter 拆分的标题层级
MD_HEADERS_TO_SPLIT_ON = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
]


def get_pdf_config(filename: str) -> dict:
    """根据 PDF 文件名获取预设的 doc_title 和 category。

    如果文件名不在映射表中，则从文件名自动推断。
    """
    if filename in PDF_MAPPING:
        return PDF_MAPPING[filename]

    # fallback: 去掉 .pdf 后缀作为 title，category 用拼音缩写
    stem = filename.replace(".pdf", "").replace(".PDF", "")
    return {"doc_title": stem, "category": stem[:20]}


def get_md_config(filename: str) -> dict:
    """根据 Markdown 文件名获取预设的 doc_title 和 category。

    如果文件名不在映射表中，则从文件名自动推断。
    """
    if filename in MD_MAPPING:
        return MD_MAPPING[filename]

    # fallback: 去掉 .md 后缀作为 title
    stem = filename.replace(".md", "").replace(".MD", "")
    return {"doc_title": stem, "category": stem[:20]}
