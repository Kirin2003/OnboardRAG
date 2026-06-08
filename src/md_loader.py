"""
Markdown 文档加载模块。

读取 data/raw/ 下的 .md 文件，返回统一格式的文档字典，
与 PDF loader 输出的 page dict 兼容。
"""

from pathlib import Path

from src.config import get_md_config


def load_markdown_files(
    raw_dir: str | Path,
    filenames: list[str] | None = None,
) -> list[dict]:
    """加载目录下的 Markdown 文件，返回文档字典列表。

    每个文档包含完整的文件文本和元数据，不做分页。
    Markdown 的结构化切分由 chunker.chunk_markdown_docs() 负责。

    Args:
        raw_dir: 原始文件目录
        filenames: 可选，指定要加载的文件名列表（如 ["统一门户账号与安全.md"]）。
                   不指定则加载目录下所有 .md 文件。

    Returns:
        [{text, source_file, doc_title, category}, ...]
    """
    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        raise FileNotFoundError(f"目录不存在: {raw_dir}")

    if filenames:
        md_files = [raw_dir / f for f in filenames if (raw_dir / f).exists()]
        missing = set(filenames) - {f.name for f in md_files}
        if missing:
            print(f"  警告: 以下 Markdown 文件未找到: {missing}")
        if not md_files:
            raise FileNotFoundError(f"指定 Markdown 文件均不存在: {filenames}")
    else:
        md_files = sorted(raw_dir.glob("*.md"))

    if not md_files:
        print("  目录下没有找到 .md 文件，跳过 Markdown 处理")
        return []

    docs = []
    for md_path in md_files:
        filename = md_path.name
        config = get_md_config(filename)

        with open(md_path, "r", encoding="utf-8") as f:
            text = f.read()

        if not text.strip():
            print(f"  警告: {filename} 内容为空，跳过")
            continue

        docs.append({
            "text": text.strip(),
            "source_file": filename,
            "doc_title": config["doc_title"],
            "category": config["category"],
        })

    return docs
