"""
PDF 文本提取模块。

使用 PyMuPDF 读取 PDF，按页提取文本和基础元数据。
"""

from pathlib import Path
from typing import Iterator

import fitz  # PyMuPDF

from src.config import get_pdf_config


class PDFPage:
    """单页 PDF 提取结果。"""

    def __init__(
        self,
        page_number: int,
        text: str,
        source_file: str,
        doc_title: str,
        category: str,
        total_pages: int = 0,
    ):
        self.page_number = page_number  # 1-based
        self.text = text
        self.source_file = source_file
        self.doc_title = doc_title
        self.category = category
        self.total_pages = total_pages

    def to_dict(self) -> dict:
        return {
            "page_number": self.page_number,
            "text": self.text,
            "source_file": self.source_file,
            "doc_title": self.doc_title,
            "category": self.category,
            "total_pages": self.total_pages,
        }


def load_pdfs(pdf_dir: str | Path) -> list[PDFPage]:
    """加载目录下所有 PDF，返回每页的 PDFPage 列表。"""
    pdf_dir = Path(pdf_dir)
    if not pdf_dir.exists():
        raise FileNotFoundError(f"PDF 目录不存在: {pdf_dir}")

    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"目录下没有找到 PDF 文件: {pdf_dir}")

    pages: list[PDFPage] = []

    for pdf_path in pdf_files:
        doc = fitz.open(str(pdf_path))
        filename = pdf_path.name
        config = get_pdf_config(filename)

        for i in range(len(doc)):
            page = doc[i]
            text = page.get_text(sort=True)  # sort=True 按阅读顺序提取
            pages.append(PDFPage(
                page_number=i + 1,
                text=text.strip(),
                source_file=filename,
                doc_title=config["doc_title"],
                category=config["category"],
                total_pages=len(doc),
            ))

        doc.close()

    return pages


def iter_pdf_pages(pdf_dir: str | Path) -> Iterator[PDFPage]:
    """迭代器版本：逐个返回 PDF 页面，适合大文件逐页处理。"""
    yield from load_pdfs(pdf_dir)
