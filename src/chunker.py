"""
Chunk 切分模块。

按语义边界（段落、标题、列表项）将文本切分为可控大小的 chunk，
尽量避免切断操作步骤和 FAQ。
"""

import re
from src.config import CHUNK_MIN_SIZE, CHUNK_MAX_SIZE, CHUNK_OVERLAP


def _split_into_segments(text: str) -> list[str]:
    """将文本拆分为语义段（段落、标题、列表项等）。"""
    if not text:
        return []

    raw_segments = text.split("\n")

    # 合并过短的相邻行（<20 字），避免把标题/短语孤立成独立段
    segments = []
    buffer = ""
    for seg in raw_segments:
        stripped = seg.strip()
        if not stripped:
            if buffer:
                segments.append(buffer)
                buffer = ""
            continue

        if buffer and len(buffer) < 20:
            buffer += "\n" + stripped
        elif buffer:
            segments.append(buffer)
            buffer = stripped
        else:
            buffer = stripped

    if buffer:
        segments.append(buffer)

    return segments


def _is_boundary_segment(segment: str) -> bool:
    """判断一个段是否是语义边界（标题、编号项开头等）。

    边界段不会被合并到上一个 chunk 的中间，尽量保持 chunk 的语义完整。
    """
    # 标题模式
    if re.match(r"^第[一二三四五六七八九十\d]+[章节条款部]", segment):
        return True
    # 编号列表项
    if re.match(r"^(\d+[\.\)、]|[一二三四五六七八九十]+[、])", segment):
        return True
    # 带括号编号
    if re.match(r"^[\(（]\d+[\)）]", segment):
        return True
    # 短标题（长度 < 30 且不含句号）
    if len(segment) < 30 and "。" not in segment and "；" not in segment:
        return True
    return False


def chunk_text(
    text: str,
    metadata: dict,
    chunk_index_start: int = 0,
) -> list[dict]:
    """将一段文本切分为多个 chunk。

    Args:
        text: 要切分的文本（单页或已合并的多页文本）
        metadata: 包含 source_file, category, doc_title, page_number 的字典
        chunk_index_start: chunk 索引起始值（跨页连续编号时使用）

    Returns:
        每个元素是一个 chunk 字典
    """
    segments = _split_into_segments(text)
    if not segments:
        return []

    chunks = []
    current_segs = []
    current_size = 0
    chunk_idx = chunk_index_start

    def _make_chunk(segs: list[str], idx: int) -> dict | None:
        """从多个段组合成一个 chunk。"""
        if not segs:
            return None
        chunk_text_content = "\n\n".join(segs)
        if len(chunk_text_content.strip()) < 20:  # 过滤太短的 chunk
            return None
        return {
            "chunk_id": f"{metadata['category']}_p{metadata['page_start']}_c{idx}",
            "text": chunk_text_content.strip(),
            "doc_title": metadata["doc_title"],
            "source_file": metadata["source_file"],
            "category": metadata["category"],
            "page_start": metadata["page_start"],
            "page_end": metadata.get("page_end", metadata["page_start"]),
        }

    for seg in segments:
        seg_len = len(seg)

        # 当前 chunk 加上这个段后超限
        if current_size + seg_len > CHUNK_MAX_SIZE and current_segs:
            # 如果当前段是边界且 chunk 已经够大了，先保存当前 chunk
            if current_size >= CHUNK_MIN_SIZE or _is_boundary_segment(seg):
                chunk = _make_chunk(current_segs, chunk_idx)
                if chunk:
                    chunks.append(chunk)
                    chunk_idx += 1

                # overlap 逻辑：保留最后一个段作为下一 chunk 的前缀
                if CHUNK_OVERLAP > 0 and len(current_segs) >= 2:
                    last_seg = current_segs[-1]
                    current_segs = [last_seg]
                    current_size = len(last_seg)
                else:
                    current_segs = []
                    current_size = 0

        current_segs.append(seg)
        current_size += seg_len

    # 最后剩余的段
    if current_segs and current_size >= 20:
        chunk = _make_chunk(current_segs, chunk_idx)
        if chunk:
            chunks.append(chunk)

    return chunks


def chunk_pages(pages: list[dict]) -> list[dict]:
    """对所有清洗后的页面进行 chunk 切分。

    将页面的 text 和元数据传入 chunker，同一页面内 chunk 连续编号。
    如果一页拆成多个 chunk，它们的 page_start/page_end 相同。
    """
    all_chunks = []

    for page in pages:
        meta = {
            "source_file": page["source_file"],
            "doc_title": page["doc_title"],
            "category": page["category"],
            "page_start": page["page_number"],
            "page_end": page["page_number"],
        }
        page_chunks = chunk_text(page["text"], meta)
        all_chunks.extend(page_chunks)

    return all_chunks
