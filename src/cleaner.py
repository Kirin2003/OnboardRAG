"""
文本清洗模块。

针对中文 PDF 提取文本的常见噪声进行清理：
- 去除多余空行和重复空格
- 识别并去除页眉页脚（重复出现的标题行、页码）
- 合并 PDF 导致的异常断行
- 保留标题、编号、步骤、FAQ 等结构化内容
"""

import re
from collections import Counter


def clean_text(text: str) -> str:
    """对单段文本执行基础清洗。"""
    if not text:
        return ""

    # 1. 统一换行为 \n
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 2. 去除重复空格（保留单个空格用于分词间隔）
    text = re.sub(r"[ \t]+", " ", text)

    # 3. 去除行首行尾空格
    lines = [line.strip() for line in text.split("\n")]

    # 4. 去除连续空行，最多保留一个空行作为段落分隔
    cleaned = []
    prev_empty = False
    for line in lines:
        if not line:
            if not prev_empty:
                cleaned.append("")
            prev_empty = True
        else:
            cleaned.append(line)
            prev_empty = False

    return "\n".join(cleaned).strip()


def remove_headers_footers(
    pages: list[dict],
    min_occurrence_ratio: float = 0.5,
) -> list[dict]:
    """基于多页统计，去除页眉页脚中的重复行。

    原理：如果一段文本在 ≥50% 的页面中出现且行数很少，
    大概率是页眉页脚内容，予以去除。

    Args:
        pages: 包含 'text' 键的页面字典列表
        min_occurrence_ratio: 判定为页眉页脚的最低出现比例
    """
    total_pages = len(pages)
    if total_pages < 2:
        return pages

    # 统计每行文本在所有页面中的出现次数
    line_counter: Counter = Counter()
    for page in pages:
        lines = page["text"].split("\n")
        # 收集每一行（去重，同页面内重复也算一次）
        for line in set(lines):
            stripped = line.strip()
            if stripped:
                line_counter[stripped] += 1

    # 找出高频行（出现页数 >= 比例 × 总页数 且长度不太长）
    threshold = max(2, int(total_pages * min_occurrence_ratio))
    header_lines = {
        line
        for line, count in line_counter.items()
        if count >= threshold and len(line) < 80
    }

    # 去除纯页码行（单独出现的数字）
    page_number_pattern = re.compile(r"^\s*\d{1,4}\s*$")

    for page in pages:
        lines = page["text"].split("\n")
        filtered = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                filtered.append(line)
                continue
            if page_number_pattern.match(stripped):
                continue
            if stripped in header_lines:
                continue
            # 去除纯空白行组成的短文本（<4个非空白字符的非中文非标题行）
            if len(stripped) < 4 and not re.search(r"[一-鿿]", stripped):
                continue
            filtered.append(line)
        page["text"] = "\n".join(filtered)

    return pages


def merge_broken_lines(text: str) -> str:
    """合并 PDF 提取导致的异常断行。

    规则：
    - 如果一行不以句号、问号、感叹号等结束，且下一行不以标点、数字编号等开头，
      则合并两行。
    - 保留明显的标题行（以编号开头的如"1."、"一、"等）。
    """
    if not text:
        return ""

    lines = text.split("\n")
    if len(lines) < 2:
        return text

    # 句子结束符
    sentence_ends = {"。", "？", "！", "；", "：", "”", "」", "）", "）", ")", "…"}
    # 行首模式：编号或特殊符号，表示新段落的开始
    new_para_starts = re.compile(
        r"^(\d+[\.\)、]|[一二三四五六七八九十]+[、．.]"
        r"|[\(（]\d+[\)）]"
        r"|第[一二三四五六七八九十\d]+[章节条款]"
        r"|[①②③④⑤⑥⑦⑧⑨⑩]"
        r"|[▶•▪✓✅]"
        r"|步骤\s*\d)"
    )

    merged = []
    buffer = ""

    for line in lines:
        stripped = line.strip()
        if not stripped:
            # 空行：刷新 buffer 并保留段落分隔
            if buffer:
                merged.append(buffer)
                buffer = ""
            merged.append("")
            continue

        # 当前行是否像是新段落的开始
        is_new_para = bool(new_para_starts.match(stripped))

        if not buffer:
            buffer = stripped
        else:
            # 判断是否应该合并
            last_char = buffer[-1] if buffer else ""
            if last_char in sentence_ends or is_new_para:
                # buffer 已经结束，当前行是新句子
                merged.append(buffer)
                buffer = stripped
            else:
                # 合并：上一行没有结束符且当前行不是新段落开始
                buffer += stripped

    if buffer:
        merged.append(buffer)

    return "\n".join(merged)


def clean_pages(pages: list[dict]) -> list[dict]:
    """对页面列表执行完整的清洗流程。"""
    # 第一步：每页文本基础清洗
    for page in pages:
        page["text"] = clean_text(page["text"])

    # 第二步：去除页眉页脚
    pages = remove_headers_footers(pages)

    # 第三步：合并异常断行
    for page in pages:
        page["text"] = merge_broken_lines(page["text"])

    # 第四步：最终清理，去除清洗后变成空页的页面
    pages = [p for p in pages if p["text"].strip()]

    return pages
