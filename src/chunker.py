"""
Chunk 切分模块（v2 — section-aware chunking）。

支持两种切分方法，由 config.CHUNK_METHOD 控制：
  - "section_aware" — 自有 section-aware 策略
  - "recursive"     — LangChain RecursiveCharacterTextSplitter

流程: document pages → segments → sections → chunks

1. _split_into_segments          — 将单段文本拆为语义段
2. _extract_segments_from_pages  — 所有页面 → 带元数据的 segment 列表
3. _detect_heading               — 判断 segment 是否为 section 级标题
4. _is_action_step               — 判断是否为操作步骤（系统手册专用）
5. _build_sections               — 根据标题将 segment 跨页归入 section
6. _chunk_section                — section 内部按长度 fallback 切分
6b._chunk_section_recursive      — section 内部用 RecursiveCharacterTextSplitter 切分
7. _build_embedding_text         — 构建含 doc_title + section_title 的 embedding 文本
8. chunk_pages                   — 主入口
"""

import re
from itertools import groupby

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import CHUNK_MIN_SIZE, CHUNK_MAX_SIZE, CHUNK_OVERLAP, CHUNK_METHOD

# ═══════════════════════════════════════════════════════════════
# 常量
# ═══════════════════════════════════════════════════════════════

# 操作动词（系统手册中用于区分操作步骤与标题）
_ACTION_VERBS = [
    "打开", "点击", "输入", "选择", "登录", "进入", "退出",
    "下载", "安装", "配置", "设置", "修改", "保存", "提交",
    "上传", "删除", "添加", "创建", "编辑", "查看", "浏览",
    "访问", "连接", "启动", "关闭", "重启", "切换", "勾选",
    "填写", "确认", "取消", "返回", "搜索", "找到", "复制",
    "粘贴", "拖动", "右键", "双击", "单击", "按下", "运行",
    "执行", "注册", "绑定", "验证", "扫描", "发送", "接收",
]

# ═══════════════════════════════════════════════════════════════
# 编译正则
# ═══════════════════════════════════════════════════════════════

# 编号开头模式（支持 1. 1、 1) （1） (1) 等格式）
_RE_NUMBERED_START = re.compile(
    r"^(\d+[\.、．)]|[(（][一二三四五六七八九十百\d]+[)）])\s*"
)

# ═══════════════════════════════════════════════════════════════
# 1. Segment 拆分（保留原逻辑）
# ═══════════════════════════════════════════════════════════════


def _split_into_segments(text: str) -> list[str]:
    """将文本拆分为语义段（段落、标题、列表项等）。"""
    if not text:
        return []

    raw_segments = text.split("\n")

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


# ═══════════════════════════════════════════════════════════════
# 2. 从页面提取 segment
# ═══════════════════════════════════════════════════════════════


def _extract_segments_from_pages(pages: list[dict]) -> list[dict]:
    """遍历所有页面，将每页文本拆为 segment 并附加元数据。

    Returns:
        [{text, page_number, source_file, doc_title, category}, ...]
    """
    all_segments = []
    for page in pages:
        text = page.get("text", "")
        if not text.strip():
            continue

        seg_texts = _split_into_segments(text)
        for seg_text in seg_texts:
            if not seg_text.strip():
                continue
            all_segments.append({
                "text": seg_text.strip(),
                "page_number": page["page_number"],
                "source_file": page["source_file"],
                "doc_title": page["doc_title"],
                "category": page["category"],
            })

    return all_segments


# ═══════════════════════════════════════════════════════════════
# 3. 标题检测
# ═══════════════════════════════════════════════════════════════


def _detect_heading(text: str, category: str) -> str | None:
    """检测文本段是否为 section 级别的标题。

    只识别"应该开启新 section 的标题"，不识别局部标签和操作步骤。

    Returns:
        str   — 标题文本
        None  — 不是标题
    """
    text = text.strip()
    if not text:
        return None

    if len(text) > 60:
        return None

    if text.endswith(("。", "；", ";")):
        return None

    # 系统手册中，操作步骤不是 section 标题
    if category != "employee_handbook" and _is_action_step(text):
        return None

    # 第X章 / 第X节 / 第X条 / 第X部分
    if re.match(r"^第[一二三四五六七八九十百\d]+[章节篇部分条款]", text):
        return text

    # （2）院办用印申请流程 / （一）登录与认证
    # 注意：前面已经过滤了操作步骤
    if re.match(r"^[\(（][一二三四五六七八九十百\d]+[\)）]\s*.+", text):
        return text

    # 1.1 标题 / 2.3.1 标题
    # 只识别多级编号（至少一个点），避免把 "1、入口" 当成 section 标题
    if re.match(r"^\d+(\.\d+)+\s*[\.、．]?\s*.+", text):
        return text

    # 一、标题 / 二、标题
    # 在员工手册里常是大标题，保留
    if re.match(r"^[一二三四五六七八九十]+[、.．]\s*.+", text):
        return text

    # 单级数字编号 "1、xxx" / "2. xxx" / "3) xxx" → 不是 section 标题
    # 这些通常是列表项或描述性条目，应保留在当前 section 内部
    if re.match(r"^\d+[\.、．)]\s*", text):
        return None

    # 局部标签（不是 section 标题）
    local_labels = {
        "入口", "操作步骤", "办理流程", "适用范围", "注意事项",
        "常见问题", "故障处理", "权限说明", "登录方式",
    }
    if text in local_labels:
        return None

    # 短文本 + 强业务关键词 → 兜底识别
    if len(text) <= 30:
        strong_section_keywords = [
            "申请流程", "办理流程", "审批流程", "登录流程",
            "报销流程", "用印申请", "请假申请", "转正申请",
            "VPN连接", "密码重置", "账号管理",
        ]
        for kw in strong_section_keywords:
            if kw in text:
                return text

    return None


# ═══════════════════════════════════════════════════════════════
# 4. 操作步骤检测（系统手册专用）
# ═══════════════════════════════════════════════════════════════


def _is_action_step(text: str) -> bool:
    """判断文本是否为操作步骤（不应作为 section/chunk 边界）。

    判定条件：
    1. 以数字编号开头（如 "1." "2、" "3)"）
    2. 包含操作动词
    """
    text = text.strip()
    if not _RE_NUMBERED_START.match(text):
        return False

    for verb in _ACTION_VERBS:
        if verb in text:
            return True

    return False


# ═══════════════════════════════════════════════════════════════
# 5. Section 构建
# ═══════════════════════════════════════════════════════════════


def _new_section(
    section_title: str,
    source_file: str,
    doc_title: str,
    category: str,
    first_page: int,
) -> dict:
    """创建空 section 字典。"""
    return {
        "section_title": section_title,
        "segments": [],
        "page_start": first_page,
        "page_end": first_page,
        "source_file": source_file,
        "doc_title": doc_title,
        "category": category,
    }


def _build_sections(
    segments: list[dict],
    category: str,
    doc_title: str,
    source_file: str,
) -> list[dict]:
    """将 segment 列表归入 section，跨页合并。

    规则：
    - 遇到标题且当前 section 已有内容 → 保存旧 section，开新 section
    - 遇到标题但当前 section 还是空的 → 直接更新 section_title
    - 其他 segment 全部归入当前 section
    """
    if not segments:
        return []

    sections: list[dict] = []

    cur = _new_section(
        section_title=doc_title,
        source_file=source_file,
        doc_title=doc_title,
        category=category,
        first_page=segments[0]["page_number"],
    )

    for seg in segments:
        title = _detect_heading(seg["text"], category)

        if title and cur["segments"]:
            # 当前 section 已有内容，保存并开新 section
            sections.append(cur)
            cur = _new_section(
                section_title=title,
                source_file=source_file,
                doc_title=doc_title,
                category=category,
                first_page=seg["page_number"],
            )
        elif title and not cur["segments"]:
            # 当前 section 还是空的，直接改名
            cur["section_title"] = title
            cur["page_start"] = seg["page_number"]
            cur["page_end"] = seg["page_number"]

        cur["segments"].append(seg)
        cur["page_end"] = seg["page_number"]

    if cur["segments"]:
        sections.append(cur)

    return sections


# ═══════════════════════════════════════════════════════════════
# 6. Section 内 Chunk 切分
# ═══════════════════════════════════════════════════════════════


def _chunk_section(section: dict, chunk_index_start: int = 0) -> list[dict]:
    """在单个 section 内按长度 fallback 切分为 chunk。

    策略：
    1. section 总长度 ≤ CHUNK_MAX_SIZE → 整个 section 作为一个 chunk
    2. 超过 → 按 segment 逐步合并，超限时保存
    3. 系统手册中连续操作步骤不从中切断：如果切分点恰好落在两
       个连续操作步骤之间，则回溯到操作流程起点之前切分
    """
    segs = section["segments"]
    if not segs:
        return []

    total_size = sum(len(s["text"]) for s in segs)

    # 整个 section 不超限 → 单 chunk
    if total_size <= CHUNK_MAX_SIZE:
        body = "\n\n".join(s["text"] for s in segs)
        if len(body.strip()) < 20:
            return []
        return [{
            "text": body.strip(),
            "doc_title": section["doc_title"],
            "category": section["category"],
            "source_file": section["source_file"],
            "page_start": section["page_start"],
            "page_end": section["page_end"],
        }]

    # 超限 → fallback 切分
    chunks = []
    current_segs: list[dict] = []
    current_size = 0
    chunk_idx = chunk_index_start

    def _emit_chunk(segs: list[dict]) -> dict | None:
        if not segs:
            return None
        body = "\n\n".join(s["text"] for s in segs)
        if len(body.strip()) < 20:
            return None
        return {
            "text": body.strip(),
            "doc_title": section["doc_title"],
            "category": section["category"],
            "source_file": section["source_file"],
            "page_start": min(s["page_number"] for s in segs),
            "page_end": max(s["page_number"] for s in segs),
        }

    for i, seg in enumerate(segs):
        seg_len = len(seg["text"])

        # 当前 chunk 加上这个 segment 后超限
        if current_size + seg_len > CHUNK_MAX_SIZE and current_segs:
            if current_size >= CHUNK_MIN_SIZE:
                # 检查是否会在操作流程中间切断
                prev_is_step = _is_action_step(current_segs[-1]["text"])
                next_is_step = _is_action_step(seg["text"])

                if prev_is_step and next_is_step:
                    # 连续操作步骤 → 回溯找到操作流程起点，
                    # 把整个流程挪到下一个 chunk
                    proc_start = len(current_segs) - 1
                    while proc_start > 0 and _is_action_step(
                        current_segs[proc_start - 1]["text"]
                    ):
                        proc_start -= 1
                    # 如果流程起点前有一行短引导语（如"请按以下步骤操作："），
                    # 也一起挪过去
                    if proc_start > 0 and len(current_segs[proc_start - 1]["text"]) < 60:
                        proc_start -= 1

                    moved_segs = current_segs[proc_start:]
                    current_segs = current_segs[:proc_start]

                    # 保存前半部分（非空时）
                    chunk = _emit_chunk(current_segs)
                    if chunk:
                        chunks.append(chunk)
                        chunk_idx += 1

                    # 后半部分（操作流程）作为新 chunk 的起点
                    current_segs = list(moved_segs)
                    current_size = sum(len(s["text"]) for s in current_segs)
                else:
                    # 常规切分
                    chunk = _emit_chunk(current_segs)
                    if chunk:
                        chunks.append(chunk)
                        chunk_idx += 1

                    # overlap 逻辑：保留最后一个段作为下一 chunk 的前缀
                    if CHUNK_OVERLAP > 0 and len(current_segs) >= 2:
                        last_seg = current_segs[-1]
                        current_segs = [last_seg]
                        current_size = len(last_seg["text"])
                    else:
                        current_segs = []
                        current_size = 0

        current_segs.append(seg)
        current_size += seg_len

    # 最后剩余
    chunk = _emit_chunk(current_segs)
    if chunk:
        chunks.append(chunk)

    return chunks


# ═══════════════════════════════════════════════════════════════
# 6b. RecursiveCharacterTextSplitter 切分
# ═══════════════════════════════════════════════════════════════


def _chunk_section_recursive(section: dict) -> list[dict]:
    """使用 LangChain RecursiveCharacterTextSplitter 在单个 section 内切分。

    与 _chunk_section 的差别：
    - 不再手动按 segment 逐步合并，而是将整个 section 文本拼接后交由
      RecursiveCharacterTextSplitter 按语义分隔符逐级切分。
    - 分隔符顺序：段落 → 换行 → 句号 → 分号 → 逗号 → 空格 → 字符
    - 切分后将每个 chunk 的文本定位回原始 segments，提取页码信息。
    """
    segs = section["segments"]
    if not segs:
        return []

    # 构建分隔符：每个 segment 之间用 \n\n 连接
    segment_texts = [s["text"] for s in segs]

    # 全部拼接
    full_text = "\n\n".join(segment_texts)

    if len(full_text.strip()) < 20:
        return []

    # 整个 section 不超限 → 单 chunk
    if len(full_text) <= CHUNK_MAX_SIZE:
        return [{
            "text": full_text.strip(),
            "doc_title": section["doc_title"],
            "category": section["category"],
            "source_file": section["source_file"],
            "page_start": section["page_start"],
            "page_end": section["page_end"],
        }]

    # 初始化 RecursiveCharacterTextSplitter
    # 分隔符按优先级排列：段落 → 换行 → 句号 → 分号 → 逗号 → 空格 → 字符
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_MAX_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "；", "，", " ", ""],
        length_function=len,
        is_separator_regex=False,
    )

    split_texts = splitter.split_text(full_text)

    # 将每个 split 文本映射回原始 segments 以获取页码
    chunks = []
    search_start = 0
    for text in split_texts:
        stripped = text.strip()
        if len(stripped) < 20:
            continue

        # 在全文中定位该 chunk
        pos = full_text.find(stripped, search_start)
        if pos == -1:
            # fallback: 用前 40 个字符定位
            prefix = stripped[:40]
            pos = full_text.find(prefix, search_start)
        if pos == -1:
            pos = search_start

        chunk_end = pos + len(stripped)
        search_start = chunk_end

        # 根据字符位置找到对应的 segment，确定页码范围
        page_start = None
        page_end = None
        offset = 0
        for seg in segs:
            seg_len = len(seg["text"])
            seg_end = offset + seg_len
            if offset < chunk_end and seg_end > pos:
                if page_start is None:
                    page_start = seg["page_number"]
                page_end = seg["page_number"]
            offset = seg_end + 2  # +2 for "\n\n"

        if page_start is None:
            page_start = section["page_start"]
            page_end = section["page_end"]

        chunks.append({
            "text": stripped,
            "doc_title": section["doc_title"],
            "category": section["category"],
            "source_file": section["source_file"],
            "page_start": page_start,
            "page_end": page_end,
        })

    return chunks


def _is_procedure_boundary_between(prev_seg: dict, next_seg: dict) -> bool:
    """判断两个连续 segment 之间是否为操作流程边界。"""
    prev_is_step = _is_action_step(prev_seg["text"])
    next_is_step = _is_action_step(next_seg["text"])
    return not (prev_is_step and next_is_step)


# ═══════════════════════════════════════════════════════════════
# 7. Embedding 文本构建
# ═══════════════════════════════════════════════════════════════


def _build_embedding_text(body: str, section_title: str, doc_title: str) -> str:
    """构建含小节标题的 embedding 文本。

    格式：
        文档：{doc_title}
        小节：{section_title}

        正文：
        {body}

    如果 section_title 等于 doc_title，省略"小节"行。
    """
    lines = [f"文档：{doc_title}"]

    if section_title and section_title != doc_title:
        lines.append(f"小节：{section_title}")

    lines.append("")
    lines.append("正文：")
    lines.append(body)

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# 8. 主入口
# ═══════════════════════════════════════════════════════════════


def chunk_pages(pages: list[dict]) -> list[dict]:
    """对所有清洗后的页面进行 section-aware chunk 切分。

    流程: document pages → segments → sections → chunks

    按 source_file 分组（不同 PDF 独立处理），每个 PDF 内：
    1. 提取所有 segment（保留页码元数据）
    2. 根据标题识别构建 section（跨页合并）
    3. section 内部按长度 fallback 切分
    4. 为每个 chunk 附加 section_title 并构建 embedding 文本
    """
    if not pages:
        return []

    # 按 (source_file, doc_title, category) 分组
    def _key(p):
        return (p["source_file"], p["doc_title"], p["category"])

    sorted_pages = sorted(pages, key=_key)

    all_chunks = []
    global_section_idx = 0

    for (source_file, doc_title, category), doc_pages in groupby(sorted_pages, key=_key):
        doc_pages = list(doc_pages)
        doc_pages.sort(key=lambda p: p["page_number"])

        # Step 1: segment 提取
        segments = _extract_segments_from_pages(doc_pages)
        if not segments:
            continue

        # Step 2: section 构建
        sections = _build_sections(segments, category, doc_title, source_file)

        # Step 3-4: chunk 切分 + embedding 文本
        _chunk_fn = _chunk_section if CHUNK_METHOD == "section_aware" else _chunk_section_recursive
        for section in sections:
            section_chunks = _chunk_fn(section)

            for chunk in section_chunks:
                body_text = chunk["text"]

                chunk_dict = {
                    "chunk_id": f"{category}_s{global_section_idx}_c{len(all_chunks)}",
                    "text": _build_embedding_text(body_text, section["section_title"], doc_title),
                    "body_text": body_text,
                    "doc_title": doc_title,
                    "section_title": section["section_title"],
                    "page_start": chunk["page_start"],
                    "page_end": chunk["page_end"],
                    "source_file": source_file,
                    "category": category,
                }
                all_chunks.append(chunk_dict)

            global_section_idx += 1

    return all_chunks
