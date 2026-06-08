"""
Evidence-level retrieval evaluation 指标模块。

提供文本归一化、char 3-gram containment、ROUGE-L recall、
partial_ratio、exact/fuzzy evidence match 以及 Hit@K / MRR / Recall@K
等 evidence-level 评测指标。

所有匹配逻辑均为纯 Python 实现，不依赖外部 NLP 库。
"""

import re
import unicodedata


# ═══════════════════════════════════════════════════════════════
# 默认阈值
# ═══════════════════════════════════════════════════════════════

DEFAULT_CONTAINMENT_THRESHOLD = 0.75
DEFAULT_ROUGE_L_THRESHOLD = 0.65
DEFAULT_PARTIAL_RATIO_THRESHOLD = 0.85


# ═══════════════════════════════════════════════════════════════
# 1. 文本归一化
# ═══════════════════════════════════════════════════════════════

def normalize_text(text: str) -> str:
    """对文本做统一归一化处理。

    步骤：
    1. unicodedata.normalize("NFKC")  — 全角→半角、兼容字符归一化
    2. 转小写
    3. 去除所有空白字符（空格、换行、制表等）
    4. 去除中英文标点符号（仅保留中文、英文、数字）

    Args:
        text: 原始文本

    Returns:
        归一化后的文本（纯中英数字符，无标点无空白）
    """
    # NFKC 归一化：全角→半角、兼容字符等
    text = unicodedata.normalize("NFKC", text)
    # 转小写
    text = text.lower()
    # 去除所有空白字符
    text = re.sub(r"\s+", "", text)
    # 去除中英文标点，只保留：
    #   一-鿿  CJK统一汉字
    #   㐀-䶿  CJK扩展A
    #   a-z           英文小写
    #   0-9           数字
    text = re.sub(r"[^一-鿿㐀-䶿a-z0-9]", "", text)
    return text


# ═══════════════════════════════════════════════════════════════
# 2. char 3-gram 工具
# ═══════════════════════════════════════════════════════════════

def char_3grams(text: str) -> set[str]:
    """生成 char 3-gram 集合。

    对长度 < 3 的文本返回空集。

    Args:
        text: 归一化后的文本

    Returns:
        char 3-gram 集合
    """
    if len(text) < 3:
        return set()
    return {text[i:i + 3] for i in range(len(text) - 2)}


def char_3gram_containment(evidence: str, chunk: str) -> float:
    """计算 evidence 的 char 3-gram 在 chunk 中的 container ratio。

    containment = |evidence_ngrams ∩ chunk_ngrams| / |evidence_ngrams|

    使用 containment（而非 Jaccard），因为 chunk 通常比 evidence 长很多。

    Args:
        evidence: 归一化后的 evidence 文本
        chunk: 归一化后的 chunk 文本

    Returns:
        containment 比例 [0, 1]
    """
    e_grams = char_3grams(evidence)
    if not e_grams:
        return 0.0
    c_grams = char_3grams(chunk)
    return len(e_grams & c_grams) / len(e_grams)


# ═══════════════════════════════════════════════════════════════
# 3. ROUGE-L recall
# ═══════════════════════════════════════════════════════════════

def _lcs_length(a: str, b: str) -> int:
    """计算两个字符串的最长公共子序列（LCS）长度。

    使用 O(m*n) 动态规划，空间优化为 O(min(m,n))。
    """
    if not a or not b:
        return 0

    # 确保 b 是较短的，减少空间
    if len(a) < len(b):
        a, b = b, a

    prev = [0] * (len(b) + 1)
    curr = [0] * (len(b) + 1)

    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr, prev

    return prev[len(b)]


def rouge_l_recall(evidence: str, chunk: str) -> float:
    """计算 ROUGE-L recall。

    rouge_l_recall = LCS(evidence, chunk) / len(evidence)

    Args:
        evidence: 归一化后的 evidence 文本
        chunk: 归一化后的 chunk 文本

    Returns:
        ROUGE-L recall 值 [0, 1]
    """
    if not evidence:
        return 0.0
    lcs = _lcs_length(evidence, chunk)
    return lcs / len(evidence)


# ═══════════════════════════════════════════════════════════════
# 4. partial_ratio（纯 Python 实现）
# ═══════════════════════════════════════════════════════════════

def partial_ratio(evidence: str, chunk: str) -> float:
    """计算 partial_ratio：滑动短串在长串上，取最佳 char 3-gram overlap。

    等价于 rapidfuzz.fuzz.partial_ratio 的逻辑，但对中文使用 char 3-gram
    而非逐字符编辑距离，更适合中文语义匹配。

    算法：
    1. 将较短的 evidence 在较长的 chunk 上逐字符滑动
    2. 对每个窗口位置，计算 evidence 与窗口文本的 char 3-gram overlap
    3. 返回最佳 overlap 比例

    Args:
        evidence: 归一化后的 evidence 文本
        chunk: 归一化后的 chunk 文本

    Returns:
        partial_ratio 值 [0, 1]
    """
    if not evidence or not chunk:
        return 0.0

    # evidence 是短串，chunk 是长串
    short, long = (evidence, chunk) if len(evidence) <= len(chunk) else (chunk, evidence)
    if len(short) < 3 or len(long) < 3:
        # 对于极短文本（<3 字符），降级为单字符容斥
        if len(short) == 0 or len(long) == 0:
            return 0.0
        short_chars = set(short)
        long_chars = set(long)
        return len(short_chars & long_chars) / len(short_chars)

    short_grams = char_3grams(short)
    if not short_grams:
        return 0.0

    best = 0.0
    # 滑动窗口：evidence 长度在 chunk 上滑动
    for i in range(len(long) - len(short) + 1):
        window = long[i:i + len(short)]
        w_grams = char_3grams(window)
        if not w_grams:
            continue
        overlap = len(short_grams & w_grams)
        total = len(short_grams)
        ratio = overlap / total
        if ratio > best:
            best = ratio

    return best


# ═══════════════════════════════════════════════════════════════
# 5. Evidence match 判定
# ═══════════════════════════════════════════════════════════════

def exact_evidence_match(evidence_quote: str, chunk_text: str) -> bool:
    """判断 evidence.quote 是否 exact match chunk 文本。

    判定条件：normalize 后的 evidence.quote 是 normalize 后的 chunk text 的子串。

    Args:
        evidence_quote: evidence 的 quote 原文
        chunk_text: retrieved chunk 的 body_text（或 text）

    Returns:
        True 如果 exact match
    """
    e_norm = normalize_text(evidence_quote)
    c_norm = normalize_text(chunk_text)
    if not e_norm:
        return False
    return e_norm in c_norm


def fuzzy_evidence_match(
    evidence_quote: str,
    chunk_text: str,
    containment_threshold: float = DEFAULT_CONTAINMENT_THRESHOLD,
    rouge_l_threshold: float = DEFAULT_ROUGE_L_THRESHOLD,
    partial_ratio_threshold: float = DEFAULT_PARTIAL_RATIO_THRESHOLD,
) -> tuple[bool, dict]:
    """判断 evidence.quote 是否 fuzzy match chunk 文本。

    判定规则（满足任一即命中）：
    1. exact_match 为 True
    2. char_3gram_containment >= containment_threshold 且 rouge_l_recall >= rouge_l_threshold
    3. partial_ratio >= partial_ratio_threshold

    Args:
        evidence_quote: evidence 的 quote 原文
        chunk_text: retrieved chunk 的 body_text（或 text）
        containment_threshold: char 3-gram containment 阈值
        rouge_l_threshold: ROUGE-L recall 阈值
        partial_ratio_threshold: partial_ratio 阈值

    Returns:
        (is_match, scores_dict)
        - is_match: 是否命中
        - scores_dict: {'containment', 'rouge_l', 'partial_ratio', 'exact_match'}
    """
    e_norm = normalize_text(evidence_quote)
    c_norm = normalize_text(chunk_text)

    scores = {
        "containment": 0.0,
        "rouge_l": 0.0,
        "partial_ratio": 0.0,
        "exact_match": False,
    }

    if not e_norm or not c_norm:
        return False, scores

    # Exact match
    if e_norm in c_norm:
        scores["exact_match"] = True
        # 对于 exact match，其他分数也设高
        scores["containment"] = 1.0
        scores["rouge_l"] = 1.0
        scores["partial_ratio"] = 1.0
        return True, scores

    # 计算三项 fuzzy 指标
    containment = char_3gram_containment(e_norm, c_norm)
    rouge_l = rouge_l_recall(e_norm, c_norm)
    pr = partial_ratio(e_norm, c_norm)

    scores["containment"] = round(containment, 4)
    scores["rouge_l"] = round(rouge_l, 4)
    scores["partial_ratio"] = round(pr, 4)

    # Rule 1: containment + rouge_l 联合
    if containment >= containment_threshold and rouge_l >= rouge_l_threshold:
        return True, scores

    # Rule 2: partial_ratio
    if pr >= partial_ratio_threshold:
        return True, scores

    return False, scores


# ═══════════════════════════════════════════════════════════════
# 6. Evidence-level 聚合指标
# ═══════════════════════════════════════════════════════════════

def _best_match_for_entry(
    chunks: list[dict],
    evidence_list: list[dict],
    match_fn,
) -> dict | None:
    """找到 Top-K chunks 中第一个命中任意 evidence 的 chunk。

    同时记录最佳匹配分数。

    Args:
        chunks: retrieved chunks（已按 rank 排序）
        evidence_list: 样本的 evidence 数组
        match_fn: 匹配函数，签名为 (evidence_quote, chunk_text) -> (bool, scores_dict)

    Returns:
        命中信息 dict 或 None
    """
    best_scores = None
    best_rank = None
    best_chunk = None
    best_evidence = None

    for rank, chunk in enumerate(chunks, start=1):
        chunk_text = chunk.get("body_text") or chunk.get("text", "")
        for ev in evidence_list:
            quote = ev.get("quote", "")
            if not quote:
                continue
            is_match, scores = match_fn(quote, chunk_text)

            # 跟踪最佳分数（即使未命中也记录，用于输出分析）
            if best_scores is None or scores.get("partial_ratio", 0) > best_scores.get("partial_ratio", 0):
                best_scores = scores
                best_rank = rank
                best_chunk = chunk
                best_evidence = ev

            if is_match:
                return {
                    "rank": rank,
                    "chunk": chunk,
                    "evidence": ev,
                    "scores": scores,
                    "is_exact": scores.get("exact_match", False),
                }

    # 无命中，但返回最佳接近匹配供分析
    if best_scores is not None:
        return {
            "rank": -1,
            "chunk": best_chunk,
            "evidence": best_evidence,
            "scores": best_scores,
            "is_exact": False,
            "is_best_effort": True,
        }

    return None


def evidence_hit_at_k(
    chunks: list[dict],
    evidence_list: list[dict],
    k: int,
    match_fn,
) -> tuple[bool, dict | None]:
    """检查 Top-K chunks 是否命中任意 evidence。

    Args:
        chunks: retrieved chunks
        evidence_list: 样本的 evidence 数组
        k: 取前 k 个 chunk
        match_fn: 匹配函数

    Returns:
        (是否命中, 最佳命中信息)
    """
    best = _best_match_for_entry(chunks[:k], evidence_list, match_fn)
    if best is None:
        return False, None
    return best["rank"] > 0, best


def evidence_mrr(
    chunks: list[dict],
    evidence_list: list[dict],
    match_fn,
) -> tuple[float, dict | None]:
    """计算 evidence MRR：第一个命中 evidence 的 chunk 的 rank 的倒数。

    Args:
        chunks: retrieved chunks
        evidence_list: 样本的 evidence 数组
        match_fn: 匹配函数

    Returns:
        (mrr_value, 最佳命中信息)
    """
    best = _best_match_for_entry(chunks, evidence_list, match_fn)
    if best is None or best["rank"] <= 0:
        return 0.0, best
    return 1.0 / best["rank"], best


def evidence_recall_at_k(
    chunks: list[dict],
    evidence_list: list[dict],
    k: int,
    match_fn,
) -> float:
    """计算 Top-K 中命中的 evidence 数量比例。

    evidence_recall@K = 命中的 evidence 数 / evidence 总数

    Args:
        chunks: retrieved chunks（前 k 个）
        evidence_list: 样本的 evidence 数组
        k: 取前 k 个 chunk
        match_fn: 匹配函数

    Returns:
        recall 值 [0, 1]
    """
    total = len(evidence_list)
    if total == 0:
        return 0.0

    matched_evidence_indices = set()
    for rank, chunk in enumerate(chunks[:k], start=1):
        chunk_text = chunk.get("body_text") or chunk.get("text", "")
        for idx, ev in enumerate(evidence_list):
            if idx in matched_evidence_indices:
                continue
            quote = ev.get("quote", "")
            if not quote:
                continue
            is_match, _ = match_fn(quote, chunk_text)
            if is_match:
                matched_evidence_indices.add(idx)

    return len(matched_evidence_indices) / total


def evidence_unit_mrr(
    chunks: list[dict],
    evidence_units: list[str],
    match_fn,
) -> float:
    """计算 evidence unit 级别的 MRR。

    从 k=1 开始逐个增加 chunk，找到首次 evidence_group_full_hit 的 k
    （即所有 unit 都被覆盖），返回 1/k。若全程未全部命中则返回 0.0。

    Args:
        chunks: retrieved chunks 列表
        evidence_units: evidence unit 文本列表
        match_fn: 匹配函数

    Returns:
        mrr_value [0, 1]
    """
    if not evidence_units:
        return 0.0
    for r in range(1, len(chunks) + 1):
        metrics = compute_evidence_unit_metrics(chunks, evidence_units, r, match_fn)
        if metrics["evidence_group_full_hit"]:
            return 1.0 / r
    return 0.0


# ═══════════════════════════════════════════════════════════════
# 7. 便捷函数：预先绑定 match_fn
# ═══════════════════════════════════════════════════════════════

def make_exact_matcher():
    """创建 exact match 函数（用于 evidence-level 评测）。"""
    def match(quote: str, chunk_text: str) -> tuple[bool, dict]:
        is_match = exact_evidence_match(quote, chunk_text)
        scores = {"exact_match": is_match, "containment": 0.0, "rouge_l": 0.0, "partial_ratio": 0.0}
        if is_match:
            scores["containment"] = 1.0
            scores["rouge_l"] = 1.0
            scores["partial_ratio"] = 1.0
        return is_match, scores
    return match


def make_rouge_l_matcher(
    threshold: float = DEFAULT_ROUGE_L_THRESHOLD,
):
    """创建纯 ROUGE-L 匹配函数（用于独立测试 ROUGE-L 效果）。

    Args:
        threshold: ROUGE-L recall 阈值，达到即认为命中

    Returns:
        match 函数，签名同 make_exact_matcher / make_fuzzy_matcher
    """
    def match(quote: str, chunk_text: str) -> tuple[bool, dict]:
        e_norm = normalize_text(quote)
        c_norm = normalize_text(chunk_text)
        rouge_l = rouge_l_recall(e_norm, c_norm)
        containment = char_3gram_containment(e_norm, c_norm)
        pr = partial_ratio(e_norm, c_norm)
        scores = {
            "exact_match": e_norm in c_norm if e_norm and c_norm else False,
            "containment": round(containment, 4),
            "rouge_l": round(rouge_l, 4),
            "partial_ratio": round(pr, 4),
        }
        is_match = rouge_l >= threshold
        return is_match, scores
    return match


def make_containment_matcher(
    threshold: float = DEFAULT_CONTAINMENT_THRESHOLD,
):
    """创建纯 char 3-gram containment 匹配函数。

    Args:
        threshold: containment 阈值，达到即认为命中
    """
    def match(quote: str, chunk_text: str) -> tuple[bool, dict]:
        e_norm = normalize_text(quote)
        c_norm = normalize_text(chunk_text)
        rouge_l = rouge_l_recall(e_norm, c_norm)
        containment = char_3gram_containment(e_norm, c_norm)
        pr = partial_ratio(e_norm, c_norm)
        scores = {
            "exact_match": e_norm in c_norm if e_norm and c_norm else False,
            "containment": round(containment, 4),
            "rouge_l": round(rouge_l, 4),
            "partial_ratio": round(pr, 4),
        }
        is_match = containment >= threshold
        return is_match, scores
    return match


def make_partial_ratio_matcher(
    threshold: float = DEFAULT_PARTIAL_RATIO_THRESHOLD,
):
    """创建纯 partial_ratio 匹配函数。

    Args:
        threshold: partial_ratio 阈值，达到即认为命中
    """
    def match(quote: str, chunk_text: str) -> tuple[bool, dict]:
        e_norm = normalize_text(quote)
        c_norm = normalize_text(chunk_text)
        rouge_l = rouge_l_recall(e_norm, c_norm)
        containment = char_3gram_containment(e_norm, c_norm)
        pr = partial_ratio(e_norm, c_norm)
        scores = {
            "exact_match": e_norm in c_norm if e_norm and c_norm else False,
            "containment": round(containment, 4),
            "rouge_l": round(rouge_l, 4),
            "partial_ratio": round(pr, 4),
        }
        is_match = pr >= threshold
        return is_match, scores
    return match


def make_fuzzy_matcher(
    containment_threshold: float = DEFAULT_CONTAINMENT_THRESHOLD,
    rouge_l_threshold: float = DEFAULT_ROUGE_L_THRESHOLD,
    partial_ratio_threshold: float = DEFAULT_PARTIAL_RATIO_THRESHOLD,
):
    """创建 fuzzy match 函数（用于 evidence-level 评测）。"""
    def match(quote: str, chunk_text: str) -> tuple[bool, dict]:
        return fuzzy_evidence_match(
            quote, chunk_text,
            containment_threshold=containment_threshold,
            rouge_l_threshold=rouge_l_threshold,
            partial_ratio_threshold=partial_ratio_threshold,
        )
    return match


# ═══════════════════════════════════════════════════════════════
# 8. Evidence Unit 支持（长 evidence 跨 chunk 评估）
# ═══════════════════════════════════════════════════════════════

def get_evidence_units(entry: dict) -> list[str]:
    """从评测样本中提取 evidence_units。

    向后兼容：
    - 如果有 `evidence_units` 字段，使用它
    - 否则从 `evidence` 数组中提取所有 quote，每个 quote 作为一个 unit

    Args:
        entry: 评测样本 dict

    Returns:
        evidence unit 文本列表
    """
    # 优先使用 evidence_units
    if "evidence_units" in entry and entry["evidence_units"]:
        return [u for u in entry["evidence_units"] if u and u.strip()]

    # 向后兼容：从 evidence 中提取
    evidence_list = entry.get("evidence", [])
    if evidence_list:
        return [ev.get("quote", "") for ev in evidence_list if ev.get("quote", "").strip()]

    return []


def build_retrieved_context(chunks: list[dict], k: int) -> str:
    """将 top-k retrieved chunks 的文本拼接为检索上下文。

    拼接时使用换行符分隔，供 evidence unit 匹配使用。

    Args:
        chunks: retrieved chunks 列表
        k: 取前 k 个 chunk

    Returns:
        拼接后的文本
    """
    texts = []
    for chunk in chunks[:k]:
        text = chunk.get("body_text") or chunk.get("text", "")
        if text:
            texts.append(text)
    return "\n".join(texts)


def match_evidence_unit_in_context(
    unit_text: str,
    context_text: str,
    match_fn,
) -> tuple[bool, dict]:
    """判断一个 evidence unit 是否在检索上下文（拼接后的 top-k chunk 文本）中命中。

    匹配流程：
    1. 先做 normalized exact containment（unit 归一化后是 context 归一化后的子串）
    2. 如果 exact 不命中，且 match_fn 不为 None，则用 match_fn 做 fuzzy matching
    3. 如果 match_fn 为 None，则不做 fuzzy matching（纯精确模式）

    Args:
        unit_text: evidence unit 原文
        context_text: 拼接后的 top-k chunk 文本
        match_fn: 匹配函数，签名为 (quote, chunk_text) -> (bool, scores_dict)
                  传 None 表示仅做精确匹配，不做模糊回退

    Returns:
        (is_match, scores_dict)
    """
    empty_scores = {"exact_match": False, "containment": 0.0, "rouge_l": 0.0, "partial_ratio": 0.0}

    if not unit_text or not context_text:
        return False, empty_scores

    u_norm = normalize_text(unit_text)
    c_norm = normalize_text(context_text)

    if not u_norm or not c_norm:
        return False, empty_scores

    # Step 1: normalized exact containment
    if u_norm in c_norm:
        return True, {
            "exact_match": True,
            "containment": 1.0,
            "rouge_l": 1.0,
            "partial_ratio": 1.0,
        }

    # Step 2: fuzzy matching (only if match_fn is provided)
    if match_fn is not None:
        return match_fn(unit_text, context_text)

    return False, empty_scores


def compute_evidence_unit_metrics(
    chunks: list[dict],
    evidence_units: list[str],
    k: int,
    match_fn,
) -> dict:
    """计算 evidence unit 级别的检索指标。

    将 top-k chunks 的文本拼接后，逐一检查每个 evidence unit 是否命中。

    unit_hit_i@k = evidence unit i 是否被 top-k context 覆盖

    Args:
        chunks: retrieved chunks 列表
        evidence_units: evidence unit 文本列表
        k: 取前 k 个 chunk
        match_fn: 匹配函数

    Returns:
        {
            "evidence_unit_total": int,
            "evidence_unit_hit_count": int,
            "evidence_unit_recall": float,
            "evidence_group_any_hit": bool (hit_count > 0),
            "evidence_group_full_hit": bool (hit_count == total),
            "matched_units": [matched unit texts],
            "missing_units": [missing unit texts],
            "unit_details": [{unit, is_match, scores}, ...],
        }
    """
    total = len(evidence_units)
    if total == 0:
        return {
            "evidence_unit_total": 0,
            "evidence_unit_hit_count": 0,
            "evidence_unit_recall": 0.0,
            "evidence_group_any_hit": False,
            "evidence_group_full_hit": False,
            "matched_units": [],
            "missing_units": [],
            "unit_details": [],
        }

    context_text = build_retrieved_context(chunks, k)

    matched_units = []
    missing_units = []
    unit_details = []
    hit_count = 0

    for unit_text in evidence_units:
        is_match, scores = match_evidence_unit_in_context(unit_text, context_text, match_fn)
        unit_details.append({
            "unit": unit_text,
            "is_match": is_match,
            "scores": scores,
        })
        if is_match:
            hit_count += 1
            matched_units.append(unit_text)
        else:
            missing_units.append(unit_text)

    return {
        "evidence_unit_total": total,
        "evidence_unit_hit_count": hit_count,
        "evidence_unit_recall": hit_count / total,
        "evidence_group_any_hit": hit_count > 0,
        "evidence_group_full_hit": hit_count == total,
        "matched_units": matched_units,
        "missing_units": missing_units,
        "unit_details": unit_details,
    }
