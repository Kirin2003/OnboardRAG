#!/usr/bin/env python3
"""
run_eval.py — OnboardRAG 评测脚本。

从 data/eval/eval_queries.jsonl 读取评测样本，执行检索评测。

支持三种模式：
  - doc:      仅 doc-level 检索评测（Hit@K, MRR）
  - evidence: 仅 evidence-level 检索评测（exact/fuzzy Hit@K, MRR, Recall@K）
  - all:      同时运行 doc + evidence 评测

用法:
    # 仅 doc-level 检索评测
    python scripts/run_eval.py --mode doc

    # 仅 evidence-level 检索评测（默认 fuzzy 三合一）
    python scripts/run_eval.py --mode evidence

    # 单独测试 ROUGE-L 匹配效果
    python scripts/run_eval.py --mode evidence --match-method rouge_l --rouge-l-threshold 0.7

    # 单独测试 containment 匹配效果
    python scripts/run_eval.py --mode evidence --match-method containment --containment-threshold 0.8

    # 单独测试 partial_ratio 匹配效果
    python scripts/run_eval.py --mode evidence --match-method partial_ratio --partial-ratio-threshold 0.85

    # 仅 exact match（跳过所有模糊匹配）
    python scripts/run_eval.py --mode evidence --exact-only

    # 全量评测（doc + evidence）
    python scripts/run_eval.py --mode all

    # 指定评测文件和输出目录
    python scripts/run_eval.py --eval-file data/eval/eval_queries.jsonl --output-dir outputs/eval/

    # 自定义 evidence 匹配阈值
    python scripts/run_eval.py --mode evidence --containment-threshold 0.8

    # 消融实验：指定检索模式
    python scripts/run_eval.py --mode all --retrieval-mode bm25   # 仅关键词检索
    python scripts/run_eval.py --mode all --retrieval-mode dense  # 仅向量检索
    python scripts/run_eval.py --mode all --retrieval-mode hybrid # 混合检索（默认）
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

# 将项目根目录加入 sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.retriever import Retriever
from src.config import RETRIEVAL_TOP_K
from src.eval_metrics import (
    evidence_hit_at_k,
    evidence_mrr,
    evidence_recall_at_k,
    get_evidence_units,
    compute_evidence_unit_metrics,
    make_exact_matcher,
    make_fuzzy_matcher,
    make_rouge_l_matcher,
    make_containment_matcher,
    make_partial_ratio_matcher,
    DEFAULT_CONTAINMENT_THRESHOLD,
    DEFAULT_ROUGE_L_THRESHOLD,
    DEFAULT_PARTIAL_RATIO_THRESHOLD,
)


# ═══════════════════════════════════════════════════════════════
# 检索评测
# ═══════════════════════════════════════════════════════════════

def _hit_at_k(expected_docs: set[str], retrieved_files: list[str], k: int) -> bool:
    """检查 top-k 检索结果中是否命中任意 expected_doc。"""
    return any(ed in rf or rf in ed
               for ed in expected_docs
               for rf in retrieved_files[:k])


def _mrr(expected_docs: set[str], retrieved_files: list[str]) -> float:
    """计算倒数排名（Mean Reciprocal Rank）。"""
    for rank, rf in enumerate(retrieved_files, start=1):
        if any(ed in rf or rf in ed for ed in expected_docs):
            return 1.0 / rank
    return 0.0


def eval_retrieval(entries: list[dict], top_k: int = RETRIEVAL_TOP_K, mode: str = "hybrid") -> dict:
    """执行检索评测。

    对每条 answerable=True 的样本，执行检索，
    计算 Hit@3、Hit@5、MRR。

    Args:
        entries: 评测样本列表
        top_k: 检索返回的 chunk 数量
        mode: 检索模式 ("hybrid" / "dense" / "bm25")

    Returns:
        包含整体指标和每条样本结果的字典
    """
    retriever = Retriever()
    results = []
    hits_3, hits_5 = 0, 0
    total_mrr = 0.0
    total_answerable = 0

    for entry in entries:
        if not entry["answerable"]:
            results.append({
                "id": entry["id"],
                "query": entry["query"],
                "intent": entry["intent"],
                "answerable": False,
            })
            continue

        total_answerable += 1
        chunks = retriever.retrieve(entry["query"], top_k=top_k, mode=mode)
        retrieved_files = [c.get("source_file", "") for c in chunks]
        expected_docs = set(entry["expected_docs"])

        hit3 = _hit_at_k(expected_docs, retrieved_files, 3)
        hit5 = _hit_at_k(expected_docs, retrieved_files, 5)
        mrr = _mrr(expected_docs, retrieved_files)

        hits_3 += int(hit3)
        hits_5 += int(hit5)
        total_mrr += mrr

        results.append({
            "id": entry["id"],
            "query": entry["query"],
            "intent": entry["intent"],
            "difficulty": entry["difficulty"],
            "answerable": True,
            "hit@3": hit3,
            "hit@5": hit5,
            "mrr": round(mrr, 4),
            "retrieved_docs": retrieved_files[:top_k],
        })

    return {
        "summary": {
            "total_samples": len(entries),
            "total_answerable": total_answerable,
            "hit@3": round(hits_3 / total_answerable, 4) if total_answerable else 0,
            "hit@5": round(hits_5 / total_answerable, 4) if total_answerable else 0,
            "mrr": round(total_mrr / total_answerable, 4) if total_answerable else 0,
        },
        "details": results,
    }


# ═══════════════════════════════════════════════════════════════
# 按意图 / 难度分组统计
# ═══════════════════════════════════════════════════════════════

def print_retrieval_report(retrieval_result: dict, mode: str = "hybrid") -> None:
    """打印检索评测报告，按意图和难度分组。"""
    summary = retrieval_result["summary"]
    details = retrieval_result["details"]

    print("\n" + "=" * 60)
    print(f"  检索评测结果（模式: {mode}）")
    print("=" * 60)
    print(f"  样本总数: {summary['total_samples']}")
    print(f"  可回答数: {summary['total_answerable']}")
    print(f"  Hit@3:    {summary['hit@3']:.2%}")
    print(f"  Hit@5:    {summary['hit@5']:.2%}")
    print(f"  MRR:      {summary['mrr']:.4f}")

    # 按意图分组
    answerable = [d for d in details if d.get("answerable")]
    by_intent = defaultdict(list)
    for d in answerable:
        by_intent[d["intent"]].append(d)

    print("\n── 按意图 ──")
    print(f"  {'意图':<16} {'数量':<6} {'Hit@3':<8} {'Hit@5':<8} {'MRR':<8}")
    for intent in ["人事与员工事务", "账号与统一门户", "内网访问", "办公平台"]:
        items = by_intent.get(intent, [])
        if not items:
            continue
        n = len(items)
        h3 = sum(d["hit@3"] for d in items) / n
        h5 = sum(d["hit@5"] for d in items) / n
        mrr = sum(d["mrr"] for d in items) / n
        print(f"  {intent:<16} {n:<6} {h3:<8.2%} {h5:<8.2%} {mrr:<8.4f}")

    # 按难度分组
    by_diff = defaultdict(list)
    for d in answerable:
        by_diff[d["difficulty"]].append(d)

    print("\n── 按难度 ──")
    print(f"  {'难度':<10} {'数量':<6} {'Hit@3':<8} {'Hit@5':<8} {'MRR':<8}")
    for diff in ["simple", "medium", "hard"]:
        items = by_diff.get(diff, [])
        if not items:
            continue
        n = len(items)
        h3 = sum(d["hit@3"] for d in items) / n
        h5 = sum(d["hit@5"] for d in items) / n
        mrr = sum(d["mrr"] for d in items) / n
        print(f"  {diff:<10} {n:<6} {h3:<8.2%} {h5:<8.2%} {mrr:<8.4f}")


# ═══════════════════════════════════════════════════════════════
# Evidence-Level 检索评测
# ═══════════════════════════════════════════════════════════════

def eval_evidence_retrieval(
    entries: list[dict],
    top_k: int = RETRIEVAL_TOP_K,
    containment_threshold: float = DEFAULT_CONTAINMENT_THRESHOLD,
    rouge_l_threshold: float = DEFAULT_ROUGE_L_THRESHOLD,
    partial_ratio_threshold: float = DEFAULT_PARTIAL_RATIO_THRESHOLD,
    mode: str = "hybrid",
    exact_only: bool = False,
    match_method: str = "fuzzy",
) -> dict:
    """执行 evidence-level 检索评测。

    只对 answerable=true 且 evidence 非空的样本计算指标。
    同时计算 exact match + 指定方法的匹配结果。
    始终返回每条的详细证据匹配信息（verbose_data）。

    Args:
        entries: 评测样本列表
        top_k: 检索返回的 chunk 数量
        containment_threshold: char 3-gram containment 阈值
        rouge_l_threshold: ROUGE-L recall 阈值
        partial_ratio_threshold: partial_ratio 阈值
        mode: 检索模式 ("hybrid" / "dense" / "bm25")
        exact_only: 仅计算 exact match，跳过第二匹配器
        match_method: 第二匹配器方法（仅在 exact_only=False 时生效）
            - "fuzzy": 三合一模糊匹配（默认）
            - "rouge_l": 纯 ROUGE-L 匹配
            - "containment": 纯 char 3-gram containment 匹配
            - "partial_ratio": 纯 partial_ratio 匹配

    Returns:
        {summary: {...}, details: [...], verbose_data: [...]}
    """
    from src.eval_metrics import normalize_text

    retriever = Retriever()
    exact_match = make_exact_matcher()

    # 根据 exact_only / match_method 选择第二匹配器
    if exact_only:
        second_match = None
        second_label = "exact"  # summary 中不产生额外字段
    elif match_method == "rouge_l":
        second_match = make_rouge_l_matcher(threshold=rouge_l_threshold)
        second_label = "rouge_l"
    elif match_method == "containment":
        second_match = make_containment_matcher(threshold=containment_threshold)
        second_label = "containment"
    elif match_method == "partial_ratio":
        second_match = make_partial_ratio_matcher(threshold=partial_ratio_threshold)
        second_label = "partial_ratio"
    else:  # "fuzzy" (default)
        second_match = make_fuzzy_matcher(
            containment_threshold=containment_threshold,
            rouge_l_threshold=rouge_l_threshold,
            partial_ratio_threshold=partial_ratio_threshold,
        )
        second_label = "fuzzy"

    results = []
    verbose_data = []  # 始终收集详细匹配信息

    # exact 计数器
    e_hit3, e_hit5, e_mrr_sum = 0, 0, 0.0
    e_rec3_sum, e_rec5_sum = 0.0, 0.0
    # fuzzy 计数器
    f_hit3, f_hit5, f_mrr_sum = 0, 0, 0.0
    f_rec3_sum, f_rec5_sum = 0.0, 0.0
    # evidence unit 计数器（基于 second_match / exact）
    eu_rec3_sum, eu_rec5_sum = 0.0, 0.0
    eu_hit3_sum, eu_hit5_sum = 0, 0
    eu_total_sum = 0
    eu_group_hit3_sum, eu_group_hit5_sum = 0, 0
    total_valid = 0

    for entry in entries:
        # 跳过不可回答或无 evidence 的样本
        if not entry.get("answerable", True):
            results.append({
                "id": entry["id"], "query": entry["query"],
                "intent": entry["intent"], "answerable": False,
            })
            continue

        evidence_list = entry.get("evidence", [])
        if not evidence_list:
            results.append({
                "id": entry["id"], "query": entry["query"],
                "intent": entry["intent"], "answerable": True,
                "evidence_count": 0, "skipped": True,
                "skip_reason": "evidence 为空",
            })
            continue

        total_valid += 1
        chunks = retriever.retrieve(entry["query"], top_k=top_k, mode=mode)

        # ── Exact match 指标 ──
        e_hit3_val, e_best3 = evidence_hit_at_k(chunks, evidence_list, 3, exact_match)
        e_hit5_val, e_best5 = evidence_hit_at_k(chunks, evidence_list, 5, exact_match)
        e_mrr_val, e_best_all = evidence_mrr(chunks, evidence_list, exact_match)
        e_rec3 = evidence_recall_at_k(chunks, evidence_list, 3, exact_match)
        e_rec5 = evidence_recall_at_k(chunks, evidence_list, 5, exact_match)

        # ── 第二匹配器指标（fuzzy / rouge_l / containment / partial_ratio）──
        if second_match is not None:
            f_hit3_val, _ = evidence_hit_at_k(chunks, evidence_list, 3, second_match)
            f_hit5_val, _ = evidence_hit_at_k(chunks, evidence_list, 5, second_match)
            f_mrr_val, f_best_all = evidence_mrr(chunks, evidence_list, second_match)
            f_rec3 = evidence_recall_at_k(chunks, evidence_list, 3, second_match)
            f_rec5 = evidence_recall_at_k(chunks, evidence_list, 5, second_match)
            best_info = f_best_all or {}
        else:
            f_hit3_val, f_hit5_val = False, False
            f_mrr_val, f_rec3, f_rec5 = 0.0, 0.0, 0.0
            best_info = e_best_all or {}

        # 累加汇总
        e_hit3 += int(e_hit3_val); e_hit5 += int(e_hit5_val)
        e_mrr_sum += e_mrr_val; e_rec3_sum += e_rec3; e_rec5_sum += e_rec5
        f_hit3 += int(f_hit3_val); f_hit5 += int(f_hit5_val)
        f_mrr_sum += f_mrr_val; f_rec3_sum += f_rec3; f_rec5_sum += f_rec5

        # ── Evidence Unit 指标 ──
        evidence_units = get_evidence_units(entry)
        unit_match_fn = second_match if second_match is not None else exact_match
        eu_metrics_k3 = compute_evidence_unit_metrics(chunks, evidence_units, 3, unit_match_fn)
        eu_metrics_k5 = compute_evidence_unit_metrics(chunks, evidence_units, 5, unit_match_fn)

        eu_total_sum += eu_metrics_k3["evidence_unit_total"]
        eu_rec3_sum += eu_metrics_k3["evidence_unit_recall"]
        eu_rec5_sum += eu_metrics_k5["evidence_unit_recall"]
        eu_hit3_sum += eu_metrics_k3["evidence_unit_hit_count"]
        eu_hit5_sum += eu_metrics_k5["evidence_unit_hit_count"]
        eu_group_hit3_sum += int(eu_metrics_k3["evidence_group_hit"])
        eu_group_hit5_sum += int(eu_metrics_k5["evidence_group_hit"])

        # best match 详细信息
        best_chunk = best_info.get("chunk", {})
        best_scores = best_info.get("scores", {})

        # truncated matched quote
        matched_quote = ""
        if best_info.get("evidence"):
            matched_quote = best_info["evidence"].get("quote", "")[:100]

        # retrieved_docs_top5
        retrieved_docs = []
        for c in chunks[:5]:
            retrieved_docs.append(c.get("source_file", ""))

        results.append({
            "id": entry["id"],
            "query": entry["query"],
            "intent": entry["intent"],
            "difficulty": entry.get("difficulty", ""),
            "eval_focus": ",".join(entry.get("eval_focus", [])),
            "expected_docs": ",".join(entry.get("expected_docs", [])),
            "evidence_count": len(evidence_list),
            "answerable": True,
            # exact
            "exact_hit@3": e_hit3_val,
            "exact_hit@5": e_hit5_val,
            "exact_mrr": round(e_mrr_val, 4),
            "exact_recall@3": round(e_rec3, 4),
            "exact_recall@5": round(e_rec5, 4),
            # 第二匹配器指标
            f"{second_label}_hit@3": f_hit3_val if second_match is not None else "",
            f"{second_label}_hit@5": f_hit5_val if second_match is not None else "",
            f"{second_label}_mrr": round(f_mrr_val, 4) if second_match is not None else "",
            f"{second_label}_recall@3": round(f_rec3, 4) if second_match is not None else "",
            f"{second_label}_recall@5": round(f_rec5, 4) if second_match is not None else "",
            # best match details
            "best_match_rank": best_info.get("rank", -1),
            "best_match_doc": best_chunk.get("source_file", ""),
            "best_match_score_containment": best_scores.get("containment", 0.0),
            "best_match_score_rouge_l": best_scores.get("rouge_l", 0.0),
            "best_match_score_partial_ratio": best_scores.get("partial_ratio", 0.0),
            "matched_evidence_quote": matched_quote,
            "retrieved_docs_top5": " | ".join(retrieved_docs),
            # evidence unit metrics
            "evidence_unit_total": eu_metrics_k5["evidence_unit_total"],
            "evidence_unit_hit_count@3": eu_metrics_k3["evidence_unit_hit_count"],
            "evidence_unit_hit_count@5": eu_metrics_k5["evidence_unit_hit_count"],
            "evidence_unit_recall@3": round(eu_metrics_k3["evidence_unit_recall"], 4),
            "evidence_unit_recall@5": round(eu_metrics_k5["evidence_unit_recall"], 4),
            "evidence_group_hit@3": eu_metrics_k3["evidence_group_hit"],
            "evidence_group_hit@5": eu_metrics_k5["evidence_group_hit"],
        })

        # ── 详细匹配信息（始终收集，每样本一条）──
        ev_details = []
        for ev in evidence_list:
            quote = ev.get("quote", "")
            ev_details.append({
                "quote": quote,
                "normalized": normalize_text(quote),
            })

        # 构建每个 chunk 的匹配详情
        chunk_details = []
        for rank, chunk in enumerate(chunks[:top_k], start=1):
            chunk_text = chunk.get("body_text", chunk.get("text", ""))
            c_norm = normalize_text(chunk_text)
            cd = {
                "rank": rank,
                "source_file": chunk.get("source_file", ""),
                "body_text": chunk_text[:300],
                "body_text_normalized": c_norm[:300],
            }

            # 对每个 evidence 检查 exact match 和 第二匹配器
            exact_matched_indices = []
            second_match_details = {}
            for idx, ev in enumerate(evidence_list):
                quote = ev.get("quote", "")
                if not quote:
                    continue
                e_norm = normalize_text(quote)
                if e_norm and e_norm in c_norm:
                    exact_matched_indices.append(idx)

                if second_match is not None:
                    is_match, scores = second_match(quote, chunk_text)
                    second_match_details[str(idx)] = {
                        "quote": quote[:80],
                        "is_match": is_match,
                        "scores": scores,
                    }

            cd["exact_matched_evidence_indices"] = exact_matched_indices
            if second_match is not None:
                cd[f"{second_label}_match_per_evidence"] = second_match_details
            chunk_details.append(cd)

        verbose_data.append({
            "id": entry["id"],
            "query": entry["query"],
            "intent": entry["intent"],
            "difficulty": entry.get("difficulty", ""),
            "expected_docs": entry.get("expected_docs", []),
            "evidence": ev_details,
            "evidence_units": evidence_units,
            "evidence_unit_total": eu_metrics_k5["evidence_unit_total"],
            "evidence_unit_matched@5": eu_metrics_k5["matched_units"],
            "evidence_unit_missing@5": eu_metrics_k5["missing_units"],
            "evidence_unit_recall@3": round(eu_metrics_k3["evidence_unit_recall"], 4),
            "evidence_unit_recall@5": round(eu_metrics_k5["evidence_unit_recall"], 4),
            "evidence_group_hit@3": eu_metrics_k3["evidence_group_hit"],
            "evidence_group_hit@5": eu_metrics_k5["evidence_group_hit"],
            "evidence_unit_details": eu_metrics_k5["unit_details"],
            "retrieved_chunks": chunk_details,
            "top_k_chunk_ids": [c.get("chunk_id", "") for c in chunks[:5]],
            "top_k_doc_names": [c.get("source_file", "") for c in chunks[:5]],
            "top_k_section_names": [c.get("doc_title", "") for c in chunks[:5]],
            "exact_summary": {
                "hit@3": e_hit3_val,
                "hit@5": e_hit5_val,
                "mrr": round(e_mrr_val, 4),
                "recall@3": round(e_rec3, 4),
                "recall@5": round(e_rec5, 4),
            },
            f"{second_label}_summary": {
                "hit@3": f_hit3_val if second_match is not None else None,
                "hit@5": f_hit5_val if second_match is not None else None,
                "mrr": round(f_mrr_val, 4) if second_match is not None else None,
                "recall@3": round(f_rec3, 4) if second_match is not None else None,
                "recall@5": round(f_rec5, 4) if second_match is not None else None,
            },
        })

    # 汇总
    n = total_valid if total_valid > 0 else 1
    summary = {
        "total_samples": len(entries),
        "total_valid": total_valid,
        "thresholds": {
            "containment": containment_threshold,
            "rouge_l": rouge_l_threshold,
            "partial_ratio": partial_ratio_threshold,
        },
        "match_method": second_label if not exact_only else "exact",
        # exact
        "exact_hit@3": round(e_hit3 / n, 4),
        "exact_hit@5": round(e_hit5 / n, 4),
        "exact_mrr": round(e_mrr_sum / n, 4),
        "exact_recall@3": round(e_rec3_sum / n, 4),
        "exact_recall@5": round(e_rec5_sum / n, 4),
        # 第二匹配器指标（动态字段名）
        f"{second_label}_hit@3": round(f_hit3 / n, 4) if second_match is not None else None,
        f"{second_label}_hit@5": round(f_hit5 / n, 4) if second_match is not None else None,
        f"{second_label}_mrr": round(f_mrr_sum / n, 4) if second_match is not None else None,
        f"{second_label}_recall@3": round(f_rec3_sum / n, 4) if second_match is not None else None,
        f"{second_label}_recall@5": round(f_rec5_sum / n, 4) if second_match is not None else None,
        # evidence unit 指标
        "evidence_unit_total_avg": round(eu_total_sum / n, 2) if n else 0,
        "evidence_unit_recall@3": round(eu_rec3_sum / n, 4) if n else 0,
        "evidence_unit_recall@5": round(eu_rec5_sum / n, 4) if n else 0,
        "evidence_unit_hit_count_avg@3": round(eu_hit3_sum / n, 2) if n else 0,
        "evidence_unit_hit_count_avg@5": round(eu_hit5_sum / n, 2) if n else 0,
        "evidence_group_hit@3": round(eu_group_hit3_sum / n, 4) if n else 0,
        "evidence_group_hit@5": round(eu_group_hit5_sum / n, 4) if n else 0,
    }

    result = {"summary": summary, "details": results, "verbose_data": verbose_data}
    return result


# ═══════════════════════════════════════════════════════════════
# Evidence 结果输出
# ═══════════════════════════════════════════════════════════════

def _group_metrics(details: list[dict], group_key: str, second_label: str = "fuzzy") -> dict:
    """按指定维度对 evidence results 分组统计。"""
    groups = defaultdict(list)
    for d in details:
        if not d.get("answerable") or d.get("skipped"):
            continue
        key_val = d.get(group_key, "unknown")
        if group_key == "eval_focus":
            key_val = key_val.split(",")[0] if key_val else "unknown"
        groups[key_val].append(d)

    result = {}
    for key, items in groups.items():
        n = len(items)
        entry = {
            "count": n,
            "exact_hit@5": round(sum(d["exact_hit@5"] for d in items) / n, 4),
            "exact_mrr": round(sum(d["exact_mrr"] for d in items) / n, 4),
            "exact_recall@5": round(sum(d["exact_recall@5"] for d in items) / n, 4),
        }
        # 第二匹配器字段（仅在非 exact_only 模式下有效）
        h5_key = f"{second_label}_hit@5"
        mrr_key = f"{second_label}_mrr"
        rec5_key = f"{second_label}_recall@5"
        if isinstance(items[0].get(h5_key), (int, float, bool)):
            entry[h5_key] = round(sum(d[h5_key] for d in items) / n, 4)
            entry[mrr_key] = round(sum(d[mrr_key] for d in items) / n, 4)
            entry[rec5_key] = round(sum(d[rec5_key] for d in items) / n, 4)
        else:
            entry[h5_key] = None
            entry[mrr_key] = None
            entry[rec5_key] = None
        result[key] = entry
    return result


def save_evidence_results(
    result: dict, output_dir: str | Path, mode: str = "hybrid",
) -> tuple[Path, Path]:
    """保存 evidence-level 评测结果。

    生成两个文件：
    - evidence_retrieval_results.csv  （每条样本一行）
    - evidence_retrieval_summary.md   （中文汇总报告）

    Args:
        result: 评测结果字典
        output_dir: 输出目录
        mode: 检索模式（用于命名和报告标题）

    Returns:
        (csv_path, summary_path)
    """
    import csv

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = result["summary"]
    details = result["details"]
    match_method = summary.get("match_method", "fuzzy")
    second_label = match_method if match_method != "exact" else "fuzzy"

    # 中文标签映射
    LABEL_CN = {"fuzzy": "模糊匹配", "rouge_l": "ROUGE-L", "containment": "包含度", "partial_ratio": "部分匹配率"}

    suffix = match_method if match_method != "fuzzy" else "full"
    if match_method == "exact":
        suffix = "exact"

    # ── CSV ──
    csv_path = output_dir / f"证据检索结果_{mode}_{suffix}.csv"
    csv_fields = [
        "id", "query", "intent", "difficulty", "eval_focus", "expected_docs",
        "evidence_count", "exact_hit@3", "exact_hit@5",
        "exact_mrr", "exact_recall@3", "exact_recall@5",
    ]
    if match_method != "exact":
        csv_fields.extend([
            f"{second_label}_hit@3", f"{second_label}_hit@5", f"{second_label}_mrr",
            f"{second_label}_recall@3", f"{second_label}_recall@5",
        ])
    csv_fields.extend([
        "best_match_rank", "best_match_doc",
        "best_match_score_containment", "best_match_score_rouge_l",
        "best_match_score_partial_ratio", "matched_evidence_quote",
        "retrieved_docs_top5",
        # evidence unit fields
        "evidence_unit_total",
        "evidence_unit_hit_count@3", "evidence_unit_hit_count@5",
        "evidence_unit_recall@3", "evidence_unit_recall@5",
        "evidence_group_hit@3", "evidence_group_hit@5",
    ])
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        for d in details:
            writer.writerow(d)

    # ── 中文汇总报告 (MD) ──
    md_path = output_dir / f"证据检索总结_{mode}_{suffix}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# 证据级检索评测总结（检索模式: {mode}）\n\n")
        f.write(f"**匹配方法:** {LABEL_CN.get(match_method, match_method)}\n\n")
        f.write(f"**匹配阈值:** 3-gram 包含度={summary['thresholds']['containment']}, "
                f"ROUGE-L={summary['thresholds']['rouge_l']}, "
                f"部分匹配率={summary['thresholds']['partial_ratio']}\n\n")

        if match_method == "exact":
            f.write("## 总体指标（仅精确匹配）\n\n")
            f.write("| 指标 | 精确匹配 |\n")
            f.write("|------|----------|\n")
            f.write(f"| Hit@3 | {summary['exact_hit@3']:.2%} |\n")
            f.write(f"| Hit@5 | {summary['exact_hit@5']:.2%} |\n")
            f.write(f"| MRR | {summary['exact_mrr']:.4f} |\n")
            f.write(f"| Recall@3 | {summary['exact_recall@3']:.2%} |\n")
            f.write(f"| Recall@5 | {summary['exact_recall@5']:.2%} |\n")
        else:
            f.write("## 总体指标\n\n")
            cn = LABEL_CN.get(match_method, match_method)
            f.write(f"| 指标 | 精确匹配 | {cn} |\n")
            f.write("|------|----------|----------|\n")
            f.write(f"| Hit@3 | {summary['exact_hit@3']:.2%} | {summary[f'{second_label}_hit@3']:.2%} |\n")
            f.write(f"| Hit@5 | {summary['exact_hit@5']:.2%} | {summary[f'{second_label}_hit@5']:.2%} |\n")
            f.write(f"| MRR | {summary['exact_mrr']:.4f} | {summary[f'{second_label}_mrr']:.4f} |\n")
            f.write(f"| Recall@3 | {summary['exact_recall@3']:.2%} | {summary[f'{second_label}_recall@3']:.2%} |\n")
            f.write(f"| Recall@5 | {summary['exact_recall@5']:.2%} | {summary[f'{second_label}_recall@5']:.2%} |\n")

        # Evidence Unit 指标
        f.write("\n## Evidence Unit 指标\n\n")
        f.write(f"| 指标 | 值 |\n")
        f.write("|------|----|\n")
        f.write(f"| 平均 Unit 数 | {summary['evidence_unit_total_avg']:.1f} |\n")
        f.write(f"| Unit Recall@3 | {summary['evidence_unit_recall@3']:.2%} |\n")
        f.write(f"| Unit Recall@5 | {summary['evidence_unit_recall@5']:.2%} |\n")
        f.write(f"| 平均命中 Unit@3 | {summary['evidence_unit_hit_count_avg@3']:.1f} |\n")
        f.write(f"| 平均命中 Unit@5 | {summary['evidence_unit_hit_count_avg@5']:.1f} |\n")
        f.write(f"| Group Hit@3 | {summary['evidence_group_hit@3']:.2%} |\n")
        f.write(f"| Group Hit@5 | {summary['evidence_group_hit@5']:.2%} |\n")

        # 按意图分组
        group_data = _group_metrics(details, "intent", second_label)
        cn = LABEL_CN.get(match_method, match_method)
        f.write("\n## 按意图分组\n\n")
        if match_method == "exact":
            f.write("| 意图 | 样本数 | 精确 Hit@5 | 精确 MRR | 精确 Recall@5 |\n")
            f.write("|------|--------|------------|----------|---------------|\n")
            for intent, m in group_data.items():
                f.write(f"| {intent} | {m['count']} | {m['exact_hit@5']:.2%} | "
                        f"{m['exact_mrr']:.4f} | {m['exact_recall@5']:.2%} |\n")
        else:
            f.write(f"| 意图 | 样本数 | 精确 Hit@5 | {cn} Hit@5 | 精确 MRR | {cn} MRR | {cn} Recall@5 |\n")
            f.write("|------|--------|------------|------------|----------|----------|---------------|\n")
            for intent, m in group_data.items():
                f.write(f"| {intent} | {m['count']} | {m['exact_hit@5']:.2%} | {m[f'{second_label}_hit@5']:.2%} | "
                        f"{m['exact_mrr']:.4f} | {m[f'{second_label}_mrr']:.4f} | {m[f'{second_label}_recall@5']:.2%} |\n")

        # 按难度分组
        group_data = _group_metrics(details, "difficulty", second_label)
        f.write("\n## 按难度分组\n\n")
        if match_method == "exact":
            f.write("| 难度 | 样本数 | 精确 Hit@5 | 精确 MRR |\n")
            f.write("|------|--------|------------|----------|\n")
            for diff, m in sorted(group_data.items()):
                f.write(f"| {diff} | {m['count']} | {m['exact_hit@5']:.2%} | "
                        f"{m['exact_mrr']:.4f} |\n")
        else:
            f.write(f"| 难度 | 样本数 | 精确 Hit@5 | {cn} Hit@5 | 精确 MRR | {cn} MRR |\n")
            f.write("|------|--------|------------|------------|----------|----------|\n")
            for diff, m in sorted(group_data.items()):
                f.write(f"| {diff} | {m['count']} | {m['exact_hit@5']:.2%} | {m[f'{second_label}_hit@5']:.2%} | "
                        f"{m['exact_mrr']:.4f} | {m[f'{second_label}_mrr']:.4f} |\n")

        # 按评测关注点分组
        group_data = _group_metrics(details, "eval_focus", second_label)
        f.write("\n## 按评测关注点分组\n\n")
        if match_method == "exact":
            f.write("| 关注点 | 样本数 | 精确 Hit@5 | 精确 MRR |\n")
            f.write("|--------|--------|------------|----------|\n")
            for focus, m in sorted(group_data.items()):
                f.write(f"| {focus} | {m['count']} | {m['exact_hit@5']:.2%} | "
                        f"{m['exact_mrr']:.4f} |\n")
        else:
            f.write(f"| 关注点 | 样本数 | 精确 Hit@5 | {cn} Hit@5 | 精确 MRR | {cn} MRR |\n")
            f.write("|--------|--------|------------|------------|----------|----------|\n")
            for focus, m in sorted(group_data.items()):
                f.write(f"| {focus} | {m['count']} | {m['exact_hit@5']:.2%} | {m[f'{second_label}_hit@5']:.2%} | "
                        f"{m['exact_mrr']:.4f} | {m[f'{second_label}_mrr']:.4f} |\n")

    return csv_path, md_path


def print_evidence_report(result: dict, mode: str = "hybrid") -> None:
    """在终端打印 evidence-level 评测报告（中文）。"""
    summary = result["summary"]
    details = result["details"]
    match_method = summary.get("match_method", "fuzzy")
    second_label = match_method if match_method != "exact" else "fuzzy"

    LABEL_CN = {"fuzzy": "模糊匹配", "rouge_l": "ROUGE-L", "containment": "包含度", "partial_ratio": "部分匹配率"}
    cn = LABEL_CN.get(match_method, match_method)

    print("\n" + "=" * 60)
    print(f"  证据级检索评测结果（检索模式: {mode}，匹配方法: {cn}）")
    print("=" * 60)
    print(f"  有效样本数: {summary['total_valid']}/{summary['total_samples']}")
    print(f"  阈值: 3-gram包含度={summary['thresholds']['containment']}, "
          f"ROUGE-L={summary['thresholds']['rouge_l']}, "
          f"部分匹配率={summary['thresholds']['partial_ratio']}")
    print()
    if match_method == "exact":
        print(f"  {'指标':<18} {'精确匹配':<10}")
        print(f"  {'-'*18} {'-'*10}")
        print(f"  {'Hit@3':<18} {summary['exact_hit@3']:<10.2%}")
        print(f"  {'Hit@5':<18} {summary['exact_hit@5']:<10.2%}")
        print(f"  {'MRR':<18} {summary['exact_mrr']:<10.4f}")
        print(f"  {'Recall@3':<18} {summary['exact_recall@3']:<10.2%}")
        print(f"  {'Recall@5':<18} {summary['exact_recall@5']:<10.2%}")
    else:
        print(f"  {'指标':<18} {'精确匹配':<10} {cn:<10}")
        print(f"  {'-'*18} {'-'*10} {'-'*10}")
        print(f"  {'Hit@3':<18} {summary['exact_hit@3']:<10.2%} {summary[f'{second_label}_hit@3']:<10.2%}")
        print(f"  {'Hit@5':<18} {summary['exact_hit@5']:<10.2%} {summary[f'{second_label}_hit@5']:<10.2%}")
        print(f"  {'MRR':<18} {summary['exact_mrr']:<10.4f} {summary[f'{second_label}_mrr']:<10.4f}")
        print(f"  {'Recall@3':<18} {summary['exact_recall@3']:<10.2%} {summary[f'{second_label}_recall@3']:<10.2%}")
        print(f"  {'Recall@5':<18} {summary['exact_recall@5']:<10.2%} {summary[f'{second_label}_recall@5']:<10.2%}")

    # Evidence Unit 指标
    print(f"\n  ── Evidence Unit 指标 ──")
    print(f"  {'平均Unit数':<16} {summary['evidence_unit_total_avg']:<10.1f}")
    print(f"  {'Unit Recall@3':<16} {summary['evidence_unit_recall@3']:<10.2%}")
    print(f"  {'Unit Recall@5':<16} {summary['evidence_unit_recall@5']:<10.2%}")
    print(f"  {'Group Hit@3':<16} {summary['evidence_group_hit@3']:<10.2%}")
    print(f"  {'Group Hit@5':<16} {summary['evidence_group_hit@5']:<10.2%}")

    # 按意图
    group_data = _group_metrics(details, "intent", second_label)
    if match_method == "exact":
        print("\n  ── 按意图 ──")
        print(f"  {'意图':<16} {'样本数':<8} {'精确H@5':<10} {'精确MRR':<10}")
        for intent, m in group_data.items():
            print(f"  {intent:<16} {m['count']:<8} {m['exact_hit@5']:<10.2%} {m['exact_mrr']:<10.4f}")
    else:
        print("\n  ── 按意图 ──")
        print(f"  {'意图':<16} {'样本数':<8} {'精确H@5':<10} {cn+'H@5':<10} {cn+'MRR':<10}")
        for intent, m in group_data.items():
            print(f"  {intent:<16} {m['count']:<8} {m['exact_hit@5']:<10.2%} {m[f'{second_label}_hit@5']:<10.2%} {m[f'{second_label}_mrr']:<10.4f}")


def print_per_query_details(verbose_data: list[dict], second_label: str) -> None:
    """逐条打印每个 query 的 expected vs retrieved 证据匹配情况。"""
    print("\n" + "=" * 60)
    print("  逐条证据匹配详情")
    print("=" * 60)

    for i, v in enumerate(verbose_data, start=1):
        exact_hit = "✓" if v["exact_summary"]["hit@5"] else "✗"
        second_key = f"{second_label}_summary"
        second_hit = "✓" if v.get(second_key, {}).get("hit@5") else "✗"

        eu_total = v.get("evidence_unit_total", 0)
        eu_recall = v.get("evidence_unit_recall@5", 0)
        eu_group = "✓" if v.get("evidence_group_hit@5") else "✗"

        print(f"\n── [{i:02d}] {v['id']} | {v['intent']} | "
              f"exact={exact_hit} {second_label}={second_hit} | "
              f"EU_recall@5={eu_recall:.0%} EU_group={eu_group} ──")
        print(f"  查询: {v['query'][:80]}")
        if v.get("expected_docs"):
            print(f"  期望文档: {', '.join(v['expected_docs'])}")

        # Evidence 列表
        print(f"  标准 Evidence ({len(v['evidence'])}条) | Evidence Units: {eu_total}")
        for j, ev in enumerate(v["evidence"]):
            print(f"    [{j}] {ev['quote'][:100]}")

        # Evidence Unit 匹配概况
        if eu_total > 0:
            matched = v.get("evidence_unit_matched@5", [])
            missing = v.get("evidence_unit_missing@5", [])
            print(f"  Evidence Unit 匹配: {len(matched)}/{eu_total} 命中, {len(missing)} 未命中")
            if missing:
                print(f"  未命中 Units (前3):")
                for mu in missing[:3]:
                    print(f"    ✗ {mu[:120]}")

        # Chunk 匹配情况
        print(f"  检索结果 (Top{len(v['retrieved_chunks'])}):")
        for c in v["retrieved_chunks"]:
            matched = c.get("exact_matched_evidence_indices", [])
            status = f"✓ ev[{','.join(map(str, matched))}]" if matched else "-"
            print(f"    rank={c['rank']} {c['source_file']:<30} {status}")


# ═══════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="OnboardRAG — 离线评测")
    parser.add_argument(
        "--eval-file",
        type=str,
        default="data/eval/eval_queries.jsonl",
        help="评测集 JSONL 文件路径（默认: data/eval/eval_queries.jsonl）",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["doc", "evidence", "all"],
        default="all",
        help="评测模式: doc（仅 doc-level）、evidence（仅 evidence-level）、all（全部，默认）",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=RETRIEVAL_TOP_K,
        help=f"检索返回的 chunk 数量（默认: {RETRIEVAL_TOP_K}）",
    )
    parser.add_argument(
        "--retrieval-mode",
        type=str,
        choices=["hybrid", "dense", "bm25"],
        default="hybrid",
        help="检索模式: hybrid（混合检索，默认）、dense（仅向量）、bm25（仅关键词）",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/eval",
        help="evidence-level 结果输出目录（默认: outputs/eval/）",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="将完整 JSON 结果保存到指定文件",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="以 JSON 格式输出结果到 stdout",
    )
    parser.add_argument(
        "--containment-threshold",
        type=float,
        default=DEFAULT_CONTAINMENT_THRESHOLD,
        help=f"char 3-gram containment 阈值（默认: {DEFAULT_CONTAINMENT_THRESHOLD}）",
    )
    parser.add_argument(
        "--rouge-l-threshold",
        type=float,
        default=DEFAULT_ROUGE_L_THRESHOLD,
        help=f"ROUGE-L recall 阈值（默认: {DEFAULT_ROUGE_L_THRESHOLD}）",
    )
    parser.add_argument(
        "--partial-ratio-threshold",
        type=float,
        default=DEFAULT_PARTIAL_RATIO_THRESHOLD,
        help=f"partial_ratio 阈值（默认: {DEFAULT_PARTIAL_RATIO_THRESHOLD}）",
    )
    parser.add_argument(
        "--exact-only",
        action="store_true",
        help="仅计算 exact match 指标，跳过第二匹配器",
    )
    parser.add_argument(
        "--match-method",
        type=str,
        choices=["fuzzy", "rouge_l", "containment", "partial_ratio"],
        default="fuzzy",
        help="evidence 第二匹配器方法（默认: fuzzy）。fuzzy=三合一, rouge_l=纯ROUGE-L, "
             "containment=纯3-gram包含度, partial_ratio=纯部分匹配率",
    )
    args = parser.parse_args()

    eval_path = Path(args.eval_file)
    if not eval_path.exists():
        print(f"评测文件不存在: {eval_path}")
        return 1

    # 加载评测集
    with open(eval_path, "r", encoding="utf-8") as f:
        entries = [json.loads(line) for line in f if line.strip()]

    print(f"加载了 {len(entries)} 条评测样本")
    print(f"评测模式: {args.mode}")
    print(f"检索模式: {args.retrieval_mode}")

    full_result: dict = {}

    # ── Doc-level 检索评测 ──
    if args.mode in ("doc", "all"):
        print("\n── Doc-Level 检索评测 ──")
        retrieval_result = eval_retrieval(entries, top_k=args.top_k, mode=args.retrieval_mode)
        print_retrieval_report(retrieval_result, mode=args.retrieval_mode)
        full_result["doc_retrieval"] = retrieval_result

    # ── Evidence-level 检索评测 ──
    if args.mode in ("evidence", "all"):
        print("\n── Evidence-Level 检索评测 ──")
        if args.exact_only:
            print("  (仅精确匹配模式)")
        evidence_result = eval_evidence_retrieval(
            entries,
            top_k=args.top_k,
            containment_threshold=args.containment_threshold,
            rouge_l_threshold=args.rouge_l_threshold,
            partial_ratio_threshold=args.partial_ratio_threshold,
            mode=args.retrieval_mode,
            exact_only=args.exact_only,
            match_method=args.match_method,
        )
        print_evidence_report(evidence_result, mode=args.retrieval_mode)

        # 逐条打印匹配详情
        second_label = evidence_result["summary"].get("match_method", "fuzzy")
        if second_label == "exact":
            second_label = "fuzzy"
        print_per_query_details(evidence_result["verbose_data"], second_label)

        full_result["evidence_retrieval"] = evidence_result

        # 保存 CSV + summary MD
        csv_path, md_path = save_evidence_results(
            evidence_result, args.output_dir, mode=args.retrieval_mode,
        )
        print(f"\n  证据详细结果 (CSV): {csv_path}")
        print(f"  证据汇总报告 (MD):  {md_path}")

        # 仅保存 exact 未命中的详匹配情到 JSON，避免干扰
        failed = [v for v in evidence_result["verbose_data"] if not v["exact_summary"]["hit@5"]]
        if failed:
            detail_json_path = Path(args.output_dir) / "证据匹配详情_未命中.json"
            detail_json_path.parent.mkdir(parents=True, exist_ok=True)
            with open(detail_json_path, "w", encoding="utf-8") as f:
                json.dump(failed, f, ensure_ascii=False, indent=2)
            print(f"  未命中详情 (JSON, {len(failed)}条): {detail_json_path}")

    # ── JSON 输出 ──
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(full_result, f, ensure_ascii=False, indent=2)
        print(f"\n完整结果已保存到: {output_path}")

    if args.json:
        print(json.dumps(full_result, ensure_ascii=False, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
