#!/usr/bin/env python3
"""
evaluate_from_retrieval_files.py — 阶段二：从检索结果文件计算指标

只读取阶段一生成的 JSONL 文件，不再调用检索器和 reranker。
计算所有 evidence-level 指标、candidate-stage 指标、overlap 分析和 error analysis。

用法:
    # 从目录加载所有 JSONL 文件
    python scripts/evaluate_from_retrieval_files.py \\
        --results-dir outputs/retrieval_results/

    # 从单个文件加载
    python scripts/evaluate_from_retrieval_files.py \\
        --results-file outputs/retrieval_results/retrieval_results_full_20260609.jsonl

    # 自定义匹配方法和阈值
    python scripts/evaluate_from_retrieval_files.py \\
        --results-dir outputs/retrieval_results/ \\
        --match-method fuzzy \\
        --containment-threshold 0.8 \\
        --output-dir outputs/eval/

    # 仅精确匹配
    python scripts/evaluate_from_retrieval_files.py \\
        --results-dir outputs/retrieval_results/ \\
        --exact-only
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.eval_metrics import (
    compute_evidence_unit_metrics,
    evidence_unit_mrr,
    get_evidence_units,
    make_fuzzy_matcher,
    make_rouge_l_matcher,
    make_containment_matcher,
    make_partial_ratio_matcher,
    DEFAULT_CONTAINMENT_THRESHOLD,
    DEFAULT_ROUGE_L_THRESHOLD,
    DEFAULT_PARTIAL_RATIO_THRESHOLD,
)


# ═══════════════════════════════════════════════════════════════
# 数据加载
# ═══════════════════════════════════════════════════════════════

def load_retrieval_results(paths: list[Path]) -> list[dict]:
    """加载一个或多个 JSONL 检索结果文件。

    Returns:
        所有行的列表（每行是一个 query+method 组合）
    """
    rows = []
    for path in paths:
        if not path.exists():
            print(f"  警告: 文件不存在，跳过: {path}")
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def load_eval_entries(evalset_path: Path) -> list[dict]:
    """加载评测集 JSONL 文件。"""
    if not evalset_path.exists():
        raise FileNotFoundError(f"评测文件不存在: {evalset_path}")
    with open(evalset_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def group_results_by_query(rows: list[dict]) -> dict[str, dict[str, dict]]:
    """将检索结果按 query_id → method 分组。

    Returns:
        {query_id: {method: row, ...}, ...}
    """
    grouped: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        qid = row["query_id"]
        method = row["method"]
        grouped[qid][method] = row
    return dict(grouped)


# ═══════════════════════════════════════════════════════════════
# 候选结果提取
# ═══════════════════════════════════════════════════════════════

def get_chunks_for_method(
    query_data: dict[str, dict],
    method: str,
    k: int | None = None,
) -> list[dict]:
    """从 query_data 中提取指定 method 的 top-k chunks。

    Args:
        query_data: {method: row}
        method: 方法名
        k: 截断值，None 表示取全部

    Returns:
        chunk 列表，每个包含 text/body_text 等字段
    """
    row = query_data.get(method)
    if row is None:
        return []
    results = row.get("results", [])
    if k is not None:
        results = results[:k]

    # 转换为 eval_metrics 期望的格式（带 body_text）
    chunks = []
    for r in results:
        chunks.append({
            "chunk_id": r.get("chunk_id", ""),
            "text": r.get("text", ""),
            "body_text": r.get("text", ""),  # eval_metrics 优先用 body_text
            "source_file": r.get("document_name", ""),
            "doc_title": r.get("section", ""),
            "score": r.get("score", 0),
            "rank": r.get("rank", 0),
        })
    return chunks


def _chunk_ids_for_method(
    query_data: dict[str, dict],
    method: str,
    k: int | None = None,
) -> set[str]:
    """获取指定 method 的 top-k chunk_id 集合。"""
    chunks = get_chunks_for_method(query_data, method, k)
    return {c["chunk_id"] for c in chunks}


# ═══════════════════════════════════════════════════════════════
# 指标计算
# ═══════════════════════════════════════════════════════════════

def compute_query_level_metrics(
    query_data: dict[str, dict],
    gold_entry: dict,
    match_fn,
    methods: list[str],
    k_values: list[int],
) -> dict:
    """计算 query-level 指标（Hit@K, MRR）。"""
    evidence_units = get_evidence_units(gold_entry)
    result = {}

    for method in methods:
        chunks = get_chunks_for_method(query_data, method)
        if not chunks:
            result[method] = {
                "hit": {f"hit@{k}": False for k in k_values},
                "mrr": 0.0,
            }
            continue

        # Evidence unit MRR
        mrr_val = evidence_unit_mrr(chunks, evidence_units, match_fn)

        # Hit@K (group_full_hit)
        hit = {}
        for k in k_values:
            eu_metrics = compute_evidence_unit_metrics(chunks, evidence_units, k, match_fn)
            hit[f"hit@{k}"] = eu_metrics["evidence_group_full_hit"]

        result[method] = {
            "hit": hit,
            "mrr": round(mrr_val, 4),
        }

    return result


def compute_evidence_unit_metrics_all(
    query_data: dict[str, dict],
    gold_entry: dict,
    match_fn,
    methods: list[str],
    k_values: list[int],
) -> dict:
    """计算 evidence-unit 级别指标。"""
    evidence_units = get_evidence_units(gold_entry)
    result = {}

    for method in methods:
        chunks = get_chunks_for_method(query_data, method)
        method_result = {}
        for k in k_values:
            eu = compute_evidence_unit_metrics(chunks, evidence_units, k, match_fn)
            method_result[f"recall@{k}"] = eu["evidence_unit_recall"]
            method_result[f"group_hit@{k}"] = eu["evidence_group_full_hit"]
            method_result[f"group_any_hit@{k}"] = eu["evidence_group_any_hit"]
            method_result[f"hit_count@{k}"] = eu["evidence_unit_hit_count"]
            method_result[f"total"] = eu["evidence_unit_total"]
            method_result[f"matched_units@{k}"] = eu["matched_units"]
            method_result[f"missing_units@{k}"] = eu["missing_units"]
        result[method] = method_result

    return result


def compute_candidate_metrics(
    query_data: dict[str, dict],
    gold_entry: dict,
    match_fn,
) -> dict:
    """计算 candidate-stage 指标。

    需要 union_candidate 结果来计算 candidate_recall@30。
    """
    evidence_units = get_evidence_units(gold_entry)

    # Union candidate pool (all chunks)
    union_chunks = get_chunks_for_method(query_data, "union_candidate")
    bm25_chunks = get_chunks_for_method(query_data, "bm25")
    dense_chunks = get_chunks_for_method(query_data, "dense")

    # RRF top-10, Rerank top-10
    rrf_10 = get_chunks_for_method(query_data, "rrf_hybrid", k=10)
    rerank_10 = get_chunks_for_method(query_data, "hybrid_rerank", k=10)

    result = {
        "bm25_candidate_recall@30": _recall_from_chunks(bm25_chunks, evidence_units, match_fn, k=30),
        "dense_candidate_recall@30": _recall_from_chunks(dense_chunks, evidence_units, match_fn, k=30),
        "union_candidate_recall@30": _recall_from_chunks(union_chunks, evidence_units, match_fn, k=30),
        "rrf_recall@10": _recall_from_chunks(rrf_10, evidence_units, match_fn, k=10),
        "rerank_recall@10": _recall_from_chunks(rerank_10, evidence_units, match_fn, k=10),
    }
    return result


def _recall_from_chunks(
    chunks: list[dict],
    evidence_units: list[str],
    match_fn,
    k: int,
) -> float:
    """计算 evidence unit recall（基于 chunk 列表，不依赖 method）。"""
    if not evidence_units:
        return 0.0
    metrics = compute_evidence_unit_metrics(chunks, evidence_units, k, match_fn)
    return metrics["evidence_unit_recall"]


def _group_full_hit(
    chunks: list[dict],
    evidence_units: list[str],
    match_fn,
    k: int,
) -> bool:
    """检查是否所有 evidence unit 都被覆盖。"""
    if not evidence_units:
        return False
    metrics = compute_evidence_unit_metrics(chunks, evidence_units, k, match_fn)
    return metrics["evidence_group_full_hit"]


# ═══════════════════════════════════════════════════════════════
# Overlap 分析
# ═══════════════════════════════════════════════════════════════

def compute_overlap_analysis(
    query_data: dict[str, dict],
    gold_entry: dict,
    match_fn,
    k: int = 10,
) -> dict:
    """计算 overlap 分析：BM25 vs Dense vs Hybrid 命中对比。

    Returns:
        包含分类信息的 dict
    """
    evidence_units = get_evidence_units(gold_entry)

    bm_chunks = get_chunks_for_method(query_data, "bm25", k=k)
    dn_chunks = get_chunks_for_method(query_data, "dense", k=k)
    rrf_chunks = get_chunks_for_method(query_data, "rrf_hybrid", k=k)
    rerank_chunks = get_chunks_for_method(query_data, "hybrid_rerank", k=k)

    bm_hit = _group_full_hit(bm_chunks, evidence_units, match_fn, k)
    dn_hit = _group_full_hit(dn_chunks, evidence_units, match_fn, k)
    rrf_hit = _group_full_hit(rrf_chunks, evidence_units, match_fn, k)

    has_rerank = bool(rerank_chunks)
    rerank_hit = _group_full_hit(rerank_chunks, evidence_units, match_fn, k) if has_rerank else None

    # 基础分类
    if bm_hit and dn_hit:
        category = "both_hit"
    elif bm_hit and not dn_hit:
        category = "bm25_only"
    elif dn_hit and not bm_hit:
        category = "dense_only"
    else:
        category = "both_miss"

    # hybrid gain/loss (vs RRF)
    hybrid_gain = rrf_hit and not bm_hit and not dn_hit
    hybrid_loss = not rrf_hit and (bm_hit or dn_hit)

    # reranker gain/loss (vs RRF) — only meaningful when reranker was run
    if has_rerank:
        rerank_gain = bool(rerank_hit) and not rrf_hit
        rerank_loss = not bool(rerank_hit) and rrf_hit
    else:
        rerank_gain = None
        rerank_loss = None

    return {
        "category": category,
        "bm25_hit": bm_hit,
        "dense_hit": dn_hit,
        "rrf_hit": rrf_hit,
        "rerank_hit": rerank_hit,
        "hybrid_gain": hybrid_gain,
        "hybrid_loss": hybrid_loss,
        "rerank_gain": rerank_gain,
        "rerank_loss": rerank_loss,
    }


# ═══════════════════════════════════════════════════════════════
# Error 分析
# ═══════════════════════════════════════════════════════════════

def compute_error_analysis(
    query_data: dict[str, dict],
    gold_entry: dict,
    match_fn,
) -> dict | None:
    """对单个 answerable query 进行 error analysis。

    记录所有与 reranker 相关的成功/失败模式（4 种），
    仅在 reranker 和 RRF 都命中时跳过（一切正常）。
    """
    evidence_units = get_evidence_units(gold_entry)
    if not evidence_units:
        return None

    # 各方法的 top-10 group_full_hit
    bm_hit10 = _group_full_hit(
        get_chunks_for_method(query_data, "bm25", k=10), evidence_units, match_fn, 10
    )
    dn_hit10 = _group_full_hit(
        get_chunks_for_method(query_data, "dense", k=10), evidence_units, match_fn, 10
    )
    rrf_hit10 = _group_full_hit(
        get_chunks_for_method(query_data, "rrf_hybrid", k=10), evidence_units, match_fn, 10
    )

    rerank_chunks = get_chunks_for_method(query_data, "hybrid_rerank", k=10)
    has_rerank = bool(rerank_chunks)
    rerank_hit10 = _group_full_hit(rerank_chunks, evidence_units, match_fn, 10) if has_rerank else False

    union_hit30 = _group_full_hit(
        get_chunks_for_method(query_data, "union_candidate", k=30), evidence_units, match_fn, 30
    )

    # 如果没有 reranker 结果，不生成 error analysis
    if not has_rerank:
        return None

    # 一切正常：RRF 和 rerank 都命中，不需要记录
    if rrf_hit10 and rerank_hit10:
        return None

    # 判断错误/优化类型（4 种模式全覆盖）
    if not union_hit30:
        error_type = "union_miss"
        error_desc = "union_candidate@30 未命中：一阶段召回失败"
    elif rrf_hit10 and not rerank_hit10:
        error_type = "rerank_negative"
        error_desc = "RRF 命中但 rerank 未命中：reranker 负优化"
    elif not rrf_hit10 and rerank_hit10:
        error_type = "rerank_positive"
        error_desc = "RRF 未命中但 rerank 命中：reranker 正优化"
    else:
        # !rrf_hit10 && !rerank_hit10 但 union_hit30=True
        error_type = "rerank_fail"
        error_desc = "union_candidate@30 命中但 rerank top10 未命中：reranker 排序失败"

    return {
        "query_id": gold_entry["id"],
        "query": gold_entry["query"],
        "intent": gold_entry.get("intent", ""),
        "difficulty": gold_entry.get("difficulty", ""),
        "gold_evidence_units": evidence_units,
        "bm25_hit@10": bm_hit10,
        "dense_hit@10": dn_hit10,
        "union_candidate_hit@30": union_hit30,
        "rrf_hit@10": rrf_hit10,
        "rerank_hit@10": rerank_hit10,
        "error_type": error_type,
        "error_description": error_desc,
    }


# ═══════════════════════════════════════════════════════════════
# 汇总统计
# ═══════════════════════════════════════════════════════════════

def aggregate_metrics(
    per_query_metrics: list[dict],
    methods: list[str],
    k_values: list[int],
) -> dict:
    """对 per-query 指标求平均，返回汇总 dict。"""
    n = len(per_query_metrics) if per_query_metrics else 1

    summary = {}

    # Query-level: Hit@K, MRR
    for method in methods:
        hit_sums = {f"hit@{k}": 0 for k in k_values}
        mrr_sum = 0.0
        for pq in per_query_metrics:
            ql = pq.get("query_level", {}).get(method, {})
            for k in k_values:
                hit_sums[f"hit@{k}"] += int(ql.get("hit", {}).get(f"hit@{k}", False))
            mrr_sum += ql.get("mrr", 0.0)

        summary[f"{method}_hit"] = {
            f"hit@{k}": round(hit_sums[f"hit@{k}"] / n, 4) for k in k_values
        }
        summary[f"{method}_mrr"] = round(mrr_sum / n, 4)

    # Evidence-unit: Recall@K, Group Hit@K
    for method in methods:
        for k in k_values:
            rec_sum = 0.0
            gh_sum = 0.0
            ga_sum = 0.0
            for pq in per_query_metrics:
                eu = pq.get("evidence_unit", {}).get(method, {})
                rec_sum += eu.get(f"recall@{k}", 0.0)
                gh_sum += int(eu.get(f"group_hit@{k}", False))
                ga_sum += int(eu.get(f"group_any_hit@{k}", False))
            summary[f"{method}_eu_recall@{k}"] = round(rec_sum / n, 4)
            summary[f"{method}_eu_group_hit@{k}"] = round(gh_sum / n, 4)
            summary[f"{method}_eu_group_any_hit@{k}"] = round(ga_sum / n, 4)

    # Candidate-stage
    cand_keys = [
        "bm25_candidate_recall@30", "dense_candidate_recall@30",
        "union_candidate_recall@30", "rrf_recall@10", "rerank_recall@10",
    ]
    for key in cand_keys:
        total = sum(pq.get("candidate", {}).get(key, 0.0) for pq in per_query_metrics)
        summary[key] = round(total / n, 4)

    # Overlap
    overlap_counts = defaultdict(int)
    gain_count, loss_count = 0, 0
    rerank_gain_count, rerank_loss_count = 0, 0
    has_rerank = False
    for pq in per_query_metrics:
        ov = pq.get("overlap", {})
        overlap_counts[ov.get("category", "both_miss")] += 1
        if ov.get("hybrid_gain"):
            gain_count += 1
        if ov.get("hybrid_loss"):
            loss_count += 1
        if ov.get("rerank_gain") is not None:
            has_rerank = True
            if ov["rerank_gain"]:
                rerank_gain_count += 1
        if ov.get("rerank_loss") is not None:
            if ov["rerank_loss"]:
                rerank_loss_count += 1

    summary["overlap"] = {
        "both_hit": overlap_counts["both_hit"],
        "bm25_only": overlap_counts["bm25_only"],
        "dense_only": overlap_counts["dense_only"],
        "both_miss": overlap_counts["both_miss"],
        "hybrid_gain": gain_count,
        "hybrid_loss": loss_count,
        "rerank_gain": rerank_gain_count if has_rerank else None,
        "rerank_loss": rerank_loss_count if has_rerank else None,
    }
    summary["overlap"]["total"] = sum(overlap_counts.values())

    # Error summary
    error_types = defaultdict(int)
    for pq in per_query_metrics:
        err = pq.get("error")
        if err:
            error_types[err["error_type"]] += 1
    summary["error_summary"] = dict(error_types)

    return summary


# ═══════════════════════════════════════════════════════════════
# 报告生成
# ═══════════════════════════════════════════════════════════════

def write_evaluation_report(
    summary: dict,
    per_query_metrics: list[dict],
    methods: list[str],
    k_values: list[int],
    match_method: str,
    thresholds: dict,
    output_path: Path,
) -> None:
    """生成中文评测报告 (Markdown)。"""
    method_labels = {
        "bm25": "BM25",
        "dense": "Dense",
        "rrf_hybrid": "RRF Hybrid",
        "hybrid_rerank": "Hybrid + Reranker",
        "union_candidate": "Union Candidate",
    }

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# OnboardRAG 检索评测报告\n\n")
        f.write(f"**匹配方法:** {match_method}\n\n")
        f.write(f"**阈值:** 3-gram 包含度={thresholds['containment']}, "
                f"ROUGE-L={thresholds['rouge_l']}, "
                f"部分匹配率={thresholds['partial_ratio']}\n\n")

        # 方法说明
        config = summary.get("config", {})
        bm_k = config.get("bm25_top_k", 30)
        dn_k = config.get("dense_top_k", 30)
        rrf_k = config.get("rrf_k", 60)
        rerank_model = config.get("reranker_model", "")
        emb_model = config.get("embedding_model", "")
        f.write("## 消融方法\n\n")
        f.write(f"| 方法 | 说明 |\n")
        f.write(f"|------|------|\n")
        f.write(f"| BM25 | 纯关键词检索，top-{bm_k} |\n")
        f.write(f"| Dense | 纯向量检索 ({emb_model})，top-{dn_k} |\n")
        f.write(f"| RRF Hybrid | BM25 top-{bm_k} + Dense top-{dn_k} → RRF (k={rrf_k}) 融合 → top-30 |\n")
        if rerank_model:
            f.write(f"| Hybrid + Reranker | BM25 top-{bm_k} ∪ Dense top-{dn_k} 去重 → {rerank_model} 精排 → top-30 |\n")
        else:
            f.write(f"| Hybrid + Reranker | (未运行) |\n")
        f.write(f"| Union Candidate | BM25 top-{bm_k} ∪ Dense top-{dn_k} 去重，候选池召回上限 |\n")
        f.write("\n")

        # ── 总体指标 ──
        f.write("## 总体指标\n\n")

        # Query-level Hit@K + MRR
        f.write("### Query-level 指标\n\n")
        header = "| 方法 | " + " | ".join(f"Hit@{k}" for k in k_values) + " | MRR |\n"
        sep = "|------|" + "|".join(["------" for _ in k_values]) + "|------|\n"
        f.write(header)
        f.write(sep)
        for method in methods:
            label = method_labels.get(method, method)
            hits = " | ".join(
                f"{summary[f'{method}_hit'][f'hit@{k}']:.2%}"
                for k in k_values
            )
            mrr = f"{summary[f'{method}_mrr']:.4f}"
            f.write(f"| {label} | {hits} | {mrr} |\n")

        # Evidence-unit Recall@K
        f.write("\n### Evidence Unit Recall\n\n")
        header = "| 方法 | " + " | ".join(f"Recall@{k}" for k in k_values) + " |\n"
        sep = "|------|" + "|".join(["------" for _ in k_values]) + "|\n"
        f.write(header)
        f.write(sep)
        for method in methods:
            label = method_labels.get(method, method)
            recalls = " | ".join(
                f"{summary[f'{method}_eu_recall@{k}']:.2%}"
                for k in k_values
            )
            f.write(f"| {label} | {recalls} |\n")

        # Evidence Group Hit@K
        f.write("\n### Evidence Group Full Hit\n\n")
        header = "| 方法 | " + " | ".join(f"Hit@{k}" for k in k_values) + " |\n"
        sep = "|------|" + "|".join(["------" for _ in k_values]) + "|\n"
        f.write(header)
        f.write(sep)
        for method in methods:
            label = method_labels.get(method, method)
            hits = " | ".join(
                f"{summary[f'{method}_eu_group_hit@{k}']:.2%}"
                for k in k_values
            )
            f.write(f"| {label} | {hits} |\n")

        # ── Candidate-stage 指标 ──
        f.write("\n## Candidate-stage 指标\n\n")
        f.write("| 指标 | 值 |\n")
        f.write("|------|----|\n")
        cand_pairs = [
            ("BM25 Candidate Recall@30", "bm25_candidate_recall@30"),
            ("Dense Candidate Recall@30", "dense_candidate_recall@30"),
            ("Union Candidate Recall@30", "union_candidate_recall@30"),
            ("RRF Recall@10", "rrf_recall@10"),
            ("Rerank Recall@10", "rerank_recall@10"),
        ]
        for label, key in cand_pairs:
            val = summary.get(key, 0.0)
            f.write(f"| {label} | {val:.2%} |\n")

        # ── Overlap 分析 ──
        f.write("\n## Overlap 分析\n\n")
        ov = summary["overlap"]
        total = ov["total"]
        f.write(f"**基于 Evidence Group Full Hit@10**  (有效样本数: {total})\n\n")
        f.write("| 分类 | 数量 | 占比 |\n")
        f.write("|------|------|------|\n")
        f.write(f"| 两者都命中 (BM25 ✓, Dense ✓) | {ov['both_hit']} | {ov['both_hit']/total*100:.1f}% |\n")
        f.write(f"| BM25 独有 (BM25 ✓, Dense ✗) | {ov['bm25_only']} | {ov['bm25_only']/total*100:.1f}% |\n")
        f.write(f"| Dense 独有 (Dense ✓, BM25 ✗) | {ov['dense_only']} | {ov['dense_only']/total*100:.1f}% |\n")
        f.write(f"| 两者都未命中 | {ov['both_miss']} | {ov['both_miss']/total*100:.1f}% |\n")
        f.write(f"| 🔵 RRF Hybrid Gain (增量) | {ov['hybrid_gain']} | {ov['hybrid_gain']/total*100:.1f}% |\n")
        f.write(f"| 🔴 RRF Hybrid Loss (损失) | {ov['hybrid_loss']} | {ov['hybrid_loss']/total*100:.1f}% |\n")
        if ov.get("rerank_gain") is not None:
            f.write(f"| 🟢 Reranker Gain (增量) | {ov['rerank_gain']} | {ov['rerank_gain']/total*100:.1f}% |\n")
        else:
            f.write(f"| 🟢 Reranker Gain (增量) | N/A | (未运行) |\n")
        if ov.get("rerank_loss") is not None:
            f.write(f"| 🟠 Reranker Loss (损失) | {ov['rerank_loss']} | {ov['rerank_loss']/total*100:.1f}% |\n")
        else:
            f.write(f"| 🟠 Reranker Loss (损失) | N/A | (未运行) |\n")

        # ── 按意图分组 ──
        f.write("\n## 按意图分组 (Evidence Group Full Hit@10)\n\n")
        _write_grouped_table(f, per_query_metrics, "intent", methods, method_labels)

        # ── 按难度分组 ──
        f.write("\n## 按难度分组 (Evidence Group Full Hit@10)\n\n")
        _write_grouped_table(f, per_query_metrics, "difficulty", methods, method_labels)


def _write_grouped_table(f, per_query_metrics, group_key, methods, method_labels):
    """按指定维度分组统计并写入 Markdown 表格。"""
    groups = defaultdict(list)
    for pq in per_query_metrics:
        key_val = pq.get(group_key, "unknown")
        groups[key_val].append(pq)

    header = "| " + group_key + " | 样本数 |"
    for method in methods:
        header += f" {method_labels.get(method, method)} Hit@10 |"
    header += "\n"
    sep = "|------|------|" + "|".join(["------" for _ in methods]) + "|\n"
    f.write(header)
    f.write(sep)

    for key, items in sorted(groups.items()):
        n = len(items)
        f.write(f"| {key} | {n} |")
        for method in methods:
            gh_sum = sum(
                int(item.get("evidence_unit", {}).get(method, {}).get("group_hit@10", False))
                for item in items
            )
            f.write(f" {gh_sum/n:.2%} |")
        f.write("\n")


def write_error_analysis_jsonl(
    per_query_metrics: list[dict],
    output_path: Path,
) -> None:
    """将 error analysis 结果写入 JSONL 文件。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for pq in per_query_metrics:
            err = pq.get("error")
            if err:
                f.write(json.dumps(err, ensure_ascii=False) + "\n")


# ═══════════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="OnboardRAG 检索评测 — 阶段二：从检索结果文件计算指标"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="检索结果 JSONL 文件所在目录（加载目录下所有 .jsonl 文件）",
    )
    parser.add_argument(
        "--results-file",
        type=str,
        default=None,
        help="单个检索结果 JSONL 文件路径",
    )
    parser.add_argument(
        "--evalset",
        type=str,
        default="data/eval/eval_queries_v2.jsonl",
        help="评测集 JSONL 文件路径（用于获取 gold evidence_units）",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/eval",
        help="输出目录（默认: outputs/eval/）",
    )
    parser.add_argument(
        "--match-method",
        type=str,
        choices=["fuzzy", "rouge_l", "containment", "partial_ratio"],
        default="fuzzy",
        help="evidence 匹配方法（默认: fuzzy）",
    )
    parser.add_argument(
        "--exact-only",
        action="store_true",
        help="仅使用精确匹配，跳过模糊匹配",
    )
    parser.add_argument(
        "--k-values",
        type=str,
        default="3,5,10,30",
        help="截断值列表，逗号分隔（默认: 3,5,10,30）",
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
    args = parser.parse_args()

    # 解析 k_values
    k_values = [int(k.strip()) for k in args.k_values.split(",")]

    # 收集 JSONL 文件
    jsonl_paths: list[Path] = []
    if args.results_dir:
        results_dir = PROJECT_ROOT / args.results_dir
        if results_dir.is_dir():
            jsonl_paths = sorted(results_dir.glob("*.jsonl"))
        else:
            print(f"错误: 目录不存在: {results_dir}")
            return 1
    if args.results_file:
        p = PROJECT_ROOT / args.results_file
        if p.exists():
            jsonl_paths.append(p)
        else:
            print(f"错误: 文件不存在: {p}")
            return 1

    if not jsonl_paths:
        print("错误: 请指定 --results-dir 或 --results-file")
        return 1

    print(f"加载了 {len(jsonl_paths)} 个检索结果文件:")
    for p in jsonl_paths:
        print(f"  - {p}")

    # 加载数据
    rows = load_retrieval_results(jsonl_paths)
    print(f"共 {len(rows)} 行检索结果")

    evalset_path = PROJECT_ROOT / args.evalset
    entries = load_eval_entries(evalset_path)
    print(f"共 {len(entries)} 条评测样本")

    # 按 query_id 分组
    grouped = group_results_by_query(rows)

    # 检测可用的 methods
    all_methods_set = set()
    for r in rows:
        all_methods_set.add(r["method"])
    methods = sorted(all_methods_set, key=lambda m: ["bm25", "dense", "rrf_hybrid", "union_candidate", "hybrid_rerank"].index(m) if m in ["bm25", "dense", "rrf_hybrid", "union_candidate", "hybrid_rerank"] else 99)

    # 移除不参与指标计算的方法（union_candidate 只用于 candidate 指标）
    metric_methods = [m for m in methods if m != "union_candidate"]
    print(f"检测到的方法: {methods}")
    print(f"参与指标计算的方法: {metric_methods}")

    # 创建匹配函数
    if args.exact_only:
        match_fn = None
        match_method = "exact"
    elif args.match_method == "rouge_l":
        match_fn = make_rouge_l_matcher(threshold=args.rouge_l_threshold)
        match_method = "rouge_l"
    elif args.match_method == "containment":
        match_fn = make_containment_matcher(threshold=args.containment_threshold)
        match_method = "containment"
    elif args.match_method == "partial_ratio":
        match_fn = make_partial_ratio_matcher(threshold=args.partial_ratio_threshold)
        match_method = "partial_ratio"
    else:
        match_fn = make_fuzzy_matcher(
            containment_threshold=args.containment_threshold,
            rouge_l_threshold=args.rouge_l_threshold,
            partial_ratio_threshold=args.partial_ratio_threshold,
        )
        match_method = "fuzzy"

    thresholds = {
        "containment": args.containment_threshold,
        "rouge_l": args.rouge_l_threshold,
        "partial_ratio": args.partial_ratio_threshold,
    }

    # ── 逐 query 计算指标 ──
    per_query_metrics = []
    error_entries = []

    for entry in entries:
        qid = entry["id"]
        query_data = grouped.get(qid, {})

        if not entry.get("answerable", True):
            continue

        if not entry.get("evidence"):
            continue

        # Query-level 指标
        ql = compute_query_level_metrics(
            query_data, entry, match_fn, metric_methods, k_values
        )

        # Evidence-unit 指标
        eu = compute_evidence_unit_metrics_all(
            query_data, entry, match_fn, metric_methods, k_values
        )

        # Candidate 指标
        cand = compute_candidate_metrics(query_data, entry, match_fn)

        # Overlap 分析
        overlap = compute_overlap_analysis(query_data, entry, match_fn, k=10)

        # Error 分析
        error = compute_error_analysis(query_data, entry, match_fn)

        pq = {
            "query_id": qid,
            "query": entry["query"],
            "intent": entry.get("intent", ""),
            "difficulty": entry.get("difficulty", ""),
            "query_level": ql,
            "evidence_unit": eu,
            "candidate": cand,
            "overlap": overlap,
            "error": error,
        }
        per_query_metrics.append(pq)

    # ── 汇总 ──
    summary = aggregate_metrics(per_query_metrics, metric_methods, k_values)
    summary["match_method"] = match_method
    summary["thresholds"] = thresholds
    summary["k_values"] = k_values
    summary["total_queries"] = len(entries)
    summary["total_answerable"] = len(per_query_metrics)
    # 从检索结果中提取 config（所有行共享同一 config，取第一行即可）
    summary["config"] = rows[0].get("config", {}) if rows else {}

    # ── 输出 ──
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    suffix = match_method if match_method != "fuzzy" else "full"

    # Metrics summary JSON
    metrics_path = output_dir / f"metrics_summary_{suffix}.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n  Metrics Summary: {metrics_path}")

    # Evaluation report MD
    report_path = output_dir / f"evaluation_report_{suffix}.md"
    write_evaluation_report(
        summary, per_query_metrics, metric_methods, k_values,
        match_method, thresholds, report_path,
    )
    print(f"  Evaluation Report: {report_path}")

    # Error analysis JSONL
    error_path = output_dir / f"error_analysis_{suffix}.jsonl"
    write_error_analysis_jsonl(per_query_metrics, error_path)
    error_count = sum(1 for pq in per_query_metrics if pq.get("error"))
    print(f"  Error Analysis ({error_count} errors): {error_path}")

    # ── 终端摘要 ──
    print("\n" + "=" * 60)
    print(f"  评测完成 (匹配方法: {match_method})")
    print("=" * 60)
    print(f"  总样本数: {summary['total_queries']}")
    print(f"  有效样本数: {summary['total_answerable']}")
    print()
    print(f"  {'方法':<20} {'Hit@10':<10} {'MRR':<10} {'EU Recall@10':<14}")
    print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*14}")
    for method in metric_methods:
        label = {"bm25": "BM25", "dense": "Dense", "rrf_hybrid": "RRF Hybrid", "hybrid_rerank": "Hybrid+Rerank"}.get(method, method)
        hit10 = summary[f"{method}_hit"]["hit@10"]
        mrr = summary[f"{method}_mrr"]
        rec10 = summary[f"{method}_eu_recall@10"]
        print(f"  {label:<20} {hit10:<10.2%} {mrr:<10.4f} {rec10:<14.2%}")

    print(f"\n  Candidate Recall:")
    print(f"    BM25 Candidate@30:    {summary['bm25_candidate_recall@30']:.2%}")
    print(f"    Dense Candidate@30:   {summary['dense_candidate_recall@30']:.2%}")
    print(f"    Union Candidate@30:   {summary['union_candidate_recall@30']:.2%}")
    print(f"    RRF Recall@10:        {summary['rrf_recall@10']:.2%}")
    print(f"    Rerank Recall@10:     {summary['rerank_recall@10']:.2%}")

    ov = summary["overlap"]
    print(f"\n  Overlap (Hit@10):")
    print(f"    Both: {ov['both_hit']}  BM25-only: {ov['bm25_only']}  Dense-only: {ov['dense_only']}  Both-miss: {ov['both_miss']}")
    print(f"    RRF Gain: {ov['hybrid_gain']}  RRF Loss: {ov['hybrid_loss']}")
    rg = ov.get('rerank_gain')
    rl = ov.get('rerank_loss')
    if rg is not None:
        print(f"    Rerank Gain: {rg}  Rerank Loss: {rl}")
    else:
        print(f"    Rerank Gain: N/A  Rerank Loss: N/A (reranker 未运行)")

    print(f"\n  Errors: {summary.get('error_summary', {})}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
