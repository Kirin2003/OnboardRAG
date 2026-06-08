#!/usr/bin/env python3
"""
analyze_hybrid.py — Hybrid 消融分析

逐 query 对比 hybrid / bm25 / dense 三种检索模式的命中情况，
分类统计 hybrid 真正的增益 (gain) 和损失 (loss)。

用法:
    conda activate rag
    python scripts/ablation/analyze_hybrid.py

    # 指定命中指标（默认 evidence_group_hit@5）
    python scripts/ablation/analyze_hybrid.py --metric evidence_group_hit@5
    python scripts/ablation/analyze_hybrid.py --metric fuzzy_hit@5
    python scripts/ablation/analyze_hybrid.py --metric evidence_unit_recall@5
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ═══════════════════════════════════════════════════════════════
# 数据加载 & 评测
# ═══════════════════════════════════════════════════════════════

def load_entries() -> list[dict]:
    eval_path = PROJECT_ROOT / "data/eval/eval_queries_v2.jsonl"
    if not eval_path.exists():
        raise FileNotFoundError(f"评测文件不存在: {eval_path}")
    with open(eval_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def run_eval_for_mode(entries: list[dict], mode: str, metric: str) -> dict[str, dict]:
    """对指定检索模式执行 evidence-level 评测，返回 {query_id: per_query_info}。

    metric 可以是:
      - evidence_group_hit@5  (bool, 是否所有 evidence unit 都被覆盖)
      - evidence_unit_recall@5 (float, evidence unit 召回比例)
      - fuzzy_hit@5           (bool, 模糊匹配是否命中任意 evidence)
    """
    from src.eval_metrics import (
        get_evidence_units,
        compute_evidence_unit_metrics,
        make_fuzzy_matcher,
    )
    from src.retriever import Retriever
    from src.config import RETRIEVAL_TOP_K

    retriever = Retriever()
    match_fn = make_fuzzy_matcher()
    top_k = RETRIEVAL_TOP_K

    results: dict[str, dict] = {}

    for entry in entries:
        eid = entry["id"]

        if not entry.get("answerable", True) or not entry.get("evidence"):
            results[eid] = {
                "id": eid,
                "query": entry["query"],
                "intent": entry["intent"],
                "difficulty": entry.get("difficulty", ""),
                "answerable": False,
                "evidence_group_hit@5": False,
                "evidence_unit_recall@5": 0.0,
                "fuzzy_hit@5": False,
            }
            continue

        chunks = retriever.retrieve(entry["query"], top_k=top_k, mode=mode)
        evidence_units = get_evidence_units(entry)

        # 计算 evidence unit 指标
        eu_k5 = compute_evidence_unit_metrics(chunks, evidence_units, 5, match_fn)
        eu_k3 = compute_evidence_unit_metrics(chunks, evidence_units, 3, match_fn)

        # 计算 fuzzy_hit@5 (evidence-level, 非 unit-level)
        from src.eval_metrics import evidence_hit_at_k
        f_hit5_val, _ = evidence_hit_at_k(chunks, entry["evidence"], 5, match_fn)

        results[eid] = {
            "id": eid,
            "query": entry["query"],
            "intent": entry["intent"],
            "difficulty": entry.get("difficulty", ""),
            "eval_focus": ",".join(entry.get("eval_focus", [])),
            "expected_docs": entry.get("expected_docs", []),
            "answerable": True,
            "evidence_group_hit@5": eu_k5["evidence_group_full_hit"],
            "evidence_group_hit@3": eu_k3["evidence_group_full_hit"],
            "evidence_unit_recall@5": eu_k5["evidence_unit_recall"],
            "evidence_unit_recall@3": eu_k3["evidence_unit_recall"],
            "evidence_unit_total": eu_k5["evidence_unit_total"],
            "fuzzy_hit@5": f_hit5_val,
        }

    return results


# ═══════════════════════════════════════════════════════════════
# 分类分析
# ═══════════════════════════════════════════════════════════════

def classify_queries(
    hy_data: dict[str, dict],
    bm_data: dict[str, dict],
    dn_data: dict[str, dict],
    metric: str,
) -> dict:
    """逐 query 分类统计。

    metric 为 bool 类型时（如 evidence_group_hit@5, fuzzy_hit@5):
        hit = True

    metric 为 float 类型时（如 evidence_unit_recall@5):
        hit = value >= 1.0  (所有 unit 全部命中)
    """
    categories = {
        "both_hit": [],       # BM25 ✓, dense ✓
        "bm25_only": [],      # BM25 ✓, dense ✗
        "dense_only": [],     # dense ✓, BM25 ✗
        "both_miss": [],      # 两者都 ✗
        "hybrid_gain": [],    # hybrid ✓ 但两者单独都 ✗
        "hybrid_loss": [],    # hybrid ✗ 但 (BM25 ✓ 或 dense ✓)
    }

    for eid in hy_data:
        hy = hy_data[eid]
        bm = bm_data[eid]
        dn = dn_data[eid]

        # 跳过不可回答的
        if not hy.get("answerable"):
            continue

        hy_val = hy[metric]
        bm_val = bm[metric]
        dn_val = dn[metric]

        # 判断 hit
        if isinstance(hy_val, bool):
            hy_hit = hy_val
            bm_hit = bm_val
            dn_hit = dn_val
        else:
            # float 类型：>= 1.0 即全部命中
            hy_hit = hy_val >= 1.0
            bm_hit = bm_val >= 1.0
            dn_hit = dn_val >= 1.0

        # ── 基础分类 ──
        info = {
            "id": eid,
            "query": hy["query"],
            "intent": hy["intent"],
            "difficulty": hy.get("difficulty", ""),
            "eval_focus": hy.get("eval_focus", ""),
            "expected_docs": hy.get("expected_docs", []),
            "evidence_unit_total": hy.get("evidence_unit_total", 0),
            "hy_hit": hy_hit,
            "bm_hit": bm_hit,
            "dn_hit": dn_hit,
            "hy_val": hy_val,
            "bm_val": bm_val,
            "dn_val": dn_val,
        }

        if bm_hit and dn_hit:
            categories["both_hit"].append(info)
        elif bm_hit and not dn_hit:
            categories["bm25_only"].append(info)
        elif dn_hit and not bm_hit:
            categories["dense_only"].append(info)
        else:  # both miss
            categories["both_miss"].append(info)

        # ── hybrid gain / loss ──
        if hy_hit and not bm_hit and not dn_hit:
            categories["hybrid_gain"].append(info)
        if not hy_hit and (bm_hit or dn_hit):
            categories["hybrid_loss"].append(info)

    return categories


# ═══════════════════════════════════════════════════════════════
# 输出
# ═══════════════════════════════════════════════════════════════

def print_report(categories: dict, metric: str) -> None:
    total = sum(len(v) for v in categories.values() if v is categories["both_hit"]
                or v is categories["bm25_only"] or v is categories["dense_only"]
                or v is categories["both_miss"])
    # 更准确的总数
    all_ids = set()
    for cat in ["both_hit", "bm25_only", "dense_only", "both_miss"]:
        for q in categories[cat]:
            all_ids.add(q["id"])
    total = len(all_ids)

    both = len(categories["both_hit"])
    bm25_only = len(categories["bm25_only"])
    dense_only = len(categories["dense_only"])
    both_miss = len(categories["both_miss"])
    gain = len(categories["hybrid_gain"])
    loss = len(categories["hybrid_loss"])

    print("\n" + "=" * 70)
    print(f"  Query 命中分类统计（指标: {metric}）")
    print("=" * 70)
    print(f"  有效样本数: {total}")
    print()
    print(f"  {'分类':<28} {'数量':<8} {'占比':<10}")
    print(f"  {'-' * 28} {'-' * 8} {'-' * 10}")
    print(f"  {'两者都命中 (BM25 ✓, dense ✓)':<28} {both:<8} {both / total * 100:.1f}%")
    print(f"  {'BM25 独有 (BM25 ✓, dense ✗)':<28} {bm25_only:<8} {bm25_only / total * 100:.1f}%")
    print(f"  {'Dense 独有 (BM25 ✗, dense ✓)':<28} {dense_only:<8} {dense_only / total * 100:.1f}%")
    print(f"  {'两者都未命中':<28} {both_miss:<8} {both_miss / total * 100:.1f}%")
    print(f"  {'─' * 46}")
    print(f"  {'🔵 hybrid_gain (增量)':<28} {gain:<8} {gain / total * 100:.1f}%")
    print(f"  {'🔴 hybrid_loss (损失)':<28} {loss:<8} {loss / total * 100:.1f}%")

    # ── hybrid_gain 详情 ──
    if gain > 0:
        print(f"\n  {'=' * 70}")
        print(f"  🔵 hybrid_gain 详情（hybrid 命中，但 BM25 和 dense 单独都未命中）")
        print(f"  {'=' * 70}")
        for q in categories["hybrid_gain"]:
            print_query_detail(q, metric)
    else:
        print(f"\n  🔵 hybrid_gain: 无（hybrid 没有比任一单路检索多命中任何 query）")

    # ── hybrid_loss 详情 ──
    if loss > 0:
        print(f"\n  {'=' * 70}")
        print(f"  🔴 hybrid_loss 详情（BM25 或 dense 命中，但 hybrid 未命中）")
        print(f"  {'=' * 70}")
        for q in categories["hybrid_loss"]:
            print_query_detail(q, metric)
    else:
        print(f"\n  🔴 hybrid_loss: 无")

    # ── 按意图分组 ──
    print(f"\n  {'─' * 46}")
    print(f"  按意图分组")
    print(f"  {'意图':<20} {'总数':<6} {'both':<6} {'bm25独':<8} {'dense独':<8} {'都miss':<8} {'gain':<6} {'loss':<6}")
    for intent in ["人事与员工事务", "账号与统一门户", "内网访问", "办公平台"]:
        by_intent = defaultdict(list)
        for cat_name, cat_list in categories.items():
            for q in cat_list:
                if q["intent"] == intent:
                    by_intent[cat_name].append(q)
        ni = sum(len(v) for v in by_intent.values())
        if ni == 0:
            continue
        print(f"  {intent:<20} {ni:<6} "
              f"{len(by_intent['both_hit']):<6} "
              f"{len(by_intent['bm25_only']):<8} "
              f"{len(by_intent['dense_only']):<8} "
              f"{len(by_intent['both_miss']):<8} "
              f"{len(by_intent['hybrid_gain']):<6} "
              f"{len(by_intent['hybrid_loss']):<6}")

    print()


def print_query_detail(q: dict, metric: str) -> None:
    """打印单条 query 的详细命中情况。"""
    status = lambda hit: "✓" if hit else "✗"
    print(f"\n  [{q['id']}] {q['intent']} | {q['difficulty']} | EU={q['evidence_unit_total']}")
    print(f"    查询: {q['query'][:100]}")
    print(f"    期望文档: {', '.join(q['expected_docs'])}")
    print(f"    eval_focus: {q['eval_focus']}")
    if isinstance(q['hy_val'], bool):
        print(f"    hybrid={status(q['hy_hit'])}  bm25={status(q['bm_hit'])}  dense={status(q['dn_hit'])}")
    else:
        print(f"    hybrid={q['hy_val']:.1%}  bm25={q['bm_val']:.1%}  dense={q['dn_val']:.1%}")


# ═══════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Hybrid 消融分析")
    parser.add_argument(
        "--metric", "-m",
        type=str,
        default="evidence_group_hit@5",
        choices=["evidence_group_hit@5", "evidence_group_hit@3",
                 "fuzzy_hit@5", "evidence_unit_recall@5", "evidence_unit_recall@3"],
        help="命中指标（默认: evidence_group_hit@5）",
    )
    args = parser.parse_args()

    print("加载评测数据...")
    entries = load_entries()
    print(f"  共 {len(entries)} 条评测样本")

    for mode, label in [("hybrid", "混合检索"), ("bm25", "仅关键词"), ("dense", "仅向量")]:
        print(f"  评测 {label} ({mode})...")
        if mode == "hybrid":
            hy_data = run_eval_for_mode(entries, mode, args.metric)
        elif mode == "bm25":
            bm_data = run_eval_for_mode(entries, mode, args.metric)
        else:
            dn_data = run_eval_for_mode(entries, mode, args.metric)

    categories = classify_queries(hy_data, bm_data, dn_data, args.metric)
    print_report(categories, args.metric)

    return 0


if __name__ == "__main__":
    sys.exit(main())
