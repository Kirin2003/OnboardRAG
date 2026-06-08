#!/usr/bin/env python3
"""
run_eval.py — OnboardRAG 评测脚本。

从 data/eval/eval_queries.jsonl 读取评测样本，执行检索评测和生成评测。

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

    # 全量评测（doc + evidence + 生成）
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
    exact_evidence_match,
    fuzzy_evidence_match,
    evidence_hit_at_k,
    evidence_mrr,
    evidence_recall_at_k,
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
# 生成评测（LLM-as-Judge）
# ═══════════════════════════════════════════════════════════════

_JUDGE_PROMPT = """你是一个严格但公正的评测员。你的任务是评估一个 RAG 系统对用户问题的回答质量。

请根据以下标准和参考依据进行评分。

【用户问题】
{query}

【参考依据（标准答案）】
{reference_answer}

【系统回答】
{generated_answer}

【评分维度】
1. 正确性 (correctness)：回答是否正确回答了用户问题？（0-2分）
   - 0 = 错误或答非所问
   - 1 = 部分正确，有遗漏或偏差
   - 2 = 完全正确

2. 完备性 (completeness)：是否遗漏了关键步骤、条件或信息？（0-2分）
   - 0 = 遗漏了全部关键信息
   - 1 = 遗漏了部分关键信息
   - 2 = 信息完备，无遗漏

3. 忠实性 (faithfulness)：回答是否严格基于参考依据，有无编造？（0-2分）
   - 0 = 大量编造或与参考依据矛盾
   - 1 = 部分内容无法在参考依据中找到
   - 2 = 完全忠实于参考依据

{extra_dimensions}

请以 JSON 格式返回评分结果：
{{"correctness": <0-2>, "completeness": <0-2>, "faithfulness": <0-2>{extra_keys}, "comment": "<简短评语>"}}
"""

_UNANSWERABLE_DIMENSIONS = """
4. 拒答正确性 (rejection)：系统是否正确地拒绝回答或引导转人工？（0-2分）
   - 0 = 编造了答案、幻觉
   - 1 = 部分拒答但仍有误导信息
   - 2 = 明确拒答并给出合理引导
"""
_UNANSWERABLE_KEYS = ', "rejection": <0-2>'


def _build_judge_prompt(entry: dict, answer: str) -> str:
    """根据样本类型构建 judge prompt。"""
    extra_dimensions = ""
    extra_keys = ""
    if not entry["answerable"]:
        extra_dimensions = _UNANSWERABLE_DIMENSIONS
        extra_keys = _UNANSWERABLE_KEYS

    return _JUDGE_PROMPT.format(
        query=entry["query"],
        reference_answer=entry["reference_answer"],
        generated_answer=answer,
        extra_dimensions=extra_dimensions,
        extra_keys=extra_keys,
    )


def eval_generation(entries: list[dict], top_k: int = 5, mode: str = "hybrid") -> dict:
    """执行生成评测（LLM-as-Judge）。

    对每条样本，先检索 + 生成答案，再用 judge LLM 打分。

    Args:
        entries: 评测样本列表
        top_k: 检索返回的 chunk 数量
        mode: 检索模式 ("hybrid" / "dense" / "bm25")
    """
    from src.query_rewriter import QueryRewriter
    from src.reranker import Reranker
    from src.generator import Generator
    from openai import OpenAI
    from src.config import LLM_BASE_URL, LLM_API_KEY, LLM_MODEL

    rewriter = QueryRewriter()
    retriever = Retriever()
    reranker = Reranker()
    generator = Generator()
    judge_client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)

    results = []
    scores = defaultdict(list)

    for i, entry in enumerate(entries):
        print(f"  [{i+1}/{len(entries)}] {entry['id']}: {entry['query'][:40]}...")

        query = rewriter.rewrite(entry["query"])
        chunks = retriever.retrieve(query, top_k=top_k, mode=mode)
        chunks = reranker.rerank(query, chunks, top_k=top_k)

        answer, sources = generator.generate(query, chunks[:top_k])

        # LLM-as-Judge 打分
        judge_prompt = _build_judge_prompt(entry, answer)
        try:
            resp = judge_client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": judge_prompt}],
                temperature=0.1,
                max_tokens=512,
            )
            judge_text = resp.choices[0].message.content.strip()
            # 提取 JSON 部分
            json_start = judge_text.find("{")
            json_end = judge_text.rfind("}") + 1
            judge_result = json.loads(judge_text[json_start:json_end]) if json_start >= 0 else {}
        except Exception as ex:
            print(f"    ⚠ Judge 评测失败: {ex}")
            judge_result = {"error": str(ex)}

        # 汇总分数
        for dim in ["correctness", "completeness", "faithfulness", "rejection"]:
            val = judge_result.get(dim, None)
            if isinstance(val, (int, float)):
                scores[dim].append(val)

        results.append({
            "id": entry["id"],
            "query": entry["query"],
            "intent": entry["intent"],
            "difficulty": entry["difficulty"],
            "answerable": entry["answerable"],
            "generated_answer": answer,
            "sources": sources,
            "judge_scores": judge_result,
            "reference_answer": entry["reference_answer"],
        })

    # 汇总
    summary = {}
    for dim, vals in scores.items():
        if vals:
            summary[f"avg_{dim}"] = round(sum(vals) / len(vals), 3)
            summary[f"count_{dim}"] = len(vals)

    return {"summary": summary, "details": results}


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


def print_generation_report(gen_result: dict) -> None:
    """打印生成评测报告。"""
    summary = gen_result["summary"]
    print("\n" + "=" * 60)
    print("  生成评测结果 (LLM-as-Judge)")
    print("=" * 60)
    for dim in ["correctness", "completeness", "faithfulness", "rejection"]:
        key = f"avg_{dim}"
        if key in summary:
            print(f"  avg_{dim}: {summary[key]:.2f}/2 (n={summary[f'count_{dim}']})")


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
    verbose: bool = False,
) -> dict:
    """执行 evidence-level 检索评测。

    只对 answerable=true 且 evidence 非空的样本计算指标。
    同时计算 exact match + 指定方法的匹配结果。

    Args:
        entries: 评测样本列表
        top_k: 检索返回的 chunk 数量
        containment_threshold: char 3-gram containment 阈值
        rouge_l_threshold: ROUGE-L recall 阈值
        partial_ratio_threshold: partial_ratio 阈值
        mode: 检索模式 ("hybrid" / "dense" / "bm25")
        exact_only: 仅计算 exact match，跳过第二匹配器（等价于 match_method="exact"）
        match_method: 第二匹配器方法（仅在 exact_only=False 时生效）
            - "fuzzy": 三合一模糊匹配（containment+ROUGE-L / partial_ratio）（默认）
            - "rouge_l": 纯 ROUGE-L 匹配
            - "containment": 纯 char 3-gram containment 匹配
            - "partial_ratio": 纯 partial_ratio 匹配
        verbose: 收集详细的证据匹配信息（用于调试）

    Returns:
        {summary: {...}, details: [...], verbose_data: [...] | None}
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
    verbose_data = [] if verbose else None

    # exact 计数器
    e_hit3, e_hit5, e_mrr_sum = 0, 0, 0.0
    e_rec3_sum, e_rec5_sum = 0.0, 0.0
    # fuzzy 计数器
    f_hit3, f_hit5, f_mrr_sum = 0, 0, 0.0
    f_rec3_sum, f_rec5_sum = 0.0, 0.0
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
        })

        # ── verbose: 仅收集 exact 未命中的样本 ──
        if verbose and not e_hit5_val:
            # 构建 evidence 详情
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
                    "body_text": chunk_text[:300],  # 截断便于阅读
                    "body_text_normalized": c_norm[:300],
                }

                # 对每个 evidence 检查 exact match 和 第二匹配器
                exact_matched_indices = []
                second_match_details = {}
                for idx, ev in enumerate(evidence_list):
                    quote = ev.get("quote", "")
                    if not quote:
                        continue
                    # exact
                    e_norm = normalize_text(quote)
                    if e_norm and e_norm in c_norm:
                        exact_matched_indices.append(idx)

                    # 第二匹配器（如果启用）
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
                "retrieved_chunks": chunk_details,
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
    }

    result = {"summary": summary, "details": results}
    if verbose:
        result["verbose_data"] = verbose_data
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


def save_evidence_verbose(verbose_data: list[dict], output_dir: str | Path) -> Path:
    """保存 verbose 证据匹配详情 JSON 文件。

    每条样本包含：
    - 标准 evidence 的原文和归一化文本
    - 每个检索到的 chunk 的原文（截断）和归一化文本
    - 每个 evidence 在哪个 chunk 中精确命中
    - fuzzy 匹配分数明细（如启用）

    Args:
        verbose_data: eval_evidence_retrieval 返回的 verbose_data
        output_dir: 输出目录

    Returns:
        json_path
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "证据匹配详情_verbose.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(verbose_data, f, ensure_ascii=False, indent=2)

    return json_path


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
        "--retrieval-only",
        action="store_true",
        help="仅执行检索评测（doc + evidence），跳过 LLM 生成评测",
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
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="输出详细证据匹配 JSON，包含每个 query 的标准 evidence 和检索到的 evidence 对比",
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
            verbose=args.verbose,
        )
        print_evidence_report(evidence_result, mode=args.retrieval_mode)
        full_result["evidence_retrieval"] = evidence_result

        # 保存 CSV + summary MD
        csv_path, md_path = save_evidence_results(
            evidence_result, args.output_dir, mode=args.retrieval_mode,
        )
        print(f"\n  证据详细结果 (CSV): {csv_path}")
        print(f"  证据汇总报告 (MD):  {md_path}")

        # 保存 verbose JSON（如果启用）
        if args.verbose and "verbose_data" in evidence_result:
            verbose_path = save_evidence_verbose(evidence_result["verbose_data"], args.output_dir)
            print(f"  证据匹配详情 (JSON): {verbose_path}")

    # ── 生成评测（LLM-as-Judge）── 仅在 all 模式下运行
    gen_result = None
    if args.mode == "all" and not args.retrieval_only:
        print("\n── 生成评测 (LLM-as-Judge) ──")
        gen_result = eval_generation(entries, top_k=args.top_k, mode=args.retrieval_mode)
        print_generation_report(gen_result)
        full_result["generation"] = gen_result

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
