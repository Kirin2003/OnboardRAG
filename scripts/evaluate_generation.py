#!/usr/bin/env python3
"""
evaluate_generation.py — OnboardRAG 生成阶段评估脚本。

使用 LLM-as-Judge 对 RAG 生成的答案进行多维度评分。

输入：
  1. evaluation_dataset.jsonl — 包含 query, intent, answerable, reference_answer,
     evidence_units, expected_behavior
  2. rag_outputs.jsonl — 包含每个 query 的 generated_answer, retrieved_chunks, citations

评估维度（第一版）：
  - correctness:        0/1/2 （null for answerable=false）
  - completeness:       0/1/2 （null for answerable=false）
  - faithfulness:       0/1/2
  - abstention_correct: true/false/null

输出：
  - per-sample 结果 JSONL
  - 按 intent 聚合的 CSV 报表

用法:
    python scripts/evaluate_generation.py \
        --eval-dataset data/eval/eval_queries_v2.jsonl \
        --rag-outputs outputs/eval/rag_outputs.jsonl \
        --output-dir outputs/eval/
"""

import argparse
import csv
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from openai import OpenAI

# 将项目根目录加入 sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import LLM_BASE_URL, LLM_API_KEY, LLM_MODEL, JUDGE_MODEL


# ═══════════════════════════════════════════════════════════════
# LLM-as-Judge Prompt
# ═══════════════════════════════════════════════════════════════

_JUDGE_SYSTEM_PROMPT = """你是一个严格、公正的 RAG 答案质量评估专家。你的任务是根据给定的评估标准，对 AI 助手生成的答案进行多维度评分。

你必须严格按照 JSON 格式输出评估结果，不要输出任何 JSON 之外的内容。"""


def _build_answerable_judge_prompt(
    query: str,
    reference_answer: str,
    evidence_units: list[str],
    generated_answer: str,
    retrieved_chunks: list[dict],
    expected_behavior: str,
) -> str:
    """构建 answerable=true 样本的 judge prompt。

    评估 correctness, completeness, faithfulness 三个维度。
    abstention_correct 设为 null。
    """
    # 格式化 evidence units
    units_text = ""
    for i, unit in enumerate(evidence_units, 1):
        units_text += f"  [{i}] {unit}\n"

    # 格式化 retrieved chunks（截断长文本，供 faithfulness 判断）
    chunks_text = ""
    for i, chunk in enumerate(retrieved_chunks[:10], 1):
        text = chunk.get("body_text", chunk.get("text", ""))[:500]
        doc = chunk.get("doc_title", chunk.get("source_file", ""))
        chunks_text += f"  [{i}] 来源: {doc}\n      内容: {text}\n"

    if not chunks_text:
        chunks_text = "（无检索结果）"

    prompt = f"""## 评估任务

请对以下 RAG 系统生成的答案进行多维度质量评估。

### 用户问题
{query}

### 问题是否可由知识库回答
是（answerable = true）

### 期望行为
{expected_behavior}

### 标准答案（参考）
{reference_answer}

### 候选关键信息点（来自知识库的标准证据）
{units_text if units_text else "（无）"}

### AI 生成的答案
{generated_answer}

### 检索到的文档片段（供判断 Faithfulness）
{chunks_text}

---

## 评分维度与标准

### 1. correctness（正确性）：0 / 1 / 2
- **2 分（完全正确）**：答案中的所有事实性陈述与标准答案和证据一致，没有事实错误。
- **1 分（部分正确）**：答案中大部分事实正确，但存在少量事实错误、张冠李戴或不准确的表述。
- **0 分（错误）**：答案存在严重事实错误，或回答了与用户问题无关的内容。

### 2. completeness（完整性）：0 / 1 / 2
- **2 分（完整）**：答案覆盖了回答问题所必需的所有关键信息点，用户看完即可完全解决问题。
- **1 分（部分完整）**：答案覆盖了部分关键信息点，但遗漏了 1-2 个对用户有实际帮助的重要信息。
- **0 分（不完整）**：答案遗漏了大部分关键信息点，用户看完仍无法有效解决问题。

  注意：只需判断回答该 query 所 **必需** 的关键点是否被覆盖，候选关键信息点中与该 query 无关的可忽略。

### 3. faithfulness（忠实性）：0 / 1 / 2
- **2 分（完全忠实）**：答案中的所有事实性陈述都可以在「检索到的文档片段」中找到明确支持。没有编造、推测或使用外部知识。
- **1 分（基本忠实）**：答案的大部分陈述有检索依据，但存在少数无法在检索片段中找到直接支持的细节。
- **0 分（不忠实）**：答案中存在明显的幻觉（编造不存在的信息），或者大量使用了检索片段之外的内容。

### 4. abstention_correct（拒答正确性）
对于 answerable=true 的问题，此字段设为 **null**。

---

## 输出格式

请严格按照以下 JSON 格式输出评估结果（不要输出任何其他内容）：

```json
{{
  "correctness": <0|1|2>,
  "completeness": <0|1|2>,
  "faithfulness": <0|1|2>,
  "abstention_correct": null,
  "reasoning": {{
    "correctness": "<简要说明 correctness 评分理由>",
    "completeness": "<简要说明 completeness 评分理由>",
    "faithfulness": "<简要说明 faithfulness 评分理由>"
  }}
}}
```"""
    return prompt


def _build_unanswerable_judge_prompt(
    query: str,
    reference_answer: str,
    generated_answer: str,
    retrieved_chunks: list[dict],
    expected_behavior: str,
) -> str:
    """构建 answerable=false 样本的 judge prompt。

    主要评估 abstention_correct 和 faithfulness。
    correctness 和 completeness 设为 null。
    """
    # 格式化 retrieved chunks
    chunks_text = ""
    for i, chunk in enumerate(retrieved_chunks[:10], 1):
        text = chunk.get("body_text", chunk.get("text", ""))[:500]
        doc = chunk.get("doc_title", chunk.get("source_file", ""))
        chunks_text += f"  [{i}] 来源: {doc}\n      内容: {text}\n"

    if not chunks_text:
        chunks_text = "（无检索结果）"

    prompt = f"""## 评估任务

请对以下 RAG 系统生成的答案进行质量评估。

这是一个 **知识库无法回答** 的问题（answerable = false），系统应该正确拒答、说明无法回答、或引导用户联系人工。

### 用户问题
{query}

### 问题是否可由知识库回答
**否**（answerable = false）——知识库中没有相关信息，或系统无权执行此操作。

### 期望行为
{expected_behavior}
- `reject_unknown`：系统应明确说明知识库中没有相关信息。
- `redirect_to_human`：系统应引导用户联系人工支持。

### 参考拒答示例
{reference_answer}

### AI 生成的答案
{generated_answer}

### 检索到的文档片段（供判断 Faithfulness）
{chunks_text}

---

## 评分维度与标准

### 1. correctness（正确性）
对于 answerable=false 的问题，此字段设为 **null**。

### 2. completeness（完整性）
对于 answerable=false 的问题，此字段设为 **null**。

### 3. faithfulness（忠实性）：0 / 1 / 2
即使问题不可回答，仍需判断答案是否忠实于检索上下文：
- **2 分（完全忠实）**：答案正确说明了知识库中无相关信息，没有编造虚假答案。或者即使给出了部分相关信息（如引导性建议），这些信息也能在检索片段中找到支持。
- **1 分（基本忠实）**：答案大体上正确拒答，但包含了少量无法在检索片段中找到支持的细节或建议。
- **0 分（不忠实）**：答案明显编造了不存在的信息，或假装能够回答问题而给出了幻觉内容。

### 4. abstention_correct（拒答正确性）：true / false
- **true**：系统正确识别了这是不可回答的问题，明确拒答或引导用户联系人工，没有编造答案。
- **false**：系统未能正确拒答：
  - 系统假装能回答，编造了答案（应拒答但未拒答）
  - 系统在应该能回答时却错误地拒答（应回答但未回答——不适用于此类样本）
  - 系统虽然说了无法回答，但同时给出了大量编造的信息

---

## 输出格式

请严格按照以下 JSON 格式输出评估结果（不要输出任何其他内容）：

```json
{{
  "correctness": null,
  "completeness": null,
  "faithfulness": <0|1|2>,
  "abstention_correct": <true|false>,
  "reasoning": {{
    "faithfulness": "<简要说明 faithfulness 评分理由>",
    "abstention_correct": "<简要说明 abstention_correct 判断理由>"
  }}
}}
```"""
    return prompt


# ═══════════════════════════════════════════════════════════════
# JSON 解析
# ═══════════════════════════════════════════════════════════════

def parse_judge_response(response_text: str) -> dict | None:
    """从 judge 响应中提取 JSON 对象。

    尝试多种策略：
    1. 直接 json.loads
    2. 提取 ```json ... ``` 代码块
    3. 提取 { ... } 最外层 JSON
    """
    if not response_text:
        return None

    text = response_text.strip()

    # Strategy 1: direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strategy 2: extract ```json ... ``` block
    code_block_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if code_block_match:
        try:
            return json.loads(code_block_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Strategy 3: find outermost { ... }
    # Find the first '{' and last '}'
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass

    return None


# ═══════════════════════════════════════════════════════════════
# 单条评估
# ═══════════════════════════════════════════════════════════════

def evaluate_sample(
    client: OpenAI,
    model: str,
    eval_entry: dict,
    rag_entry: dict,
    max_retries: int = 3,
) -> dict:
    """对单条样本执行 LLM-as-Judge 评估。

    Args:
        client: OpenAI 客户端
        model: LLM 模型名
        eval_entry: 评测数据集中的一条样本
        rag_entry: RAG 输出中的对应条目
        max_retries: JSON 解析失败时的最大重试次数

    Returns:
        包含原始字段 + 评分字段的 dict
    """
    query = eval_entry["query"]
    answerable = eval_entry.get("answerable", True)
    reference_answer = eval_entry.get("reference_answer", "")
    expected_behavior = eval_entry.get("expected_behavior", "")
    generated_answer = rag_entry.get("generated_answer", "")
    retrieved_chunks = rag_entry.get("retrieved_chunks", [])
    if isinstance(retrieved_chunks, str):
        retrieved_chunks = []

    # 提取 evidence_units
    evidence_units = eval_entry.get("evidence_units", [])
    if not evidence_units:
        # fallback: 从 evidence 中提取 quote
        evidence_list = eval_entry.get("evidence", [])
        evidence_units = [ev.get("quote", "") for ev in evidence_list if ev.get("quote", "").strip()]

    # 构建 prompt
    if answerable:
        user_prompt = _build_answerable_judge_prompt(
            query=query,
            reference_answer=reference_answer,
            evidence_units=evidence_units,
            generated_answer=generated_answer,
            retrieved_chunks=retrieved_chunks,
            expected_behavior=expected_behavior,
        )
    else:
        user_prompt = _build_unanswerable_judge_prompt(
            query=query,
            reference_answer=reference_answer,
            generated_answer=generated_answer,
            retrieved_chunks=retrieved_chunks,
            expected_behavior=expected_behavior,
        )

    # 调用 LLM judge，支持重试
    last_error = None
    last_raw = ""
    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
                max_tokens=1024,
            )
            raw_output = response.choices[0].message.content.strip()
            last_raw = raw_output

            parsed = parse_judge_response(raw_output)
            if parsed is not None:
                # 验证必要字段
                result = {
                    "id": eval_entry["id"],
                    "query": query,
                    "intent": eval_entry.get("intent", ""),
                    "answerable": answerable,
                    "expected_behavior": expected_behavior,
                    "correctness": parsed.get("correctness"),
                    "completeness": parsed.get("completeness"),
                    "faithfulness": parsed.get("faithfulness"),
                    "abstention_correct": parsed.get("abstention_correct"),
                    "reasoning": parsed.get("reasoning", {}),
                    "_judge_raw": raw_output,
                }
                return result

        except Exception as e:
            last_error = str(e)

        if attempt < max_retries:
            time.sleep(1.0 * (attempt + 1))  # 递增等待

    # 所有重试都失败
    return {
        "id": eval_entry["id"],
        "query": query,
        "intent": eval_entry.get("intent", ""),
        "answerable": answerable,
        "expected_behavior": expected_behavior,
        "correctness": None,
        "completeness": None,
        "faithfulness": None,
        "abstention_correct": None,
        "reasoning": {},
        "_judge_raw": last_raw,
        "_parse_error": last_error or "Failed to parse JSON after retries",
    }


# ═══════════════════════════════════════════════════════════════
# 聚合统计
# ═══════════════════════════════════════════════════════════════

def aggregate_results(results: list[dict]) -> dict:
    """对评估结果进行聚合统计。

    Returns:
        {
            "overall": {metric: avg, ...},
            "by_intent": {intent: {metric: avg, ...}, ...},
            "abstention_accuracy": float,
            "total_samples": int,
            "answerable_count": int,
            "unanswerable_count": int,
        }
    """
    answerable_results = [r for r in results if r.get("answerable")]
    unanswerable_results = [r for r in results if not r.get("answerable")]

    # ── Overall averages (answerable only for correctness/completeness) ──
    def safe_mean(values: list) -> float | None:
        """计算均值，跳过 None。全为 None 则返回 None。"""
        valid = [v for v in values if v is not None]
        if not valid:
            return None
        return round(sum(valid) / len(valid), 4)

    def safe_accuracy(values: list) -> float | None:
        """计算 true 的比例，跳过 None。"""
        valid = [v for v in values if v is not None]
        if not valid:
            return None
        return round(sum(1 for v in valid if v is True) / len(valid), 4)

    overall = {}
    overall["correctness"] = safe_mean([r["correctness"] for r in answerable_results])
    overall["completeness"] = safe_mean([r["completeness"] for r in answerable_results])
    overall["faithfulness"] = safe_mean([r["faithfulness"] for r in results])

    # Abstention accuracy (unanswerable only)
    overall["abstention_accuracy"] = safe_accuracy(
        [r["abstention_correct"] for r in unanswerable_results]
    )

    # ── By intent ──
    by_intent = defaultdict(lambda: defaultdict(list))
    for r in results:
        intent = r.get("intent", "unknown")
        answerable = r.get("answerable", True)
        by_intent[intent]["faithfulness"].append(r["faithfulness"])
        if answerable:
            by_intent[intent]["correctness"].append(r["correctness"])
            by_intent[intent]["completeness"].append(r["completeness"])
        else:
            by_intent[intent]["abstention_correct"].append(r["abstention_correct"])

    intent_summary = {}
    for intent, metrics in sorted(by_intent.items()):
        entry = {
            "sample_count": len(metrics["faithfulness"]),
            "correctness": safe_mean(metrics.get("correctness", [])),
            "completeness": safe_mean(metrics.get("completeness", [])),
            "faithfulness": safe_mean(metrics["faithfulness"]),
            "abstention_accuracy": safe_accuracy(metrics.get("abstention_correct", [])),
        }
        intent_summary[intent] = entry

    return {
        "overall": overall,
        "by_intent": intent_summary,
        "total_samples": len(results),
        "answerable_count": len(answerable_results),
        "unanswerable_count": len(unanswerable_results),
    }


# ═══════════════════════════════════════════════════════════════
# 输出保存
# ═══════════════════════════════════════════════════════════════

def save_results_jsonl(results: list[dict], path: Path) -> None:
    """保存 per-sample 结果到 JSONL 文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def save_report_csv(aggregation: dict, results: list[dict], path: Path) -> None:
    """保存聚合报表 CSV。

    包含两个部分：
    1. 总体指标 + 各 intent 指标（纵向）
    2. 每条样本的分数明细
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    overall = aggregation["overall"]
    by_intent = aggregation["by_intent"]

    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)

        # ── Section 1: 聚合报表 ──
        writer.writerow(["=== 聚合指标 ==="])
        writer.writerow([])
        writer.writerow([
            "维度", "样本数",
            "correctness(avg)", "completeness(avg)", "faithfulness(avg)",
            "abstention_accuracy",
        ])

        # Overall row
        overall_correctness = overall.get("correctness")
        overall_completeness = overall.get("completeness")
        overall_faithfulness = overall.get("faithfulness")
        overall_abstention = overall.get("abstention_accuracy")

        writer.writerow([
            "Overall",
            aggregation["total_samples"],
            f"{overall_correctness:.4f}" if overall_correctness is not None else "N/A",
            f"{overall_completeness:.4f}" if overall_completeness is not None else "N/A",
            f"{overall_faithfulness:.4f}" if overall_faithfulness is not None else "N/A",
            f"{overall_abstention:.4f}" if overall_abstention is not None else "N/A",
        ])

        # Per-intent rows
        for intent, metrics in by_intent.items():
            writer.writerow([
                intent,
                metrics["sample_count"],
                f"{metrics['correctness']:.4f}" if metrics["correctness"] is not None else "N/A",
                f"{metrics['completeness']:.4f}" if metrics["completeness"] is not None else "N/A",
                f"{metrics['faithfulness']:.4f}" if metrics["faithfulness"] is not None else "N/A",
                f"{metrics['abstention_accuracy']:.4f}" if metrics["abstention_accuracy"] is not None else "N/A",
            ])

        writer.writerow([])
        writer.writerow([
            f"总样本数: {aggregation['total_samples']}",
            f"可回答: {aggregation['answerable_count']}",
            f"不可回答: {aggregation['unanswerable_count']}",
        ])

        # ── Section 2: Per-sample 明细 ──
        writer.writerow([])
        writer.writerow(["=== 逐样本明细 ==="])
        writer.writerow([])
        writer.writerow([
            "id", "query", "intent", "answerable", "expected_behavior",
            "correctness", "completeness", "faithfulness", "abstention_correct",
            "reasoning_correctness", "reasoning_completeness",
            "reasoning_faithfulness", "reasoning_abstention",
        ])

        for r in results:
            reasoning = r.get("reasoning", {})
            if not isinstance(reasoning, dict):
                reasoning = {}
            writer.writerow([
                r["id"],
                r["query"][:100],
                r.get("intent", ""),
                r.get("answerable", ""),
                r.get("expected_behavior", ""),
                r.get("correctness", ""),
                r.get("completeness", ""),
                r.get("faithfulness", ""),
                r.get("abstention_correct", ""),
                reasoning.get("correctness", ""),
                reasoning.get("completeness", ""),
                reasoning.get("faithfulness", ""),
                reasoning.get("abstention_correct", ""),
            ])


# ═══════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="OnboardRAG — 生成阶段 LLM-as-Judge 评估",
    )
    parser.add_argument(
        "--eval-dataset",
        type=str,
        default="data/eval/eval_queries_v2.jsonl",
        help="评测数据集 JSONL 文件路径",
    )
    parser.add_argument(
        "--rag-outputs",
        type=str,
        required=True,
        help="RAG 输出 JSONL 文件路径（包含 generated_answer, retrieved_chunks, citations）",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/eval",
        help="评估结果输出目录（默认: outputs/eval/）",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="LLM Judge 模型（默认使用 config 中的 JUDGE_MODEL，即 glm-5.1）",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="JSON 解析失败时的最大重试次数（默认: 3）",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="限制评估样本数量（调试用，默认评估全部）",
    )
    args = parser.parse_args()

    model = args.model or JUDGE_MODEL

    # ── 加载评测数据集 ──
    eval_path = Path(args.eval_dataset)
    if not eval_path.exists():
        print(f"错误: 评测数据集文件不存在: {eval_path}")
        return 1

    with open(eval_path, "r", encoding="utf-8") as f:
        eval_entries = [json.loads(line) for line in f if line.strip()]

    print(f"加载评测数据集: {len(eval_entries)} 条样本")

    # ── 加载 RAG 输出 ──
    rag_path = Path(args.rag_outputs)
    if not rag_path.exists():
        print(f"错误: RAG 输出文件不存在: {rag_path}")
        return 1

    with open(rag_path, "r", encoding="utf-8") as f:
        rag_entries = [json.loads(line) for line in f if line.strip()]

    print(f"加载 RAG 输出: {len(rag_entries)} 条样本")

    # ── 按 id 建立索引 ──
    rag_by_id: dict[str, dict] = {}
    for entry in rag_entries:
        rag_by_id[entry.get("id", "")] = entry

    # ── 匹配样本 ──
    matched_entries = []
    skipped = 0
    for eval_entry in eval_entries:
        eid = eval_entry.get("id", "")
        if eid in rag_by_id:
            matched_entries.append((eval_entry, rag_by_id[eid]))
        else:
            skipped += 1
            print(f"  警告: 样本 {eid} 在 RAG 输出中未找到，跳过")

    print(f"成功匹配: {len(matched_entries)} 条样本")
    if skipped > 0:
        print(f"跳过: {skipped} 条（RAG 输出中无对应记录）")

    if not matched_entries:
        print("错误: 没有可评估的样本")
        return 1

    # ── 限制评估数量 ──
    if args.limit:
        matched_entries = matched_entries[:args.limit]
        print(f"限制评估数量: {args.limit} 条")

    # ── 初始化 LLM 客户端 ──
    if not LLM_API_KEY:
        print("错误: LLM_API_KEY 未设置，请在 .env 中配置")
        return 1
    if not LLM_BASE_URL:
        print("错误: LLM_BASE_URL 未设置，请在 .env 中配置")
        return 1

    client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)
    print(f"LLM Judge 模型: {model}")
    print()

    # ── 逐条评估 ──
    results = []
    answerable_count = 0
    unanswerable_count = 0

    for i, (eval_entry, rag_entry) in enumerate(matched_entries, 1):
        eid = eval_entry.get("id", "?")
        answerable = eval_entry.get("answerable", True)
        label = "A" if answerable else "U"
        print(f"[{i:03d}/{len(matched_entries)}] [{label}] {eid} — {eval_entry['query'][:60]}...", end=" ", flush=True)

        result = evaluate_sample(
            client=client,
            model=model,
            eval_entry=eval_entry,
            rag_entry=rag_entry,
            max_retries=args.max_retries,
        )
        results.append(result)

        if answerable:
            answerable_count += 1
            c = result.get("correctness", "?")
            comp = result.get("completeness", "?")
            f = result.get("faithfulness", "?")
            print(f"correctness={c} completeness={comp} faithfulness={f}")
        else:
            unanswerable_count += 1
            f = result.get("faithfulness", "?")
            a = result.get("abstention_correct", "?")
            print(f"faithfulness={f} abstention_correct={a}")

        # 每 5 条短暂暂停，避免 API 限流
        if i % 5 == 0 and i < len(matched_entries):
            time.sleep(0.5)

    # ── 聚合统计 ──
    print("\n" + "=" * 60)
    print("  生成阶段评估结果汇总")
    print("=" * 60)

    aggregation = aggregate_results(results)
    overall = aggregation["overall"]

    print(f"\n  总样本数:     {aggregation['total_samples']}")
    print(f"  可回答样本:   {aggregation['answerable_count']}")
    print(f"  不可回答样本: {aggregation['unanswerable_count']}")
    print(f"\n  ── 总体指标 ──")
    print(f"  correctness (avg):         {overall['correctness']:.4f}" if overall['correctness'] is not None else "  correctness (avg):         N/A")
    print(f"  completeness (avg):        {overall['completeness']:.4f}" if overall['completeness'] is not None else "  completeness (avg):        N/A")
    print(f"  faithfulness (avg):        {overall['faithfulness']:.4f}" if overall['faithfulness'] is not None else "  faithfulness (avg):        N/A")
    print(f"  abstention_accuracy:       {overall['abstention_accuracy']:.4f}" if overall['abstention_accuracy'] is not None else "  abstention_accuracy:       N/A")

    # 按意图
    print(f"\n  ── 按意图分组 ──")
    print(f"  {'意图':<20} {'样本数':<8} {'correctness':<12} {'completeness':<12} {'faithfulness':<12} {'abstention_acc':<14}")
    for intent, metrics in aggregation["by_intent"].items():
        c_str = f"{metrics['correctness']:.4f}" if metrics["correctness"] is not None else "N/A"
        comp_str = f"{metrics['completeness']:.4f}" if metrics["completeness"] is not None else "N/A"
        f_str = f"{metrics['faithfulness']:.4f}" if metrics["faithfulness"] is not None else "N/A"
        a_str = f"{metrics['abstention_accuracy']:.4f}" if metrics["abstention_accuracy"] is not None else "N/A"
        print(f"  {intent:<20} {metrics['sample_count']:<8} {c_str:<12} {comp_str:<12} {f_str:<12} {a_str:<14}")

    # ── 保存结果 ──
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = output_dir / "generation_eval_results.jsonl"
    save_results_jsonl(results, jsonl_path)
    print(f"\n  Per-sample 结果 (JSONL): {jsonl_path}")

    csv_path = output_dir / "generation_eval_report.csv"
    save_report_csv(aggregation, results, csv_path)
    print(f"  聚合报表 (CSV):         {csv_path}")

    print("\n评估完成。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
