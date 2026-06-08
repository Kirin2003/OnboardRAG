#!/usr/bin/env python3
"""
migrate_evidence_units.py — 将评估集中的 evidence 字段转换为 evidence_units。

对每条 answerable=True 且有 evidence 的样本：
  - 短 evidence（无编号条款）：evidence_units = [evidence_quote]
  - 长 evidence（包含编号条款如 4.1 / 4.2 / 1) / 2) 等）：按编号拆分为多个 units
  - 已有 evidence_units 的样本：保持不变

输出新文件，不覆盖原始评估集。

用法:
    python scripts/migrate_evidence_units.py

    # 指定输入输出文件
    python scripts/migrate_evidence_units.py --input data/eval/eval_queries.jsonl --output data/eval/eval_queries_v2.jsonl

    # 仅预览（不写入文件）
    python scripts/migrate_evidence_units.py --dry-run
"""

import argparse
import json
import re
import sys
from pathlib import Path

# 将项目根目录加入 sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ═══════════════════════════════════════════════════════════════
# 拆分逻辑
# ═══════════════════════════════════════════════════════════════

# 匹配编号前缀的模式
# 如: "4.1", "4.1.", "1)", "1）", "01", "一、", "（1）", "(1)", "①"
_CLAUSE_PATTERN = re.compile(
    r'(?:^|(?<=[。；;])\s*)'                          # 句首或分句边界
    r'('
    r'\d+\.\d+(?:\.\d+)*[\.、]?\s*'                  # 4.1 / 4.1.1 / 4.1、
    r'|\d+[\)）]\s*'                                   # 1) / 1）
    r'|（\d+）\s*'                                     # （1）
    r'|\(\d+\)\s*'                                    # (1)
    r'|[①②③④⑤⑥⑦⑧⑨⑩]\s*'                             # ①
    r'|[一二三四五六七八九十]+[、．.]\s*'                 # 一、
    r')',
    re.MULTILINE,
)


def _has_numbered_clauses(text: str) -> bool:
    """检查文本是否包含明显的编号条款。"""
    # 至少匹配 3 个编号才算"长条款"
    matches = _CLAUSE_PATTERN.findall(text)
    return len(matches) >= 3


def _split_by_clauses(text: str) -> list[str]:
    """按编号条款拆分文本为多个 evidence units。

    策略：
    1. 找到所有编号位置
    2. 将文本按编号边界切分
    3. 每个切分段去掉首尾空白，过滤空段
    4. 第一段（编号前的内容）作为独立 unit 或者跳过
    """
    matches = list(_CLAUSE_PATTERN.finditer(text))

    if len(matches) < 2:
        return [text]

    units = []

    # 第一个编号之前的内容
    first_match_start = matches[0].start()
    if first_match_start > 0:
        prefix = text[:first_match_start].strip()
        if prefix and len(prefix) > 5:
            units.append(prefix)

    # 按编号边界切分
    for i, m in enumerate(matches):
        start = m.end()  # 编号之后
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        clause = text[start:end].strip().rstrip("。；;，,")
        if clause and len(clause) > 3:
            # 保留编号前缀以便识别
            full_clause = (m.group() + clause).strip()
            units.append(full_clause)

    # 如果拆分结果太少（例如编号不标准），回退到原始文本
    if len(units) < 2:
        return [text]

    return units


def migrate_entry(entry: dict) -> dict:
    """对单条评测样本进行迁移。

    - 已有 evidence_units 的样本保持不变
    - 没有 evidence_units 但有 evidence 的样本自动生成
    - answerable=False 或无 evidence 的样本保持不变

    Returns:
        迁移后的 entry（浅拷贝）
    """
    entry = dict(entry)  # 浅拷贝

    # 已有 evidence_units，保持不变
    if "evidence_units" in entry and entry["evidence_units"]:
        return entry

    # 无 evidence，保持不变
    evidence_list = entry.get("evidence", [])
    if not evidence_list:
        return entry

    # 从 evidence 中提取 quotes 并生成 evidence_units
    all_units = []
    for ev in evidence_list:
        quote = ev.get("quote", "").strip()
        if not quote:
            continue

        if _has_numbered_clauses(quote):
            # 长条款：按编号拆分
            sub_units = _split_by_clauses(quote)
            all_units.extend(sub_units)
        else:
            # 短条款：整个作为 unit
            all_units.append(quote)

    entry["evidence_units"] = all_units
    return entry


# ═══════════════════════════════════════════════════════════════
# 统计信息
# ═══════════════════════════════════════════════════════════════

def print_stats(original: list[dict], migrated: list[dict]) -> None:
    """打印迁移统计信息。"""
    print("\n" + "=" * 60)
    print("  迁移统计")
    print("=" * 60)

    total = len(original)
    changed = 0
    total_units_before = 0
    total_units_after = 0

    for orig, mig in zip(original, migrated):
        old_units = len(orig.get("evidence", []))
        new_units = len(mig.get("evidence_units", []))
        total_units_before += old_units
        total_units_after += new_units
        if old_units != new_units:
            changed += 1

    print(f"  总样本数: {total}")
    print(f"  Unit 数变化的样本数: {changed}")
    print(f"  原始 evidence 条数: {total_units_before}")
    print(f"  生成 evidence_units 条数: {total_units_after}")

    # 列出变化较大的样本
    if changed > 0:
        print(f"\n  变化超过 5 个 units 的样本:")
        for orig, mig in zip(original, migrated):
            old_units = len(orig.get("evidence", []))
            new_units = len(mig.get("evidence_units", []))
            if new_units - old_units >= 5:
                print(f"    {orig['id']}: {old_units} → {new_units} units | {orig['query'][:60]}")


# ═══════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="将评估集中的 evidence 迁移为 evidence_units 格式"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default="data/eval/eval_queries.jsonl",
        help="输入评估集 JSONL 文件",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="输出文件路径（默认: 在输入文件名后加 _v2）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅预览，不写入文件",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"输入文件不存在: {input_path}")
        return 1

    # 加载原始评估集
    with open(input_path, "r", encoding="utf-8") as f:
        original = [json.loads(line) for line in f if line.strip()]

    # 迁移
    migrated = [migrate_entry(entry) for entry in original]

    # 打印统计
    print_stats(original, migrated)

    # 确定输出路径
    if args.output:
        output_path = Path(args.output)
    else:
        stem = input_path.stem
        output_path = input_path.parent / f"{stem}_v2.jsonl"

    if args.dry_run:
        print(f"\n  [DRY RUN] 将输出到: {output_path}")
        print(f"  [DRY RUN] 预览前 3 条有 evidence_units 的样本:")
        for entry in migrated:
            if entry.get("evidence_units"):
                print(f"\n  ── {entry['id']}: {entry['query'][:60]}")
                eu = entry["evidence_units"]
                print(f"    evidence_units ({len(eu)} 条):")
                for j, u in enumerate(eu[:5]):
                    print(f"      [{j}] {u[:120]}")
                if len(eu) > 5:
                    print(f"      ... (共 {len(eu)} 条)")
                if len([e for e in migrated if e.get("evidence_units")]) > 3:
                    break
        return 0

    # 写入输出文件
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in migrated:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"\n  迁移完成！输出文件: {output_path}")
    print(f"  原始文件保持不变: {input_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
