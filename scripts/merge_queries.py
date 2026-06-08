#!/usr/bin/env python3
"""将 eval_queries_v2.jsonl 的所有 query 合并成一个 JSON 数组文件。"""

import json
from pathlib import Path

INPUT = Path(__file__).resolve().parent.parent / "data" / "eval" / "eval_queries_v2.jsonl"
OUTPUT = Path(__file__).resolve().parent.parent / "data" / "eval" / "eval_queries_v2.json"


def main():
    queries = []
    with open(INPUT, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            queries.append(json.loads(line)["query"])

    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(queries, f, ensure_ascii=False, indent=2)

    print(f"已合并 {len(queries)} 条 query → {OUTPUT}")


if __name__ == "__main__":
    main()
