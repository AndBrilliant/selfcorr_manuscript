#!/usr/bin/env python3
"""Candidate replication: how often a regenerated candidate is identical to the production one,
and how often its correctness label flips (paper, Section IV-D).

Compares data/mbpp_v2/production/candidate_records.jsonl with
data/mbpp_v2/replication/candidate_records.jsonl on (model, task_id) pairs present in both.
"Identical" means the extracted code is byte-identical (same code_sha256).

Usage: python3 scripts/mbpp_v2_replication_identity.py
Writes: data/mbpp_v2/tier1_analysis/replication_identity.json
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data" / "mbpp_v2"


def load(path: Path) -> dict:
    out = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            out[(rec["model"], rec["task_id"])] = rec
    return out


def main() -> int:
    prod = load(BASE / "production" / "candidate_records.jsonl")
    repl = load(BASE / "replication" / "candidate_records.jsonl")
    both = sorted(k for k in prod if k in repl)

    def summarize(keys):
        n = len(keys)
        ident = sum(prod[k]["code_sha256"] == repl[k]["code_sha256"] for k in keys)
        flips = sum(prod[k]["T"] != repl[k]["T"] for k in keys)
        return {"pairs": n, "identical": ident, "identical_pct": round(100 * ident / n, 1),
                "label_flips": flips, "label_flips_pct": round(100 * flips / n, 1)}

    result = {"overall": summarize(both), "by_model": {}}
    for model in sorted({m for m, _ in both}):
        result["by_model"][model] = summarize([k for k in both if k[0] == model])

    out = BASE / "tier1_analysis" / "replication_identity.json"
    out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    o = result["overall"]
    print(f"pairs {o['pairs']}: identical {o['identical']} ({o['identical_pct']}%), "
          f"label flips {o['label_flips']} ({o['label_flips_pct']}%)")
    for m, r in result["by_model"].items():
        print(f"  {m}: identical {r['identical_pct']}%, flips {r['label_flips_pct']}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
