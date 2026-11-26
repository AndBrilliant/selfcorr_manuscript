#!/usr/bin/env python3
"""Sensitivity of Table III to verdicts obtained after a truncated attempt (paper, Section IV-G,
"Retries under the output cap").

Under the early output cap, some canonical evaluator verdicts were obtained after an earlier
attempt for the same (model, task, stage) had been truncated at the cap (finish_reason ==
"length" at exactly 8,192 or 16,384 completion tokens). This script recomputes paired
false acceptance on incorrect candidates (T=0, both arms parsed) three ways:
  as_reported  - Table III as published;
  dropped      - those verdicts removed;
  swapped      - those verdicts replaced by the same evaluator's uncapped round-2 verdict
                 from the repeated-evaluation run (data/mbpp_v2/repeated_eval).

Usage: python3 scripts/mbpp_v2_cap_sensitivity.py
Writes: data/mbpp_v2/analysis/cap_sensitivity.json
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data" / "mbpp_v2"
OUT = BASE / "analysis" / "cap_sensitivity.json"
CAPS = (8192, 16384)
VALID = ("ACCEPT", "REJECT")


def jsonl(path: Path):
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            yield json.loads(line)


def main() -> int:
    events = list(jsonl(BASE / "production" / "events.jsonl"))
    by_id = {e["event_id"]: e for e in events}
    by_key = defaultdict(list)
    for e in events:
        by_key[(e["model"], e["task_id"], e["stage"])].append(e)
    derived = list(jsonl(BASE / "production" / "derived_records.jsonl"))
    repeated = {(r["model"], r["task_id"]): r for r in jsonl(BASE / "repeated_eval" / "repeated_records.jsonl")}

    def is_cap(e):
        return e["finish_reason"] == "length" and (e.get("usage") or {}).get("completion_tokens") in CAPS

    def after_cap(d, stage):
        rid = d.get(stage + "_event_id")
        if not rid:
            return False
        ref = by_id[rid]
        return any(is_cap(x) and x["ts"] <= ref["ts"] and x["event_id"] != rid
                   for x in by_key[(d["model"], d["task_id"], stage)])

    result = {}
    affected_total = 0
    for model in sorted({d["model"] for d in derived}):
        rows = [d for d in derived if d["model"] == model and d["T"] == 0
                and d.get("selfcheck_verdict") in VALID and d.get("steelman_verdict") in VALID]
        result[model] = {}
        for stage in ("selfcheck", "steelman"):
            out = {}
            for mode in ("as_reported", "dropped", "swapped"):
                n = acc = 0
                for d in rows:
                    v = d[stage + "_verdict"]
                    if after_cap(d, stage):
                        if mode == "dropped":
                            continue
                        if mode == "swapped":
                            r = repeated.get((model, d["task_id"]))
                            rv = r and r.get(f"{stage}_r2_verdict")
                            if rv in VALID:
                                v = rv
                    n += 1
                    acc += v == "ACCEPT"
                out[mode] = {"accept": acc, "n": n, "false_acceptance_pct": round(100 * acc / n, 1)}
            affected = sum(1 for d in rows if after_cap(d, stage))
            affected_total += affected
            out["affected_verdicts"] = affected
            result[model][stage] = out
    result["affected_verdicts_total"] = affected_total
    OUT.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    for model, arms in result.items():
        if isinstance(arms, dict):
            for stage, o in arms.items():
                print(f"{model:15s} {stage:9s} " + "  ".join(
                    f"{m} {o[m]['false_acceptance_pct']:5.1f}%" for m in ("as_reported", "dropped", "swapped"))
                    + f"  affected {o['affected_verdicts']}")
    print("affected total", affected_total)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
