#!/usr/bin/env python3
"""Verify that all 947 reference solutions pass the production MBPP harness.

Each reference solution (data/MBPP.reference-solutions.jsonl: the ``code``
field of the canonical Austin et al. 2021 MBPP release, joined to the frozen
947-task set on task_id) is run through the exact production harness path
(ast_guard + subprocess execution + PASS-string check, 15 s timeout),
imported from mbpp_v2_pilot.py so this check can never drift from the
production code path.

Writes data/mbpp_v2/analysis/reference_harness_verification.json.
Expected result: 947/947 pass. A failure would mean the harness itself
mislabels a known-good solution, contaminating every T=0 label in the
frozen candidate ledger.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mbpp_v2_pilot import run_task_tests  # noqa: E402  (production harness)

ROOT = Path(__file__).resolve().parents[1]
TASKS_PATH = ROOT / "data" / "MBPP.jsonl"
REF_PATH = ROOT / "data" / "MBPP.reference-solutions.jsonl"
OUT_PATH = ROOT / "data" / "mbpp_v2" / "analysis" / "reference_harness_verification.json"


def main() -> int:
    tasks = {}
    with open(TASKS_PATH, encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            tasks[row["task_id"]] = row

    failures = []
    n = 0
    with open(REF_PATH, encoding="utf-8") as fh:
        for line in fh:
            ref = json.loads(line)
            task = tasks[ref["task_id"]]
            ok, detail = run_task_tests(ref["code"], task)
            n += 1
            if not ok:
                failures.append({"task_id": ref["task_id"], "detail": detail})

    result = {
        "artifact": "reference_harness_verification",
        "tasks_run": n,
        "passed": n - len(failures),
        "failed": len(failures),
        "failures": failures,
        "harness": ("scripts/mbpp_v2_pilot.py::run_task_tests "
                    "(ast_guard, 15 s timeout, PASS-string check)"),
        "inputs": {
            "tasks": "data/MBPP.jsonl (frozen 947-task set)",
            "reference_solutions": "data/MBPP.reference-solutions.jsonl",
        },
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("tasks_run", "passed", "failed")}))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
