#!/usr/bin/env python3
"""Failure-type breakdown of the incorrect candidates.

Classifies each production candidate labeled T=0 by an AST check for the
required entry point plus the recorded execution outcome (test_reason) from the
generation harness:

  missing/renamed entry point  - candidate AST does not define the function the
                                 reference asserts call
  runs, wrong values (assertion failure)  - executed; failed on assertion values
  runtime crash or pre-execution guard block  - anything else (traceback, blocked
                                  AST guard, syntax error); the generated
                                  crash_or_guard_block_split reports how many of
                                  the bucket are genuine runtime crashes versus
                                  pre-execution guard blocks

Crash records are further grouped by exception class parsed from the recorded
traceback. Deterministic; reads ledgers only, no code execution, no network.

Output: data/mbpp_v2/analysis/failure_type_breakdown.json
        data/mbpp_v2/analysis/crash_exception_breakdown.json
"""
import ast
import re
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

VALUES = "runs, wrong values (assertion failure)"
CRASH = "runtime crash or pre-execution guard block"
MISSING = "missing/renamed entry point"
CATEGORIES = [VALUES, CRASH, MISSING]


def entry_of(test):
    """Function name called by the first reference assert."""
    for line in test.splitlines():
        s = line.strip()
        if s.startswith("assert "):
            return s[len("assert "):].split("(", 1)[0].strip()
    return None


def defines_entry(code, entry):
    if not entry:
        return True
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    return any(
        isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == entry
        for n in ast.walk(tree)
    )


def exception_of(test_reason):
    if test_reason.startswith("blocked "):
        return "Blocked AST guard"
    if test_reason.startswith("Traceback"):
        lines = [x.strip() for x in test_reason.splitlines() if x.strip()]
        for l in reversed(lines):
            m = re.match(r"^([A-Za-z_][\w.]*(?:Error|Exception|Interrupt))\b", l)
            if m:
                return m.group(1)
        if lines:
            tokens = re.findall(r"([A-Za-z_][\w.]*(?:Error|Exception|Interrupt))\b", lines[-1])
            if tokens:
                return tokens[0]
    return "Other exception"


def main() -> int:
    tasks = {
        t["task_id"]: t
        for t in (json.loads(l) for l in (ROOT / "data/MBPP.jsonl").read_text().splitlines() if l.strip())
    }
    incorrect = []
    for l in (ROOT / "data/mbpp_v2/production/candidate_records.jsonl").read_text().splitlines():
        if not l.strip():
            continue
        r = json.loads(l)
        if r.get("T") == 0:
            incorrect.append(r)

    per_model = Counter()          # model -> category (insertion order = first occurrence)
    model_counts = {}
    exception_counts = Counter()
    crash_total = 0

    for r in incorrect:
        model_counts[r["model"]] = None
        if not defines_entry(r["code"], entry_of(tasks[r["task_id"]]["test"])):
            cat = MISSING
        elif r.get("test_reason", "").startswith("assertion failed"):
            cat = VALUES
        else:
            cat = CRASH
            crash_total += 1
            exception_counts[exception_of(r.get("test_reason", ""))] += 1
        key = (r["model"], cat)
        per_model[key] += 1

    breakdown = Counter({VALUES: 0, CRASH: 0, MISSING: 0})
    for (m, cat), n in per_model.items():
        breakdown[cat] += n

    out = ROOT / "data/mbpp_v2/analysis"
    out.mkdir(parents=True, exist_ok=True)

    per_model_out = {}
    for m in model_counts:
        counts = {cat: per_model[(m, cat)] for cat in CATEGORIES if per_model[(m, cat)]}
        per_model_out[m] = counts

    (out / "failure_type_breakdown.json").write_text(json.dumps({
        "total_incorrect": len(incorrect),
        "breakdown": {cat: breakdown[cat] for cat in CATEGORIES},
        "per_model": per_model_out,
        "method": "AST check for required entry-point definition + test_reason classification (crash-or-guard-block vs assertion)",
        "crash_or_guard_block_split": {
            "runtime_crash": crash_total - exception_counts.get("Blocked AST guard", 0),
            "pre_execution_guard_block": exception_counts.get("Blocked AST guard", 0),
        },
    }, indent=1))
    (out / "crash_exception_breakdown.json").write_text(json.dumps({
        "total_crash_or_guard_block": crash_total,
        "by_exception": dict(exception_counts),
        "method": "test_reason string classification of the crash-or-guard-block bucket; see crash_or_guard_block_split in failure_type_breakdown.json for the runtime-crash vs pre-execution-guard-block split",
    }, indent=1))

    print(f"incorrect candidates: {len(incorrect)}")
    print(json.dumps({cat: breakdown[cat] for cat in CATEGORIES}))
    print("crash_or_guard_block:", crash_total, "|", dict(exception_counts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
