#!/usr/bin/env python3
"""Failure-type-stratified false acceptance and arm contrast (production).

Classifies every production candidate labeled T=0 with the identical rules as
scripts/mbpp_v2_failure_types.py (AST entry-point check plus recorded
test_reason), then restricts to the paired stratum (both arms parsed) and
reports, per failure category:

  n_incorrect              paired incorrect candidates in the category
  selfcheck_fa / steelman_fa   false-acceptance counts and Wilson 95% intervals
  discordant               self-check-only / steelman-only accepts (kappa cells)
  mcnemar_p                exact two-sided McNemar p for the discordant pair

Deterministic; reads ledgers only, no code execution, no network.

Output: data/mbpp_v2/analysis/failure_type_fa.json
"""
import ast
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


def category_of(task, r):
    if not defines_entry(r.get("code") or "", entry_of(task["test"])):
        return MISSING
    if (r.get("test_reason") or "").startswith("assertion failed"):
        return VALUES
    return CRASH


def wilson(count, n, z=1.959963985):
    """95% Wilson score interval as (lo, hi) proportions."""
    if n == 0:
        return (0.0, 0.0)
    p = count / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def mcnemar_exact_p(a, b):
    from scipy.stats import binomtest
    n = a + b
    if n == 0:
        return 1.0
    # binomtest's default alternative is two-sided; its p-value is already the exact
    # two-sided McNemar value and must not be doubled.
    return min(1.0, binomtest(min(a, b), n, 0.5).pvalue)


def main() -> int:
    tasks = {
        t["task_id"]: t
        for t in (json.loads(l) for l in (ROOT / "data/MBPP.jsonl").read_text().splitlines() if l.strip())
    }
    code_by = {}
    for l in (ROOT / "data/mbpp_v2/production/candidate_records.jsonl").read_text().splitlines():
        if not l.strip():
            continue
        r = json.loads(l)
        code_by[(r["model"], r["task_id"])] = r.get("code") or ""

    cat_by = {}
    for l in (ROOT / "data/mbpp_v2/production/derived_records.jsonl").read_text().splitlines():
        if not l.strip():
            continue
        r = json.loads(l)
        if r.get("T") != 0:
            continue
        probe = {"code": code_by.get((r["model"], r["task_id"]), ""),
                 "test_reason": r.get("test_reason") or ""}
        cat_by[(r["model"], r["task_id"])] = category_of(tasks[r["task_id"]], probe)

    strata = {cat: {"n": 0, "sc_fa": 0, "st_fa": 0, "sc_only": 0, "st_only": 0}
              for cat in CATEGORIES}
    for l in (ROOT / "data/mbpp_v2/production/derived_records.jsonl").read_text().splitlines():
        if not l.strip():
            continue
        r = json.loads(l)
        key = (r["model"], r["task_id"])
        if r.get("T") != 0 or key not in cat_by:
            continue
        if r.get("selfcheck_verdict") not in ("ACCEPT", "REJECT") or \
                r.get("steelman_verdict") not in ("ACCEPT", "REJECT"):
            continue
        s = strata[cat_by[key]]
        s["n"] += 1
        if r["selfcheck_verdict"] == "ACCEPT":
            s["sc_fa"] += 1
        if r["steelman_verdict"] == "ACCEPT":
            s["st_fa"] += 1
        if r["selfcheck_verdict"] == "ACCEPT" and r["steelman_verdict"] == "REJECT":
            s["sc_only"] += 1
        if r["selfcheck_verdict"] == "REJECT" and r["steelman_verdict"] == "ACCEPT":
            s["st_only"] += 1

    out = {}
    for cat in CATEGORIES:
        s = strata[cat]
        n = s["n"]
        lo_sc, hi_sc = wilson(s["sc_fa"], n)
        lo_st, hi_st = wilson(s["st_fa"], n)
        out[cat] = {
            "n_incorrect": n,
            "selfcheck_fa": s["sc_fa"],
            "steelman_fa": s["st_fa"],
            "selfcheck_fa_wilson95": [round(100 * lo_sc, 1), round(100 * hi_sc, 1)],
            "steelman_fa_wilson95": [round(100 * lo_st, 1), round(100 * hi_st, 1)],
            "discordant_selfcheck_only": s["sc_only"],
            "discordant_steelman_only": s["st_only"],
            "mcnemar_exact_p": mcnemar_exact_p(s["sc_only"], s["st_only"]),
        }

    dest = ROOT / "data/mbpp_v2/analysis/failure_type_fa.json"
    dest.write_text(json.dumps({
        "schema": "mbpp-v2-failure-type-fa/1",
        "scope": "production paired stratum (both arms parsed, ground truth assigned); "
                 "categories use the identical classification as failure_type_breakdown.json",
        "categories": out,
    }, indent=2) + "\n")
    print(dest)
    for cat in CATEGORIES:
        c = out[cat]
        print(cat, c["n_incorrect"],
              f"SC {c['selfcheck_fa']}/{c['n_incorrect']} {c['selfcheck_fa_wilson95']}",
              f"ST {c['steelman_fa']}/{c['n_incorrect']} {c['steelman_fa_wilson95']}",
              f"kappa {c['discordant_selfcheck_only']}/{c['discordant_steelman_only']} p={c['mcnemar_exact_p']:.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
