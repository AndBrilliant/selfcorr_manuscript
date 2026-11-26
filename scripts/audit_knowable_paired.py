#!/usr/bin/env python3
"""Headline numbers of the knowability audit that are not in followup.json.

Offline; no network; executes no candidate code (the arity check is static, via ast).
Inputs: data/mbpp_v2/crossmodel_audit_all/labels.jsonl, production/derived_records.jsonl,
production/candidate_records.jsonl, cross_eval/cross_records.jsonl,
confidence_eval/records.jsonl, data/MBPP.jsonl.

On agreed-knowable errors (both auditors label KNOWABLE):
  * author (the generating family's own evaluator, production verdict) false acceptance,
    and the per-candidate mean false acceptance of the other four families, the paired
    difference with a task-clustered bootstrap 95% interval (2,000 resamples, seed
    20261124), and an exact two-sided sign test on candidates the author accepts and most
    others reject versus the reverse ("most" = more than half of the parsed other-family
    verdicts); for self-check and steelman.
On agreed-not-knowable errors:
  * the share where the reference tests call the entry point with an argument count its
    signature cannot accept (static check);
  * in the confidence-rating control (own family, self-check arm), the share rated 6 or 7
    ("likely or certainly correct") and the share rated 4 or less ("doubt").
Output: data/mbpp_v2/analysis/audit_knowable_paired.json
"""
import ast
import json
import random
from collections import defaultdict
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data/mbpp_v2"
OK = ("KNOWABLE", "NOT_KNOWABLE")


def jl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def classes():
    labs = defaultdict(dict)
    for r in jl(D / "crossmodel_audit_all/labels.jsonl"):
        labs[(r["generator"], r["task_id"])][r["auditor"]] = r["label"]
    auditors = sorted({a for v in labs.values() for a in v})
    cls = {}
    for k, v in labs.items():
        ls = [v.get(a) for a in auditors]
        if all(l in OK for l in ls):
            cls[k] = ("AGREED_" + ls[0]) if len(set(ls)) == 1 else "SPLIT"
        else:
            cls[k] = "NO_CLASS"
    return cls


def sign_p(b, c):
    n = b + c
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0


def paired(mem, der, cross, arm, seed=20261124, B=2000):
    by = defaultdict(list)
    for r in cross:
        if r.get(f"{arm}_verdict") in ("ACCEPT", "REJECT"):
            by[(r["generator"], r["task_id"])].append(r[f"{arm}_verdict"] == "ACCEPT")
    rows = []
    for k in sorted(mem):
        d = der.get(k)
        if not d or d.get(f"{arm}_extraction") != "parsed" or d.get(f"{arm}_verdict") not in ("ACCEPT", "REJECT"):
            continue
        if not by.get(k):
            continue
        own = d[f"{arm}_verdict"] == "ACCEPT"
        oth = sum(by[k]) / len(by[k])
        rows.append((k[1], own, oth))
    n = len(rows)
    own_r = sum(r[1] for r in rows) / n
    oth_r = sum(r[2] for r in rows) / n
    b = sum(1 for r in rows if r[1] and r[2] < 0.5)
    c = sum(1 for r in rows if (not r[1]) and r[2] > 0.5)
    tasks = defaultdict(list)
    for r in rows:
        tasks[r[0]].append(r)
    tk = sorted(tasks)
    rng = random.Random(seed)
    diffs = []
    for _ in range(B):
        s = [x for t in (rng.choice(tk) for _ in tk) for x in tasks[t]]
        diffs.append(100 * (sum(x[1] for x in s) - sum(x[2] for x in s)) / len(s))
    diffs.sort()
    return {"n": n, "author_fa_pct": round(100 * own_r, 1), "others_mean_fa_pct": round(100 * oth_r, 1),
            "diff_pts": round(100 * (own_r - oth_r), 1),
            "diff_ci95": [round(diffs[int(0.025 * B)], 1), round(diffs[int(0.975 * B) - 1], 1)],
            "author_accepts_most_others_reject": b, "reverse": c, "sign_test_p": sign_p(b, c)}


def entry_and_calls(test, name_hint=None):
    calls = []
    for line in test.splitlines():
        line = line.strip()
        if not line.startswith("assert"):
            continue
        try:
            tree = ast.parse(line)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                calls.append((node.func.id, len(node.args), len(node.keywords)))
                break
    return calls


def arity_mismatch(code, test):
    try:
        mod = ast.parse(code)
    except SyntaxError:
        return None
    defs = {n.name: n for n in mod.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    calls = entry_and_calls(test)
    if not calls:
        return None
    name = calls[0][0]
    f = defs.get(name)
    if f is None:
        return None
    a = f.args
    pos = a.posonlyargs + a.args
    req = len(pos) - len(a.defaults)
    mx = None if a.vararg else len(pos)
    for (nm, npos, nkw) in calls:
        if nm != name:
            continue
        if npos + nkw < req and not a.kwarg or (mx is not None and npos > mx):
            return True
    return False


def main():
    cls = classes()
    der = {(r["model"], r["task_id"]): r for r in jl(D / "production/derived_records.jsonl")}
    cross = jl(D / "cross_eval/cross_records.jsonl")
    kn = {k for k, v in cls.items() if v == "AGREED_KNOWABLE"}
    nk = {k for k, v in cls.items() if v == "AGREED_NOT_KNOWABLE"}
    out = {"class_counts": {c: sum(v == c for v in cls.values()) for c in set(cls.values())},
           "knowable_paired": {arm: paired(kn, der, cross, arm) for arm in ("selfcheck", "steelman")}}

    tasks = {json.loads(l)["task_id"]: json.loads(l) for l in open(ROOT / "data/MBPP.jsonl") if l.strip()}
    code = {(c["model"], c["task_id"]): c.get("code") or "" for c in jl(D / "production/candidate_records.jsonl")}
    mm = [arity_mismatch(code.get(k, ""), tasks[k[1]]["test"]) for k in sorted(nk)]
    out["not_knowable_arity_mismatch"] = {"n": len(nk), "mismatch": sum(m is True for m in mm),
                                          "pct": round(100 * sum(m is True for m in mm) / len(nk), 1)}

    conf = [r for r in jl(D / "confidence_eval/records.jsonl")
            if r.get("arm") == "selfcheck" and (r["model"], r["task_id"]) in nk
            and isinstance(r.get("confidence"), int)]
    n = len(conf)
    out["not_knowable_confidence_selfcheck"] = {
        "n": n, "rated_6_or_7_pct": round(100 * sum(r["confidence"] >= 6 for r in conf) / n, 1),
        "rated_4_or_less_pct": round(100 * sum(r["confidence"] <= 4 for r in conf) / n, 1)}
    p = D / "analysis/audit_knowable_paired.json"
    p.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
