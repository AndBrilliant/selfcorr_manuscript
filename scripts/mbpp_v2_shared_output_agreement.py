#!/usr/bin/env python3
"""Output agreement for the shared-blind-spot test.

Refines scripts/mbpp_v2_shared_blindspot.py: among cross rows where evaluator E's own
candidate on the task is also incorrect, splits by whether E's and generator G's incorrect
programs return the same values on every reference-assert input ("same output"), i.e.
whether E's own code embodies the same reading as G's. For each assert of the form
`assert <call> == <expected>` the left-hand call is evaluated (otherwise the whole assert
expression); a raised exception is recorded by its type, a timeout as TIMEOUT.

Reports self-check and steelman false acceptance for three groups (own code right; own
code wrong with the same outputs; own code wrong with different outputs) and
Mantel-Haenszel odds ratios against "own code right", stratified by (E, G, difficulty) as
in the primary test, with task-clustered bootstrap intervals (2000 resamples, seed
20261123). Computed after data collection, as a follow-up to the primary test; it
executes the frozen candidates in subprocesses with a 10 s timeout and makes no network
calls.

Outputs: data/mbpp_v2/analysis/shared_output_agreement.json, *_summary.txt
"""
import ast
import json
import multiprocessing as mp
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from mbpp_v2_error_severity_audit import split_parts  # noqa: E402
from mbpp_v2_shared_blindspot import mh, wilson, FAMS  # noqa: E402

D = ROOT / "data/mbpp_v2"
OUT = D / "analysis"
B, SEED = 2000, 20261123
SAFE = "import sys, json\nsys.setrecursionlimit(10000)\n"


def lhs(a):
    try:
        node = ast.parse(a.strip()).body[0]
        t = node.test
        if isinstance(t, ast.Compare) and len(t.ops) == 1 and isinstance(t.ops[0], ast.Eq):
            return ast.unparse(t.left)
        return ast.unparse(t)
    except Exception:
        return None


def run(arg):
    code, setup, exprs = arg
    body = ["__o = []"]
    for e in exprs:
        if e is None:
            body.append("__o.append('UNPARSED')")
            continue
        body += ["try:", f"    __o.append(repr({e}))", "except BaseException as __x:",
                 "    __o.append('EXC:' + type(__x).__name__)"]
    body.append("print('__OUT__' + json.dumps(__o))")
    src = SAFE + code + "\n\n" + setup + "\n\n" + "\n".join(body) + "\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
        fh.write(src)
        p = fh.name
    try:
        r = subprocess.run([sys.executable, p], capture_output=True, timeout=10, text=True)
        for line in r.stdout.splitlines()[::-1]:
            if line.startswith("__OUT__"):
                return tuple(json.loads(line[7:]))
        return ("CRASH",)
    except subprocess.TimeoutExpired:
        return ("TIMEOUT",)
    finally:
        Path(p).unlink(missing_ok=True)


def main():
    tasks = {json.loads(l)["task_id"]: json.loads(l) for l in open(ROOT / "data/MBPP.jsonl") if l.strip()}
    cands = [json.loads(l) for l in open(D / "production/candidate_records.jsonl") if l.strip()]
    bad = [c for c in cands if c.get("T") == 0 and c.get("code")]
    jobs, keys = [], []
    for c in bad:
        setup, asserts = split_parts(tasks[c["task_id"]]["test"])
        jobs.append((c["code"], setup, [lhs(a) for a in asserts]))
        keys.append((c["model"], c["task_id"]))
    with mp.Pool(16) as pool:
        sig = dict(zip(keys, pool.map(run, jobs, chunksize=4)))

    T = {(r["model"], r["task_id"]): r["T"] for r in map(json.loads, open(D / "production/derived_records.jsonl"))}
    rows, drops = [], Counter()
    for r in map(json.loads, open(D / "cross_eval/cross_records.jsonl")):
        E, G, t = r["evaluator"], r["generator"], r["task_id"]
        tg, te = T.get((G, t)), T.get((E, t))
        if tg != 0 or te is None:
            continue
        others = [T.get((f, t)) for f in FAMS if f not in (E, G)]
        if None in others:
            continue
        if te == 0:
            se, sg = sig.get((E, t)), sig.get((G, t))
            if se is None or sg is None:
                drops["no_signature"] += 1
                continue
            grp = "same_output" if se == sg else "different_output"
        else:
            grp = "own_right"
        rows.append(dict(E=E, G=G, t=t, diff=sum(x == 0 for x in others), grp=grp,
                         sc=r.get("selfcheck_verdict"), st=r.get("steelman_verdict")))

    tasks_ = sorted({r["t"] for r in rows}); tix = {t: i for i, t in enumerate(tasks_)}
    strata = [(E, G, k) for E in FAMS for G in FAMS if E != G for k in range(4)]
    six = {s: i for i, s in enumerate(strata)}
    rng = np.random.default_rng(SEED)
    W = np.stack([np.bincount(rng.integers(0, len(tasks_), len(tasks_)), minlength=len(tasks_))
                  for _ in range(B)]).astype(float)
    res = {"n_signatures": len(sig), "drops": dict(drops)}
    L = ["Output agreement within the shared-blind-spot test (MBPP cross-evaluation)", ""]
    for arm, key in (("selfcheck", "sc"), ("steelman", "st")):
        ar = [r for r in rows if r[key] in ("ACCEPT", "REJECT")]
        o = {}
        for g in ("own_right", "same_output", "different_output"):
            s = [r for r in ar if r["grp"] == g]; k = sum(r[key] == "ACCEPT" for r in s)
            o[g] = {"n": len(s), "fa_pct": round(100 * k / len(s), 1) if s else None, "wilson95": wilson(k, len(s))}
        for g in ("same_output", "different_output"):
            X = np.zeros((len(tasks_), len(strata), 4))
            for r in ar:
                if r["grp"] not in (g, "own_right"):
                    continue
                fa = r[key] == "ACCEPT"
                j = (0 if fa else 1) if r["grp"] == g else (2 if fa else 3)
                X[tix[r["t"]], six[(r["E"], r["G"], r["diff"])], j] += 1
            pt = float(mh(X.sum(0)[None])[0])
            bs = mh(np.einsum("bt,tsc->bsc", W, X)); bs = np.sort(bs[~np.isnan(bs)])
            o[f"mh_or_{g}_vs_own_right"] = {"or": round(pt, 3),
                                            "ci95": [round(float(bs[int(.025 * len(bs))]), 3),
                                                     round(float(bs[int(.975 * len(bs)) - 1]), 3)]}
        res[arm] = o
        L.append(f"[{arm}] FA own right {o['own_right']['fa_pct']}% (n={o['own_right']['n']}); "
                 f"own wrong, same output {o['same_output']['fa_pct']}% (n={o['same_output']['n']}); "
                 f"own wrong, different output {o['different_output']['fa_pct']}% (n={o['different_output']['n']})")
        L.append(f"  MH OR same output vs own right {o['mh_or_same_output_vs_own_right']}; "
                 f"different output vs own right {o['mh_or_different_output_vs_own_right']}")
    (OUT / "shared_output_agreement.json").write_text(json.dumps(res, indent=2))
    (OUT / "shared_output_agreement_summary.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
