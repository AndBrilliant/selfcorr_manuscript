#!/usr/bin/env python3
"""Analysis of the round-8 placebo-assert control (specified with the run).

Three self-check conditions on the same frozen candidates, paired:
  none     production self-check (description only)
  placebo  description + an assert on the same input whose expected value comes from the
           candidate itself (no information about correctness)
  real     description + the task's first reference assert (information control)
Only candidates with a placebo and a parsed ACCEPT/REJECT in all three conditions are used.

Per family and pooled (counts summed): false acceptance (FA) on incorrect candidates,
true acceptance (TA) on correct ones, d' = z(TA) - z(FA), c = -(z(TA)+z(FA))/2 (+0.5
correction only at 0 or 1). Paired exact McNemar on FA (placebo vs none, real vs none).
Delta d' and delta c with task-clustered bootstrap (2,000 resamples, seed 20261125).

Strata of incorrect candidates:
  exposed  fails the shown (first) reference assert: the real assert shows the right value,
           the placebo shows the candidate's own wrong value
  hidden   passes the shown assert: placebo and real assert are the same line
Output: data/mbpp_v2/analysis/round8_placebo_analysis.json
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from math import comb
from pathlib import Path
from statistics import NormalDist

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data/mbpp_v2"
FAMS = ["gpt-5.4", "ds-v4-flash", "glm-5.3-flash", "qwen38-flash", "gpt-6-luna"]
z = NormalDist().inv_cdf


def jl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def last_by(rows, key):
    out = {}
    for r in rows:
        out[key(r)] = r
    return out


def rate(k, n):
    return (k + 0.5) / (n + 1) if k in (0, n) else k / n


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


placebo = last_by(jl(D / "round8_placebo/records.jsonl"), lambda r: (r["model"], r["task_id"]))
real = last_by([r for r in jl(D / "assert_visible_eval/records.jsonl") if r["arm"] == "selfcheck"],
               lambda r: (r["model"], r["task_id"]))
prod = last_by(jl(D / "production/derived_records.jsonl"), lambda r: (r["model"], r["task_id"]))
pl_lines = {(r["model"], r["task_id"]): r for r in jl(D / "round8_placebo/placebos.jsonl")}

rows = []
for k, p in placebo.items():
    if k not in real or k not in prod:
        continue
    vn = prod[k].get("selfcheck_verdict") if prod[k].get("selfcheck_extraction") == "parsed" else None
    vp, vr = p.get("verdict"), real[k].get("verdict")
    if not all(v in ("ACCEPT", "REJECT") for v in (vn, vp, vr)):
        continue
    T = prod[k]["T"]
    rows.append({"model": k[0], "task_id": k[1], "T": T, "stratum": None,
                 "none": vn == "ACCEPT", "placebo": vp == "ACCEPT", "real": vr == "ACCEPT"})

# An incorrect candidate is "hidden" if it passes the shown assert. That is recorded in the
# error-severity audit (per_assert_failed index 0), which the paper already uses.
sev = {(r["model"], r["task_id"]): r for r in jl(D / "analysis/error_severity_audit.jsonl")}
for r in rows:
    if r["T"] == 0:
        s = sev.get((r["model"], r["task_id"]))
        if s is None or not s.get("per_assert_failed"):
            r["stratum"] = None
        else:
            r["stratum"] = "exposed" if s["per_assert_failed"][0] else "hidden"


def sdt(rs, cond, stratum=None):
    c = [r for r in rs if r["T"] == 1]
    e = [r for r in rs if r["T"] == 0 and (stratum is None or r["stratum"] == stratum)]
    if not c or not e:
        return None
    ta = rate(sum(r[cond] for r in c), len(c))
    fa = rate(sum(r[cond] for r in e), len(e))
    return {"n_err": len(e), "n_cor": len(c), "FA": fa, "TA": ta,
            "d": z(ta) - z(fa), "c": -(z(ta) + z(fa)) / 2}


def block(rs, stratum=None, boot=True, seed=20261125):
    out = {}
    for cond in ("none", "placebo", "real"):
        s = sdt(rs, cond, stratum)
        out[cond] = None if s is None else {k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items()}
    if out["none"] is None:
        return out
    e = [r for r in rs if r["T"] == 0 and (stratum is None or r["stratum"] == stratum)]
    for cond in ("placebo", "real"):
        b = sum(1 for r in e if r["none"] and not r[cond])   # accepted without, rejected with
        cc = sum(1 for r in e if not r["none"] and r[cond])
        out[f"FA_mcnemar_{cond}_vs_none"] = {"none_only": b, "with_only": cc, "p": round(mcnemar(b, cc), 5)}
    if boot:
        by = defaultdict(list)
        for r in rs:
            by[r["task_id"]].append(r)
        ts = sorted(by)
        rng = random.Random(seed)
        acc = {f"{q}_{cond}": [] for q in ("dd", "dc") for cond in ("placebo", "real")}
        for _ in range(2000):
            sample = [r for t in (rng.choice(ts) for _ in ts) for r in by[t]]
            s0 = sdt(sample, "none", stratum)
            if s0 is None:
                continue
            for cond in ("placebo", "real"):
                s1 = sdt(sample, cond, stratum)
                acc[f"dd_{cond}"].append(s1["d"] - s0["d"])
                acc[f"dc_{cond}"].append(s1["c"] - s0["c"])
        for cond in ("placebo", "real"):
            for q in ("dd", "dc"):
                a = sorted(acc[f"{q}_{cond}"])
                est = out[cond]["d" if q == "dd" else "c"] - out["none"]["d" if q == "dd" else "c"]
                out[f"{q}_{cond}"] = {"est": round(est, 3),
                                      "ci95": [round(a[int(.025 * (len(a) - 1))], 3), round(a[int(.975 * (len(a) - 1))], 3)]}
    return out


res = {"n_rows": len(rows), "by_family": {}, "pooled": {}}
for m in FAMS:
    rs = [r for r in rows if r["model"] == m]
    res["by_family"][m] = {"all": block(rs, boot=True), "exposed": block(rs, "exposed", boot=False),
                           "hidden": block(rs, "hidden", boot=False)}
res["pooled"] = {"all": block(rows), "exposed": block(rows, "exposed"), "hidden": block(rows, "hidden")}
res["strata_counts"] = {m: {s: sum(1 for r in rows if r["model"] == m and r["stratum"] == s) for s in ("exposed", "hidden", None)} for m in FAMS}
(D / "analysis").mkdir(exist_ok=True)
(D / "analysis/round8_placebo_analysis.json").write_text(json.dumps(res, indent=1, default=str) + "\n")
print(json.dumps({"n_rows": res["n_rows"], "strata": res["strata_counts"]}, default=str))
for scope in ("all", "exposed", "hidden"):
    p = res["pooled"][scope]
    print(f"\nPOOLED {scope}: " + " | ".join(
        f"{c}: FA {p[c]['FA']:.3f} TA {p[c]['TA']:.3f} d' {p[c]['d']:.2f} c {p[c]['c']:.2f} (n_err {p[c]['n_err']})" for c in ("none", "placebo", "real")))
    for cond in ("placebo", "real"):
        print(f"  {cond} vs none: dd {p['dd_'+cond]}, dc {p['dc_'+cond]}, McNemar FA {p['FA_mcnemar_'+cond+'_vs_none']}")
print()
for m in FAMS:
    a = res["by_family"][m]["all"]
    print(m, " | ".join(f"{c}: FA {a[c]['FA']:.3f} TA {a[c]['TA']:.3f} d' {a[c]['d']:.2f}" for c in ("none", "placebo", "real")),
          "| dd placebo", a["dd_placebo"], "McNemar p", a["FA_mcnemar_placebo_vs_none"]["p"])
