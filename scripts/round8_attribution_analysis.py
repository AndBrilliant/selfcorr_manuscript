#!/usr/bin/env python3
"""Analysis of the round-8 attribution pilot (none / self / user framing).

Per family and pooled (counts summed): acceptance of incorrect candidates (FA) overall
and split by audit class (knowable / not knowable), acceptance of correct ones (TA),
d' and c. Changes self-none and user-none with task-clustered bootstrap intervals
(2,000 resamples, seed 20261125); paired exact McNemar on FA. Candidates parsed under all
three framings only. Output: data/mbpp_v2/analysis/round8_attribution_analysis.json
"""
from __future__ import annotations
import json, random
from collections import defaultdict
from math import comb
from pathlib import Path
from statistics import NormalDist
ROOT = Path(__file__).resolve().parents[1]; D = ROOT / "data/mbpp_v2"
FAMS = ["gpt-5.4", "ds-v4-flash", "glm-5.3-flash", "qwen38-flash", "gpt-6-luna"]
z = NormalDist().inv_cdf
def jl(p): return [json.loads(l) for l in open(p) if l.strip()]
last = {}
for r in jl(D / "round8_attribution/records.jsonl"):
    last[(r["model"], r["task_id"], r["frame"])] = r
labs = defaultdict(dict)
for r in jl(D / "crossmodel_audit_all/labels.jsonl"):
    labs[(r["generator"], r["task_id"])][r["auditor"]] = r["label"]
cls = {k: list(v.values())[0] for k, v in labs.items() if len(v) == 2 and len(set(v.values())) == 1}
rows = []
for (m, t) in {(k[0], k[1]) for k in last}:
    v = {f: last.get((m, t, f), {}).get("verdict") for f in ("none", "self", "user")}
    if not all(x in ("ACCEPT", "REJECT") for x in v.values()):
        continue
    T = last[(m, t, "none")]["T"]
    rows.append({"model": m, "task_id": t, "T": T, "cls": None if T else cls.get((m, t)),
                 **{f: v[f] == "ACCEPT" for f in v}})
def rate(k, n): return (k + .5) / (n + 1) if k in (0, n) else k / n
def mcn(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)
def sdt(rs, f, sub=None):
    c = [r for r in rs if r["T"] == 1]; e = [r for r in rs if r["T"] == 0 and (sub is None or r["cls"] == sub)]
    if not c or not e: return None
    ta, fa = rate(sum(r[f] for r in c), len(c)), rate(sum(r[f] for r in e), len(e))
    return {"n_err": len(e), "n_cor": len(c), "FA": fa, "TA": ta, "d": z(ta) - z(fa), "c": -(z(ta) + z(fa)) / 2}
def block(rs, sub=None, boot=True):
    out = {f: sdt(rs, f, sub) for f in ("none", "self", "user")}
    if out["none"] is None: return out
    e = [r for r in rs if r["T"] == 0 and (sub is None or r["cls"] == sub)]
    c = [r for r in rs if r["T"] == 1]
    for f in ("self", "user"):
        out[f"FA_mcnemar_{f}"] = {"none_only": sum(r["none"] and not r[f] for r in e), f"{f}_only": sum(r[f] and not r["none"] for r in e),
                                  "p": round(mcn(sum(r["none"] and not r[f] for r in e), sum(r[f] and not r["none"] for r in e)), 4)}
        out[f"TA_mcnemar_{f}"] = {"p": round(mcn(sum(r["none"] and not r[f] for r in c), sum(r[f] and not r["none"] for r in c)), 4)}
    if boot:
        by = defaultdict(list)
        for r in rs: by[r["task_id"]].append(r)
        ts = sorted(by); rng = random.Random(20261125); acc = defaultdict(list)
        for _ in range(2000):
            s = [r for t in (rng.choice(ts) for _ in ts) for r in by[t]]
            s0 = sdt(s, "none", sub)
            if not s0: continue
            for f in ("self", "user"):
                s1 = sdt(s, f, sub)
                if s1: acc[f"dd_{f}"].append(s1["d"] - s0["d"]); acc[f"dc_{f}"].append(s1["c"] - s0["c"])
        for k, a in acc.items():
            a.sort(); f = k.split("_")[1]; q = "d" if k.startswith("dd") else "c"
            out[k] = {"est": round(out[f][q] - out["none"][q], 3), "ci95": [round(a[int(.025 * (len(a) - 1))], 3), round(a[int(.975 * (len(a) - 1))], 3)]}
    return out
res = {"n": len(rows), "pooled": {"all": block(rows), "knowable": block(rows, "KNOWABLE"), "not_knowable": block(rows, "NOT_KNOWABLE")},
       "by_family": {m: {"all": block([r for r in rows if r["model"] == m]),
                         "knowable": block([r for r in rows if r["model"] == m], "KNOWABLE", boot=False)} for m in FAMS}}
(D / "analysis/round8_attribution_analysis.json").write_text(json.dumps(res, indent=1, default=str) + "\n")
def show(name, b):
    print(f"{name}: " + " | ".join(f"{f}: FA {100*b[f]['FA']:.1f} TA {100*b[f]['TA']:.1f} d' {b[f]['d']:.2f} c {b[f]['c']:.2f}" for f in ("none", "self", "user")) + f"  (n_err {b['none']['n_err']}, n_cor {b['none']['n_cor']})")
    for f in ("self", "user"):
        print(f"    {f}-none: dd {b.get('dd_'+f)}  dc {b.get('dc_'+f)}  FA McNemar {b['FA_mcnemar_'+f]}  TA McNemar p {b['TA_mcnemar_'+f]['p']}")
print("rows", len(rows))
for k in ("all", "knowable", "not_knowable"): show("POOLED " + k, res["pooled"][k])
for m in FAMS: show(m, res["by_family"][m]["all"]); kb = res["by_family"][m]["knowable"]; print(f"    knowable FA none/self/user: {kb['none']['FA']:.2f} {kb['self']['FA']:.2f} {kb['user']['FA']:.2f} (n {kb['none']['n_err']})")
