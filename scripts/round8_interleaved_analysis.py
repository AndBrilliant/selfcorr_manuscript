#!/usr/bin/env python3
"""Scores the interleaved self-check vs steelman run against PREREGISTRATION_round8_pending.md (Run A).

Candidates with a parsed ACCEPT/REJECT in both arms; pooled over families with counts summed.
d' = z(TA) - z(FA), c = -(z(TA)+z(FA))/2, +0.5 correction only at 0 or 1. Intervals: task-clustered
bootstrap, 2,000 resamples, seed 20261125. Audit classes: both auditors agree.
A1: pooled delta c (steelman - self-check) > 0 with 95% interval above 0.
A2: pooled delta d' on not-knowable errors, 95% interval contains 0.
A3: steelman false acceptance lower than self-check in >= 4 of 5 families.
Also reported (not predicted): delta d' on knowable errors; comparison with production.
Output: data/mbpp_v2/analysis/round8_interleaved_analysis.json
"""
from __future__ import annotations
import json, random
from collections import defaultdict
from pathlib import Path
from statistics import NormalDist
ROOT = Path(__file__).resolve().parents[1]; D = ROOT / "data/mbpp_v2"
FAMS = ["gpt-5.4", "ds-v4-flash", "glm-5.3-flash", "qwen38-flash", "gpt-6-luna"]
z = NormalDist().inv_cdf
def jl(p): return [json.loads(l) for l in open(p) if l.strip()]
last = {}
for r in jl(D / "round8_interleaved/records.jsonl"): last[(r["model"], r["task_id"], r["arm"])] = r
labs = defaultdict(dict)
for r in jl(D / "crossmodel_audit_all/labels.jsonl"): labs[(r["generator"], r["task_id"])][r["auditor"]] = r["label"]
cls = {k: list(v.values())[0] for k, v in labs.items() if len(v) == 2 and len(set(v.values())) == 1}
rows = []
for (m, t) in {(k[0], k[1]) for k in last}:
    a, b = last.get((m, t, "selfcheck")), last.get((m, t, "steelman"))
    if not a or not b or a["verdict"] not in ("ACCEPT", "REJECT") or b["verdict"] not in ("ACCEPT", "REJECT"): continue
    rows.append({"model": m, "task_id": t, "T": a["T"], "cls": None if a["T"] else cls.get((m, t)),
                 "sc": a["verdict"] == "ACCEPT", "st": b["verdict"] == "ACCEPT"})
def rate(k, n): return (k + .5) / (n + 1) if k in (0, n) else k / n
def sdt(rs, arm, sub=None):
    c = [r for r in rs if r["T"] == 1]; e = [r for r in rs if r["T"] == 0 and (sub is None or r["cls"] == sub)]
    if not c or not e: return None
    ta, fa = rate(sum(r[arm] for r in c), len(c)), rate(sum(r[arm] for r in e), len(e))
    return {"n_err": len(e), "n_cor": len(c), "FA": fa, "TA": ta, "d": z(ta) - z(fa), "c": -(z(ta) + z(fa)) / 2}
def delta(rs, sub=None, seed=20261125):
    s0, s1 = sdt(rs, "sc", sub), sdt(rs, "st", sub)
    by = defaultdict(list)
    for r in rs: by[r["task_id"]].append(r)
    ts = sorted(by); rng = random.Random(seed); dd, dc = [], []
    for _ in range(2000):
        s = [r for t in (rng.choice(ts) for _ in ts) for r in by[t]]
        a, b = sdt(s, "sc", sub), sdt(s, "st", sub)
        if a and b: dd.append(b["d"] - a["d"]); dc.append(b["c"] - a["c"])
    dd.sort(); dc.sort(); q = lambda x, p: round(x[int(p * (len(x) - 1))], 3)
    return {"selfcheck": s0, "steelman": s1, "dd": round(s1["d"] - s0["d"], 3), "dd_ci95": [q(dd, .025), q(dd, .975)],
            "dc": round(s1["c"] - s0["c"], 3), "dc_ci95": [q(dc, .025), q(dc, .975)]}
res = {"n_rows": len(rows), "all": delta(rows), "not_knowable": delta(rows, "NOT_KNOWABLE"), "knowable": delta(rows, "KNOWABLE"),
       "by_family": {m: delta([r for r in rows if r["model"] == m]) for m in FAMS}}
A1 = res["all"]["dc_ci95"][0] > 0
A2 = res["not_knowable"]["dd_ci95"][0] <= 0 <= res["not_knowable"]["dd_ci95"][1]
fa_lower = {m: res["by_family"][m]["steelman"]["FA"] < res["by_family"][m]["selfcheck"]["FA"] for m in FAMS}
A3 = sum(fa_lower.values()) >= 4
contradicted = res["not_knowable"]["dd_ci95"][0] > 0
res["score"] = {"A1": A1, "A2": A2, "A3": A3, "A3_families_lower": fa_lower, "P2_supported": A1 and A2, "P2_contradicted": contradicted}
(D / "analysis/round8_interleaved_analysis.json").write_text(json.dumps(res, indent=1, default=str) + "\n")
print("rows", len(rows))
for k in ("all", "not_knowable", "knowable"):
    x = res[k]; print(f"{k}: FA sc {100*x['selfcheck']['FA']:.1f} st {100*x['steelman']['FA']:.1f} | TA sc {100*x['selfcheck']['TA']:.1f} st {100*x['steelman']['TA']:.1f} | dd {x['dd']} {x['dd_ci95']} | dc {x['dc']} {x['dc_ci95']} (n_err {x['selfcheck']['n_err']})")
for m in FAMS:
    x = res["by_family"][m]; print(f"  {m}: FA {100*x['selfcheck']['FA']:.1f} -> {100*x['steelman']['FA']:.1f}, dd {x['dd']} {x['dd_ci95']}, dc {x['dc']} {x['dc_ci95']}")
print("SCORE", res["score"])
