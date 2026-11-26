#!/usr/bin/env python3
"""Signal-detection change (steelman minus self-check) split by audit class.

Author-family verdicts from production. Incorrect candidates are split by the cross-model
audit (AGREED_KNOWABLE / AGREED_NOT_KNOWABLE); correct candidates are shared across both
classes. Only candidates with a parsed ACCEPT/REJECT in both arms are used (paired).
d' = z(TA) - z(FA), c = -(z(TA)+z(FA))/2, pooled over families (counts summed), with a
log-linear correction (+0.5) only if a rate is 0 or 1. Interval: task-clustered bootstrap
(2,000 resamples, seed 20261124). Specified after the data (reviewer request, round 8).
Output: data/mbpp_v2/analysis/sdt_by_knowability.json
"""
import json, math, random, sys
from collections import defaultdict
from pathlib import Path
from statistics import NormalDist
ROOT = Path(__file__).resolve().parents[1]; D = ROOT / "data/mbpp_v2"
z = NormalDist().inv_cdf
def jl(p): return [json.loads(l) for l in open(p) if l.strip()]
labs = defaultdict(dict)
for r in jl(D / "crossmodel_audit_all/labels.jsonl"):
    labs[(r["generator"], r["task_id"])][r["auditor"]] = r["label"]
cls = {}
for k, v in labs.items():
    ls = list(v.values())
    if len(ls) == 2 and len(set(ls)) == 1 and ls[0] in ("KNOWABLE", "NOT_KNOWABLE"):
        cls[k] = ls[0]
der = jl(D / "production/derived_records.jsonl")
rows = []
for r in der:
    if all(r.get(f"{a}_extraction") == "parsed" and r.get(f"{a}_verdict") in ("ACCEPT", "REJECT") for a in ("selfcheck", "steelman")):
        k = (r["model"], r["task_id"])
        grp = "correct" if r["T"] == 1 else cls.get(k)
        if grp:
            rows.append((r["task_id"], r["model"], grp, r["selfcheck_verdict"] == "ACCEPT", r["steelman_verdict"] == "ACCEPT"))
def rate(k, n):
    return (k + 0.5) / (n + 1) if k in (0, n) else k / n
def sdt(rs, cls_):
    out = {}
    for i, arm in ((3, "selfcheck"), (4, "steelman")):
        c = [r for r in rs if r[2] == "correct"]; e = [r for r in rs if r[2] == cls_]
        if not c or not e: return None
        ta = rate(sum(r[i] for r in c), len(c)); fa = rate(sum(r[i] for r in e), len(e))
        out[arm] = {"TA": ta, "FA": fa, "evidence_bits": math.log2(ta / fa), "d": z(ta) - z(fa), "c": -(z(ta) + z(fa)) / 2, "n_err": len(e), "n_cor": len(c)}
    out["dd"] = out["steelman"]["d"] - out["selfcheck"]["d"]; out["dc"] = out["steelman"]["c"] - out["selfcheck"]["c"]
    return out
by_task = defaultdict(list)
for r in rows: by_task[r[0]].append(r)
tasks = sorted(by_task); rng = random.Random(20261124)
res = {}
for cl in ("KNOWABLE", "NOT_KNOWABLE"):
    est = sdt(rows, cl); bd, bc = [], []
    for _ in range(2000):
        rs = [r for t in (rng.choice(tasks) for _ in tasks) for r in by_task[t]]
        s = sdt(rs, cl)
        if s: bd.append(s["dd"]); bc.append(s["dc"])
    bd.sort(); bc.sort(); q = lambda a, p: a[int(p * (len(a) - 1))]
    res[cl] = {"selfcheck": {k: round(v, 3) for k, v in est["selfcheck"].items()},
               "steelman": {k: round(v, 3) for k, v in est["steelman"].items()},
               "delta_d_prime": round(est["dd"], 3), "delta_d_prime_ci95": [round(q(bd, .025), 3), round(q(bd, .975), 3)],
               "delta_c": round(est["dc"], 3), "delta_c_ci95": [round(q(bc, .025), 3), round(q(bc, .975), 3)]}
(D / "analysis/sdt_by_knowability.json").write_text(json.dumps(res, indent=1) + "\n")
print(json.dumps(res, indent=1))
