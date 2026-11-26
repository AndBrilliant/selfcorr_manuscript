#!/usr/bin/env python3
"""Scores the generator-sees-assert run against PREREGISTRATION_round8_pending.md (Run B).

Last record per (family, task, arm). Incorrect candidates (full test fails) are split by whether
they fail the shown assert: (i) fails shown, (ii) passes shown, fails hidden only.
B1: self-check FA on (i) higher than the information control's 24/1,066 (one-sided Fisher, p < 0.05).
B2: self-check FA on (ii) above 60% (pooled).
B3: steelman vs self-check delta c > 0 (task-clustered bootstrap 95% interval reported).
Output: data/mbpp_v2/analysis/round8_gen_assert_analysis.json
"""
from __future__ import annotations
import json, random, sys
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path
from statistics import NormalDist
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import scripts.round8_common as rc
from scripts.round8_common import pilot
from scipy.stats import fisher_exact
D = ROOT / "data/mbpp_v2/round8_gen_assert"; z = NormalDist().inv_cdf
FAMS = rc.FAMILIES
cand = {(c["model"], c["task_id"]): c for c in map(json.loads, open(D / "candidate_records.jsonl"))}
recs = [json.loads(l) for l in open(D / "eval_records.jsonl")]
groups = defaultdict(list)
for r in recs: groups[(r["model"], r["task_id"], r["arm"])].append(r)
last = {k: v[-1] for k, v in groups.items()}
dups = [v for v in groups.values() if len(v) > 1]
dup_disagree = sum(1 for v in dups if len({x["verdict"] for x in v}) > 1)
tasks = pilot.load_task_map()
inc = [k for k, c in cand.items() if c["T"] == 0]
if __name__ == "__main__":
    with Pool(16) as p:
        fails = p.map(rc.assert_fails, [(cand[k]["code"], rc.split_parts(tasks[k[1]]["test"])[0], cand[k]["shown_assert"]) for k in inc], chunksize=4)
    stratum = {k: ("i" if f else "ii") for k, f in zip(inc, fails)}
    rows = []
    for (m, t), c in cand.items():
        a, b = last.get((m, t, "selfcheck")), last.get((m, t, "steelman"))
        if not a or not b: continue
        rows.append({"model": m, "task_id": t, "T": c["T"], "s": stratum.get((m, t)),
                     "sc": a["verdict"], "st": b["verdict"]})
    def fa(rs, arm, s):
        x = [r for r in rs if r["T"] == 0 and r["s"] == s and r[arm] in ("ACCEPT", "REJECT")]
        return sum(r[arm] == "ACCEPT" for r in x), len(x)
    k1, n1 = fa(rows, "sc", "i"); k2, n2 = fa(rows, "sc", "ii")
    odds, p1 = fisher_exact([[k1, n1 - k1], [24, 1066 - 24]], alternative="greater")
    def rate(k, n): return (k + .5) / (n + 1) if k in (0, n) else k / n
    pr = [r for r in rows if r["sc"] in ("ACCEPT", "REJECT") and r["st"] in ("ACCEPT", "REJECT")]
    def sdt(rs, arm):
        c = [r for r in rs if r["T"] == 1]; e = [r for r in rs if r["T"] == 0]
        ta, f = rate(sum(r[arm] == "ACCEPT" for r in c), len(c)), rate(sum(r[arm] == "ACCEPT" for r in e), len(e))
        return z(ta) - z(f), -(z(ta) + z(f)) / 2, ta, f
    d0, c0, ta0, f0 = sdt(pr, "sc"); d1, c1, ta1, f1 = sdt(pr, "st")
    by = defaultdict(list)
    for r in pr: by[r["task_id"]].append(r)
    ts = sorted(by); rng = random.Random(20261125); bc, bd = [], []
    for _ in range(2000):
        s = [r for t in (rng.choice(ts) for _ in ts) for r in by[t]]
        a, b = sdt(s, "sc"), sdt(s, "st"); bc.append(b[1] - a[1]); bd.append(b[0] - a[0])
    bc.sort(); bd.sort(); q = lambda x, p: round(x[int(p * (len(x) - 1))], 3)
    fam = {}
    for m in FAMS:
        rs = [r for r in rows if r["model"] == m]
        fam[m] = {"FA_i": fa(rs, "sc", "i"), "FA_ii": fa(rs, "sc", "ii"), "FA_i_steelman": fa(rs, "st", "i")}
    res = {"duplicates": {"keys": len(dups), "disagreeing": dup_disagree},
           "strata_counts": dict(Counter(stratum.values())),
           "B1": {"selfcheck_FA_i": [k1, n1], "rate": round(k1 / n1, 4), "reference": [24, 1066], "p_one_sided": p1, "supported": p1 < 0.05},
           "B2": {"selfcheck_FA_ii": [k2, n2], "rate": round(k2 / n2, 4), "supported": k2 / n2 > 0.60},
           "B3": {"dc": round(c1 - c0, 3), "dc_ci95": [q(bc, .025), q(bc, .975)], "dd": round(d1 - d0, 3), "dd_ci95": [q(bd, .025), q(bd, .975)],
                  "TA": [round(ta0, 4), round(ta1, 4)], "FA_all": [round(f0, 4), round(f1, 4)], "supported": c1 - c0 > 0},
           "steelman_FA_i": fa(rows, "st", "i"), "steelman_FA_ii": fa(rows, "st", "ii"), "by_family": fam}
    (ROOT / "data/mbpp_v2/analysis/round8_gen_assert_analysis.json").write_text(json.dumps(res, indent=1, default=str) + "\n")
    print(json.dumps(res, indent=1, default=str))
