#!/usr/bin/env python3
"""P2 where the evaluator has the information: steelman minus self-check on errors the evaluator
can see, in three data sets (analysis specified after the data, using existing records only).

A. Information control (production candidates, first reference assert shown to the evaluator):
   errors that fail the shown assert, plus all correct candidates.
B. Generator sees assert (new candidates; same assert shown to the evaluator; arms interleaved):
   errors that fail the shown assert, plus all correct candidates.
C. Five-family interleaved rerun, errors both auditors judged knowable from the description.
Paired (parsed in both arms), pooled over families with counts summed; d' = z(TA)-z(FA),
c = -(z(TA)+z(FA))/2; task-clustered bootstrap, 2,000 resamples, seed 20261126.
Also reports whether the two arms of A ran interleaved in time.
Output: data/mbpp_v2/analysis/p2_information_present.json
"""
from __future__ import annotations
import json, random, sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path
from statistics import NormalDist
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import scripts.round8_common as rc
from scripts.round8_common import pilot
D = ROOT / "data/mbpp_v2"; z = NormalDist().inv_cdf
def jl(p): return [json.loads(l) for l in open(p) if l.strip()]
def rate(k, n): return (k + .5) / (n + 1) if k in (0, n) else k / n
def sdt(rs, arm):
    c = [r for r in rs if r["T"] == 1]; e = [r for r in rs if r["T"] == 0]
    ta, fa = rate(sum(r[arm] for r in c), len(c)), rate(sum(r[arm] for r in e), len(e))
    return z(ta) - z(fa), -(z(ta) + z(fa)) / 2, ta, fa, len(e), len(c)
def contrast(rows, seed=20261126):
    d0, c0, ta0, fa0, ne, nc = sdt(rows, "sc"); d1, c1, ta1, fa1, _, _ = sdt(rows, "st")
    by = defaultdict(list)
    for r in rows: by[r["task_id"]].append(r)
    ts = sorted(by); rng = random.Random(seed); bd, bc = [], []
    for _ in range(2000):
        s = [r for t in (rng.choice(ts) for _ in ts) for r in by[t]]
        a, b = sdt(s, "sc"), sdt(s, "st"); bd.append(b[0] - a[0]); bc.append(b[1] - a[1])
    bd.sort(); bc.sort(); q = lambda x, p: round(x[int(p * (len(x) - 1))], 3)
    return {"n_err": ne, "n_cor": nc, "FA": [round(fa0, 4), round(fa1, 4)], "TA": [round(ta0, 4), round(ta1, 4)],
            "d_selfcheck": round(d0, 3), "d_steelman": round(d1, 3),
            "dd": round(d1 - d0, 3), "dd_ci95": [q(bd, .025), q(bd, .975)], "dc": round(c1 - c0, 3), "dc_ci95": [q(bc, .025), q(bc, .975)]}
if __name__ == "__main__":
    tasks = pilot.load_task_map(); out = {}
    # A
    prod = rc.load_candidates()
    av = defaultdict(dict)
    for r in jl(D / "assert_visible_eval/records.jsonl"): av[(r["model"], r["task_id"])][r["arm"]] = r["verdict"]
    inc = [k for k, c in prod.items() if c["T"] == 0 and c.get("code") and k in av]
    with Pool(16) as p:
        fails = p.map(rc.assert_fails, [(prod[k]["code"], rc.split_parts(tasks[k[1]]["test"])[0], rc.first_assert(tasks[k[1]])) for k in inc], chunksize=4)
    fs = {k for k, f in zip(inc, fails) if f}
    rows = []
    for k, v in av.items():
        if v.get("selfcheck") in ("ACCEPT", "REJECT") and v.get("steelman") in ("ACCEPT", "REJECT"):
            T = prod[k]["T"]
            if T == 1 or k in fs:
                rows.append({"task_id": k[1], "T": T, "sc": v["selfcheck"] == "ACCEPT", "st": v["steelman"] == "ACCEPT"})
    out["A_information_control_fails_shown"] = contrast(rows)
    ev = jl(D / "assert_visible_eval/events.jsonl")
    seq = [e["stage"] for e in sorted(ev, key=lambda e: e["ts"]) if e["stage"] in ("selfcheck", "steelman") or "selfcheck" in e["stage"] or "steelman" in e["stage"]]
    sw = sum(1 for a, b in zip(seq, seq[1:]) if a != b)
    out["A_arm_switches_in_time_order"] = {"events": len(seq), "switches": sw}
    # B
    gc = {(c["model"], c["task_id"]): c for c in jl(D / "round8_gen_assert/candidate_records.jsonl")}
    last = {}
    for r in jl(D / "round8_gen_assert/eval_records.jsonl"): last[(r["model"], r["task_id"], r["arm"])] = r["verdict"]
    ginc = [k for k, c in gc.items() if c["T"] == 0]
    with Pool(16) as p:
        gf = p.map(rc.assert_fails, [(gc[k]["code"], rc.split_parts(tasks[k[1]]["test"])[0], gc[k]["shown_assert"]) for k in ginc], chunksize=4)
    gfs = {k for k, f in zip(ginc, gf) if f}
    rows = []
    for k, c in gc.items():
        a, b = last.get((k[0], k[1], "selfcheck")), last.get((k[0], k[1], "steelman"))
        if a in ("ACCEPT", "REJECT") and b in ("ACCEPT", "REJECT") and (c["T"] == 1 or k in gfs):
            rows.append({"task_id": k[1], "T": c["T"], "sc": a == "ACCEPT", "st": b == "ACCEPT"})
    out["B_generator_sees_assert_fails_shown"] = contrast(rows)
    # C
    ia = json.load(open(D / "analysis/round8_interleaved_analysis.json"))["knowable"]
    out["C_interleaved_knowable"] = {"n_err": ia["selfcheck"]["n_err"], "dd": ia["dd"], "dd_ci95": ia["dd_ci95"], "dc": ia["dc"], "dc_ci95": ia["dc_ci95"]}
    (D / "analysis/p2_information_present.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))
