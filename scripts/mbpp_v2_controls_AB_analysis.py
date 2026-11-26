"""Analysis of the confidence-rating and reasoning-off controls (paper Sec. IV-F/IV-G).

Confidence control: per family and arm, empirical ROC AUC from the 1-7 ratings
(P(rating of a correct candidate > rating of an incorrect one), ties counted half), with a
task-clustered bootstrap 95% interval, and the steelman-minus-self-check AUC difference on
the same candidates with its interval. No equal-variance assumption is needed.

Reasoning controls: for each family run with a changed reasoning setting, false and true
acceptance in that run vs. production on the same candidates, exact
McNemar on incorrect candidates, and d' in each condition.
Writes data/mbpp_v2/analysis/controls_AB_analysis.json.
"""
import json, random
from math import comb
from pathlib import Path
from statistics import NormalDist

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data/mbpp_v2"
N = NormalDist()
FAMS = ["gpt-5.4", "ds-v4-flash", "glm-5.3-flash", "qwen38-flash", "gpt-6-luna"]


def auc(pos, neg):
    if not pos or not neg:
        return None
    s = 0.0
    for p in pos:
        for n in neg:
            s += 1.0 if p > n else 0.5 if p == n else 0.0
    return s / (len(pos) * len(neg))


def mcnemar(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def dprime(tau, kap, n1, n0):
    tau = min(max(tau, 0.5 / n1), 1 - 0.5 / n1); kap = min(max(kap, 0.5 / n0), 1 - 0.5 / n0)
    return N.inv_cdf(tau) - N.inv_cdf(kap)


def confidence(out):
    p = D / "confidence_eval/records.jsonl"
    if not p.exists():
        return
    rec = {}
    for l in open(p):
        if l.strip():
            r = json.loads(l)
            if r.get("confidence") is not None:
                rec[(r["model"], r["task_id"], r["arm"])] = (r["T"], r["confidence"])
    random.seed(20261123)
    res = {}
    for m in FAMS:
        tids = sorted({t for (mm, t, a) in rec if mm == m and (m, t, "selfcheck") in rec and (m, t, "steelman") in rec})
        if not tids:
            continue
        def aucs(ts):
            o = {}
            for a in ("selfcheck", "steelman"):
                pos = [rec[(m, t, a)][1] for t in ts if rec[(m, t, a)][0] == 1]
                neg = [rec[(m, t, a)][1] for t in ts if rec[(m, t, a)][0] == 0]
                o[a] = auc(pos, neg)
            return o
        base = aucs(tids)
        bs = [aucs([random.choice(tids) for _ in tids]) for _ in range(1000)]
        def ci(f):
            v = sorted(x for x in (f(b) for b in bs) if x is not None)
            return [round(v[int(0.025 * len(v))], 3), round(v[int(0.975 * len(v)) - 1], 3)]
        res[m] = {"n_tasks": len(tids),
                  "auc_selfcheck": round(base["selfcheck"], 3), "auc_selfcheck_ci95": ci(lambda b: b["selfcheck"]),
                  "auc_steelman": round(base["steelman"], 3), "auc_steelman_ci95": ci(lambda b: b["steelman"]),
                  "delta_auc": round(base["steelman"] - base["selfcheck"], 3),
                  "delta_auc_ci95": ci(lambda b: b["steelman"] - b["selfcheck"] if None not in b.values() else None)}
        print("confidence", m, res[m])
    # Blind spot in the sense of Lakkaraju et al. (AAAI 2017): an incorrect candidate
    # (T=0) rated 6 or 7 ("likely" / "certainly correct") on the 1-7 scale.
    for m in FAMS:
        for a in ("selfcheck", "steelman"):
            neg = [c for (mm, t, aa), (tt, c) in rec.items() if mm == m and aa == a and tt == 0]
            if neg and m in res:
                res[m][f"confident_error_pct_{a}"] = round(100 * sum(c >= 6 for c in neg) / len(neg), 1)
                res[m][f"n_incorrect_{a}"] = len(neg)
        if m in res:
            print("confident errors", m, {k: v for k, v in res[m].items() if k.startswith(("confident", "n_incorrect"))})
    out["confidence"] = res


def _sdt(rows, i):
    c = [q for q in rows if q[0] == 1]; n = [q for q in rows if q[0] == 0]
    t = sum(q[i] for q in c) / len(c); k = sum(q[i] for q in n) / len(n)
    t = min(max(t, .5 / len(c)), 1 - .5 / len(c)); k = min(max(k, .5 / len(n)), 1 - .5 / len(n))
    return N.inv_cdf(t) - N.inv_cdf(k), -(N.inv_cdf(t) + N.inv_cdf(k)) / 2, 100 * k, 100 * t


def reasoning_off(out):
    """Production vs. changed reasoning setting on the same candidates: FA, TA, McNemar,
    d' and criterion change with paired bootstrap 95% intervals (2,000 resamples,
    seed 20261123 reset per run folder)."""
    base = D / "reasoning_off_eval"
    if not base.exists():
        return
    prod = {(r["model"], r["task_id"]): r for r in map(json.loads, open(D / "production/derived_records.jsonl"))}
    res = {}
    for sub in ("gpt-5.4__effort-low", "ds-v4-flash", "qwen38-flash", "gpt-6-luna"):
        if not (base / sub).exists():
            continue
        m = sub.split("__")[0]
        r = [json.loads(l) for l in open(base / sub / "records.jsonl") if l.strip()]
        run = {(x["task_id"], x["arm"]): x["verdict"] == "ACCEPT" for x in r if x["verdict"] in ("ACCEPT", "REJECT")}
        random.seed(20261123)
        for a in ("selfcheck", "steelman"):
            rows = [(p["T"], p[a + "_verdict"] == "ACCEPT", run[(t, a)]) for (mm, t), p in prod.items()
                    if mm == m and p["T"] in (0, 1) and p.get(a + "_extraction") == "parsed" and (t, a) in run]
            x0, x1 = _sdt(rows, 1), _sdt(rows, 2)
            inc = [q for q in rows if q[0] == 0]
            b = sum(1 for q in inc if q[1] and not q[2]); c = sum(1 for q in inc if not q[1] and q[2])
            bs = []
            for _ in range(2000):
                smp = [random.choice(rows) for _ in rows]; u, v = _sdt(smp, 1), _sdt(smp, 2)
                bs.append((v[0] - u[0], v[1] - u[1]))
            dd = sorted(q[0] for q in bs); dc = sorted(q[1] for q in bs)
            res[f"{sub}|{a}"] = {"FA": [round(x0[2], 1), round(x1[2], 1)], "TA": [round(x0[3], 1), round(x1[3], 1)],
                                 "b_c": [b, c], "p": mcnemar(b, c), "dprime": [round(x0[0], 2), round(x1[0], 2)],
                                 "d_delta": round(x1[0] - x0[0], 2), "d_ci": [round(dd[50], 2), round(dd[1949], 2)],
                                 "c_delta": round(x1[1] - x0[1], 2), "c_ci": [round(dc[50], 2), round(dc[1949], 2)]}
            print("reasoning", sub, a, res[f"{sub}|{a}"])
    out["reasoning"] = res


if __name__ == "__main__":
    out = {}
    confidence(out); reasoning_off(out)
    (D / "analysis/controls_AB_analysis.json").write_text(json.dumps(out, indent=1))
