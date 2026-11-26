#!/usr/bin/env python3
"""Shared-blind-spot test on the cross-evaluation matrix.

Question: when evaluator family E falsely accepts another family G's incorrect code on
task t, is that more likely when E's own frozen candidate for t is also incorrect,
beyond what task difficulty explains?

Inputs
  data/mbpp_v2/cross_eval/cross_records.jsonl          (evaluator != generator, 20 pairs)
  data/mbpp_v2/production/derived_records.jsonl        (each family's own candidate, T)
  data/mbpp_v2/analysis/error_severity_audit.jsonl     (per-assert failures, T=0 only)

Unit: cross row with G's candidate T=0 and E's own candidate T in {0,1}; verdicts other
than ACCEPT/REJECT dropped (per arm). Outcome FA = verdict == ACCEPT.
Exposures: own_wrong = E's own T == 0 (primary); same_way = own_wrong and E's and G's
per_assert_failed lists identical, compared with own_wrong = 0 (secondary).
Difficulty: number of the other three families (excluding E and G) with T = 0 on t (0..3).
Primary statistic: Mantel-Haenszel common odds ratio, strata (E, G, difficulty).
95% CI: task-clustered percentile bootstrap, 2000 resamples, seed 20261123.
Decision rule (self-check, primary): CI entirely above 1 -> "supported"; CI includes 1 ->
"not supported"; CI entirely below 1 -> "reversed".

Computed after data collection; analysis plan fixed before results were seen.
Deterministic; reads ledgers only; no code execution, no network.
Outputs: data/mbpp_v2/analysis/shared_blindspot.json, shared_blindspot_summary.txt
"""
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data/mbpp_v2"
OUT = D / "analysis"
FAMS = ["gpt-5.4", "ds-v4-flash", "glm-5.3-flash", "qwen38-flash", "gpt-6-luna"]
B, SEED = 2000, 20261123
V = {"ACCEPT", "REJECT"}


def jl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def wilson(k, n, z=1.959964):
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(100 * (c - h), 1), round(100 * (c + h), 1)]


def mh(cells):
    """cells: array (S, 4) of a,b,c,d = exposed&FA, exposed&noFA, unexp&FA, unexp&noFA."""
    a, b, c, d = cells[..., 0], cells[..., 1], cells[..., 2], cells[..., 3]
    n = a + b + c + d
    ok = ((a + b) > 0) & ((c + d) > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        num = np.where(ok, a * d / np.where(n > 0, n, 1), 0).sum(-1)
        den = np.where(ok, b * c / np.where(n > 0, n, 1), 0).sum(-1)
        return np.where(den > 0, num / den, np.nan)


def main():
    T = {(r["model"], r["task_id"]): r["T"] for r in jl(D / "production/derived_records.jsonl")}
    pa = {(r["model"], r["task_id"]): tuple(r["per_assert_failed"])
          for r in jl(OUT / "error_severity_audit.jsonl")}
    cross = jl(D / "cross_eval/cross_records.jsonl")
    drops = Counter()
    rows = []
    for r in cross:
        E, G, t = r["evaluator"], r["generator"], r["task_id"]
        drops["cross_rows_total"] += 1
        tg, te = T.get((G, t)), T.get((E, t))
        if tg is None or te is None:
            drops["null_T_E_or_G"] += 1
            continue
        if tg != 0:
            drops["generator_correct"] += 1
            continue
        others = [f for f in FAMS if f not in (E, G)]
        to = [T.get((f, t)) for f in others]
        if any(x is None for x in to):
            drops["null_T_other_family"] += 1
            continue
        diff = sum(1 for x in to if x == 0)
        own_wrong = int(te == 0)
        same = None
        if own_wrong:
            pe, pg = pa.get((E, t)), pa.get((G, t))
            same = int(pe is not None and pg is not None and pe == pg)
            if pe is None or pg is None:
                drops["missing_per_assert"] += 1
        rows.append(dict(E=E, G=G, t=t, diff=diff, own_wrong=own_wrong, same=same,
                         sc=r.get("selfcheck_verdict"), st=r.get("steelman_verdict")))
    drops["analysis_rows_before_verdict_filter"] = len(rows)

    tasks = sorted({r["t"] for r in rows})
    tix = {t: i for i, t in enumerate(tasks)}
    strata = [(E, G, k) for E in FAMS for G in FAMS if E != G for k in range(4)]
    six = {s: i for i, s in enumerate(strata)}
    rng = np.random.default_rng(SEED)
    W = np.stack([np.bincount(rng.integers(0, len(tasks), len(tasks)), minlength=len(tasks))
                  for _ in range(B)]).astype(float)

    res = {"drops": dict(drops), "n_tasks": len(tasks)}
    summ = []
    for arm, key in (("selfcheck", "sc"), ("steelman", "st")):
        ar = [r for r in rows if r[key] in V]
        res[f"{arm}_rows"] = len(ar)
        res[f"{arm}_unparsed_dropped"] = len(rows) - len(ar)
        out = {}
        # raw tables
        raw = {}
        for scope in ["all"] + FAMS:
            sub = ar if scope == "all" else [r for r in ar if r["E"] == scope]
            tab = {}
            for lvl in (1, 0):
                s = [r for r in sub if r["own_wrong"] == lvl]
                k = sum(r[key] == "ACCEPT" for r in s)
                tab["own_wrong" if lvl else "own_right"] = {
                    "n": len(s), "fa": k, "fa_pct": round(100 * k / len(s), 1) if s else None,
                    "wilson95": wilson(k, len(s))}
            raw[scope] = tab
        s = [r for r in ar if r["same"] == 1]
        k = sum(r[key] == "ACCEPT" for r in s)
        raw["all"]["same_way"] = {"n": len(s), "fa": k,
                                  "fa_pct": round(100 * k / len(s), 1) if s else None,
                                  "wilson95": wilson(k, len(s))}
        out["raw"] = raw

        def tensor(sel, expo):
            X = np.zeros((len(tasks), len(strata), 4))
            for r in sel:
                e = expo(r)
                if e is None:
                    continue
                fa = r[key] == "ACCEPT"
                j = (0 if fa else 1) if e else (2 if fa else 3)
                X[tix[r["t"]], six[(r["E"], r["G"], r["diff"])], j] += 1
            return X

        def estimate(X, fams=None):
            idx = [i for i, s in enumerate(strata) if fams is None or s[0] in fams]
            Xs = X[:, idx, :]
            pt = Xs.sum(0)
            point = float(mh(pt[None])[0])
            usable = int((((pt[:, 0] + pt[:, 1]) > 0) & ((pt[:, 2] + pt[:, 3]) > 0)).sum())
            bs = mh(np.einsum("bt,tsc->bsc", W, Xs))
            bs = np.sort(bs[~np.isnan(bs)])
            ci = [float(bs[int(0.025 * len(bs))]), float(bs[int(0.975 * len(bs)) - 1])]
            return {"mh_or": round(point, 3), "ci95": [round(ci[0], 3), round(ci[1], 3)],
                    "strata_used": usable, "strata_skipped": len(idx) - usable,
                    "bootstrap_valid": int(len(bs))}

        Xp = tensor(ar, lambda r: r["own_wrong"])
        Xs = tensor(ar, lambda r: None if (r["own_wrong"] == 1 and r["same"] != 1) else
                    (1 if r["own_wrong"] == 1 else 0))
        out["primary_own_wrong"] = estimate(Xp)
        out["secondary_same_way"] = estimate(Xs)
        out["per_evaluator_own_wrong"] = {E: estimate(Xp, {E}) for E in FAMS}
        # Robustness (added after the primary result): split by difficulty level.
        for name, ks in (("difficulty_0to2", {0, 1, 2}), ("difficulty_3", {3})):
            idx = [i for i, st in enumerate(strata) if st[2] in ks]
            out[f"own_wrong_{name}"] = round(float(mh(Xp[:, idx, :].sum(0)[None])[0]), 3)
        out["difficulty_distribution"] = dict(sorted(Counter(r["diff"] for r in ar).items()))
        lo, hi = out["primary_own_wrong"]["ci95"]
        out["decision"] = ("supported" if lo > 1 else "reversed" if hi < 1 else "not supported")
        res[arm] = out

    res["decision_rule_primary_selfcheck"] = res["selfcheck"]["decision"]
    (OUT / "shared_blindspot.json").write_text(json.dumps(res, indent=2))

    L = ["Shared-blind-spot test (cross-evaluation, MBPP)", ""]
    L.append(f"Row accounting: {json.dumps(res['drops'])}; tasks={res['n_tasks']}")
    for arm in ("selfcheck", "steelman"):
        o = res[arm]
        L.append("")
        L.append(f"[{arm}] rows={res[arm+'_rows']} (unparsed dropped {res[arm+'_unparsed_dropped']}); "
                 f"difficulty dist {o['difficulty_distribution']}")
        a = o["raw"]["all"]
        L.append(f"  raw FA: own code wrong {a['own_wrong']['fa_pct']}% (n={a['own_wrong']['n']}, "
                 f"{a['own_wrong']['wilson95']}) vs own code right {a['own_right']['fa_pct']}% "
                 f"(n={a['own_right']['n']}, {a['own_right']['wilson95']}); same way "
                 f"{a['same_way']['fa_pct']}% (n={a['same_way']['n']})")
        p = o["primary_own_wrong"]
        L.append(f"  PRIMARY MH OR own_wrong = {p['mh_or']} {p['ci95']} "
                 f"(strata used {p['strata_used']}, skipped {p['strata_skipped']}) -> {o['decision']}")
        s = o["secondary_same_way"]
        L.append(f"  SECONDARY MH OR same_way = {s['mh_or']} {s['ci95']} "
                 f"(strata used {s['strata_used']}, skipped {s['strata_skipped']})")
        L.append(f"  robustness: OR difficulty 0-2 = {o['own_wrong_difficulty_0to2']}, "
                 f"difficulty 3 = {o['own_wrong_difficulty_3']}")
        for E in FAMS:
            e = o["per_evaluator_own_wrong"][E]
            rw = o["raw"][E]
            L.append(f"    {E:14s} OR {e['mh_or']} {e['ci95']}  raw FA wrong/right "
                     f"{rw['own_wrong']['fa_pct']}%/{rw['own_right']['fa_pct']}%")
    L += ["", f"DECISION (primary, self-check): {res['decision_rule_primary_selfcheck']}"]
    (OUT / "shared_blindspot_summary.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
