#!/usr/bin/env python3
"""Supplementary MBPP statistics reported in the paper that are not in mbpp_v2_analysis.json.

- steelman reduction in false acceptance and true-acceptance cost, unrounded (Abstract, Sec. IV-C)
- verdict-accuracy discordance (verdict scored correct when it matches T), exact McNemar (Sec. IV-C)
- precision of accepted candidates and lift over the paired base rate (Sec. IV-C)
- completion sensitivity of the primary incorrect-stratum McNemar contrast over every
  completion of the missing pairs consistent with the observed verdicts, with five-family
  Holm adjustment (Appendix A)

Reads data/mbpp_v2/production/derived_records.jsonl only. No model calls.
Writes data/mbpp_v2/analysis/supplementary_stats.json.
"""
import json
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FAMS = ["gpt-5.4", "ds-v4-flash", "glm-5.3-flash", "qwen38-flash", "gpt-6-luna"]


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def holm(pvals):
    order = sorted(pvals, key=pvals.get)
    m, out, run = len(order), {}, 0.0
    for i, k in enumerate(order):
        run = max(run, min(1.0, (m - i) * pvals[k]))
        out[k] = run
    return out


rows = [json.loads(l) for l in open(ROOT / "data/mbpp_v2/production/derived_records.jsonl")]
acc = lambda r, arm: r[f"{arm}_verdict"] == "ACCEPT"
parsed = lambda r, arm: r.get(f"{arm}_extraction") == "parsed"

result = {"families": {}}
primary = {}
missing = {}
for f in FAMS:
    fr = [r for r in rows if r["model"] == f and r["T"] in (0, 1)]
    paired = [r for r in fr if parsed(r, "selfcheck") and parsed(r, "steelman")]
    t0 = [r for r in paired if r["T"] == 0]
    t1 = [r for r in paired if r["T"] == 1]
    k = {a: sum(acc(r, a) for r in t0) for a in ("selfcheck", "steelman")}
    t = {a: sum(acc(r, a) for r in t1) for a in ("selfcheck", "steelman")}
    b = sum(acc(r, "selfcheck") and not acc(r, "steelman") for r in t0)
    c = sum(acc(r, "steelman") and not acc(r, "selfcheck") for r in t0)
    primary[f] = (b, c)
    ok = lambda r, a: acc(r, a) == (r["T"] == 1)
    fixes = sum(ok(r, "steelman") and not ok(r, "selfcheck") for r in paired)
    intro = sum(ok(r, "selfcheck") and not ok(r, "steelman") for r in paired)
    base = len(t1) / len(paired)
    prec = {a: sum(r["T"] == 1 for r in paired if acc(r, a)) / sum(acc(r, a) for r in paired)
            for a in ("selfcheck", "steelman")}
    missing[f] = [r for r in fr if r["T"] == 0 and not (parsed(r, "selfcheck") and parsed(r, "steelman"))]
    result["families"][f] = {
        "n_incorrect_paired": len(t0), "n_correct_paired": len(t1),
        "false_accept": k, "true_accept": t,
        "kappa_drop_points": round(100 * (k["selfcheck"] - k["steelman"]) / len(t0), 3),
        "tau_cost_points": round(100 * (t["selfcheck"] - t["steelman"]) / len(t1), 3),
        "primary_discordance_sc_only_st_only": [b, c], "primary_mcnemar_p": mcnemar(b, c),
        "verdict_accuracy_discordance_st_fixes_st_introduces": [fixes, intro],
        "verdict_accuracy_mcnemar_p": mcnemar(fixes, intro),
        "base_rate_correct_pct": round(100 * base, 2),
        "precision_accepted_pct": {a: round(100 * v, 2) for a, v in prec.items()},
        "lift_over_base_points": {a: round(100 * (v - base), 2) for a, v in prec.items()},
        "missing_incorrect_pairs": len(missing[f]),
    }

pp = {f: mcnemar(*primary[f]) for f in FAMS}
result["primary_holm"] = holm(pp)


def completions(f):
    """(b, c) totals over every completion of the missing T=0 pairs consistent with observed verdicts."""
    opts = [primary[f]]
    for r in missing[f]:
        sc = [acc(r, "selfcheck")] if parsed(r, "selfcheck") else [True, False]
        st = [acc(r, "steelman")] if parsed(r, "steelman") else [True, False]
        new = set()
        for b, c in opts:
            for x in sc:
                for y in st:
                    new.add((b + (x and not y), c + (y and not x)))
        opts = sorted(new)
    return opts


sens = {}
for f in FAMS:
    best = worst = None
    for b, c in completions(f):
        ph = holm({**pp, f: mcnemar(b, c)})[f]
        if best is None or ph < best[2]:
            best = (b, c, ph)
        if worst is None or ph > worst[2]:
            worst = (b, c, ph)
    sens[f] = {"best": {"b": best[0], "c": best[1], "p_holm": best[2]},
               "worst": {"b": worst[0], "c": worst[1], "p_holm": worst[2]}}
result["completion_sensitivity"] = sens

out = ROOT / "data/mbpp_v2/analysis/supplementary_stats.json"
out.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
for f in FAMS:
    d = result["families"][f]
    print(f, d["kappa_drop_points"], d["tau_cost_points"], d["verdict_accuracy_discordance_st_fixes_st_introduces"],
          round(d["verdict_accuracy_mcnemar_p"], 3), d["lift_over_base_points"], sens[f])
