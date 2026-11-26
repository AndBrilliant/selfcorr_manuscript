#!/usr/bin/env python3
"""Within-class correlation of repeated verdicts (P5 in statistical-error terms).

For each family x arm x correctness class, the three temperature-0 verdicts on a candidate
(production call r1, repeats r2, r3) are treated as k=3 binary measurements of the same item.
rho is the one-way ANOVA intraclass correlation, (MSB - MSW) / (MSB + (k-1) MSW), on candidates
with all three verdicts parsed; N_eff = k / (1 + (k-1) rho) is the number of independent verdicts
the three are worth. Interval: candidate bootstrap (1,000 resamples, seed 20261125). Classes with
almost no variance (all verdicts identical) give an undefined rho and are reported as such.
Output: data/mbpp_v2/analysis/repeated_icc.json
"""
from __future__ import annotations
import json, random
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; D = ROOT / "data/mbpp_v2"
FAMS = ["gpt-5.4", "ds-v4-flash", "glm-5.3-flash", "qwen38-flash", "gpt-6-luna"]
prod = {(r["model"], r["task_id"]): r for r in map(json.loads, open(D / "production/derived_records.jsonl"))}
rep = {(r["model"], r["task_id"]): r for r in map(json.loads, open(D / "repeated_eval/repeated_records.jsonl")) if r.get("production_candidate", True)}
def icc(items, k=3):
    n = len(items)
    if n < 3: return None
    grand = sum(map(sum, items)) / (n * k)
    msb = k * sum((sum(x) / k - grand) ** 2 for x in items) / (n - 1)
    msw = sum((v - sum(x) / k) ** 2 for x in items for v in x) / (n * (k - 1))
    den = msb + (k - 1) * msw
    return None if den == 0 else (msb - msw) / den
out = {}
for m in FAMS:
    for arm in ("selfcheck", "steelman"):
        for T in (0, 1):
            items = []
            for key, p in prod.items():
                if key[0] != m or p["T"] != T or key not in rep: continue
                r = rep[key]
                vs = [p.get(f"{arm}_verdict") if p.get(f"{arm}_extraction") == "parsed" else None,
                      r.get(f"{arm}_r2_verdict") if r.get(f"{arm}_r2_extraction") == "parsed" else None,
                      r.get(f"{arm}_r3_verdict") if r.get(f"{arm}_r3_extraction") == "parsed" else None]
                if all(v in ("ACCEPT", "REJECT") for v in vs):
                    items.append([v == "ACCEPT" for v in vs])
            est = icc(items)
            rng = random.Random(20261125); bs = []
            for _ in range(1000):
                b = icc([rng.choice(items) for _ in items])
                if b is not None: bs.append(b)
            bs.sort()
            ci = [round(bs[int(.025 * (len(bs) - 1))], 3), round(bs[int(.975 * (len(bs) - 1))], 3)] if bs else None
            out[f"{m}|{arm}|T={T}"] = {"n": len(items), "rho": None if est is None else round(est, 3), "rho_ci95": ci,
                                       "N_eff_of_3": None if est is None else round(3 / (1 + 2 * est), 2),
                                       "all_identical_share": round(sum(len(set(x)) == 1 for x in items) / len(items), 3) if items else None}
(D / "analysis/repeated_icc.json").write_text(json.dumps(out, indent=1) + "\n")
for k, v in out.items(): print(k, v)
