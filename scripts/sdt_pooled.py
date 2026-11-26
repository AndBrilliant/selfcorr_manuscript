#!/usr/bin/env python3
"""Pooled change in sensitivity d' across the five families (paper, Section IV-C).

Inverse-variance pooling of the per-family steelman-minus-self-check change in d', with each
family's standard error approximated from its 95% bootstrap interval as (upper - lower) / 3.92.
Reads data/mbpp_v2/analysis/signal_detection.json (run signal_detection.py first).

Usage: python3 scripts/sdt_pooled.py
Writes: data/mbpp_v2/analysis/sdt_pooled.json
"""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "mbpp_v2" / "analysis" / "signal_detection.json"
OUT = ROOT / "data" / "mbpp_v2" / "analysis" / "sdt_pooled.json"


def pool(d, key, ci_key):
    num = den = 0.0
    for fam in d.values():
        lo, hi = fam[ci_key]
        se = (hi - lo) / 3.92
        w = 1 / se ** 2
        num += w * fam[key]
        den += w
    est, se = num / den, math.sqrt(1 / den)
    # Cochran's Q and I^2 across families (heterogeneity of the per-family changes)
    q = sum((1 / ((f[ci_key][1] - f[ci_key][0]) / 3.92) ** 2) * (f[key] - est) ** 2 for f in d.values())
    k = len(d) - 1
    i2 = max(0.0, (q - k) / q) * 100 if q > 0 else 0.0
    p = math.exp(-q / 2) * sum((q / 2) ** i / math.factorial(i) for i in range(k // 2)) if k % 2 == 0 else None
    from statistics import NormalDist
    nd = NormalDist()
    # Two one-sided tests (TOST) against symmetric margins; margins chosen after the data.
    tost = {str(m): round(max(1 - nd.cdf((est + m) / se), nd.cdf((est - m) / se)), 4) for m in (0.15, 0.2, 0.25)}
    return {"pooled": round(est, 3), "ci90": [round(est - 1.645 * se, 3), round(est + 1.645 * se, 3)],
            "tost_p_by_margin": tost, "ci95": [round(est - 1.96 * se, 3), round(est + 1.96 * se, 3)],
            "mde80_pooled": round(2.8 * se, 3),
            "mde80_per_family": {k: round(2.8 * (f[ci_key][1] - f[ci_key][0]) / 3.92, 3) for k, f in d.items()},
            "cochran_q": round(q, 2), "df": k, "q_p": None if p is None else round(p, 3), "i2_pct": round(i2, 1)}


def main():
    d = json.loads(SRC.read_text(encoding="utf-8"))
    out = {"delta_d_prime": pool(d, "delta_d_prime", "delta_d_prime_ci95"),
           "delta_c": pool(d, "delta_c", "delta_c_ci95"),
           "method": "inverse-variance pooling; per-family SE = (upper - lower) / 3.92 of the 95% bootstrap interval"}
    OUT.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
