#!/usr/bin/env python3
"""Signal-detection view of the MBPP verdicts (paper Sec. IV-C, Table "sdt").

For each family and arm, on the paired set: sensitivity d' = z(tau) - z(kappa) and
criterion c = -(z(tau) + z(kappa)) / 2, where z is the standard normal quantile.
Higher c means a less lenient (more rejecting) evaluator. Steelman-minus-self-check
changes in d' and c get 95% stratified paired bootstrap intervals (10,000 resamples,
seed 20261122; candidates resampled within the T=0 and T=1 strata).

Reads data/mbpp_v2/production/derived_records.jsonl. Writes
data/mbpp_v2/analysis/signal_detection.json. No model calls.
"""
import json
import random
from pathlib import Path
from statistics import NormalDist

ROOT = Path(__file__).resolve().parents[1]
FAMS = ["ds-v4-flash", "glm-5.3-flash", "qwen38-flash", "gpt-5.4", "gpt-6-luna"]
z = NormalDist().inv_cdf


def sdt(p0, p1, i):
    k = sum(x[i] for x in p0) / len(p0)
    t = sum(x[i] for x in p1) / len(p1)
    k = min(max(k, 0.5 / len(p0)), 1 - 0.5 / len(p0))  # guard against 0/1 in resamples
    t = min(max(t, 0.5 / len(p1)), 1 - 0.5 / len(p1))
    return z(t) - z(k), -(z(t) + z(k)) / 2


rows = [json.loads(l) for l in open(ROOT / "data/mbpp_v2/production/derived_records.jsonl")]
rng = random.Random(20261122)
out = {}
for f in FAMS:
    rs = [r for r in rows if r["model"] == f and r["T"] in (0, 1)
          and r["selfcheck_extraction"] == "parsed" and r["steelman_extraction"] == "parsed"]
    pair = lambda r: (r["selfcheck_verdict"] == "ACCEPT", r["steelman_verdict"] == "ACCEPT")
    p0 = [pair(r) for r in rs if r["T"] == 0]
    p1 = [pair(r) for r in rs if r["T"] == 1]
    (d_sc, c_sc), (d_st, c_st) = sdt(p0, p1, 0), sdt(p0, p1, 1)
    bd, bc = [], []
    for _ in range(10000):
        q0 = [rng.choice(p0) for _ in p0]
        q1 = [rng.choice(p1) for _ in p1]
        (a, b), (c, d) = sdt(q0, q1, 0), sdt(q0, q1, 1)
        bd.append(c - a)
        bc.append(d - b)
    bd.sort(); bc.sort()
    out[f] = {
        "d_prime": {"selfcheck": round(d_sc, 3), "steelman": round(d_st, 3)},
        "criterion_c": {"selfcheck": round(c_sc, 3), "steelman": round(c_st, 3)},
        "delta_d_prime": round(d_st - d_sc, 3), "delta_d_prime_ci95": [round(bd[250], 3), round(bd[9749], 3)],
        "delta_c": round(c_st - c_sc, 3), "delta_c_ci95": [round(bc[250], 3), round(bc[9749], 3)],
    }
    print(f, out[f])
(ROOT / "data/mbpp_v2/analysis/signal_detection.json").write_text(json.dumps(out, indent=1) + "\n")
