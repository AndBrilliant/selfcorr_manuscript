"""Verdict information at an unbiased threshold, and reasoning tokens per call (Sec. IV).

(1) For each family and arm, d' is computed from the paired acceptance rates, and the
    verdict information I(T;A|G) is recomputed for the same d' at an unbiased threshold
    (c = 0: tau = Phi(d'/2), kappa = Phi(-d'/2)) under the equal-variance model, at the
    family's observed base rate.
(2) Median reasoning tokens per production evaluator call, from the provider usage
    records (completion_tokens_details.reasoning_tokens).
Computed after data collection, in response to review.
Writes data/mbpp_v2/analysis/threshold_and_reasoning.json.
"""
import ast, json, math, statistics as st
from pathlib import Path
from statistics import NormalDist

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data/mbpp_v2"
N = NormalDist()
h = lambda p: 0.0 if p <= 0 or p >= 1 else -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


def mi(rho, tau, kap):
    pa = rho * tau + (1 - rho) * kap
    return h(pa) - (rho * h(tau) + (1 - rho) * h(kap))


prod = [json.loads(l) for l in open(D / "production/derived_records.jsonl") if l.strip()]
fams = ["gpt-5.4", "ds-v4-flash", "glm-5.3-flash", "qwen38-flash", "gpt-6-luna"]
out = {"unbiased_threshold": {}, "reasoning_tokens_median": {}}
for m in fams:
    rs = [r for r in prod if r["model"] == m and r["T"] in (0, 1)
          and r["selfcheck_extraction"] == "parsed" and r["steelman_extraction"] == "parsed"]
    rho = sum(r["T"] for r in rs) / len(rs)
    t1 = [r for r in rs if r["T"] == 1]; t0 = [r for r in rs if r["T"] == 0]
    for arm in ("selfcheck", "steelman"):
        tau = sum(r[f"{arm}_verdict"] == "ACCEPT" for r in t1) / len(t1)
        kap = sum(r[f"{arm}_verdict"] == "ACCEPT" for r in t0) / len(t0)
        dp = N.inv_cdf(tau) - N.inv_cdf(kap)
        obs, unb = mi(rho, tau, kap), mi(rho, N.cdf(dp / 2), N.cdf(-dp / 2))
        out["unbiased_threshold"][f"{m}|{arm}"] = {
            "d_prime": round(dp, 3), "I_observed_millibits": round(1000 * obs, 1),
            "I_unbiased_millibits": round(1000 * unb, 1), "ratio": round(unb / obs, 2),
            "unbiased_pct_of_H": round(100 * unb / h(rho), 2)}
R = {}
for l in open(D / "production/events.jsonl"):
    e = json.loads(l)
    if e["stage"] not in ("selfcheck", "steelman") or not e.get("usage"):
        continue
    u = ast.literal_eval(e["usage"]) if isinstance(e["usage"], str) else e["usage"]
    r = (u.get("completion_tokens_details") or {}).get("reasoning_tokens")
    if r is not None:
        R.setdefault(f"{e['model']}|{e['stage']}", []).append(r)
out["reasoning_tokens_median"] = {k: st.median(v) for k, v in sorted(R.items())}
u = out["unbiased_threshold"].values()
out["summary"] = {"I_unbiased_range_millibits": [min(x["I_unbiased_millibits"] for x in u), max(x["I_unbiased_millibits"] for x in u)],
                  "unbiased_pct_of_H_range": [min(x["unbiased_pct_of_H"] for x in u), max(x["unbiased_pct_of_H"] for x in u)],
                  "ratio_range": [min(x["ratio"] for x in u), max(x["ratio"] for x in u)]}
print(json.dumps(out["summary"]), json.dumps(out["reasoning_tokens_median"]))
(D / "analysis/threshold_and_reasoning.json").write_text(json.dumps(out, indent=1))
