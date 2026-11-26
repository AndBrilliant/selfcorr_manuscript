#!/usr/bin/env python3
"""Fixed-denominator accounting for the MBPP paired analysis (Appendix A).

Regenerates the exclusion accounting table and the missing-record
extreme-imputation bounds quoted in the paper, directly from the
production ledger. Deterministic; no randomness, no network.

Exclusion categories (disjoint, exhaustive over the 4,735 pairs):
  generation ceiling  - generation call ended at the provider output ceiling (T unknown)
  evaluator ceiling   - an evaluator call ended at the output ceiling
  grammar             - evaluator call completed but carried no strict terminal verdict
Pairs in none of these categories are paired.
"""
import json
from collections import defaultdict
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROD = ROOT / "data/mbpp_v2/production"


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    x = min(b, c)
    return min(1.0, 2.0 * sum(comb(n, i) for i in range(x + 1)) / 2**n)


def holm_adjust(pvals):
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    adj = [0.0] * len(pvals)
    running = 0.0
    m = len(pvals)
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adj[i] = running
    return adj


def main():
    records = [
        json.loads(l)
        for l in (PROD / "derived_records.jsonl").read_text().splitlines()
        if l.strip()
    ]

    # evaluator-arm finish reasons from the event ledger
    length_events = set()
    for l in (PROD / "events.jsonl").read_text().splitlines():
        if not l.strip():
            continue
        e = json.loads(l)
        if e["stage"] in ("selfcheck", "steelman") and e["finish_reason"] == "length":
            length_events.add(e["event_id"])

    fams = defaultdict(lambda: {
        "gen_ceiling": 0,
        "ceil_pairs": 0, "gram_pairs": 0,
        "sc_arm": 0, "st_arm": 0, "both_arm": 0,
        "missing_T0": 0,
    })
    paired_far = defaultdict(lambda: {"b": 0, "c": 0})

    for r in records:
        f = fams[r["model"]]
        if r.get("generation_finish_reason") == "length" and r.get("T") is None:
            f["gen_ceiling"] += 1
            continue

        bad = []          # arms whose verdict is unusable (any cause)
        ceiling = False
        for arm, eid_key in (
            ("selfcheck", "selfcheck_event_id"),
            ("steelman", "steelman_event_id"),
        ):
            if r.get(f"{arm}_extraction") == "parsed":
                continue
            if r.get(f"{arm}_extraction") == "not_run":
                continue
            bad.append(arm)
            if r.get(eid_key) in length_events:
                ceiling = True

        if not bad:
            if r["T"] == 0:
                sc, st = r["selfcheck_verdict"], r["steelman_verdict"]
                if sc == "ACCEPT" and st == "REJECT":
                    paired_far[r["model"]]["b"] += 1
                elif sc == "REJECT" and st == "ACCEPT":
                    paired_far[r["model"]]["c"] += 1
            continue

        # excluded pair with known ground truth
        if r["T"] == 0:
            f["missing_T0"] += 1
        if len(bad) == 2:
            f["both_arm"] += 1
        elif bad[0] == "selfcheck":
            f["sc_arm"] += 1
        else:
            f["st_arm"] += 1
        if ceiling:
            f["ceil_pairs"] += 1
        else:
            f["gram_pairs"] += 1

    names = list(fams)
    total = defaultdict(int)
    print("family paired gen_ceiling eval_ceiling grammar arms")
    for m in names:
        f = fams[m]
        paired = sum(
            1 for r in records
            if r["model"] == m and r.get("T") in (0, 1)
            and r.get("selfcheck_extraction") == "parsed"
            and r.get("steelman_extraction") == "parsed"
        )
        arm_str = ", ".join(
            x for x in (
                f"sc:{f['sc_arm']}" if f["sc_arm"] else "",
                f"st:{f['st_arm']}" if f["st_arm"] else "",
                f"both:{f['both_arm']}" if f["both_arm"] else "",
            ) if x
        )
        print(f"{m} {paired} {f['gen_ceiling']} {f['ceil_pairs']} {f['gram_pairs']} {arm_str}")
        total["paired"] += paired
        total["gen"] += f["gen_ceiling"]
        total["ceil"] += f["ceil_pairs"]
        total["gram"] += f["gram_pairs"]
        total["excl"] += f["gen_ceiling"] + f["ceil_pairs"] + f["gram_pairs"]
    print(f"TOTAL {total['paired']} {total['gen']} {total['ceil']} {total['gram']} {total['excl']} excl")

    # extreme-imputation bounds on the paired FAR McNemar contrast
    b0 = {m: paired_far[m]["b"] for m in names}
    c0 = {m: paired_far[m]["c"] for m in names}
    m0 = {m: fams[m]["missing_T0"] for m in names}

    def adjusted(adversarial):
        ps = [mcnemar_exact(b0[m], c0[m] + (m0[m] if adversarial else 0)) for m in names]
        return ps, holm_adjust(ps)

    print("\nunconstrained extreme imputation (every missing T=0 pair maximally adversarial,\n"
          " ignoring observed verdicts; the paper reports the completions consistent with the\n"
          " observed verdicts, see scripts/mbpp_v2_supplementary_stats.py):")
    _, adj = adjusted(True)
    for m, a in zip(names, adj):
        print(f"  {m} worst_holm_p={a:.4g}")

    print("\npartial adversarial grid (other families at observed values):")
    obs_adj = adjusted(False)[1]
    for m in sorted(names, key=lambda x: obs_adj[names.index(x)])[:2]:
        grid = ", ".join(
            f"{k} adv -> {holm_adjust([mcnemar_exact(b0[x], c0[x] + (k if x == m else 0)) for x in names])[names.index(m)]:.4g}"
            for k in range(0, m0[m] + 1)
        )
        print(f"  {m} (missing T=0 pairs: {m0[m]}): {grid}")


if __name__ == "__main__":
    main()
