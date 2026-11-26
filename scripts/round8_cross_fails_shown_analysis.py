#!/usr/bin/env python3
"""Scores the cross-family check (prediction in round8_cross_family_fails_shown.py).

Author family's self-check false acceptance on the errors that fail the shown assert (from the
generator-sees-assert run, last record per item) against the other four families' pooled false
acceptance on the same candidates. One-sided Fisher exact test (author higher), p < 0.05.
Also: by authoring family, and paired within candidate (author vs mean of others).
Output: data/mbpp_v2/analysis/round8_cross_fails_shown_analysis.json
"""
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path
from scipy.stats import fisher_exact
ROOT = Path(__file__).resolve().parents[1]; D = ROOT / "data/mbpp_v2"
cross = {}
for l in open(D / "round8_cross_fails_shown/records.jsonl"):
    r = json.loads(l); cross[(r["evaluator"], r["author"], r["task_id"])] = r
own = {}
for l in open(D / "round8_gen_assert/eval_records.jsonl"):
    r = json.loads(l)
    if r["arm"] == "selfcheck": own[(r["model"], r["task_id"])] = r
items = sorted({(a, t) for (_, a, t) in cross})
ok = lambda v: v in ("ACCEPT", "REJECT")
ak = an = ok_ = on = 0
by = defaultdict(lambda: [0, 0, 0, 0])
for a, t in items:
    o = own.get((a, t))
    if o and ok(o["verdict"]):
        ak += o["verdict"] == "ACCEPT"; an += 1; by[a][0] += o["verdict"] == "ACCEPT"; by[a][1] += 1
    for (e, a2, t2), r in cross.items():
        if a2 == a and t2 == t and ok(r["verdict"]):
            ok_ += r["verdict"] == "ACCEPT"; on += 1; by[a][2] += r["verdict"] == "ACCEPT"; by[a][3] += 1
odds, p = fisher_exact([[ak, an - ak], [ok_, on - ok_]], alternative="greater")
by_eval = defaultdict(lambda: [0, 0])
for (e, a, t), r in cross.items():
    if ok(r["verdict"]): by_eval[e][0] += r["verdict"] == "ACCEPT"; by_eval[e][1] += 1
res = {"candidates": len(items), "author": [ak, an, round(ak / an, 4)], "others": [ok_, on, round(ok_ / on, 4)],
       "fisher_one_sided_p": p, "odds_ratio": odds, "self_specific_supported": bool(p < 0.05),
       "by_author": {a: {"author": v[:2], "others": v[2:]} for a, v in by.items()},
       "by_evaluator": {e: v for e, v in by_eval.items()},
       "unparsed_cross": sum(1 for r in cross.values() if not ok(r["verdict"]))}
(D / "analysis/round8_cross_fails_shown_analysis.json").write_text(json.dumps(res, indent=1, default=str) + "\n")
print(json.dumps(res, indent=1, default=str))
