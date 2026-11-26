#!/usr/bin/env python3
"""Round 8 pilot: who wrote the code? (attribution framing). Specified before the run.

The production prompts never say who wrote the candidate. This pilot adds one sentence
after the first line of the production self-check prompt, and nothing else:
  none   (production wording)
  self   "You wrote this solution earlier."
  user   "The user wrote this solution."
All three framings of every sampled candidate are shuffled together into one batch.

Sample, per family (seed 20261125): every incorrect candidate both auditors judged
knowable from the description; 40 random incorrect candidates both judged not knowable;
40 random correct candidates. About 1,350 calls.

Prediction (written before the run): attribution adds no information about correctness,
so it may move the acceptance threshold in either direction (deference to the user,
or favouring one's own code) but should not change sensitivity d'. No prediction on the
direction of the threshold shift.

Usage:
  python scripts/round8_attribution_pilot.py --show 1
  python scripts/round8_attribution_pilot.py --limit 3 --cap 0.5   # dry run
  python scripts/round8_attribution_pilot.py --cap 6               # full pilot
Output: data/mbpp_v2/round8_attribution/
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.round8_common as rc  # noqa: E402
from scripts.round8_common import pilot  # noqa: E402

OUT = ROOT / "data/mbpp_v2/round8_attribution"
pilot.SCHEMA_VERSION = "round8-attribution/1"
FIRST = "You are evaluating one immutable Python candidate solution."
FRAMES = {"none": None, "self": "You wrote this solution earlier.", "user": "The user wrote this solution."}
SEED = 20261125


def render(frame, task, code):
    base = pilot.selfcheck_prompt(task, code)
    assert base.startswith(FIRST + "\n"), "production prompt changed"
    assert code and code.strip() and code in base, "candidate code missing"
    if FRAMES[frame] is None:
        return base
    out = base.replace(FIRST, FIRST + " " + FRAMES[frame], 1)
    assert out.count(FRAMES[frame]) == 1
    return out


def sample(cands):
    labels = {}
    for r in map(json.loads, open(ROOT / "data/mbpp_v2/crossmodel_audit_all/labels.jsonl")):
        labels.setdefault((r["generator"], r["task_id"]), {})[r["auditor"]] = r["label"]
    agreed = {k: v.popitem()[1] for k, v in labels.items() if len(v) == 2 and len(set(v.values())) == 1}
    rng = random.Random(SEED)
    keys = []
    for m in rc.FAMILIES:
        fam = sorted((k for k, c in cands.items() if k[0] == m and c.get("code")),
                     key=lambda k: int(k[1].split("/")[1]))
        know = [k for k in fam if cands[k]["T"] == 0 and agreed.get(k) == "KNOWABLE"]
        hidden = [k for k in fam if cands[k]["T"] == 0 and agreed.get(k) == "NOT_KNOWABLE"]
        correct = [k for k in fam if cands[k]["T"] == 1]
        keys += know + rng.sample(hidden, 40) + rng.sample(correct, 40)
    return keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--cap", type=float, default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--min-credits", type=float, default=5.0)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    tasks, cands = pilot.load_task_map(), rc.load_candidates()
    keys = sample(cands)
    if a.show:
        k = keys[0]
        for f in FRAMES:
            print("=" * 70, k, f, "\n", render(f, tasks[k[1]], cands[k]["code"])[:400])
        return
    if a.cap is None:
        raise SystemExit("--cap is required for any run that makes calls")
    (OUT / "sample.json").write_text(json.dumps([list(k) for k in keys]) + "\n")
    rec = OUT / "records.jsonl"
    done = rc.done_keys(rec, ("model", "task_id", "frame"))
    if a.limit:
        keys = [k for m in rc.FAMILIES for k in [k for k in keys if k[0] == m][:a.limit]]
    work = [(k, f) for k in keys for f in FRAMES if (k[0], k[1], f) not in done]
    cap = rc.Cap(a.cap, a.min_credits)

    def fn(item):
        (m, t), f = item
        c = cands[(m, t)]
        ev = pilot.call_model(m, render(f, tasks[t], c["code"]), f"selfcheck_attr_{f}", t, OUT, 3)
        cap.add(ev)
        v = pilot.strict_verdict(ev.get("raw_response")) if not ev.get("error") else None
        r = {"model": m, "task_id": t, "frame": f, "T": c["T"], "code_sha256": c["code_sha256"],
             "verdict": v, "event_id": ev.get("event_id"), "prompt_sha256": ev.get("prompt_sha256"),
             "finish_reason": ev.get("finish_reason"), "error": ev.get("error")}
        with rc.LOCK:
            pilot.append_jsonl(rec, r)

    rc.run_work(work, fn, cap, a.workers, OUT, "attribution")


if __name__ == "__main__":
    main()
