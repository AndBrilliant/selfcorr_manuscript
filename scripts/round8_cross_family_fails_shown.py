#!/usr/bin/env python3
"""Round 8 follow-up: do other families accept the errors that fail the shown assert?
Specified before the run.

In the generator-sees-assert run, the author family's self-check accepted 41 of 124 errors
that fail the very assert shown to both generator and evaluator (33.1%). Those 124 are a
selected set (errors the generator made while seeing the example), so the rate could reflect
case difficulty rather than a self-specific blind spot. Here each of the other four families
judges the same frozen candidates with the same prompt (self-check, assert shown), once.

Prediction: if the blind spot is self-specific, the other families' pooled false acceptance
on these candidates is lower than the author family's 41/124 (one-sided Fisher exact test,
p < 0.05). If it is similar, the 33% reflects case difficulty, and the paper says so.

Usage:
  python scripts/round8_cross_family_fails_shown.py --show 1
  python scripts/round8_cross_family_fails_shown.py --cap 3
Output: data/mbpp_v2/round8_cross_fails_shown/
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.round8_common as rc  # noqa: E402
from scripts.round8_common import pilot  # noqa: E402
import scripts.round8_generator_sees_assert as g  # noqa: E402

OUT = ROOT / "data/mbpp_v2/round8_cross_fails_shown"
pilot.SCHEMA_VERSION = "round8-cross-fails-shown/1"
SRC = ROOT / "data/mbpp_v2/round8_gen_assert"


def targets():
    """Incorrect generator-sees-assert candidates that fail the shown assert."""
    tasks = pilot.load_task_map()
    cand = [json.loads(l) for l in open(SRC / "candidate_records.jsonl")]
    inc = [c for c in cand if c["T"] == 0]
    from multiprocessing import Pool
    with Pool(16) as p:
        fails = p.map(rc.assert_fails, [(c["code"], rc.split_parts(tasks[c["task_id"]]["test"])[0], c["shown_assert"]) for c in inc], chunksize=4)
    return [c for c, f in zip(inc, fails) if f]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", type=int, default=0)
    ap.add_argument("--cap", type=float, default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--min-credits", type=float, default=5.0)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    tasks = pilot.load_task_map()
    tg = targets()
    assert len(tg) == 126, f"expected 126 candidates failing the shown assert, found {len(tg)}"
    work = [(e, c) for c in tg for e in rc.FAMILIES if e != c["model"]]
    if a.show:
        e, c = work[0]
        print("evaluator", e, "author", c["model"], c["task_id"], "\n", g.eval_prompt("selfcheck", tasks[c["task_id"]], c["code"]))
        print(f"\n{len(work)} calls")
        return
    if a.cap is None:
        raise SystemExit("--cap is required for any run that makes calls")
    rec = OUT / "records.jsonl"
    done = rc.done_keys(rec, ("evaluator", "author", "task_id"))
    work = [(e, c) for e, c in work if (e, c["model"], c["task_id"]) not in done]
    cap = rc.Cap(a.cap, a.min_credits)

    def fn(item):
        e, c = item
        if pilot.sha256_text(c["code"]) != c["code_sha256"]:
            raise RuntimeError("frozen candidate hash mismatch")
        ev = pilot.call_model(e, g.eval_prompt("selfcheck", tasks[c["task_id"]], c["code"]),
                              "selfcheck_cross_fails_shown", c["task_id"], OUT, 3)
        cap.add(ev)
        v = pilot.strict_verdict(ev.get("raw_response")) if not ev.get("error") else None
        r = {"evaluator": e, "author": c["model"], "task_id": c["task_id"], "T": c["T"],
             "code_sha256": c["code_sha256"], "verdict": v, "event_id": ev.get("event_id"),
             "prompt_sha256": ev.get("prompt_sha256"), "finish_reason": ev.get("finish_reason"),
             "error": ev.get("error")}
        with rc.LOCK:
            pilot.append_jsonl(rec, r)

    rc.run_work(work, fn, cap, a.workers, OUT, "cross_fails_shown")


if __name__ == "__main__":
    main()
