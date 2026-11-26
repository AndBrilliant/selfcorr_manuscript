#!/usr/bin/env python3
"""Round 8, run 2: interleaved self-check vs steelman, all five families, one replicate.
Specified before the run.

Production ran the two arms in separate batches, and identical prompts moved acceptance
by up to 2.4 points between batches. Here both arms, for every frozen production
candidate of every family, are shuffled together with a fixed seed and sent through one
worker pool, so batch drift falls equally on both arms. Prompts are the production
prompts exactly (mbpp_v2_pilot.selfcheck_prompt / steelman_prompt); all families via
OpenRouter.

Analysis: per family and pooled, steelman minus self-check in false acceptance (exact
McNemar), d' and c (task-clustered bootstrap), on candidates parsed in both arms.

Usage:
  python3 scripts/round8_interleaved_sc_st.py --show 2            # print prompts, no calls
  python3 scripts/round8_interleaved_sc_st.py --limit 10 --cap 3  # dry run
  python3 scripts/round8_interleaved_sc_st.py --cap 30            # full run (resumable)
Output: data/mbpp_v2/round8_interleaved/
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.round8_common as rc  # noqa: E402
from scripts.round8_common import pilot  # noqa: E402

OUT = ROOT / "data/mbpp_v2/round8_interleaved"
pilot.SCHEMA_VERSION = "round8-interleaved/1"
ARMS = {"selfcheck": pilot.selfcheck_prompt, "steelman": pilot.steelman_prompt}


def render(arm, task, code):
    prompt = ARMS[arm](task, code)
    assert code and code.strip(), "empty candidate code"
    assert code in prompt, "candidate code missing from prompt"
    assert "reference assertion" not in prompt and "assert " not in task["prompt"], \
        "production prompts must not show an assert"
    return prompt


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
    keys = sorted((k for k, c in cands.items() if k[0] in rc.FAMILIES and c.get("code")),
                  key=lambda k: (k[0], int(k[1].split("/")[1])))
    if a.show:
        for k in [k for k in keys if cands[k]["T"] == 0][:a.show]:
            for arm in ARMS:
                print("=" * 70, k, arm, "\n", render(arm, tasks[k[1]], cands[k]["code"]))
        return
    if a.cap is None:
        raise SystemExit("--cap is required for any run that makes calls")
    rec = OUT / "records.jsonl"
    done = rc.done_keys(rec, ("model", "task_id", "arm"))
    work = []
    for m in rc.FAMILIES:
        ts = [k for k in keys if k[0] == m]
        if a.limit:
            ts = ts[:a.limit]
        work += [(k, arm) for k in ts for arm in ARMS if (k[0], k[1], arm) not in done]
    cap = rc.Cap(a.cap, a.min_credits)

    def fn(item):
        (m, t), arm = item
        c = cands[(m, t)]
        prompt = render(arm, tasks[t], c["code"])
        ev = pilot.call_model(m, prompt, f"{arm}_interleaved", t, OUT, 3)
        cap.add(ev)
        v = pilot.strict_verdict(ev.get("raw_response")) if not ev.get("error") else None
        r = {"model": m, "task_id": t, "arm": arm, "T": c["T"], "code_sha256": c["code_sha256"],
             "verdict": v, "event_id": ev.get("event_id"), "prompt_sha256": ev.get("prompt_sha256"),
             "finish_reason": ev.get("finish_reason"), "error": ev.get("error")}
        with rc.LOCK:
            pilot.append_jsonl(rec, r)

    rc.run_work(work, fn, cap, a.workers, OUT, "interleaved")


if __name__ == "__main__":
    main()
