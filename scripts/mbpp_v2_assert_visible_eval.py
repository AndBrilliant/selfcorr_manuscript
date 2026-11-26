#!/usr/bin/env python3
"""Evaluator-only one-assert-visible control (paper Sec. IV-G).

Each family's frozen production candidate is re-evaluated under the self-check and
steelman prompts, with the task description extended by the task's first reference
assert. Candidates and their correctness labels are unchanged, so any change in
acceptance is due to the visible example alone.

Every prompt is checked to contain the assert before it is sent; a prompt without it
stops the run. (The earlier generator-and-evaluator control, mbpp_v2_assert_visible.py,
sent the unmodified production prompts because pilot.run_pair builds its own prompts.)

Usage:  python3 scripts/mbpp_v2_assert_visible_eval.py [--models ...] [--workers 8] [--limit N]
Output: data/mbpp_v2/assert_visible_eval/events.jsonl, records.jsonl  (resumable)
"""
import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.mbpp_v2_pilot as pilot

MODELS = ["gpt-5.4", "ds-v4-flash", "glm-5.3-flash", "qwen38-flash", "gpt-6-luna"]
ARMS = {"selfcheck": pilot.selfcheck_prompt, "steelman": pilot.steelman_prompt}
LOCK = threading.Lock()


def first_assert(task: dict) -> str:
    for line in task["test"].splitlines():
        if line.lstrip().startswith("assert "):
            return line.strip()
    raise ValueError(f"no assert in {task['task_id']}")


def with_assert(task: dict) -> dict:
    t = dict(task)
    t["prompt"] = (task["prompt"] + "\n\nFor example, the solution must satisfy this "
                   "reference assertion:\n" + first_assert(task))
    return t


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="data/mbpp_v2/assert_visible_eval")
    args = ap.parse_args()

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    pilot.SCHEMA_VERSION = "mbpp-v2-assert-visible-eval/1"
    tasks = pilot.load_task_map()
    cands = {}
    for l in open(ROOT / "data/mbpp_v2/production/candidate_records.jsonl"):
        if l.strip():
            c = json.loads(l)
            cands[(c["model"], c["task_id"])] = c
    done = set()
    rec_path = out / "records.jsonl"
    if rec_path.exists():
        for l in open(rec_path):
            if l.strip():
                r = json.loads(l)
                done.add((r["model"], r["task_id"], r["arm"]))

    work = []
    for m in args.models:
        tids = sorted((t for (mm, t) in cands if mm == m), key=lambda t: int(t.split("/")[1]))
        if args.limit:
            tids = tids[:args.limit]
        for tid in tids:
            for arm in ARMS:
                if (m, tid, arm) not in done:
                    work.append((m, tid, arm))
    print(f"assert-visible evaluator control: {len(work)} calls to run, {len(done)} already done", flush=True)

    def run(item):
        m, tid, arm = item
        task = tasks[tid]
        c = cands[(m, tid)]
        prompt = ARMS[arm](with_assert(task), c["code"])
        if first_assert(task) not in prompt:
            raise RuntimeError(f"assert missing from prompt for {m} {tid} {arm}")
        ev = pilot.call_model(m, prompt, f"{arm}_assert_visible", tid, out, 3)
        verdict = pilot.strict_verdict(ev.get("raw_response")) if not ev.get("error") else None
        rec = {"model": m, "task_id": tid, "arm": arm, "T": c["T"], "verdict": verdict,
               "event_id": ev.get("event_id"), "finish_reason": ev.get("finish_reason"),
               "prompt_sha256": ev.get("prompt_sha256"), "error": ev.get("error")}
        with LOCK:
            pilot.append_jsonl(rec_path, rec)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(run, work))
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
