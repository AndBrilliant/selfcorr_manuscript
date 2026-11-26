#!/usr/bin/env python3
"""Round 8, run 1: the generator sees one reference assert; self-evaluation is then
tested on the errors that remain. Specified before the run.

Generation: the production generation prompt extended with the task's first reference
assert, in the wording of the information control. Correctness T is the full reference
test (as in production); T_hidden is the test without the shown assert (secondary).
Evaluation: the same family judges its own new candidate under self-check and steelman,
shown the same description and the same assert the generator saw. Arms are interleaved.
All families via OpenRouter.

Phases (each resumable; evaluation never starts for a candidate before it is frozen):
  gen   one generation call per (family, task); candidate frozen with SHA-256, tested offline
  eval  self-check and steelman on every frozen candidate, shuffled together

Analysis: error rate with the assert shown vs production; on the remaining errors,
false acceptance, d' and c by family and arm; how many remaining errors fail only hidden
asserts; mechanical arity check on the remaining errors.

Usage:
  python3 scripts/round8_generator_sees_assert.py gen --show 2
  python3 scripts/round8_generator_sees_assert.py gen --limit 10 --cap 3     # dry run
  python3 scripts/round8_generator_sees_assert.py eval --limit 10 --cap 3    # dry run
  python3 scripts/round8_generator_sees_assert.py gen --cap 30
  python3 scripts/round8_generator_sees_assert.py eval --cap 30
Output: data/mbpp_v2/round8_gen_assert/
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

OUT = ROOT / "data/mbpp_v2/round8_gen_assert"
pilot.SCHEMA_VERSION = "round8-gen-assert/1"
ARMS = {"selfcheck": pilot.selfcheck_prompt, "steelman": pilot.steelman_prompt}


def hidden_task(task):
    """Task whose test omits the first reference assert line (None if nothing is left)."""
    shown = rc.first_assert(task)
    lines, removed = [], False
    for line in task["test"].splitlines():
        if not removed and line.strip() == shown:
            removed = True
            continue
        lines.append(line)
    if not any(l.lstrip().startswith("assert ") for l in lines):
        return None
    t = dict(task)
    t["test"] = "\n".join(lines)
    return t


def gen_prompt(task):
    shown = rc.first_assert(task)
    prompt = pilot.generation_prompt(rc.with_shown(task, shown))
    assert shown in prompt, "shown assert missing from generation prompt"
    assert prompt.startswith(task["prompt"]), "generation prompt must extend the production prompt"
    return prompt


def eval_prompt(arm, task, code):
    shown = rc.first_assert(task)
    prompt = ARMS[arm](rc.with_shown(task, shown), code)
    assert code and code.strip(), "empty candidate code"
    assert code in prompt, "candidate code missing from prompt"
    assert shown in prompt, "shown assert missing from evaluation prompt"
    return prompt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["gen", "eval"])
    ap.add_argument("--show", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--cap", type=float, default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--min-credits", type=float, default=5.0)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    tasks = pilot.load_task_map()
    prod = rc.load_candidates()
    tids = sorted({t for (m, t) in prod if m in rc.FAMILIES}, key=lambda t: int(t.split("/")[1]))
    cand_path = OUT / "candidate_records.jsonl"

    if a.phase == "gen":
        if a.show:
            for t in tids[:a.show]:
                print("=" * 70, t, "\n", gen_prompt(tasks[t]))
            return
        if a.cap is None:
            raise SystemExit("--cap is required for any run that makes calls")
        done = rc.done_keys(OUT / "generation_records.jsonl", ("model", "task_id"))
        work = []
        for m in rc.FAMILIES:
            ts = tids[:a.limit] if a.limit else tids
            work += [(m, t) for t in ts if (m, t) not in done]
        cap = rc.Cap(a.cap, a.min_credits)

        def fn(item):
            m, t = item
            task = tasks[t]
            ev = pilot.call_model(m, gen_prompt(task), "generation_assert_shown", t, OUT, 3)
            cap.add(ev)
            code = pilot.extract_code(ev.get("raw_response"), task) if not ev.get("error") else None
            g = {"model": m, "task_id": t, "event_id": ev.get("event_id"), "error": ev.get("error"),
                 "finish_reason": ev.get("finish_reason"), "has_code": code is not None}
            if code is not None:
                ok, why = pilot.run_task_tests(code, task)
                ht = hidden_task(task)
                okh = pilot.run_task_tests(code, ht)[0] if ht else None
                c = {"model": m, "task_id": t, "generation_event_id": ev.get("event_id"),
                     "code": code, "code_sha256": pilot.sha256_text(code), "T": int(ok),
                     "T_hidden": (int(okh) if okh is not None else None), "test_reason": why,
                     "shown_assert": rc.first_assert(task)}
                with rc.LOCK:
                    pilot.append_jsonl(cand_path, c)
            with rc.LOCK:
                pilot.append_jsonl(OUT / "generation_records.jsonl", g)

        rc.run_work(work, fn, cap, a.workers, OUT, "gen_assert:gen")
        return

    # eval phase
    if not cand_path.exists():
        raise SystemExit("run the gen phase first")
    cands = {}
    for r in map(json.loads, open(cand_path)):
        cands[(r["model"], r["task_id"])] = r
    keys = sorted(cands, key=lambda k: (k[0], int(k[1].split("/")[1])))
    if a.show:
        for k in [k for k in keys if cands[k]["T"] == 0][:a.show]:
            for arm in ARMS:
                print("=" * 70, k, arm, "\n", eval_prompt(arm, tasks[k[1]], cands[k]["code"]))
        return
    if a.cap is None:
        raise SystemExit("--cap is required for any run that makes calls")
    rec = OUT / "eval_records.jsonl"
    done = rc.done_keys(rec, ("model", "task_id", "arm"))
    work = []
    for m in rc.FAMILIES:
        ks = [k for k in keys if k[0] == m]
        if a.limit:
            ks = ks[:a.limit]
        work += [(k, arm) for k in ks for arm in ARMS if (k[0], k[1], arm) not in done]
    cap = rc.Cap(a.cap, a.min_credits)

    def efn(item):
        (m, t), arm = item
        c = cands[(m, t)]
        if pilot.sha256_text(c["code"]) != c["code_sha256"]:
            raise RuntimeError(f"frozen candidate hash mismatch {m} {t}")
        ev = pilot.call_model(m, eval_prompt(arm, tasks[t], c["code"]), f"{arm}_assert_shown", t, OUT, 3)
        cap.add(ev)
        v = pilot.strict_verdict(ev.get("raw_response")) if not ev.get("error") else None
        r = {"model": m, "task_id": t, "arm": arm, "T": c["T"], "T_hidden": c["T_hidden"],
             "code_sha256": c["code_sha256"], "verdict": v, "event_id": ev.get("event_id"),
             "prompt_sha256": ev.get("prompt_sha256"), "finish_reason": ev.get("finish_reason"),
             "error": ev.get("error")}
        with rc.LOCK:
            pilot.append_jsonl(rec, r)

    rc.run_work(work, efn, cap, a.workers, OUT, "gen_assert:eval")


if __name__ == "__main__":
    main()
