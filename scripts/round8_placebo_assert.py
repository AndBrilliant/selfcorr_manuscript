#!/usr/bin/env python3
"""Round 8, run 3: placebo-assert control (self-check only). Specified before the run.

The information control showed each evaluator the task's first reference assert. Here
the evaluator sees an assert in exactly the same position and wording, on the same input,
but with the expected value taken from running the candidate itself. The placebo
therefore agrees with the candidate whatever the candidate's correctness and carries no
information about T beyond the code the evaluator already has.

Prediction (P2/P3): the placebo may move the acceptance threshold; it should not raise
sensitivity d' the way the real assert did. Analysis: paired with production self-check
on the same candidates; d', c and false acceptance by family, as for the information
control.

A candidate gets no placebo (and no call) when running it on the first assert's input
raises, times out, or returns a value whose repr is not a Python literal; those are
counted and reported. Every placebo is verified to pass against the candidate before use.

Usage:
  python3 scripts/round8_placebo_assert.py --build            # offline: build placebos, no calls
  python3 scripts/round8_placebo_assert.py --show 3           # print 3 rendered prompts, no calls
  python3 scripts/round8_placebo_assert.py --limit 10 --cap 2 # dry run: 10 tasks per family
  python3 scripts/round8_placebo_assert.py --cap 15           # full run (resumable)
Output: data/mbpp_v2/round8_placebo/
"""
from __future__ import annotations

import argparse
import ast
import json
import multiprocessing as mp
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.round8_common as rc  # noqa: E402
from scripts.round8_common import pilot  # noqa: E402
from scripts.round8_common import split_parts, assert_fails, lhs, run_exprs  # noqa: E402

OUT = ROOT / "data/mbpp_v2/round8_placebo"
pilot.SCHEMA_VERSION = "round8-placebo/1"


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    tasks = pilot.load_task_map()
    cands = rc.load_candidates()
    keys, jobs = [], []
    for (m, t), c in sorted(cands.items()):
        if m not in rc.FAMILIES or not c.get("code"):
            continue
        setup, _ = split_parts(tasks[t]["test"])
        keys.append((m, t))
        jobs.append((c["code"], setup, [lhs(rc.first_assert(tasks[t]))]))
    with mp.Pool(16) as pool:
        outs = pool.map(run_exprs, jobs, chunksize=4)
    rows, why = [], Counter()
    verify = []
    for (m, t), o in zip(keys, outs):
        e = lhs(rc.first_assert(tasks[t]))
        val = o[0] if o and o[0] not in ("CRASH", "TIMEOUT") else None
        reason = None
        if e is None:
            reason = "unparsed_assert"
        elif val is None:
            reason = (o[0] if o else "CRASH").lower()
        elif val.startswith("EXC:") or val == "UNPARSED":
            reason = "candidate_raises"
        else:
            try:
                ast.literal_eval(val)
            except Exception:
                reason = "non_literal_output"
        if reason:
            why[reason] += 1
            rows.append({"model": m, "task_id": t, "placebo": None, "reason": reason})
            continue
        placebo = f"assert {e} == {val}"
        rows.append({"model": m, "task_id": t, "placebo": placebo, "reason": None,
                     "same_as_reference": None})
        verify.append(len(rows) - 1)
    # verify every placebo passes against its own candidate
    vjobs = []
    for i in verify:
        r = rows[i]
        setup, _ = split_parts(tasks[r["task_id"]]["test"])
        vjobs.append((cands[(r["model"], r["task_id"])]["code"], setup, r["placebo"]))
    with mp.Pool(16) as pool:
        failed = pool.map(assert_fails, vjobs, chunksize=4)
    for i, f in zip(verify, failed):
        if f:
            why["placebo_fails_own_candidate"] += 1
            rows[i]["reason"], rows[i]["placebo"] = "placebo_fails_own_candidate", None
        else:
            rows[i]["same_as_reference"] = (cands[(rows[i]["model"], rows[i]["task_id"])]["T"] == 1)
    with open(OUT / "placebos.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    usable = Counter(r["model"] for r in rows if r["placebo"])
    summary = {"candidates": len(rows), "usable_by_family": dict(usable),
               "excluded": dict(why)}
    (OUT / "placebos_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


def load_placebos():
    p = OUT / "placebos.jsonl"
    if not p.exists():
        raise SystemExit("run --build first")
    return {(r["model"], r["task_id"]): r for r in map(json.loads, open(p)) if r["placebo"]}


def render(task, code, placebo):
    prompt = pilot.selfcheck_prompt(rc.with_shown(task, placebo), code)
    assert code and code.strip(), "empty candidate code"
    assert code in prompt, "candidate code missing from prompt"
    assert placebo in prompt, "placebo assert missing from prompt"
    assert "For example, the solution must satisfy this reference assertion:" in prompt
    return prompt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--show", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--cap", type=float, default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--min-credits", type=float, default=5.0)
    a = ap.parse_args()
    if a.build:
        return build()
    tasks, cands, pl = pilot.load_task_map(), rc.load_candidates(), load_placebos()
    keys = sorted(pl, key=lambda k: (k[0], int(k[1].split("/")[1])))
    if a.show:
        for k in [k for k in keys if cands[k]["T"] == 0][:a.show]:
            print("=" * 70, k, "T=0\n", render(tasks[k[1]], cands[k]["code"], pl[k]["placebo"]))
        return
    if a.cap is None:
        raise SystemExit("--cap is required for any run that makes calls")
    rec = OUT / "records.jsonl"
    done = rc.done_keys(rec, ("model", "task_id"))
    work = []
    for m in rc.FAMILIES:
        ts = [k for k in keys if k[0] == m]
        if a.limit:
            ts = ts[:a.limit]
        work += [k for k in ts if k not in done]
    cap = rc.Cap(a.cap, a.min_credits)

    def fn(k):
        m, t = k
        c = cands[k]
        prompt = render(tasks[t], c["code"], pl[k]["placebo"])
        ev = pilot.call_model(m, prompt, "selfcheck_placebo", t, OUT, 3)
        cap.add(ev)
        v = pilot.strict_verdict(ev.get("raw_response")) if not ev.get("error") else None
        r = {"model": m, "task_id": t, "arm": "selfcheck_placebo", "T": c["T"],
             "code_sha256": c["code_sha256"], "placebo": pl[k]["placebo"], "verdict": v,
             "event_id": ev.get("event_id"), "prompt_sha256": ev.get("prompt_sha256"),
             "finish_reason": ev.get("finish_reason"), "error": ev.get("error")}
        with rc.LOCK:
            pilot.append_jsonl(rec, r)

    rc.run_work(work, fn, cap, a.workers, OUT, "placebo")


if __name__ == "__main__":
    main()
