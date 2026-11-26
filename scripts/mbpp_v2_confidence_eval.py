#!/usr/bin/env python3
"""Confidence-rating control (paper Sec. IV-C/IV-F).

Every frozen MBPP production candidate is re-evaluated under the self-check and steelman
prompts, unchanged except for the closing instruction: the evaluator gives its verdict
and then a confidence from 1 (certainly incorrect) to 7 (certainly correct). The ratings
give an empirical ROC per family and prompt, so sensitivity can be compared across
prompts without the equal-variance assumption.

Usage:  python3 scripts/mbpp_v2_confidence_eval.py [--models ...] [--workers 4] [--limit N]
Output: data/mbpp_v2/confidence_eval/events.jsonl, records.jsonl  (resumable)
"""
import argparse, json, re, sys, threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import scripts.mbpp_v2_pilot as pilot

MODELS = ["gpt-5.4", "ds-v4-flash", "glm-5.3-flash", "qwen38-flash", "gpt-6-luna"]
OLD_END = "End with exactly one final line containing only one of:\nVERDICT: ACCEPT\nVERDICT: REJECT"
NEW_END = ("End with exactly two final lines. The first contains only one of:\n"
           "VERDICT: ACCEPT\nVERDICT: REJECT\n"
           "The second contains only CONFIDENCE: followed by one integer from 1 to 7, where "
           "1 means the candidate is certainly incorrect and 7 means it is certainly correct.")
LOCK = threading.Lock()


def prompt_for(arm, task, code):
    base = (pilot.selfcheck_prompt if arm == "selfcheck" else pilot.steelman_prompt)(task, code)
    if base.count(OLD_END) != 1:
        raise RuntimeError("production prompt ending not found; harness changed")
    return base.replace(OLD_END, NEW_END)


def parse(resp):
    if resp is None:
        return None, None
    lines = [l.strip() for l in resp.rstrip().splitlines() if l.strip()]
    if len(lines) < 2:
        return None, None
    v = re.fullmatch(r"VERDICT\s*:\s*(ACCEPT|REJECT)", lines[-2], re.I)
    c = re.fullmatch(r"CONFIDENCE\s*:\s*([1-7])", lines[-1], re.I)
    if not v or not c:
        return None, None
    if sum(bool(re.fullmatch(r"VERDICT\s*:\s*(ACCEPT|REJECT)", l, re.I)) for l in lines) != 1:
        return None, None
    return v.group(1).upper(), int(c.group(1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="data/mbpp_v2/confidence_eval")
    a = ap.parse_args()
    out = ROOT / a.out; out.mkdir(parents=True, exist_ok=True)
    pilot.SCHEMA_VERSION = "mbpp-v2-confidence-eval/1"
    tasks = pilot.load_task_map()
    cands = {(c["model"], c["task_id"]): c for c in map(json.loads, open(ROOT / "data/mbpp_v2/production/candidate_records.jsonl"))}
    rec = out / "records.jsonl"; done = set()
    if rec.exists():
        for l in open(rec):
            if l.strip():
                r = json.loads(l); done.add((r["model"], r["task_id"], r["arm"]))
    work = []
    for m in a.models:
        tids = sorted((t for (mm, t) in cands if mm == m), key=lambda t: int(t.split("/")[1]))
        if a.limit:
            tids = tids[:a.limit]
        work += [(m, t, arm) for t in tids for arm in ("selfcheck", "steelman") if (m, t, arm) not in done]
    print(f"confidence control: {len(work)} calls to run, {len(done)} already done", flush=True)

    def run(item):
        m, t, arm = item
        p = prompt_for(arm, tasks[t], cands[(m, t)]["code"])
        if "CONFIDENCE:" not in p:
            raise RuntimeError("confidence instruction missing")
        ev = pilot.call_model(m, p, f"{arm}_confidence", t, out, 3)
        v, c = parse(ev.get("raw_response")) if not ev.get("error") else (None, None)
        with LOCK:
            pilot.append_jsonl(rec, {"model": m, "task_id": t, "arm": arm, "T": cands[(m, t)]["T"],
                                     "verdict": v, "confidence": c, "event_id": ev.get("event_id"),
                                     "finish_reason": ev.get("finish_reason"), "error": ev.get("error")})

    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        list(pool.map(run, work))
    print("done", flush=True)


if __name__ == "__main__":
    main()
