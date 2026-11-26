#!/usr/bin/env python3
"""Shared machinery for the round-8 runs (placebo assert, interleaved self-check vs
steelman, generator-sees-assert). Specified before any of these runs.

Guards, in every runner:
  * every family goes through OpenRouter (as in the three earlier controls), so every
    event carries provider-reported cost;
  * a hard cap on summed provider-reported cost: once reached, no new call starts;
  * pre-call assertions on every rendered prompt (raise -> the whole run stops);
  * OpenRouter credits are read before and after (the key itself is never printed);
  * work items are shuffled with a fixed seed, so arms and families are interleaved;
  * ledgers are append-only and resumable; nothing is overwritten.
"""
from __future__ import annotations

import ast
import json
import random
import subprocess
import tempfile
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.mbpp_v2_pilot as pilot  # noqa: E402

FAMILIES = ["gpt-5.4", "ds-v4-flash", "glm-5.3-flash", "qwen38-flash", "gpt-6-luna"]
SEED = 20261124
LOCK = threading.Lock()

# Route GPT-5.4 through OpenRouter, as the information, confidence and neutral controls did.
pilot.MODEL_SPECS["gpt-5.4"] = {
    "provider": "openrouter",
    "model_id": "openai/gpt-5.4",
    "base_url": "https://openrouter.ai/api/v1/chat/completions",
    "key_file": "openrouter",
}


def first_assert(task: dict) -> str:
    for line in task["test"].splitlines():
        if line.lstrip().startswith("assert "):
            return line.strip()
    raise ValueError(f"no assert in {task['task_id']}")


def with_shown(task: dict, assert_line: str) -> dict:
    """Same wording as the information control (mbpp_v2_assert_visible_eval.with_assert)."""
    t = dict(task)
    t["prompt"] = (task["prompt"] + "\n\nFor example, the solution must satisfy this "
                   "reference assertion:\n" + assert_line)
    return t


def load_candidates() -> dict:
    cands = {}
    for l in open(ROOT / "data/mbpp_v2/production/candidate_records.jsonl"):
        if l.strip():
            c = json.loads(l)
            cands[(c["model"], c["task_id"])] = c
    return cands


def credits() -> float | None:
    key = (Path.home() / ".keys" / "openrouter").read_text().strip()
    try:
        r = requests.get("https://openrouter.ai/api/v1/credits",
                         headers={"Authorization": f"Bearer {key}"}, timeout=30)
        d = r.json()["data"]
        return round(float(d["total_credits"]) - float(d["total_usage"]), 4)
    except Exception as e:  # report, do not fail the run on a read error
        print(f"credits read failed: {type(e).__name__}", flush=True)
        return None


class Cap:
    """Stops new calls when (a) logged cost reaches the cap, (b) OpenRouter credits fall
    below the floor (re-read every CHECK_EVERY calls), or (c) the provider reports
    insufficient credits (HTTP 402 or an error mentioning credits)."""
    CHECK_EVERY = 200

    def __init__(self, usd: float, min_credits: float = 5.0):
        self.usd, self.min_credits = usd, min_credits
        self.spent, self.calls, self.stopped, self.reason = 0.0, 0, False, None

    def stop(self, reason: str):
        if not self.stopped:
            self.stopped, self.reason = True, reason
            print(f"STOPPING: {reason}; no new calls will start", flush=True)

    def add(self, ev: dict):
        c = ((ev.get("usage") or {}).get("cost")) or 0.0
        err = str(ev.get("error") or "").lower()
        with LOCK:
            self.spent += float(c)
            self.calls += 1
            n = self.calls
            if self.spent >= self.usd:
                self.stop(f"cost cap reached: ${self.spent:.2f} logged >= ${self.usd:.2f}")
            if ev.get("http_status") == 402 or "credit" in err or "insufficient" in err:
                self.stop("provider reports insufficient credits")
        if n % self.CHECK_EVERY == 0:
            left = credits()
            print(f"[{n} calls] logged ${self.spent:.2f}; credits left {left}", flush=True)
            if left is not None and left < self.min_credits:
                self.stop(f"credits {left} below floor {self.min_credits}")


def done_keys(path: Path, fields: tuple) -> set:
    """Keys already finished. A record with a transport/provider error (out of credits,
    HTTP failure) is NOT finished and is retried on resume; a provider-side length
    truncation is deterministic at temperature 0 and stays final, as in production.
    Analyses take the last record per key."""
    s = set()
    if path.exists():
        for l in open(path):
            if l.strip():
                r = json.loads(l)
                if r.get("error") and r.get("finish_reason") != "length":
                    continue
                s.add(tuple(r[f] for f in fields))
    return s


def run_work(work: list, fn, cap: Cap, workers: int, out: Path, label: str):
    """Shuffle (interleave) and run; fn(item) does assertions, the call and the record."""
    random.Random(SEED).shuffle(work)
    before = credits()
    if before is not None and before < cap.min_credits:
        raise SystemExit(f"credits {before} already below floor {cap.min_credits}; not starting")
    print(f"[{label}] {len(work)} calls to run; cap ${cap.usd:.2f} logged; "
          f"OpenRouter credits before: {before}", flush=True)

    def guarded(item):
        if cap.stopped:
            return
        fn(item)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(guarded, work))
    after = credits()
    summary = {"label": label, "calls_made": cap.calls, "logged_cost_usd": round(cap.spent, 4),
               "credits_before": before, "credits_after": after,
               "credits_delta": (round(before - after, 4) if before is not None and after is not None else None),
               "cap_usd": cap.usd, "stopped": cap.stopped, "stop_reason": cap.reason,
               "credit_floor": cap.min_credits}
    with LOCK:
        pilot.append_jsonl(out / "run_summaries.jsonl", summary)
    print(json.dumps(summary), flush=True)
    if summary["credits_delta"] and cap.spent:
        print(f"[{label}] billed/logged ratio this session: {summary['credits_delta'] / cap.spent:.2f}", flush=True)


# --- Execution helpers, copied verbatim in behaviour from mbpp_v2_error_severity_audit.py
# --- (split_parts, job) and mbpp_v2_shared_output_agreement.py (lhs, run), so the round-8
# --- runners need only mbpp_v2_pilot.py.
SAFE_JOB = "import sys\nsys.setrecursionlimit(10000)\n"
SAFE_RUN = "import sys, json\nsys.setrecursionlimit(10000)\n"
# Fixed hash seed: set and dict reprs print in a reproducible order across runs.
import os as _os  # noqa: E402
CHILD_ENV = dict(_os.environ, PYTHONHASHSEED="0")


def split_parts(raw: str) -> tuple[str, list[str]]:
    lines = [l for l in raw.splitlines() if l.strip()]
    while lines and (lines[-1].strip() == "pass" or lines[-1].startswith("def check(")):
        lines.pop()
    asserts = [l for l in lines if l.lstrip().startswith("assert ")]
    setup = "\n".join(l for l in lines if not l.lstrip().startswith("assert "))
    return setup, asserts


def assert_fails(arg) -> bool:
    code, setup, a = arg
    src = SAFE_JOB + code + "\n\n" + setup + ("\n\n" if setup else "") + a + "\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
        fh.write(src)
        path = fh.name
    try:
        r = subprocess.run([sys.executable, path], capture_output=True, timeout=10, env=CHILD_ENV)
        return r.returncode != 0
    except subprocess.TimeoutExpired:
        return True
    finally:
        Path(path).unlink(missing_ok=True)


def lhs(a):
    try:
        node = ast.parse(a.strip()).body[0]
        t = node.test
        if isinstance(t, ast.Compare) and len(t.ops) == 1 and isinstance(t.ops[0], ast.Eq):
            return ast.unparse(t.left)
        return ast.unparse(t)
    except Exception:
        return None


def run_exprs(arg):
    code, setup, exprs = arg
    body = ["__o = []"]
    for e in exprs:
        if e is None:
            body.append("__o.append('UNPARSED')")
            continue
        body += ["try:", f"    __o.append(repr({e}))", "except BaseException as __x:",
                 "    __o.append('EXC:' + type(__x).__name__)"]
    body.append("print('__OUT__' + json.dumps(__o))")
    src = SAFE_RUN + code + "\n\n" + setup + "\n\n" + "\n".join(body) + "\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
        fh.write(src)
        p = fh.name
    try:
        r = subprocess.run([sys.executable, p], capture_output=True, timeout=10, text=True, env=CHILD_ENV)
        for line in r.stdout.splitlines()[::-1]:
            if line.startswith("__OUT__"):
                return tuple(json.loads(line[7:]))
        return ("CRASH",)
    except subprocess.TimeoutExpired:
        return ("TIMEOUT",)
    finally:
        Path(p).unlink(missing_ok=True)
