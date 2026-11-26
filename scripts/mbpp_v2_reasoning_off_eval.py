#!/usr/bin/env python3
"""Reasoning-off control (paper Sec. IV-F).

One reasoning family's frozen MBPP production candidates are re-evaluated under the
unchanged production self-check and steelman prompts with reasoning disabled through
OpenRouter ("reasoning": {"effort": "none"}). Production verdicts on the same candidates
are the reasoning-on condition.

With --effort low|medium|high the same script runs the reverse test: reasoning switched
ON for a family that ran without it in production (GPT-5.4).

Usage:
  python3 scripts/mbpp_v2_reasoning_off_eval.py --check --model glm-5.3-flash
  python3 scripts/mbpp_v2_reasoning_off_eval.py --model glm-5.3-flash [--effort none] [--workers 4] [--limit N]
Output: data/mbpp_v2/reasoning_off_eval/<model>/ for effort none,
        data/mbpp_v2/reasoning_off_eval/<model>__effort-<e>/ otherwise  (resumable)
"""
import argparse, json, sys, threading, time, uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import scripts.mbpp_v2_pilot as pilot

LOCK = threading.Lock()
REASONING = {"effort": "none"}  # replaced by --effort


def check(model):
    mid = pilot.MODEL_SPECS[model]["model_id"]
    data = requests.get("https://openrouter.ai/api/v1/models", timeout=60).json()["data"]
    m = next((x for x in data if x.get("id") == mid), None)
    if m is None:
        print(f"{mid}: not found in OpenRouter model list"); return
    print(mid, "| supported_parameters:", m.get("supported_parameters"),
          "| reasoning:", json.dumps(m.get("reasoning") or m.get("architecture", {}).get("reasoning")))


def call(model, prompt, stage, tid, out):
    spec = pilot.MODEL_SPECS[model]
    if spec["provider"] != "openrouter":
        raise RuntimeError("this control is defined for OpenRouter routes only; route the model through OpenRouter")
    key = (Path.home() / ".keys" / spec["key_file"]).read_text().strip()
    body = {"model": spec["model_id"], "messages": [{"role": "user", "content": prompt}],
            "temperature": 0, "reasoning": REASONING}
    base = {"schema_version": "mbpp-v2-reasoning-off/1", "ts": pilot.utc_now(), "stage": stage,
            "task_id": tid, "model": model, "provider_model_id": spec["model_id"],
            "prompt_sha256": pilot.sha256_text(prompt), "temperature": 0, "reasoning_request": REASONING}
    for attempt in range(1, 4):
        try:
            r = requests.post(spec["base_url"], headers={"Authorization": f"Bearer {key}",
                              "Content-Type": "application/json"}, json=body, timeout=1800)
            try:
                p = r.json()
            except ValueError:
                p = {}
            ch = (p.get("choices") or [{}])[0]
            raw = (ch.get("message") or {}).get("content")
            ev = dict(base, event_id=str(uuid.uuid4()), transport_attempt=attempt, http_status=r.status_code,
                      raw_response=raw if isinstance(raw, str) else None, finish_reason=ch.get("finish_reason"),
                      usage=p.get("usage"), error=None if isinstance(raw, str) else str(p.get("error") or "no content")[:500])
        except requests.RequestException as e:
            ev = dict(base, event_id=str(uuid.uuid4()), transport_attempt=attempt, http_status=None,
                      raw_response=None, finish_reason=None, usage=None, error=f"{type(e).__name__}: {e}"[:500])
        with LOCK:
            pilot.append_jsonl(out / "events.jsonl", ev)
        if ev["error"] is None or ev["finish_reason"] == "length" or (ev["http_status"] and 400 <= ev["http_status"] < 500 and ev["http_status"] != 429):
            return ev
        time.sleep(min(2 ** attempt, 8))
    return ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--effort", default="none", choices=["none", "low", "medium", "high"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    if a.check:
        check(a.model); return
    global REASONING
    REASONING = {"effort": a.effort}
    sub = a.model if a.effort == "none" else f"{a.model}__effort-{a.effort}"
    out = ROOT / "data/mbpp_v2/reasoning_off_eval" / sub; out.mkdir(parents=True, exist_ok=True)
    tasks = pilot.load_task_map()
    cands = {(c["model"], c["task_id"]): c for c in map(json.loads, open(ROOT / "data/mbpp_v2/production/candidate_records.jsonl"))}
    rec = out / "records.jsonl"; done = set()
    if rec.exists():
        for l in open(rec):
            if l.strip():
                r = json.loads(l); done.add((r["task_id"], r["arm"]))
    tids = sorted((t for (m, t) in cands if m == a.model), key=lambda t: int(t.split("/")[1]))
    if a.limit:
        tids = tids[:a.limit]
    work = [(t, arm) for t in tids for arm in ("selfcheck", "steelman") if (t, arm) not in done]
    print(f"reasoning control ({a.model}, effort={a.effort}): {len(work)} calls to run, {len(done)} already done", flush=True)

    def run(item):
        t, arm = item
        f = pilot.selfcheck_prompt if arm == "selfcheck" else pilot.steelman_prompt
        ev = call(a.model, f(tasks[t], cands[(a.model, t)]["code"]), f"{arm}_reasoning_{a.effort}", t, out)
        u = ev.get("usage") or {}
        rt = (u.get("completion_tokens_details") or {}).get("reasoning_tokens")
        with LOCK:
            pilot.append_jsonl(rec, {"model": a.model, "effort": a.effort, "task_id": t, "arm": arm, "T": cands[(a.model, t)]["T"],
                                     "verdict": pilot.strict_verdict(ev.get("raw_response")) if not ev.get("error") else None,
                                     "reasoning_tokens": rt, "completion_tokens": u.get("completion_tokens"),
                                     "event_id": ev.get("event_id"), "error": ev.get("error")})

    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        list(pool.map(run, work))
    print("done", flush=True)


if __name__ == "__main__":
    main()
