"""Cross-model audit of MBPP v2 "incorrect" labels — CORRECTED RERUN (v3).

Purpose
-------
Decide, with two outside-family frontier auditors, whether each of 650
sampled MBPP-v2 INCORRECT candidates is knowable from the task description
alone. Rerun of the void 2026-11-24 run (archived at
data/mbpp_v2/crossmodel_audit_INVALID_empty_code/ — the first runner shipped
every prompt with an EMPTY candidate program because sample rows lacked the
code field).

Auditor selection rule (Step 1, per RUN_crossmodel_audit_FIX.md)
----------------------------------------------------------------
Exactly two auditors, chosen for cost:
  1. anthropic/claude-opus-5.5 — exact id looked up in the OpenRouter models
     list and recorded; Mythos-tier models excluded.
  2. x-ai/grok-4.7.
Both are called with OpenRouter reasoning effort "low" (the first run burned
2.74M hidden Grok reasoning tokens at provider-default). Both ids are written
to data/mbpp_v2/crossmodel_audit/auditors.json before any call.

Sample (Step 2)
---------------
Same deterministic sample as the void run: population T == 0 with non-empty
code; per generator family sorted by task_id; 130 drawn per family with
random.Random(20261124).sample(...). The regenerated sample is verified
row-for-row identical to the archived sample.jsonl before the run proceeds.

Safety checks — before EVERY call, abort the whole run on any failure:
  (a) code non-empty and containing "def ";
  (b) sha256(code) equals the sample row's code_sha256;
  (c) the rendered prompt contains that exact code string.

Cost controls: OpenRouter credit balance recorded before the run; the running
sum of per-response `cost` is checked against a $15 stop; attempts hard-stop
at 1,400 (1,300 planned calls + up to 100 retries). Max output tokens 2,000,
temperature 0. Run after data collection; analysis plan fixed before results
were seen.
"""

import argparse
import concurrent.futures
import hashlib
import json
import os
import random
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO, "data", "mbpp_v2", "crossmodel_audit")
INVALID_DIR = os.path.join(REPO, "data", "mbpp_v2", "crossmodel_audit_INVALID_empty_code")
KEY_PATH = os.path.expanduser("~/.keys/openrouter")
OR_LIST = "https://openrouter.ai/api/v1/models"
OR_CHAT = "https://openrouter.ai/api/v1/chat/completions"
OR_CREDITS = "https://openrouter.ai/api/v1/credits"
MAX_TOKENS = 2000
TEMPERATURE = 0.0
MAX_RETRIES = 3
ATTEMPT_BUDGET = 1400          # 1,300 planned + up to 100 retries
COST_STOP = 15.0               # USD, from per-response `cost`
WORKERS_PER_AUDITOR = 8
GENERATORS = ["gpt-5.4", "ds-v4-flash", "glm-5.3-flash", "qwen38-flash", "gpt-6-luna"]
SAMPLE_PER_FAMILY = 130
SEED = 20261124
AUDITORS = ["anthropic/claude-opus-5.5", "x-ai/grok-4.7"]

PROMPT_TEMPLATE = """You are auditing a labeled programming benchmark. A program was written from a short task description. Hidden reference tests marked it INCORRECT. Your job is to decide whether that error was knowable from the description alone.

[TASK DESCRIPTION]
{prompt}
(The required function name is: {entry_point})

[CANDIDATE PROGRAM]
{code}

[HIDDEN REFERENCE TESTS]
{tests}

Question: could a careful, competent programmer who saw ONLY the task description and the function name have known that this program is wrong?

Answer KNOWABLE if the program violates something the description states or clearly implies (wrong algorithm, wrong result for a case the description covers, mishandled input the description describes).

Answer NOT_KNOWABLE if the program is a defensible reading of the description, and the tests fail it only because of something the description does not state (an unstated return type or container, argument order or format, indexing convention, ordering, edge-case behavior, or another interpretation a reasonable reader could hold).

Reply with one short sentence of reasoning, then end with exactly one final line containing only one of:
LABEL: KNOWABLE
LABEL: NOT_KNOWABLE"""
PROMPT_SHA256 = hashlib.sha256(PROMPT_TEMPLATE.encode()).hexdigest()

_lock = threading.Lock()
_attempts = [0]
_cost_total = [0.0]


def log(*a):
    with _lock:
        print(datetime.now(timezone.utc).strftime("%H:%M:%S"), *a, flush=True)


def read_key():
    with open(KEY_PATH) as f:
        return f.read().strip().splitlines()[0].strip()


def http_json(url, payload=None, key=None, timeout=600):
    req = urllib.request.Request(url, data=json.dumps(payload).encode() if payload else None)
    req.add_header("Content-Type", "application/json")
    if key:
        req.add_header("Authorization", "Bearer " + key)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def load_jsonl_local(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def append_line(path, obj):
    with _lock:
        with open(path, "a") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def pick_auditors(key):
    """Look up the two exact ids; write auditors.json before any call."""
    out = os.path.join(OUT_DIR, "auditors.json")
    if os.path.exists(out):
        return json.load(open(out))["auditors"]
    lst = http_json(OR_LIST, key=key)["data"]
    ids = {m["id"]: m for m in lst}
    doc = {"queried_at": datetime.now(timezone.utc).isoformat(),
           "rule": "two auditors per RUN_crossmodel_audit_FIX.md: anthropic/claude-opus-5.5 "
                   "(exact id from the models list; Mythos-tier models excluded) and "
                   "x-ai/grok-4.7, both called with reasoning effort 'low'; Gemini dropped.",
           "auditors": [], "details": {}}
    for mid in AUDITORS:
        if mid not in ids:
            raise SystemExit(f"auditor {mid} not found in OpenRouter models list - aborting")
        m = ids[mid]
        doc["auditors"].append(mid)
        doc["details"][mid] = {"name": m.get("name"), "context_length": m.get("context_length"),
                               "pricing": m.get("pricing")}
    with open(out, "w") as f:
        json.dump(doc, f, indent=1)
    return doc["auditors"]


def build_sample(sample_path=None):
    out = sample_path or os.path.join(OUT_DIR, "sample.jsonl")
    if os.path.exists(out):
        rows = load_jsonl_local(out)
    else:
        cands = load_jsonl_local(os.path.join(REPO, "data/mbpp_v2/production/candidate_records.jsonl"))
        by_fam = {}
        for c in cands:
            if str(c["T"]) == "0" and (c.get("code") or "").strip():
                by_fam.setdefault(c["model"], []).append(c)
        rows = []
        for fam in GENERATORS:
            pool = sorted(by_fam.get(fam, []), key=lambda c: c["task_id"])
            if len(pool) < SAMPLE_PER_FAMILY:
                raise SystemExit(f"family {fam} has only {len(pool)} incorrect candidates")
            for c in random.Random(SEED).sample(pool, SAMPLE_PER_FAMILY):
                rows.append({"generator": fam, "task_id": c["task_id"],
                             "code_sha256": c.get("code_sha256") or hashlib.sha256(c["code"].encode()).hexdigest()})
        with open(out, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    old = os.path.join(INVALID_DIR, "sample.jsonl") if not sample_path else None
    if old and os.path.exists(old):
        old_rows = load_jsonl_local(old)
        if old_rows != rows:
            raise SystemExit("regenerated sample differs from the archived sample - aborting")
        log("sample verified identical to archived sample:", len(rows))
    return rows


def build_inputs():
    mbpp = {}
    for line in open(os.path.join(REPO, "data/MBPP.jsonl")):
        r = json.loads(line)
        mbpp[r["task_id"]] = r
    sev = {}
    for line in open(os.path.join(REPO, "data/mbpp_v2/analysis/error_severity_audit.jsonl")):
        r = json.loads(line)
        sev[(r["model"], r["task_id"])] = r
    return mbpp, sev


def code_lookup_table():
    table = {}
    for c in load_jsonl_local(os.path.join(REPO, "data/mbpp_v2/production/candidate_records.jsonl")):
        table[(c["model"], c["task_id"])] = c.get("code") or ""
    return table


def render_prompt(prompt, entry_point, code, test, flags):
    assert_lines = [l.strip() for l in (test or "").splitlines() if l.strip().startswith("assert")]
    if flags and len(flags) != len(assert_lines):
        flags = (list(flags) + [False] * len(assert_lines))[: len(assert_lines)]
    if not flags:
        flags = [False] * len(assert_lines)
    tests = "\n".join("%s  # %s" % (l, "FAILED" if f else "passed") for l, f in zip(assert_lines, flags))
    return PROMPT_TEMPLATE.format(prompt=prompt, entry_point=entry_point, code=code, tests=tests)


def parse_label(text):
    hits = []
    for line in (text or "").splitlines():
        s = line.strip()
        if s.startswith("LABEL:"):
            tail = s[len("LABEL:"):].strip()
            if tail in ("KNOWABLE", "NOT_KNOWABLE"):
                hits.append(tail)
    return hits[-1] if hits else "unparsed"


def credit_snapshot(key):
    try:
        d = http_json(OR_CREDITS, key=key).get("data", {})
        return {"total_credits": d.get("total_credits"), "total_usage": d.get("total_usage")}
    except Exception as e:
        return {"error": str(e)[:150]}


def audit_one(key, auditor, row, mbpp, sev, code_table, ev_path, lab_path):
    task = row["task_id"]
    code = code_table[(row["generator"], task)]
    if not code.strip() or "def " not in code:
        raise RuntimeError("(a) code empty/missing 'def ' for %s %s - aborting run" % (row["generator"], task))
    sha = hashlib.sha256(code.encode()).hexdigest()
    if sha != row["code_sha256"]:
        raise RuntimeError("(b) code sha mismatch for %s %s - aborting run" % (row["generator"], task))
    mb = mbpp.get(task)
    s = sev.get((row["generator"], task))
    if mb is None or s is None:
        raise RuntimeError("missing mbpp/severity input for %s %s - aborting run" % (row["generator"], task))
    prompt = render_prompt(mb["prompt"], mb["entry_point"], code, mb["test"], s.get("per_assert_failed"))
    if code not in prompt:
        raise RuntimeError("(c) code not present in rendered prompt for %s %s - aborting run" % (row["generator"], task))
    psha = hashlib.sha256(prompt.encode()).hexdigest()
    attempt = 0
    while attempt < MAX_RETRIES:
        attempt += 1
        with _lock:
            _attempts[0] += 1
            n = _attempts[0]
        if n > ATTEMPT_BUDGET:
            raise RuntimeError("attempt budget %d exhausted - aborting run" % ATTEMPT_BUDGET)
        try:
            t0 = time.time()
            resp = http_json(OR_CHAT, {
                "model": auditor,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": TEMPERATURE,
                "max_tokens": MAX_TOKENS,
                "reasoning": {"effort": "low"},
                "usage": {"include": True},
            }, key=key)
            ch = (resp.get("choices") or [{}])[0]
            raw = (ch.get("message") or {}).get("content") or ""
            usage = resp.get("usage") or {}
            cost = usage.get("cost")
            prov_model = resp.get("model")
            append_line(ev_path, {
                "ts": datetime.now(timezone.utc).isoformat(),
                "auditor_requested": auditor, "auditor_returned": prov_model,
                "generator": row["generator"], "task_id": task,
                "prompt_sha256": psha, "attempt": attempt,
                "latency_s": round(time.time() - t0, 2),
                "usage": usage, "cost": cost,
                "finish_reason": ch.get("finish_reason"),
                "raw": raw,
            })
            with _lock:
                if cost is not None:
                    _cost_total[0] += float(cost)
            if _cost_total[0] > COST_STOP:
                raise RuntimeError("running cost $%.2f passed $%.2f - stopping run" % (_cost_total[0], COST_STOP))
            label = parse_label(raw)
            append_line(lab_path, {"auditor": auditor, "generator": row["generator"],
                                   "task_id": task, "label": label, "attempt": attempt,
                                   "cost": cost})
            return
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError) as e:
            code_ = getattr(e, "code", None)
            retryable = code_ in (429, 500, 502, 503, 504) or not isinstance(e, urllib.error.HTTPError)
            append_line(ev_path, {"ts": datetime.now(timezone.utc).isoformat(),
                                  "auditor_requested": auditor, "generator": row["generator"],
                                  "task_id": task, "attempt": attempt,
                                  "error": str(e)[:300], "http_status": code_})
            if attempt >= MAX_RETRIES or not retryable:
                append_line(lab_path, {"auditor": auditor, "generator": row["generator"],
                                       "task_id": task, "label": "unparsed",
                                       "note": "api failure after %d attempts: %s" % (attempt, str(e)[:120])})
                return
            time.sleep(min(60, 5 * 2 ** attempt))


def main():
    global OUT_DIR, COST_STOP, ATTEMPT_BUDGET
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="output folder under data/mbpp_v2/ (default crossmodel_audit)")
    ap.add_argument("--sample", default=None, help="pre-built sample.jsonl to audit (skips rebuild + archive check)")
    ap.add_argument("--cost-stop", type=float, default=COST_STOP)
    ap.add_argument("--attempt-budget", type=int, default=ATTEMPT_BUDGET)
    args = ap.parse_args()
    if args.out:
        OUT_DIR = args.out if os.path.isabs(args.out) else os.path.join(REPO, args.out)
    COST_STOP = args.cost_stop
    ATTEMPT_BUDGET = args.attempt_budget
    os.makedirs(OUT_DIR, exist_ok=True)
    key = read_key()
    auditors = pick_auditors(key)
    for _a in auditors:
        if "fable" in _a.lower() or "mythos" in _a.lower():
            raise SystemExit("AUDITOR GUARD: forbidden model id %s - aborting run" % _a)
    log("auditors:", auditors)
    log("credit snapshot:", json.dumps(credit_snapshot(key)))
    sample = build_sample(args.sample)
    mbpp, sev = build_inputs()
    code_table = code_lookup_table()
    ev_path = os.path.join(OUT_DIR, "events.jsonl")
    lab_path = os.path.join(OUT_DIR, "labels.jsonl")
    done = set()
    if os.path.exists(lab_path):
        for r in load_jsonl_local(lab_path):
            done.add((r["auditor"], r["task_id"], r["generator"]))
    todo = [(aud, row) for aud in auditors for row in sample
            if (aud, row["task_id"], row["generator"]) not in done]
    log("todo: %d calls (%d already labeled); attempt cap %d, cost stop $%.2f" %
        (len(todo), len(done), ATTEMPT_BUDGET, COST_STOP))
    failures = 0
    stopped = None
    pools = {}
    for aud in auditors:
        pools[aud] = concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS_PER_AUDITOR)
    futs = []
    for aud in auditors:
        for row in sample:
            if (aud, row["task_id"], row["generator"]) in done:
                continue
            futs.append((aud, pools[aud].submit(audit_one, key, aud, row, mbpp, sev,
                                                code_table, ev_path, lab_path)))
    done_n = 0
    for aud, f in futs:
        if stopped:
            f.cancel()
            continue
        try:
            f.result()
        except RuntimeError as e:
            msg = str(e)
            if "stopping run" in msg or "aborting run" in msg or "budget" in msg:
                log("RUN STOPPED:", msg[:200])
                stopped = msg
                continue
            failures += 1
            log("worker error:", msg[:200])
        except Exception as e:
            failures += 1
            log("worker exception:", repr(e)[:200])
        done_n += 1
        if done_n % 100 == 0:
            log("progress %d/%d, attempts=%d, cost=$%.2f" % (done_n, len(futs), _attempts[0], _cost_total[0]))
    for pool in pools.values():
        pool.shutdown(wait=False, cancel_futures=True)
    labels = sum(1 for _ in open(lab_path)) if os.path.exists(lab_path) else 0
    log("DONE: labels=%d attempts=%d cost=$%.2f worker_errors=%d stopped=%s" %
        (labels, _attempts[0], _cost_total[0], failures, bool(stopped)))


if __name__ == "__main__":
    main()
