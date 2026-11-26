"""Interleaved rerun (paper Sec. IV-F, Table VIII): runner.
Balanced blocks of 10 (5 arms x 2 candidate families), CONC concurrent requests,
full per-event log including response-body model and cached tokens. The design called
for one uninterrupted window; the run completed in three sessions (resume logic skips
completed calls), as reported in the paper.
Dry run: --single-block runs ONE block then exits."""
import json, hashlib, time, sys, os, re, random
import urllib.request
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "data/mbpp_v2/interleaved_rerun"
KEY = os.environ.get("OPENROUTER_API_KEY", "")  # set before running; not needed for analysis
API = "https://openrouter.ai/api/v1/chat/completions"
MODELS = {"qwen": "qwen/qwen3.8-flash", "luna": "openai/gpt-6-luna"}
OTHER = {"qwen": "luna", "luna": "qwen"}
CONC = 16

def steelman_prompt(p, c):
    return ("You are a neutral arbiter judging a debate between two expert Python programmers.\n\n"
            f"[PROBLEM]\n{p}\n\n[CANDIDATE]\n{c}\n\n"
            "Do the following in order:\n"
            "1. STEELMAN AGAINST: give the strongest case that the candidate is wrong.\n"
            "2. STEELMAN FOR: give the strongest case that the candidate is correct.\n"
            "3. Decide which case survives scrutiny. Judge only the candidate as given; "
            "do not provide a corrected version.\n"
            "End with exactly one final line containing only one of:\n"
            "VERDICT: ACCEPT\nVERDICT: REJECT")

def control_prompt(p, c):
    return ("You are performing a careful neutral verification of one immutable Python "
            "candidate solution.\n\n"
            f"[PROBLEM]\n{p}\n\n[CANDIDATE]\n{c}\n\n"
            "Verify the candidate against the specification step by step: check the "
            "logic, trace the behavior on the requirements, and confirm whether it "
            "satisfies the specification. Do not modify, repair, rewrite, or provide "
            "an alternative solution, and do not presume the candidate is correct or "
            "incorrect; verify it.\n"
            "End with exactly one final line containing only one of:\n"
            "VERDICT: ACCEPT\nVERDICT: REJECT")

def arm_spec(arm, cf):
    # returns (evaluator_model_id, prompt_fn, arm_family_for_prompt)
    if arm in ("steel_self_a", "steel_self_b"):  return MODELS[cf], steelman_prompt, "steelman"
    if arm == "control_self":                    return MODELS[cf], control_prompt, "control"
    if arm == "steel_partner":                   return MODELS[OTHER[cf]], steelman_prompt, "steelman"
    if arm == "control_partner":                 return MODELS[OTHER[cf]], control_prompt, "control"
    raise ValueError(arm)

VERDICT_RX = re.compile(r"VERDICT:\s*(ACCEPT|REJECT)\s*$", re.M)

def strict_verdict(text):
    if not text: return None
    ms = VERDICT_RX.findall(text)
    return ms[-1] if ms else None

def call(model, prompt):
    import time as _t
    for attempt in range(3):
        try:
            return _call1(model, prompt)
        except Exception as e:
            if attempt == 2: raise
            _t.sleep(20 if "Name or service" in str(e) else 5)

def _call1(model, prompt):
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}],
                       "temperature": 0}).encode()
    req = urllib.request.Request(API, data=body, headers={
        "Authorization": "Bearer " + KEY, "Content-Type": "application/json",
        "HTTP-Referer": "https://ad-research.org", "X-Title": "selfcorr-interleaved-rerun"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.loads(r.read())
    return d, time.time() - t0

def main():
    pilot = "--single-block" in sys.argv
    prose = {}
    for line in open(BASE / "data/MBPP.jsonl"):
        d = json.loads(line); prose[d["task_id"]] = d["prompt"]
    code = {}
    for line in open(BASE / "data/mbpp_v2/production/candidate_records.jsonl"):
        d = json.loads(line)
        fam = {"qwen38-flash": "qwen", "gpt-6-luna": "luna"}.get(d["model"])
        if fam: code[(fam, d["task_id"])] = d

    work = [json.loads(l) for l in open(OUT / "worklist.jsonl")]
    meta = json.load(open(OUT / "worklist_meta.json"))
    rng = random.Random(meta["seed"] + 1)
    rng.shuffle(work)

    # balanced blocks of 10: 5 arms x 2 cf, position-mixed
    blocks = [work[i:i+10] for i in range(0, len(work), 10)]
    # enforce structure: block b takes items b*5..b*5+5 of each cf — simpler: blocks are
    # already shuffled; balance guaranteed by construction of the work list (each task
    # contributes all 5 arms x3 reps; a random block is NOT balanced. Build balanced:
    tasks_by_cf = {}
    for w in work:
        tasks_by_cf.setdefault((w["cf"], w["task"]), []).append(w)
    order = sorted({(w["cf"], w["task"]) for w in work})
    rng.shuffle(order)
    blocks = []
    for i in range(0, len(order), 2):
        grp = []
        for (cf, task) in order[i:i+2]:
            grp.extend(tasks_by_cf[(cf, task)])  # 15 items: 5 arms x 3 reps
        rng.shuffle(grp)
        blocks.append(grp)
    if pilot: blocks = blocks[:1]

    done = set()
    ev_path = OUT / "events.jsonl"
    if ev_path.exists():
        for l in open(ev_path):
            try:
                d = json.loads(l)
                if not d.get("error") and d.get("verdict"): done.add(d["event_key"])
            except Exception: pass
    print(f"blocks: {len(blocks)} | items: {sum(len(b) for b in blocks)} | already done: {len(done)}", flush=True)

    import threading
    from concurrent.futures import ThreadPoolExecutor
    lock = threading.Lock()
    fh = open(ev_path, "a")
    req_index = [len(done)]
    t_window = time.time()

    def one(w, bi):
        key = f"{w['cf']}|{w['task']}|{w['arm']}|{w['rep']}"
        if key in done: return
        model, pfn, pkind = arm_spec(w["arm"], w["cf"])
        rec = code[(w["cf"], w["task"])]
        prompt = pfn(prose[w["task"]], rec["code"])
        with lock:
            ri = req_index[0]; req_index[0] += 1
        ev = {"event_key": key, "request_index": ri, "block_index": bi,
              "arm": w["arm"], "cand_family": w["cf"], "evaluator": model,
              "prompt_kind": pkind, "task_id": w["task"], "replicate": w["rep"],
              "T": rec["T"], "code_sha256": rec["code_sha256"],
              "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
              "enqueue_ts": time.time()}
        try:
            d, dur = call(model, prompt)
            ch = d["choices"][0]
            u = d.get("usage") or {}
            ev.update({"response_ts": time.time(), "duration_s": round(dur, 3),
                       "response_model": d.get("model"),
                       "generation_id": d.get("id"),
                       "finish_reason": ch.get("finish_reason"),
                       "prompt_tokens": u.get("prompt_tokens"),
                       "completion_tokens": u.get("completion_tokens"),
                       "cached_tokens": ((u.get("prompt_tokens_details") or {}).get("cached_tokens")),
                       "raw_response": (ch.get("message") or {}).get("content"),
                       "verdict": strict_verdict((ch.get("message") or {}).get("content"))})
        except Exception as e:
            ev.update({"error": f"{type(e).__name__}: {e}"[:300], "response_ts": time.time()})
        with lock:
            fh.write(json.dumps(ev) + "\n"); fh.flush()

    with ThreadPoolExecutor(max_workers=CONC) as ex:
        futs = []
        for bi, blk in enumerate(blocks):
            for w in blk:
                futs.append(ex.submit(one, w, bi))
            if bi % 100 == 0:
                print(f"submitted through block {bi}/{len(blocks)} | {time.time()-t_window:.0f}s", flush=True)
        for f in futs:
            f.result()
    fh.close()
    print("WINDOW COMPLETE", flush=True)

if __name__ == "__main__":
    main()
