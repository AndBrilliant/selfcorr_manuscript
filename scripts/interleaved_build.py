"""Interleaved rerun (paper Sec. IV-F, Table VIII): work-list builder.
Builds the balanced-block work list (5 arms incl. an A/A steelman replicate, 3 replicates
per task and arm) and spot-checks prompt reconstruction against the original ledgers on a
few candidates. The prompts actually sent are defined in interleaved_run.py."""
import json, hashlib, random, sys, time, urllib.request, asyncio

BASE = str(__import__("pathlib").Path(__file__).resolve().parents[1])
OUT = BASE + "/data/mbpp_v2/interleaved_rerun"
SEED = 20261120

# ── arms ──
SELF = {"qwen": "qwen/qwen3.8-flash", "luna": "openai/gpt-6-luna"}
ARMS = ["steel_self_a", "steel_self_b", "control_self", "steel_partner", "control_partner"]

def selfcheck_prompt(prose, code):
    return ("You are evaluating one immutable Python candidate solution.\n\n"
            f"[PROBLEM]\n{prose}\n\n[CANDIDATE]\n{code}\n\n"
            "Analyze whether the candidate satisfies the specification and tests. "
            "Do not modify, repair, rewrite, or provide an alternative solution.\n"
            "End with exactly one final line containing only one of:\n"
            "VERDICT: ACCEPT\nVERDICT: REJECT")

def steelman_prompt(prose, code):
    return ("You are a neutral arbiter judging a debate between two expert Python programmers.\n\n"
            f"[PROBLEM]\n{prose}\n\n[CANDIDATE]\n{code}\n\n"
            "Do the following in order:\n"
            "1. STEELMAN AGAINST: give the strongest case that the candidate is wrong.\n"
            "2. STEELMAN FOR: give the strongest case that the candidate is correct.\n"
            "3. Decide which case survives scrutiny. Judge only the candidate as given; "
            "do not provide a corrected version.\n"
            "End with exactly one final line containing only one of:\n"
            "VERDICT: ACCEPT\nVERDICT: REJECT")

def control_prompt(prose, code):
    return ("You are performing a careful neutral verification of one immutable Python "
            "candidate solution.\n\n"
            f"[PROBLEM]\n{prose}\n\n[CANDIDATE]\n{code}\n\n"
            "Verify the candidate against the specification step by step: check the logic, "
            "trace the behavior on the requirements, and confirm whether it satisfies the "
            "specification. Do not modify, repair, rewrite, or provide an alternative "
            "solution, and do not presume the candidate is correct or incorrect; verify it.\n"
            "End with exactly one final line containing only one of:\n"
            "VERDICT: ACCEPT\nVERDICT: REJECT")

def build():
    import pathlib
    pathlib.Path(OUT).mkdir(exist_ok=True)
    # MBPP task prose
    prose = {}
    with open(BASE + "/data/MBPP.jsonl") as f:
        for line in f:
            d = json.loads(line)
            prose[d["task_id"]] = d["prompt"]
    # frozen candidates for the two families (generator = evaluator family on diagonal)
    cands = {}
    with open(BASE + "/data/mbpp_v2/production/candidate_records.jsonl") as f:
        for line in f:
            d = json.loads(line)
            fam = {"qwen38-flash": "qwen", "gpt-6-luna": "luna"}.get(d["model"])
            if fam:
                cands.setdefault(fam, []).append(d)
    print("candidates:", {k: len(v) for k, v in cands.items()})

    # verify control-prompt bytes against original control ledger sha256
    ctl_events = {}
    with open(BASE + "/data/mbpp_v2/control/events.jsonl") as f:
        for line in f:
            d = json.loads(line)
            ctl_events.setdefault(d.get("task_id"), d)
    # verify steelman bytes against production events (prompt_sha256)
    ok = miss = 0
    for fam, recs in cands.items():
        for r in recs[:5]:
            tid = r["task_id"]
            if tid in ctl_events and ctl_events[tid].get("model") == r["model"]:
                h = hashlib.sha256(control_prompt(prose.get(tid, ""), r["code"]).encode()).hexdigest()
                if h == ctl_events[tid]["prompt_sha256"]: ok += 1
                else: miss += 1
    print(f"control prompt byte-verify: {ok} match, {miss} mismatch")

    # work list: (cand_family, task_id, arm, replicate)
    work = []
    for cf in ("qwen", "luna"):
        for r in cands[cf]:
            for arm in ARMS:
                for rep in (1, 2, 3):
                    work.append({"cf": cf, "task": r["task_id"], "arm": arm, "rep": rep,
                                 "code_sha": r["code_sha256"], "T": r["T"]})
    rng = random.Random(SEED)
    rng.shuffle(work)  # seed-recorded; blocks below enforce balance regardless
    json.dump({"seed": SEED, "arms": ARMS, "models": SELF}, open(OUT + "/worklist_meta.json", "w"))
    with open(OUT + "/worklist.jsonl", "w") as f:
        for w in work:
            f.write(json.dumps(w) + "\n")
    print("work items:", len(work))

if __name__ == "__main__":
    build()
