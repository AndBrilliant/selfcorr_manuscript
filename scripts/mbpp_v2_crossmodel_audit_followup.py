#!/usr/bin/env python3
"""Follow-up to the cross-model audit: false acceptance and the shared-error split by
whether an error was knowable from the description.

Input: a labels.jsonl (auditor, generator, task_id, label); default
data/mbpp_v2/crossmodel_audit_all/labels.jsonl (all 1,986), or pass
data/mbpp_v2/crossmodel_audit/labels.jsonl for the first 650-candidate audit.
Class per sampled candidate: AGREED_KNOWABLE / AGREED_NOT_KNOWABLE when every auditor
parsed and all labels agree; SPLIT when all parsed and they disagree; NO_CLASS otherwise.

Reports, per class:
  1. self-check and steelman false acceptance of the generating family's own evaluator
     (production verdicts), Wilson 95% intervals;
  2. cross-family false acceptance (other evaluators, cross_records);
  3. the shared-error split of mbpp_v2_shared_output_agreement.py restricted to the
     class's sampled candidates: false acceptance when the evaluator's own code is right,
     wrong with the same outputs, wrong with different outputs.
Analysis plan fixed before the corrected labels were seen. Executes frozen candidates
offline (10 s timeout); no network.
Outputs: followup.json and followup_summary.txt next to the labels file.
"""
import json
import math
import multiprocessing as mp
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from mbpp_v2_error_severity_audit import split_parts  # noqa: E402
from mbpp_v2_shared_output_agreement import lhs, run  # noqa: E402

D = ROOT / "data/mbpp_v2"
AUD = D / "crossmodel_audit_all"
FAMS = ["gpt-5.4", "ds-v4-flash", "glm-5.3-flash", "qwen38-flash", "gpt-6-luna"]
OK = ("KNOWABLE", "NOT_KNOWABLE")


def wilson(k, n, z=1.959964):
    if not n:
        return {"n": 0, "fa_pct": None, "ci95": None}
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return {"n": n, "fa_pct": round(100 * p, 1), "ci95": [round(100 * (c - h), 1), round(100 * (c + h), 1)]}


def jl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def main(labels_path=AUD / "labels.jsonl"):
    labs = defaultdict(dict)
    for r in jl(labels_path):
        labs[(r["generator"], r["task_id"])][r["auditor"]] = r["label"]
    auditors = sorted({a for v in labs.values() for a in v})
    cls = {}
    for k, v in labs.items():
        ls = [v.get(a) for a in auditors]
        if all(l in OK for l in ls):
            cls[k] = ("AGREED_" + ls[0]) if len(set(ls)) == 1 else "SPLIT"
        else:
            cls[k] = "NO_CLASS"
    classes = ["AGREED_KNOWABLE", "AGREED_NOT_KNOWABLE", "SPLIT", "NO_CLASS"]

    der = {(r["model"], r["task_id"]): r for r in jl(D / "production/derived_records.jsonl")}
    T = {k: r["T"] for k, r in der.items()}
    cross = jl(D / "cross_eval/cross_records.jsonl")

    # signatures for the sampled candidates and the evaluators' own wrong code on those tasks
    tasks = {json.loads(l)["task_id"]: json.loads(l) for l in open(ROOT / "data/MBPP.jsonl") if l.strip()}
    code = {(c["model"], c["task_id"]): c.get("code") or "" for c in jl(D / "production/candidate_records.jsonl")}
    need = set(cls)
    for (g, t) in cls:
        for e in FAMS:
            if T.get((e, t)) == 0:
                need.add((e, t))
    keys, jobs = [], []
    for k in sorted(need):
        if code.get(k):
            setup, asserts = split_parts(tasks[k[1]]["test"])
            keys.append(k)
            jobs.append((code[k], setup, [lhs(a) for a in asserts]))
    with mp.Pool(16) as pool:
        sig = dict(zip(keys, pool.map(run, jobs, chunksize=4)))

    out = {"auditors": auditors, "class_counts": dict(Counter(cls.values()))}
    L = [f"Cross-model audit follow-up; auditors: {', '.join(auditors)}",
         f"classes: {out['class_counts']}", ""]
    for c in classes:
        mem = [k for k, v in cls.items() if v == c]
        o = {}
        for arm in ("selfcheck", "steelman"):
            rows = [der[k] for k in mem if k in der and der[k].get(f"{arm}_extraction") == "parsed"
                    and der[k].get(f"{arm}_verdict") in ("ACCEPT", "REJECT")]
            o[f"own_{arm}"] = wilson(sum(r[f"{arm}_verdict"] == "ACCEPT" for r in rows), len(rows))
            cr = [r for r in cross if (r["generator"], r["task_id"]) in set(mem)
                  and r.get(f"{arm}_verdict") in ("ACCEPT", "REJECT")]
            o[f"cross_{arm}"] = wilson(sum(r[f"{arm}_verdict"] == "ACCEPT" for r in cr), len(cr))
            grp = defaultdict(lambda: [0, 0])
            for r in cr:
                E, G, t = r["evaluator"], r["generator"], r["task_id"]
                te = T.get((E, t))
                if te == 1:
                    g = "own_right"
                elif te == 0 and (E, t) in sig and (G, t) in sig:
                    g = "own_wrong_same_output" if sig[(E, t)] == sig[(G, t)] else "own_wrong_different_output"
                else:
                    continue
                grp[g][0] += r[f"{arm}_verdict"] == "ACCEPT"
                grp[g][1] += 1
            o[f"shared_split_{arm}"] = {g: wilson(*v) for g, v in grp.items()}
        out[c] = o
        L.append(f"[{c}] n={len(mem)}")
        for arm in ("selfcheck", "steelman"):
            L.append(f"  {arm}: own FA {o[f'own_{arm}']}; cross FA {o[f'cross_{arm}']}")
            L.append(f"    shared split: {o[f'shared_split_{arm}']}")
    od = Path(labels_path).parent
    (od / "followup.json").write_text(json.dumps(out, indent=1))
    (od / "followup_summary.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else AUD / "labels.jsonl")
