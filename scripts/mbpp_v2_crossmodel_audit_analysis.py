"""Analysis for the corrected two-auditor MBPP v2 cross-model audit.

Classes are defined on parsed labels only (never overwritten):
  KNOWABLE_agreed      both auditors parsed, both KNOWABLE
  NOT_KNOWABLE_agreed  both auditors parsed, both NOT_KNOWABLE
  split                both parsed, they disagree (reported, FA reported too)
  no_class             any unparsed label
Cohen's kappa between the two auditors replaces Fleiss'.
Report sections (a)-(g) as originally specified, plus:
  (h) sanity: responses matching /\b(empty|missing (the )?function|does not
      define|no code)\b/i per auditor — must be near zero.
Analysis plan fixed before results were seen (2026-11-24).
"""

import json
import math
import os
import random
import re
from collections import Counter, defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUD = os.path.join(REPO, "data", "mbpp_v2", "crossmodel_audit")
INVALID = os.path.join(REPO, "data", "mbpp_v2", "crossmodel_audit_INVALID_empty_code")
BOOTSTRAP_N = 2000
BOOTSTRAP_SEED = 20261124
EMPTY_RX = re.compile(r"\b(empty|missing (the )?function|does not define|no code)\b", re.I)
CLASSES = ["KNOWABLE_agreed", "NOT_KNOWABLE_agreed", "split", "no_class"]


def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return None
    p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return {"p": round(p, 4), "lo": round(center - half, 4), "hi": round(center + half, 4), "k": k, "n": n}


def cohen_kappa(a, b):
    cats = sorted(set(a) | set(b))
    n = len(a)
    if n == 0:
        return None
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in cats)
    if pe == 1:
        return 1.0 if po == 1 else 0.0
    return round((po - pe) / (1 - pe), 4)


def load_jsonl(path):
    out = []
    if os.path.exists(path):
        for line in open(path):
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def main():
    import sys
    dirs = sys.argv[1].split(",") if len(sys.argv) > 1 else [AUD]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else AUD
    labels, events = [], []
    for d in dirs:
        labels += load_jsonl(os.path.join(REPO, d, "labels.jsonl") if not os.path.isabs(d) else os.path.join(d, "labels.jsonl"))
        events += load_jsonl(os.path.join(REPO, d, "events.jsonl") if not os.path.isabs(d) else os.path.join(d, "events.jsonl"))
    sample = []
    for d in dirs:
        sample += load_jsonl(os.path.join(REPO, d, "sample.jsonl") if not os.path.isabs(d) else os.path.join(d, "sample.jsonl"))
    auditors = sorted({r["auditor"] for r in labels}) or json.load(open(os.path.join(dirs[0], "auditors.json") if not os.path.isabs(dirs[0]) else os.path.join(dirs[0], "auditors.json")))["auditors"]
    a1, a2 = auditors[0], auditors[1]

    lab = {}
    for r in labels:
        lab[(r["auditor"], r["generator"], r["task_id"])] = r["label"]

    # (a) unparsed per auditor
    unparsed = {a: sum(1 for r in labels if r["auditor"] == a and r["label"] == "unparsed") for a in (a1, a2)}

    # (b) Cohen's kappa on rows where both parsed
    l1, l2 = [], []
    for row in sample:
        x = lab.get((a1, row["generator"], row["task_id"]))
        y = lab.get((a2, row["generator"], row["task_id"]))
        if x in ("KNOWABLE", "NOT_KNOWABLE") and y in ("KNOWABLE", "NOT_KNOWABLE"):
            l1.append(x)
            l2.append(y)
    kappa = cohen_kappa(l1, l2)
    kappa_n = len(l1)

    # (c) classes
    consensus = {}
    for row in sample:
        x = lab.get((a1, row["generator"], row["task_id"]))
        y = lab.get((a2, row["generator"], row["task_id"]))
        ok_x = x in ("KNOWABLE", "NOT_KNOWABLE")
        ok_y = y in ("KNOWABLE", "NOT_KNOWABLE")
        key = (row["generator"], row["task_id"])
        if ok_x and ok_y:
            if x == y:
                consensus[key] = ("KNOWABLE_agreed" if x == "KNOWABLE" else "NOT_KNOWABLE_agreed")
            else:
                consensus[key] = "split"
        else:
            consensus[key] = "no_class"

    # (d) shares
    families = sorted({r["generator"] for r in sample})

    def class_shares(rows):
        out = {}
        for cl in CLASSES:
            sel = [r for r in rows if consensus[(r["generator"], r["task_id"])] == cl]
            out[cl] = wilson(len(sel), len(rows))
        return out
    shares_overall = class_shares(sample)
    shares_by_fam = {f: class_shares([r for r in sample if r["generator"] == f]) for f in families}
    class_counts_by_family = {f: {cl: sum(1 for r in sample if r["generator"] == f
                                          and consensus[(r["generator"], r["task_id"])] == cl)
                                  for cl in CLASSES} for f in families}

    # derived records (self-check / steelman)
    drv = {}
    for d in load_jsonl(os.path.join(REPO, "data", "mbpp_v2", "production", "derived_records.jsonl")):
        drv[(d["model"], d["task_id"])] = d

    def arm_fa(rows, arm):
        vkey, ekey = "%s_verdict" % arm, "%s_extraction" % arm
        tot_k = tot_n = 0
        for row in rows:
            d = drv.get((row["generator"], row["task_id"]))
            if not d or d.get(ekey) != "parsed" or d.get(vkey) not in ("ACCEPT", "REJECT"):
                continue
            tot_n += 1
            tot_k += 1 if d[vkey] == "ACCEPT" else 0
        return wilson(tot_k, tot_n)

    def arm_fa_by_class(rows, arm):
        out = {}
        for cl in ["KNOWABLE_agreed", "split", "NOT_KNOWABLE_agreed"]:
            sel = [r for r in rows if consensus[(r["generator"], r["task_id"])] == cl]
            out[cl] = arm_fa(sel, arm)
        return out

    # (e) self-check + steelman FA, overall + per family
    fa = {}
    for arm in ("selfcheck", "steelman"):
        fa[arm] = {"overall": arm_fa_by_class(sample, arm),
                   "by_family": {f: arm_fa_by_class([r for r in sample if r["generator"] == f], arm) for f in families}}

    # (f) cross-evaluation matrix
    cross = load_jsonl(os.path.join(REPO, "data", "mbpp_v2", "cross_eval", "cross_records.jsonl"))
    cross_fa = {}
    for arm in ("selfcheck", "steelman"):
        vkey = "%s_verdict" % arm

        def cross_by_class(rows):
            out = {}
            for cl in ["KNOWABLE_agreed", "split", "NOT_KNOWABLE_agreed"]:
                sel = [r for r in rows if consensus.get((r["generator"], r["task_id"])) == cl]
                vv = [r for r in sel if r.get(vkey) in ("ACCEPT", "REJECT")]
                k = sum(1 for r in vv if r[vkey] == "ACCEPT")
                out[cl] = wilson(k, len(vv))
            return out
        rows_all = [r for r in cross if (r["generator"], r["task_id"]) in {(s["generator"], s["task_id"]) for s in sample}
                    and r["evaluator"] != r["generator"]]
        cross_fa[arm] = {"overall": cross_by_class(rows_all),
                         "by_family": {f: cross_by_class([r for r in rows_all if r["generator"] == f]) for f in families}}

    # (g) task-clustered bootstrap: selfcheck FA, KNOWABLE_agreed minus NOT_KNOWABLE_agreed
    cand_by_task = defaultdict(list)
    for row in sample:
        cl = consensus[(row["generator"], row["task_id"])]
        if cl in ("KNOWABLE_agreed", "NOT_KNOWABLE_agreed"):
            d = drv.get((row["generator"], row["task_id"]))
            if d and d.get("selfcheck_extraction") == "parsed" and d.get("selfcheck_verdict") in ("ACCEPT", "REJECT"):
                cand_by_task[row["task_id"]].append((cl, d["selfcheck_verdict"] == "ACCEPT"))

    def rate(pool, cl):
        vals = [acc for c, acc in pool if c == cl]
        return (sum(vals) / len(vals)) if vals else None

    rng = random.Random(BOOTSTRAP_SEED)
    task_list = sorted(cand_by_task.keys())
    diffs = []
    for _ in range(BOOTSTRAP_N):
        pool = [x for t in [rng.choice(task_list) for _ in range(len(task_list))] for x in cand_by_task[t]]
        rk, rn = rate(pool, "KNOWABLE_agreed"), rate(pool, "NOT_KNOWABLE_agreed")
        if rk is None or rn is None:
            continue
        diffs.append(rk - rn)
    diffs.sort()
    allpool = [x for t in task_list for x in cand_by_task[t]]
    pk, pn = rate(allpool, "KNOWABLE_agreed"), rate(allpool, "NOT_KNOWABLE_agreed")
    boot = {"point_estimate_diff": round(pk - pn, 4) if (pk is not None and pn is not None) else None,
            "ci95": [round(diffs[int(0.025 * len(diffs))], 4), round(diffs[int(0.975 * len(diffs)) - 1], 4)],
            "n_resamples": len(diffs), "n_tasks": len(task_list), "seed": BOOTSTRAP_SEED}

    # (h) sanity: empty-program mentions per auditor
    sanity = {a: sum(1 for r in events if r.get("auditor_requested") == a and EMPTY_RX.search(r.get("raw") or ""))
              for a in (a1, a2)}
    total_cost = sum(float(r.get("cost") or 0) for r in events)
    retries = sum(1 for r in events if r.get("attempt", 1) > 1)

    result = {"auditors": [a1, a2], "n_sample": len(sample), "n_calls": len(events),
              "retries": retries, "total_cost_usd": round(total_cost, 2),
              "a_unparsed_by_auditor": unparsed,
              "b_cohen_kappa": {"kappa": kappa, "n": kappa_n},
              "c_class_counts": {cl: sum(1 for v in consensus.values() if v == cl) for cl in CLASSES},
              "d_shares": {"overall": shares_overall, "by_family": shares_by_fam},
              "d_class_counts_by_family": class_counts_by_family,
              "e_false_acceptance": fa,
              "f_cross_matrix_false_acceptance": cross_fa,
              "g_selfcheck_fa_diff_KAgreed_minus_NKAgreed": boot,
              "h_empty_program_mentions": sanity}
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "analysis.json"), "w") as f:
        json.dump(result, f, indent=1)

    L = []
    L.append("MBPP v2 cross-model audit (corrected rerun) - summary")
    L.append("auditors: %s, %s (both reasoning effort low)" % (a1, a2))
    L.append("sample: %d incorrect candidates across %d families, calls: %d, retries: %d, total cost: $%.2f"
             % (len(sample), len(families), len(events), retries, total_cost))
    L.append("")
    L.append("(a) unparsed per auditor:")
    L.append("    %s: %d" % (a1, unparsed[a1]))
    L.append("    %s: %d" % (a2, unparsed[a2]))
    L.append("")
    L.append("(b) agreement (both parsed, n=%d):" % kappa_n)
    L.append("    Cohen kappa = %s" % kappa)
    L.append("")
    L.append("(c) class counts (parsed labels only, no overwriting):")
    for cl in CLASSES:
        L.append("    %s: %d" % (cl, result["c_class_counts"][cl]))
    L.append("")
    L.append("(d) class shares, Wilson 95%%:")
    for cl in CLASSES:
        s = shares_overall[cl]
        if s:
            L.append("    %s: %.3f [%.3f, %.3f] (n=%d)" % (cl, s["p"], s["lo"], s["hi"], s["n"]))
    L.append("    per-family class counts:")
    for f_ in families:
        L.append("      %s: %s" % (f_, json.dumps(class_counts_by_family[f_])))
    L.append("")
    for arm in ("selfcheck", "steelman"):
        L.append("(%s) false acceptance of incorrect candidates:" % arm)
        for cl in ["KNOWABLE_agreed", "split", "NOT_KNOWABLE_agreed"]:
            s = fa[arm]["overall"][cl]
            if s:
                L.append("    overall %s: FA %.3f [%.3f, %.3f] (n=%d)" % (cl, s["p"], s["lo"], s["hi"], s["n"]))
        for f_ in families:
            for cl in ["KNOWABLE_agreed", "NOT_KNOWABLE_agreed"]:
                s = fa[arm]["by_family"][f_][cl]
                if s:
                    L.append("      %s %s: FA %.3f [%.3f, %.3f] (n=%d)" % (f_, cl, s["p"], s["lo"], s["hi"], s["n"]))
        L.append("")
    for arm in ("selfcheck", "steelman"):
        L.append("(cross matrix, %s arm) false acceptance by other evaluators (row unit):" % arm)
        for cl in ["KNOWABLE_agreed", "split", "NOT_KNOWABLE_agreed"]:
            s = cross_fa[arm]["overall"][cl]
            if s:
                L.append("    overall %s: FA %.3f [%.3f, %.3f] (n=%d)" % (cl, s["p"], s["lo"], s["hi"], s["n"]))
        L.append("")
    L.append("(g) self-check FA difference, KNOWABLE_agreed minus NOT_KNOWABLE_agreed:")
    L.append("    diff = %s, task-clustered bootstrap 95%% CI %s (%d resamples of %d tasks, seed %d)"
             % (boot["point_estimate_diff"], boot["ci95"], boot["n_resamples"], boot["n_tasks"], BOOTSTRAP_SEED))
    L.append("")
    L.append("(h) sanity - empty/missing-program mentions per auditor (must be ~0):")
    for a in (a1, a2):
        L.append("    %s: %d" % (a, sanity[a]))
    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
