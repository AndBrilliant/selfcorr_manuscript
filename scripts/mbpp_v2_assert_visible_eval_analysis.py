"""Analysis of the evaluator-only one-assert-visible control (paper Sec. IV-G).

For each family and arm, on the same frozen candidates judged in production and with the
first reference assert visible:
  - false acceptance kappa and true acceptance tau, production vs. assert-visible;
  - exact McNemar test on incorrect candidates (production verdict vs. assert-visible
    verdict), Holm-adjusted across the ten family-arm cells;
  - kappa with the assert visible, split by whether the incorrect candidate fails the
    shown assert itself (run under the frozen harness) or passes it and fails a hidden one.
Writes data/mbpp_v2/assert_visible_eval/assert_visible_eval_analysis.json.
"""
import json, sys
from pathlib import Path
from math import comb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.mbpp_v2_pilot as pilot

D = ROOT / "data/mbpp_v2"
OUT = D / "assert_visible_eval"
FAMS = ["gpt-5.4", "ds-v4-flash", "glm-5.3-flash", "qwen38-flash", "gpt-6-luna"]
ARMS = ["selfcheck", "steelman"]


def first_assert_task(task):
    lines = task["test"].splitlines()
    keep, seen = [], False
    for l in lines:
        if l.lstrip().startswith("assert "):
            if not seen:
                keep.append(l); seen = True
        else:
            keep.append(l)
    t = dict(task); t["test"] = "\n".join(keep)
    return t


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def main():
    tasks = pilot.load_task_map()
    prod = {(r["model"], r["task_id"]): r for r in map(json.loads, open(D / "production/derived_records.jsonl"))}
    cands = {(c["model"], c["task_id"]): c for c in map(json.loads, open(D / "production/candidate_records.jsonl"))}
    av = {}
    for l in open(OUT / "records.jsonl"):
        if l.strip():
            r = json.loads(l)
            av[(r["model"], r["task_id"], r["arm"])] = r["verdict"]
    pa = {(r["model"], r["task_id"]): r["per_assert_failed"]
          for r in map(json.loads, open(D / "analysis/error_severity_audit.jsonl"))}
    shown_fail = {}
    for (m, t), c in cands.items():
        if c["T"] == 0 and c.get("code"):
            ok, _ = pilot.run_task_tests(c["code"], first_assert_task(tasks[t]))
            shown_fail[(m, t)] = not ok
    res, ps = {}, []
    for m in FAMS:
        for arm in ARMS:
            rows = []
            for (mm, t), p in prod.items():
                if mm != m or p["T"] not in (0, 1) or p.get(f"{arm}_extraction") != "parsed":
                    continue
                v = av.get((m, t, arm))
                if v not in ("ACCEPT", "REJECT"):
                    continue
                rows.append((t, p["T"], p[f"{arm}_verdict"] == "ACCEPT", v == "ACCEPT"))
            inc = [r for r in rows if r[1] == 0]; cor = [r for r in rows if r[1] == 1]
            b = sum(1 for r in inc if r[2] and not r[3])   # production accept -> visible reject
            c = sum(1 for r in inc if not r[2] and r[3])   # production reject -> visible accept
            fail = [r for r in inc if shown_fail.get((m, r[0]))]
            hid = [r for r in inc if shown_fail.get((m, r[0])) is False]
            pct = lambda xs, i: round(100 * sum(x[i] for x in xs) / len(xs), 1) if xs else None
            res[f"{m}|{arm}"] = {
                "n_incorrect": len(inc), "n_correct": len(cor),
                "kappa_production": pct(inc, 2), "kappa_visible": pct(inc, 3),
                "tau_production": pct(cor, 2), "tau_visible": pct(cor, 3),
                "mcnemar_b_c": [b, c], "p": mcnemar(b, c),
                "n_fail_shown_assert": len(fail), "kappa_visible_fail_shown": pct(fail, 3),
                "n_fail_hidden_only": len(hid), "kappa_visible_fail_hidden_only": pct(hid, 3)}
            allr = [r for r in rows]
            flips = [r for r in allr if r[2] != r[3]]
            from statistics import NormalDist as _N
            def _dp(t, k, n1, n0):
                t = min(max(t / 100, .5 / n1), 1 - .5 / n1); k = min(max(k / 100, .5 / n0), 1 - .5 / n0)
                return round(_N().inv_cdf(t) - _N().inv_cdf(k), 2)
            v = res[f"{m}|{arm}"]
            v["dprime_production"] = _dp(v["tau_production"], v["kappa_production"], len(cor), len(inc))
            v["dprime_visible"] = _dp(v["tau_visible"], v["kappa_visible"], len(cor), len(inc))
            # Same hidden-only subset, description-only verdict (production), paired McNemar.
            v["kappa_production_fail_hidden_only"] = pct(hid, 2)
            hb = sum(1 for r in hid if r[2] and not r[3]); hc = sum(1 for r in hid if not r[2] and r[3])
            v["hidden_only_mcnemar_b_c"] = [hb, hc]; v["hidden_only_p"] = mcnemar(hb, hc)
            # d' with correctness defined by the hidden (not shown) asserts only: an incorrect
            # candidate that fails only the shown assert counts as correct under this label.
            # Per-assert outcomes from analysis/error_severity_audit.jsonl (index 0 = shown assert).
            hidT = {r[0]: (r[1] == 1 or not any(pa.get((m, r[0]), [True, True])[1:])) for r in rows}
            inc2 = [r for r in rows if not hidT[r[0]]]; cor2 = [r for r in rows if hidT[r[0]]]
            v["hidden_label_n_incorrect"] = len(inc2); v["hidden_label_n_correct"] = len(cor2)
            v["hidden_label_kappa_production"] = pct(inc2, 2); v["hidden_label_kappa_visible"] = pct(inc2, 3)
            v["hidden_label_tau_production"] = pct(cor2, 2); v["hidden_label_tau_visible"] = pct(cor2, 3)
            v["hidden_label_dprime_production"] = _dp(v["hidden_label_tau_production"], v["hidden_label_kappa_production"], len(cor2), len(inc2))
            v["hidden_label_dprime_visible"] = _dp(v["hidden_label_tau_visible"], v["hidden_label_kappa_visible"], len(cor2), len(inc2))
            v["shown_index_agreement"] = round(100 * sum(1 for r in inc if (m, r[0]) in pa and pa[(m, r[0])][0] == shown_fail.get((m, r[0]))) / max(1, sum(1 for r in inc if (m, r[0]) in pa)), 1)
            res[f"{m}|{arm}"]["verdict_change_pct"] = round(100 * len(flips) / len(allr), 1)
            res[f"{m}|{arm}"]["changes_toward_truth_pct"] = round(100 * sum(1 for r in flips if r[3] == (r[1] == 1)) / len(flips), 1) if flips else None
            ps.append((f"{m}|{arm}", res[f"{m}|{arm}"]["p"]))
    ps.sort(key=lambda x: x[1]); k = len(ps); run = 0
    for i, (key, p) in enumerate(ps):
        run = max(run, min(1.0, (k - i) * p)); res[key]["p_holm"] = run
    for key, v in res.items():
        print(key, v)
    (OUT / "assert_visible_eval_analysis.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
