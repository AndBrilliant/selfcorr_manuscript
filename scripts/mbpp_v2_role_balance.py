#!/usr/bin/env python3
"""Evaluator and generator roles in the cross-evaluation matrix.

For each model family, on incorrect candidates only and excluding same-family cells:
  evaluator rate = mean false acceptance it gives the four other families' candidates
                   (row mean of the matrix);
  generator rate = mean false acceptance its own candidates receive from the four other
                   families' evaluators (column mean);
  gap            = evaluator rate - generator rate (percentage points);
  relative gap   = gap / mean(evaluator rate, generator rate), in percent.
Each generator rate is measured on that family's own incorrect candidates, so it
reflects which tasks the family got wrong as well as how its incorrect code reads.
95% intervals: task-clustered percentile bootstrap, 2,000 resamples, seed 20261123.
Computed after data collection; not part of the original analysis plan.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data/mbpp_v2"
OUT = D / "cross_eval"
MODELS = ["gpt-5.4", "ds-v4-flash", "glm-5.3-flash", "qwen38-flash", "gpt-6-luna"]
NAMES = {"gpt-5.4": "GPT-5.4", "ds-v4-flash": "DeepSeek", "glm-5.3-flash": "GLM",
         "qwen38-flash": "Qwen", "gpt-6-luna": "GPT-6-Luna"}
V = {"ACCEPT", "REJECT"}
B, SEED = 2000, 20261123


def load_prod() -> dict:
    prod = {}
    for l in (D / "production/derived_records.jsonl").read_text().splitlines():
        if l.strip():
            r = json.loads(l)
            prod[(r["model"], r["task_id"])] = r
    return prod


def load_cross(arm: str, prod: dict) -> pd.DataFrame:
    rows = []
    for l in (D / "cross_eval/cross_records.jsonl").read_text().splitlines():
        if not l.strip():
            continue
        r = json.loads(l)
        if r.get(f"{arm}_verdict") not in V:
            continue
        p = prod.get((r["generator"], r["task_id"]))
        if not p or p.get("T") != 0:
            continue
        rows.append((r["evaluator"], r["generator"], r["task_id"],
                     int(r[f"{arm}_verdict"] == "ACCEPT")))
    return pd.DataFrame(rows, columns=["evaluator", "generator", "task_id", "accept"])


def roles(df: pd.DataFrame) -> dict:
    cell = df.groupby(["evaluator", "generator"]).accept.mean() * 100
    out = {}
    for m in MODELS:
        ev = float(np.mean([cell[(m, g)] for g in MODELS if g != m]))
        ge = float(np.mean([cell[(e, m)] for e in MODELS if e != m]))
        out[m] = (ev, ge, ev - ge, (ev - ge) / ((ev + ge) / 2) * 100)
    return out


def main() -> None:
    prod = load_prod()
    rng = np.random.default_rng(SEED)
    result, lines = {}, ["Evaluator and generator roles (incorrect candidates, off-diagonal cells)",
                         f"Task-clustered bootstrap: {B} resamples, seed {SEED}", ""]
    for arm in ["selfcheck", "steelman"]:
        df = load_cross(arm, prod)
        base = roles(df)
        groups = {t: g for t, g in df.groupby("task_id")}
        tasks = np.array(list(groups))
        boots = {m: [] for m in MODELS}
        for _ in range(B):
            s = pd.concat([groups[t] for t in rng.choice(tasks, len(tasks))])
            st = roles(s)
            for m in MODELS:
                boots[m].append(st[m])
        result[arm] = {"n_records": int(len(df)), "n_tasks": int(len(tasks)), "models": {}}
        lines.append(f"--- {arm} (n={len(df)}, tasks={len(tasks)})")
        for m in sorted(MODELS, key=lambda m: abs(base[m][2])):
            ev, ge, gap, rel = base[m]
            b = np.array(boots[m])
            gl, gh = np.percentile(b[:, 2], [2.5, 97.5])
            rl, rh = np.percentile(b[:, 3], [2.5, 97.5])
            result[arm]["models"][m] = {
                "evaluator_rate": round(ev, 2), "generator_rate": round(ge, 2),
                "gap": round(gap, 2), "gap_ci95": [round(gl, 2), round(gh, 2)],
                "relative_gap_pct": round(rel, 2), "relative_gap_ci95": [round(rl, 2), round(rh, 2)]}
            lines.append(f"{NAMES[m]:11s} evaluator {ev:5.1f}  generator {ge:5.1f}  "
                         f"gap {gap:+5.1f} [{gl:+.1f},{gh:+.1f}]  relative {rel:+5.1f}% [{rl:+.1f},{rh:+.1f}]")
        lines.append("")
    (OUT / "role_balance.json").write_text(json.dumps(result, indent=1))
    (OUT / "role_balance_summary.txt").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
