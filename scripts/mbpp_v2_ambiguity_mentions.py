#!/usr/bin/env python3
"""Do evaluators flag ambiguity in the incorrect code they accept?

For every production evaluator verdict (self-check and steelman) on a frozen candidate
with assigned ground truth, searches the evaluator's raw response for language that flags
an ambiguous or underspecified task (fixed regular expression below, set before the
search was run). Reports, per family and arm, the share of responses with such a flag
among incorrect candidates accepted, incorrect candidates rejected, and correct
candidates, plus pooled counts.

Reads ledgers only; no network, no code execution. Computed after data collection.
Outputs: data/mbpp_v2/analysis/ambiguity_mentions.json and _summary.txt
"""
import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data/mbpp_v2"
FAMS = ["gpt-5.4", "ds-v4-flash", "glm-5.3-flash", "qwen38-flash", "gpt-6-luna"]
PAT = re.compile(
    r"ambigu|underspecif|under-specif|not specified|unspecified|unclear|could be interpreted|"
    r"can be interpreted|interpretation|depends on (the )?(interpretation|reading|whether)|"
    r"one reading|another reading|1-index|one-index|0-index|zero-index|not (clearly )?stated|"
    r"does not specify|doesn't specify|isn't specified|is not specified",
    re.I)


def main():
    ev = {}
    for l in open(D / "production/events.jsonl"):
        if l.strip():
            e = json.loads(l)
            ev[e["event_id"]] = e.get("raw_response") or ""
    cells = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for l in open(D / "production/derived_records.jsonl"):
        if not l.strip():
            continue
        r = json.loads(l)
        if r["T"] not in (0, 1):
            continue
        for arm in ("selfcheck", "steelman"):
            v = r.get(f"{arm}_verdict")
            if r.get(f"{arm}_extraction") != "parsed" or v not in ("ACCEPT", "REJECT"):
                continue
            txt = ev.get(r.get(f"{arm}_event_id"), "")
            grp = "correct" if r["T"] == 1 else ("incorrect_accepted" if v == "ACCEPT" else "incorrect_rejected")
            c = cells[(r["model"], arm)][grp]
            c[0] += 1
            c[1] += bool(PAT.search(txt))
    out, L = {}, ["Ambiguity flags in evaluator responses (MBPP production)", ""]
    pooled = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for (m, arm), g in sorted(cells.items()):
        out[f"{m}|{arm}"] = {k: {"n": n, "flagged": f, "pct": round(100 * f / n, 1) if n else None}
                            for k, (n, f) in g.items()}
        for k, (n, f) in g.items():
            pooled[arm][k][0] += n
            pooled[arm][k][1] += f
        L.append(f"{m:14s} {arm:9s} " + "  ".join(
            f"{k} {out[f'{m}|{arm}'][k]['pct']}% (n={out[f'{m}|{arm}'][k]['n']})" for k in
            ("incorrect_accepted", "incorrect_rejected", "correct") if k in g))
    out["pooled"] = {arm: {k: {"n": n, "flagged": f, "pct": round(100 * f / n, 1)} for k, (n, f) in g.items()}
                     for arm, g in pooled.items()}
    L.append("")
    for arm, g in out["pooled"].items():
        L.append(f"POOLED {arm}: " + "  ".join(f"{k} {v['pct']}% ({v['flagged']}/{v['n']})" for k, v in g.items()))
    (D / "analysis/ambiguity_mentions.json").write_text(json.dumps(out, indent=1))
    (D / "analysis/ambiguity_mentions_summary.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
