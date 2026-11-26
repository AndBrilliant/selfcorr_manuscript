#!/usr/bin/env python3
"""Interleaved rerun: session timing and Holm correction (paper, Section IV-F).

- Session timing from response timestamps in data/mbpp_v2/interleaved_rerun/events.jsonl:
  total span, the two gaps between sessions, session sizes, and how many blocks span a gap.
- Holm correction over the four prompt contrasts of Table IX (each evaluator on its own and on
  the other family's candidates) using the exact McNemar p-values in
  data/mbpp_v2/analysis/interleaved_results.json (run interleaved_analysis.py first).

Usage: python3 scripts/interleaved_sessions.py
Writes: data/mbpp_v2/analysis/interleaved_sessions.json
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVENTS = ROOT / "data" / "mbpp_v2" / "interleaved_rerun" / "events.jsonl"
RESULTS = ROOT / "data" / "mbpp_v2" / "analysis" / "interleaved_results.json"
OUT = ROOT / "data" / "mbpp_v2" / "analysis" / "interleaved_sessions.json"
CONTRASTS = ["prompt: Qwen judging own", "prompt: Qwen judging GPT-6-Luna",
             "prompt: GPT-6-Luna judging own", "prompt: GPT-6-Luna judging Qwen"]


def main() -> int:
    events = [json.loads(line) for line in EVENTS.open(encoding="utf-8")]
    ts = sorted(float(e["response_ts"]) for e in events)
    gaps = sorted(range(len(ts) - 1), key=lambda i: ts[i + 1] - ts[i])[-2:]
    cut1, cut2 = sorted(gaps)
    bounds = [ts[cut1], ts[cut2]]

    def session(t: float) -> int:
        return 0 if t <= bounds[0] else (1 if t <= bounds[1] else 2)

    blocks = defaultdict(set)
    for e in events:
        blocks[e["block_index"]].add(session(float(e["response_ts"])))

    res = json.loads(RESULTS.read_text(encoding="utf-8"))
    ps = sorted((res[k]["p"], k) for k in CONTRASTS)
    holm, running = {}, 0.0
    for rank, (p, k) in enumerate(ps):
        running = max(running, min(1.0, p * (len(ps) - rank)))
        holm[k] = round(running, 4)

    result = {
        "events": len(ts),
        "span_hours": round((ts[-1] - ts[0]) / 3600, 2),
        "gap_hours": [round((ts[i + 1] - ts[i]) / 3600, 2) for i in (cut1, cut2)],
        "session_sizes": [cut1 + 1, cut2 - cut1, len(ts) - cut2 - 1],
        "blocks": len(blocks),
        "blocks_split_across_sessions": sum(len(s) > 1 for s in blocks.values()),
        "holm_adjusted_p_four_prompt_contrasts": holm,
    }
    OUT.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
