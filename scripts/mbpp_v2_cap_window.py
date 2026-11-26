#!/usr/bin/env python3
"""Output-cap window of the MBPP production run (paper, Section IV-A and Appendix C).

The run began with an explicit output cap, which was later removed. The request cap is not
recorded in the ledger; capped attempts are identified as finish_reason == "length" at exactly
8,192 or 16,384 completion tokens. Some retries before the restart ran with a larger limit (a few
responses exceed 8,192 tokens), so those exact-value hits are a lower bound on capped attempts. This script reports those attempts by model, the time of the
last one, and how many task-model pairs have all their referenced events before the run restart
(07:00 UTC, 5 November 2026; the ledger is silent from 06:32 to 07:18).

Usage: python3 scripts/mbpp_v2_cap_window.py
Writes: data/mbpp_v2/analysis/cap_window.json
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROD = ROOT / "data" / "mbpp_v2" / "production"
OUT = ROOT / "data" / "mbpp_v2" / "analysis" / "cap_window.json"
CUT = "2026-11-05T07:00"
CAPS = (8192, 16384)


def main() -> int:
    events = {}
    with (PROD / "events.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            e = json.loads(line)
            events[e["event_id"]] = e

    def tokens(e):
        return (e.get("usage") or {}).get("completion_tokens")

    capped = [e for e in events.values() if e["finish_reason"] == "length" and tokens(e) in CAPS]
    derived = [json.loads(line) for line in (PROD / "derived_records.jsonl").open(encoding="utf-8")]
    keys = ("generation_event_id", "selfcheck_event_id", "steelman_event_id")
    under_cap = mixed = referenced_capped = 0
    for d in derived:
        ts = [events[d[k]]["ts"] for k in keys if d.get(k) and d[k] in events]
        referenced_capped += sum(1 for k in keys if d.get(k) in events
                                 and events[d[k]]["finish_reason"] == "length"
                                 and tokens(events[d[k]]) in CAPS)
        if ts and max(ts) < CUT:
            under_cap += 1
        elif ts and min(ts) < CUT:
            mixed += 1

    result = {
        "capped_attempts": len(capped),
        "capped_attempts_by_model_and_cap": {f"{m} @ {t}": n for (m, t), n in
                                             sorted(Counter((e["model"], tokens(e)) for e in capped).items())},
        "last_capped_attempt_ts": max(e["ts"] for e in capped),
        "pairs_total": len(derived),
        "pairs_completed_under_cap": under_cap,
        "pairs_straddling_restart": mixed,
        "referenced_events_truncated_at_cap": referenced_capped,
    }
    OUT.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
