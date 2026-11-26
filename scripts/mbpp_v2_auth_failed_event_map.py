#!/usr/bin/env python3
"""Build the cross-eval auth-failure event-id map.

When every transport attempt for an evaluator call fails, mbpp_v2_pilot.call_model
returns a record with a fresh event_id that is never written to events.jsonl
(see the final return in call_model). The 35 GPT-5.4 cross-eval records in
auth_failed_records.jsonl therefore carry selfcheck/steelman event ids with no
matching event. This script links each such record id to the logged HTTP 401
attempts it summarizes.

Matching rule (deterministic, verified unambiguous on the archived data):
  * logged attempts are grouped by (task_id, stage, prompt_sha256), in log order;
  * groups for the same (task_id, stage) are ordered by their first timestamp;
  * auth-failed records for the same task_id are ordered by their timestamp;
  * the i-th record maps to the i-th group for each stage.

Usage: python3 scripts/mbpp_v2_auth_failed_event_map.py [--dir data/mbpp_v2/cross_eval]
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build(directory: Path) -> list[dict]:
    records = [json.loads(l) for l in (directory / "auth_failed_records.jsonl").open(encoding="utf-8") if l.strip()]
    events = [json.loads(l) for l in (directory / "events.jsonl").open(encoding="utf-8") if l.strip()]

    groups: dict[tuple, list[dict]] = {}
    for event in events:
        if event.get("http_status") != 401:
            continue
        groups.setdefault((event["task_id"], event["stage"], event["prompt_sha256"]), []).append(event)
    by_task_stage: dict[tuple, list[list[dict]]] = collections.defaultdict(list)
    for (task_id, stage, _), attempts in groups.items():
        by_task_stage[(task_id, stage)].append(attempts)
    for key in by_task_stage:
        by_task_stage[key].sort(key=lambda attempts: attempts[0]["ts"])

    records_by_task: dict[str, list[dict]] = collections.defaultdict(list)
    for record in sorted(records, key=lambda r: r["ts"]):
        records_by_task[record["task_id"]].append(record)

    rows = []
    for record in records:
        index = records_by_task[record["task_id"]].index(record)
        for stage in ("selfcheck", "steelman"):
            candidates = by_task_stage[(record["task_id"], f"cross_{stage}")]
            if len(candidates) != len(records_by_task[record["task_id"]]):
                raise SystemExit(f"ambiguous match for {record['task_id']} {stage}")
            attempts = candidates[index]
            rows.append({
                "record_event_id": record[f"{stage}_event_id"],
                "generator": record["generator"],
                "evaluator": record["evaluator"],
                "task_id": record["task_id"],
                "stage": f"cross_{stage}",
                "logged_event_ids": [a["event_id"] for a in attempts],
                "logged_http_status": [a["http_status"] for a in attempts],
            })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="data/mbpp_v2/cross_eval")
    args = parser.parse_args()
    directory = ROOT / args.dir
    rows = build(directory)
    out = directory / "auth_failed_event_map.jsonl"
    out.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8")
    print(f"wrote {len(rows)} rows to {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
