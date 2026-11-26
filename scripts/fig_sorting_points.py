#!/usr/bin/env python3
"""Build data/mbpp_v2/analysis/fig_sorting_points.json, the input of fig_sorting.py.

Each point: [label, group, delta_c, delta_c_ci95, delta_dprime, delta_dprime_ci95], read
from the analysis outputs (offline; run the listed analysis scripts first):
  (a) round8_interleaved_analysis.json   not_knowable
  (b) p2_information_present.json        C_interleaved_knowable
  (c) p2_information_present.json        A_information_control_fails_shown
  (d) p2_information_present.json        B_generator_sees_assert_fails_shown
  (e,f) round8_attribution_analysis.json pooled.all dc_self/dd_self, dc_user/dd_user
  (g,h) round8_placebo_analysis.json     pooled.exposed dc/dd_placebo, dc/dd_real
Usage: python scripts/fig_sorting_points.py
       python scripts/fig_sorting.py data/mbpp_v2/analysis/fig_sorting_points.json fig_sorting.pdf
"""
import json
from pathlib import Path

A = Path(__file__).resolve().parents[1] / "data/mbpp_v2/analysis"


def j(name):
    return json.load(open(A / name))


def flat(lbl, g, o):
    return [lbl, g, o["dc"], o["dc_ci95"], o["dd"], o["dd_ci95"]]


def est(lbl, g, c, d):
    return [lbl, g, c["est"], c["ci95"], d["est"], d["ci95"]]


il, p2 = j("round8_interleaved_analysis.json"), j("p2_information_present.json")
at, pl = j("round8_attribution_analysis.json")["pooled"]["all"], j("round8_placebo_analysis.json")["pooled"]["exposed"]
pts = [
    flat("Steelman, convention unstated", "prompt", il["not_knowable"]),
    flat("Steelman, error visible in prose", "prompt", p2["C_interleaved_knowable"]),
    flat("Steelman, assert shown", "prompt", p2["A_information_control_fails_shown"]),
    flat("Steelman, assert seen by generator", "prompt", p2["B_generator_sees_assert_fails_shown"]),
    est('"You wrote this"', "attr", at["dc_self"], at["dd_self"]),
    est('"The user wrote this"', "attr", at["dc_user"], at["dd_user"]),
    est("Placebo assert", "assert", pl["dc_placebo"], pl["dd_placebo"]),
    est("Reference assert", "assert", pl["dc_real"], pl["dd_real"]),
]
(A / "fig_sorting_points.json").write_text(json.dumps(pts, indent=1))
print(json.dumps(pts))
