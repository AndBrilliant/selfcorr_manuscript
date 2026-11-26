# Limits of Self-Correction in LLMs: code and data

Code, prompts and model outputs for *"Limits of Self-Correction in LLMs: Prompting Moves the
Threshold, Information Moves Discrimination."*

Every number in the paper can be regenerated from the files here, offline, with the scripts listed
below (the HumanEval same-context figures are tallied from a collection output; see the table).
No API key or network access is needed for the analysis.

## Setup

```
pip install -r requirements.txt
```

Run every command from the repository root; scripts write into `data/`.

Python versions: the test results were recorded on Python 3.9, and two scripts depend on the
interpreter. `mbpp_v2_error_severity_audit.py` reproduces its committed output only on 3.9 (on
3.10+, four candidates that call `int.bit_count()` pass instead of failing). `convention_audit.py`
reproduces its committed output only on 3.10 or later. Run each of these two on its version.

## What's here

- `data/MBPP.jsonl`: the 947 MBPP tasks used (prompt, reference asserts, entry point).
  `data/MBPP.reference-solutions.jsonl` holds the reference solutions.
- `data/mbpp_v2/production/`: the main MBPP run (5 model families × 947 tasks).
  `events.jsonl` has every API response; `candidate_records.jsonl` has one solution per
  (model, task) with its test result; `derived_records.jsonl` has the self-check and steelman verdicts.
- One folder per supplementary experiment under `data/mbpp_v2/`:
  - `repeated_eval/`, `replication/`: repeated evaluation and candidate replication;
  - `control/`: neutral re-examination control; `cross_eval/`: cross-evaluation matrix;
  - `assert_visible_eval/`: information control (one reference assert shown);
    `assert_visible_qwen/`: same-prompt Qwen replicate;
  - `confidence_eval/`: 1–7 confidence ratings; `reasoning_off_eval/`: reasoning off or low;
  - `interleaved_rerun/`: first, two-family interleaved rerun;
  - `crossmodel_audit/` (first 650), `crossmodel_audit_rest/` (remaining 1,336),
    `crossmodel_audit_all/` (combined analysis), `crossmodel_audit_INVALID_empty_code/` (invalid
    first attempt, record only): the knowability audit;
  - `round8_interleaved/`: five-family interleaved self-check vs steelman rerun;
  - `round8_gen_assert/`: generator sees the assert; `round8_cross_fails_shown/`: other
    families judge the errors that fail the shown assert;
  - `round8_placebo/`: placebo-assert control; `round8_attribution/`: attribution pilot.
- `data/mbpp_v2/analysis/`, `moderator_analysis/`, `tier1_analysis/`: analysis outputs.
- `data/humaneval/`: the HumanEval experiment (candidates, evaluator responses, results).
- `scripts/`: the analysis scripts (table below) and the scripts used to collect the data.
- `PREREGISTRATION_round8_pending.md`: predictions for the five-family interleaved rerun, the
  generator-sees-assert run and the attribution pilot, committed before their outcomes were analysed.

## Where each result comes from

Paper references are to sections (§), tables and appendices (App.) of the manuscript.

| Paper | What | Script (`scripts/`) | Output (`data/`) |
|---|---|---|---|
| §4.2, App. D.2 | HumanEval rates, intervals, McNemar, bootstrap | `humaneval/paired_contrasts.py` | `humaneval/paired_contrasts.json`, `humaneval/paired_contrasts_summary.txt` |
| §4.2, App. D.2 | same-context comparison (6/9, 4/9; 148/155) | `humaneval/harness_context.py` (collection; counts tallied from its output) | `humaneval/context/context_same.jsonl` |
| App. B.1, App. B.3 | output-cap window: capped attempts, 1,686 pairs completed under the cap | `mbpp_v2_cap_window.py` | `mbpp_v2/analysis/cap_window.json` |
| §4.1 | reference solutions pass the test harness (947/947) | `mbpp_v2_reference_harness_check.py` | `mbpp_v2/analysis/reference_harness_verification.json` |
| §4.3, App. D.1 | false and true acceptance, Wilson intervals, McNemar, Holm | `mbpp_v2_analysis.py` | `mbpp_v2/analysis/mbpp_v2_analysis.json`, `mbpp_v2/analysis/mbpp_v2_analysis_summary.txt` |
| §4.3, App. D.1 | steelman drop, true-acceptance cost, precision and lift, missing-pair bounds | `mbpp_v2_supplementary_stats.py` | `mbpp_v2/analysis/supplementary_stats.json` |
| App. D.1 | discrimination-gap change, bootstrap intervals | `discrimination_bootstrap.py` | `mbpp_v2/analysis/discrimination_bootstrap.json` |
| §4.3 | signal detection d′ and criterion c by family | `signal_detection.py` | `mbpp_v2/analysis/signal_detection.json` |
| §4.3, Abstract | pooled Δd′ and Δc, equivalence test, Cochran's Q, minimum detectable Δd′ | `sdt_pooled.py` (after `signal_detection.py`) | `mbpp_v2/analysis/sdt_pooled.json` |
| §4.3 | pooled Δd′ and Δc split by audit class (knowable / not knowable) | `sdt_by_knowability.py` | `mbpp_v2/analysis/sdt_by_knowability.json` |
| App. D.3 | joint acceptance; moderation analysis (definitions: `mbpp_v2_moderator_spec.md`) | `mbpp_v2_moderator_analysis.py` | `mbpp_v2/moderator_analysis/` |
| App. D.3 | repeated evaluation; candidate replication | `mbpp_v2_tier1_analysis.py` | `mbpp_v2/tier1_analysis/` |
| App. D.3, App. E.2 | replication: identical candidates (33.9%), label flips (6.2%) | `mbpp_v2_replication_identity.py` | `mbpp_v2/tier1_analysis/replication_identity.json` |
| §4.4, Apps. A.4 and D.3 | correlation of repeated verdicts and effective sample size | `repeated_icc.py` | `mbpp_v2/analysis/repeated_icc.json` |
| §4.4, Apps. A.3 and A.4 | framework quantities: E, slack, I(T;A\|G), repeated verdicts, naive vs. empirical LR | `theory_quantities.py`, `theory_extra.py` | `mbpp_v2/analysis/theory_quantities.json`, `theory_extra.json` |
| App. A.4 | moderators as candidate Z (permutation null) | `theory_calibrated.py` | `mbpp_v2/analysis/theory_calibrated.json` |
| App. A.4 | naive benchmark from round-specific ratios (computed after data collection) | `naive_product_check.py` | `mbpp_v2/analysis/naive_product_check.json` |
| App. E.2 | repeated-verdict quantities without superseded copies (computed after data collection) | `superseded_repeated_sensitivity.py` | `mbpp_v2/analysis/superseded_repeated_sensitivity.json` |
| App. G.1 | neutral re-examination control | `mbpp_v2_control_analysis.py` | `mbpp_v2/control/control_analysis_summary.txt` |
| App. G.1 | first, two-family interleaved rerun (exploratory) | `interleaved_analysis.py` | `mbpp_v2/analysis/interleaved_results.json` |
| App. G.1 | first interleaved rerun: session timing, split blocks, Holm-adjusted p | `interleaved_sessions.py` (after `interleaved_analysis.py`) | `mbpp_v2/analysis/interleaved_sessions.json` |
| §4.5, App. F | cross-evaluation matrix, discordance, crossed logistic model | `mbpp_v2_cross_analysis.py`, `mbpp_v2_crossed_model.py` | `mbpp_v2/cross_eval/` |
| App. G.1 | evaluator and generator roles from the cross-evaluation matrix (computed after data collection) | `mbpp_v2_role_balance.py` | `mbpp_v2/cross_eval/role_balance.json`, `role_balance_summary.txt` |
| App. B.3 | retries under the output cap: sensitivity analysis | `mbpp_v2_cap_sensitivity.py` | `mbpp_v2/analysis/cap_sensitivity.json` |
| App. E.2 | failures that fail every assert (84.4%) | `mbpp_v2_error_severity_audit.py` (Python 3.9) | `mbpp_v2/analysis/error_severity_audit.jsonl` |
| App. E.2 | failure types and false acceptance by type | `mbpp_v2_failure_types.py`, `mbpp_v2_failure_type_fa.py` | `mbpp_v2/analysis/failure_type_breakdown.json`, `crash_exception_breakdown.json`, `failure_type_fa.json` |
| App. E.2 | convention-tolerant audit | `convention_audit.py` (Python 3.10+; then `--validate`), `convention_kappa.py` | `mbpp_v2/analysis/convention_audit*.json`, `convention_kappa.json` |
| App. G.1 | same-prompt Qwen replicate (the intended one-assert prompt was not sent; see the script's docstring) | `mbpp_v2_assert_visible_analysis.py` | `mbpp_v2/assert_visible_qwen/` |
| §4.6, App. G.2 | information control: one reference assert shown to the evaluator (run) | `mbpp_v2_assert_visible_eval.py` | `mbpp_v2/assert_visible_eval/events.jsonl`, `records.jsonl` |
| §4.6, App. G.2 | information control: FA/TA, McNemar with Holm, fails-shown vs hidden-only, verdict changes, d′ with correctness defined by the hidden asserts | `mbpp_v2_assert_visible_eval_analysis.py` | `mbpp_v2/assert_visible_eval/assert_visible_eval_analysis.json` |
| App. G.1 | confidence control: 1–7 ratings (run) | `mbpp_v2_confidence_eval.py` | `mbpp_v2/confidence_eval/` |
| App. G.1 | reasoning control: reasoning off (DeepSeek, Qwen, GPT-6-Luna) or low (GPT-5.4) (run) | `mbpp_v2_reasoning_off_eval.py` | `mbpp_v2/reasoning_off_eval/<model>/` |
| §2.1, App. G.1 | ROC areas from confidence ratings; confident errors; reasoning-control d′ and criterion changes | `mbpp_v2_controls_AB_analysis.py` | `mbpp_v2/analysis/controls_AB_analysis.json` |
| §3.6, Tables 1–2 | predictions P1–P5, the scorecard and the design table (numbers from the rows here) | — | — |
| §4.5, App. F | shared blind spot: evaluator's own code wrong vs right, Mantel–Haenszel OR stratified by difficulty (decision rule fixed before the run) | `mbpp_v2_shared_blindspot.py` | `mbpp_v2/analysis/shared_blindspot.json`, `_summary.txt` |
| §4.5, App. F | shared blind spot by output agreement: own code wrong with same vs different outputs (executes frozen candidates offline) | `mbpp_v2_shared_output_agreement.py` | `mbpp_v2/analysis/shared_output_agreement.json`, `_summary.txt` |
| §4.3, App. E.1 | knowability audit (two auditors outside the study; 650, then the remaining 1,336) (run) | `mbpp_v2_crossmodel_audit.py` | `mbpp_v2/crossmodel_audit/`, `mbpp_v2/crossmodel_audit_rest/` |
| App. E.1 | first 650-candidate audit | `mbpp_v2_crossmodel_audit_analysis.py` | `mbpp_v2/crossmodel_audit/summary.txt`, `analysis.json` |
| §4.3, App. E.1 | audit over all 1,986 (agreement, class shares, false acceptance by class) | `mbpp_v2_crossmodel_audit_analysis.py data/mbpp_v2/crossmodel_audit,data/mbpp_v2/crossmodel_audit_rest data/mbpp_v2/crossmodel_audit_all` | `mbpp_v2/crossmodel_audit_all/summary.txt`, `analysis.json` |
| §4.3, §4.5, App. E.1 | audit follow-up: false acceptance and the shared-output split by audit class | `mbpp_v2_crossmodel_audit_followup.py` (all; pass `data/mbpp_v2/crossmodel_audit/labels.jsonl` for the first 650) | `mbpp_v2/crossmodel_audit_all/followup*`, `mbpp_v2/crossmodel_audit/followup*` |
| §4.3, App. E.1 | knowable errors: author vs other families (paired difference, sign test); arity mismatches and confidence ratings on not-knowable errors | `audit_knowable_paired.py` | `mbpp_v2/analysis/audit_knowable_paired.json` |
| §4.4 | verdict information at an unbiased threshold; reasoning tokens per call | `mbpp_v2_threshold_and_reasoning.py` | `mbpp_v2/analysis/threshold_and_reasoning.json` |
| §4.5, App. G.1 | five-family interleaved rerun (run; preregistered) | `round8_interleaved_sc_st.py` | `mbpp_v2/round8_interleaved/` |
| §4.5, App. G.1 | five-family interleaved rerun: Δc, Δd′ by audit class, per-family direction | `round8_interleaved_analysis.py` | `mbpp_v2/analysis/round8_interleaved_analysis.json` |
| App. G.4 | generator sees the assert (run; preregistered) | `round8_generator_sees_assert.py` | `mbpp_v2/round8_gen_assert/` |
| App. G.4 | generator sees the assert: error rates, acceptance of errors failing the shown assert | `round8_gen_assert_analysis.py` | `mbpp_v2/analysis/round8_gen_assert_analysis.json` |
| App. G.4 | other families judge the errors that fail the shown assert (run; prediction in the script's docstring) | `round8_cross_family_fails_shown.py` | `mbpp_v2/round8_cross_fails_shown/` |
| App. G.4 | cross-family check: other families vs author, one-sided Fisher | `round8_cross_fails_shown_analysis.py` | `mbpp_v2/analysis/round8_cross_fails_shown_analysis.json` |
| App. G.3 | placebo-assert control (run) | `round8_placebo_assert.py` | `mbpp_v2/round8_placebo/` |
| §4.6, App. G.3 | placebo vs reference assert: Δd′, Δc, false acceptance on exposed and hidden-only errors | `round8_placebo_analysis.py` | `mbpp_v2/analysis/round8_placebo_analysis.json` |
| App. G.5 | attribution pilot (run; preregistered) | `round8_attribution_pilot.py` | `mbpp_v2/round8_attribution/` |
| App. G.5 | attribution pilot: FA, TA, d′, c by framing | `round8_attribution_analysis.py` | `mbpp_v2/analysis/round8_attribution_analysis.json` |
| §4.6, Table 2 | steelman with the information present (information control, generator-sees-assert, exposed errors in the interleaved rerun) | `p2_information_present.py` (after `round8_interleaved_analysis.py`) | `mbpp_v2/analysis/p2_information_present.json` |
| §4.6, Fig. 1 | sorting figure: Δc and Δd′ for every manipulation | `fig_sorting_points.py`, then `fig_sorting.py data/mbpp_v2/analysis/fig_sorting_points.json fig_sorting.pdf` | `mbpp_v2/analysis/fig_sorting_points.json` |
| App. B.3 | exclusions and accounting | `mbpp_v2_accounting.py` | prints the table |
| App. B.3 | map of the auth-failed cross-evaluation records to their events | `mbpp_v2_auth_failed_event_map.py` | `mbpp_v2/cross_eval/` |
| (not used in the paper) | ambiguity-language flags in evaluator responses; regex too broad to separate the arms | `mbpp_v2_ambiguity_mentions.py` | `mbpp_v2/analysis/ambiguity_mentions.json` |

Run order where one script reads another's output:

- `signal_detection.py` before `sdt_pooled.py`;
- `mbpp_v2_error_severity_audit.py` before `mbpp_v2_shared_blindspot.py`,
  `mbpp_v2_assert_visible_eval_analysis.py` and `round8_placebo_analysis.py`;
- `convention_audit.py` before `convention_kappa.py`;
- `interleaved_analysis.py` before `interleaved_sessions.py`;
- the combined `mbpp_v2_crossmodel_audit_analysis.py` run before `mbpp_v2_crossmodel_audit_followup.py`;
- `round8_interleaved_analysis.py` before `p2_information_present.py`;
- the analysis scripts above before `fig_sorting_points.py`.

## How the data were collected

These scripts call model APIs; the analysis doesn't need them (collection also needs `openai`).

- **MBPP main run:** `mbpp_v2_pilot.py` (via `run_mbpp_v2_production.sh`). GPT-5.4 through the OpenAI
  API; the other four families through OpenRouter; temperature 0. Each solution is tested by running the
  task's asserts in a separate Python 3.9 process with a 15-second timeout.
- **Control and cross-evaluation:** `mbpp_v2_cross_control.py`. **Repeated evaluation:**
  `mbpp_v2_repeated_eval.py`. **Same-prompt Qwen replicate:** `mbpp_v2_assert_visible.py` (intended as a one-assert control, but it sent the production prompts). **Information control:** `mbpp_v2_assert_visible_eval.py`.
  **Confidence and reasoning controls:** `mbpp_v2_confidence_eval.py`, `mbpp_v2_reasoning_off_eval.py`.
  **First interleaved rerun:** `interleaved_build.py`, `interleaved_run.py`.
  **Knowability audit:** `mbpp_v2_crossmodel_audit.py`.
- **Later runs** (all through OpenRouter, sharing `round8_common.py`: cost cap, credit floor,
  shuffled interleaving): `round8_interleaved_sc_st.py`, `round8_generator_sees_assert.py`,
  `round8_cross_family_fails_shown.py`, `round8_placebo_assert.py`, `round8_attribution_pilot.py`.
- **HumanEval:** `scripts/humaneval/` (`generate_baseline.py`, the evaluator arms in `arms/`, and
  `harness_context.py` for the same-context comparison). The HumanEval test suites are not
  redistributed (task prompts appear in the candidate records).

Notes:

- The MBPP main run started with a maximum output length (8,192 tokens for DeepSeek and Qwen, 16,384 for GLM, with
  some retries allowed more; values inferred from the token counts of truncated responses, since the
  production ledger does not record the request limit; no GPT response reached it). A response that hit it was retried (up to three attempts), and pairs still truncated were re-run
  after the limit was removed; 1,686 of 4,735 pairs were completed under the limit. The script here is the
  final version without the limit. The script used for the candidate-replication run is not included.
- The OpenAI key was invalidated during the neutral control and cross-evaluation runs. The 747 affected control
  tasks were re-run, and 35 GPT-5.4 cross-evaluation records were re-run through OpenRouter. The failed
  calls are in `control/auth_failed_records.jsonl` and `cross_eval/auth_failed_*.jsonl`. The
  information-control ledger (`assert_visible_eval/events.jsonl`) also holds 1,048 GPT-5.4 calls to
  OpenAI's API that failed authentication; they returned no verdicts, and every GPT-5.4 verdict in that
  control came through OpenRouter. In the error text of these failed calls, the masked key echoed by the API is replaced by
  `[REDACTED]`; nothing else in those ledgers was changed.
- Credentials were removed from `production/events.jsonl` before release.
- Some script and output names are historical: "pilot" is the main harness, "tier1" is repeated
  evaluation and replication, "round8" marks the later runs.
- `round8_gen_assert/candidate_records_rebuilt.jsonl` is an offline rebuild of
  `candidate_records.jsonl` (identical test results and code hashes), kept as a check;
  `round8_placebo/placebos_summary.json` and `interleaved_rerun/worklist_meta.json` are written by
  the collection scripts.
