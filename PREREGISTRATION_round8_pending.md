# Preregistration: predictions for the two pending round-8 runs

Written 2026-11-25, before either outcome was analysed.

**Disclosure: order of events.**

1. The author started the runs (the interleaved run first, with the generator-sees-assert
   evaluation queued after it).
2. Partway through the interleaved run, the author sent its partial output (3,281
   records) to the AI assistant that wrote the runners, to check that the run was
   working. The assistant checked truncations, prompt hashes, ordering and duplicates,
   and reported only that the run was working. It computed no acceptance rate, d′ or c,
   and showed or explained no outcome to the author.
3. The author then registered these predictions, having read only the run
   specifications and no output of either run.

No evaluation record of the generator-sees-assert run existed at registration. The figures used below as reference points (the
2.2% information-control rate and the 126/373 strata, the latter from code execution
alone, with no evaluator verdict) come from runs already completed and reported.

All tests are on candidates with a parsed ACCEPT/REJECT, pooled over the five families
(counts summed) unless stated. Intervals are 95% task-clustered bootstrap intervals
(2,000 resamples, seed 20261125). d′ = z(TA) − z(FA) and c = −(z(TA) + z(FA))/2, with a
+0.5 correction only at rates of 0 or 1.

## Run A: interleaved self-check vs steelman (all five families, one batch)

Prediction (P2): steelman raises the threshold without a detectable change in
discrimination on errors the description hides.

- **A1 (threshold).** Pooled Δc (steelman − self-check) is positive, with its 95% interval
  above 0.
- **A2 (discrimination, hidden errors).** On the errors both auditors judged not knowable,
  the pooled Δd′ 95% interval contains 0.
- **A3 (replication).** Steelman's false acceptance is lower than self-check's in at
  least 4 of 5 families (direction only).

Decision rule: P2 is supported if A1 and A2 both hold. It is contradicted if the Δd′
interval on hidden errors lies entirely above 0. A1 failing (no threshold shift once batch
drift is removed) would mean the production steelman effect was partly drift, and the
paper would say so.

Not predicted: the exposed-error Δd′ (production +0.43, −0.02 to 0.90). It is reported
whatever it is.

## Run B: generator sees one reference assert; the same family evaluates its own code with the same assert shown

Correctness T is the full reference test. The incorrect candidates are split into
(i) those that fail the shown assert (126 in the rebuilt candidate set) and (ii) those
that pass it and fail hidden asserts only (373).

- **B1 (self-specific blind spot, errors the evaluator can see).** On stratum (i), the
  evaluator has the failing case in front of it. In the information control (production
  candidates, another moment's reading), pooled self-check false acceptance on errors
  failing the shown assert was 2.2% (n = 1,066). Prediction: self-check false acceptance
  on stratum (i) is **higher** than that, because these are errors the model made while
  seeing the same assert, so its misreading of the case persists into evaluation.
  Test: one-sided Fisher exact test of stratum (i) FA against 24/1,066; supported if
  p < 0.05.
- **B2 (convention-only errors).** On stratum (ii), self-check false acceptance is high,
  at the level of row 5 of the design table (about 83%); predicted pooled FA above 60%.
- **B3 (prompting, again).** Steelman vs self-check on these candidates: Δc > 0. No
  prediction on Δd′ (the strata are small).

Decision rule for B1: supported if the Fisher test gives p < 0.05. If stratum (i) FA is at
or below the information control's rate, the self-specific blind spot does not survive
when the evaluator sees the failing case, and the paper will say that plainly.

## Reporting

All of these outcomes are reported whatever they turn out to be, in the design table
(rows 6 and 7) and the text, with this file cited by its SHA-256.

## Run C (added 2026-11-25, before the run was started): attribution pilot

The production prompts never say who wrote the candidate. Run C adds one sentence after the
first line of the self-check prompt: none, "You wrote this solution earlier.", or "The user
wrote this solution.", on a fixed sample (every agreed-knowable error, 40 agreed-hidden
errors and 40 correct candidates per family; 449 candidates, 1,347 calls), all framings
interleaved in one batch (script `round8_attribution_pilot.py`).

- **C1.** Attribution adds no information about correctness. Pooled d′ under "self" and
  under "user" each differs from "none" by an amount whose 95% interval contains 0.
- **C2.** No prediction on the direction of any threshold shift (deference to the user, or
  favouring one's own code); the author considers the two equally likely. Δc is reported
  whatever it is.
