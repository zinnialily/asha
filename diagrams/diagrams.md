# Diagrams

What each image in this folder shows, and the script that produced it. Every chart here was
generated from a real run of the project's code — nothing is a mockup.

## `distance_only_by_scenario.png`

- Shows: what percent of the time the model's #1 recommendation differs from what a
  plain-distance-only ranking would have picked, one bar per scenario (46.7%–63.3%).
- Why it matters: this is the direct answer to "does scoring on blood level and services actually
  change anything, or could we just rank by distance?" — yes, substantially, more than half the
  time overall.
- Produced by: `modeling/demo/distance_only_sensitivity.py` (240 sampled real locations, fixed
  seed, reproducible).

## `pymcdm_validation.png`

- Shows: a parity plot — our entropy-weight and TOPSIS-score outputs (x-axis) plotted against an
  independent, third-party library's outputs (y-axis) for the same inputs. Every point sits
  exactly on the diagonal.
- Why it matters: this isn't our code checking itself — it's proof the entropy-weighting and
  TOPSIS formulas in `modeling/scoring/scoring.py` match the published, textbook versions of those
  algorithms exactly (max difference: 1.6e-12, pure floating-point noise), not just that our own
  tests pass.
- Produced by: `modeling/demo/generate_testing_diagrams.py`, reusing the comparison from
  `modeling/demo/validate_against_reference_implementation.py`.

## `hand_written_test_cases.png`

- Shows: a checklist of 7 situations with an "obvious right answer" (e.g. "two facilities
  equally close, only one has what's needed → that one should win"), each with a pass/fail
  result. Currently 7/7 pass.
- Why it matters: this catches model logic that's technically correct math but behaves
  nonsensically in a situation anyone could eyeball and judge. The 7 cases come directly from
  `model-design.md (draft)`'s own testing plan, formalized as real, runnable assertions instead
  of prose.
- Produced by: `modeling/demo/hand_written_test_cases.py`.

## `sensitivity_sweep.png`

- Shows: two panels — how often the #1 recommendation changes when the time window (45/60/75
  min) or the severity boost factor (x1.5/x2/x3) is nudged away from the current default, split by
  scenario.
- Why it matters: the 60-minute window and the x2 boost are judgment calls, not measured facts.
  This checks whether the model's output is fragile to those exact numbers (bad — means the
  specific choice matters too much) or roughly stable (good — means the ranking reflects something
  real about the candidates). Result: small, single-digit-percent changes across the range tested.
  `severe_anemia`'s flat 0% is a real measured result, not missing data — explained by hand-written
  test case 7: the severity boost has no effect when only the unconfirmed fallback tier is
  reachable, which this sample happened to hit consistently.
- Produced by: `modeling/demo/generate_testing_diagrams.py`, reusing
  `modeling/demo/sensitivity_sweep.py`'s sampling and sweep logic.

## What these four diagrams do and don't prove

Together: the math is textbook-correct (`pymcdm_validation.png`), the model behaves sensibly on
clear-cut cases (`hand_written_test_cases.png`), the output isn't fragile to the judgment calls
baked into it (`sensitivity_sweep.png`), and the extra criteria beyond distance meaningfully change
the outcome rather than being decorative (`distance_only_by_scenario.png`).

None of this proves the model recommends the objectively *best* real-world facility for a given
patient — that would need real patient-outcome data, which `model-design.md (draft)` establishes
doesn't exist for this project. These four diagrams show the model is built correctly and behaves
predictably, not that it has been medically validated.
