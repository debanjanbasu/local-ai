# Judgment-head confidence, refusal and `/v1/decisions` acceptance

This file has two parts. **Part 1 was written and committed before any new
held-out, test or out-of-domain result was computed**; it fixes the splits,
the calibration and refusal procedure and the acceptance criteria. Part 2
records the measured results and the decision, and is appended later
without editing Part 1.

## Part 1: pre-registered protocol (frozen before evaluation)

### Heads under evaluation

| Id | Artifact (SHA-256) | Declared scope |
| --- | --- | --- |
| `devtools` | `b3227ce477a2b010fc1c14a335653562f16ac27a90e04fd13d65893801ae7539` | two-option Yes/No code-diff predicates: CodeReviewer `needs_comment`, CommitPackFT `message_match` |
| `hard-s0` | `24971f5fadb02fb0770f9beb372e5f0ef1bb650e4215064cafb5b901203ac39e` | Kev hard-v1 synthetic predicate/choice/score questions |
| `hard-s1`, `hard-s2` | seeds 1 and 2 (hashes in `local-engine/README.md`) | as `hard-s0`; reported, not candidates |

`devtools` and `hard-s0` are the candidates. No head is retrained.

### Splits (fixed before evaluation)

Every row is a native decision rendering (`DecisionRequest::render`) whose
features come from the server's capture contract: 60-row blocks of
output-normalized hidden states after each option line and the `Decision:`
cue, rounded through FP16.

`devtools` head:

- **CAL** (fit calibration and the refusal threshold, nothing else): the
  existing devtools-v1 calibration split (306 rows, groups carved from Kev
  train, used only for the exported temperature) plus new CodeReviewer rows
  from the projects of Kev's *development* partition (which only selected the
  checkpoint epoch), excluding every row Kev used.
- **TEST** (scored once, after the calibration artifact is frozen): new
  CodeReviewer rows from the projects of Kev's *test* partition, excluding
  every row Kev used, plus the existing 234-row devtools-v1 test split.
  Neither was used for training, selection or calibration.
- **OOD**: rows outside the declared scope rendered through the same path:
  Kev devtools-v1 development/test `prompt_injection`, `aegis_unsafe` and
  `flakeflagger_flaky` predicates, and Kev hard-v1 predicate rows.

`hard-s0` head: Kev hard-v1's locked test is not available locally (it
needs a remote regeneration). Its calibration split (template 3, 1,880 rows,
used only to fit the exported temperature) is split by `group_id` hash into
**CAL-h** and **TEST-h** halves; calibration is refitted from raw logits
(logit × exported temperature), so the exported temperature carries no
TEST-h information into the refit. OOD: all devtools rows above.

### Confidence and refusal procedure

1. Raw logits are recomputed from features with the exported head (raw =
   logit × exported temperature).
2. Calibration maps are fitted on CAL only. Candidates: temperature scaling,
   and for two-option predicates Platt scaling of the Yes-minus-No logit
   difference (slope and bias). The method with the lowest 5-fold grouped
   cross-validated NLL on CAL is selected, then refitted on all of CAL.
3. `confidence` is the calibrated probability of the selected answer
   (top-label). For predicates OpenAI returns only `probability`
   (calibrated P(Yes)); refusal still uses the top-label confidence.
4. Refusal threshold τ: target selective risk **10%**. τ is the smallest
   candidate threshold (the distinct CAL confidences) whose answered CAL rows
   number at least 50 and whose Wilson 95% upper bound on selective risk is at
   most 0.10. A question whose confidence is below τ is answered `refusal`.
5. Scope: a declarative check (question kind, exact question text or
   template, option set, state framing, token limit) refuses anything outside
   the head's declared scope. A feature gate (Mahalanobis distance of the
   head's 256-dim query projection of the `Decision:` row, diagonal
   covariance from CAL) refuses inputs above the CAL 99th percentile.

### Metrics

Accuracy (no abstention) with Wilson 95% intervals; NLL; Brier (binary
`(p_yes - y)^2` for predicates, multi-class sum otherwise); top-label ECE with
15 equal-width bins; selective risk/coverage curve and AURC; selective risk
and coverage at τ. Reported per split, per question family, raw and
calibrated, against the train-majority baseline.

### Acceptance criteria for serving `POST /v1/decisions`

A candidate head may back `/v1/decisions` only if **all** hold on its TEST
split (OOD where stated):

- **A1 accuracy:** overall accuracy's Wilson 95% lower bound exceeds the
  train-majority baseline by at least 10 points, and every declared
  question family's Wilson lower bound exceeds its own majority baseline.
- **A2 calibration:** calibrated top-label ECE ≤ 0.05, calibrated NLL lower
  than the uniform predictor's, and calibrated Brier ≤ 0.20 (binary) or
  ≤ 0.40 (multi-class).
- **A3 selective risk:** at τ, selective risk's Wilson 95% upper bound ≤
  0.15 and coverage ≥ 25%.
- **A4 no hidden family:** at τ, every declared question family either has
  selective risk ≤ 0.15 or coverage ≤ 5%.
- **A5 scope:** the declarative check refuses 100% of OOD rows; the feature
  gate alone refuses ≥ 80% of OOD rows while refusing ≤ 5% of TEST rows.
- **A6 integrity:** the calibration artifact validates its head hash and its
  own content hash at load; native and HTTP answers agree with the offline
  scoring of the same features.

If any criterion fails, `/v1/decisions` stays refused and only the tooling
and measurements are shipped. Nothing here bears on MTP acceptance or
coding quality.

## Part 2: results

(Appended after evaluation.)
