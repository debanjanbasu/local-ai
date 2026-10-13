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

Recorded after evaluation; Part 1 above is unchanged. **Decision: no head
passes, so `POST /v1/decisions` stays refused.** Only the tooling and these
measurements ship.

### Data and provenance

Every feature row comes from the serving path (`Engine::capture_decision`
via `judgment-capture`, binary SHA-256 `fef579b1a76bb8391cd2f2256f7999a72dd92817f00a95a4f426c106c8ad91ee`), one GPU-lock hold
per chunk of at most 160 rows.

| Set | Rows | Use | Source and hashes |
| --- | ---: | --- | --- |
| devtools-v1 stored features | 3,533 | `devtools`: CAL (calibration, 306), TEST (test, 234), train baseline; `hard-*`: OOD | Kev devtools-v1 as prepared for the head; manifest `5084534a…eb0db` (the head's own capture) |
| hard-v1 stored features | 8,479 | `hard-*`: CAL-h / TEST-h (calibration split by `group_id` hash), train baseline; `devtools`: OOD (`*/noul` rows of calibration and validation) | manifest `dabe32c4…15d63` |
| `cr-new` | 1,280 | `devtools`: CAL (480, 9 Kev-development projects), TEST (800, 15 Kev-test projects); `hard-*`: OOD | CodeReviewer `cls-valid`/`cls-test` (Zenodo 6900648, CC-BY-4.0, Li et al. 2022), permissive-licence projects only, every hunk Kev used excluded, label-balanced; requests `33765761…c8686`, features `7e5268c6…4f270` |
| `kev-ood` | 450 | `devtools` and `hard-*`: OOD | Kev devtools-v1 development/test `prompt_injection`, `aegis_unsafe`, `flakeflagger_flaky` (150 each, label-balanced); requests `7534dba9…a7a1f`, features `29a40a42…4177a` |

Integrity checks before scoring: the 16 devtools-v1 and 24 hard-v1 rows
re-expressed as native requests recaptured **bit-identical** to the stored
features (`judgment_calibrate.py parity`), with this binary and with the
earlier one; every new manifest row matches its request id and order and has
`positions × 5120` FP16 values. A capture run interrupted earlier (other
binary, `kev-ood` 160 rows, no `cr-new` chunk finished) was discarded and
everything recaptured with one binary; its finished chunk was bit-identical.
No public data was fetched in this run (the Zenodo members, Kev manifests and
licence table already on disk were re-verified by hash when the sets were
built).

### Deviations from Part 1

- A debugging run before any TEST scoring used devtools-v1 *validation*
  (checkpoint selection) in place of TEST, and scored the stored features
  only. It touched no `cr-new` row and not the devtools-v1 test split.
- `hard-s0` scope: Part 1 says "exact question text or template". Kev
  hard-v1's calibration split uses held-out templates, so a scope of train
  questions alone refused about two thirds of in-distribution rows. The
  declarative scope used is the exact question texts and option counts of
  train **plus CAL-h**. This was chosen after a preliminary run whose report
  *did* contain TEST-h metrics (the same numbers as below), so it is not
  blind; it changes only the declarative refusal rate, and the hard heads
  fail A1–A5 regardless (their feature gate refuses no OOD row).
- `devtools` OOD hard-v1 predicate rows: Part 1 names no partition; all
  non-train `*/noul` rows (calibration and validation, 695) were used.
- The `hard-*` OOD set is every devtools-side row (devtools-v1 calibration
  and test, `cr-new`, `kev-ood`; 2,270 rows).

### `devtools` head (`b3227ce4…`), the only candidate near acceptance

Calibration selected on CAL by 5-fold grouped CV NLL: temperature 0.5208 vs
Platt 0.5236, so temperature scaling, T = 1.7540 on raw logits. Refusal
threshold τ = 0.7644 (CAL: 30.8% coverage, 6.2% risk, Wilson upper 9.97%).
Feature gate: CAL 99th percentile, 3.644.

| Split | Rows | Majority | Accuracy (Wilson 95%) | Raw NLL | Cal NLL | Raw ECE | Cal ECE | Cal Brier | AURC | Coverage@τ | Risk@τ (Wilson 95%) | Gate refused | Scope refused |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| CAL | 786 | 0.496 | 0.709 (0.676–0.739) | 0.5472 | 0.5204 | 0.0774 | 0.0297 | 0.1809 | 0.1465 | 0.308 | 0.062 (0.038–0.100) | 0.009 | 0.000 |
| TEST | 1,034 | 0.501 | 0.699 (0.671–0.726) | 0.5923 | 0.5608 | 0.0764 | 0.0415 | 0.1951 | 0.1807 | 0.269 | 0.126 (0.092–0.170) | 0.004 | 0.000 |
| OOD | 1,145 | 0.479 | 0.688 (0.661–0.714) | 0.9337 | 0.6978 | 0.1790 | 0.1041 | 0.2259 | 0.2455 | 0.574 | 0.272 (0.240–0.308) | 0.190 | 1.000 |

Uniform NLL is 0.6931 throughout; Brier is binary `(p_yes − y)²`.

| Split | Family | Rows | Majority | Accuracy | Cal ECE | Coverage@τ | Risk@τ |
|---|---|---:|---:|---:|---:|---:|---:|
| CAL | codereviewer_needs_comment | 636 | 0.500 | 0.654 | 0.0309 | 0.159 | 0.109 |
| CAL | commitpackft_message | 150 | 0.480 | 0.940 | 0.0421 | 0.940 | 0.028 |
| TEST | codereviewer_needs_comment | 918 | 0.500 | 0.666 (0.634–0.695) | 0.0467 | 0.185 | **0.200** (0.147–0.266) |
| TEST | commitpackft_message | 116 | 0.509 | 0.966 (0.915–0.987) | 0.0218 | 0.931 | 0.009 |
| OOD | aegis_unsafe | 150 | 0.500 | 0.780 | 0.0498 | 0.633 | 0.158 |
| OOD | flakeflagger_flaky | 150 | 0.500 | 0.653 | 0.1500 | 0.267 | 0.325 |
| OOD | prompt_injection | 150 | 0.500 | 0.647 | 0.1478 | 0.527 | 0.304 |
| OOD | judge/noul | 314 | 0.503 | 0.608 | 0.2368 | 0.682 | 0.407 |
| OOD | tradeoff/noul | 314 | 0.420 | 0.799 | 0.0416 | 0.615 | 0.124 |
| OOD | probability/noul | 26 | 0.538 | 0.500 | 0.3723 | 0.692 | 0.444 |
| OOD | temporal_numeric/noul | 41 | 0.463 | 0.512 | 0.3221 | 0.439 | 0.444 |

Risk at fixed coverage (calibrated-confidence ranking):

| Split | 100% | 90% | 80% | 70% | 60% | 50% | 40% | 30% | 20% | 10% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CAL | 0.291 | 0.272 | 0.253 | 0.220 | 0.210 | 0.173 | 0.118 | 0.055 | 0.013 | 0.000 |
| TEST | 0.301 | 0.288 | 0.273 | 0.243 | 0.229 | 0.211 | 0.193 | 0.161 | 0.053 | 0.000 |
| OOD | 0.312 | 0.297 | 0.287 | 0.269 | 0.269 | 0.269 | 0.251 | 0.224 | 0.218 | 0.184 |

| Criterion | Result | Evidence (TEST unless stated) |
| --- | --- | --- |
| A1 accuracy | pass | lower bound 0.671 ≥ 0.501 + 0.10; CodeReviewer 0.634 > 0.500, CommitPackFT 0.915 > 0.509 |
| A2 calibration | pass | ECE 0.0415 ≤ 0.05; NLL 0.5608 < 0.6931; Brier 0.1951 ≤ 0.20 |
| A3 selective risk | **fail** | risk 0.126 but Wilson upper 0.170 > 0.15 (coverage 0.269 ≥ 0.25) |
| A4 no hidden family | **fail** | CodeReviewer risk 0.200 at coverage 0.185 (> 0.15 and > 5%); the pooled threshold is carried by CommitPackFT |
| A5 scope | **fail** | declarative check refuses 100% of OOD and 0% of TEST, but the feature gate refuses only 19.0% of OOD (needs ≥ 80%); TEST 0.4% |
| A6 integrity | not reached | the calibration artifact's head and content hashes are checked by the tool's tests; no server path loads it |

The CAL-chosen threshold did not transfer to new CodeReviewer projects
(CodeReviewer risk 0.109 on CAL, 0.200 on TEST): on code-review judgment the
head's confidence ranks errors only weakly, and the head's query projection
does not separate other Yes/No predicates (prompt injection, unsafe content,
flaky tests, synthetic hard-v1) from in-scope ones. Only the declarative
check, not the model, refuses out-of-scope questions.

### `hard-*` heads (Kev hard-v1, development-only)

Temperature scaling only (multi-class). TEST-h is half of hard-v1's
calibration split; OOD is every devtools-side row.

| Head | T | τ | TEST-h acc (Wilson 95%) | Majority | Cal NLL (uniform 1.276) | Cal ECE | Cal Brier | Coverage@τ / risk | Gate refused OOD / TEST | Result |
|---|---:|---:|---|---:|---:|---:|---:|---|---|---|
| `hard-s0` (`24971f5f…`) | 1.586 | none | 0.650 (0.619–0.680) | 0.321 | 0.866 | 0.0457 | 0.471 | 0 (all refused) | 0.000 / 0.015 | A1–A5 fail |
| `hard-s1` (`11bf6f74…`) | 1.538 | none | 0.621 (0.589–0.652) | 0.321 | 0.885 | 0.0506 | 0.485 | 0 (all refused) | 0.000 / 0.018 | A1–A5 fail |
| `hard-s2` (`657cde86…`) | 1.322 | 0.849 | 0.634 (0.603–0.665) | 0.321 | 0.865 | 0.0388 | 0.470 | 0.181 / 0.072 (upper 0.121) | 0.000 / 0.016 | A1–A5 fail (A3: coverage 18% < 25%) |

A1 fails on `probability/noul` (8 TEST-h rows, 0.375) and
`temporal_numeric/noul` (12 rows, 0.333) for every seed; multi-class Brier
≈ 0.47 exceeds 0.40; for `hard-s0`/`hard-s1` no CAL-h threshold reaches a
Wilson upper risk of 10% with 50 answered rows, so they would refuse
everything. The declarative scope refuses 100% of OOD and 2.4% of TEST-h
(question texts seen only in TEST-h); the feature gate refuses no OOD row.
On devtools-side rows these heads score 0.583–0.606.

### Refusal and out-of-scope semantics (as evaluated)

A served answer would have been, per question: out of declared scope (the
artifact's declarative check: kind, option set, question text or pattern,
state framing, token limit) → `refusal`; feature-gate distance above the
CAL 99th percentile → `refusal`; calibrated top-label confidence below τ →
`refusal`; otherwise the answer with `confidence` (and, for predicates,
calibrated `probability` of Yes). The evaluation shows the model-side parts
of this (τ and the gate) do not hold on new data, so nothing is served.

### Artifacts (not committed; in the evaluating worktree's `.amp/in/`)

| Item | SHA-256 |
| --- | --- |
| `tools/judgment_calibrate.py` used | `8ebc9a45731fa439e71844be8212c84f4b55a0bf53f427a30127e0cd8f6ac200` |
| `devtools` scores / report / calibration artifact (content hash) | `0d1d18c6a299093f0972f20a15cf2b911b2adf831d99a5cdb40bacefd6bd9938` / `02fa2ff748c692bbe7f4955be887fa1fc2dbf5659db0078f30fec3c8f3586a07` / `7896ed60d739f24f56fd9c40b15ed8ad818d0ff189b22ad7fdeb825bda639113` |
| `hard-s0` scores / report / artifact content | `8fe24cb188c2e00c5ea7f675985ab5b1dd5c75f7bb0d8bbf1f059d546a3cebaa` / `060fec6d0b0c74b72d707bbc38b51f0c61e223197e3d108136df78de1ed7256f` / `ff3b2d891261311df9fa4fa41a7bdd366cda2777693820953e99b16c2c45e219` |
| `hard-s1` scores / report / artifact content | `83e1b21d9c7b16d069458bc4e046b9a73680c6c48593cb78eed8724ecbfe3905` / `18f6cb1b92de218efe45b0e68757b4cc9a61392b8c8d4c10e5b3d55e3018edff` / `65002dc19a6f85f3e2a2412f8fccb4f8afe9a12247392327b2b6b7a8bf2173a3` |
| `hard-s2` scores / report / artifact content | `0cdc88b94295bc84714e6a37790d219ec6eaa8000caab56a68278ca013f21ed2` / `9edaa952852d08af371329b247978adb236ee6f1f93389462c0cf057f48bcf47` / `751825f2a6417a5fa9529eb317a7c68b0a7fdf0725354c2439f6d9486b714811` |

Every calibration artifact records `acceptance.passed: false`.

### Limitations

One checkpoint and its existing heads; no head was retrained or re-selected.
CodeReviewer labels ("needs a reviewer comment") are noisy, and TEST is
dominated by them by design. hard-v1's locked test is unavailable, so TEST-h
is half of a split whose temperature the exported head was fitted on (the
refit starts from raw logits). TEST-h families with fewer than 15 rows make
A1 fragile for the hard heads. None of this bears on MTP or coding quality.

## Part 3: retrained head on fresh splits (protocol v2)

Parts 1 and 2 are unchanged. Part 3 has the same shape: **3a was written and
committed before any CAL, TEST or OOD row below was scored** (features may be
captured before the commit; no head scores them before it). 3b records the
results afterwards without editing 3a. Every split Part 2 scored (devtools-v1
calibration/test, `cr-new`, `kev-ood`, hard-v1 `*/noul`) has been looked at,
so none of it is CAL, TEST or OOD here: it may only train, select or develop.

### 3a. Pre-registration (frozen before scoring)

#### Integrity of the Part 2 inputs, recaptured

- `kev-ood` requests rebuilt with `judgment_calibrate.py build-ood`:
  `7534dba97bb3bfbf0138eeba8462be7da9df3d4677e098ee0a99322b9e4a7a1f`, as
  documented. Recaptured through `judgment-capture` (binary
  `f87a6f8b54c7f14d7c905ae33b058f3e2aa8d7f91e59d1654f54aca4d88e6104`, built from
  this branch), 3 GPU-lock holds of 150 rows: features manifest
  `29a40a425945ef4298a9e685039995911fdb17a1af0e38c8dc69622e80b4177a`, **identical
  to the documented manifest**, so the pinned renderer and capture path are
  unchanged.
- Zenodo 6900648 fetched by a private, internet-enabled CPU Kaggle kernel
  (`tools/kaggle_judgment_job.py --fetch-public-data`, staged `convert.py`
  `27fa406b…0aad24`; nothing fetched from this machine): archive
  `86d054de…41fff5`, `cls-valid.jsonl` `ca2725be…ad33`, `cls-test.jsonl`
  `de62a2ce…d5d9c`, all the pinned values. `cr-new` requests rebuilt with
  `build-codereviewer`: `337657614b9ae9ea3e78d38bde6c4eb6901ac6626d1fe06b62922e5a765c8686`,
  as documented. `cr-new` features were **not** recaptured: capture runs at
  ~110 prompt tokens/s (378,585 tokens, about an hour of shared GPU time), and
  `kev-ood` already shows the capture path reproduces bit for bit.
- The stored devtools-v1 and hard-v1 feature manifests still hash to
  `5084534a…eb0db` and `dabe32c4…15d63`.

#### Diagnosis of the Part 2 failures (from looked-at data only)

- **A3/A4 are CodeReviewer.** Part 2's pooled τ (0.764) was fitted on a CAL of
  636 CodeReviewer and 150 CommitPackFT rows. CommitPackFT is answered at 93%
  coverage with ≈1% risk on any split; CodeReviewer was answered at 16–19%
  coverage with 10.9% risk on CAL and 20.0% on TEST. One threshold cannot be
  right for a 96%-accurate and a 66%-accurate family at once, and the pooled
  CAL bound let the noisy family through.
- **No single project drives it.** Re-scoring the stored devtools-v1
  validation/test rows (22 Kev development/test projects; temperature refitted
  on the stored calibration split) with the Part 2 head: CodeReviewer accuracy
  per project ranges 0.30–0.90 and the rows answered at τ are spread thinly
  (0–5 per project) with errors in most projects; CodeReviewer risk at τ was
  0.132 on the stored calibration split (38 answered) and 0.333 on the stored
  test split (30 answered). The head's CodeReviewer confidence ranks errors
  weakly and inconsistently across projects; this is consistent with
  CodeReviewer's noisy human label ("a reviewer commented"), not with one bad
  project.
- **Why the gate missed OOD.** The Part 2 gate is a diagonal Mahalanobis
  distance on the head's 256-dim *query projection*, which training shapes to
  rank the options of in-scope rows, not to describe the input; OOD queries
  land inside the in-scope spread. Fitted on the stored calibration split alone
  it refuses 69% of hard-v1 `*/noul`, 33% of `kev-ood` and 14% of CommitPackFT
  change-type questions (same diffs, another question); fitted on Part 2's
  wider CAL it refused 19%. On the raw `Decision:` hidden state, a PCA
  (64 components, fitted on the devtools-v1 train rows) Mahalanobis distance
  plus residual energy, thresholded at the stored calibration split's 99th
  percentile, refuses **100%** of all three of those looked-at OOD sets while
  refusing 1.7% of the stored test split. This was developed only on
  looked-at data (`dev-ood`: 150 Kev *train* change-type rows, requests
  `684d154f…db09`, captured for this purpose and used nowhere else).

#### Candidate

One new head, `devtools-v2`, trained by `tools/judgment_train.py` (unchanged)
with the Part 2 head's frozen config (`epochs=40 lr=3e-5 weight_decay=0.01
batch_size=64 seed=0 width=5120 head_dim=256`), weights on `train`, epoch by
`validation` NLL, on CPU PyTorch. No second config or seed is tried; its
sha256 is recorded in 3b. The trainer's own `calibration`/`test` splits are
bookkeeping only (the exported temperature is ignored: calibration is refitted
from raw logits on CAL). The Part 2 `devtools` head is scored on the same
splits for reference only: it trained on Kev train projects, which include the
v2 CAL/TEST projects, so it is not a candidate.

Declared scope, unchanged from Part 2: two-option Yes/No code-diff
predicates, CodeReviewer `needs_comment` (exact question) and CommitPackFT
`message_match` (question pattern), states starting `Diff:` or
`Context before hunk:`, at most 2048 prompt tokens.

#### Data and roles (`tools/judgment_reeval.py`, seed `local-ai-decisions-v2`)

All new requests are native predicate/choice renderings captured through
`judgment-capture`. Sources: Zenodo 6900648 (CC-BY-4.0, Li et al. 2022),
projects with permissive licences in Kev's licence table only; CommitPackFT
(`bigcode/commitpackft@fc56fe33`, MIT, permissive per-row repository
licences, fetched by the same Kaggle kernel; summary `36a9fe70…d9e23`), 20
languages: Kev's 12 plus kotlin, swift, scala, haskell, lua, perl, dart and
elixir, none of which Kev sampled.

CodeReviewer project roles: Kev *development* projects (10) → `validation`;
Kev *test* projects (16, Part 2's TEST) → `train`; Kev *train* projects plus
the 6 permissive projects Kev never used are ordered by seeded hash: first 12
→ **CAL**, next 20 → **TEST**, rest (98) → `train`. Every row of a CAL/TEST
project is removed from training (including Kev's own rows, which trained the
Part 2 head). New rows exclude every hunk Kev or `cr-new` used.

CommitPackFT roles: repository components (union of each commit's
repositories) that share no repository with any Kev devtools-v1 row, assigned
by seeded component hash (70% train, 10% validation, 10% CAL, 10% TEST); one
commit per component, languages round-robin; Kev's message-match construction
(the negative shows another component's subject in the same language and
role); labels balanced.

| Set | Requests (SHA-256) | Rows | Role |
| --- | --- | ---: | --- |
| `cr2:cal` | `cr-v2` `f0c27860d3768648871296ffa69be3ff36a59b7847eff0fb179d2de3b297612a` | 400 | **CAL**, 11 projects with rows: LLK-scratch-blocks, ProteoWizard-pwiz, aws-amazon-ecs-agent, aws-aws-sdk-go, bitpay-bitcore, graphite-project-graphite-web, istio-tools, lucas-clemente-quic-go, openfoodfacts-openfoodfacts-androidapp, vavr-io-vavr, vimeo-psalm |
| `cr2:test` | same file | 500 | **TEST**, 17 projects: AthenZ-athenz, Automattic-mongoose, Azure-autorest, DataDog-dd-trace-dotnet, NLog-NLog, OPAE-opae-sdk, aspnet-KestrelHttpServer, aws-copilot-cli, influxdata-flux, kubernetes-sigs-cluster-api-provider-aws, kythe-kythe, nats-io-nats-server, open-osrs-runelite, open-telemetry-opentelemetry-go, opencontainers-runc, realm-realm-js, thoughtbot-upcase |
| `cr2:train` | same file | 400 | train (83 train-role projects) |
| `cm2:cal` / `cm2:test` | `cm-v2` `e6748cf452b4c925d9b70d64d8bcb01f9d4f28a5edd6a59953a1f01f2a814c73` | 250 / 350 | **CAL** / **TEST** (new repository components, 20 languages) |
| `cm2:train` / `cm2:validation` | same file | 400 / 100 | train / validation |
| `ood2:ood` | `ood-v2` `e1ef6ff82bf5d6f81e7cab5f682095b6fc7e727e638a8bc0d07560c3d1e53eaa` | 500 | **OOD**: 100 each of `prompt_injection`, `aegis_unsafe`, `flakeflagger_flaky` (Kev development/test rows `kev-ood` did not select), when2call actions and CommitPackFT change types (Kev development/test choices); none scored before |
| devtools-v1 stored features | manifest `5084534a…eb0db` | 3,533 | train (train and test splits), `validation` (development-project CodeReviewer), trainer bookkeeping; rows of CAL/TEST projects dropped |

Training directory: `judgment_reeval.py assemble` with the role file
`d262b6ba205f6f08af9d4151117ebc8aa2094dcaf0158c3bd790d47d7872fca1`:
devtools-v1 train/test → train, calibration → trainer calibration,
development CodeReviewer → validation, development CommitPackFT → trainer
test, every `codereviewer/<CAL or TEST project>` group dropped; `cr2:train`
and `cm2:train` → train; `cm2:validation` → validation. The train-majority
baseline uses the training directory's `train` rows.

#### Changes to the confidence and refusal procedure (criteria unchanged)

1. Calibration **per declared family** (the family is known declaratively from
   the question): temperature or Platt by 5-fold grouped CV NLL on that
   family's CAL rows, refitted on all of them.
2. Refusal threshold **per declared family**: Part 1's rule (smallest CAL
   confidence with ≥ 50 answered CAL rows and Wilson 95% upper risk ≤ 0.10) on
   that family's CAL rows; if none qualifies the family is refused outright.
3. Feature gate: PCA of the **training** rows' raw `Decision:` hidden state
   (64 components); distance = (Mahalanobis in the components + residual
   energy / mean residual variance) / 5120; threshold = CAL 99th percentile.
4. Declarative scope check unchanged (`DEVTOOLS_SCOPE`).

Metrics, the train-majority baseline and criteria **A1–A6 are exactly Part
1's**: A3 on the pooled TEST answers under the per-family thresholds, A4 per
declared family, A5 on `ood2`. One `judgment_reeval.py evaluate` run fits on
CAL and scores TEST and OOD once; nothing is refitted after. If any
criterion fails, `/v1/decisions` stays refused. Nothing here bears on MTP or
coding quality.

#### Frozen before scoring (appended to 3a, nothing above edited)

- `devtools-v2` head (`judgment_train.py`, frozen config, CPU torch):
  `34d0708873125c09aa59f444d7c40e64ac2af8b81ca989c7970ace9e67c83b81`; epoch 8
  of 40 by validation NLL (0.4173 on 250 validation rows). Training directory
  manifest `a1b7fa3a…85b2cc` (4,001 rows: train 3,338 = 1,472 CodeReviewer +
  1,866 CommitPackFT; validation 250; trainer bookkeeping 413), assembled with
  the role file above from the stored devtools-v1 features and the finished
  `cr-v2`/`cm-v2` capture chunks holding their train/validation rows (the
  same rows, in the same order, as the merged directories).
- Gate: `9a5a4eb811822e4d1c64dde6006b9d92f7f0d5ebeb757d8b99e113e7d21f4fb1`
  (64 components, fitted on the 3,338 training rows only).

### 3b. Results (recorded after the single scoring run; 3a unchanged)

**`devtools-v2` passes A1–A5 on the fresh TEST and OOD splits.** The pass
has a narrow meaning: CAL found no CodeReviewer threshold meeting the rule,
so **every CodeReviewer question is refused**, and the answered set is
CommitPackFT message-match. A6 is in 3c.

#### Data and provenance

All capture through `judgment-capture` binary `f87a6f8b…e6104` (every chunk
records it), `tools/judgment_calibrate.py capture` with one
`timeout 1800 lockf -k [-t 900] /tmp/local-ai-gpu.lock` hold per chunk
(13 holds, 2,900 rows, 6,704 s of GPU time). Every merged manifest matches its
request ids and order and has `positions × 5120` FP16 values per row. A
reboot and lock time-outs interrupted the chains; interrupted chunks were
re-captured whole (resumable per chunk), never merged partially.

| Set | Rows | Features manifest (SHA-256) |
| --- | ---: | --- |
| `cr-v2` | 1,300 | `da9c5b65b22aab301546e645f4e38cb0fc027e77b9f53561556e76a36d59d31b` |
| `cm-v2` | 1,100 | `86511af5078d45c3408856b28a03b04eaa7e10e0972aa335ef0e0fd8cb5cc884` |
| `ood-v2` | 500 | `e1dd32d660c6dabfba3defb76ba152c6f4157e40d4de9b89dd361fa4c4c07dfc` |

Re-assembling the training directory from the merged `cr-v2`/`cm-v2`
directories reproduces the frozen training manifest `a1b7fa3a…85b2cc`
exactly. Scores `956947eb…7aff98` (head `34d07088…`, gate `9a5a4eb8…`);
report `8df898be…3ead8`; calibration artifact content hash
`7cf043ef40d773f1ed38ed1a0d0332a04e114494ad9d438be9f20afc908c13a2`
(`acceptance.passed: true`); `judgment_reeval.py` `676660a4…1d81557`,
`judgment_calibrate.py` `9920e31d…31ffd`.

#### Fitted on CAL

| Family | Method (5-fold grouped CV NLL) | Parameters | Threshold τ |
| --- | --- | --- | --- |
| CodeReviewer `needs_comment` | Platt (0.6140 vs temperature 0.6227) | slope 1.1001, bias 0.5672 on raw logit(Yes) − logit(No) | **none qualifies → refused** |
| CommitPackFT `message_match` | temperature (0.1430 vs Platt 0.1478) | T = 1.4790 on raw logits | 0.5084 |

Gate threshold (CAL 99th percentile): 2.1578.

#### Metrics

Uniform NLL 0.6931 (binary splits); Brier is binary `(p_yes − y)²` for
CAL/TEST (OOD includes choices, so its Brier is multi-class).

| Split | Rows | Majority | Accuracy (Wilson 95%) | Raw NLL | Cal NLL | Raw ECE | Cal ECE | Cal Brier | AURC | Coverage@τ | Risk@τ (Wilson 95%) | Gate refused | Scope refused |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| CAL | 650 | 0.500 | 0.775 (0.742–0.806) | 0.4458 | 0.4217 | 0.0649 | 0.0205 | 0.1416 | 0.0809 | 0.385 | 0.036 (0.019–0.067) | 0.009 | 0.000 |
| TEST | 850 | 0.500 | 0.787 (0.758–0.813) | 0.3811 | 0.3798 | 0.0341 | 0.0374 | 0.1288 | 0.0651 | 0.409 | 0.009 (0.003–0.025) | 0.015 | 0.000 |
| OOD | 500 | 0.408 | 0.604 (0.560–0.646) | 0.9179 | 0.9018 | 0.0730 | 0.0483 | 0.4961 | 0.2092 | 0.674 | 0.264 (0.220–0.314) | **1.000** | **1.000** |

| Split | Family | Rows | Majority | Accuracy (Wilson 95%) | Cal ECE | Coverage@τ | Risk@τ | Gate refused |
|---|---|---:|---:|---|---:|---:|---:|---:|
| CAL | codereviewer_needs_comment | 400 | 0.500 | 0.657 (0.610–0.702) | 0.0295 | 0.000 | – | 0.013 |
| CAL | commitpackft_message | 250 | 0.500 | 0.964 (0.933–0.981) | 0.0212 | 1.000 | 0.036 | 0.004 |
| TEST | codereviewer_needs_comment | 500 | 0.500 | 0.646 (0.603–0.687) | 0.0548 | 0.000 | – | 0.008 |
| TEST | commitpackft_message | 350 | 0.500 | 0.989 (0.971–0.996) | 0.0319 | 0.994 | 0.009 | 0.026 |
| OOD | aegis_unsafe | 100 | 0.480 | 0.810 | 0.0930 | 0.980 | 0.184 | 1.000 |
| OOD | commitpackft_type | 100 | 0.130 | 0.330 | 0.0794 | 0.050 | 0.600 | 1.000 |
| OOD | flakeflagger_flaky | 100 | 0.530 | 0.590 | 0.1157 | 0.990 | 0.414 | 1.000 |
| OOD | prompt_injection | 100 | 0.530 | 0.790 | 0.0968 | 1.000 | 0.210 | 1.000 |
| OOD | when2call_action | 100 | 0.370 | 0.500 | 0.1103 | 0.350 | 0.171 | 1.000 |

OOD coverage/risk are what the head *would* answer if the declarative check
and the gate did not refuse every OOD row (the pooled CAL temperature and the
strictest finite τ; diagnostic only).

Risk at fixed coverage (calibrated-confidence ranking):

| Split | 10% | 20% | 30% | 40% | 50% | 60% | 70% | 80% | 90% | 100% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CAL | 0.000 | 0.000 | 0.015 | 0.027 | 0.058 | 0.087 | 0.134 | 0.169 | 0.195 | 0.225 |
| TEST | 0.000 | 0.006 | 0.004 | 0.003 | 0.024 | 0.063 | 0.109 | 0.149 | 0.190 | 0.213 |
| OOD | 0.040 | 0.100 | 0.147 | 0.195 | 0.216 | 0.243 | 0.269 | 0.320 | 0.353 | 0.396 |

CodeReviewer accuracy per TEST project (30 rows each, 28 for four): 0.47–0.83
(AthenZ 0.79, mongoose 0.82, autorest 0.60, dd-trace-dotnet 0.47, NLog 0.47,
opae-sdk 0.67, KestrelHttpServer 0.70, copilot-cli 0.79, flux 0.61,
cluster-api-provider-aws 0.63, kythe 0.47, nats-server 0.57, runelite 0.47,
opentelemetry-go 0.83, runc 0.64, realm-js 0.80, upcase 0.70); none answered.

#### Criteria

| Criterion | Result | Evidence (TEST unless stated) |
| --- | --- | --- |
| A1 accuracy | pass | lower bound 0.758 ≥ 0.500 + 0.10; CodeReviewer 0.603 > 0.500; CommitPackFT 0.971 > 0.500 |
| A2 calibration | pass | ECE 0.0374 ≤ 0.05; NLL 0.3798 < 0.6931; Brier 0.1288 ≤ 0.20 (CodeReviewer alone: ECE 0.0548) |
| A3 selective risk | pass | risk 0.009, Wilson upper 0.025 ≤ 0.15; coverage 0.409 ≥ 0.25 |
| A4 no hidden family | pass | CodeReviewer coverage 0.000 (≤ 5%); CommitPackFT risk 0.009 at coverage 0.994 |
| A5 scope | pass | declarative check refuses 100% of OOD, 0% of TEST; gate alone refuses 100% of OOD and 1.5% of TEST |
| A6 integrity | see 3c | |

#### What changed, honestly

The Part 2 head, scored on the same fresh splits with the same v2 procedure
(reference only: it trained on Kev rows of the v2 CAL/TEST CodeReviewer
projects), **also passes A1–A5**: TEST accuracy 0.782 (0.753–0.809), raw NLL
0.4089, calibrated ECE 0.0401, Brier 0.1322, risk 0.020 (upper 0.041) at
coverage 0.411, CodeReviewer refused, gate refusing 100% of OOD and 1.5% of
TEST. So the pass comes from the **procedure**, not from the retraining:
per-family thresholds (which refuse the family the head cannot rank) and a
gate on the raw hidden state (which separates out-of-scope questions; the
query-projection gate could not). Retraining on more projects and eight new
languages helped modestly and in the expected direction (TEST raw NLL 0.381
vs 0.409, raw ECE 0.034 vs 0.054, CommitPackFT accuracy 0.989, answered risk
0.009 vs 0.020) and did **not** make CodeReviewer usable: 0.646 on new
projects, essentially the Part 2 head's level, with no CAL threshold able to
reach 10% selective risk on 50 rows.

#### Limitations

- What could be served is a CommitPackFT message-match predicate ("does this
  commit message describe this diff?"); CodeReviewer questions are always
  refused. Message-match negatives are another commit's subject, which is
  easy; TEST accuracy 0.989 says little about subtler mismatches.
- One head, one seed, one config; CAL 650 and TEST 850 rows; CommitPackFT TEST
  components are new repositories, CodeReviewer TEST projects are new to the
  head but were Kev train projects (their Kev rows trained the Part 2 head).
- The gate is tested on five OOD sources, two of them near (same diffs,
  another question; when2call); questions phrased like the in-scope ones
  about non-diff input are refused only by the declarative state-prefix
  check, and paraphrases of the in-scope questions are refused by the exact
  question check, not measured by the model.
- `cr-new` features were not recaptured (request hash reproduced; `kev-ood`
  recapture was bit-identical).
- Nothing here bears on MTP acceptance or coding quality.
