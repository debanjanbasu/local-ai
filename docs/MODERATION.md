# Moderation: calibrated heads on EmbeddingGemma 2

`POST /v1/moderations` scores text with small calibrated heads (one logistic
regression per category) on the frozen native EmbeddingGemma 2 text
embeddings ([EMBEDDINGS.md](EMBEDDINGS.md)). No new large model is involved.
It is opt-in by installation: the route exists only when both the embedding
model and the moderation artifact are installed; otherwise it is `404`.

**It scores only the categories that passed a pre-registered evaluation**
(see [Acceptance criteria](#acceptance-criteria-pre-registered) and
[Results](#results)). Every other category is reported as *not scored*, never
as a guessed score. It is not a replacement for OpenAI's omni-moderation
model: different training data, a different and narrower label mapping, and
text only.

## Install

```bash
mkdir -p models/moderation
cp local-engine/assets/moderation/moderation-heads.json models/moderation/
```

The server verifies on startup that the file's SHA-256 equals the pinned
`local_engine::moderation::MODERATION_HEADS_SHA256`, that it names the exact
EmbeddingGemma 2 payload it was trained on (`a7c15de9…d4296c5`, checked
against the loaded encoder), and that every head is well formed; a bad
artifact stops startup. Startup logs the scored categories.

## API

`POST /v1/moderations`, per the pinned `CreateModerationRequest` /
`CreateModerationResponse`:

- `input`: a string or an array of strings (one result each, at most 256),
  or an array of multi-modal objects (one result for the whole array; its
  `text` parts are joined with newlines). **`image_url` parts are refused
  with `400`** until native image embeddings land; the integration point is
  `moderations.rs`: embed the image with the vision encoder and score it with
  heads trained and evaluated on image embeddings (the text heads must not be
  reused on image vectors). At most 300,000 tokens per request.
- `model`: optional; `embeddinggemma-2-moderation-1`, or one of the OpenAI
  names (`omni-moderation-latest`, `omni-moderation-2024-09-26`,
  `text-moderation-latest`, `text-moderation-stable`) accepted as aliases so
  SDK defaults work. The response's `model` is always
  `embeddinggemma-2-moderation-1`.
- Response: all 13 categories, as the spec requires. For a **scored**
  category: `category_scores` is the calibrated probability,
  `categories` is `score >= threshold` (thresholds below), and
  `category_applied_input_types` is `["text"]`. For an **unscored** category:
  `category_applied_input_types` is `[]` (the score applies to no input),
  `category_scores` is `0`, and `categories` is `false` — or `null` for
  `illicit` and `illicit/violent`, the only categories whose type allows it.
  An unscored category never sets `flagged`. The unscored categories are also
  listed in the `x-moderation-unscored-categories` response header. Clients
  must read `category_applied_input_types` (or the header) before treating a
  `false` as "evaluated and clean".
- `flagged`: whether any scored category is flagged.

Scoring: the text is prefixed with the model card's classification prompt
(`task: classification | query: `) and embedded (768-d, normalized). Inputs
over 512 tokens (prompt included) are split into 512-token windows that each
carry the prompt; a category's score is the maximum over windows. The heads
were fitted on the first window of each training text. Requests share the
embedding route's queue (at most 8 admitted, `503` + `Retry-After` beyond)
and its yielding to Bonsai decode.

## Data and provenance

Downloaded by a private Kaggle CPU kernel (internet on) at pinned revisions;
every file below is checked by `tools/moderation_data.py` against the
kernel's manifest before use.

| Source | License | Revision | Use |
| --- | --- | --- | --- |
| NVIDIA Aegis 2.0 (`nvidia/Aegis-AI-Content-Safety-Dataset-2.0`) | CC-BY-4.0 | `d86bb8be` | train / calibration / test |
| Civil Comments (`google/civil_comments`, the Jigsaw Unintended Bias data) | CC0-1.0 | `f2970eb3` | train / calibration / test (stratified sample of the train split) |
| OpenAI moderation evaluation set (`openai/moderation-api-release`, `samples-1680.jsonl.gz`) | MIT | `f4ab51b5` | out-of-distribution test only |
| ToxicChat 0124 test (`lmsys/toxic-chat`) | CC-BY-NC-4.0 | `29df8e4d` | out-of-distribution report only (never trained on) |

Source file SHA-256s: Aegis `train.json` `154fba82…13e02f6`,
`validation.json` `a97200e2…951dfcd`, `test.json` `b0a6d602…9318f35`; Civil
Comments `data/train-00000-of-00002.parquet` `c20f01c3…d6ccefe`,
`data/train-00001-of-00002.parquet` `a76af69d…6d99ccb` (1,804,874 rows;
sample: every row with identity_attack, threat, sexual_explicit, insult or
toxicity ≥ 0.5 up to 8,000 each, severe_toxicity up to 3,000, plus 25,000
uniformly random rows, seed 20261013: 55,628 rows); OpenAI
`samples-1680.jsonl.gz` `ef3f7de8…63ab9a8`; ToxicChat
`toxic-chat_annotation_test.csv` `3c2e4988…3c4c9b`. The full hashes, the
derived dataset's SHA-256 and every feature chunk's SHA-256 are recorded in
the artifact's `provenance`.

Splits: Aegis test → test, validation → calibration, its train file 80 / 10
/ 10 (train / calibration / test) by id hash; Civil Comments 70 / 15 / 15 by
row hash. A text is kept only in its most held-out split (OOD, test,
calibration, train), so nothing evaluated or calibrated on is fitted on
(2,247 duplicate or leaking rows dropped).

## Category mapping

Aegis 2.0 (per row: prompt-only rows use the prompt; when the response is
safe, the prompt carries the categories and the response is a negative; when
both are unsafe, prompt and response together, since the categories may come
from either; rows whose only categories are `Needs Caution`/`Other` are
dropped). Civil Comments: annotator fraction ≥ 0.5 positive, < 0.1 negative,
in between not annotated.

| Spec category | Aegis 2.0 | Civil Comments | OpenAI eval (OOD) | Notes |
| --- | --- | --- | --- | --- |
| hate | Hate/Identity Hate | identity_attack | H | |
| hate/threatening | Hate + (Threat or Violence) | identity_attack and threat | H2 | weak: few positives |
| harassment | Harassment or Threat | insult or threat | HR | insult ≠ harassment exactly |
| harassment/threatening | Threat | threat | — | |
| illicit | Criminal Planning/Confessions, Illegal Activity, Controlled/Regulated Substances, Fraud/Deception, Malware, Guns and Illegal Weapons | assumed negative | — | Aegis includes confessions and requests, not only instructions |
| illicit/violent | Guns and Illegal Weapons, or Criminal Planning + Violence | assumed negative | — | approximation |
| self-harm | Suicide and Self Harm | assumed negative | SH | |
| self-harm/intent | — | — | — | **no data: unsupported** |
| self-harm/instructions | — | — | — | **no data: unsupported** |
| sexual | Sexual or Sexual (minor) | sexual_explicit | S | |
| sexual/minors | Sexual (minor) | negative only if sexual_explicit < 0.1 | S3 | few positives |
| violence | Violence or Threat | not annotated | V | OpenAI 2022 V is "promotes or glorifies"; current spec is "depicts" |
| violence/graphic | — | — | V2 | **no training data: unsupported** |

"Assumed negative" means news comments are taken not to contain that
category; this is noise the evaluation inherits, documented here.

## Method

1. Features: `target/release/moderation-embed features` (native Metal
   encoder, the server's exact prompt and first window), 768-d.
2. `tools/moderation_train.py fit`: per category with enough data, L2
   logistic regression on `train` (scikit-learn, `C` from {0.3, 1, 3, 10, 30}
   by calibration log-loss); Platt scaling (`p = sigmoid(a·logit + c)`) fitted
   on `calibration`; the `flagged` threshold chosen on `calibration` as the
   highest-recall threshold with precision ≥ 0.80 (recall ≥ 0.30 required).
3. `tools/moderation_train.py evaluate`: test and OOD scored once; criteria
   applied; failing categories become unsupported in the artifact, with the
   failed criteria as the reason.

## Acceptance criteria (pre-registered)

Committed before any head was fitted and before the test split was scored.
A category ships only if all of these hold:

- Data: ≥ 100 train positives and ≥ 30 calibration positives (else not
  fitted).
- Calibration: some threshold reaches precision ≥ 0.80 with recall ≥ 0.30.
- In-distribution test (Aegis 2.0 + Civil Comments test split), at the
  calibration-chosen threshold:
  - ≥ 30 test positives (else "insufficient evaluation data");
  - ROC AUC ≥ 0.90;
  - precision ≥ 0.70;
  - recall ≥ 0.50;
  - ECE (10 equal-width bins) ≤ 0.05.
- Out of distribution (OpenAI 2022 evaluation set), where the category maps
  and has ≥ 20 positives: ROC AUC ≥ 0.80. Categories without OpenAI-eval
  labels (illicit, illicit/violent, harassment/threatening) have no OOD gate.
- ToxicChat is reported (AUC of the maximum score, precision and recall of
  `flagged` against its binary toxicity label), not gated.

Confidence intervals (Wilson 95%) are reported for precision and recall;
gating uses the point estimates.

## Results

Pending: filled in from the single evaluation run.
