//! EXPERIMENTAL: capture judgment-head features through the serving path.
//!
//! ```text
//! judgment-capture --out DIR [--max-tokens N] REQUESTS.jsonl
//! judgment-capture --validate-only [--max-tokens N] REQUESTS.jsonl
//! ```
//!
//! Each input line is one typed decision (unknown fields and blank lines are
//! rejected):
//!
//! ```text
//! {"id": "<string>", "state": "<text>", "question": "<text>",
//!  "kind": "predicate" | "choice" | "score",
//!  "options": [<string or bool>, ...],   // choice values / score labels; absent for predicate
//!  "metadata": {<object, optional>}}
//! ```
//!
//! Every row is rendered by `DecisionRequest::render`, tokenized and checked
//! (`DecisionCapture::prepare`) before the model loads. Capture then runs
//! `Engine::capture_decision`, the same renderer, endpoint resolution,
//! 60-row capture blocks and FP16 rounding a served decision scores, so
//! scoring these features with a head reproduces that head's decisions.
//!
//! Output in `--out DIR` (absent or empty, never resumed), in the
//! `mtp-capture features` format that `tools/judgment_train.py` reads:
//! `feature-NNNNNN.bin` (option rows then the decision row, 5120 LE FP16
//! values each) and `features.jsonl` lines `{"id", "token_ids", "positions",
//! "width", "feature_file", "metadata"}`. `metadata` is the input object
//! plus `native_render`: `{"capture": "Engine::capture_decision", "renderer",
//! "text_sha256", "byte_ends", "options"}`. Files go through `.partial` and a
//! manifest line is written only after its file is complete.

use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use local_engine::Engine;
use local_engine::bonsai::BonsaiPackage;
use local_engine::bonsai_tokenizer::BonsaiTokenizer;
use local_engine::judgment::{
    DECISION_RENDERER, DecisionCapture, DecisionKind, DecisionRequest, DecisionValue,
};
use local_engine::resources::Resources;

const USAGE: &str = "usage:
  judgment-capture --out DIR [--max-tokens N] REQUESTS.jsonl
  judgment-capture --validate-only [--max-tokens N] REQUESTS.jsonl
      rows {id, state, question, kind: predicate|choice|score, options?, metadata?};
      captures option and Decision: rows with Engine::capture_decision";

const WIDTH: usize = local_engine::bonsai::WIDTH;
const MANIFEST: &str = "features.jsonl";

struct Options {
    out: Option<PathBuf>,
    input: PathBuf,
    max_tokens: usize,
    validate_only: bool,
}

fn parse(args: &[String]) -> Result<Options, String> {
    let (mut out, mut inputs, mut max_tokens, mut validate_only) = (None, Vec::new(), 2048, false);
    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--out" => out = Some(PathBuf::from(iter.next().ok_or("--out needs a value")?)),
            "--max-tokens" => {
                let text = iter.next().ok_or("--max-tokens needs a value")?;
                max_tokens = text
                    .parse()
                    .map_err(|_| format!("--max-tokens: {text:?} is not a whole number"))?;
            }
            "--validate-only" => validate_only = true,
            flag if flag.starts_with("--") => return Err(format!("unknown option {flag}")),
            path => inputs.push(PathBuf::from(path)),
        }
    }
    let [input] = <[PathBuf; 1]>::try_from(inputs).map_err(|_| USAGE.to_owned())?;
    if out.is_none() && !validate_only {
        return Err("--out is required (or --validate-only)".into());
    }
    if !(2..=local_engine::bonsai::TRAINING_CONTEXT).contains(&max_tokens) {
        return Err(format!(
            "--max-tokens must be within 2..={}",
            local_engine::bonsai::TRAINING_CONTEXT
        ));
    }
    Ok(Options {
        out,
        input,
        max_tokens,
        validate_only,
    })
}

/// One input line.
#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct Row {
    id: String,
    state: String,
    question: String,
    kind: Kind,
    #[serde(default)]
    options: Option<Vec<Value>>,
    #[serde(default)]
    metadata: Option<serde_json::Map<String, serde_json::Value>>,
}

#[derive(serde::Deserialize, Clone, Copy)]
#[serde(rename_all = "lowercase")]
enum Kind {
    Predicate,
    Choice,
    Score,
}

#[derive(serde::Deserialize, Clone)]
#[serde(untagged)]
enum Value {
    Bool(bool),
    Text(String),
}

impl Row {
    /// The typed request; option rules beyond shape are the renderer's.
    fn request(&self) -> Result<DecisionRequest, String> {
        let kind = match (self.kind, &self.options) {
            (Kind::Predicate, None) => DecisionKind::Predicate,
            (Kind::Predicate, Some(_)) => return Err("a predicate takes no options".into()),
            (Kind::Choice, Some(options)) => DecisionKind::Choice(
                options
                    .iter()
                    .map(|value| match value {
                        Value::Bool(value) => DecisionValue::Bool(*value),
                        Value::Text(text) => DecisionValue::Text(text.clone()),
                    })
                    .collect(),
            ),
            (Kind::Score, Some(options)) => DecisionKind::Score(
                options
                    .iter()
                    .map(|value| match value {
                        Value::Text(text) => Ok(text.clone()),
                        Value::Bool(_) => Err("score levels must be strings"),
                    })
                    .collect::<Result<_, _>>()?,
            ),
            (_, None) => return Err("choice and score rows need options".into()),
        };
        if self.id.is_empty() {
            return Err("id is empty".into());
        }
        Ok(DecisionRequest {
            state: self.state.clone(),
            question: self.question.clone(),
            kind,
        })
    }
}

/// A validated row.
struct Job {
    row: Row,
    request: DecisionRequest,
    prepared: DecisionCapture,
}

fn jobs(input: &Path, tokenizer: &BonsaiTokenizer, max_tokens: usize) -> Result<Vec<Job>, String> {
    let file = std::fs::File::open(input).map_err(|e| format!("{}: {e}", input.display()))?;
    let mut ids = std::collections::HashSet::new();
    let mut jobs = Vec::new();
    for (index, line) in std::io::BufReader::new(file).lines().enumerate() {
        let at = format!("{}:{}", input.display(), index + 1);
        let line = line.map_err(|e| format!("{at}: {e}"))?;
        let row: Row =
            serde_json::from_str(&line).map_err(|e| format!("{at}: invalid row: {e}"))?;
        let request = row.request().map_err(|e| format!("{at}: {e}"))?;
        let prepared = DecisionCapture::prepare(tokenizer, &request)
            .map_err(|e| format!("{at} (id {:?}): {e}", row.id))?;
        if prepared.tokens.len() > max_tokens {
            return Err(format!(
                "{at} (id {:?}): {} tokens, over --max-tokens {max_tokens} (not truncated)",
                row.id,
                prepared.tokens.len()
            ));
        }
        if !ids.insert(row.id.clone()) {
            return Err(format!("{at}: duplicate id {:?}", row.id));
        }
        jobs.push(Job {
            row,
            request,
            prepared,
        });
    }
    if jobs.is_empty() {
        return Err(format!("{}: no rows", input.display()));
    }
    Ok(jobs)
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = ring::digest::digest(&ring::digest::SHA256, bytes);
    digest
        .as_ref()
        .iter()
        .fold(String::with_capacity(64), |mut hex, byte| {
            use std::fmt::Write as _;
            let _ = write!(hex, "{byte:02x}");
            hex
        })
}

/// The row's output metadata: its own, plus how it was rendered.
fn metadata(job: &Job) -> serde_json::Value {
    let rendered = &job.prepared.rendered;
    let mut metadata = job.row.metadata.clone().unwrap_or_default();
    metadata.insert(
        "native_render".into(),
        serde_json::json!({
            "capture": "Engine::capture_decision",
            "renderer": DECISION_RENDERER,
            "text_sha256": sha256_hex(rendered.text.as_bytes()),
            "byte_ends": rendered.byte_ends,
            "options": rendered.options.iter().map(DecisionValue::rendered).collect::<Vec<_>>(),
        }),
    );
    serde_json::Value::Object(metadata)
}

fn write_features(dir: &Path, name: &str, features: &[f32]) -> Result<(), String> {
    let path = dir.join(name);
    let partial = dir.join(format!("{name}.partial"));
    let bytes: Vec<u8> = features
        .iter()
        .flat_map(|&value| half::f16::from_f32(value).to_le_bytes())
        .collect();
    let result = (|| -> std::io::Result<()> {
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&partial)?;
        file.write_all(&bytes)?;
        file.sync_all()?;
        if path.exists() {
            return Err(std::io::Error::other(format!("{name} already exists")));
        }
        std::fs::rename(&partial, &path)
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&partial);
    }
    result.map_err(|e| format!("{}: {e}", path.display()))
}

fn run(options: &Options) -> Result<(), String> {
    let resources = Resources::discover(None, false).map_err(|e| e.to_string())?;
    let tokenizer = BonsaiPackage::open(&resources.model)
        .and_then(|package| BonsaiTokenizer::from_package(&package))
        .map_err(|e| e.to_string())?;
    let jobs = jobs(&options.input, &tokenizer, options.max_tokens)?;
    if options.validate_only {
        let tokens: usize = jobs.iter().map(|job| job.prepared.tokens.len()).sum();
        println!(
            "{}",
            serde_json::json!({"valid": true, "rows": jobs.len(), "tokens": tokens})
        );
        return Ok(());
    }
    let out = options.out.as_deref().ok_or("--out is required")?;
    if out.exists() {
        let mut entries = std::fs::read_dir(out).map_err(|e| format!("{}: {e}", out.display()))?;
        if entries.next().is_some() {
            return Err(format!("{} is not empty", out.display()));
        }
    } else {
        std::fs::create_dir_all(out).map_err(|e| format!("{}: {e}", out.display()))?;
    }
    let manifest_path = out.join(MANIFEST);
    let mut manifest = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&manifest_path)
        .map_err(|e| format!("{}: {e}", manifest_path.display()))?;
    let started = Instant::now();
    let mut engine =
        Engine::from_resources(&Resources::discover(None, true).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
    eprintln!("loaded engine in {:.1}s", started.elapsed().as_secs_f64());
    let started = Instant::now();
    for (index, job) in jobs.iter().enumerate() {
        let captured = engine
            .capture_decision(&job.request)
            .map_err(|e| format!("row {index} (id {:?}): {e}", job.row.id))?;
        if captured.tokens != job.prepared.tokens
            || captured.positions != job.prepared.positions
            || captured.features.len() != captured.positions.len() * WIDTH
        {
            return Err(format!(
                "row {index} (id {:?}): capture disagrees with validation",
                job.row.id
            ));
        }
        let name = format!("feature-{index:06}.bin");
        write_features(out, &name, &captured.features)?;
        let line = serde_json::json!({
            "id": job.row.id,
            "token_ids": captured.tokens,
            "positions": captured.positions,
            "width": WIDTH,
            "feature_file": name,
            "metadata": metadata(job),
        });
        writeln!(manifest, "{line}")
            .and_then(|()| manifest.flush())
            .map_err(|e| format!("{}: {e}", manifest_path.display()))?;
        eprintln!(
            "row {}/{} ({} tokens, {} positions)",
            index + 1,
            jobs.len(),
            captured.tokens.len(),
            captured.positions.len()
        );
    }
    println!(
        "{}",
        serde_json::json!({
            "rows": jobs.len(),
            "manifest": manifest_path.display().to_string(),
            "seconds": started.elapsed().as_secs_f64(),
        })
    );
    Ok(())
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match parse(&args).and_then(|options| run(&options)) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("judgment-capture: {error}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Row, parse};
    use local_engine::judgment::{DecisionKind, DecisionValue};

    fn row(json: &str) -> Result<Row, String> {
        serde_json::from_str(json).map_err(|e| e.to_string())
    }

    #[test]
    fn rows_map_to_typed_requests_and_refuse_mismatched_options() {
        let predicate = row(r#"{"id":"a","state":"s","question":"q","kind":"predicate"}"#).unwrap();
        assert_eq!(predicate.request().unwrap().kind, DecisionKind::Predicate);
        let choice =
            row(r#"{"id":"b","state":"s","question":"q","kind":"choice","options":[false,"x"]}"#)
                .unwrap();
        assert_eq!(
            choice.request().unwrap().kind,
            DecisionKind::Choice(vec![
                DecisionValue::Bool(false),
                DecisionValue::Text("x".into())
            ])
        );
        let score = row(
            r#"{"id":"c","state":"s","question":"q","kind":"score","options":["lo","hi"],"metadata":{"split":"test"}}"#,
        )
        .unwrap();
        assert_eq!(
            score.request().unwrap().kind,
            DecisionKind::Score(vec!["lo".into(), "hi".into()])
        );
        for bad in [
            r#"{"id":"d","state":"s","question":"q","kind":"predicate","options":["x"]}"#,
            r#"{"id":"e","state":"s","question":"q","kind":"choice"}"#,
            r#"{"id":"f","state":"s","question":"q","kind":"score","options":[true,"x"]}"#,
            r#"{"id":"","state":"s","question":"q","kind":"predicate"}"#,
        ] {
            assert!(
                row(bad).and_then(|r| r.request().map(|_| ())).is_err(),
                "{bad}"
            );
        }
        assert!(row(r#"{"id":"g","state":"s","question":"q","kind":"predicate","x":1}"#).is_err());
        assert!(row(r#"{"id":"h","state":"s","question":"q","kind":"maybe"}"#).is_err());
    }

    #[test]
    fn options_need_an_output_or_validate_only() {
        let args = |list: &[&str]| list.iter().map(|s| (*s).to_owned()).collect::<Vec<_>>();
        assert!(parse(&args(&["rows.jsonl"])).is_err());
        assert!(parse(&args(&["--validate-only", "rows.jsonl"])).is_ok());
        assert!(parse(&args(&["--out", "d", "rows.jsonl"])).is_ok());
        assert!(parse(&args(&["--out", "d", "--max-tokens", "1", "rows.jsonl"])).is_err());
        assert!(parse(&args(&["--out", "d", "a", "b"])).is_err());
    }
}
