//! Self-distillation data for the MTP draft head (see `tools/mtp_train`).
//!
//! ```text
//! mtp-capture tables DIR
//! mtp-capture signs DIR
//! mtp-capture capture --out DIR [options] CORPUS.jsonl...
//! mtp-capture parity  --out DIR [--doc N] [--positions P,..] [--depth D] CORPUS.jsonl...
//! mtp-capture features --out DIR [--max-tokens N] [--rows N] ROWS.jsonl
//! mtp-capture features --validate-only [--out DIR] [--max-tokens N] ROWS.jsonl
//! ```
//!
//! Paths resolve like `local-ai bonsai`: the target is discovered from the
//! working directory, and `parity` loads the head discovery finds beside it.
//!
//! # `features`: frozen-target hidden features at selected positions
//!
//! Input: one JSON object per line (blank lines and unknown fields rejected):
//!
//! ```text
//! {"id": "<string>", "text": "<raw text>", "positions": [<int>, ...], "metadata": <any JSON, optional>}
//! {"id": "<string>", "text": "<raw text>", "token_end_offsets": [<int>, ...], "metadata": <any JSON, optional>}
//! ```
//!
//! `text` is tokenized with the target's own Bonsai tokenizer, as-is: no chat
//! template, no BOS/EOS added (special-token text inside `text` still maps to
//! its special id). A row gives exactly one of:
//!
//! - `positions`: zero-based indices into those token ids;
//! - `token_end_offsets`: exclusive UTF-8 byte endpoints into `text`. One
//!   encode of the whole text resolves each to the index of the token ending
//!   exactly there; an endpoint that is zero, past the text, inside a UTF-8
//!   character or inside a (merged) token is an error, never rounded.
//!
//! Either way the positions must be strictly increasing, each below the token
//! count; the last token (`len - 1`) is allowed. A row needs
//! `2..=--max-tokens` tokens (default 2048); longer rows are an error, never
//! truncated. `metadata` (e.g. `{"label": ..., "split": ...}`) is passed
//! through untouched. Ids must be non-empty and unique. Every row is
//! validated before the model loads.
//!
//! `--validate-only` stops there: it tokenizes and checks every row, prints a
//! JSON summary (rows, vectors, token counts, rows per string
//! `metadata.split`), and neither loads the model for inference nor writes
//! anything. With `--out` it also checks that the directory would be accepted.
//!
//! Output in `--out DIR`, which must be absent or empty (no resume):
//!
//! - `feature-NNNNNN.bin` for input row `N` (zero-based, six digits, never
//!   derived from `id`): `positions.len()` consecutive vectors of 5120
//!   little-endian FP16 values, in `positions` order, no header. Vector `i` is
//!   the target's output-normalized final hidden (`output_norm` applied,
//!   unrotated Qwen basis) of the row whose input is token `positions[i]` —
//!   the hidden that produces the logits predicting token `positions[i] + 1`.
//!   File size is exactly `positions.len() * 5120 * 2` bytes.
//! - `features.jsonl`, one line per captured row, in input order:
//!   `{"id", "token_ids", "positions", "width": 5120, "feature_file":
//!   "feature-NNNNNN.bin", "metadata": <input metadata or null>}`.
//!
//! Each binary is written to `feature-NNNNNN.bin.partial`, checked for
//! size, and renamed; its manifest line is appended only afterwards. On
//! failure the run stops: manifest lines name only complete files, and a
//! leftover `.partial` is never listed.

use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use local_engine::bonsai::BonsaiPackage;
use local_engine::bonsai_native::capture::{
    HEAD_SIGN_WIDTHS, MtpCapture, export_head_tables, export_signs,
};
use local_engine::bonsai_tokenizer::{BonsaiTokenizer, ChatMessage};
use local_engine::resources::Resources;
use local_engine::{MtpMode, MtpSettings};

const USAGE: &str = "usage:
  mtp-capture tables DIR
      write embed_tokens/lm_head F16 tables (unrotated basis) and meta.json
  mtp-capture signs DIR
      write the target's Hadamard signs-{5120,6144,17408}.bin (raw i8 ±1) and
      rotate-check-*.bin test vectors for a ternary head (tools/mtp_train)
  mtp-capture capture --out DIR [--max-docs N] [--max-positions N] [--skip-docs N]
                      [--max-doc-tokens N] [--min-doc-tokens N] [--rows N] [--top-k N]
                      CORPUS.jsonl...
      one shard-NNNNNN.bin per document; existing shards are kept (resume)
  mtp-capture parity --out DIR [--doc N] [--positions P,..] [--depth D] CORPUS.jsonl...
      capture one document and record the runtime head's draft chains
  mtp-capture features --out DIR [--max-tokens N] [--rows N] ROWS.jsonl
      output-normalized hidden (5120 × LE fp16) at selected token positions;
      rows {id, text, positions:[int] | token_end_offsets:[int], metadata?},
      raw text, no template; token_end_offsets are exclusive UTF-8 byte ends
      that must each end a token exactly. DIR must be absent or empty; writes
      feature-NNNNNN.bin per input row and features.jsonl {id, token_ids,
      positions, width, feature_file, metadata}. Rows over --max-tokens
      (default 2048) are errors, never truncated.
  mtp-capture features --validate-only [--out DIR] [--max-tokens N] ROWS.jsonl
      tokenize and validate every row, print a JSON summary; no capture, no
      output written

Corpus lines are JSON objects with `messages` (chat template, no thinking)
or `text` (raw). Files are read round-robin, one document from each in turn.";

struct Options {
    out: PathBuf,
    corpus: Vec<PathBuf>,
    max_docs: usize,
    max_positions: usize,
    skip_docs: usize,
    max_doc_tokens: usize,
    min_doc_tokens: usize,
    rows: usize,
    top_k: usize,
    doc: usize,
    positions: Vec<usize>,
    depth: usize,
}

fn parse(args: &[String]) -> Result<Options, String> {
    let mut options = Options {
        out: PathBuf::new(),
        corpus: Vec::new(),
        max_docs: usize::MAX,
        max_positions: usize::MAX,
        skip_docs: 0,
        max_doc_tokens: 2048,
        min_doc_tokens: 64,
        rows: DEFAULT_ROWS,
        top_k: 8,
        doc: 0,
        positions: vec![16, 48, 96, 160, 255],
        depth: 3,
    };
    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        let mut value = || {
            iter.next()
                .cloned()
                .ok_or_else(|| format!("{arg} needs a value"))
        };
        let number = |text: String| {
            text.parse::<usize>()
                .map_err(|_| format!("{arg}: {text:?} is not a whole number"))
        };
        match arg.as_str() {
            "--out" => options.out = PathBuf::from(value()?),
            "--max-docs" => options.max_docs = number(value()?)?,
            "--max-positions" => options.max_positions = number(value()?)?,
            "--skip-docs" => options.skip_docs = number(value()?)?,
            "--max-doc-tokens" => options.max_doc_tokens = number(value()?)?,
            "--min-doc-tokens" => options.min_doc_tokens = number(value()?)?,
            "--rows" => options.rows = number(value()?)?,
            "--top-k" => options.top_k = number(value()?)?,
            "--doc" => options.doc = number(value()?)?,
            "--depth" => options.depth = number(value()?)?,
            "--positions" => {
                options.positions = value()?
                    .split(',')
                    .map(|part| number(part.to_owned()))
                    .collect::<Result<_, _>>()?;
            }
            flag if flag.starts_with("--") => return Err(format!("unknown option {flag}")),
            path => options.corpus.push(PathBuf::from(path)),
        }
    }
    if options.out.as_os_str().is_empty() {
        return Err("--out is required".into());
    }
    if options.corpus.is_empty() {
        return Err("at least one corpus file is required".into());
    }
    Ok(options)
}

/// Within `local_metal::bonsai::DEFAULT_SMALL_BATCH_MAX`, so the per-row
/// logits path runs on the small-batch kernels.
const DEFAULT_ROWS: usize = 60;

/// One corpus document: where it came from and its rendered text.
struct Document {
    source: String,
    line: usize,
    text: String,
}

/// Read every corpus file round-robin, yielding rendered documents.
fn documents(paths: &[PathBuf]) -> local_engine::Result<impl Iterator<Item = Document>> {
    let mut readers = paths
        .iter()
        .map(|path| {
            let file = std::fs::File::open(path)?;
            Ok((
                path.display().to_string(),
                std::io::BufReader::new(file).lines().enumerate(),
            ))
        })
        .collect::<local_engine::Result<Vec<_>>>()?;
    let mut turn = 0;
    Ok(std::iter::from_fn(move || {
        while !readers.is_empty() {
            let index = turn % readers.len();
            let (source, lines) = &mut readers[index];
            match lines.next() {
                Some((line, Ok(text))) => {
                    turn += 1;
                    if let Some(text) = render(&text) {
                        return Some(Document {
                            source: source.clone(),
                            line,
                            text,
                        });
                    }
                }
                Some((_, Err(_))) | None => {
                    drop(readers.remove(index));
                }
            }
        }
        None
    }))
}

fn render(line: &str) -> Option<String> {
    let value: serde_json::Value = serde_json::from_str(line).ok()?;
    if let Some(messages) = value.get("messages").and_then(|m| m.as_array()) {
        let messages: Vec<ChatMessage<'_>> = messages
            .iter()
            .filter_map(|message| {
                Some(ChatMessage {
                    role: message.get("role")?.as_str()?,
                    content: message.get("content")?.as_str()?,
                    reasoning_content: None,
                })
            })
            .collect();
        return BonsaiTokenizer::chat_messages(&messages, false).ok();
    }
    value
        .get("text")
        .and_then(|text| text.as_str())
        .filter(|text| !text.trim().is_empty())
        .map(str::to_owned)
}

fn head_settings(resources: &Resources) -> local_engine::Result<MtpSettings> {
    match &resources.mtp {
        MtpMode::Head(settings) => Ok(settings.clone()),
        _ => Err(local_engine::Error::InvalidArgument(format!(
            "parity needs an MTP head: {}",
            resources.mtp_reason
        ))),
    }
}

fn write_shard(
    capture: &mut MtpCapture,
    tokens: &[u32],
    top_k: usize,
    path: &Path,
) -> local_engine::Result<()> {
    let partial = path.with_extension("partial");
    let mut out = std::io::BufWriter::new(std::fs::File::create(&partial)?);
    capture.capture_document(tokens, top_k, &mut out)?;
    out.flush()?;
    drop(out);
    std::fs::rename(&partial, path)?;
    Ok(())
}

fn capture(options: &Options) -> local_engine::Result<()> {
    let resources = Resources::discover(None, false)?;
    let tokenizer = BonsaiTokenizer::from_package(&BonsaiPackage::open(&resources.model)?)?;
    let started = Instant::now();
    let mut capture =
        MtpCapture::open(&resources.model, None, options.rows, options.max_doc_tokens)?;
    eprintln!("loaded target in {:.1}s", started.elapsed().as_secs_f64());
    std::fs::create_dir_all(&options.out)?;
    let mut meta = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(options.out.join("meta.jsonl"))?;
    let started = Instant::now();
    let (mut docs, mut positions, mut captured) = (0_usize, 0_usize, 0_usize);
    for (index, document) in documents(&options.corpus)?
        .skip(options.skip_docs)
        .enumerate()
    {
        if docs >= options.max_docs || positions >= options.max_positions {
            break;
        }
        let mut tokens = tokenizer.encode(&document.text)?;
        tokens.truncate(options.max_doc_tokens);
        if tokens.len() < options.min_doc_tokens.max(2) {
            continue;
        }
        let shard = options
            .out
            .join(format!("shard-{:06}.bin", index + options.skip_docs));
        docs += 1;
        positions += tokens.len();
        if shard.is_file() {
            continue;
        }
        write_shard(&mut capture, &tokens, options.top_k, &shard)?;
        captured += tokens.len();
        writeln!(
            meta,
            "{}",
            serde_json::json!({
                "shard": shard.file_name().map(|name| name.to_string_lossy()),
                "source": document.source,
                "line": document.line,
                "tokens": tokens.len(),
            })
        )?;
        let seconds = started.elapsed().as_secs_f64();
        eprintln!(
            "doc {docs} ({} tokens): {positions} positions, {:.1} captured positions/s",
            tokens.len(),
            captured as f64 / seconds
        );
    }
    println!(
        "{}",
        serde_json::json!({
            "documents": docs,
            "positions": positions,
            "captured_positions": captured,
            "seconds": started.elapsed().as_secs_f64(),
            "positions_per_second": captured as f64 / started.elapsed().as_secs_f64(),
        })
    );
    Ok(())
}

fn parity(options: &Options) -> local_engine::Result<()> {
    let resources = Resources::discover_with_depth(None, true, options.depth.clamp(1, 4))?;
    let settings = head_settings(&resources)?;
    let tokenizer = BonsaiTokenizer::from_package(&BonsaiPackage::open(&resources.model)?)?;
    let document = documents(&options.corpus)?
        .nth(options.doc)
        .ok_or_else(|| local_engine::Error::InvalidArgument("no such document".into()))?;
    let mut tokens = tokenizer.encode(&document.text)?;
    tokens.truncate(options.max_doc_tokens);
    let mut capture = MtpCapture::open(
        &resources.model,
        Some(&settings),
        options.rows,
        options.max_doc_tokens,
    )?;
    std::fs::create_dir_all(&options.out)?;
    write_shard(
        &mut capture,
        &tokens,
        options.top_k,
        &options.out.join("shard-000000.bin"),
    )?;
    let mut chains = Vec::new();
    for &position in options
        .positions
        .iter()
        .filter(|&&p| p > 0 && p + options.depth < tokens.len())
    {
        chains.push(capture.draft_chain(&tokens, position, options.depth, options.top_k)?);
    }
    let report = serde_json::json!({
        "head": settings.path,
        "source": document.source,
        "line": document.line,
        "tokens": tokens,
        "chains": chains,
    });
    std::fs::write(
        options.out.join("parity.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    println!("wrote {} chains to {}", chains.len(), options.out.display());
    Ok(())
}

/// Hidden width of every feature vector.
const FEATURE_WIDTH: usize = local_engine::bonsai::WIDTH;
const FEATURE_MANIFEST: &str = "features.jsonl";

struct FeatureOptions {
    /// Required unless `validate_only`.
    out: Option<PathBuf>,
    input: PathBuf,
    max_tokens: usize,
    rows: usize,
    validate_only: bool,
}

fn parse_features(args: &[String]) -> Result<FeatureOptions, String> {
    let (mut out, mut inputs) = (None, Vec::new());
    let (mut max_tokens, mut rows, mut validate_only) = (2048, DEFAULT_ROWS, false);
    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        let mut value = || {
            iter.next()
                .cloned()
                .ok_or_else(|| format!("{arg} needs a value"))
        };
        let number = |text: String| {
            text.parse::<usize>()
                .map_err(|_| format!("{arg}: {text:?} is not a whole number"))
        };
        match arg.as_str() {
            "--out" => out = Some(PathBuf::from(value()?)),
            "--max-tokens" => max_tokens = number(value()?)?,
            "--rows" => rows = number(value()?)?,
            "--validate-only" => validate_only = true,
            flag if flag.starts_with("--") => return Err(format!("unknown option {flag}")),
            path => inputs.push(PathBuf::from(path)),
        }
    }
    if out.is_none() && !validate_only {
        return Err("--out is required (or --validate-only)".into());
    }
    let [input] = <[PathBuf; 1]>::try_from(inputs)
        .map_err(|_| "features takes exactly one ROWS.jsonl".to_owned())?;
    if !(2..=local_engine::bonsai::TRAINING_CONTEXT).contains(&max_tokens) {
        return Err(format!(
            "--max-tokens must be within 2..={}",
            local_engine::bonsai::TRAINING_CONTEXT
        ));
    }
    if !(3..=MtpCapture::MAX_ROWS).contains(&rows) {
        return Err(format!(
            "--rows must be within 3..={}",
            MtpCapture::MAX_ROWS
        ));
    }
    Ok(FeatureOptions {
        out,
        input,
        max_tokens,
        rows,
        validate_only,
    })
}

/// One `features` input line.
#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct FeatureRow {
    id: String,
    text: String,
    /// Absent is `None`; an explicit `null` is rejected, not read as absent.
    #[serde(default, deserialize_with = "present")]
    positions: Option<Vec<usize>>,
    #[serde(default, deserialize_with = "present")]
    token_end_offsets: Option<Vec<usize>>,
    #[serde(default)]
    metadata: Option<serde_json::Value>,
}

fn present<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> Result<Option<Vec<usize>>, D::Error> {
    <Vec<usize> as serde::Deserialize>::deserialize(deserializer).map(Some)
}

/// How a row selects its feature positions.
#[derive(Debug, PartialEq, Eq)]
enum Selection<'a> {
    Positions(&'a [usize]),
    TokenEndOffsets(&'a [usize]),
}

impl FeatureRow {
    /// The row's one position selector; both, neither or an empty id is an
    /// error.
    fn selection(&self) -> Result<Selection<'_>, String> {
        if self.id.is_empty() {
            return Err("id is empty".into());
        }
        match (&self.positions, &self.token_end_offsets) {
            (Some(positions), None) => Ok(Selection::Positions(positions)),
            (None, Some(ends)) if ends.is_empty() => Err("token_end_offsets is empty".into()),
            (None, Some(ends)) => Ok(Selection::TokenEndOffsets(ends)),
            (Some(_), Some(_)) => {
                Err("give exactly one of positions or token_end_offsets, not both".into())
            }
            (None, None) => Err("row needs positions or token_end_offsets".into()),
        }
    }
}

/// A validated, tokenized input row.
struct FeatureJob {
    row: FeatureRow,
    tokens: Vec<u32>,
    positions: Vec<usize>,
}

/// Check one row's positions against its tokens, before any model work.
fn check_positions(tokens: usize, positions: &[usize], max_tokens: usize) -> Result<(), String> {
    if tokens < 2 {
        return Err(format!("text has {tokens} tokens, need at least 2"));
    }
    if tokens > max_tokens {
        return Err(format!(
            "text has {tokens} tokens, over --max-tokens {max_tokens} (not truncated)"
        ));
    }
    if positions.is_empty() {
        return Err("positions is empty".into());
    }
    if let Some(&[before, after]) = positions
        .array_windows::<2>()
        .find(|[before, after]| before >= after)
    {
        return Err(format!(
            "positions must be strictly increasing: {before} then {after}"
        ));
    }
    let last = positions[positions.len() - 1];
    if last >= tokens {
        return Err(format!(
            "position {last} is past the final token index {}",
            tokens - 1
        ));
    }
    Ok(())
}

/// Parse, tokenize and validate every input row.
fn feature_jobs(
    input: &Path,
    tokenizer: &BonsaiTokenizer,
    max_tokens: usize,
) -> Result<Vec<FeatureJob>, String> {
    let file = std::fs::File::open(input).map_err(|e| format!("{}: {e}", input.display()))?;
    let mut ids = std::collections::HashSet::new();
    let mut jobs = Vec::new();
    for (index, line) in std::io::BufReader::new(file).lines().enumerate() {
        let at = format!("{}:{}", input.display(), index + 1);
        let line = line.map_err(|e| format!("{at}: {e}"))?;
        let row: FeatureRow =
            serde_json::from_str(&line).map_err(|e| format!("{at}: invalid row: {e}"))?;
        let resolved = match row.selection() {
            Ok(Selection::Positions(positions)) => tokenizer
                .encode(&row.text)
                .map(|tokens| (tokens, positions.to_vec())),
            Ok(Selection::TokenEndOffsets(ends)) => {
                tokenizer.encode_with_token_ends(&row.text, ends)
            }
            Err(message) => Err(local_engine::Error::InvalidArgument(message)),
        };
        let (tokens, positions) = resolved.map_err(|e| format!("{at} (id {:?}): {e}", row.id))?;
        if !ids.insert(row.id.clone()) {
            return Err(format!("{at}: duplicate id {:?}", row.id));
        }
        check_positions(tokens.len(), &positions, max_tokens)
            .map_err(|e| format!("{at} (id {:?}): {e}", row.id))?;
        jobs.push(FeatureJob {
            row,
            tokens,
            positions,
        });
    }
    if jobs.is_empty() {
        return Err(format!("{}: no rows", input.display()));
    }
    Ok(jobs)
}

/// Accept `out` only if it is absent or an empty directory: an earlier
/// run's files are never resumed, overwritten or mixed in. Returns whether
/// it exists.
fn check_feature_dir(out: &Path) -> Result<bool, String> {
    if !out.exists() {
        return Ok(false);
    }
    let mut entries = std::fs::read_dir(out).map_err(|e| format!("{}: {e}", out.display()))?;
    if entries.next().is_some() {
        return Err(format!(
            "{} is not empty; features needs a fresh output directory",
            out.display()
        ));
    }
    Ok(true)
}

/// [`check_feature_dir`], then create `out` if it is absent.
fn prepare_feature_dir(out: &Path) -> Result<(), String> {
    if check_feature_dir(out)? {
        Ok(())
    } else {
        std::fs::create_dir_all(out).map_err(|e| format!("{}: {e}", out.display()))
    }
}

/// The `--validate-only` report: counts only, nothing written.
fn validation_summary(jobs: &[FeatureJob], max_tokens: usize) -> serde_json::Value {
    let lengths = jobs.iter().map(|job| job.tokens.len());
    let mut splits = std::collections::BTreeMap::<String, usize>::new();
    for job in jobs {
        let split = job
            .row
            .metadata
            .as_ref()
            .and_then(|metadata| metadata.get("split"))
            .and_then(serde_json::Value::as_str);
        *splits
            .entry(split.unwrap_or("<none>").to_owned())
            .or_default() += 1;
    }
    serde_json::json!({
        "valid": true,
        "rows": jobs.len(),
        "vectors": jobs.iter().map(|job| job.positions.len()).sum::<usize>(),
        "tokens": {
            "total": lengths.clone().sum::<usize>(),
            "min": lengths.clone().min(),
            "max": lengths.max(),
            "limit": max_tokens,
        },
        "rows_by_split": splits,
    })
}

fn feature_file_name(index: usize) -> String {
    format!("feature-{index:06}.bin")
}

/// Capture one row into `dir/name` through a `.partial` file, checking its
/// size before the rename.
fn write_feature_file(
    capture: &mut MtpCapture,
    job: &FeatureJob,
    dir: &Path,
    name: &str,
) -> local_engine::Result<()> {
    let path = dir.join(name);
    let partial = dir.join(format!("{name}.partial"));
    let result = (|| {
        let file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&partial)?;
        let mut out = std::io::BufWriter::new(file);
        capture.write_features(&job.tokens, &job.positions, &mut out)?;
        let file = out
            .into_inner()
            .map_err(|error| local_engine::Error::Io(error.into_error()))?;
        file.sync_all()?;
        let expected = (job.positions.len() * FEATURE_WIDTH * 2) as u64;
        let actual = file.metadata()?.len();
        if actual != expected {
            return Err(local_engine::Error::InvalidFormat(format!(
                "{name}: wrote {actual} bytes, expected {expected}"
            )));
        }
        if path.exists() {
            return Err(local_engine::Error::InvalidArgument(format!(
                "{name} already exists"
            )));
        }
        std::fs::rename(&partial, &path)?;
        Ok(())
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&partial);
    }
    result
}

fn features(options: &FeatureOptions) -> Result<(), String> {
    let resources = Resources::discover(None, false).map_err(|e| e.to_string())?;
    let tokenizer = BonsaiPackage::open(&resources.model)
        .and_then(|package| BonsaiTokenizer::from_package(&package))
        .map_err(|e| e.to_string())?;
    let jobs = feature_jobs(&options.input, &tokenizer, options.max_tokens)?;
    if options.validate_only {
        if let Some(out) = &options.out {
            check_feature_dir(out)?;
        }
        println!("{}", validation_summary(&jobs, options.max_tokens));
        return Ok(());
    }
    let out = options.out.as_deref().ok_or("--out is required")?;
    prepare_feature_dir(out)?;
    let manifest_path = out.join(FEATURE_MANIFEST);
    let mut manifest = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&manifest_path)
        .map_err(|e| format!("{}: {e}", manifest_path.display()))?;
    let started = Instant::now();
    let mut capture = MtpCapture::open(&resources.model, None, options.rows, options.max_tokens)
        .map_err(|e| e.to_string())?;
    eprintln!("loaded target in {:.1}s", started.elapsed().as_secs_f64());
    let started = Instant::now();
    let mut vectors = 0;
    for (index, job) in jobs.iter().enumerate() {
        let name = feature_file_name(index);
        write_feature_file(&mut capture, job, out, &name)
            .map_err(|e| format!("row {index} (id {:?}): {e}", job.row.id))?;
        let line = serde_json::json!({
            "id": job.row.id,
            "token_ids": job.tokens,
            "positions": job.positions,
            "width": FEATURE_WIDTH,
            "feature_file": name,
            "metadata": job.row.metadata,
        });
        writeln!(manifest, "{line}")
            .and_then(|()| manifest.flush())
            .map_err(|e| format!("{}: {e}", manifest_path.display()))?;
        vectors += job.positions.len();
        eprintln!(
            "row {}/{} ({} tokens, {} positions)",
            index + 1,
            jobs.len(),
            job.tokens.len(),
            job.positions.len()
        );
    }
    println!(
        "{}",
        serde_json::json!({
            "rows": jobs.len(),
            "vectors": vectors,
            "manifest": manifest_path.display().to_string(),
            "seconds": started.elapsed().as_secs_f64(),
        })
    );
    Ok(())
}

fn run(args: &[String]) -> Result<(), String> {
    let Some((command, rest)) = args.split_first() else {
        return Err(USAGE.into());
    };
    let result = match command.as_str() {
        "tables" => {
            let [dir] = rest else {
                return Err(USAGE.into());
            };
            Resources::discover(None, false).and_then(|resources| {
                export_head_tables(&resources.model, Path::new(dir)).map(|meta| println!("{meta}"))
            })
        }
        "signs" => {
            let [dir] = rest else {
                return Err(USAGE.into());
            };
            Resources::discover(None, false).and_then(|resources| {
                export_signs(&resources.model, Path::new(dir), &HEAD_SIGN_WIDTHS)
                    .map(|meta| println!("{meta}"))
            })
        }
        "capture" => capture(&parse(rest)?),
        "parity" => parity(&parse(rest)?),
        "features" => return features(&parse_features(rest)?),
        _ => return Err(USAGE.into()),
    };
    result.map_err(|error| error.to_string())
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("{message}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        FeatureRow, Selection, check_feature_dir, check_positions, feature_file_name,
        parse_features,
    };

    #[test]
    fn feature_positions_allow_final_token_and_reject_bad_lists() {
        assert!(check_positions(4, &[0, 3], 8).is_ok());
        assert!(check_positions(4, &[3], 4).is_ok());
        assert!(check_positions(4, &[4], 8).is_err());
        assert!(check_positions(4, &[2, 2], 8).is_err());
        assert!(check_positions(4, &[2, 1], 8).is_err());
        assert!(check_positions(4, &[], 8).is_err());
        assert!(check_positions(1, &[0], 8).is_err());
        assert!(check_positions(9, &[0], 8).is_err());
    }

    #[test]
    fn feature_rows_are_strict_and_keep_metadata_opaque() -> serde_json::Result<()> {
        let row: FeatureRow = serde_json::from_str(
            r#"{"id":"a/../b","text":"hi there","positions":[0,1],"metadata":{"label":1,"split":"dev"}}"#,
        )?;
        assert_eq!(row.selection(), Ok(Selection::Positions(&[0, 1])));
        assert_eq!(
            row.metadata,
            Some(serde_json::json!({"label":1,"split":"dev"}))
        );
        let row: FeatureRow = serde_json::from_str(r#"{"id":"x","text":"t","positions":[1]}"#)?;
        assert_eq!(row.metadata, None);
        for bad in [
            r#"{"id":"x","text":"t","positions":[-1]}"#,
            r#"{"id":"x","text":"t","positions":[1.5]}"#,
            r#"{"id":"x","text":"t","positions":[1],"label":0}"#,
            r#"{"id":1,"text":"t","positions":[1]}"#,
            r#"{"text":"t","positions":[1]}"#,
            r#"{"id":"x","text":"t","positions":null}"#,
            r#"{"id":"x","text":"t","token_end_offsets":null}"#,
            r#"{"id":"x","text":"t","token_end_offsets":[-1]}"#,
            "",
        ] {
            assert!(serde_json::from_str::<FeatureRow>(bad).is_err(), "{bad}");
        }
        assert_eq!(feature_file_name(7), "feature-000007.bin");
        Ok(())
    }

    #[test]
    fn feature_rows_select_exactly_one_of_positions_or_byte_ends() -> serde_json::Result<()> {
        let row: FeatureRow = serde_json::from_str(
            r#"{"id":"a","text":"é\n","token_end_offsets":[2,3],"metadata":{"target":1}}"#,
        )?;
        assert_eq!(row.selection(), Ok(Selection::TokenEndOffsets(&[2, 3])));
        for bad in [
            r#"{"id":"a","text":"t","positions":[0],"token_end_offsets":[1]}"#,
            r#"{"id":"a","text":"t"}"#,
            r#"{"id":"a","text":"t","token_end_offsets":[]}"#,
            r#"{"id":"","text":"t","positions":[0]}"#,
        ] {
            let row: FeatureRow = serde_json::from_str(bad)?;
            assert!(row.selection().is_err(), "{bad}");
        }
        Ok(())
    }

    #[test]
    fn feature_options_need_out_and_one_input() {
        let args = |list: &[&str]| list.iter().map(|&s| s.to_owned()).collect::<Vec<_>>();
        let options = parse_features(&args(&["--out", "d", "--max-tokens", "64", "r.jsonl"]));
        assert!(options.is_ok_and(|o| o.max_tokens == 64 && o.input.as_os_str() == "r.jsonl"));
        assert!(parse_features(&args(&["r.jsonl"])).is_err());
        assert!(parse_features(&args(&["--out", "d"])).is_err());
        assert!(parse_features(&args(&["--out", "d", "a", "b"])).is_err());
        assert!(parse_features(&args(&["--out", "d", "--resume", "a"])).is_err());
        let options = parse_features(&args(&["--validate-only", "r.jsonl"]));
        assert!(options.is_ok_and(|o| o.validate_only && o.out.is_none()));
        assert!(parse_features(&args(&["--validate-only"])).is_err());
        for (flag, value) in [
            ("--rows", "2"),
            ("--rows", "129"),
            ("--max-tokens", "1"),
            ("--max-tokens", "262145"),
        ] {
            assert!(parse_features(&args(&["--validate-only", flag, value, "r.jsonl"])).is_err());
        }
        for (rows, tokens) in [("3", "2"), ("128", "262144")] {
            assert!(
                parse_features(&args(&[
                    "--validate-only",
                    "--rows",
                    rows,
                    "--max-tokens",
                    tokens,
                    "r.jsonl",
                ]))
                .is_ok()
            );
        }
    }

    #[test]
    fn validate_only_output_check_never_creates_the_directory() -> std::io::Result<()> {
        let root = std::env::temp_dir().join(format!("mtp-capture-check-{}", std::process::id()));
        let absent = root.join("absent");
        assert_eq!(check_feature_dir(&absent), Ok(false));
        assert!(!absent.exists());
        std::fs::create_dir_all(&root)?;
        assert_eq!(check_feature_dir(&root), Ok(true));
        std::fs::write(root.join("x"), b"")?;
        assert!(check_feature_dir(&root).is_err());
        std::fs::remove_dir_all(&root)
    }
}
