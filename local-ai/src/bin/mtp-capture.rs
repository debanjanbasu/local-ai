//! Self-distillation data for the MTP draft head (see `tools/mtp_train`).
//!
//! ```text
//! mtp-capture tables DIR
//! mtp-capture signs DIR
//! mtp-capture capture --out DIR [options] CORPUS.jsonl...
//! mtp-capture parity  --out DIR [--doc N] [--positions P,..] [--depth D] CORPUS.jsonl...
//! ```
//!
//! Paths resolve like `local-ai bonsai`: the target is discovered from the
//! working directory, and `parity` loads the head discovery finds beside it.

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

/// `local_metal::bonsai::DEFAULT_SMALL_BATCH_MAX`: the largest verify block the
/// per-row logits path runs on its small-batch kernels.
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
