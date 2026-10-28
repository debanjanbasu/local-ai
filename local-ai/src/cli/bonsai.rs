use std::io::Write as _;
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use local_engine::DflashPrecision;
use serde_json::json;

use crate::GenerateParams;
use crate::bonsai::{BonsaiPackage, validate_profile};
use crate::bonsai_model::BonsaiEngine;
use crate::bonsai_native::KvOptions;
use crate::bonsai_ngram::NgramSettings;
use crate::bonsai_tokenizer::BonsaiTokenizer;
use crate::resources::{PREFILL_CHUNK, Resources};

/// What `--export` writes instead of generating. The variants are the kinds as
/// the command line names them, and the directory an export writes belongs to
/// its kind, so `index` cannot be given one.
#[derive(Debug, PartialEq, Eq)]
enum ExportKind {
    Index,
    MtpHead {
        dir: PathBuf,
    },
    DflashHead {
        dir: PathBuf,
        precision: DflashPrecision,
    },
}

#[derive(Debug, PartialEq, Eq)]
enum OutputMode {
    Stream,
    Json,
    Tokenize,
    Export(ExportKind),
}

/// The kinds `--export` accepts, for every message that has to name them.
const EXPORT_KINDS: &str = "index, mtp-head=DIR, dflash-head=DIR, dflash-head-int4=DIR";

/// How an [`OutputMode::Export`] is spelled, for its error messages.
const fn export_flag(kind: &ExportKind) -> &'static str {
    match kind {
        ExportKind::Index => "--export index",
        ExportKind::MtpHead { .. } => "--export mtp-head=DIR",
        ExportKind::DflashHead {
            precision: DflashPrecision::Int8,
            ..
        } => "--export dflash-head=DIR",
        ExportKind::DflashHead {
            precision: DflashPrecision::Int4,
            ..
        } => "--export dflash-head-int4=DIR",
    }
}

/// The `--export` grammar: a kind, with the directory it writes joined by `=`
/// as `--export mtp-head=DIR`. A directory on a kind that writes no file is
/// rejected rather than ignored, so `--export index=/tmp/x` cannot quietly do
/// less than it was asked for.
fn parse_export(requested: &str) -> Result<ExportKind, String> {
    let (kind, dir) = requested
        .split_once('=')
        .map_or((requested, None), |(kind, dir)| (kind, Some(dir)));
    match (kind, dir) {
        ("", _) => Err(format!("--export requires a kind ({EXPORT_KINDS})")),
        ("index", None) => Ok(ExportKind::Index),
        ("index", Some(_)) => Err("--export index takes no directory".into()),
        ("mtp-head", None) => Err("--export mtp-head requires =DIR".into()),
        ("mtp-head", Some("")) => Err("--export mtp-head requires a non-empty =DIR".into()),
        ("mtp-head", Some(dir)) => Ok(ExportKind::MtpHead {
            dir: PathBuf::from(dir),
        }),
        ("dflash-head" | "dflash-head-int4", None) => Err(format!("--export {kind} requires =DIR")),
        ("dflash-head" | "dflash-head-int4", Some("")) => {
            Err(format!("--export {kind} requires a non-empty =DIR"))
        }
        ("dflash-head", Some(dir)) => Ok(ExportKind::DflashHead {
            dir: PathBuf::from(dir),
            precision: DflashPrecision::Int8,
        }),
        ("dflash-head-int4", Some(dir)) => Ok(ExportKind::DflashHead {
            dir: PathBuf::from(dir),
            precision: DflashPrecision::Int4,
        }),
        (kind, _) => Err(format!(
            "unknown --export kind: {kind} (expected {EXPORT_KINDS})"
        )),
    }
}

#[derive(Debug)]
struct Args {
    max_tokens: usize,
    prompt: String,
    prompt_file: Option<PathBuf>,
    raw: bool,
    thinking: bool,
    greedy: bool,
    output: OutputMode,
}

fn usage() {
    eprintln!("Usage: local-ai bonsai [options] <prompt>");
    eprintln!("Bonsai 2 27B text inference: native Metal straight from the GGUF.");
    eprintln!("  --max-tokens N     Output cap (default: 8192)");
    eprintln!("  --prompt-file PATH Read a long UTF-8 prompt instead of positional text");
    eprintln!("  --raw              No chat template (and so no reasoning)");
    eprintln!("  --greedy           Disable sampling for reference comparisons");
    eprintln!("  --json             Emit token IDs, stop reason, and measured timings");
    eprintln!("  --tokenize         Emit prompt token IDs without loading the model");
    eprintln!("  --export KIND      Write an artifact instead of generating; takes no");
    eprintln!("                     other options. Kinds:");
    eprintln!("                       index");
    eprintln!("                         emit the checked GGUF index JSON on stdout");
    eprintln!("                       mtp-head=DIR");
    eprintln!("                         pack a trained ternary head from");
    eprintln!("                         DIR/model_mtp_ternary.safetensors into");
    eprintln!("                         DIR/mtp-head-ptq1-v1.bin (no engine, no");
    eprintln!("                         checkpoint: head file only)");
    eprintln!("                       dflash-head=DIR");
    eprintln!("                         convert a DFlash2 drafter from");
    eprintln!("                         DIR/model.safetensors and DIR/config.json");
    eprintln!("                         into DIR/dflash2-drafter-int8-v1.bin");
    eprintln!("                         (per-row int8; no engine, no checkpoint)");
    eprintln!("                       dflash-head-int4=DIR");
    eprintln!("                         the same into DIR/dflash2-drafter-int4-v1.bin");
    eprintln!("                         (4-bit, a scale and bias per 64 columns)");
}

fn parse(args: &[String]) -> Result<Args, String> {
    let mut result = Args {
        max_tokens: crate::DEFAULT_MAX_OUTPUT_TOKENS,
        prompt: String::new(),
        prompt_file: None,
        raw: false,
        thinking: true,
        greedy: false,
        output: OutputMode::Stream,
    };
    let mut positional = Vec::new();
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--max-tokens" | "--prompt-file" => {
                let flag = args[index].as_str();
                index += 1;
                let value = args
                    .get(index)
                    .ok_or_else(|| format!("{flag} requires a value"))?;
                match flag {
                    "--prompt-file" => result.prompt_file = Some(PathBuf::from(value)),
                    _ => {
                        result.max_tokens = value.parse().map_err(|_| "invalid --max-tokens")?;
                    }
                }
            }
            "--raw" => result.raw = true,
            "--greedy" => result.greedy = true,
            "--json" => {
                if let OutputMode::Export(kind) = &result.output {
                    return Err(format!("--json is incompatible with {}", export_flag(kind)));
                }
                if !matches!(result.output, OutputMode::Tokenize) {
                    result.output = OutputMode::Json;
                }
            }
            "--tokenize" => {
                if let OutputMode::Export(kind) = &result.output {
                    return Err(format!(
                        "--tokenize is incompatible with {}",
                        export_flag(kind)
                    ));
                }
                result.output = OutputMode::Tokenize;
            }
            flag if flag == "--export" || flag.starts_with("--export=") => {
                let requested = match flag.strip_prefix("--export=") {
                    Some(kind) => kind,
                    None => {
                        index += 1;
                        args.get(index).filter(|value| !value.starts_with('-'))
                    }
                    .ok_or_else(|| format!("--export requires a kind ({EXPORT_KINDS})"))?,
                };
                let kind = parse_export(requested)?;
                let claimed = match &result.output {
                    OutputMode::Stream => None,
                    OutputMode::Json => Some("--json"),
                    OutputMode::Tokenize => Some("--tokenize"),
                    OutputMode::Export(previous) => Some(export_flag(previous)),
                };
                if let Some(claimed) = claimed {
                    return Err(format!(
                        "{} is incompatible with {claimed}",
                        export_flag(&kind)
                    ));
                }
                result.output = OutputMode::Export(kind);
            }
            "--help" | "-h" => return Err(String::new()),
            "--" => {
                positional.extend_from_slice(&args[index + 1..]);
                break;
            }
            flag if flag.starts_with('-') => return Err(format!("unknown option: {flag}")),
            value => positional.push(value.to_owned()),
        }
        index += 1;
    }
    result.thinking = !result.raw;
    if let OutputMode::Export(kind) = &result.output {
        if !positional.is_empty()
            || result.prompt_file.is_some()
            || result.raw
            || result.greedy
            || result.max_tokens != crate::DEFAULT_MAX_OUTPUT_TOKENS
        {
            return Err(format!("{} takes no other options", export_flag(kind)));
        }
    } else if positional.is_empty() == result.prompt_file.is_none() {
        return Err("supply a prompt or --prompt-file, not both".into());
    }
    result.prompt = positional.join(" ");
    Ok(result)
}

/// Writes the artifact an [`ExportKind`] names and returns. No kind opens
/// the engine: `index` checks the GGUF, `mtp-head` packs the head file alone,
/// and `dflash-head` converts the drafter checkpoint alone.
fn export(resources: &Resources, kind: &ExportKind) -> crate::Result<()> {
    match kind {
        ExportKind::Index => {
            let package = BonsaiPackage::open(&resources.model)?;
            validate_profile(&package)?;
            println!("{}", package.export_index(&resources.model)?);
        }
        ExportKind::MtpHead { dir } => {
            // No engine and no checkpoint: this packs the head file alone.
            println!("{}", json!(local_engine::export_head(dir)?));
        }
        ExportKind::DflashHead { dir, precision } => {
            println!(
                "{}",
                json!(local_engine::export_dflash_head(dir, *precision)?)
            );
        }
    }
    Ok(())
}

/// Speculation head and depth for the JSON record, when enabled.
fn mtp_summary(engine: &BonsaiEngine) -> Option<(serde_json::Value, serde_json::Value)> {
    let policy = &engine.info().policy["mtp"];
    engine
        .mtp_enabled()
        .then(|| (policy["head"].clone(), policy["depth"].clone()))
}

#[expect(clippy::too_many_lines)]
fn run(args: &Args) -> crate::Result<()> {
    // Exports and tokenizing load neither the MTP head nor the drafter (an
    // export may be writing one), so only generation requires them.
    let generates = matches!(args.output, OutputMode::Stream | OutputMode::Json);
    let resources = Resources::discover(None, generates)?;
    if let OutputMode::Export(kind) = &args.output {
        return export(&resources, kind);
    }
    let text = args
        .prompt_file
        .as_ref()
        .map_or_else(|| Ok(args.prompt.clone()), std::fs::read_to_string)?;
    if matches!(args.output, OutputMode::Tokenize) {
        let package = BonsaiPackage::open(&resources.model)?;
        let tokenizer = BonsaiTokenizer::from_package(&package)?;
        let rendered = if args.raw {
            text
        } else {
            BonsaiTokenizer::chat_prompt(&text, args.thinking)?
        };
        let tokens = tokenizer.encode(&rendered)?;
        println!(
            "{}",
            json!({"prompt": rendered, "prompt_tokens": tokens.len(), "token_ids": tokens})
        );
        return Ok(());
    }
    let started = Instant::now();
    let mut engine = BonsaiEngine::open_with_speculation(
        &resources.model,
        None,
        PREFILL_CHUNK,
        None,
        &resources.mtp,
        &resources.dflash,
        NgramSettings::default(),
        KvOptions::default(),
    )?;
    engine.configure_prompt_cache(
        resources.prompt_cache_dir.clone(),
        resources.disk_budget_bytes,
        resources.prompt_cache_write_bytes_per_second,
    )?;
    let load_seconds = started.elapsed().as_secs_f64();
    eprintln!(
        "{}",
        json!({"experimental_bonsai": {"engine": engine.info(), "policy": resources.policy()}, "load_seconds": load_seconds})
    );
    let prompt_ids = engine.encode_prompt(&text, args.raw, args.thinking)?;
    let params = GenerateParams {
        max_tokens: args.max_tokens,
        temperature: if args.greedy {
            0.0
        } else {
            crate::DEFAULT_TEMPERATURE
        },
        ..GenerateParams::default()
    };
    let output = engine.generate(&prompt_ids, &params, |piece| {
        if matches!(args.output, OutputMode::Json) {
            return true;
        }
        print!("{piece}");
        std::io::stdout().flush().is_ok()
    })?;
    let stats = &output.stats;
    let first_token = stats.first_token.map(|duration| duration.as_secs_f64());
    let decode_seconds = first_token.map(|first| stats.elapsed.as_secs_f64() - first);
    let decode_rate = decode_seconds
        .filter(|&seconds| seconds > 0.0 && stats.sampled_tokens > 1)
        .map(|seconds| (stats.sampled_tokens - 1) as f64 / seconds);
    let kflash = engine.kflash_enabled().then(|| {
        json!({
            "cycles": stats.kflash.cycles,
            "mtp_cycles": stats.kflash.mtp_cycles,
            "declined_trees": stats.kflash.declined_trees,
            "lookup_cycles": stats.kflash.lookup_cycles,
            "verified_rows": stats.kflash.verified_rows,
            "rows_histogram": stats.kflash.rows_histogram,
            "accepted_tokens": stats.kflash.accepted_tokens,
            "off_chain_cycles": stats.kflash.off_chain_cycles,
            "tokens_per_cycle": (stats.kflash.cycles > 0)
                .then(|| (stats.kflash.accepted_tokens + stats.kflash.cycles) as f64
                    / stats.kflash.cycles as f64),
            "expected_tokens_per_cycle": (stats.kflash.cycles > 0)
                .then(|| stats.kflash.expected_tokens / stats.kflash.cycles as f64),
            "catch_up_rows": stats.kflash.catch_up_rows,
            "catch_up_seconds": stats.kflash.catch_up.as_secs_f64(),
            "drafting_seconds": stats.kflash.drafting.as_secs_f64(),
            "verification_seconds": stats.kflash.verification.as_secs_f64(),
            "commit_seconds": stats.kflash.commit.as_secs_f64(),
        })
    });
    let record = json!({
        "context": engine.info().context,
        "prefill_chunk_size": engine.info().prefill_chunk_size,
        "prefill_buckets": engine.info().policy["prefill_buckets"],
        "attention_kernel": engine.attention_kernel(),
        "kv_cache": engine.info().policy["kv_cache"],
        "ngram_policy": engine.info().policy["ngram"],
        "model": resources.model, "raw": args.raw, "thinking": args.thinking,
        "sampling": {
            "temperature": params.temperature, "top_k": params.top_k,
            "top_p": params.top_p, "min_p": params.min_p, "seed": params.seed,
            "presence_penalty": params.presence_penalty,
            "repetition_penalty": params.repetition_penalty, "max_tokens": params.max_tokens,
        },
        "prompt_tokens": stats.prompt_tokens, "cached_tokens": stats.reused_prompt_tokens,
        "prefilled_tokens": stats.prompt_tokens - stats.reused_prompt_tokens,
        "generated_tokens": stats.generated_tokens,
        "sampled_tokens": stats.sampled_tokens, "stop_reason": output.stop_reason,
        "prefill_seconds": stats.prefill.as_secs_f64(),
        "prefill_tokens_per_second": (stats.prefill.as_secs_f64() > 0.0)
            .then(|| (stats.prompt_tokens - stats.reused_prompt_tokens) as f64
                / stats.prefill.as_secs_f64()),
        "first_token_seconds": first_token, "decode_seconds": decode_seconds,
        "decode_tokens_per_second": decode_rate,
        "sampling_seconds": stats.sampling.as_secs_f64(),
        "gpu_seconds": stats.gpu.as_secs_f64(),
        "elapsed_seconds": stats.elapsed.as_secs_f64(),
        "mtp": mtp_summary(&engine).map(|(head, depth)| json!({
            "head": head, "depth": depth,
            "rounds": stats.mtp.rounds,
            "proposed_tokens": stats.mtp.proposed_tokens,
            "accepted_tokens": stats.mtp.accepted_tokens,
            "verified_tokens": stats.mtp.verified_tokens,
            "acceptance_rate": (stats.mtp.proposed_tokens > 0)
                .then(|| stats.mtp.accepted_tokens as f64 / stats.mtp.proposed_tokens as f64),
            "drafting_seconds": stats.mtp.drafting.as_secs_f64(),
            "verification_seconds": stats.mtp.verification.as_secs_f64(),
            "commit_seconds": stats.mtp.commit.as_secs_f64(),
        })),
        "kflash": kflash,
        "ngram": {
            "rounds": stats.ngram.rounds,
            "proposed_tokens": stats.ngram.proposed_tokens,
            "accepted_tokens": stats.ngram.accepted_tokens,
            "acceptance_rate": (stats.ngram.proposed_tokens > 0)
                .then(|| stats.ngram.accepted_tokens as f64 / stats.ngram.proposed_tokens as f64),
            "lookup_seconds": stats.ngram.lookup.as_secs_f64(),
            "lookup_microseconds_per_sample": (stats.sampled_tokens > 0)
                .then(|| stats.ngram.lookup.as_secs_f64() * 1_000_000.0 / stats.sampled_tokens as f64),
        },
    });
    if matches!(args.output, OutputMode::Json) {
        let mut record = record;
        record["text"] = json!(output.text);
        record["final_text"] = if args.raw {
            serde_json::Value::Null
        } else {
            json!(engine.final_answer(&output.token_ids, args.thinking)?)
        };
        record["token_ids"] = json!(output.token_ids);
        record["prompt_token_ids"] = json!(prompt_ids);
        println!("{record}");
    } else {
        println!();
        eprintln!("{record}");
    }
    Ok(())
}

pub fn main_with_args(args: &[String]) -> ExitCode {
    match parse(args) {
        Ok(args) => match run(&args) {
            Ok(()) => ExitCode::SUCCESS,
            Err(error) => {
                eprintln!("error: {error}");
                ExitCode::FAILURE
            }
        },
        Err(error) => {
            if !error.is_empty() {
                eprintln!("error: {error}");
            }
            usage();
            if error.is_empty() {
                ExitCode::SUCCESS
            } else {
                ExitCode::FAILURE
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| (*value).into()).collect()
    }

    #[test]
    fn export_kinds_select_their_mode() {
        for (command, kind) in [
            ("--export index", ExportKind::Index),
            ("--export=index", ExportKind::Index),
            (
                "--export mtp-head=/tmp/head",
                ExportKind::MtpHead {
                    dir: PathBuf::from("/tmp/head"),
                },
            ),
            (
                "--export=mtp-head=/tmp/head",
                ExportKind::MtpHead {
                    dir: PathBuf::from("/tmp/head"),
                },
            ),
            (
                "--export dflash-head=/tmp/drafter",
                ExportKind::DflashHead {
                    dir: PathBuf::from("/tmp/drafter"),
                    precision: DflashPrecision::Int8,
                },
            ),
            (
                "--export dflash-head-int4=/tmp/drafter",
                ExportKind::DflashHead {
                    dir: PathBuf::from("/tmp/drafter"),
                    precision: DflashPrecision::Int4,
                },
            ),
        ] {
            let parsed = parse(&args(&command.split(' ').collect::<Vec<_>>())).expect(command);
            assert_eq!(parsed.output, OutputMode::Export(kind), "{command}");
        }
    }

    #[test]
    fn export_grammar_errors() {
        for (input, expected) in [
            (
                vec!["--export", "index=/tmp/head"],
                "--export index takes no directory",
            ),
            (
                vec!["--export", "mtp-head"],
                "--export mtp-head requires =DIR",
            ),
            (
                vec!["--export", "mtp-head="],
                "--export mtp-head requires a non-empty =DIR",
            ),
            (
                vec!["--export", "dflash-head"],
                "--export dflash-head requires =DIR",
            ),
            (
                vec!["--export", "dflash-head="],
                "--export dflash-head requires a non-empty =DIR",
            ),
            (
                vec!["--export", "dflash-head-int4"],
                "--export dflash-head-int4 requires =DIR",
            ),
            (
                vec!["--export", "mtp-head-zstd=/tmp/head"],
                "unknown --export kind: mtp-head-zstd (expected index, mtp-head=DIR, dflash-head=DIR, dflash-head-int4=DIR)",
            ),
            (
                vec!["--export", "mtp-head-ternary=/tmp/head"],
                "unknown --export kind: mtp-head-ternary (expected index, mtp-head=DIR, dflash-head=DIR, dflash-head-int4=DIR)",
            ),
            (
                vec!["--export", "foo"],
                "unknown --export kind: foo (expected index, mtp-head=DIR, dflash-head=DIR, dflash-head-int4=DIR)",
            ),
            (
                vec!["--export=foo"],
                "unknown --export kind: foo (expected index, mtp-head=DIR, dflash-head=DIR, dflash-head-int4=DIR)",
            ),
            (
                vec!["--export="],
                "--export requires a kind (index, mtp-head=DIR, dflash-head=DIR, dflash-head-int4=DIR)",
            ),
            (
                vec!["--export"],
                "--export requires a kind (index, mtp-head=DIR, dflash-head=DIR, dflash-head-int4=DIR)",
            ),
            (
                vec!["--export", "--json"],
                "--export requires a kind (index, mtp-head=DIR, dflash-head=DIR, dflash-head-int4=DIR)",
            ),
        ] {
            let error = parse(&args(&input)).expect_err(&format!("{input:?}"));
            assert_eq!(error, expected, "{input:?}");
        }
    }

    #[test]
    fn export_refuses_anything_that_decodes() {
        for (extra, expected) in [
            (vec!["hello"], "--export index takes no other options"),
            (
                vec!["--prompt-file", "/tmp/prompt.txt"],
                "--export index takes no other options",
            ),
            (
                vec!["--max-tokens", "16"],
                "--export index takes no other options",
            ),
            (vec!["--greedy"], "--export index takes no other options"),
            (vec!["--raw"], "--export index takes no other options"),
            (vec!["--json"], "--json is incompatible with --export index"),
            (
                vec!["--tokenize"],
                "--tokenize is incompatible with --export index",
            ),
        ] {
            let mut input = vec!["--export", "index"];
            input.extend(extra.iter().copied());
            let error = parse(&args(&input)).expect_err(&format!("{extra:?}"));
            assert_eq!(error, expected, "{extra:?}");
        }
    }

    #[test]
    fn export_is_incompatible_with_the_other_output_modes() {
        for (input, expected) in [
            (
                vec!["--json", "--export", "index"],
                "--export index is incompatible with --json",
            ),
            (
                vec!["--tokenize", "--export", "index"],
                "--export index is incompatible with --tokenize",
            ),
            (
                vec!["--export", "mtp-head=/tmp/head", "--export", "index"],
                "--export index is incompatible with --export mtp-head=DIR",
            ),
        ] {
            let error = parse(&args(&input)).expect_err(&format!("{input:?}"));
            assert_eq!(error, expected, "{input:?}");
        }
    }

    #[test]
    fn thinking_is_always_on_and_only_a_raw_prompt_drops_it() {
        assert!(parse(&args(&["hello"])).expect("default").thinking);
        let raw = parse(&args(&["--raw", "hello"])).expect("raw");
        assert!(!raw.thinking);
    }

    #[test]
    fn retired_and_model_selection_flags_are_unknown_options() {
        // One pinned model is discovered, never named, each automatic policy
        // replaced a tuning flag, and nothing can be turned off; all of them
        // are plain unknown options.
        for flag in [
            "--model",
            "--thinking",
            "--no-thinking",
            "--no-speculation",
            "--mtp-depth",
            "--max-queue",
            "--mtp",
            "--mtp-head",
            "--no-mtp",
            "--prefill-chunk",
            "--attention-kernel",
            "--kv-cache",
            "--kv-initial",
            "--ngram",
            "--no-ngram",
            "--ngram-max",
            "--ngram-min-match",
            "--prompt-cache-checkpoints",
            "--context",
            "--repeat",
        ] {
            assert_eq!(
                parse(&args(&[flag, "hello"])).expect_err(flag),
                format!("unknown option: {flag}"),
                "{flag}"
            );
        }
    }
}
