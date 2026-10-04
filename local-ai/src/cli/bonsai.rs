use std::io::Write as _;
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use serde_json::json;

use crate::GenerateParams;
use crate::bonsai::{BonsaiPackage, DEFAULT_BONSAI_GGUF, validate_profile};
use crate::bonsai_model::BonsaiEngine;
use crate::bonsai_native::KvOptions;
use crate::bonsai_ngram::NgramSettings;
use crate::bonsai_tokenizer::BonsaiTokenizer;
use crate::resources::{PREFILL_CHUNK, Resources};
use crate::{DEFAULT_MTP_DEPTH, MAX_MTP_DEPTH};

/// What `--export` writes instead of generating. The variants are the kinds as
/// the command line names them, and the directory an export writes belongs to
/// its kind, so `index` cannot be given one.
#[derive(Debug, PartialEq, Eq)]
enum ExportKind {
    Index,
    MtpHead { dir: PathBuf, zstd: bool },
}

#[derive(Debug, PartialEq, Eq)]
enum OutputMode {
    Stream,
    Json,
    Tokenize,
    Export(ExportKind),
}

/// The kinds `--export` accepts, for every message that has to name them.
const EXPORT_KINDS: &str = "index, mtp-head=DIR, mtp-head-zstd=DIR";

/// How an [`OutputMode::Export`] is spelled, for its error messages.
const fn export_flag(kind: &ExportKind) -> &'static str {
    match kind {
        ExportKind::Index => "--export index",
        ExportKind::MtpHead { zstd: false, .. } => "--export mtp-head=DIR",
        ExportKind::MtpHead { zstd: true, .. } => "--export mtp-head-zstd=DIR",
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
        (kind @ ("mtp-head" | "mtp-head-zstd"), None) => {
            Err(format!("--export {kind} requires =DIR"))
        }
        (kind @ ("mtp-head" | "mtp-head-zstd"), Some("")) => {
            Err(format!("--export {kind} requires a non-empty =DIR"))
        }
        (kind @ ("mtp-head" | "mtp-head-zstd"), Some(dir)) => Ok(ExportKind::MtpHead {
            dir: PathBuf::from(dir),
            zstd: kind == "mtp-head-zstd",
        }),
        (kind, _) => Err(format!(
            "unknown --export kind: {kind} (expected {EXPORT_KINDS})"
        )),
    }
}

#[allow(clippy::struct_excessive_bools)]
#[derive(Debug)]
struct Args {
    model: PathBuf,
    max_tokens: usize,
    prompt: String,
    prompt_file: Option<PathBuf>,
    raw: bool,
    thinking: bool,
    no_thinking: bool,
    greedy: bool,
    output: OutputMode,
    model_explicit: bool,
    no_speculation: bool,
    mtp_depth: usize,
}

fn usage() {
    eprintln!("Usage: local-ai bonsai [options] <prompt>");
    eprintln!("Bonsai 2 27B text inference: native Metal straight from the GGUF.");
    eprintln!("  --model PATH       Override automatic pinned-model discovery");
    eprintln!("  --max-tokens N     Output cap (default: 8192)");
    eprintln!("  --prompt-file PATH Read a long UTF-8 prompt instead of positional text");
    eprintln!("  --raw              No chat template");
    eprintln!("  --no-thinking      Skip the checkpoint's xhigh reasoning (default: on);");
    eprintln!("                     lower reasoning quality in our checks");
    eprintln!("  --greedy           Disable sampling for reference comparisons");
    eprintln!("  --json             Emit token IDs, stop reason, and measured timings");
    eprintln!("  --tokenize         Emit prompt token IDs without loading the model");
    eprintln!("  --export KIND      Write an artifact instead of generating; accepts");
    eprintln!("                     only --model. Kinds:");
    eprintln!("                       index");
    eprintln!("                         emit the checked GGUF index JSON on stdout");
    eprintln!("                       mtp-head=DIR");
    eprintln!("                         quantize the MTP head to DIR/mtp-head-int8-v2.bin");
    eprintln!("                         (no engine, no checkpoint: head file only)");
    eprintln!("                       mtp-head-zstd=DIR");
    eprintln!("                         same artifact, zstd level 19: 16.3% smaller on the");
    eprintln!("                         wire, byte-identical once inflated. Costs ~425 MB");
    eprintln!("                         of anonymous RAM at load, so the stored form is");
    eprintln!("                         what discovery prefers.");
    eprintln!("  --no-speculation   A/B baseline: disable MTP and suffix lookup");
    eprintln!(
        "  --mtp-depth N      Drafts per speculative round (1-{MAX_MTP_DEPTH}, \
         default: {DEFAULT_MTP_DEPTH}); measures what the confidence gate leaves \
         unused, not a tuned default"
    );
}

#[allow(clippy::too_many_lines)]
fn parse(args: &[String]) -> Result<Args, String> {
    let mut result = Args {
        model: PathBuf::from(DEFAULT_BONSAI_GGUF),
        max_tokens: crate::DEFAULT_MAX_OUTPUT_TOKENS,
        prompt: String::new(),
        prompt_file: None,
        raw: false,
        thinking: true,
        no_thinking: false,
        greedy: false,
        output: OutputMode::Stream,
        model_explicit: false,
        no_speculation: false,
        mtp_depth: DEFAULT_MTP_DEPTH,
    };
    let mut positional = Vec::new();
    let mut index = 0;
    while index < args.len() {
        if matches!(
            args[index].as_str(),
            "--prefill-chunk"
                | "--attention-kernel"
                | "--kv-cache"
                | "--kv-initial"
                | "--mtp"
                | "--mtp-head"
                | "--no-mtp"
                | "--ngram"
                | "--no-ngram"
                | "--ngram-max"
                | "--ngram-min-match"
                | "--prompt-cache-checkpoints"
                | "--context"
                | "--repeat"
        ) {
            // A flag that moved is told where its intent went: the flag that
            // replaced it, or the default that absorbed it. The rest keep the
            // bare rejection, because there is nothing better to say.
            let flag = args[index].as_str();
            return Err(match flag {
                "--mtp" => format!(
                    "unknown option: {flag} (no replacement: MTP is on by default, and \
                     --no-speculation opts out)"
                ),
                "--mtp-head" => format!(
                    "unknown option: {flag} (no replacement flag: the head is discovered \
                     beside the model, and --export mtp-head=DIR writes a quantized one)"
                ),
                "--no-mtp" => format!(
                    "unknown option: {flag} (replaced by --no-speculation, which disables \
                     MTP and suffix lookup)"
                ),
                flag => format!("unknown option: {flag}"),
            });
        }
        match args[index].as_str() {
            "--model" | "--max-tokens" | "--mtp-depth" | "--prompt-file" => {
                let flag = args[index].as_str();
                index += 1;
                let value = args
                    .get(index)
                    .ok_or_else(|| format!("{flag} requires a value"))?;
                match flag {
                    "--model" => {
                        result.model = PathBuf::from(value);
                        result.model_explicit = true;
                    }
                    "--prompt-file" => result.prompt_file = Some(PathBuf::from(value)),
                    "--mtp-depth" => result.mtp_depth = parse_mtp_depth(value)?,
                    _ => {
                        result.max_tokens = value.parse().map_err(|_| "invalid --max-tokens")?;
                    }
                }
            }
            "--raw" => result.raw = true,
            "--no-speculation" => result.no_speculation = true,
            "--no-thinking" => result.no_thinking = true,
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
    result.thinking = !result.raw && !result.no_thinking;
    if let OutputMode::Export(kind) = &result.output {
        if !positional.is_empty()
            || result.prompt_file.is_some()
            || result.raw
            || result.no_thinking
            || result.greedy
            || result.max_tokens != crate::DEFAULT_MAX_OUTPUT_TOKENS
            || result.mtp_depth != DEFAULT_MTP_DEPTH
        {
            return Err(format!("{} accepts only --model", export_flag(kind)));
        }
    } else if positional.is_empty() == result.prompt_file.is_none() {
        return Err("supply a prompt or --prompt-file, not both".into());
    }
    result.prompt = positional.join(" ");
    Ok(result)
}

/// Parse a `--mtp-depth` value as a draft depth inside the engine's range.
///
/// The range is the engine's, not a second copy of it: a depth the MTP
/// settings would refuse is refused at the flag too, so an unusable depth is
/// named by the person who passed it instead of decoding at a depth nobody
/// asked for.
fn parse_mtp_depth(value: &str) -> Result<usize, String> {
    let depth: usize = value
        .parse()
        .map_err(|_| format!("invalid --mtp-depth: {value:?} is not a whole number"))?;
    if !(1..=MAX_MTP_DEPTH).contains(&depth) {
        return Err(format!(
            "--mtp-depth must be between 1 and {MAX_MTP_DEPTH}, not {depth}"
        ));
    }
    Ok(depth)
}

/// Writes the artifact an [`ExportKind`] names and returns. Neither kind opens
/// the engine: `index` checks the GGUF, and the head kinds transform the head
/// file alone.
fn export(resources: &Resources, kind: &ExportKind) -> crate::Result<()> {
    match kind {
        ExportKind::Index => {
            let package = BonsaiPackage::open(&resources.model)?;
            validate_profile(&package)?;
            println!("{}", package.export_index(&resources.model)?);
        }
        ExportKind::MtpHead { dir, zstd } => {
            let source = resources.mtp_source.as_deref().ok_or_else(|| {
                crate::Error::InvalidArgument(format!(
                    "no BF16 MTP head found beside {}; exporting quantizes it",
                    resources.model.display()
                ))
            })?;
            // No engine and no checkpoint: this transforms the head file alone.
            let artifact = if *zstd {
                local_engine::export_head_zstd(source, dir)?
            } else {
                local_engine::export_head(source, dir)?
            };
            println!("{}", json!(artifact));
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

#[allow(clippy::too_many_lines)]
fn run(args: &Args) -> crate::Result<()> {
    let resources = Resources::discover_with_depth(
        args.model_explicit.then_some(args.model.as_path()),
        !args.no_speculation,
        args.mtp_depth,
    )?;
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
    let mut engine = BonsaiEngine::open_with_options(
        &resources.model,
        None,
        PREFILL_CHUNK,
        None,
        &resources.mtp,
        if args.no_speculation {
            NgramSettings {
                enabled: false,
                ..NgramSettings::default()
            }
        } else {
            NgramSettings::default()
        },
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
#[allow(clippy::expect_used)]
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
                    zstd: false,
                },
            ),
            (
                "--export mtp-head-zstd=/tmp/head",
                ExportKind::MtpHead {
                    dir: PathBuf::from("/tmp/head"),
                    zstd: true,
                },
            ),
            (
                "--export=mtp-head-zstd=/tmp/head",
                ExportKind::MtpHead {
                    dir: PathBuf::from("/tmp/head"),
                    zstd: true,
                },
            ),
        ] {
            let parsed = parse(&args(&command.split(' ').collect::<Vec<_>>())).expect(command);
            assert_eq!(parsed.output, OutputMode::Export(kind), "{command}");
        }
        let with_model =
            parse(&args(&["--export", "index", "--model", "/tmp/model.gguf"])).expect("model");
        assert_eq!(with_model.output, OutputMode::Export(ExportKind::Index));
        assert!(with_model.model_explicit);
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
                vec!["--export", "mtp-head-zstd"],
                "--export mtp-head-zstd requires =DIR",
            ),
            (
                vec!["--export", "foo"],
                "unknown --export kind: foo (expected index, mtp-head=DIR, mtp-head-zstd=DIR)",
            ),
            (
                vec!["--export=foo"],
                "unknown --export kind: foo (expected index, mtp-head=DIR, mtp-head-zstd=DIR)",
            ),
            (
                vec!["--export="],
                "--export requires a kind (index, mtp-head=DIR, mtp-head-zstd=DIR)",
            ),
            (
                vec!["--export"],
                "--export requires a kind (index, mtp-head=DIR, mtp-head-zstd=DIR)",
            ),
            (
                vec!["--export", "--json"],
                "--export requires a kind (index, mtp-head=DIR, mtp-head-zstd=DIR)",
            ),
        ] {
            let error = parse(&args(&input)).expect_err(&format!("{input:?}"));
            assert_eq!(error, expected, "{input:?}");
        }
    }

    #[test]
    fn export_refuses_anything_that_decodes() {
        for (extra, expected) in [
            (vec!["hello"], "--export index accepts only --model"),
            (
                vec!["--prompt-file", "/tmp/prompt.txt"],
                "--export index accepts only --model",
            ),
            (
                vec!["--max-tokens", "16"],
                "--export index accepts only --model",
            ),
            (vec!["--greedy"], "--export index accepts only --model"),
            (vec!["--raw"], "--export index accepts only --model"),
            (vec!["--no-thinking"], "--export index accepts only --model"),
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
    fn mtp_depth_defaults_to_the_shipped_depth() {
        assert_eq!(
            parse(&args(&["hello"])).expect("unset").mtp_depth,
            DEFAULT_MTP_DEPTH
        );
    }

    #[test]
    fn mtp_depth_accepts_every_depth_the_engine_accepts() {
        for depth in 1..=MAX_MTP_DEPTH {
            let flag = depth.to_string();
            let parsed = parse(&args(&["--mtp-depth", &flag, "hello"])).expect(&flag);
            assert_eq!(parsed.mtp_depth, depth, "{flag}");
        }
    }

    #[test]
    fn mtp_depth_rejects_a_non_integer() {
        for value in ["2.5", "-1", "three", ""] {
            let error = parse(&args(&["--mtp-depth", value, "hello"])).expect_err(value);
            assert_eq!(
                error,
                format!("invalid --mtp-depth: {value:?} is not a whole number")
            );
        }
    }

    #[test]
    fn mtp_depth_rejects_zero_and_anything_above_the_maximum() {
        for depth in [0, MAX_MTP_DEPTH + 1, 99] {
            let flag = depth.to_string();
            let error = parse(&args(&["--mtp-depth", &flag, "hello"])).expect_err(&flag);
            assert_eq!(
                error,
                format!("--mtp-depth must be between 1 and {MAX_MTP_DEPTH}, not {depth}")
            );
        }
    }

    #[test]
    fn export_refuses_a_depth_it_cannot_honour() {
        let error = parse(&args(&["--export", "index", "--mtp-depth", "2"])).expect_err("depth");
        assert_eq!(error, "--export index accepts only --model");
    }

    #[test]
    fn thinking_is_not_a_flag_because_it_already_is_the_default() {
        assert_eq!(
            parse(&args(&["--thinking", "hello"])).expect_err("rejected"),
            "unknown option: --thinking"
        );
        let raw = parse(&args(&["--raw", "--no-thinking", "hello"])).expect("raw");
        assert!(!raw.thinking);
    }

    #[test]
    fn a_replaced_mtp_flag_names_the_flag_that_replaced_it() {
        assert_eq!(
            parse(&args(&["--no-mtp", "hello"])).expect_err("rejected"),
            "unknown option: --no-mtp (replaced by --no-speculation, which disables \
             MTP and suffix lookup)"
        );
    }

    #[test]
    fn a_rejected_mtp_flag_that_a_default_absorbed_says_so() {
        for (flag, hint) in [
            (
                "--mtp",
                "unknown option: --mtp (no replacement: MTP is on by default, and \
                 --no-speculation opts out)",
            ),
            (
                "--mtp-head",
                "unknown option: --mtp-head (no replacement flag: the head is discovered \
                 beside the model, and --export mtp-head=DIR writes a quantized one)",
            ),
        ] {
            assert_eq!(
                parse(&args(&[flag, "hello"])).expect_err(flag),
                hint,
                "{flag}"
            );
        }
    }

    #[test]
    fn a_rejected_flag_with_nothing_to_point_at_keeps_the_bare_message() {
        for flag in [
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
