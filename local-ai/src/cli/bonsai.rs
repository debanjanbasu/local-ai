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

enum OutputMode {
    Stream,
    Json,
    Tokenize,
    ExportIndex,
    ExportMtpHead,
    ExportMtpHeadZstd,
}

/// Output modes that decode nothing: they are mutually exclusive, and with a
/// prompt, and with any sampling flag.
const fn is_export(output: &OutputMode) -> bool {
    matches!(
        output,
        OutputMode::ExportIndex | OutputMode::ExportMtpHead | OutputMode::ExportMtpHeadZstd
    )
}

/// The flag that selected an [`is_export`] mode, for its error messages.
const fn export_flag(output: &OutputMode) -> &'static str {
    match output {
        OutputMode::ExportMtpHead => "--export-mtp-head",
        OutputMode::ExportMtpHeadZstd => "--export-mtp-head-zstd",
        _ => "--export-index",
    }
}

#[allow(clippy::struct_excessive_bools)]
struct Args {
    model: PathBuf,
    max_tokens: usize,
    prompt: String,
    prompt_file: Option<PathBuf>,
    raw: bool,
    thinking: bool,
    greedy: bool,
    output: OutputMode,
    export_dir: Option<PathBuf>,
    model_explicit: bool,
    no_speculation: bool,
}

fn usage() {
    eprintln!("Usage: local-ai bonsai [options] <prompt>");
    eprintln!("Bonsai 2 27B text inference: native Metal straight from the GGUF.");
    eprintln!("  --model PATH       Override automatic pinned-model discovery");
    eprintln!("  --max-tokens N     Output cap (default: 8192)");
    eprintln!("  --prompt-file PATH Read a long UTF-8 prompt instead of positional text");
    eprintln!("  --raw              No chat template");
    eprintln!("  --thinking         Checkpoint's xhigh thinking (default)");
    eprintln!("  --no-thinking      Skip thinking; lower reasoning quality in our checks");
    eprintln!("  --greedy           Disable sampling for reference comparisons");
    eprintln!("  --json             Emit token IDs, stop reason, and measured timings");
    eprintln!("  --tokenize         Emit prompt token IDs without loading the model");
    eprintln!("  --export-index     Emit the checked GGUF index JSON");
    eprintln!("  --export-mtp-head DIR");
    eprintln!("                     Quantize the MTP head to DIR/mtp-head-int8-v2.bin");
    eprintln!("                     (no engine, no checkpoint: head file only)");
    eprintln!("  --export-mtp-head-zstd DIR");
    eprintln!("                     Same artifact, zstd level 19: 16.3% smaller on the");
    eprintln!("                     wire, byte-identical once inflated. Costs ~425 MB");
    eprintln!("                     of anonymous RAM at load, so the stored form is");
    eprintln!("                     what discovery prefers.");
    eprintln!("  --no-speculation   A/B baseline: disable MTP and suffix lookup");
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
        greedy: false,
        output: OutputMode::Stream,
        export_dir: None,
        model_explicit: false,
        no_speculation: false,
    };
    let mut positional = Vec::new();
    let mut thinking_override = None;
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
                | "--mtp-depth"
                | "--no-mtp"
                | "--ngram"
                | "--no-ngram"
                | "--ngram-max"
                | "--ngram-min-match"
                | "--prompt-cache-checkpoints"
                | "--context"
                | "--repeat"
        ) {
            return Err(format!("unknown option: {}", args[index]));
        }
        match args[index].as_str() {
            "--model"
            | "--max-tokens"
            | "--prompt-file"
            | "--export-mtp-head"
            | "--export-mtp-head-zstd" => {
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
                    "--max-tokens" => {
                        result.max_tokens = value.parse().map_err(|_| "invalid --max-tokens")?;
                    }
                    flag @ ("--export-mtp-head" | "--export-mtp-head-zstd") => {
                        let compressed = flag == "--export-mtp-head-zstd";
                        if !matches!(result.output, OutputMode::Stream) {
                            return Err(format!(
                                "{flag} is incompatible with \
                                 --json/--tokenize/--export-index/--export-mtp-head/--export-mtp-head-zstd"
                            ));
                        }
                        result.export_dir = Some(PathBuf::from(value));
                        result.output = if compressed {
                            OutputMode::ExportMtpHeadZstd
                        } else {
                            OutputMode::ExportMtpHead
                        };
                    }
                    _ => unreachable!(),
                }
            }
            "--raw" => result.raw = true,
            "--no-speculation" => result.no_speculation = true,
            "--thinking" => thinking_override = Some(true),
            "--no-thinking" => thinking_override = Some(false),
            "--greedy" => result.greedy = true,
            "--json" => {
                if is_export(&result.output) {
                    return Err(format!(
                        "--json is incompatible with {}",
                        export_flag(&result.output)
                    ));
                }
                if !matches!(result.output, OutputMode::Tokenize) {
                    result.output = OutputMode::Json;
                }
            }
            "--tokenize" => {
                if is_export(&result.output) {
                    return Err(format!(
                        "--tokenize is incompatible with {}",
                        export_flag(&result.output)
                    ));
                }
                result.output = OutputMode::Tokenize;
            }
            "--export-index" => {
                if !matches!(result.output, OutputMode::Stream) {
                    return Err(
                        "--export-index is incompatible with --json/--tokenize/--export-mtp-head"
                            .into(),
                    );
                }
                result.output = OutputMode::ExportIndex;
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
    if result.raw && thinking_override == Some(true) {
        return Err("--thinking requires the chat template, not --raw".into());
    }
    result.thinking = !result.raw && thinking_override.unwrap_or(true);
    if is_export(&result.output) {
        if !positional.is_empty()
            || result.prompt_file.is_some()
            || result.raw
            || thinking_override.is_some()
            || result.greedy
            || result.max_tokens != crate::DEFAULT_MAX_OUTPUT_TOKENS
        {
            return Err(format!(
                "{} accepts only --model",
                export_flag(&result.output)
            ));
        }
    } else if positional.is_empty() == result.prompt_file.is_none() {
        return Err("supply a prompt or --prompt-file, not both".into());
    }
    result.prompt = positional.join(" ");
    Ok(result)
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
    let resources = Resources::discover(
        args.model_explicit.then_some(args.model.as_path()),
        !args.no_speculation,
    )?;
    if matches!(args.output, OutputMode::ExportIndex) {
        let package = BonsaiPackage::open(&resources.model)?;
        validate_profile(&package)?;
        println!("{}", package.export_index(&resources.model)?);
        return Ok(());
    }
    if matches!(
        args.output,
        OutputMode::ExportMtpHead | OutputMode::ExportMtpHeadZstd
    ) {
        let directory = args.export_dir.as_deref().ok_or_else(|| {
            crate::Error::InvalidArgument(format!(
                "{} requires a directory",
                export_flag(&args.output)
            ))
        })?;
        let source = resources.mtp_source.as_deref().ok_or_else(|| {
            crate::Error::InvalidArgument(format!(
                "no BF16 MTP head found beside {}; exporting quantizes it",
                resources.model.display()
            ))
        })?;
        // No engine and no checkpoint: this transforms the head file alone.
        let artifact = if matches!(args.output, OutputMode::ExportMtpHeadZstd) {
            local_engine::export_head_zstd(source, directory)?
        } else {
            local_engine::export_head(source, directory)?
        };
        println!("{}", json!(artifact));
        return Ok(());
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
