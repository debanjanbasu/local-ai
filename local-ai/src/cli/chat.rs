use std::io::Write as _;
use std::process::ExitCode;

use serde_json::json;

use crate::GenerateParams;
use crate::bonsai_model::BonsaiEngine;
use crate::bonsai_native::KvOptions;
use crate::bonsai_ngram::NgramSettings;
use crate::resources::{PREFILL_CHUNK, Resources};

#[derive(Debug)]
struct Args {
    max_tokens: Option<usize>,
    thinking: bool,
    raw: bool,
    greedy: bool,
    prompt: String,
}

fn usage() {
    eprintln!("Usage: local-ai chat [options] <prompt>");
    eprintln!();
    eprintln!("  --max-tokens N    Maximum generated tokens (default: 8192)");
    eprintln!("  --greedy          Disable sampling");
    eprintln!("  --raw             Skip the chat template (and so reasoning)");
}

fn parse(args: &[String]) -> Result<Args, String> {
    let mut max_tokens = None;
    let mut raw = false;
    let mut greedy = false;
    let mut prompt = Vec::new();
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--max-tokens" => {
                index += 1;
                let value = args.get(index).ok_or("--max-tokens requires a value")?;
                max_tokens = Some(value.parse().map_err(|_| "invalid --max-tokens")?);
            }
            "--raw" => raw = true,
            "--greedy" => greedy = true,
            "--help" | "-h" => return Err(String::new()),
            "--" => {
                prompt.extend_from_slice(&args[index + 1..]);
                break;
            }
            flag if flag.starts_with('-') => return Err(format!("unknown option: {flag}")),
            value => prompt.push(value.to_owned()),
        }
        index += 1;
    }
    if prompt.is_empty() {
        return Err("prompt is required".into());
    }
    Ok(Args {
        max_tokens,
        thinking: !raw,
        raw,
        greedy,
        prompt: prompt.join(" "),
    })
}

fn run(args: &Args) -> crate::Result<()> {
    let resources = Resources::discover(None, true)?;
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
    eprintln!(
        "{}",
        json!({"experimental_bonsai": {"engine": engine.info(), "policy": resources.policy()}})
    );
    let ids = engine.encode_prompt(&args.prompt, args.raw, !args.raw && args.thinking)?;
    let mut params = GenerateParams::default();
    if let Some(max_tokens) = args.max_tokens {
        params.max_tokens = max_tokens;
    }
    if args.greedy {
        params.temperature = 0.0;
    }
    let output = engine.generate(&ids, &params, |piece| {
        print!("{piece}");
        std::io::stdout().flush().is_ok()
    })?;
    println!();
    if output.text.is_empty() {
        Err(crate::Error::Generation("model produced no text".into()))
    } else {
        Ok(())
    }
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
    fn defaults_to_bonsai_and_parses_options() {
        let args = parse(&args(&["--greedy", "--max-tokens", "16", "hi"])).expect("options");
        assert!(args.thinking && args.greedy && !args.raw);
        assert_eq!(args.max_tokens, Some(16));
    }

    #[test]
    fn validates_option_bounds_and_prompt_modes() {
        for invalid in [
            vec!["--prefill-chunk", "0", "hi"],
            vec!["--context", "4096", "hi"],
            vec!["--kv-cache", "q8", "hi"],
            vec!["--kv-initial", "1024", "hi"],
            vec!["--attention-kernel", "simd", "hi"],
            vec!["--prompt-cache-checkpoints", "2", "hi"],
            vec!["--max-queue", "8", "hi"],
            vec!["--no-speculation", "hi"],
            vec!["--no-mtp", "hi"],
            vec!["--no-ngram", "hi"],
            vec!["--mtp-depth", "2", "hi"],
            vec!["--no-thinking", "hi"],
            vec!["--image", "cat.png", "hi"],
        ] {
            assert!(parse(&args(&invalid)).is_err(), "{invalid:?}");
        }
        let raw = parse(&args(&["--raw", "hi"])).expect("raw prompt");
        assert!(raw.raw && !raw.thinking);
        assert_eq!(
            parse(&args(&["--", "--literal"])).expect("literal").prompt,
            "--literal"
        );
    }

    #[test]
    fn thinking_is_not_a_flag_because_it_is_always_on() {
        for flag in ["--thinking", "--no-thinking"] {
            assert_eq!(
                parse(&args(&[flag, "hi"])).expect_err("rejected"),
                format!("unknown option: {flag}")
            );
        }
    }
}
