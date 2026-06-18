//! `compress` subcommand: convert a GGUF model directory into a self-describing
//! `.lma` archive (zstd-compressed, with the `config.json` embedded).
//!
//! Usage:
//!   local-ai compress --model <model-dir> [--out <path.lma>] [--level <1-22>]

use std::path::PathBuf;
use std::process::ExitCode;

use crate::gguf::GGUFModel;
use crate::lma::compress_gguf_to_lma;

/// Default GGUF weights filename inside a model directory.
const DEFAULT_GGUF: &str = "gemma-4-E2B-it-qat-UD-Q2_K_XL.gguf";
/// Default zstd compression level. Level 22 (max) gives the smallest archive —
/// the model is compressed once and shipped, so compression time is irrelevant.
const DEFAULT_LEVEL: i32 = 22;

struct Args {
    model: PathBuf,
    out: Option<PathBuf>,
    level: i32,
}

fn print_usage() {
    eprintln!("Usage: compress --model <model-dir> [--out <path.lma>] [--level <1-22>]");
    eprintln!();
    eprintln!("Options:");
    eprintln!("  --model <path>   Path to the model directory (required)");
    eprintln!("  --out <path>     Output .lma path (default: <model-dir>/model.lma)");
    eprintln!("  --level <N>      zstd compression level 1-22 (default: 22)");
    eprintln!("  --help           Show this help message");
}

fn parse_args(argv: &[String]) -> Option<Args> {
    if argv.is_empty() {
        print_usage();
        return None;
    }

    let mut model: Option<PathBuf> = None;
    let mut out: Option<PathBuf> = None;
    let mut level = DEFAULT_LEVEL;
    let mut i = 0;

    while i < argv.len() {
        match argv[i].as_str() {
            "--help" | "-h" => {
                print_usage();
                return None;
            }
            "--model" | "-m" => {
                i += 1;
                model = argv.get(i).map(PathBuf::from);
            }
            "--out" | "-o" => {
                i += 1;
                out = argv.get(i).map(PathBuf::from);
            }
            "--level" | "-l" => {
                i += 1;
                let Some(v) = argv.get(i).and_then(|v| v.parse::<i32>().ok()) else {
                    eprintln!("error: --level requires an integer");
                    return None;
                };
                level = v;
            }
            other => {
                eprintln!("error: unexpected argument {other:?}");
                print_usage();
                return None;
            }
        }
        i += 1;
    }

    let model = model.or_else(|| {
        eprintln!("error: --model is required");
        print_usage();
        None
    })?;

    Some(Args { model, out, level })
}

fn run(args: &Args) -> crate::Result<PathBuf> {
    let gguf_path = args.model.join(DEFAULT_GGUF);
    // Recompress mode: no GGUF source, but an existing archive — rewrite it
    // frame-by-frame at the requested level (byte-verified, atomic).
    let lma_path = args.model.join("model.lma");
    if !gguf_path.exists() && lma_path.exists() {
        eprintln!(
            "recompressing {} at zstd level {}",
            lma_path.display(),
            args.level
        );
        crate::lma::recompress_lma(&lma_path, args.level)?;
        return Ok(lma_path);
    }
    let gguf = GGUFModel::open(&gguf_path)?;

    let config_path = args.model.join("config.json");
    let config_json = std::fs::read_to_string(&config_path)
        .map_err(|e| crate::Error::Io(format!("{}: {e}", config_path.display())))?;

    // Bundle the multimodal companion and tokenizer when present, so the
    // archive is the only file the engine needs.
    let companion_path = args.model.join("mmproj-BF16.gguf");
    let companion = if companion_path.exists() {
        eprintln!("bundling multimodal companion {}", companion_path.display());
        Some(GGUFModel::open(&companion_path)?)
    } else {
        None
    };
    let tokenizer_json = std::fs::read(args.model.join("tokenizer.json")).ok();
    if tokenizer_json.is_some() {
        eprintln!("bundling tokenizer.json");
    }

    let out_path = args
        .out
        .clone()
        .unwrap_or_else(|| args.model.join("model.lma"));

    compress_gguf_to_lma(
        &gguf,
        &config_json,
        companion.as_ref(),
        tokenizer_json.as_deref(),
        &out_path,
        args.level,
    )?;
    Ok(out_path)
}

/// Entry point for the `compress` subcommand.
#[must_use]
pub fn main_with_args(argv: &[String]) -> ExitCode {
    let Some(parsed) = parse_args(argv) else {
        return ExitCode::FAILURE;
    };
    match run(&parsed) {
        Ok(out_path) => {
            eprintln!("wrote {}", out_path.display());
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("compress failed: {e}");
            ExitCode::FAILURE
        }
    }
}
