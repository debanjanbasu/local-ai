//! Shared CLI implementation for the current stable top-level text target:
//! `gemma-4-e2b-qat`.
//!
//! Primary usage:
//!   cargo run -p local-engine --bin local-ai -- chat "Your prompt here"
//!
//! Compatibility wrapper:
//!   cargo run --example chat -- --model ~/path/to/model "Your prompt here"
//!
//! Notes:
//! - `Engine::new` accepts the currently supported top-level text targets.
//! - `generate_chat()` selects the family-appropriate prompt wrapper.

use std::path::PathBuf;
use std::process::ExitCode;

use crate::{Engine, EngineConfig, GenerateParams, MediaInput, MultimodalPrompt};

const DEFAULT_MODEL_DIR: &str = "models/gemma-4-e2b-it";

struct Args {
    model: PathBuf,
    max_tokens: Option<usize>,
    images: Vec<PathBuf>,
    audio: Vec<PathBuf>,
    videos: Vec<PathBuf>,
    prompt: String,
}

fn print_usage() {
    eprintln!(
        "Usage: chat [--model PATH] [--image PATH ...] [--audio PATH ...] [--video FRAME_OR_DIR ...] [--max-tokens N] <prompt>"
    );
    eprintln!();
    eprintln!("Options:");
    eprintln!("  --model <path>       Model directory (default: {DEFAULT_MODEL_DIR})");
    eprintln!("  --max-tokens <N>     Output length cap (default: model/context maximum)");
    eprintln!("  --image <path>       Add an image input (repeatable)");
    eprintln!("  --audio <path>       Add a WAV audio input (repeatable; PCM16/float32)");
    eprintln!(
        "  --video <path>       Add a sampled video input (repeatable; decoded frame image or frame directory)"
    );
    eprintln!("  --help               Show this help message");
    eprintln!();
    eprintln!(
        "The CLI auto-selects max context, model/context-max output, quality sampling, KV cache, and prompt caching for best default quality/performance."
    );
}

#[allow(clippy::too_many_lines)]
fn parse_args(argv: &[String]) -> Option<Args> {
    let arg_vec: Vec<String> = argv.to_vec();

    if arg_vec.is_empty() {
        print_usage();
        return None;
    }

    let mut model: PathBuf = PathBuf::from(DEFAULT_MODEL_DIR);
    let mut max_tokens: Option<usize> = None;
    let mut images = Vec::new();
    let mut audio = Vec::new();
    let mut videos = Vec::new();
    let mut prompt_parts: Vec<String> = Vec::new();
    let mut i = 0;

    while i < arg_vec.len() {
        match arg_vec[i].as_str() {
            "--help" | "-h" => {
                print_usage();
                return None;
            }
            "--model" | "-m" => {
                i += 1;
                let Some(val) = arg_vec.get(i) else {
                    eprintln!("Missing value for --model");
                    return None;
                };
                model = PathBuf::from(val);
            }
            "--max-tokens" | "-n" => {
                i += 1;
                let Some(val) = arg_vec.get(i) else {
                    eprintln!("Missing value for --max-tokens");
                    return None;
                };
                max_tokens = match val.parse() {
                    Ok(n) => Some(n),
                    Err(e) => {
                        eprintln!("Invalid --max-tokens value: {e}");
                        return None;
                    }
                };
            }
            // Kept as hidden compatibility aliases for scripts; new users should
            // rely on the engine defaults instead of hand-tuning these knobs.
            "--temperature" | "-t" => {
                i += 1;
                if arg_vec
                    .get(i)
                    .and_then(|val| val.parse::<f32>().ok())
                    .is_none()
                {
                    eprintln!("Invalid or missing --temperature value");
                    return None;
                }
                eprintln!("Note: --temperature is deprecated in the CLI; using engine defaults.");
            }
            "--context" => {
                i += 1;
                if arg_vec
                    .get(i)
                    .and_then(|val| val.parse::<usize>().ok())
                    .is_none()
                {
                    eprintln!("Invalid or missing --context value");
                    return None;
                }
                eprintln!("Note: --context is deprecated in the CLI; context is auto-sized.");
            }
            "--boundary-handoff" => {
                eprintln!("Note: --boundary-handoff is no longer needed; handoff is automatic.");
            }
            "--image" => {
                i += 1;
                let Some(val) = arg_vec.get(i) else {
                    eprintln!("Missing value for --image");
                    return None;
                };
                images.push(PathBuf::from(val));
            }
            "--audio" => {
                i += 1;
                let Some(val) = arg_vec.get(i) else {
                    eprintln!("Missing value for --audio");
                    return None;
                };
                audio.push(PathBuf::from(val));
            }
            "--video" => {
                i += 1;
                let Some(val) = arg_vec.get(i) else {
                    eprintln!("Missing value for --video");
                    return None;
                };
                videos.push(PathBuf::from(val));
            }
            other => {
                if other.starts_with('-') {
                    eprintln!("Unknown option: {other}");
                    print_usage();
                    return None;
                }
                prompt_parts.push(other.to_owned());
            }
        }
        i += 1;
    }

    if prompt_parts.is_empty() {
        eprintln!("Error: prompt argument is required");
        print_usage();
        return None;
    }

    Some(Args {
        model,
        max_tokens,
        images,
        audio,
        videos,
        prompt: prompt_parts.join(" "),
    })
}

fn run(arg_vec: Args) -> std::result::Result<(), Box<dyn std::error::Error>> {
    let config = EngineConfig {
        model_dir: arg_vec.model,
        max_context_length: 0,
    };

    let mut engine = Engine::new(&config)?;

    let params = arg_vec
        .max_tokens
        .map_or_else(GenerateParams::default, |max_tokens| GenerateParams {
            max_tokens,
            ..GenerateParams::default()
        });

    if let Some(max_tokens) = arg_vec.max_tokens {
        eprintln!("Generating (max {max_tokens} tokens)...");
    } else {
        eprintln!("Generating (model/context max output)...");
    }

    let output =
        if arg_vec.images.is_empty() && arg_vec.audio.is_empty() && arg_vec.videos.is_empty() {
            engine.generate_chat(&arg_vec.prompt, &params)?
        } else {
            let media = arg_vec
                .images
                .into_iter()
                .map(|path| MediaInput::Image { path })
                .chain(
                    arg_vec
                        .audio
                        .into_iter()
                        .map(|path| MediaInput::Audio { path }),
                )
                .chain(
                    arg_vec
                        .videos
                        .into_iter()
                        .map(|path| MediaInput::Video { path }),
                )
                .collect();
            engine.generate_multimodal_chat(
                &MultimodalPrompt {
                    text: arg_vec.prompt,
                    media,
                },
                &params,
            )?
        };
    println!("{output}");

    Ok(())
}

#[must_use]
pub fn main_with_args(argv: &[String]) -> ExitCode {
    let requested_help = argv
        .iter()
        .any(|arg| matches!(arg.as_str(), "--help" | "-h"));
    let Some(opts) = parse_args(argv) else {
        return if requested_help {
            ExitCode::SUCCESS
        } else {
            ExitCode::FAILURE
        };
    };

    if let Err(e) = run(opts) {
        eprintln!("Error: {e}");
        return ExitCode::FAILURE;
    }

    ExitCode::SUCCESS
}

#[must_use]
pub fn main_from_env() -> ExitCode {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    main_with_args(&argv)
}
