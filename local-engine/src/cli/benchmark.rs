use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use crate::tokenizer::Tokenizer;
use crate::{
    DecodedRgbImage, DecodedVideoFrame, Engine, EngineConfig, GenerateParams, MediaInput,
    MultimodalPrompt, PcmAudio,
};

const DEFAULT_MODEL_DIR: &str = "models/gemma-4-e2b-it";

#[allow(clippy::struct_excessive_bools)]
struct Args {
    model: PathBuf,
    tokens: usize,
    prompt: String,
    raw: bool,
    greedy: bool,
    suite: bool,
    cache_suite: bool,
    multimodal_suite: bool,
}

struct BenchCase {
    name: &'static str,
    prompt: &'static str,
    tokens: usize,
}

struct BenchResult {
    name: &'static str,
    seconds: f64,
    output_tokens: usize,
    tokens_per_second: f64,
}

struct MultimodalBenchCase {
    name: &'static str,
    prompt: &'static str,
    tokens: usize,
    media: fn() -> Vec<MediaInput>,
}

const BENCH_SUITE: &[BenchCase] = &[
    BenchCase {
        name: "short_factual",
        prompt: "Answer in one paragraph: why is the sky blue?",
        tokens: 64,
    },
    BenchCase {
        name: "reasoning",
        prompt: "A train leaves at 08:10 and travels for 2 hours 35 minutes. It waits 18 minutes, then travels another 47 minutes. What time does it arrive? Explain briefly.",
        tokens: 96,
    },
    BenchCase {
        name: "long_instruction",
        prompt: "Write a practical checklist for preparing an iPhone app to download and run a large on-device multimodal model. Cover installation, verification, media decoding, inference, cancellation, and error handling.",
        tokens: 192,
    },
    BenchCase {
        name: "code_explanation",
        prompt: "Explain how to implement an LRU cache in Rust. Include the main data structures, ownership concerns, and a small pseudocode sketch.",
        tokens: 160,
    },
];

const CACHE_SUITE: &[BenchCase] = &[
    BenchCase {
        name: "a_miss",
        prompt: "You are helping build an iOS app that downloads a large on-device multimodal model. Summarize the install-time steps for model download, checksum verification, archive unpacking, Core ML preparation, and first-run readiness.",
        tokens: 96,
    },
    BenchCase {
        name: "b_miss",
        prompt: "You are helping build an iOS app that runs a local multimodal model. Summarize the runtime steps for image decoding, video frame sampling, audio PCM handoff, inference cancellation, and user-visible errors.",
        tokens: 96,
    },
    BenchCase {
        name: "a_lru_hit",
        prompt: "You are helping build an iOS app that downloads a large on-device multimodal model. Summarize the install-time steps for model download, checksum verification, archive unpacking, Core ML preparation, and first-run readiness.",
        tokens: 96,
    },
    BenchCase {
        name: "a_shared_prefix",
        prompt: "You are helping build an iOS app that downloads a large on-device multimodal model. Summarize the install-time steps for model download, checksum verification, archive unpacking, Core ML preparation, first-run readiness, retry UI, and background task scheduling.",
        tokens: 96,
    },
];

const MULTIMODAL_SUITE: &[MultimodalBenchCase] = &[
    MultimodalBenchCase {
        name: "image_decoded",
        prompt: "Describe the image. Mention the dominant colors and visible pattern.",
        tokens: 96,
        media: image_media,
    },
    MultimodalBenchCase {
        name: "image_decoded_cache_hit",
        prompt: "Look at the same image again. Answer with a concise caption.",
        tokens: 96,
        media: image_media,
    },
    MultimodalBenchCase {
        name: "audio_pcm",
        prompt: "Describe the audio signal in one short paragraph.",
        tokens: 96,
        media: audio_media,
    },
    MultimodalBenchCase {
        name: "video_decoded_frames",
        prompt: "Describe how the simple video changes over time.",
        tokens: 96,
        media: video_media,
    },
];

fn synthetic_image(phase: u8) -> DecodedRgbImage {
    let width = 224;
    let height = 224;
    let mut rgb = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        for x in 0..width {
            let stripe = ((x / 28 + y / 28 + usize::from(phase)) % 2) as u8;
            if stripe == 0 {
                rgb.extend_from_slice(&[230, 48u8.saturating_add(phase * 20), 64]);
            } else {
                rgb.extend_from_slice(&[32, 112, 220u8.saturating_sub(phase * 12)]);
            }
        }
    }
    DecodedRgbImage {
        width: width as u32,
        height: height as u32,
        rgb,
    }
}

fn synthetic_audio() -> PcmAudio {
    let sample_rate = 16_000;
    let seconds = 2;
    let sample_count = sample_rate * seconds;
    let mut samples = Vec::with_capacity(sample_count as usize);
    for idx in 0..sample_count {
        let t = idx as f32 / sample_rate as f32;
        let envelope = (1.0 - t / seconds as f32).max(0.2);
        let tone_a = (2.0 * std::f32::consts::PI * 440.0 * t).sin();
        let tone_b = (2.0 * std::f32::consts::PI * 660.0 * t).sin();
        samples.push(0.2f32.mul_add(tone_b, 0.35 * tone_a) * envelope);
    }
    PcmAudio {
        samples,
        sample_rate,
        channels: 1,
    }
}

fn image_media() -> Vec<MediaInput> {
    vec![MediaInput::DecodedImage {
        image: synthetic_image(0),
    }]
}

fn audio_media() -> Vec<MediaInput> {
    vec![MediaInput::PcmAudio {
        audio: synthetic_audio(),
    }]
}

fn video_media() -> Vec<MediaInput> {
    let frames = (0..4)
        .map(|idx| DecodedVideoFrame {
            image: synthetic_image(idx as u8),
            timestamp_seconds: idx,
        })
        .collect();
    vec![MediaInput::DecodedVideo { frames }]
}

fn print_usage() {
    eprintln!(
        "Usage: benchmark [--model PATH] [--tokens N] [--prompt TEXT] [--raw] [--greedy] [--suite] [--cache-suite] [--multimodal-suite]"
    );
    eprintln!();
    eprintln!("Options:");
    eprintln!("  --model <path>   Model directory (default: {DEFAULT_MODEL_DIR})");
    eprintln!("  --tokens <n>     Max tokens to generate (default: 128)");
    eprintln!("  --prompt <text>  Prompt to use (default: built-in)");
    eprintln!("  --raw            Benchmark raw continuation instead of chat templating");
    eprintln!("  --greedy         Use deterministic greedy sampling instead of quality defaults");
    eprintln!("  --suite          Run the built-in multi-prompt default-path suite");
    eprintln!("  --cache-suite    Run prompt-cache miss/hit/shared-prefix cases");
    eprintln!("  --multimodal-suite");
    eprintln!("                   Run deterministic in-memory image/audio/video cases");
    eprintln!("  --help           Show this help message");
}

fn parse_arg(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|arg| arg == flag)
        .and_then(|idx| args.get(idx + 1))
        .cloned()
}

fn parse_args(argv: &[String]) -> Option<Args> {
    if argv
        .iter()
        .any(|arg| matches!(arg.as_str(), "--help" | "-h"))
    {
        print_usage();
        return None;
    }

    let model =
        parse_arg(argv, "--model").map_or_else(|| PathBuf::from(DEFAULT_MODEL_DIR), PathBuf::from);

    let tokens = parse_arg(argv, "--tokens")
        .and_then(|value| value.parse().ok())
        .unwrap_or(128);
    let prompt = parse_arg(argv, "--prompt")
        .unwrap_or_else(|| "Explain the theory of general relativity in simple terms.".to_owned());
    let raw = argv.iter().any(|arg| arg == "--raw");
    let greedy = argv.iter().any(|arg| arg == "--greedy");
    let suite = argv.iter().any(|arg| arg == "--suite");
    let cache_suite = argv.iter().any(|arg| arg == "--cache-suite");
    let multimodal_suite = argv.iter().any(|arg| arg == "--multimodal-suite");

    Some(Args {
        model,
        tokens,
        prompt,
        raw,
        greedy,
        suite,
        cache_suite,
        multimodal_suite,
    })
}

fn count_output_tokens(model_dir: &std::path::Path, output: &str) -> usize {
    Tokenizer::from_model_dir(model_dir)
        .and_then(|tokenizer| tokenizer.encode(output, false))
        .map_or_else(|_| output.split_whitespace().count(), |tokens| tokens.len())
}

fn generate_once(
    engine: &mut Engine,
    model_dir: &std::path::Path,
    prompt: &str,
    tokens: usize,
    raw: bool,
    greedy: bool,
) -> Result<(String, BenchResult), String> {
    let mut params = GenerateParams {
        max_tokens: tokens,
        ..GenerateParams::default()
    };
    if greedy {
        params.temperature = 0.0;
    }

    let gen_start = Instant::now();
    let output = if raw {
        engine.generate(prompt, &params)
    } else {
        engine.generate_chat(prompt, &params)
    }
    .map_err(|err| format!("Generation failed: {err}"))?;
    let seconds = gen_start.elapsed().as_secs_f64();
    let output_tokens = count_output_tokens(model_dir, &output);
    let tokens_per_second = if seconds > 0.0 {
        output_tokens as f64 / seconds
    } else {
        0.0
    };
    Ok((
        output,
        BenchResult {
            name: "single",
            seconds,
            output_tokens,
            tokens_per_second,
        },
    ))
}

fn generate_multimodal_once(
    engine: &mut Engine,
    model_dir: &std::path::Path,
    case: &MultimodalBenchCase,
    greedy: bool,
) -> Result<BenchResult, String> {
    let mut params = GenerateParams {
        max_tokens: case.tokens,
        ..GenerateParams::default()
    };
    if greedy {
        params.temperature = 0.0;
    }

    let prompt = MultimodalPrompt {
        text: case.prompt.to_owned(),
        media: (case.media)(),
    };

    let gen_start = Instant::now();
    let output = engine
        .generate_multimodal_chat(&prompt, &params)
        .map_err(|err| format!("Multimodal generation failed for {}: {err}", case.name))?;
    let seconds = gen_start.elapsed().as_secs_f64();
    let output_tokens = count_output_tokens(model_dir, &output);
    let tokens_per_second = if seconds > 0.0 {
        output_tokens as f64 / seconds
    } else {
        0.0
    };
    Ok(BenchResult {
        name: case.name,
        seconds,
        output_tokens,
        tokens_per_second,
    })
}

fn print_suite_summary(results: &[BenchResult]) {
    let total_seconds: f64 = results.iter().map(|result| result.seconds).sum();
    let total_tokens: usize = results.iter().map(|result| result.output_tokens).sum();
    let aggregate_tps = if total_seconds > 0.0 {
        total_tokens as f64 / total_seconds
    } else {
        0.0
    };
    eprintln!("\n=== Suite Summary ===");
    eprintln!("case,seconds,output_tokens,tokens_per_second");
    for result in results {
        eprintln!(
            "{},{:.2},{},{:.1}",
            result.name, result.seconds, result.output_tokens, result.tokens_per_second
        );
    }
    eprintln!("total,{total_seconds:.2},{total_tokens},{aggregate_tps:.1}");
}

#[allow(clippy::too_many_lines)]
fn run(args: Args) -> Result<(), String> {
    eprintln!("=== Inference Benchmark ===");
    eprintln!("Model: {}", args.model.display());
    if args.cache_suite {
        eprintln!("Suite: prompt-cache miss/hit/shared-prefix suite");
    } else if args.multimodal_suite {
        eprintln!("Suite: deterministic multimodal image/audio/video suite");
    } else if args.suite {
        eprintln!("Suite: built-in default-path prompt suite");
    } else {
        eprintln!("Max tokens: {}", args.tokens);
    }
    eprintln!(
        "Mode: {}",
        if args.raw { "raw continuation" } else { "chat" }
    );
    eprintln!(
        "Sampling: {}",
        if args.greedy {
            "greedy"
        } else {
            "quality defaults"
        }
    );
    eprintln!("Prompt: \"{}\"", args.prompt);
    eprintln!();

    let load_start = Instant::now();
    let model_dir = args.model;
    let config = EngineConfig {
        model_dir: model_dir.clone(),
        max_context_length: 0,
    };
    let mut engine = Engine::new(&config).map_err(|err| format!("Failed to load model: {err}"))?;
    let load_time = load_start.elapsed();
    eprintln!("Load time: {:.2}s", load_time.as_secs_f64());
    if args.multimodal_suite {
        let support = engine.multimodal_support();
        eprintln!(
            "Multimodal support: image={}, audio={}, video={}",
            support.supports_images(),
            support.supports_audio(),
            support.supports_video()
        );
    }

    if args.multimodal_suite {
        if args.raw {
            return Err("--multimodal-suite always uses chat templating; remove --raw".into());
        }
        let mut results = Vec::with_capacity(MULTIMODAL_SUITE.len());
        for case in MULTIMODAL_SUITE {
            engine.reset();
            eprintln!("\n--- Case: {} (max {} tokens) ---", case.name, case.tokens);
            let result = generate_multimodal_once(&mut engine, &model_dir, case, args.greedy)?;
            eprintln!(
                "{}: {:.2}s, {} tokens, {:.1} tok/s",
                result.name, result.seconds, result.output_tokens, result.tokens_per_second
            );
            results.push(result);
        }
        print_suite_summary(&results);
        return Ok(());
    }

    if args.suite || args.cache_suite {
        let cases = if args.cache_suite {
            CACHE_SUITE
        } else {
            BENCH_SUITE
        };
        let mut results = Vec::with_capacity(cases.len());
        for case in cases {
            if args.suite {
                engine.reset();
            }
            eprintln!("\n--- Case: {} (max {} tokens) ---", case.name, case.tokens);
            let (_, mut result) = generate_once(
                &mut engine,
                &model_dir,
                case.prompt,
                case.tokens,
                args.raw,
                args.greedy,
            )?;
            result.name = case.name;
            eprintln!(
                "{}: {:.2}s, {} tokens, {:.1} tok/s",
                result.name, result.seconds, result.output_tokens, result.tokens_per_second
            );
            results.push(result);
        }

        print_suite_summary(&results);
        return Ok(());
    }

    let (output, result) = generate_once(
        &mut engine,
        &model_dir,
        &args.prompt,
        args.tokens,
        args.raw,
        args.greedy,
    )?;

    eprintln!();
    eprintln!("=== Results ===");
    eprintln!("Generation time: {:.2}s", result.seconds);
    eprintln!("Output tokens: {}", result.output_tokens);
    eprintln!("Speed: {:.1} tok/s", result.tokens_per_second);
    eprintln!();
    eprintln!("Output:");
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

    if let Err(err) = run(opts) {
        eprintln!("{err}");
        return ExitCode::FAILURE;
    }

    ExitCode::SUCCESS
}

#[must_use]
pub fn main_from_env() -> ExitCode {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    main_with_args(&argv)
}
