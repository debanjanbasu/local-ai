use std::process::ExitCode;

fn print_usage() {
    eprintln!("Usage: local-ai <subcommand> [args]");
    eprintln!();
    eprintln!("Available subcommands:");
    eprintln!("  benchmark  Run inference speed benchmark");
    eprintln!("  chat       Generate text from a prompt");
    eprintln!("  serve      Run the continuous batching demo");
    eprintln!("  compress   Compress a model for on-device inference");
    eprintln!();
    eprintln!("Use `local-ai <subcommand> --help` for per-command options.");
}

fn dispatch(command: &str, argv: &[String]) -> ExitCode {
    match command {
        "benchmark" => local_engine::cli::benchmark::main_with_args(argv),
        "chat" => local_engine::cli::chat::main_with_args(argv),
        "serve" => local_engine::cli::serve::main_with_args(argv),
        "compress" => local_engine::cli::compress::main_with_args(argv),
        _ => {
            eprintln!("Unknown subcommand: {command}");
            print_usage();
            ExitCode::FAILURE
        }
    }
}

fn main() -> ExitCode {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let Some((command, rest)) = argv.split_first() else {
        print_usage();
        return ExitCode::FAILURE;
    };
    if matches!(command.as_str(), "--help" | "-h") {
        print_usage();
        return ExitCode::SUCCESS;
    }
    dispatch(command, rest)
}
