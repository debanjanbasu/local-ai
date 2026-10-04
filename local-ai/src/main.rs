use std::process::ExitCode;

fn usage() {
    eprintln!("Usage: local-ai <command> [args]");
    eprintln!();
    eprintln!("  chat       Run Bonsai 2 27B text inference with default settings");
    eprintln!("  bonsai     Run Bonsai 2 27B inference with backend, MTP and profiling options");
    eprintln!("  serve      Run the OpenAI-compatible server");
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some((command, rest)) = args.split_first() else {
        usage();
        return ExitCode::FAILURE;
    };
    match command.as_str() {
        "chat" => local_ai::cli::chat::main_with_args(rest),
        "bonsai" => local_ai::cli::bonsai::main_with_args(rest),
        "serve" => local_ai::cli::serve::main_with_args(rest),
        "--help" | "-h" => {
            usage();
            ExitCode::SUCCESS
        }
        _ => {
            eprintln!("error: unknown command {command:?}");
            usage();
            ExitCode::FAILURE
        }
    }
}
