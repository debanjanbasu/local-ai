use std::path::PathBuf;

use super::Args;

pub(super) fn usage() {
    eprintln!("Usage: local-ai serve [options]");
    eprintln!();
    eprintln!("  --model PATH       Override automatic pinned-model discovery");
    eprintln!("  --host IP          Bind address (default: 127.0.0.1)");
    eprintln!("  --port N           TCP/UDP port (default: 8080)");
    eprintln!("  --api-key KEY      Require bearer authentication");
    eprintln!("  --no-thinking      Disable thinking");
}

pub(super) fn parse(args: &[String]) -> Result<Args, String> {
    let mut parsed = Args {
        model: None,
        host: "127.0.0.1".parse().map_err(|_| "invalid default host")?,
        port: 8080,
        thinking: true,
        api_key: None,
    };
    let mut index = 0;
    while index < args.len() {
        let flag = args[index].as_str();
        if matches!(flag, "--help" | "-h") {
            return Err(String::new());
        }
        if matches!(flag, "--thinking" | "--no-thinking") {
            parsed.thinking = flag == "--thinking";
            index += 1;
            continue;
        }
        index += 1;
        let value = args
            .get(index)
            .ok_or_else(|| format!("{flag} requires a value"))?;
        match flag {
            "--model" => parsed.model = Some(PathBuf::from(value)),
            "--host" => parsed.host = value.parse().map_err(|_| "invalid --host")?,
            "--port" => parsed.port = value.parse().map_err(|_| "invalid --port")?,
            "--api-key" => parsed.api_key = Some(value.clone()),
            _ => return Err(format!("unknown option: {flag}")),
        }
        index += 1;
    }
    Ok(parsed)
}
