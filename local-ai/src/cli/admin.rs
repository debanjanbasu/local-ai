//! `local-ai admin`: the owner's command line for authentication.
//!
//! It works on the same native services store the server uses, so changes
//! take effect on a running server at once.

use std::path::PathBuf;
use std::process::ExitCode;

use local_services::Store;
use local_services::admin::{Actor, Admin, Page};

fn usage() {
    eprintln!("Usage: local-ai admin <command> [options]");
    eprintln!();
    eprintln!(
        "  create-key [--name NAME] [--expires-in-seconds N]  Create an admin API key, print it \
         once on stdout, and turn authentication on"
    );
    eprintln!("  list-keys          List admin API keys (redacted)");
    eprintln!("  delete-key ID      Delete an admin API key (authentication stays on)");
    eprintln!("  status             Show whether authentication is on");
    eprintln!("  disable-auth       Turn authentication off (keys are kept)");
    eprintln!("  enable-auth        Turn authentication back on (needs a live admin key)");
    eprintln!();
    eprintln!(
        "  With authentication on, every request needs `Authorization: Bearer <key>`: project \
         keys (created through the Administration API's service accounts) for the API, admin \
         keys for /v1/organization. See docs/ADMIN.md."
    );
}

/// The services store the server opens (see `serve::services`).
fn store() -> Result<Admin, String> {
    let home = std::env::var_os("HOME")
        .filter(|home| !home.is_empty())
        .map(PathBuf::from)
        .ok_or("HOME must name the user's directory")?;
    let path = home.join("Library/Application Support/local-ai/services");
    Store::open(&path)
        .map(Admin::new)
        .map_err(|error| format!("cannot open native services at {}: {error}", path.display()))
}

fn run(args: &[String]) -> Result<(), String> {
    let Some((command, rest)) = args.split_first() else {
        return Err(String::new());
    };
    let text = |error: local_services::Error| error.to_string();
    match (command.as_str(), rest) {
        ("create-key", options) => {
            let (mut name, mut expires) = ("Admin key".to_owned(), None);
            let mut index = 0;
            while index < options.len() {
                let value = options
                    .get(index + 1)
                    .ok_or_else(|| format!("{} requires a value", options[index]))?;
                match options[index].as_str() {
                    "--name" => name.clone_from(value),
                    "--expires-in-seconds" => {
                        expires = Some(value.parse::<i64>().map_err(|_| {
                            format!("--expires-in-seconds must be a whole number, not {value:?}")
                        })?);
                    }
                    other => return Err(format!("unknown option: {other}")),
                }
                index += 2;
            }
            let created = store()?
                .create_admin_key(&name, expires, &Actor::Owner)
                .map_err(text)?;
            eprintln!(
                "created admin API key {} ({}); authentication is now on. This is the only time \
                 the key is shown:",
                created.object.id, created.object.redacted_value
            );
            println!("{}", created.value);
        }
        ("list-keys", []) => {
            let admin = store()?;
            let mut page = Page {
                limit: Some(local_services::admin::MAX_LIST_LIMIT),
                ..Page::default()
            };
            loop {
                let list = admin.list_admin_keys(&page).map_err(text)?;
                for key in &list.data {
                    println!(
                        "{}\t{}\t{}\tcreated {}\texpires {}\tlast used {}",
                        key.id,
                        key.redacted_value,
                        key.name.as_deref().unwrap_or(""),
                        key.created_at,
                        key.expires_at
                            .map_or_else(|| "never".to_owned(), |at| at.to_string()),
                        key.last_used_at
                            .map_or_else(|| "never".to_owned(), |at| at.to_string()),
                    );
                }
                if !list.has_more {
                    break;
                }
                page.after = list.last_id;
            }
        }
        ("delete-key", [id]) => {
            store()?.delete_admin_key(id, &Actor::Owner).map_err(text)?;
            eprintln!("deleted admin API key {id}");
        }
        ("status", []) => {
            let on = store()?.auth_enabled().map_err(text)?;
            println!("authentication: {}", if on { "on" } else { "off" });
        }
        ("disable-auth", []) => {
            store()?.disable_auth().map_err(text)?;
            eprintln!("authentication is off: any process that can reach the server may use it");
        }
        ("enable-auth", []) => {
            store()?.enable_auth().map_err(text)?;
            eprintln!("authentication is on");
        }
        ("--help" | "-h", _) => return Err(String::new()),
        (other, _) => return Err(format!("unknown or malformed admin command: {other}")),
    }
    Ok(())
}

pub fn main_with_args(args: &[String]) -> ExitCode {
    match run(args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            if !error.is_empty() {
                eprintln!("error: {error}");
            }
            usage();
            if error.is_empty() && !args.is_empty() {
                ExitCode::SUCCESS
            } else {
                ExitCode::FAILURE
            }
        }
    }
}
