//! Hosted tool permissions and limits.
//!
//! Safety: every command run here is a fixed, harmless, strictly bounded one
//! (`echo`, `wc`, `printf` of a fixed size, `ulimit` queries, `sleep` stopped
//! by a short deadline, one 2 MiB file of zeros, at most two background
//! `sleep`s, one `perl` that leaves its process group and sleeps until the
//! deadline). Refusals are checked on inert strings that are validated and
//! never executed.

#![expect(clippy::expect_used, clippy::panic)]

use std::path::Path;
use std::time::{Duration, Instant};

use local_services::tools::shell::{self, Outcome};
use local_services::tools::{
    Capability, Limits, Output, Permissions, Request, SearchMode, ToolError, Toolbox, fetch,
};

const SECOND: Duration = Duration::from_secs(1);

fn toolbox(root: &Path, write: bool, shell: bool) -> Toolbox {
    Toolbox::new(
        Permissions {
            root: Some(root.to_owned()),
            write,
            shell,
            fetch_domains: Vec::new(),
        },
        Limits::default(),
    )
    .expect("toolbox")
}

fn text(output: Output) -> String {
    match output {
        Output::Text { text, .. } => text,
        Output::Shell(runs) => shell::model_text(&runs),
    }
}

fn runs(output: Output) -> Vec<shell::CommandRun> {
    match output {
        Output::Shell(runs) => runs,
        Output::Text { text, .. } => panic!("expected shell output, got {text}"),
    }
}

fn sh(commands: &[&str], timeout: Option<Duration>) -> Request {
    Request::Shell {
        commands: commands
            .iter()
            .map(|command| (*command).to_owned())
            .collect(),
        timeout,
        max_output_bytes: None,
    }
}

#[tokio::test]
async fn default_permissions_grant_nothing() {
    let tools = Toolbox::new(Permissions::default(), Limits::default()).expect("toolbox");
    assert!(!tools.any());
    for request in [
        Request::List { path: None },
        Request::Write {
            path: "a.txt".into(),
            content: "x".into(),
        },
        sh(&["echo never-run"], None),
        Request::Fetch {
            url: "https://example.com/".into(),
        },
    ] {
        let error = tools.run(request, SECOND).await.expect_err("refused");
        assert!(matches!(error, ToolError::Denied(_)), "{error}");
    }
}

#[tokio::test]
async fn read_only_root_refuses_writes_and_shell() {
    let root = tempfile::tempdir().expect("root");
    let tools = toolbox(root.path(), false, false);
    assert!(tools.allows(Capability::Read));
    assert!(!tools.allows(Capability::Write));
    let error = tools
        .run(
            Request::Write {
                path: "new.txt".into(),
                content: "x".into(),
            },
            SECOND,
        )
        .await
        .expect_err("write refused");
    assert!(error.to_string().contains("--tools-allow-write"), "{error}");
    assert!(!root.path().join("new.txt").exists());
    let error = tools
        .run(sh(&["echo never-run"], None), SECOND)
        .await
        .expect_err("shell refused");
    assert!(matches!(error, ToolError::Denied(_)), "{error}");
}

#[test]
fn write_and_shell_need_a_root() {
    for (write, shell) in [(true, false), (false, true)] {
        let error = Toolbox::new(
            Permissions {
                root: None,
                write,
                shell,
                fetch_domains: Vec::new(),
            },
            Limits::default(),
        )
        .expect_err("needs a root");
        assert!(error.to_string().contains("root"), "{error}");
    }
}

#[tokio::test]
async fn list_read_search_and_truncation() {
    let root = tempfile::tempdir().expect("root");
    std::fs::write(root.path().join("a.txt"), "alpha\nbeta\ngamma\n").expect("write");
    std::fs::create_dir(root.path().join("sub")).expect("dir");
    std::fs::write(root.path().join("sub/b.txt"), "beta again\n").expect("write");
    let tools = toolbox(root.path(), false, false);
    let listed = text(
        tools
            .run(Request::List { path: None }, SECOND)
            .await
            .expect("list"),
    );
    assert!(
        listed.contains("a.txt\t17 bytes") && listed.contains("sub/b.txt"),
        "{listed}"
    );
    let read = text(
        tools
            .run(
                Request::Read {
                    path: "a.txt".into(),
                    start_line: Some(2),
                    end_line: Some(2),
                },
                SECOND,
            )
            .await
            .expect("read"),
    );
    assert!(read.starts_with("2: beta\n"), "{read}");
    let found = text(
        tools
            .run(
                Request::Search {
                    pattern: "beta".into(),
                    path: None,
                    mode: SearchMode::Literal,
                },
                SECOND,
            )
            .await
            .expect("search"),
    );
    assert!(found.contains("a.txt:2: beta") && found.contains("sub/b.txt:1: beta again"));

    std::fs::write(root.path().join("big.txt"), "x".repeat(10_000)).expect("write");
    let small = Toolbox::new(
        Permissions {
            root: Some(root.path().to_owned()),
            ..Permissions::default()
        },
        Limits {
            max_output_bytes: 64,
            ..Limits::default()
        },
    )
    .expect("toolbox");
    let output = small
        .run(
            Request::Read {
                path: "big.txt".into(),
                start_line: None,
                end_line: None,
            },
            SECOND,
        )
        .await
        .expect("read");
    let Output::Text { text, truncated } = output else {
        panic!("text expected");
    };
    assert!(truncated && text.contains("[truncated"), "{text}");
}

#[tokio::test]
async fn path_escapes_are_refused() {
    let outer = tempfile::tempdir().expect("outer");
    let root = outer.path().join("root");
    std::fs::create_dir(&root).expect("root");
    let secret = outer.path().join("secret.txt");
    std::fs::write(&secret, "outside").expect("secret");
    std::fs::create_dir(root.join(".git")).expect("git");
    std::os::unix::fs::symlink(&secret, root.join("link.txt")).expect("link");
    std::os::unix::fs::symlink(outer.path(), root.join("linkdir")).expect("linkdir");
    let tools = toolbox(&root, true, false);
    let escapes = [
        "../secret.txt",
        "sub/../../secret.txt",
        "/etc/hosts",
        secret.to_str().expect("utf-8"),
        ".git/config",
        "link.txt",
        "linkdir/secret.txt",
    ];
    for path in escapes {
        let read = tools
            .run(
                Request::Read {
                    path: path.into(),
                    start_line: None,
                    end_line: None,
                },
                SECOND,
            )
            .await;
        assert!(read.is_err(), "read of {path} must be refused: {read:?}");
        let write = tools
            .run(
                Request::Write {
                    path: path.into(),
                    content: "overwritten".into(),
                },
                SECOND,
            )
            .await;
        assert!(write.is_err(), "write of {path} must be refused: {write:?}");
        let edit = tools
            .run(
                Request::Edit {
                    path: path.into(),
                    old: "outside".into(),
                    new: "edited".into(),
                },
                SECOND,
            )
            .await;
        assert!(edit.is_err(), "edit of {path} must be refused: {edit:?}");
    }
    assert_eq!(std::fs::read_to_string(&secret).expect("secret"), "outside");
    assert!(!root.join(".git/config").exists());
}

#[tokio::test]
async fn write_and_edit_inside_the_root() {
    let root = tempfile::tempdir().expect("root");
    let tools = toolbox(root.path(), true, false);
    tools
        .run(
            Request::Write {
                path: "notes/today.txt".into(),
                content: "one two two\n".into(),
            },
            SECOND,
        )
        .await
        .expect("write");
    let error = tools
        .run(
            Request::Edit {
                path: "notes/today.txt".into(),
                old: "two".into(),
                new: "three".into(),
            },
            SECOND,
        )
        .await
        .expect_err("ambiguous edit");
    assert!(error.to_string().contains("2 times"), "{error}");
    tools
        .run(
            Request::Edit {
                path: "notes/today.txt".into(),
                old: "one".into(),
                new: "zero".into(),
            },
            SECOND,
        )
        .await
        .expect("edit");
    assert_eq!(
        std::fs::read_to_string(root.path().join("notes/today.txt")).expect("read"),
        "zero two two\n"
    );
    // A second name for a file (here inside the root, standing in for one
    // outside it) makes it unwritable through the tools.
    std::fs::hard_link(
        root.path().join("notes/today.txt"),
        root.path().join("alias.txt"),
    )
    .expect("hard link");
    let error = tools
        .run(
            Request::Write {
                path: "alias.txt".into(),
                content: "changed\n".into(),
            },
            SECOND,
        )
        .await
        .expect_err("refused");
    assert!(matches!(error, ToolError::Denied(_)), "{error}");
    assert_eq!(
        std::fs::read_to_string(root.path().join("notes/today.txt")).expect("read"),
        "zero two two\n"
    );
}

#[tokio::test]
async fn writes_never_follow_a_symlink_swapped_in_at_the_last_component() {
    let root = tempfile::tempdir().expect("root");
    let target = root.path().join("target.txt");
    std::fs::write(&target, "original\n").expect("target");
    std::fs::write(root.path().join("x"), "plain\n").expect("x");
    // A racing process keeps swapping `x` between a regular file and a
    // symlink to `target.txt` (inside the root, so cap-std alone would
    // follow it), by atomic renames, for a bounded number of writes. The
    // race is probabilistic; following the link loses it in most runs.
    let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let swapper = {
        let stop = std::sync::Arc::clone(&stop);
        let dir = root.path().to_owned();
        std::thread::spawn(move || {
            while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                let _ = std::os::unix::fs::symlink("target.txt", dir.join(".link"));
                let _ = std::fs::rename(dir.join(".link"), dir.join("x"));
                let _ = std::fs::write(dir.join(".file"), "plain\n");
                let _ = std::fs::rename(dir.join(".file"), dir.join("x"));
            }
        })
    };
    let tools = toolbox(root.path(), true, false);
    for round in 0..10_000 {
        let request = if round % 2 == 0 {
            Request::Write {
                path: "x".into(),
                content: "written\n".into(),
            }
        } else {
            Request::Edit {
                path: "x".into(),
                old: "original".into(),
                new: "edited".into(),
            }
        };
        let _ = tools.run(request, SECOND).await;
    }
    stop.store(true, std::sync::atomic::Ordering::Relaxed);
    swapper.join().expect("swapper");
    assert_eq!(
        std::fs::read_to_string(&target).expect("target"),
        "original\n"
    );
}

#[tokio::test]
async fn shell_runs_harmless_commands_in_the_root() {
    let root = tempfile::tempdir().expect("root");
    std::fs::write(root.path().join("a.txt"), "hello\n").expect("write");
    let tools = toolbox(root.path(), false, true);
    let runs = runs(
        tools
            .run(sh(&["ls", "wc -c a.txt", "exit 3"], None), 10 * SECOND)
            .await
            .expect("shell"),
    );
    assert_eq!(runs[0].stdout, "a.txt\n");
    assert_eq!(runs[0].outcome, Outcome::Exit(0));
    assert!(runs[1].stdout.contains("6 a.txt"), "{:?}", runs[1]);
    assert_eq!(runs[2].outcome, Outcome::Exit(3));
}

#[tokio::test]
async fn shell_deadline_kills_a_sleep() {
    let root = tempfile::tempdir().expect("root");
    let tools = toolbox(root.path(), false, true);
    let started = Instant::now();
    let runs = runs(
        tools
            .run(sh(&["sleep 30"], Some(SECOND)), 10 * SECOND)
            .await
            .expect("shell"),
    );
    assert_eq!(runs[0].outcome, Outcome::TimedOut);
    assert!(started.elapsed() < 5 * SECOND, "{:?}", started.elapsed());
}

#[tokio::test]
async fn shell_output_is_capped_with_a_marker() {
    let root = tempfile::tempdir().expect("root");
    let tools = toolbox(root.path(), false, true);
    let output = tools
        .run(
            Request::Shell {
                commands: vec!["printf '%5000s' x".into()],
                timeout: None,
                max_output_bytes: Some(100),
            },
            10 * SECOND,
        )
        .await
        .expect("shell");
    let runs = runs(output);
    assert!(runs[0].truncated);
    assert!(
        runs[0]
            .stdout
            .contains("[truncated: 100 of 5000 bytes shown"),
        "{}",
        runs[0].stdout
    );
}

#[tokio::test]
async fn shell_rlimits_are_set_in_the_child() {
    let root = tempfile::tempdir().expect("root");
    let limits = Limits {
        shell_cpu_seconds: 3,
        shell_open_files: 40,
        shell_file_bytes: 512 * 1024,
        ..Limits::default()
    };
    let tools = Toolbox::new(
        Permissions {
            root: Some(root.path().to_owned()),
            shell: true,
            ..Permissions::default()
        },
        limits,
    )
    .expect("toolbox");
    let runs = runs(
        tools
            .run(
                sh(
                    &["ulimit -t; ulimit -n; ulimit -f; ulimit -c; ulimit -u"],
                    None,
                ),
                10 * SECOND,
            )
            .await
            .expect("shell"),
    );
    let lines: Vec<&str> = runs[0].stdout.lines().collect();
    // `ulimit -f` reports 1024-byte blocks here.
    assert_eq!(&lines[..4], ["3", "40", "512", "0"], "{lines:?}");
    // The process limit is the user's current count plus a small margin,
    // never unlimited.
    let processes: u64 = lines[4].parse().expect("a number");
    assert!(processes > limits.shell_processes, "{processes}");
}

#[tokio::test]
async fn shell_scratch_directory_is_bounded() {
    let root = tempfile::tempdir().expect("root");
    let tools = Toolbox::new(
        Permissions {
            root: Some(root.path().to_owned()),
            shell: true,
            ..Permissions::default()
        },
        Limits {
            shell_scratch_bytes: 1024 * 1024,
            ..Limits::default()
        },
    )
    .expect("toolbox");
    let started = Instant::now();
    // 2 MiB, once, against a 1 MiB quota; the sleep only keeps the command
    // alive until the quota check sees the file.
    let runs = runs(
        tools
            .run(
                sh(
                    &["head -c 2097152 /dev/zero > \"$TMPDIR/big\" && sleep 5"],
                    None,
                ),
                10 * SECOND,
            )
            .await
            .expect("shell"),
    );
    assert_eq!(runs[0].outcome, Outcome::Signal(9), "{:?}", runs[0]);
    assert!(
        runs[0]
            .stderr
            .contains("[killed: the scratch directory passed"),
        "{:?}",
        runs[0]
    );
    assert!(started.elapsed() < 4 * SECOND, "{:?}", started.elapsed());
}

#[tokio::test]
async fn shell_has_no_network() {
    let root = tempfile::tempdir().expect("root");
    let tools = toolbox(root.path(), false, true);
    // A connection attempt to a closed localhost port. Outside the sandbox
    // it is refused by the port; inside, the sandbox denies the connect
    // itself, which is what distinguishes the two.
    let probe = r#"/usr/bin/perl -e 'use Socket; socket(my $s, PF_INET, SOCK_STREAM, 0) or die "socket: $!\n"; connect($s, pack_sockaddr_in(9, inet_aton("127.0.0.1"))) or die "connect: $!\n"; print "connected\n";'"#;
    let runs = runs(
        tools
            .run(sh(&[probe], None), 10 * SECOND)
            .await
            .expect("shell"),
    );
    assert_ne!(runs[0].outcome, Outcome::Exit(0));
    assert!(!runs[0].stdout.contains("connected"));
    assert!(
        runs[0].stderr.contains("Operation not permitted"),
        "{:?}",
        runs[0]
    );
}

#[tokio::test]
async fn shell_writes_are_confined() {
    let outer = tempfile::tempdir().expect("outer");
    let root = outer.path().join("root");
    std::fs::create_dir_all(root.join(".git")).expect("root");
    let outside = outer.path().join("outside.txt");
    let command = format!(
        "echo x > inside.txt; echo x > .git/inside.txt; echo x > '{}'; echo x > \"$TMPDIR/scratch.txt\" && echo scratch-ok",
        outside.display()
    );
    let read_only = toolbox(&root, false, true);
    let run = runs(
        read_only
            .run(sh(&[&command], None), 10 * SECOND)
            .await
            .expect("shell"),
    );
    assert!(run[0].stdout.contains("scratch-ok"), "{:?}", run[0]);
    assert!(!root.join("inside.txt").exists());
    assert!(!outside.exists());
    let writable = toolbox(&root, true, true);
    writable
        .run(sh(&[&command], None), 10 * SECOND)
        .await
        .expect("shell");
    assert!(root.join("inside.txt").exists());
    assert!(!root.join(".git/inside.txt").exists());
    assert!(!outside.exists());
    // A nested repository's `.git`, in any case, is as closed as the root's;
    // names that only resemble it are not.
    std::fs::create_dir_all(root.join("sub/.git")).expect("nested .git");
    writable
        .run(
            sh(
                &[
                    "echo x > sub/.git/config",
                    "echo x > sub/.GIT/hooks",
                    "mkdir -p deep/.Git",
                    "echo x > sub/.github; echo x > sub/x.git",
                ],
                None,
            ),
            10 * SECOND,
        )
        .await
        .expect("shell");
    assert!(!root.join("sub/.git/config").exists());
    assert!(!root.join("sub/.git/hooks").exists());
    assert!(!root.join("deep/.Git").exists());
    assert!(root.join("sub/.github").exists());
    assert!(root.join("sub/x.git").exists());
    // A hard link would make a file outside the root writable through it.
    let source = outer.path().join("source.txt");
    std::fs::write(&source, "outside\n").expect("source");
    let link = format!("ln '{}' hard.txt", source.display());
    let run = runs(
        writable
            .run(sh(&[&link], None), 10 * SECOND)
            .await
            .expect("shell"),
    );
    assert_ne!(run[0].outcome, Outcome::Exit(0), "{:?}", run[0]);
    assert!(!root.join("hard.txt").exists());
}

#[tokio::test]
async fn shell_kills_its_process_group_when_the_command_exits() {
    let root = tempfile::tempdir().expect("root");
    let tools = toolbox(root.path(), false, true);
    let started = Instant::now();
    // Two fixed background children keep stdout open. The call can only
    // return before its 10 s deadline if they were killed with the group.
    let runs = runs(
        tools
            .run(
                sh(&["sleep 30 & sleep 30 & echo started"], Some(10 * SECOND)),
                20 * SECOND,
            )
            .await
            .expect("shell"),
    );
    assert_eq!(runs[0].outcome, Outcome::Exit(0));
    assert!(runs[0].stdout.starts_with("started"), "{:?}", runs[0]);
    assert!(started.elapsed() < 5 * SECOND, "{:?}", started.elapsed());
}

#[tokio::test]
async fn shell_deadline_kills_a_leader_that_left_its_group() {
    let root = tempfile::tempdir().expect("root");
    let tools = toolbox(root.path(), false, true);
    let started = Instant::now();
    // One escape attempt: the shell becomes perl (same PID, the group's
    // leader), which moves itself into this test process's group, out of
    // reach of the group kill. The hang guard bounds the test either way;
    // dropping the call kills the leader by PID.
    let escape = "echo $$; exec /usr/bin/perl -e 'setpgrp(0, getpgrp(getppid())) or die \"setpgrp: $!\\n\"; sleep 1e9'";
    let call = tools.run(sh(&[escape], Some(SECOND)), 10 * SECOND);
    let runs = runs(
        tokio::time::timeout(8 * SECOND, call)
            .await
            .expect("the call outlived its deadline")
            .expect("shell"),
    );
    assert_eq!(runs[0].outcome, Outcome::TimedOut, "{:?}", runs[0]);
    assert!(runs[0].stderr.is_empty(), "{:?}", runs[0]);
    assert!(started.elapsed() < 5 * SECOND, "{:?}", started.elapsed());
    let pid = runs[0].stdout.trim();
    let status = std::process::Command::new("/bin/ps")
        .args(["-o", "stat=", "-p", pid])
        .output()
        .expect("ps");
    let status = String::from_utf8_lossy(&status.stdout);
    assert!(
        status.trim().is_empty() || status.trim().starts_with('Z'),
        "{status:?}"
    );
}

#[tokio::test]
async fn dropping_a_shell_call_kills_it() {
    let root = tempfile::tempdir().expect("root");
    let tools = toolbox(root.path(), true, true);
    let call = tools.run(sh(&["echo $$ > pid.txt; exec sleep 30"], None), 20 * SECOND);
    let abandoned = tokio::time::timeout(Duration::from_millis(1500), call).await;
    assert!(abandoned.is_err(), "the call should still be running");
    let pid = std::fs::read_to_string(root.path().join("pid.txt")).expect("pid");
    tokio::time::sleep(Duration::from_millis(300)).await;
    // `ps` prints nothing for a process that is gone and `Z` for one that is
    // dead and not yet reaped; a live `sleep` would show `S`.
    let status = std::process::Command::new("/bin/ps")
        .args(["-o", "stat=", "-p", pid.trim()])
        .output()
        .expect("ps");
    let status = String::from_utf8_lossy(&status.stdout);
    assert!(
        status.trim().is_empty() || status.trim().starts_with('Z'),
        "{status:?}"
    );
}

#[tokio::test]
async fn malformed_shell_arguments_are_refused_before_running() {
    let root = tempfile::tempdir().expect("root");
    let tools = toolbox(root.path(), false, true);
    let too_many: Vec<&str> = vec!["true"; shell::MAX_COMMANDS + 1];
    for request in [
        sh(&[], None),
        sh(&["   "], None),
        sh(&["echo a\0b"], None),
        sh(&too_many, None),
    ] {
        let error = tools.run(request, SECOND).await.expect_err("invalid");
        assert!(matches!(error, ToolError::Invalid(_)), "{error}");
    }
}

#[test]
fn fetch_targets_are_checked_without_network() {
    let domains = vec!["example.com".to_owned()];
    let target = fetch::target("https://docs.example.com/a?b=1", &domains).expect("allowed");
    assert_eq!(
        (
            target.https,
            target.host.as_str(),
            target.port,
            target.path.as_str()
        ),
        (true, "docs.example.com", 443, "/a?b=1")
    );
    for url in [
        "https://example.org/",
        "https://badexample.com/",
        "https://127.0.0.1/",
        "http://[::1]/",
        "https://user@example.com/",
        "ftp://example.com/",
    ] {
        assert!(
            matches!(fetch::target(url, &domains), Err(ToolError::Denied(_))),
            "{url}"
        );
    }
    assert!(fetch::target("file:///etc/passwd", &domains).is_err());
    for address in [
        "127.0.0.1",
        "10.1.2.3",
        "192.168.0.1",
        "169.254.169.254",
        "100.64.0.1",
        "0.0.0.0",
        "::1",
        "fe80::1",
        "fd00::1",
        "::ffff:127.0.0.1",
        "2001:db8::1",
    ] {
        assert!(!fetch::is_public(address.parse().expect("ip")), "{address}");
    }
    for address in ["93.184.215.14", "2606:4700::6810:84e5"] {
        assert!(fetch::is_public(address.parse().expect("ip")), "{address}");
    }
    assert!(fetch::check_domain("Example.com").is_err());
    assert!(fetch::check_domain("10.0.0.1").is_err());
}

#[tokio::test]
async fn fetch_refuses_names_resolving_to_loopback() {
    let tools = Toolbox::new(
        Permissions {
            fetch_domains: vec!["localhost".into()],
            ..Permissions::default()
        },
        Limits::default(),
    )
    .expect("toolbox");
    let error = tools
        .run(
            Request::Fetch {
                url: "http://localhost:9/".into(),
            },
            5 * SECOND,
        )
        .await
        .expect_err("loopback refused");
    assert!(
        matches!(&error, ToolError::Denied(message) if message.contains("non-public")),
        "{error}"
    );
}

#[tokio::test]
async fn shell_inherits_no_descriptors_beyond_its_standard_streams() {
    use std::os::fd::{AsRawFd as _, FromRawFd as _, OwnedFd};
    let root = tempfile::tempdir().expect("root");
    let file = std::fs::File::open("/etc/hosts").expect("open");
    // A descriptor without close-on-exec, as a C library might leave one.
    // SAFETY: `dup` of an open descriptor; the result is owned below.
    #[expect(unsafe_code)]
    let leaked = unsafe { OwnedFd::from_raw_fd(libc::dup(file.as_raw_fd())) };
    let tools = toolbox(root.path(), false, true);
    let runs = runs(
        tools
            .run(sh(&["ls /dev/fd"], None), 10 * SECOND)
            .await
            .expect("shell"),
    );
    let open: Vec<&str> = runs[0].stdout.lines().collect();
    assert!(
        !open.contains(&leaked.as_raw_fd().to_string().as_str()),
        "{open:?}"
    );
}
