#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::env;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

const SDK: &str = "macosx";

/// Shaders in this library. A listed file that is missing fails the build:
/// skipping it would only surface as a missing kernel at startup.
const SHADER_FILES: &[&str] = &[
    "bonsai.metal",
    "bonsai_ops.metal",
    "bonsai_small_batch.metal",
    "bonsai_int8.metal",
    "bonsai_ane.metal",
    "sampling.metal",
    "draft.metal",
    "embedding.metal",
    "embedding_vision.metal",
    "bonsai_vision.metal",
    "embedding_audio.metal",
    "kokoro.metal",
    "whisper.metal",
];

/// The tensor kernels are compiled only from this Metal language version on.
const TENSOR_METAL_VERSION: u32 = 400;

const TOOLCHAIN_HINT: &str = "install Xcode (or the Command Line Tools: `xcode-select --install`) \
     and its Metal toolchain (`xcodebuild -downloadComponent MetalToolchain`); \
     DEVELOPER_DIR and SDKROOT select which ones xcrun uses";

fn xcrun() -> Command {
    let mut command = Command::new("xcrun");
    command.args(["-sdk", SDK]);
    command
}

/// The Metal language version the SDK compiles these shaders for. This is
/// the first use of the toolchain, so its failure names the fix.
fn metal_version() -> Option<u32> {
    let output = xcrun()
        .args(["metal", "-E", "-dM", "-x", "metal", "/dev/null", "-o", "-"])
        .stderr(Stdio::piped())
        .output()
        .unwrap_or_else(|e| panic!("cannot run xcrun ({e}): {TOOLCHAIN_HINT}"));
    if !output.status.success() {
        eprint!("{}", String::from_utf8_lossy(&output.stderr));
        panic!(
            "the Metal compiler is not usable ({}): {TOOLCHAIN_HINT}",
            output.status
        );
    }
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .find_map(|line| line.strip_prefix("#define __METAL_VERSION__ "))
        .and_then(|version| version.trim().parse().ok())
}

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let shader_dir = Path::new("../shaders");
    // Sources are embedded only where a debugger can use them.
    let record_sources = env::var("PROFILE").is_ok_and(|profile| profile == "debug");

    println!("cargo:rerun-if-changed=../shaders/");
    println!("cargo:rerun-if-env-changed=DEVELOPER_DIR");
    println!("cargo:rerun-if-env-changed=SDKROOT");

    match metal_version() {
        Some(version) if version >= TENSOR_METAL_VERSION => {}
        version => println!(
            "cargo:warning=the macOS SDK compiles Metal {}, but the tensor matmul and \
             attention kernels need Metal 4.0: they are left out, so inference falls back \
             to the slower SIMD kernels; build with the macOS 26 SDK or newer",
            version.map_or_else(
                || "of an unknown version".to_owned(),
                |v| format!("{}.{}", v / 100, v % 100 / 10)
            )
        ),
    }

    // Start every compile, then wait for all of them. Their diagnostics go
    // straight to the build log: the probe above already caught a missing
    // toolchain, and piping twelve outputs could block a compiler. Collecting
    // is what starts every compile before the first wait.
    #[allow(clippy::needless_collect)]
    let compiles = SHADER_FILES
        .iter()
        .map(|shader| {
            let shader_path = shader_dir.join(shader);
            assert!(
                shader_path.is_file(),
                "listed shader {} is missing",
                shader_path.display()
            );
            let air_path = out_dir.join(shader.replace(".metal", ".air"));
            let mut command = xcrun();
            // Fast math (the default) also assumes no INF or NaN, but online
            // softmax starts from -INFINITY and the selectors must see
            // non-finite logits. Relaxed keeps the other freedoms.
            command.args(["metal", "-c", "-fmetal-math-mode=relaxed"]);
            if record_sources {
                command.arg("-frecord-sources");
            }
            command
                .arg("-I")
                .arg(shader_dir)
                .arg(&shader_path)
                .arg("-o")
                .arg(&air_path)
                .stdout(Stdio::null())
                .stderr(Stdio::inherit());
            let child = command.spawn().unwrap_or_else(|e| {
                panic!("cannot run xcrun to compile {shader} ({e}): {TOOLCHAIN_HINT}")
            });
            (shader, child, air_path)
        })
        .collect::<Vec<_>>();
    let air_files = compiles
        .into_iter()
        .map(|(shader, mut child, air_path)| {
            let status = child.wait().unwrap_or_else(|e| {
                panic!("waiting for the Metal compiler on {shader} failed: {e}")
            });
            assert!(
                status.success(),
                "compiling {shader} failed ({status}); see the compiler output above"
            );
            air_path
        })
        .collect::<Vec<_>>();

    let metallib_path = out_dir.join("shaders.metallib");
    let status = xcrun()
        .arg("metallib")
        .args(&air_files)
        .arg("-o")
        .arg(&metallib_path)
        .status()
        .unwrap_or_else(|e| panic!("cannot run xcrun metallib ({e}): {TOOLCHAIN_HINT}"));
    assert!(
        status.success(),
        "linking shaders.metallib failed ({status})"
    );
}
