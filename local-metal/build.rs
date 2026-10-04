#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let shader_dir = Path::new("../shaders");
    let sdk = "macosx";

    println!("cargo:rerun-if-changed=../shaders/");

    let shader_files: &[&str] = &[
        "bonsai.metal",
        "bonsai_ops.metal",
        "bonsai_small_batch.metal",
        "sampling.metal",
    ];

    let mut air_files = Vec::new();

    for shader in shader_files {
        let shader_path = shader_dir.join(shader);
        if !shader_path.exists() {
            continue;
        }
        let air_path = out_dir.join(shader.replace(".metal", ".air"));
        let status = Command::new("xcrun")
            .args([
                "-sdk",
                sdk,
                "metal",
                "-c",
                "-frecord-sources",
                "-I",
                shader_dir.to_str().unwrap(),
                shader_path.to_str().unwrap(),
                "-o",
                air_path.to_str().unwrap(),
            ])
            .status()
            .unwrap_or_else(|e| panic!("Failed to run xcrun metal compiler: {e}"));
        assert!(
            status.success(),
            "Metal shader compilation failed for {shader}"
        );
        air_files.push(air_path);
    }

    if air_files.is_empty() {
        return;
    }

    let metallib_path = out_dir.join("shaders.metallib");
    let mut cmd = Command::new("xcrun");
    cmd.args(["-sdk", sdk, "metallib"]);
    for air in &air_files {
        cmd.arg(air);
    }
    cmd.args(["-o", metallib_path.to_str().unwrap()]);
    let status = cmd
        .status()
        .unwrap_or_else(|e| panic!("Failed to run xcrun metallib: {e}"));
    assert!(status.success(), "Metal library linking failed");
}
