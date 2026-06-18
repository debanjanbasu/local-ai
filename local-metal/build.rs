#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Xcode SDK matching the cargo target: macOS, iOS device, or iOS simulator.
fn metal_sdk() -> &'static str {
    let os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let abi = env::var("CARGO_CFG_TARGET_ABI").unwrap_or_default();
    match (os.as_str(), abi.as_str()) {
        ("ios", "sim") => "iphonesimulator",
        ("ios", _) => "iphoneos",
        _ => "macosx",
    }
}

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let shader_dir = Path::new("../shaders");
    let sdk = metal_sdk();

    println!("cargo:rerun-if-changed=../shaders/");

    let shader_files: &[&str] = &[
        "rms_norm.metal",
        "gelu.metal",
        "softmax.metal",
        "embedding.metal",
        "rope.metal",
        "matvec_f16.metal",
        "matvec_quant.metal",
        "attention_sliding.metal",
        "attention_flash.metal",
        "attention_flash_tq.metal",
        "silu.metal",
        "elementwise_mul.metal",
        // MoE / mamba / IQ shaders — removed (not needed for Gemma4 QAT)
        "residual_add.metal",
        "scale_in_place.metal",
        "qk_norm.metal",
        "clipped_linear.metal",
        "vision.metal",
        "audio.metal",
        "matmul_f16.metal",
        "flash_attention_prefill.metal",
        "flash_attention_prefill_masked.metal",
        "bf16_to_fp16.metal",
        "dequantize_kv.metal",
        "flash_decoding.metal",
        "tq2_dequant.metal",
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
