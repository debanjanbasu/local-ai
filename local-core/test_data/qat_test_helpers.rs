pub fn config_path() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("test_data/gemma4_e2b_qat_config.json")
}
