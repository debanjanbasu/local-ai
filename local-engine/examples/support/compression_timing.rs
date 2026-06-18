use std::path::Path;
use std::time::Instant;

use local_engine::compress_pipeline::{
    CompressionPlan, CompressionReportError, CompressionStage, CompressionStageTiming,
};

#[allow(dead_code)]
#[must_use]
pub fn file_size(path: &Path) -> u64 {
    std::fs::metadata(path).map_or(0, |metadata| metadata.len())
}

#[allow(dead_code)]
#[must_use]
pub fn stage_timing_since(
    run_start: Instant,
    stage: CompressionStage,
    started_at: Instant,
    finished_at: Instant,
    input_bytes: u64,
    output_bytes: u64,
) -> CompressionStageTiming {
    CompressionStageTiming::new(
        stage,
        started_at.saturating_duration_since(run_start),
        finished_at.saturating_duration_since(run_start),
        input_bytes,
        output_bytes,
    )
}

pub fn render_stage_timing_report(
    label: &str,
    plan: &CompressionPlan,
    timings: &[CompressionStageTiming],
) -> Result<String, CompressionReportError> {
    let report = plan.stage_timing_report(timings)?;
    Ok(format!(
        "{label} timing stages: {:?}\n{label} overlaps CPU/GPU: {}\n{report}",
        plan.stages,
        plan.overlaps_cpu_gpu_work(),
    ))
}
