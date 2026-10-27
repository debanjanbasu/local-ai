//! Per-position calibration of the drafter's probabilities into acceptance
//! probabilities, refitted from what the target accepted.
//!
//! The drafter's conditional probability `q` of a node is not the chance the
//! target accepts it; for each depth `d` the calibration is a logistic
//! regression `p = σ(α_d + β_d · logit q)` over the nodes the target tested
//! (those whose parent it accepted), starting from `p = q` (`α = 0, β = 1`),
//! refitted every [`REFIT_CYCLES`] cycles on the last
//! [`CALIBRATION_RECORDS`] records of each depth, and persisted per model,
//! drafter and chip, so the next start begins from what this Mac learned.

use std::collections::VecDeque;
use std::path::{Path, PathBuf};

use local_metal::dflash::{DFLASH_MAX_TREE_ROWS, DFLASH_SLOTS, DflashCalibration, DflashTree};

use super::tree::tested_nodes;

/// Cycles between refits.
pub const REFIT_CYCLES: usize = 16;
/// Records kept per depth: a refit sees the most recent ones.
pub const CALIBRATION_RECORDS: usize = 2048;
/// A depth is refitted only with this many records, and this many of each
/// outcome; with fewer it keeps its current fit.
const MIN_RECORDS: usize = 50;
const MIN_OUTCOMES: usize = 5;

/// Fit `p = σ(α + β x)` to `(x = logit q, accepted)` records by Newton's
/// method (a 1e-6 ridge keeps it defined), as `sim.py`'s `fit_logistic`.
/// Too few records, or of one outcome, return `current`.
#[must_use]
pub fn fit_depth(records: &[(f32, bool)], current: (f32, f32)) -> (f32, f32) {
    let hits = records.iter().filter(|(_, hit)| *hit).count();
    if records.len() < MIN_RECORDS || hits < MIN_OUTCOMES || records.len() - hits < MIN_OUTCOMES {
        return current;
    }
    let (mut alpha, mut beta) = (0.0_f64, 1.0_f64);
    for _ in 0..50 {
        let (mut grad0, mut grad1) = (0.0_f64, 0.0_f64);
        let (mut h00, mut h01, mut h11) = (1e-6_f64, 0.0_f64, 1e-6_f64);
        for &(logit, hit) in records {
            let logit = f64::from(logit);
            let prob = 1.0 / (1.0 + (-beta.mul_add(logit, alpha)).exp());
            let weight = prob.mul_add(1.0 - prob, 1e-9);
            let residual = prob - f64::from(u8::from(hit));
            grad0 += residual;
            grad1 = residual.mul_add(logit, grad1);
            h00 += weight;
            h01 = weight.mul_add(logit, h01);
            h11 = (weight * logit).mul_add(logit, h11);
        }
        let det = h00.mul_add(h11, -(h01 * h01));
        if det.abs() < 1e-300 {
            return current;
        }
        let step0 = h11.mul_add(grad0, -(h01 * grad1)) / det;
        let step1 = h00.mul_add(grad1, -(h01 * grad0)) / det;
        alpha -= step0;
        beta -= step1;
        if !(alpha.is_finite() && beta.is_finite()) {
            return current;
        }
        if step0.abs().max(step1.abs()) < 1e-8 {
            break;
        }
    }
    (alpha as f32, beta as f32)
}

/// The calibration cache file for one model, drafter and chip:
/// `~/Library/Caches/local-ai/dflash/<model>-<drafter digest prefix>-<chip>.json`.
#[must_use]
pub fn calibration_cache_path(model: &str, drafter_sha256: &str, chip: &str) -> Option<PathBuf> {
    let clean = |text: &str| -> String {
        text.chars()
            .map(|c| {
                if c.is_ascii_alphanumeric() || c == '.' {
                    c
                } else {
                    '-'
                }
            })
            .collect()
    };
    let home = std::env::var_os("HOME")?;
    Some(
        PathBuf::from(home)
            .join("Library/Caches/local-ai/dflash")
            .join(format!(
                "{}-{}-{}.json",
                clean(model),
                clean(drafter_sha256.get(..16).unwrap_or(drafter_sha256)),
                clean(chip)
            )),
    )
}

/// Acceptance records and the calibration fitted to them.
pub struct Calibrator {
    /// Index `d` holds depth `d`'s `(logit q, accepted)` records.
    records: Vec<VecDeque<(f32, bool)>>,
    cycles: usize,
    calibration: DflashCalibration,
    path: Option<PathBuf>,
}

impl Default for Calibrator {
    fn default() -> Self {
        Self {
            records: vec![VecDeque::new(); DFLASH_MAX_TREE_ROWS],
            cycles: 0,
            calibration: DflashCalibration::default(),
            path: None,
        }
    }
}

impl Calibrator {
    /// A calibrator persisted at `path`, starting from the fit saved there
    /// (or from `p = q` when there is none, or it does not parse).
    #[must_use]
    pub fn persistent(path: PathBuf) -> Self {
        let mut calibrator = Self::default();
        if let Some(saved) = std::fs::read(&path)
            .ok()
            .and_then(|bytes| serde_json::from_slice::<serde_json::Value>(&bytes).ok())
        {
            let read = |key: &str| -> Option<Vec<f32>> {
                saved[key]
                    .as_array()?
                    .iter()
                    .map(|v| v.as_f64().map(|v| v as f32))
                    .collect()
            };
            if let (Some(alpha), Some(beta)) = (read("alpha"), read("beta"))
                && alpha.len() == DFLASH_MAX_TREE_ROWS
                && beta.len() == DFLASH_MAX_TREE_ROWS
                && alpha.iter().chain(&beta).all(|v| v.is_finite())
            {
                calibrator.calibration.alpha.copy_from_slice(&alpha);
                calibrator.calibration.beta.copy_from_slice(&beta);
            }
        }
        calibrator.path = Some(path);
        calibrator
    }

    /// The current fit, for [`super::DflashDrafter::set_calibration`].
    #[must_use]
    pub const fn calibration(&self) -> &DflashCalibration {
        &self.calibration
    }

    /// Where the fit persists, if anywhere.
    #[must_use]
    pub fn path(&self) -> Option<&Path> {
        self.path.as_deref()
    }

    /// Records held for `depth`.
    #[must_use]
    pub fn records(&self, depth: usize) -> usize {
        self.records.get(depth).map_or(0, VecDeque::len)
    }

    /// One tested node at `depth` (1..=7) with conditional log probability
    /// `log_q`, and whether the target accepted it.
    pub fn record(&mut self, depth: usize, log_q: f32, accepted: bool) {
        if !(1..=DFLASH_SLOTS).contains(&depth) {
            return;
        }
        let q = log_q.exp().clamp(1e-7, 1.0 - 1e-6);
        let records = &mut self.records[depth];
        if records.len() == CALIBRATION_RECORDS {
            records.pop_front();
        }
        records.push_back((q.ln() - (-q).ln_1p(), accepted));
    }

    /// Record one verified cycle (see [`tested_nodes`]: `rows` rows of
    /// `tree` verified, `target` the target's tokens along the accepted
    /// path), refitting every [`REFIT_CYCLES`] cycles. Returns the drafts
    /// accepted and whether the calibration changed, in which case upload it.
    pub fn record_cycle(
        &mut self,
        tree: &DflashTree,
        rows: usize,
        target: &[u32],
    ) -> (usize, bool) {
        let (accepted, records) = tested_nodes(tree, rows, target);
        for (depth, log_q, hit) in records {
            self.record(depth, log_q, hit);
        }
        self.cycles += 1;
        let refit = self.cycles.is_multiple_of(REFIT_CYCLES) && self.refit();
        (accepted, refit)
    }

    /// Refit every depth from its records and persist the result; returns
    /// whether anything changed.
    pub fn refit(&mut self) -> bool {
        let mut changed = false;
        for depth in 1..=DFLASH_SLOTS {
            let records: Vec<_> = self.records[depth].iter().copied().collect();
            let current = (self.calibration.alpha[depth], self.calibration.beta[depth]);
            let fitted = fit_depth(&records, current);
            if fitted != current {
                (self.calibration.alpha[depth], self.calibration.beta[depth]) = fitted;
                changed = true;
            }
        }
        if changed && let Err(error) = self.save() {
            eprintln!("dflash: cannot persist the calibration: {error}");
        }
        changed
    }

    /// Write the fit to its cache file, if it has one.
    pub fn save(&self) -> std::io::Result<()> {
        let Some(path) = &self.path else {
            return Ok(());
        };
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let body = serde_json::json!({
            "alpha": self.calibration.alpha,
            "beta": self.calibration.beta,
        });
        let temp = path.with_extension(format!("{}.tmp", std::process::id()));
        std::fs::write(&temp, body.to_string())?;
        std::fs::rename(&temp, path)
    }
}
