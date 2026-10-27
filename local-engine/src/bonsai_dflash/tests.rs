use std::path::{Path, PathBuf};

use local_metal::batch::CommandBatch;
use local_metal::buffer::MetalBuffer;
use local_metal::context::MetalContext;
use local_metal::dflash::DflashKernels;
use local_metal::shaders::ShaderLibrary;

use super::artifact::spec;
use super::calibration::fit_depth;
use super::tree::calibrated;
use super::*;

fn xorshift(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

fn uniform(state: &mut u64) -> f32 {
    (xorshift(state) >> 40) as f32 / (1_u64 << 24) as f32
}

/// A lattice whose conditionals are log-softmaxes of random logits at a
/// random temperature per row, so some rows are confident and some flat.
fn synthetic_lattice(state: &mut u64) -> DflashLattice {
    let mut lattice: DflashLattice = bytemuck::Zeroable::zeroed();
    for s in 0..DFLASH_SLOTS {
        for k in 0..DFLASH_TOPK {
            lattice.cand[s][k] = (s * 1000 + k * 7 + 1) as u32;
            lattice.logit[s][k] = -(k as f32);
        }
        for row in &mut lattice.cond_logq[s] {
            let temperature = 4.0_f32.mul_add(uniform(state), 0.3);
            let raw: Vec<f32> = (0..DFLASH_TOPK)
                .map(|_| uniform(state) * 6.0 / temperature)
                .collect();
            let peak = raw.iter().copied().fold(f32::MIN, f32::max);
            let sum: f32 = raw.iter().map(|x| (x - peak).exp()).sum();
            for (value, raw) in row.iter_mut().zip(&raw) {
                *value = raw - peak - sum.ln();
            }
        }
    }
    lattice
}

fn synthetic_calibration(state: &mut u64) -> DflashCalibration {
    let mut calibration = DflashCalibration::default();
    for d in 1..=DFLASH_SLOTS {
        calibration.alpha[d] = uniform(state).mul_add(1.6, -1.0);
        calibration.beta[d] = uniform(state).mul_add(0.8, 0.5);
    }
    calibration
}

/// `(depth, slot, parent)` per tree row.
type Shape = Vec<(u32, u32, i32)>;

struct SimCase {
    name: String,
    anchor: u32,
    lattice: DflashLattice,
    calibration: DflashCalibration,
    /// `(tag, [(depth, slot, parent)])`
    trees: Vec<(String, Shape)>,
}

/// Trees `sim.py`'s `build_tree` built over four real Metal-free fp32
/// lattices (prose, code, essay and agent anchors) with no calibration and
/// with that prompt's leave-one-out calibration.
fn sim_cases() -> Vec<SimCase> {
    let cases: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/sim_trees.json")).expect("fixture");
    let floats = |value: &serde_json::Value| -> Vec<f32> {
        value
            .as_array()
            .expect("array")
            .iter()
            .map(|v| v.as_f64().expect("number") as f32)
            .collect()
    };
    cases
        .as_array()
        .expect("cases")
        .iter()
        .map(|case| {
            let mut lattice: DflashLattice = bytemuck::Zeroable::zeroed();
            for (s, row) in case["cand"].as_array().expect("cand").iter().enumerate() {
                for (k, id) in row.as_array().expect("row").iter().enumerate() {
                    lattice.cand[s][k] = id.as_u64().expect("id") as u32;
                }
            }
            for (s, slot) in case["cond_logq"]
                .as_array()
                .expect("clq")
                .iter()
                .enumerate()
            {
                for (j, row) in slot.as_array().expect("slot").iter().enumerate() {
                    lattice.cond_logq[s][j].copy_from_slice(&floats(row));
                }
            }
            let mut calibration = DflashCalibration::default();
            calibration
                .alpha
                .copy_from_slice(&floats(&case["calibration"]["alpha"]));
            calibration
                .beta
                .copy_from_slice(&floats(&case["calibration"]["beta"]));
            let trees = case["trees"]
                .as_object()
                .expect("trees")
                .iter()
                .map(|(tag, nodes)| {
                    let nodes = nodes
                        .as_array()
                        .expect("nodes")
                        .iter()
                        .map(|node| {
                            let n = node.as_array().expect("node");
                            (
                                n[0].as_u64().expect("depth") as u32,
                                n[1].as_u64().expect("slot") as u32,
                                n[2].as_i64().expect("parent") as i32,
                            )
                        })
                        .collect();
                    (tag.clone(), nodes)
                })
                .collect();
            SimCase {
                name: case["name"].as_str().expect("name").to_owned(),
                anchor: case["anchor"].as_u64().expect("anchor") as u32,
                lattice,
                calibration,
                trees,
            }
        })
        .collect()
}

fn shape(nodes: &[TreeNode]) -> Shape {
    nodes
        .iter()
        .map(|node| (node.depth, node.slot, node.parent))
        .collect()
}

#[test]
fn reference_tree_matches_sim() {
    for case in sim_cases() {
        for (tag, expected) in &case.trees {
            let calibration = if tag == "raw" {
                DflashCalibration::default()
            } else {
                case.calibration
            };
            let nodes = build_tree_reference(&case.lattice, case.anchor, &calibration, 8);
            assert_eq!(&shape(&nodes), expected, "{} {tag}", case.name);
        }
    }
}

/// Every connected subtree holding the root with `extra` more nodes: the
/// largest sum of path probabilities among them. Each subtree is enumerated
/// once, by choosing frontier entries in increasing index order.
fn best_subtree(lattice: &DflashLattice, calibration: &DflashCalibration, extra: usize) -> f64 {
    #[derive(Clone)]
    struct Frontier {
        depth: usize,
        row: usize,
        slot: usize,
        score: f64,
    }
    fn walk(
        lattice: &DflashLattice,
        calibration: &DflashCalibration,
        frontier: &[Frontier],
        start: usize,
        left: usize,
        sum: f64,
    ) -> f64 {
        if left == 0 {
            return sum;
        }
        let mut best = f64::NEG_INFINITY;
        for index in start..frontier.len() {
            let node = &frontier[index];
            let mut next = frontier.to_vec();
            if node.depth < DFLASH_SLOTS {
                for (c, &lq) in lattice.cond_logq[node.depth][node.slot].iter().enumerate() {
                    let lq = f64::from(lq);
                    next.push(Frontier {
                        depth: node.depth + 1,
                        row: node.slot,
                        slot: c,
                        score: node.score
                            + calibrated(
                                lq,
                                calibration.alpha[node.depth + 1],
                                calibration.beta[node.depth + 1],
                            ),
                    });
                }
            }
            let _ = node.row;
            best = best.max(walk(
                lattice,
                calibration,
                &next,
                index + 1,
                left - 1,
                sum + node.score.exp(),
            ));
        }
        best
    }
    let root: Vec<Frontier> = (0..DFLASH_TOPK)
        .map(|c| Frontier {
            depth: 1,
            row: 0,
            slot: c,
            score: calibrated(
                f64::from(lattice.cond_logq[0][0][c]),
                calibration.alpha[1],
                calibration.beta[1],
            ),
        })
        .collect();
    walk(lattice, calibration, &root, 0, extra, 0.0)
}

#[test]
fn reference_tree_is_prefix_optimal_with_parents_first() {
    let mut state = 0x9e37_79b9_7f4a_7c15;
    for trial in 0..6 {
        let lattice = synthetic_lattice(&mut state);
        let calibration = if trial % 2 == 0 {
            DflashCalibration::default()
        } else {
            synthetic_calibration(&mut state)
        };
        let full = build_tree_reference(&lattice, 7, &calibration, DFLASH_MAX_TREE_ROWS);
        assert_eq!(full.len(), DFLASH_MAX_TREE_ROWS);
        for (row, node) in full.iter().enumerate().skip(1) {
            let parent = node.parent as usize;
            assert!(parent < row, "parent after child");
            assert_eq!(node.depth, full[parent].depth + 1);
            assert!(node.path_log_p <= full[parent].path_log_p);
            assert_eq!(
                node.token,
                lattice.cand[node.depth as usize - 1][node.slot as usize]
            );
        }
        for rows in 1..=DFLASH_MAX_TREE_ROWS {
            let small = build_tree_reference(&lattice, 7, &calibration, rows);
            assert_eq!(small, full[..rows], "the {rows}-row tree is the prefix");
        }
        for extra in 1..=3 {
            let built: f64 = full[1..=extra].iter().map(|n| n.path_log_p.exp()).sum();
            let best = best_subtree(&lattice, &calibration, extra);
            assert!(
                (built - best).abs() <= 1e-12 * best.max(1.0),
                "trial {trial}, {extra} nodes: built {built}, best {best}"
            );
        }
    }
}

#[test]
fn tested_nodes_follow_the_accepted_path() {
    // anchor(0) -> a(1) -> b(2); a -> c(3); anchor -> d(4)
    let mut tree: DflashTree = bytemuck::Zeroable::zeroed();
    tree.rows = 5;
    let rows = [(-1, 0, 100), (0, 1, 10), (1, 2, 20), (1, 2, 30), (0, 1, 40)];
    for (row, &(parent, depth, token)) in rows.iter().enumerate() {
        tree.parent[row] = parent;
        tree.depth[row] = depth;
        tree.token[row] = token;
        tree.log_q[row] = -0.1 * row as f32;
    }
    // The target takes a, then c, then something else.
    let (accepted, records) = tested_nodes(&tree, 5, &[10, 30, 99]);
    assert_eq!(accepted, 2);
    let summary: Vec<_> = records.iter().map(|&(d, _, hit)| (d, hit)).collect();
    assert_eq!(summary, [(1, true), (1, false), (2, false), (2, true)]);
    // Only the first three rows verified: c is not in the tree.
    let (accepted, records) = tested_nodes(&tree, 3, &[10, 30]);
    assert_eq!(accepted, 1);
    assert_eq!(records.len(), 2);
    // The target rejects every first draft.
    let (accepted, records) = tested_nodes(&tree, 5, &[55]);
    assert_eq!((accepted, records.len()), (0, 2));
}

#[test]
fn calibration_fit_recovers_a_logistic() {
    let mut state = 0x1234_5678_9abc_def1;
    let (alpha, beta) = (-0.6_f32, 0.75_f32);
    let records: Vec<(f32, bool)> = (0..2048)
        .map(|_| {
            let x = uniform(&mut state).mul_add(10.0, -6.0);
            let p = 1.0 / (1.0 + (-beta.mul_add(x, alpha)).exp());
            (x, uniform(&mut state) < p)
        })
        .collect();
    let (a, b) = fit_depth(&records, (0.0, 1.0));
    assert!(
        (a - alpha).abs() < 0.15 && (b - beta).abs() < 0.1,
        "{a} {b}"
    );
    // Too few records, or of one outcome, keep the current fit.
    assert_eq!(fit_depth(&records[..40], (0.3, 0.9)), (0.3, 0.9));
    let all_hits: Vec<_> = records.iter().map(|&(x, _)| (x, true)).collect();
    assert_eq!(fit_depth(&all_hits, (0.0, 1.0)), (0.0, 1.0));
}

#[test]
fn calibrator_refits_every_sixteen_cycles_and_persists() {
    let directory = tempfile::tempdir().expect("temp dir");
    let path = directory.path().join("calibration.json");
    let mut calibrator = Calibrator::persistent(path.clone());
    assert_eq!(*calibrator.calibration(), DflashCalibration::default());
    // A chain tree whose first draft the target takes a third of the time,
    // whatever the drafter's probability (0.1 to 0.9): calibration must pull
    // p toward 1/3.
    let mut tree: DflashTree = bytemuck::Zeroable::zeroed();
    tree.rows = 3;
    tree.parent = [-1, 0, 1, -1, -1, -1, -1, -1];
    tree.depth = [0, 1, 2, 0, 0, 0, 0, 0];
    tree.token = [1, 2, 3, 0, 0, 0, 0, 0];
    let q = |cycle: usize| 0.1_f32.mul_add((cycle % 9) as f32, 0.1);
    let mut refits = Vec::new();
    for cycle in 0..96 {
        tree.log_q[1] = q(cycle).ln();
        let target: &[u32] = if cycle % 3 == 0 { &[2, 9] } else { &[9] };
        let (_, refit) = calibrator.record_cycle(&tree, 3, target);
        refits.push(refit);
    }
    // Depth 1 reaches 50 records at cycle 50; the first refit after is 63.
    assert!(refits[..63].iter().all(|refit| !refit));
    assert!(refits[63]);
    assert!(
        refits
            .iter()
            .enumerate()
            .all(|(i, &r)| !r || (i + 1) % REFIT_CYCLES == 0)
    );
    let fitted = *calibrator.calibration();
    let mean_p = (0..9)
        .map(|cycle| calibrated(f64::from(q(cycle).ln()), fitted.alpha[1], fitted.beta[1]).exp())
        .sum::<f64>()
        / 9.0;
    assert!((mean_p - 1.0 / 3.0).abs() < 0.05, "{mean_p}");
    assert_eq!(fitted.alpha[2], 0.0, "depth 2 has too few records");
    let reloaded = Calibrator::persistent(path);
    assert_eq!(*reloaded.calibration(), fitted);
}

#[test]
fn calibration_cache_path_names_model_drafter_and_chip() {
    let path = calibration_cache_path(
        "Ternary-Bonsai-2-27B-PTQ1_0",
        &"ab".repeat(32),
        "Apple M4 Pro",
    )
    .expect("HOME");
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .expect("name");
    assert_eq!(
        name,
        "Ternary-Bonsai-2-27B-PTQ1-0-abababababababab-Apple-M4-Pro.json"
    );
    assert!(
        path.to_string_lossy()
            .contains("Library/Caches/local-ai/dflash")
    );
}

#[test]
fn policy_requires_the_artifact() {
    let missing = Path::new("/nonexistent/dflash2-drafter-int8-v1.bin");
    let error = DflashMode::Auto
        .resolve_with_default(missing)
        .expect_err("a missing drafter must stop startup");
    assert!(error.to_string().contains("dflash2-drafter-int8-v1.bin"), "{error}");
    assert!(DflashMode::Drafter(missing.to_owned()).resolve().is_err());
    assert_eq!(
        DflashMode::Off("off by request".into()).resolve().ok(),
        Some(DflashResolution::Fallback("off by request".into()))
    );
    assert_eq!(
        Path::new(DEFAULT_BONSAI_DFLASH_ARTIFACT).file_name(),
        Some(std::ffi::OsStr::new(DFLASH_HEAD_ARTIFACT))
    );
}

#[test]
fn artifact_spec_names_every_tensor_once() {
    let sections = spec();
    assert_eq!(sections.len(), 128);
    let mut names: Vec<_> = sections.iter().map(|(name, _)| name.as_str()).collect();
    names.sort_unstable();
    names.dedup();
    assert_eq!(names.len(), sections.len());
    // The header (fixed fields, the name table and offsets) fits its page.
    let header = 8
        + 4
        + 4
        + 8
        + 8
        + 64
        + 4
        + sections
            .iter()
            .map(|(name, _)| 1 + name.len())
            .sum::<usize>()
        + 16 * sections.len();
    assert!(header <= crate::bonsai_mtp::PAGE, "{header}");
    let payload: usize = sections.iter().map(|(_, bytes)| bytes).sum();
    assert_eq!(payload, 2_053_443_584);
}

/// `None` when this machine has no Metal device.
fn gpu() -> Option<(MetalContext, ShaderLibrary)> {
    let context = MetalContext::new().ok()?;
    let shaders = ShaderLibrary::new(context.device()).expect("shaders");
    Some((context, shaders))
}

fn gpu_tree(
    context: &MetalContext,
    kernels: &DflashKernels,
    lattice: &DflashLattice,
    anchor: u32,
    calibration: &DflashCalibration,
    rows: usize,
) -> DflashTree {
    let device = context.device();
    let lattice = MetalBuffer::from_slice(device, std::slice::from_ref(lattice)).expect("lattice");
    let tokens = MetalBuffer::from_slice(device, &[anchor, MASK_TOKEN]).expect("tokens");
    let calibration =
        MetalBuffer::from_slice(device, std::slice::from_ref(calibration)).expect("calibration");
    let tree = MetalBuffer::empty(device, size_of::<DflashTree>()).expect("tree");
    let mut batch = CommandBatch::new(context).expect("batch");
    kernels
        .tree(&mut batch, &lattice, &tokens, &calibration, &tree, rows)
        .expect("encode tree");
    batch.commit_and_wait().expect("tree");
    // SAFETY: the batch completed; nothing else uses the buffer.
    unsafe { tree.as_slice::<DflashTree>()[0] }
}

fn assert_same_tree(gpu: &DflashTree, reference: &[TreeNode], label: &str) {
    assert_eq!(gpu.rows as usize, reference.len(), "{label}");
    let mut expected = 1.0_f64;
    for (row, node) in reference.iter().enumerate() {
        assert_eq!(gpu.parent[row], node.parent, "{label} row {row}");
        assert_eq!(gpu.depth[row], node.depth, "{label} row {row}");
        assert_eq!(gpu.slot[row], node.slot, "{label} row {row}");
        assert_eq!(gpu.token[row], node.token, "{label} row {row}");
        assert!(
            (f64::from(gpu.path_log_p[row]) - node.path_log_p).abs() < 1e-4,
            "{label}"
        );
        if row > 0 {
            expected += node.path_log_p.exp();
            let parent = node.parent as usize;
            assert_eq!(
                gpu.ancestors[row],
                gpu.ancestors[parent] | (1 << row),
                "{label}"
            );
        }
        assert!(
            (f64::from(gpu.expected[row]) - expected).abs() < 1e-4,
            "{label} E({}) {} against {expected}",
            row + 1,
            gpu.expected[row]
        );
    }
}

#[test]
fn gpu_tree_matches_the_reference() {
    let Some((context, shaders)) = gpu() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let kernels = DflashKernels::new(&context, &shaders).expect("kernels");
    let mut state = 0x0bad_5eed_1234_5678;
    for trial in 0..12 {
        let lattice = synthetic_lattice(&mut state);
        let calibration = if trial % 3 == 0 {
            DflashCalibration::default()
        } else {
            synthetic_calibration(&mut state)
        };
        for rows in 1..=DFLASH_MAX_TREE_ROWS {
            let reference = build_tree_reference(&lattice, 42, &calibration, rows);
            let tree = gpu_tree(&context, &kernels, &lattice, 42, &calibration, rows);
            assert_same_tree(&tree, &reference, &format!("trial {trial}, {rows} rows"));
        }
    }
    for case in sim_cases() {
        for (tag, expected) in &case.trees {
            let calibration = if tag == "raw" {
                DflashCalibration::default()
            } else {
                case.calibration
            };
            let reference = build_tree_reference(&case.lattice, case.anchor, &calibration, 8);
            let tree = gpu_tree(
                &context,
                &kernels,
                &case.lattice,
                case.anchor,
                &calibration,
                8,
            );
            assert_same_tree(&tree, &reference, &format!("{} {tag}", case.name));
            assert_eq!(&shape(&reference), expected);
        }
    }
}

fn trace_dir() -> Option<PathBuf> {
    std::env::var_os("DFLASH_TRACE").map(PathBuf::from)
}

/// The installed artifact loads, and a copy with one flipped payload byte or
/// a truncated copy is refused with the reason.
#[test]
#[ignore = "requires the installed DFlash2 drafter and a Metal device"]
fn installed_drafter_loads_and_corruption_is_refused() {
    let (context, shaders) = gpu().expect("Metal");
    let installed = Path::new(DEFAULT_BONSAI_DFLASH_ARTIFACT);
    let drafter = DflashDrafter::load(&context, &shaders, installed).expect("installed drafter");
    eprintln!(
        "loaded {} ({} weight bytes, sha256 {})",
        drafter.path().display(),
        drafter.weight_bytes(),
        drafter.payload_sha256()
    );
    drop(drafter);
    let directory = tempfile::tempdir_in(installed.parent().expect("models dir")).expect("temp");
    let corrupt = directory.path().join(DFLASH_HEAD_ARTIFACT);
    // An APFS clone: one modified page, not a 2 GB copy.
    let status = std::process::Command::new("cp")
        .arg("-c")
        .arg(installed)
        .arg(&corrupt)
        .status()
        .expect("cp");
    assert!(status.success());
    {
        use std::io::{Read, Seek, SeekFrom, Write};
        let mut file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(&corrupt)
            .expect("open clone");
        let at = 1 << 30;
        let mut byte = [0_u8];
        file.seek(SeekFrom::Start(at)).expect("seek");
        file.read_exact(&mut byte).expect("read");
        file.seek(SeekFrom::Start(at)).expect("seek");
        file.write_all(&[byte[0] ^ 0x40]).expect("write");
    }
    let error = DflashDrafter::load(&context, &shaders, &corrupt)
        .err()
        .expect("corrupt drafter refused")
        .to_string();
    assert!(error.contains("payload hashes to"), "{error}");
    let truncated = directory.path().join("truncated.bin");
    let bytes = std::fs::read(installed.parent().expect("dir").join(DFLASH_HEAD_ARTIFACT))
        .map(|bytes| bytes[..1 << 20].to_vec())
        .expect("read head");
    std::fs::write(&truncated, bytes).expect("write truncated");
    let error = DflashDrafter::load(&context, &shaders, &truncated)
        .err()
        .expect("truncated drafter refused")
        .to_string();
    assert!(error.contains("header records"), "{error}");
    let foreign = DflashDrafter::load(
        &context,
        &shaders,
        Path::new(crate::bonsai_mtp::DEFAULT_BONSAI_MTP_ARTIFACT),
    )
    .err()
    .expect("MTP head refused")
    .to_string();
    assert!(
        foreign.contains("is not a DFlash2 drafter artifact"),
        "{foreign}"
    );
}

/// The parity and acceptance harness: on every captured trace prompt
/// (`$DFLASH_TRACE/caps*/`), feed the drafter the target's taps one
/// committed position at a time, as decoding would, draft at every anchor,
/// and write each anchor's lattice and 8-row tree to
/// `$DFLASH_OUT/<prompt>.b8.bin` for the Python comparison against the fp32
/// reference (`drafter_lib.py`) and the acceptance simulation (`sim.py`).
/// Also reports the measured GPU time of the passes.
#[test]
#[ignore = "requires the Bonsai GGUF, the installed drafter, the trace captures and a Metal device"]
#[expect(clippy::too_many_lines)]
fn drafter_lattices_on_captures() {
    use local_metal::bonsai::BonsaiKernels;
    use std::io::Write;

    let Some(trace) = trace_dir() else {
        eprintln!("skipping: set DFLASH_TRACE to the trace directory");
        return;
    };
    let out = PathBuf::from(std::env::var_os("DFLASH_OUT").expect("DFLASH_OUT"));
    std::fs::create_dir_all(&out).expect("output dir");
    let only: Option<String> = std::env::var("DFLASH_ONLY").ok();
    let limit: usize = std::env::var("DFLASH_LIMIT")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(usize::MAX);
    let package = crate::bonsai::BonsaiPackage::open(crate::bonsai::DEFAULT_BONSAI_GGUF)
        .expect("open Bonsai GGUF");
    let (context, shaders) = gpu().expect("Metal");
    let kernels = BonsaiKernels::new(&context, &shaders).expect("kernels");
    let embedding = package
        .metal_tensor(context.device(), "token_embd.weight")
        .expect("embedding");
    let output = package
        .metal_tensor(context.device(), "output.weight")
        .expect("output");
    let rotation = package.hadamard().upload(&context, 5120).expect("signs");
    let shared = DflashShared {
        kernels: &kernels,
        embedding: embedding.ptq1_matrix().expect("embedding matrix"),
        output: output.ptq1_matrix().expect("output matrix"),
        rotation: &rotation,
    };
    let mut drafter = DflashDrafter::load(
        &context,
        &shaders,
        Path::new(DEFAULT_BONSAI_DFLASH_ARTIFACT),
    )
    .expect("drafter");
    let captures = [
        ("caps/000", "prose"),
        ("caps/001", "explanation"),
        ("caps/002", "arithmetic"),
        ("caps/003", "code"),
        ("caps/004", "essay"),
        ("caps/005", "thinking"),
        ("caps-agent/000", "agent"),
    ];
    let mut draft_times = Vec::new();
    let mut append_times = Vec::new();
    for (capture, name) in captures {
        if only.as_deref().is_some_and(|only| only != name) {
            continue;
        }
        let meta: serde_json::Value = serde_json::from_slice(
            &std::fs::read(trace.join(format!("{capture}.json"))).expect("capture json"),
        )
        .expect("capture meta");
        let ids = |key: &str| -> Vec<u32> {
            meta[key]
                .as_array()
                .expect("ids")
                .iter()
                .map(|id| id.as_u64().expect("id") as u32)
                .collect()
        };
        let prompt = ids("prompt_ids");
        let tokens: Vec<u32> = prompt.iter().copied().chain(ids("gen_ids")).collect();
        let (n, n0) = (tokens.len(), prompt.len());
        // `<capture>.taps` is F32 [6, n, 5120]: the embedding, then the five tap layers.
        let raw = std::fs::read(trace.join(format!("{capture}.taps"))).expect("taps");
        assert_eq!(raw.len(), 6 * n * 5120 * 4, "{name} taps size");
        let floats: &[f32] = bytemuck::cast_slice(&raw);
        let mut taps = vec![0.0_f32; n * 5 * 5120];
        for position in 0..n {
            for layer in 0..5 {
                let from = ((layer + 1) * n + position) * 5120;
                let to = (position * 5 + layer) * 5120;
                taps[to..to + 5120].copy_from_slice(&floats[from..from + 5120]);
            }
        }
        let taps = MetalBuffer::from_slice(context.device(), &taps).expect("taps buffer");
        let row_bytes = 5 * 5120 * 4;
        drafter.reset();
        let mut batch = CommandBatch::new(&context).expect("batch");
        drafter
            .append_context(&mut batch, &shared, &taps, 0, n0, TapFormat::F32)
            .expect("prompt context");
        batch.commit_and_wait().expect("prompt context");
        let mut file = std::io::BufWriter::new(
            std::fs::File::create(out.join(format!("{name}.b8.bin"))).expect("output file"),
        );
        let anchors = (n0..n - 1).take(limit);
        for position in anchors {
            let elapsed = drafter
                .draft_and_wait(
                    &context,
                    &shared,
                    &DraftAnchor::Token(tokens[position]),
                    position,
                    DFLASH_MAX_TREE_ROWS,
                )
                .expect("draft");
            draft_times.push(elapsed.as_secs_f64() * 1e3);
            let lattice = drafter.read_lattice();
            let tree = drafter.read_tree();
            if position == n0 {
                // The same pass with the anchor written on the GPU.
                let anchor = MetalBuffer::from_slice(context.device(), &[0, tokens[position]])
                    .expect("anchor buffer");
                let device = DraftAnchor::Device {
                    buffer: &anchor,
                    offset: 4,
                };
                drafter
                    .draft_and_wait(&context, &shared, &device, position, DFLASH_MAX_TREE_ROWS)
                    .expect("device-anchor draft");
                assert_eq!(
                    bytemuck::bytes_of(&drafter.read_lattice()),
                    bytemuck::bytes_of(&lattice),
                    "{name}: a GPU anchor drafts what a host anchor does"
                );
                assert_eq!(drafter.read_tree(), tree);
            }
            file.write_all(&(position as u32).to_le_bytes())
                .expect("write");
            file.write_all(bytemuck::bytes_of(&lattice)).expect("write");
            file.write_all(bytemuck::bytes_of(&tree)).expect("write");
            let mut batch = CommandBatch::new(&context).expect("batch");
            drafter
                .append_context(
                    &mut batch,
                    &shared,
                    &taps,
                    position * row_bytes,
                    1,
                    TapFormat::F32,
                )
                .expect("append");
            let elapsed = batch.commit_async().wait_with_gpu_time().expect("append");
            append_times.push(elapsed.as_secs_f64() * 1e3);
        }
        file.flush().expect("flush");
        eprintln!("{name}: {} anchors", n - 1 - n0);
    }
    let median = |times: &mut Vec<f64>| {
        times.sort_by(f64::total_cmp);
        times.get(times.len() / 2).copied().unwrap_or(0.0)
    };
    let passes = draft_times.len();
    eprintln!(
        "draft pass GPU ms: median {:.3} over {passes}; one-row context append GPU ms: median {:.3}",
        median(&mut draft_times),
        median(&mut append_times)
    );
}
