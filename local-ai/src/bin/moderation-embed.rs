//! Features and native scores for the moderation heads (see
//! `docs/MODERATION.md`).
//!
//! ```text
//! moderation-embed features DATASET.jsonl OUT_PREFIX [--start N] [--count M]
//! moderation-embed score TEXTS.jsonl
//! ```
//!
//! The encoder loads from `models/embeddinggemma2-text` under the working
//! directory.
//!
//! - `features`: rows `start..start+count` of a `tools/moderation_data.py`
//!   dataset (`{"id", "text", ...}` per line). Each text gets the moderation
//!   prompt and its first window (exactly what the server scores for inputs
//!   up to one window). Writes `OUT_PREFIX.f32` (768 little-endian F32 per
//!   row, L2-normalized) and `OUT_PREFIX.json` (`ids`, token counts, window
//!   counts, the encoder payload SHA-256).
//! - `score`: `{"text"}` per line; prints the installed artifact's
//!   per-category scores and flags per line, as the server computes them
//!   (all windows, maximum per category).

use std::io::Write as _;
use std::path::Path;
use std::time::Instant;

use local_engine::embedding::{EmbeddingModel, installed_model};
use local_engine::moderation::{self, CATEGORIES, ModerationHeads, installed_heads};
use serde_json::{Value, json};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

fn open() -> Result<EmbeddingModel> {
    let dir = installed_model(Path::new("models"))
        .ok_or("models/embeddinggemma2-text is not installed (see docs/EMBEDDINGS.md)")?;
    Ok(EmbeddingModel::open(&dir)?)
}

fn rows(path: &str) -> Result<Vec<Value>> {
    Ok(std::fs::read_to_string(path)?
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(serde_json::from_str)
        .collect::<std::result::Result<_, _>>()?)
}

fn flag(args: &[String], name: &str) -> Result<Option<usize>> {
    match args.iter().position(|a| a == name) {
        Some(i) => Ok(Some(args.get(i + 1).ok_or("missing value")?.parse()?)),
        None => Ok(None),
    }
}

fn features(dataset: &str, prefix: &str, args: &[String]) -> Result<()> {
    let all = rows(dataset)?;
    let start = flag(args, "--start")?.unwrap_or(0).min(all.len());
    let count = flag(args, "--count")?.unwrap_or(all.len());
    let rows = &all[start..(start + count).min(all.len())];
    let mut model = open()?;
    let started = Instant::now();
    let mut ids = Vec::with_capacity(rows.len());
    let mut first = Vec::with_capacity(rows.len());
    let (mut tokens, mut window_counts) = (Vec::new(), Vec::new());
    for row in rows {
        let text = row["text"].as_str().ok_or("row without text")?;
        let windows = moderation::windows(&model, text)?;
        tokens.push(windows[0].len());
        window_counts.push(windows.len());
        first.push(windows.into_iter().next().unwrap_or_default());
        ids.push(row["id"].clone());
    }
    let mut out = std::io::BufWriter::new(std::fs::File::create(format!("{prefix}.f32"))?);
    for batch in first.chunks(256) {
        let vectors = model.embed_ids(batch, &mut || Ok(()))?;
        for vector in vectors {
            for value in vector {
                out.write_all(&value.to_le_bytes())?;
            }
        }
    }
    out.flush()?;
    let total: usize = tokens.iter().sum();
    std::fs::write(
        format!("{prefix}.json"),
        serde_json::to_vec(&json!({
            "start":start,"rows":rows.len(),"ids":ids,"tokens":tokens,"windows":window_counts,
            "embedding_payload_sha256":model.payload_sha256(),"prompt":moderation::PROMPT,
            "window_tokens":moderation::WINDOW_TOKENS,
        }))?,
    )?;
    eprintln!(
        "{} rows, {total} tokens in {:.1?}",
        rows.len(),
        started.elapsed()
    );
    Ok(())
}

fn score(path: &str) -> Result<()> {
    let mut model = open()?;
    let heads_path = installed_heads(Path::new("models"))
        .ok_or("models/moderation/moderation-heads.json is not installed")?;
    let heads = ModerationHeads::open(&heads_path, model.payload_sha256())?;
    for row in rows(path)? {
        let text = row["text"].as_str().ok_or("row without text")?;
        let windows = moderation::windows(&model, text)?;
        let vectors = model.embed_ids(&windows, &mut || Ok(()))?;
        let scores = heads.score(&vectors);
        let mut out = serde_json::Map::new();
        for (index, name) in CATEGORIES.iter().enumerate() {
            out.insert(
                (*name).to_owned(),
                json!({"score":scores.scores[index],"flag":scores.flags[index]}),
            );
        }
        println!(
            "{}",
            json!({"flagged":scores.flagged(),"windows":windows.len(),"categories":out})
        );
    }
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("features") if args.len() >= 3 => features(&args[1], &args[2], &args[3..]),
        Some("score") if args.len() == 2 => score(&args[1]),
        _ => Err(
            "usage: moderation-embed features DATASET.jsonl OUT_PREFIX [--start N] \
                  [--count M] | score TEXTS.jsonl"
                .into(),
        ),
    }
}
