//! Parity, retrieval and throughput checks for the native `EmbeddingGemma` 2
//! text encoder (see `docs/EMBEDDINGS.md`).
//!
//! ```text
//! embedding-eval parity FIXTURE.jsonl
//! embedding-eval retrieval TASK.json...
//! embedding-eval bench [--texts N] [--tokens T]
//! ```
//!
//! The model loads from `models/embeddinggemma2-text` under the working
//! directory. Fixtures come from the reference kernel (sentence-transformers
//! 6.1.0, transformers 5.19.0, FP32):
//!
//! - `parity`: one JSON object per line, `{"text", "ids", "emb"}` with the
//!   reference token ids (`<bos>` … `<eos>`, truncated at 512 tokens as the
//!   reference was) and the reference normalized 768-d embedding. Reports
//!   token-id agreement (native ids truncated the same way) and the cosine of the native embedding of the
//!   reference ids to the reference embedding (mean and minimum).
//! - `retrieval`: `{"task", "queries", "query_ids", "docs", "doc_ids",
//!   "qrels"}` with prompts already applied; inputs are truncated at 512
//!   tokens like the reference runs; reports nDCG@10 and recall@100 at 768
//!   dimensions (exact cosine search).
//! - `bench`: tokens per second for packed batches of synthetic inputs.

use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

use local_engine::embedding::{EmbeddingModel, MAX_INPUT_TOKENS, installed_model};
use serde::Deserialize;
use serde_json::Value;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[derive(Deserialize)]
struct ParityRow {
    text: String,
    ids: Vec<u32>,
    emb: Vec<f32>,
}

#[derive(Deserialize)]
struct Task {
    #[serde(rename = "task")]
    name: String,
    queries: Vec<String>,
    query_ids: Vec<String>,
    docs: Vec<String>,
    doc_ids: Vec<String>,
    qrels: HashMap<String, HashMap<String, f64>>,
}

fn open() -> Result<EmbeddingModel> {
    let dir = installed_model(Path::new("models"))
        .ok_or("models/embeddinggemma2-text is not installed (see docs/EMBEDDINGS.md)")?;
    let started = Instant::now();
    let model = EmbeddingModel::open(&dir)?;
    eprintln!("loaded {} in {:.2?}", dir.display(), started.elapsed());
    Ok(model)
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn parity(path: &str) -> Result<()> {
    let rows: Vec<ParityRow> = std::fs::read_to_string(path)?
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(serde_json::from_str)
        .collect::<std::result::Result<_, _>>()?;
    let mut model = open()?;
    let (mut compared, mut equal, mut first_diff) = (0, 0, None);
    for (i, row) in rows.iter().enumerate() {
        let ids = reference_truncate(model.tokenize(&row.text)?);
        compared += 1;
        if ids == row.ids {
            equal += 1;
        } else if first_diff.is_none() {
            first_diff = Some((i, ids.len(), row.ids.len()));
        }
    }
    println!("token ids identical: {equal}/{compared} (first difference: {first_diff:?})");
    let inputs: Vec<Vec<u32>> = rows.iter().map(|row| row.ids.clone()).collect();
    let started = Instant::now();
    let embeddings = model.embed_ids(&inputs, &mut || Ok(()))?;
    let elapsed = started.elapsed();
    let cosines: Vec<f32> = embeddings
        .iter()
        .zip(&rows)
        .map(|(native, row)| dot(native, &row.emb))
        .collect();
    let mean = cosines.iter().map(|c| f64::from(*c)).sum::<f64>() / cosines.len() as f64;
    let (worst, min) =
        cosines.iter().enumerate().fold(
            (0, f32::INFINITY),
            |acc, (i, c)| if *c < acc.1 { (i, *c) } else { acc },
        );
    let tokens: usize = inputs.iter().map(Vec::len).sum();
    // Packing independence: the first 64 inputs one at a time against packed.
    let mut alone_min = f32::INFINITY;
    let mut identical = 0;
    for (ids, packed) in inputs.iter().zip(&embeddings).take(64) {
        let single = model.embed_ids(std::slice::from_ref(ids), &mut || Ok(()))?;
        alone_min = alone_min.min(dot(&single[0], packed));
        identical += usize::from(single[0] == *packed);
    }
    println!(
        "packing independence over 64 inputs: {identical} bitwise identical, min cosine {alone_min:.7}"
    );
    println!(
        "cosine to reference over {} inputs: mean {mean:.6}, min {min:.6} (input {worst}, {} tokens)",
        cosines.len(),
        inputs[worst].len()
    );
    println!(
        "{tokens} tokens in {elapsed:.2?}: {:.0} tokens/s",
        tokens as f64 / elapsed.as_secs_f64()
    );
    Ok(())
}

fn ndcg_recall(
    ranked: &[usize],
    doc_ids: &[String],
    qid: &str,
    rels: &HashMap<String, f64>,
) -> (f64, f64) {
    let names: Vec<&str> = ranked
        .iter()
        .map(|&j| doc_ids[j].as_str())
        .filter(|d| *d != qid)
        .take(100)
        .collect();
    let dcg: f64 = names
        .iter()
        .take(10)
        .enumerate()
        .map(|(i, d)| rels.get(*d).copied().unwrap_or(0.0) / ((i + 2) as f64).log2())
        .sum();
    let mut ideal: Vec<f64> = rels.values().copied().collect();
    ideal.sort_by(|a, b| b.total_cmp(a));
    let idcg: f64 = ideal
        .iter()
        .take(10)
        .enumerate()
        .map(|(i, g)| g / ((i + 2) as f64).log2())
        .sum();
    let hits = names.iter().filter(|d| rels.contains_key(**d)).count();
    (
        if idcg > 0.0 { dcg / idcg } else { 0.0 },
        hits as f64 / rels.len() as f64,
    )
}

/// nDCG@10 expected under a uniformly random order among exactly tied scores
/// (identical documents embed identically here, so a corpus with duplicate
/// texts, such as `CosQA`, otherwise scores by the tie-break rule). `scored`
/// is sorted by descending score.
fn tie_aware_ndcg(
    scored: &[(f32, usize)],
    doc_ids: &[String],
    qid: &str,
    rels: &HashMap<String, f64>,
) -> f64 {
    let kept: Vec<(f32, f64)> = scored
        .iter()
        .filter(|(_, j)| doc_ids[*j] != qid)
        .map(|(score, j)| (*score, rels.get(&doc_ids[*j]).copied().unwrap_or(0.0)))
        .collect();
    let mut dcg = 0.0;
    let mut start = 0;
    while start < kept.len() && start < 10 {
        let mut end = start + 1;
        while end < kept.len() && kept[end].0.to_bits() == kept[start].0.to_bits() {
            end += 1;
        }
        let mean_gain = kept[start..end].iter().map(|(_, g)| g).sum::<f64>() / (end - start) as f64;
        dcg += (start..end.min(10))
            .map(|rank| mean_gain / ((rank + 2) as f64).log2())
            .sum::<f64>();
        start = end;
    }
    let mut ideal: Vec<f64> = rels.values().copied().collect();
    ideal.sort_by(|a, b| b.total_cmp(a));
    let idcg: f64 = ideal
        .iter()
        .take(10)
        .enumerate()
        .map(|(i, g)| g / ((i + 2) as f64).log2())
        .sum();
    if idcg > 0.0 { dcg / idcg } else { 0.0 }
}

/// The reference kernels ran sentence-transformers with `max_seq_length`
/// 512: `<bos>`, the first 510 text tokens, `<eos>`.
const REFERENCE_MAX_TOKENS: usize = 512;

fn reference_truncate(mut ids: Vec<u32>) -> Vec<u32> {
    if ids.len() > REFERENCE_MAX_TOKENS {
        ids.truncate(REFERENCE_MAX_TOKENS - 1);
        ids.push(1);
    }
    ids
}

fn embed_all(
    model: &mut EmbeddingModel,
    texts: &[String],
) -> Result<(Vec<Vec<f32>>, usize, usize)> {
    let mut ids = Vec::with_capacity(texts.len());
    let mut truncated = 0;
    for text in texts {
        let row = model.tokenize(text)?;
        truncated += usize::from(row.len() > REFERENCE_MAX_TOKENS);
        ids.push(reference_truncate(row));
    }
    let tokens = ids.iter().map(Vec::len).sum();
    let mut out = Vec::with_capacity(ids.len());
    for chunk in ids.chunks(256) {
        out.extend(model.embed_ids(chunk, &mut || Ok(()))?);
    }
    Ok((out, tokens, truncated))
}

fn retrieval(paths: &[String]) -> Result<()> {
    let mut model = open()?;
    for path in paths {
        let task: Task = serde_json::from_str(&std::fs::read_to_string(path)?)?;
        let started = Instant::now();
        let (docs, doc_tokens, truncated) = embed_all(&mut model, &task.docs)?;
        let doc_time = started.elapsed();
        let (queries, _, _) = embed_all(&mut model, &task.queries)?;
        let (mut ndcg, mut recall, mut counted, mut tied) = (0.0, 0.0, 0, 0.0);
        for (query, qid) in queries.iter().zip(&task.query_ids) {
            let Some(rels) = task.qrels.get(qid) else {
                continue;
            };
            let mut scored: Vec<(f32, usize)> = docs
                .iter()
                .enumerate()
                .map(|(j, d)| (dot(query, d), j))
                .collect();
            scored.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
            let ranked: Vec<usize> = scored.iter().take(120).map(|(_, j)| *j).collect();
            let (n, r) = ndcg_recall(&ranked, &task.doc_ids, qid, rels);
            tied += tie_aware_ndcg(&scored, &task.doc_ids, qid, rels);
            ndcg += n;
            recall += r;
            counted += 1;
        }
        println!(
            "{}: nDCG@10 {:.4} (ties in index order; {:.4} expected under random tie order), \
             recall@100 {:.4} over {counted} queries, {} docs \
             ({doc_tokens} tokens, {truncated} truncated to {REFERENCE_MAX_TOKENS}) embedded in {doc_time:.2?} \
             ({:.0} tokens/s)",
            task.name,
            ndcg / f64::from(counted),
            tied / f64::from(counted),
            recall / f64::from(counted),
            task.docs.len(),
            doc_tokens as f64 / doc_time.as_secs_f64(),
        );
    }
    Ok(())
}

fn bench(args: &[String]) -> Result<()> {
    let mut texts = 64_usize;
    let mut tokens = 256_usize;
    let mut it = args.iter();
    while let Some(flag) = it.next() {
        let value: usize = it.next().ok_or("missing value")?.parse()?;
        match flag.as_str() {
            "--texts" => texts = value,
            "--tokens" => tokens = value.clamp(2, MAX_INPUT_TOKENS),
            _ => return Err(format!("unknown flag {flag}").into()),
        }
    }
    let mut model = open()?;
    let inputs: Vec<Vec<u32>> = (0..texts)
        .map(|i| {
            let mut ids = vec![2_u32];
            ids.extend((0..tokens - 2).map(|j| 1000 + ((i * 7919 + j * 104_729) % 200_000) as u32));
            ids.push(1);
            ids
        })
        .collect();
    model.embed_ids(&inputs[..1], &mut || Ok(()))?;
    let mut best = f64::INFINITY;
    for _ in 0..3 {
        let started = Instant::now();
        model.embed_ids(&inputs, &mut || Ok(()))?;
        best = best.min(started.elapsed().as_secs_f64());
    }
    let total = (texts * tokens) as f64;
    println!(
        "{texts} inputs x {tokens} tokens: best {:.1} ms, {:.0} tokens/s, {:.1} inputs/s",
        best * 1e3,
        total / best,
        texts as f64 / best
    );
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("parity") if args.len() == 2 => parity(&args[1]),
        Some("retrieval") if args.len() >= 2 => retrieval(&args[1..]),
        Some("bench") => bench(&args[1..]),
        _ => {
            let value = Value::String(
                "usage: embedding-eval parity FIXTURE.jsonl | retrieval TASK.json... | bench [--texts N] [--tokens T]"
                    .into(),
            );
            Err(value.as_str().unwrap_or_default().into())
        }
    }
}
