//! Retrieval quality of the vector store search path: BM25, dense and
//! hybrid rankings compared on the same store (see `docs/EMBEDDINGS.md`).
//!
//! ```text
//! vector-store-eval prepare SCRATCH_DIR TASK.json...   (no GPU)
//! vector-store-eval measure SCRATCH_DIR TASK.json...   (loads the encoder)
//! ```
//!
//! `prepare` turns each task (`{"task", "queries", "query_ids", "docs",
//! "doc_ids", "qrels"}`, the retrieval fixtures of `embedding-eval`, prompts
//! applied) into a fresh services store under `SCRATCH_DIR/<task>` (which
//! must not exist): every distinct document text one File (`title\n\ntext`,
//! the prompt removed) attached to one vector store with the default
//! chunking and indexed for BM25 by [`VectorIndex::index_next`] without an
//! embedding model. `measure` opens it with the native encoder from
//! `models/embeddinggemma2-text` and embeds every chunk through the
//! backfill ([`VectorIndex::embed_next`], the path an existing store takes
//! when the model is installed), then compares the rankings. Queries are sent without
//! their prompt (the store adds its own) through
//! [`VectorIndex::search_with`], 50 results each; results are reduced to
//! distinct files in order and scored by nDCG@10 against the qrels, a file
//! being relevant with the best grade of the documents it holds (exact
//! duplicate texts share one file, so ties between them do not count).

use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use local_ai::cli::vector_embedder::ModelChunkEmbedder;
use local_engine::activity::EngineActivity;
use local_engine::embedding::{EmbeddingModel, installed_model};
use local_services::Store;
use local_services::files::{CreateFile, FilePurpose};
use local_services::vector_stores::{
    CreateFileBatch, CreateVectorStore, MAX_BATCH_FILES, Ranker, Ranking, RankingOptions,
    SearchRequest, VectorIndex,
};
use serde::Deserialize;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

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

/// `title: T | text: X` → `T\n\nX` (just `X` for the title `none`).
fn document(prompted: &str) -> String {
    let rest = prompted.strip_prefix("title: ").unwrap_or(prompted);
    match rest.split_once(" | text: ") {
        Some(("none", text)) => text.to_owned(),
        Some((title, text)) => format!("{title}\n\n{text}"),
        None => rest.to_owned(),
    }
}

/// `task: … | query: Q` → `Q`.
fn query(prompted: &str) -> &str {
    prompted
        .split_once(" | query: ")
        .map_or(prompted, |(_, query)| query)
}

const RANKINGS: [(&str, Ranking); 7] = [
    ("BM25", Ranking::Lexical),
    ("dense", Ranking::Dense),
    (
        "RRF k=60 1:1",
        Ranking::Fused {
            k: 60.0,
            text_weight: 1.0,
            embedding_weight: 1.0,
        },
    ),
    (
        "RRF k=60 1:2",
        Ranking::Fused {
            k: 60.0,
            text_weight: 1.0,
            embedding_weight: 2.0,
        },
    ),
    (
        "RRF k=60 1:3",
        Ranking::Fused {
            k: 60.0,
            text_weight: 1.0,
            embedding_weight: 3.0,
        },
    ),
    (
        "RRF k=60 2:1",
        Ranking::Fused {
            k: 60.0,
            text_weight: 2.0,
            embedding_weight: 1.0,
        },
    ),
    (
        "RRF k=20 1:1",
        Ranking::Fused {
            k: 20.0,
            text_weight: 1.0,
            embedding_weight: 1.0,
        },
    ),
];

fn ndcg10(ranked: &[&str], rels: &HashMap<String, f64>) -> f64 {
    let dcg: f64 = ranked
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
    if idcg > 0.0 { dcg / idcg } else { 0.0 }
}

#[derive(serde::Serialize, Deserialize)]
struct Prepared {
    vector_store: String,
    files: usize,
    file_of_doc: HashMap<String, String>,
}

fn prepare(scratch: &Path, path: &str) -> Result<()> {
    let task: Task = serde_json::from_slice(&std::fs::read(path)?)?;
    let root = scratch.join(&task.name);
    if root.exists() {
        return Err(format!("{} exists; give a fresh scratch directory", root.display()).into());
    }
    let store = Store::open(&root)?;
    // One File per distinct text; documents map to it.
    let started = Instant::now();
    let mut by_text: HashMap<&str, String> = HashMap::new();
    let mut file_of_doc: HashMap<String, String> = HashMap::new();
    let mut file_ids = Vec::new();
    for (doc_id, text) in task.doc_ids.iter().zip(&task.docs) {
        let file_id = if let Some(file_id) = by_text.get(text.as_str()) {
            file_id.clone()
        } else {
            let file = store.create_file(
                &CreateFile {
                    filename: format!("{doc_id}.txt"),
                    purpose: FilePurpose::Assistants,
                    expires_after: None,
                },
                document(text).as_bytes(),
            )?;
            by_text.insert(text, file.id.clone());
            file_ids.push(file.id.clone());
            file.id
        };
        file_of_doc.insert(doc_id.clone(), file_id);
    }
    let vector_store = store.create_vector_store(&CreateVectorStore {
        name: Some(task.name.clone()),
        ..CreateVectorStore::default()
    })?;
    for batch in file_ids.chunks(MAX_BATCH_FILES) {
        store.create_vector_store_file_batch(
            &vector_store.id,
            &CreateFileBatch {
                file_ids: Some(batch.to_vec()),
                ..CreateFileBatch::default()
            },
        )?;
    }
    let index = VectorIndex::open(&store)?;
    while index.index_next()? > 0 {}
    eprintln!(
        "{}: {} documents as {} files, indexed for BM25 in {:.1?}",
        task.name,
        task.docs.len(),
        file_ids.len(),
        started.elapsed()
    );
    let prepared = Prepared {
        vector_store: vector_store.id,
        files: file_ids.len(),
        file_of_doc,
    };
    std::fs::write(root.join("eval.json"), serde_json::to_vec(&prepared)?)?;
    Ok(())
}

fn measure(scratch: &Path, path: &str, embedder: &Arc<ModelChunkEmbedder>) -> Result<()> {
    let task: Task = serde_json::from_slice(&std::fs::read(path)?)?;
    let root = scratch.join(&task.name);
    let prepared: Prepared = serde_json::from_slice(&std::fs::read(root.join("eval.json"))?)?;
    let store = Store::open(&root)?;
    let index = VectorIndex::open(&store)?;
    index.set_embedder(Some(Arc::clone(embedder) as _));
    let started = Instant::now();
    let mut files = 0;
    loop {
        match index.embed_next()? {
            0 => break,
            n => files += n,
        }
    }
    let chunks = index.check()?.chunks;
    eprintln!(
        "{}: embedded {files} files ({chunks} chunks) through the backfill in {:.1?}",
        task.name,
        started.elapsed()
    );
    let vector_store = prepared.vector_store;
    // Relevance per file: the best grade among its documents.
    let mut qrels: HashMap<&str, HashMap<String, f64>> = HashMap::new();
    for (qid, rels) in &task.qrels {
        let entry = qrels.entry(qid.as_str()).or_default();
        for (doc_id, grade) in rels {
            if let Some(file_id) = prepared.file_of_doc.get(doc_id.as_str()) {
                let slot = entry.entry(file_id.clone()).or_insert(0.0);
                *slot = slot.max(*grade);
            }
        }
    }
    println!(
        "| {} ({} queries, {} files, {chunks} chunks) | nDCG@10 | mean search ms |",
        task.name,
        task.queries.len(),
        prepared.files
    );
    println!("| --- | ---: | ---: |");
    for (name, ranking) in RANKINGS {
        let (mut total, mut counted) = (0.0, 0_u32);
        let started = Instant::now();
        for (qid, prompted) in task.query_ids.iter().zip(&task.queries) {
            let Some(rels) = qrels.get(qid.as_str()) else {
                continue;
            };
            let mut request = SearchRequest::new(query(prompted));
            request.max_num_results = Some(50);
            request.ranking_options = Some(RankingOptions {
                ranker: Ranker::Auto,
                score_threshold: 0.0,
            });
            let results = index.search_with(&vector_store, &request, ranking)?;
            let mut ranked: Vec<&str> = Vec::new();
            for hit in &results.data {
                if !ranked.contains(&hit.file_id.as_str()) {
                    ranked.push(&hit.file_id);
                }
            }
            total += ndcg10(&ranked, rels);
            counted += 1;
        }
        let elapsed = started.elapsed();
        println!(
            "| {name} | {:.2} | {:.1} |",
            100.0 * total / f64::from(counted.max(1)),
            elapsed.as_secs_f64() * 1000.0 / f64::from(counted.max(1))
        );
    }
    println!();
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let usage = "usage: vector-store-eval prepare|measure SCRATCH_DIR TASK.json...";
    let [mode, scratch, tasks @ ..] = args.as_slice() else {
        return Err(usage.into());
    };
    if tasks.is_empty() {
        return Err(usage.into());
    }
    match mode.as_str() {
        "prepare" => {
            for task in tasks {
                prepare(Path::new(scratch), task)?;
            }
            return Ok(());
        }
        "measure" => {}
        _ => return Err(usage.into()),
    }
    let dir = installed_model(Path::new("models"))?;
    let model = EmbeddingModel::open(&dir)?;
    let embedder = Arc::new(ModelChunkEmbedder::new(
        Arc::new(Mutex::new(model)),
        Arc::new(EngineActivity::default()),
    ));
    for task in tasks {
        measure(Path::new(scratch), task, &embedder)?;
    }
    Ok(())
}
