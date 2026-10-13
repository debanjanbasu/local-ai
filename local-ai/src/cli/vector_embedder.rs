//! The native `EmbeddingGemma` 2 text encoder as the vector stores' chunk
//! embedder ([`local_services::vector_stores::ChunkEmbedder`]).
//!
//! Chunks are embedded with the model card's document prompt
//! (`title: none | text: {chunk}`: chunks have no title of their own) and
//! queries with its retrieval query prompt (`task: search result | query:
//! {query}`), at the full 768 dimensions. An input longer than the encoder's
//! [`MAX_INPUT_TOKENS`] is embedded from its first tokens (the closing
//! `<eos>` kept); the vector store's chunks are counted in words, so a large
//! static chunk size can exceed it.
//!
//! The model is shared with `POST /v1/embeddings`: it is locked for at most
//! about [`LOCK_TOKENS`] tokens at a time, and between command buffers the
//! work yields to Bonsai decode steps ([`EngineActivity::yield_to_decode`]),
//! so background indexing never delays a decode step by more than one short
//! submission, and an API embedding request waits for at most one group.

use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;

use local_engine::activity::EngineActivity;
use local_engine::embedding::{
    EMBEDDING_DIMENSIONS, EMBEDDING_MODEL_ID, EmbeddingModel, EmbeddingPrompt, MAX_INPUT_TOKENS,
};
use local_services::vector_stores::ChunkEmbedder;

/// Tokens embedded per hold of the model lock (about 0.5 s on an idle GPU).
pub const LOCK_TOKENS: usize = 4_096;
/// Longest wait for one decode step before the next submission goes ahead,
/// so embeddings are slowed, never starved.
const DECODE_YIELD_LIMIT: Duration = Duration::from_millis(500);
/// The model card's query task for retrieval.
const QUERY_TASK: &str = "search result";

/// The shared encoder, embedding at low priority beside Bonsai.
pub struct ModelChunkEmbedder {
    model: Arc<Mutex<EmbeddingModel>>,
    activity: Arc<EngineActivity>,
    space: String,
}

impl ModelChunkEmbedder {
    /// Wrap the shared model; `activity` is the engine's decode activity.
    #[must_use]
    pub fn new(model: Arc<Mutex<EmbeddingModel>>, activity: Arc<EngineActivity>) -> Self {
        let digest = model
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .payload_sha256()
            .chars()
            .take(16)
            .collect::<String>();
        Self {
            model,
            activity,
            // Model, weights, width, storage and prompts: vectors stored
            // under any other name are embedded again.
            space: format!(
                "{EMBEDDING_MODEL_ID}/{digest}/{EMBEDDING_DIMENSIONS}/f16/doc:title-none/\
                 query:{QUERY_TASK}"
            ),
        }
    }

    fn embed(
        &self,
        texts: &[&str],
        prompt: EmbeddingPrompt<'_>,
        stop: &dyn Fn() -> bool,
    ) -> local_services::Result<Vec<Vec<f32>>> {
        let ids = {
            let model = self.model.lock().unwrap_or_else(PoisonError::into_inner);
            texts
                .iter()
                .map(|text| model.tokenize(&prompt.format(text)).map(fit))
                .collect::<local_engine::Result<Vec<_>>>()
                .map_err(failure)?
        };
        let mut vectors = Vec::with_capacity(ids.len());
        let mut start = 0;
        while start < ids.len() {
            let mut end = start + 1;
            let mut tokens = ids[start].len();
            while end < ids.len() && tokens + ids[end].len() <= LOCK_TOKENS {
                tokens += ids[end].len();
                end += 1;
            }
            let mut pace = || {
                if stop() {
                    return Err(local_engine::Error::Generation(
                        "vector store embedding stopped".into(),
                    ));
                }
                self.activity.yield_to_decode(DECODE_YIELD_LIMIT);
                Ok(())
            };
            // Yield before taking the GPU too, not only between submissions.
            pace().map_err(failure)?;
            let mut model = self.model.lock().unwrap_or_else(PoisonError::into_inner);
            let group = model.embed_ids(&ids[start..end], &mut pace);
            drop(model);
            vectors.extend(group.map_err(failure)?);
            start = end;
        }
        Ok(vectors)
    }
}

impl ChunkEmbedder for ModelChunkEmbedder {
    fn space(&self) -> &str {
        &self.space
    }

    fn dimensions(&self) -> usize {
        EMBEDDING_DIMENSIONS
    }

    fn embed_documents(
        &self,
        texts: &[&str],
        stop: &dyn Fn() -> bool,
    ) -> local_services::Result<Vec<Vec<f32>>> {
        self.embed(texts, EmbeddingPrompt::Document { title: None }, stop)
    }

    fn embed_queries(&self, queries: &[&str]) -> local_services::Result<Vec<Vec<f32>>> {
        self.embed(
            queries,
            EmbeddingPrompt::Query { task: QUERY_TASK },
            &|| false,
        )
    }
}

/// Keep at most [`MAX_INPUT_TOKENS`] ids: the first ones and the closing
/// `<eos>`.
fn fit(mut ids: Vec<u32>) -> Vec<u32> {
    if ids.len() > MAX_INPUT_TOKENS
        && let Some(&last) = ids.last()
    {
        ids.truncate(MAX_INPUT_TOKENS - 1);
        ids.push(last);
    }
    ids
}

#[allow(clippy::needless_pass_by_value)] // used as `map_err(failure)`
fn failure(error: local_engine::Error) -> local_services::Error {
    local_services::Error::Io(std::io::Error::other(format!("embedding failed: {error}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn long_inputs_keep_their_head_and_eos() {
        let ids: Vec<u32> = (0..u32::try_from(MAX_INPUT_TOKENS + 10).unwrap_or(0)).collect();
        let last = *ids.last().unwrap_or(&0);
        let fitted = fit(ids);
        assert_eq!(fitted.len(), MAX_INPUT_TOKENS);
        assert_eq!(fitted[0], 0);
        assert_eq!(fitted[MAX_INPUT_TOKENS - 1], last);
        assert_eq!(fit(vec![2, 5, 1]), vec![2, 5, 1]);
    }
}
