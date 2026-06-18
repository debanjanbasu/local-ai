use std::collections::{HashMap, VecDeque};

use crate::request_queue::RequestQueue;
pub use crate::request_queue::{InferenceRequest, RequestPriority};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PrefixIdentity {
    pub model_id: String,
    pub tokenizer_id: String,
    pub prompt_hash: u64,
}

impl PrefixIdentity {
    #[must_use]
    pub fn new(
        model_id: impl Into<String>,
        tokenizer_id: impl Into<String>,
        prompt_hash: u64,
    ) -> Self {
        Self {
            model_id: model_id.into(),
            tokenizer_id: tokenizer_id.into(),
            prompt_hash,
        }
    }
}

impl From<&InferenceRequest> for PrefixIdentity {
    fn from(request: &InferenceRequest) -> Self {
        Self::new(
            request.model_id.clone(),
            request.tokenizer_id.clone(),
            request.prompt_hash,
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrefixEntry {
    pub ref_count: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AdmittedLane {
    Running,
    Swapped,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Admission {
    pub lane: AdmittedLane,
    pub request: InferenceRequest,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ReleaseOutcome {
    pub released: InferenceRequest,
    pub promoted: Option<InferenceRequest>,
}

#[derive(Debug, Default)]
pub struct SharedPrefixCache {
    entries: HashMap<PrefixIdentity, PrefixEntry>,
}

impl SharedPrefixCache {
    pub fn record(&mut self, request: &InferenceRequest) {
        self.entries
            .entry(PrefixIdentity::from(request))
            .and_modify(|entry| entry.ref_count += 1)
            .or_insert(PrefixEntry { ref_count: 1 });
    }

    pub fn release(&mut self, request: &InferenceRequest) {
        let key = PrefixIdentity::from(request);
        if let Some(entry) = self.entries.get_mut(&key) {
            if entry.ref_count > 1 {
                entry.ref_count -= 1;
            } else {
                self.entries.remove(&key);
            }
        }
    }

    #[must_use]
    pub fn ref_count(&self, model_id: &str, tokenizer_id: &str, prompt_hash: u64) -> usize {
        self.entries
            .get(&PrefixIdentity::new(model_id, tokenizer_id, prompt_hash))
            .map_or(0, |entry| entry.ref_count)
    }
}

#[derive(Debug)]
pub struct ContinuousBatcher {
    queue: RequestQueue,
    shared_prefix_cache: SharedPrefixCache,
    max_running: usize,
    max_swapped: usize,
    running_ids: VecDeque<String>,
    swapped_ids: VecDeque<String>,
    admitted: HashMap<String, (AdmittedLane, InferenceRequest)>,
}

impl ContinuousBatcher {
    #[must_use]
    pub fn new(max_running: usize, max_swapped: usize) -> Self {
        Self {
            queue: RequestQueue::default(),
            shared_prefix_cache: SharedPrefixCache::default(),
            max_running,
            max_swapped,
            running_ids: VecDeque::new(),
            swapped_ids: VecDeque::new(),
            admitted: HashMap::new(),
        }
    }

    pub fn enqueue(&mut self, request: InferenceRequest) {
        self.shared_prefix_cache.record(&request);
        self.queue.push(request);
    }

    pub fn admit_next(&mut self) -> Option<InferenceRequest> {
        self.admit_next_with_lane()
            .map(|admission| admission.request)
    }

    pub(crate) fn admit_next_with_lane(&mut self) -> Option<Admission> {
        if self.running_ids.len() < self.max_running {
            let request = self.queue.pop_next()?;
            self.running_ids.push_back(request.request_id.clone());
            self.admitted.insert(
                request.request_id.clone(),
                (AdmittedLane::Running, request.clone()),
            );
            return Some(Admission {
                lane: AdmittedLane::Running,
                request,
            });
        }
        if self.swapped_ids.len() < self.max_swapped {
            let request = self.queue.pop_next()?;
            self.swapped_ids.push_back(request.request_id.clone());
            self.admitted.insert(
                request.request_id.clone(),
                (AdmittedLane::Swapped, request.clone()),
            );
            return Some(Admission {
                lane: AdmittedLane::Swapped,
                request,
            });
        }
        None
    }

    pub fn release(&mut self, request_id: &str) -> Option<InferenceRequest> {
        self.release_with_promotion(request_id)
            .map(|outcome| outcome.released)
    }

    pub(crate) fn release_with_promotion(&mut self, request_id: &str) -> Option<ReleaseOutcome> {
        if let Some((lane, request)) = self.admitted.remove(request_id) {
            let promoted = match lane {
                AdmittedLane::Running => {
                    let removed = Self::remove_from_ids(&mut self.running_ids, request_id);
                    debug_assert!(
                        removed,
                        "running request should remain ordered until release"
                    );
                    self.promote_next_swapped()
                }
                AdmittedLane::Swapped => {
                    let removed = Self::remove_from_ids(&mut self.swapped_ids, request_id);
                    debug_assert!(
                        removed,
                        "swapped request should remain ordered until release"
                    );
                    None
                }
            };
            self.shared_prefix_cache.release(&request);
            return Some(ReleaseOutcome {
                released: request,
                promoted,
            });
        }

        let removed = self.queue.remove(request_id)?;
        self.shared_prefix_cache.release(&removed);
        Some(ReleaseOutcome {
            released: removed,
            promoted: None,
        })
    }

    #[must_use]
    pub fn running(&self) -> usize {
        self.running_ids.len()
    }

    #[must_use]
    pub fn swapped(&self) -> usize {
        self.swapped_ids.len()
    }

    #[must_use]
    pub const fn shared_prefix_cache(&self) -> &SharedPrefixCache {
        &self.shared_prefix_cache
    }

    /// Check if there are pending decode requests that could be processed
    /// between prefill chunks. O(1) check.
    #[must_use]
    pub fn has_pending_decodes(&self) -> bool {
        !self.queue.is_empty() || self.running() > 0
    }

    /// Get the number of pending requests in the queue.
    #[must_use]
    pub fn pending_count(&self) -> usize {
        self.queue.len()
    }

    pub fn admitted_requests(&self) -> impl Iterator<Item = &InferenceRequest> + '_ {
        self.admitted.values().map(|(_, request)| request)
    }

    fn promote_next_swapped(&mut self) -> Option<InferenceRequest> {
        if self.running_ids.len() >= self.max_running {
            return None;
        }

        let request_id = self.swapped_ids.pop_front()?;
        #[allow(clippy::expect_used)]
        let (lane, request) = self
            .admitted
            .get_mut(&request_id)
            .expect("swapped request should remain admitted until promotion");
        *lane = AdmittedLane::Running;
        self.running_ids.push_back(request_id);
        Some(request.clone())
    }

    fn remove_from_ids(ids: &mut VecDeque<String>, request_id: &str) -> bool {
        let Some(index) = ids.iter().position(|candidate| candidate == request_id) else {
            return false;
        };
        ids.remove(index);
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_has_pending_decodes_empty() {
        let batcher = ContinuousBatcher::new(4, 2);
        assert!(!batcher.has_pending_decodes());
    }

    #[test]
    fn test_pending_count_empty() {
        let batcher = ContinuousBatcher::new(4, 2);
        assert_eq!(batcher.pending_count(), 0);
    }
}
