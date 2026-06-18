use std::collections::VecDeque;

use crate::DEFAULT_MAX_OUTPUT_TOKENS;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueueLane {
    Interactive,
    Background,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequestPriority {
    Interactive,
    Background,
}

#[derive(Debug, Clone)]
pub struct InferenceRequest {
    pub request_id: String,
    pub model_id: String,
    pub tokenizer_id: String,
    pub prompt_hash: u64,
    pub priority: RequestPriority,
    pub prompt_tokens: usize,
    pub max_tokens: usize,
}

impl PartialEq for InferenceRequest {
    fn eq(&self, other: &Self) -> bool {
        self.request_id == other.request_id
            && self.model_id == other.model_id
            && self.tokenizer_id == other.tokenizer_id
            && self.prompt_hash == other.prompt_hash
            && self.priority == other.priority
            && self.prompt_tokens == other.prompt_tokens
            && self.max_tokens == other.max_tokens
    }
}

impl Eq for InferenceRequest {}

impl InferenceRequest {
    #[must_use]
    pub fn new(
        request_id: impl Into<String>,
        model_id: impl Into<String>,
        tokenizer_id: impl Into<String>,
        prompt_hash: u64,
        priority: RequestPriority,
    ) -> Self {
        Self {
            request_id: request_id.into(),
            model_id: model_id.into(),
            tokenizer_id: tokenizer_id.into(),
            prompt_hash,
            priority,
            prompt_tokens: 0,
            max_tokens: DEFAULT_MAX_OUTPUT_TOKENS,
        }
    }

    #[must_use]
    pub const fn with_token_limits(mut self, prompt_tokens: usize, max_tokens: usize) -> Self {
        self.prompt_tokens = prompt_tokens;
        self.max_tokens = max_tokens;
        self
    }
}

#[derive(Debug, Default)]
pub struct RequestQueue {
    interactive: VecDeque<InferenceRequest>,
    background: VecDeque<InferenceRequest>,
}

impl RequestQueue {
    pub fn push(&mut self, request: InferenceRequest) {
        match request.priority {
            RequestPriority::Interactive => self.interactive.push_back(request),
            RequestPriority::Background => self.background.push_back(request),
        }
    }

    pub fn pop_next(&mut self) -> Option<InferenceRequest> {
        self.interactive
            .pop_front()
            .or_else(|| self.background.pop_front())
    }

    pub fn remove(&mut self, request_id: &str) -> Option<InferenceRequest> {
        Self::remove_from_queue(&mut self.interactive, request_id)
            .or_else(|| Self::remove_from_queue(&mut self.background, request_id))
    }

    #[must_use]
    pub fn lane_for(&self, request_id: &str) -> Option<QueueLane> {
        if self
            .interactive
            .iter()
            .any(|request| request.request_id == request_id)
        {
            Some(QueueLane::Interactive)
        } else if self
            .background
            .iter()
            .any(|request| request.request_id == request_id)
        {
            Some(QueueLane::Background)
        } else {
            None
        }
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.interactive.is_empty() && self.background.is_empty()
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.interactive.len() + self.background.len()
    }

    fn remove_from_queue(
        queue: &mut VecDeque<InferenceRequest>,
        request_id: &str,
    ) -> Option<InferenceRequest> {
        let index = queue
            .iter()
            .position(|request| request.request_id == request_id)?;
        queue.remove(index)
    }
}
