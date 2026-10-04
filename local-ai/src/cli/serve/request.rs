use serde::Deserialize;
use serde_json::Value;

use local_engine::{ChatMessage, ChatRequest, CompletionRequest, Sampling};

use crate::GenerateParams;

#[derive(Deserialize)]
struct GenerateRequest {
    prompt: Option<String>,
    messages: Option<Vec<Message>>,
    max_tokens: Option<usize>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<usize>,
    min_p: Option<f32>,
    presence_penalty: Option<f32>,
    frequency_penalty: Option<f32>,
    seed: Option<u64>,
    #[serde(default)]
    stream: bool,
    user: Option<String>,
    session_id: Option<String>,
}

impl GenerateRequest {
    fn params(&self) -> GenerateParams {
        let defaults = GenerateParams::default();
        GenerateParams {
            max_tokens: self.max_tokens.unwrap_or(defaults.max_tokens),
            temperature: self.temperature.unwrap_or(defaults.temperature),
            top_p: self.top_p.unwrap_or(defaults.top_p),
            top_k: self.top_k.unwrap_or(defaults.top_k),
            min_p: self.min_p.unwrap_or(defaults.min_p),
            presence_penalty: self.presence_penalty.unwrap_or(defaults.presence_penalty),
            repetition_penalty: self
                .frequency_penalty
                .map_or(defaults.repetition_penalty, |value| 1.0 + value.max(0.0)),
            seed: self.seed.unwrap_or(defaults.seed),
        }
    }
}

#[derive(Deserialize)]
struct Message {
    role: String,
    content: Value,
    reasoning_content: Option<String>,
}

pub(super) enum GenerationRequest {
    Chat(ChatRequest),
    Completion(CompletionRequest),
}

pub(super) struct PreparedGeneration {
    pub(super) stream: bool,
    pub(super) request: GenerationRequest,
}

pub(super) fn prepare_generation(
    body: &[u8],
    chat: bool,
    thinking: bool,
) -> crate::Result<PreparedGeneration> {
    let request: GenerateRequest = serde_json::from_slice(body)
        .map_err(|error| crate::Error::InvalidArgument(format!("invalid JSON request: {error}")))?;
    let params = request.params();
    let session = request.session_id.or(request.user);
    let generation = if chat {
        let messages = request
            .messages
            .as_deref()
            .ok_or_else(|| crate::Error::InvalidArgument("messages are required".into()))?;
        let messages = messages
            .iter()
            .map(|message| {
                Ok(ChatMessage {
                    role: message.role.clone(),
                    content: message_text(&message.content)?,
                    reasoning_content: message.reasoning_content.clone(),
                })
            })
            .collect::<crate::Result<Vec<_>>>()?;
        GenerationRequest::Chat(ChatRequest {
            messages,
            max_tokens: params.max_tokens,
            sampling: Sampling(params),
            thinking,
            session,
        })
    } else {
        let prompt = request
            .prompt
            .ok_or_else(|| crate::Error::InvalidArgument("prompt is required".into()))?;
        GenerationRequest::Completion(CompletionRequest {
            prompt,
            max_tokens: params.max_tokens,
            sampling: Sampling(params),
            session,
        })
    };
    Ok(PreparedGeneration {
        stream: request.stream,
        request: generation,
    })
}

pub(super) fn message_text(value: &Value) -> crate::Result<String> {
    if let Some(text) = value.as_str() {
        return Ok(text.to_owned());
    }
    let items = value
        .as_array()
        .ok_or_else(|| crate::Error::InvalidArgument("message content must be text".into()))?;
    let mut text = String::new();
    for item in items {
        if item.get("type").and_then(Value::as_str) != Some("text") {
            return Err(crate::Error::InvalidArgument(
                "media content is not supported; Bonsai 2 is text-only".into(),
            ));
        }
        text.push_str(item.get("text").and_then(Value::as_str).ok_or_else(|| {
            crate::Error::InvalidArgument("text content requires a text string".into())
        })?);
    }
    Ok(text)
}
