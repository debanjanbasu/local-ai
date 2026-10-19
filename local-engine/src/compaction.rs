//! Model-facing halves of conversation compaction and reasoning summaries.
//!
//! Both are ordinary generations of the one loaded checkpoint: no second
//! model runs. This module owns only what the engine owns — how the request
//! is phrased and rendered and how a compacted summary is rendered back into
//! a prompt. *When* to compact, which items are kept verbatim and how much
//! output space to reserve belong to the caller's session layer (see
//! `local_services::compaction`), which counts with
//! [`Engine::count_chat_tokens`](crate::Engine::count_chat_tokens) or
//! [`EngineHandle::count_chat_tokens`](crate::EngineHandle::count_chat_tokens)
//! against the context in [`EngineInfo`](crate::EngineInfo).
//!
//! Both requests run with thinking off, so the summary is the generation's
//! visible content and never its raw reasoning; a request is answered by
//! collecting [`Event::Content`](crate::Event::Content) pieces only.

use crate::{
    ChatMessage, ChatRequest, GenerateParams, ReasoningEffort, ResponseFormat, Sampling, ToolChoice,
};
use crate::{ToolDefinition, runtime::DEFAULT_MAX_OUTPUT_TOKENS};

/// The output budget of one compaction summary.
pub const COMPACTION_MAX_TOKENS: usize = 2_048;

/// Sampling temperature for summaries: low, because a summary should restate,
/// not invent.
const SUMMARY_TEMPERATURE: f32 = 0.3;

/// The instruction that ends a compaction request, after the history it
/// summarizes.
pub const COMPACTION_PROMPT: &str = "\
CONTEXT CHECKPOINT COMPACTION. The conversation above will be removed from your context \
and replaced by what you write now, so write a summary that lets you continue the task \
without it. If the conversation begins with an earlier summary, fold it in. Record, \
precisely and without inventing anything:
- the user's goals, requirements and constraints;
- decisions made, and why;
- work completed, with the exact file paths, names, identifiers, commands, values and \
results later steps depend on;
- each tool call made and what it returned, when it matters later;
- everything unresolved: open questions, failures, pending work, and the next step.
Write plain text without preamble. Do not answer or continue the task here.";

/// The header a compacted summary is rendered under, so the model reads it as
/// a summary of removed context, never as the user's words or its own
/// reasoning.
pub const SUMMARY_HEADER: &str = "\
[Context compacted] Earlier turns of this conversation were removed to save context. \
This is the summary written when they were removed:";

/// How much detail a reasoning summary carries (`reasoning.summary`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SummaryDetail {
    /// The default length: one short paragraph.
    Auto,
    /// One to three sentences.
    Concise,
    /// Several short paragraphs, one per main step.
    Detailed,
}

impl SummaryDetail {
    /// The detail named by `OpenAI`'s `auto`, `concise` or `detailed`.
    #[must_use]
    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "auto" => Some(Self::Auto),
            "concise" => Some(Self::Concise),
            "detailed" => Some(Self::Detailed),
            _ => None,
        }
    }

    /// The `OpenAI` name of this detail level.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Concise => "concise",
            Self::Detailed => "detailed",
        }
    }

    /// The output budget of a summary at this detail.
    #[must_use]
    pub const fn max_tokens(self) -> usize {
        match self {
            Self::Concise => 160,
            Self::Auto => 320,
            Self::Detailed => 900,
        }
    }

    const fn instruction(self) -> &'static str {
        match self {
            Self::Concise => "in one to three short sentences",
            Self::Auto => "in one short paragraph",
            Self::Detailed => "in a few short paragraphs, one for each main step or consideration",
        }
    }
}

fn summary_request(
    messages: Vec<ChatMessage>,
    tools: Vec<ToolDefinition>,
    session: Option<String>,
    max_tokens: usize,
) -> ChatRequest {
    ChatRequest {
        messages,
        max_tokens,
        sampling: Sampling(GenerateParams {
            temperature: SUMMARY_TEMPERATURE,
            ..GenerateParams::default()
        }),
        thinking: false,
        effort: ReasoningEffort::default(),
        preserve_thinking: true,
        session,
        // The conversation's tools are rendered, so the prompt shares its
        // prefix with the conversation's own turns and the prompt cache can
        // serve it, but no call may be made: the answer is the summary.
        tool_choice: if tools.is_empty() {
            ToolChoice::Auto
        } else {
            ToolChoice::None
        },
        tools,
        parallel_tool_calls: true,
        response_format: ResponseFormat::Text,
        stop: Vec::new(),
    }
}

/// The request that summarizes `history` for a context checkpoint.
///
/// `history` is rendered exactly as the conversation's own turns are (an
/// earlier summary first, as [`summary_message`] renders it), followed by
/// [`COMPACTION_PROMPT`]. `tools` and `session` should be the conversation's,
/// so the prompt cache can reuse the prefix those turns already prefilled.
/// The summary is the visible content of the generation.
#[must_use]
pub fn compaction_request(
    mut history: Vec<ChatMessage>,
    tools: Vec<ToolDefinition>,
    session: Option<String>,
) -> ChatRequest {
    history.push(ChatMessage {
        role: "user".into(),
        content: COMPACTION_PROMPT.into(),
        ..ChatMessage::default()
    });
    summary_request(history, tools, session, COMPACTION_MAX_TOKENS)
}

/// How a compacted `summary` is rendered back into a prompt: one user
/// message under [`SUMMARY_HEADER`], in place of the turns it replaced.
#[must_use]
pub fn summary_message(summary: &str) -> ChatMessage {
    ChatMessage {
        role: "user".into(),
        content: format!("{SUMMARY_HEADER}\n\n{summary}"),
        ..ChatMessage::default()
    }
}

/// The request that summarizes a generation's raw `reasoning` at `detail`.
///
/// The reasoning is quoted to the model as text to summarize; the summary is
/// the visible content of the generation, and the raw reasoning itself is
/// never returned as one.
#[must_use]
pub fn reasoning_summary_request(reasoning: &str, detail: SummaryDetail) -> ChatRequest {
    let content = format!(
        "Below is the private reasoning an assistant wrote before answering. Summarize it {}, \
         in the first person, as a reader-facing account of what was considered and \
         concluded. State only what the reasoning contains. Reply with the summary alone.\n\n\
         <reasoning>\n{reasoning}\n</reasoning>",
        detail.instruction()
    );
    let message = ChatMessage {
        role: "user".into(),
        content,
        ..ChatMessage::default()
    };
    summary_request(
        vec![message],
        Vec::new(),
        None,
        detail.max_tokens().min(DEFAULT_MAX_OUTPUT_TOKENS),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Engine;

    #[test]
    fn compaction_requests_render_the_history_then_the_prompt_without_thinking() {
        let history = vec![
            summary_message("Earlier: the user wants a CSV parser."),
            ChatMessage {
                role: "user".into(),
                content: "Add quoting support.".into(),
                ..ChatMessage::default()
            },
        ];
        let request = compaction_request(history, Vec::new(), Some("s".into()));
        assert!(!request.thinking);
        assert_eq!(request.max_tokens, COMPACTION_MAX_TOKENS);
        assert_eq!(request.session.as_deref(), Some("s"));
        assert_eq!(request.messages.len(), 3);
        assert!(request.messages[0].content.starts_with(SUMMARY_HEADER));
        let rendered =
            Engine::render_chat_with_tools(&request.messages, request.thinking, &request.tools)
                .unwrap_or_default();
        let summary = rendered.find("CSV parser").unwrap_or(usize::MAX);
        let ask = rendered.find("Add quoting").unwrap_or(0);
        let prompt = rendered.find("CONTEXT CHECKPOINT").unwrap_or(0);
        assert!(summary < ask && ask < prompt, "{rendered}");
    }

    #[test]
    fn reasoning_summaries_quote_the_reasoning_and_bound_the_output() {
        for (name, detail) in [
            ("auto", SummaryDetail::Auto),
            ("concise", SummaryDetail::Concise),
            ("detailed", SummaryDetail::Detailed),
        ] {
            assert_eq!(SummaryDetail::parse(name), Some(detail));
            assert_eq!(detail.name(), name);
            let request = reasoning_summary_request("2 + 2 is 4.", detail);
            assert!(!request.thinking);
            assert_eq!(request.max_tokens, detail.max_tokens());
            assert!(request.messages[0].content.contains("2 + 2 is 4."));
            assert_eq!(request.tools.len(), 0);
        }
        assert_eq!(SummaryDetail::parse("verbose"), None);
        assert!(SummaryDetail::Concise.max_tokens() < SummaryDetail::Detailed.max_tokens());
    }
}
