//! Model identifiers: what `GET /v1/models` lists, what
//! `GET /v1/models/{model}` retrieves, and what every route's `model` member
//! accepts. One table, so every route agrees on what each name means.
//!
//! | model | id | aliases | routes |
//! |-------|----|---------|--------|
//! | Bonsai | [`BONSAI_MODEL_ID`] | none | Chat Completions, Responses, compaction, input tokens, decisions, batches |
//! | embeddings | [`EMBEDDING_MODEL_ID`] | the `OpenAI` embedding model names | `/v1/embeddings` |
//! | moderation | [`MODERATION_MODEL_ID`] | the `OpenAI` moderation model names | `/v1/moderations` |
//! | speech | [`SPEECH_MODEL_ID`] | the `OpenAI` speech model names | `/v1/audio/speech` |
//! | transcription | [`TRANSCRIPTION_MODEL_ID`] | the `OpenAI` transcription model names | `/v1/audio/transcriptions`, `/v1/audio/translations` |
//!
//! The aliases let a client keep the `OpenAI` model name it was written for;
//! the local model answers, under its own id.
//!
//! Routes of the optional models require their own model (moderation lets it
//! be omitted). The Bonsai generation routes, batch lines included, accept an
//! omitted `model` (batch lines must name one) or any name that is not
//! another served model's, because many `OpenAI` clients send a configured
//! name the server cannot know; a name of another local model is a mistake
//! and is refused.

use axum::http::{Method, StatusCode};
use serde_json::{Value, json};

use local_engine::embedding::EMBEDDING_MODEL_ID;
use local_engine::moderation::MODERATION_MODEL_ID;
use local_engine::speech::SPEECH_MODEL_ID;
use local_engine::transcription::TRANSCRIPTION_MODEL_ID;

use super::{embeddings, moderations, speech, transcription};

/// The id this server gives the pinned Ternary Bonsai 2 27B checkpoint
/// (`prism-ml/Ternary-Bonsai-2-27B-gguf`, PTQ1).
pub(super) const BONSAI_MODEL_ID: &str = "ternary-bonsai-2-27b";

/// `OpenAI` embedding model names, accepted for [`EMBEDDING_MODEL_ID`].
pub(super) const EMBEDDING_ALIASES: [&str; 3] = [
    "text-embedding-3-small",
    "text-embedding-3-large",
    "text-embedding-ada-002",
];

/// `OpenAI` speech model names, accepted for [`SPEECH_MODEL_ID`].
pub(super) const SPEECH_ALIASES: [&str; 3] = ["tts-1", "tts-1-hd", "gpt-4o-mini-tts"];

/// `OpenAI` moderation model names, accepted for [`MODERATION_MODEL_ID`].
pub(super) const MODERATION_ALIASES: [&str; 4] = [
    "omni-moderation-latest",
    "omni-moderation-2024-09-26",
    "text-moderation-latest",
    "text-moderation-stable",
];

/// `OpenAI` transcription model names, accepted for [`TRANSCRIPTION_MODEL_ID`].
pub(super) const TRANSCRIPTION_ALIASES: [&str; 3] =
    ["whisper-1", "gpt-4o-transcribe", "gpt-4o-mini-transcribe"];

/// A model this server can serve.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Served {
    Bonsai,
    Embeddings,
    Moderation,
    Speech,
    Transcription,
}

/// Which model `name` names, by id or alias, installed or not; `bonsai` is
/// the loaded Bonsai id ([`BONSAI_MODEL_ID`] outside tests).
pub(super) fn resolve(name: &str, bonsai: &str) -> Option<Served> {
    if name == bonsai {
        Some(Served::Bonsai)
    } else if name == EMBEDDING_MODEL_ID || EMBEDDING_ALIASES.contains(&name) {
        Some(Served::Embeddings)
    } else if name == MODERATION_MODEL_ID || MODERATION_ALIASES.contains(&name) {
        Some(Served::Moderation)
    } else if name == SPEECH_MODEL_ID || SPEECH_ALIASES.contains(&name) {
        Some(Served::Speech)
    } else if name == TRANSCRIPTION_MODEL_ID || TRANSCRIPTION_ALIASES.contains(&name) {
        Some(Served::Transcription)
    } else {
        None
    }
}

/// Check the `model` of a Bonsai generation request: absent or any name but
/// another served model's (see the module docs).
pub(super) fn check_generation(model: Option<&str>) -> crate::Result<()> {
    model
        .and_then(generation_refusal)
        .map_or(Ok(()), |refusal| {
            Err(crate::Error::InvalidArgument(format!("model {refusal}")))
        })
}

/// Why `name` is refused as a generation model, after the word "model":
/// `None` unless it names another served model.
pub(super) fn generation_refusal(name: &str) -> Option<String> {
    match resolve(name, BONSAI_MODEL_ID)? {
        Served::Bonsai => None,
        other => Some(format!(
            "{name:?} is {}, not a text generation model; use {BONSAI_MODEL_ID:?} here",
            route_of(other)
        )),
    }
}

/// [`check_generation`] for a `model` member of any JSON type: a non-string
/// is refused as the contract requires a string.
pub(super) fn check_generation_value(model: Option<&Value>) -> crate::Result<()> {
    match model {
        None | Some(Value::Null) => Ok(()),
        Some(Value::String(name)) => check_generation(Some(name)),
        Some(_) => Err(crate::Error::InvalidArgument(
            "model must be a string".into(),
        )),
    }
}

/// Where a non-Bonsai model is served, for the refusal.
const fn route_of(model: Served) -> &'static str {
    match model {
        Served::Bonsai => "the text generation model",
        Served::Embeddings => "served by POST /v1/embeddings",
        Served::Moderation => "served by POST /v1/moderations",
        Served::Speech => "served by POST /v1/audio/speech",
        Served::Transcription => "served by POST /v1/audio/transcriptions",
    }
}

/// Which optional models this server runs beside Bonsai (independent
/// installs, so independent flags).
#[derive(Clone, Copy, Debug, Default)]
#[allow(clippy::struct_excessive_bools)]
pub(super) struct Installed {
    pub(super) embeddings: bool,
    pub(super) moderation: bool,
    pub(super) speech: bool,
    pub(super) transcription: bool,
}

/// The models this server serves, as `GET /v1/models` lists them.
#[derive(Clone, Copy, Debug)]
pub(super) struct Catalog<'a> {
    /// The loaded Bonsai id.
    pub(super) bonsai: &'a str,
    /// Tokens one request may hold (`BonsaiInfo::context`).
    pub(super) context: usize,
    /// Registration time of every model, stable for this server.
    pub(super) created: u64,
    pub(super) installed: Installed,
}

impl Catalog<'_> {
    /// The model object of `model`, or `None` when it is not installed.
    ///
    /// Bonsai's `context_length` (the name `OpenRouter` and LM Studio read)
    /// and `max_model_len` (vLLM's) are both the per-request admission limit
    /// from `BonsaiInfo::context`, prompt plus output, not the checkpoint's
    /// training length: it is what a request can actually use on this
    /// machine.
    fn entry(&self, model: Served) -> Option<Value> {
        let (created, installed) = (self.created, self.installed);
        match model {
            Served::Bonsai => Some(json!({
                "id":self.bonsai,"object":"model","created":created,"owned_by":"local",
                "context_length":self.context,"max_model_len":self.context
            })),
            Served::Embeddings => installed
                .embeddings
                .then(|| embeddings::model_json(created)),
            Served::Moderation => installed
                .moderation
                .then(|| moderations::model_json(created)),
            Served::Speech => installed.speech.then(|| speech::model_json(created)),
            Served::Transcription => installed
                .transcription
                .then(|| transcription::model_json(created)),
        }
    }

    /// `GET /v1/models`: Bonsai, then every installed optional model (each
    /// served by its own route).
    pub(super) fn list(&self) -> Value {
        let data: Vec<Value> = [
            Served::Bonsai,
            Served::Embeddings,
            Served::Moderation,
            Served::Speech,
            Served::Transcription,
        ]
        .into_iter()
        .filter_map(|model| self.entry(model))
        .collect();
        json!({"object":"list","data":data})
    }

    /// `GET` and `DELETE /v1/models/{model}`, `model` being the rest of the
    /// path.
    ///
    /// `GET` answers the model object of an installed model, by id or alias
    /// (an alias answers the model it stands for, under its own id), and 404
    /// for anything else. `DELETE` is for fine-tuned models, which this
    /// server has none of: an installed model is refused with 403, since it
    /// is a base model that cannot be deleted, and any other name is 404.
    pub(super) fn answer(
        &self,
        method: &Method,
        model: &str,
    ) -> Result<Value, (StatusCode, String)> {
        let entry = resolve(model, self.bonsai).and_then(|served| self.entry(served));
        let not_found = || {
            (
                StatusCode::NOT_FOUND,
                format!("model {model:?} not found; GET /v1/models lists the models served here"),
            )
        };
        match *method {
            Method::GET => entry.ok_or_else(not_found),
            Method::DELETE => match entry {
                Some(_) => Err((
                    StatusCode::FORBIDDEN,
                    format!(
                        "model {model:?} is a base model, not a fine-tuned one; it cannot be \
                         deleted (this server has no fine-tuned models)"
                    ),
                )),
                None => Err(not_found()),
            },
            _ => Err((StatusCode::NOT_FOUND, "route not found".into())),
        }
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;

    fn catalog(installed: Installed) -> Catalog<'static> {
        Catalog {
            bonsai: BONSAI_MODEL_ID,
            context: 32768,
            created: 1234,
            installed,
        }
    }

    const ALL: Installed = Installed {
        embeddings: true,
        moderation: true,
        speech: true,
        transcription: true,
    };

    #[test]
    fn every_listed_model_is_retrieved_by_its_id() {
        let catalog = catalog(ALL);
        let list = catalog.list();
        let data = list["data"].as_array().expect("data");
        assert_eq!(data.len(), 5);
        assert_eq!(data[0]["id"], BONSAI_MODEL_ID);
        for model in data {
            let id = model["id"].as_str().expect("id");
            assert_eq!(catalog.answer(&Method::GET, id).expect(id), *model);
            // And each id is what its own route accepts.
            assert!(resolve(id, BONSAI_MODEL_ID).is_some(), "{id}");
        }
    }

    #[test]
    fn aliases_retrieve_the_model_they_stand_for() {
        let catalog = catalog(ALL);
        let whisper = catalog.answer(&Method::GET, "whisper-1").expect("alias");
        assert_eq!(whisper["id"], TRANSCRIPTION_MODEL_ID);
        for alias in MODERATION_ALIASES {
            let moderation = catalog.answer(&Method::GET, alias).expect(alias);
            assert_eq!(moderation["id"], MODERATION_MODEL_ID);
        }
    }

    #[test]
    fn uninstalled_and_unknown_names_are_not_found() {
        let catalog = catalog(Installed::default());
        for name in [EMBEDDING_MODEL_ID, SPEECH_MODEL_ID, "whisper-1", "gpt-4o"] {
            let (status, _) = catalog.answer(&Method::GET, name).expect_err(name);
            assert_eq!(status, StatusCode::NOT_FOUND, "{name}");
        }
    }

    #[test]
    fn deleting_refuses_base_models_and_misses_unknown_ones() {
        let catalog = catalog(ALL);
        for name in [BONSAI_MODEL_ID, EMBEDDING_MODEL_ID, "whisper-1"] {
            let (status, message) = catalog.answer(&Method::DELETE, name).expect_err(name);
            assert_eq!(status, StatusCode::FORBIDDEN, "{name}");
            assert!(message.contains("fine-tuned"), "{message}");
        }
        let (status, _) = catalog
            .answer(&Method::DELETE, "ft:gpt-4o-mini:acme::abc")
            .expect_err("unknown");
        assert_eq!(status, StatusCode::NOT_FOUND);
    }

    #[test]
    fn generation_refuses_only_other_local_models() {
        for name in [BONSAI_MODEL_ID, "gpt-4o", "anything"] {
            check_generation(Some(name)).expect(name);
        }
        check_generation(None).expect("omitted");
        for name in [
            EMBEDDING_MODEL_ID,
            MODERATION_MODEL_ID,
            "omni-moderation-latest",
            SPEECH_MODEL_ID,
            TRANSCRIPTION_MODEL_ID,
            "whisper-1",
        ] {
            let error = check_generation(Some(name)).expect_err(name);
            assert!(error.to_string().contains("served by"), "{error}");
        }
        assert!(check_generation_value(Some(&json!(7))).is_err());
        check_generation_value(Some(&Value::Null)).expect("null");
    }
}
