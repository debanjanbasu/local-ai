//! CPU-only tests of the `file_search` declaration and of turning citation
//! markers into citations as answer text streams.

use std::fmt::Write as _;

use serde_json::{Value, json};

use local_engine::ToolOutput;
use local_services::vector_stores::{Query, Ranker, Ranking};

use super::{Citations, Config, DEFAULT_MAX_NUM_RESULTS, NAME, Piece};

fn declare(tool: &Value) -> crate::Result<(Config, local_engine::ToolDefinition, Value)> {
    Config::declare(tool.as_object().expect("object"))
}

#[test]
fn a_declaration_is_echoed_with_its_defaults_and_offered_as_one_strict_function() {
    let (config, definition, echo) =
        declare(&json!({"type":"file_search","vector_store_ids":["vs_a","vs_b"]})).expect("ok");
    assert_eq!(
        echo,
        json!({"type":"file_search","vector_store_ids":["vs_a","vs_b"],"max_num_results":DEFAULT_MAX_NUM_RESULTS,
               "ranking_options":{"ranker":"auto","score_threshold":0.0},"filters":null})
    );
    assert_eq!(definition.name, NAME);
    assert!(definition.strict);
    assert_eq!(definition.parameters["required"], json!(["queries"]));
    assert!(!config.needs_embeddings());
    // The model's queries fill in the rest of the search.
    let request = config
        .request(&json!({"queries":["alpha","beta"]}))
        .expect("request");
    assert_eq!(
        request.query,
        Query::Many(vec!["alpha".into(), "beta".into()])
    );
    assert_eq!(request.max_num_results, Some(DEFAULT_MAX_NUM_RESULTS));
    assert!(config.request(&json!({"queries":[]})).is_err());
    assert!(config.request(&json!({"queries":"alpha"})).is_err());

    // The client's choices are honoured.
    let (config, _, echo) = declare(&json!({"type":"file_search","vector_store_ids":["vs_a"],
        "max_num_results":50,"filters":{"type":"eq","key":"lang","value":"en"},
        "ranking_options":{"ranker":"none","score_threshold":0.25}}))
    .expect("ok");
    let request = config.request(&json!({"queries":["q"]})).expect("request");
    assert_eq!(request.max_num_results, Some(50));
    assert!(request.filters.is_some());
    let options = request.ranking_options.expect("options");
    assert_eq!(
        (options.ranker, options.score_threshold),
        (Ranker::None, 0.25)
    );
    assert_eq!(
        echo["ranking_options"],
        json!({"ranker":"none","score_threshold":0.25})
    );
    let (config, _, echo) = declare(&json!({"type":"file_search","vector_store_ids":["vs_a"],
        "ranking_options":{"hybrid_search":{"embedding_weight":0.75,"text_weight":0.25}}}))
    .expect("ok");
    assert!(config.needs_embeddings());
    assert_eq!(
        config.ranking(),
        Some(Ranking::Fused {
            k: 60.0,
            text_weight: 0.25,
            embedding_weight: 0.75
        })
    );
    assert_eq!(
        echo["ranking_options"]["hybrid_search"],
        json!({"embedding_weight":0.75,"text_weight":0.25})
    );
}

#[test]
fn malformed_declarations_are_refused_with_the_reason() {
    for (tool, expected) in [
        (json!({"type":"file_search"}), "vector_store_ids"),
        (
            json!({"type":"file_search","vector_store_ids":[]}),
            "vector_store_ids",
        ),
        (
            json!({"type":"file_search","vector_store_ids":[""]}),
            "non-empty",
        ),
        (
            json!({"type":"file_search","vector_store_ids":["vs"],"max_num_results":0}),
            "1 to 50",
        ),
        (
            json!({"type":"file_search","vector_store_ids":["vs"],"max_num_results":51}),
            "1 to 50",
        ),
        (
            json!({"type":"file_search","vector_store_ids":["vs"],"ranking_options":{"ranker":"default-2024-11-15"}}),
            "default-2024-11-15 is not available",
        ),
        (
            json!({"type":"file_search","vector_store_ids":["vs"],"ranking_options":{"score_threshold":2}}),
            "score_threshold",
        ),
        (
            json!({"type":"file_search","vector_store_ids":["vs"],"ranking_options":{"ranker":"none","hybrid_search":{"embedding_weight":1,"text_weight":1}}}),
            "needs ranker auto",
        ),
        (
            json!({"type":"file_search","vector_store_ids":["vs"],"ranking_options":{"hybrid_search":{"embedding_weight":0,"text_weight":0}}}),
            "positive weight",
        ),
        (
            json!({"type":"file_search","vector_store_ids":["vs"],"filters":{"type":"near","key":"a","value":1}}),
            "filters",
        ),
        (
            json!({"type":"file_search","vector_store_ids":["vs"],"rewrite_query":true}),
            "\"rewrite_query\" is not supported",
        ),
    ] {
        let error = declare(&tool).expect_err("refused").to_string();
        assert!(error.contains(expected), "{tool}: {error}");
    }
}

/// Citations of results 1 (`a.md`) and 2 (`b.md`).
fn citations() -> Citations {
    let mut citations = Citations::default();
    citations.add(&ToolOutput {
        call_id: "c".into(),
        name: NAME.into(),
        content: String::new(),
        detail: json!({"status":"completed","first":1,"results":[
            {"file_id":"file-a","filename":"a.md","score":0.9,"text":"x","attributes":{}},
            {"file_id":"file-b","filename":"b.md","score":0.8,"text":"y","attributes":{}}
        ]}),
    });
    citations
}

/// The text `pieces` stream as, with each citation as `<file_id,...>`.
fn stream(citations: &mut Citations, pieces: &[&str]) -> String {
    let mut out = String::new();
    for piece in pieces {
        for piece in citations.filter(piece) {
            match piece {
                Piece::Text(text) => out.push_str(&text),
                Piece::Cite(cited) => {
                    let ids: Vec<String> = cited.into_iter().map(|(id, _)| id).collect();
                    let _ = write!(out, "<{}>", ids.join(","));
                }
            }
        }
    }
    out.push_str(&citations.flush());
    out
}

#[test]
fn markers_become_citations_wherever_the_stream_splits_them() {
    let text = "The answer is 42 【1】. Also 【1, 2】 and【2†b.md】!\nDone ";
    let expected = "The answer is 42<file-a>. Also<file-a,file-b> and<file-b>!\nDone ";
    // Every split point of the text, by characters.
    let chars: Vec<char> = text.chars().collect();
    for split in 0..=chars.len() {
        let (head, tail): (String, String) = (
            chars[..split].iter().collect(),
            chars[split..].iter().collect(),
        );
        let mut citations = citations();
        assert_eq!(
            stream(&mut citations, &[&head, &tail]),
            expected,
            "split at {split}"
        );
    }
    // One character at a time.
    let mut citations = citations();
    let singles: Vec<String> = chars.iter().map(char::to_string).collect();
    let singles: Vec<&str> = singles.iter().map(String::as_str).collect();
    assert_eq!(stream(&mut citations, &singles), expected);
}

#[test]
fn text_that_cites_nothing_of_this_response_is_left_as_written() {
    for text in [
        "a 【3】 b",
        "【注意】 read this",
        "half 【1",
        "list [1] stays",
        "【1, x】",
        "【】",
    ] {
        let mut citations = citations();
        assert_eq!(stream(&mut citations, &[text]), text, "{text}");
    }
    // An opening bracket never closed within the marker limit is text.
    let long = format!("【{}】 tail", "1".repeat(80));
    let mut citations = citations();
    assert_eq!(stream(&mut citations, &[&long]), long);
    // Without results, nothing is held back at all.
    let mut empty = Citations::default();
    assert_eq!(
        empty.filter("trailing 【1"),
        vec![Piece::Text("trailing 【1".into())]
    );
}
