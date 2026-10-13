//! Store-level eval tests: lifecycle, pagination, errors, and run execution
//! driven through real batches whose lines are settled by hand (standing in
//! for the batch worker), including a restart in the middle of a run and
//! cancellation.

use std::path::PathBuf;

use serde_json::{Value, json};

use super::*;
use crate::batches::{LineOutcome, NextLine};

const MODEL: &str = "local";

struct Scratch {
    _dir: tempfile::TempDir,
    root: PathBuf,
    store: Store,
}

impl Scratch {
    fn new() -> Self {
        let dir = tempfile::tempdir().expect("temporary directory");
        let root = dir
            .path()
            .canonicalize()
            .expect("canonical temporary directory")
            .join("services");
        let store = Store::open(&root).expect("store opens");
        Self {
            _dir: dir,
            root,
            store,
        }
    }

    /// A second handle, as a restarted server would open it.
    fn reopen(&self) -> Store {
        Store::open(&self.root).expect("store reopens")
    }
}

fn eval_request(criteria: Value) -> CreateEval {
    CreateEval {
        name: Some("arith".to_owned()),
        metadata: None,
        data_source_config: json!({
            "type": "custom",
            "item_schema": {"type": "object", "properties": {"q": {"type": "string"},
                            "a": {"type": "string"}}, "required": ["q", "a"]},
            "include_sample_schema": true,
        }),
        testing_criteria: serde_json::from_value(criteria).expect("criteria array"),
    }
}

fn string_check() -> Value {
    json!({"type": "string_check", "name": "exact", "input": "{{sample.output_text}}",
           "reference": "{{item.a}}", "operation": "eq"})
}

fn score_model() -> Value {
    json!({"type": "score_model", "name": "judge", "model": MODEL, "range": [0, 10],
           "pass_threshold": 5,
           "input": [{"role": "user", "content": "Rate {{sample.output_text}} for {{item.q}}"}]})
}

fn rows() -> Value {
    json!([
        {"item": {"q": "1+1", "a": "2"}},
        {"item": {"q": "2+2", "a": "4"}},
        {"item": {"q": "3+3", "a": "6"}},
    ])
}

fn sampling_run(content: &Value) -> CreateEvalRun {
    CreateEvalRun {
        name: Some("bonsai".to_owned()),
        metadata: None,
        data_source: json!({
            "type": "completions",
            "model": MODEL,
            "input_messages": {"type": "template",
                               "template": [{"role": "user", "content": "{{item.q}}="}]},
            "source": {"type": "file_content", "content": content},
        }),
    }
}

fn chat_answer(text: &str) -> LineOutcome {
    LineOutcome::Response {
        status_code: 200,
        body: json!({
            "object": "chat.completion",
            "model": MODEL,
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": text}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7,
                      "prompt_tokens_details": {"cached_tokens": 1}},
        }),
    }
}

/// Stand in for the batch worker: run every pending line of `batch` with
/// `answer` (by request body), then finish the batch.
fn run_batch(store: &Store, batch: &str, answer: impl Fn(&Map<String, Value>) -> LineOutcome) {
    let lease = store.lease_batch(batch).expect("lease").expect("free");
    loop {
        match store.next_batch_line(&lease).expect("next line") {
            NextLine::Line(line) => {
                store
                    .settle_batch_line(&lease, line.line, &answer(&line.body))
                    .expect("settled");
            }
            NextLine::Finish => {
                store.finish_batch(&lease).expect("finished");
                return;
            }
            NextLine::Done => return,
        }
    }
}

fn last_user_text(body: &Map<String, Value>) -> String {
    body["messages"]
        .as_array()
        .and_then(|messages| messages.last())
        .and_then(|message| message["content"].as_str())
        .unwrap_or_default()
        .to_owned()
}

/// Sampling answers `2`, `5`, `6`; the judge scores 8 unless the answer is 5.
fn answer(body: &Map<String, Value>) -> LineOutcome {
    let text = last_user_text(body);
    if body.contains_key("response_format") {
        let score = if text.starts_with("Rate 5 ") { 1 } else { 8 };
        return chat_answer(&format!("{{\"result\": {score}}}"));
    }
    chat_answer(match text.as_str() {
        "1+1=" => "2",
        "2+2=" => "5",
        _ => "6",
    })
}

fn advance(store: &Store, run: &str) -> Advance {
    let lease = store.lease_eval_run(run).expect("lease").expect("free");
    store.advance_eval_run(&lease, MODEL).expect("advances")
}

#[test]
fn evals_are_created_listed_updated_and_deleted() {
    let scratch = Scratch::new();
    let store = &scratch.store;
    let mut ids = Vec::new();
    for _ in 0..3 {
        ids.push(
            store
                .create_eval(&eval_request(json!([string_check()])), MODEL)
                .expect("created")
                .id,
        );
    }
    let eval = store.get_eval(&ids[0]).expect("found");
    assert_eq!(eval.name, "arith");
    assert_eq!(
        eval.data_source_config["schema"]["required"],
        json!(["item", "sample"])
    );
    assert_eq!(eval.testing_criteria, vec![string_check()]);

    let first = store
        .list_evals(&ListEvals {
            limit: Some(2),
            ..ListEvals::default()
        })
        .expect("listed");
    assert_eq!(
        first.data.iter().map(|e| e.id.clone()).collect::<Vec<_>>(),
        ids[..2]
    );
    assert!(first.has_more);
    let rest = store
        .list_evals(&ListEvals {
            after: first.last_id,
            ..ListEvals::default()
        })
        .expect("listed");
    assert_eq!(rest.data.len(), 1);
    assert!(!rest.has_more);
    let newest = store
        .list_evals(&ListEvals {
            order: Some(Order::Desc),
            limit: Some(1),
            ..ListEvals::default()
        })
        .expect("listed");
    assert_eq!(newest.first_id.as_deref(), Some(ids[2].as_str()));

    let mut metadata = Metadata::new();
    metadata.insert("team".to_owned(), "local".to_owned());
    let updated = store
        .update_eval(
            &ids[0],
            &UpdateEval {
                name: Some("renamed".to_owned()),
                metadata: Some(metadata.clone()),
            },
        )
        .expect("updated");
    assert_eq!(
        (updated.name.as_str(), &updated.metadata),
        ("renamed", &metadata)
    );
    let by_update = store
        .list_evals(&ListEvals {
            order_by: Some(EvalOrderBy::UpdatedAt),
            order: Some(Order::Desc),
            limit: Some(1),
            ..ListEvals::default()
        })
        .expect("listed");
    // Updated in the same second as created: ties break by creation.
    assert_eq!(by_update.data.len(), 1);

    assert!(matches!(
        store.list_evals(&ListEvals {
            limit: Some(0),
            ..ListEvals::default()
        }),
        Err(Error::InvalidArgument(_))
    ));
    assert!(matches!(
        store.list_evals(&ListEvals {
            after: Some("eval_missing".into()),
            ..ListEvals::default()
        }),
        Err(Error::NotFound(_))
    ));
    let deleted = store.delete_eval(&ids[1]).expect("deleted");
    assert_eq!(
        serde_json::to_value(&deleted).expect("json"),
        json!({"object": "eval.deleted", "deleted": true, "eval_id": ids[1]})
    );
    assert!(matches!(store.get_eval(&ids[1]), Err(Error::NotFound(_))));
}

#[test]
fn invalid_evals_and_runs_store_nothing() {
    let scratch = Scratch::new();
    let store = &scratch.store;
    for criteria in [
        json!([]),
        json!([string_check(), string_check()]),
        json!([{"type": "python", "name": "p", "source": "x"}]),
    ] {
        assert!(matches!(
            store.create_eval(&eval_request(criteria), MODEL),
            Err(Error::InvalidArgument(_))
        ));
    }
    assert_eq!(
        store
            .list_evals(&ListEvals::default())
            .expect("listed")
            .data
            .len(),
        0
    );

    let eval = store
        .create_eval(&eval_request(json!([string_check()])), MODEL)
        .expect("created");
    let refused = [
        // An item that does not match the item schema.
        sampling_run(&json!([{"item": {"q": "1+1"}}])),
        // No rows.
        sampling_run(&json!([])),
        // Another model.
        CreateEvalRun {
            data_source: json!({"type": "completions", "model": "gpt-4o",
                                "input_messages": {"type": "template", "template": [{"role": "user", "content": "x"}]},
                                "source": {"type": "file_content", "content": rows()}}),
            ..CreateEvalRun::default()
        },
        // A missing file.
        CreateEvalRun {
            data_source: json!({"type": "jsonl", "source": {"type": "file_id", "id": "file-missing"}}),
            ..CreateEvalRun::default()
        },
    ];
    for request in refused {
        assert!(
            matches!(
                store.create_eval_run(&eval.id, &request, MODEL),
                Err(Error::InvalidArgument(_))
            ),
            "{:?} refused",
            request.data_source
        );
    }
    assert_eq!(
        store
            .list_eval_runs(&eval.id, &ListEvalRuns::default())
            .expect("listed")
            .data
            .len(),
        0
    );
    assert!(matches!(
        store.create_eval_run("eval_missing", &sampling_run(&rows()), MODEL),
        Err(Error::NotFound(_))
    ));
}

#[test]
#[allow(clippy::too_many_lines)]
fn a_jsonl_run_grades_without_generating() {
    let scratch = Scratch::new();
    let store = &scratch.store;
    let eval = store
        .create_eval(
            &eval_request(json!([
                string_check(),
                {"type": "text_similarity", "name": "close", "input": "{{sample.output_text}}",
                 "reference": "{{item.a}}", "evaluation_metric": "rouge_l", "pass_threshold": 0.5},
            ])),
            MODEL,
        )
        .expect("created");
    let file = store
        .create_file(
            &CreateFile { filename: "rows.jsonl".into(), purpose: FilePurpose::Evals, expires_after: None },
            concat!(
                "{\"item\":{\"q\":\"1+1\",\"a\":\"2\"},\"sample\":{\"output_text\":\"2\",\"model\":\"m\"}}\n",
                "{\"item\":{\"q\":\"2+2\",\"a\":\"4\"},\"sample\":{\"output_text\":\"five\"}}\n",
                "{\"item\":{\"q\":\"3+3\",\"a\":\"6\"}}\n",
            )
            .as_bytes(),
        )
        .expect("file stored");
    let run = store
        .create_eval_run(
            &eval.id,
            &CreateEvalRun {
                data_source: json!({"type": "jsonl", "source": {"type": "file_id", "id": file.id}}),
                ..CreateEvalRun::default()
            },
            MODEL,
        )
        .expect("run created");
    assert_eq!(run.status, RunStatus::Queued);
    assert_eq!(run.model, None);
    assert_eq!(
        store.active_eval_runs().expect("active"),
        vec![run.id.clone()]
    );
    // A file deleted after the run was created changes nothing.
    store.delete_file(&file.id).expect("deleted");
    assert_eq!(advance(store, &run.id), Advance::Finished);
    let run = store.get_eval_run(&eval.id, &run.id).expect("found");
    assert_eq!(run.status, RunStatus::Completed);
    assert_eq!(
        run.result_counts,
        ResultCounts {
            total: 3,
            errored: 1,
            failed: 1,
            passed: 1
        }
    );
    assert_eq!(run.per_model_usage, None);
    assert_eq!(
        run.per_testing_criteria_results,
        Some(vec![
            CriterionResult {
                testing_criteria: "exact".into(),
                passed: 1,
                failed: 2
            },
            CriterionResult {
                testing_criteria: "close".into(),
                passed: 1,
                failed: 2
            },
        ])
    );
    let items = store
        .list_eval_output_items(&eval.id, &run.id, &ListOutputItems::default())
        .expect("listed");
    assert_eq!(items.data.len(), 3);
    assert_eq!(items.data[0]["status"], "pass");
    assert_eq!(items.data[0]["sample"]["model"], "m");
    assert_eq!(items.data[1]["results"][1]["score"], 0.0);
    // The third row has no sample: its graders could not run.
    assert_eq!(
        items.data[2]["results"][0]["error"]["code"],
        "invalid_template"
    );
    let passing = store
        .list_eval_output_items(
            &eval.id,
            &run.id,
            &ListOutputItems {
                status: Some(OutputItemStatus::Pass),
                ..ListOutputItems::default()
            },
        )
        .expect("listed");
    assert_eq!(passing.data.len(), 1);
    let id = items.data[1]["id"].as_str().expect("id");
    assert_eq!(
        store
            .get_eval_output_item(&eval.id, &run.id, id)
            .expect("found"),
        items.data[1]
    );
    let page = store
        .list_eval_output_items(
            &eval.id,
            &run.id,
            &ListOutputItems {
                after: Some(id.to_owned()),
                order: Some(Order::Desc),
                ..ListOutputItems::default()
            },
        )
        .expect("listed");
    assert_eq!(page.data, vec![items.data[0].clone()]);
    assert!(matches!(
        store.get_eval_output_item(&eval.id, &run.id, "outputitem_missing"),
        Err(Error::NotFound(_))
    ));
}

#[test]
fn a_sampling_run_generates_and_grades_through_batches() {
    let scratch = Scratch::new();
    let store = &scratch.store;
    let eval = store
        .create_eval(&eval_request(json!([string_check(), score_model()])), MODEL)
        .expect("created");
    let run = store
        .create_eval_run(&eval.id, &sampling_run(&rows()), MODEL)
        .expect("created");
    assert_eq!(run.model.as_deref(), Some(MODEL));

    let Advance::Started(sampling) = advance(store, &run.id) else {
        panic!("a sampling batch is started");
    };
    let batch = store.get_batch(&sampling).expect("batch");
    assert_eq!(batch.request_counts.total, 3);
    assert_eq!(batch.metadata["eval_batch"], format!("{}:sampling", run.id));
    assert_eq!(
        store.get_eval_run(&eval.id, &run.id).expect("run").status,
        RunStatus::InProgress
    );
    assert_eq!(advance(store, &run.id), Advance::Waiting(sampling.clone()));
    run_batch(store, &sampling, answer);

    let Advance::Started(grading) = advance(store, &run.id) else {
        panic!("a grading batch is started");
    };
    assert_eq!(
        store
            .get_batch(&grading)
            .expect("batch")
            .request_counts
            .total,
        3
    );
    run_batch(store, &grading, answer);
    assert_eq!(advance(store, &run.id), Advance::Finished);

    let run = store.get_eval_run(&eval.id, &run.id).expect("run");
    assert_eq!(run.status, RunStatus::Completed);
    assert_eq!(
        run.result_counts,
        ResultCounts {
            total: 3,
            errored: 0,
            failed: 1,
            passed: 2
        }
    );
    assert_eq!(
        run.per_model_usage,
        Some(vec![ModelUsage {
            model_name: MODEL.into(),
            invocation_count: 6,
            prompt_tokens: 30,
            completion_tokens: 12,
            total_tokens: 42,
            cached_tokens: 6,
        }])
    );
    let items = store
        .list_eval_output_items(&eval.id, &run.id, &ListOutputItems::default())
        .expect("listed");
    let wrong = &items.data[1];
    assert_eq!(wrong["status"], "fail");
    assert_eq!(
        wrong["sample"]["output"],
        json!([{"role": "assistant", "content": "5"}])
    );
    assert_eq!(
        wrong["sample"]["input"],
        json!([{"role": "user", "content": "2+2="}])
    );
    // A completions source samples through Chat Completions, whose seed is
    // deprecated: the sample reports the fixed one it ran with.
    assert_eq!(wrong["sample"]["seed"], crate::evals::CHAT_COMPLETIONS_SEED);
    assert_eq!(wrong["results"][1]["score"], 1.0);
    assert_eq!(wrong["results"][1]["passed"], false);
    assert_eq!(
        wrong["results"][1]["sample"]["input"],
        json!([{"role": "user", "content": "Rate 5 for 2+2"}])
    );
    assert_eq!(items.data[0]["results"][1]["score"], 8.0);
    // A finished run can be deleted; its eval then too.
    store.delete_eval_run(&eval.id, &run.id).expect("deleted");
    assert!(matches!(
        store.get_eval_run(&eval.id, &run.id),
        Err(Error::NotFound(_))
    ));
}

#[test]
fn a_restarted_store_resumes_a_run_where_it_stopped() {
    let scratch = Scratch::new();
    let store = &scratch.store;
    let eval = store
        .create_eval(&eval_request(json!([string_check(), score_model()])), MODEL)
        .expect("created");
    let run = store
        .create_eval_run(&eval.id, &sampling_run(&rows()), MODEL)
        .expect("created");
    let Advance::Started(sampling) = advance(store, &run.id) else {
        panic!("a sampling batch is started");
    };
    // The "process" stops after settling one line.
    {
        let lease = store.lease_batch(&sampling).expect("lease").expect("free");
        let NextLine::Line(line) = store.next_batch_line(&lease).expect("next") else {
            panic!("a line");
        };
        store
            .settle_batch_line(&lease, line.line, &answer(&line.body))
            .expect("settled");
    }
    // A run lease held by a live owner keeps others out.
    let held = store.lease_eval_run(&run.id).expect("lease").expect("free");
    assert!(store.lease_eval_run(&run.id).expect("lease").is_none());
    drop(held);

    let restarted = scratch.reopen();
    assert_eq!(
        restarted.active_eval_runs().expect("active"),
        vec![run.id.clone()]
    );
    assert_eq!(
        advance(&restarted, &run.id),
        Advance::Waiting(sampling.clone())
    );
    run_batch(&restarted, &sampling, answer);
    let Advance::Started(grading) = advance(&restarted, &run.id) else {
        panic!("a grading batch is started");
    };
    // A crash after creating the grading batch but before recording it:
    // the batch is found again rather than created twice.
    restarted
        .connection()
        .expect("connection")
        .execute(
            "UPDATE eval_runs SET batch_id = NULL WHERE id = ?1",
            [&run.id],
        )
        .expect("forgotten");
    assert_eq!(
        advance(&restarted, &run.id),
        Advance::Started(grading.clone())
    );
    run_batch(&restarted, &grading, answer);
    assert_eq!(advance(&restarted, &run.id), Advance::Finished);
    let done = restarted.get_eval_run(&eval.id, &run.id).expect("run");
    assert_eq!(
        done.result_counts,
        ResultCounts {
            total: 3,
            errored: 0,
            failed: 1,
            passed: 2
        }
    );
    let batches = restarted
        .list_batches(&crate::batches::ListBatches::default())
        .expect("batches");
    assert_eq!(batches.data.len(), 2, "one batch per phase");
    // Advancing a finished run again changes nothing.
    assert_eq!(advance(&restarted, &run.id), Advance::Finished);
    let items = restarted
        .list_eval_output_items(&eval.id, &run.id, &ListOutputItems::default())
        .expect("listed");
    assert_eq!(items.data.len(), 3);
}

#[test]
fn cancelling_a_run_cancels_its_batch_and_keeps_no_results() {
    let scratch = Scratch::new();
    let store = &scratch.store;
    let eval = store
        .create_eval(&eval_request(json!([string_check()])), MODEL)
        .expect("created");
    let run = store
        .create_eval_run(&eval.id, &sampling_run(&rows()), MODEL)
        .expect("created");
    let Advance::Started(sampling) = advance(store, &run.id) else {
        panic!("a sampling batch is started");
    };
    assert!(matches!(
        store.delete_eval(&eval.id),
        Err(Error::InvalidArgument(_))
    ));
    assert!(matches!(
        store.delete_eval_run(&eval.id, &run.id),
        Err(Error::InvalidArgument(_))
    ));
    let canceled = store.cancel_eval_run(&eval.id, &run.id).expect("canceled");
    assert_eq!(canceled.run.status, RunStatus::Canceled);
    assert_eq!(canceled.batch.as_deref(), Some(sampling.as_str()));
    assert_eq!(
        store.get_batch(&sampling).expect("batch").status,
        crate::batches::BatchStatus::Cancelling
    );
    run_batch(store, &sampling, answer);
    assert_eq!(advance(store, &run.id), Advance::Finished);
    assert_eq!(
        store.active_eval_runs().expect("active"),
        Vec::<String>::new()
    );
    // Repeating is harmless.
    assert_eq!(
        store
            .cancel_eval_run(&eval.id, &run.id)
            .expect("again")
            .batch,
        None
    );
    assert_eq!(
        store
            .list_eval_output_items(&eval.id, &run.id, &ListOutputItems::default())
            .expect("listed")
            .data
            .len(),
        0
    );
    let listed = store
        .list_eval_runs(
            &eval.id,
            &ListEvalRuns {
                status: Some(RunStatus::Canceled),
                ..ListEvalRuns::default()
            },
        )
        .expect("listed");
    assert_eq!(listed.data.len(), 1);
    store
        .delete_eval(&eval.id)
        .expect("deleted with its finished runs");
}

#[test]
fn a_run_canceled_before_its_batch_is_recorded_cancels_that_batch() {
    let scratch = Scratch::new();
    let store = &scratch.store;
    let eval = store
        .create_eval(&eval_request(json!([string_check()])), MODEL)
        .expect("created");
    let run = store
        .create_eval_run(&eval.id, &sampling_run(&rows()), MODEL)
        .expect("created");
    let lease = store.lease_eval_run(&run.id).expect("lease").expect("free");
    let row = load_run(&store.connection().expect("connection"), &run.id).expect("row");
    store.cancel_eval_run(&eval.id, &run.id).expect("canceled");
    let body = json!({"model": MODEL, "messages": [{"role": "user", "content": "1+1="}]});
    let lines = vec![(
        "item-0".to_owned(),
        body.as_object().cloned().expect("object"),
    )];
    let step = store
        .start_batch(
            &row,
            Phase::Sampling,
            Endpoint::ChatCompletions,
            &lines,
            MODEL,
        )
        .expect("started");
    assert_eq!(step, Some(Advance::Finished));
    let batch = store
        .find_batch(BATCH_KEY, &format!("{}:sampling", run.id))
        .expect("found")
        .expect("created");
    assert_eq!(
        store.get_batch(&batch).expect("batch").status,
        crate::batches::BatchStatus::Cancelling
    );
    drop(lease);
}

#[test]
fn a_batch_cancelled_elsewhere_fails_the_run() {
    let scratch = Scratch::new();
    let store = &scratch.store;
    let eval = store
        .create_eval(&eval_request(json!([string_check()])), MODEL)
        .expect("created");
    let run = store
        .create_eval_run(&eval.id, &sampling_run(&rows()), MODEL)
        .expect("created");
    let Advance::Started(sampling) = advance(store, &run.id) else {
        panic!("a sampling batch is started");
    };
    store.cancel_batch(&sampling).expect("cancelled");
    run_batch(store, &sampling, answer);
    assert_eq!(advance(store, &run.id), Advance::Finished);
    let run = store.get_eval_run(&eval.id, &run.id).expect("run");
    assert_eq!(run.status, RunStatus::Failed);
    assert_eq!(run.error.expect("error").code, "batch_cancelled");
}

#[test]
fn failed_generations_make_rows_errored_not_the_run() {
    let scratch = Scratch::new();
    let store = &scratch.store;
    let eval = store
        .create_eval(&eval_request(json!([string_check()])), MODEL)
        .expect("created");
    let run = store
        .create_eval_run(&eval.id, &sampling_run(&rows()), MODEL)
        .expect("created");
    let Advance::Started(sampling) = advance(store, &run.id) else {
        panic!("a sampling batch is started");
    };
    run_batch(store, &sampling, |body| {
        if last_user_text(body) == "3+3=" {
            LineOutcome::Response {
                status_code: 400,
                body: json!({"error": {"message": "context overflow", "type": "invalid_request_error", "code": null}}),
            }
        } else {
            answer(body)
        }
    });
    // No model graders: straight from sampling to scoring.
    assert_eq!(advance(store, &run.id), Advance::Finished);
    let run = store.get_eval_run(&eval.id, &run.id).expect("run");
    assert_eq!(
        run.result_counts,
        ResultCounts {
            total: 3,
            errored: 1,
            failed: 1,
            passed: 1
        }
    );
    let items = store
        .list_eval_output_items(&eval.id, &run.id, &ListOutputItems::default())
        .expect("listed");
    assert_eq!(
        items.data[2]["sample"]["error"],
        json!({"code": "invalid_request_error", "message": "context overflow"})
    );
    assert_eq!(items.data[2]["results"][0]["error"]["code"], "sample_error");
    assert_eq!(run.per_model_usage.expect("usage")[0].invocation_count, 3);
}

#[test]
fn runs_page_by_creation_and_filter_by_status() {
    let scratch = Scratch::new();
    let store = &scratch.store;
    let eval = store
        .create_eval(&eval_request(json!([string_check()])), MODEL)
        .expect("created");
    let ids: Vec<String> = (0..3)
        .map(|_| {
            store
                .create_eval_run(&eval.id, &sampling_run(&rows()), MODEL)
                .expect("created")
                .id
        })
        .collect();
    let page = store
        .list_eval_runs(
            &eval.id,
            &ListEvalRuns {
                limit: Some(2),
                ..ListEvalRuns::default()
            },
        )
        .expect("listed");
    assert_eq!(
        page.data.iter().map(|r| r.id.clone()).collect::<Vec<_>>(),
        ids[..2]
    );
    assert!(page.has_more);
    let desc = store
        .list_eval_runs(
            &eval.id,
            &ListEvalRuns {
                order: Some(Order::Desc),
                after: Some(ids[2].clone()),
                ..ListEvalRuns::default()
            },
        )
        .expect("listed");
    assert_eq!(
        desc.data.iter().map(|r| r.id.clone()).collect::<Vec<_>>(),
        vec![ids[1].clone(), ids[0].clone()]
    );
    let queued = store
        .list_eval_runs(
            &eval.id,
            &ListEvalRuns {
                status: Some(RunStatus::Completed),
                ..ListEvalRuns::default()
            },
        )
        .expect("listed");
    assert_eq!(queued.data.len(), 0);
    let other = store
        .create_eval(&eval_request(json!([string_check()])), MODEL)
        .expect("created");
    assert!(matches!(
        store.get_eval_run(&other.id, &ids[0]),
        Err(Error::NotFound(_))
    ));
    assert!(matches!(
        store.list_eval_runs(
            &other.id,
            &ListEvalRuns {
                after: Some(ids[0].clone()),
                ..ListEvalRuns::default()
            }
        ),
        Err(Error::NotFound(_))
    ));
}
