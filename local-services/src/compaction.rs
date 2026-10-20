//! Conversation compaction: the session-layer policy over conversation items.
//!
//! A long conversation is compacted by summarizing its older items with the
//! loaded model and keeping the rest verbatim. This module decides *when* to
//! compact and *what* is kept; it runs no model and renders nothing. The
//! engine owns the summary request, its rendering and the token counts (see
//! `local_engine::compaction`), and the caller wires the two together.
//!
//! Items are the conversation item forms of [`crate::conversations`]
//! (`message`, `reasoning`, `function_call`, `function_call_output`) plus the
//! `compaction` item this produces. The rules:
//!
//! - A `function_call` and its `function_call_output` are never separated: no
//!   cut falls while a call is unanswered, so an unanswered call (pending
//!   work) is always kept verbatim, as is every item after it.
//! - An assistant turn — reasoning, message and calls — is never cut inside.
//! - The leading `system` and `developer` messages (the standing task
//!   instructions) and the latest `user` message (the current request) are
//!   always kept verbatim, as are the newest items up to a token budget.
//! - Raw reasoning is only ever summarized as part of the history; it is
//!   never relabeled as, or returned as, a summary.
//!
//! A compaction item is self-contained: it carries the summary *and* the
//! kept items ([`Compacted`]), so a prompt is rendered from the latest
//! compaction item onward and every item before it is skipped, however the
//! client or store kept them. Re-compacting folds the earlier summary in.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// The `type` of a compaction item.
pub const COMPACTION: &str = "compaction";

/// What a compaction item stands for: the summary of the items it replaced,
/// and the items it keeps verbatim, in order.
///
/// Rendered as the kept leading instructions (if any), then the summary, then
/// the remaining kept items. The serialized form is what a caller seals into
/// the item's `encrypted_content`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Compacted {
    /// The model-written summary of the replaced items.
    pub summary: String,
    /// Items kept verbatim, oldest first.
    pub items: Vec<Value>,
}

/// Which items one compaction replaces and which it keeps.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Plan {
    /// Items `[..split]` are summarized (after any earlier summary).
    pub split: usize,
    /// The items kept verbatim: the leading instructions and the latest user
    /// message when they fall before `split`, then every item from `split`.
    pub kept: Vec<Value>,
}

fn kind(item: &Value) -> &str {
    item.get("type")
        .and_then(Value::as_str)
        .unwrap_or("message")
}

fn role(item: &Value) -> Option<&str> {
    item.get("role").and_then(Value::as_str)
}

fn call_id(item: &Value) -> Option<&str> {
    item.get("call_id").and_then(Value::as_str)
}

/// Whether `item` is a compaction item.
#[must_use]
pub fn is_compaction(item: &Value) -> bool {
    kind(item) == COMPACTION
}

/// The index of the latest compaction item in `items`, from which a prompt
/// is rendered.
#[must_use]
pub fn latest(items: &[Value]) -> Option<usize> {
    items.iter().rposition(is_compaction)
}

/// The prompt length at which a conversation is compacted when its caller
/// names no threshold of its own (`context_management` without a
/// `compact_threshold`). Chosen by the compaction threshold sweep in
/// docs/BONSAI.md.
pub const DEFAULT_COMPACT_THRESHOLD: usize = 32_000;

/// Whether a prompt of `prompt_tokens` should be compacted before it is
/// generated from: it has reached the caller's `threshold`, or it leaves less
/// than `reserve` output tokens in a `context`-token window.
#[must_use]
pub const fn should_compact(
    prompt_tokens: usize,
    threshold: Option<usize>,
    reserve: usize,
    context: usize,
) -> bool {
    let over = match threshold {
        Some(threshold) => prompt_tokens >= threshold,
        None => false,
    };
    over || prompt_tokens.saturating_add(reserve) > context
}

/// How many leading instruction messages `items` starts with (Codex sends
/// several): kept, and never the only thing summarized.
fn head(items: &[Value]) -> usize {
    items
        .iter()
        .take_while(|item| {
            kind(item) == "message" && matches!(role(item), Some("system" | "developer"))
        })
        .count()
}

/// Every index `cut` at which `items[..cut]` may be summarized and
/// `items[cut..]` kept, ascending: no call open across it, no turn split.
fn cuts(items: &[Value]) -> Vec<usize> {
    let mut open: Vec<&str> = Vec::new();
    let mut cuts = Vec::new();
    for (index, item) in items.iter().enumerate() {
        match kind(item) {
            "function_call" => open.extend(call_id(item)),
            "function_call_output" => {
                if let Some(id) = call_id(item) {
                    open.retain(|open| *open != id);
                }
            }
            _ => {}
        }
        let cut = index + 1;
        let next = items.get(cut);
        let same_turn = match next.map(kind) {
            // A call belongs to the assistant turn before it.
            Some("function_call") => {
                matches!(kind(item), "reasoning" | "function_call")
                    || (kind(item) == "message" && role(item) == Some("assistant"))
            }
            // An output answers a call, so it never starts what is kept.
            Some("function_call_output") => true,
            _ => false,
        };
        // Reasoning belongs to the output that follows it.
        if open.is_empty() && !same_turn && kind(item) != "reasoning" {
            cuts.push(cut);
        }
    }
    cuts
}

/// Plan one compaction of `items`: the earliest cut whose kept items count
/// at most `keep_tokens` by `count`, or the latest cut when none does.
///
/// `count` measures the prompt tokens of the items a cut would keep (the
/// leading instructions, the latest user message and the tail, so always a
/// renderable prompt when `items` is one); it is called a logarithmic number
/// of times. `None` when nothing can be summarized: no cut leaves more than
/// the leading instructions behind it.
pub fn plan<E>(
    items: &[Value],
    keep_tokens: usize,
    mut count: impl FnMut(&[Value]) -> Result<usize, E>,
) -> Result<Option<Plan>, E> {
    let head = head(items);
    let cuts: Vec<usize> = cuts(items).into_iter().filter(|cut| *cut > head).collect();
    let latest_user = items
        .iter()
        .rposition(|item| kind(item) == "message" && role(item) == Some("user"));
    let kept = |split: usize| {
        let mut kept: Vec<Value> = items.get(..head).unwrap_or_default().to_vec();
        if let Some(user) = latest_user.filter(|user| *user < split)
            && let Some(item) = items.get(user)
        {
            kept.push(item.clone());
        }
        kept.extend_from_slice(items.get(split..).unwrap_or_default());
        kept
    };
    let (mut low, mut high) = (0, cuts.len());
    // The first cut whose kept items fit; they shrink as the cut moves later.
    while low < high {
        let middle = low + (high - low) / 2;
        if count(&kept(cuts[middle]))? <= keep_tokens {
            high = middle;
        } else {
            low = middle + 1;
        }
    }
    let Some(&split) = cuts.get(low).or_else(|| cuts.last()) else {
        return Ok(None);
    };
    Ok(Some(Plan {
        split,
        kept: kept(split),
    }))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::unnecessary_wraps)]
mod tests {
    use serde_json::json;

    use super::*;

    fn user(text: &str) -> Value {
        json!({"type":"message","role":"user","content":text})
    }

    fn assistant(text: &str) -> Value {
        json!({"type":"message","role":"assistant","content":text})
    }

    fn call(id: &str) -> Value {
        json!({"type":"function_call","call_id":id,"name":"f","arguments":"{}"})
    }

    fn output(id: &str) -> Value {
        json!({"type":"function_call_output","call_id":id,"output":"ok"})
    }

    fn reasoning() -> Value {
        json!({"type":"reasoning","summary":[],"content":[{"type":"reasoning_text","text":"r"}]})
    }

    /// One token per item, so budgets read as item counts.
    fn by_items(items: &[Value]) -> Result<usize, ()> {
        Ok(items.len())
    }

    #[test]
    fn calls_and_outputs_are_never_separated_and_turns_never_split() {
        let items = vec![
            user("a"),
            reasoning(),
            assistant("checking"),
            call("1"),
            call("2"),
            output("1"),
            output("2"),
            assistant("done"),
            user("b"),
        ];
        // Inside the call group nothing may be cut: not between the
        // reasoning and its message, the message and its calls, the calls
        // and their outputs, or the two outputs.
        assert_eq!(cuts(&items), vec![1, 7, 8, 9]);
        for keep in 0..items.len() {
            let plan = plan(&items, keep, by_items).unwrap().unwrap();
            let summarized = &items[..plan.split];
            let ids = |slice: &[Value], kind_name: &str| -> Vec<String> {
                slice
                    .iter()
                    .filter(|item| kind(item) == kind_name)
                    .filter_map(|item| call_id(item).map(str::to_owned))
                    .collect()
            };
            assert_eq!(
                ids(summarized, "function_call"),
                ids(summarized, "function_call_output")
            );
        }
    }

    #[test]
    fn unanswered_calls_and_everything_after_them_are_kept() {
        let items = vec![
            user("a"),
            assistant("x"),
            user("b"),
            call("9"),
            user("c"),
            assistant("y"),
        ];
        let plan = plan(&items, 0, by_items).unwrap().unwrap();
        assert_eq!(plan.split, 3);
        assert_eq!(plan.kept, vec![call("9"), user("c"), assistant("y")]);
    }

    #[test]
    fn instructions_and_the_latest_request_are_kept_verbatim() {
        let system = json!({"type":"message","role":"developer","content":"Be terse."});
        let items = vec![
            system.clone(),
            user("task"),
            assistant("step 1"),
            call("1"),
            output("1"),
            assistant("step 2"),
            call("2"),
            output("2"),
            assistant("step 3"),
        ];
        let plan = plan(&items, 3, by_items).unwrap().unwrap();
        assert_eq!(plan.split, 8);
        assert_eq!(plan.kept, vec![system, user("task"), assistant("step 3")]);
    }

    #[test]
    fn the_budget_picks_the_earliest_cut_that_fits() {
        let items = vec![
            user("1"),
            assistant("1"),
            user("2"),
            assistant("2"),
            user("3"),
            assistant("3"),
        ];
        let mut calls = 0;
        let plan = plan(&items, 2, |slice: &[Value]| {
            calls += 1;
            Ok::<_, ()>(slice.len())
        })
        .unwrap()
        .unwrap();
        assert_eq!(plan.split, 4);
        assert_eq!(plan.kept, items[4..].to_vec());
        assert!(calls <= 3, "binary search, not a scan: {calls}");
        // A budget nothing fits still summarizes up to the latest cut.
        let all = super::plan(&items, 0, by_items).unwrap().unwrap();
        assert_eq!(all.split, 6);
        assert_eq!(all.kept, vec![user("3")]);
    }

    #[test]
    fn nothing_to_summarize_is_none() {
        let system = json!({"type":"message","role":"system","content":"s"});
        assert_eq!(plan(&[], 0, by_items).unwrap(), None);
        assert_eq!(plan(&[system], 0, by_items).unwrap(), None);
        assert_eq!(plan(&[call("1")], 0, by_items).unwrap(), None);
    }

    #[test]
    fn the_latest_compaction_item_starts_the_prompt() {
        let compaction = |id: &str| json!({"type":"compaction","id":id,"encrypted_content":"e"});
        let items = vec![
            user("a"),
            compaction("cmp_1"),
            user("b"),
            compaction("cmp_2"),
        ];
        assert_eq!(latest(&items), Some(3));
        assert_eq!(latest(&items[..3]), Some(1));
        assert_eq!(latest(&items[..1]), None);
    }

    #[test]
    fn compaction_is_due_at_the_threshold_or_without_room_for_output() {
        assert!(!should_compact(999, Some(1000), 100, 4096));
        assert!(should_compact(1000, Some(1000), 100, 4096));
        assert!(!should_compact(3996, None, 100, 4096));
        assert!(should_compact(3997, None, 100, 4096));
        assert!(should_compact(usize::MAX, None, usize::MAX, 4096));
    }
}
