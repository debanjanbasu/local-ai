//! Template strings: `{{ item.path }}` and `{{ sample.path }}` references in
//! grader inputs and sampling messages.
//!
//! A reference is a dotted path from the `item` namespace (the data source
//! row) or the `sample` namespace (the sample being graded). A segment that
//! is an unsigned integer indexes an array. A string value renders as is and
//! any other value as compact JSON. A reference to a value that does not
//! exist is an error for that row, never an empty string. Text outside
//! `{{ }}` (and an unclosed `{{`) is literal.

use serde_json::Value;

/// The namespaces one rendering reads; `sample` is `None` when there is no
/// sample (yet).
#[derive(Clone, Copy, Debug)]
pub struct Namespaces<'a> {
    pub item: &'a Value,
    pub sample: Option<&'a Value>,
}

/// Each `{{ ... }}` reference in `template`, trimmed.
fn references(template: &str) -> impl Iterator<Item = (&str, &str, &str)> {
    // (literal before, reference, rest)
    let mut rest = Some(template);
    std::iter::from_fn(move || {
        let text = rest.take()?;
        let (before, after) = text.split_once("{{")?;
        let (reference, tail) = after.split_once("}}")?;
        rest = Some(tail);
        Some((before, reference.trim(), tail))
    })
}

/// Refuse a template whose references are malformed or name an unknown
/// namespace. Whether a value exists is known only per row.
pub fn check(template: &str) -> Result<(), String> {
    for (_, reference, _) in references(template) {
        check_path(reference)?;
    }
    Ok(())
}

/// Refuse a malformed path or one outside `item` and `sample`.
pub fn check_path(path: &str) -> Result<(), String> {
    let mut segments = path.split('.');
    match segments.next() {
        Some("item" | "sample") => {}
        _ => {
            return Err(format!(
                "template reference {{{{{path}}}}} must start with item. or sample."
            ));
        }
    }
    if path.split('.').any(str::is_empty) {
        return Err(format!(
            "template reference {{{{{path}}}}} has an empty segment"
        ));
    }
    Ok(())
}

/// The value `path` names.
pub fn lookup<'a>(path: &str, namespaces: Namespaces<'a>) -> Result<&'a Value, String> {
    check_path(path)?;
    let mut segments = path.split('.');
    let mut value = match segments.next() {
        Some("item") => namespaces.item,
        _ => namespaces
            .sample
            .ok_or_else(|| format!("{path} is referenced but this row has no sample"))?,
    };
    for segment in segments {
        let next = match value {
            Value::Object(map) => map.get(segment),
            Value::Array(items) => segment.parse::<usize>().ok().and_then(|i| items.get(i)),
            _ => None,
        };
        value = next.ok_or_else(|| format!("{path} does not exist in this row"))?;
    }
    Ok(value)
}

/// `template` with every reference replaced by its value.
pub fn render(template: &str, namespaces: Namespaces<'_>) -> Result<String, String> {
    let mut out = String::with_capacity(template.len());
    let mut tail = template;
    for (before, reference, rest) in references(template) {
        out.push_str(before);
        match lookup(reference, namespaces)? {
            Value::String(text) => out.push_str(text),
            other => out.push_str(&other.to_string()),
        }
        tail = rest;
    }
    out.push_str(tail);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn references_render_strings_raw_and_other_values_as_json() {
        let item = json!({"question": "2+2?", "tags": ["math", 4], "nested": {"n": 1}});
        let sample = json!({"output_text": "4"});
        let namespaces = Namespaces {
            item: &item,
            sample: Some(&sample),
        };
        assert_eq!(
            render(
                "Q: {{item.question}} A: {{ sample.output_text }} T: {{item.tags.1}} {{item.nested}}",
                namespaces
            ),
            Ok("Q: 2+2? A: 4 T: 4 {\"n\":1}".to_owned())
        );
        assert_eq!(
            render("no refs {{ unclosed", namespaces),
            Ok("no refs {{ unclosed".to_owned())
        );
        assert!(render("{{item.missing}}", namespaces).is_err());
        let without = Namespaces {
            item: &item,
            sample: None,
        };
        assert!(render("{{sample.output_text}}", without).is_err());
    }

    #[test]
    fn unknown_namespaces_and_empty_segments_are_refused() {
        assert!(check("{{item.a}} and {{sample.output_text}}").is_ok());
        assert!(check("{{items.a}}").is_err());
        assert!(check("{{item..a}}").is_err());
        assert!(check("{{ }}").is_err());
    }
}
