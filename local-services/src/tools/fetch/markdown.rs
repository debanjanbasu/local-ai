//! Response text for the model: HTML converted to Markdown, other text
//! decoded from its declared charset.
//!
//! The conversion (`html-to-markdown-rs`, an html5ever-based converter,
//! chosen over `htmd`, `fast_html2md` and `mdka` on real pages: the best
//! output, with tables, code blocks and absolute links, and a depth limit
//! where `htmd` overflowed the stack and `mdka` took seconds on deeply
//! nested markup) drops scripts, styles, comments, images, forms and
//! navigation chrome, keeps headings, paragraphs, lists, links (made
//! absolute against the page URL and any `<base>`), code blocks and tables,
//! and stops descending at a fixed depth so hostile nesting cannot exhaust
//! the stack. It costs milliseconds per page (about 70 ms for a 1 MB
//! rustdoc page).

use std::borrow::Cow;

use encoding_rs::Encoding;

/// Elements dropped with their content, on top of the converter's own
/// navigation, form and hidden-content removal.
const EXCLUDED: &[&str] = &[
    "script",
    "style",
    "noscript",
    "template",
    "svg",
    "iframe",
    "nav",
    "aside",
    "[role=navigation]",
];
/// Depth past which markup is skipped.
const MAX_DEPTH: usize = 512;
/// Stack of the conversion thread: room for the converter's recursion to
/// [`MAX_DEPTH`] with margin (reserved, not committed, memory).
pub const CONVERSION_STACK: usize = 32 << 20;

/// The kind of text a response body holds, by its media type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Kind {
    /// `text/html` or `application/xhtml+xml`: converted to Markdown.
    Html { charset: Option<String> },
    /// Other text (`text/*`, JSON, XML, JavaScript): decoded from its
    /// charset.
    Text { charset: Option<String> },
    /// Anything else: shown as UTF-8, lossily, as before.
    Other,
}

impl Kind {
    /// The kind a `Content-Type` value names.
    #[must_use]
    pub fn of(content_type: Option<&str>) -> Self {
        let Some(content_type) = content_type else {
            return Self::Other;
        };
        let mut parameters = content_type.split(';');
        let essence = parameters
            .next()
            .unwrap_or_default()
            .trim()
            .to_ascii_lowercase();
        let charset = parameters.find_map(|parameter| {
            let (name, value) = parameter.split_once('=')?;
            name.trim()
                .eq_ignore_ascii_case("charset")
                .then(|| value.trim().trim_matches('"').to_owned())
        });
        if essence == "text/html" || essence == "application/xhtml+xml" {
            Self::Html { charset }
        } else if essence.starts_with("text/")
            || essence.ends_with("+json")
            || essence.ends_with("+xml")
            || matches!(
                essence.as_str(),
                "application/json" | "application/xml" | "application/javascript"
            )
        {
            Self::Text { charset }
        } else {
            Self::Other
        }
    }
}

/// `bytes` decoded from `charset` (a byte-order mark wins, as in browsers);
/// UTF-8, with invalid sequences replaced, when it names none or an unknown
/// one.
#[must_use]
pub fn decode<'a>(bytes: &'a [u8], charset: Option<&str>) -> Cow<'a, str> {
    let encoding = charset
        .and_then(|label| Encoding::for_label(label.as_bytes()))
        .unwrap_or(encoding_rs::UTF_8);
    encoding.decode(bytes).0
}

/// `html`, from a page at `base_url`, as Markdown. The charset is the
/// header's, else a `<meta>` declaration in the first 1024 bytes, else
/// UTF-8.
#[must_use]
pub fn convert(html: &[u8], charset: Option<&str>, base_url: &str) -> String {
    let declared = charset.map(str::to_owned).or_else(|| meta_charset(html));
    let text = decode(html, declared.as_deref());
    let options = html_to_markdown_rs::ConversionOptions {
        base_url: Some(base_url.to_owned()),
        extract_metadata: false,
        skip_images: true,
        max_depth: Some(MAX_DEPTH),
        preprocessing: html_to_markdown_rs::PreprocessingOptions {
            preset: html_to_markdown_rs::PreprocessingPreset::Aggressive,
            ..html_to_markdown_rs::PreprocessingOptions::default()
        },
        exclude_selectors: EXCLUDED
            .iter()
            .map(|selector| (*selector).to_owned())
            .collect(),
        ..html_to_markdown_rs::ConversionOptions::default()
    };
    html_to_markdown_rs::convert(&text, options)
        .ok()
        .and_then(|result| result.content)
        .map(|markdown| markdown.trim().to_owned())
        .unwrap_or_default()
}

/// [`convert`] on a thread of its own, with a known stack, awaited without
/// blocking. A caller that stops waiting leaves it to finish its bounded
/// work.
pub async fn convert_on_thread(
    html: Vec<u8>,
    charset: Option<String>,
    base_url: String,
) -> Result<String, super::ToolError> {
    let (sender, receiver) = tokio::sync::oneshot::channel();
    std::thread::Builder::new()
        .name("local-ai-markdown".into())
        .stack_size(CONVERSION_STACK)
        .spawn(move || {
            let _ = sender.send(convert(&html, charset.as_deref(), &base_url));
        })
        .map_err(|error| super::ToolError::Failed(format!("cannot convert the page: {error}")))?;
    receiver
        .await
        .map_err(|_| super::ToolError::Failed("the page could not be converted to Markdown".into()))
}

/// The charset a `<meta charset>` or `<meta http-equiv=content-type>` in the
/// first 1024 bytes declares (the HTML prescan, simplified).
fn meta_charset(html: &[u8]) -> Option<String> {
    let head = html.get(..html.len().min(1024)).unwrap_or_default();
    let head = String::from_utf8_lossy(head).to_ascii_lowercase();
    head.split("<meta").skip(1).find_map(|tag| {
        let tag = tag.split('>').next().unwrap_or_default();
        let (_, rest) = tag.split_once("charset")?;
        let value = rest.trim_start().strip_prefix('=')?.trim_start();
        let value = value.trim_start_matches(['"', '\'']);
        let end = value
            .find(|c: char| matches!(c, '"' | '\'' | ';' | '/') || c.is_ascii_whitespace())
            .unwrap_or(value.len());
        let label = value.get(..end)?;
        Encoding::for_label(label.as_bytes()).map(|_| label.to_owned())
    })
}

#[cfg(test)]
mod tests {
    use super::{Kind, convert, decode};

    const PAGE: &str = r#"<!doctype html>
<html><head><title>Fixture</title>
<style>body { color: red }</style>
<script>var secret = "SCRIPT-TEXT";</script>
</head><body>
<nav><a href="/home">NAV-LINK</a> <a href="/about">About</a></nav>
<header><nav><ul><li><a href="/x">MENU-ITEM</a></li></ul></nav></header>
<!-- COMMENT-TEXT -->
<main>
<h1>Title &amp; more</h1>
<p>A paragraph with a <a href="guide/intro.html">relative link</a>, an
<a href="https://other.example/page">absolute one</a> and <code>inline code</code>.</p>
<h2>List</h2>
<ul><li>first</li><li>second <em>item</em></li></ul>
<ol><li>one</li><li>two</li></ol>
<pre><code class="language-rust">fn main() {
    println!("hi");
}</code></pre>
<table><thead><tr><th>Name</th><th>Value</th></tr></thead>
<tbody><tr><td>alpha</td><td>1</td></tr><tr><td>beta</td><td>2</td></tr></tbody></table>
<img src="/logo.png" alt="LOGO-ALT">
<form><input name="q"><button>SEARCH-BUTTON</button></form>
</main>
<footer><nav><a href="/legal">FOOTER-NAV</a></nav></footer>
</body></html>"#;

    #[test]
    fn html_becomes_clean_markdown() {
        let markdown = convert(PAGE.as_bytes(), None, "https://docs.example.com/a/b.html");
        for kept in [
            "# Title & more",
            "## List",
            "[relative link](https://docs.example.com/a/guide/intro.html)",
            "[absolute one](https://other.example/page)",
            "`inline code`",
            "- first",
            "- second *item*",
            "1. one",
            "2. two",
            "```rust\nfn main() {\n    println!(\"hi\");\n}\n```",
            "| Name  | Value |",
            "| alpha | 1     |",
            "| beta  | 2     |",
        ] {
            assert!(markdown.contains(kept), "missing {kept:?} in:\n{markdown}");
        }
        for dropped in [
            "SCRIPT-TEXT",
            "color: red",
            "NAV-LINK",
            "MENU-ITEM",
            "COMMENT-TEXT",
            "LOGO-ALT",
            "SEARCH-BUTTON",
            "FOOTER-NAV",
            "<",
        ] {
            assert!(
                !markdown.contains(dropped),
                "kept {dropped:?} in:\n{markdown}"
            );
        }
    }

    #[test]
    fn charsets_are_respected() {
        // "café" in windows-1252, declared by the header or by <meta>.
        let body =
            b"<html><head><meta charset=\"windows-1252\"></head><body><p>caf\xe9</p></body></html>";
        assert_eq!(convert(body, None, "https://a.test/"), "café");
        let undeclared = b"<p>caf\xe9</p>";
        assert_eq!(
            convert(undeclared, Some("ISO-8859-1"), "https://a.test/"),
            "café"
        );
        let shift_jis = b"<meta http-equiv=\"Content-Type\" content=\"text/html; charset=shift_jis\"><p>\x93\xfa\x96\x7b</p>";
        assert_eq!(convert(shift_jis, None, "https://a.test/"), "日本");
        assert_eq!(decode(b"caf\xe9", Some("latin1")), "café");
        assert_eq!(decode("café".as_bytes(), None), "café");
        assert_eq!(decode(b"caf\xe9", Some("no-such-charset")), "caf\u{fffd}");
    }

    #[test]
    fn media_types_choose_the_conversion() {
        for (content_type, expected) in [
            ("text/html", Kind::Html { charset: None }),
            (
                "Application/XHTML+XML; charset=\"Shift_JIS\"",
                Kind::Html {
                    charset: Some("Shift_JIS".into()),
                },
            ),
            ("application/json", Kind::Text { charset: None }),
            ("application/ld+json", Kind::Text { charset: None }),
            (
                "text/plain; format=flowed; charset=koi8-r",
                Kind::Text {
                    charset: Some("koi8-r".into()),
                },
            ),
            ("image/png", Kind::Other),
            ("application/octet-stream", Kind::Other),
        ] {
            assert_eq!(Kind::of(Some(content_type)), expected, "{content_type}");
        }
        assert_eq!(Kind::of(None), Kind::Other);
    }

    #[test]
    fn hostile_nesting_neither_overflows_nor_stalls() {
        let deep = "<div>".repeat(100_000) + "deep text";
        let started = std::time::Instant::now();
        let markdown = std::thread::Builder::new()
            .stack_size(super::CONVERSION_STACK)
            .spawn(move || convert(deep.as_bytes(), None, "https://a.test/"))
            .expect("thread")
            .join()
            .expect("no overflow");
        assert!(markdown.len() < 100, "{markdown}");
        assert!(started.elapsed().as_secs() < 10);
    }
}
