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
//!
//! The page's main content is extracted first ([`super::extract`]). Both
//! steps are guarded: an extraction that panics converts the whole page,
//! and a conversion that panics or fails gives the page's plain text
//! ([`plain_text`]), so a page the libraries cannot handle still reads.

use std::borrow::Cow;

use encoding_rs::Encoding;

/// Elements dropped with their content, on top of the converter's own
/// navigation, form and hidden-content removal.
pub const EXCLUDED: &[&str] = &[
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
pub const MAX_DEPTH: usize = 512;
/// Stack of the conversion thread: room for the converter's recursion to
/// [`MAX_DEPTH`] with margin (reserved, not committed, memory).
pub const CONVERSION_STACK: usize = 32 << 20;

/// The kind of text a response body holds, by its media type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Kind {
    /// `text/html` or `application/xhtml+xml`: converted to Markdown.
    Html { charset: Option<String> },
    /// `application/pdf`, or a URL path ending in `.pdf` with a media type
    /// that is neither HTML nor text: its text is extracted
    /// ([`super::pdf`]).
    Pdf,
    /// Other text (`text/*`, JSON, XML, JavaScript): decoded from its
    /// charset.
    Text { charset: Option<String> },
    /// Anything else: shown as UTF-8, lossily, as before.
    Other,
}

impl Kind {
    /// The kind a `Content-Type` value names for a response to `path` (the
    /// request's path and query).
    #[must_use]
    pub fn of(content_type: Option<&str>, path: &str) -> Self {
        let path = path.split(['?', '#']).next().unwrap_or_default();
        let pdf_path = path
            .get(path.len().saturating_sub(4)..)
            .is_some_and(|end| end.eq_ignore_ascii_case(".pdf"));
        let Some(content_type) = content_type else {
            return if pdf_path { Self::Pdf } else { Self::Other };
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
        } else if essence == "application/pdf" || essence == "application/x-pdf" {
            Self::Pdf
        } else if essence.starts_with("text/")
            || essence.ends_with("+json")
            || essence.ends_with("+xml")
            || matches!(
                essence.as_str(),
                "application/json" | "application/xml" | "application/javascript"
            )
        {
            Self::Text { charset }
        } else if pdf_path {
            Self::Pdf
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

/// How a page was turned into text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum How {
    /// Its main content, as Markdown.
    MainContent,
    /// The whole page, as Markdown: no main content stood out, or the
    /// extraction failed.
    WholePage,
    /// Plain text: the Markdown conversion failed.
    PlainText,
}

/// `html`, from a page at `base_url`, as Markdown, and how. The charset is
/// the header's, else a `<meta>` declaration in the first 1024 bytes, else
/// UTF-8.
#[must_use]
pub fn convert(html: &[u8], charset: Option<&str>, base_url: &str) -> (String, How) {
    let declared = charset.map(str::to_owned).or_else(|| meta_charset(html));
    let text = decode(html, declared.as_deref());
    convert_with(&text, base_url, super::extract::main_content, to_markdown)
}

/// [`convert`] with its two steps given: `extract` the main content (or
/// `None` for the whole page), then convert it to `markdown` (`None` on
/// failure). A panic in either is caught (the panic hook still reports
/// it).
fn convert_with(
    html: &str,
    base_url: &str,
    extract: fn(&str, &str) -> Option<String>,
    markdown: fn(&str, &str) -> Option<String>,
) -> (String, How) {
    let main = std::panic::catch_unwind(|| extract(html, base_url))
        .ok()
        .flatten();
    let (source, how) = main
        .as_deref()
        .map_or((html, How::WholePage), |main| (main, How::MainContent));
    match std::panic::catch_unwind(|| markdown(source, base_url)) {
        Ok(Some(markdown)) => (markdown, how),
        _ => (plain_text(source), How::PlainText),
    }
}

/// `html` as Markdown, links resolved against `base_url`; `None` if the
/// converter fails.
fn to_markdown(html: &str, base_url: &str) -> Option<String> {
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
    let markdown = html_to_markdown_rs::convert(html, options).ok()?.content?;
    Some(markdown.trim().to_owned())
}

/// Elements whose content is not text.
const RAW_TEXT: &[&str] = &["script", "style", "noscript", "template", "svg", "iframe"];
/// Elements that end a line of text.
const BLOCKS: &[&str] = &[
    "address",
    "article",
    "aside",
    "blockquote",
    "br",
    "dd",
    "div",
    "dl",
    "dt",
    "figure",
    "footer",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "header",
    "hr",
    "li",
    "main",
    "nav",
    "ol",
    "p",
    "pre",
    "section",
    "table",
    "td",
    "th",
    "tr",
    "ul",
];

/// `html`'s text without markup, for when conversion fails: tags dropped
/// (block elements end a line), comments and the content of scripts,
/// styles and the like dropped, common character references decoded, and
/// runs of blank space collapsed. One scan of the string, nothing that can
/// fail.
#[must_use]
pub fn plain_text(html: &str) -> String {
    // ASCII lowercasing keeps every byte offset.
    let lower = html.to_ascii_lowercase();
    let mut text = String::with_capacity(html.len() / 2);
    let mut at = 0;
    while let Some(open) = lower.get(at..).and_then(|rest| rest.find('<')) {
        let open = at + open;
        decode_references(html.get(at..open).unwrap_or_default(), &mut text);
        let tag = lower.get(open..).unwrap_or_default();
        if tag.starts_with("<!--") {
            at = tag.find("-->").map_or(html.len(), |end| open + end + 3);
            continue;
        }
        at = tag.find('>').map_or(html.len(), |end| open + end + 1);
        let name: String = tag
            .get(1..)
            .unwrap_or_default()
            .trim_start_matches('/')
            .chars()
            .take_while(char::is_ascii_alphanumeric)
            .collect();
        if !tag.starts_with("</") && RAW_TEXT.contains(&name.as_str()) {
            let close = format!("</{name}");
            at = lower
                .get(at..)
                .and_then(|rest| rest.find(&close))
                .map_or(html.len(), |close| at + close);
        } else if BLOCKS.contains(&name.as_str()) {
            text.push('\n');
        }
    }
    decode_references(html.get(at..).unwrap_or_default(), &mut text);
    let mut collapsed = String::with_capacity(text.len());
    for line in text.lines() {
        let words: Vec<&str> = line.split_whitespace().collect();
        if !words.is_empty() {
            collapsed.push_str(&words.join(" "));
            collapsed.push('\n');
        } else if !collapsed.is_empty() && !collapsed.ends_with("\n\n") {
            collapsed.push('\n');
        }
    }
    collapsed.trim().to_owned()
}

/// Append `raw` to `text` with the common named and all numeric character
/// references decoded; others are kept as written.
fn decode_references(raw: &str, text: &mut String) {
    let mut rest = raw;
    while let Some(amp) = rest.find('&') {
        text.push_str(rest.get(..amp).unwrap_or_default());
        let after = rest.get(amp + 1..).unwrap_or_default();
        let reference = after
            .find(';')
            .filter(|end| *end <= 10)
            .and_then(|end| Some((after.get(..end)?, end)));
        let decoded = reference.and_then(|(name, _)| match name {
            "amp" => Some('&'),
            "lt" => Some('<'),
            "gt" => Some('>'),
            "quot" => Some('"'),
            "apos" => Some('\''),
            "nbsp" => Some(' '),
            _ => {
                let number = name.strip_prefix('#')?;
                let code = match number.strip_prefix(['x', 'X']) {
                    Some(hex) => u32::from_str_radix(hex, 16).ok()?,
                    None => number.parse().ok()?,
                };
                char::from_u32(code)
            }
        });
        if let (Some(character), Some((_, end))) = (decoded, reference) {
            text.push(character);
            rest = after.get(end + 1..).unwrap_or_default();
        } else {
            text.push('&');
            rest = after;
        }
    }
    text.push_str(rest);
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
    use super::{How, Kind, convert, convert_with, decode, plain_text, to_markdown};

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
        let (markdown, how) = convert(PAGE.as_bytes(), None, "https://docs.example.com/a/b.html");
        // Too little text to score: converted whole.
        assert_eq!(how, How::WholePage);
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
        assert_eq!(convert(body, None, "https://a.test/").0, "café");
        let undeclared = b"<p>caf\xe9</p>";
        assert_eq!(
            convert(undeclared, Some("ISO-8859-1"), "https://a.test/").0,
            "café"
        );
        let shift_jis = b"<meta http-equiv=\"Content-Type\" content=\"text/html; charset=shift_jis\"><p>\x93\xfa\x96\x7b</p>";
        assert_eq!(convert(shift_jis, None, "https://a.test/").0, "日本");
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
            assert_eq!(
                Kind::of(Some(content_type), "/"),
                expected,
                "{content_type}"
            );
        }
        assert_eq!(Kind::of(None, "/"), Kind::Other);
        // PDFs by media type, or by a `.pdf` path when the media type says
        // neither HTML nor text.
        assert_eq!(Kind::of(Some("application/pdf"), "/paper"), Kind::Pdf);
        assert_eq!(Kind::of(Some("Application/X-PDF"), "/"), Kind::Pdf);
        for path in ["/rfc/rfc9114.pdf", "/a/B.PDF?download=1", "/x.pdf#page=2"] {
            assert_eq!(Kind::of(Some("application/octet-stream"), path), Kind::Pdf);
            assert_eq!(Kind::of(None, path), Kind::Pdf);
        }
        assert_eq!(
            Kind::of(Some("text/html"), "/viewer.pdf"),
            Kind::Html { charset: None }
        );
        assert_eq!(Kind::of(Some("image/png"), "/pdf"), Kind::Other);
    }

    fn fails(_: &str, _: &str) -> Option<String> {
        std::panic::resume_unwind(Box::new("a library bug"))
    }

    #[test]
    fn failures_fall_back_to_the_whole_page_then_to_plain_text() {
        let page = "<html><body><main><h1>Kept &amp; well</h1><p>Body&nbsp;text &#x263A; \
                    &bogus; &#169;</p><script>SCRIPT-TEXT</script><!-- COMMENT --></main>\
                    <STYLE>STYLE-TEXT</STYLE><ul><li>one</li><li>two</li></ul></body></html>";
        let url = "https://a.test/";
        // An extraction that panics: the whole page is converted.
        let (markdown, how) = convert_with(page, url, fails, to_markdown);
        assert_eq!(how, How::WholePage);
        assert!(markdown.starts_with("# Kept & well"), "{markdown}");
        // A conversion that panics or fails: the page's plain text.
        let plain = "Kept & well\n\nBody text \u{263a} &bogus; \u{a9}\n\none\n\ntwo";
        for markdown in [
            fails as fn(&str, &str) -> Option<String>,
            |_: &str, _: &str| None,
        ] {
            assert_eq!(
                convert_with(page, url, |_, _| None, markdown),
                (plain.to_owned(), How::PlainText)
            );
        }
        // After an extraction, the plain text is the main content's.
        let (text, how) = convert_with(page, url, |_, _| Some("<p>MAIN</p>".into()), fails);
        assert_eq!((text.as_str(), how), ("MAIN", How::PlainText));
        // Unclosed markup ends the text rather than failing.
        assert_eq!(plain_text("a<b"), "a");
        assert_eq!(plain_text("a<script>b"), "a");
        assert_eq!(plain_text("a<!-- b"), "a");
    }

    #[test]
    fn hostile_nesting_neither_overflows_nor_stalls() {
        let deep = "<div>".repeat(100_000) + "deep text";
        let started = std::time::Instant::now();
        let markdown = std::thread::Builder::new()
            .stack_size(super::CONVERSION_STACK)
            .spawn(move || convert(deep.as_bytes(), None, "https://a.test/").0)
            .expect("thread")
            .join()
            .expect("no overflow");
        assert!(markdown.len() < 100, "{markdown}");
        assert!(started.elapsed().as_secs() < 10);
    }
}
