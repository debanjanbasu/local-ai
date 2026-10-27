//! Main-content extraction: the part of a page that holds its article or
//! main text, chosen before Markdown conversion so that headers, footers,
//! cookie banners, sidebars and other chrome do not reach the model.
//!
//! - The scope is the page's `<main>` (or `role="main"`) when it holds at
//!   least a quarter of the page's text, else its largest `<article>` when
//!   that does, else the whole body.
//! - Within the scope, `dom_smoothie` (a port of Mozilla's Readability)
//!   scores blocks by their text, commas and link density and keeps the
//!   best one with its related siblings. It runs with its `Raw` policy:
//!   Readability's stricter passes drop elements by class name and clean
//!   blocks conditionally, which removed rustdoc's method signatures and
//!   most of Wikipedia's section headings, so they are off; the converter
//!   already drops navigation, forms, scripts and images.
//! - Readability removes the page title from the text and turns `<h1>`
//!   into `<h2>`, so the title comes back as the one `<h1>`. Its byline
//!   heuristic would remove the first short element whose class or id
//!   names an author (MDN's `<dt id="alt-authority">`, say), so it is
//!   given a byline and never looks.
//! - Permalink anchors (`§`, `¶`) and long-URL buttons without text (a
//!   playground link per code example) are dropped.
//! - Nothing is lost for want of a good guess: a page with little text, a
//!   larger or deeper document than the bounds below, or an extraction
//!   that keeps less than a quarter of the page's text gives `None`, and
//!   the whole page is converted.
//!
//! Links come back absolute (against the page URL and any `<base>`); code
//! blocks and tables are kept. It costs less than the conversion (about
//! 50 ms for rustdoc's 1 MB `Vec` page).

use dom_query::Document;
use dom_smoothie::{Config, ParsePolicy, Readability};

/// Text (non-whitespace bytes) below which a page is converted whole:
/// there is little chrome to save and too little text to score.
const MIN_PAGE_TEXT: usize = 500;
/// The share of the page's text, in percent, a scope must hold to be used
/// and an extraction must keep to be returned (else the whole page is
/// converted).
const MIN_SHARE_PERCENT: usize = 25;
/// URL length past which a link without text is taken for a button.
const MAX_BUTTON_URL: usize = 128;
/// Elements past which a document is converted whole, unparsed.
const MAX_ELEMENTS: usize = 100_000;
/// Nesting past which a document is converted whole, unparsed (the
/// converter's own depth limit).
const MAX_DEPTH: usize = super::markdown::MAX_DEPTH;

/// The main content of `html`, a page at `page_url`, as HTML with its
/// title as the `<h1>` and absolute links; `None` to convert the whole
/// page.
#[must_use]
pub fn main_content(html: &str, page_url: &str) -> Option<String> {
    if !within_bounds(html) {
        return None;
    }
    let document = Document::from(html);
    // What the converter drops goes first, so the page's text is the text
    // the converter would keep.
    document
        .select(&super::markdown::EXCLUDED.join(", "))
        .remove();
    remove_control_links(&document);
    let page = text_len(&document.select("body").text());
    if page < MIN_PAGE_TEXT {
        return None;
    }
    if let Some(scope) = scope(&document, page) {
        document.select("body").set_html(scope);
    }
    // A byline from the metadata stops Readability from looking for one in
    // the text and removing it.
    document
        .select("head")
        .append_html(r#"<meta name="author" content="local-ai">"#);
    // Classes are kept: the converter reads code languages from them.
    let config = Config {
        disable_json_ld: true,
        keep_classes: true,
        ..Config::default()
    };
    let article = Readability::with_document(document, Some(page_url), Some(config))
        .ok()?
        .parse_with_policy(ParsePolicy::Raw)
        .ok()?;
    if !enough(text_len(&article.text_content), page) {
        return None;
    }
    let title = article.title.trim();
    let content = article.content.to_string();
    if title.is_empty() || article.text_content.trim_start().starts_with(title) {
        return Some(content);
    }
    let title = title
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;");
    Some(format!("<h1>{title}</h1>{content}"))
}

/// Remove links that are controls rather than references: permalink
/// anchors (a `§`, `¶` or `#`, or nothing, linking within the page) and
/// buttons without text whose URL is long, such as rustdoc's "Run code"
/// playground links, each carrying its example's code again (a quarter of
/// the `Vec` page's Markdown).
fn remove_control_links(document: &Document) {
    for link in document.select("a[href]").nodes() {
        let href = link.attr("href").unwrap_or_default();
        let text = link.text();
        let text = text.trim();
        let permalink = href.starts_with('#') && matches!(text, "" | "§" | "¶" | "#");
        let button = text.is_empty() && href.len() > MAX_BUTTON_URL;
        if permalink || button {
            link.remove_from_parent();
        }
    }
}

/// The HTML of the page's main element or article, if one holds enough of
/// its text.
fn scope(document: &Document, page: usize) -> Option<String> {
    ["main, [role=main]", "article"]
        .into_iter()
        .find_map(|selector| {
            document
                .select(selector)
                .nodes()
                .iter()
                .map(|node| (text_len(&node.text()), node))
                .max_by_key(|(text, _)| *text)
                .filter(|(text, _)| enough(*text, page))
                .map(|(_, node)| node.html().to_string())
        })
}

/// Whether `kept` bytes of text are enough of a page's `page`.
const fn enough(kept: usize, page: usize) -> bool {
    kept.saturating_mul(100) >= page.saturating_mul(MIN_SHARE_PERCENT)
}

/// Non-whitespace bytes of `text`.
fn text_len(text: &str) -> usize {
    text.split_whitespace().map(str::len).sum()
}

/// Elements that hold no content or close implicitly: they do not deepen
/// the nesting [`within_bounds`] counts.
const FLAT: &[&str] = &[
    "area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source",
    "track", "wbr", "p", "li", "dt", "dd", "tr", "td", "th", "option", "optgroup", "rb", "rt",
    "rp",
];
/// Elements whose content is text, not markup.
const RAW_TEXT: &[&str] = &[
    "script", "style", "textarea", "title", "xmp", "iframe", "noembed", "noframes",
];

/// Whether `html` has at most [`MAX_ELEMENTS`] elements nested at most
/// [`MAX_DEPTH`] deep, by a scan of its tags before any parse: parsing
/// costs time that grows with the square of the nesting. An approximate
/// count (implicitly closed elements other than [`FLAT`] ones count as
/// open) errs towards converting the whole page.
fn within_bounds(html: &str) -> bool {
    // ASCII lowercasing keeps every byte offset.
    let lower = html.to_ascii_lowercase();
    let mut elements = 0;
    let mut depth: usize = 0;
    let mut at = 0;
    while let Some(open) = lower.get(at..).and_then(|rest| rest.find('<')) {
        at += open + 1;
        let rest = lower.get(at..).unwrap_or_default();
        let (closing, rest) = rest
            .strip_prefix('/')
            .map_or((false, rest), |rest| (true, rest));
        let name = rest
            .split(|c: char| !c.is_ascii_alphanumeric())
            .next()
            .unwrap_or_default();
        let flat = FLAT.contains(&name);
        if !name.starts_with(|c: char| c.is_ascii_alphabetic()) || (closing && flat) {
            continue;
        }
        if closing {
            depth = depth.saturating_sub(1);
        } else if RAW_TEXT.contains(&name) {
            let close = format!("</{name}");
            at = lower
                .get(at..)
                .and_then(|rest| rest.find(&close))
                .map_or(lower.len(), |end| at + end);
        } else {
            elements += 1;
            depth += usize::from(!flat);
            if elements > MAX_ELEMENTS || depth > MAX_DEPTH {
                return false;
            }
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::super::markdown::{How, convert};
    use super::main_content;

    /// A paragraph of article text, with the commas Readability scores.
    fn paragraph(topic: &str) -> String {
        format!(
            "<p>The {topic} section explains, in some detail, how the system behaves under \
             load, why its queues stay short, and what an operator should watch when the \
             numbers drift. It is written for people who run it, not for its authors.</p>"
        )
    }

    fn article_page() -> String {
        format!(
            r##"<!doctype html><html><head><title>Queue tuning guide - Example Docs</title>
<base href="https://docs.example.com/guide/"></head><body>
<div id="cookie-banner">COOKIE-BANNER: we use cookies <button>Accept</button></div>
<header><div class="logo">SITE-LOGO</div><div class="menu"><a href="/a">MENU-A</a>
<a href="/b">MENU-B</a></div></header>
<div class="layout">
<div class="sidebar"><h3>SIDEBAR-POPULAR</h3><ul><li><a href="/p1">Popular one</a></li>
<li><a href="/p2">Popular two</a></li></ul></div>
<article><h1>Queue tuning guide</h1>
<p class="byline" id="author">By Ada Example</p>
{p1}{p2}
<h2 id="usage"><a class="doc-anchor" href="#usage">§</a>Usage</h2>
<pre><code class="language-rust">fn main() {{ tune(); }}</code></pre>
<a class="test-arrow" title="Run code" href="https://play.example.com/?code={code}"></a>
<table><tr><th>Setting</th><th>Default</th></tr><tr><td>depth</td><td>64</td></tr></table>
{p3}
<p>Read <a href="limits.html">the limits page</a> next.</p>
<dl><dt id="alt-authority"><code>&lt;alt-authority&gt;</code></dt><dd>An alternative.</dd></dl>
</article></div>
<footer><p>FOOTER-COPYRIGHT 2026, all rights reserved, by Example Corp.</p>
<div><a href="/legal">FOOTER-LEGAL</a></div></footer>
</body></html>"##,
            p1 = paragraph("first"),
            p2 = paragraph("second"),
            p3 = paragraph("third"),
            code = "fn+main()+%7B+tune();+%7D".repeat(12),
        )
    }

    #[test]
    fn an_article_keeps_its_text_and_drops_the_chrome() {
        let page = article_page();
        let (markdown, how) = convert(page.as_bytes(), None, "https://docs.example.com/x");
        assert_eq!(how, How::MainContent, "{markdown}");
        for kept in [
            "# Queue tuning guide",
            "By Ada Example",
            "The first section explains",
            "The third section explains",
            "```rust\nfn main() { tune(); }\n```",
            "| Setting | Default |",
            "| depth   | 64      |",
            "[the limits page](https://docs.example.com/guide/limits.html)",
            "`<alt-authority>`",
            "## Usage\n",
        ] {
            assert!(markdown.contains(kept), "missing {kept:?} in:\n{markdown}");
        }
        for dropped in [
            "COOKIE-BANNER",
            "SITE-LOGO",
            "MENU-A",
            "SIDEBAR-POPULAR",
            "FOOTER-COPYRIGHT",
            "FOOTER-LEGAL",
            "§",
            "play.example.com",
        ] {
            assert!(
                !markdown.contains(dropped),
                "kept {dropped:?} in:\n{markdown}"
            );
        }
    }

    #[test]
    fn a_page_without_landmarks_is_scored() {
        // Div soup: no <main>, <article>, <nav> or <footer> to go by.
        let links = (0..40)
            .map(|n| format!(r#"<div class="item"><a href="/s/{n}">MENU-LINK-{n}</a></div>"#))
            .collect::<Vec<_>>()
            .concat();
        let page = format!(
            r#"<html><head><title>Release notes</title></head><body>
<div id="top">{links}</div>
<div id="wrap"><div class="content"><h2>Release notes</h2>{}{}{}{}</div>
<div class="related"><a href="/r1">RELATED-ONE</a> <a href="/r2">RELATED-TWO</a></div></div>
<div id="bottom"><a href="/c">BOTTOM-CONTACT</a></div></body></html>"#,
            paragraph("install"),
            paragraph("upgrade"),
            paragraph("rollback"),
            paragraph("support"),
        );
        let main = main_content(&page, "https://example.com/notes").expect("extracted");
        let (markdown, how) = convert(page.as_bytes(), None, "https://example.com/notes");
        assert_eq!(how, How::MainContent);
        for kept in [
            "Release notes",
            "The install section",
            "The support section",
        ] {
            assert!(markdown.contains(kept), "missing {kept:?} in:\n{markdown}");
        }
        for dropped in ["MENU-LINK-7", "BOTTOM-CONTACT"] {
            assert!(!main.contains(dropped), "kept {dropped:?} in:\n{main}");
        }
    }

    #[test]
    fn too_little_found_converts_the_whole_page() {
        // The only scored block is a short paragraph; most of the text is in
        // a list Readability does not score, in a branch of its own.
        let items = (0..30)
            .map(|n| format!("<li>Changelog entry {n}: fixed a crash, a leak and a typo.</li>"))
            .collect::<Vec<_>>()
            .concat();
        let page = format!(
            "<html><head><title>Changes</title></head><body><div id=\"intro\"><div><div><p>\
             Recent changes, newest first, as listed below.</p></div></div></div><div \
             id=\"list\"><div><div><ul>{items}</ul></div></div></div></body></html>"
        );
        assert_eq!(main_content(&page, "https://example.com/"), None);
        let (markdown, how) = convert(page.as_bytes(), None, "https://example.com/");
        assert_eq!(how, How::WholePage);
        assert!(markdown.contains("Recent changes"), "{markdown}");
        assert!(markdown.contains("Changelog entry 29"), "{markdown}");
        // A short page is not scored at all.
        let short = "<html><body><main><p>Just a line.</p></main><footer>F</footer></body></html>";
        assert_eq!(main_content(short, "https://example.com/"), None);
    }

    #[test]
    fn oversized_documents_are_not_scored() {
        let wide = format!(
            "<html><body>{}{}</body></html>",
            paragraph("only").repeat(4),
            "<i>x</i>".repeat(super::MAX_ELEMENTS)
        );
        assert_eq!(main_content(&wide, "https://example.com/"), None);
        let deep = format!(
            "<html><body>{}{}</body></html>",
            "<div>".repeat(super::MAX_DEPTH + 10),
            paragraph("deep").repeat(4)
        );
        assert_eq!(main_content(&deep, "https://example.com/"), None);
    }
}
