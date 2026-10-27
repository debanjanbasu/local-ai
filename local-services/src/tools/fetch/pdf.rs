//! PDF text for the model: each page's text in the order the file draws
//! it, under a `## Page N` heading.
//!
//! `lopdf` (MIT) parses the file: it bounds what any one stream may
//! decompress to and how deeply objects nest, repairs damaged
//! cross-reference tables, and opens files encrypted with an empty user
//! password (most "protected" PDFs). It was chosen over `pdf-extract`
//! (better line layout, but on `lopdf` 0.42 with unbounded stream
//! decoding, an `unwrap` on page content, and twice the time) and
//! `pdf_oxide` (image, layout and logging crates as dependencies, and an
//! interleaved title block on the arXiv test paper). `lopdf`'s own
//! `extract_text` runs a page into one line, so the text is laid out here
//! from the content stream's text operators: a line break where the
//! baseline moves by more than half a line, a blank line where it moves by
//! more than one and a half, and a space at a `TJ` gap of a fifth of an em
//! or more.
//!
//! Not read: text drawn by form `XObject`s, and scanned pages (images; a
//! file with no text says so). A file that needs a password, or that
//! cannot be parsed, gives a one-line explanation instead of text, as does
//! a parser panic (caught per file and per page).

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::panic::AssertUnwindSafe;

use lopdf::content::Content;
use lopdf::{Document, Encoding, LoadOptions, Object, ObjectId};

/// Bytes any one stream may decompress to: object streams while loading,
/// a page's content, a font's character map.
const STREAM_LIMIT: usize = 8 << 20;
/// Pages read at most.
pub const MAX_PAGES: usize = 300;
/// A `TJ` adjustment (thousandths of an em; negative moves right) wide
/// enough to read as a space.
const SPACE_ADJUSTMENT: f32 = -200.0;

/// `pdf`'s text (`cut` if its read stopped at a limit), at most
/// `max_bytes` of it, with markers for what was cut, and whether anything
/// was. Reading stops at the next page once `stopped` turns true (the
/// caller gave up).
#[must_use]
pub fn to_text(
    pdf: &[u8],
    cut: bool,
    max_bytes: usize,
    stopped: &dyn Fn() -> bool,
) -> (String, bool) {
    let options = LoadOptions {
        max_decompressed_size: Some(STREAM_LIMIT),
        ..LoadOptions::default()
    };
    let loaded = std::panic::catch_unwind(|| Document::load_mem_with_options(pdf, options));
    let document = match loaded {
        Ok(Ok(document)) => document,
        Ok(Err(error)) if cut => {
            return (
                format!(
                    "[the PDF is larger than the {} bytes read and cannot be read without the \
                     rest: {error}]",
                    pdf.len()
                ),
                true,
            );
        }
        Ok(Err(error)) => return (format!("[the PDF cannot be read: {error}]"), false),
        Err(_) => {
            return (
                "[the PDF cannot be read: its structure broke the reader]".into(),
                cut,
            );
        }
    };
    if document.is_encrypted() && document.encryption_state.is_none() {
        return (
            "[the PDF is encrypted with a password; its text cannot be read]".into(),
            false,
        );
    }
    let pages = document.get_pages();
    let mut text = format!("[converted from PDF to text: {} pages]\n", pages.len());
    let mut found = false;
    let mut read = 0;
    for (number, page) in pages.iter().take(MAX_PAGES) {
        if stopped() || text.len() >= max_bytes {
            break;
        }
        read += 1;
        let _ = write!(text, "\n## Page {number}\n\n");
        match std::panic::catch_unwind(AssertUnwindSafe(|| page_text(&document, *page))) {
            Ok(Ok(page)) => {
                found |= !page.trim().is_empty();
                clean(&page, &mut text);
            }
            Ok(Err(error)) => {
                let _ = writeln!(text, "[this page cannot be read: {error}]");
            }
            Err(_) => text.push_str("[this page cannot be read: its content broke the reader]\n"),
        }
    }
    if !found && read == pages.len() {
        return (
            format!(
                "[the PDF's {} pages hold no text: it may be scanned images, which are not read]",
                pages.len()
            ),
            cut,
        );
    }
    let mut truncated = false;
    if text.len() > max_bytes {
        let mut end = max_bytes;
        while !text.is_char_boundary(end) {
            end -= 1;
        }
        text.truncate(end);
        let _ = write!(text, "\n[truncated: the text was cut at {max_bytes} bytes]");
        truncated = true;
    } else if read < pages.len() {
        let _ = write!(
            text,
            "\n[truncated: only the first {read} of {} pages were read]",
            pages.len()
        );
        truncated = true;
    }
    if cut {
        let _ = write!(
            text,
            "\n[truncated: the PDF was cut at {} bytes; pages past the cut are missing]",
            pdf.len()
        );
    }
    (text.trim_end().to_owned(), truncated || cut)
}

/// `page`'s text, laid out from its content stream's text operators.
fn page_text(document: &Document, page: ObjectId) -> lopdf::Result<String> {
    let encodings: BTreeMap<Vec<u8>, Encoding> = document
        .get_page_fonts(page)?
        .into_iter()
        .filter_map(|(name, font)| {
            Some((
                name,
                font.get_font_encoding_with_limit(document, STREAM_LIMIT)
                    .ok()?,
            ))
        })
        .collect();
    let content = Content::decode(&document.get_page_content_with_limit(page, STREAM_LIMIT)?)?;
    let mut layout = Layout::default();
    let mut encoding = None;
    for operation in &content.operations {
        let operands = &operation.operands;
        let number = |index: usize| {
            operands
                .get(index)
                .and_then(|operand| operand.as_float().ok())
                .unwrap_or(0.0)
        };
        let matrix = || {
            [
                number(0),
                number(1),
                number(2),
                number(3),
                number(4),
                number(5),
            ]
        };
        match operation.operator.as_str() {
            "q" => layout.saved.push(layout.ctm),
            "Q" => layout.ctm = layout.saved.pop().unwrap_or(IDENTITY),
            "cm" => layout.ctm = multiply(matrix(), layout.ctm),
            "BT" => layout.line = IDENTITY,
            "Tf" => {
                encoding = operands
                    .first()
                    .and_then(|name| name.as_name().ok())
                    .and_then(|name| encodings.get(name));
                layout.size = number(1);
            }
            "TL" => layout.leading = number(0),
            "Td" => layout.move_line(number(0), number(1)),
            "TD" => {
                layout.leading = -number(1);
                layout.move_line(number(0), number(1));
            }
            "Tm" => layout.line = matrix(),
            "T*" => layout.move_line(0.0, -layout.leading),
            "Tj" => layout.show(encoding, operands.first()),
            "'" => {
                layout.move_line(0.0, -layout.leading);
                layout.show(encoding, operands.first());
            }
            "\"" => {
                layout.move_line(0.0, -layout.leading);
                layout.show(encoding, operands.get(2));
            }
            "TJ" => {
                let items = operands.first().and_then(|array| array.as_array().ok());
                for item in items.into_iter().flatten() {
                    if matches!(item, Object::String(..)) {
                        layout.show(encoding, Some(item));
                    } else if item.as_float().is_ok_and(|gap| gap <= SPACE_ADJUSTMENT) {
                        layout.space();
                    }
                }
            }
            _ => {}
        }
    }
    Ok(layout.text)
}

type Matrix = [f32; 6];

const IDENTITY: Matrix = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0];

/// `first` then `second`, as PDF composes transformations.
fn multiply(first: Matrix, second: Matrix) -> Matrix {
    let [m0, m1, m2, m3, m4, m5] = first;
    let [n0, n1, n2, n3, n4, n5] = second;
    [
        m0.mul_add(n0, m1 * n2),
        m0.mul_add(n1, m1 * n3),
        m2.mul_add(n0, m3 * n2),
        m2.mul_add(n1, m3 * n3),
        m4.mul_add(n0, m5.mul_add(n2, n4)),
        m4.mul_add(n1, m5.mul_add(n3, n5)),
    ]
}

/// Where text is drawn, enough to tell lines and paragraphs apart.
struct Layout {
    text: String,
    /// The current transformation matrix, and those saved by `q`.
    ctm: Matrix,
    saved: Vec<Matrix>,
    /// The text line matrix, the font size and the leading.
    line: Matrix,
    size: f32,
    leading: f32,
    /// The baseline and line height, in device space, of the last text.
    last: Option<(f32, f32)>,
}

impl Default for Layout {
    fn default() -> Self {
        Self {
            text: String::new(),
            ctm: IDENTITY,
            saved: Vec::new(),
            line: IDENTITY,
            size: 0.0,
            leading: 0.0,
            last: None,
        }
    }
}

impl Layout {
    /// `Td`: the next line, offset from the start of this one.
    fn move_line(&mut self, x: f32, y: f32) {
        self.line = multiply([1.0, 0.0, 0.0, 1.0, x, y], self.line);
    }

    /// Show `string`, after a line break or blank line if the baseline
    /// moved.
    fn show(&mut self, encoding: Option<&Encoding>, string: Option<&Object>) {
        let (Some(encoding), Some(Object::String(bytes, _))) = (encoding, string) else {
            return;
        };
        let [x, y] = [self.line[4], self.line[5]];
        let baseline = self.ctm[1].mul_add(x, self.ctm[3].mul_add(y, self.ctm[5]));
        let height = (self.size * self.line[3].abs() * self.ctm[3].abs()).max(f32::EPSILON);
        if let Some((last, last_height)) = self.last {
            let moved = (baseline - last).abs();
            let line = height.max(last_height);
            if moved > 1.5 * line {
                self.end_line();
                if !self.text.is_empty() && !self.text.ends_with("\n\n") {
                    self.text.push('\n');
                }
            } else if moved > 0.5 * line {
                self.end_line();
            }
        }
        self.last = Some((baseline, height));
        let _ = encoding.write_to_string(bytes, &mut self.text);
    }

    fn end_line(&mut self) {
        if !self.text.is_empty() && !self.text.ends_with('\n') {
            self.text.push('\n');
        }
    }

    fn space(&mut self) {
        if !self.text.is_empty() && !self.text.ends_with(char::is_whitespace) {
            self.text.push(' ');
        }
    }
}

/// Append `page` to `text`: ligatures spelled out, control characters
/// dropped, lines without trailing blanks, and a final line break.
fn clean(page: &str, text: &mut String) {
    for line in page.trim().lines() {
        for character in line.trim_end().chars() {
            match character {
                '\u{fb00}' => text.push_str("ff"),
                '\u{fb01}' => text.push_str("fi"),
                '\u{fb02}' => text.push_str("fl"),
                '\u{fb03}' => text.push_str("ffi"),
                '\u{fb04}' => text.push_str("ffl"),
                '\u{fb05}' | '\u{fb06}' => text.push_str("st"),
                '\t' => text.push(' '),
                character if character.is_control() => {}
                character => text.push(character),
            }
        }
        text.push('\n');
    }
}

#[cfg(test)]
pub mod tests {
    use lopdf::content::{Content, Operation};
    use lopdf::{
        Document, EncryptionState, EncryptionVersion, Object, Permissions, Stream, dictionary,
    };

    use super::to_text;

    /// A PDF of one page per entry of `pages`, each drawing its operations
    /// with Helvetica as `F1`; encrypted with `(owner, user)` passwords if
    /// given.
    pub fn pdf_of(pages: Vec<Vec<Operation>>, passwords: Option<(&str, &str)>) -> Vec<u8> {
        let mut document = Document::with_version("1.5");
        let pages_id = document.new_object_id();
        let font = document.add_object(dictionary! {
            "Type" => "Font", "Subtype" => "Type1", "BaseFont" => "Helvetica",
        });
        let resources = document.add_object(dictionary! { "Font" => dictionary! { "F1" => font } });
        let kids: Vec<Object> = pages
            .into_iter()
            .map(|operations| {
                let content = Content { operations }.encode().expect("content");
                let content = document.add_object(Stream::new(dictionary! {}, content));
                document
                    .add_object(dictionary! {
                        "Type" => "Page", "Parent" => pages_id, "Contents" => content,
                    })
                    .into()
            })
            .collect();
        let count = i64::try_from(kids.len()).expect("page count");
        document.objects.insert(
            pages_id,
            Object::Dictionary(dictionary! {
                "Type" => "Pages", "Kids" => kids, "Count" => count, "Resources" => resources,
                "MediaBox" => vec![0.into(), 0.into(), 612.into(), 792.into()],
            }),
        );
        let catalog = document.add_object(dictionary! { "Type" => "Catalog", "Pages" => pages_id });
        document.trailer.set("Root", catalog);
        document.trailer.set(
            "ID",
            Object::Array(vec![
                Object::string_literal(b"local-ai".to_vec()),
                Object::string_literal(b"test".to_vec()),
            ]),
        );
        if let Some((owner_password, user_password)) = passwords {
            let state = EncryptionState::try_from(EncryptionVersion::V2 {
                document: &document,
                owner_password,
                user_password,
                key_length: 128,
                permissions: Permissions::default(),
            })
            .expect("encryption");
            document.encrypt(&state).expect("encrypt");
        }
        let mut bytes = Vec::new();
        document.save_to(&mut bytes).expect("save");
        bytes
    }

    /// Operations drawing `lines` 14 points apart from the top left, with
    /// a paragraph gap before the last.
    pub fn lines(lines: &[&str]) -> Vec<Operation> {
        let mut operations = vec![
            Operation::new("BT", vec![]),
            Operation::new("Tf", vec!["F1".into(), 12.into()]),
            Operation::new("TL", vec![14.into()]),
            Operation::new("Td", vec![72.into(), 720.into()]),
        ];
        for (index, line) in lines.iter().enumerate() {
            if index + 1 == lines.len() && index > 0 {
                operations.push(Operation::new("Td", vec![0.into(), (-40).into()]));
            } else if index > 0 {
                operations.push(Operation::new("T*", vec![]));
            }
            operations.push(Operation::new("Tj", vec![Object::string_literal(*line)]));
        }
        operations.push(Operation::new("ET", vec![]));
        operations
    }

    fn read(pdf: &[u8], max_bytes: usize) -> (String, bool) {
        to_text(pdf, false, max_bytes, &|| false)
    }

    #[test]
    fn a_pdf_reads_as_pages_of_lines() {
        let mut first = lines(&["Attention Is All You Need", "Abstract", "Last paragraph."]);
        // A word gap as a TJ adjustment, kerning that is not one, and a
        // superscript a few points up that stays on its line.
        first.splice(
            first.len() - 1..first.len() - 1,
            [
                Operation::new("Td", vec![0.into(), (-28).into()]),
                Operation::new(
                    "TJ",
                    vec![Object::Array(vec![
                        Object::string_literal("Hel"),
                        (-30).into(),
                        Object::string_literal("lo"),
                        (-250).into(),
                        Object::string_literal("World"),
                    ])],
                ),
                Operation::new("Ts", vec![4.into()]),
                Operation::new("Td", vec![60.into(), 4.into()]),
                Operation::new("Tj", vec![Object::string_literal("1")]),
            ],
        );
        let pdf = pdf_of(vec![first, lines(&["Second page text"])], None);
        assert_eq!(
            read(&pdf, 4096),
            (
                "[converted from PDF to text: 2 pages]\n\n## Page 1\n\nAttention Is All You Need\n\
                 Abstract\n\nLast paragraph.\n\nHello World1\n\n## Page 2\n\nSecond page text"
                    .to_owned(),
                false
            )
        );
        let mut cleaned = String::new();
        super::clean("e\u{fb03}cient\u{0}  \n\u{fb01}ne\t.", &mut cleaned);
        assert_eq!(cleaned, "efficient\nfine .\n");
    }

    #[test]
    fn locked_scanned_and_broken_pdfs_explain_themselves() {
        let text = || vec![lines(&["Private text"])];
        let locked = pdf_of(text(), Some(("owner", "secret")));
        assert_eq!(
            read(&locked, 4096),
            (
                "[the PDF is encrypted with a password; its text cannot be read]".to_owned(),
                false
            )
        );
        // An empty user password (permissions only) opens without asking.
        let open = pdf_of(text(), Some(("owner", "")));
        assert!(
            read(&open, 4096).0.ends_with("\n\nPrivate text"),
            "{:?}",
            read(&open, 4096)
        );
        let drawing = vec![vec![
            Operation::new("m", vec![0.into(), 0.into()]),
            Operation::new("l", vec![100.into(), 100.into()]),
            Operation::new("S", vec![]),
        ]];
        assert_eq!(
            read(&pdf_of(drawing, None), 4096).0,
            "[the PDF's 1 pages hold no text: it may be scanned images, which are not read]"
        );
        for broken in [
            &b"%PDF-1.7\nnot really a PDF"[..],
            b"<html>a page</html>",
            b"",
        ] {
            let (message, truncated) = read(broken, 4096);
            assert!(
                message.starts_with("[the PDF cannot be read: "),
                "{message}"
            );
            assert!(!truncated);
        }
        // A file cut at the read limit: repaired, or explained.
        let whole = pdf_of(vec![lines(&["Early page"]), lines(&["Late page"])], None);
        let half = whole.get(..whole.len() / 2).expect("half");
        let (message, truncated) = to_text(half, true, 4096, &|| false);
        assert!(truncated);
        assert!(
            message.starts_with("[the PDF is larger than")
                || message.ends_with("; pages past the cut are missing]"),
            "{message}"
        );
    }

    #[test]
    fn text_pages_and_cancellation_are_bounded() {
        let pages = (0..super::MAX_PAGES + 5)
            .map(|_| lines(&["A page of text"]))
            .collect();
        let pdf = pdf_of(pages, None);
        let (text, truncated) = read(&pdf, 1 << 20);
        assert!(truncated);
        assert!(
            text.ends_with(&format!(
                "\n[truncated: only the first {} of {} pages were read]",
                super::MAX_PAGES,
                super::MAX_PAGES + 5
            )),
            "{text}"
        );
        let (text, truncated) = read(&pdf, 200);
        assert!(truncated);
        assert!(
            text.ends_with("\n[truncated: the text was cut at 200 bytes]"),
            "{text}"
        );
        // A caller that stopped waiting: no page is read.
        let (text, _) = to_text(&pdf, false, 1 << 20, &|| true);
        assert!(!text.contains("## Page"), "{text}");
    }
}
