//! Lark grammars (the syntax of `OpenAI` custom tool `format`s) translated
//! into one regular expression for a raw string value.
//!
//! A raw value is constrained by regular expressions intersected with the
//! delimiters it may never contain (see [`super::RAW_BASE`]), so a grammar is
//! supported exactly when its language is regular: every rule and terminal is
//! inlined, and a rule that reaches itself is refused. Codex's `apply_patch`
//! grammar is regular, as are most line-oriented formats.
//!
//! The language is the one the grammar derives when each terminal matches
//! any string of its own language: there is no separate lexing pass, so
//! terminal priorities and longest-match lexing do not narrow it. Rule
//! aliases (`-> name`), `?`/`!` rule prefixes and priorities only shape the
//! parse tree and are accepted and ignored.
//!
//! Supported: rules and terminals; `|` alternatives (also on continuation
//! lines); grouping `( )`; `[ ]` and `?` optional; `*`, `+`, `~ n` and
//! `~ n..m` repetition; string literals (with the `i` flag) and `"a".."z"`
//! ranges; `/regex/` terminals with the `i` and `s` flags; `//` comments;
//! and `%import common.NAME` (or `%import common (A, B)`) for the common
//! terminals in [`COMMON`]. Everything else (`%ignore`, `%declare`,
//! `%override`, `%extend`, templates, other imports, recursion, regex
//! anchors, lookarounds and backreferences) is refused with the reason, never
//! enforced partially.

use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;

use llguidance::derivre::RegexBuilder;

use super::regex_literal;

/// Bounds keeping translation small: the grammar text, the inlined regular
/// expression, and repetition counts.
const MAX_SOURCE: usize = 65_536;
const MAX_REGEX: usize = 262_144;
const MAX_REPEAT: u32 = 1_024;

/// The terminals of Lark's `common` module this translation provides, as
/// regular expressions (Lark's own definitions).
const COMMON: &[(&str, &str)] = &[
    ("LF", r"\n"),
    ("CR", r"\r"),
    ("NEWLINE", r"(?:\r?\n)+"),
    ("WS", r"[ \t\x{0C}\r\n]+"),
    ("WS_INLINE", r"[ \t]+"),
    ("DIGIT", r"[0-9]"),
    ("HEXDIGIT", r"[a-fA-F0-9]"),
    ("INT", r"[0-9]+"),
    ("SIGNED_INT", r"[+\-]?[0-9]+"),
    ("LCASE_LETTER", r"[a-z]"),
    ("UCASE_LETTER", r"[A-Z]"),
    ("LETTER", r"[a-zA-Z]"),
    ("WORD", r"[a-zA-Z]+"),
    ("CNAME", r"[_a-zA-Z][_a-zA-Z0-9]*"),
];

/// Translate a Lark grammar into an anchored regular expression for the
/// whole value, or say exactly why it cannot be enforced.
pub(in crate::tools) fn to_regex(source: &str) -> Result<String, String> {
    if source.len() > MAX_SOURCE {
        return Err(format!("grammar longer than {MAX_SOURCE} bytes"));
    }
    let tokens = statements(tokenize(source)?);
    let mut definitions = HashMap::new();
    for statement in tokens {
        parse_statement(&statement, &mut definitions)?;
    }
    if !definitions.contains_key("start") {
        return Err("the grammar has no start rule".into());
    }
    let mut translator = Translator {
        definitions: &definitions,
        done: HashMap::new(),
        active: HashSet::new(),
    };
    let regex = translator.name("start")?;
    RegexBuilder::new()
        .mk_regex(&regex)
        .map_err(|error| format!("unsupported regular expression: {error}"))?;
    Ok(regex)
}

#[derive(Clone, Debug, PartialEq)]
enum Token {
    Name(String),
    Str(String, bool),
    Regex(String, String),
    Number(u32),
    Directive(String),
    Colon,
    Pipe,
    Open(char),
    Close(char),
    Op(char),
    Tilde,
    Dot,
    DotDot,
    Arrow,
    Comma,
    Newline,
}

#[allow(clippy::too_many_lines)] // One scanner over the Lark token syntax.
fn tokenize(source: &str) -> Result<Vec<Token>, String> {
    let mut tokens = Vec::new();
    let mut characters = source.chars().peekable();
    while let Some(character) = characters.next() {
        match character {
            '\n' => tokens.push(Token::Newline),
            ' ' | '\t' | '\r' => {}
            '/' if characters.peek() == Some(&'/') => {
                while characters.peek().is_some_and(|next| *next != '\n') {
                    characters.next();
                }
            }
            '#' => return Err("'#' comments are not Lark syntax; use //".into()),
            '"' => {
                let mut text = String::new();
                loop {
                    match characters.next() {
                        None | Some('\n') => return Err("unterminated string literal".into()),
                        Some('"') => break,
                        Some('\\') => text.push(string_escape(&mut characters)?),
                        Some(other) => text.push(other),
                    }
                }
                let insensitive = characters.peek() == Some(&'i');
                if insensitive {
                    characters.next();
                }
                tokens.push(Token::Str(text, insensitive));
            }
            '/' => {
                let mut body = String::new();
                loop {
                    match characters.next() {
                        None | Some('\n') => return Err("unterminated regular expression".into()),
                        Some('/') => break,
                        Some('\\') => {
                            let next =
                                characters.next().ok_or("unterminated regular expression")?;
                            body.push('\\');
                            body.push(next);
                        }
                        Some(other) => body.push(other),
                    }
                }
                let mut flags = String::new();
                while characters.peek().is_some_and(char::is_ascii_lowercase) {
                    flags.extend(characters.next());
                }
                tokens.push(Token::Regex(body, flags));
            }
            '%' => {
                let mut name = String::new();
                while characters.peek().is_some_and(char::is_ascii_alphabetic) {
                    name.extend(characters.next());
                }
                tokens.push(Token::Directive(name));
            }
            ':' => tokens.push(Token::Colon),
            '|' => tokens.push(Token::Pipe),
            '(' | '[' => tokens.push(Token::Open(character)),
            ')' | ']' => tokens.push(Token::Close(character)),
            '?' | '*' | '+' | '!' => tokens.push(Token::Op(character)),
            '~' => tokens.push(Token::Tilde),
            ',' => tokens.push(Token::Comma),
            '.' => {
                if characters.peek() == Some(&'.') {
                    characters.next();
                    tokens.push(Token::DotDot);
                } else {
                    tokens.push(Token::Dot);
                }
            }
            '-' if characters.peek() == Some(&'>') => {
                characters.next();
                tokens.push(Token::Arrow);
            }
            '{' | '}' => return Err("templates are not supported".into()),
            character if character.is_ascii_digit() => {
                let mut digits = String::from(character);
                while characters.peek().is_some_and(char::is_ascii_digit) {
                    digits.extend(characters.next());
                }
                let number = digits
                    .parse()
                    .map_err(|_| format!("number {digits} is too large"))?;
                tokens.push(Token::Number(number));
            }
            character if character == '_' || character.is_ascii_alphabetic() => {
                let mut name = String::from(character);
                while characters
                    .peek()
                    .is_some_and(|next| *next == '_' || next.is_ascii_alphanumeric())
                {
                    name.extend(characters.next());
                }
                tokens.push(Token::Name(name));
            }
            other => return Err(format!("unexpected character {other:?}")),
        }
    }
    Ok(tokens)
}

/// A string literal escape, as Python reads it.
fn string_escape(
    characters: &mut std::iter::Peekable<std::str::Chars<'_>>,
) -> Result<char, String> {
    let escape = characters.next().ok_or("unterminated string literal")?;
    let hex = |characters: &mut std::iter::Peekable<std::str::Chars<'_>>, digits: usize| {
        let text: String = characters.by_ref().take(digits).collect();
        u32::from_str_radix(&text, 16)
            .ok()
            .filter(|_| text.len() == digits)
            .and_then(char::from_u32)
            .ok_or_else(|| format!("invalid \\{escape} escape in a string literal"))
    };
    Ok(match escape {
        'n' => '\n',
        't' => '\t',
        'r' => '\r',
        '0' => '\0',
        'f' => '\u{0C}',
        'v' => '\u{0B}',
        '\\' | '"' | '\'' => escape,
        'x' => hex(characters, 2)?,
        'u' => hex(characters, 4)?,
        'U' => hex(characters, 8)?,
        other => return Err(format!("unsupported escape \\{other} in a string literal")),
    })
}

/// Split tokens into statements: a newline ends one unless it is inside
/// brackets or the next line continues it with `|`.
fn statements(tokens: Vec<Token>) -> Vec<Vec<Token>> {
    let mut statements = Vec::new();
    let mut current = Vec::new();
    let mut depth = 0usize;
    let mut tokens = tokens.into_iter().peekable();
    while let Some(token) = tokens.next() {
        match token {
            Token::Newline => {
                while tokens.peek() == Some(&Token::Newline) {
                    tokens.next();
                }
                if depth > 0 || tokens.peek() == Some(&Token::Pipe) {
                    continue;
                }
                if !current.is_empty() {
                    statements.push(std::mem::take(&mut current));
                }
            }
            token => {
                match token {
                    Token::Open(_) => depth += 1,
                    Token::Close(_) => depth = depth.saturating_sub(1),
                    _ => {}
                }
                current.push(token);
            }
        }
    }
    if !current.is_empty() {
        statements.push(current);
    }
    statements
}

#[derive(Debug)]
enum Expr {
    Alt(Vec<Self>),
    Seq(Vec<Self>),
    Repeat(Box<Self>, u32, Option<u32>),
    Literal(String, bool),
    Regex(String),
    Name(String),
}

/// A defined name: whether it is a terminal, and its expansion.
struct Definition {
    terminal: bool,
    expr: Expr,
}

fn is_terminal(name: &str) -> bool {
    name.trim_start_matches('_')
        .starts_with(|character: char| character.is_ascii_uppercase())
}

fn parse_statement(
    statement: &[Token],
    definitions: &mut HashMap<String, Definition>,
) -> Result<(), String> {
    if let Some(Token::Directive(directive)) = statement.first() {
        return match directive.as_str() {
            "import" => import(&statement[1..], definitions),
            other => Err(format!("%{other} is not supported")),
        };
    }
    let mut position = 0;
    if matches!(statement.first(), Some(Token::Op('?' | '!'))) {
        position += 1;
    }
    let Some(Token::Name(name)) = statement.get(position) else {
        return Err("expected a rule or terminal definition".into());
    };
    position += 1;
    if statement.get(position) == Some(&Token::Dot) {
        if !matches!(statement.get(position + 1), Some(Token::Number(_))) {
            return Err(format!("{name}: expected a priority number after '.'"));
        }
        position += 2;
    }
    if statement.get(position) != Some(&Token::Colon) {
        return Err(format!("{name}: expected ':'"));
    }
    let mut parser = Parser {
        tokens: &statement[position + 1..],
        position: 0,
    };
    let expr = parser
        .alternatives()
        .map_err(|error| format!("{name}: {error}"))?;
    if let Some(token) = parser.tokens.get(parser.position) {
        return Err(format!("{name}: unexpected {token:?}"));
    }
    insert(definitions, name, expr)
}

fn insert(
    definitions: &mut HashMap<String, Definition>,
    name: &str,
    expr: Expr,
) -> Result<(), String> {
    let definition = Definition {
        terminal: is_terminal(name),
        expr,
    };
    if definitions.insert(name.to_owned(), definition).is_some() {
        return Err(format!("{name} is defined twice"));
    }
    Ok(())
}

/// `%import common.NAME [-> ALIAS]` or `%import common (A, B, ...)`.
fn import(tokens: &[Token], definitions: &mut HashMap<String, Definition>) -> Result<(), String> {
    let common = |name: &str| {
        COMMON
            .iter()
            .find(|(candidate, _)| *candidate == name)
            .map(|(_, regex)| Expr::Regex((*regex).to_owned()))
            .ok_or_else(|| format!("%import of common.{name} is not supported"))
    };
    match tokens {
        [Token::Name(module), Token::Dot, Token::Name(name)] if module == "common" => {
            insert(definitions, name, common(name)?)
        }
        [
            Token::Name(module),
            Token::Dot,
            Token::Name(name),
            Token::Arrow,
            Token::Name(alias),
        ] if module == "common" => insert(definitions, alias, common(name)?),
        [
            Token::Name(module),
            Token::Open('('),
            names @ ..,
            Token::Close(')'),
        ] if module == "common" => {
            for (index, token) in names.iter().enumerate() {
                match (index % 2, token) {
                    (0, Token::Name(name)) => insert(definitions, name, common(name)?)?,
                    (1, Token::Comma) => {}
                    _ => return Err("malformed %import list".into()),
                }
            }
            Ok(())
        }
        [Token::Name(module), ..] if module != "common" => Err(format!(
            "%import from {module:?} is not supported: only the common module"
        )),
        _ => Err("malformed %import".into()),
    }
}

struct Parser<'t> {
    tokens: &'t [Token],
    position: usize,
}

impl Parser<'_> {
    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.position)
    }

    fn next(&mut self) -> Option<&Token> {
        let token = self.tokens.get(self.position);
        self.position += 1;
        token
    }

    fn alternatives(&mut self) -> Result<Expr, String> {
        let mut alternatives = vec![self.sequence()?];
        while self.peek() == Some(&Token::Pipe) {
            self.position += 1;
            alternatives.push(self.sequence()?);
        }
        Ok(if alternatives.len() == 1 {
            alternatives.remove(0)
        } else {
            Expr::Alt(alternatives)
        })
    }

    fn sequence(&mut self) -> Result<Expr, String> {
        let mut items = Vec::new();
        loop {
            match self.peek() {
                None | Some(Token::Pipe | Token::Close(_)) => break,
                Some(Token::Arrow) => {
                    // An alias names the tree node; the language is the same.
                    self.position += 1;
                    match self.next() {
                        Some(Token::Name(_)) => break,
                        _ => return Err("expected a name after '->'".into()),
                    }
                }
                _ => items.push(self.item()?),
            }
        }
        Ok(Expr::Seq(items))
    }

    fn item(&mut self) -> Result<Expr, String> {
        let atom = self.atom()?;
        Ok(match self.peek() {
            Some(Token::Op('?')) => {
                self.position += 1;
                Expr::Repeat(Box::new(atom), 0, Some(1))
            }
            Some(Token::Op('*')) => {
                self.position += 1;
                Expr::Repeat(Box::new(atom), 0, None)
            }
            Some(Token::Op('+')) => {
                self.position += 1;
                Expr::Repeat(Box::new(atom), 1, None)
            }
            Some(Token::Tilde) => {
                self.position += 1;
                let minimum = self.count()?;
                let maximum = if self.peek() == Some(&Token::DotDot) {
                    self.position += 1;
                    self.count()?
                } else {
                    minimum
                };
                if minimum > maximum {
                    return Err(format!("repetition ~ {minimum}..{maximum} is empty"));
                }
                Expr::Repeat(Box::new(atom), minimum, Some(maximum))
            }
            _ => atom,
        })
    }

    fn count(&mut self) -> Result<u32, String> {
        match self.next() {
            Some(Token::Number(count)) if *count <= MAX_REPEAT => Ok(*count),
            Some(Token::Number(_)) => Err(format!("repetition above {MAX_REPEAT}")),
            _ => Err("expected a repetition count after '~'".into()),
        }
    }

    fn atom(&mut self) -> Result<Expr, String> {
        match self.next().cloned() {
            Some(Token::Open(open)) => {
                let inner = self.alternatives()?;
                let close = if open == '(' { ')' } else { ']' };
                if self.next() != Some(&Token::Close(close)) {
                    return Err(format!("expected '{close}'"));
                }
                Ok(if open == '[' {
                    Expr::Repeat(Box::new(inner), 0, Some(1))
                } else {
                    inner
                })
            }
            Some(Token::Str(text, insensitive)) => {
                if self.peek() != Some(&Token::DotDot) {
                    return Ok(Expr::Literal(text, insensitive));
                }
                self.position += 1;
                let Some(Token::Str(end, _)) = self.next().cloned() else {
                    return Err("expected a string after '..'".into());
                };
                let (Some(low), Some(high)) = (single(&text), single(&end)) else {
                    return Err("a \"..\" range needs single-character strings".into());
                };
                if low > high {
                    return Err("a \"..\" range must not be reversed".into());
                }
                Ok(Expr::Regex(format!(
                    "[\\x{{{:x}}}-\\x{{{:x}}}]",
                    u32::from(low),
                    u32::from(high)
                )))
            }
            Some(Token::Regex(body, flags)) => python_regex(&body, &flags).map(Expr::Regex),
            Some(Token::Name(name)) => Ok(Expr::Name(name)),
            Some(token) => Err(format!("unexpected {token:?}")),
            None => Err("unexpected end of definition".into()),
        }
    }
}

fn single(text: &str) -> Option<char> {
    let mut characters = text.chars();
    let character = characters.next()?;
    characters.next().is_none().then_some(character)
}

/// A Lark `/regex/flags` terminal (Python `re` syntax) as an equivalent
/// regular expression for the engine, or why it has no exact translation.
fn python_regex(body: &str, flags: &str) -> Result<String, String> {
    let mut out = String::with_capacity(body.len() + 8);
    let mut characters = body.chars().peekable();
    let mut class = false;
    while let Some(character) = characters.next() {
        match character {
            '\\' => {
                let next = characters
                    .next()
                    .ok_or("regex ends with a lone backslash")?;
                if next.is_ascii_digit() && next != '0' {
                    return Err("backreferences are not supported in regex terminals".into());
                }
                match (next, class) {
                    ('b', true) => out.push_str(r"\x{8}"),
                    ('b' | 'B' | 'A' | 'Z' | 'z' | 'G', false) => {
                        return Err(format!(
                            "\\{next} anchors are not supported in regex terminals"
                        ));
                    }
                    ('/', _) => out.push_str(r"\/"),
                    _ => {
                        out.push('\\');
                        out.push(next);
                    }
                }
            }
            '[' if !class => {
                class = true;
                out.push('[');
                if characters.peek() == Some(&'^') {
                    out.push('^');
                    characters.next();
                }
                if characters.peek() == Some(&']') {
                    return Err("a ']'-leading character class is not supported".into());
                }
            }
            ']' if class => {
                class = false;
                out.push(']');
            }
            _ if class => {
                // The engine's classes nest and have set operators; Python's
                // do not.
                if character == '-' && characters.peek() == Some(&'-') {
                    return Err("'--' inside a character class is not supported".into());
                }
                if matches!(character, '[' | '&' | '~') {
                    out.push('\\');
                }
                push_plain(&mut out, character);
            }
            '(' if characters.peek() == Some(&'?') => {
                characters.next();
                if characters.next() != Some(':') {
                    return Err(
                        "only (?:...) groups are supported in regex terminals (no lookarounds, \
                         named groups or inline flags)"
                            .into(),
                    );
                }
                out.push_str("(?:");
            }
            '^' | '$' => {
                return Err("anchors are not supported in regex terminals".into());
            }
            character => push_plain(&mut out, character),
        }
    }
    if class {
        return Err("unterminated character class".into());
    }
    let mut regex = out;
    for flag in flags.chars() {
        regex = match flag {
            'i' => format!("(?i:{regex})"),
            's' => format!("(?s:{regex})"),
            other => return Err(format!("regex flag {other:?} is not supported")),
        };
    }
    RegexBuilder::new()
        .mk_regex(&regex)
        .map_err(|error| format!("unsupported regex /{body}/: {error}"))?;
    Ok(format!("(?:{regex})"))
}

/// A character outside an escape, kept readable inside a Lark `/.../`.
fn push_plain(out: &mut String, character: char) {
    match character {
        '/' => out.push_str(r"\/"),
        character if character.is_control() => {
            let _ = write!(out, "\\x{{{:x}}}", u32::from(character));
        }
        character => out.push(character),
    }
}

struct Translator<'d> {
    definitions: &'d HashMap<String, Definition>,
    done: HashMap<String, String>,
    active: HashSet<String>,
}

impl Translator<'_> {
    fn name(&mut self, name: &str) -> Result<String, String> {
        if let Some(regex) = self.done.get(name) {
            return Ok(regex.clone());
        }
        let definition = self
            .definitions
            .get(name)
            .ok_or_else(|| format!("{name} is used but not defined"))?;
        if !self.active.insert(name.to_owned()) {
            return Err(format!(
                "{name} is recursive: only grammars with a regular language (no recursive rules) \
                 can be enforced"
            ));
        }
        let regex = self.expr(&definition.expr, definition.terminal, name)?;
        self.active.remove(name);
        self.done.insert(name.to_owned(), regex.clone());
        Ok(regex)
    }

    fn expr(&mut self, expr: &Expr, terminal: bool, owner: &str) -> Result<String, String> {
        let regex = match expr {
            Expr::Alt(alternatives) => {
                let parts = alternatives
                    .iter()
                    .map(|alternative| self.expr(alternative, terminal, owner))
                    .collect::<Result<Vec<_>, _>>()?;
                format!("(?:{})", parts.join("|"))
            }
            Expr::Seq(items) => {
                let mut regex = String::from("(?:");
                for item in items {
                    regex.push_str(&self.expr(item, terminal, owner)?);
                    Self::check(regex.len())?;
                }
                regex.push(')');
                regex
            }
            Expr::Repeat(inner, minimum, maximum) => {
                let inner = self.expr(inner, terminal, owner)?;
                match (minimum, maximum) {
                    (0, Some(1)) => format!("{inner}?"),
                    (0, None) => format!("{inner}*"),
                    (1, None) => format!("{inner}+"),
                    (minimum, Some(maximum)) if minimum == maximum => {
                        format!("{inner}{{{minimum}}}")
                    }
                    (minimum, Some(maximum)) => format!("{inner}{{{minimum},{maximum}}}"),
                    (minimum, None) => format!("{inner}{{{minimum},}}"),
                }
            }
            Expr::Literal(text, false) => format!("(?:{})", regex_literal(text)),
            Expr::Literal(text, true) => format!("(?i:{})", regex_literal(text)),
            Expr::Regex(regex) => regex.clone(),
            Expr::Name(name) => {
                if terminal && !is_terminal(name) {
                    return Err(format!("terminal {owner} refers to rule {name}"));
                }
                self.name(name)?
            }
        };
        Self::check(regex.len())?;
        Ok(regex)
    }

    fn check(length: usize) -> Result<(), String> {
        if length > MAX_REGEX {
            return Err(format!(
                "the grammar expands to a regular expression longer than {MAX_REGEX} bytes"
            ));
        }
        Ok(())
    }
}
