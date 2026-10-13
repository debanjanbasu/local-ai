//! English grapheme-to-phoneme conversion for Kokoro, ported from `misaki`
//! 0.9.4 (`misaki/en.py`, Apache-2.0) and reading its own gold and silver
//! lexicons from the artifact.
//!
//! Differences from the original, all because nothing GPL or model-based is
//! linked here:
//!
//! - `misaki` tokenizes and part-of-speech tags with spaCy. Here a rule
//!   tokenizer splits punctuation the way spaCy does for ordinary prose and a
//!   few context rules stand in for the tagger where the lexicon needs a tag
//!   (determiner `a`, heteronyms such as `read`/`close`/`live`, proper nouns,
//!   `used to`). Heteronyms without such a cue take the lexicon's default.
//! - Words the lexicons do not know fall back, in `misaki` with Kokoro's
//!   pipeline, to espeak-ng (GPL). Here they are segmented into known words
//!   and spelled-out letter sounds ([`Lexicon::fallback`]), which is worse on
//!   unusual names but keeps the binary permissive. `docs/SPEECH.md` reports
//!   the agreement with `misaki` on a public-domain sample.

use std::collections::HashMap;
use std::sync::Arc;

use serde_json::Value;

const PRIMARY: char = 'ˈ';
const SECONDARY: char = 'ˌ';
const VOWELS: &str = "AIOQWYaiuæɑɒɔəɛɜɪʊʌᵻ";
const CONSONANTS: &str = "bdfhjklmnpstvwzðŋɡɹɾʃʒʤʧθ";
const US_TAUS: &str = "AIOWYiuæɑəɛɪɹʊʌ";
const DIPHTHONGS: &str = "AIOQWYʤʧ";
const PUNCTS: &str = ";:,.!?—…\"“”";
const NON_QUOTE_PUNCTS: &str = ";:,.!?—…";
const SUBTOKEN_JUNKS: &str = "',-._‘’/";
const ORDINALS: [&str; 4] = ["st", "nd", "rd", "th"];

fn is_vowel(c: char) -> bool {
    VOWELS.contains(c)
}

/// One output token: the text it came from, its phonemes (`None` when
/// unknown), and whether whitespace follows it.
#[derive(Clone, Debug, PartialEq)]
pub struct Token {
    pub text: String,
    pub phonemes: Option<String>,
    pub whitespace: bool,
}

#[derive(Clone, Debug)]
struct MToken {
    text: String,
    tag: String,
    whitespace: bool,
    is_head: bool,
    alias: Option<String>,
    phonemes: Option<String>,
    stress: Option<f32>,
    currency: Option<char>,
    num_flags: String,
    prespace: bool,
    rating: Option<u8>,
}

impl MToken {
    fn new(text: &str, tag: &str, whitespace: bool) -> Self {
        Self {
            text: text.to_owned(),
            tag: tag.to_owned(),
            whitespace,
            is_head: true,
            alias: None,
            phonemes: None,
            stress: None,
            currency: None,
            num_flags: String::new(),
            prespace: false,
            rating: None,
        }
    }
}

#[derive(Clone, Copy, Default)]
struct Context {
    future_vowel: Option<bool>,
    future_to: bool,
}

#[derive(Clone)]
enum Entry {
    Plain(String),
    Tagged(HashMap<String, Option<String>>),
}

fn lower(s: &str) -> String {
    s.to_lowercase()
}

fn upper(s: &str) -> String {
    s.to_uppercase()
}

/// Python `str.capitalize`.
fn capitalize(s: &str) -> String {
    let mut chars = s.chars();
    chars.next().map_or_else(String::new, |first| {
        first.to_uppercase().collect::<String>() + &chars.as_str().to_lowercase()
    })
}

fn is_alpha(s: &str) -> bool {
    !s.is_empty() && s.chars().all(char::is_alphabetic)
}

fn is_digits(s: &str) -> bool {
    !s.is_empty() && s.chars().all(|c| c.is_ascii_digit())
}

fn lexicon_char(c: char) -> bool {
    c == '\'' || c == '-' || c.is_ascii_alphabetic()
}

fn count(ps: &str, set: &str) -> bool {
    ps.chars().any(|c| set.contains(c))
}

fn apply_stress(ps: &str, stress: Option<f32>) -> String {
    let Some(stress) = stress else {
        return ps.to_owned();
    };
    let has_primary = ps.contains(PRIMARY);
    let has_secondary = ps.contains(SECONDARY);
    let has_any = has_primary || has_secondary;
    let restress = |ps: &str| -> String {
        let chars: Vec<char> = ps.chars().collect();
        let mut keyed: Vec<(f64, char)> = chars
            .iter()
            .enumerate()
            .map(|(i, c)| (i as f64, *c))
            .collect();
        for (i, c) in chars.iter().enumerate() {
            if *c == PRIMARY || *c == SECONDARY {
                if let Some(j) = (i..chars.len()).find(|j| is_vowel(chars[*j])) {
                    keyed[i].0 = j as f64 - 0.5;
                }
            }
        }
        keyed.sort_by(|a, b| a.0.total_cmp(&b.0));
        keyed.into_iter().map(|(_, c)| c).collect()
    };
    if stress < -1.0 {
        ps.replace([PRIMARY, SECONDARY], "")
    } else if stress == -1.0 || ((stress == 0.0 || stress == -0.5) && has_primary) {
        ps.replace(SECONDARY, "").replace(PRIMARY, &SECONDARY.to_string())
    } else if (stress == 0.0 || stress == 0.5 || stress == 1.0) && !has_any {
        if !count(ps, VOWELS) {
            return ps.to_owned();
        }
        restress(&format!("{SECONDARY}{ps}"))
    } else if stress >= 1.0 && !has_primary && has_secondary {
        ps.replace(SECONDARY, &PRIMARY.to_string())
    } else if stress > 1.0 && !has_any {
        if !count(ps, VOWELS) {
            return ps.to_owned();
        }
        restress(&format!("{PRIMARY}{ps}"))
    } else {
        ps.to_owned()
    }
}

fn stress_weight(ps: &str) -> usize {
    ps.chars()
        .map(|c| if DIPHTHONGS.contains(c) { 2 } else { 1 })
        .sum()
}

fn parent_tag(tag: &str) -> &str {
    if tag.starts_with("VB") {
        "VERB"
    } else if tag.starts_with("NN") {
        "NOUN"
    } else if tag.starts_with("ADV") || tag.starts_with("RB") {
        "ADV"
    } else if tag.starts_with("ADJ") || tag.starts_with("JJ") {
        "ADJ"
    } else {
        tag
    }
}

type Lookup = (Option<String>, Option<u8>);

/// The `misaki` lexicon (gold, then silver) for one accent.
pub struct Lexicon {
    british: bool,
    golds: HashMap<String, Entry>,
    silvers: HashMap<String, Entry>,
    /// Lowercase alphabetic lexicon keys of 3+ letters (fallback segmentation).
    pieces: HashMap<String, String>,
}

fn grow(raw: HashMap<String, Entry>) -> HashMap<String, Entry> {
    let mut grown = HashMap::with_capacity(raw.len() * 2);
    for (k, v) in &raw {
        if k.chars().count() < 2 {
            continue;
        }
        if *k == lower(k) {
            let cap = capitalize(k);
            if *k != cap {
                grown.insert(cap, v.clone());
            }
        } else if *k == capitalize(&lower(k)) {
            grown.insert(lower(k), v.clone());
        }
    }
    grown.extend(raw);
    grown
}

fn parse(json: &[u8]) -> crate::Result<HashMap<String, Entry>> {
    let value: HashMap<String, Value> = serde_json::from_slice(json)?;
    let mut out = HashMap::with_capacity(value.len());
    for (k, v) in value {
        let entry = match v {
            Value::String(s) => Entry::Plain(s),
            Value::Object(map) => Entry::Tagged(
                map.into_iter()
                    .map(|(tag, ps)| (tag, ps.as_str().map(str::to_owned)))
                    .collect(),
            ),
            _ => {
                return Err(crate::Error::InvalidFormat(format!(
                    "speech lexicon entry {k} is neither a string nor a map"
                )));
            }
        };
        out.insert(k, entry);
    }
    Ok(out)
}

impl Lexicon {
    /// Build from the gold and silver JSON lexicons.
    ///
    /// # Errors
    ///
    /// Malformed JSON.
    pub fn new(gold: &[u8], silver: &[u8], british: bool) -> crate::Result<Self> {
        let golds = grow(parse(gold)?);
        let silvers = grow(parse(silver)?);
        let mut pieces = HashMap::new();
        for source in [&silvers, &golds] {
            for (k, v) in source {
                if let Entry::Plain(ps) = v
                    && k.chars().count() >= 3
                    && k.chars().all(|c| c.is_ascii_lowercase())
                {
                    pieces.insert(k.clone(), ps.clone());
                }
            }
        }
        Ok(Self {
            british,
            golds,
            silvers,
            pieces,
        })
    }

    fn gold_plain(&self, key: &str) -> Option<String> {
        match self.golds.get(key)? {
            Entry::Plain(s) => Some(s.clone()),
            Entry::Tagged(map) => map.get("DEFAULT").cloned().flatten(),
        }
    }

    fn get_nnp(&self, word: &str) -> Lookup {
        let mut ps = String::new();
        for c in word.chars().filter(|c| c.is_alphabetic()) {
            match self.gold_plain(&c.to_uppercase().to_string()) {
                Some(p) => ps.push_str(&p),
                None => return (None, None),
            }
        }
        let ps = apply_stress(&ps, Some(0.0));
        let out = match ps.rfind(SECONDARY) {
            Some(at) => format!("{}{PRIMARY}{}", &ps[..at], &ps[at + SECONDARY.len_utf8()..]),
            None => ps,
        };
        (Some(out), Some(3))
    }

    #[allow(clippy::too_many_lines)]
    fn get_special_case(&self, word: &str, tag: &str, stress: Option<f32>, ctx: Context) -> Lookup {
        let symbols = |w: &str| match w {
            "%" => Some("percent"),
            "&" => Some("and"),
            "+" => Some("plus"),
            "@" => Some("at"),
            _ => None,
        };
        if tag == "ADD" && (word == "." || word == "/") {
            let name = if word == "." { "dot" } else { "slash" };
            return self.lookup(name, None, Some(-0.5), Some(ctx));
        }
        if let Some(name) = symbols(word) {
            return self.lookup(name, None, None, Some(ctx));
        }
        let stripped = word.trim_matches('.');
        if stripped.contains('.')
            && is_alpha(&word.replace('.', ""))
            && word.split('.').map(|p| p.chars().count()).max().unwrap_or(0) < 3
        {
            return self.get_nnp(word);
        }
        match word {
            "a" | "A" => {
                return (Some(if tag == "DT" { "ɐ" } else { "ˈA" }.to_owned()), Some(4));
            }
            "am" | "Am" | "AM" => {
                if tag.starts_with("NN") {
                    return self.get_nnp(word);
                } else if ctx.future_vowel.is_none()
                    || word != "am"
                    || stress.is_some_and(|s| s > 0.0)
                {
                    return (self.gold_plain("am"), Some(4));
                }
                return (Some("ɐm".to_owned()), Some(4));
            }
            "an" | "An" | "AN" => {
                if word == "AN" && tag.starts_with("NN") {
                    return self.get_nnp(word);
                }
                return (Some("ɐn".to_owned()), Some(4));
            }
            "I" if tag == "PRP" => return (Some(format!("{SECONDARY}I")), Some(4)),
            "by" | "By" | "BY" if parent_tag(tag) == "ADV" => {
                return (Some("bˈI".to_owned()), Some(4));
            }
            "to" | "To" => return (self.to_form(ctx), Some(4)),
            "TO" if tag == "TO" || tag == "IN" => return (self.to_form(ctx), Some(4)),
            "in" | "In" => return (Some(self.in_form(tag, ctx)), Some(4)),
            "IN" if tag != "NNP" => return (Some(self.in_form(tag, ctx)), Some(4)),
            "the" | "The" => return (Some(the_form(ctx)), Some(4)),
            "THE" if tag == "DT" => return (Some(the_form(ctx)), Some(4)),
            "used" | "Used" | "USED" => {
                if let Some(Entry::Tagged(map)) = self.golds.get("used") {
                    let key = if (tag == "VBD" || tag == "JJ") && ctx.future_to {
                        "VBD"
                    } else {
                        "DEFAULT"
                    };
                    return (map.get(key).cloned().flatten(), Some(4));
                }
            }
            _ => {}
        }
        if tag == "IN" {
            let lw = lower(word);
            if lw == "vs" || lw == "vs." {
                return self.lookup("versus", None, None, Some(ctx));
            }
        }
        (None, None)
    }

    fn to_form(&self, ctx: Context) -> Option<String> {
        match ctx.future_vowel {
            None => self.gold_plain("to"),
            Some(false) => Some("tə".to_owned()),
            Some(true) => Some("tʊ".to_owned()),
        }
    }

    fn in_form(&self, tag: &str, ctx: Context) -> String {
        let _ = self;
        if ctx.future_vowel.is_none() || tag != "IN" {
            format!("{PRIMARY}ɪn")
        } else {
            "ɪn".to_owned()
        }
    }

    fn is_known(&self, word: &str) -> bool {
        if self.golds.contains_key(word)
            || matches!(word, "%" | "&" | "+" | "@")
            || self.silvers.contains_key(word)
        {
            return true;
        }
        if !is_alpha(word) || !word.chars().all(lexicon_char) {
            return false;
        }
        if word.chars().count() == 1 {
            return true;
        }
        if word == upper(word) && self.golds.contains_key(&lower(word)) {
            return true;
        }
        let rest: String = word.chars().skip(1).collect();
        rest == upper(&rest)
    }

    fn lookup(&self, word: &str, tag: Option<&str>, stress: Option<f32>, ctx: Option<Context>) -> Lookup {
        let mut word = word.to_owned();
        let mut is_nnp = None;
        if word == upper(&word) && !self.golds.contains_key(&word) {
            word = lower(&word);
            is_nnp = Some(tag == Some("NNP"));
        }
        let mut entry = self.golds.get(&word).cloned();
        let mut rating = 4;
        if entry.is_none() && is_nnp != Some(true) {
            entry = self.silvers.get(&word).cloned();
            rating = 3;
        }
        let ps = match entry {
            None => None,
            Some(Entry::Plain(s)) => Some(s),
            Some(Entry::Tagged(map)) => {
                let key: Option<String> = if ctx.is_some_and(|c| c.future_vowel.is_none())
                    && map.contains_key("None")
                {
                    Some("None".to_owned())
                } else {
                    match tag {
                        Some(t) if map.contains_key(t) => Some(t.to_owned()),
                        Some(t) => Some(parent_tag(t).to_owned()),
                        None => None,
                    }
                };
                match key.as_deref().and_then(|k| map.get(k)) {
                    Some(value) => value.clone(),
                    None => map.get("DEFAULT").cloned().flatten(),
                }
            }
        };
        if ps.is_none() || (is_nnp == Some(true) && !ps.as_deref().unwrap_or("").contains(PRIMARY)) {
            let (nnp, r) = self.get_nnp(&word);
            if nnp.is_some() {
                return (nnp, r);
            }
        }
        match ps {
            Some(ps) => (Some(apply_stress(&ps, stress)), Some(rating)),
            None => (None, None),
        }
    }

    fn suffix_s(&self, stem: Option<String>) -> Option<String> {
        let stem = stem?;
        let last = stem.chars().last()?;
        Some(if "ptkfθ".contains(last) {
            stem + "s"
        } else if "szʃʒʧʤ".contains(last) {
            stem + if self.british { "ɪz" } else { "ᵻz" }
        } else {
            stem + "z"
        })
    }

    fn stem_s(&self, word: &str, tag: Option<&str>, stress: Option<f32>, ctx: Option<Context>) -> Lookup {
        let n = word.chars().count();
        if n < 3 || !word.ends_with('s') {
            return (None, None);
        }
        let stem = if !word.ends_with("ss") && self.is_known(&word[..word.len() - 1]) {
            word[..word.len() - 1].to_owned()
        } else if (word.ends_with("'s")
            || (n > 4 && word.ends_with("es") && !word.ends_with("ies")))
            && self.is_known(&word[..word.len() - 2])
        {
            word[..word.len() - 2].to_owned()
        } else if n > 4 && word.ends_with("ies") && self.is_known(&format!("{}y", &word[..word.len() - 3])) {
            format!("{}y", &word[..word.len() - 3])
        } else {
            return (None, None);
        };
        let (ps, rating) = self.lookup(&stem, tag, stress, ctx);
        (self.suffix_s(ps), rating)
    }

    fn suffix_ed(&self, stem: Option<String>) -> Option<String> {
        let stem = stem?;
        let chars: Vec<char> = stem.chars().collect();
        let last = *chars.last()?;
        Some(if "pkfθʃsʧ".contains(last) {
            stem + "t"
        } else if last == 'd' {
            stem + if self.british { "ɪd" } else { "ᵻd" }
        } else if last != 't' {
            stem + "d"
        } else if self.british || chars.len() < 2 {
            stem + "ɪd"
        } else if US_TAUS.contains(chars[chars.len() - 2]) {
            chars[..chars.len() - 1].iter().collect::<String>() + "ɾᵻd"
        } else {
            stem + "ᵻd"
        })
    }

    fn stem_ed(&self, word: &str, tag: Option<&str>, stress: Option<f32>, ctx: Option<Context>) -> Lookup {
        let n = word.chars().count();
        if n < 4 || !word.ends_with('d') {
            return (None, None);
        }
        let stem = if !word.ends_with("dd") && self.is_known(&word[..word.len() - 1]) {
            &word[..word.len() - 1]
        } else if n > 4
            && word.ends_with("ed")
            && !word.ends_with("eed")
            && self.is_known(&word[..word.len() - 2])
        {
            &word[..word.len() - 2]
        } else {
            return (None, None);
        };
        let (ps, rating) = self.lookup(stem, tag, stress, ctx);
        (self.suffix_ed(ps), rating)
    }

    fn suffix_ing(&self, stem: Option<String>) -> Option<String> {
        let stem = stem?;
        let chars: Vec<char> = stem.chars().collect();
        let last = *chars.last()?;
        if self.british {
            if last == 'ə' || last == 'ː' {
                return None;
            }
        } else if chars.len() > 1 && last == 't' && US_TAUS.contains(chars[chars.len() - 2]) {
            return Some(chars[..chars.len() - 1].iter().collect::<String>() + "ɾɪŋ");
        }
        Some(stem + "ɪŋ")
    }

    fn stem_ing(&self, word: &str, tag: Option<&str>, stress: Option<f32>, ctx: Option<Context>) -> Lookup {
        let n = word.chars().count();
        if n < 5 || !word.ends_with("ing") {
            return (None, None);
        }
        let base = &word[..word.len() - 3];
        let doubled = {
            let b: Vec<char> = base.chars().collect();
            (b.len() >= 2 && b[b.len() - 1] == b[b.len() - 2] && "bcdgklmnprstvxz".contains(b[b.len() - 1]))
                || word.ends_with("cking")
        };
        let stem = if n > 5 && self.is_known(base) {
            base.to_owned()
        } else if self.is_known(&format!("{base}e")) {
            format!("{base}e")
        } else if n > 5 && doubled && self.is_known(&word[..word.len() - 4]) {
            word[..word.len() - 4].to_owned()
        } else {
            return (None, None);
        };
        let (ps, rating) = self.lookup(&stem, tag, stress, ctx);
        (self.suffix_ing(ps), rating)
    }

    fn get_word(&self, word: &str, tag: &str, stress: Option<f32>, ctx: Context) -> Lookup {
        let special = self.get_special_case(word, tag, stress, ctx);
        if special.0.is_some() {
            return special;
        }
        let wl = lower(word);
        let mut word = word.to_owned();
        let rest: String = word.chars().skip(1).collect();
        if word.chars().count() > 1
            && is_alpha(&word.replace('\'', ""))
            && word != wl
            && (tag != "NNP" || word.chars().count() > 7)
            && !self.golds.contains_key(&word)
            && !self.silvers.contains_key(&word)
            && (word == upper(&word) || rest == lower(&rest))
            && (self.golds.contains_key(&wl)
                || self.silvers.contains_key(&wl)
                || self.stem_s(&wl, Some(tag), stress, Some(ctx)).0.is_some()
                || self.stem_ed(&wl, Some(tag), stress, Some(ctx)).0.is_some()
                || self.stem_ing(&wl, Some(tag), stress, Some(ctx)).0.is_some())
        {
            word = wl;
        }
        if self.is_known(&word) {
            return self.lookup(&word, Some(tag), stress, Some(ctx));
        }
        if word.ends_with("s'") && self.is_known(&format!("{}'s", &word[..word.len() - 2])) {
            return self.lookup(&format!("{}'s", &word[..word.len() - 2]), Some(tag), stress, Some(ctx));
        }
        if word.ends_with('\'') && self.is_known(&word[..word.len() - 1]) {
            return self.lookup(&word[..word.len() - 1], Some(tag), stress, Some(ctx));
        }
        let s = self.stem_s(&word, Some(tag), stress, Some(ctx));
        if s.0.is_some() {
            return s;
        }
        let ed = self.stem_ed(&word, Some(tag), stress, Some(ctx));
        if ed.0.is_some() {
            return ed;
        }
        let ing = self.stem_ing(&word, Some(tag), Some(stress.unwrap_or(0.5)), Some(ctx));
        if ing.0.is_some() {
            return ing;
        }
        (None, None)
    }

    fn append_currency(&self, ps: String, currency: Option<char>) -> String {
        let unit = match currency {
            Some('$') => "dollar",
            Some('£') => "pound",
            Some('€') => "euro",
            _ => return ps,
        };
        match self.stem_s(&format!("{unit}s"), None, None, None).0 {
            Some(c) => format!("{ps} {c}"),
            None => ps,
        }
    }

    fn is_number(word: &str, is_head: bool) -> bool {
        if !word.chars().any(|c| c.is_ascii_digit()) {
            return false;
        }
        let mut w = word;
        for s in ["ing", "'d", "ed", "'s", "st", "nd", "rd", "th", "s"] {
            if let Some(stripped) = w.strip_suffix(s) {
                w = stripped;
                break;
            }
        }
        w.chars()
            .enumerate()
            .all(|(i, c)| c.is_ascii_digit() || c == ',' || c == '.' || (is_head && i == 0 && c == '-'))
    }

    fn is_currency(word: &str) -> bool {
        if !word.contains('.') {
            return true;
        }
        if word.matches('.').count() > 1 {
            return false;
        }
        let cents = word.split('.').nth(1).unwrap_or("");
        cents.chars().count() < 3
    }

    #[allow(clippy::too_many_lines)]
    fn get_number(&self, word: &str, currency: Option<char>, is_head: bool, num_flags: &str) -> Lookup {
        let suffix_len = word
            .chars()
            .rev()
            .take_while(|c| c.is_ascii_lowercase() || *c == '\'')
            .count();
        let split = word.len() - word.chars().rev().take(suffix_len).map(char::len_utf8).sum::<usize>();
        let suffix = if suffix_len > 0 { Some(&word[split..]) } else { None };
        let mut word = &word[..split];
        let mut result: Vec<(String, u8)> = Vec::new();
        if let Some(rest) = word.strip_prefix('-') {
            if let (Some(p), Some(r)) = self.lookup("minus", None, None, None) {
                result.push((p, r));
            }
            word = rest;
        }
        let extend = |text: &str, first: bool, result: &mut Vec<(String, u8)>| {
            let splits: Vec<&str> = text
                .split(|c: char| !c.is_ascii_lowercase())
                .filter(|s| !s.is_empty())
                .collect();
            for (i, w) in splits.iter().enumerate() {
                if *w != "and" || num_flags.contains('&') {
                    if first && i == 0 && splits.len() > 1 && *w == "one" && num_flags.contains('a') {
                        result.push(("ə".to_owned(), 4));
                    } else {
                        let stress = if *w == "point" { Some(-2.0) } else { None };
                        if let (Some(p), Some(r)) = self.lookup(w, None, stress, None) {
                            result.push((p, r));
                        }
                    }
                } else if num_flags.contains('n') && !result.is_empty() {
                    let last = result.len() - 1;
                    result[last].0.push_str("ən");
                }
            }
        };
        let digit = |n: &str| super::numbers::cardinal_str(n);
        if is_digits(word) && suffix.is_some_and(|s| ORDINALS.contains(&s)) {
            extend(&super::numbers::ordinal_str(word), true, &mut result);
        } else if result.is_empty()
            && word.chars().count() == 4
            && !matches!(currency, Some('$' | '£' | '€'))
            && is_digits(word)
        {
            extend(&super::numbers::year_str(word), true, &mut result);
        } else if !is_head && !word.contains('.') {
            let num = word.replace(',', "");
            let chars: Vec<char> = num.chars().collect();
            if chars.first() == Some(&'0') || chars.len() > 3 {
                for c in &chars {
                    extend(&digit(&c.to_string()), false, &mut result);
                }
            } else if chars.len() == 3 && !num.ends_with("00") {
                extend(&digit(&chars[0].to_string()), true, &mut result);
                if chars[1] == '0' {
                    if let (Some(p), Some(r)) = self.lookup("O", None, Some(-2.0), None) {
                        result.push((p, r));
                    }
                    extend(&digit(&chars[2].to_string()), false, &mut result);
                } else {
                    extend(&digit(&num[1..]), false, &mut result);
                }
            } else {
                extend(&digit(&num), true, &mut result);
            }
        } else if word.matches('.').count() > 1 || !is_head {
            let mut first = true;
            for num in word.replace(',', "").split('.') {
                if num.is_empty() {
                } else if num.starts_with('0')
                    || (num.chars().count() != 2 && num.chars().skip(1).any(|c| c != '0'))
                {
                    for c in num.chars() {
                        extend(&digit(&c.to_string()), false, &mut result);
                    }
                } else {
                    extend(&digit(num), first, &mut result);
                }
                first = false;
            }
        } else if matches!(currency, Some('$' | '£' | '€')) && Self::is_currency(word) {
            let units = match currency {
                Some('$') => ("dollar", "cent"),
                Some('£') => ("pound", "pence"),
                _ => ("euro", "cent"),
            };
            let cleaned = word.replace(',', "");
            let mut pairs: Vec<(u64, &str)> = cleaned
                .split('.')
                .zip([units.0, units.1])
                .map(|(n, u)| (n.parse::<u64>().unwrap_or(0), u))
                .collect();
            if pairs.len() > 1 {
                if pairs[1].0 == 0 {
                    pairs.truncate(1);
                } else if pairs[0].0 == 0 {
                    pairs.remove(0);
                }
            }
            for (i, (num, unit)) in pairs.iter().enumerate() {
                if i > 0
                    && let (Some(p), Some(r)) = self.lookup("and", None, None, None)
                {
                    result.push((p, r));
                }
                extend(&super::numbers::cardinal(*num), i == 0, &mut result);
                let unit_ps = if *num != 1 && *unit != "pence" {
                    self.stem_s(&format!("{unit}s"), None, None, None)
                } else {
                    self.lookup(unit, None, None, None)
                };
                if let (Some(p), Some(r)) = unit_ps {
                    result.push((p, r));
                }
            }
        } else {
            let text = if is_digits(word) {
                digit(word)
            } else if !word.contains('.') {
                digit(&word.replace(',', ""))
            } else {
                let w = word.replace(',', "");
                if let Some(frac) = w.strip_prefix('.') {
                    let digits: Vec<String> =
                        frac.chars().map(|c| digit(&c.to_string())).collect();
                    format!("point {}", digits.join(" "))
                } else {
                    super::numbers::float_str(&w)
                }
            };
            extend(&text, true, &mut result);
        }
        if result.is_empty() {
            return (None, None);
        }
        let rating = result.iter().map(|(_, r)| *r).min();
        let joined = result.into_iter().map(|(p, _)| p).collect::<Vec<_>>().join(" ");
        match suffix {
            Some("s" | "'s") => (self.suffix_s(Some(joined)), rating),
            Some("ed" | "'d") => (self.suffix_ed(Some(joined)), rating),
            Some("ing") => (self.suffix_ing(Some(joined)), rating),
            _ => (Some(joined), rating),
        }
    }

    fn call(&self, tk: &MToken, ctx: Context) -> Lookup {
        let word = tk.alias.as_deref().unwrap_or(&tk.text).replace(['‘', '’'], "'");
        let stress = if word == lower(&word) {
            None
        } else if word == upper(&word) {
            Some(2.0)
        } else {
            Some(0.5)
        };
        let (ps, rating) = self.get_word(&word, &tk.tag, stress, ctx);
        if let Some(ps) = ps {
            return (Some(apply_stress(&self.append_currency(ps, tk.currency), tk.stress)), rating);
        }
        if Self::is_number(&word, tk.is_head) {
            let (ps, rating) = self.get_number(&word, tk.currency, tk.is_head, &tk.num_flags);
            return (ps.map(|ps| apply_stress(&ps, tk.stress)), rating);
        }
        (None, None)
    }

    /// Pronunciation of a word no lexicon knows: the cheapest segmentation
    /// into known words of 3+ letters and spelled letter sounds, first
    /// primary stress kept and later ones made secondary. A heuristic
    /// stand-in for `misaki`'s espeak-ng fallback.
    #[must_use]
    pub fn fallback(&self, word: &str) -> Option<String> {
        let w: Vec<char> = lower(word).chars().filter(char::is_ascii_alphabetic).collect();
        if w.is_empty() {
            return None;
        }
        let n = w.len();
        // Short all-caps words are initialisms: spell the letter names.
        if word.chars().count() <= 4 && word == upper(word) && is_alpha(word) {
            return self.get_nnp(word).0;
        }
        let mut best: Vec<Option<(u32, usize, String)>> = vec![None; n + 1];
        best[0] = Some((0, 0, String::new()));
        for start in 0..n {
            let Some((cost, _, _)) = best[start].clone() else {
                continue;
            };
            for end in (start + 1)..=n {
                let piece: String = w[start..end].iter().collect();
                let option = if end - start >= 3 {
                    self.pieces.get(&piece).map(|ps| (3, ps.clone()))
                } else {
                    None
                }
                .or_else(|| letter_sound(&piece).map(|ps| (2 + 2 * (end - start) as u32, ps.to_owned())));
                if let Some((c, ps)) = option {
                    let total = cost + c;
                    if best[end].as_ref().is_none_or(|(b, _, _)| total < *b) {
                        best[end] = Some((total, start, ps));
                    }
                }
            }
        }
        best[n].as_ref()?;
        let mut parts = Vec::new();
        let mut at = n;
        while at > 0 {
            let (_, start, ps) = best[at].clone()?;
            parts.push(ps);
            at = start;
        }
        parts.reverse();
        let joined: String = parts.concat();
        // One primary stress: the first one (or on the first vowel), others secondary.
        let mut seen = false;
        let mut out: String = joined
            .chars()
            .map(|c| {
                if c == PRIMARY {
                    if seen {
                        SECONDARY
                    } else {
                        seen = true;
                        PRIMARY
                    }
                } else {
                    c
                }
            })
            .collect();
        if !seen {
            out = apply_stress(&out, Some(2.0));
        }
        Some(out)
    }
}

/// Letter-sound spellings used by [`Lexicon::fallback`].
fn letter_sound(piece: &str) -> Option<&'static str> {
    Some(match piece {
        "a" => "æ",
        "b" => "b",
        "c" => "k",
        "d" => "d",
        "e" => "ɛ",
        "f" => "f",
        "g" => "ɡ",
        "h" => "h",
        "i" => "ɪ",
        "j" => "ʤ",
        "k" => "k",
        "l" => "l",
        "m" => "m",
        "n" => "n",
        "o" => "ɑ",
        "p" => "p",
        "q" => "k",
        "r" => "ɹ",
        "s" => "s",
        "t" => "t",
        "u" => "ʌ",
        "v" => "v",
        "w" => "w",
        "x" => "ks",
        "y" => "i",
        "z" => "z",
        "sh" => "ʃ",
        "ch" => "ʧ",
        "th" => "θ",
        "ph" => "f",
        "ck" => "k",
        "ng" => "ŋ",
        "qu" => "kw",
        "ee" => "i",
        "ea" => "i",
        "oo" => "u",
        "ou" => "W",
        "ow" => "O",
        "ai" | "ay" => "A",
        "oa" => "O",
        "ie" => "i",
        "ey" => "i",
        "er" => "əɹ",
        "ar" => "ɑɹ",
        "or" => "ɔɹ",
        "ll" => "l",
        "ss" => "s",
        "tt" => "t",
        "nn" => "n",
        "mm" => "m",
        "rr" => "ɹ",
        "ff" => "f",
        "dd" => "d",
        "pp" => "p",
        "gh" => "",
        "wh" => "w",
        "kn" => "n",
        "wr" => "ɹ",
        _ => return None,
    })
}

fn the_form(ctx: Context) -> String {
    if ctx.future_vowel == Some(true) {
        "ði".to_owned()
    } else {
        "ðə".to_owned()
    }
}

/// Split one whitespace-free chunk into spaCy-like tokens: leading and
/// trailing punctuation off, abbreviations ("Mr.") and decimals kept whole.
fn split_chunk(chunk: &str, lexicon: &Lexicon) -> Vec<String> {
    const PREFIX: &str = "\"“‘'([{¿¡$£€#*";
    const SUFFIX: &str = "\"”’')]}.,!?;:…%*";
    let mut prefixes = Vec::new();
    let mut rest = chunk;
    while let Some(c) = rest.chars().next() {
        if PREFIX.contains(c) && rest.chars().count() > 1 {
            // A leading apostrophe stays on lexicon words ("'em", "'Merica").
            if (c == '\'' || c == '‘') && lexicon.golds.contains_key(rest.trim_end_matches(|c: char| SUFFIX.contains(c) && c != '\'')) {
                break;
            }
            prefixes.push(c.to_string());
            rest = &rest[c.len_utf8()..];
        } else {
            break;
        }
    }
    let mut suffixes = Vec::new();
    loop {
        if rest.ends_with("...") && rest.len() > 3 {
            suffixes.push("...".to_owned());
            rest = &rest[..rest.len() - 3];
            continue;
        }
        let Some(c) = rest.chars().last() else { break };
        if !SUFFIX.contains(c) || rest.chars().count() == 1 {
            break;
        }
        if c == '.' {
            // Abbreviations and initialisms keep their period.
            let core = rest.trim_end_matches('.');
            if lexicon.golds.contains_key(rest)
                || (core.contains('.') && is_alpha(&core.replace('.', "")))
            {
                break;
            }
        }
        if (c == '\'' || c == '’') && rest.chars().count() > 2 {
            // Plural possessive ("dogs'") stays attached.
            let before: Vec<char> = rest.chars().rev().skip(1).take(1).collect();
            if before.first().is_some_and(|b| *b == 's' || *b == 'S') {
                break;
            }
        }
        suffixes.push(c.to_string());
        rest = &rest[..rest.len() - c.len_utf8()];
    }
    let mut out = prefixes;
    if !rest.is_empty() {
        out.push(rest.to_owned());
    }
    out.extend(suffixes.into_iter().rev());
    out
}

/// `misaki`'s subtoken regex as a scanner (letters, apostrophe-joined
/// words, numbers, camelCase and acronym boundaries, hyphen runs).
fn subtokenize(word: &str) -> Vec<String> {
    let c: Vec<char> = word.chars().collect();
    let n = c.len();
    let apos = |x: char| x == '\'' || x == '‘' || x == '’';
    let mut out = Vec::new();
    let mut i = 0;
    while i < n {
        // ^['‘’]+
        if i == 0 && apos(c[0]) {
            let mut j = 0;
            while j < n && apos(c[j]) {
                j += 1;
            }
            out.push(c[..j].iter().collect());
            i = j;
            continue;
        }
        // \p{Lu}(?=\p{Lu}\p{Ll})
        if c[i].is_uppercase() && i + 2 < n && c[i + 1].is_uppercase() && c[i + 2].is_lowercase() {
            out.push(c[i].to_string());
            i += 1;
            continue;
        }
        // (?:^-)?(?:\d?[,.]?\d)+
        {
            let sep = |x: char| x == ',' || x == '.';
            let start = if i == 0 && c[0] == '-' { 1 } else { i };
            let mut k = start;
            if k + 1 < n && sep(c[k]) && c[k + 1].is_ascii_digit() {
                k += 1;
            }
            if k < n && c[k].is_ascii_digit() {
                while k < n {
                    if c[k].is_ascii_digit() {
                        k += 1;
                    } else if sep(c[k]) && k + 1 < n && c[k + 1].is_ascii_digit() && c[k - 1].is_ascii_digit() {
                        k += 1;
                    } else {
                        break;
                    }
                }
                out.push(c[i..k].iter().collect());
                i = k;
                continue;
            }
        }
        // [-_]+
        if c[i] == '-' || c[i] == '_' {
            let mut j = i;
            while j < n && (c[j] == '-' || c[j] == '_') {
                j += 1;
            }
            out.push(c[i..j].iter().collect());
            i = j;
            continue;
        }
        // ['‘’]{2,}
        if apos(c[i]) && i + 1 < n && apos(c[i + 1]) {
            let mut j = i;
            while j < n && apos(c[j]) {
                j += 1;
            }
            out.push(c[i..j].iter().collect());
            i = j;
            continue;
        }
        if c[i].is_alphabetic() {
            // Letter run with apostrophe joins.
            let mut j = i;
            while j < n {
                if c[j].is_alphabetic() {
                    j += 1;
                } else if apos(c[j]) && j + 1 < n && c[j + 1].is_alphabetic() {
                    j += 2;
                } else {
                    break;
                }
            }
            // camelCase: the shortest prefix ending in a lowercase letter
            // followed by an uppercase one.
            let camel = (i..j.saturating_sub(1))
                .find(|k| c[*k].is_lowercase() && c[k + 1].is_uppercase());
            let end = camel.map_or(j, |k| k + 1);
            out.push(c[i..end].iter().collect());
            i = end;
            continue;
        }
        if !apos(c[i]) && !c[i].is_ascii_digit() {
            out.push(c[i].to_string());
            i += 1;
            continue;
        }
        // ['‘’]+$
        if apos(c[i]) && c[i..].iter().all(|x| apos(*x)) {
            out.push(c[i..].iter().collect());
            break;
        }
        i += 1;
    }
    out
}

fn punct_tag(text: &str, opening: bool) -> Option<&'static str> {
    Some(match text {
        "." | "!" | "?" | "?!" | "!?" => ".",
        "," => ",",
        ":" | ";" | "-" | "--" | "—" | "–" | "..." | "…" => ":",
        "(" | "[" | "{" => "-LRB-",
        ")" | "]" | "}" => "-RRB-",
        "“" => "``",
        "”" => "''",
        "\"" => {
            if opening {
                "``"
            } else {
                "''"
            }
        }
        "$" | "£" | "€" => "$",
        "#" => "#",
        "*" | "~" | "^" | "|" | "<" | ">" | "=" | "/" | "\\" => "NFP",
        _ => return None,
    })
}

const MODALS: [&str; 24] = [
    "to", "will", "would", "can", "could", "should", "shall", "may", "might", "must", "do",
    "does", "did", "don't", "doesn't", "didn't", "won't", "can't", "cannot", "let's", "please",
    "wouldn't", "couldn't", "shouldn't",
];
const DETERMINERS: [&str; 15] = [
    "the", "a", "an", "this", "that", "these", "those", "my", "your", "his", "her", "its",
    "our", "their", "every",
];
const PREPOSITIONS: [&str; 16] = [
    "of", "in", "at", "for", "on", "with", "by", "from", "about", "like", "after", "before",
    "into", "over", "upon", "under",
];
const PRONOUNS: [&str; 7] = ["i", "you", "he", "she", "it", "we", "they"];
const AUXILIARIES: [&str; 11] = [
    "have", "has", "had", "having", "was", "were", "is", "are", "be", "been", "being",
];

/// Rule tags standing in for spaCy's tagger (see the module docs).
fn tag_words(tokens: &mut [MToken]) {
    let mut previous: Option<String> = None;
    let mut sentence_start = true;
    for i in 0..tokens.len() {
        let text = tokens[i].text.clone();
        let opening = i == 0 || tokens[i - 1].whitespace;
        if let Some(tag) = punct_tag(&text, opening) {
            tokens[i].tag = tag.to_owned();
            if tag == "." {
                sentence_start = true;
            }
            if tag != "``" && tag != "''" && tag != "-LRB-" {
                previous = None;
            }
            continue;
        }
        if text.chars().any(|c| c.is_ascii_digit()) && Lexicon::is_number(&text, true) {
            tokens[i].tag = "CD".to_owned();
            previous = None;
            sentence_start = false;
            continue;
        }
        let lw = lower(&text);
        let next_is_word = tokens
            .get(i + 1)
            .is_some_and(|t| t.text.chars().any(char::is_alphanumeric));
        let tag = if lw == "a" {
            if text == "a" || next_is_word { "DT" } else { "NN" }
        } else if text == "I" {
            "PRP"
        } else if lw == "to" {
            "TO"
        } else if ["in", "on", "at", "by", "of", "for", "with", "from", "vs", "vs."].contains(&lw.as_str()) {
            "IN"
        } else if lw == "used" {
            "VBD"
        } else if lw == "that" || lw == "that's" {
            // Demonstrative (stressed) before punctuation, after a
            // preposition, or contracted; conjunction or relative otherwise.
            let next_is_punct = tokens
                .get(i + 1)
                .is_none_or(|t| punct_tag(&t.text, false).is_some());
            if lw == "that's"
                || next_is_punct
                || previous.as_deref().is_some_and(|p| PREPOSITIONS.contains(&p))
            {
                "DT"
            } else {
                "IN"
            }
        } else if text.chars().count() > 1 && text == upper(&text) && is_alpha(&text) {
            "NNP"
        } else if !sentence_start
            && text.chars().next().is_some_and(char::is_uppercase)
            && is_alpha(&text)
        {
            "NNP"
        } else {
            match previous.as_deref() {
                Some(p) if MODALS.contains(&p) => "VB",
                Some(p) if DETERMINERS.contains(&p) => "NN",
                Some(p) if PRONOUNS.contains(&p) => "VBD",
                Some(p) if AUXILIARIES.contains(&p) => "VBN",
                _ => "",
            }
        };
        tokens[i].tag = tag.to_owned();
        previous = Some(lw);
        sentence_start = false;
    }
}

/// The `misaki` English G2P for one accent.
pub struct G2p {
    lexicon: Arc<Lexicon>,
    /// For British: the American lexicon, consulted (and its phonemes mapped
    /// to British symbols) for words the British lexicons lack, before the
    /// segmentation fallback.
    alternate: Option<Arc<Lexicon>>,
}

/// American lexicon phonemes in British symbols (the vowel and rhotic
/// differences of `misaki`'s two phoneme sets).
fn us_to_gb(ps: &str) -> String {
    let chars: Vec<char> = ps.chars().collect();
    let mut out = String::with_capacity(ps.len());
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        let next = chars.get(i + 1).copied();
        let after = chars.get(i + 2).copied();
        if next == Some('ɹ') && "ɜɑɔɛɪʊə".contains(c) && !after.is_some_and(is_vowel) {
            out.push_str(match c {
                'ɜ' => "ɜː",
                'ɑ' => "ɑː",
                'ɔ' => "ɔː",
                'ɛ' => "ɛː",
                'ɪ' => "ɪə",
                'ʊ' => "ʊə",
                _ => "ə",
            });
            i += 2;
            continue;
        }
        out.push(match c {
            'æ' => 'a',
            'O' => 'Q',
            'ɾ' => 't',
            'ᵻ' => 'ɪ',
            'ɑ' => 'ɒ',
            other => other,
        });
        i += 1;
    }
    out
}

fn merge_texts(tokens: &[MToken]) -> String {
    let mut text = String::new();
    for (i, t) in tokens.iter().enumerate() {
        text.push_str(&t.text);
        if i + 1 < tokens.len() && t.whitespace {
            text.push(' ');
        }
    }
    text
}

fn merge(tokens: &[MToken], unk: Option<&str>) -> MToken {
    let mut stresses: Vec<f32> = tokens.iter().filter_map(|t| t.stress).collect();
    stresses.sort_by(f32::total_cmp);
    stresses.dedup();
    let currencies: Vec<char> = tokens.iter().filter_map(|t| t.currency).collect();
    let phonemes = unk.map(|unk| {
        let mut ps = String::new();
        for t in tokens {
            if t.prespace && !ps.is_empty() && !ps.ends_with(char::is_whitespace) && t.phonemes.as_deref().is_some_and(|p| !p.is_empty()) {
                ps.push(' ');
            }
            ps.push_str(t.phonemes.as_deref().unwrap_or(unk));
        }
        ps
    });
    let tag = tokens
        .iter()
        .rev()
        .max_by_key(|t| t.text.chars().map(|c| if c.is_lowercase() || !c.is_alphabetic() { 1 } else { 2 }).sum::<usize>())
        .map(|t| t.tag.clone())
        .unwrap_or_default();
    let mut flags: Vec<char> = tokens.iter().flat_map(|t| t.num_flags.chars()).collect();
    flags.sort_unstable();
    flags.dedup();
    MToken {
        text: merge_texts(tokens),
        tag,
        whitespace: tokens.last().is_some_and(|t| t.whitespace),
        is_head: tokens.first().is_none_or(|t| t.is_head),
        alias: None,
        phonemes,
        stress: if stresses.len() == 1 { Some(stresses[0]) } else { None },
        currency: currencies.into_iter().max(),
        num_flags: flags.into_iter().collect(),
        prespace: tokens.first().is_some_and(|t| t.prespace),
        rating: tokens.iter().map(|t| t.rating).min().flatten(),
    }
}

enum Word {
    One(MToken),
    Group(Vec<MToken>),
}

fn token_context(ctx: Context, ps: Option<&str>, token: &MToken) -> Context {
    let mut vowel = ctx.future_vowel;
    if let Some(ps) = ps {
        for c in ps.chars() {
            if NON_QUOTE_PUNCTS.contains(c) {
                vowel = None;
                break;
            }
            if is_vowel(c) {
                vowel = Some(true);
                break;
            }
            if CONSONANTS.contains(c) {
                vowel = Some(false);
                break;
            }
        }
    }
    let future_to = token.text == "to"
        || token.text == "To"
        || (token.text == "TO" && (token.tag == "TO" || token.tag == "IN"));
    Context {
        future_vowel: vowel,
        future_to,
    }
}

fn resolve_tokens(tokens: &mut [MToken]) {
    let text = merge_texts(tokens);
    let kinds: std::collections::BTreeSet<u8> = text
        .chars()
        .filter(|c| !SUBTOKEN_JUNKS.contains(*c))
        .map(|c| {
            if c.is_alphabetic() {
                0
            } else if c.is_ascii_digit() {
                1
            } else {
                2
            }
        })
        .collect();
    let prespace = text.contains(' ') || text.contains('/') || kinds.len() > 1;
    let last = tokens.len() - 1;
    for (i, tk) in tokens.iter_mut().enumerate() {
        if tk.phonemes.is_none() {
            if i == last && tk.text.chars().count() == 1 && NON_QUOTE_PUNCTS.contains(tk.text.as_str()) {
                tk.phonemes = Some(tk.text.clone());
                tk.rating = Some(3);
            } else if tk.text.chars().all(|c| SUBTOKEN_JUNKS.contains(c)) {
                tk.phonemes = Some(String::new());
                tk.rating = Some(3);
            }
        } else if i > 0 {
            tk.prespace = prespace;
        }
    }
    if prespace {
        return;
    }
    let mut indices: Vec<(bool, usize, usize)> = tokens
        .iter()
        .enumerate()
        .filter_map(|(i, tk)| {
            tk.phonemes
                .as_deref()
                .filter(|p| !p.is_empty())
                .map(|p| (p.contains(PRIMARY), stress_weight(p), i))
        })
        .collect();
    if indices.len() == 2 && tokens[indices[0].2].text.chars().count() == 1 {
        let i = indices[1].2;
        tokens[i].phonemes = tokens[i].phonemes.as_deref().map(|p| apply_stress(p, Some(-0.5)));
        return;
    }
    let primaries = indices.iter().filter(|(b, _, _)| *b).count();
    if indices.len() < 2 || primaries <= indices.len().div_ceil(2) {
        return;
    }
    indices.sort_unstable();
    let half = indices.len() / 2;
    for (_, _, i) in indices.into_iter().take(half) {
        tokens[i].phonemes = tokens[i].phonemes.as_deref().map(|p| apply_stress(p, Some(-0.5)));
    }
}

impl G2p {
    #[must_use]
    pub const fn new(lexicon: Arc<Lexicon>) -> Self {
        Self {
            lexicon,
            alternate: None,
        }
    }

    /// A G2P that consults `alternate` (the American lexicon, for a British
    /// G2P) for words its own lexicons lack.
    #[must_use]
    pub const fn with_alternate(lexicon: Arc<Lexicon>, alternate: Arc<Lexicon>) -> Self {
        Self {
            lexicon,
            alternate: Some(alternate),
        }
    }

    fn unknown(&self, tk: &MToken, ctx: Context) -> Option<String> {
        if let Some(alternate) = &self.alternate
            && let (Some(ps), _) = alternate.call(tk, ctx)
        {
            return Some(us_to_gb(&ps));
        }
        self.lexicon.fallback(tk.alias.as_deref().unwrap_or(&tk.text))
    }

    #[must_use]
    pub fn british(&self) -> bool {
        self.lexicon.british
    }

    fn tokenize(&self, text: &str) -> Vec<MToken> {
        let mut tokens = Vec::new();
        let chars: Vec<(usize, char)> = text.char_indices().collect();
        let mut i = 0;
        while i < chars.len() {
            if chars[i].1.is_whitespace() {
                i += 1;
                continue;
            }
            let start = chars[i].0;
            let mut j = i;
            while j < chars.len() && !chars[j].1.is_whitespace() {
                j += 1;
            }
            let end = chars.get(j).map_or(text.len(), |c| c.0);
            let pieces = split_chunk(&text[start..end], &self.lexicon);
            let count = pieces.len();
            for (k, piece) in pieces.into_iter().enumerate() {
                tokens.push(MToken::new(&piece, "", k + 1 == count && j < chars.len()));
            }
            i = j;
        }
        tag_words(&mut tokens);
        tokens
    }

    fn retokenize(tokens: Vec<MToken>) -> Vec<Word> {
        let mut words: Vec<Word> = Vec::new();
        let mut currency: Option<char> = None;
        let count = tokens.len();
        let tags: Vec<String> = tokens.iter().map(|t| t.tag.clone()).collect();
        for (i, token) in tokens.into_iter().enumerate() {
            let mut tks: Vec<MToken> = if token.alias.is_none() && token.phonemes.is_none() {
                subtokenize(&token.text)
                    .into_iter()
                    .map(|t| {
                        let mut m = MToken::new(&t, &token.tag, false);
                        m.num_flags.clone_from(&token.num_flags);
                        m.stress = token.stress;
                        m
                    })
                    .collect()
            } else {
                vec![token.clone()]
            };
            if tks.is_empty() {
                continue;
            }
            let last = tks.len() - 1;
            tks[last].whitespace = token.whitespace;
            let len = tks.len();
            for j in 0..len {
                let mut tk = tks[j].clone();
                if tk.alias.is_some() || tk.phonemes.is_some() {
                } else if tk.tag == "$" && matches!(tk.text.as_str(), "$" | "£" | "€") {
                    currency = tk.text.chars().next();
                    tk.phonemes = Some(String::new());
                    tk.rating = Some(4);
                } else if tk.tag == ":" && (tk.text == "-" || tk.text == "–") {
                    tk.phonemes = Some("—".to_owned());
                    tk.rating = Some(3);
                } else if ["." , ",", "-LRB-", "-RRB-", "``", "\"\"", "''", ":", "$", "#", "NFP"].contains(&tk.tag.as_str())
                    && !tk.text.chars().all(|c| c.is_ascii_alphabetic())
                {
                    tk.phonemes = Some(match tk.tag.as_str() {
                        "-LRB-" => "(".to_owned(),
                        "-RRB-" => ")".to_owned(),
                        "``" => "“".to_owned(),
                        "\"\"" | "''" => "”".to_owned(),
                        _ => tk.text.chars().filter(|c| PUNCTS.contains(*c)).collect(),
                    });
                    tk.rating = Some(4);
                } else if currency.is_some() {
                    if tk.tag != "CD" {
                        currency = None;
                    } else if j + 1 == len && (i + 1 == count || tags[i + 1] != "CD") {
                        tk.currency = currency;
                    }
                } else if 0 < j
                    && j < len - 1
                    && tk.text == "2"
                    && tks[j - 1].text.chars().last().is_some_and(char::is_alphabetic)
                    && tks[j + 1].text.chars().next().is_some_and(char::is_alphabetic)
                {
                    tk.alias = Some("to".to_owned());
                }
                if tk.alias.is_some() || tk.phonemes.is_some() {
                    words.push(Word::One(tk));
                } else if let Some(Word::Group(group)) = words.last_mut()
                    && !group.last().is_some_and(|t| t.whitespace)
                {
                    tk.is_head = false;
                    group.push(tk);
                } else if tk.whitespace {
                    words.push(Word::One(tk));
                } else {
                    words.push(Word::Group(vec![tk]));
                }
            }
        }
        words
            .into_iter()
            .map(|w| match w {
                Word::Group(mut g) if g.len() == 1 => Word::One(g.remove(0)),
                other => other,
            })
            .collect()
    }

    /// Phonemize text: one [`Token`] per word or punctuation mark. Unknown
    /// words use [`Lexicon::fallback`] when `fallback` is set and are left
    /// `None` otherwise (`misaki` with `fallback=None`).
    #[must_use]
    pub fn tokens(&self, text: &str, fallback: bool) -> Vec<Token> {
        let tokens = self.tokenize(text.trim_start());
        let mut words = Self::retokenize(tokens);
        let mut ctx = Context::default();
        for w in words.iter_mut().rev() {
            match w {
                Word::One(tk) => {
                    if tk.phonemes.is_none() {
                        let (ps, rating) = self.lexicon.call(tk, ctx);
                        tk.phonemes = ps;
                        tk.rating = rating;
                    }
                    if tk.phonemes.is_none() && fallback {
                        tk.phonemes = self.unknown(tk, ctx);
                    }
                    ctx = token_context(ctx, tk.phonemes.as_deref(), tk);
                }
                Word::Group(group) => {
                    let (mut left, mut right) = (0, group.len());
                    let mut should_fallback = false;
                    while left < right {
                        let merged = if group[left..right]
                            .iter()
                            .any(|t| t.alias.is_some() || t.phonemes.is_some())
                        {
                            None
                        } else {
                            Some(merge(&group[left..right], None))
                        };
                        let (ps, rating) = merged
                            .as_ref()
                            .map_or((None, None), |m| self.lexicon.call(m, ctx));
                        if let Some(ps) = ps {
                            group[left].phonemes = Some(ps.clone());
                            group[left].rating = rating;
                            for x in &mut group[left + 1..right] {
                                x.phonemes = Some(String::new());
                                x.rating = rating;
                            }
                            if let Some(m) = &merged {
                                ctx = token_context(ctx, Some(&ps), m);
                            }
                            right = left;
                            left = 0;
                        } else if left + 1 < right {
                            left += 1;
                        } else {
                            right -= 1;
                            let tk = &mut group[right];
                            if tk.phonemes.is_none() {
                                if tk.text.chars().all(|c| SUBTOKEN_JUNKS.contains(c)) {
                                    tk.phonemes = Some(String::new());
                                    tk.rating = Some(3);
                                } else if fallback {
                                    should_fallback = true;
                                    break;
                                }
                            }
                            left = 0;
                        }
                    }
                    if should_fallback {
                        let merged = merge(group, None);
                        group[0].phonemes = self.unknown(&merged, ctx);
                        for x in &mut group[1..] {
                            x.phonemes = Some(String::new());
                        }
                    } else {
                        resolve_tokens(group);
                    }
                }
            }
        }
        words
            .into_iter()
            .map(|w| match w {
                Word::One(tk) => tk,
                Word::Group(g) => merge(&g, Some("\u{2753}")),
            })
            .map(|tk| Token {
                text: tk.text,
                phonemes: tk.phonemes.map(|p| p.replace('ɾ', "T").replace('ʔ', "t")),
                whitespace: tk.whitespace,
            })
            .collect()
    }

    /// The phoneme string `misaki` prints (`G2P.__call__`), unknown words as
    /// `❓` when `fallback` is off.
    #[must_use]
    pub fn phonemize(&self, text: &str, fallback: bool) -> String {
        let mut out = String::new();
        for tk in self.tokens(text, fallback) {
            out.push_str(tk.phonemes.as_deref().unwrap_or("\u{2753}"));
            if tk.whitespace {
                out.push(' ');
            }
        }
        out
    }
}
