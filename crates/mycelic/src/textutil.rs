//! The retrieval tokenizer: lower-cased, accent-folded content tokens, stop words removed, possessives stripped,
//! lightly stemmed.  A port of the copy Mycelic took from NeuralGraph's `chat_memory.textutil`; only what
//! retrieval uses came across (the chit-chat gate and entity normalisation had no caller in Mycelic).
//!
//! Python's notion of whitespace (`str.split()`, `str.strip()`, `\s`) is kept, which counts the information
//! separators U+001C..U+001F as whitespace where Rust's `char::is_whitespace` does not.

use regex::Regex;
use std::sync::LazyLock;
use unicode_normalization::UnicodeNormalization;
use unicode_normalization::char::canonical_combining_class;

const STOP: &[&str] = &[
    "the", "and", "for", "are", "but", "not", "you", "all", "can", "had", "her", "was", "one", "our", "with", "that",
    "this", "have", "from", "they", "what", "when", "where", "which", "who", "how", "did", "does", "has", "his", "she",
    "him", "them", "been", "were", "will", "would", "could", "should", "about", "into", "your", "just", "like", "some",
    "than", "then", "there", "their", "also", "very", "really", "much", "many", "more", "yeah", "yes", "hey", "thanks",
    "thank", "sounds", "sure", "good", "great", "awesome", "love", "know", "its", "it's", "i'm", "don't", "can't",
    "get", "got", "let", "lets", "let's", "okay", "well", "still",
];

static TOKEN_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"[a-z0-9][a-z0-9'\-]*").expect("token regex"));
static POSSESSIVE_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"'s\b|'\b").expect("possessive regex"));
static STOP_SET: LazyLock<std::collections::HashSet<&'static str>> = LazyLock::new(|| STOP.iter().copied().collect());

/// Python's `str.isspace()` for one character.
pub fn py_isspace(c: char) -> bool {
    c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)
}

/// Python's `" ".join(text.split())`.
pub fn collapse_ws(text: &str) -> String {
    text.split(py_isspace).filter(|s| !s.is_empty()).collect::<Vec<_>>().join(" ")
}

/// Python's `str.strip()` / `str.rstrip()` with no argument.
pub fn py_strip(s: &str) -> &str {
    s.trim_matches(py_isspace)
}

pub fn py_rstrip(s: &str) -> &str {
    s.trim_end_matches(py_isspace)
}

/// Collapse runs of whitespace to one space and trim (`normalize_ws`).
pub fn normalize_ws(text: &str) -> String {
    collapse_ws(text)
}

const VOWELS: &[u8] = b"aeiou";

fn is_vowel(b: u8) -> bool {
    VOWELS.contains(&b)
}

fn cvc_short(w: &[u8]) -> bool {
    w.len() == 3 && !is_vowel(w[0]) && is_vowel(w[1]) && !is_vowel(w[2]) && !b"wxy".contains(&w[2])
}

/// Porter step-1b clean-up after removing -ed / -ing.
fn after_strip(w: &str) -> String {
    let b = w.as_bytes();
    if w.ends_with("at") || w.ends_with("bl") || w.ends_with("iz") {
        return format!("{w}e");
    }
    if b.len() > 2 {
        let last = b[b.len() - 1];
        if last == b[b.len() - 2] && !b"lsz".contains(&last) && !is_vowel(last) {
            return w[..w.len() - 1].to_string();
        }
    }
    if cvc_short(b) {
        return format!("{w}e");
    }
    w.to_string()
}

fn has_vowel(s: &str) -> bool {
    s.bytes().any(is_vowel)
}

/// Small Porter-style stemmer (lives -> live, moved -> move, running -> run, cities -> city).  Tokens are ASCII
/// (`[a-z0-9'-]`), so byte positions are character positions.
pub fn stem(word: &str) -> String {
    let w = word;
    let n = w.len();
    if n > 4 && w.ends_with("ies") {
        return format!("{}y", &w[..n - 3]);
    }
    if n > 5 && w.ends_with("ing") && has_vowel(&w[..n - 3]) {
        return after_strip(&w[..n - 3]);
    }
    if n > 4 && w.ends_with("ed") && has_vowel(&w[..n - 2]) {
        return after_strip(&w[..n - 2]);
    }
    if n > 3 && w.ends_with("es") && b"sxzh".contains(&w.as_bytes()[n - 3]) {
        return w[..n - 2].to_string();
    }
    if n > 3 && w.ends_with('s') && !w.ends_with("ss") {
        return w[..n - 1].to_string();
    }
    w.to_string()
}

/// Lower-case and strip diacritics (Zürich -> zurich) so keyword matching survives accents.
pub fn fold(text: &str) -> String {
    let stripped: String = text.nfkd().filter(|c| canonical_combining_class(*c) == 0).collect();
    stripped.to_lowercase()
}

fn is_number(t: &str) -> bool {
    !t.is_empty() && t.bytes().all(|b| b.is_ascii_digit())
}

/// Content tokens (>= 3 chars or any number, stop words removed, possessives stripped, lightly stemmed).
pub fn tokenize_with(text: &str, do_stem: bool, keep_numbers: bool) -> Vec<String> {
    let folded = fold(text);
    let cleaned = POSSESSIVE_RE.replace_all(&folded, "");
    let mut out = Vec::new();
    for m in TOKEN_RE.find_iter(&cleaned) {
        let t = m.as_str().trim_matches(['\'', '-']);
        if t.is_empty() || STOP_SET.contains(t) {
            continue;
        }
        let numeric = is_number(t);
        if t.len() < 3 && !(keep_numbers && numeric) {
            continue;
        }
        out.push(if do_stem && !numeric { stem(t) } else { t.to_string() });
    }
    out
}

pub fn tokenize(text: &str) -> Vec<String> {
    tokenize_with(text, true, true)
}

/// Python's `textutil.clip`-style shortening used for consolidation and conclusion texts: whitespace collapsed,
/// at most `limit` characters, an ellipsis when cut.
pub fn clip_collapsed(text: &str, limit: usize) -> String {
    let t = collapse_ws(text);
    if t.chars().count() <= limit {
        return t;
    }
    let head: String = t.chars().take(limit.saturating_sub(1)).collect();
    format!("{}…", py_rstrip(&head))
}
