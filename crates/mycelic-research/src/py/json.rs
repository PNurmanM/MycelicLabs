//! JSON text exactly as the Python research code wrote it.
//!
//! The artifacts were written with ``json.dumps(row)`` (one line per run) and ``json.dump(obj, fh, indent=n)``:
//! ``", "``/``": "`` separators (``","`` between items once indented), ``repr()`` floats (``1e-05``, ``1e+16``,
//! ``100.0``) and ``ensure_ascii=True``.  Writing the same bytes is what lets a rerun be diffed against the
//! committed rows.  ``round``/``fmt_fixed`` reproduce Python's ``round(x, n)`` and ``f"{x:.nf}"``: both round
//! the exact binary value half-to-even, as Rust's ``{:.n}`` formatting does.
//!
//! Adapted from ``crates/mycelic/src/pyjson.rs`` (the service's copy); the research crate does not depend on
//! the service.

use serde_json::Value;
use std::fmt::Write as _;

/// How to lay out the text; the constructors name the ``json.dumps`` call each one reproduces.
#[derive(Clone, Copy, Debug)]
pub struct Style {
    pub sort_keys: bool,
    pub indent: Option<usize>,
    pub item_sep: &'static str,
    pub key_sep: &'static str,
    pub ensure_ascii: bool,
}

impl Style {
    /// ``json.dumps(v, ensure_ascii=False)``
    pub const DEFAULT: Style =
        Style { sort_keys: false, indent: None, item_sep: ", ", key_sep: ": ", ensure_ascii: false };
    /// ``json.dumps(v, ensure_ascii=False, sort_keys=True)``: database columns
    pub const SORTED: Style = Style { sort_keys: true, ..Style::DEFAULT };
    /// ``json.dumps(v, ensure_ascii=False, indent=1)``: MCP tool results
    pub const INDENT1: Style = Style { indent: Some(1), item_sep: ",", ..Style::DEFAULT };
    /// ``json.dumps(v, ensure_ascii=False, separators=(",", ":"))``
    pub const COMPACT: Style = Style { item_sep: ",", key_sep: ":", ..Style::DEFAULT };
    /// ``json.dumps(v)``: ASCII only
    pub const ASCII: Style = Style { ensure_ascii: true, ..Style::DEFAULT };
    /// ``json.dump(v, fh, indent=n)`` (ASCII only)
    pub const fn indent(n: usize) -> Style {
        Style { sort_keys: false, indent: Some(n), item_sep: ",", key_sep: ": ", ensure_ascii: true }
    }
}

/// ``json.dumps(v, ensure_ascii=False)``
pub fn dumps(v: &Value) -> String {
    dumps_with(v, Style::DEFAULT)
}

/// ``json.dumps(v, ensure_ascii=False, sort_keys=True)``
pub fn dumps_sorted(v: &Value) -> String {
    dumps_with(v, Style::SORTED)
}

pub fn dumps_with(v: &Value, style: Style) -> String {
    let mut out = String::new();
    write_value(&mut out, v, style, 0);
    out
}

fn newline_indent(out: &mut String, style: Style, level: usize) {
    if let Some(n) = style.indent {
        out.push('\n');
        for _ in 0..n * level {
            out.push(' ');
        }
    }
}

fn write_value(out: &mut String, v: &Value, style: Style, level: usize) {
    match v {
        Value::Null => out.push_str("null"),
        Value::Bool(true) => out.push_str("true"),
        Value::Bool(false) => out.push_str("false"),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                let _ = write!(out, "{i}");
            } else if let Some(u) = n.as_u64() {
                let _ = write!(out, "{u}");
            } else {
                out.push_str(&float_json(n.as_f64().unwrap_or(f64::NAN)));
            }
        }
        Value::String(s) => write_str(out, s, style.ensure_ascii),
        Value::Array(items) => {
            if items.is_empty() {
                out.push_str("[]");
                return;
            }
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push_str(style.item_sep);
                }
                newline_indent(out, style, level + 1);
                write_value(out, item, style, level + 1);
            }
            newline_indent(out, style, level);
            out.push(']');
        }
        Value::Object(map) => {
            if map.is_empty() {
                out.push_str("{}");
                return;
            }
            out.push('{');
            let mut entries: Vec<(&String, &Value)> = map.iter().collect();
            if style.sort_keys {
                entries.sort_by(|a, b| a.0.cmp(b.0));
            }
            for (i, (k, item)) in entries.into_iter().enumerate() {
                if i > 0 {
                    out.push_str(style.item_sep);
                }
                newline_indent(out, style, level + 1);
                write_str(out, k, style.ensure_ascii);
                out.push_str(style.key_sep);
                write_value(out, item, style, level + 1);
            }
            newline_indent(out, style, level);
            out.push('}');
        }
    }
}

fn write_str(out: &mut String, s: &str, ensure_ascii: bool) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c if ensure_ascii && (c as u32) > 0x7e => {
                let mut buf = [0u16; 2];
                for unit in c.encode_utf16(&mut buf) {
                    let _ = write!(out, "\\u{unit:04x}");
                }
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

/// A float as ``json.dumps`` writes it (``repr`` for finite values, ``NaN``/``Infinity`` otherwise).
pub fn float_json(x: f64) -> String {
    if x.is_nan() {
        "NaN".to_string()
    } else if x.is_infinite() {
        if x > 0.0 { "Infinity".to_string() } else { "-Infinity".to_string() }
    } else {
        float_repr(x)
    }
}

/// Python's ``repr(float)``: the shortest digits that round-trip, positional between 1e-4 and 1e16, exponent
/// form (at least two exponent digits, explicit sign) outside, and always a ``.0`` on integral positional values.
pub fn float_repr(x: f64) -> String {
    let negative = x.is_sign_negative();
    let mut buf = ryu::Buffer::new();
    let text = buf.format_finite(x.abs());
    let (mantissa, exp) = match text.split_once(['e', 'E']) {
        Some((m, e)) => (m, e.parse::<i32>().unwrap_or(0)),
        None => (text, 0),
    };
    let (int_part, frac_part) = mantissa.split_once('.').unwrap_or((mantissa, ""));
    let mut digits: String = format!("{int_part}{frac_part}");
    let mut decpt = int_part.len() as i32 + exp;
    let leading = digits.len() - digits.trim_start_matches('0').len();
    digits.drain(..leading);
    decpt -= leading as i32;
    let trimmed = digits.trim_end_matches('0').len();
    digits.truncate(trimmed);
    let sign = if negative { "-" } else { "" };
    if digits.is_empty() {
        return format!("{sign}0.0");
    }
    let n = digits.len() as i32;
    if decpt <= -4 || decpt > 16 {
        let (first, rest) = digits.split_at(1);
        let e = decpt - 1;
        let mant = if rest.is_empty() { first.to_string() } else { format!("{first}.{rest}") };
        let esign = if e < 0 { '-' } else { '+' };
        format!("{sign}{mant}e{esign}{:02}", e.abs())
    } else if decpt <= 0 {
        format!("{sign}0.{}{digits}", "0".repeat((-decpt) as usize))
    } else if decpt < n {
        let (a, b) = digits.split_at(decpt as usize);
        format!("{sign}{a}.{b}")
    } else {
        format!("{sign}{digits}{}.0", "0".repeat((decpt - n) as usize))
    }
}

/// Python's ``round(x, ndigits)`` for a float.
pub fn round(x: f64, ndigits: usize) -> f64 {
    if !x.is_finite() {
        return x;
    }
    fmt_fixed(x, ndigits).parse().unwrap_or(x)
}

/// Python's ``f"{x:.{digits}f}"``.
pub fn fmt_fixed(x: f64, digits: usize) -> String {
    format!("{x:.digits$}")
}

/// Python's ``repr(str)``: single quotes unless the text holds a single quote and no double quote; escapes for
/// backslashes, the chosen quote, ``\n``/``\r``/``\t`` and other non-printable characters.  Error messages copied
/// from the Python service embed values this way (``unknown layer 'x'``).
pub fn repr_str(s: &str) -> String {
    let quote = if s.contains('\'') && !s.contains('"') { '"' } else { '\'' };
    let mut out = String::with_capacity(s.len() + 2);
    out.push(quote);
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            c if is_printable(c) => out.push(c),
            c => {
                let u = c as u32;
                if u <= 0xff {
                    let _ = write!(out, "\\x{u:02x}");
                } else if u <= 0xffff {
                    let _ = write!(out, "\\u{u:04x}");
                } else {
                    let _ = write!(out, "\\U{u:08x}");
                }
            }
        }
    }
    out.push(quote);
    out
}

/// Python's ``str.isprintable`` for one character, close enough for error messages: everything but control,
/// separator (other than the plain space) and unassigned/format code points.
fn is_printable(c: char) -> bool {
    if c == ' ' {
        return true;
    }
    if c.is_control() || c.is_whitespace() {
        return false;
    }
    !matches!(c as u32, 0xad | 0x600..=0x605 | 0x61c | 0x6dd | 0x70f | 0x200b..=0x200f | 0x202a..=0x202e | 0x2060..=0x2064
        | 0x2066..=0x206f | 0xfeff | 0xfff9..=0xfffb | 0xd800..=0xdfff | 0xe000..=0xf8ff)
}

/// A Python tuple of strings as ``repr`` prints it: ``('a', 'b')``; a one-element tuple keeps its comma.
pub fn repr_tuple(items: &[&str]) -> String {
    let inner: Vec<String> = items.iter().map(|s| repr_str(s)).collect();
    if inner.len() == 1 { format!("({},)", inner[0]) } else { format!("({})", inner.join(", ")) }
}

/// A Python list of strings as ``repr`` prints it: ``['a', 'b']``.
pub fn repr_list(items: &[&str]) -> String {
    let inner: Vec<String> = items.iter().map(|s| repr_str(s)).collect();
    format!("[{}]", inner.join(", "))
}

/// Python truthiness of a JSON value (``payload or {}``).
pub fn truthy(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().map(|f| f != 0.0).unwrap_or(true),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

/// Number of Unicode code points, which is what Python's ``len(str)`` counts.
pub fn char_len(s: &str) -> usize {
    s.chars().count()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repr_examples() {
        assert_eq!(float_repr(1e16), "1e+16");
        assert_eq!(float_repr(1e15), "1000000000000000.0");
        assert_eq!(float_repr(1e-5), "1e-05");
        assert_eq!(float_repr(0.0001), "0.0001");
        assert_eq!(float_repr(-0.0), "-0.0");
        assert_eq!(float_repr(1.5e-7), "1.5e-07");
        assert_eq!(repr_str("x"), "'x'");
        assert_eq!(repr_str("it's"), "\"it's\"");
        assert_eq!(repr_tuple(&["team", "org"]), "('team', 'org')");
    }
}
