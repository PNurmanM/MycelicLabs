//! Python's format-spec mini-language, for the f-strings the document generators are written in.
//!
//! `f(x, ".3f")` is `f"{x:.3f}"`, `i(n, ",")` is `f"{n:,}"`, and so on.  Fixed and exponent forms round the
//! exact binary value half-to-even exactly as CPython's `format()` does (Rust's `{:.n}`/`{:.ne}` formatting does
//! the same); `g` follows CPython's rule of choosing the form from the exponent after rounding; `%` multiplies by
//! 100 in floating point first, as CPython does.  Only the parts of the grammar the generators use are
//! implemented: `[[fill]align][sign][0][width][,][.precision][type]` with types `f % e g d` and none.

use super::json::float_repr;

#[derive(Debug, Default, Clone)]
struct Spec {
    fill: char,
    align: Option<char>,
    sign: Option<char>,
    zero: bool,
    width: usize,
    comma: bool,
    precision: Option<usize>,
    ty: Option<char>,
}

fn parse(spec: &str) -> Spec {
    let chars: Vec<char> = spec.chars().collect();
    let mut s = Spec { fill: ' ', ..Default::default() };
    let mut i = 0;
    let is_align = |c: char| matches!(c, '<' | '>' | '^' | '=');
    if chars.len() >= 2 && is_align(chars[1]) {
        s.fill = chars[0];
        s.align = Some(chars[1]);
        i = 2;
    } else if !chars.is_empty() && is_align(chars[0]) {
        s.align = Some(chars[0]);
        i = 1;
    }
    if i < chars.len() && matches!(chars[i], '+' | '-' | ' ') {
        s.sign = Some(chars[i]);
        i += 1;
    }
    if i < chars.len() && chars[i] == '0' {
        s.zero = true;
        i += 1;
    }
    let start = i;
    while i < chars.len() && chars[i].is_ascii_digit() {
        i += 1;
    }
    if i > start {
        s.width = chars[start..i].iter().collect::<String>().parse().unwrap();
    }
    if i < chars.len() && chars[i] == ',' {
        s.comma = true;
        i += 1;
    }
    if i < chars.len() && chars[i] == '.' {
        i += 1;
        let start = i;
        while i < chars.len() && chars[i].is_ascii_digit() {
            i += 1;
        }
        s.precision = Some(chars[start..i].iter().collect::<String>().parse().unwrap());
    }
    if i < chars.len() {
        s.ty = Some(chars[i]);
        i += 1;
    }
    assert!(i == chars.len(), "unsupported format spec {spec:?}");
    s
}

/// Insert thousands separators into the integer part of a digit string (no sign).
fn group(digits: &str) -> String {
    let (int_part, rest) = match digits.find(|c: char| !c.is_ascii_digit()) {
        Some(p) => digits.split_at(p),
        None => (digits, ""),
    };
    let n = int_part.len();
    let mut out = String::with_capacity(n + n / 3 + rest.len());
    for (k, c) in int_part.chars().enumerate() {
        if k > 0 && (n - k) % 3 == 0 {
            out.push(',');
        }
        out.push(c);
    }
    out.push_str(rest);
    out
}

fn pad(body: String, negative: bool, spec: &Spec, numeric: bool) -> String {
    let sign = if negative {
        "-"
    } else {
        match spec.sign {
            Some('+') => "+",
            Some(' ') => " ",
            _ => "",
        }
    };
    let len = sign.chars().count() + body.chars().count();
    if len >= spec.width {
        return format!("{sign}{body}");
    }
    let n = spec.width - len;
    let (fill, align) = if spec.zero && spec.align.is_none() {
        ('0', '=')
    } else {
        (spec.fill, spec.align.unwrap_or(if numeric { '>' } else { '<' }))
    };
    let fills = |k: usize| std::iter::repeat_n(fill, k).collect::<String>();
    match align {
        '<' => format!("{sign}{body}{}", fills(n)),
        '^' => format!("{}{sign}{body}{}", fills(n / 2), fills(n - n / 2)),
        '=' => format!("{sign}{}{body}", fills(n)),
        _ => format!("{}{sign}{body}", fills(n)),
    }
}

/// Mantissa and exponent of `{:.pe}` (Rust prints `1.23e4`), as (mantissa text, exponent).
fn exp_parts(x: f64, p: usize) -> (String, i32) {
    let s = format!("{x:.p$e}");
    let (m, e) = s.split_once('e').expect("exponent form");
    (m.to_string(), e.parse().expect("exponent"))
}

fn exp_form(x: f64, p: usize) -> String {
    let (m, e) = exp_parts(x, p);
    let esign = if e < 0 { '-' } else { '+' };
    format!("{m}e{esign}{:02}", e.abs())
}

fn strip_zeros(s: &str) -> String {
    if s.contains('.') { s.trim_end_matches('0').trim_end_matches('.').to_string() } else { s.to_string() }
}

/// `format(x, 'g')`-style formatting of a non-negative finite value with `p` significant digits.
fn general(x: f64, p: usize, keep_point: bool) -> String {
    let p = p.max(1);
    if x == 0.0 {
        return if keep_point { "0.0".to_string() } else { "0".to_string() };
    }
    let (_, e) = exp_parts(x, p - 1);
    if -4 <= e && (e as i64) < p as i64 {
        let decimals = (p as i64 - 1 - e as i64) as usize;
        let s = strip_zeros(&format!("{x:.decimals$}"));
        if keep_point && !s.contains('.') { format!("{s}.0") } else { s }
    } else {
        let (m, e) = exp_parts(x, p - 1);
        let m = strip_zeros(&m);
        let esign = if e < 0 { '-' } else { '+' };
        format!("{m}e{esign}{:02}", e.abs())
    }
}

/// `format(x, spec)` for a Python float.
pub fn f(x: f64, spec: &str) -> String {
    let s = parse(spec);
    let negative = x.is_sign_negative() && !x.is_nan();
    let a = x.abs();
    let mut body = if x.is_nan() {
        "nan".to_string()
    } else if x.is_infinite() {
        "inf".to_string()
    } else {
        match s.ty {
            Some('f') | Some('F') => {
                let p = s.precision.unwrap_or(6);
                format!("{a:.p$}")
            }
            Some('%') => {
                let p = s.precision.unwrap_or(6);
                let v = a * 100.0;
                format!("{v:.p$}")
            }
            Some('e') => exp_form(a, s.precision.unwrap_or(6)),
            Some('g') => general(a, s.precision.unwrap_or(6), false),
            None => match s.precision {
                // no type with a precision: like 'g' but never drops to a bare integer
                Some(p) => general(a, p, true),
                None => float_repr(a),
            },
            Some(t) => panic!("unsupported float format type {t:?}"),
        }
    };
    if s.ty == Some('%') && x.is_nan() {
        body = "nan".to_string();
    }
    if s.comma {
        body = group(&body);
    }
    if s.ty == Some('%') {
        body.push('%');
    }
    pad(body, negative, &s, true)
}

/// `format(n, spec)` for a Python int (`d`, `,`, width).  A float type formats the int as a float.
pub fn i(n: i64, spec: &str) -> String {
    let s = parse(spec);
    match s.ty {
        Some('f') | Some('F') | Some('%') | Some('e') | Some('g') => return f(n as f64, spec),
        Some('d') | None => {}
        Some(t) => panic!("unsupported int format type {t:?}"),
    }
    let mut body = n.unsigned_abs().to_string();
    if s.comma {
        body = group(&body);
    }
    pad(body, n < 0, &s, true)
}

/// `format(text, spec)` for a str (width/alignment only).
pub fn s(text: &str, spec: &str) -> String {
    let sp = parse(spec);
    let mut text = text.to_string();
    if let Some(p) = sp.precision {
        text = text.chars().take(p).collect();
    }
    let len = text.chars().count();
    if len >= sp.width {
        return text;
    }
    let n = sp.width - len;
    let fills = |k: usize| std::iter::repeat_n(sp.fill, k).collect::<String>();
    match sp.align.unwrap_or('<') {
        '>' => format!("{}{text}", fills(n)),
        '^' => format!("{}{text}{}", fills(n / 2), fills(n - n / 2)),
        _ => format!("{text}{}", fills(n)),
    }
}

/// `str(x)` / `f"{x}"` for a float: its repr.
pub fn repr(x: f64) -> String {
    if x.is_nan() {
        "nan".to_string()
    } else if x.is_infinite() {
        if x > 0.0 { "inf".to_string() } else { "-inf".to_string() }
    } else {
        float_repr(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn python_examples() {
        assert_eq!(f(0.125, ".2f"), "0.12");
        assert_eq!(f(-0.0001, "+.3f"), "-0.000");
        assert_eq!(f(0.0, "+.3f"), "+0.000");
        assert_eq!(f(12345.678, ".2e"), "1.23e+04");
        assert_eq!(f(0.555, ".0%"), "56%");
        assert_eq!(f(0.25, ".1%"), "25.0%");
        assert_eq!(f(1234.5, ".3g"), "1.23e+03");
        assert_eq!(f(0.000123, ".3g"), "0.000123");
        assert_eq!(f(2.0, "g"), "2");
        assert_eq!(f(0.8, "g"), "0.8");
        assert_eq!(f(1234567.0, ",.0f"), "1,234,567");
        assert_eq!(i(1234567, ","), "1,234,567");
        assert_eq!(i(7, "3d"), "  7");
        assert_eq!(f(f64::NAN, ".1%"), "nan%");
        assert_eq!(f(-1.5e-5, "+.2e"), "-1.50e-05");
        assert_eq!(s("ab", "6s"), "ab    ");
    }
}
