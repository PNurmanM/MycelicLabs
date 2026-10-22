//! Score the direct model measurements and place them on the capability axis (`score_live.py`).
//!
//! Output: for each measured model, the five operator accuracies, plus the value of `q` whose simulated tier vector
//! best matches those accuracies (least squares over the measured components).  That `q` is the *measured anchor*;
//! every other position on the axis in this study remains an assumption.

use crate::live_tasks::{key_dir, read_json, write_json};
use crate::models::{HardShape, Tier, tier_from_q};
use crate::np::math;
use crate::py::json::{self as pyjson, Style};
use crate::py::value::{fl, py_str};
use crate::runner::art;
use anyhow::Result;
use serde_json::{Map, Value, json};
use std::collections::HashMap;
use std::path::Path;

/// (model, answer file in the artifacts directory), in `FILES` order.
pub const FILES: [(&str, &str); 3] = [
    ("claude-haiku-4.5", "live_answers_haiku.json"),
    ("claude-sonnet-5", "live_answers_sonnet.json"),
    ("claude-opus-5", "live_answers_opus.json"),
];

pub const MEASURED_FIELDS: [&str; 6] =
    ["extract_recall", "entity_fidelity", "causal_check", "temporal_check", "entity_check", "dedup_check"];

/// `getattr(tier, field)` for the measured fields.
pub fn tier_field(t: &Tier, field: &str) -> f64 {
    match field {
        "extract_recall" => t.extract_recall,
        "entity_fidelity" => t.entity_fidelity,
        "causal_check" => t.causal_check,
        "temporal_check" => t.temporal_check,
        "entity_check" => t.entity_check,
        "dedup_check" => t.dedup_check,
        other => panic!("not a measured tier field: {other:?}"),
    }
}

/// Python's `str.strip()`: `str.isspace` also counts the information separators U+001C..U+001F, which Rust's
/// `char::is_whitespace` does not.
pub(crate) fn py_strip(s: &str) -> &str {
    s.trim_matches(|c: char| c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c))
}

/// `str(v).strip().lower()` of an answer field.
fn norm(v: Option<&Value>) -> String {
    py_strip(&v.map(py_str).unwrap_or_default()).to_lowercase()
}

/// `int(v)` for a value `json.load` produced, None where Python raises.  Strings follow `int(str)`: surrounding
/// whitespace, an optional sign, ASCII digits with single underscores between them.
fn py_int(v: Option<&Value>) -> Option<i64> {
    match v? {
        Value::Bool(b) => Some(i64::from(*b)),
        Value::Number(n) => n.as_i64().or_else(|| n.as_f64().filter(|x| x.is_finite()).map(|x| x.trunc() as i64)),
        Value::String(s) => {
            let t = py_strip(s);
            let (neg, digits) = match t.strip_prefix('-') {
                Some(rest) => (true, rest),
                None => (false, t.strip_prefix('+').unwrap_or(t)),
            };
            let ok = !digits.is_empty()
                && !digits.starts_with('_')
                && !digits.ends_with('_')
                && !digits.contains("__")
                && digits.chars().all(|c| c.is_ascii_digit() || c == '_');
            if !ok {
                return None;
            }
            let n: i64 = digits.replace('_', "").parse().ok()?;
            Some(if neg { -n } else { n })
        }
        _ => None,
    }
}

/// `_index(items)`: answers by `str(id)`, skipping anything that is not an object.
fn index(items: Option<&Value>) -> HashMap<String, &Map<String, Value>> {
    let mut out = HashMap::new();
    for x in items.and_then(Value::as_array).into_iter().flatten() {
        if let Value::Object(o) = x {
            out.insert(o.get("id").map(py_str).unwrap_or_else(|| "None".into()), o);
        }
    }
    out
}

fn items<'a>(tasks: &'a Value, group: &str) -> &'a [Value] {
    tasks["tasks"][group].as_array().map(Vec::as_slice).unwrap_or_default()
}

fn id_of(item: &Value) -> String {
    py_str(&item["id"])
}

/// A present, non-empty answer object (`if not g` skips both a missing and an empty one).
fn answered<'a>(index: &HashMap<String, &'a Map<String, Value>>, id: &str) -> Option<&'a Map<String, Value>> {
    index.get(id).copied().filter(|g| !g.is_empty())
}

fn ratio(a: i64, b: i64) -> f64 {
    a as f64 / b.max(1) as f64
}

/// Balanced accuracy of a two-way answer (T2, T4) and the number of items.
fn balanced(tasks: &Value, ans: &Value, key: &Value, group: &str, positive: &str, negative: &str) -> (f64, i64) {
    let a = index(ans.get(group));
    let (mut tp, mut tn, mut pos, mut neg) = (0, 0, 0, 0);
    for it in items(tasks, group) {
        let id = id_of(it);
        let got = answered(&a, &id).map(|g| norm(g.get("answer"))).unwrap_or_default();
        if key[&id]["answer"] == positive {
            pos += 1;
            tp += i64::from(got == positive);
        } else {
            neg += 1;
            tn += i64::from(got == negative);
        }
    }
    (0.5 * (ratio(tp, pos) + ratio(tn, neg)), pos + neg)
}

/// `score_one(tasks, ans, key)`: the operator accuracies of one model's answers, in the Python's key order.
pub fn score_one(tasks: &Value, ans: &Value, key: &Value) -> Map<String, Value> {
    let mut out = Map::new();

    // T1 extraction: predicate recall and entity fidelity
    let a1 = index(ans.get("T1"));
    let (mut pr_ok, mut ent_ok, mut n1) = (0, 0, 0);
    for it in items(tasks, "T1") {
        let id = id_of(it);
        n1 += 1;
        let Some(g) = answered(&a1, &id) else { continue };
        let kk = &key[&id];
        if Value::String(norm(g.get("predicate"))) == kk["predicate"] {
            pr_ok += 1;
        }
        if norm(g.get("entity")) == py_str(&kk["entity"]).to_lowercase() {
            ent_ok += 1;
        }
    }
    out.insert("extract_recall".into(), fl(ratio(pr_ok, n1)));
    out.insert("entity_fidelity".into(), fl(ratio(ent_ok, n1)));
    out.insert("n_T1".into(), json!(n1));

    // T2 causal check (balanced accuracy)
    let (causal, n2) = balanced(tasks, ans, key, "T2", "yes", "no");
    out.insert("causal_check".into(), fl(causal));
    out.insert("n_T2".into(), json!(n2));

    // T3 temporal check (3-way, macro accuracy), classes in first-seen order
    let a3 = index(ans.get("T3"));
    let mut per: indexmap::IndexMap<String, [i64; 2]> = indexmap::IndexMap::new();
    for it in items(tasks, "T3") {
        let id = id_of(it);
        let got = answered(&a3, &id).map(|g| norm(g.get("answer"))).unwrap_or_default();
        let gold = py_str(&key[&id]["answer"]);
        let cell = per.entry(gold.clone()).or_insert([0, 0]);
        cell[1] += 1;
        cell[0] += i64::from(got == gold);
    }
    let per_class: Vec<f64> = per.values().map(|v| ratio(v[0], v[1])).collect();
    let temporal = if per.is_empty() { 0.0 } else { math::mean(&per_class) };
    out.insert("temporal_check".into(), fl(temporal));
    out.insert("n_T3".into(), json!(per.values().map(|v| v[1]).sum::<i64>()));

    // T4 entity linking (balanced accuracy)
    let (linking, n4) = balanced(tasks, ans, key, "T4", "same", "different");
    out.insert("entity_check".into(), fl(linking));
    out.insert("n_T4".into(), json!(n4));

    // T5 independence: exact-count accuracy; an unreadable answer counts as an error of the full gold count
    let a5 = index(ans.get("T5"));
    let (mut ok, mut n5) = (0, 0);
    let mut errs: Vec<f64> = Vec::new();
    for it in items(tasks, "T5") {
        let id = id_of(it);
        n5 += 1;
        let gold = py_int(Some(&key[&id]["answer"])).expect("integer gold count");
        // `int(g.get("answer"))` raises for a missing answer object as well as an unparseable answer
        match py_int(a5.get(&id).and_then(|g| g.get("answer"))) {
            None => errs.push(gold as f64),
            Some(v) => {
                ok += i64::from(v == gold);
                errs.push((v - gold).abs() as f64);
            }
        }
    }
    out.insert("dedup_check".into(), fl(ratio(ok, n5)));
    out.insert("dedup_mae".into(), fl(if errs.is_empty() { 0.0 } else { math::mean(&errs) }));
    out.insert("n_T5".into(), json!(n5));
    out
}

/// `fit_q(meas, shape)`: the q on a 199-point grid whose tier best matches the measured accuracies (least squares;
/// the first minimum wins).
pub fn fit_q(meas: &Map<String, Value>, shape: HardShape) -> f64 {
    let (mut best, mut best_q) = (1e9, 0.5);
    for q in math::linspace(0.01, 1.0, 199) {
        let t = tier_from_q(q, None, None, None, shape);
        let mut err = 0.0;
        for f in MEASURED_FIELDS {
            if let Some(m) = meas.get(f).and_then(Value::as_f64) {
                err += crate::py::pow(tier_field(&t, f) - m, 2.0);
            }
        }
        if err < best {
            best = err;
            best_q = q;
        }
    }
    best_q
}

/// `main()`: score every answer file present in `art_dir` against `art_dir/live_tasks.json` and
/// `key_dir/live_key.json`, write `out_dir/live_calibration.json` (indent 2) and return the result.
pub fn run(art_dir: &Path, key_dir: &Path, out_dir: &Path) -> Result<Value> {
    let tasks = read_json(&art_dir.join("live_tasks.json"))?;
    let key = read_json(&key_dir.join("live_key.json"))?;
    let counts: Map<String, Value> = tasks["tasks"]
        .as_object()
        .into_iter()
        .flatten()
        .map(|(k, v)| (k.clone(), json!(v.as_array().map_or(0, Vec::len))))
        .collect();
    let mut per = Map::new();
    for (model, file) in FILES {
        let path = art_dir.join(file);
        if !path.exists() {
            continue;
        }
        let ans = read_json(&path)?;
        let mut s = score_one(&tasks, &ans, &key);
        let q = fit_q(&s, HardShape::Convex);
        s.insert("fitted_q".into(), fl(q));
        let t = tier_from_q(q, None, None, None, HardShape::Convex);
        let simulated: Map<String, Value> =
            MEASURED_FIELDS.iter().map(|&f| (f.to_string(), fl(pyjson::round(tier_field(&t, f), 4)))).collect();
        s.insert("simulated_at_fitted_q".into(), Value::Object(simulated));
        per.insert(model.to_string(), Value::Object(s));
    }
    let res = json!({"task_counts": counts, "measurements": per});
    write_json(out_dir, "live_calibration.json", &res, 2)?;
    Ok(res)
}

/// `python -m research.mycelic.score_live`
pub fn main(_args: &[String]) -> Result<()> {
    let art_dir = art();
    let res = run(&art_dir, &key_dir(), &art_dir)?;
    println!("{}", pyjson::dumps_with(&res, Style::indent(2)));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn int_parsing_follows_python() {
        let s = |x: &str| py_int(Some(&json!(x)));
        assert_eq!(s(" 3 "), Some(3));
        assert_eq!(s("+4"), Some(4));
        assert_eq!(s("-1_0"), Some(-10));
        assert_eq!(s("3.0"), None);
        assert_eq!(s("1__0"), None);
        assert_eq!(s(""), None);
        assert_eq!(py_int(Some(&json!(2.9))), Some(2));
        assert_eq!(py_int(Some(&json!(true))), Some(1));
        assert_eq!(py_int(Some(&Value::Null)), None);
        assert_eq!(py_int(None), None);
        assert_eq!(py_strip("\u{1c} x\t"), "x");
    }
}
