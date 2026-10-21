//! Aggregation, uncertainty and significance over the raw JSONL rows (`analysis.py`).
//!
//! Raw per-run metrics are never overwritten; every aggregate carries a bootstrap CI; every architecture
//! comparison is PAIRED on (scale, seed); with few seeds an exact sign test is reported rather than a t-test.

use crate::np::Generator;
use crate::np::math;
use crate::py::fmt;
use crate::py::value::{Row, is_float, num, py_cmp, py_eq, py_str};
use crate::runner::art;
use serde_json::Value;
use std::path::Path;

/// Keep the first row per key (default (arch, scale, seed)); rows missing a key field are kept.
pub fn dedupe(rows: Vec<Row>, key: &[&str]) -> Vec<Row> {
    let mut seen: Vec<Vec<Value>> = Vec::new();
    let mut out = Vec::new();
    for r in rows {
        let k: Option<Vec<Value>> = key.iter().map(|x| r.get(*x).cloned()).collect();
        match k {
            None => out.push(r),
            Some(k) => {
                if seen.iter().any(|s| s.iter().zip(&k).all(|(a, b)| py_eq(a, b))) {
                    continue;
                }
                seen.push(k);
                out.push(r);
            }
        }
    }
    out
}

/// `dedupe(rows)` on (arch, scale, seed).
pub fn dedupe_default(rows: Vec<Row>) -> Vec<Row> {
    dedupe(rows, &["arch", "scale", "seed"])
}

/// Every row of a JSONL file (missing file -> no rows).
pub fn load_path(p: &Path) -> Vec<Row> {
    let Ok(text) = std::fs::read_to_string(p) else { return Vec::new() };
    text.lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .map(|l| match serde_json::from_str::<Value>(l).expect("jsonl row") {
            Value::Object(m) => m,
            other => panic!("not a row: {other}"),
        })
        .collect()
}

/// `load(fname)` from the artifacts directory.
pub fn load(fname: &str) -> Vec<Row> {
    load_path(&art().join(fname))
}

/// `boot_ci(x, n=4000, alpha=0.05, seed=0)`: (mean, lo, hi) of a percentile bootstrap.
pub fn boot_ci(x: &[f64]) -> (f64, f64, f64) {
    boot_ci_with(x, 4000, 0.05, 0)
}

pub fn boot_ci_with(x: &[f64], n: usize, alpha: f64, seed: u64) -> (f64, f64, f64) {
    if x.is_empty() {
        return (0.0, 0.0, 0.0);
    }
    if x.len() == 1 {
        return (x[0], x[0], x[0]);
    }
    let mut rng = Generator::new(seed);
    let k = x.len();
    let idx = rng.integers_n(0, k as i64, n * k);
    let means: Vec<f64> = idx
        .chunks(k)
        .map(|row| {
            let vals: Vec<f64> = row.iter().map(|&i| x[i as usize]).collect();
            math::mean(&vals)
        })
        .collect();
    (
        math::mean(x),
        math::percentile(&means, 100.0 * alpha / 2.0),
        math::percentile(&means, 100.0 * (1.0 - alpha / 2.0)),
    )
}

/// Exact two-sided sign test: (n positive, n nonzero, p).
pub fn sign_test(diffs: &[f64]) -> (usize, usize, f64) {
    let d: Vec<f64> = diffs.iter().copied().filter(|x| x.abs() > 1e-12).collect();
    let n = d.len();
    if n == 0 {
        return (0, 0, 1.0);
    }
    let k = d.iter().filter(|&&x| x > 0.0).count();
    let comb = |n: usize, r: usize| -> u128 {
        let mut c: u128 = 1;
        for i in 0..r {
            c = c * (n - i) as u128 / (i + 1) as u128;
        }
        c
    };
    let tail: u128 = (0..=k.min(n - k)).map(|i| comb(n, i)).sum();
    // Python: 2.0 * tail / (2 ** n), float arithmetic on exact integers
    let p = 1.0f64.min(2.0 * tail as f64 / (2.0f64).powi(n as i32));
    (k, n, p)
}

/// The result of a paired comparison (`paired`), `n == 0` when no key is shared.
#[derive(Clone, Debug, Default)]
pub struct Paired {
    pub n: usize,
    pub mean_a: f64,
    pub mean_b: f64,
    pub mean_diff: f64,
    pub ci_lo: f64,
    pub ci_hi: f64,
    pub wins: usize,
    pub n_nonzero: usize,
    pub sign_p: f64,
}

/// `paired(rows, a, b, metric, group, key)`: a minus b on the keys both have, sorted by key.
pub fn paired_by(rows: &[Row], a: &Value, b: &Value, metric: &str, group: &str, key: &[&str]) -> Paired {
    let side = |which: &Value| -> Vec<(Vec<Value>, f64)> {
        let mut m: Vec<(Vec<Value>, f64)> = Vec::new();
        for r in rows {
            if !r.get(group).is_some_and(|g| py_eq(g, which)) {
                continue;
            }
            let k: Vec<Value> = key.iter().map(|x| r[*x].clone()).collect();
            let v = num(&r[metric]).expect("metric");
            // dict comprehension: a later row with the same key replaces the value
            match m.iter_mut().find(|e| e.0.iter().zip(&k).all(|(p, q)| py_eq(p, q))) {
                Some(e) => e.1 = v,
                None => m.push((k, v)),
            }
        }
        m
    };
    let ka = side(a);
    let kb = side(b);
    let mut common: Vec<(Vec<Value>, f64, f64)> = ka
        .iter()
        .filter_map(|(k, va)| {
            kb.iter().find(|(k2, _)| k2.iter().zip(k).all(|(p, q)| py_eq(p, q))).map(|(_, vb)| (k.clone(), *va, *vb))
        })
        .collect();
    if common.is_empty() {
        return Paired::default();
    }
    common.sort_by(|x, y| py_cmp(&Value::Array(x.0.clone()), &Value::Array(y.0.clone())));
    let da: Vec<f64> = common.iter().map(|c| c.1).collect();
    let db: Vec<f64> = common.iter().map(|c| c.2).collect();
    let diff: Vec<f64> = da.iter().zip(&db).map(|(x, y)| x - y).collect();
    let (m, lo, hi) = boot_ci(&diff);
    let (k, n, p) = sign_test(&diff);
    Paired {
        n: common.len(),
        mean_a: math::mean(&da),
        mean_b: math::mean(&db),
        mean_diff: m,
        ci_lo: lo,
        ci_hi: hi,
        wins: k,
        n_nonzero: n,
        sign_p: p,
    }
}

/// `paired(rows, a, b, metric)` on the arch column, keyed by (scale, seed).
pub fn paired(rows: &[Row], a: &str, b: &str, metric: &str) -> Paired {
    paired_by(rows, &Value::from(a), &Value::from(b), metric, "arch", &["scale", "seed"])
}

/// `agg(rows, metrics, by)`: one row per group (sorted by the group key) with the mean and bootstrap CI of every
/// metric (`m`, `m_lo`, `m_hi`) and `n_runs`.
pub fn agg(rows: &[Row], metrics: &[&str], by: &[&str]) -> Vec<Row> {
    let mut keys: Vec<Vec<Value>> = Vec::new();
    for r in rows {
        let k: Vec<Value> = by.iter().map(|b| r[*b].clone()).collect();
        if !keys.iter().any(|x| x.iter().zip(&k).all(|(p, q)| py_eq(p, q) && p.is_string() == q.is_string())) {
            keys.push(k);
        }
    }
    keys.sort_by(|x, y| py_cmp(&Value::Array(x.clone()), &Value::Array(y.clone())));
    let mut out = Vec::new();
    for k in keys {
        let sub: Vec<&Row> = rows.iter().filter(|r| by.iter().zip(&k).all(|(b, v)| py_eq(&r[*b], v))).collect();
        let mut d = Row::new();
        for (b, v) in by.iter().zip(&k) {
            d.insert(b.to_string(), v.clone());
        }
        d.insert("n_runs".into(), Value::from(sub.len()));
        for m in metrics {
            let vals: Vec<f64> = sub.iter().filter_map(|r| r.get(*m)).map(|v| num(v).expect("metric")).collect();
            let (mean, lo, hi) = boot_ci(&vals);
            d.insert(m.to_string(), crate::py::value::fl(mean));
            d.insert(format!("{m}_lo"), crate::py::value::fl(lo));
            d.insert(format!("{m}_hi"), crate::py::value::fl(hi));
        }
        out.push(d);
    }
    out
}

/// A markdown table (`md_table`): `cols = [(key, header, decimals)]`, optional descending sort.
pub fn md_table(
    rows: &[Row],
    cols: &[(&str, &str, usize)],
    sort_by: Option<&str>,
    reverse: bool,
    ci_for: &[&str],
) -> String {
    let mut rs: Vec<&Row> = rows.iter().collect();
    if let Some(s) = sort_by {
        let zero = Value::from(0);
        rs.sort_by(|a, b| {
            let o = py_cmp(a.get(s).unwrap_or(&zero), b.get(s).unwrap_or(&zero));
            if reverse { o.reverse() } else { o }
        });
    }
    let head = format!("| {} |", cols.iter().map(|c| c.1).collect::<Vec<_>>().join(" | "));
    let sep =
        format!("|{}|", (0..cols.len()).map(|i| if i == 0 { "---" } else { "---:" }).collect::<Vec<_>>().join("|"));
    let mut lines = vec![head, sep];
    for r in rs {
        let mut cells = Vec::new();
        for &(k, _, dec) in cols {
            let v = r.get(k).cloned().unwrap_or(Value::String(String::new()));
            let s = if is_float(&v) {
                let x = v.as_f64().unwrap();
                let mut s = if x.abs() < 1e5 { fmt::f(x, &format!(".{dec}f")) } else { fmt::f(x, ".2e") };
                if ci_for.contains(&k) && r.contains_key(&format!("{k}_lo")) {
                    let lo = r[&format!("{k}_lo")].as_f64().unwrap();
                    let hi = r[&format!("{k}_hi")].as_f64().unwrap();
                    s += &format!(
                        " <sub>[{}, {}]</sub>",
                        fmt::f(lo, &format!(".{dec}f")),
                        fmt::f(hi, &format!(".{dec}f"))
                    );
                }
                s
            } else {
                py_str(&v)
            };
            cells.push(s);
        }
        lines.push(format!("| {} |", cells.join(" | ")));
    }
    lines.join("\n")
}

pub const HEADLINE_COLS: [(&str, &str, usize); 15] = [
    ("arch", "architecture", 0),
    ("average_precision", "AP", 4),
    ("found_anywhere_in_register", "found", 3),
    ("recall_at_100", "R@100", 3),
    ("rare_signal_recall", "rare R", 3),
    ("false_discovery_rate", "FDR", 3),
    ("decoy_acceptance_all", "decoy acc", 3),
    ("independent_evidence_accuracy", "indep acc", 3),
    ("lineage_accuracy", "lineage", 3),
    ("contradiction_f1", "contra F1", 3),
    ("information_loss", "info loss", 3),
    ("compute_units", "compute", 2),
    ("tokens_total", "tokens", 2),
    ("wall_seconds", "wall s", 1),
    ("privacy_exposure_fraction", "privacy", 4),
];

/// `summary(fname)`: the headline table of one artifact file, by architecture.
pub fn summary(fname: &str) -> String {
    let rows = load(fname);
    if rows.is_empty() {
        return format!("(no rows in {fname})");
    }
    let metrics: Vec<&str> = HEADLINE_COLS[1..].iter().map(|c| c.0).collect();
    let a = agg(&rows, &metrics, &["arch"]);
    let mut cols: Vec<(&str, &str, usize)> = vec![("arch", "arch", 0)];
    cols.extend(metrics.iter().map(|m| (*m, *m, 4)));
    md_table(&a, &cols, Some("average_precision"), true, &[])
}
