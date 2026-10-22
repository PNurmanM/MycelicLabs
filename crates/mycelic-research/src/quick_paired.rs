//! Paired A/B of one variant against a base architecture on identical worlds (`quick_paired.py`): the same
//! (scale, seed) worlds for both, so seed-to-seed variance cancels, and the paired difference reported with a
//! bootstrap interval and an exact sign test.  Rows go to `artifacts/quick_<tag>.jsonl`.

use crate::analysis::{boot_ci, sign_test};
use crate::cli::{Args, list};
use crate::experiments::append_rows;
use crate::models;
use crate::py::fmt;
use crate::py::json::round;
use crate::py::value::{Row, f, fl};
use crate::runner::{self, Cal};
use anyhow::Result;
use rayon::prelude::*;
use serde_json::{Map, Value, json};
use std::time::Instant;

pub const METRICS: [(&str, &str, bool); 14] = [
    ("found_anywhere_in_register", "found", true),
    ("average_precision", "AP", true),
    ("recall_at_100", "R@100", true),
    ("rare_signal_recall", "rare recall", true),
    ("evidence_coverage_2links", "evidence cov.", true),
    ("false_discovery_rate", "FDR", false),
    ("decoy_acceptance_all", "decoy acc.", false),
    ("independent_evidence_accuracy", "indep. acc.", true),
    ("lineage_accuracy", "lineage", true),
    ("contradiction_f1", "contra F1", true),
    ("compute_units", "compute", false),
    ("inference_calls", "calls", false),
    ("raw_text_exposure_fraction", "raw text out", false),
    ("claim_exposure_fraction", "claims out", false),
];

/// `run_pair`: base and variant on the same worlds; with `ranker_ab` the ranker is off for the base side and
/// forced on for the variant side.
#[allow(clippy::too_many_arguments)]
pub fn run_pair(
    cal: &Cal,
    base: &str,
    variant: &str,
    cfg_over: Option<&Map<String, Value>>,
    scale: usize,
    seeds: &[u64],
    tag: &str,
    alloc_name: &str,
    ranker_ab: bool,
) -> Result<Vec<Row>> {
    let alloc = models::allocation_default(alloc_name);
    let blocks: Vec<Vec<Row>> = seeds
        .par_iter()
        .map(|&seed| {
            let w = runner::build_world_default(scale, seed);
            let mut out = Vec::new();
            for (side, (label, arch, over)) in
                [(base, base, None), (variant, variant, cfg_over)].into_iter().enumerate()
            {
                let t1 = Instant::now();
                let mut c = cal.clone();
                if ranker_ab {
                    // the variant may be the base itself, so switch on the SIDE, not the label
                    c.force_ranker = Some(side == 1);
                }
                let res = match over {
                    Some(o) if runner::arch_spec(arch).is_some() => {
                        runner::run_arch(&c, arch, &w, &alloc, seed, Some(o))
                    }
                    _ => runner::run_arch(&c, arch, &w, &alloc, seed, None),
                };
                let met = runner::evaluate(&w, &res.res);
                let mut r = Row::new();
                r.insert("arch".into(), json!(label));
                r.insert("variant_of".into(), json!(arch));
                r.insert("cfg_over".into(), over.map_or(Value::Null, |o| Value::Object(o.clone())));
                r.insert("scale".into(), json!(scale));
                r.insert("seed".into(), json!(seed));
                r.insert("tag".into(), json!(tag));
                let ranker = match c.force_ranker {
                    Some(true) => "forced-on",
                    Some(false) => "forced-off",
                    None => "cal",
                };
                r.insert("ranker".into(), json!(ranker));
                r.insert(
                    "ranker_l2".into(),
                    c.d.get("ranker").and_then(|x| x.get("l2")).cloned().unwrap_or(Value::Null),
                );
                r.insert("runtime_s".into(), fl(round(t1.elapsed().as_secs_f64(), 1)));
                for (k, v) in &met {
                    r.insert((*k).into(), fl(*v));
                }
                println!(
                    "  seed {seed} {label:24} found={} AP={} cov={} rare={} cu={}",
                    fmt::f(met["found_anywhere_in_register"], ".3f"),
                    fmt::f(met["average_precision"], ".4f"),
                    fmt::f(met["evidence_coverage_2links"], ".3f"),
                    fmt::f(met["rare_signal_recall"], ".3f"),
                    fmt::f(met["compute_units"], ".2e"),
                );
                out.push(r);
            }
            out
        })
        .collect();
    let rows: Vec<Row> = blocks.into_iter().flatten().collect();
    append_rows(&format!("quick_{tag}.jsonl"), &rows)?;
    Ok(rows)
}

/// `report(rows, base, variant)`: the paired table, one line per metric.
pub fn report(rows: &[Row], base: &str, variant: &str) -> String {
    let side = |name: &str| -> Vec<(i64, &Row)> {
        let mut m: Vec<(i64, &Row)> = Vec::new();
        for r in rows.iter().filter(|r| r["arch"] == json!(name)) {
            let s = r["seed"].as_i64().unwrap();
            match m.iter_mut().find(|e| e.0 == s) {
                Some(e) => e.1 = r,
                None => m.push((s, r)),
            }
        }
        m
    };
    let kb = side(base);
    let kv = side(variant);
    let mut seeds: Vec<i64> = kb.iter().map(|e| e.0).filter(|s| kv.iter().any(|e| e.0 == *s)).collect();
    seeds.sort();
    fn get<'a>(m: &[(i64, &'a Row)], s: i64) -> &'a Row {
        m.iter().find(|e| e.0 == s).unwrap().1
    }
    let mut lines = vec![
        format!("**{variant} vs {base}** — {} paired seeds", seeds.len()),
        String::new(),
        "| metric | base | variant | Δ | 95% CI | better on | sign p | verdict |".to_string(),
        "|---|---:|---:|---:|---|---:|---:|---|".to_string(),
    ];
    for (key, label, higher_better) in METRICS {
        let b: Vec<f64> = seeds.iter().map(|&s| f(get(&kb, s), key)).collect();
        let v: Vec<f64> = seeds.iter().map(|&s| f(get(&kv, s), key)).collect();
        let d: Vec<f64> = v.iter().zip(&b).map(|(x, y)| x - y).collect();
        let (m, lo, hi) = boot_ci(&d);
        let (k, n, p) = sign_test(&d);
        let wins = if higher_better { k } else { n - k };
        let verdict = if m.abs() < 1e-12 {
            "same"
        } else if (lo > 0.0 && higher_better) || (hi < 0.0 && !higher_better) {
            "**better**"
        } else if (hi < 0.0 && higher_better) || (lo > 0.0 && !higher_better) {
            "**worse**"
        } else {
            "inside noise"
        };
        let spec = if key == "compute_units" || key == "inference_calls" { ".2e" } else { ".4f" };
        let sg = |x: f64| if x >= 0.0 { "+" } else { "" };
        lines.push(format!(
            "| {label} | {} | {} | {}{} | [{}{}, {}{}] | {wins}/{n} | {} | {verdict} |",
            fmt::f(crate::np::math::mean(&b), spec),
            fmt::f(crate::np::math::mean(&v), spec),
            sg(m),
            fmt::f(m, spec),
            sg(lo),
            fmt::f(lo, spec),
            sg(hi),
            fmt::f(hi, spec),
            fmt::f(p, ".3f"),
        ));
    }
    lines.join("\n")
}

/// `python3 -m research.mycelic.quick_paired --tag T [--base B] [--variant-arch V] [--cfg JSON] [--scale N]
/// [--seeds 0,1,2] [--alloc A] [--ranker cal|off|on|ab]`
pub fn main(args: &[String]) -> Result<()> {
    let a = Args::parse(args)?;
    let tag = a.required("tag")?;
    let base = a.get("base").unwrap_or("H_mycelic_full");
    let mut cal = Cal::load();
    let ranker = a.get("ranker").unwrap_or("cal");
    cal.force_ranker = match ranker {
        "off" => Some(false),
        "on" => Some(true),
        _ => None,
    };
    let mut over: Option<Map<String, Value>> = a.get("cfg").map(serde_json::from_str).transpose()?;
    let variant = a.get("variant_arch").unwrap_or(base);
    let label = a.get("variant_arch").map(str::to_string).unwrap_or_else(|| format!("{base}+{tag}"));
    if ranker == "ab" && over.is_none() {
        over = Some(Map::new()); // same config; the ranker is the only difference
    }
    let scale: usize = a.get("scale").unwrap_or("10000").parse()?;
    let seeds: Vec<u64> = list(a.get("seeds").unwrap_or("0,1,2"))?;
    let mut rows = run_pair(
        &cal,
        base,
        variant,
        over.as_ref(),
        scale,
        &seeds,
        tag,
        a.get("alloc").unwrap_or("back-loaded"),
        ranker == "ab",
    )?;
    // relabel the variant rows so the report can tell them apart
    for (i, r) in rows.iter_mut().enumerate() {
        if i % 2 == 1 {
            r.insert("arch".into(), json!(label));
        }
    }
    println!();
    println!("{}", report(&rows, base, &label));
    Ok(())
}
