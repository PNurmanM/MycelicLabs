//! Multi-arm paired runs on identical worlds, each arm with its own ranker (`arm_paired.py`).  An arm is
//! `label|architecture|ranker|HierConfig overrides`; the ranker is `none` (the hand-set logistic), `cal` (what
//! calibration.json holds, with its per-architecture adoption) or the path of a calibration file whose ranker is
//! forced on.  Rows go to `artifacts/quick_<tag>.jsonl`; every arm is reported against the first.

use crate::cli::{Args, list};
use crate::experiments::append_rows;
use crate::models;
use crate::py::fmt;
use crate::py::json::round;
use crate::py::value::{Row, fl};
use crate::quick_paired::report;
use crate::runner::{self, Cal, art};
use anyhow::{Result, bail};
use rayon::prelude::*;
use serde_json::{Map, Value, json};
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Clone, Debug)]
pub struct Arm {
    pub label: String,
    pub arch: String,
    pub ranker: String,
    pub cfg: Option<Map<String, Value>>,
}

pub fn parse_arm(spec: &str) -> Result<Arm> {
    let parts: Vec<&str> = spec.splitn(4, '|').collect();
    if parts.len() < 3 {
        bail!("arm needs label|arch|ranker[|cfg-json]: {spec:?}");
    }
    let cfg = match parts.get(3).map(|s| s.trim()) {
        Some(s) if !s.is_empty() => Some(serde_json::from_str::<Map<String, Value>>(s)?),
        _ => None,
    };
    let arch = parts[1].trim();
    if runner::arch_spec(arch).is_none() {
        bail!("unknown architecture {arch:?}");
    }
    Ok(Arm { label: parts[0].trim().into(), arch: arch.into(), ranker: parts[2].trim().into(), cfg })
}

/// The calibration an arm runs with (`_set_ranker`).
fn arm_cal(base: &Cal, spec: &str) -> Result<Cal> {
    let mut c = base.clone();
    match spec {
        "none" => c.force_ranker = Some(false),
        "cal" => c.force_ranker = None,
        path => {
            let p = Path::new(path);
            let mut full: PathBuf = if p.is_absolute() { p.to_path_buf() } else { art().parent().unwrap().join(p) };
            if !full.exists() {
                full = art().join(p.file_name().unwrap_or_default());
            }
            let d: Value = serde_json::from_str(&std::fs::read_to_string(&full)?)?;
            c.d.insert("ranker".into(), d["ranker"].clone());
            c.force_ranker = Some(true);
        }
    }
    Ok(c)
}

pub fn run_arms(cal: &Cal, arms: &[Arm], scale: usize, seeds: &[u64], tag: &str, alloc_name: &str) -> Result<Vec<Row>> {
    let alloc = models::allocation_default(alloc_name);
    let cals: Vec<Cal> = arms.iter().map(|a| arm_cal(cal, &a.ranker)).collect::<Result<_>>()?;
    let blocks: Vec<Vec<Row>> = seeds
        .par_iter()
        .map(|&seed| {
            let w = runner::build_world_default(scale, seed);
            arms.iter()
                .zip(&cals)
                .map(|(arm, c)| {
                    let t1 = Instant::now();
                    let over = arm.cfg.as_ref().filter(|m| !m.is_empty());
                    let res = runner::run_arch(c, &arm.arch, &w, &alloc, seed, over);
                    let met = runner::evaluate(&w, &res.res);
                    let mut r = Row::new();
                    r.insert("arch".into(), json!(arm.label));
                    r.insert("variant_of".into(), json!(arm.arch));
                    r.insert("cfg_over".into(), arm.cfg.clone().map_or(Value::Null, Value::Object));
                    r.insert("ranker".into(), json!(arm.ranker));
                    r.insert("scale".into(), json!(scale));
                    r.insert("seed".into(), json!(seed));
                    r.insert("tag".into(), json!(tag));
                    r.insert("runtime_s".into(), fl(round(t1.elapsed().as_secs_f64(), 1)));
                    for (k, v) in &met {
                        r.insert((*k).into(), fl(*v));
                    }
                    println!(
                        "  seed {seed} {:24} found={} AP={} cov={} rare={} decoy={} cu={} calls={}",
                        arm.label,
                        fmt::f(met["found_anywhere_in_register"], ".3f"),
                        fmt::f(met["average_precision"], ".4f"),
                        fmt::f(met["evidence_coverage_2links"], ".3f"),
                        fmt::f(met["rare_signal_recall"], ".3f"),
                        fmt::f(met["decoy_acceptance_all"], ".3f"),
                        fmt::f(met["compute_units"], ".2e"),
                        fmt::f(met["inference_calls"], ".2e"),
                    );
                    r
                })
                .collect()
        })
        .collect();
    let rows: Vec<Row> = blocks.into_iter().flatten().collect();
    append_rows(&format!("quick_{tag}.jsonl"), &rows)?;
    Ok(rows)
}

/// `python3 -m research.mycelic.arm_paired --tag T --arm SPEC [--arm SPEC ...] [--scale N] [--seeds S] [--alloc A]`
pub fn main(args: &[String]) -> Result<()> {
    let a = Args::parse(args)?;
    let tag = a.required("tag")?;
    let arms: Vec<Arm> = a.all("arm").into_iter().map(parse_arm).collect::<Result<_>>()?;
    if arms.is_empty() {
        bail!("the following arguments are required: --arm");
    }
    let scale: usize = a.get("scale").unwrap_or("10000").parse()?;
    let seeds: Vec<u64> = list(a.get("seeds").unwrap_or("0,1,2"))?;
    let rows = run_arms(&Cal::load(), &arms, scale, &seeds, tag, a.get("alloc").unwrap_or("back-loaded"))?;
    for arm in &arms[1..] {
        println!();
        println!("{}", report(&rows, &arms[0].label, &arm.label));
    }
    Ok(())
}
