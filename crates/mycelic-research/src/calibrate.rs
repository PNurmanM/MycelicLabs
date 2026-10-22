//! Hyper-parameter calibration of the v1 knobs (`calibrate.py`).
//!
//! Every system's main operating knob is tuned on CALIBRATION seeds (500-502), disjoint from the evaluation
//! seeds, by the same procedure and objective, then frozen.  `calibrate` rewrites calibration.json from scratch
//! (dropping the vNext knobs and the learned ranker), so run it only to reproduce v1.  No ranker is active in
//! any of these runs: the Python never installed one in this process.

use crate::calibrator::write_json;
use crate::models;
use crate::np::math;
use crate::org::{ENT, USER};
use crate::py::json::round;
use crate::py::value::fl;
use crate::runner::{self, Cal, World, art};
use crate::systems::{self, HierRunner};
use anyhow::Result;
use rayon::prelude::*;
use serde_json::{Map, Value, json};
use std::time::Instant;

pub const CAL_SEEDS: [u64; 3] = [500, 501, 502];
pub const CAL_SCALE: usize = 10_000;
pub const OBJECTIVE: &str = "average_precision";

type Metrics = crate::evalm::Metrics;

fn mean_of(vals: &[Metrics], key: &str) -> f64 {
    let v: Vec<f64> = vals.iter().map(|m| m[key]).collect();
    math::mean(&v)
}

fn worlds(scale: usize, seeds: &[u64]) -> Vec<World> {
    seeds.par_iter().map(|&s| runner::build_world_default(scale, s)).collect()
}

fn over_seeds(ws: &[World], f: impl Fn(&World) -> Metrics + Sync + Send) -> Vec<Metrics> {
    ws.par_iter().map(f).collect()
}

fn hier(cal: &Cal, w: &World, over: &Map<String, Value>) -> Metrics {
    let alloc = models::allocation_default("back-loaded");
    let ul = w.user_layer(&alloc[USER], w.seed);
    let (r, _) = HierRunner::new(&w.corpus, alloc, cal.hier_cfg(over), w.seed, &ul, &w.near_miss, None).run();
    runner::evaluate(w, &r)
}

fn ct_grid(ws: &[World]) -> (Vec<Value>, i64) {
    let alloc = models::allocation_default("back-loaded");
    let (mut ct, mut cb, mut cv) = (Vec::new(), 0i64, -1.0);
    for b in [0usize, 2000, 6000, 20000, 60000] {
        let vals = over_seeds(ws, |w| {
            let ul = w.user_layer(&alloc[USER], w.seed);
            runner::evaluate(w, &systems::central_triage(&w.corpus, &alloc, w.seed, &ul, 2, 0.35, 6000, b, None))
        });
        let v = mean_of(&vals, OBJECTIVE);
        let cu: Vec<f64> = vals.iter().map(|m| m["compute_units"]).collect();
        ct.push(json!({"kernel_ko_cap": b, OBJECTIVE: fl(round(v, 5)),
                       "found": fl(round(mean_of(&vals, "found_anywhere_in_register"), 4)), "cu": fl(math::mean(&cu))}));
        if v > cv {
            cv = v;
            cb = b as i64;
        }
    }
    (ct, cb)
}

/// `calibrate()`: the hierarchy's triage prior x question budget, map-reduce's kernel budget, the centralised
/// triage control's evidence budget and flat RAG's token budget.
pub fn calibrate(cal: &Cal) -> Result<Map<String, Value>> {
    let ws = worlds(CAL_SCALE, &CAL_SEEDS);
    let alloc = models::allocation_default("back-loaded");
    let mut out = Map::new();
    out.insert("scale".into(), json!(CAL_SCALE));
    out.insert("seeds".into(), json!(CAL_SEEDS));
    out.insert("objective".into(), json!(OBJECTIVE));
    out.insert("alloc".into(), json!("back-loaded"));

    let (mut best, mut best_v) = ((0.0, 0.0), -1.0);
    let mut grid = Vec::new();
    for w in [0.0, 1.2, 2.0, 3.0, 4.0] {
        for qf in [0.25, 0.55, 1.0] {
            let over = json!({"downward_retrieval": true, "questions": true, "cross_links": false,
                              "triage_prior_weight": w, "question_frac": qf});
            let vals = over_seeds(&ws, |wd| hier(cal, wd, over.as_object().unwrap()));
            let v = mean_of(&vals, OBJECTIVE);
            let cu: Vec<f64> = vals.iter().map(|m| m["compute_units"]).collect();
            grid.push(json!({"triage_prior_weight": w, "question_frac": qf, OBJECTIVE: fl(round(v, 5)),
                             "recall_at_100": fl(round(mean_of(&vals, "recall_at_100"), 4)),
                             "found": fl(round(mean_of(&vals, "found_anywhere_in_register"), 4)),
                             "cu": fl(math::mean(&cu))}));
            if v > best_v {
                best_v = v;
                best = (w, qf);
            }
        }
    }
    out.insert("hier_grid".into(), Value::Array(grid));
    out.insert("triage_prior_weight".into(), fl(best.0));
    out.insert("question_frac".into(), fl(best.1));

    let (mut mr, mut bb, mut bv) = (Vec::new(), 900i64, -1.0);
    for b in [300usize, 900, 2500, 8000] {
        let vals = over_seeds(&ws, |w| {
            let ul = w.user_layer(&alloc[USER], w.seed);
            runner::evaluate(w, &systems::map_reduce(&w.corpus, &alloc, w.seed, b, &ul, true, None))
        });
        let v = mean_of(&vals, OBJECTIVE);
        let cu: Vec<f64> = vals.iter().map(|m| m["compute_units"]).collect();
        mr.push(json!({"kernel_ko_budget": b, OBJECTIVE: fl(round(v, 5)),
                       "found": fl(round(mean_of(&vals, "found_anywhere_in_register"), 4)), "cu": fl(math::mean(&cu))}));
        if v > bv {
            bv = v;
            bb = b as i64;
        }
    }
    out.insert("mr_grid".into(), Value::Array(mr));
    out.insert("mr_budget".into(), json!(bb));

    let (ct, cb) = ct_grid(&ws);
    out.insert("ct_grid".into(), Value::Array(ct));
    out.insert("ct_kernel_ko_cap".into(), json!(cb));

    let (mut fr, mut fb, mut fv) = (Vec::new(), 120_000i64, -1.0);
    for b in [60_000i64, 120_000, 400_000, 1_000_000] {
        let vals = over_seeds(&ws, |w| {
            runner::evaluate(w, &systems::flat_rag(&w.corpus, &alloc[ENT], w.seed, b, &w.near_miss, true, None))
        });
        let v = mean_of(&vals, OBJECTIVE);
        let cu: Vec<f64> = vals.iter().map(|m| m["compute_units"]).collect();
        fr.push(json!({"token_budget": b, OBJECTIVE: fl(round(v, 5)),
                       "found": fl(round(mean_of(&vals, "found_anywhere_in_register"), 4)), "cu": fl(math::mean(&cu))}));
        if v > fv {
            fv = v;
            fb = b;
        }
    }
    out.insert("rag_grid".into(), Value::Array(fr));
    out.insert("flat_budget".into(), json!(fb));
    write_json(&art().join("calibration.json"), &Value::Object(out.clone()), 2)?;
    Ok(out)
}

fn read_cal() -> Map<String, Value> {
    std::fs::read_to_string(art().join("calibration.json"))
        .ok()
        .and_then(|t| serde_json::from_str(&t).ok())
        .unwrap_or_default()
}

/// `calibrate_ct()`: refit the centralised-triage control's evidence budget and merge it in.
pub fn calibrate_ct() -> Result<Map<String, Value>> {
    let ws = worlds(CAL_SCALE, &CAL_SEEDS);
    let mut cal = read_cal();
    let (ct, cb) = ct_grid(&ws);
    cal.insert("ct_grid".into(), Value::Array(ct));
    cal.insert("ct_kernel_ko_cap".into(), json!(cb));
    write_json(&art().join("calibration.json"), &Value::Object(cal.clone()), 2)?;
    Ok(cal)
}

/// `calibrate_evidence()`: the evidence-quality features (dispersion, synchrony, attribution), fitted on the
/// same held-out seeds and merged into calibration.json.
pub fn calibrate_evidence(cal: &Cal) -> Result<Map<String, Value>> {
    let ws = worlds(CAL_SCALE, &CAL_SEEDS);
    let mut calj = read_cal();
    let tpw = calj.get("triage_prior_weight").and_then(Value::as_f64).unwrap_or(0.0);
    let qf = calj.get("question_frac").and_then(Value::as_f64).unwrap_or(0.25);
    let mut combos: Vec<(f64, f64, f64)> = Vec::new();
    for wd in [0.0, 0.8, 1.5, 2.5] {
        for wa in [0.0, 2.5] {
            combos.push((wd, 0.0, wa));
        }
    }
    combos.push((1.5, 1.5, 2.5));
    combos.push((2.5, 1.5, 0.0));
    let (mut grid, mut best, mut bv) = (Vec::new(), (0.0, 0.0, 0.0), -1.0);
    for (wd, ws_, wa) in combos {
        let over = json!({"downward_retrieval": true, "questions": true, "cross_links": true,
                          "triage_prior_weight": tpw, "question_frac": qf, "w_dispersion": wd,
                          "w_synchrony": ws_, "verify_evidence": wa > 0.0, "w_attribution": wa});
        let vals = over_seeds(&ws, |w| hier(cal, w, over.as_object().unwrap()));
        let v = mean_of(&vals, OBJECTIVE);
        let cu: Vec<f64> = vals.iter().map(|m| m["compute_units"]).collect();
        grid.push(json!({"w_dispersion": wd, "w_synchrony": ws_, "w_attribution": wa, OBJECTIVE: fl(round(v, 5)),
                         "found": fl(round(mean_of(&vals, "found_anywhere_in_register"), 4)),
                         "recall_at_100": fl(round(mean_of(&vals, "recall_at_100"), 4)), "cu": fl(math::mean(&cu))}));
        if v > bv {
            bv = v;
            best = (wd, ws_, wa);
        }
    }
    calj.insert("evidence_grid".into(), Value::Array(grid));
    calj.insert("w_dispersion".into(), fl(best.0));
    calj.insert("w_synchrony".into(), fl(best.1));
    calj.insert("w_attribution".into(), fl(best.2));
    calj.insert("verify_evidence".into(), json!(best.2 > 0.0));
    write_json(&art().join("calibration.json"), &Value::Object(calj.clone()), 2)?;
    Ok(calj)
}

/// `python3 -m research.mycelic.calibrate [ct|evidence]`
pub fn main(args: &[String]) -> Result<()> {
    let t0 = Instant::now();
    let cal = Cal::load();
    if args.iter().any(|a| a == "ct") {
        let c = calibrate_ct()?;
        println!("ct_kernel_ko_cap = {}", c["ct_kernel_ko_cap"]);
        for row in c["ct_grid"].as_array().unwrap() {
            println!("   {}", crate::py::value::py_repr(row));
        }
        return Ok(());
    }
    if args.iter().any(|a| a == "evidence") {
        let c = calibrate_evidence(&cal)?;
        let keys = ["w_dispersion", "w_synchrony", "w_attribution", "verify_evidence"];
        let sel: Map<String, Value> = keys.iter().map(|k| (k.to_string(), c[*k].clone())).collect();
        println!("{}", crate::py::json::dumps_with(&Value::Object(sel), crate::py::json::Style::indent(2)));
        println!("\nevidence calibration took {:.0}s", t0.elapsed().as_secs_f64());
        return Ok(());
    }
    let c = calibrate(&cal)?;
    let sel: Map<String, Value> =
        c.iter().filter(|(k, _)| !k.ends_with("grid")).map(|(k, v)| (k.clone(), v.clone())).collect();
    println!("{}", crate::py::json::dumps_with(&Value::Object(sel), crate::py::json::Style::indent(2)));
    println!("\ncalibration took {:.0}s", t0.elapsed().as_secs_f64());
    Ok(())
}
