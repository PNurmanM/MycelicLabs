//! Per-pattern loss accounting (`loss_accounting.py`): where does each hidden pattern die?
//!
//! Every discoverable gold pattern is traced through the stages of a system's pipeline and the stage at which
//! it is lost is recorded with enough metadata (rare or common, regions, facets, witnesses) to slice the losses.
//! Hierarchy stages, in order: extracted, sketch_visible, in_triage, questioned, descent_reached, in_pool,
//! candidate, matched_any, matched_tau, in_register.  The centralised systems have no sketch, triage, question
//! or descent stage.  `first_loss` is the first failing stage in order (diagnostic); `loss_stage` the first
//! failing stage after the last passing one ('' for a reported pattern) - the tables use `loss_stage`.
//!
//! Output: `artifacts/loss_funnel.jsonl`, one row per (arch, scale, seed, pid) plus a summary row per run.

use crate::corpus::Pattern;
use crate::evalm::match_sets;
use crate::experiments::append_rows;
use crate::ops::{Hypothesis, chain_span, stem_rep_map};
use crate::org::REGION;
use crate::py::value::{Row, fl};
use crate::runner::{self, Cal, World, arch_spec, art};
use crate::systems::{HierRunnerState, RunResult, UserLayer};
use anyhow::Result;
use rustc_hash::{FxHashMap, FxHashSet};
use serde_json::{Map, Value, json};
use std::collections::HashSet;
use std::path::PathBuf;
use std::time::Instant;

pub const HIER_STAGES: [&str; 10] = [
    "extracted",
    "sketch_visible",
    "in_triage",
    "questioned",
    "descent_reached",
    "in_pool",
    "candidate",
    "matched_any",
    "matched_tau",
    "in_register",
];
pub const FLAT_STAGES: [&str; 6] = ["extracted", "in_pool", "candidate", "matched_any", "matched_tau", "in_register"];

pub const TAU: f64 = 0.5;

/// `artifacts/loss_funnel.jsonl`
pub fn funnel_path() -> PathBuf {
    art().join("loss_funnel.jsonl")
}

/// (facets with >= 1 correct claim, facets with >= 1 claim of any kind).  A facet counts as extracted only if a
/// genuine (non-spurious) claim of one of its records carries BOTH the right anchor and the right predicate.
fn facets_extracted(ul: &UserLayer, p: &Pattern) -> (usize, usize) {
    let ex = &ul.ex;
    let mut rid_rows: FxHashMap<i64, Vec<usize>> = FxHashMap::default();
    for (i, &r) in ex.rid.iter().enumerate() {
        if !ex.spurious[i] {
            rid_rows.entry(r).or_default().push(i);
        }
    }
    let (mut ok, mut anyk) = (0, 0);
    for (j, fr) in p.facet_records.iter().enumerate() {
        let want = p.preds[j] as i64;
        let (mut hit, mut seen) = (false, false);
        'facet: for &r in fr {
            for &i in rid_rows.get(&(r as i64)).map(Vec::as_slice).unwrap_or(&[]) {
                seen = true;
                if ex.anchor[i] as i64 == p.anchor as i64 && ex.pred[i] as i64 == want {
                    hit = true;
                    break 'facet;
                }
            }
        }
        ok += usize::from(hit);
        anyk += usize::from(seen);
    }
    (ok, anyk)
}

/// Re-run the kernel's triage predicate for ONE anchor, keeping the reason it fails.
fn sketch_diagnosis(world: &World, h: &HierRunnerState, anchor: i64) -> Vec<(&'static str, Value)> {
    let org = &world.corpus.org;
    let a32 = anchor as i32;
    let mut rows: Vec<(usize, usize, i64, i64)> = Vec::new();
    let (mut tot, mut n_sites_any) = (0i64, 0usize);
    for (&st, ent) in &h.site_sketch {
        let Some(&(t, mask, _, _)) = ent.get(&a32) else { continue };
        n_sites_any += 1;
        tot += t;
        if mask != 0 {
            rows.push((st, org.ancestor_at(st, REGION).unwrap(), mask, t));
        }
    }
    let mut out = vec![
        ("sk_sites_present", json!(n_sites_any)),
        ("sk_sites_with_bit", json!(rows.len())),
        ("sk_total_mentions", json!(tot)),
    ];
    if rows.is_empty() {
        out.push(("sketch_fail", json!("no_site_bit")));
        return out;
    }
    let total = 1.max(tot);
    let foreign: Vec<&(usize, usize, i64, i64)> =
        rows.iter().filter(|r| r.3 as f64 / total as f64 <= h.foreign_share).collect();
    out.push(("sk_foreign_sites", json!(foreign.len())));
    if foreign.len() < 2 {
        out.push(("sketch_fail", json!("foreign_lt2")));
        return out;
    }
    let regs: FxHashSet<usize> = foreign.iter().map(|r| r.1).collect();
    out.push(("sk_foreign_regions", json!(regs.len())));
    if regs.len() < 2 {
        out.push(("sketch_fail", json!("regions_lt2")));
        return out;
    }
    let um = foreign.iter().fold(0i64, |m, r| m | r.2);
    let (span, _) = chain_span(um);
    out.push(("sk_span", json!(span)));
    if span < 2 {
        out.push(("sketch_fail", json!("span_lt2")));
        return out;
    }
    out.push(("sketch_fail", json!("")));
    out
}

/// (index of the best primary-matching hypothesis, gold links it shares).
fn best_match(hyps: &[&Hypothesis], p: &Pattern, stem: &[i32]) -> (Option<usize>, usize) {
    let m = match_sets(hyps, &[p], stem, false, false);
    let Some(idx) = m.get(&p.pid) else { return (None, 0) };
    let mut best = idx[0];
    for &i in &idx[1..] {
        if hyps[i].conf > hyps[best].conf {
            best = i;
        }
    }
    let ps: HashSet<usize> = p.preds.iter().copied().collect();
    let shared = hyps[best].preds.iter().filter(|x| ps.contains(x)).collect::<HashSet<_>>().len();
    (Some(best), shared)
}

fn shares_link(x: &Hypothesis, p: &Pattern) -> usize {
    let ps: HashSet<usize> = p.preds.iter().copied().collect();
    x.preds.iter().filter(|q| ps.contains(q)).collect::<HashSet<_>>().len()
}

fn base_row(arch: &str, scale: usize, seed: u64, p: &Pattern) -> Row {
    let mut r = Row::new();
    r.insert("arch".into(), json!(arch));
    r.insert("scale".into(), json!(scale));
    r.insert("seed".into(), json!(seed));
    r.insert("pid".into(), json!(p.pid));
    r.insert("anchor".into(), json!(p.anchor));
    r.insert("chain".into(), json!(p.chain));
    r.insert("n_facets".into(), json!(p.preds.len()));
    r.insert("rare".into(), json!(p.rare));
    r.insert("n_regions".into(), json!(p.n_regions));
    r.insert("n_witnesses".into(), json!(p.facet_records.iter().map(Vec::len).sum::<usize>()));
    r.insert("contradicted".into(), json!(p.contradicted));
    r
}

/// Candidate / match / register stages shared by both funnels.
fn match_stages(r: &mut Row, res: &RunResult, full: &[&Hypothesis], p: &Pattern, stem: &[i32]) {
    let a = p.anchor as i64;
    let ranked: Vec<&Hypothesis> = res.ranked();
    let ranked_idx: Vec<usize> = (0..ranked.len()).filter(|&i| ranked[i].conf >= TAU).collect();
    let ranked_tau: Vec<&Hypothesis> = ranked_idx.iter().map(|&i| ranked[i]).collect();
    let cand: Vec<&&Hypothesis> =
        full.iter().filter(|x| x.anchor == a && x.chain == p.chain as i64 && shares_link(x, p) > 0).collect();
    let (bi, shared) = best_match(full, p, stem);
    r.insert("candidate".into(), json!(!cand.is_empty()));
    r.insert("candidate_links".into(), json!(cand.iter().map(|x| shares_link(x, p)).max().unwrap_or(0)));
    let best_conf = cand.iter().map(|x| x.conf).reduce(|a, b| if b > a { b } else { a }).unwrap_or(-1.0);
    r.insert("candidate_conf".into(), fl(best_conf));
    r.insert("matched_any".into(), json!(bi.is_some()));
    r.insert("match_links".into(), json!(shared));
    r.insert("match_conf".into(), fl(bi.map_or(-1.0, |i| full[i].conf)));
    r.insert("matched_tau".into(), json!(bi.is_some_and(|i| full[i].conf >= TAU)));
    let (bi2, _) = best_match(&ranked_tau, p, stem);
    r.insert("in_register".into(), json!(bi2.is_some()));
    r.insert("rank".into(), json!(bi2.map_or(-1, |i| ranked_idx[i] as i64)));
}

fn truthy(r: &Row, s: &str) -> bool {
    r.get(s).is_some_and(crate::py::value::truthy)
}

/// `_loss_fields`: the two readings of where a pattern was lost.
fn loss_fields(r: &mut Row, stages: &[&str]) {
    let first = stages.iter().find(|s| !truthy(r, s)).copied().unwrap_or("");
    r.insert("first_loss".into(), json!(first));
    if truthy(r, "in_register") {
        r.insert("loss_stage".into(), json!(""));
        return;
    }
    let last_pass = stages.iter().rposition(|s| truthy(r, s)).map_or(-1, |i| i as i64);
    let stage =
        stages[(last_pass + 1) as usize..].iter().find(|s| !truthy(r, s)).copied().unwrap_or(stages[stages.len() - 1]);
    r.insert("loss_stage".into(), json!(stage));
}

fn hier_rows(arch: &str, world: &World, res: &RunResult, h: &HierRunnerState, ul: &UserLayer) -> Vec<Row> {
    let c = &world.corpus;
    let stem = stem_rep_map(c);
    let q_anchors: FxHashSet<i64> = res.questions.iter().map(|q| q.anchor).collect();
    let mut q_order: FxHashMap<i64, usize> = FxHashMap::default();
    for (i, &a) in h.question_order.iter().enumerate() {
        q_order.entry(a).or_insert(i);
    }
    let mut tg: Vec<(f64, i64)> = h.triage_gain.iter().map(|(&a, &g)| (g, a)).collect();
    tg.sort_by(|x, y| y.partial_cmp(x).unwrap());
    let triage_rank: FxHashMap<i64, usize> = tg.iter().enumerate().map(|(i, &(_, a))| (a, i)).collect();
    let full: Vec<&Hypothesis> = res.full_hyps.as_ref().expect("full").iter().map(|&i| &res.store[i]).collect();
    let mut rows = Vec::new();
    for &pid in &world.gold.discoverable {
        let p = &c.patterns[pid];
        let a = p.anchor as i64;
        let (ok, anyk) = facets_extracted(ul, p);
        let mut r = base_row(arch, world.scale, world.seed, p);
        r.insert("facets_extracted".into(), json!(ok));
        r.insert("facets_seen".into(), json!(anyk));
        r.insert("extracted".into(), json!(ok >= 2));
        for (k, v) in sketch_diagnosis(world, h, a) {
            r.insert(k.into(), v);
        }
        r.insert("sketch_visible".into(), json!(r["sketch_fail"] == json!("")));
        r.insert("in_triage".into(), json!(h.triage_gain.contains_key(&a)));
        r.insert("triage_rank".into(), json!(triage_rank.get(&a).map_or(-1, |&x| x as i64)));
        r.insert("question_rank".into(), json!(q_order.get(&a).map_or(-1, |&x| x as i64)));
        r.insert("questioned".into(), json!(q_anchors.contains(&a)));
        let facet_users: FxHashSet<usize> = p.facet_users.iter().flatten().copied().collect();
        let empty = FxHashSet::default();
        let reached = h.reached_users.get(&a).unwrap_or(&empty);
        r.insert("users_reached".into(), json!(reached.len()));
        let fur = reached.iter().filter(|u| facet_users.contains(u)).count();
        r.insert("facet_users_reached".into(), json!(fur));
        r.insert(
            "facets_reached".into(),
            json!(p.facet_users.iter().filter(|fu| fu.iter().any(|u| reached.contains(u))).count()),
        );
        r.insert("descent_reached".into(), json!(fur > 0));
        let links: FxHashSet<(i64, i64)> = p.preds.iter().map(|&pr| (pr as i64, a)).collect();
        let lip = links.iter().filter(|l| res.retained.contains(l)).count();
        r.insert("links_in_pool".into(), json!(lip));
        r.insert("in_pool".into(), json!(lip >= 2));
        match_stages(&mut r, res, &full, p, &stem);
        loss_fields(&mut r, &HIER_STAGES);
        rows.push(r);
    }
    rows
}

fn flat_rows(arch: &str, world: &World, res: &RunResult, ul: Option<&UserLayer>) -> Vec<Row> {
    let c = &world.corpus;
    let stem = stem_rep_map(c);
    let full: Vec<&Hypothesis> = match &res.full_hyps {
        Some(f) if !f.is_empty() => f.iter().map(|&i| &res.store[i]).collect(),
        _ => res.ranked(),
    };
    // the flat pools carry evidence ids per object, so "extracted" can be read off the pool
    let mut ev_of: FxHashMap<(i64, i64), FxHashSet<i64>> = FxHashMap::default();
    for &k in &res.kernel_kos {
        let ko = &res.arena[k];
        ev_of.entry((ko.pred, ko.anchor)).or_default().extend(ko.evidence.iter().copied());
    }
    let mut rows = Vec::new();
    for &pid in &world.gold.discoverable {
        let p = &c.patterns[pid];
        let a = p.anchor as i64;
        let mut ok = 0;
        for (j, fr) in p.facet_records.iter().enumerate() {
            if ev_of.get(&(p.preds[j] as i64, a)).is_some_and(|ev| fr.iter().any(|&x| ev.contains(&(x as i64)))) {
                ok += 1;
            }
        }
        let mut r = base_row(arch, world.scale, world.seed, p);
        r.insert("facets_extracted".into(), json!(ok));
        let links: FxHashSet<(i64, i64)> = p.preds.iter().map(|&pr| (pr as i64, a)).collect();
        let lip = links.iter().filter(|l| res.retained.contains(l)).count();
        r.insert("links_in_pool".into(), json!(lip));
        match ul {
            Some(ul) => {
                // B4 and the oracle consume the same edge extraction as the hierarchy
                let (ok2, anyk) = facets_extracted(ul, p);
                r.insert("facets_extracted".into(), json!(ok2));
                r.insert("facets_seen".into(), json!(anyk));
                r.insert("extracted".into(), json!(ok2 >= 2));
            }
            // A2 extracts per chunk with the kernel tier and keeps no extraction: the stage is not measured
            None => {
                r.insert("extracted".into(), Value::Null);
            }
        }
        r.insert("in_pool".into(), json!(lip >= 2));
        match_stages(&mut r, res, &full, p, &stem);
        let stages: &[&str] = if ul.is_some() { &FLAT_STAGES } else { &FLAT_STAGES[1..] };
        loss_fields(&mut r, stages);
        rows.push(r);
    }
    rows
}

fn summary(arch: &str, rows: &[Row], world: &World, res: &RunResult) -> Row {
    let stages: Vec<&str> = if rows[0].contains_key("sketch_visible") {
        HIER_STAGES.to_vec()
    } else if !rows[0]["extracted"].is_null() {
        FLAT_STAGES.to_vec()
    } else {
        FLAT_STAGES[1..].to_vec()
    };
    let mut out = Row::new();
    out.insert("arch".into(), json!(arch));
    out.insert("scale".into(), json!(world.scale));
    out.insert("seed".into(), json!(world.seed));
    out.insert("summary".into(), json!(true));
    out.insert("n_gold".into(), json!(rows.len()));
    out.insert("n_rare".into(), json!(rows.iter().filter(|r| r["rare"] == json!(true)).count()));
    out.insert("stages".into(), json!(stages));
    for s in &stages {
        let vals: Vec<&Value> = rows.iter().filter_map(|r| r.get(*s)).filter(|v| !v.is_null()).collect();
        let k = vals.iter().filter(|v| crate::py::value::truthy(v)).count();
        out.insert(format!("surv_{s}"), json!(k));
        out.insert(
            format!("frac_{s}"),
            if vals.is_empty() { Value::Null } else { fl(k as f64 / 1.max(vals.len()) as f64) },
        );
        out.insert(
            format!("rare_surv_{s}"),
            json!(rows.iter().filter(|r| truthy(r, s) && r["rare"] == json!(true)).count()),
        );
    }
    let mut fl_counts = Map::new();
    for r in rows {
        let ls = r["loss_stage"].as_str().unwrap();
        let key = if ls.is_empty() { "reported" } else { ls };
        let n = fl_counts.get(key).and_then(Value::as_i64).unwrap_or(0) + 1;
        fl_counts.insert(key.into(), json!(n));
    }
    out.insert("loss_stage_counts".into(), Value::Object(fl_counts));
    out.insert("n_reported".into(), json!(rows.iter().filter(|r| truthy(r, "in_register")).count()));
    if rows[0].contains_key("sketch_fail") {
        let mut sf = Map::new();
        for r in rows {
            if !truthy(r, "sketch_visible") {
                let k = r["sketch_fail"].as_str().unwrap().to_string();
                let n = sf.get(&k).and_then(Value::as_i64).unwrap_or(0) + 1;
                sf.insert(k, json!(n));
            }
        }
        out.insert("sketch_fail_counts".into(), Value::Object(sf));
    }
    let ranked = res.ranked();
    out.insert("n_hypotheses".into(), json!(ranked.len()));
    out.insert("n_hypotheses_tau".into(), json!(ranked.iter().filter(|x| x.conf >= TAU).count()));
    out.insert("kernel_pool".into(), json!(res.kernel_kos.len()));
    out.insert("n_questions".into(), json!(res.questions.len()));
    out.insert("compute_units".into(), fl(res.meter.cu()));
    out
}

/// The funnel rows (per pattern, then the summary) of one architecture on one world.
pub fn arch_rows(cal: &Cal, arch: &str, name: &str, world: &World, cfg_over: Option<&Map<String, Value>>) -> Vec<Row> {
    let alloc = runner::back_loaded();
    let kind = arch_spec(arch).map_or("hier", |s| s.0);
    let run = runner::run_arch(cal, arch, world, &alloc, world.seed, cfg_over);
    let mut rows = if kind == "hier" {
        let ul = world.user_layer(&alloc[crate::org::USER], world.seed);
        hier_rows(name, world, &run.res, run.hier.as_ref().expect("hierarchy state"), &ul)
    } else {
        let ul = matches!(kind, "central_triage" | "oracle" | "map_reduce" | "naive_enum")
            .then(|| world.user_layer(&alloc[crate::org::USER], world.seed));
        flat_rows(name, world, &run.res, ul.as_deref())
    };
    let summ = summary(name, &rows, world, &run.res);
    rows.push(summ);
    rows
}

pub const DEFAULT_ARCHS: [&str; 4] = ["H_mycelic_full", "B4_central_triage", "Y_oracle_retrieval", "A2_chunked_ctx"];

/// `run(scales, seeds)`: append every row to the funnel file, world by world.
pub fn run(cal: &Cal, scales: &[usize], seeds: &[u64], archs: &[&str], tag: &str) -> Result<Vec<Row>> {
    use rayon::prelude::*;
    let mut all = Vec::new();
    for &scale in scales {
        let blocks: Vec<Vec<Row>> = seeds
            .par_iter()
            .map(|&seed| {
                let t0 = Instant::now();
                let w = runner::build_world_default(scale, seed);
                let rows: Vec<Vec<Row>> =
                    archs.par_iter().map(|a| arch_rows(cal, a, &format!("{a}{tag}"), &w, None)).collect();
                for (a, rs) in archs.iter().zip(&rows) {
                    let s = rs.last().unwrap();
                    let st: Vec<String> = s["stages"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .filter_map(|x| {
                            let x = x.as_str().unwrap();
                            s[&format!("frac_{x}")]
                                .as_f64()
                                .map(|f| format!("{}={}", &x[..x.len().min(8)], crate::py::fmt::f(f, ".2f")))
                        })
                        .collect();
                    println!("  {a:22} {}", st.join(" "));
                }
                println!("scale={scale} seed={seed} done in {:.0}s", t0.elapsed().as_secs_f64());
                rows.into_iter().flatten().collect()
            })
            .collect();
        for b in blocks {
            append_rows("loss_funnel.jsonl", &b)?;
            all.extend(b);
        }
    }
    Ok(all)
}

/// `python3 -m research.mycelic.loss_accounting [scales [seeds]]`
pub fn main(args: &[String]) -> Result<()> {
    let cal = Cal::load();
    let parse = |s: &str| -> Vec<u64> { s.split(',').map(|x| x.trim().parse().expect("number")).collect() };
    if let Some(sc) = args.first() {
        let scales: Vec<usize> = parse(sc).into_iter().map(|x| x as usize).collect();
        let seeds = args.get(1).map_or(vec![0], |s| parse(s));
        run(&cal, &scales, &seeds, &DEFAULT_ARCHS, "")?;
    } else {
        run(&cal, &[10_000], &[0, 1, 2, 3, 4], &DEFAULT_ARCHS, "")?;
        run(&cal, &[50_000], &[0, 1, 2], &DEFAULT_ARCHS, "")?;
    }
    Ok(())
}
