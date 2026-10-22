//! The experiment suite E1-E12 (`experiments.py`): each experiment appends one JSON row per run to its own
//! JSONL file in the artifacts directory.  Nothing here aggregates; `analysis` reads the rows back.
//!
//! The Python ran worlds and architectures one after another.  Every run seeds its own generators from
//! (seed, architecture), so here worlds and runs execute in parallel (rayon) and the rows are written in the
//! Python's order: the output does not depend on thread scheduling.  `runtime_s` is the run's own wall time.
//!
//! Each experiment ran in its own `python3 -m research.mycelic.experiments <e>` process (run_suite.sh and
//! final_rerun.sh start one per experiment), which matters for the one place the Python leaked state between
//! runs: the ranker global.  E1d's centralised-triage runs call the system directly and therefore ran with no
//! ranker; E6's map-reduce frontier ran with the ranker the preceding hierarchy runs had installed (the
//! hierarchy's).  Both are reproduced as such.

use crate::corpus::CorpusCfg;
use crate::evalm::Metrics;
use crate::models::{self, ALLOCATIONS, HardShape, Tier, tier_from_q};
use crate::org::{DEPT, TEAM, USER, default_fanin};
use crate::py::json as pyjson;
use crate::py::value::{Row, fl};
use crate::runner::{self, Cal, World, WorldOpts, archs, art};
use crate::systems::{self, HierRunner, RunResult};
use anyhow::{Result, bail};
use rayon::prelude::*;
use serde_json::{Map, Value, json};
use std::io::Write as _;
use std::time::Instant;

pub const EVAL_SEEDS: [u64; 5] = [0, 1, 2, 3, 4];
pub const SCALES: [usize; 3] = [2_000, 10_000, 50_000];

/// E2: one feature removed (or added) at a time from the full system, in `ABLATIONS` order.
pub fn ablations() -> Vec<(&'static str, Value)> {
    vec![
        ("full", json!({})),
        ("-lineage", json!({"lineage": false})),
        ("-independence", json!({"independence": false})),
        ("-contradiction", json!({"contradiction": false})),
        ("-temporal", json!({"temporal": false})),
        ("-adaptive_routing", json!({"adaptive_routing": false})),
        ("-questions", json!({"questions": false})),
        ("-question_targeting", json!({"question_targeting": false})),
        ("-downward_retrieval", json!({"downward_retrieval": false, "questions": false})),
        ("-cross_links", json!({"cross_links": false})),
        ("-sketch_channel", json!({"sketch_channel": false})),
        ("-adaptive_abstraction", json!({"adaptive_abstraction": false})),
        ("-triage_prior", json!({"triage_prior_weight": 0.0})),
        ("+evidence_verification", json!({"verify_evidence": true})),
        ("+source_dispersion", json!({"w_dispersion": 1.5})),
        ("+chain_completion", json!({"chain_completion": true})),
        ("-foreign_filter", json!({"foreign_only_evidence": false})),
        ("-synthesis_restriction", json!({"restrict_synthesis_to_triage": false})),
    ]
}

/// E5: adversarial conditions (corpus, hierarchy and org overrides), in `ADVERSARIAL` order.
pub fn adversarial() -> Vec<(&'static str, Value)> {
    vec![
        ("clean", json!({})),
        ("high_cross_site_noise", json!({"corpus": {"p_cross_site_entity": 0.15}})),
        ("decoy_heavy", json!({"corpus": {"decoy_ratio": 3.0}})),
        ("duplicate_flood", json!({"corpus": {"p_dup": 0.30}})),
        ("stale_flood", json!({"corpus": {"p_stale": 0.25}})),
        ("false_report_flood", json!({"corpus": {"p_false": 0.15}})),
        ("rare_only", json!({"corpus": {"rare_fraction": 1.0}})),
        ("malicious_nodes_5pct", json!({"cfg": {"malicious_frac": 0.05}})),
        ("unavailable_sites_20pct", json!({"cfg": {"unavailable_frac": 0.20}})),
        ("approx_index_bloom", json!({"cfg": {"bloom_fp": 0.02}})),
        ("extreme_imbalance", json!({"org": {"n_regions": 3}})),
    ]
}

fn obj(v: &Value) -> Map<String, Value> {
    v.as_object().cloned().unwrap_or_default()
}

/// `_row(name, world, res, extra)`: identity fields, the experiment's tags, then every metric (a metric with
/// the same name as an earlier field replaces its value in place, as the Python dict merge does).
pub fn row(name: &str, world: &World, m: &Metrics, extra: Vec<(&str, Value)>) -> Row {
    let mut r = Row::new();
    r.insert("arch".into(), Value::from(name));
    r.insert("n_users".into(), Value::from(world.corpus.org.user_ids.len()));
    r.insert("n_records".into(), Value::from(world.corpus.recs.len()));
    r.insert("n_gold".into(), Value::from(world.gold.discoverable.len()));
    for (k, v) in extra {
        r.insert(k.into(), v);
    }
    for (k, v) in m {
        r.insert((*k).into(), fl(*v));
    }
    r
}

/// Python's `round(time.time() - t, 2)`.
fn runtime(t: Instant, digits: usize) -> Value {
    fl(pyjson::round(t.elapsed().as_secs_f64(), digits))
}

/// Append rows to `artifacts/<fname>` as `json.dumps(row)` lines.
pub fn append_rows(fname: &str, rows: &[Row]) -> Result<()> {
    let path = art().join(fname);
    let mut fh = std::fs::OpenOptions::new().create(true).append(true).open(&path)?;
    for r in rows {
        writeln!(fh, "{}", pyjson::dumps_with(&Value::Object(r.clone()), pyjson::Style::ASCII))?;
    }
    Ok(())
}

/// One run of `name` on `world`, with the per-architecture ranker (`_run_named`).
fn run_named(
    cal: &Cal,
    name: &str,
    w: &World,
    alloc: &[Tier],
    seed: u64,
    cfg_over: Option<&Map<String, Value>>,
) -> RunResult {
    runner::run_arch(cal, name, w, alloc, seed, cfg_over).res
}

/// A hierarchy run with the hierarchy's ranker and a configuration (E2, E6, E8).
fn run_hier(cal: &Cal, w: &World, alloc: &[Tier], seed: u64, cfg: &Map<String, Value>) -> RunResult {
    let ranker = cal.ranker_for("H_mycelic_full");
    let ul = w.user_layer(&alloc[USER], seed);
    let (res, _) =
        HierRunner::new(&w.corpus, alloc.to_vec(), cal.hier_cfg(cfg), seed, &ul, &w.near_miss, ranker.as_ref()).run();
    res
}

fn print_row(prefix: &str, name: &str, r: &Row) {
    let g = |k: &str| r.get(k).and_then(Value::as_f64).unwrap_or(f64::NAN);
    println!(
        "  {prefix} {name:22} AP={} found={} R@100={} cu={}",
        crate::py::fmt::f(g("average_precision"), ".4f"),
        crate::py::fmt::f(g("found_anywhere_in_register"), ".2f"),
        crate::py::fmt::f(g("recall_at_100"), ".3f"),
        crate::py::fmt::f(g("compute_units"), ".2e"),
    );
}

/// Build one world per spec and run `per_world` on each, in parallel; rows come back in spec order.
fn over_worlds<S: Sync, F>(specs: &[S], build: impl Fn(&S) -> World + Sync + Send, per_world: F) -> Vec<Row>
where
    F: Fn(&S, &World) -> Vec<Row> + Sync + Send,
{
    let blocks: Vec<Vec<Row>> = specs
        .par_iter()
        .map(|s| {
            let w = build(s);
            let rows = per_world(s, &w);
            w.clear_cache();
            rows
        })
        .collect();
    blocks.into_iter().flatten().collect()
}

/// Run `f` over items in parallel, keeping their order.
fn par_rows<T: Sync>(items: &[T], f: impl Fn(&T) -> Row + Sync + Send) -> Vec<Row> {
    items.par_iter().map(f).collect()
}

fn alloc_back_loaded() -> Vec<Tier> {
    models::allocation_default("back-loaded")
}

/// E1 - every architecture across scales (also E1c with more seeds).
pub fn e1_baselines(cal: &Cal, scales: &[usize], seeds: &[u64], fname: &str) -> Result<Vec<Row>> {
    let alloc = alloc_back_loaded();
    let specs: Vec<(usize, u64)> = scales.iter().flat_map(|&s| seeds.iter().map(move |&d| (s, d))).collect();
    let names: Vec<&str> = archs().iter().map(|a| a.0).collect();
    let rows = over_worlds(
        &specs,
        |&(s, d)| runner::build_world_default(s, d),
        |&(scale, seed), w| {
            w.user_layer(&alloc[USER], seed);
            par_rows(&names, |a| {
                let t = Instant::now();
                let res = run_named(cal, a, w, &alloc, seed, None);
                let m = runner::evaluate(w, &res);
                let r = row(
                    a,
                    w,
                    &m,
                    vec![
                        ("scale", json!(scale)),
                        ("seed", json!(seed)),
                        ("alloc", json!("back-loaded")),
                        ("runtime_s", runtime(t, 2)),
                    ],
                );
                print_row(&format!("{scale:6}/{seed}"), a, &r);
                r
            })
        },
    );
    append_rows(fname, &rows)?;
    Ok(rows)
}

/// E1b - architectures added after the first E1 run, same worlds (also E11 with the verified hierarchy).
pub fn e1b_extra(cal: &Cal, scales: &[usize], seeds: &[u64], names: &[&str], fname: &str) -> Result<Vec<Row>> {
    let alloc = alloc_back_loaded();
    let specs: Vec<(usize, u64)> = scales.iter().flat_map(|&s| seeds.iter().map(move |&d| (s, d))).collect();
    let rows = over_worlds(
        &specs,
        |&(s, d)| runner::build_world_default(s, d),
        |&(scale, seed), w| {
            w.user_layer(&alloc[USER], seed);
            par_rows(names, |a| {
                let t = Instant::now();
                let res = run_named(cal, a, w, &alloc, seed, None);
                let m = runner::evaluate(w, &res);
                let r = row(
                    a,
                    w,
                    &m,
                    vec![
                        ("scale", json!(scale)),
                        ("seed", json!(seed)),
                        ("alloc", json!("back-loaded")),
                        ("runtime_s", runtime(t, 2)),
                    ],
                );
                print_row(&format!("{scale:6}/{seed}"), a, &r);
                r
            })
        },
    );
    append_rows(fname, &rows)?;
    Ok(rows)
}

/// E2 - ablations: one feature removed at a time from the full system.
pub fn e2_ablations(cal: &Cal, scale: usize, seeds: &[u64]) -> Result<Vec<Row>> {
    let alloc = alloc_back_loaded();
    let abl = ablations();
    let rows = over_worlds(
        seeds,
        |&d| runner::build_world_default(scale, d),
        |&seed, w| {
            w.user_layer(&alloc[USER], seed);
            par_rows(&abl, |(name, over)| {
                let mut cfg = obj(&json!({"downward_retrieval": true, "questions": true, "cross_links": true}));
                cfg.extend(obj(over));
                let t = Instant::now();
                let res = run_hier(cal, w, &alloc, seed, &cfg);
                let m = runner::evaluate(w, &res);
                let r = row(
                    &format!("H{name}"),
                    w,
                    &m,
                    vec![
                        ("scale", json!(scale)),
                        ("seed", json!(seed)),
                        ("ablation", json!(name)),
                        ("alloc", json!("back-loaded")),
                        ("runtime_s", runtime(t, 2)),
                    ],
                );
                print_row(&format!("{seed}"), name, &r);
                r
            })
        },
    );
    append_rows("e2_ablations.jsonl", &rows)?;
    Ok(rows)
}

/// E3 - compute allocation across levels.
pub fn e3_allocation(cal: &Cal, scale: usize, seeds: &[u64]) -> Result<Vec<Row>> {
    let jobs: Vec<(&str, &str)> = ALLOCATIONS
        .iter()
        .flat_map(|(a, _)| ["H_mycelic_full", "B2_map_reduce", "E_hier_lineage"].into_iter().map(move |n| (*a, n)))
        .collect();
    let rows = over_worlds(
        seeds,
        |&d| runner::build_world_default(scale, d),
        |&seed, w| {
            par_rows(&jobs, |&(aname, arch)| {
                let alloc = models::allocation_default(aname);
                let t = Instant::now();
                let res = run_named(cal, arch, w, &alloc, seed, None);
                let m = runner::evaluate(w, &res);
                let r = row(
                    arch,
                    w,
                    &m,
                    vec![
                        ("scale", json!(scale)),
                        ("seed", json!(seed)),
                        ("alloc", json!(aname)),
                        ("tiers", json!(models::allocation_tiers(aname).join("->"))),
                        ("runtime_s", runtime(t, 2)),
                    ],
                );
                print_row(&format!("{seed} {aname}"), arch, &r);
                r
            })
        },
    );
    append_rows("e3_allocation.jsonl", &rows)?;
    Ok(rows)
}

/// E3b - marginal value of capability at each level: every level small-7b, then one upgraded to frontier.
pub fn e3b_level_marginal(cal: &Cal, scale: usize, seeds: &[u64]) -> Result<Vec<Row>> {
    let levels = ["user", "team", "dept", "site", "region", "enterprise"];
    let jobs: Vec<(i64, &str)> =
        (-1..6).flat_map(|up| ["H_mycelic_full", "B2_map_reduce"].into_iter().map(move |n| (up, n))).collect();
    let rows = over_worlds(
        seeds,
        |&d| runner::build_world_default(scale, d),
        |&seed, w| {
            par_rows(&jobs, |&(up, arch)| {
                let mut names = ["small-7b"; 6];
                if up >= 0 {
                    names[up as usize] = "frontier";
                }
                let alloc: Vec<Tier> = names.iter().map(|n| models::anchor(n, HardShape::Convex)).collect();
                let t = Instant::now();
                let res = run_named(cal, arch, w, &alloc, seed, None);
                let m = runner::evaluate(w, &res);
                let lvl = if up < 0 { "none" } else { levels[up as usize] };
                let r = row(
                    arch,
                    w,
                    &m,
                    vec![
                        ("scale", json!(scale)),
                        ("seed", json!(seed)),
                        ("upgraded_level", json!(lvl)),
                        ("alloc", json!(names.join("+"))),
                        ("runtime_s", runtime(t, 2)),
                    ],
                );
                print_row(&format!("{seed} upgrade={lvl:11}"), arch, &r);
                r
            })
        },
    );
    append_rows("e3b_level_marginal.jsonl", &rows)?;
    Ok(rows)
}

/// The uniform capability-q allocation of E3c.
pub fn uniform_q_alloc(q: f64) -> Vec<Tier> {
    let t = tier_from_q(
        q,
        Some(&format!("q{q:.2}")),
        Some(3.0 * crate::py::pow(40.0, q)),
        Some((8000.0 * crate::py::pow(128.0, q)) as i64),
        HardShape::Convex,
    );
    vec![t; 6]
}

/// E3c - uniform capability sweep: the function from operator quality to architecture quality.
pub fn e3c_q_sweep(cal: &Cal, scale: usize, seeds: &[u64]) -> Result<Vec<Row>> {
    let qs = [0.10, 0.28, 0.44, 0.58, 0.72, 0.88, 1.00];
    let jobs: Vec<(f64, &str)> = qs
        .iter()
        .flat_map(|&q| ["H_mycelic_full", "B2_map_reduce", "E_hier_lineage"].into_iter().map(move |n| (q, n)))
        .collect();
    let rows = over_worlds(
        seeds,
        |&d| runner::build_world_default(scale, d),
        |&seed, w| {
            par_rows(&jobs, |&(q, arch)| {
                let alloc = uniform_q_alloc(q);
                let res = run_named(cal, arch, w, &alloc, seed, None);
                let m = runner::evaluate(w, &res);
                let r = row(
                    arch,
                    w,
                    &m,
                    vec![
                        ("scale", json!(scale)),
                        ("seed", json!(seed)),
                        ("q", fl(q)),
                        ("alloc", json!(format!("uniform-q{q:.2}"))),
                    ],
                );
                print_row(&format!("{seed} q={q:.2}"), arch, &r);
                r
            })
        },
    );
    append_rows("e3c_q_sweep.jsonl", &rows)?;
    Ok(rows)
}

/// E4 - users per team.
pub fn e4_fanin(cal: &Cal, scale: usize, seeds: &[u64]) -> Result<Vec<Row>> {
    let alloc = alloc_back_loaded();
    let specs: Vec<(u64, i64)> =
        seeds.iter().flat_map(|&d| [6, 8, 10, 12, 15].into_iter().map(move |t| (d, t))).collect();
    let rows = over_worlds(
        &specs,
        |&(d, ts)| {
            let mut fan = default_fanin();
            fan[TEAM].0 = ts as f64;
            runner::build_world(scale, d, &WorldOpts { fanin: Some(fan), ..Default::default() })
        },
        |&(seed, ts), w| {
            w.user_layer(&alloc[USER], seed);
            par_rows(&["H_mycelic_full", "E_hier_lineage", "B2_map_reduce"], |arch| {
                let res = run_named(cal, arch, w, &alloc, seed, None);
                let m = runner::evaluate(w, &res);
                let r = row(
                    arch,
                    w,
                    &m,
                    vec![
                        ("scale", json!(scale)),
                        ("seed", json!(seed)),
                        ("team_size", json!(ts)),
                        ("n_teams", json!(w.corpus.org.levels[TEAM].len())),
                        ("alloc", json!("back-loaded")),
                    ],
                );
                print_row(&format!("{seed} team={ts:2}"), arch, &r);
                r
            })
        },
    );
    append_rows("e4_fanin.jsonl", &rows)?;
    Ok(rows)
}

/// E4b - teams per department.
pub fn e4b_dept_fanin(cal: &Cal, scale: usize, seeds: &[u64]) -> Result<Vec<Row>> {
    let alloc = alloc_back_loaded();
    let specs: Vec<(u64, i64)> = seeds.iter().flat_map(|&d| [4, 7, 10, 14].into_iter().map(move |t| (d, t))).collect();
    let rows = over_worlds(
        &specs,
        |&(d, tpd)| {
            let mut fan = default_fanin();
            fan[DEPT].0 = tpd as f64;
            runner::build_world(scale, d, &WorldOpts { fanin: Some(fan), ..Default::default() })
        },
        |&(seed, tpd), w| {
            w.user_layer(&alloc[USER], seed);
            par_rows(&["H_mycelic_full", "E_hier_lineage"], |arch| {
                let res = run_named(cal, arch, w, &alloc, seed, None);
                let m = runner::evaluate(w, &res);
                let r = row(
                    arch,
                    w,
                    &m,
                    vec![
                        ("scale", json!(scale)),
                        ("seed", json!(seed)),
                        ("teams_per_dept", json!(tpd)),
                        ("n_depts", json!(w.corpus.org.levels[DEPT].len())),
                        ("alloc", json!("back-loaded")),
                    ],
                );
                print_row(&format!("{seed} t/dept={tpd:2}"), arch, &r);
                r
            })
        },
    );
    append_rows("e4b_dept_fanin.jsonl", &rows)?;
    Ok(rows)
}

/// The world options of an E5 condition.
pub fn adversarial_world(spec: &Value) -> WorldOpts {
    let mut o = WorldOpts::default();
    if let Some(c) = spec.get("corpus").and_then(Value::as_object) {
        let mut cfg = CorpusCfg::default();
        for (k, v) in c {
            cfg.set(k, v);
        }
        o.corpus = Some(cfg);
    }
    if let Some(n) = spec.get("org").and_then(|x| x.get("n_regions")).and_then(Value::as_u64) {
        o.n_regions = Some(n as usize);
    }
    o
}

/// E5 - adversarial conditions.
pub fn e5_adversarial(cal: &Cal, scale: usize, seeds: &[u64]) -> Result<Vec<Row>> {
    let alloc = alloc_back_loaded();
    let conds = adversarial();
    let specs: Vec<(u64, usize)> = seeds.iter().flat_map(|&d| (0..conds.len()).map(move |c| (d, c))).collect();
    let rows = over_worlds(
        &specs,
        |&(d, c)| runner::build_world(scale, d, &adversarial_world(&conds[c].1)),
        |&(seed, c), w| {
            let (cond, spec) = &conds[c];
            let cfg = spec.get("cfg").and_then(Value::as_object).cloned();
            w.user_layer(&alloc[USER], seed);
            par_rows(&["H_mycelic_full", "B2_map_reduce", "B_long_context", "E_hier_lineage"], |arch| {
                let res = run_named(cal, arch, w, &alloc, seed, cfg.as_ref());
                let m = runner::evaluate(w, &res);
                let r = row(
                    arch,
                    w,
                    &m,
                    vec![
                        ("scale", json!(scale)),
                        ("seed", json!(seed)),
                        ("condition", json!(cond)),
                        ("alloc", json!("back-loaded")),
                    ],
                );
                print_row(&format!("{seed} {cond:24}"), arch, &r);
                r
            })
        },
    );
    append_rows("e5_adversarial.jsonl", &rows)?;
    Ok(rows)
}

/// E6 - cost / quality frontier: the hierarchy's question budget and map-reduce's kernel budget.
pub fn e6_frontier(cal: &Cal, scale: usize, seeds: &[u64]) -> Result<Vec<Row>> {
    let alloc = alloc_back_loaded();
    // the map-reduce frontier ran after the hierarchy's runs had installed the hierarchy's ranker
    let ranker = cal.ranker_for("H_mycelic_full");
    #[derive(Clone, Copy)]
    enum Job {
        Qf(f64),
        Mr(usize),
    }
    let mut jobs: Vec<Job> = [0.0, 0.05, 0.15, 0.35, 0.55, 1.0].into_iter().map(Job::Qf).collect();
    jobs.extend([300usize, 900, 2500, 8000, 20000].into_iter().map(Job::Mr));
    let rows = over_worlds(
        seeds,
        |&d| runner::build_world_default(scale, d),
        |&seed, w| {
            let ul = w.user_layer(&alloc[USER], seed);
            par_rows(&jobs, |job| match *job {
                Job::Qf(qf) => {
                    let cfg = obj(&json!({"downward_retrieval": true, "questions": qf > 0.0, "cross_links": true,
                                      "question_frac": qf.max(1e-6)}));
                    let res = run_hier(cal, w, &alloc, seed, &cfg);
                    let m = runner::evaluate(w, &res);
                    let r = row(
                        "H_frontier",
                        w,
                        &m,
                        vec![
                            ("scale", json!(scale)),
                            ("seed", json!(seed)),
                            ("question_frac", fl(qf)),
                            ("alloc", json!("back-loaded")),
                            ("knob", json!("question_frac")),
                        ],
                    );
                    print_row(&format!("{seed} qfrac={qf:.2}"), "H_frontier", &r);
                    r
                }
                Job::Mr(b) => {
                    let res = systems::map_reduce(&w.corpus, &alloc, seed, b, &ul, true, ranker.as_ref());
                    let m = runner::evaluate(w, &res);
                    let r = row(
                        "B2_frontier",
                        w,
                        &m,
                        vec![
                            ("scale", json!(scale)),
                            ("seed", json!(seed)),
                            ("mr_budget", json!(b)),
                            ("alloc", json!("back-loaded")),
                            ("knob", json!("mr_budget")),
                        ],
                    );
                    print_row(&format!("{seed} mr_budget={b}"), "B2_frontier", &r);
                    r
                }
            })
        },
    );
    append_rows("e6_frontier.jsonl", &rows)?;
    Ok(rows)
}

/// E7 - capability-shape sensitivity: the allocation conclusion under convex, linear and concave hard skills.
pub fn e7_shape(cal: &Cal, scale: usize, seeds: &[u64]) -> Result<Vec<Row>> {
    let shapes = [HardShape::Convex, HardShape::Linear, HardShape::Concave];
    let specs: Vec<(HardShape, u64)> = shapes.iter().flat_map(|&s| seeds.iter().map(move |&d| (s, d))).collect();
    let jobs: Vec<(&str, &str)> =
        ["flat-small", "naive-ladder", "back-loaded", "kernel-only", "two-step", "front-loaded"]
            .into_iter()
            .flat_map(|a| ["H_mycelic_full", "B2_map_reduce"].into_iter().map(move |n| (a, n)))
            .collect();
    let rows = over_worlds(
        &specs,
        |&(_, d)| runner::build_world_default(scale, d),
        |&(shape, seed), w| {
            par_rows(&jobs, |&(aname, arch)| {
                let alloc = models::allocation(aname, shape);
                let res = run_named(cal, arch, w, &alloc, seed, None);
                let m = runner::evaluate(w, &res);
                let r = row(
                    arch,
                    w,
                    &m,
                    vec![
                        ("scale", json!(scale)),
                        ("seed", json!(seed)),
                        ("shape", json!(shape.name())),
                        ("alloc", json!(aname)),
                    ],
                );
                print_row(&format!("{:8} {seed} {aname:14}", shape.name()), arch, &r);
                r
            })
        },
    );
    append_rows("e7_shape.jsonl", &rows)?;
    Ok(rows)
}

/// E8 - semantic cross-links against the strict org tree.
pub fn e8_crosslinks(cal: &Cal, scales: &[usize], seeds: &[u64]) -> Result<Vec<Row>> {
    let alloc = alloc_back_loaded();
    let specs: Vec<(usize, u64)> = scales.iter().flat_map(|&s| seeds.iter().map(move |&d| (s, d))).collect();
    let rows = over_worlds(
        &specs,
        |&(s, d)| runner::build_world_default(s, d),
        |&(scale, seed), w| {
            w.user_layer(&alloc[USER], seed);
            par_rows(&[(false, 0i64), (true, 4)], |&(xl, deg)| {
                let cfg = obj(&json!({"downward_retrieval": true, "questions": true, "cross_links": xl,
                                  "cross_link_degree": 1.max(deg)}));
                let res = run_hier(cal, w, &alloc, seed, &cfg);
                let m = runner::evaluate(w, &res);
                let r = row(
                    "H",
                    w,
                    &m,
                    vec![
                        ("scale", json!(scale)),
                        ("seed", json!(seed)),
                        ("cross_links", json!(xl)),
                        ("degree", json!(deg)),
                        ("alloc", json!("back-loaded")),
                    ],
                );
                print_row(&format!("{scale} {seed} xlinks={xl}"), "H", &r);
                r
            })
        },
    );
    append_rows("e8_crosslinks.jsonl", &rows)?;
    Ok(rows)
}

/// E1d - the centralised-triage control with its calibrated evidence budget (no ranker: see the module notes).
pub fn e1d_b4_capped(cal: &Cal, scales: &[usize], seeds: &[u64]) -> Result<Vec<Row>> {
    let alloc = alloc_back_loaded();
    let cap = cal.int("ct_kernel_ko_cap") as usize;
    let specs: Vec<(usize, u64)> = scales.iter().flat_map(|&s| seeds.iter().map(move |&d| (s, d))).collect();
    let rows = over_worlds(
        &specs,
        |&(s, d)| runner::build_world_default(s, d),
        |&(scale, seed), w| {
            let ul = w.user_layer(&alloc[USER], seed);
            let res = systems::central_triage(&w.corpus, &alloc, seed, &ul, 2, 0.35, 6000, cap, None);
            let m = runner::evaluate(w, &res);
            let r = row(
                "B4_central_triage_capped",
                w,
                &m,
                vec![
                    ("scale", json!(scale)),
                    ("seed", json!(seed)),
                    ("alloc", json!("back-loaded")),
                    ("kernel_ko_cap", json!(cap)),
                ],
            );
            print_row(&format!("{scale:6}/{seed}"), "B4_capped", &r);
            vec![r]
        },
    );
    append_rows("e1d_b4_capped.jsonl", &rows)?;
    Ok(rows)
}

/// E10 - one scale beyond the brief (100,000 users).
pub fn e10_scale_trend(cal: &Cal, scales: &[usize], seeds: &[u64]) -> Result<Vec<Row>> {
    let names = [
        "A_flat_rag",
        "B_long_context",
        "B2_map_reduce",
        "A2_chunked_ctx",
        "E_hier_lineage",
        "G_hier_questions",
        "H_mycelic_full",
        "J_mycelic_verified",
        "Y_oracle_retrieval",
    ];
    e1b_extra(cal, scales, seeds, &names, "e10_scale_trend.jsonl")
}

/// E12 - provenance integrity: does a report's cited evidence name the entity the report is about?
pub fn e12_provenance(cal: &Cal, scales: &[usize], seeds: &[u64]) -> Result<Vec<Row>> {
    let alloc = alloc_back_loaded();
    let names = [
        "A_flat_rag",
        "A2_chunked_ctx",
        "B2_map_reduce",
        "B4_central_triage",
        "E_hier_lineage",
        "H_mycelic_full",
        "J_mycelic_verified",
    ];
    let specs: Vec<(usize, u64)> = scales.iter().flat_map(|&s| seeds.iter().map(move |&d| (s, d))).collect();
    let rows = over_worlds(
        &specs,
        |&(s, d)| runner::build_world_default(s, d),
        |&(scale, seed), w| {
            w.user_layer(&alloc[USER], seed);
            par_rows(&names, |a| {
                let res = run_named(cal, a, w, &alloc, seed, None);
                let recs = &w.corpus.recs;
                let (mut ok, mut tot) = (0usize, 0usize);
                let mut per_hyp: Vec<f64> = Vec::new();
                for &i in res.hypotheses.iter().take(400) {
                    let hy = &res.store[i];
                    let ev: Vec<usize> = hy
                        .evidence
                        .iter()
                        .filter(|&&e| e >= 0 && (e as usize) < recs.len())
                        .map(|&e| e as usize)
                        .collect();
                    if ev.is_empty() {
                        continue;
                    }
                    let m: Vec<f64> =
                        ev.iter().map(|&e| if recs.anchor[e] as i64 == hy.anchor { 1.0 } else { 0.0 }).collect();
                    ok += m.iter().filter(|&&x| x > 0.0).count();
                    tot += m.len();
                    per_hyp.push(crate::np::math::mean(&m));
                }
                let frac = |pred: &dyn Fn(f64) -> bool| -> f64 {
                    if per_hyp.is_empty() {
                        0.0
                    } else {
                        let v: Vec<f64> = per_hyp.iter().map(|&x| if pred(x) { 1.0 } else { 0.0 }).collect();
                        crate::np::math::mean(&v)
                    }
                };
                let m = runner::evaluate(w, &res);
                let r = row(
                    a,
                    w,
                    &m,
                    vec![
                        ("scale", json!(scale)),
                        ("seed", json!(seed)),
                        ("alloc", json!("back-loaded")),
                        ("evidence_names_claimed_entity", fl(ok as f64 / 1.max(tot) as f64)),
                        ("reports_with_no_matching_evidence", fl(frac(&|x| x == 0.0))),
                        ("reports_fully_attributed", fl(frac(&|x| x >= 0.999))),
                        ("n_reports_checked", json!(per_hyp.len())),
                    ],
                );
                println!(
                    "  {scale:6}/{seed} {a:22} evidence-names-entity={} fully-attributed={} none-attributed={}",
                    crate::py::fmt::f(crate::py::value::f(&r, "evidence_names_claimed_entity"), ".3f"),
                    crate::py::fmt::f(crate::py::value::f(&r, "reports_fully_attributed"), ".3f"),
                    crate::py::fmt::f(crate::py::value::f(&r, "reports_with_no_matching_evidence"), ".3f"),
                );
                r
            })
        },
    );
    append_rows("e12_provenance.jsonl", &rows)?;
    Ok(rows)
}

/// E9 - privacy / propagation-volume accounting (raw text, claims, sketch metadata).
pub fn e9_privacy(cal: &Cal, scales: &[usize], seeds: &[u64]) -> Result<Vec<Row>> {
    let alloc = alloc_back_loaded();
    let names: Vec<&str> = archs().iter().map(|a| a.0).collect();
    let specs: Vec<(usize, u64)> = scales.iter().flat_map(|&s| seeds.iter().map(move |&d| (s, d))).collect();
    let rows = over_worlds(
        &specs,
        |&(s, d)| runner::build_world_default(s, d),
        |&(scale, seed), w| {
            w.user_layer(&alloc[USER], seed);
            par_rows(&names, |a| {
                let res = run_named(cal, a, w, &alloc, seed, None);
                let m = runner::evaluate(w, &res);
                let r = row(
                    a,
                    w,
                    &m,
                    vec![("scale", json!(scale)), ("seed", json!(seed)), ("alloc", json!("back-loaded"))],
                );
                print_row(&format!("{scale:6}/{seed}"), a, &r);
                r
            })
        },
    );
    append_rows("e9_privacy.jsonl", &rows)?;
    Ok(rows)
}

/// The experiments by CLI name (`ALL`), in the Python's order.
pub const ALL: [&str; 18] = [
    "e1", "e1b", "e1c", "e2", "e3", "e3b", "e3c", "e4", "e4b", "e5", "e6", "e7", "e8", "e9", "e10", "e11", "e1d", "e12",
];

/// Run one experiment by name with its default parameters.
pub fn run_one(cal: &Cal, name: &str) -> Result<()> {
    let seeds3 = [0u64, 1, 2];
    match name {
        "e1" => e1_baselines(cal, &SCALES, &EVAL_SEEDS, "e1_baselines.jsonl").map(drop),
        "e1c" => e1_baselines(cal, &[2_000, 10_000], &[5, 6, 7, 8, 9], "e1_baselines.jsonl").map(drop),
        "e1b" => e1b_extra(
            cal,
            &SCALES,
            &EVAL_SEEDS,
            &["A2_chunked_ctx", "B4_central_triage", "I_mycelic_completion", "Z2_naive_enumerate"],
            "e1b_extra.jsonl",
        )
        .map(drop),
        "e11" => e1b_extra(cal, &SCALES, &EVAL_SEEDS, &["J_mycelic_verified"], "e1b_extra.jsonl").map(drop),
        "e2" => e2_ablations(cal, 10_000, &EVAL_SEEDS).map(drop),
        "e3" => e3_allocation(cal, 10_000, &seeds3).map(drop),
        "e3b" => e3b_level_marginal(cal, 10_000, &seeds3).map(drop),
        "e3c" => e3c_q_sweep(cal, 10_000, &seeds3).map(drop),
        "e4" => e4_fanin(cal, 10_000, &seeds3).map(drop),
        "e4b" => e4b_dept_fanin(cal, 10_000, &seeds3).map(drop),
        "e5" => e5_adversarial(cal, 10_000, &seeds3).map(drop),
        "e6" => e6_frontier(cal, 10_000, &seeds3).map(drop),
        "e7" => e7_shape(cal, 10_000, &seeds3).map(drop),
        "e8" => e8_crosslinks(cal, &SCALES, &seeds3).map(drop),
        "e9" => e9_privacy(cal, &SCALES, &[0, 1]).map(drop),
        "e10" => e10_scale_trend(cal, &[100_000], &seeds3).map(drop),
        "e1d" => e1d_b4_capped(cal, &SCALES, &EVAL_SEEDS).map(drop),
        "e12" => e12_provenance(cal, &[10_000, 50_000], &seeds3).map(drop),
        other => bail!("unknown experiment {other:?}; known: {ALL:?}"),
    }
}

/// `python3 -m research.mycelic.experiments e1 e2 ...` (default e1).  Each experiment loads the calibration
/// afresh, as each ran in its own process.
pub fn main(args: &[String]) -> Result<()> {
    let which: Vec<String> = if args.is_empty() { vec!["e1".into()] } else { args.to_vec() };
    for w in &which {
        println!("=== {w} ===");
        let t = Instant::now();
        run_one(&Cal::load(), w)?;
        println!("=== {w} done in {:.0}s ===", t.elapsed().as_secs_f64());
    }
    Ok(())
}
