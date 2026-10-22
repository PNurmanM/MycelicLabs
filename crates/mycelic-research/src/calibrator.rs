//! Learned confidence calibration over kernel-side features (`calibrator.py`).
//!
//! Dumps every candidate a synthesis produced (pre-register-cut), labels it against gold, fits a logistic (or a
//! small gradient-boosted tree ensemble) on the CALIBRATION seeds (500-502) and measures how much discovery a
//! better ranking alone recovers on the evaluation seeds under the same register cap.  Nothing here reads raw
//! text: every feature is a statistic the kernel already holds.
//!
//! Reproducibility: the dumps, the tree ensemble, leave-one-seed-out selection and per-architecture adoption
//! are bit-exact with the Python.  The logistic's Newton iterations went through BLAS (`@`) and LAPACK
//! (`np.linalg.solve`), whose summation order is the library's own; here they are plain dot products and an LU
//! solve with partial pivoting, so fitted weights agree with the Python's to rounding, not to the bit.

use crate::evalm::match_sets;
use crate::models;
use crate::np::math;
use crate::ops::{FEATURES, Hypothesis, Ranker, stem_rep_map};
use crate::org::USER;
use crate::py::json as pyjson;
use crate::py::value::{Row, fl, num};
use crate::runner::{self, Cal, arch_spec, archs, art};
use crate::systems::HierRunner;
use anyhow::{Context, Result};
use serde_json::{Map, Value, json};
use std::io::Write as _;
use std::time::Instant;

pub const CAL_SEEDS: [u64; 3] = [500, 501, 502];
pub const EVAL_SEEDS: [u64; 5] = [0, 1, 2, 3, 4];
pub const TAU: f64 = 0.5;

/// A small, fixed set of interactions chosen a priori from the mechanism, not searched over.
pub const INTERACTIONS: [(&str, &str); 6] = [
    ("verified", "min_sup"),
    ("dispersion", "n_sites"),
    ("n_links", "min_sup"),
    ("tspan", "n_links"),
    ("from_question", "n_sites"),
    ("rank_in_anchor", "n_same_anchor"),
];

/// `_features(h)`: the candidate's features in `FEATURES` order (hand_conf = conf when the candidate has none).
fn features(h: &Hypothesis) -> Vec<(&'static str, f64)> {
    match &h.feat {
        Some(f) => FEATURES.iter().map(|&k| (k, f.get(k))).collect(),
        None => FEATURES.iter().map(|&k| (k, if k == "hand_conf" { h.conf } else { 0.0 })).collect(),
    }
}

/// A dense row-major matrix.
#[derive(Clone, Debug)]
pub struct Mat {
    pub n: usize,
    pub d: usize,
    pub a: Vec<f64>,
}

impl Mat {
    fn row(&self, i: usize) -> &[f64] {
        &self.a[i * self.d..(i + 1) * self.d]
    }
}

/// `_design(rows, interactions)`
pub fn design(rows: &[Row], interactions: bool) -> Mat {
    let d0 = FEATURES.len();
    let d = d0 + if interactions { INTERACTIONS.len() } else { 0 };
    let mut a = Vec::with_capacity(rows.len() * d);
    for r in rows {
        let x: Vec<f64> = FEATURES.iter().map(|f| r.get(*f).and_then(num).unwrap_or(0.0)).collect();
        a.extend_from_slice(&x);
        if interactions {
            for (p, q) in INTERACTIONS {
                let ip = FEATURES.iter().position(|f| *f == p).unwrap();
                let iq = FEATURES.iter().position(|f| *f == q).unwrap();
                a.push(x[ip] * x[iq]);
            }
        }
    }
    Mat { n: rows.len(), d, a }
}

/// `_standardise(X, mu, sd)`: column means and standard deviations accumulate row by row, as numpy reduces
/// over axis 0 of a C-ordered matrix; a constant column gets sd 1.
pub fn standardise(x: &Mat, given: Option<(&[f64], &[f64])>) -> (Mat, Vec<f64>, Vec<f64>) {
    let (mu, sd) = match given {
        Some((m, s)) => (m.to_vec(), s.to_vec()),
        None => {
            let n = x.n as f64;
            let mut sum = vec![0.0; x.d];
            for i in 0..x.n {
                for (s, v) in sum.iter_mut().zip(x.row(i)) {
                    *s += v;
                }
            }
            let mu: Vec<f64> = sum.iter().map(|s| s / n).collect();
            let mut sq = vec![0.0; x.d];
            for i in 0..x.n {
                for ((s, v), m) in sq.iter_mut().zip(x.row(i)).zip(&mu) {
                    let dv = v - m;
                    *s += dv * dv;
                }
            }
            let sd: Vec<f64> = sq.iter().map(|s| (s / n).sqrt()).map(|s| if s < 1e-6 { 1.0 } else { s }).collect();
            (mu, sd)
        }
    };
    let mut a = x.a.clone();
    for i in 0..x.n {
        for j in 0..x.d {
            a[i * x.d + j] = (a[i * x.d + j] - mu[j]) / sd[j];
        }
    }
    (Mat { n: x.n, d: x.d, a }, mu, sd)
}

// Matrix code indexes rows and columns by number; iterator chains would obscure the algebra.
#[allow(clippy::needless_range_loop)]
fn solve(mut h: Vec<Vec<f64>>, mut g: Vec<f64>) -> Vec<f64> {
    let n = g.len();
    for col in 0..n {
        let piv = (col..n).max_by(|&a, &b| h[a][col].abs().partial_cmp(&h[b][col].abs()).unwrap()).unwrap();
        h.swap(col, piv);
        g.swap(col, piv);
        let p = h[col][col];
        for r in col + 1..n {
            let f = h[r][col] / p;
            if f != 0.0 {
                for c in col..n {
                    h[r][c] -= f * h[col][c];
                }
                g[r] -= f * g[col];
            }
        }
    }
    let mut x = vec![0.0; n];
    for r in (0..n).rev() {
        let mut s = g[r];
        for c in r + 1..n {
            s -= h[r][c] * x[c];
        }
        x[r] = s / h[r][r];
    }
    x
}

/// `fit_logistic(X, y, l2, iters=60, row_weight)`: Newton's method with L2 and class weights.
#[allow(clippy::needless_range_loop)]
pub fn fit_logistic(x: &Mat, y: &[f64], l2: f64, row_weight: Option<&[f64]>) -> Vec<f64> {
    let (n, d) = (x.n, x.d);
    let mut w = vec![0.0; d + 1];
    let mut reg = vec![l2; d + 1];
    reg[0] = 0.0;
    // class weights so a 5% positive rate does not collapse to "always no"
    let pos = 1.max(math::sum(y) as i64) as f64;
    let mut wt: Vec<f64> = y.iter().map(|&v| if v > 0.0 { (n as f64 - pos) / pos } else { 1.0 }).collect();
    if let Some(rw) = row_weight {
        for (a, b) in wt.iter_mut().zip(rw) {
            *a *= b;
        }
    }
    let xb = |i: usize, j: usize| if j == 0 { 1.0 } else { x.a[i * d + j - 1] };
    for _ in 0..60 {
        let p: Vec<f64> = (0..n)
            .map(|i| {
                let mut z = 0.0;
                for (j, wj) in w.iter().enumerate() {
                    z += xb(i, j) * wj;
                }
                1.0 / (1.0 + (-z).exp())
            })
            .collect();
        let mut g = vec![0.0; d + 1];
        for i in 0..n {
            let r = wt[i] * (p[i] - y[i]);
            for (j, gj) in g.iter_mut().enumerate() {
                *gj += xb(i, j) * r;
            }
        }
        for j in 0..=d {
            g[j] += reg[j] * w[j];
        }
        let s: Vec<f64> = (0..n).map(|i| wt[i] * p[i] * (1.0 - p[i])).collect();
        let mut h = vec![vec![0.0; d + 1]; d + 1];
        for i in 0..n {
            for a in 0..=d {
                let xa = xb(i, a) * s[i];
                if xa == 0.0 {
                    continue;
                }
                for b in 0..=d {
                    h[a][b] += xa * xb(i, b);
                }
            }
        }
        for j in 0..=d {
            h[j][j] += reg[j];
        }
        let step = solve(h, g);
        for (wj, sj) in w.iter_mut().zip(&step) {
            *wj -= sj;
        }
        if step.iter().fold(0.0f64, |m, s| m.max(s.abs())) < 1e-7 {
            break;
        }
    }
    w
}

pub fn predict(x: &Mat, w: &[f64]) -> Vec<f64> {
    (0..x.n)
        .map(|i| {
            let mut z = w[0];
            for (j, v) in x.row(i).iter().enumerate() {
                z += v * w[j + 1];
            }
            1.0 / (1.0 + (-z).exp())
        })
        .collect()
}

// ---- gradient-boosted depth-2 trees on histogram splits (logistic loss) ----

#[derive(Clone, Debug)]
pub enum Node {
    Leaf(f64),
    Split { f: usize, t: f64, l: Box<Node>, r: Box<Node> },
}

impl Node {
    pub fn to_json(&self) -> Value {
        match self {
            Node::Leaf(v) => json!({ "leaf": fl(*v) }),
            Node::Split { f, t, l, r } => json!({"f": f, "t": fl(*t), "l": l.to_json(), "r": r.to_json()}),
        }
    }
}

fn sum_sel(v: &[f64], idx: &[usize]) -> f64 {
    let s: Vec<f64> = idx.iter().map(|&i| v[i]).collect();
    math::sum(&s)
}

fn best_split(x: &Mat, rows: &[usize], g: &[f64], h: &[f64], lam: f64) -> (f64, i64, f64) {
    let mut best = (0.0, -1i64, 0.0);
    let big_g = sum_sel(g, rows);
    let big_h = sum_sel(h, rows);
    let base = big_g * big_g / (big_h + lam);
    let qgrid = math::linspace(0.05, 0.95, 19);
    for j in 0..x.d {
        let mut col: Vec<f64> = rows.iter().map(|&i| x.a[i * x.d + j]).collect();
        col.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mut qs: Vec<f64> = qgrid.iter().map(|&q| math::quantile_sorted(&col, q)).collect();
        qs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        qs.dedup();
        for &t in &qs {
            let left: Vec<usize> = rows.iter().copied().filter(|&i| x.a[i * x.d + j] <= t).collect();
            let gl = sum_sel(g, &left);
            let hl = sum_sel(h, &left);
            if hl < 1e-9 || (big_h - hl) < 1e-9 {
                continue;
            }
            let gain = gl * gl / (hl + lam) + crate::py::pow(big_g - gl, 2.0) / (big_h - hl + lam) - base;
            if gain > best.0 {
                best = (gain, j as i64, t);
            }
        }
    }
    best
}

fn grow(x: &Mat, rows: &[usize], g: &[f64], h: &[f64], depth: usize, lam: f64) -> Node {
    let leaf = || Node::Leaf(-sum_sel(g, rows) / (sum_sel(h, rows) + lam));
    if depth == 0 || rows.len() < 20 {
        return leaf();
    }
    let (gain, j, t) = best_split(x, rows, g, h, lam);
    if j < 0 || gain <= 0.0 {
        return leaf();
    }
    let j = j as usize;
    let (l, r): (Vec<usize>, Vec<usize>) = rows.iter().partition(|&&i| x.a[i * x.d + j] <= t);
    Node::Split {
        f: j,
        t,
        l: Box::new(grow(x, &l, g, h, depth - 1, lam)),
        r: Box::new(grow(x, &r, g, h, depth - 1, lam)),
    }
}

fn tree_pred(node: &Node, x: &Mat, i: usize) -> f64 {
    match node {
        Node::Leaf(v) => *v,
        Node::Split { f, t, l, r } => {
            if x.a[i * x.d + f] <= *t {
                tree_pred(l, x, i)
            } else {
                tree_pred(r, x, i)
            }
        }
    }
}

/// `fit_gbdt(X, y, rounds=150, lr=0.1, depth=2, lam=1.0)`
pub fn fit_gbdt(x: &Mat, y: &[f64], rounds: usize, lr: f64, row_weight: Option<&[f64]>) -> Vec<Node> {
    let n = y.len();
    let pos = 1.max(math::sum(y) as i64) as f64;
    let wpos = (n as f64 - pos) / pos;
    let mut wt: Vec<f64> = y.iter().map(|&v| if v > 0.0 { wpos } else { 1.0 }).collect();
    if let Some(rw) = row_weight {
        for (a, b) in wt.iter_mut().zip(rw) {
            *a *= b;
        }
    }
    let mut f = vec![0.0; n];
    let all: Vec<usize> = (0..n).collect();
    let mut trees = Vec::new();
    for _ in 0..rounds {
        let p: Vec<f64> = f.iter().map(|&v: &f64| 1.0 / (1.0 + (-v).exp())).collect();
        let g: Vec<f64> = (0..n).map(|i| wt[i] * (p[i] - y[i])).collect();
        let h: Vec<f64> = (0..n).map(|i| wt[i] * p[i] * (1.0 - p[i])).collect();
        let t = grow(x, &all, &g, &h, 2, 1.0);
        for (i, fi) in f.iter_mut().enumerate() {
            *fi += lr * tree_pred(&t, x, i);
        }
        trees.push(t);
    }
    trees
}

pub fn gbdt_predict(trees: &[Node], lr: f64, x: &Mat) -> Vec<f64> {
    let mut f = vec![0.0; x.n];
    for t in trees {
        for (i, fi) in f.iter_mut().enumerate() {
            *fi += lr * tree_pred(t, x, i);
        }
    }
    f.iter().map(|&v: &f64| 1.0 / (1.0 + (-v).exp())).collect()
}

/// `auc(score, y)`: ranks from numpy's default argsort (ties broken its way), NaN without both classes.
pub fn auc(score: &[f64], y: &[f64]) -> f64 {
    let order = math::argsort_quick(score);
    let mut ranks = vec![0.0; score.len()];
    for (r, &i) in order.iter().enumerate() {
        ranks[i] = (r + 1) as f64;
    }
    let npos = math::sum(y);
    let nneg = y.len() as f64 - npos;
    if npos == 0.0 || nneg == 0.0 {
        return f64::NAN;
    }
    let pos_ranks: Vec<f64> = (0..y.len()).filter(|&i| y[i] > 0.0).map(|i| ranks[i]).collect();
    (math::sum(&pos_ranks) - npos * (npos + 1.0) / 2.0) / (npos * nneg)
}

/// `_found_under_cap`: (found, rare found, kept) if each run's candidates were ranked by `score`, kept while
/// `score >= thresh` and cut at the run's own register cap.
fn found_under_cap(rows: &[&Row], score: &[f64], thresh: f64) -> (usize, usize, usize) {
    let mut by_run: Vec<((String, i64), Vec<usize>)> = Vec::new();
    for (i, r) in rows.iter().enumerate() {
        let k = (r["arch"].as_str().unwrap().to_string(), r["seed"].as_i64().unwrap());
        match by_run.iter_mut().find(|e| e.0 == k) {
            Some(e) => e.1.push(i),
            None => by_run.push((k, vec![i])),
        }
    }
    let (mut found, mut rare_found, mut n_kept) = (0, 0, 0);
    for (_, mut idx) in by_run {
        idx.sort_by(|&a, &b| (-score[a]).partial_cmp(&-score[b]).unwrap());
        let cap = rows[idx[0]]["cap"].as_u64().unwrap() as usize;
        let kept: Vec<usize> = idx.into_iter().filter(|&i| score[i] >= thresh).take(cap).collect();
        n_kept += kept.len();
        let mut seen: Vec<i64> = Vec::new();
        for i in kept {
            if rows[i]["gold"] == json!(true) {
                let pid = rows[i]["pid"].as_i64().unwrap();
                if !seen.contains(&pid) {
                    seen.push(pid);
                    found += 1;
                    rare_found += usize::from(rows[i]["rare"] == json!(true));
                }
            }
        }
    }
    (found, rare_found, n_kept)
}

fn row_weight(rows: &[&Row], decoy_weight: f64) -> Vec<f64> {
    rows.iter().map(|r| if r.get("decoy") == Some(&json!(true)) { decoy_weight } else { 1.0 }).collect()
}

fn labels(rows: &[&Row]) -> Vec<f64> {
    rows.iter().map(|r| if r["gold"] == json!(true) { 1.0 } else { 0.0 }).collect()
}

fn seed_of(r: &Row) -> u64 {
    r["seed"].as_u64().unwrap()
}

/// `hyp_features<tag>.jsonl`
pub fn dump_path(tag: &str) -> std::path::PathBuf {
    art().join(format!("hyp_features{tag}.jsonl"))
}

pub fn load_dump(tag: &str) -> Result<Vec<Row>> {
    let p = dump_path(tag);
    let text = std::fs::read_to_string(&p).with_context(|| format!("{}", p.display()))?;
    Ok(text
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str::<Map<String, Value>>(l).unwrap())
        .collect())
}

/// `dump(...)`: one row per candidate (pre-register-cut for the hierarchy), labelled against gold.  The ranker
/// is off for the hierarchy (its features must be the un-ranked ones); a non-hierarchy architecture installs
/// its own choice, and - as in the Python, where that choice stayed in a module global - a hierarchy dumped
/// after it in the same run inherits it.
pub fn dump(
    cal: &Cal,
    scale: usize,
    seeds: &[u64],
    arch_list: &[&str],
    cfg_over: Option<&Map<String, Value>>,
    tag: &str,
) -> Result<()> {
    let alloc = models::allocation_default("back-loaded");
    let mut fh = std::fs::File::create(dump_path(tag))?;
    let mut current: Option<Ranker> = None;
    for &seed in seeds {
        let t0 = Instant::now();
        let w = runner::build_world_default(scale, seed);
        let stem = stem_rep_map(&w.corpus);
        let real: Vec<&crate::corpus::Pattern> = w.gold.discoverable.iter().map(|&i| &w.corpus.patterns[i]).collect();
        let decoys: Vec<&crate::corpus::Pattern> = w.corpus.patterns.iter().filter(|p| !p.real).collect();
        for &arch in arch_list {
            let (kind, spec) = arch_spec(arch).expect("registered architecture");
            let (res, list, cap) = if kind == "hier" {
                let mut over = spec.as_object().cloned().unwrap_or_default();
                if let Some(c) = cfg_over {
                    over.extend(c.clone());
                }
                let cfg = cal.hier_cfg(&over);
                let ul = w.user_layer(&alloc[USER], seed);
                let (res, state) =
                    HierRunner::new(&w.corpus, alloc.clone(), cfg, seed, &ul, &w.near_miss, current.as_ref()).run();
                let full = res.full_hyps.clone().unwrap();
                (res, full, state.max_reports)
            } else {
                current = cal.ranker_for(arch);
                let r = runner::run_arch(cal, arch, &w, &alloc, seed, None).res;
                let list = r.hypotheses.clone();
                (r, list, 6000.min(600.max(w.corpus.entities.len())))
            };
            let hyps: Vec<&Hypothesis> = list.iter().map(|&i| &res.store[i]).collect();
            let m_real = match_sets(&hyps, &real, &stem, false, false);
            let m_dec = match_sets(&hyps, &decoys, &stem, false, false);
            let mut gold_idx: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
            for (pid, idxs) in &m_real {
                for &i in idxs {
                    gold_idx.insert(i, *pid);
                }
            }
            let dec_idx: std::collections::HashSet<usize> = m_dec.values().flatten().copied().collect();
            for (i, h) in hyps.iter().enumerate() {
                let mut row = Row::new();
                row.insert("arch".into(), json!(arch));
                row.insert("scale".into(), json!(scale));
                row.insert("seed".into(), json!(seed));
                row.insert("i".into(), json!(i));
                row.insert("cap".into(), json!(cap));
                row.insert("conf".into(), fl(h.conf));
                row.insert("gold".into(), json!(gold_idx.contains_key(&i)));
                row.insert("pid".into(), json!(gold_idx.get(&i).map_or(-1, |&p| p as i64)));
                row.insert("decoy".into(), json!(dec_idx.contains(&i)));
                row.insert("anchor".into(), json!(h.anchor));
                row.insert("chain".into(), json!(h.chain));
                row.insert("rare".into(), json!(gold_idx.get(&i).is_some_and(|&p| w.corpus.patterns[p].rare)));
                for (k, v) in features(h) {
                    row.insert(k.into(), fl(v));
                }
                writeln!(fh, "{}", pyjson::dumps_with(&Value::Object(row), pyjson::Style::ASCII))?;
            }
            println!(
                "  seed {seed} {arch:20} {} candidates, {} gold-matching, {} patterns matched ({:.0}s)",
                hyps.len(),
                gold_idx.len(),
                m_real.len(),
                t0.elapsed().as_secs_f64()
            );
        }
    }
    Ok(())
}

/// The model the selection grid chose (`fit_logistic` weights or a tree ensemble).
enum Model {
    Logistic(Vec<f64>),
    Gbdt(Vec<Node>),
}

impl Model {
    fn predict(&self, x: &Mat) -> Vec<f64> {
        match self {
            Model::Logistic(w) => predict(x, w),
            Model::Gbdt(t) => gbdt_predict(t, 0.1, x),
        }
    }
}

/// `_loso_found`: leave-one-calibration-seed-out discovery under the real cap, less a charge per admitted
/// decoy, summed over the held-out seeds (evaluation seeds are never read).
fn loso_found(
    by_tag: &[Vec<Row>],
    l2: f64,
    interactions: bool,
    gbdt: bool,
    decoy_weight: f64,
    decoy_penalty: f64,
) -> f64 {
    let mut total = 0.0;
    for held in CAL_SEEDS {
        let tr: Vec<&Row> =
            by_tag.iter().flatten().filter(|r| CAL_SEEDS.contains(&seed_of(r)) && seed_of(r) != held).collect();
        let tr_owned: Vec<Row> = tr.iter().map(|r| (*r).clone()).collect();
        let (xs, mu, sd) = standardise(&design(&tr_owned, interactions), None);
        let y = labels(&tr);
        let rw = row_weight(&tr, decoy_weight);
        let model = if gbdt {
            Model::Gbdt(fit_gbdt(&xs, &y, 150, 0.1, Some(&rw)))
        } else {
            Model::Logistic(fit_logistic(&xs, &y, l2, Some(&rw)))
        };
        for rows in by_tag {
            let te: Vec<Row> = rows
                .iter()
                .filter(|r| seed_of(r) == held && r["arch"].as_str().unwrap().starts_with("H_"))
                .cloned()
                .collect();
            if te.is_empty() {
                continue;
            }
            let (xt, _, _) = standardise(&design(&te, interactions), Some((&mu, &sd)));
            let p = model.predict(&xt);
            let k = te.iter().filter(|r| r["hand_conf"].as_f64().unwrap() >= 0.5).count();
            let order: Vec<usize> = math::argsort_desc(&p).into_iter().take(k).collect();
            let cap = te[0]["cap"].as_u64().unwrap() as usize;
            let mut seen: Vec<i64> = Vec::new();
            let mut dec = 0usize;
            for &i in order.iter().take(cap) {
                if te[i]["gold"] == json!(true) {
                    let pid = te[i]["pid"].as_i64().unwrap();
                    if !seen.contains(&pid) {
                        seen.push(pid);
                    }
                } else if te[i].get("decoy") == Some(&json!(true)) {
                    dec += 1;
                }
            }
            total += seen.len() as f64 - decoy_penalty * dec as f64 / 1.max(seen.len()) as f64;
        }
    }
    total
}

fn read_cal_file() -> Result<Map<String, Value>> {
    let p = art().join("calibration.json");
    Ok(serde_json::from_str(&std::fs::read_to_string(&p).with_context(|| format!("{}", p.display()))?)?)
}

/// `json.dump(obj, fh, indent=n)`
pub fn write_json(path: &std::path::Path, v: &Value, indent: usize) -> Result<()> {
    std::fs::write(path, pyjson::dumps_with(v, pyjson::Style::indent(indent)))?;
    Ok(())
}

/// `fit_and_store`: fit the shared ranker on the calibration seeds of the dumps and write it into
/// calibration.json.
pub fn fit_and_store(
    tags: &[&str],
    l2: f64,
    interactions: bool,
    gbdt: bool,
    decoy_weight: f64,
) -> Result<Map<String, Value>> {
    let mut tr: Vec<Row> = Vec::new();
    for tag in tags {
        if dump_path(tag).exists() {
            tr.extend(load_dump(tag)?.into_iter().filter(|r| CAL_SEEDS.contains(&seed_of(r))));
        }
    }
    let refs: Vec<&Row> = tr.iter().collect();
    let (xs, mu, sd) = standardise(&design(&tr, interactions), None);
    let y = labels(&refs);
    let rw = row_weight(&refs, decoy_weight);
    let mut names: Vec<String> = FEATURES.iter().map(|s| s.to_string()).collect();
    if interactions {
        names.extend(INTERACTIONS.iter().map(|(a, b)| format!("{a}*{b}")));
    }
    let fv = |v: &[f64]| Value::Array(v.iter().map(|&x| fl(x)).collect());
    let ranker = if gbdt {
        let trees = fit_gbdt(&xs, &y, 150, 0.1, Some(&rw));
        json!({"kind": "gbdt", "features": names, "trees": trees.iter().map(Node::to_json).collect::<Vec<_>>(),
               "lr": 0.1, "bias": 0.0, "weights": vec![0.0; names.len()], "mu": fv(&mu), "sd": fv(&sd),
               "l2": l2, "interactions": interactions, "n_train": tr.len(), "seeds": CAL_SEEDS,
               "train_archs": "all", "source_dumps": tags})
    } else {
        let w = fit_logistic(&xs, &y, l2, Some(&rw));
        json!({"kind": "logistic", "features": names, "bias": fl(w[0]), "weights": fv(&w[1..]),
               "mu": fv(&mu), "sd": fv(&sd), "l2": l2, "interactions": interactions, "decoy_weight": decoy_weight,
               "n_train": tr.len(), "seeds": CAL_SEEDS, "train_archs": "all", "source_dumps": tags})
    };
    let mut cal = read_cal_file().unwrap_or_default();
    cal.insert("ranker".into(), ranker.clone());
    write_json(&art().join("calibration.json"), &Value::Object(cal), 1)?;
    Ok(ranker.as_object().unwrap().clone())
}

/// `select_and_store`: choose l2, interactions (and logistic vs trees) by leave-one-seed-out discovery on the
/// calibration seeds, then fit on all three and store.
pub fn select_and_store(tags: &[&str], decoy_weights: &[f64], decoy_penalty: f64) -> Result<Map<String, Value>> {
    let mut by_tag: Vec<Vec<Row>> = Vec::new();
    for tag in tags {
        if dump_path(tag).exists() {
            by_tag.push(load_dump(tag)?);
        }
    }
    let mut grid: Vec<Value> = Vec::new();
    let mut best: (f64, (bool, f64, bool, f64)) = (-1e9, (false, 1.0, false, 1.0));
    for l2 in [0.3, 1.0, 3.0] {
        for inter in [false, true] {
            for &dw in decoy_weights {
                let v = loso_found(&by_tag, l2, inter, false, dw, decoy_penalty);
                grid.push(json!({"kind": "logistic", "l2": l2, "interactions": inter, "decoy_weight": dw, "loso_objective": fl(v)}));
                if v > best.0 {
                    best = (v, (false, l2, inter, dw));
                }
            }
        }
    }
    for &dw in decoy_weights.iter().take(1) {
        let v = loso_found(&by_tag, 1.0, false, true, dw, decoy_penalty);
        grid.push(json!({"kind": "gbdt", "decoy_weight": dw, "loso_objective": fl(v)}));
        if v > best.0 {
            best = (v, (true, 1.0, false, dw));
        }
    }
    let (gbdt, l2, inter, dw) = best.1;
    let mut r = fit_and_store(tags, l2, inter, gbdt, dw)?;
    r.insert("selection_grid".into(), Value::Array(grid));
    let mut cal = read_cal_file()?;
    cal.insert("ranker".into(), Value::Object(r.clone()));
    write_json(&art().join("calibration.json"), &Value::Object(cal), 1)?;
    Ok(r)
}

pub const SELECT_ARCHS: [&str; 9] = [
    "H_mycelic_full",
    "A2_chunked_ctx",
    "B4_central_triage",
    "Y_oracle_retrieval",
    "A_flat_rag",
    "B2_map_reduce",
    "G_hier_questions",
    "J_mycelic_verified",
    "H_mycelic_lean",
];

/// `select_archs`: per architecture, on the calibration seeds, discovery with the fitted ranker against the
/// hand-set logistic; an architecture adopts the ranker only if it is not worse, and every hierarchy variant
/// (but the v1 reference) shares the hierarchy's decision.
pub fn select_archs(
    cal: &Cal,
    arch_list: &[&str],
    scale: usize,
    seeds: &[u64],
) -> Result<(Vec<String>, Map<String, Value>)> {
    use rayon::prelude::*;
    let alloc = models::allocation_default("back-loaded");
    let per_seed: Vec<Vec<(f64, f64)>> = seeds
        .par_iter()
        .map(|&seed| {
            let w = runner::build_world_default(scale, seed);
            arch_list
                .par_iter()
                .map(|a| {
                    let mut out = [0.0; 2];
                    for (k, on) in [false, true].into_iter().enumerate() {
                        let mut c = cal.clone();
                        c.force_ranker = Some(on);
                        let r = runner::run_arch(&c, a, &w, &alloc, seed, None).res;
                        out[k] = runner::evaluate(&w, &r)["found_anywhere_in_register"];
                    }
                    (out[0], out[1])
                })
                .collect()
        })
        .collect();
    let mut chosen: Vec<String> = Vec::new();
    let mut table = Map::new();
    for (j, a) in arch_list.iter().enumerate() {
        let off: Vec<f64> = per_seed.iter().map(|s| s[j].0).collect();
        let on: Vec<f64> = per_seed.iter().map(|s| s[j].1).collect();
        let (on, off) = (math::mean(&on), math::mean(&off));
        table.insert(a.to_string(), json!({"found_ranker": fl(on), "found_hand": fl(off), "adopt": on >= off}));
        if on >= off {
            chosen.push(a.to_string());
        }
    }
    if chosen.iter().any(|c| c == "H_mycelic_full") {
        for (k, kind, _) in archs() {
            if kind == "hier" && k != "H_mycelic_prev" && !chosen.iter().any(|c| c == k) {
                chosen.push(k.to_string());
            }
        }
        chosen.sort();
    }
    let mut calj = read_cal_file()?;
    calj.insert("ranker_archs".into(), json!(chosen));
    calj.insert("ranker_arch_table".into(), Value::Object(table.clone()));
    write_json(&art().join("calibration.json"), &Value::Object(calj), 1)?;
    Ok((chosen, table))
}

/// `evaluate_calibrator(tag, arch_filter, train_archs)`: fit on the calibration seeds of a dump, evaluate on
/// its evaluation seeds (AUCs, and discovery under the register cap per architecture).
pub fn evaluate_calibrator_with(tag: &str, arch_filter: Option<&str>) -> Result<Value> {
    let rows = load_dump(tag)?;
    let tr: Vec<&Row> = rows.iter().filter(|r| CAL_SEEDS.contains(&seed_of(r))).collect();
    let te: Vec<&Row> = rows
        .iter()
        .filter(|r| arch_filter.is_none_or(|a| r["arch"] == json!(a)))
        .filter(|r| EVAL_SEEDS.contains(&seed_of(r)))
        .collect();
    let stored = read_cal_file().ok().and_then(|c| c.get("ranker").cloned()).unwrap_or(json!({}));
    let inter = stored.get("interactions").and_then(Value::as_bool).unwrap_or(false);
    let l2 = stored.get("l2").and_then(Value::as_f64).unwrap_or(1.0);
    let tr_owned: Vec<Row> = tr.iter().map(|r| (*r).clone()).collect();
    let te_owned: Vec<Row> = te.iter().map(|r| (*r).clone()).collect();
    let ytr = labels(&tr);
    let yte = labels(&te);
    let (xtr, mu, sd) = standardise(&design(&tr_owned, inter), None);
    let (xte, _, _) = standardise(&design(&te_owned, inter), Some((&mu, &sd)));
    let w = fit_logistic(&xtr, &ytr, l2, None);
    let p_te = predict(&xte, &w);
    let conf_te: Vec<f64> = te.iter().map(|r| r["conf"].as_f64().unwrap()).collect();
    // the denominator is every discoverable gold pattern in the world (the funnel's summary rows)
    let mut n_gold_world: Vec<(u64, i64)> = Vec::new();
    let scale0 = te[0]["scale"].clone();
    for r in crate::analysis::load_path(&crate::loss_accounting::funnel_path()) {
        if r.get("summary") == Some(&json!(true)) && r["scale"] == scale0 {
            let s = r["seed"].as_u64().unwrap();
            let g = r["n_gold"].as_i64().unwrap();
            match n_gold_world.iter_mut().find(|e| e.0 == s) {
                Some(e) => e.1 = g,
                None => n_gold_world.push((s, g)),
            }
        }
    }
    let mut n_gold: Vec<((String, u64), Vec<i64>)> = Vec::new();
    for r in &te {
        if r["gold"] == json!(true) {
            let k = (r["arch"].as_str().unwrap().to_string(), seed_of(r));
            let pid = r["pid"].as_i64().unwrap();
            match n_gold.iter_mut().find(|e| e.0 == k) {
                Some(e) => {
                    if !e.1.contains(&pid) {
                        e.1.push(pid)
                    }
                }
                None => n_gold.push((k, vec![pid])),
            }
        }
    }
    let mut names: Vec<String> = vec!["bias".into()];
    names.extend(FEATURES.iter().map(|s| s.to_string()));
    if inter {
        names.extend(INTERACTIONS.iter().map(|(a, b)| format!("{a}*{b}")));
    }
    let mut weights = Map::new();
    for (n, x) in names.iter().zip(&w) {
        weights.insert(n.clone(), fl(*x));
    }
    let nanv = |x: f64| if x.is_nan() { Value::String(NAN.into()) } else { fl(x) };
    let mut out = Map::new();
    out.insert("tag".into(), json!(tag));
    out.insert("arch".into(), json!(arch_filter.unwrap_or("all")));
    out.insert("n_train".into(), json!(tr.len()));
    out.insert("n_test".into(), json!(te.len()));
    out.insert("pos_rate_train".into(), fl(math::mean(&ytr)));
    out.insert("auc_conf".into(), nanv(auc(&conf_te, &yte)));
    out.insert("auc_learned".into(), nanv(auc(&p_te, &yte)));
    out.insert("weights".into(), Value::Object(weights));
    let mut arch_names: Vec<String> = te.iter().map(|r| r["arch"].as_str().unwrap().to_string()).collect();
    arch_names.sort();
    arch_names.dedup();
    let mut per_arch = Map::new();
    for arch in arch_names {
        let idx: Vec<usize> = (0..te.len()).filter(|&i| te[i]["arch"] == json!(arch)).collect();
        let sub: Vec<&Row> = idx.iter().map(|&i| te[i]).collect();
        let mut seeds_here: Vec<u64> = sub.iter().map(|r| seed_of(r)).collect();
        seeds_here.sort();
        seeds_here.dedup();
        let matchable: usize = n_gold.iter().filter(|e| e.0.0 == arch).map(|e| e.1.len()).sum();
        let world_total: i64 =
            seeds_here.iter().map(|s| n_gold_world.iter().find(|e| e.0 == *s).map_or(0, |e| e.1)).sum();
        let gold_total = if world_total != 0 { world_total } else { matchable as i64 };
        let cs: Vec<f64> = idx.iter().map(|&i| conf_te[i]).collect();
        let ps: Vec<f64> = idx.iter().map(|&i| p_te[i]).collect();
        let ys: Vec<f64> = idx.iter().map(|&i| yte[i]).collect();
        let base = found_under_cap(&sub, &cs, TAU);
        let rank_only = found_under_cap(&sub, &ps, 0.0);
        let thresholded = found_under_cap(&sub, &ps, TAU);
        let g = 1.max(gold_total) as f64;
        per_arch.insert(
            arch.clone(),
            json!({
                "gold_patterns": gold_total,
                "matchable_patterns": matchable,
                "found_baseline": fl(base.0 as f64 / g),
                "found_learned_rank_only": fl(rank_only.0 as f64 / g),
                "found_learned_thresholded": fl(thresholded.0 as f64 / g),
                "kept_baseline": base.2,
                "kept_learned_thresholded": thresholded.2,
                "auc_conf": nanv(auc(&cs, &ys)),
                "auc_learned": nanv(auc(&ps, &ys)),
            }),
        );
    }
    out.insert("per_arch".into(), Value::Object(per_arch));
    Ok(Value::Object(out))
}

/// A NaN cannot live in a JSON value; it travels as this marker and [`dumps_nan`] writes Python's `NaN`.
pub const NAN: &str = "__python_nan__";

/// `json.dumps` text of a value that may carry [`NAN`] markers.
pub fn dumps_nan(v: &Value, style: pyjson::Style) -> String {
    pyjson::dumps_with(v, style).replace(&format!("\"{NAN}\""), "NaN")
}

/// `evaluate_calibrator(tag)` over every architecture of the dump.
pub fn evaluate_calibrator(tag: &str) -> Value {
    evaluate_calibrator_with(tag, None).unwrap_or_else(|e| panic!("evaluate_calibrator({tag:?}): {e:#}"))
}

/// The vNext knobs the final dump runs the hierarchy with (`final_rerun.sh`).
pub fn frozen_knobs(cal: &Cal) -> Map<String, Value> {
    let mut m = Map::new();
    for k in
        ["question_frac", "batched_descent", "local_reextract", "link_time", "strict_targeting", "triage_target_chains"]
    {
        if let Some(v) = cal.d.get(k) {
            m.insert(k.into(), v.clone());
        }
    }
    m
}

/// The final rerun's candidate dump and ranker evaluation into `ranker_eval_final.json`.
pub fn final_eval(cal: &Cal) -> Result<()> {
    let _ = std::fs::remove_file(art().join("ranker_eval_final.json"));
    for tag in ["_finalH", "_finalC"] {
        let _ = std::fs::remove_file(dump_path(tag));
    }
    let seeds: Vec<u64> = CAL_SEEDS.iter().chain(EVAL_SEEDS.iter()).copied().collect();
    dump(cal, 10_000, &seeds, &["H_mycelic_full"], Some(&frozen_knobs(cal)), "_finalH")?;
    dump(cal, 10_000, &seeds, &["A2_chunked_ctx", "B4_central_triage", "Y_oracle_retrieval"], None, "_finalC")?;
    let mut out = Map::new();
    for tag in ["_finalH", "_finalC"] {
        let r = evaluate_calibrator_with(tag, None)?;
        let mut short = r.as_object().unwrap().clone();
        short.shift_remove("weights");
        println!("{tag} {}", dumps_nan(&Value::Object(short), pyjson::Style::ASCII));
        out.insert(tag.into(), r);
    }
    std::fs::write(art().join("ranker_eval_final.json"), dumps_nan(&Value::Object(out), pyjson::Style::indent(1)))?;
    Ok(())
}

fn tags_arg(args: &[String], default: &[&str]) -> Vec<String> {
    match args.get(1) {
        Some(t) => t.split(',').map(str::to_string).collect(),
        None => default.iter().map(|s| s.to_string()).collect(),
    }
}

/// `python3 -m research.mycelic.calibrator [archs|select|store|dump|fit|final] [tags]`
pub fn main(args: &[String]) -> Result<()> {
    let cmd = args.first().map(String::as_str).unwrap_or("fit");
    let cal = Cal::load();
    match cmd {
        "archs" => {
            let (chosen, table) = select_archs(&cal, &SELECT_ARCHS, 10_000, &CAL_SEEDS)?;
            for (a, v) in &table {
                println!(
                    "  {a:22} hand={} ranker={} adopt={}",
                    crate::py::fmt::f(v["found_hand"].as_f64().unwrap(), ".3f"),
                    crate::py::fmt::f(v["found_ranker"].as_f64().unwrap(), ".3f"),
                    if v["adopt"] == json!(true) { "True" } else { "False" }
                );
            }
            println!("ranker_archs = {}", crate::py::value::py_repr(&json!(chosen)));
        }
        "select" => {
            let tags = tags_arg(args, &["", "_qf1"]);
            let refs: Vec<&str> = tags.iter().map(String::as_str).collect();
            let r = select_and_store(&refs, &[1.0], 0.0)?;
            println!("selection grid: {}", crate::py::value::py_repr(&r["selection_grid"]));
            println!("stored ranker: l2 {} interactions {} n_train {}", r["l2"], r["interactions"], r["n_train"]);
        }
        "store" => {
            let tags = tags_arg(args, &[""]);
            let refs: Vec<&str> = tags.iter().map(String::as_str).collect();
            let r = fit_and_store(&refs, 1.0, true, false, 1.0)?;
            println!("stored ranker fitted on {} candidates", r["n_train"]);
        }
        "dump" => {
            let seeds: Vec<u64> = CAL_SEEDS.iter().chain(EVAL_SEEDS.iter()).copied().collect();
            dump(
                &cal,
                10_000,
                &seeds,
                &["H_mycelic_full", "A2_chunked_ctx", "B4_central_triage", "Y_oracle_retrieval"],
                None,
                "",
            )?;
            let mut over = Map::new();
            over.insert("question_frac".into(), json!(1.0));
            dump(&cal, 10_000, &seeds, &["H_mycelic_full"], Some(&over), "_qf1")?;
        }
        "final" => final_eval(&cal)?,
        _ => {
            for tag in ["", "_qf1"] {
                if !dump_path(tag).exists() {
                    continue;
                }
                let r = evaluate_calibrator_with(tag, None)?;
                let mut short = r.as_object().unwrap().clone();
                let weights = short.shift_remove("weights");
                println!("{}", dumps_nan(&Value::Object(short), pyjson::Style::indent(1)));
                println!("weights: {}", dumps_nan(&weights.unwrap_or(Value::Null), pyjson::Style::indent(1)));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::tags_arg;

    #[test]
    fn calibrator_cli_passes_dump_tags() {
        // a `store` that dropped its tags once trained on the wrong dump (test_mycelic.py)
        let args = |v: &[&str]| v.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        assert_eq!(tags_arg(&args(&["store", "_a,_b"]), &[""]), vec!["_a", "_b"]);
        assert_eq!(tags_arg(&args(&["store"]), &[""]), vec![""]);
        assert_eq!(tags_arg(&args(&["select", "_hybH,_v3C,_J"]), &["", "_qf1"]), vec!["_hybH", "_v3C", "_J"]);
        assert_eq!(tags_arg(&args(&["select"]), &["", "_qf1"]), vec!["", "_qf1"]);
    }
}
