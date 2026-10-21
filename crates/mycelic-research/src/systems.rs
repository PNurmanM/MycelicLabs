//! The architectures under test (`systems.py`).
//!
//! All of them consume the same corpus through the same operators in `ops`:
//!
//! * A  `flat_rag`            lexical retrieval over raw text -> one kernel call
//! * B  `long_context`        largest possible raw-text context -> one kernel call
//! * B2 `map_reduce`          cheap extraction over every record, global importance rank, one strong kernel call
//! * A2 `chunked_long_context` the kernel over every causal-predicate record, chunk by chunk
//! * B4 `central_triage`      the hierarchy's own triage algorithm, run centrally
//! * C  `recursive_summary`   text-level summarisation up the org tree
//! * D..J hierarchical        structured abstraction with feature flags ([`HierConfig`])
//! * Y, Z, Z2                 reference controls (oracle retrieval, random rank, naive enumeration)

use crate::corpus::{CAUSAL_CHAINS, Corpus, SCHEMA, n_causal_pred};
use crate::models::{Tier, usd_per_mtok};
use crate::np::Generator;
use crate::np::math;
use crate::ops::{
    self, BaseRate, ExtractResult, ForeignScore, Hypothesis, Ko, KoId, LocalBaseRate, MAX_EVIDENCE, MAX_SIGS,
    MAX_USERS, Ranker, SynthOpts, TOK_PER_HYP, TOK_PER_KO, TOK_PER_KO_NOLIN, TOK_PER_QUESTION, TOK_PER_RECORD,
    TOK_PROMPT_OVERHEAD, chain_span, chains_with_span, merge_into, stem_rep_map, valid_chain_path,
};
use crate::org::{DEPT, ENT, Org, REGION, SITE, TEAM, USER};
use crate::py::PySet;
use indexmap::IndexMap;
use rustc_hash::{FxHashMap, FxHashSet};
use serde_json::{Value, json};
use std::collections::HashSet;

// ---------------------------------------------------------------------------------------------------------------
// Metering
// ---------------------------------------------------------------------------------------------------------------

#[derive(Clone, Debug, Default)]
pub struct LevelMeter {
    pub calls: i64,
    pub tok_in: i64,
    pub tok_out: i64,
    pub cu: f64,
    pub usd: f64,
    pub max_ctx: i64,
    pub lat: Vec<f64>,
    pub tier: String,
    pub concurrency: i64,
}

/// Token, call, compute and latency accounting per stage, in the order stages were first metered.
#[derive(Clone, Debug, Default)]
pub struct Meter {
    pub levels: IndexMap<String, LevelMeter>,
}

impl Meter {
    pub fn add(&mut self, stage: &str, tier: &Tier, tok_in: i64, tok_out: i64, calls: i64) {
        let m = self.levels.entry(stage.to_string()).or_insert_with(|| LevelMeter {
            tier: tier.name.clone(),
            concurrency: tier.concurrency,
            ..Default::default()
        });
        m.calls += calls;
        m.tok_in += tok_in;
        m.tok_out += tok_out;
        m.cu += tier.cu(tok_in, tok_out);
        m.usd += usd_per_mtok(&tier.name) * (tok_in + tok_out) as f64 / 1e6;
        let per_in = tok_in.div_euclid(1.max(calls));
        let per_out = tok_out.div_euclid(1.max(calls));
        m.max_ctx = m.max_ctx.max(per_in);
        if m.lat.len() < 4000 {
            m.lat.push(tier.seconds(per_in, per_out));
        }
    }

    pub fn tok_in(&self) -> i64 {
        self.levels.values().map(|m| m.tok_in).sum()
    }

    pub fn tok_out(&self) -> i64 {
        self.levels.values().map(|m| m.tok_out).sum()
    }

    pub fn calls(&self) -> i64 {
        self.levels.values().map(|m| m.calls).sum()
    }

    /// Python's `sum()` over the stages: left to right.
    pub fn cu(&self) -> f64 {
        self.levels.values().fold(0.0, |a, m| a + m.cu)
    }

    pub fn usd(&self) -> f64 {
        self.levels.values().fold(0.0, |a, m| a + m.usd)
    }

    /// Critical path: stages run in sequence, calls within a stage in parallel up to the fleet's concurrency.
    pub fn wall_seconds(&self) -> f64 {
        let mut tot = 0.0;
        for m in self.levels.values() {
            if m.calls == 0 {
                continue;
            }
            let per = if m.lat.is_empty() { 0.0 } else { math::mean(&m.lat) };
            tot += (m.calls as f64 / 1.max(m.concurrency) as f64).ceil() * per;
        }
        tot
    }

    pub fn call_latency_pct(&self, p: f64) -> f64 {
        let all: Vec<f64> = self.levels.values().flat_map(|m| m.lat.iter().copied()).collect();
        if all.is_empty() { 0.0 } else { math::percentile(&all, p) }
    }

    pub fn max_ctx(&self) -> i64 {
        self.levels.values().map(|m| m.max_ctx).max().unwrap_or(0)
    }
}

#[derive(Clone, Debug)]
pub struct Question {
    pub qid: i64,
    pub anchor: i64,
    pub target_preds: Vec<usize>,
    pub why: String,
    pub expected_gain: f64,
    pub cost_est: f64,
    pub urgency: f64,
    pub well_targeted: bool,
    pub answered: bool,
    pub n_new_evidence: usize,
    pub changed_conclusion: bool,
    pub resolved_contradiction: bool,
}

/// What a run produced.  `hypotheses` (the register, in rank order) and `full_hyps` (everything the last
/// synthesis formed, before the register cut) index into `store`; knowledge objects live in `arena`.
#[derive(Clone, Debug, Default)]
pub struct RunResult {
    pub name: String,
    pub store: Vec<Hypothesis>,
    pub hypotheses: Vec<usize>,
    pub full_hyps: Option<Vec<usize>>,
    pub meter: Meter,
    /// (pred, anchor) reaching the kernel
    pub retained: FxHashSet<(i64, i64)>,
    pub arena: Vec<Ko>,
    pub kernel_kos: Vec<KoId>,
    pub questions: Vec<Question>,
    /// hypothesis index groups (positions in `hypotheses`)
    pub families: Vec<Vec<usize>>,
    /// run notes, as the Python `notes` dict (without `full_hyps`)
    pub notes: serde_json::Map<String, Value>,
    /// raw record TEXT read by anything other than the owning user agent
    pub exposed_raw_records: i64,
    /// extracted CLAIMS (no surface text) that left the user node
    pub claims_leaving_node: i64,
    /// index/sketch METADATA entries that left the node
    pub sketch_entries_leaving_node: i64,
    /// underlying records represented by whatever left the node
    pub propagated_records: i64,
}

impl RunResult {
    pub fn hyp(&self, i: usize) -> &Hypothesis {
        &self.store[self.hypotheses[i]]
    }

    pub fn ranked(&self) -> Vec<&Hypothesis> {
        self.hypotheses.iter().map(|&i| &self.store[i]).collect()
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Shared: user-level extraction (the "map" everybody may pay for)
// ---------------------------------------------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct UserLayer {
    /// claims, sorted by user
    pub ex: ExtractResult,
    pub imp: Vec<f64>,
    pub nov: Vec<f64>,
    pub uid_start: FxHashMap<usize, usize>,
    pub uid_count: FxHashMap<usize, usize>,
    pub sketch_tokens: i64,
}

/// `user_extract`: every record through the user tier's extraction, with novelty and importance.
pub fn user_extract(corpus: &Corpus, tier: &Tier, rng: &mut Generator, near_miss: &[i32]) -> UserLayer {
    let n_rec = corpus.recs.len();
    let rids: Vec<i64> = (0..n_rec as i64).collect();
    let ex = ops::extract_default(corpus, &rids, tier, rng, near_miss);
    let n_ent = corpus.entities.len();
    let base = BaseRate::new(&ex, n_ent);
    let nov = base.novelty(&ex.pred, &ex.anchor);
    let org = &corpus.org;
    let site_of: Vec<i64> =
        ex.uid.iter().map(|&u| org.ancestor_at(u as usize, SITE).map_or(-1, |x| x as i64)).collect();
    let team_of: Vec<i64> =
        ex.uid.iter().map(|&u| org.ancestor_at(u as usize, TEAM).map_or(-1, |x| x as i64)).collect();
    let fs = ForeignScore::new(&ex, &site_of);
    let lb_site = LocalBaseRate::new(&ex, &site_of, n_ent);
    let lb_team = LocalBaseRate::new(&ex, &team_of, n_ent);
    // The SITE is the natural "is this entity normally seen here" context; the team signal is noisier and
    // only a small correction (taking the max of the two destroyed the ranking: research log, iteration 4).
    let fss = fs.score();
    let lsn = lb_site.novelty();
    let ltn = lb_team.novelty();
    let lnov: Vec<f64> = (0..ex.len()).map(|i| 0.70 * fss[i] + 0.22 * lsn[i] + 0.08 * ltn[i]).collect();
    let sketch_tok = lb_site.sketch_tokens(200) + lb_team.sketch_tokens(200) + fs.sketch_tokens();
    let imp = ops::importance_score(&ex.pred, &nov, &ex.t, &ex.polarity, corpus.n_days, Some(&lnov));
    let uid: Vec<i64> = ex.uid.clone();
    let order = math::argsort_stable(&uid);
    let ex = ex.take(&order);
    let imp: Vec<f64> = order.iter().map(|&i| imp[i]).collect();
    let nov: Vec<f64> = order.iter().map(|&i| nov[i]).collect();
    let mut uid_start = FxHashMap::default();
    let mut uid_count = FxHashMap::default();
    for (i, &u) in ex.uid.iter().enumerate() {
        uid_start.entry(u as usize).or_insert(i);
        *uid_count.entry(u as usize).or_insert(0usize) += 1;
    }
    // `np.maximum(nov, lnov)` takes the re-ordered novelty against the extraction-ordered local novelty, element
    // by element; the Python did exactly this, so it is kept.
    let nov: Vec<f64> = nov.iter().zip(&lnov).map(|(&a, &b)| if a >= b || b.is_nan() { a } else { b }).collect();
    UserLayer { ex, imp, nov, uid_start, uid_count, sketch_tokens: sketch_tok }
}

fn branches_of(org: &Org, u: usize) -> [Option<PySet>; 6] {
    let mut br: [Option<PySet>; 6] = Default::default();
    for lvl in [TEAM, DEPT, SITE, REGION] {
        if let Some(a) = org.ancestor_at(u, lvl) {
            br[lvl] = Some(PySet::from_one(a as i64));
        }
    }
    br
}

/// `_kos_from_claims`: merged objects for one user's selected claims, appended to `arena`.
fn kos_from_claims(
    arena: &mut Vec<Ko>,
    ul: &UserLayer,
    idx: &[usize],
    owner: usize,
    level: usize,
    keep_lineage: bool,
    org: &Org,
) -> Vec<KoId> {
    let ex = &ul.ex;
    let mut out: IndexMap<(i64, i64), KoId> = IndexMap::new();
    for &i in idx {
        let pred = ex.pred[i] as i64;
        let anchor = ex.anchor[i] as i64;
        let u = ex.uid[i] as usize;
        let t = ex.t[i] as i64;
        let pol = ex.polarity[i] as i64;
        match out.get(&(pred, anchor)) {
            None => {
                let k = Ko {
                    pred,
                    anchor,
                    tmin: t,
                    tmax: t,
                    polarity: pol,
                    n_raw: 1,
                    sigs: PySet::from_one(ex.sig[i]),
                    evidence: vec![ex.rid[i]],
                    branches: if keep_lineage { branches_of(org, u) } else { Default::default() },
                    lineage: if keep_lineage { vec![u] } else { Vec::new() },
                    owner,
                    level,
                    conf: 0.55,
                    importance: ul.imp[i],
                    novelty: ul.nov[i],
                    contra: 0,
                    revisions: 0,
                    q_tag: -1,
                    origin_users: if keep_lineage { PySet::from_one(u as i64) } else { PySet::new() },
                    pos_tmax: if pol > 0 { t } else { -1 },
                    neg_tmax: if pol < 0 { t } else { -1 },
                };
                arena.push(k);
                out.insert((pred, anchor), arena.len() - 1);
            }
            Some(&id) => {
                let k = &mut arena[id];
                k.tmin = k.tmin.min(t);
                k.tmax = k.tmax.max(t);
                k.n_raw += 1;
                if k.sigs.len() < MAX_SIGS {
                    k.sigs.add(ex.sig[i]);
                }
                if k.evidence.len() < MAX_EVIDENCE {
                    k.evidence.push(ex.rid[i]);
                }
                if k.polarity != pol {
                    k.contra += 1;
                }
                k.importance = k.importance.max(ul.imp[i]);
                k.novelty = k.novelty.max(ul.nov[i]);
                if pol > 0 {
                    k.pos_tmax = k.pos_tmax.max(t);
                } else {
                    k.neg_tmax = k.neg_tmax.max(t);
                }
            }
        }
    }
    out.into_values().collect()
}

// ---------------------------------------------------------------------------------------------------------------
// Hierarchical family
// ---------------------------------------------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct HierConfig {
    pub budgets: [usize; 5],
    // feature flags (ablation switches)
    pub lineage: bool,
    pub independence: bool,
    pub contradiction: bool,
    pub temporal: bool,
    pub adaptive_routing: bool,
    pub questions: bool,
    pub question_targeting: bool,
    pub downward_retrieval: bool,
    pub cross_links: bool,
    pub confidence: bool,
    pub adaptive_abstraction: bool,
    pub n_questions: usize,
    pub question_frac: f64,
    pub max_questions: usize,
    pub n_descents: usize,
    pub descent_fanout: usize,
    // vNext candidates (each a paired experiment)
    pub link_time: String,
    pub stale_rule: String,
    pub strict_targeting: bool,
    pub triage_target_chains: String,
    pub cross_link_degree: usize,
    pub cross_link_budget: usize,
    pub bloom_fp: f64,
    pub retrieval_depth: usize,
    pub sketch_channel: bool,
    pub sketch_cap: usize,
    pub sketch_min_support: i64,
    pub foreign_share: f64,
    pub foreign_only_evidence: bool,
    pub restrict_synthesis_to_triage: bool,
    pub pin_importance: f64,
    pub triage_prior_weight: f64,
    pub w_dispersion: f64,
    pub w_synchrony: f64,
    pub verify_evidence: bool,
    pub verify_top_n: usize,
    pub verify_sample: usize,
    pub w_attribution: f64,
    pub chain_completion: bool,
    pub completion_budget: usize,
    // adversarial / robustness knobs
    pub malicious_frac: f64,
    pub unavailable_frac: f64,
    pub unavailable_level: usize,
    pub triage_budget: usize,
    pub batched_descent: bool,
    pub merge_descent_evidence: bool,
    pub local_reextract: bool,
    pub sketch_weak_bits: bool,
    pub weak_gain_factor: f64,
    pub descent_leaf_cap: usize,
    pub descent_frontier_cap: usize,
    pub max_kernel_kos: usize,
    /// 0 -> scale with the entity namespace
    pub max_reports: usize,
}

impl Default for HierConfig {
    fn default() -> Self {
        HierConfig {
            budgets: [6, 18, 42, 90, 130],
            lineage: true,
            independence: true,
            contradiction: true,
            temporal: true,
            adaptive_routing: true,
            questions: true,
            question_targeting: true,
            downward_retrieval: true,
            cross_links: false,
            confidence: true,
            adaptive_abstraction: true,
            n_questions: 220,
            question_frac: 0.55,
            max_questions: 4000,
            n_descents: 40,
            descent_fanout: 3,
            link_time: "min".into(),
            stale_rule: "any".into(),
            strict_targeting: false,
            triage_target_chains: "none".into(),
            cross_link_degree: 4,
            cross_link_budget: 6,
            bloom_fp: 0.0,
            retrieval_depth: USER,
            sketch_channel: true,
            sketch_cap: 4000,
            sketch_min_support: 2,
            foreign_share: 0.35,
            foreign_only_evidence: true,
            restrict_synthesis_to_triage: true,
            pin_importance: 3.0,
            triage_prior_weight: 1.2,
            w_dispersion: 0.0,
            w_synchrony: 0.0,
            verify_evidence: false,
            verify_top_n: 120,
            verify_sample: 6,
            w_attribution: 0.0,
            chain_completion: false,
            completion_budget: 400,
            malicious_frac: 0.0,
            unavailable_frac: 0.0,
            unavailable_level: SITE,
            triage_budget: 6000,
            batched_descent: false,
            merge_descent_evidence: false,
            local_reextract: false,
            sketch_weak_bits: false,
            weak_gain_factor: 0.7,
            descent_leaf_cap: 140,
            descent_frontier_cap: 400,
            max_kernel_kos: 900,
            max_reports: 0,
        }
    }
}

fn as_bool(key: &str, v: &Value) -> bool {
    match v {
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().is_some_and(|x| x != 0.0),
        _ => panic!("{key}: expected a boolean, got {v}"),
    }
}

fn as_f64(key: &str, v: &Value) -> f64 {
    match v {
        Value::Bool(b) => f64::from(u8::from(*b)),
        Value::Number(n) => n.as_f64().unwrap(),
        _ => panic!("{key}: expected a number, got {v}"),
    }
}

fn as_usize(key: &str, v: &Value) -> usize {
    match v {
        Value::Number(n) => n.as_u64().map(|x| x as usize).or_else(|| n.as_f64().map(|x| x as usize)).unwrap(),
        Value::Bool(b) => usize::from(*b),
        _ => panic!("{key}: expected an integer, got {v}"),
    }
}

impl HierConfig {
    /// `setattr(cfg, key, value)` from a JSON override.
    pub fn set(&mut self, key: &str, v: &Value) {
        match key {
            "lineage" => self.lineage = as_bool(key, v),
            "independence" => self.independence = as_bool(key, v),
            "contradiction" => self.contradiction = as_bool(key, v),
            "temporal" => self.temporal = as_bool(key, v),
            "adaptive_routing" => self.adaptive_routing = as_bool(key, v),
            "questions" => self.questions = as_bool(key, v),
            "question_targeting" => self.question_targeting = as_bool(key, v),
            "downward_retrieval" => self.downward_retrieval = as_bool(key, v),
            "cross_links" => self.cross_links = as_bool(key, v),
            "confidence" => self.confidence = as_bool(key, v),
            "adaptive_abstraction" => self.adaptive_abstraction = as_bool(key, v),
            "n_questions" => self.n_questions = as_usize(key, v),
            "question_frac" => self.question_frac = as_f64(key, v),
            "max_questions" => self.max_questions = as_usize(key, v),
            "n_descents" => self.n_descents = as_usize(key, v),
            "descent_fanout" => self.descent_fanout = as_usize(key, v),
            "link_time" => self.link_time = v.as_str().expect("link_time").to_string(),
            "stale_rule" => self.stale_rule = v.as_str().expect("stale_rule").to_string(),
            "strict_targeting" => self.strict_targeting = as_bool(key, v),
            "triage_target_chains" => self.triage_target_chains = v.as_str().expect("triage_target_chains").into(),
            "cross_link_degree" => self.cross_link_degree = as_usize(key, v),
            "cross_link_budget" => self.cross_link_budget = as_usize(key, v),
            "bloom_fp" => self.bloom_fp = as_f64(key, v),
            "retrieval_depth" => self.retrieval_depth = as_usize(key, v),
            "sketch_channel" => self.sketch_channel = as_bool(key, v),
            "sketch_cap" => self.sketch_cap = as_usize(key, v),
            "sketch_min_support" => self.sketch_min_support = as_usize(key, v) as i64,
            "foreign_share" => self.foreign_share = as_f64(key, v),
            "foreign_only_evidence" => self.foreign_only_evidence = as_bool(key, v),
            "restrict_synthesis_to_triage" => self.restrict_synthesis_to_triage = as_bool(key, v),
            "pin_importance" => self.pin_importance = as_f64(key, v),
            "triage_prior_weight" => self.triage_prior_weight = as_f64(key, v),
            "w_dispersion" => self.w_dispersion = as_f64(key, v),
            "w_synchrony" => self.w_synchrony = as_f64(key, v),
            "verify_evidence" => self.verify_evidence = as_bool(key, v),
            "verify_top_n" => self.verify_top_n = as_usize(key, v),
            "verify_sample" => self.verify_sample = as_usize(key, v),
            "w_attribution" => self.w_attribution = as_f64(key, v),
            "chain_completion" => self.chain_completion = as_bool(key, v),
            "completion_budget" => self.completion_budget = as_usize(key, v),
            "malicious_frac" => self.malicious_frac = as_f64(key, v),
            "unavailable_frac" => self.unavailable_frac = as_f64(key, v),
            "unavailable_level" => self.unavailable_level = as_usize(key, v),
            "triage_budget" => self.triage_budget = as_usize(key, v),
            "batched_descent" => self.batched_descent = as_bool(key, v),
            "merge_descent_evidence" => self.merge_descent_evidence = as_bool(key, v),
            "local_reextract" => self.local_reextract = as_bool(key, v),
            "sketch_weak_bits" => self.sketch_weak_bits = as_bool(key, v),
            "weak_gain_factor" => self.weak_gain_factor = as_f64(key, v),
            "descent_leaf_cap" => self.descent_leaf_cap = as_usize(key, v),
            "descent_frontier_cap" => self.descent_frontier_cap = as_usize(key, v),
            "max_kernel_kos" => self.max_kernel_kos = as_usize(key, v),
            "max_reports" => self.max_reports = as_usize(key, v),
            "budgets" => {
                let a: Vec<usize> = v.as_array().expect("budgets").iter().map(|x| as_usize(key, x)).collect();
                self.budgets.copy_from_slice(&a);
            }
            other => panic!("unknown HierConfig field {other:?}"),
        }
    }
}

/// One site's sketch entry: total mentions, per operational predicate counts, users, time span.
#[derive(Clone, Debug)]
struct SkEntry {
    count: i64,
    preds: IndexMap<i64, i64>,
    n_users: i64,
    tmin: i64,
    tmax: i64,
}

/// The hierarchy: an org tree of agents, upward propagation, the sketch channel, descent and the kernel.
/// (`Hierarchy` and `HierRunner` of the Python, in one struct: they share the random generator.)
pub struct HierRunner<'w> {
    c: &'w Corpus,
    org: &'w Org,
    alloc: Vec<Tier>,
    pub cfg: HierConfig,
    rng: Generator,
    ul: &'w UserLayer,
    near_miss: &'w [i32],
    stem: Vec<i32>,
    ranker: Option<&'w Ranker>,
    pub meter: Meter,
    pub arena: Vec<Ko>,
    kos_at: Vec<Vec<KoId>>,
    anchor_count: Vec<FxHashMap<i32, i64>>,
    anchor_causal: Vec<FxHashMap<i32, i64>>,
    sketch: Vec<Option<IndexMap<i32, SkEntry>>>,
    /// site -> anchor -> (count, mask, tmin, tmax), in site processing order
    pub site_sketch: IndexMap<usize, IndexMap<i32, (i64, i64, i64, i64)>>,
    site_weak: IndexMap<usize, IndexMap<i32, i64>>,
    pub sketch_entries: i64,
    pub sketch_dropped: i64,
    pub triage_entries: i64,
    pub triage_weak_used: i64,
    pub triage_sites: FxHashMap<i64, Vec<usize>>,
    pub triage_home: FxHashMap<i64, usize>,
    /// anchor -> triage gain, in insertion order
    pub triage_gain: IndexMap<i64, f64>,
    pub completion_targets: usize,
    pub index_entries: i64,
    pub questions: Vec<Question>,
    pub propagated: i64,
    pub raw_reads: i64,
    pub reextract_records: i64,
    route_ledger: IndexMap<usize, Vec<(usize, i64)>>,
    read_ledger: IndexMap<usize, Vec<(i64, i64, i64)>>,
    /// anchor -> users a descent for that anchor queried (loss accounting)
    pub reached_users: FxHashMap<i64, FxHashSet<usize>>,
    /// the question queue's anchor order before the budget cut
    pub question_order: Vec<i64>,
    /// the current synthesis's candidates (`full_hyps` is `full` ordered over `store`)
    pub store: Vec<Hypothesis>,
    pub full: Vec<usize>,
    pub unavailable: FxHashSet<usize>,
    pub malicious_nodes: FxHashSet<usize>,
}

/// A hypothesis's identity across syntheses: (anchor, sorted predicates).
fn hkey(h: &Hypothesis) -> (i64, Vec<usize>) {
    let mut p = h.preds.clone();
    p.sort();
    (h.anchor, p)
}

/// `_weak_targets` / the missing links of a candidate's chain around its observed span.
fn missing_links(hy: &Hypothesis) -> Vec<usize> {
    let mut missing = Vec::new();
    if hy.chain >= 0 {
        let ch = &SCHEMA.chain_ids[hy.chain as usize];
        let mut pos: Vec<usize> = hy.preds.iter().filter_map(|p| ch.iter().position(|x| x == p)).collect();
        pos.sort();
        if let (Some(&first), Some(&last)) = (pos.first(), pos.last()) {
            for &c in &ch[first.saturating_sub(1)..ch.len().min(last + 2)] {
                if !hy.preds.contains(&c) {
                    missing.push(c);
                }
            }
        }
    }
    missing
}

impl<'w> HierRunner<'w> {
    pub fn new(
        corpus: &'w Corpus,
        alloc: Vec<Tier>,
        cfg: HierConfig,
        seed: u64,
        ul: &'w UserLayer,
        near_miss: &'w [i32],
        ranker: Option<&'w Ranker>,
    ) -> Self {
        let n = corpus.org.nodes.len();
        HierRunner {
            c: corpus,
            org: &corpus.org,
            alloc,
            cfg,
            rng: Generator::new(90_000 + seed),
            ul,
            near_miss,
            stem: stem_rep_map(corpus),
            ranker,
            meter: Meter::default(),
            arena: Vec::new(),
            kos_at: vec![Vec::new(); n],
            anchor_count: vec![FxHashMap::default(); n],
            anchor_causal: vec![FxHashMap::default(); n],
            sketch: vec![None; n],
            site_sketch: IndexMap::new(),
            site_weak: IndexMap::new(),
            sketch_entries: 0,
            sketch_dropped: 0,
            triage_entries: 0,
            triage_weak_used: 0,
            triage_sites: FxHashMap::default(),
            triage_home: FxHashMap::default(),
            triage_gain: IndexMap::new(),
            completion_targets: 0,
            index_entries: 0,
            questions: Vec::new(),
            propagated: 0,
            raw_reads: 0,
            reextract_records: 0,
            route_ledger: IndexMap::new(),
            read_ledger: IndexMap::new(),
            reached_users: FxHashMap::default(),
            question_order: Vec::new(),
            store: Vec::new(),
            full: Vec::new(),
            unavailable: FxHashSet::default(),
            malicious_nodes: FxHashSet::default(),
        }
    }

    // -- upward pass ---------------------------------------------------------------------------------------------

    pub fn build_user_layer(&mut self) {
        let ul = self.ul;
        let org = self.org;
        let b0 = self.cfg.budgets[0];
        let ex = &ul.ex;
        let tier = self.alloc[USER].clone();
        let noise = self.rng.normal_n(0.0, tier.salience_noise, ex.len());
        let score: Vec<f64> = ul.imp.iter().zip(&noise).map(|(a, b)| a + b).collect();
        for &u in &org.user_ids {
            let Some(&s) = ul.uid_start.get(&u) else {
                self.kos_at[u] = Vec::new();
                continue;
            };
            let n = ul.uid_count[&u];
            let order = math::argsort_desc(&score[s..s + n]);
            let keep: Vec<usize> = order.into_iter().take(b0).map(|i| i + s).collect();
            let mut kos = kos_from_claims(&mut self.arena, ul, &keep, u, USER, self.cfg.lineage, org);
            // retention loss during abstraction
            if !kos.is_empty() && tier.abstract_retention < 1.0 {
                let m = self.rng.random_n(kos.len());
                kos = kos.into_iter().zip(m).filter(|(_, x)| *x < tier.abstract_retention).map(|(k, _)| k).collect();
            }
            self.propagated += kos.iter().map(|&k| self.arena[k].n_raw).sum::<i64>();
            self.kos_at[u] = kos;
        }
        // anchor index (cheap metadata, no model call): what makes a targeted descent possible
        for &u in &org.user_ids {
            let Some(&s) = ul.uid_start.get(&u) else {
                continue;
            };
            let n = ul.uid_count[&u];
            let mut acnt: FxHashMap<i32, i64> = FxHashMap::default();
            let mut accz: FxHashMap<i32, i64> = FxHashMap::default();
            for i in s..s + n {
                *acnt.entry(ex.anchor[i]).or_insert(0) += 1;
                if ops::causal_mask(ex.pred[i] as i64) {
                    *accz.entry(ex.anchor[i]).or_insert(0) += 1;
                }
            }
            self.index_entries += acnt.len() as i64;
            self.anchor_count[u] = acnt;
            self.anchor_causal[u] = accz;
            if self.cfg.sketch_channel {
                // per entity: total mentions + counts of each OPERATIONAL predicate
                let mut sk: IndexMap<i32, SkEntry> = IndexMap::new();
                for i in s..s + n {
                    let ti = ex.t[i] as i64;
                    let e = sk.entry(ex.anchor[i]).or_insert_with(|| SkEntry {
                        count: 0,
                        preds: IndexMap::new(),
                        n_users: 1,
                        tmin: ti,
                        tmax: ti,
                    });
                    e.count += 1;
                    e.tmin = e.tmin.min(ti);
                    e.tmax = e.tmax.max(ti);
                    let pi = ex.pred[i] as i64;
                    if ops::causal_mask(pi) {
                        *e.preds.entry(pi).or_insert(0) += 1;
                    }
                }
                self.sketch_entries += sk.len() as i64;
                self.sketch[u] = Some(sk);
            }
        }
    }

    pub fn aggregate_level(&mut self, level: usize) {
        let org = self.org;
        let tier = self.alloc[level].clone();
        let budget = self.cfg.budgets[level];
        let mut n_calls = 0i64;
        let (mut tin, mut tout) = (0i64, 0i64);
        let tok_ko = if self.cfg.lineage { TOK_PER_KO } else { TOK_PER_KO_NOLIN };
        for &nid in &org.levels[level] {
            if self.cfg.unavailable_frac > 0.0
                && level == self.cfg.unavailable_level
                && self.rng.random() < self.cfg.unavailable_frac
            {
                // a site/region is temporarily unreachable: nothing propagates and it cannot answer a descent
                self.kos_at[nid] = Vec::new();
                self.anchor_count[nid] = FxHashMap::default();
                self.anchor_causal[nid] = FxHashMap::default();
                self.sketch[nid] = Some(IndexMap::new());
                self.unavailable.insert(nid);
                continue;
            }
            let kids = &org.nodes[nid].children;
            let mut pool: Vec<KoId> = Vec::new();
            let mut acnt: FxHashMap<i32, i64> = FxHashMap::default();
            let mut accz: FxHashMap<i32, i64> = FxHashMap::default();
            for &c in kids {
                pool.extend(self.kos_at[c].iter().copied());
                for (&a, &n) in &self.anchor_count[c] {
                    *acnt.entry(a).or_insert(0) += n;
                }
                for (&a, &n) in &self.anchor_causal[c] {
                    *accz.entry(a).or_insert(0) += n;
                }
            }
            self.index_entries += acnt.len() as i64;
            self.anchor_count[nid] = acnt;
            self.anchor_causal[nid] = accz;
            if self.cfg.sketch_channel {
                let mut sk: IndexMap<i32, SkEntry> = IndexMap::new();
                for &ch in kids {
                    let Some(csk) = self.sketch[ch].as_ref() else { continue };
                    for (&a2, e2) in csk {
                        match sk.get_mut(&a2) {
                            None => {
                                sk.insert(a2, e2.clone());
                            }
                            Some(e) => {
                                e.count += e2.count;
                                for (&pp, &cc) in &e2.preds {
                                    *e.preds.entry(pp).or_insert(0) += cc;
                                }
                                e.n_users += e2.n_users;
                                e.tmin = e.tmin.min(e2.tmin);
                                e.tmax = e.tmax.max(e2.tmax);
                            }
                        }
                    }
                }
                if sk.len() > self.cfg.sketch_cap {
                    let mut ranked: Vec<(i32, SkEntry)> = sk.into_iter().collect();
                    ranked.sort_by_key(|(_, e)| -(e.preds.len() as i64 * 4 + e.n_users));
                    self.sketch_dropped += (ranked.len() - self.cfg.sketch_cap) as i64;
                    ranked.truncate(self.cfg.sketch_cap);
                    sk = ranked.into_iter().collect();
                }
                self.sketch_entries += sk.len() as i64;
                if level == SITE {
                    // At the site boundary the counts collapse to a support-thresholded bitmask: single stray
                    // mentions are dropped, a real facet (several reports of the SAME predicate) survives.
                    let ms = self.cfg.sketch_min_support;
                    let mut ent: IndexMap<i32, (i64, i64, i64, i64)> = IndexMap::new();
                    let mut weak: IndexMap<i32, i64> = IndexMap::new();
                    for (&a2, e2) in &sk {
                        let (mut mask, mut wmask) = (0i64, 0i64);
                        for (&pp, &cc) in &e2.preds {
                            if cc >= ms && pp < 63 {
                                mask |= 1 << pp;
                            } else if self.cfg.sketch_weak_bits && 0 < cc && cc < ms && pp < 63 {
                                wmask |= 1 << pp;
                            }
                        }
                        ent.insert(a2, (e2.count, mask, e2.tmin, e2.tmax));
                        if wmask != 0 {
                            weak.insert(a2, wmask);
                        }
                    }
                    self.site_sketch.insert(nid, ent);
                    if self.cfg.sketch_weak_bits {
                        self.site_weak.insert(nid, weak);
                    }
                }
                self.sketch[nid] = Some(sk);
            }
            if pool.is_empty() {
                self.kos_at[nid] = Vec::new();
                continue;
            }
            n_calls += 1;
            tin += pool.len() as i64 * tok_ko + TOK_PROMPT_OVERHEAD;
            let mut merged: IndexMap<(i64, i64), KoId> = IndexMap::new();
            for &k in &pool {
                let key = self.arena[k].key();
                match merged.get(&key) {
                    None => {
                        let ko = &mut self.arena[k];
                        ko.owner = nid;
                        ko.level = level;
                        if self.cfg.lineage {
                            ko.lineage.push(nid);
                        }
                        merged.insert(key, k);
                    }
                    Some(&m) => {
                        if self.rng.random() < tier.abstract_distortion {
                            continue; // lost in merge
                        }
                        merge_into(&mut self.arena, m, k, self.cfg.lineage);
                    }
                }
            }
            let items: Vec<KoId> = merged.into_values().collect();
            // observable importance: novelty + breadth
            for &k in &items {
                let ko = &mut self.arena[k];
                let breadth =
                    if self.cfg.lineage { (ko.branch_len(TEAM) as f64).ln_1p() } else { (ko.n_raw as f64).ln_1p() };
                ko.importance = ko.importance + 0.25 * ko.novelty + 0.10 * breadth + 0.0;
                // a claim seen in several distinct sub-branches but rare overall is the weak cross-cutting signal
                if self.cfg.adaptive_abstraction && self.cfg.lineage && ko.branch_len(TEAM) >= 2 && ko.novelty > 0.55 {
                    ko.importance += 0.45;
                }
            }
            let noise = self.rng.normal_n(0.0, tier.salience_noise, items.len());
            let sc: Vec<f64> = items.iter().zip(&noise).map(|(&k, n)| self.arena[k].importance + n).collect();
            let keep_idx: Vec<usize> = math::argsort_desc(&sc).into_iter().take(budget).collect();
            let mut kept: Vec<KoId> = keep_idx.iter().map(|&i| items[i]).collect();
            if tier.abstract_retention < 1.0 && !kept.is_empty() {
                let m = self.rng.random_n(kept.len());
                kept = kept.into_iter().zip(m).filter(|(_, x)| *x < tier.abstract_retention).map(|(k, _)| k).collect();
            }
            if self.cfg.contradiction {
                for &k in &kept {
                    if self.arena[k].contra > 0 && self.rng.random() > tier.contradiction_acc {
                        self.arena[k].contra = 0;
                    }
                }
            }
            if self.cfg.malicious_frac > 0.0 && self.rng.random() < self.cfg.malicious_frac {
                // A compromised aggregator forwards fabricated objects with inflated support and strips the
                // contradictions it saw.
                for &k in &kept {
                    let base = -90_000_000 - 40 * nid as i64;
                    let ko = &mut self.arena[k];
                    ko.contra = 0;
                    ko.sigs = PySet::from_iter_seq(base..base + 12);
                    ko.n_raw = ko.n_raw.max(12);
                    ko.importance += 1.5;
                    if self.rng.random() < 0.5 {
                        self.arena[k].pred = self.rng.integers(0, 34);
                    }
                }
                self.malicious_nodes.insert(nid);
            }
            tout += kept.len() as i64 * tok_ko;
            self.kos_at[nid] = kept;
        }
        if n_calls > 0 {
            self.meter.add(&format!("L{level}-aggregate"), &tier, tin, tout, n_calls);
        }
    }

    // -- cross links -------------------------------------------------------------------------------------------

    /// Semantic overlay: departments that share top anchors exchange their top objects.
    pub fn cross_link(&mut self, level: usize) {
        let org = self.org;
        let tier = self.alloc[level].clone();
        let nodes = &org.levels[level];
        if nodes.len() < 2 {
            return;
        }
        let mut by_anchor: IndexMap<i64, Vec<usize>> = IndexMap::new();
        for &nid in nodes {
            let mut ts: Vec<KoId> = self.kos_at[nid].clone();
            ts.sort_by(|&a, &b| (-self.arena[a].importance).partial_cmp(&-self.arena[b].importance).unwrap());
            ts.truncate(self.cfg.cross_link_budget);
            let tops: Vec<i64> = ts.iter().map(|&k| self.arena[k].anchor).collect();
            for a in PySet::from_iter_seq(tops).iter() {
                by_anchor.entry(a).or_default().push(nid);
            }
        }
        let (mut sent, mut calls, mut tin, mut tout) = (0i64, 0i64, 0i64, 0i64);
        for (&a, group) in &by_anchor {
            if group.len() < 2 || group.len() > 60 {
                continue;
            }
            for &nid in group.iter().take(self.cfg.cross_link_degree * 8) {
                let peers: Vec<usize> =
                    group.iter().copied().filter(|&p| p != nid).take(self.cfg.cross_link_degree).collect();
                if peers.is_empty() {
                    continue;
                }
                calls += 1;
                let incoming: Vec<KoId> = peers
                    .iter()
                    .flat_map(|&p| self.kos_at[p].iter().copied().filter(|&k| self.arena[k].anchor == a))
                    .collect();
                if incoming.is_empty() {
                    continue;
                }
                tin += incoming.len() as i64 * TOK_PER_KO + TOK_PROMPT_OVERHEAD;
                let own: Vec<KoId> = self.kos_at[nid].iter().copied().filter(|&k| self.arena[k].anchor == a).collect();
                let mut preds: Vec<usize> = own
                    .iter()
                    .chain(incoming.iter())
                    .map(|&k| self.arena[k].pred as usize)
                    .collect::<HashSet<usize>>()
                    .into_iter()
                    .collect();
                preds.sort();
                let (valid, _) = valid_chain_path(&preds);
                if valid && preds.len() >= 2 {
                    for &k in &own {
                        self.arena[k].importance += 0.9;
                        self.arena[k].q_tag = 1;
                    }
                    sent += incoming.len() as i64;
                    tout += 8;
                }
            }
        }
        if calls > 0 {
            self.meter.add(&format!("L{level}-crosslink"), &tier, tin, tout, calls);
        }
        self.propagated += sent;
    }

    // -- downward retrieval ------------------------------------------------------------------------------------

    /// Targeted descent guided by the propagated sketches: breadth-balanced (top-k per parent), seeded at the
    /// sites the triage flagged, the entity's home branch down-ranked.
    pub fn descend(
        &mut self,
        anchor: i64,
        target_preds: &[usize],
        budget_nodes: usize,
        start_nodes: Option<&[usize]>,
        avoid: Option<&[usize]>,
    ) -> Vec<KoId> {
        let org = self.org;
        let mut found: Vec<KoId> = Vec::new();
        let mut frontier: Vec<usize> = vec![org.root];
        let seeds = PySet::from_iter_seq(start_nodes.unwrap_or(&[]).iter().map(|&s| s as i64));
        let avoid_set: FxHashSet<usize> = avoid.unwrap_or(&[]).iter().copied().collect();
        let mut avoid_anc: FxHashSet<usize> = FxHashSet::default();
        for &a in &avoid_set {
            avoid_anc.insert(a);
            avoid_anc.extend(org.ancestors(a));
        }
        let per_parent = |level: usize| -> usize {
            match level {
                REGION => 12,
                USER => budget_nodes + 5,
                _ => budget_nodes + 2,
            }
        };
        let a32 = anchor as i32;
        for level in [REGION, SITE, DEPT, TEAM, USER] {
            let mut nxt: Vec<(f64, usize)> = Vec::new();
            for &nid in &frontier {
                let node = &org.nodes[nid];
                if node.level == USER {
                    continue;
                }
                let kids = &node.children;
                let tier = &self.alloc[1.max(node.level)];
                let mut scored: Vec<(f64, usize)> = Vec::new();
                for &ch in kids {
                    let nc = self.anchor_count[ch].get(&a32).copied().unwrap_or(0);
                    let ncz = self.anchor_causal[ch].get(&a32).copied().unwrap_or(0);
                    if nc == 0 {
                        if self.cfg.bloom_fp > 0.0 && self.rng.random() < self.cfg.bloom_fp {
                            scored.push((1e-6, ch));
                        }
                        continue;
                    }
                    let mut sc = 2.0 * ncz as f64 + nc as f64;
                    if avoid_anc.contains(&ch) {
                        sc *= 0.12;
                    }
                    scored.push((sc, ch));
                }
                if scored.is_empty() {
                    continue;
                }
                scored.sort_by(|a, b| b.partial_cmp(a).unwrap());
                let k = per_parent(level);
                if self.cfg.adaptive_routing {
                    if self.rng.random() > tier.route_quality && scored.len() > 1 {
                        scored.remove(0);
                    }
                } else {
                    self.rng.shuffle(&mut scored);
                }
                nxt.extend(scored.into_iter().take(k));
                if self.cfg.batched_descent {
                    self.route_ledger.entry(nid).or_default().push((kids.len(), anchor));
                } else {
                    let tier = tier.clone();
                    self.meter.add("descend-route", &tier, kids.len() as i64 * 6 + TOK_PROMPT_OVERHEAD, 20, 1);
                }
            }
            // per-parent quota has guaranteed breadth; a global sort keeps the best when the frontier is cut
            nxt.sort_by(|a, b| b.partial_cmp(a).unwrap());
            let mut ordered: Vec<usize> = nxt.into_iter().map(|(_, n)| n).collect();
            if level == SITE && !seeds.is_empty() {
                let mut seen: FxHashSet<usize> = FxHashSet::default();
                ordered = seeds.iter().map(|s| s as usize).chain(ordered).filter(|n| seen.insert(*n)).collect();
            }
            ordered.truncate(self.cfg.descent_frontier_cap);
            frontier = ordered;
            if frontier.is_empty() {
                return found;
            }
            if org.nodes[frontier[0]].level == self.cfg.retrieval_depth {
                break;
            }
        }
        let leaf_tier = self.alloc[USER].clone();
        let ul = self.ul;
        let leaf_cap = self.cfg.descent_leaf_cap;
        for &nid in frontier.iter().take(leaf_cap) {
            if !avoid_anc.is_empty()
                && self.cfg.foreign_only_evidence
                && org.ancestor_at(nid, SITE).is_some_and(|s| avoid_set.contains(&s))
            {
                // the entity's own home site is what the organisation already knows
                continue;
            }
            if org.nodes[nid].level == USER {
                self.reached_users.entry(anchor).or_default().insert(nid);
                let Some(&st) = ul.uid_start.get(&nid) else { continue };
                let n = ul.uid_count[&nid];
                let mut sel: Vec<usize> = (0..n).filter(|&i| ul.ex.anchor[st + i] as i64 == anchor).collect();
                if !target_preds.is_empty() {
                    let pm: Vec<bool> =
                        sel.iter().map(|&i| target_preds.contains(&(ul.ex.pred[st + i] as usize))).collect();
                    // soft targeting falls back to everything the user holds on the anchor when nothing matches
                    if pm.iter().any(|&x| x) || self.cfg.strict_targeting {
                        sel = sel.into_iter().zip(pm).filter(|(_, m)| *m).map(|(i, _)| i).collect();
                    }
                }
                if self.cfg.batched_descent {
                    self.read_ledger.entry(nid).or_default().push((sel.len() as i64, sel.len() as i64, anchor));
                } else {
                    let k = sel.len() as i64;
                    self.meter.add(
                        "descend-read",
                        &leaf_tier,
                        k * TOK_PER_RECORD + TOK_PROMPT_OVERHEAD + 12,
                        k * TOK_PER_KO,
                        1,
                    );
                }
                if sel.is_empty() {
                    continue;
                }
                let idx: Vec<usize> = sel.iter().map(|&i| i + st).collect();
                let mut kos = kos_from_claims(&mut self.arena, ul, &idx, nid, USER, self.cfg.lineage, org);
                if self.cfg.local_reextract {
                    let more = self.reextract(nid, anchor, target_preds, st, n);
                    kos.extend(more);
                }
                for &k in &kos {
                    self.arena[k].importance += 1.2;
                    self.arena[k].q_tag = 2;
                }
                found.extend(kos);
                self.raw_reads += n as i64;
            } else {
                let kos = self.kos_at[nid].clone();
                let hit: Vec<KoId> = kos
                    .iter()
                    .copied()
                    .filter(|&k| {
                        self.arena[k].anchor == anchor
                            && (target_preds.is_empty() || target_preds.contains(&(self.arena[k].pred as usize)))
                    })
                    .collect();
                if self.cfg.batched_descent {
                    self.read_ledger.entry(nid).or_default().push((kos.len() as i64, hit.len() as i64, anchor));
                } else {
                    self.meter.add(
                        "descend-read",
                        &leaf_tier,
                        kos.len() as i64 * TOK_PER_KO + TOK_PROMPT_OVERHEAD,
                        hit.len() as i64 * TOK_PER_KO,
                        1,
                    );
                }
                found.extend(hit);
            }
        }
        found
    }

    /// Re-read this user's own notes about `anchor` that produced no claim on the first pass.
    fn reextract(&mut self, uid: usize, anchor: i64, target_preds: &[usize], st: usize, n: usize) -> Vec<KoId> {
        let c = self.c;
        let org = self.org;
        let ui = org.user_ids.partition_point(|&u| u < uid);
        if ui >= org.user_ids.len() || org.user_ids[ui] != uid {
            return Vec::new();
        }
        let (a, b) = (c.user_slices[ui] as usize, c.user_slices[ui + 1] as usize);
        if b <= a {
            return Vec::new();
        }
        let cand: Vec<usize> = (a..b).filter(|&r| c.recs.anchor[r] as i64 == anchor).collect();
        if cand.is_empty() {
            return Vec::new();
        }
        let already: FxHashSet<i64> = self.ul.ex.rid[st..st + n].iter().copied().collect();
        let rids: Vec<i64> = cand.iter().map(|&r| r as i64).filter(|r| !already.contains(r)).collect();
        if rids.is_empty() {
            return Vec::new();
        }
        let tier = self.alloc[USER].clone();
        let ex = ops::extract_default(c, &rids, &tier, &mut self.rng, self.near_miss);
        self.reextract_records += rids.len() as i64;
        if self.cfg.batched_descent {
            self.read_ledger.entry(uid).or_default().push((rids.len() as i64, ex.len() as i64, anchor));
        } else {
            self.meter.add(
                "descend-reread",
                &tier,
                rids.len() as i64 * TOK_PER_RECORD,
                ex.len() as i64 * TOK_PER_KO,
                1,
            );
        }
        let mut out: IndexMap<(i64, i64), KoId> = IndexMap::new();
        for i in 0..ex.len() {
            if ex.anchor[i] as i64 != anchor {
                continue;
            }
            if !target_preds.is_empty() && !target_preds.contains(&(ex.pred[i] as usize)) {
                continue;
            }
            let key = (ex.pred[i] as i64, anchor);
            let t = ex.t[i] as i64;
            let pol = ex.polarity[i] as i64;
            match out.get(&key) {
                None => {
                    let lin = self.cfg.lineage;
                    self.arena.push(Ko {
                        pred: key.0,
                        anchor,
                        tmin: t,
                        tmax: t,
                        polarity: pol,
                        n_raw: 1,
                        sigs: PySet::from_one(ex.sig[i]),
                        evidence: vec![ex.rid[i]],
                        branches: if lin { branches_of(org, uid) } else { Default::default() },
                        lineage: if lin { vec![uid] } else { Vec::new() },
                        owner: uid,
                        level: USER,
                        conf: 0.55,
                        importance: 0.5,
                        novelty: 0.5,
                        contra: 0,
                        revisions: 0,
                        q_tag: 3,
                        origin_users: if lin { PySet::from_one(uid as i64) } else { PySet::new() },
                        pos_tmax: if pol > 0 { t } else { -1 },
                        neg_tmax: if pol < 0 { t } else { -1 },
                    });
                    out.insert(key, self.arena.len() - 1);
                }
                Some(&id) => {
                    let k = &mut self.arena[id];
                    k.tmin = k.tmin.min(t);
                    k.tmax = k.tmax.max(t);
                    k.n_raw += 1;
                    if k.sigs.len() < MAX_SIGS {
                        k.sigs.add(ex.sig[i]);
                    }
                    if k.evidence.len() < MAX_EVIDENCE {
                        k.evidence.push(ex.rid[i]);
                    }
                    if k.polarity != pol {
                        k.contra += 1;
                    }
                }
            }
        }
        out.into_values().collect()
    }

    /// Meter the batched descent: one call per routed node and one per queried user.
    pub fn flush_descent_ledger(&mut self) {
        let org = self.org;
        for (nid, entries) in std::mem::take(&mut self.route_ledger) {
            let tier = self.alloc[1.max(org.nodes[nid].level)].clone();
            let tok_in = TOK_PROMPT_OVERHEAD + entries.iter().map(|(k, _)| *k as i64 * 6).sum::<i64>();
            self.meter.add("descend-route", &tier, tok_in, 20 * entries.len() as i64, 1);
        }
        let leaf_tier = self.alloc[USER].clone();
        for (nid, entries) in std::mem::take(&mut self.read_ledger) {
            let (tok_in, tok_out) = if org.nodes[nid].level == USER {
                (
                    TOK_PROMPT_OVERHEAD
                        + 12 * entries.len() as i64
                        + entries.iter().map(|e| e.0 * TOK_PER_RECORD).sum::<i64>(),
                    entries.iter().map(|e| e.1 * TOK_PER_KO).sum::<i64>(),
                )
            } else {
                // a non-leaf node's object store is read once, whatever the number of anchors asked about
                (
                    TOK_PROMPT_OVERHEAD + entries.iter().map(|e| e.0).max().unwrap() * TOK_PER_KO,
                    entries.iter().map(|e| e.1 * TOK_PER_KO).sum::<i64>(),
                )
            };
            self.meter.add("descend-read", &leaf_tier, tok_in, tok_out, 1);
        }
    }

    // -- the kernel --------------------------------------------------------------------------------------------

    fn synth_opts(&self) -> SynthOpts<'_> {
        let mut o = SynthOpts::new(self.c.entities.len(), self.cfg.max_reports);
        o.use_lineage = self.cfg.lineage;
        o.use_dedup = self.cfg.independence;
        o.use_temporal = self.cfg.temporal;
        o.link_time = &self.cfg.link_time;
        o.stale_rule = &self.cfg.stale_rule;
        o.stem_rep = Some(&self.stem);
        o.w_dispersion = self.cfg.w_dispersion;
        o.w_synchrony = self.cfg.w_synchrony;
        o.ranker = self.ranker;
        o
    }

    /// Synthesize over `pool`; the result becomes the current full list and the returned register.
    fn synthesize_pool(&mut self, pool: &[KoId], kernel_tier: &Tier) -> Vec<usize> {
        let syn = {
            let mut rng = std::mem::replace(&mut self.rng, Generator::new(0));
            let opts = self.synth_opts();
            let s = ops::synthesize(&self.arena, pool, kernel_tier, &mut rng, &opts);
            self.rng = rng;
            s
        };
        let kept = syn.kept();
        self.full = (0..syn.hyps.len()).collect();
        self.store = syn.hyps;
        kept
    }

    pub fn run(mut self) -> (RunResult, HierRunnerState) {
        let c = self.c;
        let users = c.org.user_ids.len() as i64;
        let user_tier = self.alloc[USER].clone();
        self.meter.add(
            "L0-extract",
            &user_tier,
            c.recs.len() as i64 * TOK_PER_RECORD + users * TOK_PROMPT_OVERHEAD,
            self.ul.ex.len() as i64 * 10,
            users,
        );
        self.build_user_layer();
        self.aggregate_level(TEAM);
        self.aggregate_level(DEPT);
        if self.cfg.cross_links {
            self.cross_link(DEPT);
        }
        self.aggregate_level(SITE);
        self.aggregate_level(REGION);

        let kernel_tier = self.alloc[ENT].clone();
        let mut pool: Vec<KoId> = Vec::new();
        for &r in &c.org.levels[REGION] {
            pool.extend(self.kos_at[r].iter().copied());
        }
        pool.sort_by(|&a, &b| (-self.arena[a].importance).partial_cmp(&-self.arena[b].importance).unwrap());
        pool.truncate(self.cfg.max_kernel_kos);
        let tok_ko = if self.cfg.lineage { TOK_PER_KO } else { TOK_PER_KO_NOLIN };
        self.meter.add("L5-kernel", &kernel_tier, pool.len() as i64 * tok_ko + TOK_PROMPT_OVERHEAD, 0, 1);

        if self.cfg.max_reports == 0 {
            // a risk register holds roughly one entry per tracked entity
            self.cfg.max_reports = 6000.min(600.max(c.entities.len()));
        }
        let mut hyps = self.synthesize_pool(&pool, &kernel_tier);
        self.meter.add("L5-kernel", &kernel_tier, 0, hyps.len() as i64 * TOK_PER_HYP, 0);
        self.enrich();
        ops::sort_by_conf(&self.store, &mut hyps);

        // Evidence recovered by a descent is ACCUMULATED into the working pool.
        if self.cfg.downward_retrieval {
            (hyps, pool) = self.descend_round(hyps, pool, &kernel_tier);
        }
        if self.cfg.questions {
            (hyps, pool) = self.question_round(hyps, pool, &kernel_tier);
        }
        if self.cfg.chain_completion {
            (hyps, pool) = self.completion_round(hyps, pool, &kernel_tier);
        }
        if self.cfg.verify_evidence {
            hyps = self.verify_evidence(hyps, &kernel_tier);
        }
        let fams = self.families(&hyps, &kernel_tier);

        let retained: FxHashSet<(i64, i64)> = pool.iter().map(|&k| self.arena[k].key()).collect();
        let claims_out: i64 = c.org.user_ids.iter().map(|&u| self.kos_at[u].len() as i64).sum();
        let mut notes = serde_json::Map::new();
        notes.insert("kernel_kos".into(), json!(pool.len()));
        notes.insert("sketch_dropped".into(), json!(self.sketch_dropped));
        notes.insert("raw_records_reread_locally".into(), json!(self.raw_reads));
        notes.insert("records_reextracted".into(), json!(self.reextract_records));
        notes.insert("triage_weak_candidates".into(), json!(self.triage_weak_used));
        let state = HierRunnerState {
            site_sketch: std::mem::take(&mut self.site_sketch),
            triage_gain: std::mem::take(&mut self.triage_gain),
            question_order: std::mem::take(&mut self.question_order),
            reached_users: std::mem::take(&mut self.reached_users),
            foreign_share: self.cfg.foreign_share,
            max_reports: self.cfg.max_reports,
        };
        let res = RunResult {
            name: "hier".into(),
            store: std::mem::take(&mut self.store),
            hypotheses: hyps,
            full_hyps: Some(std::mem::take(&mut self.full)),
            meter: std::mem::take(&mut self.meter),
            retained,
            arena: std::mem::take(&mut self.arena),
            kernel_kos: pool,
            questions: std::mem::take(&mut self.questions),
            families: fams,
            notes,
            exposed_raw_records: 0,
            claims_leaving_node: claims_out,
            sketch_entries_leaving_node: self.sketch_entries,
            propagated_records: self.propagated,
        };
        (res, state)
    }

    /// Attach anchor-level context the kernel already holds to every candidate's features, then re-apply the
    /// ranker (over the full list, which it re-sorts).
    fn enrich(&mut self) {
        let mut qcount: FxHashMap<i64, i64> = FxHashMap::default();
        for q in &self.questions {
            *qcount.entry(q.anchor).or_insert(0) += 1;
        }
        let mut tot: FxHashMap<i64, i64> = FxHashMap::default();
        for ent in self.site_sketch.values() {
            for (&a2, e) in ent {
                *tot.entry(a2 as i64).or_insert(0) += e.0;
            }
        }
        for &i in &self.full {
            let hy = &mut self.store[i];
            let Some(f) = hy.feat.as_mut() else { continue };
            let a = hy.anchor;
            f.set("triage_gain", self.triage_gain.get(&a).copied().unwrap_or(0.0));
            f.set("sk_total", (tot.get(&a).copied().unwrap_or(0) as f64).ln_1p());
            f.set("n_foreign_sites", self.triage_sites.get(&a).map_or(0, |v| v.len()) as f64);
            f.set("users_reached", (self.reached_users.get(&a).map_or(0, |v| v.len()) as f64).ln_1p());
            f.set("n_questions_anchor", qcount.get(&a).copied().unwrap_or(0) as f64);
        }
        if self.ranker.is_some() {
            let full = self.full.clone();
            ops::apply_ranker_to(self.ranker, &mut self.store, &full);
            let mut full = full;
            ops::sort_by_conf(&self.store, &mut full);
            self.full = full;
        }
    }

    fn merge_new(&mut self, new_kos: Vec<KoId>) -> Vec<KoId> {
        if !self.cfg.merge_descent_evidence {
            return new_kos;
        }
        // merged per polarity too, so the kernel's contradiction balance is exactly what it was
        let mut merged: IndexMap<(i64, i64, i64), KoId> = IndexMap::new();
        for k in new_kos {
            let ko = &self.arena[k];
            let key = (ko.pred, ko.anchor, ko.polarity);
            match merged.get(&key) {
                None => {
                    merged.insert(key, k);
                }
                Some(&m) => merge_into(&mut self.arena, m, k, self.cfg.lineage),
            }
        }
        merged.into_values().collect()
    }

    fn descend_round(&mut self, hyps: Vec<usize>, pool: Vec<KoId>, kernel_tier: &Tier) -> (Vec<usize>, Vec<KoId>) {
        let mut targets: Vec<(i64, Vec<usize>)> = Vec::new();
        for &i in &hyps {
            let hy = &self.store[i];
            if hy.hallucinated {
                continue;
            }
            let missing = missing_links(hy);
            let why = !missing.is_empty() || hy.n_indep < 3 || hy.contra != 0;
            if why {
                targets.push((hy.anchor, missing));
            }
        }
        targets.truncate(self.cfg.n_descents);
        let mut new_kos: Vec<KoId> = Vec::new();
        for (anchor, missing) in &targets {
            let got = self.descend(*anchor, missing, self.cfg.descent_fanout, None, None);
            new_kos.extend(got);
        }
        self.flush_descent_ledger();
        if new_kos.is_empty() {
            return (hyps, pool);
        }
        let mut merged = pool;
        merged.extend(self.merge_new(new_kos));
        self.meter.add("L5-kernel-redo", kernel_tier, merged.len() as i64 * TOK_PER_KO + TOK_PROMPT_OVERHEAD, 0, 1);
        let mut hyps2 = self.synthesize_pool(&merged, kernel_tier);
        self.meter.add("L5-kernel-redo", kernel_tier, 0, hyps2.len() as i64 * TOK_PER_HYP, 0);
        self.enrich();
        ops::sort_by_conf(&self.store, &mut hyps2);
        (hyps2, merged)
    }

    fn question_round(&mut self, hyps: Vec<usize>, pool: Vec<KoId>, kernel_tier: &Tier) -> (Vec<usize>, Vec<KoId>) {
        let org = self.org;
        let mut qs: Vec<Question> = Vec::new();
        let mut qid = 0i64;
        let mut triage_tp: FxHashMap<i64, Vec<usize>> = FxHashMap::default();
        // before = {hkey: hyp}: later duplicates replace the value
        let mut before: FxHashMap<(i64, Vec<usize>), (f64, Vec<usize>, i64)> = FxHashMap::default();
        for &i in &hyps {
            let h = &self.store[i];
            before.insert(hkey(h), (h.conf, h.preds.clone(), h.contra));
        }
        // 1. missing-link / weak-support / contradiction questions
        let mut by_conf = hyps.clone();
        ops::sort_by_conf(&self.store, &mut by_conf);
        for &i in &by_conf {
            let (hallucinated, mut missing, n_indep, contra, conf, h_anchor) = {
                let hy = &self.store[i];
                (hy.hallucinated, missing_links(hy), hy.n_indep, hy.contra, hy.conf, hy.anchor)
            };
            if hallucinated {
                continue;
            }
            let why = if !missing.is_empty() {
                "missing chain link"
            } else if n_indep < 3 {
                "weak independent support"
            } else if contra != 0 {
                "contradictory evidence"
            } else {
                ""
            };
            if why.is_empty() {
                continue;
            }
            let b = |x: bool| if x { 1.0 } else { 0.0 };
            let gain = (0.5 * b(!missing.is_empty()) + 0.3 * b(n_indep < 3) + 0.2 * b(contra > 0))
                * (1.0 - (conf - 0.5).abs() * 1.2);
            let well = self.rng.random() < kernel_tier.question_quality;
            let mut anchor = h_anchor;
            if !(well && self.cfg.question_targeting) {
                anchor = self.rng.integers(0, self.c.entities.len() as i64);
                missing = Vec::new();
            }
            qs.push(Question {
                qid,
                anchor,
                target_preds: missing,
                why: why.into(),
                expected_gain: gain,
                cost_est: (self.cfg.descent_fanout * 4) as f64,
                urgency: conf,
                well_targeted: well,
                answered: false,
                n_new_evidence: 0,
                changed_conclusion: false,
                resolved_contradiction: false,
            });
            qid += 1;
        }
        // 2. Sketch-driven triage (channel A): entities FOREIGN at >= 2 sites in >= 2 regions with a repeated
        //    operational predicate, the predicates on one causal chain.  No single record can pass this test.
        let mut by_anchor: FxHashMap<i64, FxHashSet<i64>> = FxHashMap::default();
        for &k in &pool {
            by_anchor.entry(self.arena[k].anchor).or_default().insert(self.arena[k].pred);
        }
        let mut cands: Vec<(f64, i64, usize, i64, usize)> = Vec::new();
        if self.cfg.sketch_channel {
            let mut per_ent: IndexMap<i32, Vec<(usize, usize, i64, i64, i64)>> = IndexMap::new();
            let mut tot_ent: FxHashMap<i32, i64> = FxHashMap::default();
            let mut n_entries = 0i64;
            let mut weak_ent: IndexMap<i32, Vec<(usize, usize, i64)>> = IndexMap::new();
            for (&st, ent) in &self.site_sketch {
                let reg = org.ancestor_at(st, REGION).expect("site without region");
                for (&a2, &(tot, mask, t0, t1)) in ent {
                    *tot_ent.entry(a2).or_insert(0) += tot;
                    if mask != 0 {
                        per_ent.entry(a2).or_default().push((st, reg, mask, t0, t1));
                        n_entries += 1;
                    }
                }
                if let (true, Some(w)) = (self.cfg.sketch_weak_bits, self.site_weak.get(&st)) {
                    for (&a2, &wm) in w {
                        weak_ent.entry(a2).or_default().push((st, reg, wm));
                    }
                }
            }
            let n_weak: i64 = weak_ent.values().map(|v| v.len() as i64).sum();
            self.meter.add("L5-triage", kernel_tier, n_entries * 5 + n_weak + TOK_PROMPT_OVERHEAD, 0, 1);
            self.triage_entries = n_entries;
            self.triage_weak_used = 0;
            let keys: Vec<i64> = per_ent.keys().map(|&a| a as i64).collect();
            let mut anchors = PySet::from_dict_keys(&keys);
            let other = if self.cfg.sketch_weak_bits {
                PySet::from_dict_keys(&weak_ent.keys().map(|&a| a as i64).collect::<Vec<_>>())
            } else {
                PySet::new()
            };
            anchors = anchors.union(&other);
            for a2 in anchors.iter() {
                let a32 = a2 as i32;
                let rows: Vec<(usize, usize, i64, i64, i64)> = per_ent.get(&a32).cloned().unwrap_or_default();
                let total = 1.max(tot_ent.get(&a32).copied().unwrap_or(1));
                // a site is "foreign" for this entity if it holds only a small share of its mentions
                let mut site_tot: IndexMap<usize, i64> = IndexMap::new();
                for &(st, ..) in &rows {
                    site_tot.insert(st, self.site_sketch[&st][&a32].0);
                }
                let mut foreign: Vec<(usize, usize, i64, i64, i64)> = rows
                    .iter()
                    .copied()
                    .filter(|r| site_tot[&r.0] as f64 / total as f64 <= self.cfg.foreign_share)
                    .collect();
                let mut used_weak = false;
                if self.cfg.sketch_weak_bits && !foreign.is_empty() {
                    // corroboration: a weak bit counts only where the entity already shows a STRONG foreign bit
                    // in another region
                    let strong_regs: FxHashSet<usize> = foreign.iter().map(|r| r.1).collect();
                    for &(st, reg, wm) in weak_ent.get(&a32).map(|v| v.as_slice()).unwrap_or(&[]) {
                        let stt = self.site_sketch.get(&st).and_then(|e| e.get(&a32)).map_or(0, |e| e.0);
                        if stt as f64 / total as f64 <= self.cfg.foreign_share
                            && !strong_regs.contains(&reg)
                            && !site_tot.contains_key(&st)
                        {
                            let e_ = self.site_sketch[&st][&a32];
                            foreign.push((st, reg, wm, e_.2, e_.3));
                            site_tot.insert(st, stt);
                            used_weak = true;
                        }
                    }
                }
                if foreign.len() < 2 {
                    continue;
                }
                let regs: FxHashSet<usize> = foreign.iter().map(|r| r.1).collect();
                if regs.len() < 2 {
                    continue;
                }
                let umask = foreign.iter().fold(0i64, |m, r| m | r.2);
                let (span, _) = chain_span(umask);
                if span < 2 {
                    continue;
                }
                let seen = by_anchor.get(&a2).map_or(0, |s| s.len());
                let mut gain = (0.6 * span as f64 + 0.35 * regs.len() as f64 + 0.3 * if seen < 2 { 1.0 } else { 0.0 })
                    / (1.0 + 0.25 * (total as f64).ln_1p());
                if used_weak {
                    gain *= self.cfg.weak_gain_factor;
                    self.triage_weak_used += 1;
                }
                cands.push((gain, a2, regs.len(), total, seen));
                self.triage_sites.insert(a2, foreign.iter().map(|r| r.0).collect());
                self.triage_gain.insert(a2, gain);
                let mut home = 0usize;
                let mut best = i64::MIN;
                for (&st, &v) in &site_tot {
                    if v > best {
                        best = v;
                        home = st;
                    }
                }
                self.triage_home.insert(a2, home);
                if self.cfg.triage_target_chains == "span2" {
                    // the question names the predicates of every chain the foreign sketch bits flag
                    let mut tp: Vec<usize> = chains_with_span(umask, 2)
                        .into_iter()
                        .flat_map(|ci| SCHEMA.chain_ids[ci].iter().copied())
                        .collect::<HashSet<usize>>()
                        .into_iter()
                        .collect();
                    tp.sort();
                    triage_tp.insert(a2, tp);
                }
            }
        }
        cands.sort_by(|a, b| b.partial_cmp(a).unwrap());
        for &(gain, a2, nreg, tot, seen) in cands.iter().take(self.cfg.triage_budget) {
            qs.push(Question {
                qid,
                anchor: a2,
                target_preds: triage_tp.get(&a2).cloned().unwrap_or_default(),
                why: format!(
                    "sketch: causal span across {nreg} regions, {tot} mentions, {seen} facets already at kernel"
                ),
                expected_gain: gain,
                cost_est: (self.cfg.descent_fanout * 4) as f64,
                urgency: 0.5,
                well_targeted: true,
                answered: false,
                n_new_evidence: 0,
                changed_conclusion: false,
                resolved_contradiction: false,
            });
            qid += 1;
        }
        qs.sort_by(|a, b| {
            let ka = -(a.expected_gain / 1.0f64.max(a.cost_est));
            let kb = -(b.expected_gain / 1.0f64.max(b.cost_est));
            ka.partial_cmp(&kb).unwrap()
        });
        // the question budget scales with the number of entities the triage flags
        let nq =
            self.cfg.max_questions.min(self.cfg.n_questions.max((self.cfg.question_frac * qs.len() as f64) as usize));
        self.question_order = qs.iter().map(|q| q.anchor).collect();
        qs.truncate(nq);
        self.meter.add(
            "L5-questions",
            kernel_tier,
            pool.len() as i64 * TOK_PER_KO + TOK_PROMPT_OVERHEAD,
            qs.len() as i64 * TOK_PER_QUESTION,
            1,
        );
        let mut new_kos: Vec<KoId> = Vec::new();
        for q in qs.iter_mut() {
            let start = self.triage_sites.get(&q.anchor).cloned();
            let avoid = self.triage_home.get(&q.anchor).map(|&h| vec![h]);
            let got =
                self.descend(q.anchor, &q.target_preds, self.cfg.descent_fanout, start.as_deref(), avoid.as_deref());
            q.answered = true;
            q.n_new_evidence = got.len();
            new_kos.extend(got);
        }
        self.flush_descent_ledger();
        self.questions = qs;
        if new_kos.is_empty() {
            return (hyps, pool);
        }
        let mut merged = pool.clone();
        merged.extend(self.merge_new(new_kos));
        if self.cfg.restrict_synthesis_to_triage {
            // The triage is the candidate generator: proposing a chain for every entity the kernel holds an
            // object about buries the real candidates.
            let mut keep: FxHashSet<i64> = self.questions.iter().map(|q| q.anchor).collect();
            keep.extend(
                pool.iter()
                    .filter(|&&k| self.arena[k].importance > self.cfg.pin_importance)
                    .map(|&k| self.arena[k].anchor),
            );
            merged.retain(|&k| keep.contains(&self.arena[k].anchor));
        }
        self.meter.add("L5-kernel-q", kernel_tier, merged.len() as i64 * TOK_PER_KO + TOK_PROMPT_OVERHEAD, 0, 1);
        let mut hyps2 = self.synthesize_pool(&merged, kernel_tier);
        self.meter.add("L5-kernel-q", kernel_tier, 0, hyps2.len() as i64 * TOK_PER_HYP, 0);
        self.enrich();
        ops::sort_by_conf(&self.store, &mut hyps2);
        if self.cfg.triage_prior_weight > 0.0 && !self.triage_gain.is_empty() {
            // Combine the cheap structural prior with the expensive verification.
            let gs: Vec<f64> =
                hyps2.iter().map(|&i| self.triage_gain.get(&self.store[i].anchor).copied().unwrap_or(0.0)).collect();
            if gs.len() > 2 && math::std(&gs, 0) > 1e-9 {
                let mean = math::mean(&gs);
                let sd = math::std(&gs, 0);
                for (&i, &g) in hyps2.iter().zip(&gs) {
                    let zz = (g - mean) / sd;
                    let x = &mut self.store[i];
                    x.prior = zz;
                    let lo = (1e-6f64.max(x.conf) / 1e-6f64.max(1.0 - x.conf)).ln();
                    x.conf = 1.0 / (1.0 + (-(lo + self.cfg.triage_prior_weight * zz)).exp());
                }
                ops::sort_by_conf(&self.store, &mut hyps2);
            }
        }
        let mut after: IndexMap<(i64, Vec<usize>), usize> = IndexMap::new();
        for &i in &hyps2 {
            after.insert(hkey(&self.store[i]), i);
        }
        let mut after_by_anchor: FxHashMap<i64, Vec<(&(i64, Vec<usize>), usize)>> = FxHashMap::default();
        for (key, &i) in &after {
            after_by_anchor.entry(self.store[i].anchor).or_default().push((key, i));
        }
        for q in self.questions.iter_mut() {
            let Some(list) = after_by_anchor.get(&q.anchor) else { continue };
            for &(key, i) in list {
                let hy = &self.store[i];
                match before.get(key) {
                    None => q.changed_conclusion = true,
                    Some(old) => {
                        let (op, np): (HashSet<usize>, HashSet<usize>) =
                            (old.1.iter().copied().collect(), hy.preds.iter().copied().collect());
                        if (old.0 - hy.conf).abs() > 0.05 || op != np {
                            q.changed_conclusion = true;
                        }
                        if old.2 > 0 && hy.contra == 0 {
                            q.resolved_contradiction = true;
                        }
                    }
                }
            }
        }
        (hyps2, merged)
    }

    /// Once two links of a chain are established for an entity, go looking for the MISSING links specifically.
    fn completion_round(&mut self, hyps: Vec<usize>, pool: Vec<KoId>, kernel_tier: &Tier) -> (Vec<usize>, Vec<KoId>) {
        let mut targets: Vec<(i64, Vec<usize>)> = Vec::new();
        let mut seen: FxHashSet<i64> = FxHashSet::default();
        let mut by_conf = hyps.clone();
        ops::sort_by_conf(&self.store, &mut by_conf);
        for &i in &by_conf {
            let hy = &self.store[i];
            if hy.hallucinated || hy.chain < 0 || hy.preds.len() < 2 || seen.contains(&hy.anchor) {
                continue;
            }
            let missing: Vec<usize> =
                SCHEMA.chain_ids[hy.chain as usize].iter().copied().filter(|p| !hy.preds.contains(p)).collect();
            if missing.is_empty() {
                continue;
            }
            seen.insert(hy.anchor);
            targets.push((hy.anchor, missing));
            if targets.len() >= self.cfg.completion_budget {
                break;
            }
        }
        if targets.is_empty() {
            return (hyps, pool);
        }
        let mut new_kos: Vec<KoId> = Vec::new();
        for (anchor, missing) in &targets {
            let start = self.triage_sites.get(anchor).cloned();
            let avoid = self.triage_home.get(anchor).map(|&h| vec![h]);
            let got = self.descend(*anchor, missing, self.cfg.descent_fanout + 2, start.as_deref(), avoid.as_deref());
            new_kos.extend(got);
        }
        self.flush_descent_ledger();
        if new_kos.is_empty() {
            return (hyps, pool);
        }
        let mut merged = pool.clone();
        merged.extend(self.merge_new(new_kos));
        let mut keep: FxHashSet<i64> = targets.iter().map(|t| t.0).collect();
        keep.extend(
            pool.iter().filter(|&&k| self.arena[k].importance > self.cfg.pin_importance).map(|&k| self.arena[k].anchor),
        );
        if self.cfg.restrict_synthesis_to_triage {
            let hyp_anchors: FxHashSet<i64> = hyps.iter().map(|&i| self.store[i].anchor).collect();
            merged.retain(|&k| keep.contains(&self.arena[k].anchor) || hyp_anchors.contains(&self.arena[k].anchor));
        }
        self.meter.add("L5-kernel-complete", kernel_tier, merged.len() as i64 * TOK_PER_KO + TOK_PROMPT_OVERHEAD, 0, 1);
        let mut hyps2 = self.synthesize_pool(&merged, kernel_tier);
        self.meter.add("L5-kernel-complete", kernel_tier, 0, hyps2.len() as i64 * TOK_PER_HYP, 0);
        self.enrich();
        ops::sort_by_conf(&self.store, &mut hyps2);
        self.completion_targets = targets.len();
        (hyps2, merged)
    }

    /// Re-read a sample of each top candidate's ORIGINAL notes with the kernel's own extractor and check they
    /// name the entity the candidate is about.
    fn verify_evidence(&mut self, mut hyps: Vec<usize>, kernel_tier: &Tier) -> Vec<usize> {
        let mut top: Vec<usize> = (0..hyps.len()).collect();
        top.sort_by(|&a, &b| (-self.store[hyps[a]].conf).partial_cmp(&-self.store[hyps[b]].conf).unwrap());
        top.truncate(self.cfg.verify_top_n);
        let (mut tin, mut tout, mut calls) = (0i64, 0i64, 0i64);
        for &pos in &top {
            let i = hyps[pos];
            let ev: Vec<i64> = self.store[i].evidence.iter().take(self.cfg.verify_sample).copied().collect();
            if ev.is_empty() {
                continue;
            }
            calls += 1;
            tin += ev.len() as i64 * TOK_PER_RECORD + 40;
            tout += 8;
            let ex = ops::extract(self.c, &ev, kernel_tier, &mut self.rng, self.near_miss, 0.08, 7);
            let hy = &mut self.store[i];
            hy.attribution = if ex.is_empty() {
                0.0
            } else {
                let hits: Vec<f64> = ex.anchor.iter().map(|&a| if a as i64 == hy.anchor { 1.0 } else { 0.0 }).collect();
                math::mean(&hits)
            };
            if self.ranker.is_some() {
                // with a fitted ranker active, attribution is a FEATURE the ranker was trained on
                let att = hy.attribution;
                if let Some(f) = hy.feat.as_mut() {
                    f.set("attribution", att);
                    continue;
                }
            }
            let lo = (1e-6f64.max(hy.conf) / 1e-6f64.max(1.0 - hy.conf)).ln()
                + self.cfg.w_attribution * (hy.attribution - 0.5);
            hy.conf = 1.0 / (1.0 + (-lo).exp());
        }
        if calls > 0 {
            self.meter.add("L5-verify", kernel_tier, tin, tout, calls);
        }
        if self.ranker.is_some() {
            ops::apply_ranker_to(self.ranker, &mut self.store, &hyps);
        }
        ops::sort_by_conf(&self.store, &mut hyps);
        hyps
    }

    fn families(&mut self, hyps: &[usize], tier: &Tier) -> Vec<Vec<usize>> {
        let mut by_chain: IndexMap<i64, Vec<usize>> = IndexMap::new();
        for (pos, &i) in hyps.iter().enumerate() {
            let hy = &self.store[i];
            if hy.chain >= 0 && !hy.hallucinated {
                by_chain.entry(hy.chain).or_default().push(pos);
            }
        }
        let mut fams = Vec::new();
        for ids in by_chain.values() {
            let anchors: FxHashSet<i64> = ids.iter().map(|&p| self.store[hyps[p]].anchor).collect();
            if ids.len() >= 2 && anchors.len() >= 2 && self.rng.random() < tier.causal_check {
                fams.push(ids.clone());
            }
        }
        if !fams.is_empty() {
            self.meter.add(
                "L5-families",
                tier,
                hyps.len() as i64 * TOK_PER_HYP + TOK_PROMPT_OVERHEAD,
                fams.len() as i64 * 30,
                1,
            );
        }
        fams
    }
}

/// What the loss accounting reads from a finished hierarchy run besides its result.
#[derive(Clone, Debug, Default)]
pub struct HierRunnerState {
    pub site_sketch: IndexMap<usize, IndexMap<i32, (i64, i64, i64, i64)>>,
    pub triage_gain: IndexMap<i64, f64>,
    pub question_order: Vec<i64>,
    pub reached_users: FxHashMap<i64, FxHashSet<usize>>,
    pub foreign_share: f64,
    pub max_reports: usize,
}

// ---------------------------------------------------------------------------------------------------------------
// Centralised baselines
// ---------------------------------------------------------------------------------------------------------------

/// Inverted index restricted to the predicate surface forms, concatenated over the causal predicates.
fn causal_candidates(corpus: &Corpus) -> Vec<i64> {
    let pred: Vec<i64> = corpus.recs.pred.iter().map(|&p| p as i64).collect();
    let order = math::argsort_stable(&pred);
    let nc = n_causal_pred() as i64;
    order.into_iter().filter(|&i| pred[i] < nc).map(|i| i as i64).collect()
}

fn max_reports_for(corpus: &Corpus) -> usize {
    6000.min(600.max(corpus.entities.len()))
}

/// `_flat_kos`: one object per (pred, anchor) over a centrally held extraction, appended to `arena`.
fn flat_kos(arena: &mut Vec<Ko>, corpus: &Corpus, ex: &ExtractResult, keep_lineage: bool) -> Vec<KoId> {
    let org = &corpus.org;
    let mut out: IndexMap<(i64, i64), KoId> = IndexMap::new();
    for i in 0..ex.len() {
        let pred = ex.pred[i] as i64;
        let anchor = ex.anchor[i] as i64;
        let u = ex.uid[i] as usize;
        let t = ex.t[i] as i64;
        let pol = ex.polarity[i] as i64;
        match out.get(&(pred, anchor)) {
            None => {
                arena.push(Ko {
                    pred,
                    anchor,
                    tmin: t,
                    tmax: t,
                    polarity: pol,
                    n_raw: 1,
                    sigs: PySet::from_one(ex.sig[i]),
                    evidence: vec![ex.rid[i]],
                    branches: if keep_lineage { branches_of(org, u) } else { Default::default() },
                    lineage: vec![u],
                    owner: org.root,
                    level: ENT,
                    conf: 0.55,
                    importance: 0.5,
                    novelty: 0.5,
                    contra: 0,
                    revisions: 0,
                    q_tag: -1,
                    origin_users: if keep_lineage { PySet::from_one(u as i64) } else { PySet::new() },
                    pos_tmax: if pol > 0 { t } else { -1 },
                    neg_tmax: if pol < 0 { t } else { -1 },
                });
                out.insert((pred, anchor), arena.len() - 1);
            }
            Some(&id) => {
                let k = &mut arena[id];
                k.tmin = k.tmin.min(t);
                k.tmax = k.tmax.max(t);
                k.n_raw += 1;
                if k.sigs.len() < MAX_SIGS {
                    k.sigs.add(ex.sig[i]);
                }
                if k.evidence.len() < MAX_EVIDENCE {
                    k.evidence.push(ex.rid[i]);
                }
                if keep_lineage {
                    for lvl in [TEAM, DEPT, SITE, REGION] {
                        if let Some(a) = org.ancestor_at(u, lvl) {
                            k.branches[lvl].get_or_insert_with(PySet::new).add(a as i64);
                        }
                    }
                    if k.origin_users.len() < MAX_USERS {
                        k.origin_users.add(u as i64);
                    }
                }
                if k.polarity != pol {
                    k.contra += 1;
                }
                if pol > 0 {
                    k.pos_tmax = k.pos_tmax.max(t);
                } else {
                    k.neg_tmax = k.neg_tmax.max(t);
                }
            }
        }
    }
    out.into_values().collect()
}

fn result_from(name: &str, syn: ops::Synthesis, meter: Meter, arena: Vec<Ko>, kos: Vec<KoId>, full: bool) -> RunResult {
    let kept = syn.kept();
    let n = syn.hyps.len();
    let retained = kos.iter().map(|&k| arena[k].key()).collect();
    RunResult {
        name: name.into(),
        store: syn.hyps,
        hypotheses: kept,
        full_hyps: if full { Some((0..n).collect()) } else { None },
        meter,
        retained,
        arena,
        kernel_kos: kos,
        ..Default::default()
    }
}

/// BASELINE A.  Schema-aware lexical retrieval over raw text, a sample up to the kernel's token budget, one call.
pub fn flat_rag(
    corpus: &Corpus,
    kernel_tier: &Tier,
    seed: u64,
    token_budget: i64,
    near_miss: &[i32],
    expand_schema: bool,
    ranker: Option<&Ranker>,
) -> RunResult {
    let mut rng = Generator::new(31_000 + seed);
    let mut meter = Meter::default();
    let cand: Vec<i64> =
        if expand_schema { causal_candidates(corpus) } else { (0..corpus.recs.len() as i64).collect() };
    let n_read = cand.len().min(1.max(token_budget.div_euclid(TOK_PER_RECORD)) as usize);
    let sel: Vec<i64> = rng.choice_without_replace(cand.len(), n_read).into_iter().map(|i| cand[i as usize]).collect();
    let ex = ops::extract_default(corpus, &sel, kernel_tier, &mut rng, near_miss);
    meter.add("retrieve", kernel_tier, 0, 0, 0);
    meter.add("L5-kernel", kernel_tier, n_read as i64 * TOK_PER_RECORD + TOK_PROMPT_OVERHEAD, ex.len() as i64 * 10, 1);
    let mut arena = Vec::new();
    let kos = flat_kos(&mut arena, corpus, &ex, true);
    let stem = stem_rep_map(corpus);
    let mut opts = SynthOpts::new(corpus.entities.len(), max_reports_for(corpus));
    opts.stem_rep = Some(&stem);
    opts.ranker = ranker;
    let syn = ops::synthesize(&arena, &kos, kernel_tier, &mut rng, &opts);
    meter.add("L5-kernel", kernel_tier, 0, syn.n_kept as i64 * TOK_PER_HYP, 0);
    let mut r = result_from("flat_rag", syn, meter, arena, kos, false);
    r.propagated_records = n_read as i64;
    r.exposed_raw_records = n_read as i64;
    r.notes.insert("records_read".into(), json!(n_read));
    r
}

/// BASELINE B.  Fill the kernel's whole context window with an unfiltered sample of raw records.
pub fn long_context(
    corpus: &Corpus,
    kernel_tier: &Tier,
    seed: u64,
    near_miss: &[i32],
    ranker: Option<&Ranker>,
) -> RunResult {
    let budget = kernel_tier.ctx - TOK_PROMPT_OVERHEAD;
    let mut r = flat_rag(corpus, kernel_tier, seed, budget, near_miss, false, ranker);
    r.name = "long_context".into();
    r
}

/// BASELINE B2, the strongest centralised competitor: the hierarchy's extraction cost, every claim pooled,
/// ranked by GLOBAL importance, the top handed to one strong kernel call.
pub fn map_reduce(
    corpus: &Corpus,
    alloc: &[Tier],
    seed: u64,
    kernel_ko_budget: usize,
    ul: &UserLayer,
    keep_lineage: bool,
    ranker: Option<&Ranker>,
) -> RunResult {
    let mut rng = Generator::new(41_000 + seed);
    let mut meter = Meter::default();
    let users = corpus.org.user_ids.len() as i64;
    meter.add(
        "L0-extract",
        &alloc[USER],
        corpus.recs.len() as i64 * TOK_PER_RECORD + users * TOK_PROMPT_OVERHEAD,
        ul.ex.len() as i64 * 10,
        users,
    );
    let kernel_tier = &alloc[ENT];
    let noise = rng.normal_n(0.0, kernel_tier.salience_noise, ul.imp.len());
    let score: Vec<f64> = ul.imp.iter().zip(&noise).map(|(a, b)| a + b).collect();
    let take = ul.ex.len().min(kernel_ko_budget * 12);
    let sel: Vec<usize> = math::argsort_desc(&score).into_iter().take(take).collect();
    let mut arena = Vec::new();
    let mut kos = flat_kos(&mut arena, corpus, &ul.ex.take(&sel), keep_lineage);
    kos.sort_by(|&a, &b| {
        let ka = -(arena[a].novelty + 0.1 * arena[a].n_raw as f64);
        let kb = -(arena[b].novelty + 0.1 * arena[b].n_raw as f64);
        ka.partial_cmp(&kb).unwrap()
    });
    kos.truncate(kernel_ko_budget);
    meter.add("L5-kernel", kernel_tier, kos.len() as i64 * TOK_PER_KO + TOK_PROMPT_OVERHEAD, 0, 1);
    let stem = stem_rep_map(corpus);
    let mut opts = SynthOpts::new(corpus.entities.len(), max_reports_for(corpus));
    opts.use_lineage = keep_lineage;
    opts.stem_rep = Some(&stem);
    opts.ranker = ranker;
    let syn = ops::synthesize(&arena, &kos, kernel_tier, &mut rng, &opts);
    meter.add("L5-kernel", kernel_tier, 0, syn.n_kept as i64 * TOK_PER_HYP, 0);
    let mut r = result_from("map_reduce", syn, meter, arena, kos, false);
    r.propagated_records = ul.ex.len() as i64;
    r.claims_leaving_node = ul.ex.len() as i64;
    r.notes.insert("pooled_claims".into(), json!(ul.ex.len()));
    r
}

/// BASELINE C.  Text-level recursive summarisation: each level keeps a fixed fraction, no structure.
pub fn recursive_summary(
    corpus: &Corpus,
    alloc: &[Tier],
    seed: u64,
    ul: &UserLayer,
    near_miss: &[i32],
    ranker: Option<&Ranker>,
) -> RunResult {
    let cfg = HierConfig {
        budgets: [4, 10, 20, 35, 60],
        lineage: false,
        independence: false,
        contradiction: false,
        temporal: false,
        adaptive_routing: false,
        questions: false,
        downward_retrieval: false,
        cross_links: false,
        adaptive_abstraction: false,
        sketch_channel: false,
        max_kernel_kos: 200,
        ..Default::default()
    };
    let (mut r, _) = HierRunner::new(corpus, alloc.to_vec(), cfg, seed, ul, near_miss, ranker).run();
    r.name = "recursive_summary".into();
    r
}

/// UPPER BOUND.  Every extracted claim, no budgets, same reasoning operators, strongest kernel.
pub fn oracle_retrieval(
    corpus: &Corpus,
    alloc: &[Tier],
    seed: u64,
    ul: &UserLayer,
    ranker: Option<&Ranker>,
) -> RunResult {
    let mut rng = Generator::new(77_000 + seed);
    let mut meter = Meter::default();
    let users = corpus.org.user_ids.len() as i64;
    meter.add(
        "L0-extract",
        &alloc[USER],
        corpus.recs.len() as i64 * TOK_PER_RECORD + users * TOK_PROMPT_OVERHEAD,
        ul.ex.len() as i64 * 10,
        users,
    );
    let mut arena = Vec::new();
    let kos = flat_kos(&mut arena, corpus, &ul.ex, true);
    let kt = &alloc[ENT];
    meter.add("L5-kernel", kt, kos.len() as i64 * TOK_PER_KO + TOK_PROMPT_OVERHEAD, 0, 1);
    let stem = stem_rep_map(corpus);
    let mut opts = SynthOpts::new(corpus.entities.len(), max_reports_for(corpus));
    opts.stem_rep = Some(&stem);
    opts.ranker = ranker;
    let syn = ops::synthesize(&arena, &kos, kt, &mut rng, &opts);
    meter.add("L5-kernel", kt, 0, syn.n_kept as i64 * TOK_PER_HYP, 0);
    let mut r = result_from("oracle_retrieval", syn, meter, arena, kos, true);
    r.propagated_records = ul.ex.len() as i64;
    r.claims_leaving_node = ul.ex.len() as i64;
    r
}

/// CONTROL.  The same candidates, ranked at random.
pub fn random_rank(mut res: RunResult, seed: u64) -> RunResult {
    let mut rng = Generator::new(88_000 + seed);
    rng.shuffle(&mut res.hypotheses);
    res.name = format!("{}_randrank", res.name);
    res
}

/// CONTROL B4: the centralised twin of the hierarchy's own algorithm - one claim pool, the same entity-level
/// relational triage, no propagation budget, no routing, no descent.
#[allow(clippy::too_many_arguments)]
pub fn central_triage(
    corpus: &Corpus,
    alloc: &[Tier],
    seed: u64,
    ul: &UserLayer,
    sketch_min_support: i64,
    foreign_share: f64,
    max_anchors: usize,
    kernel_ko_cap: usize,
    ranker: Option<&Ranker>,
) -> RunResult {
    let mut rng = Generator::new(55_000 + seed);
    let mut meter = Meter::default();
    let org = &corpus.org;
    let users = org.user_ids.len() as i64;
    meter.add(
        "L0-extract",
        &alloc[USER],
        corpus.recs.len() as i64 * TOK_PER_RECORD + users * TOK_PROMPT_OVERHEAD,
        ul.ex.len() as i64 * 10,
        users,
    );
    let ex = &ul.ex;
    let site_of: Vec<i64> =
        ex.uid.iter().map(|&u| org.ancestor_at(u as usize, SITE).map_or(-1, |x| x as i64)).collect();
    let n_ent = corpus.entities.len() as i64;
    let key: Vec<i64> = (0..ex.len())
        .map(|i| site_of[i] * (n_ent * 64) + ex.anchor[i] as i64 * 64 + (ex.pred[i] as i64).min(63))
        .collect();
    let (uk, _, cnt) = math::unique_index_counts(&key);
    let mut mask: IndexMap<(i64, i64), i64> = IndexMap::new();
    let mut tot: FxHashMap<(i64, i64), i64> = FxHashMap::default();
    let mut ent_tot: FxHashMap<i64, i64> = FxHashMap::default();
    for (&k, &c) in uk.iter().zip(&cnt) {
        let (s_, e_, p_) = (k.div_euclid(n_ent * 64), k.rem_euclid(n_ent * 64).div_euclid(64), k.rem_euclid(64));
        let c = c as i64;
        *tot.entry((s_, e_)).or_insert(0) += c;
        *ent_tot.entry(e_).or_insert(0) += c;
        if p_ < 34 && c >= sketch_min_support {
            *mask.entry((s_, e_)).or_insert(0) |= 1 << p_;
        }
    }
    let mut per_ent: IndexMap<i64, Vec<(i64, i64, i64)>> = IndexMap::new();
    for (&(s_, e_), &m_) in &mask {
        let reg = if s_ >= 0 { org.ancestor_at(s_ as usize, REGION).map_or(-1, |r| r as i64) } else { -1 };
        per_ent.entry(e_).or_default().push((s_, reg, m_));
    }
    let kt = &alloc[ENT];
    meter.add("L5-triage", kt, mask.len() as i64 * 5 + TOK_PROMPT_OVERHEAD, 0, 1);

    let mut cands: Vec<(f64, i64, Vec<i64>)> = Vec::new();
    for (&e_, rows) in &per_ent {
        let total = 1.max(ent_tot.get(&e_).copied().unwrap_or(1));
        let foreign: Vec<&(i64, i64, i64)> =
            rows.iter().filter(|r| tot[&(r.0, e_)] as f64 / total as f64 <= foreign_share).collect();
        if foreign.len() < 2 {
            continue;
        }
        let regs: FxHashSet<i64> = foreign.iter().map(|r| r.1).collect();
        if regs.len() < 2 {
            continue;
        }
        let um = foreign.iter().fold(0i64, |m, r| m | r.2);
        let (span, _) = chain_span(um);
        if span < 2 {
            continue;
        }
        let gain = (0.6 * span as f64 + 0.35 * regs.len() as f64) / (1.0 + 0.25 * (total as f64).ln_1p());
        cands.push((gain, e_, foreign.iter().map(|r| r.0).collect()));
    }
    cands.sort_by(|a, b| b.partial_cmp(a).unwrap());
    cands.truncate(max_anchors);
    let keep_ent: FxHashSet<i64> = cands.iter().map(|c| c.1).collect();
    let gain_of: FxHashMap<i64, f64> = cands.iter().map(|c| (c.1, c.0)).collect();
    let nfor_of: FxHashMap<i64, usize> = cands.iter().map(|c| (c.1, c.2.len())).collect();
    let mut home_of: FxHashMap<i64, i64> = FxHashMap::default();
    for c in &cands {
        let e_ = c.1;
        let best = per_ent[&e_].iter().map(|r| (tot[&(r.0, e_)], r.0)).max();
        home_of.insert(e_, best.map_or(-1, |b| b.1));
    }
    let sel: Vec<usize> = (0..ex.len())
        .filter(|&i| keep_ent.contains(&(ex.anchor[i] as i64)))
        .filter(|&i| site_of[i] != *home_of.get(&(ex.anchor[i] as i64)).unwrap_or(&-1))
        .collect();
    let mut arena = Vec::new();
    let mut kos = if sel.is_empty() { Vec::new() } else { flat_kos(&mut arena, corpus, &ex.take(&sel), true) };
    if kernel_ko_cap > 0 && kos.len() > kernel_ko_cap {
        // the same evidence-selection discipline the hierarchy's propagation budget forces on it
        kos.sort_by(|&a, &b| {
            let ka = -(arena[a].novelty + 0.35 * (arena[a].n_raw as f64).ln_1p());
            let kb = -(arena[b].novelty + 0.35 * (arena[b].n_raw as f64).ln_1p());
            ka.partial_cmp(&kb).unwrap()
        });
        kos.truncate(kernel_ko_cap);
    }
    meter.add("L5-kernel", kt, kos.len() as i64 * TOK_PER_KO + TOK_PROMPT_OVERHEAD, 0, 1);
    let mr = max_reports_for(corpus);
    let stem = stem_rep_map(corpus);
    let mut opts = SynthOpts::new(corpus.entities.len(), mr);
    opts.stem_rep = Some(&stem);
    opts.ranker = ranker;
    let syn = ops::synthesize(&arena, &kos, kt, &mut rng, &opts);
    let mut hyps = syn.kept();
    let mut store = syn.hyps;
    let mut full: Vec<usize> = (0..store.len()).collect();
    for hy in store.iter_mut() {
        let a = hy.anchor;
        if let Some(f) = hy.feat.as_mut() {
            f.set("triage_gain", gain_of.get(&a).copied().unwrap_or(0.0));
            f.set("sk_total", (ent_tot.get(&a).copied().unwrap_or(0) as f64).ln_1p());
            f.set("n_foreign_sites", nfor_of.get(&a).copied().unwrap_or(0) as f64);
        }
    }
    if ranker.is_some() {
        ops::apply_ranker_to(ranker, &mut store, &full);
        ops::sort_by_conf(&store, &mut full);
        hyps = full.iter().copied().take(mr).collect();
    }
    meter.add("L5-kernel", kt, 0, hyps.len() as i64 * TOK_PER_HYP, 0);
    let retained = kos.iter().map(|&k| arena[k].key()).collect();
    let mut notes = serde_json::Map::new();
    notes.insert("triage_candidates".into(), json!(cands.len()));
    notes.insert("sketch_entries".into(), json!(mask.len()));
    notes.insert("triage_anchors".into(), json!(cands.iter().map(|c| c.1).collect::<Vec<_>>()));
    RunResult {
        name: "central_triage".into(),
        store,
        hypotheses: hyps,
        full_hyps: Some(full),
        meter,
        retained,
        arena,
        kernel_kos: kos,
        notes,
        propagated_records: ex.len() as i64,
        claims_leaving_node: ex.len() as i64,
        ..Default::default()
    }
}

/// BASELINE A2, "just use more context": the kernel over every causal-predicate record in context-sized chunks,
/// objects pooled and merged, one synthesis at the end.
pub fn chunked_long_context(
    corpus: &Corpus,
    alloc: &[Tier],
    seed: u64,
    near_miss: &[i32],
    max_chunks: usize,
    expand_schema: bool,
    ranker: Option<&Ranker>,
) -> RunResult {
    let mut rng = Generator::new(61_000 + seed);
    let mut meter = Meter::default();
    let kt = &alloc[ENT];
    let mut cand: Vec<i64> =
        if expand_schema { causal_candidates(corpus) } else { (0..corpus.recs.len() as i64).collect() };
    let per_chunk = 1.max((kt.ctx - TOK_PROMPT_OVERHEAD).div_euclid(TOK_PER_RECORD)) as usize;
    let n_chunks = max_chunks.min((cand.len() as f64 / per_chunk as f64).ceil() as usize);
    rng.shuffle(&mut cand);
    let mut arena = Vec::new();
    let mut kos: Vec<KoId> = Vec::new();
    let mut read = 0usize;
    for c in 0..n_chunks {
        let lo = (c * per_chunk).min(cand.len());
        let hi = ((c + 1) * per_chunk).min(cand.len());
        let sel = &cand[lo..hi];
        if sel.is_empty() {
            break;
        }
        let ex = ops::extract_default(corpus, sel, kt, &mut rng, near_miss);
        read += sel.len();
        meter.add(
            "L5-kernel-chunk",
            kt,
            sel.len() as i64 * TOK_PER_RECORD + TOK_PROMPT_OVERHEAD,
            ex.len() as i64 * 10,
            1,
        );
        kos.extend(flat_kos(&mut arena, corpus, &ex, true));
    }
    // merge across chunks so an entity seen in several chunks becomes one object
    let mut merged: IndexMap<(i64, i64), KoId> = IndexMap::new();
    for k in kos {
        let key = arena[k].key();
        match merged.get(&key) {
            None => {
                merged.insert(key, k);
            }
            Some(&m) => merge_into(&mut arena, m, k, true),
        }
    }
    let pool: Vec<KoId> = merged.into_values().collect();
    meter.add("L5-kernel", kt, pool.len() as i64 * TOK_PER_KO + TOK_PROMPT_OVERHEAD, 0, 1);
    let stem = stem_rep_map(corpus);
    let mut opts = SynthOpts::new(corpus.entities.len(), max_reports_for(corpus));
    opts.stem_rep = Some(&stem);
    opts.ranker = ranker;
    let syn = ops::synthesize(&arena, &pool, kt, &mut rng, &opts);
    meter.add("L5-kernel", kt, 0, syn.n_kept as i64 * TOK_PER_HYP, 0);
    let mut r = result_from("chunked_long_context", syn, meter, arena, pool, true);
    r.propagated_records = read as i64;
    r.exposed_raw_records = read as i64;
    r.notes.insert("chunks".into(), json!(n_chunks));
    r.notes.insert("records_read".into(), json!(read));
    r
}

/// CONTROL Z2: is `found` gameable by enumerating everything?  The best chain span per entity, ranked by
/// raw mention volume, with no checks of any kind.
pub fn naive_enumerate(corpus: &Corpus, alloc: &[Tier], ul: &UserLayer) -> RunResult {
    let mut meter = Meter::default();
    let users = corpus.org.user_ids.len() as i64;
    meter.add(
        "L0-extract",
        &alloc[USER],
        corpus.recs.len() as i64 * TOK_PER_RECORD + users * TOK_PROMPT_OVERHEAD,
        ul.ex.len() as i64 * 10,
        users,
    );
    let ex = &ul.ex;
    let key: Vec<i64> = (0..ex.len()).map(|i| ex.anchor[i] as i64 * 64 + (ex.pred[i] as i64).min(63)).collect();
    let (uk, _, cnt) = math::unique_index_counts(&key);
    let mut per_ent: IndexMap<i64, Vec<i64>> = IndexMap::new();
    let mut tot: FxHashMap<i64, i64> = FxHashMap::default();
    for (&k, &c) in uk.iter().zip(&cnt) {
        let (e_, p_) = (k.div_euclid(64), k.rem_euclid(64));
        per_ent.entry(e_).or_default().push(p_);
        *tot.entry(e_).or_insert(0) += c as i64;
    }
    let kt = &alloc[ENT];
    let mut store: Vec<Hypothesis> = Vec::new();
    for (&e_, preds) in &per_ent {
        let mask = preds.iter().filter(|&&p| p < 34).fold(0i64, |m, &p| m | (1 << p));
        let (span, ci) = chain_span(mask);
        if span < 2 || ci < 0 {
            continue;
        }
        let on: Vec<usize> =
            SCHEMA.chain_ids[ci as usize].iter().copied().filter(|&p| preds.contains(&(p as i64))).collect();
        let t = tot.get(&e_).copied().unwrap_or(1);
        store.push(Hypothesis::enumerated(e_, on, t, (t as f64).ln_1p(), ci));
    }
    let mut order: Vec<usize> = (0..store.len()).collect();
    ops::sort_by_conf(&store, &mut order);
    order.truncate(max_reports_for(corpus));
    meter.add("L5-kernel", kt, per_ent.len() as i64 * 6 + TOK_PROMPT_OVERHEAD, order.len() as i64 * TOK_PER_HYP, 1);
    RunResult {
        name: "naive_enumerate".into(),
        store,
        hypotheses: order,
        meter,
        propagated_records: ex.len() as i64,
        claims_leaving_node: ex.len() as i64,
        ..Default::default()
    }
}

/// `CAUSAL_CHAINS[ci]` length, for callers outside `ops`.
pub fn chain_len(ci: usize) -> usize {
    CAUSAL_CHAINS[ci].len()
}
