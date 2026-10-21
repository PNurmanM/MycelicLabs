//! Shared cognitive operators (`ops.py`).
//!
//! Every architecture under test calls *these* operators; architectures differ only in which records or
//! objects reach which operator call and which tier runs it, never in the quality of the operator.  That is the
//! fairness contract.
//!
//! Leakage contract: operators may read ground-truth record fields to *simulate* a noisy model, but their output
//! never exposes `kind`, `group`, `facet`, `veracity` or `event`.
//!
//! Knowledge objects live in an arena (`Vec<Ko>`, addressed by `KoId`) because the Python objects are shared by
//! reference: a team merges a user's object into itself, a department mutates the same object, the kernel pool
//! holds it.  Indices into one arena reproduce that aliasing exactly.

use crate::corpus::{CAUSAL_CHAINS, Corpus, SCHEMA, n_causal_pred, prefix5};
use crate::models::Tier;
use crate::np::Generator;
use crate::np::math;
use crate::org::{REGION, SITE};
use crate::py::PySet;
use serde_json::Value;
use std::collections::{HashMap, HashSet};

// token cost constants (measured from the generator's own output, see calibrate_token_costs)
pub const TOK_PER_RECORD: i64 = 24;
pub const TOK_PER_KO: i64 = 26;
pub const TOK_PER_KO_NOLIN: i64 = 15;
pub const TOK_PER_HYP: i64 = 40;
pub const TOK_PER_QUESTION: i64 = 30;
pub const TOK_PROMPT_OVERHEAD: i64 = 120;

/// Mean surface tokens per record, measured on the actual corpus.
pub fn calibrate_token_costs(corpus: &Corpus, n: usize) -> f64 {
    let mut rng = Generator::new(7);
    let ids = rng.integers_n(0, corpus.recs.len() as i64, n.min(corpus.recs.len()));
    let v: Vec<f64> = ids.iter().map(|&i| (corpus.tokens(i as usize).len() + 8) as f64).collect();
    math::mean(&v)
}

// ---------------------------------------------------------------------------------------------------------------
// Extraction
// ---------------------------------------------------------------------------------------------------------------

/// Predicate confusion targets: neighbours in the same causal chain (`NEIGHBOUR_TAB`, -1 = none).
fn neighbour_table() -> Vec<[i64; 4]> {
    let mut nbs: Vec<Vec<usize>> = vec![Vec::new(); SCHEMA.predicates.len()];
    for ch in CAUSAL_CHAINS {
        for (pos, p) in ch.iter().enumerate() {
            let id = SCHEMA.pred_id[p];
            if pos > 0 {
                nbs[id].push(SCHEMA.pred_id[ch[pos - 1]]);
            }
            if pos + 1 < ch.len() {
                nbs[id].push(SCHEMA.pred_id[ch[pos + 1]]);
            }
        }
    }
    nbs.iter()
        .map(|nb| {
            let mut row = [-1i64; 4];
            for (i, &x) in nb.iter().take(4).enumerate() {
                row[i] = x as i64;
            }
            row
        })
        .collect()
}

static NEIGHBOUR_TAB: std::sync::LazyLock<Vec<[i64; 4]>> = std::sync::LazyLock::new(neighbour_table);

/// Columnar extraction output for a set of records.
#[derive(Clone, Debug, Default)]
pub struct ExtractResult {
    /// source record id (retrieval pointer)
    pub rid: Vec<i64>,
    pub uid: Vec<i64>,
    pub pred: Vec<i16>,
    pub anchor: Vec<i32>,
    pub t: Vec<i32>,
    pub polarity: Vec<i8>,
    /// near-duplicate signature (text-derived)
    pub sig: Vec<i64>,
    /// the operator invented this claim
    pub spurious: Vec<bool>,
}

impl ExtractResult {
    pub fn len(&self) -> usize {
        self.rid.len()
    }

    pub fn is_empty(&self) -> bool {
        self.rid.is_empty()
    }

    pub fn take(&self, idx: &[usize]) -> ExtractResult {
        ExtractResult {
            rid: idx.iter().map(|&i| self.rid[i]).collect(),
            uid: idx.iter().map(|&i| self.uid[i]).collect(),
            pred: idx.iter().map(|&i| self.pred[i]).collect(),
            anchor: idx.iter().map(|&i| self.anchor[i]).collect(),
            t: idx.iter().map(|&i| self.t[i]).collect(),
            polarity: idx.iter().map(|&i| self.polarity[i]).collect(),
            sig: idx.iter().map(|&i| self.sig[i]).collect(),
            spurious: idx.iter().map(|&i| self.spurious[i]).collect(),
        }
    }

    fn extend(&mut self, o: &ExtractResult) {
        self.rid.extend_from_slice(&o.rid);
        self.uid.extend_from_slice(&o.uid);
        self.pred.extend_from_slice(&o.pred);
        self.anchor.extend_from_slice(&o.anchor);
        self.t.extend_from_slice(&o.t);
        self.polarity.extend_from_slice(&o.polarity);
        self.sig.extend_from_slice(&o.sig);
        self.spurious.extend_from_slice(&o.spurious);
    }
}

/// For each entity, a lexically similar different entity id (`_near_miss_map`).
pub fn near_miss_map(corpus: &Corpus, rng: &mut Generator) -> Vec<i32> {
    let ents = &corpus.entities;
    let mut stems: Vec<(&str, Vec<i32>)> = Vec::new();
    let mut index: HashMap<&str, usize> = HashMap::new();
    for (i, e) in ents.iter().enumerate() {
        let key = prefix5(e);
        match index.get(key) {
            Some(&k) => stems[k].1.push(i as i32),
            None => {
                index.insert(key, stems.len());
                stems.push((key, vec![i as i32]));
            }
        }
    }
    let mut out: Vec<i32> = (0..ents.len() as i32).collect();
    for (_, arr) in &stems {
        if arr.len() < 2 {
            continue;
        }
        let n = arr.len() as i64;
        let perm = rng.permutation(arr.len());
        for &p in &perm {
            let shifted = arr[((p + 1) % n) as usize];
            out[arr[p as usize] as usize] = shifted;
        }
    }
    out
}

/// Deterministic uniform(0,1) keyed on an integer (splitmix64), `_hash01`.
pub fn hash01(a: i64, salt: i64) -> f64 {
    let mut x = (a as u64).wrapping_add((salt & 0x0FFF_FFFF_FFFF_FFFF) as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    x ^= x >> 30;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^= x >> 31;
    (x >> 11) as f64 / (1u64 << 53) as f64
}

/// Map every entity onto the representative of its lexical stem group.
pub fn stem_rep_map(corpus: &Corpus) -> Vec<i32> {
    let mut first: HashMap<&str, i32> = HashMap::new();
    corpus.entities.iter().enumerate().map(|(i, e)| *first.entry(prefix5(e)).or_insert(i as i32)).collect()
}

/// Simulated text -> claim extraction at a given tier (`extract`).
///
/// Error modes, all tier-governed: miss, predicate slip to a chain neighbour, entity slip to a lexically
/// similar entity (per agent and entity, not per record), spurious claims, and signature collision between
/// independent near-identical reports.
pub fn extract(
    corpus: &Corpus,
    rids: &[i64],
    tier: &Tier,
    rng: &mut Generator,
    near_miss: &[i32],
    sig_collision: f64,
    entity_salt: i64,
) -> ExtractResult {
    let recs = &corpus.recs;
    let n = rids.len();
    let keep = rng.random_n(n);
    let idx: Vec<usize> = (0..n).filter(|&i| keep[i] < tier.extract_recall).collect();
    let src: Vec<usize> = idx.iter().map(|&i| rids[i] as usize).collect();
    let mut pred: Vec<i16> = src.iter().map(|&r| recs.pred[r]).collect();
    let mut anchor: Vec<i32> = src.iter().map(|&r| recs.anchor[r]).collect();
    let t: Vec<i32> = src.iter().map(|&r| recs.t[r] as i32).collect();
    let pol: Vec<i8> = src.iter().map(|&r| recs.polarity[r]).collect();
    let mut sig: Vec<i64> = src.iter().map(|&r| recs.event[r] as i64).collect();
    let m = idx.len();

    // predicate slip
    let slip = rng.random_n(m);
    let pick = rng.integers_n(0, 4, m);
    for i in 0..m {
        let cand = NEIGHBOUR_TAB[pred[i] as usize][pick[i] as usize];
        if slip[i] < tier.pred_confusion && cand >= 0 {
            pred[i] = cand as i16;
        }
    }

    // Entity resolution errors are per (AGENT, ENTITY), not per record: an agent that mis-links "nimbus-pay" once
    // mis-links it every time.  Per-record errors would make a 7-record facet survive with probability p^7 and
    // turn the study into a referendum on extraction noise.
    let salt = (tier.q * 1e6) as i64 + entity_salt;
    let thresh = 1.0 - tier.entity_fidelity;
    for i in 0..m {
        let ukey = recs.uid[src[i]] as i64 * 1_000_003 + anchor[i] as i64;
        if hash01(ukey, salt + 11) < thresh {
            anchor[i] = near_miss[anchor[i] as usize];
        }
    }

    // signature collision between independent near-identical reports
    let coll = rng.random_n(m);
    for i in 0..m {
        if coll[i] < sig_collision {
            let key = (pred[i] as i64 * 1_000_003 + anchor[i] as i64) * 97 + (t[i] as i64).div_euclid(3);
            sig[i] = -key - 1; // negative space: collided pseudo-signatures
        }
    }

    let mut res = ExtractResult {
        rid: idx.iter().map(|&i| rids[i]).collect(),
        uid: src.iter().map(|&r| recs.uid[r] as i64).collect(),
        pred,
        anchor,
        t,
        polarity: pol,
        sig,
        spurious: vec![false; m],
    };

    // spurious claims
    let n_spur = rng.binomial(n as i64, (1.0 - tier.extract_precision).max(0.0)) as usize;
    if n_spur > 0 {
        let src = rng.integers_n(0, 1.max(m) as i64, n_spur);
        if m == 0 {
            return res;
        }
        let preds = rng.integers_n(0, SCHEMA.predicates.len() as i64, n_spur);
        let anchors = rng.integers_n(0, corpus.entities.len() as i64, n_spur);
        let sp = ExtractResult {
            rid: src.iter().map(|&s| res.rid[s as usize]).collect(),
            uid: src.iter().map(|&s| res.uid[s as usize]).collect(),
            pred: preds.iter().map(|&p| p as i16).collect(),
            anchor: anchors.iter().map(|&a| a as i32).collect(),
            t: src.iter().map(|&s| res.t[s as usize]).collect(),
            polarity: vec![1; n_spur],
            sig: (0..n_spur as i64).map(|i| -2_000_000_000 - i).collect(),
            spurious: vec![true; n_spur],
        };
        res.extend(&sp);
    }
    res
}

/// `extract` with the default signature-collision rate and no entity salt.
pub fn extract_default(
    corpus: &Corpus,
    rids: &[i64],
    tier: &Tier,
    rng: &mut Generator,
    near_miss: &[i32],
) -> ExtractResult {
    extract(corpus, rids, tier, rng, near_miss, 0.08, 0)
}

// ---------------------------------------------------------------------------------------------------------------
// Knowledge objects
// ---------------------------------------------------------------------------------------------------------------

pub type KoId = usize;

/// One propagated knowledge object.  `lineage` is the node path from the originating node up to the current
/// owner; `branches` records, per org level, the distinct ancestor nodes that contributed.
#[derive(Clone, Debug)]
pub struct Ko {
    pub pred: i64,
    pub anchor: i64,
    pub tmin: i64,
    pub tmax: i64,
    pub polarity: i64,
    pub n_raw: i64,
    pub sigs: PySet,
    pub evidence: Vec<i64>,
    /// per level (TEAM..REGION), the contributing nodes
    pub branches: [Option<PySet>; 6],
    pub lineage: Vec<usize>,
    pub owner: usize,
    pub level: usize,
    pub conf: f64,
    pub importance: f64,
    pub novelty: f64,
    pub contra: i64,
    pub revisions: i64,
    pub q_tag: i64,
    pub origin_users: PySet,
    pub pos_tmax: i64,
    pub neg_tmax: i64,
}

impl Ko {
    pub fn key(&self) -> (i64, i64) {
        (self.pred, self.anchor)
    }

    pub fn branch(&self, level: usize) -> Option<&PySet> {
        self.branches[level].as_ref()
    }

    pub fn branch_len(&self, level: usize) -> usize {
        self.branches[level].as_ref().map_or(0, |s| s.len())
    }
}

pub const MAX_EVIDENCE: usize = 6;
pub const MAX_SIGS: usize = 64;
pub const MAX_USERS: usize = 32;

/// `_merge_into(a, b)`: fold object `b` into object `a` (both in the arena).
pub fn merge_into(arena: &mut [Ko], a: KoId, b: KoId, keep_lineage: bool) {
    let bk = arena[b].clone();
    let ak = &mut arena[a];
    ak.tmin = ak.tmin.min(bk.tmin);
    ak.tmax = ak.tmax.max(bk.tmax);
    ak.pos_tmax = ak.pos_tmax.max(bk.pos_tmax);
    ak.neg_tmax = ak.neg_tmax.max(bk.neg_tmax);
    ak.n_raw += bk.n_raw;
    if ak.sigs.len() < MAX_SIGS {
        ak.sigs.update_set(&bk.sigs);
    }
    if ak.evidence.len() < MAX_EVIDENCE {
        let room = MAX_EVIDENCE - ak.evidence.len();
        ak.evidence.extend(bk.evidence.iter().take(room));
    }
    if keep_lineage {
        for (lvl, s) in bk.branches.iter().enumerate() {
            if let Some(s) = s {
                ak.branches[lvl].get_or_insert_with(PySet::new).update_set(s);
            }
        }
        if ak.origin_users.len() < MAX_USERS {
            ak.origin_users.update_set(&bk.origin_users);
        }
    }
    if ak.polarity != bk.polarity {
        ak.contra += 1;
    }
    ak.revisions += 1;
    ak.importance = ak.importance.max(bk.importance);
    ak.novelty = ak.novelty.max(bk.novelty);
    ak.conf = 0.99f64.min(1.0 - (1.0 - ak.conf) * crate::py::pow(1.0 - bk.conf, 0.5));
}

// ---------------------------------------------------------------------------------------------------------------
// Novelty / importance (observable features only, no ground truth)
// ---------------------------------------------------------------------------------------------------------------

pub fn causal_mask(pred: i64) -> bool {
    (pred as usize) < n_causal_pred()
}

/// Global observed frequency of (pred, anchor) pairs from what the user layer extracted (`BaseRate`).
pub struct BaseRate {
    pub n_ent: i64,
    pub keys: Vec<i64>,
    pub cnts: Vec<usize>,
    pub total: f64,
}

impl BaseRate {
    pub fn new(ex: &ExtractResult, n_ent: usize) -> BaseRate {
        let key: Vec<i64> = (0..ex.len()).map(|i| ex.pred[i] as i64 * n_ent as i64 + ex.anchor[i] as i64).collect();
        let (u, _, cnt) = math::unique_index_counts(&key);
        let total = cnt.iter().sum::<usize>() as f64 + 1.0;
        BaseRate { n_ent: n_ent as i64, keys: u, cnts: cnt, total }
    }

    pub fn novelty(&self, pred: &[i16], anchor: &[i32]) -> Vec<f64> {
        let lt = self.total.ln();
        pred.iter()
            .zip(anchor)
            .map(|(&p, &a)| {
                let key = p as i64 * self.n_ent + a as i64;
                let pos = self.keys.partition_point(|&k| k < key).min(self.keys.len().saturating_sub(1));
                let c = if self.keys.get(pos) == Some(&key) { self.cnts[pos] as f64 } else { 1.0 };
                (self.total / c).ln() / lt
            })
            .collect()
    }
}

/// Per-org-unit frequency of (pred, anchor): "is this entity normally seen here" (`LocalBaseRate`).
pub struct LocalBaseRate {
    pub anchor_count: Vec<f64>,
    pub count_per_record: Vec<f64>,
    pub unit_total: Vec<f64>,
    pub n_units: usize,
    pub n_entries: usize,
}

impl LocalBaseRate {
    pub fn new(ex: &ExtractResult, unit_of_record: &[i64], n_ent: usize) -> LocalBaseRate {
        let n = ex.len();
        let acomb: Vec<i64> = (0..n).map(|i| unit_of_record[i] * (1i64 << 34) + ex.anchor[i] as i64).collect();
        let (au, ainv, acnt) = math::unique_inverse_counts(&acomb);
        let comb: Vec<i64> = (0..n)
            .map(|i| unit_of_record[i] * (1i64 << 34) + (ex.pred[i] as i64 * n_ent as i64 + ex.anchor[i] as i64))
            .collect();
        let (uc, inv, cnt) = math::unique_inverse_counts(&comb);
        let (un, uinv, ucnt) = math::unique_inverse_counts(unit_of_record);
        LocalBaseRate {
            anchor_count: ainv.iter().map(|&j| acnt[j] as f64).collect(),
            count_per_record: inv.iter().map(|&j| cnt[j] as f64).collect(),
            unit_total: uinv.iter().map(|&j| ucnt[j] as f64).collect(),
            n_units: un.len(),
            n_entries: uc.len() + au.len(),
        }
    }

    pub fn novelty(&self) -> Vec<f64> {
        (0..self.unit_total.len())
            .map(|i| {
                let ut = self.unit_total[i];
                let lg = 3.0f64.max(ut).ln();
                let ent = (ut / self.anchor_count[i]).ln() / lg;
                let pair = (ut / self.count_per_record[i]).ln() / lg;
                0.75 * ent + 0.25 * pair
            })
            .collect()
    }

    /// Serialised size of the sketches that travel up/down (3 tokens per entry).
    pub fn sketch_tokens(&self, cap: usize) -> i64 {
        (3 * self.n_entries.min(cap * self.n_units)) as i64
    }
}

/// "This entity belongs somewhere else": an entity's concentration across sites times how foreign this
/// record's site is to it (`ForeignScore`).
pub struct ForeignScore {
    pub conc: Vec<f64>,
    pub share: Vec<f64>,
    pub n_entries: usize,
}

impl ForeignScore {
    pub fn new(ex: &ExtractResult, site_of_record: &[i64]) -> ForeignScore {
        let n = ex.len();
        let anc: Vec<i64> = ex.anchor.iter().map(|&a| a as i64).collect();
        let comb: Vec<i64> = (0..n).map(|i| site_of_record[i] * (1i64 << 34) + anc[i]).collect();
        let (uc, inv, cnt) = math::unique_inverse_counts(&comb);
        let (ua, ainv, acnt) = math::unique_inverse_counts(&anc);
        let share: Vec<f64> = (0..n).map(|i| cnt[inv[i]] as f64 / 1.0f64.max(acnt[ainv[i]] as f64)).collect();
        let mut best = vec![0.0f64; ua.len()];
        let last = ua.len().saturating_sub(1);
        for (j, &u) in uc.iter().enumerate() {
            let ent = u & ((1i64 << 34) - 1);
            let pos = ua.partition_point(|&a| a < ent).min(last);
            let pair_share = cnt[j] as f64 / 1.0f64.max(acnt[pos] as f64);
            if pair_share > best[pos] || best[pos].is_nan() {
                best[pos] = pair_share;
            }
        }
        ForeignScore { conc: ainv.iter().map(|&j| best[j]).collect(), share, n_entries: uc.len() }
    }

    pub fn score(&self) -> Vec<f64> {
        self.conc.iter().zip(&self.share).map(|(c, s)| c * (1.0 - s)).collect()
    }

    pub fn sketch_tokens(&self) -> i64 {
        (3 * self.n_entries) as i64
    }
}

/// Observable importance heuristic shared by every architecture (`importance_score`, default weights, no
/// watch-list - nothing in the suite sets one).
pub fn importance_score(
    pred: &[i16],
    novelty: &[f64],
    t: &[i32],
    polarity: &[i8],
    n_days: i64,
    local_novelty: Option<&[f64]>,
) -> Vec<f64> {
    let (w_causal, w_novelty, w_local, w_interaction, w_recency, w_negative) = (0.25, 0.15, 0.55, 1.10, 0.10, 0.10);
    (0..pred.len())
        .map(|i| {
            let cz = if causal_mask(pred[i] as i64) { 1.0 } else { 0.0 };
            let neg = if polarity[i] < 0 { 1.0 } else { 0.0 };
            let mut s = w_causal * cz
                + w_novelty * novelty[i]
                + w_recency * (t[i] as f64 / 1.max(n_days) as f64)
                + w_negative * neg;
            if let Some(ln) = local_novelty {
                // The interaction is load-bearing: an operational event about an entity not normally seen in
                // this unit.  Either feature alone is common; together they are rare.
                s = s + w_local * ln[i] + w_interaction * (cz * ln[i]);
            }
            s
        })
        .collect()
}

// ---------------------------------------------------------------------------------------------------------------
// Synthesis: candidate hypotheses and the four structural checks
// ---------------------------------------------------------------------------------------------------------------

/// The ranker's input features, in `calibrator.FEATURES` order.
pub const FEATURES: [&str; 39] = [
    "n_links",
    "min_sup",
    "mean_sup",
    "max_sup",
    "spread",
    "multi",
    "n_indep",
    "n_sites",
    "n_regions",
    "contra",
    "conflict",
    "dispersion",
    "synchrony",
    "verified",
    "penalty",
    "tspan",
    "max_lag",
    "min_lag",
    "mean_lag",
    "neg_lag",
    "log_n_kos",
    "from_question",
    "q_evidence_frac",
    "attribution",
    "hand_conf",
    "n_same_anchor",
    "rank_in_anchor",
    "n_origin_users",
    "min_link_users",
    "echo_ratio",
    "frac_single_witness",
    "lag_cv",
    "span_per_link",
    "chain_len_frac",
    "triage_gain",
    "sk_total",
    "n_foreign_sites",
    "users_reached",
    "n_questions_anchor",
];

pub fn feature_index(name: &str) -> Option<usize> {
    FEATURES.iter().position(|&f| f == name)
}

/// A candidate's feature vector (`Hypothesis.feat`).
#[derive(Clone, Debug, PartialEq)]
pub struct Feat(pub [f64; 39]);

impl Feat {
    pub fn get(&self, name: &str) -> f64 {
        feature_index(name).map_or(0.0, |i| self.0[i])
    }

    pub fn set(&mut self, name: &str, v: f64) {
        let i = feature_index(name).unwrap_or_else(|| panic!("unknown feature {name}"));
        self.0[i] = v;
    }
}

#[derive(Clone, Debug)]
pub struct Hypothesis {
    pub anchor: i64,
    pub preds: Vec<usize>,
    pub kos: Vec<KoId>,
    pub evidence: Vec<i64>,
    pub n_indep: i64,
    pub n_branch_regions: usize,
    pub n_branch_sites: usize,
    pub conf: f64,
    pub contra: i64,
    pub tspan: (i64, i64),
    pub chain: i64,
    pub hallucinated: bool,
    pub prior: f64,
    /// sources spread over sites, per link
    pub dispersion: f64,
    /// links whose reports all land on one day
    pub synchrony: f64,
    /// fraction of evidence that names this entity (-1 = not checked)
    pub attribution: f64,
    pub from_question: i64,
    pub min_sup: i64,
    pub spread: i64,
    pub multi: i64,
    pub conflict: f64,
    pub verified: bool,
    pub penalty: f64,
    pub link_sup: Vec<i64>,
    pub link_lag: Vec<i64>,
    pub feat: Option<Feat>,
}

impl Hypothesis {
    fn plain(anchor: i64, preds: Vec<usize>, n_indep: i64, conf: f64, chain: i64) -> Hypothesis {
        Hypothesis {
            anchor,
            preds,
            kos: Vec::new(),
            evidence: Vec::new(),
            n_indep,
            n_branch_regions: 1,
            n_branch_sites: 1,
            conf,
            contra: 0,
            tspan: (0, 0),
            chain,
            hallucinated: false,
            prior: 0.0,
            dispersion: 0.0,
            synchrony: 0.0,
            attribution: -1.0,
            from_question: -1,
            min_sup: 0,
            spread: 0,
            multi: 0,
            conflict: 0.0,
            verified: false,
            penalty: 0.0,
            link_sup: Vec::new(),
            link_lag: Vec::new(),
            feat: None,
        }
    }

    /// The enumeration control's hypothesis (no objects, no checks).
    pub fn enumerated(anchor: i64, preds: Vec<usize>, n_indep: i64, conf: f64, chain: i64) -> Hypothesis {
        Hypothesis::plain(anchor, preds, n_indep, conf, chain)
    }
}

/// Longest run of predicates from ONE causal chain present in a bitmask (`_chain_span`): (span, chain).
pub fn chain_span(mask: i64) -> (usize, i64) {
    let mut best = 0usize;
    let mut bci: i64 = -1;
    for (ci, ids) in SCHEMA.chain_ids.iter().enumerate() {
        let n = ids.iter().filter(|&&i| i < 63 && (mask >> i) & 1 == 1).count();
        if n > best {
            best = n;
            bci = ci as i64;
        }
    }
    (best, bci)
}

/// Every causal chain with at least k of its predicates in the bitmask.
pub fn chains_with_span(mask: i64, k: usize) -> Vec<usize> {
    SCHEMA
        .chain_ids
        .iter()
        .enumerate()
        .filter(|(_, ids)| ids.iter().filter(|&&i| i < 63 && (mask >> i) & 1 == 1).count() >= k)
        .map(|(ci, _)| ci)
        .collect()
}

/// Group a link's objects into time clusters no wider than `slack` days: (earliest tmin, independent signatures,
/// objects) per cluster, earliest first.
fn time_clusters(arena: &[Ko], ks: &[KoId], slack: i64) -> Vec<(i64, usize, usize)> {
    let mut sorted: Vec<KoId> = ks.to_vec();
    sorted.sort_by_key(|&k| arena[k].tmin);
    let mut out = Vec::new();
    let mut cur_t: Option<i64> = None;
    let mut cur_sigs: HashSet<i64> = HashSet::new();
    let mut cur_n = 0usize;
    for k in sorted {
        let ko = &arena[k];
        if cur_t.is_none_or(|t| ko.tmin > t + slack) {
            if let Some(t) = cur_t {
                out.push((t, cur_sigs.len(), cur_n));
            }
            cur_t = Some(ko.tmin);
            cur_sigs.clear();
            cur_n = 0;
        }
        cur_sigs.extend(ko.sigs.iter());
        cur_n += 1;
    }
    if let Some(t) = cur_t {
        out.push((t, cur_sigs.len(), cur_n));
    }
    out
}

/// Most independent signatures, then most objects, then earliest (the first such cluster).
fn heaviest_cluster(cl: &[(i64, usize, usize)]) -> (i64, usize, usize) {
    let mut best = cl[0];
    for &c in &cl[1..] {
        if (c.1, c.2, -c.0) > (best.1, best.2, -best.0) {
            best = c;
        }
    }
    best
}

/// True iff preds form a set of >= 2 distinct predicates lying on one chain (`_valid_chain_path`).
pub fn valid_chain_path(preds: &[usize]) -> (bool, i64) {
    let mut best = (false, -1);
    for (ci, ids) in SCHEMA.chain_ids.iter().enumerate() {
        let pos: Vec<usize> = preds.iter().filter_map(|p| ids.iter().position(|x| x == p)).collect();
        let distinct: HashSet<usize> = pos.iter().copied().collect();
        if pos.len() >= 2 && distinct.len() == pos.len() {
            if pos.iter().max().unwrap() - pos.iter().min().unwrap() <= preds.len() + 1 {
                return (true, ci as i64);
            }
            best = (true, ci as i64);
        }
    }
    best
}

// --------------------------------------------------------------------------------------------------------------
// The shared calibrated ranker.  When a fitted ranker is active it REPLACES the hand-set confidence for every
// system alike: one shared operator, fitted on seeds 500-502 and frozen.
// --------------------------------------------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub enum Tree {
    Leaf(f64),
    Split { f: usize, t: f64, l: Box<Tree>, r: Box<Tree> },
}

impl Tree {
    pub fn from_json(v: &Value) -> Tree {
        if let Some(leaf) = v.get("leaf") {
            return Tree::Leaf(leaf.as_f64().expect("leaf"));
        }
        Tree::Split {
            f: v["f"].as_u64().expect("f") as usize,
            t: v["t"].as_f64().expect("t"),
            l: Box::new(Tree::from_json(&v["l"])),
            r: Box::new(Tree::from_json(&v["r"])),
        }
    }

    pub fn score(&self, x: &[f64]) -> f64 {
        let mut node = self;
        loop {
            match node {
                Tree::Leaf(v) => return *v,
                Tree::Split { f, t, l, r } => node = if x[*f] <= *t { l } else { r },
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct Ranker {
    pub gbdt: bool,
    /// feature names, `a*b` for an interaction
    pub features: Vec<String>,
    pub weights: Vec<f64>,
    pub bias: f64,
    pub mu: Vec<f64>,
    pub sd: Vec<f64>,
    pub lr: f64,
    pub trees: Vec<Tree>,
    /// the JSON it was read from (`calibration.json["ranker"]`)
    pub json: Value,
}

impl Ranker {
    pub fn from_json(v: &Value) -> Ranker {
        let fl = |k: &str| -> Vec<f64> {
            v.get(k)
                .and_then(Value::as_array)
                .map(|a| a.iter().map(|x| x.as_f64().unwrap()).collect())
                .unwrap_or_default()
        };
        Ranker {
            gbdt: v.get("kind").and_then(Value::as_str) == Some("gbdt"),
            features: v["features"].as_array().unwrap().iter().map(|x| x.as_str().unwrap().to_string()).collect(),
            weights: fl("weights"),
            bias: v.get("bias").and_then(Value::as_f64).unwrap_or(0.0),
            mu: fl("mu"),
            sd: fl("sd"),
            lr: v.get("lr").and_then(Value::as_f64).unwrap_or(0.0),
            trees: v
                .get("trees")
                .and_then(Value::as_array)
                .map(|a| a.iter().map(Tree::from_json).collect())
                .unwrap_or_default(),
            json: v.clone(),
        }
    }

    fn value(feat: &Feat, f: &str) -> f64 {
        match f.split_once('*') {
            Some((a, b)) => feat.get(a) * feat.get(b),
            None => feat.get(f),
        }
    }

    /// `ranker_score(feat, r)`
    pub fn score(&self, feat: &Feat) -> f64 {
        if self.gbdt {
            let x: Vec<f64> = self
                .features
                .iter()
                .zip(self.mu.iter().zip(&self.sd))
                .map(|(f, (mu, sd))| (Self::value(feat, f) - mu) / sd)
                .collect();
            // Python's sum() over a generator: left to right from 0
            let mut z = 0.0;
            for t in &self.trees {
                z += self.lr * t.score(&x);
            }
            return 1.0 / (1.0 + (-z).exp());
        }
        let mut z = self.bias;
        for (i, f) in self.features.iter().enumerate() {
            let v = Self::value(feat, f);
            z += self.weights[i] * (v - self.mu[i]) / self.sd[i];
        }
        1.0 / (1.0 + (-z).exp())
    }
}

/// Anchor-level context every candidate carries: how many candidates share its entity, and where it sits among
/// them by the hand score.
pub fn annotate_anchor_context(hyps: &mut [Hypothesis]) {
    let mut by_anchor: Vec<(i64, Vec<usize>)> = Vec::new();
    let mut index: HashMap<i64, usize> = HashMap::new();
    for (i, h) in hyps.iter().enumerate() {
        if !h.hallucinated && h.feat.is_some() {
            let slot = *index.entry(h.anchor).or_insert_with(|| {
                by_anchor.push((h.anchor, Vec::new()));
                by_anchor.len() - 1
            });
            by_anchor[slot].1.push(i);
        }
    }
    for (_, hs) in by_anchor.iter_mut() {
        hs.sort_by(|&a, &b| {
            let ka = -hyps[a].feat.as_ref().unwrap().get("hand_conf");
            let kb = -hyps[b].feat.as_ref().unwrap().get("hand_conf");
            ka.partial_cmp(&kb).unwrap()
        });
        let n = hs.len() as f64;
        for (i, &h) in hs.iter().enumerate() {
            let f = hyps[h].feat.as_mut().unwrap();
            f.set("n_same_anchor", n);
            f.set("rank_in_anchor", i as f64);
        }
    }
}

/// `apply_ranker_to(hyps)` over the candidates at `idx`: keep the number the hand logistic put at or above 0.5,
/// but let the learned score decide which candidates fill those slots.
pub fn apply_ranker_to(ranker: Option<&Ranker>, hyps: &mut [Hypothesis], idx: &[usize]) {
    let Some(r) = ranker else { return };
    if idx.is_empty() {
        return;
    }
    let real: Vec<usize> = idx.iter().copied().filter(|&i| !hyps[i].hallucinated && hyps[i].feat.is_some()).collect();
    let k = real.iter().filter(|&&i| hyps[i].feat.as_ref().unwrap().get("hand_conf") >= 0.5).count();
    let mut scored: Vec<(f64, usize)> = real.iter().map(|&i| (r.score(hyps[i].feat.as_ref().unwrap()), i)).collect();
    scored.sort_by(|a, b| (-a.0).partial_cmp(&-b.0).unwrap());
    for (pos, (p, i)) in scored.into_iter().enumerate() {
        hyps[i].conf = if pos < k { 0.5 + 0.5 * p } else { 0.5 * p };
    }
}

/// Stable sort of candidate indices by descending confidence (`list.sort(key=lambda h: -h.conf)`).
pub fn sort_by_conf(hyps: &[Hypothesis], idx: &mut [usize]) {
    idx.sort_by(|&a, &b| (-hyps[a].conf).partial_cmp(&-hyps[b].conf).unwrap());
}

/// Options of `synthesize` (Python keyword arguments with their defaults).
#[derive(Clone, Debug)]
pub struct SynthOpts<'a> {
    pub min_preds: usize,
    pub use_lineage: bool,
    pub use_dedup: bool,
    pub use_temporal: bool,
    pub use_entity_check: bool,
    pub n_entities: usize,
    pub max_reports: usize,
    pub min_link_support: usize,
    pub w_dispersion: f64,
    pub w_synchrony: f64,
    pub stem_rep: Option<&'a [i32]>,
    pub question_tag: i64,
    pub link_time: &'a str,
    pub stale_rule: &'a str,
    pub ranker: Option<&'a Ranker>,
}

impl<'a> SynthOpts<'a> {
    pub fn new(n_entities: usize, max_reports: usize) -> Self {
        SynthOpts {
            min_preds: 2,
            use_lineage: true,
            use_dedup: true,
            use_temporal: true,
            use_entity_check: true,
            n_entities,
            max_reports,
            min_link_support: 1,
            w_dispersion: 0.0,
            w_synchrony: 0.0,
            stem_rep: None,
            question_tag: -1,
            link_time: "min",
            stale_rule: "any",
            ranker: None,
        }
    }
}

/// What `synthesize` produced: every candidate (sorted by confidence: the `full_out` list) and how many of them
/// the register keeps (`out[:max_reports]`).
pub struct Synthesis {
    pub hyps: Vec<Hypothesis>,
    pub n_kept: usize,
}

impl Synthesis {
    /// Indices of the register (the returned list), in order.
    pub fn kept(&self) -> Vec<usize> {
        (0..self.n_kept).collect()
    }
}

fn union_len<'a>(sets: impl Iterator<Item = &'a PySet>) -> usize {
    let mut u: HashSet<i64> = HashSet::new();
    for s in sets {
        u.extend(s.iter());
    }
    u.len()
}

/// `synthesize`: enumerate candidate strategic patterns per anchor and chain, and verify them with four checks
/// (causal, temporal incl. staleness, dedup, entity) that a tier passes with its own probability and otherwise
/// fails open.  See `ops.py` for the full rationale of each step.
pub fn synthesize(arena: &[Ko], kos: &[KoId], tier: &Tier, rng: &mut Generator, opts: &SynthOpts) -> Synthesis {
    let schema = &*SCHEMA;
    // Entity linking: a weak tier collapses lexically similar names onto one identity, per entity.
    let mut amap: HashMap<i64, i64> = HashMap::new();
    if let (Some(stem_rep), true) = (opts.stem_rep, opts.use_entity_check) {
        let mut uniq: Vec<i64> = kos.iter().map(|&k| arena[k].anchor).collect::<HashSet<i64>>().into_iter().collect();
        uniq.sort();
        let salt = (tier.q * 1e6) as i64 + 7717;
        for a in uniq {
            if hash01(a, salt) < 1.0 - tier.entity_check && stem_rep[a as usize] as i64 != a {
                amap.insert(a, stem_rep[a as usize] as i64);
            }
        }
    }
    let mut by_anchor: Vec<(i64, Vec<KoId>)> = Vec::new();
    {
        let mut index: HashMap<i64, usize> = HashMap::new();
        for &k in kos {
            let a = *amap.get(&arena[k].anchor).unwrap_or(&arena[k].anchor);
            let slot = *index.entry(a).or_insert_with(|| {
                by_anchor.push((a, Vec::new()));
                by_anchor.len() - 1
            });
            by_anchor[slot].1.push(k);
        }
    }

    let chain_pos: Vec<HashMap<usize, usize>> =
        schema.chain_ids.iter().map(|ids| ids.iter().enumerate().map(|(i, &p)| (p, i)).collect()).collect();

    let mut out: Vec<Hypothesis> = Vec::new();

    let support = |ks: &[KoId]| union_len(ks.iter().map(|&k| &arena[k].sigs));

    let mk = |anchor: i64,
              plist: &[usize],
              pred_kos: &[(usize, Vec<KoId>)],
              chain: i64,
              penalty: f64,
              verified: bool,
              rng: &mut Generator,
              out: &mut Vec<Hypothesis>| {
        let get = |p: usize| -> &Vec<KoId> { &pred_kos.iter().find(|(q, _)| *q == p).unwrap().1 };
        // ---- per-link independent support: a chain is only as strong as its weakest link ----
        let mut link_sup: Vec<i64> = Vec::new();
        let mut link_modal: Vec<i64> = Vec::new();
        let mut disp: Vec<f64> = Vec::new();
        let mut sync: Vec<f64> = Vec::new();
        for &p in plist {
            let ks = get(p);
            let mut sg: HashSet<i64> = HashSet::new();
            let mut st: HashSet<i64> = HashSet::new();
            // region -> raw count, in first-seen order (max() breaks ties by that order)
            let mut rc: Vec<(i64, i64)> = Vec::new();
            let (mut tlo, mut thi, mut nraw) = (1_000_000_000i64, -1i64, 0i64);
            for &k in ks {
                let ko = &arena[k];
                sg.extend(ko.sigs.iter());
                nraw += ko.n_raw;
                tlo = tlo.min(ko.tmin);
                thi = thi.max(ko.tmax);
                if opts.use_lineage {
                    if let Some(st_set) = ko.branch(SITE) {
                        st.extend(st_set.iter());
                    }
                    if let Some(kr) = ko.branch(REGION) {
                        for rr in kr.iter() {
                            match rc.iter_mut().find(|e| e.0 == rr) {
                                Some(e) => e.1 += ko.n_raw,
                                None => rc.push((rr, ko.n_raw)),
                            }
                        }
                    }
                }
            }
            let mut modal = -1i64;
            let mut best = i64::MIN;
            for &(r, c) in &rc {
                if c > best {
                    best = c;
                    modal = r;
                }
            }
            link_modal.push(modal);
            // SOURCE DISPERSION: reports spread over sites, or several departments of one loud site?
            disp.push(if opts.use_lineage { st.len() as f64 / 1.0f64.max(sg.len() as f64) } else { 1.0 });
            // SYNCHRONY: many reports landing in the same day or two are one announcement fanned out
            let span = 0.max(thi - tlo);
            sync.push(if nraw >= 4 && span <= 1 { 1.0 } else { 0.0 });
            if !opts.use_dedup || rng.random() > tier.dedup_check {
                link_sup.push(ks.iter().map(|&k| arena[k].n_raw).sum()); // echoes counted
            } else {
                link_sup.push(sg.len() as i64);
            }
        }
        if plist.len() < opts.min_preds {
            return;
        }
        let members: Vec<KoId> = plist.iter().flat_map(|&p| get(p).iter().copied()).collect();
        let mut n_indep = union_len(members.iter().map(|&k| &arena[k].sigs)) as i64;
        if !opts.use_dedup || rng.random() > tier.dedup_check {
            n_indep = members.iter().map(|&k| arena[k].n_raw).sum();
        }
        let mut regions: HashSet<i64> = HashSet::new();
        let mut sites: HashSet<i64> = HashSet::new();
        if opts.use_lineage {
            for &k in &members {
                if let Some(s) = arena[k].branch(REGION) {
                    regions.extend(s.iter());
                }
                if let Some(s) = arena[k].branch(SITE) {
                    sites.extend(s.iter());
                }
            }
        }
        let mut ev: Vec<i64> = Vec::new();
        for &k in &members {
            ev.extend(arena[k].evidence.iter().take(3));
        }
        // Contradiction recomputed from the polarities in front of the kernel, as a BALANCE of evidence.
        let mut contra = 0i64;
        let mut minority: Vec<f64> = Vec::new();
        for &p in plist {
            let (mut npos, mut nneg, mut flips) = (0i64, 0i64, 0i64);
            for &k in get(p) {
                let ko = &arena[k];
                flips += ko.contra;
                if ko.polarity < 0 {
                    nneg += 1.max(ko.n_raw);
                } else {
                    npos += 1.max(ko.n_raw);
                }
            }
            nneg += flips;
            let (lo_, hi_) = (npos.min(nneg), npos.max(nneg));
            minority.push(lo_ as f64 / 1.0f64.max((lo_ + hi_) as f64));
            let flagged = lo_ >= 2 && lo_ as f64 >= 0.25 * hi_ as f64;
            if rng.random() < tier.contradiction_acc {
                contra += if flagged { 1 } else { 0 };
            } else {
                contra += if flagged { 0 } else { 1 }; // a weak tier misreads both ways
            }
        }
        let conflict = if minority.is_empty() { 0.0 } else { math::mean(&minority) };
        let (spread, multi) = if opts.use_lineage {
            let distinct: HashSet<i64> = link_modal.iter().copied().filter(|&r| r >= 0).collect();
            (regions.len() as i64, distinct.len() as i64)
        } else {
            let raw: i64 = members.iter().map(|&k| arena[k].n_raw).sum();
            (4.min(1.max(raw / 4)), 1)
        };
        let min_sup = link_sup.iter().copied().min().unwrap_or(0);
        let mean_disp = if disp.is_empty() { 1.0 } else { math::mean(&disp) };
        let frac_sync = if sync.is_empty() { 0.0 } else { math::mean(&sync) };
        let z = -3.25
            + 0.35 * 4.min(plist.len() as i64) as f64
            + 0.45 * 4.min(min_sup) as f64
            + 0.30 * 3.min(0.max(spread - 1)) as f64
            + 0.95 * 3.min(0.max(multi - 1)) as f64
            + (if verified { 0.55 } else { 0.0 })
            - 0.30 * 3.min(contra) as f64
            - 3.2 * conflict
            - 3.0 * penalty
            + opts.w_dispersion * (mean_disp - 0.5)
            - opts.w_synchrony * frac_sync;
        let conf = 1.0 / (1.0 + (-z).exp());
        let tm: Vec<i64> = plist.iter().map(|&p| get(p).iter().map(|&k| arena[k].tmin).min().unwrap()).collect();
        let mut lags: Vec<i64> = tm.windows(2).map(|w| w[1] - w[0]).collect();
        if lags.is_empty() {
            lags.push(0);
        }
        let tmin_all = members.iter().map(|&k| arena[k].tmin).min().unwrap();
        let tmax_all = members.iter().map(|&k| arena[k].tmax).max().unwrap();
        let tspan = tmax_all - tmin_all;
        let mut f = Feat([0.0; 39]);
        f.set("n_links", plist.len() as f64);
        f.set("min_sup", min_sup as f64);
        f.set("mean_sup", math::mean_int(&link_sup));
        f.set("max_sup", *link_sup.iter().max().unwrap() as f64);
        f.set("spread", spread as f64);
        f.set("multi", multi as f64);
        f.set("n_indep", 50.min(n_indep) as f64);
        f.set("n_sites", sites.len() as f64);
        f.set("n_regions", regions.len() as f64);
        f.set("contra", contra as f64);
        f.set("conflict", conflict);
        f.set("dispersion", mean_disp);
        f.set("synchrony", frac_sync);
        f.set("verified", if verified { 1.0 } else { 0.0 });
        f.set("penalty", penalty);
        f.set("tspan", tspan as f64);
        f.set("max_lag", *lags.iter().max().unwrap() as f64);
        f.set("min_lag", *lags.iter().min().unwrap() as f64);
        f.set("mean_lag", math::mean_int(&lags));
        f.set("neg_lag", if *lags.iter().min().unwrap() < 0 { 1.0 } else { 0.0 });
        f.set("log_n_kos", (members.len() as f64).ln_1p());
        f.set("from_question", if opts.question_tag >= 0 { 1.0 } else { 0.0 });
        let qev: Vec<f64> = members.iter().map(|&k| if arena[k].q_tag >= 0 { 1.0 } else { 0.0 }).collect();
        f.set("q_evidence_frac", math::mean(&qev));
        f.set("attribution", -1.0);
        f.set("hand_conf", conf);
        // evidence-shape features: who reported it, and how independently
        let mut users: HashSet<i64> = HashSet::new();
        let mut nraw_all = 0i64;
        let mut sigs_all: HashSet<i64> = HashSet::new();
        let mut link_users: Vec<f64> = Vec::new();
        let mut link_raw: Vec<i64> = Vec::new();
        for &p in plist {
            let mut lu: HashSet<i64> = HashSet::new();
            let mut lr = 0i64;
            for &k in get(p) {
                lu.extend(arena[k].origin_users.iter());
                lr += arena[k].n_raw;
                sigs_all.extend(arena[k].sigs.iter());
            }
            link_users.push(lu.len() as f64);
            users.extend(lu);
            nraw_all += lr;
            link_raw.push(lr);
        }
        f.set("n_origin_users", users.len() as f64);
        f.set("min_link_users", link_users.iter().copied().fold(f64::INFINITY, f64::min));
        f.set("echo_ratio", nraw_all as f64 / 1.0f64.max(sigs_all.len() as f64));
        let single: Vec<f64> = link_raw.iter().map(|&x| if x <= 1 { 1.0 } else { 0.0 }).collect();
        f.set("frac_single_witness", math::mean(&single));
        f.set(
            "lag_cv",
            if lags.len() > 1 { math::std_int(&lags, 0) / 1.0f64.max(math::mean_int(&lags).abs()) } else { 0.0 },
        );
        f.set("span_per_link", tspan as f64 / 1.0f64.max(plist.len() as f64));
        f.set(
            "chain_len_frac",
            if chain >= 0 { plist.len() as f64 / CAUSAL_CHAINS[chain as usize].len() as f64 } else { 0.0 },
        );
        ev.truncate(24);
        out.push(Hypothesis {
            anchor,
            preds: plist.to_vec(),
            kos: members,
            evidence: ev,
            n_indep,
            n_branch_regions: regions.len(),
            n_branch_sites: sites.len(),
            conf,
            contra,
            tspan: (tmin_all, tmax_all),
            chain,
            hallucinated: false,
            prior: 0.0,
            dispersion: mean_disp,
            synchrony: frac_sync,
            attribution: -1.0,
            from_question: opts.question_tag,
            min_sup,
            spread,
            multi,
            conflict,
            verified,
            penalty,
            link_lag: tm.windows(2).map(|w| w[1] - w[0]).collect(),
            link_sup,
            feat: Some(f),
        });
    };

    let depth = 2.max(tier.synth_depth) as usize;
    for (anchor, group) in &by_anchor {
        // pred -> objects, in first-seen order
        let mut preds: Vec<(usize, Vec<KoId>)> = Vec::new();
        for &k in group {
            let p = arena[k].pred as usize;
            match preds.iter_mut().find(|e| e.0 == p) {
                Some(e) => e.1.push(k),
                None => preds.push((p, vec![k])),
            }
        }
        if preds.len() < opts.min_preds {
            continue;
        }
        let causal_ok = rng.random() < tier.causal_check;
        if causal_ok {
            for (ci, pos) in chain_pos.iter().enumerate() {
                let mut on: Vec<usize> = preds.iter().map(|e| e.0).filter(|p| pos.contains_key(p)).collect();
                if on.len() < opts.min_preds {
                    continue;
                }
                on.sort_by_key(|p| pos[p]);
                let get = |p: usize| -> &Vec<KoId> { &preds.iter().find(|(q, _)| *q == p).unwrap().1 };
                // (1) thin links first: a single stray mention must not complete a plausible causal story
                if opts.use_dedup && rng.random() < tier.dedup_check {
                    let strong: Vec<usize> =
                        on.iter().copied().filter(|&p| support(get(p)) >= opts.min_link_support).collect();
                    if strong.len() >= opts.min_preds {
                        on = strong;
                    }
                }
                // (2) temporal: the heaviest chain-ordered, time-ordered sub-path
                if opts.use_temporal {
                    let (clusters, tmins): (Vec<Vec<(i64, usize, usize)>>, Vec<i64>) = if opts.link_time == "modal"
                        || opts.link_time == "hybrid"
                    {
                        let cl: Vec<Vec<(i64, usize, usize)>> =
                            on.iter().map(|&p| time_clusters(arena, get(p), 3)).collect();
                        let tm = cl.iter().map(|c| heaviest_cluster(c).0).collect();
                        (cl, tm)
                    } else {
                        (Vec::new(), on.iter().map(|&p| get(p).iter().map(|&k| arena[k].tmin).min().unwrap()).collect())
                    };
                    if rng.random() < tier.temporal_check {
                        let mut path: Vec<usize> = Vec::new();
                        if opts.link_time == "hybrid" {
                            // one state per (link, time cluster) for links whose witnesses disagree on a time
                            let mut states: Vec<(usize, i64, f64)> = Vec::new();
                            for (i2, &p) in on.iter().enumerate() {
                                let heavy = heaviest_cluster(&clusters[i2]);
                                if heavy.1 >= 2 {
                                    states.push((i2, heavy.0, (support(get(p)) as f64).ln_1p() + 0.35));
                                } else {
                                    for &(t0, nsig, _) in &clusters[i2] {
                                        states.push((i2, t0, (nsig as f64).ln_1p() + 0.35));
                                    }
                                }
                            }
                            let mut sbest: Vec<f64> = states.iter().map(|s| s.2).collect();
                            let mut sprev: Vec<i64> = vec![-1; states.len()];
                            for si in 0..states.len() {
                                let (li, ti, wi) = states[si];
                                for sj in 0..si {
                                    let (lj, tj, _) = states[sj];
                                    if lj < li && tj <= ti + 3 && sbest[sj] + wi > sbest[si] {
                                        sbest[si] = sbest[sj] + wi;
                                        sprev[si] = sj as i64;
                                    }
                                }
                            }
                            let mut end = math::argmax(&sbest) as i64;
                            while end >= 0 {
                                path.push(on[states[end as usize].0]);
                                end = sprev[end as usize];
                            }
                        } else {
                            let wts: Vec<f64> = on.iter().map(|&p| (support(get(p)) as f64).ln_1p() + 0.35).collect();
                            let mut best = wts.clone();
                            let mut prev: Vec<i64> = vec![-1; on.len()];
                            for i2 in 0..on.len() {
                                for j2 in 0..i2 {
                                    if tmins[j2] <= tmins[i2] + 3 && best[j2] + wts[i2] > best[i2] {
                                        best[i2] = best[j2] + wts[i2];
                                        prev[i2] = j2 as i64;
                                    }
                                }
                            }
                            let mut end = math::argmax(&best) as i64;
                            while end >= 0 {
                                path.push(on[end as usize]);
                                end = prev[end as usize];
                            }
                        }
                        path.reverse();
                        if path.len() < opts.min_preds {
                            continue;
                        }
                        on = path;
                    }
                    // staleness: the chain was retracted later than asserted
                    let mem0: Vec<KoId> = on.iter().flat_map(|&p| get(p).iter().copied()).collect();
                    let neg = mem0.iter().map(|&k| arena[k].neg_tmax).max().unwrap_or(-1);
                    let pos_t = if opts.stale_rule == "strong" {
                        // one routine positive mention after the retraction must not revive a retracted chain
                        let strong = mem0
                            .iter()
                            .filter(|&&k| arena[k].pos_tmax >= 0 && arena[k].sigs.len() >= 2)
                            .map(|&k| arena[k].pos_tmax)
                            .max()
                            .unwrap_or(-1);
                        if strong < 0 { mem0.iter().map(|&k| arena[k].pos_tmax).max().unwrap_or(-1) } else { strong }
                    } else {
                        mem0.iter().map(|&k| arena[k].pos_tmax).max().unwrap_or(-1)
                    };
                    if neg > pos_t && pos_t >= 0 && rng.random() < tier.temporal_check {
                        continue;
                    }
                }
                on.truncate(depth);
                mk(*anchor, &on, &preds, ci as i64, 0.0, true, rng, &mut out);
            }
        } else {
            let mut ordered: Vec<(usize, f64)> = preds
                .iter()
                .map(|(p, ks)| (*p, ks.iter().map(|&k| arena[k].importance).fold(f64::NEG_INFINITY, f64::max)))
                .collect();
            ordered.sort_by(|a, b| (-a.1).partial_cmp(&-b.1).unwrap());
            ordered.truncate(depth);
            let plist: Vec<usize> = ordered.iter().map(|x| x.0).collect();
            let mut ci = -1i64;
            for (cj, pos) in chain_pos.iter().enumerate() {
                if plist.iter().filter(|p| pos.contains_key(p)).count() >= 2 {
                    ci = cj as i64;
                    break;
                }
            }
            mk(*anchor, &plist, &preds, ci, 0.06, false, rng, &mut out);
        }
    }

    // --- hallucination: unsupported hypotheses ---
    let n_hall = rng.binomial(1.max(out.len()) as i64, tier.hallucination);
    for _ in 0..n_hall {
        let ci = rng.integers(0, CAUSAL_CHAINS.len() as i64) as usize;
        let ch = &schema.chain_ids[ci];
        let l = rng.integers(2, 4.min(ch.len()) as i64 + 1) as usize;
        let sp = rng.integers(0, (ch.len() - l + 1) as i64) as usize;
        let anchor = rng.integers(0, opts.n_entities as i64);
        let n_indep = rng.integers(1, 5);
        let conf = 0.3 + 0.4 * rng.random();
        let mut h = Hypothesis::plain(anchor, ch[sp..sp + l].to_vec(), n_indep, conf, ci as i64);
        h.hallucinated = true;
        out.push(h);
    }
    annotate_anchor_context(&mut out);
    if opts.ranker.is_some() {
        let all: Vec<usize> = (0..out.len()).collect();
        apply_ranker_to(opts.ranker, &mut out, &all);
    }
    let mut order: Vec<usize> = (0..out.len()).collect();
    sort_by_conf(&out, &mut order);
    let mut slots: Vec<Option<Hypothesis>> = out.into_iter().map(Some).collect();
    let hyps: Vec<Hypothesis> = order.iter().map(|&i| slots[i].take().unwrap()).collect();
    let n_kept = hyps.len().min(opts.max_reports);
    Synthesis { hyps, n_kept }
}
