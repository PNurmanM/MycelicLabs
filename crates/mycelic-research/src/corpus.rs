//! Synthetic enterprise corpus with hidden, verifiable ground truth (`corpus.py`).
//!
//! Design goals, each a defence against a way of rigging the benchmark in the hierarchy's favour:
//!
//! 1. A record's *content* is a structured tuple (predicate, anchor entity, aux entity, timestamp, event id,
//!    polarity); its surface text is a lossy, synonym-noised realisation of that tuple, read by every system
//!    through the same tier-dependent extraction operator.
//! 2. A strategic pattern is a contiguous path in a causal predicate schema, bound to one anchor entity, with
//!    non-decreasing timestamps and facets scattered across distant org branches: invisible to any single
//!    team, equally visible to any system that gets the facet records into one context.
//! 3. Decoys share the surface statistics of real patterns and differ only in verifiable structure (D1 entity
//!    coincidence, D2 temporal scramble, D3 near-miss entity, D4 duplicate inflation, D5 stale chain).
//! 4. Rare-signal patterns are supported by 1-2 users per facet out of the whole enterprise.
//!
//! The record table is held column by column (`Recs`), in the dtypes of `REC_DTYPE`.

use crate::np::Generator;
use crate::np::math;
use crate::org::{Org, REGION, SITE};
use blake2::Blake2bVar;
use blake2::digest::{Update, VariableOutput};
use serde_json::{Map, Value, json};
use std::collections::{HashMap, HashSet};
use std::sync::LazyLock;

// --------------------------------------------------------------------------------------------------------------
// Predicate / causal schema
// --------------------------------------------------------------------------------------------------------------

/// Each chain is an ordered list of predicate names; a real pattern is a contiguous sub-path of one chain.
pub const CAUSAL_CHAINS: [&[&str]; 7] = [
    &[
        "supplier_substitution",
        "component_defect",
        "yield_drop",
        "shipment_delay",
        "customer_escalation",
        "revenue_miss",
    ],
    &["cert_rotation", "auth_failure", "api_error_spike", "checkout_drop", "revenue_miss"],
    &["policy_change", "manual_workaround", "data_exposure_risk", "audit_finding", "regulatory_notice"],
    &["attrition_spike", "backlog_growth", "sla_breach", "contract_penalty", "account_churn"],
    &["firmware_update", "sensor_drift", "false_alarm", "unplanned_downtime", "output_shortfall"],
    &["price_increase", "margin_compression", "discount_escalation", "forecast_revision"],
    &["network_change", "latency_regression", "batch_job_overrun", "reporting_delay", "close_delay"],
];

/// Background predicates that carry no strategic meaning.
pub const BACKGROUND_PREDICATES: [&str; 16] = [
    "status_update",
    "meeting_note",
    "ticket_triage",
    "doc_review",
    "onboarding",
    "budget_note",
    "training_session",
    "tool_upgrade",
    "headcount_note",
    "travel_note",
    "vendor_checkin",
    "release_note",
    "backup_completed",
    "access_request",
    "inventory_count",
    "patch_window",
];

/// Surface synonyms of the causal predicates: several ways of saying the same thing.
const PRED_SURFACE_CAUSAL: [(&str, [&str; 3]); 34] = [
    ("supplier_substitution", ["second source part swap", "alt vendor substitution", "supplier switched"]),
    ("component_defect", ["defective lot found", "component out of spec", "part failure rate up"]),
    ("yield_drop", ["line yield fell", "first pass yield down", "scrap rate climbing"]),
    ("shipment_delay", ["shipment slipped", "delivery pushed out", "freight delayed"]),
    ("customer_escalation", ["customer escalated", "exec escalation opened", "account raised severity"]),
    ("revenue_miss", ["booked below plan", "revenue shortfall", "missed quarter target"]),
    ("cert_rotation", ["certificate rotated", "tls cert rollover", "key rotation performed"]),
    ("auth_failure", ["auth failures spiked", "token validation errors", "login rejections up"]),
    ("api_error_spike", ["5xx rate elevated", "api errors surged", "error budget burned"]),
    ("checkout_drop", ["checkout conversion fell", "cart completion down", "payment success rate dipped"]),
    ("policy_change", ["policy updated", "control tightened", "new approval rule"]),
    ("manual_workaround", ["team used a workaround", "manual bypass in use", "shadow process adopted"]),
    ("data_exposure_risk", ["unencrypted export observed", "data left the boundary", "exposure risk flagged"]),
    ("audit_finding", ["audit finding raised", "control gap logged", "assessor noted issue"]),
    ("regulatory_notice", ["regulator inquiry received", "compliance notice", "supervisory letter"]),
    ("attrition_spike", ["several resignations", "attrition above baseline", "team lost engineers"]),
    ("backlog_growth", ["queue depth rising", "backlog grew", "work in progress piling up"]),
    ("sla_breach", ["sla missed", "response time out of target", "breached service level"]),
    ("contract_penalty", ["penalty clause triggered", "service credits owed", "contractual fine"]),
    ("account_churn", ["customer did not renew", "account churned", "logo lost"]),
    ("firmware_update", ["firmware pushed", "controller flashed", "device image updated"]),
    ("sensor_drift", ["sensor readings drifting", "calibration off", "telemetry bias observed"]),
    ("false_alarm", ["spurious alarm", "nuisance trip", "false positive alert"]),
    ("unplanned_downtime", ["line stopped unexpectedly", "unplanned outage", "asset went down"]),
    ("output_shortfall", ["output below plan", "production short", "throughput miss"]),
    ("price_increase", ["price raised", "list price up", "surcharge applied"]),
    ("margin_compression", ["gross margin down", "margin squeezed", "cost of goods up"]),
    ("discount_escalation", ["deeper discounting", "discount approvals rising", "price concessions"]),
    ("forecast_revision", ["forecast cut", "plan revised down", "guidance changed"]),
    ("network_change", ["network config change", "routing updated", "firewall rule change"]),
    ("latency_regression", ["p99 latency up", "response times regressed", "slow queries"]),
    ("batch_job_overrun", ["nightly job overran", "batch window exceeded", "etl ran long"]),
    ("reporting_delay", ["report late", "dashboard stale", "metrics delayed"]),
    ("close_delay", ["month end close slipped", "books closed late", "financial close delayed"]),
];

pub const FILLER_TEXT: &str = "per the runbook during the window as discussed with the team \
ticket reference attached no further action for now follow up next \
week owner assigned pending review see thread for details minor \
impact expected nothing blocking so far context below";

pub const ENTITY_KINDS: [&str; 6] = ["vendor", "component", "system", "product", "process", "site_asset"];
pub const ENTITY_STEMS: [&str; 28] = [
    "nimbus",
    "veridian",
    "kestrel",
    "orion",
    "cobalt",
    "tessera",
    "halcyon",
    "meridian",
    "pellucid",
    "quarry",
    "basalt",
    "lumen",
    "arbor",
    "cinder",
    "solace",
    "vantage",
    "thistle",
    "umbra",
    "wexford",
    "zephyr",
    "granite",
    "marlin",
    "osprey",
    "peregrine",
    "sable",
    "tundra",
    "vireo",
    "wisteria",
];
pub const ENTITY_SUFFIX: [&str; 15] =
    ["", "-pay", "-core", "-edge", "x", "-ii", "-pro", "-lite", "-eu", "-na", "-apac", "-v2", "-mk3", "-hub", "-net"];

pub const KIND_ROUTINE: i8 = 0;
pub const KIND_FACET: i8 = 1;
pub const KIND_DECOY: i8 = 2;
pub const KIND_DUP: i8 = 3;
pub const KIND_STALE: i8 = 4;
pub const KIND_FALSE: i8 = 5;
pub const KIND_ANOMALY: i8 = 6;
pub const KIND_NOTABLE: i8 = 7;
pub const KIND_NAMES: [&str; 8] = ["routine", "facet", "decoy", "dup", "stale", "false", "anomaly", "notable"];

/// The predicate schema, derived once: `PREDICATES`, `PRED_ID`, `CHAIN_OF_PRED`, `PRED_SURFACE`.
pub struct Schema {
    pub predicates: Vec<&'static str>,
    pub n_causal: usize,
    pub pred_id: HashMap<&'static str, usize>,
    /// predicate id -> [(chain, position)]
    pub chain_of_pred: HashMap<usize, Vec<(usize, usize)>>,
    /// predicate id -> surface forms
    pub surface: Vec<Vec<String>>,
    /// chain -> predicate ids in order
    pub chain_ids: Vec<Vec<usize>>,
    pub filler: Vec<&'static str>,
}

pub static SCHEMA: LazyLock<Schema> = LazyLock::new(|| {
    let mut predicates: Vec<&'static str> = Vec::new();
    for ch in CAUSAL_CHAINS {
        for &p in ch {
            if !predicates.contains(&p) {
                predicates.push(p);
            }
        }
    }
    let n_causal = predicates.len();
    predicates.extend(BACKGROUND_PREDICATES);
    let pred_id: HashMap<&'static str, usize> = predicates.iter().enumerate().map(|(i, &p)| (p, i)).collect();
    let mut chain_of_pred: HashMap<usize, Vec<(usize, usize)>> = HashMap::new();
    for (ci, ch) in CAUSAL_CHAINS.iter().enumerate() {
        for (pos, p) in ch.iter().enumerate() {
            chain_of_pred.entry(pred_id[p]).or_default().push((ci, pos));
        }
    }
    let mut surface: Vec<Vec<String>> = vec![Vec::new(); predicates.len()];
    for (p, forms) in PRED_SURFACE_CAUSAL {
        surface[pred_id[p]] = forms.iter().map(|s| s.to_string()).collect();
    }
    for bp in BACKGROUND_PREDICATES {
        let spaced = bp.replace('_', " ");
        surface[pred_id[bp]] = vec![spaced.clone(), format!("{spaced} logged"), format!("routine {spaced}")];
    }
    let chain_ids = CAUSAL_CHAINS.iter().map(|ch| ch.iter().map(|p| pred_id[p]).collect()).collect();
    Schema {
        predicates,
        n_causal,
        pred_id,
        chain_of_pred,
        surface,
        chain_ids,
        filler: FILLER_TEXT.split_whitespace().collect(),
    }
});

/// `PRED_ID[name]`
pub fn pred_id(name: &str) -> usize {
    SCHEMA.pred_id[name]
}

/// Number of causal (operational) predicates: 34.
pub fn n_causal_pred() -> usize {
    SCHEMA.n_causal
}

// --------------------------------------------------------------------------------------------------------------
// The record table
// --------------------------------------------------------------------------------------------------------------

/// `REC_DTYPE`, column by column.
#[derive(Clone, Debug, Default)]
pub struct Recs {
    pub uid: Vec<i32>,
    pub t: Vec<i16>,
    pub pred: Vec<i16>,
    pub anchor: Vec<i32>,
    pub aux: Vec<i32>,
    pub event: Vec<i32>,
    pub kind: Vec<i8>,
    pub veracity: Vec<i8>,
    pub polarity: Vec<i8>,
    /// pattern id (>= 0), decoy (-2 - pid) or -1
    pub group: Vec<i32>,
    pub facet: Vec<i8>,
    pub superseded: Vec<i8>,
    pub salience: Vec<f32>,
}

impl Recs {
    pub fn len(&self) -> usize {
        self.uid.len()
    }

    pub fn is_empty(&self) -> bool {
        self.uid.is_empty()
    }

    fn zeros(n: usize) -> Recs {
        Recs {
            uid: vec![0; n],
            t: vec![0; n],
            pred: vec![0; n],
            anchor: vec![0; n],
            aux: vec![0; n],
            event: vec![0; n],
            kind: vec![0; n],
            veracity: vec![0; n],
            polarity: vec![0; n],
            group: vec![0; n],
            facet: vec![0; n],
            superseded: vec![0; n],
            salience: vec![0.0; n],
        }
    }

    fn extend(&mut self, o: &Recs) {
        self.uid.extend_from_slice(&o.uid);
        self.t.extend_from_slice(&o.t);
        self.pred.extend_from_slice(&o.pred);
        self.anchor.extend_from_slice(&o.anchor);
        self.aux.extend_from_slice(&o.aux);
        self.event.extend_from_slice(&o.event);
        self.kind.extend_from_slice(&o.kind);
        self.veracity.extend_from_slice(&o.veracity);
        self.polarity.extend_from_slice(&o.polarity);
        self.group.extend_from_slice(&o.group);
        self.facet.extend_from_slice(&o.facet);
        self.superseded.extend_from_slice(&o.superseded);
        self.salience.extend_from_slice(&o.salience);
    }

    fn take(&self, order: &[usize]) -> Recs {
        Recs {
            uid: order.iter().map(|&i| self.uid[i]).collect(),
            t: order.iter().map(|&i| self.t[i]).collect(),
            pred: order.iter().map(|&i| self.pred[i]).collect(),
            anchor: order.iter().map(|&i| self.anchor[i]).collect(),
            aux: order.iter().map(|&i| self.aux[i]).collect(),
            event: order.iter().map(|&i| self.event[i]).collect(),
            kind: order.iter().map(|&i| self.kind[i]).collect(),
            veracity: order.iter().map(|&i| self.veracity[i]).collect(),
            polarity: order.iter().map(|&i| self.polarity[i]).collect(),
            group: order.iter().map(|&i| self.group[i]).collect(),
            facet: order.iter().map(|&i| self.facet[i]).collect(),
            superseded: order.iter().map(|&i| self.superseded[i]).collect(),
            salience: order.iter().map(|&i| self.salience[i]).collect(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Pattern {
    pub pid: usize,
    pub chain: usize,
    pub start_pos: usize,
    pub preds: Vec<usize>,
    pub anchor: usize,
    /// record ids per facet
    pub facet_records: Vec<Vec<usize>>,
    /// user node ids per facet
    pub facet_users: Vec<Vec<usize>>,
    pub facet_times: Vec<i64>,
    pub real: bool,
    /// 0 real, 1..5 decoy kinds
    pub decoy_type: i64,
    pub rare: bool,
    pub n_regions: usize,
    pub n_sites: usize,
    pub contradicted: bool,
    /// variant-family id (cross-region variants), -1 for none
    pub family: i64,
}

/// The corpus generator's configuration (`DEFAULT_CFG`), with the per-experiment overrides of E5.
#[derive(Clone, Debug)]
pub struct CorpusCfg {
    pub recs_per_user_median: f64,
    pub recs_per_user_sigma: f64,
    pub recs_per_user_lo: i64,
    pub recs_per_user_hi: i64,
    pub n_days: i64,
    pub entities_per_1k_users: i64,
    pub min_entities: i64,
    /// fraction of records that echo another user's event
    pub p_dup: f64,
    pub p_stale: f64,
    pub p_false: f64,
    pub p_anomaly: f64,
    pub p_cross_site_entity: f64,
    pub p_notable: f64,
    pub patterns_per_10k_users: f64,
    pub min_patterns: i64,
    /// decoys per real pattern
    pub decoy_ratio: f64,
    /// fraction of real patterns that are rare-signal
    pub rare_fraction: f64,
    pub facet_support_common: (i64, i64),
    pub facet_support_rare: (i64, i64),
    pub pattern_len_choices: [i64; 3],
    pub p_pattern_contradicted: f64,
    pub variant_family_fraction: f64,
    /// D4 decoys are echoed by this many users
    pub dup_inflation_users: (i64, i64),
}

impl Default for CorpusCfg {
    fn default() -> Self {
        CorpusCfg {
            recs_per_user_median: 28.0,
            recs_per_user_sigma: 0.55,
            recs_per_user_lo: 6,
            recs_per_user_hi: 160,
            n_days: 180,
            entities_per_1k_users: 60,
            min_entities: 150,
            p_dup: 0.09,
            p_stale: 0.06,
            p_false: 0.035,
            p_anomaly: 0.02,
            p_cross_site_entity: 0.05,
            p_notable: 0.012,
            patterns_per_10k_users: 20.0,
            min_patterns: 40,
            decoy_ratio: 1.25,
            rare_fraction: 0.40,
            facet_support_common: (3, 9),
            facet_support_rare: (1, 2),
            pattern_len_choices: [3, 4, 5],
            p_pattern_contradicted: 0.25,
            variant_family_fraction: 0.35,
            dup_inflation_users: (12, 40),
        }
    }
}

impl CorpusCfg {
    /// Apply one override from an experiment spec (`{"p_dup": 0.30}`), as `dict.update` would.
    pub fn set(&mut self, key: &str, v: &Value) {
        let f = || v.as_f64().unwrap_or_else(|| panic!("corpus cfg {key}: not a number"));
        match key {
            "p_dup" => self.p_dup = f(),
            "p_stale" => self.p_stale = f(),
            "p_false" => self.p_false = f(),
            "p_anomaly" => self.p_anomaly = f(),
            "p_cross_site_entity" => self.p_cross_site_entity = f(),
            "p_notable" => self.p_notable = f(),
            "decoy_ratio" => self.decoy_ratio = f(),
            "rare_fraction" => self.rare_fraction = f(),
            "patterns_per_10k_users" => self.patterns_per_10k_users = f(),
            "p_pattern_contradicted" => self.p_pattern_contradicted = f(),
            "variant_family_fraction" => self.variant_family_fraction = f(),
            other => panic!("unsupported corpus cfg override {other:?}"),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Corpus {
    pub org: Org,
    pub recs: Recs,
    /// (n_users + 1) offsets into recs, which are sorted by uid
    pub user_slices: Vec<i64>,
    pub patterns: Vec<Pattern>,
    pub entities: Vec<String>,
    pub entity_kind: Vec<i8>,
    pub n_days: i64,
    pub seed: u64,
    pub cfg: CorpusCfg,
    pub notable_records: Vec<usize>,
    /// family id -> pattern ids, in creation order
    pub families: Vec<(i64, Vec<usize>)>,
    pub contradiction_pairs: Vec<(usize, usize)>,
}

fn blake2b_le(data: &[u8], digest_size: usize) -> u64 {
    let mut h = Blake2bVar::new(digest_size).expect("blake2b digest size");
    h.update(data);
    let mut out = [0u8; 8];
    h.finalize_variable(&mut out[..digest_size]).expect("blake2b output");
    u64::from_le_bytes(out)
}

impl Corpus {
    /// Surface realisation.  Synonym and filler choices are keyed on the EVENT id, so an echo repeats the
    /// original wording almost verbatim while two independent observations of the same (predicate, anchor)
    /// differ in filler: duplicate detection is a text-similarity problem every system faces.
    pub fn tokens(&self, rid: usize) -> Vec<String> {
        let r = &self.recs;
        let h = blake2b_le(&(r.event[rid] as i64 as u64).to_le_bytes(), 8);
        let pred = r.pred[rid] as usize;
        let forms = &SCHEMA.surface[pred];
        let surf = &forms[(h % forms.len() as u64) as usize];
        let mut toks: Vec<String> = surf.split_whitespace().map(str::to_string).collect();
        toks.push(self.entities[r.anchor[rid] as usize].clone());
        if r.aux[rid] >= 0 {
            toks.push(self.entities[r.aux[rid] as usize].clone());
        }
        let filler = &SCHEMA.filler;
        let nf = 4 + (h >> 7) % 7;
        for i in 0..nf {
            toks.push(filler[((h >> (3 * i + 11)) % filler.len() as u64) as usize].to_string());
        }
        // paraphrase noise so echoes are near- not exact-duplicates
        let hr = blake2b_le(&(rid as u64).to_le_bytes(), 4);
        if hr % 100 < 35 && toks.len() > 3 {
            let last = toks.len() - 1;
            toks[last] = filler[((hr >> 7) % filler.len() as u64) as usize].to_string();
        }
        if r.polarity[rid] < 0 {
            toks.push("not".to_string());
        }
        toks
    }

    pub fn text(&self, rid: usize) -> String {
        let node = &self.org.nodes[self.recs.uid[rid] as usize];
        format!("[d{:03}] {} :: {}", self.recs.t[rid], node.name, self.tokens(rid).join(" "))
    }

    /// `Corpus.stats()`
    pub fn stats(&self) -> Value {
        let kinds: Vec<i64> = self.recs.kind.iter().map(|&k| k as i64).collect();
        let k = math::bincount(&kinds, 8);
        let real: Vec<&Pattern> = self.patterns.iter().filter(|p| p.real).collect();
        let dec: Vec<&Pattern> = self.patterns.iter().filter(|p| !p.real).collect();
        let mut kmap = Map::new();
        for (i, name) in KIND_NAMES.iter().enumerate() {
            kmap.insert(name.to_string(), json!(k[i]));
        }
        let mut dt = Map::new();
        for t in 1..=5 {
            dt.insert(t.to_string(), json!(dec.iter().filter(|p| p.decoy_type == t).count()));
        }
        let spread = if real.is_empty() {
            json!({})
        } else {
            let nr: Vec<i64> = real.iter().map(|p| p.n_regions as i64).collect();
            json!({
                "mean": crate::py::json::round(math::mean_int(&nr), 2),
                "min": nr.iter().min(),
                "max": nr.iter().max(),
            })
        };
        json!({
            "n_records": self.recs.len(),
            "n_users": self.org.user_ids.len(),
            "recs_per_user_mean": crate::py::json::round(self.recs.len() as f64 / self.org.user_ids.len() as f64, 1),
            "kinds": kmap,
            "n_entities": self.entities.len(),
            "n_real_patterns": real.len(),
            "n_rare_patterns": real.iter().filter(|p| p.rare).count(),
            "n_decoys": dec.len(),
            "decoy_types": dt,
            "n_families": self.families.len(),
            "n_notable": self.notable_records.len(),
            "n_contradictions": self.contradiction_pairs.len(),
            "pattern_region_spread": spread,
        })
    }
}

fn mk_entities(n: usize) -> (Vec<String>, Vec<i8>) {
    let mut names = Vec::new();
    let mut kinds = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    let mut i = 0usize;
    while names.len() < n {
        let stem = ENTITY_STEMS[i % ENTITY_STEMS.len()];
        let suf = ENTITY_SUFFIX[(i / ENTITY_STEMS.len()) % ENTITY_SUFFIX.len()];
        let extra = if i < ENTITY_STEMS.len() * ENTITY_SUFFIX.len() { String::new() } else { i.to_string() };
        let nm = format!("{stem}{suf}{extra}");
        if seen.insert(nm.clone()) {
            names.push(nm);
            kinds.push((i % ENTITY_KINDS.len()) as i8);
        }
        i += 1;
    }
    (names, kinds)
}

/// The first five characters of a name (`e[:5]`).
pub fn prefix5(s: &str) -> &str {
    let end = s.char_indices().nth(5).map(|(i, _)| i).unwrap_or(s.len());
    &s[..end]
}

/// Pick a different entity whose name shares the stem (lexically similar).
fn near_miss(entities: &[String], eid: usize, rng: &mut Generator, pool: &[usize]) -> usize {
    let base = &entities[eid];
    let stem = base.trim_end_matches(|c: char| c.is_ascii_digit()).split('-').next().unwrap_or("");
    let stem5 = prefix5(stem);
    let cands: Vec<usize> = pool.iter().copied().filter(|&j| j != eid && entities[j].starts_with(stem5)).collect();
    if !cands.is_empty() {
        return cands[rng.integers(0, cands.len() as i64) as usize];
    }
    pool[rng.integers(0, pool.len() as i64) as usize]
}

/// A block of records waiting for ids, with the pattern and facet it belongs to.
struct Pending {
    rows: Recs,
    pid: usize,
    facet: usize,
}

/// `build_corpus(org, seed, cfg)`
pub fn build_corpus(org: Org, seed: u64, cfg: Option<CorpusCfg>) -> Corpus {
    let c = cfg.unwrap_or_default();
    let schema = &*SCHEMA;
    let mut rng = Generator::new(1_000_003 + seed);
    let users: Vec<usize> = org.user_ids.clone();
    let nu = users.len();
    let uidx: HashMap<usize, usize> = users.iter().enumerate().map(|(i, &u)| (u, i)).collect();
    let n_days = c.n_days;

    let n_ent = (c.min_entities as usize).max((c.entities_per_1k_users as f64 * nu as f64 / 1000.0) as usize);
    let (entities, ent_kind) = mk_entities(n_ent);

    // ---- per-user record budget ----
    let mu = c.recs_per_user_median.ln();
    let cnt: Vec<i64> = rng
        .lognormal_n(mu, c.recs_per_user_sigma, nu)
        .into_iter()
        .map(|v| math::rint(v).clamp(c.recs_per_user_lo as f64, c.recs_per_user_hi as f64) as i64)
        .collect();
    let total_bg: usize = cnt.iter().sum::<i64>() as usize;

    // Entity affinity: a small GLOBAL pool visible everywhere, the rest PARTITIONED into site-private blocks, so
    // "the same entity in three regions" is an anomaly rather than the norm.
    let n_global = 6.max(n_ent / 12);
    let global_pool: Vec<usize> = (0..n_global).collect();
    let site_ids: Vec<usize> = org.levels[SITE].clone();
    let mut private: Vec<usize> = (n_global..n_ent).collect();
    rng.shuffle(&mut private);
    let per = 6.max(private.len() / 1.max(site_ids.len()));
    let mut site_pool: HashMap<usize, Vec<usize>> = HashMap::new();
    for (i, &s) in site_ids.iter().enumerate() {
        let lo = (i * per).min(private.len());
        let hi = ((i + 1) * per).min(private.len());
        let mut blk: Vec<usize> = private[lo..hi].to_vec();
        if blk.len() < 6 {
            let start = private.len().saturating_sub(per);
            blk = private[start..].to_vec();
        }
        let picks = rng.choice_without_replace(n_global, 4.min(n_global));
        blk.extend(picks.iter().map(|&k| global_pool[k as usize]));
        site_pool.insert(s, blk);
    }
    let private_pool = private;
    let user_site: Vec<usize> = users.iter().map(|&u| org.ancestor_at(u, SITE).expect("user without site")).collect();
    let user_region: Vec<usize> =
        users.iter().map(|&u| org.ancestor_at(u, REGION).expect("user without region")).collect();

    // ---------------- background records ----------------
    let mut recs = Recs::zeros(total_bg);
    let bg_ids: Vec<i64> = BACKGROUND_PREDICATES.iter().map(|p| schema.pred_id[p] as i64).collect();
    {
        let mut k = 0;
        for (ui, &u) in users.iter().enumerate() {
            for _ in 0..cnt[ui] {
                recs.uid[k] = u as i32;
                k += 1;
            }
        }
    }
    for (k, t) in rng.integers_n(0, n_days, total_bg).into_iter().enumerate() {
        recs.t[k] = t as i16;
    }
    // most background records use background predicates; a slice uses causal predicates too (so causal
    // predicates are NOT a give-away signal)
    let use_causal: Vec<bool> = rng.random_n(total_bg).into_iter().map(|x| x < 0.22).collect();
    let causal_pick = rng.integers_n(0, schema.n_causal as i64, total_bg);
    let bg_pick = rng.integers_n(0, bg_ids.len() as i64, total_bg);
    for k in 0..total_bg {
        recs.pred[k] = if use_causal[k] { causal_pick[k] } else { bg_ids[bg_pick[k] as usize] } as i16;
    }
    // anchors: 70% site-local pool, 20% global pool, 10% anywhere
    let mut anchors = vec![0i32; total_bg];
    let which = rng.random_n(total_bg);
    let site_of_rec: Vec<usize> = {
        let mut v = Vec::with_capacity(total_bg);
        for (ui, &s) in user_site.iter().enumerate() {
            for _ in 0..cnt[ui] {
                v.push(s);
            }
        }
        v
    };
    let mut recs_of_site: HashMap<usize, Vec<usize>> = HashMap::new();
    for (k, &s) in site_of_rec.iter().enumerate() {
        recs_of_site.entry(s).or_default().push(k);
    }
    for s in &site_ids {
        let Some(ks) = recs_of_site.get(s) else { continue };
        let pool = &site_pool[s];
        let draws = rng.integers_n(0, pool.len() as i64, ks.len());
        for (&k, &d) in ks.iter().zip(&draws) {
            anchors[k] = pool[d as usize] as i32;
        }
    }
    let gm: Vec<usize> = (0..total_bg).filter(|&k| which[k] >= 0.70 && which[k] < 0.86).collect();
    let draws = rng.integers_n(0, global_pool.len() as i64, gm.len());
    for (&k, &d) in gm.iter().zip(&draws) {
        anchors[k] = global_pool[d as usize] as i32;
    }
    // Benign cross-site traffic: shared suppliers, transferred staff, joint projects, M&A overlap.  Without it
    // the only entities visible in more than one region would be the planted patterns.
    let xm: Vec<usize> =
        (0..total_bg).filter(|&k| which[k] >= 0.86 && which[k] < 0.86 + c.p_cross_site_entity).collect();
    if !xm.is_empty() {
        let pick_site = rng.integers_n(0, site_ids.len() as i64, xm.len());
        let vals: Vec<i32> = pick_site
            .iter()
            .map(|&ps| {
                let pool = &site_pool[&site_ids[ps as usize]];
                pool[rng.integers(0, pool.len() as i64) as usize] as i32
            })
            .collect();
        for (&k, &v) in xm.iter().zip(&vals) {
            anchors[k] = v;
        }
    }
    // `which >= 1.01` never holds (the entity namespace is partitioned by site); numpy draws nothing for it
    recs.anchor = anchors;
    let aux_coin = rng.random_n(total_bg);
    let aux_pick = rng.integers_n(0, n_ent as i64, total_bg);
    for k in 0..total_bg {
        recs.aux[k] = if aux_coin[k] < 0.5 { aux_pick[k] as i32 } else { -1 };
    }
    for k in 0..total_bg {
        recs.event[k] = k as i32;
        recs.kind[k] = KIND_ROUTINE;
        recs.veracity[k] = 1;
        recs.polarity[k] = 1;
        recs.group[k] = -1;
        recs.facet[k] = -1;
    }
    for (k, x) in rng.random_n(total_bg).into_iter().enumerate() {
        recs.salience[k] = x as f32 * 0.25f32;
    }

    // ---- mark duplicates / stale / false / anomaly / notable in place ----
    let r = rng.random_n(total_bg);
    let base2 = c.p_dup + c.p_stale;
    let base3 = base2 + c.p_false;
    let base4 = base3 + c.p_anomaly;
    let dup_idx: Vec<usize> = (0..total_bg).filter(|&k| r[k] < c.p_dup).collect();
    let st_idx: Vec<usize> = (0..total_bg).filter(|&k| r[k] >= c.p_dup && r[k] < c.p_dup + c.p_stale).collect();
    let fl_idx: Vec<usize> = (0..total_bg).filter(|&k| r[k] >= base2 && r[k] < base2 + c.p_false).collect();
    let an_idx: Vec<usize> = (0..total_bg).filter(|&k| r[k] >= base3 && r[k] < base3 + c.p_anomaly).collect();
    let nt_idx: Vec<usize> = (0..total_bg).filter(|&k| r[k] >= base4 && r[k] < base4 + c.p_notable).collect();

    // Echoes circulate locally: a duplicate copies a record from the SAME SITE (global copying would smear every
    // entity across every region and destroy the locality the index depends on).
    let site_rec: Vec<i64> = site_of_rec.iter().map(|&s| s as i64).collect();
    let sorder = math::argsort_stable(&site_rec);
    let mut site_slot: HashMap<i64, (usize, usize)> = HashMap::new();
    {
        let mut k = 0;
        while k < sorder.len() {
            let v = site_rec[sorder[k]];
            let start = k;
            while k < sorder.len() && site_rec[sorder[k]] == v {
                k += 1;
            }
            site_slot.insert(v, (start, k - start));
        }
    }
    let local_src = |idx: &[usize], rng: &mut Generator| -> Vec<usize> {
        idx.iter()
            .map(|&r_| {
                let (a, b) = site_slot[&site_rec[r_]];
                sorder[a + rng.integers(0, b as i64) as usize]
            })
            .collect()
    };

    let src = local_src(&dup_idx, &mut rng);
    let src_pred: Vec<i16> = src.iter().map(|&s| recs.pred[s]).collect();
    let src_anchor: Vec<i32> = src.iter().map(|&s| recs.anchor[s]).collect();
    let src_event: Vec<i32> = src.iter().map(|&s| recs.event[s]).collect();
    let src_t: Vec<i64> = src.iter().map(|&s| recs.t[s] as i64).collect();
    let shift = rng.integers_n(0, 6, dup_idx.len());
    for (j, &k) in dup_idx.iter().enumerate() {
        recs.kind[k] = KIND_DUP;
        recs.pred[k] = src_pred[j];
        recs.anchor[k] = src_anchor[j];
        // SAME event id -> not independent
        recs.event[k] = src_event[j];
        recs.t[k] = (src_t[j] + shift[j]).clamp(0, n_days - 1) as i16;
    }

    let stale_t = rng.integers_n(0, 1.max(n_days / 3), st_idx.len());
    for (j, &k) in st_idx.iter().enumerate() {
        recs.kind[k] = KIND_STALE;
        recs.superseded[k] = 1;
        recs.t[k] = stale_t[j] as i16;
    }
    for &k in &fl_idx {
        recs.kind[k] = KIND_FALSE;
        recs.veracity[k] = 0;
        recs.polarity[k] = -1;
    }
    let an_sal = rng.random_n(an_idx.len());
    for (j, &k) in an_idx.iter().enumerate() {
        recs.kind[k] = KIND_ANOMALY;
        recs.salience[k] = 0.45f32 + an_sal[j] as f32 * 0.2f32;
    }
    let nt_sal = rng.random_n(nt_idx.len());
    for (j, &k) in nt_idx.iter().enumerate() {
        recs.kind[k] = KIND_NOTABLE;
        recs.salience[k] = 0.7f32 + nt_sal[j] as f32 * 0.3f32;
    }

    // contradiction pairs: for each false record, find/emit a true counterpart
    let mut contradiction_pairs: Vec<(usize, usize)> = Vec::new();
    if !fl_idx.is_empty() {
        let partner = local_src(&fl_idx, &mut rng);
        for (&f, &p) in fl_idx.iter().zip(&partner) {
            if recs.uid[f] == recs.uid[p] {
                continue;
            }
            recs.pred[f] = recs.pred[p];
            recs.anchor[f] = recs.anchor[p];
            // distinct event: a genuine conflict
            recs.event[f] = (total_bg + f) as i32;
            contradiction_pairs.push((p, f));
        }
    }

    // ---------------- strategic patterns ----------------
    let n_pat = (c.min_patterns).max(crate::models::py_round(c.patterns_per_10k_users * nu as f64 / 10_000.0));
    let n_decoy = crate::models::py_round(n_pat as f64 * c.decoy_ratio);
    let mut patterns: Vec<Pattern> = Vec::new();
    let mut next_event: i64 = total_bg as i64 * 2 + 10;
    let region_ids: Vec<usize> = org.levels[REGION].clone();
    let users_by_region: HashMap<usize, Vec<usize>> = region_ids
        .iter()
        .map(|&rid| (rid, (0..nu).filter(|&i| user_region[i] == rid).collect::<Vec<usize>>()))
        .collect();

    let pick_users = |k: i64, prefer: &[usize], rng: &mut Generator| -> Vec<usize> {
        let mut out = Vec::with_capacity(k as usize);
        for i in 0..k as usize {
            let reg = prefer[i % prefer.len()];
            let pool = &users_by_region[&reg];
            if pool.is_empty() {
                out.push(rng.integers(0, nu as i64) as usize);
            } else {
                out.push(pool[rng.integers(0, pool.len() as i64) as usize]);
            }
        }
        out
    };

    let mut pending: Vec<Pending> = Vec::new();

    // Anchors are drawn WITHOUT replacement across real patterns and decoys: sharing an entity would make one
    // reported hypothesis simultaneously a discovery and a decoy acceptance, a scoring ambiguity.
    let mut used_anchors: HashSet<usize> = HashSet::new();
    let mut take_anchor = |rng: &mut Generator| -> usize {
        for _ in 0..500 {
            let a = if rng.random() < 0.8 {
                private_pool[rng.integers(0, private_pool.len() as i64) as usize]
            } else {
                rng.integers(0, n_global as i64) as usize
            };
            if used_anchors.insert(a) {
                return a;
            }
        }
        let a = rng.integers(0, n_ent as i64) as usize;
        used_anchors.insert(a);
        a
    };

    let rare_flags: Vec<bool> = rng.random_n(n_pat as usize).into_iter().map(|x| x < c.rare_fraction).collect();
    let fam_flags: Vec<bool> =
        rng.random_n(n_pat as usize).into_iter().map(|x| x < c.variant_family_fraction).collect();
    let mut families: Vec<(i64, Vec<usize>)> = Vec::new();
    let mut fam_next: i64 = 0;
    let mut fam_open: Vec<(i64, usize, usize)> = Vec::new();
    let len_choices = c.pattern_len_choices;

    let mk_rows = |k: usize| Recs::zeros(k);

    for pi in 0..n_pat as usize {
        let rare = rare_flags[pi];
        let mut ci = rng.integers(0, CAUSAL_CHAINS.len() as i64) as usize;
        let mut chain = CAUSAL_CHAINS[ci];
        let mut l = len_choices[rng.choice_index(len_choices.len())] as usize;
        l = l.min(chain.len());
        let mut sp = rng.integers(0, (chain.len() - l + 1) as i64) as usize;
        let mut fam: i64 = -1;
        if !fam_open.is_empty() && rng.random() < 0.55 {
            let pick = fam_open[rng.integers(0, fam_open.len() as i64) as usize];
            fam = pick.0;
            ci = pick.1;
            sp = pick.2;
            chain = CAUSAL_CHAINS[ci];
            l = l.min(chain.len() - sp);
        } else if fam_flags[pi] {
            fam = fam_next;
            fam_next += 1;
            fam_open.push((fam, ci, sp));
        }
        let preds: Vec<usize> = (0..l).map(|j| schema.pred_id[chain[sp + j]]).collect();
        let anchor = take_anchor(&mut rng);
        // scatter facets across regions: at least 2 distinct regions
        let k_reg = region_ids.len().min(2 + rng.integers(0, 3) as usize);
        let regs: Vec<usize> =
            rng.choice_without_replace(region_ids.len(), k_reg).into_iter().map(|i| region_ids[i as usize]).collect();
        let (lo, hi) = if rare { c.facet_support_rare } else { c.facet_support_common };
        let t0 = rng.integers(0, 1.max(n_days - 20 * l as i64));
        let pid = patterns.len();
        let mut pat = Pattern {
            pid,
            chain: ci,
            start_pos: sp,
            preds: preds.clone(),
            anchor,
            facet_records: Vec::new(),
            facet_users: Vec::new(),
            facet_times: Vec::new(),
            real: true,
            decoy_type: 0,
            rare,
            n_regions: 0,
            n_sites: 0,
            contradicted: false,
            family: fam,
        };
        for (j, &pr) in preds.iter().enumerate() {
            let k = rng.integers(lo, hi + 1);
            let us = pick_users(k, &[regs[j % regs.len()]], &mut rng);
            let t = (n_days - 1).min(t0 + j as i64 * rng.integers(4, 18));
            let mut rows = mk_rows(k as usize);
            let sal = rng.random_n(k as usize);
            for (q, &ui) in us.iter().enumerate() {
                rows.uid[q] = users[ui] as i32;
                rows.t[q] = t as i16;
                rows.pred[q] = pr as i16;
                rows.anchor[q] = anchor as i32;
                rows.aux[q] = -1;
                rows.event[q] = (next_event + q as i64) as i32;
                rows.kind[q] = KIND_FACET;
                rows.veracity[q] = 1;
                rows.polarity[q] = 1;
                rows.group[q] = pid as i32;
                rows.facet[q] = j as i8;
                rows.salience[q] = 0.30f32 + 0.1f32 * sal[q] as f32;
            }
            next_event += k;
            pending.push(Pending { rows, pid, facet: j });
            pat.facet_users.push(us.iter().map(|&i| users[i]).collect());
            pat.facet_times.push(t);
        }
        if rng.random() < c.p_pattern_contradicted {
            pat.contradicted = true;
            let j = rng.integers(0, l as i64) as usize;
            let us = pick_users(1, &[regs[j % regs.len()]], &mut rng);
            let mut rows = mk_rows(1);
            rows.uid[0] = users[us[0]] as i32;
            rows.t[0] = (n_days - 1).min(pat.facet_times[j] + 3) as i16;
            rows.pred[0] = preds[j] as i16;
            rows.anchor[0] = anchor as i32;
            rows.aux[0] = -1;
            rows.event[0] = next_event as i32;
            next_event += 1;
            rows.kind[0] = KIND_FACET;
            rows.veracity[0] = 0;
            rows.polarity[0] = -1;
            rows.group[0] = pid as i32;
            rows.facet[0] = j as i8;
            rows.salience[0] = 0.35f32;
            pending.push(Pending { rows, pid, facet: j });
        }
        if fam >= 0 {
            match families.iter_mut().find(|f| f.0 == fam) {
                Some(f) => f.1.push(pid),
                None => families.push((fam, vec![pid])),
            }
        }
        patterns.push(pat);
    }

    // ---------------- decoys ----------------
    let all_pool: Vec<usize> = (0..n_ent).collect();
    for di in 0..n_decoy as usize {
        let dtype = 1 + (di % 5) as i64;
        let mut l = len_choices[rng.choice_index(len_choices.len())] as usize;
        let ci = rng.integers(0, CAUSAL_CHAINS.len() as i64) as usize;
        let chain = CAUSAL_CHAINS[ci];
        l = l.min(chain.len());
        let sp = rng.integers(0, (chain.len() - l + 1) as i64) as usize;
        let anchor = take_anchor(&mut rng);
        let k_reg = region_ids.len().min(2 + rng.integers(0, 3) as usize);
        let regs: Vec<usize> =
            rng.choice_without_replace(region_ids.len(), k_reg).into_iter().map(|i| region_ids[i as usize]).collect();
        let t0 = rng.integers(0, 1.max(n_days - 20 * l as i64));
        let pid = patterns.len();
        let (preds, times, anchors): (Vec<usize>, Vec<i64>, Vec<usize>) = match dtype {
            1 => {
                // entity coincidence: predicates from mixed chains
                let mut preds = Vec::new();
                for _ in 0..l {
                    let cj = rng.integers(0, CAUSAL_CHAINS.len() as i64) as usize;
                    let pos = rng.integers(0, CAUSAL_CHAINS[cj].len() as i64) as usize;
                    preds.push(schema.pred_id[CAUSAL_CHAINS[cj][pos]]);
                }
                let times = (0..l).map(|_| rng.integers(0, n_days)).collect();
                (preds, times, vec![anchor; l])
            }
            2 => {
                // temporal scramble
                let preds = (0..l).map(|j| schema.pred_id[chain[sp + j]]).collect();
                let mut times: Vec<i64> =
                    (0..l).map(|j| (n_days - 1).min(t0 + j as i64 * rng.integers(4, 18))).collect();
                times.sort_by(|a, b| b.cmp(a));
                (preds, times, vec![anchor; l])
            }
            3 => {
                // near-miss entities
                let preds = (0..l).map(|j| schema.pred_id[chain[sp + j]]).collect();
                let times = (0..l).map(|j| (n_days - 1).min(t0 + j as i64 * rng.integers(4, 18))).collect();
                let mut anchors = vec![anchor];
                for _ in 0..l - 1 {
                    anchors.push(near_miss(&entities, anchor, &mut rng, &all_pool));
                }
                (preds, times, anchors)
            }
            4 => {
                // duplicate inflation (one event, many echoes)
                let preds = (0..l).map(|j| schema.pred_id[chain[sp + j]]).collect();
                let times = (0..l).map(|j| (n_days - 1).min(t0 + j as i64 * rng.integers(4, 18))).collect();
                (preds, times, vec![anchor; l])
            }
            _ => {
                // stale chain: every facet later retracted
                let preds = (0..l).map(|j| schema.pred_id[chain[sp + j]]).collect();
                let half = n_days / 2;
                let times = (0..l).map(|j| half.min(t0 % 1.max(half) + j as i64 * rng.integers(3, 9))).collect();
                (preds, times, vec![anchor; l])
            }
        };
        let mut pat = Pattern {
            pid,
            chain: ci,
            start_pos: sp,
            preds: preds.clone(),
            anchor,
            facet_records: Vec::new(),
            facet_users: Vec::new(),
            facet_times: Vec::new(),
            real: false,
            decoy_type: dtype,
            rare: false,
            n_regions: 0,
            n_sites: 0,
            contradicted: false,
            family: -1,
        };
        for (j, &pr) in preds.iter().enumerate() {
            let k = if dtype == 4 {
                let (lo4, hi4) = c.dup_inflation_users;
                rng.integers(lo4, hi4 + 1)
            } else {
                let (lo, hi) = c.facet_support_common;
                rng.integers(lo, hi + 1)
            };
            let us = pick_users(k, &regs, &mut rng);
            let mut rows = mk_rows(k as usize);
            for (q, &ui) in us.iter().enumerate() {
                rows.uid[q] = users[ui] as i32;
                rows.t[q] = times[j] as i16;
                rows.pred[q] = pr as i16;
                rows.anchor[q] = anchors[j] as i32;
                rows.aux[q] = -1;
                if dtype == 4 {
                    // ONE event id for all echoes
                    rows.event[q] = next_event as i32;
                    rows.kind[q] = KIND_DUP;
                } else {
                    rows.event[q] = (next_event + q as i64) as i32;
                    rows.kind[q] = KIND_DECOY;
                }
                rows.veracity[q] = 1;
                rows.polarity[q] = 1;
                rows.group[q] = -2 - pid as i32;
                rows.facet[q] = j as i8;
            }
            next_event += if dtype == 4 { 1 } else { k };
            let sal = rng.random_n(k as usize);
            for (slot, x) in rows.salience.iter_mut().zip(&sal) {
                *slot = 0.30f32 + 0.1f32 * *x as f32;
            }
            pending.push(Pending { rows, pid, facet: j });
            pat.facet_users.push(us.iter().map(|&i| users[i]).collect());
            pat.facet_times.push(times[j]);
            if dtype == 5 {
                // a later, independent retraction of the same claim
                let kr = 1.max(k / 2);
                let ur = pick_users(kr, &regs, &mut rng);
                let tr = (n_days - 1).min(times[j] + rng.integers(30, 70));
                let mut rr = mk_rows(kr as usize);
                for (q, &ui) in ur.iter().enumerate() {
                    rr.uid[q] = users[ui] as i32;
                    rr.t[q] = tr as i16;
                    rr.pred[q] = pr as i16;
                    rr.anchor[q] = anchors[j] as i32;
                    rr.aux[q] = -1;
                    rr.event[q] = (next_event + q as i64) as i32;
                    rr.kind[q] = KIND_DECOY;
                    rr.veracity[q] = 1;
                    rr.polarity[q] = -1;
                    rr.group[q] = -2 - pid as i32;
                    rr.facet[q] = j as i8;
                    rr.salience[q] = 0.3f32;
                }
                next_event += kr;
                pending.push(Pending { rows: rr, pid, facet: j });
            }
        }
        patterns.push(pat);
    }

    // ---------------- assemble ----------------
    let base = recs.len();
    let mut all_recs = recs;
    let mut cursor = base;
    for p in &pending {
        all_recs.extend(&p.rows);
        let pat = &mut patterns[p.pid];
        while pat.facet_records.len() <= p.facet {
            pat.facet_records.push(Vec::new());
        }
        pat.facet_records[p.facet].extend(cursor..cursor + p.rows.len());
        cursor += p.rows.len();
    }

    // sort by uid so per-user slices are contiguous
    let uid64: Vec<i64> = all_recs.uid.iter().map(|&u| u as i64).collect();
    let order = math::argsort_stable(&uid64);
    let mut inv = vec![0usize; order.len()];
    for (newpos, &old) in order.iter().enumerate() {
        inv[old] = newpos;
    }
    let all_recs = all_recs.take(&order);
    for pat in patterns.iter_mut() {
        for fr in pat.facet_records.iter_mut() {
            for r in fr.iter_mut() {
                *r = inv[*r];
            }
        }
    }
    let contradiction_pairs: Vec<(usize, usize)> = contradiction_pairs.iter().map(|&(a, b)| (inv[a], inv[b])).collect();
    let notable: Vec<usize> = nt_idx.iter().map(|&i| inv[i]).collect();

    let mut counts = vec![0i64; nu];
    for &u in &all_recs.uid {
        counts[uidx[&(u as usize)]] += 1;
    }
    let mut slices = vec![0i64; nu + 1];
    for i in 0..nu {
        slices[i + 1] = slices[i] + counts[i];
    }

    // spread stats for patterns
    for pat in patterns.iter_mut() {
        let mut regs: HashSet<usize> = HashSet::new();
        let mut sites: HashSet<usize> = HashSet::new();
        for fr in &pat.facet_records {
            for &rid in fr {
                let u = all_recs.uid[rid] as usize;
                if let Some(x) = org.ancestor_at(u, REGION) {
                    regs.insert(x);
                }
                if let Some(x) = org.ancestor_at(u, SITE) {
                    sites.insert(x);
                }
            }
        }
        pat.n_regions = regs.len();
        pat.n_sites = sites.len();
    }

    Corpus {
        org,
        recs: all_recs,
        user_slices: slices,
        patterns,
        entities,
        entity_kind: ent_kind,
        n_days,
        seed,
        cfg: c,
        notable_records: notable,
        families,
        contradiction_pairs,
    }
}

// --------------------------------------------------------------------------------------------------------------
// Ground truth / gold answers
// --------------------------------------------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct Gold {
    /// real pattern ids a perfect system finds
    pub discoverable: Vec<usize>,
    pub rare: Vec<usize>,
    pub contradicted: Vec<usize>,
    /// variant families with >= 2 discoverable members, in family order
    pub families: Vec<(i64, Vec<usize>)>,
    pub decoys: Vec<usize>,
    /// pattern -> true supporting record ids
    pub evidence: HashMap<usize, Vec<usize>>,
    /// pattern -> independent sources (distinct events)
    pub independent_sources: HashMap<usize, usize>,
    pub notable_records: Vec<usize>,
    /// pattern with the most independent sources (-1 if none)
    pub top_supported: i64,
}

/// `make_gold(corpus, min_facets=3)`
pub fn make_gold(corpus: &Corpus, min_facets: usize) -> Gold {
    let mut disc = Vec::new();
    let mut rare = Vec::new();
    let mut contra = Vec::new();
    let mut ev: HashMap<usize, Vec<usize>> = HashMap::new();
    let mut ind: Vec<(usize, usize)> = Vec::new();
    for p in &corpus.patterns {
        if !p.real {
            continue;
        }
        let present = p.facet_records.iter().filter(|fr| !fr.is_empty()).count();
        if present < min_facets.min(p.preds.len()) {
            continue;
        }
        disc.push(p.pid);
        let rows: Vec<usize> = p.facet_records.iter().flatten().copied().collect();
        let n_events = rows.iter().map(|&r| corpus.recs.event[r]).collect::<HashSet<i32>>().len();
        ev.insert(p.pid, rows);
        ind.push((p.pid, n_events));
        if p.rare {
            rare.push(p.pid);
        }
        if p.contradicted {
            contra.push(p.pid);
        }
    }
    let disc_set: HashSet<usize> = disc.iter().copied().collect();
    let fams: Vec<(i64, Vec<usize>)> = corpus
        .families
        .iter()
        .map(|(f, ps)| (*f, ps.iter().copied().filter(|p| disc_set.contains(p)).collect::<Vec<usize>>()))
        .filter(|(_, ps)| ps.len() >= 2)
        .collect();
    let decoys = corpus.patterns.iter().filter(|p| !p.real).map(|p| p.pid).collect();
    // max(ind, key=ind.get): the first pattern with the largest count, in insertion order
    let mut top: i64 = -1;
    let mut best = 0usize;
    for &(pid, n) in &ind {
        if top < 0 || n > best {
            top = pid as i64;
            best = n;
        }
    }
    Gold {
        discoverable: disc,
        rare,
        contradicted: contra,
        families: fams,
        decoys,
        evidence: ev,
        independent_sources: ind.into_iter().collect(),
        notable_records: corpus.notable_records.clone(),
        top_supported: top,
    }
}
