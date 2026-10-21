//! World construction, the architecture registry and the calibrated knobs (`runner.py`).

use crate::corpus::{Corpus, CorpusCfg, Gold, build_corpus, make_gold};
use crate::evalm::{self, Metrics};
use crate::models::{Tier, allocation_default};
use crate::np::Generator;
use crate::ops::{Ranker, near_miss_map, stem_rep_map};
use crate::org::{ENT, Fanin, USER, build_org};
use crate::py::json as pyjson;
use crate::systems::{self, HierConfig, HierRunner, RunResult, UserLayer};
use serde_json::{Map, Value, json};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

/// The repository root: `MYCELIC_ROOT`, else the checkout this crate was built from.
pub fn repo_root() -> PathBuf {
    match std::env::var_os("MYCELIC_ROOT") {
        Some(p) => PathBuf::from(p),
        None => PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..").join(".."),
    }
}

/// `research/mycelic/artifacts` (`ART`); `MYCELIC_ART` overrides it, which is how a rerun writes elsewhere.
pub fn art() -> PathBuf {
    match std::env::var_os("MYCELIC_ART") {
        Some(p) => PathBuf::from(p),
        None => repo_root().join("research").join("mycelic").join("artifacts"),
    }
}

pub struct World {
    pub corpus: Corpus,
    pub gold: Gold,
    pub near_miss: Vec<i32>,
    pub stem: Vec<i32>,
    pub seed: u64,
    pub scale: usize,
    ul_cache: Mutex<HashMap<String, Arc<UserLayer>>>,
}

impl World {
    /// The user layer for a tier, built once per world (`World.user_layer`).
    pub fn user_layer(&self, tier: &Tier, seed: u64) -> Arc<UserLayer> {
        let key = format!("{}:{}", tier.name, seed);
        if let Some(ul) = self.ul_cache.lock().unwrap().get(&key) {
            return ul.clone();
        }
        let mut rng = Generator::new(50_000 + seed * 13 + (tier.q * 1000.0) as u64);
        let ul = Arc::new(systems::user_extract(&self.corpus, tier, &mut rng, &self.near_miss));
        self.ul_cache.lock().unwrap().insert(key, ul.clone());
        ul
    }

    pub fn clear_cache(&self) {
        self.ul_cache.lock().unwrap().clear();
    }
}

/// Generator options of a world (the `corpus_cfg` and `org_kwargs` of `build_world`).
#[derive(Clone, Debug, Default)]
pub struct WorldOpts {
    pub corpus: Option<CorpusCfg>,
    pub fanin: Option<[Fanin; 6]>,
    pub n_regions: Option<usize>,
}

pub fn build_world(scale: usize, seed: u64, opts: &WorldOpts) -> World {
    let org = build_org(scale, seed, opts.fanin, opts.n_regions.unwrap_or(7));
    let corpus = build_corpus(org, seed, opts.corpus.clone());
    let gold = make_gold(&corpus, 3);
    let near_miss = near_miss_map(&corpus, &mut Generator::new(99 + seed));
    let stem = stem_rep_map(&corpus);
    World { corpus, gold, near_miss, stem, seed, scale, ul_cache: Mutex::new(HashMap::new()) }
}

pub fn build_world_default(scale: usize, seed: u64) -> World {
    build_world(scale, seed, &WorldOpts::default())
}

pub const BASE_BUDGETS: [usize; 5] = [6, 18, 42, 90, 130];

/// Calibrated knobs, fitted on held-out seeds: `calibration.json` over the built-in defaults, minus the grids
/// (`load_calibration`).  Every entry point uses the same operating points.
#[derive(Clone, Debug)]
pub struct Cal {
    pub d: Map<String, Value>,
    /// `FORCE_RANKER`: None -> the per-architecture choice; Some(on) -> forced
    pub force_ranker: Option<bool>,
}

impl Cal {
    pub fn load() -> Cal {
        Cal::from_path(&art().join("calibration.json"))
    }

    pub fn from_path(p: &std::path::Path) -> Cal {
        let mut d: Map<String, Value> = json!({
            "triage_prior_weight": 0.0, "question_frac": 0.25, "mr_budget": 900,
            "flat_budget": 1_000_000, "ct_kernel_ko_cap": 0,
            "w_dispersion": 0.0, "w_synchrony": 0.0, "w_attribution": 0.0
        })
        .as_object()
        .unwrap()
        .clone();
        if let Ok(text) = std::fs::read_to_string(p) {
            let file: Map<String, Value> = serde_json::from_str(&text).expect("calibration.json");
            for (k, v) in file {
                if !k.ends_with("grid") {
                    d.insert(k, v);
                }
            }
        }
        Cal { d, force_ranker: None }
    }

    pub fn f64(&self, k: &str) -> f64 {
        self.d.get(k).and_then(Value::as_f64).unwrap_or(0.0)
    }

    pub fn int(&self, k: &str) -> i64 {
        self.d.get(k).and_then(|v| v.as_i64().or_else(|| v.as_f64().map(|x| x as i64))).unwrap_or(0)
    }

    /// `ranker_for(arch)`: the fitted ranker if this architecture adopted it on the calibration seeds.
    pub fn ranker_for(&self, arch: &str) -> Option<Ranker> {
        let r = self.d.get("ranker")?;
        if r.is_null() {
            return None;
        }
        if let Some(force) = self.force_ranker {
            return force.then(|| Ranker::from_json(r));
        }
        match self.d.get("ranker_archs").and_then(Value::as_array) {
            None => Some(Ranker::from_json(r)),
            Some(sel) => sel.iter().any(|x| x.as_str() == Some(arch)).then(|| Ranker::from_json(r)),
        }
    }

    /// `hier_cfg(**kw)`: the calibrated hierarchy, then the overrides.
    pub fn hier_cfg(&self, over: &Map<String, Value>) -> HierConfig {
        let mut c = HierConfig { budgets: BASE_BUDGETS, ..Default::default() };
        c.triage_prior_weight = self.f64("triage_prior_weight");
        c.question_frac = self.f64("question_frac");
        c.w_dispersion = self.f64("w_dispersion");
        c.w_synchrony = self.f64("w_synchrony");
        c.w_attribution = self.f64("w_attribution");
        // vNext knobs, frozen in calibration.json once their paired experiments were read
        for k in
            ["batched_descent", "local_reextract", "strict_targeting", "sketch_weak_bits", "merge_descent_evidence"]
        {
            if let Some(v) = self.d.get(k) {
                c.set(k, &Value::Bool(pyjson::truthy(v)));
            }
        }
        for k in ["link_time", "triage_target_chains"] {
            if let Some(v) = self.d.get(k) {
                let s = match v {
                    Value::String(s) => s.clone(),
                    other => other.to_string(),
                };
                c.set(k, &Value::String(s));
            }
        }
        for (k, v) in over {
            c.set(k, v);
        }
        c
    }
}

/// The hierarchy as benchmarked before the vNext work, run inside the new suite for the old-vs-new comparison.
pub fn v1_hier_cfg() -> Value {
    json!({"downward_retrieval": true, "questions": true, "cross_links": true,
           "question_frac": 0.25, "batched_descent": false,
           "local_reextract": false, "link_time": "min",
           "strict_targeting": false, "triage_target_chains": "none",
           "sketch_weak_bits": false, "merge_descent_evidence": false})
}

/// `ARCHS`, in registry order: (name, kind, hierarchy overrides).
pub fn archs() -> Vec<(&'static str, &'static str, Value)> {
    let h = |v: Value| v;
    vec![
        ("A_flat_rag", "flat_rag", json!({})),
        ("B_long_context", "long_context", json!({})),
        ("B2_map_reduce", "map_reduce", json!({})),
        ("A2_chunked_ctx", "chunked_ctx", json!({})),
        ("B4_central_triage", "central_triage", json!({})),
        ("C_recursive_sum", "recursive", json!({})),
        (
            "D_hier_nolineage",
            "hier",
            h(json!({"lineage": false, "independence": false, "contradiction": false,
                     "downward_retrieval": false, "questions": false, "cross_links": false})),
        ),
        ("E_hier_lineage", "hier", json!({"downward_retrieval": false, "questions": false, "cross_links": false})),
        ("F_hier_retrieval", "hier", json!({"downward_retrieval": true, "questions": false, "cross_links": false})),
        ("G_hier_questions", "hier", json!({"downward_retrieval": true, "questions": true, "cross_links": false})),
        ("H_mycelic_full", "hier", json!({"downward_retrieval": true, "questions": true, "cross_links": true})),
        (
            "I_mycelic_completion",
            "hier",
            json!({"downward_retrieval": true, "questions": true, "cross_links": true, "chain_completion": true}),
        ),
        (
            "J_mycelic_verified",
            "hier",
            json!({"downward_retrieval": true, "questions": true, "cross_links": true, "verify_evidence": true}),
        ),
        ("H_mycelic_prev", "hier", v1_hier_cfg()),
        (
            "H_mycelic_lean",
            "hier",
            json!({"downward_retrieval": true, "questions": true, "cross_links": true,
                   "strict_targeting": true, "triage_target_chains": "span2"}),
        ),
        ("Y_oracle_retrieval", "oracle", json!({})),
        ("Z_random_rank", "randrank", json!({})),
        ("Z2_naive_enumerate", "naive_enum", json!({})),
    ]
}

pub fn arch_spec(name: &str) -> Option<(&'static str, Value)> {
    archs().into_iter().find(|a| a.0 == name).map(|a| (a.1, a.2))
}

/// The result of one run plus what the loss accounting reads from a hierarchy run.
pub struct ArchRun {
    pub res: RunResult,
    pub hier: Option<systems::HierRunnerState>,
}

/// `run_arch` / `experiments._run_named`: one architecture on one world.  `cfg_over` merges into a hierarchy's
/// own overrides; `ranker_arch` names the architecture whose ranker choice applies (the name itself by default).
pub fn run_arch(
    cal: &Cal,
    name: &str,
    world: &World,
    alloc: &[Tier],
    seed: u64,
    cfg_over: Option<&Map<String, Value>>,
) -> ArchRun {
    let (kind, spec) = arch_spec(name).unwrap_or(("hier", json!({})));
    let ranker = cal.ranker_for(if arch_spec(name).is_some() { name } else { "H_mycelic_full" });
    let ranker = ranker.as_ref();
    let c = &world.corpus;
    let kt = &alloc[ENT];
    let mr_budget = cal.int("mr_budget") as usize;
    let res = match kind {
        "flat_rag" => systems::flat_rag(c, kt, seed, cal.int("flat_budget"), &world.near_miss, true, ranker),
        "long_context" => systems::long_context(c, kt, seed, &world.near_miss, ranker),
        "map_reduce" => {
            let ul = world.user_layer(&alloc[USER], seed);
            systems::map_reduce(c, alloc, seed, mr_budget, &ul, true, ranker)
        }
        "chunked_ctx" => systems::chunked_long_context(c, alloc, seed, &world.near_miss, 64, true, ranker),
        "central_triage" => {
            let ul = world.user_layer(&alloc[USER], seed);
            systems::central_triage(c, alloc, seed, &ul, 2, 0.35, 6000, cal.int("ct_kernel_ko_cap") as usize, ranker)
        }
        "oracle" => {
            let ul = world.user_layer(&alloc[USER], seed);
            systems::oracle_retrieval(c, alloc, seed, &ul, ranker)
        }
        "naive_enum" => {
            let ul = world.user_layer(&alloc[USER], seed);
            systems::naive_enumerate(c, alloc, &ul)
        }
        "randrank" => {
            let ul = world.user_layer(&alloc[USER], seed);
            systems::random_rank(systems::map_reduce(c, alloc, seed, mr_budget, &ul, true, ranker), seed)
        }
        "recursive" => {
            let ul = world.user_layer(&alloc[USER], seed);
            systems::recursive_summary(c, alloc, seed, &ul, &world.near_miss, ranker)
        }
        _ => {
            let mut over = spec.as_object().cloned().unwrap_or_default();
            if let Some(extra) = cfg_over {
                for (k, v) in extra {
                    over.insert(k.clone(), v.clone());
                }
            }
            let cfg = cal.hier_cfg(&over);
            let ul = world.user_layer(&alloc[USER], seed);
            let (res, state) = HierRunner::new(c, alloc.to_vec(), cfg, seed, &ul, &world.near_miss, ranker).run();
            return ArchRun { res, hier: Some(state) };
        }
    };
    ArchRun { res, hier: None }
}

/// `evaluate` with the defaults (`lenient=False, tau=0.5`).
pub fn evaluate(world: &World, res: &RunResult) -> Metrics {
    evalm::evaluate(&world.corpus, &world.gold, res, false, 0.5)
}

/// The default allocation.
pub fn back_loaded() -> Vec<Tier> {
    allocation_default("back-loaded")
}
