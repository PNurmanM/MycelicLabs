//! Direct measurement of the operator that actually binds: CANDIDATE DISCRIMINATION (`live_rank.py`).
//!
//! The five primitive-operator tasks (`live_tasks`) saturate at 1.00 for frontier-class models.  That is a real
//! result - those primitives are solved - but it means they cannot explain why enterprise-scale discovery is hard,
//! and they cannot discriminate between model tiers where the architecture needs it.
//!
//! The operator that does bind is the kernel's judgement over a *candidate list*: given several hundred
//! entity-chain hypotheses, each with its evidence statistics, which ones are genuine emerging cross-organisational
//! risks?  In the simulator this is a calibrated logistic over (links, weakest-link independent support, branch
//! spread, links-in-distinct-branches, chain validity, contradictions).  If a real frontier model ranks these
//! materially better than that logistic, the simulator understates the value of top-tier compute at the kernel and
//! every allocation conclusion has to be revisited.  If it does not, the simulation is fair.
//!
//! The candidates are taken from a real run of the benchmark, so the difficulty, the class balance and the feature
//! distributions are the ones the architecture actually produces - not a hand-made puzzle.

use crate::corpus::{Corpus, SCHEMA};
use crate::evalm::match_sets;
use crate::live_tasks::{causal_chains_json, key_dir, py_prefix, read_json, write_json};
use crate::models::allocation_default;
use crate::np::{Generator, math};
use crate::ops::{Hypothesis, Ko};
use crate::org::{REGION, SITE, USER};
use crate::py::json::{self as pyjson, Style};
use crate::py::value::{fl, py_str, truthy};
use crate::runner::{Cal, art, build_world_default};
use crate::systems::HierRunner;
use anyhow::Result;
use indexmap::IndexMap;
use serde_json::{Map, Value, json};
use std::collections::{BTreeSet, HashMap, HashSet};
use std::path::Path;

pub const N_ITEMS: usize = 60;

/// Per-predicate evidence statistics of one candidate (the `by_pred` dict of `_render`).
struct LinkStats {
    sigs: HashSet<i64>,
    sites: HashSet<i64>,
    regions: HashSet<i64>,
    tmin: i64,
    tmax: i64,
    raw: i64,
    neg: i64,
}

/// `_render(h, corpus, rank, rich)`: one candidate as the model sees it.  `rich` adds the work notes behind the
/// statistics, so we can ask whether the ceiling is in the reasoner or in what the abstraction kept.
fn render(h: &Hypothesis, arena: &[Ko], corpus: &Corpus, rank: usize, rich: bool) -> Value {
    let mut by_pred: IndexMap<i64, LinkStats> = IndexMap::new();
    for &id in &h.kos {
        let k = &arena[id];
        let d = by_pred.entry(k.pred).or_insert_with(|| LinkStats {
            sigs: HashSet::new(),
            sites: HashSet::new(),
            regions: HashSet::new(),
            tmin: 1_000_000_000,
            tmax: -1,
            raw: 0,
            neg: 0,
        });
        d.sigs.extend(k.sigs.iter());
        d.sites.extend(k.branch(SITE).into_iter().flat_map(|s| s.iter()));
        d.regions.extend(k.branch(REGION).into_iter().flat_map(|s| s.iter()));
        d.tmin = d.tmin.min(k.tmin);
        d.tmax = d.tmax.max(k.tmax);
        d.raw += k.n_raw;
        d.neg += i64::from(k.polarity < 0);
    }
    // links in order of first sighting; the sort is stable, so ties keep the order the objects were met in
    let mut order: Vec<(&i64, &LinkStats)> = by_pred.iter().collect();
    order.sort_by_key(|(_, d)| d.tmin);
    let links: Vec<Value> = order
        .into_iter()
        .map(|(&p, d)| {
            json!({
                "event": SCHEMA.predicates[p as usize],
                "first_seen_day": d.tmin,
                "last_seen_day": d.tmax,
                "independent_sources": d.sigs.len(),
                "total_reports": d.raw,
                "distinct_sites": d.sites.len(),
                "distinct_regions": d.regions.len(),
                "contradicting_reports": d.neg,
            })
        })
        .collect();
    let n_links = links.len();
    let mut out = json!({
        "id": format!("C-{rank}"),
        "entity": corpus.entities[h.anchor as usize],
        "links": links,
        "n_links": n_links,
        "distinct_regions_total": h.n_branch_regions,
        "distinct_sites_total": h.n_branch_sites,
        "contradiction_count": h.contra,
    });
    if rich {
        let mut seen = HashSet::new();
        let notes: Vec<String> = h
            .kos
            .iter()
            .flat_map(|&id| arena[id].evidence.iter().take(3))
            .map(|&r| corpus.text(r as usize))
            .filter(|t| seen.insert(t.clone()))
            .take(24)
            .collect();
        out["source_notes"] = json!(notes);
    }
    out
}

/// `build(seed, scale, rich)`: run the full hierarchy on one world, take up to 60 of its register's candidates (a
/// quarter real, the rest spread over the simulator's confidence range, one per entity) and write the task file to
/// `art_dir/live_rank_tasks{_rich}.json` and the key to `key_dir/live_rank_key{_rich}.json`.  `cal` is the
/// calibration the hierarchy runs with (`Cal::load()`, i.e. `artifacts/calibration.json`, in the Python); it is a
/// parameter so that writing elsewhere does not also change the operating point.
pub fn build(seed: u64, scale: usize, rich: bool, cal: &Cal, art_dir: &Path, key_dir: &Path) -> Result<Value> {
    let w = build_world_default(scale, seed);
    let alloc = allocation_default("back-loaded");
    let mut over = Map::new();
    for k in ["downward_retrieval", "questions", "cross_links"] {
        over.insert(k.into(), Value::Bool(true));
    }
    // `experiments._hier(...)`: the calibrated knobs, then the overrides
    let cfg = cal.hier_cfg(&over);
    let ul = w.user_layer(&alloc[USER], seed);
    // No ranker: the Python constructed HierRunner directly, and nothing in `python -m research.mycelic.live_rank`
    // sets the module-global `ops.RANKER` (only run_arch / the experiment drivers do), so it was None.
    let (res, _) = HierRunner::new(&w.corpus, alloc, cfg, seed, &ul, &w.near_miss, None).run();
    let hyps = res.ranked();
    let discoverable: HashSet<usize> = w.gold.discoverable.iter().copied().collect();
    let real: Vec<_> = w.corpus.patterns.iter().filter(|p| discoverable.contains(&p.pid)).collect();
    let matched = match_sets(&hyps, &real, &w.stem, false, false);
    let gold_idx: BTreeSet<usize> = matched.values().flatten().copied().collect();

    let mut rng = Generator::new(seed);
    let n_gold = gold_idx.len().min(N_ITEMS / 4);
    let gold: Vec<usize> = gold_idx.iter().copied().take(n_gold).collect();
    let others: Vec<usize> = (0..hyps.len())
        .filter(|i| !gold_idx.contains(i) && !hyps[*i].hallucinated && !hyps[*i].kos.is_empty())
        .collect();
    // sample distractors from across the simulator's own confidence range, so the model is not handed an easy
    // split (a stable sort on -conf, as `sorted(key=lambda i: -conf)`)
    let mut others_sorted = others;
    others_sorted.sort_by(|&a, &b| py_float_cmp(-hyps[a].conf, -hyps[b].conf));
    let take = N_ITEMS - n_gold;
    if others_sorted.len() > take {
        let pick = math::linspace(0.0, (others_sorted.len() - 1) as f64, take);
        others_sorted = pick.into_iter().map(|x| others_sorted[x as usize]).collect();
    }
    // At most one candidate per entity.  A repeated entity lets a model compare two hypotheses about the same thing
    // and infer which is the real one from the contrast - a hint that has nothing to do with the operator being
    // measured.  (A measurement subject reported using exactly this.)
    let mut seen_entity = HashSet::new();
    let chosen: Vec<usize> =
        gold.into_iter().chain(others_sorted).filter(|&ci| seen_entity.insert(hyps[ci].anchor)).collect();
    let order = rng.permutation(chosen.len());
    let mut items = Vec::with_capacity(chosen.len());
    let mut key = Map::new();
    for (rank, &o) in order.iter().enumerate() {
        let ci = chosen[o as usize];
        let item = render(hyps[ci], &res.arena, &w.corpus, rank, rich);
        key.insert(
            py_str(&item["id"]),
            json!({"is_real": gold_idx.contains(&ci), "simulator_confidence": fl(hyps[ci].conf)}),
        );
        items.push(item);
    }
    let spec = json!({
        "seed": seed, "scale": scale,
        "n_items": items.len(),
        "causal_chains": causal_chains_json(),
        "instructions_summary":
            "Each candidate is one entity plus the operational events reported \
             about it across the enterprise. Decide which candidates are \
             genuine emerging cross-organisational risks.",
        "candidates": items,
    });
    let suffix = if rich { "_rich" } else { "" };
    write_json(art_dir, &format!("live_rank_tasks{suffix}.json"), &spec, 1)?;
    write_json(key_dir, &format!("live_rank_key{suffix}.json"), &Value::Object(key), 1)?;
    Ok(spec)
}

/// The order `sorted` sees between two float keys: by `<`, so -0.0 and 0.0 tie (and keep their order).
fn py_float_cmp(a: f64, b: f64) -> std::cmp::Ordering {
    a.partial_cmp(&b).unwrap_or(std::cmp::Ordering::Equal)
}

/// `_ap(ranked_ids, key)`: average precision of a ranking against the key's real candidates.
fn average_precision(ranked: &[String], key: &Map<String, Value>) -> f64 {
    let gold: Vec<&String> = key.iter().filter(|(_, k)| truthy(&k["is_real"])).map(|(i, _)| i).collect();
    if gold.is_empty() {
        return 0.0;
    }
    let pos: HashMap<&str, usize> = ranked.iter().enumerate().map(|(i, r)| (r.as_str(), i)).collect();
    let mut found: Vec<usize> = gold.iter().filter_map(|g| pos.get(g.as_str()).copied()).collect();
    found.sort();
    let mut ap = 0.0;
    for (r, idx) in found.into_iter().enumerate() {
        ap += (r + 1) as f64 / (idx + 1) as f64;
    }
    ap / gold.len() as f64
}

/// `s.split("-r")[0]`: the model name of a repeat run `<model>-rN`.
fn base_name(name: &str) -> &str {
    name.split("-r").next().unwrap_or(name)
}

/// `score(rich)`: every answer file in `art_dir` against `art_dir/live_rank_tasks{_rich}.json` and
/// `key_dir/live_rank_key{_rich}.json`, with the simulator's logistic and a random ranking as references; writes
/// `out_dir/live_rank_results{_rich}.json` (indent 2) and returns the result.
pub fn score(rich: bool, art_dir: &Path, key_dir: &Path, out_dir: &Path) -> Result<Value> {
    let suffix = if rich { "_rich" } else { "" };
    let spec = read_json(&art_dir.join(format!("live_rank_tasks{suffix}.json")))?;
    let key_value = read_json(&key_dir.join(format!("live_rank_key{suffix}.json")))?;
    let key = key_value.as_object().cloned().unwrap_or_default();
    let ids: Vec<String> = spec["candidates"].as_array().into_iter().flatten().map(|c| py_str(&c["id"])).collect();
    let n_gold = key.values().filter(|k| truthy(&k["is_real"])).count();
    let mut out = Map::new();
    out.insert("n_items".into(), json!(ids.len()));
    out.insert("n_real".into(), json!(n_gold));

    // reference 1: the simulator's own logistic (stable sort on -confidence)
    let conf = |i: &String| key[i.as_str()]["simulator_confidence"].as_f64().unwrap_or(0.0);
    let mut sim = ids.clone();
    sim.sort_by(|a, b| py_float_cmp(-conf(a), -conf(b)));
    out.insert("simulator_logistic_ap".into(), fl(average_precision(&sim, &key)));
    // reference 2: random
    let mut rng = Generator::new(5);
    let aps: Vec<f64> = (0..2000)
        .map(|_| {
            let mut r = ids.clone();
            rng.shuffle(&mut r);
            average_precision(&r, &key)
        })
        .collect();
    out.insert("random_ap".into(), fl(math::mean(&aps)));

    // answer files in sorted (code point) order, as `sorted(os.listdir(ART))`
    let mut files: Vec<String> =
        std::fs::read_dir(art_dir)?.filter_map(|e| e.ok().and_then(|e| e.file_name().into_string().ok())).collect();
    files.sort();
    let prefix = format!("live_rank{suffix}_answers_");
    let mut models = Map::new();
    for name in files {
        let Some(rest) = name.strip_prefix(&prefix) else { continue };
        if !rich && name.contains("_rich") {
            continue;
        }
        // fn[len(pref):-len(".json")]: the last five code points go whatever they are
        let model = py_prefix(rest, rest.chars().count().saturating_sub(5)).to_string();
        let a = read_json(&art_dir.join(&name))?;
        let returned = a.get("ranking").and_then(Value::as_array).map(Vec::as_slice).unwrap_or_default();
        let mut ranked: Vec<String> = returned.iter().map(py_str).filter(|x| key.contains_key(x)).collect();
        let ranked_set: HashSet<String> = ranked.iter().cloned().collect();
        ranked.extend(ids.iter().filter(|i| !ranked_set.contains(*i)).cloned());
        let picks: HashSet<String> =
            a.get("real").and_then(Value::as_array).into_iter().flatten().map(py_str).collect();
        let tp = picks.iter().filter(|p| key.get(p.as_str()).is_some_and(|k| truthy(&k["is_real"]))).count();
        let precision = tp as f64 / picks.len().max(1) as f64;
        let recall = tp as f64 / n_gold.max(1) as f64;
        models.insert(
            model,
            json!({
                "ap": fl(average_precision(&ranked, &key)),
                "selection_precision": fl(precision),
                "selection_recall": fl(recall),
                "selection_f1": fl(2.0 * precision * recall / (precision + recall).max(1e-9)),
                "n_selected": picks.len(),
                "n_ranked_returned": returned.len(),
            }),
        );
    }
    // Repeats are named "<model>-rN".  A single run per cell cannot carry a mechanism claim, so repeats are
    // aggregated per model with their spread reported rather than averaged away.
    let mut by_name: IndexMap<&str, Vec<&Value>> = IndexMap::new();
    for (k, v) in &models {
        by_name.entry(base_name(k)).or_default().push(v);
    }
    let mut aggregated = Map::new();
    for (base, runs) in by_name {
        let field = |f: &str| runs.iter().map(|v| v[f].as_f64().unwrap_or(0.0)).collect::<Vec<f64>>();
        let (aps, f1s) = (field("ap"), field("selection_f1"));
        let min = |x: &[f64]| x.iter().copied().fold(f64::INFINITY, f64::min);
        let max = |x: &[f64]| x.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        aggregated.insert(
            base.to_string(),
            json!({
                "n_runs": runs.len(),
                "ap_mean": fl(math::mean(&aps)),
                "ap_min": fl(min(&aps)),
                "ap_max": fl(max(&aps)),
                "selection_f1_mean": fl(math::mean(&f1s)),
                "selection_f1_min": fl(min(&f1s)),
                "selection_f1_max": fl(max(&f1s)),
            }),
        );
    }
    out.insert("models".into(), Value::Object(models));
    out.insert("models_aggregated".into(), Value::Object(aggregated));
    let out = Value::Object(out);
    write_json(out_dir, &format!("live_rank_results{suffix}.json"), &out, 2)?;
    Ok(out)
}

/// `python -m research.mycelic.live_rank [score] [rich]`: build the candidate list (seed 901, 10,000 users), or
/// score the answers with `score`; `rich` selects the variant with the source notes.
pub fn main(args: &[String]) -> Result<()> {
    let rich = args.iter().any(|a| a == "rich");
    let art_dir = art();
    if args.iter().any(|a| a == "score") {
        let res = score(rich, &art_dir, &key_dir(), &art_dir)?;
        println!("{}", pyjson::dumps_with(&res, Style::indent(2)));
    } else {
        let spec = build(901, 10_000, rich, &Cal::load(), &art_dir, &key_dir())?;
        println!("wrote live_rank_tasks.json with {} candidates", spec["n_items"]);
        let first = pyjson::dumps_with(&spec["candidates"][0], Style::indent(1));
        println!("{}", py_prefix(&first, 900));
    }
    Ok(())
}
