//! The properties the study's conclusions depend on (the port of `test_mycelic.py`): determinism, the leakage
//! contract, the fairness contract, generator invariants and metric sanity.  The report-generator tests live
//! with the generators (`tests/docs_identity.rs`) and the calibrator's command line in `calibrator.rs`.

use mycelic_research::corpus::{CAUSAL_CHAINS, Corpus, KIND_FACET, build_corpus, make_gold, pred_id};
use mycelic_research::models::{self, HardShape, tier_from_q};
use mycelic_research::np::{Generator, math};
use mycelic_research::ops::{self, ExtractResult, chain_span};
use mycelic_research::org::{DEPT, ENT, REGION, SITE, TEAM, USER, build_org_default};
use mycelic_research::runner::{self, Cal, World};
use mycelic_research::systems::{HierConfig, HierRunner, Meter, RunResult};
use std::collections::{HashMap, HashSet};

const SMALL: usize = 1200;

fn small_world() -> World {
    runner::build_world_default(SMALL, 0)
}

fn small7b() -> models::Tier {
    models::anchor("small-7b", HardShape::Convex)
}

// ---- org ----------------------------------------------------------------------------------------------------

#[test]
fn org_is_deterministic() {
    let (a, b) = (build_org_default(SMALL, 3), build_org_default(SMALL, 3));
    assert_eq!(a.user_ids.len(), b.user_ids.len());
    assert_eq!(
        a.nodes.iter().map(|n| n.parent).collect::<Vec<_>>(),
        b.nodes.iter().map(|n| n.parent).collect::<Vec<_>>()
    );
}

#[test]
fn org_has_six_levels_populated() {
    let o = build_org_default(10_000, 0);
    for lvl in [USER, TEAM, DEPT, SITE, REGION] {
        assert!(!o.levels[lvl].is_empty(), "level {lvl} empty");
    }
    assert_eq!(o.levels[ENT].len(), 1);
}

#[test]
fn org_fanin_is_heterogeneous() {
    // a balanced k-ary tree would make the whole study meaningless
    let o = build_org_default(50_000, 0);
    for lvl in [TEAM, DEPT, SITE] {
        let fan: Vec<i64> = o.levels[lvl].iter().map(|&n| o.nodes[n].children.len() as i64).collect();
        assert!(math::std_int(&fan, 0) > 1.0, "level {lvl} too uniform");
        assert!(*fan.iter().max().unwrap() > 2 * math::median_int(&fan) as i64, "level {lvl} has no large nodes");
    }
}

#[test]
fn every_user_reaches_the_root() {
    let o = build_org_default(5_000, 1);
    for &u in o.user_ids.iter().step_by(97) {
        assert_eq!(o.ancestor_at(u, ENT), Some(o.root));
        for lvl in [TEAM, DEPT, SITE, REGION] {
            assert!(o.ancestor_at(u, lvl).is_some());
        }
    }
}

// ---- corpus -------------------------------------------------------------------------------------------------

fn small_corpus() -> Corpus {
    build_corpus(build_org_default(SMALL, 0), 0, None)
}

#[test]
fn corpus_is_deterministic() {
    let (a, b) = (small_corpus(), small_corpus());
    assert_eq!(a.recs.pred, b.recs.pred);
    assert_eq!(a.recs.anchor, b.recs.anchor);
    assert_eq!(a.text(17), b.text(17));
}

#[test]
fn discoverable_patterns_span_multiple_regions() {
    // a pattern visible inside one branch would not test cross-organisational discovery
    let c = small_corpus();
    let gold = make_gold(&c, 3);
    for &pid in &gold.discoverable {
        assert!(c.patterns[pid].n_regions >= 2, "pattern {pid}");
    }
}

#[test]
fn no_single_user_holds_a_whole_pattern() {
    let c = small_corpus();
    for p in c.patterns.iter().filter(|p| p.real) {
        let mut per_user: HashMap<i32, HashSet<usize>> = HashMap::new();
        for (j, fr) in p.facet_records.iter().enumerate() {
            for &r in fr {
                per_user.entry(c.recs.uid[r]).or_default().insert(j);
            }
        }
        for (u, facets) in per_user {
            assert!(facets.len() < 2.max(p.preds.len()), "user {u} holds most of pattern {}", p.pid);
        }
    }
    assert!(c.recs.kind.contains(&KIND_FACET));
}

#[test]
fn all_five_decoy_classes_present() {
    let c = small_corpus();
    let kinds: HashSet<i64> = c.patterns.iter().filter(|p| !p.real).map(|p| p.decoy_type).collect();
    assert_eq!(kinds, HashSet::from([1, 2, 3, 4, 5]));
}

fn jaccard(a: &[String], b: &[String]) -> f64 {
    let (a, b): (HashSet<&String>, HashSet<&String>) = (a.iter().collect(), b.iter().collect());
    a.intersection(&b).count() as f64 / 1.max(a.union(&b).count()) as f64
}

#[test]
fn echoes_repeat_wording_independents_do_not() {
    let c = small_corpus();
    let mut ev: Vec<(i32, Vec<usize>)> = Vec::new();
    let mut index: HashMap<i32, usize> = HashMap::new();
    for i in 0..c.recs.len() {
        let e = c.recs.event[i];
        let slot = *index.entry(e).or_insert_with(|| {
            ev.push((e, Vec::new()));
            ev.len() - 1
        });
        ev[slot].1.push(i);
    }
    let shared: Vec<&Vec<usize>> = ev.iter().map(|e| &e.1).filter(|v| v.len() > 1).take(40).collect();
    assert!(!shared.is_empty(), "no echoed events in the corpus");
    let sims: Vec<f64> = shared.iter().map(|g| jaccard(&c.tokens(g[0]), &c.tokens(g[1]))).collect();
    assert!(math::mean(&sims) > 0.7, "echoes should be near-duplicates");
    let mut rng = Generator::new(0);
    let mut other = Vec::new();
    for _ in 0..60 {
        let ij = rng.integers_n(0, c.recs.len() as i64, 2);
        let (i, j) = (ij[0] as usize, ij[1] as usize);
        if c.recs.event[i] == c.recs.event[j] {
            continue;
        }
        other.push(jaccard(&c.tokens(i), &c.tokens(j)));
    }
    assert!(math::mean(&other) < 0.55, "independent records should not look like duplicates");
}

#[test]
fn entity_locality() {
    // most entities must belong somewhere, or index routing is a giveaway on one side and useless on the other
    let c = small_corpus();
    let mut d: HashMap<i32, HashSet<usize>> = HashMap::new();
    for i in 0..c.recs.len() {
        d.entry(c.recs.anchor[i]).or_default().insert(c.org.ancestor_at(c.recs.uid[i] as usize, REGION).unwrap());
    }
    let multi = d.values().filter(|v| v.len() >= 2).count();
    assert!(multi as f64 / 1.max(d.len()) as f64 > 0.30, "entities too local; triage trivial");
    let gold = make_gold(&c, 3);
    let gold_anchors: HashSet<usize> = gold.discoverable.iter().map(|&p| c.patterns[p].anchor).collect();
    assert!((gold_anchors.len() as f64 / 1.max(multi) as f64) < 0.5, "multi-region entities are almost all gold");
}

// ---- operators ----------------------------------------------------------------------------------------------

#[test]
fn extraction_output_carries_no_ground_truth() {
    // The leakage contract: an operator may READ truth to simulate noise, but nothing it returns exposes it.
    // The destructuring below fails to compile if a field is added to ExtractResult.
    let w = small_world();
    let mut rng = Generator::new(0);
    let ids: Vec<i64> = (0..500).collect();
    let ex = ops::extract_default(&w.corpus, &ids, &small7b(), &mut rng, &w.near_miss);
    let ExtractResult { rid, uid, pred, anchor, t, polarity, sig, spurious } = ex;
    let n = rid.len();
    assert!(
        [uid.len(), pred.len(), anchor.len(), t.len(), polarity.len(), sig.len(), spurious.len()]
            .iter()
            .all(|&l| l == n)
    );
}

#[test]
fn stronger_tier_extracts_better() {
    let w = small_world();
    let ids: Vec<i64> = (0..3000).collect();
    let lo = ops::extract_default(
        &w.corpus,
        &ids,
        &models::anchor("edge-3b", HardShape::Convex),
        &mut Generator::new(1),
        &w.near_miss,
    );
    let hi = ops::extract_default(
        &w.corpus,
        &ids,
        &models::anchor("frontier-plus", HardShape::Convex),
        &mut Generator::new(1),
        &w.near_miss,
    );
    assert!(hi.len() > lo.len());
}

#[test]
fn entity_errors_are_per_agent_not_per_mention() {
    // the same agent must read the same entity the same way every time
    let w = small_world();
    let ids: Vec<i64> = (0..w.corpus.recs.len() as i64).collect();
    let ex = ops::extract_default(&w.corpus, &ids, &small7b(), &mut Generator::new(0), &w.near_miss);
    let mut seen: HashMap<(i64, i32), HashSet<i32>> = HashMap::new();
    for i in 0..ex.len() {
        if ex.spurious[i] {
            continue; // an invented claim has no true anchor
        }
        let true_anchor = w.corpus.recs.anchor[ex.rid[i] as usize];
        seen.entry((ex.uid[i], true_anchor)).or_default().insert(ex.anchor[i]);
    }
    assert_eq!(seen.values().filter(|v| v.len() > 1).count(), 0, "an agent resolved the same entity two ways");
}

#[test]
fn chain_span_finds_the_longest_run() {
    let mut mask = 0i64;
    for p in &CAUSAL_CHAINS[0][..4] {
        mask |= 1 << pred_id(p);
    }
    assert_eq!(chain_span(mask), (4, 0));
    assert_eq!(chain_span(0).0, 0);
}

#[test]
fn capability_vector_is_monotone_in_q() {
    let mut prev: Option<models::Tier> = None;
    for q in math::linspace(0.05, 1.0, 12) {
        let t = tier_from_q(q, None, None, None, HardShape::Convex);
        if let Some(p) = &prev {
            assert!(t.extract_recall >= p.extract_recall - 1e-9);
            assert!(t.causal_check >= p.causal_check - 1e-9);
            assert!(t.hallucination <= p.hallucination + 1e-9);
        }
        prev = Some(t);
    }
}

// ---- systems ------------------------------------------------------------------------------------------------

fn run(w: &World, arch: &str) -> RunResult {
    runner::run_arch(&Cal::load(), arch, w, &runner::back_loaded(), 0, None).res
}

#[test]
fn runs_are_deterministic() {
    let w = small_world();
    let (a, b) = (run(&w, "H_mycelic_full"), run(&w, "H_mycelic_full"));
    let (ma, mb) = (runner::evaluate(&w, &a), runner::evaluate(&w, &b));
    for k in ["average_precision", "found_anywhere_in_register", "compute_units", "tokens_total"] {
        assert_eq!(ma[k], mb[k], "{k}");
    }
}

#[test]
fn hierarchy_never_moves_raw_text_off_the_user_node() {
    assert_eq!(run(&small_world(), "H_mycelic_full").exposed_raw_records, 0);
}

#[test]
fn flat_baselines_do_move_raw_text() {
    assert!(run(&small_world(), "A_flat_rag").exposed_raw_records > 0);
}

#[test]
fn random_rank_has_the_same_candidates_as_its_parent() {
    let w = small_world();
    let (base, rr) = (run(&w, "B2_map_reduce"), run(&w, "Z_random_rank"));
    let anchors = |r: &RunResult| {
        let mut v: Vec<i64> = r.ranked().iter().map(|h| h.anchor).collect();
        v.sort();
        v
    };
    assert_eq!(anchors(&base), anchors(&rr));
    assert_eq!(
        runner::evaluate(&w, &base)["found_anywhere_in_register"],
        runner::evaluate(&w, &rr)["found_anywhere_in_register"]
    );
}

#[test]
fn oracle_has_near_total_evidence_coverage() {
    let w = small_world();
    assert!(runner::evaluate(&w, &run(&w, "Y_oracle_retrieval"))["evidence_coverage_2links"] > 0.9);
}

#[test]
fn every_call_is_metered() {
    let r = run(&small_world(), "H_mycelic_full");
    let m = &r.meter;
    assert!(m.calls() > 0 && m.tok_in() > 0 && m.cu() > 0.0);
    for (stage, v) in &m.levels {
        if v.calls > 0 {
            assert!(v.tok_in > 0, "{stage} unmetered");
        }
    }
}

#[test]
fn kernel_context_is_comparable_across_architectures() {
    let w = small_world();
    let ctx: Vec<f64> = ["B2_map_reduce", "E_hier_lineage", "H_mycelic_full"]
        .iter()
        .map(|a| runner::evaluate(&w, &run(&w, a))["kernel_context_tokens"])
        .filter(|&v| v > 0.0)
        .collect();
    let (mx, mn) = (ctx.iter().cloned().fold(f64::MIN, f64::max), ctx.iter().cloned().fold(f64::MAX, f64::min));
    assert!(mx / mn.max(1.0) < 6.0, "kernel contexts differ wildly: {ctx:?}");
}

#[test]
fn ablating_lineage_destroys_lineage_accuracy() {
    let w = small_world();
    let alloc = runner::back_loaded();
    let cfg = HierConfig {
        lineage: false,
        independence: false,
        downward_retrieval: false,
        questions: false,
        ..Default::default()
    };
    let ul = w.user_layer(&alloc[USER], 0);
    let (r, _) = HierRunner::new(&w.corpus, alloc, cfg, 0, &ul, &w.near_miss, None).run();
    assert_eq!(runner::evaluate(&w, &r)["lineage_accuracy"], 0.0);
}

#[test]
fn descent_does_not_broadcast() {
    // A targeted descent must touch a bounded number of nodes, and the bound must NOT grow with the enterprise;
    // checked at 10k, where the org is big enough for the claim to mean something.
    let w = runner::build_world_default(10_000, 0);
    let alloc = runner::back_loaded();
    let cfg = HierConfig::default();
    let leaf_cap = cfg.descent_leaf_cap;
    let ul = w.user_layer(&alloc[USER], 0);
    let mut h = HierRunner::new(&w.corpus, alloc, cfg, 0, &ul, &w.near_miss, None);
    h.build_user_layer();
    for lvl in [TEAM, DEPT, SITE, REGION] {
        h.aggregate_level(lvl);
    }
    let before = h.meter.levels.get("descend-read").map_or(0, |m| m.calls);
    let anchor = w.corpus.patterns[w.gold.discoverable[0]].anchor as i64;
    h.descend(anchor, &[], 3, None, None);
    let after = h.meter.levels["descend-read"].calls - before;
    assert!(after as usize <= leaf_cap, "descent exceeded its own leaf cap");
    assert!((after as f64) < 0.02 * w.corpus.org.user_ids.len() as f64, "descent touched too much of the org");
}

// ---- metrics ------------------------------------------------------------------------------------------------

#[test]
fn empty_report_scores_zero_not_nan() {
    let w = small_world();
    let r = RunResult { name: "empty".into(), meter: Meter::default(), ..Default::default() };
    let m = runner::evaluate(&w, &r);
    for (k, v) in &m {
        assert!(!v.is_nan(), "{k} is NaN");
    }
    assert_eq!(m["discovery_recall"], 0.0);
    assert_eq!(m["average_precision"], 0.0);
}

#[test]
fn strict_match_never_exceeds_primary() {
    let w = small_world();
    for a in ["A_flat_rag", "H_mycelic_full", "B2_map_reduce"] {
        let m = runner::evaluate(&w, &run(&w, a));
        assert!(m["discovery_recall_strict"] <= m["discovery_recall"] + 1e-9, "{a}");
    }
}

#[test]
fn found_is_at_least_recall_at_100() {
    let w = small_world();
    for a in ["A_flat_rag", "H_mycelic_full"] {
        let m = runner::evaluate(&w, &run(&w, a));
        assert!(m["found_anywhere_in_register"] + 1e-9 >= m["recall_at_100"], "{a}");
    }
}
