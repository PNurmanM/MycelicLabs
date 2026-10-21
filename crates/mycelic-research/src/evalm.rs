//! Metrics (`evalm.py`).  Raw measurements first; composites only as a convenience.
//!
//! A reported hypothesis matches gold pattern p iff `hyp.anchor == p.anchor` and
//! * primary (every headline number): `|hyp.preds & p.preds| >= max(2, ceil(len(p.preds) / 2))`
//! * strict (`*_strict`): `|hyp.preds & p.preds| >= min(3, len(p.preds))`.

use crate::corpus::{Corpus, Gold, Pattern};
use crate::np::math;
use crate::ops::{Hypothesis, stem_rep_map};
use crate::org::REGION;
use crate::py::PySet;
use crate::systems::RunResult;
use indexmap::IndexMap;
use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::HashSet;

/// `_match_sets`: gold pattern id -> sorted indices of the hypotheses that match it, in pattern order.
pub fn match_sets(
    hyps: &[&Hypothesis],
    patterns: &[&Pattern],
    stem: &[i32],
    lenient: bool,
    strict: bool,
) -> IndexMap<usize, Vec<usize>> {
    let mut by_anchor: FxHashMap<i64, Vec<usize>> = FxHashMap::default();
    for (i, h) in hyps.iter().enumerate() {
        by_anchor.entry(h.anchor).or_default().push(i);
        if lenient {
            by_anchor.entry(stem[h.anchor as usize] as i64).or_default().push(i);
        }
    }
    let mut out = IndexMap::new();
    for p in patterns {
        let mut cands: FxHashSet<usize> = by_anchor.get(&(p.anchor as i64)).into_iter().flatten().copied().collect();
        if lenient {
            cands.extend(by_anchor.get(&(stem[p.anchor] as i64)).into_iter().flatten().copied());
        }
        let ps: HashSet<usize> = p.preds.iter().copied().collect();
        let need = if strict { 3.min(p.preds.len()) } else { 2.max(p.preds.len().div_ceil(2)) };
        let mut hit: Vec<usize> = cands
            .into_iter()
            .filter(|&i| {
                let hs: HashSet<usize> = hyps[i].preds.iter().copied().collect();
                hs.intersection(&ps).count() >= need
            })
            .collect();
        if !hit.is_empty() {
            hit.sort();
            out.insert(p.pid, hit);
        }
    }
    out
}

struct Ranked {
    ap: f64,
    n_found: f64,
    recall_at: [f64; 3],
    precision_at: [f64; 3],
    r_precision: f64,
}

/// Average precision / recall@K over the ranked risk register.
fn ranked_metrics(ranked: &[&Hypothesis], real: &[&Pattern], stem: &[i32], lenient: bool, strict: bool) -> Ranked {
    let m = match_sets(ranked, real, stem, lenient, strict);
    let mut order: Vec<(usize, usize)> = m.iter().map(|(&pid, idxs)| (pid, *idxs.iter().min().unwrap())).collect();
    order.sort_by_key(|x| x.1);
    let n_gold = 1.max(real.len()) as f64;
    let mut ap = 0.0;
    for (rank_pos, &(_, idx)) in order.iter().enumerate() {
        ap += (rank_pos + 1) as f64 / (idx + 1) as f64;
    }
    ap /= n_gold;
    let mut recall_at = [0.0; 3];
    let mut precision_at = [0.0; 3];
    for (j, k) in [40usize, 100, 300].into_iter().enumerate() {
        let hit = order.iter().filter(|x| x.1 < k).count() as f64;
        recall_at[j] = hit / n_gold;
        precision_at[j] = hit / k.min(1.max(ranked.len())) as f64;
    }
    let rp = order.iter().filter(|x| (x.1 as f64) < n_gold).count() as f64 / n_gold;
    Ranked { ap, n_found: order.len() as f64, recall_at, precision_at, r_precision: rp }
}

/// The metrics of one run, in the Python dict's key order.
pub type Metrics = IndexMap<&'static str, f64>;

fn max_by_conf<'a>(hyps: &[&'a Hypothesis], idxs: &[usize]) -> (usize, &'a Hypothesis) {
    let mut best = idxs[0];
    for &i in &idxs[1..] {
        if hyps[i].conf > hyps[best].conf {
            best = i;
        }
    }
    (best, hyps[best])
}

/// `evaluate(corpus, gold, res, lenient=False, tau=0.5)`
pub fn evaluate(corpus: &Corpus, gold: &Gold, res: &RunResult, lenient: bool, tau: f64) -> Metrics {
    let stem = stem_rep_map(corpus);
    let ranked: Vec<&Hypothesis> = res.ranked();
    let hyps: Vec<&Hypothesis> = ranked.iter().copied().filter(|h| h.conf >= tau).collect();
    let pats: FxHashMap<usize, &Pattern> = corpus.patterns.iter().map(|p| (p.pid, p)).collect();
    let real: Vec<&Pattern> = gold.discoverable.iter().map(|i| pats[i]).collect();
    // D4 chains are genuine causal chains that are merely over-supported: scored separately, not as decoys
    let decoys: Vec<&Pattern> = corpus.patterns.iter().filter(|p| !p.real && p.decoy_type != 4).collect();
    let dup_infl: Vec<&Pattern> = corpus.patterns.iter().filter(|p| !p.real && p.decoy_type == 4).collect();

    let m_real = match_sets(&hyps, &real, &stem, lenient, false);
    let m_real_strict = match_sets(&hyps, &real, &stem, lenient, true);
    let m_dec_raw = match_sets(&hyps, &decoys, &stem, lenient, false);
    // a report that correctly identifies the real chain is not also a decoy acceptance
    let real_hyp: FxHashSet<usize> = m_real.values().flatten().copied().collect();
    let m_dec: IndexMap<usize, Vec<usize>> = m_dec_raw
        .into_iter()
        .map(|(pid, idxs)| (pid, idxs.into_iter().filter(|i| !real_hyp.contains(i)).collect::<Vec<_>>()))
        .filter(|(_, v)| !v.is_empty())
        .collect();
    let m_dup = match_sets(&hyps, &dup_infl, &stem, lenient, false);
    let dup_hyp: FxHashSet<usize> = m_dup.values().flatten().copied().collect();

    let n_rep = hyps.len() as i64 - dup_hyp.len() as i64;
    let matched_hyp: FxHashSet<usize> = m_real.values().flatten().copied().collect();
    let dec_hyp: FxHashSet<usize> = m_dec.values().flatten().copied().collect();

    let rec = m_real.len() as f64 / 1.max(real.len()) as f64;
    let prec = matched_hyp.len() as f64 / 1.max(n_rep) as f64;
    let f1 = 2.0 * rec * prec / 1e-9f64.max(rec + prec);
    let fdr = 1.0 - prec;

    let rare: Vec<&&Pattern> = real.iter().filter(|p| p.rare).collect();
    let rare_rec = rare.iter().filter(|p| m_real.contains_key(&p.pid)).count() as f64 / 1.max(rare.len()) as f64;
    let common: Vec<&&Pattern> = real.iter().filter(|p| !p.rare).collect();
    let common_rec = common.iter().filter(|p| m_real.contains_key(&p.pid)).count() as f64 / 1.max(common.len()) as f64;

    // decoy acceptance, by type
    let mut dec_acc = [0.0f64; 6];
    let mut dec_n = [0usize; 6];
    for p in &decoys {
        dec_n[p.decoy_type as usize] += 1;
        if m_dec.contains_key(&p.pid) {
            dec_acc[p.decoy_type as usize] += 1.0;
        }
    }
    for t in [1, 2, 3, 5] {
        dec_acc[t] /= 1.max(dec_n[t]) as f64;
    }

    // duplicate inflation: does the system over-count independent support?
    let mut dup_errs: Vec<f64> = Vec::new();
    for (pid, idxs) in &m_dup {
        let p = pats[pid];
        let true_ind = p.facet_records.iter().flatten().map(|&r| corpus.recs.event[r]).collect::<HashSet<i32>>().len();
        let (_, h) = max_by_conf(&hyps, idxs);
        dup_errs.push(4.0f64.min(h.n_indep as f64 / 1.max(true_ind) as f64));
    }
    let dup_err = if dup_errs.is_empty() { 1.0 } else { math::mean(&dup_errs) };

    let hall = hyps.iter().filter(|h| h.hallucinated).count();
    let hall_rate = hall as f64 / 1.max(n_rep) as f64;
    let unsupported = (0..hyps.len()).filter(|i| !matched_hyp.contains(i) && !dec_hyp.contains(i)).count();

    // --- cross-branch discovery / lineage / provenance ---
    let (mut cb_ok, mut cb_n, mut lin_num, mut lin_den, mut prov_ok) = (0usize, 0usize, 0usize, 0usize, 0usize);
    let mut ind_err: Vec<f64> = Vec::new();
    let mut ev_p: Vec<f64> = Vec::new();
    let mut ev_r: Vec<f64> = Vec::new();
    for (pid, idxs) in &m_real {
        let p = pats[pid];
        let (_, h) = max_by_conf(&hyps, idxs);
        if p.n_regions >= 2 {
            cb_n += 1;
            if h.n_branch_regions >= 2 {
                cb_ok += 1;
            }
        }
        let mut true_regions: FxHashSet<i64> = FxHashSet::default();
        for &r in p.facet_records.iter().flatten() {
            if let Some(a) = corpus.org.ancestor_at(corpus.recs.uid[r] as usize, REGION) {
                true_regions.insert(a as i64);
            }
        }
        let mut rep_regions = PySet::new();
        for &k in &h.kos {
            if let Some(s) = res.arena[k].branch(REGION) {
                rep_regions.update_set(s);
            }
        }
        if !rep_regions.is_empty() {
            lin_num += rep_regions.iter().filter(|r| true_regions.contains(r)).count();
            lin_den += rep_regions.len();
        }
        if !h.evidence.is_empty() && !h.kos.is_empty() && h.kos.iter().any(|&k| !res.arena[k].lineage.is_empty()) {
            prov_ok += 1;
        }
        let true_ind = *gold.independent_sources.get(pid).unwrap_or(&1) as f64;
        ind_err.push(2.0f64.min((h.n_indep as f64 - true_ind).abs() / 1.0f64.max(true_ind)));
        let gev: FxHashSet<i64> = gold.evidence.get(pid).into_iter().flatten().map(|&r| r as i64).collect();
        let rev: FxHashSet<i64> = h.evidence.iter().copied().collect();
        let inter = rev.iter().filter(|r| gev.contains(r)).count() as f64;
        if !rev.is_empty() {
            ev_p.push(inter / rev.len() as f64);
        }
        ev_r.push(inter / 1.max(gev.len().min(24)) as f64);
    }

    // --- contradiction detection (pattern level) ---
    let (mut tp, mut fp, mut fn_) = (0usize, 0usize, 0usize);
    for (pid, idxs) in &m_real {
        let (_, h) = max_by_conf(&hyps, idxs);
        let truth = pats[pid].contradicted;
        let pred_c = h.contra > 0;
        if truth && pred_c {
            tp += 1;
        } else if pred_c && !truth {
            fp += 1;
        } else if truth && !pred_c {
            fn_ += 1;
        }
    }
    let c_prec = tp as f64 / 1.max(tp + fp) as f64;
    let c_rec = tp as f64 / 1.max(tp + fn_) as f64;
    let c_f1 = 2.0 * c_prec * c_rec / 1e-9f64.max(c_prec + c_rec);

    // --- strongest-independent-support question, over the conclusions actually reached ---
    let mut top_ok = 0.0;
    if !m_real.is_empty() && gold.top_supported >= 0 && m_real.contains_key(&(gold.top_supported as usize)) {
        let (mut best_pid, mut best_val) = (-1i64, -1.0f64);
        for (pid, idxs) in &m_real {
            let v = idxs.iter().map(|&i| hyps[i].n_indep).max().unwrap() as f64;
            if v > best_val {
                best_val = v;
                best_pid = *pid as i64;
            }
        }
        top_ok = if best_pid == gold.top_supported { 1.0 } else { 0.0 };
    }

    // --- cross-region variant families ---
    let mut fam_ok = 0usize;
    let fam_n = gold.families.len();
    if fam_n > 0 {
        let mut rep_groups: Vec<FxHashSet<usize>> = Vec::new();
        for grp in &res.families {
            let mut pids: FxHashSet<usize> = FxHashSet::default();
            for i in grp {
                for (pid, idxs) in &m_real {
                    if idxs.contains(i) {
                        pids.insert(*pid);
                    }
                }
            }
            if pids.len() >= 2 {
                rep_groups.push(pids);
            }
        }
        for (_, members) in &gold.families {
            let ms: FxHashSet<usize> = members.iter().copied().collect();
            if rep_groups.iter().any(|g| ms.iter().filter(|m| g.contains(m)).count() >= 2) {
                fam_ok += 1;
            }
        }
    }
    let fam_rec = fam_ok as f64 / 1.max(fam_n) as f64;

    // --- evidence coverage: did the kernel ever HOLD the evidence?  (retrieval, measured on the pool) ---
    let mut held: FxHashMap<i64, FxHashSet<i64>> = FxHashMap::default();
    for &k in &res.kernel_kos {
        held.entry(res.arena[k].anchor).or_default().insert(res.arena[k].pred);
    }
    let (mut cov2, mut cov3) = (0usize, 0usize);
    for p in &real {
        let got = held.get(&(p.anchor as i64)).map_or(0, |s| {
            p.preds.iter().collect::<HashSet<_>>().iter().filter(|&&&x| s.contains(&(x as i64))).count()
        });
        cov2 += usize::from(got >= 2);
        cov3 += usize::from(got >= 3);
    }

    // --- information loss / compression ---
    let mut facet_types: FxHashSet<(i64, i64)> = FxHashSet::default();
    for p in corpus.patterns.iter().filter(|p| p.real) {
        for fr in p.facet_records.iter().take(p.preds.len()) {
            for &r in fr {
                facet_types.insert((corpus.recs.pred[r] as i64, corpus.recs.anchor[r] as i64));
            }
        }
    }
    let kept = &res.retained;
    let facet_survival =
        facet_types.iter().filter(|t| kept.contains(t)).count() as f64 / 1.max(facet_types.len()) as f64;
    let all_types: FxHashSet<(i64, i64)> =
        (0..corpus.recs.len()).map(|r| (corpus.recs.pred[r] as i64, corpus.recs.anchor[r] as i64)).collect();
    let type_survival = all_types.iter().filter(|t| kept.contains(t)).count() as f64 / 1.max(all_types.len()) as f64;

    let meter = &res.meter;
    let in_tokens_if_raw = corpus.recs.len() as f64 * 24.0;
    let kernel_ctx = meter.levels.get("L5-kernel").map_or(0, |m| m.max_ctx);
    let compression = in_tokens_if_raw / 1.max(kernel_ctx) as f64;
    let tok_in = meter.tok_in();
    let tok_out = meter.tok_out();
    let cu = crate::py::json::round(meter.cu(), 2);
    let usd = crate::py::json::round(meter.usd(), 4);
    let wall = crate::py::json::round(meter.wall_seconds(), 2);
    let p50 = crate::py::json::round(meter.call_latency_pct(50.0), 4);
    let p95 = crate::py::json::round(meter.call_latency_pct(95.0), 4);

    // --- questions ---
    let qs = &res.questions;
    let nq = qs.len() as f64;
    let q_util = if qs.is_empty() { 0.0 } else { qs.iter().filter(|q| q.changed_conclusion).count() as f64 / nq };
    let q_evid = if qs.is_empty() { 0.0 } else { qs.iter().map(|q| q.n_new_evidence).sum::<usize>() as f64 / nq };
    let q_targeted = if qs.is_empty() { 0.0 } else { qs.iter().filter(|q| q.well_targeted).count() as f64 / nq };

    let n_correct = m_real.len();
    let n_recs = 1.max(corpus.recs.len()) as f64;
    let mut out: Metrics = IndexMap::new();
    out.insert("discovery_recall", rec);
    out.insert("discovery_recall_strict", m_real_strict.len() as f64 / 1.max(real.len()) as f64);
    out.insert("discovery_precision", prec);
    out.insert("discovery_f1", f1);
    out.insert("false_discovery_rate", fdr);
    out.insert("rare_signal_recall", rare_rec);
    out.insert("common_signal_recall", common_rec);
    out.insert("n_reported", n_rep as f64);
    out.insert("n_correct", n_correct as f64);
    out.insert("n_gold", real.len() as f64);
    out.insert("hallucination_rate", hall_rate);
    out.insert("unsupported_reports", unsupported as f64);
    out.insert("decoy_acceptance_all", m_dec.len() as f64 / 1.max(decoys.len()) as f64);
    out.insert("decoy_D1_entity_coincidence", dec_acc[1]);
    out.insert("decoy_D2_temporal_scramble", dec_acc[2]);
    out.insert("decoy_D3_near_miss_entity", dec_acc[3]);
    out.insert("decoy_D5_stale_chain", dec_acc[5]);
    out.insert("dup_inflation_reported", m_dup.len() as f64 / 1.max(dup_infl.len()) as f64);
    out.insert("dup_inflation_support_error", dup_err);
    out.insert("cross_branch_discovery_accuracy", cb_ok as f64 / 1.max(cb_n) as f64);
    out.insert("lineage_accuracy", lin_num as f64 / 1.max(lin_den) as f64);
    out.insert("provenance_preservation", prov_ok as f64 / 1.max(m_real.len()) as f64);
    out.insert("independent_evidence_error", if ind_err.is_empty() { 1.0 } else { math::mean(&ind_err) });
    out.insert(
        "independent_evidence_accuracy",
        if ind_err.is_empty() { 0.0 } else { 1.0 - math::mean(&ind_err) / 2.0 },
    );
    out.insert("evidence_precision", if ev_p.is_empty() { 0.0 } else { math::mean(&ev_p) });
    out.insert("evidence_recall", if ev_r.is_empty() { 0.0 } else { math::mean(&ev_r) });
    out.insert("contradiction_f1", c_f1);
    out.insert("contradiction_precision", c_prec);
    out.insert("contradiction_recall", c_rec);
    out.insert("top_support_accuracy", top_ok);
    out.insert("family_recall", fam_rec);
    out.insert("evidence_coverage_2links", cov2 as f64 / 1.max(real.len()) as f64);
    out.insert("evidence_coverage_3links", cov3 as f64 / 1.max(real.len()) as f64);
    out.insert("facet_type_survival", facet_survival);
    out.insert("information_loss", 1.0 - facet_survival);
    out.insert("claim_type_survival", type_survival);
    out.insert("compression_ratio", compression);
    out.insert("kernel_context_tokens", kernel_ctx as f64);
    out.insert("max_context_tokens", meter.max_ctx() as f64);
    out.insert("tokens_in", tok_in as f64);
    out.insert("tokens_out", tok_out as f64);
    out.insert("tokens_total", (tok_in + tok_out) as f64);
    out.insert("inference_calls", meter.calls() as f64);
    out.insert("compute_units", cu);
    out.insert("usd_estimate", usd);
    out.insert("wall_seconds", wall);
    out.insert("p50_call_s", p50);
    out.insert("p95_call_s", p95);
    out.insert("tau", tau);
    out.insert("n_questions", nq);
    out.insert("question_utility", q_util);
    out.insert("question_new_evidence", q_evid);
    out.insert("question_targeting", q_targeted);
    out.insert("privacy_propagated_records", res.propagated_records as f64);
    out.insert("privacy_exposure_fraction", res.propagated_records as f64 / n_recs);
    out.insert("raw_records_leaving_node", res.exposed_raw_records as f64);
    out.insert("raw_text_exposure_fraction", res.exposed_raw_records as f64 / n_recs);
    out.insert("claims_leaving_node", res.claims_leaving_node as f64);
    out.insert("claim_exposure_fraction", res.claims_leaving_node as f64 / n_recs);
    out.insert("sketch_entries_leaving_node", res.sketch_entries_leaving_node as f64);
    let rank_p = ranked_metrics(&ranked, &real, &stem, lenient, false);
    let rank_s = ranked_metrics(&ranked, &real, &stem, lenient, true);
    out.insert("average_precision", rank_p.ap);
    out.insert("average_precision_strict", rank_s.ap);
    out.insert("r_precision", rank_p.r_precision);
    out.insert("n_candidates", ranked.len() as f64);
    out.insert("random_ap_baseline", real.len() as f64 / 1.max(ranked.len()) as f64);
    out.insert("ap_lift_over_random", rank_p.ap / 1e-9f64.max(real.len() as f64 / 1.max(ranked.len()) as f64));
    for (j, (r, p)) in [
        ("recall_at_40", "precision_at_40"),
        ("recall_at_100", "precision_at_100"),
        ("recall_at_300", "precision_at_300"),
    ]
    .into_iter()
    .enumerate()
    {
        out.insert(r, rank_p.recall_at[j]);
        out.insert(p, rank_p.precision_at[j]);
    }
    out.insert("found_anywhere_in_register", rank_p.n_found / 1.max(real.len()) as f64);
    out.insert("cost_per_correct_discovery", cu / 1.max(n_correct) as f64);
    out.insert("usd_per_correct_discovery", usd / 1.max(n_correct) as f64);
    // executive answer accuracy: macro average over the executive question set
    let exec = [f1, top_ok, c_f1, fam_rec, cb_ok as f64 / 1.max(cb_n) as f64];
    out.insert("executive_answer_accuracy", math::mean(&exec));
    out
}

pub const HEADLINE: [&str; 19] = [
    "discovery_recall",
    "discovery_precision",
    "discovery_f1",
    "false_discovery_rate",
    "rare_signal_recall",
    "cross_branch_discovery_accuracy",
    "decoy_acceptance_all",
    "independent_evidence_accuracy",
    "lineage_accuracy",
    "contradiction_f1",
    "information_loss",
    "question_utility",
    "compute_units",
    "tokens_total",
    "inference_calls",
    "wall_seconds",
    "cost_per_correct_discovery",
    "privacy_exposure_fraction",
    "executive_answer_accuracy",
];
