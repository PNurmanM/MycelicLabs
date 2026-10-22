//! Every number the vNext deliverables quote, computed from raw artifacts (`vnext_data.py`).
//!
//! Nothing in here is typed in by hand: the OLD rows come from `artifacts/v1/` (the benchmark as it was before the
//! vNext work, archived by `final_rerun.sh`), the NEW rows from `artifacts/`, the change ledger from the paired
//! experiment files `artifacts/quick_<tag>.jsonl`, the funnels from `loss_funnel.jsonl` in both places, the ranker
//! evaluation from `ranker_eval_final.json` and the frozen configuration from `calibration.json`'s provenance
//! block.
//!
//! ```sh
//! mycelic-research vnext_data      # prints every table
//! ```
//!
//! Seed-keyed dicts are `IndexMap`s: Python dicts keep insertion order and a later row replaces an earlier one in
//! place, and the order in which per-seed values are summed decides the last bit of a mean.

use crate::analysis::{boot_ci, dedupe_default, load_path, sign_test};
use crate::loss_report::{flag, has, int, is_summary, loss_label, stage_label, stages_for, text};
use crate::np::math;
use crate::py::fmt;
use crate::py::value::{self, Row, py_str, truthy};
use crate::runner::art;
use indexmap::IndexMap;
use serde_json::Value;
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

/// `V1`: the archived pre-vNext artifacts.
pub fn v1() -> PathBuf {
    art().join("v1")
}

/// (key, label, higher is better)
pub const METRICS: [(&str, &str, bool); 14] = [
    ("found_anywhere_in_register", "found", true),
    ("evidence_coverage_2links", "evidence cov.", true),
    ("rare_signal_recall", "rare recall", true),
    ("common_signal_recall", "common recall", true),
    ("average_precision", "AP", true),
    ("false_discovery_rate", "FDR", false),
    ("decoy_acceptance_all", "decoy acc. (all)", false),
    ("decoy_D1_entity_coincidence", "decoy D1 entity", false),
    ("decoy_D2_temporal_scramble", "decoy D2 scramble", false),
    ("decoy_D3_near_miss_entity", "decoy D3 near-miss", false),
    ("decoy_D5_stale_chain", "decoy D5 stale", false),
    ("compute_units", "compute", false),
    ("inference_calls", "calls (metering)", false),
    ("max_context_tokens", "kernel prompt tokens (max)", false),
];
pub const CORE: [&str; 11] = [
    "found_anywhere_in_register",
    "evidence_coverage_2links",
    "rare_signal_recall",
    "average_precision",
    "false_discovery_rate",
    "decoy_acceptance_all",
    "decoy_D5_stale_chain",
    "decoy_D2_temporal_scramble",
    "compute_units",
    "inference_calls",
    "max_context_tokens",
];
pub const DECOYS: [&str; 4] =
    ["decoy_D1_entity_coincidence", "decoy_D2_temporal_scramble", "decoy_D3_near_miss_entity", "decoy_D5_stale_chain"];

/// Rows keyed by seed, in first-insertion order (a Python `{r["seed"]: r for r in rows ...}`).
pub type BySeed<'a> = IndexMap<i64, &'a Row>;

fn metric(r: &Row, k: &str) -> f64 {
    value::f(r, k)
}

fn label_of(k: &str) -> &'static str {
    METRICS.iter().find(|m| m.0 == k).map(|m| m.1).unwrap_or_else(|| panic!("no label for {k}"))
}

fn higher_better(k: &str) -> bool {
    METRICS.iter().find(|m| m.0 == k).map(|m| m.2).unwrap_or_else(|| panic!("no direction for {k}"))
}

// ---------------------------------------------------------------------------
// loading
// ---------------------------------------------------------------------------

/// `_rows(fname, base)`: a JSONL file under `base` (missing file -> no rows).
pub fn rows(fname: &str, base: &Path) -> Vec<Row> {
    load_path(&base.join(fname))
}

/// Headline rows: the first row per (arch, scale, seed), as the report reads them.
pub fn e1_rows(base: &Path) -> Vec<Row> {
    let mut r = rows("e1_baselines.jsonl", base);
    r.extend(rows("e1b_extra.jsonl", base));
    dedupe_default(r)
}

/// `by_seed(rows, arch, scale)`
pub fn by_seed<'a>(rows: &'a [Row], arch: &str, scale: i64) -> BySeed<'a> {
    let mut out = IndexMap::new();
    for r in rows.iter().filter(|r| text(r, "arch") == arch && int(r, "scale") == scale) {
        out.insert(int(r, "seed"), r);
    }
    out
}

/// The seeds two seed-keyed dicts share, ascending (`sorted(set(a) & set(b))`).
pub fn common_seeds<A, B>(a: &IndexMap<i64, A>, b: &IndexMap<i64, B>) -> Vec<i64> {
    let mut s: Vec<i64> = a.keys().filter(|k| b.contains_key(*k)).copied().collect();
    s.sort();
    s
}

fn fmt_metric(key: &str, v: f64) -> String {
    if matches!(key, "compute_units" | "inference_calls" | "max_context_tokens") {
        fmt::f(v, ".2e")
    } else {
        fmt::f(v, ".3f")
    }
}

/// (mean delta, ci text, wins text, verdict) for a paired difference vector.
pub fn verdict(d: &[f64], higher_better: bool) -> (String, String, String, String) {
    if d.is_empty() {
        return ("—".into(), "—".into(), "—".into(), "no rows".into());
    }
    let (m, mut lo, mut hi) = boot_ci(d);
    let (k, n, _) = sign_test(d);
    let wins = if higher_better { k } else { n - k };
    let v = if m.abs() < 1e-12 {
        "same"
    } else if (lo > 0.0 && higher_better) || (hi < 0.0 && !higher_better) {
        "**better**"
    } else if (hi < 0.0 && higher_better) || (lo > 0.0 && !higher_better) {
        "**worse**"
    } else {
        "inside noise"
    };
    let ci = if d.len() < 5 {
        lo = np_min(d);
        hi = np_max(d);
        format!("[{}, {}] (range)", fmt::f(lo, "+.3f"), fmt::f(hi, "+.3f"))
    } else {
        format!("[{}, {}]", fmt::f(lo, "+.3f"), fmt::f(hi, "+.3f"))
    };
    (fmt::f(m, "+.3f"), ci, format!("{wins}/{n}"), v.to_string())
}

/// `np.min`: NaN propagates.
fn np_min(d: &[f64]) -> f64 {
    d.iter().copied().fold(f64::INFINITY, |a, x| if x.is_nan() || a.is_nan() { f64::NAN } else { a.min(x) })
}

/// `np.max`: NaN propagates.
fn np_max(d: &[f64]) -> f64 {
    d.iter().copied().fold(f64::NEG_INFINITY, |a, x| if x.is_nan() || a.is_nan() { f64::NAN } else { a.max(x) })
}

/// `np.mean` of a metric over rows, in the order given (NaN when there are none, as numpy returns).
pub fn mean_of<'a>(rows: impl IntoIterator<Item = &'a Row>, k: &str) -> f64 {
    let v: Vec<f64> = rows.into_iter().map(|r| metric(r, k)).collect();
    math::mean(&v)
}

/// Per metric: (mean a, mean b, paired deltas b-a) over the common seeds.
pub fn paired_rows(a: &BySeed, b: &BySeed, keys: &[&str]) -> IndexMap<String, (f64, f64, Vec<f64>)> {
    let seeds = common_seeds(a, b);
    let mut out = IndexMap::new();
    for &k in keys {
        let va: Vec<f64> = seeds.iter().map(|s| metric(a[s], k)).collect();
        let vb: Vec<f64> = seeds.iter().map(|s| metric(b[s], k)).collect();
        let d: Vec<f64> = vb.iter().zip(&va).map(|(y, x)| y - x).collect();
        let ma = if seeds.is_empty() { f64::NAN } else { math::mean(&va) };
        let mb = if seeds.is_empty() { f64::NAN } else { math::mean(&vb) };
        out.insert(k.to_string(), (ma, mb, d));
    }
    out
}

// ---------------------------------------------------------------------------
// OLD vs NEW
// ---------------------------------------------------------------------------

/// The systems `old_new_table` compares by default: (name, OLD arch, NEW arch).
pub const OLD_NEW_ARCHS: [(&str, &str, &str); 4] = [
    ("Mycelic hierarchy", "H_mycelic_full", "H_mycelic_full"),
    ("A2 chunked long context", "A2_chunked_ctx", "A2_chunked_ctx"),
    ("B4 central triage", "B4_central_triage", "B4_central_triage"),
    ("Y oracle retrieval", "Y_oracle_retrieval", "Y_oracle_retrieval"),
];

/// OLD (artifacts/v1) against NEW (artifacts) on the same worlds, on the default systems and metrics.  `seeds`
/// restricts both sides, e.g. to the confirmation panel 5-9 that no development decision ever read.
pub fn old_new_table(scale: i64, seeds: Option<&[i64]>) -> String {
    old_new_table_with(scale, &OLD_NEW_ARCHS, &CORE, seeds)
}

/// `old_new_table(scale, archs, keys, seeds)`
pub fn old_new_table_with(scale: i64, archs: &[(&str, &str, &str)], keys: &[&str], seeds: Option<&[i64]>) -> String {
    let mut old = e1_rows(&v1());
    let mut new = e1_rows(&art());
    if let Some(sel) = seeds {
        old.retain(|r| sel.contains(&int(r, "seed")));
        new.retain(|r| sel.contains(&int(r, "seed")));
    }
    if old.is_empty() || new.is_empty() {
        return "_(old or new headline rows missing: run final_rerun.sh)_".to_string();
    }
    let mut lines = vec![
        "| system | metric | OLD | NEW | Δ | 95% CI | better on | verdict |".to_string(),
        "|---|---|---:|---:|---:|---|---:|---|".to_string(),
    ];
    for &(name, oa, na) in archs {
        let a = by_seed(&old, oa, scale);
        let b = by_seed(&new, na, scale);
        if a.is_empty() || b.is_empty() {
            lines.push(format!("| {name} | — | — | — | — | — | — | no rows |"));
            continue;
        }
        let pr = paired_rows(&a, &b, keys);
        for &k in keys {
            let (ma, mb, d) = &pr[k];
            let (mut dm, ci, wins, v) = verdict(d, higher_better(k));
            if matches!(k, "compute_units" | "inference_calls" | "max_context_tokens") {
                // `if ma`: a zero mean has no ratio (NaN is truthy in Python and prints as nan)
                dm = if *ma != 0.0 { format!("{}%", fmt::f((mb / ma - 1.0) * 100.0, "+.0f")) } else { "—".into() };
            }
            lines.push(format!(
                "| {name} | {} | {} | {} | {dm} | {ci} | {wins} | {v} |",
                label_of(k),
                fmt_metric(k, *ma),
                fmt_metric(k, *mb)
            ));
        }
    }
    let first_old = by_seed(&old, archs[0].1, scale);
    let first_new = by_seed(&new, archs[0].2, scale);
    let (n_old, n_new) = (first_old.len(), first_new.len());
    // the register size is read from the rows (n_candidates = entries kept), not typed
    let caps = |m: &BySeed| -> String {
        let c: BTreeSet<i64> = m.values().map(|r| metric(r, "n_candidates") as i64).collect();
        format!("[{}]", c.iter().map(i64::to_string).collect::<Vec<_>>().join(", "))
    };
    lines.push(String::new());
    lines.push(format!(
        "paired seeds at {}: {} (old rows {n_old}, new rows {n_new}); register entries kept: OLD {}, NEW {} (the cap \
         is min(6000, max(600, n_entities)) in both).",
        fmt::i(scale, ","),
        n_old.min(n_new),
        caps(&first_old),
        caps(&first_new)
    ));
    lines.join("\n")
}

/// `H_mycelic_prev` inside the NEW suite against the archived OLD H rows: the v1 configuration should reproduce,
/// and this is the check.
pub fn prev_reproduction_table(scale: i64) -> String {
    let old_rows = e1_rows(&v1());
    let new_rows = e1_rows(&art());
    let old = by_seed(&old_rows, "H_mycelic_full", scale);
    let new = by_seed(&new_rows, "H_mycelic_prev", scale);
    if old.is_empty() || new.is_empty() {
        return "_(no H_mycelic_prev rows yet)_".to_string();
    }
    let pr = paired_rows(&old, &new, &CORE);
    let mut lines = vec![
        "| metric | OLD H_mycelic_full | NEW H_mycelic_prev | max abs. Δ over seeds |".to_string(),
        "|---|---:|---:|---:|".to_string(),
    ];
    for k in CORE {
        let (ma, mb, d) = &pr[k];
        let abs: Vec<f64> = d.iter().map(|x| x.abs()).collect();
        let mx = if d.is_empty() { f64::NAN } else { np_max(&abs) };
        lines.push(format!(
            "| {} | {} | {} | {} |",
            label_of(k),
            fmt_metric(k, *ma),
            fmt_metric(k, *mb),
            fmt_metric(k, mx)
        ));
    }
    lines.join("\n")
}

/// The controls `gap_table` sets against the hierarchy by default.
pub const GAP_CONTROLS: [&str; 5] =
    ["A2_chunked_ctx", "B4_central_triage", "Y_oracle_retrieval", "H_mycelic_lean", "H_mycelic_prev"];

/// NEW hierarchy against every control on the same NEW worlds.
pub fn gap_table(scale: i64) -> String {
    gap_table_with(scale, "H_mycelic_full", &GAP_CONTROLS)
}

/// `gap_table(scale, hier, controls)`
pub fn gap_table_with(scale: i64, hier: &str, controls: &[&str]) -> String {
    let new = e1_rows(&art());
    let h = by_seed(&new, hier, scale);
    if h.is_empty() {
        return "_(no new rows yet)_".to_string();
    }
    let mut lines = vec![
        "| system | found | evidence cov. | rare recall | AP | decoy acc. | D5 stale | compute | calls | kernel prompt \
         (max tokens) | found gap to H | compute ratio to H |"
            .to_string(),
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|".to_string(),
    ];
    let row = |name: &str, d: &BySeed, lines: &mut Vec<String>| {
        let mut seeds: Vec<i64> = d.keys().copied().collect();
        seeds.sort();
        let m = |k: &str| mean_of(seeds.iter().map(|s| d[s]), k);
        let common = common_seeds(d, &h);
        let gap = if common.is_empty() {
            f64::NAN
        } else {
            let v: Vec<f64> = common
                .iter()
                .map(|s| metric(d[s], "found_anywhere_in_register") - metric(h[s], "found_anywhere_in_register"))
                .collect();
            math::mean(&v)
        };
        let cr = if seeds.iter().any(|s| h.contains_key(s)) {
            let hc = mean_of(seeds.iter().filter(|s| h.contains_key(*s)).map(|s| h[s]), "compute_units");
            m("compute_units") / value::max2(1e-9, hc)
        } else {
            f64::NAN
        };
        lines.push(format!(
            "| {name} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {}x |",
            fmt::f(m("found_anywhere_in_register"), ".3f"),
            fmt::f(m("evidence_coverage_2links"), ".3f"),
            fmt::f(m("rare_signal_recall"), ".3f"),
            fmt::f(m("average_precision"), ".3f"),
            fmt::f(m("decoy_acceptance_all"), ".3f"),
            fmt::f(m("decoy_D5_stale_chain"), ".3f"),
            fmt::f(m("compute_units"), ".2e"),
            fmt::f(m("inference_calls"), ".2e"),
            fmt::f(m("max_context_tokens"), ".2e"),
            fmt::f(gap, "+.3f"),
            fmt::f(cr, ".2f")
        ));
    };
    row(hier, &h, &mut lines);
    for c in controls {
        let d = by_seed(&new, c, scale);
        if !d.is_empty() {
            row(c, &d, &mut lines);
        }
    }
    lines.join("\n")
}

// ---------------------------------------------------------------------------
// change ledger from the paired experiment files
// ---------------------------------------------------------------------------

/// `artifacts/quick_<tag>.jsonl`
pub fn quick(tag: &str) -> Vec<Row> {
    rows(&format!("quick_{tag}.jsonl"), &art())
}

/// How one arm of a paired file is selected (`_side`'s `sel`).
#[derive(Clone, Copy, Debug)]
pub enum Sel<'a> {
    /// row parity: 0 = base, 1 = variant (quick_paired files interleave the two under one label)
    Parity(usize),
    /// the arm_paired label in the `arch` column
    Label(&'a str),
}

/// One arm of a paired file.  quick_paired files interleave base/variant rows under one label when the variant is
/// a ranker switch, so an int selects by row parity (0 = base, 1 = variant); a string selects the arm_paired
/// label.  A tag 'a|b' names two files, one per side.
pub fn side(tag: &str, sel: Sel) -> IndexMap<i64, Row> {
    let mut out = IndexMap::new();
    for (i, r) in quick(tag).into_iter().enumerate() {
        let keep = match sel {
            Sel::Parity(p) => i % 2 == p,
            Sel::Label(l) => text(&r, "arch") == l,
        };
        if keep {
            out.insert(int(&r, "seed"), r);
        }
    }
    out
}

/// `_arms(tag, base, variant)`: both sides of a paired file; `None` for the base means interleaved (0, 1).
pub fn arms(tag: &str, base: Option<Sel>, variant: Option<Sel>) -> (IndexMap<i64, Row>, IndexMap<i64, Row>) {
    let (ta, tb) = match tag.split_once('|') {
        Some((a, b)) if !b.is_empty() => (a, b),
        Some((a, _)) => (a, a),
        None => (tag, tag),
    };
    let (base, variant) = match base {
        None => (Sel::Parity(0), Sel::Parity(1)),
        Some(b) => (b, variant.expect("a labelled base needs a variant")),
    };
    (side(ta, base), side(tb, variant))
}

/// Borrowed view of an owned seed map.
pub fn borrow_map(m: &IndexMap<i64, Row>) -> BySeed<'_> {
    m.iter().map(|(k, v)| (*k, v)).collect()
}

/// (tag, base selector or None for interleaved, variant selector, scale, change, status, evidence seeds)
pub type LedgerRow =
    (&'static str, Option<Sel<'static>>, Option<Sel<'static>>, i64, &'static str, &'static str, &'static str);

pub const LEDGER: [LedgerRow; 24] = {
    use Sel::{Label as L, Parity as P};
    [
        ("rk_H", None, None, 10_000, "ranker v1: learned order inside the hand gate", "superseded", "eval 0-4"),
        ("rk2_H", None, None, 10_000, "ranker v2: learned top-K, anchor-context features", "superseded", "eval 0-4"),
        ("v3_H", None, None, 10_000, "ranker v3 + question_frac 0.65 + batched descent", "ACCEPTED", "eval 0-4"),
        ("seldw_H", Some(L("v3")), Some(L("v3dw")), 10_000, "decoy-weighted ranker selection", "rejected", "eval 0-4"),
        ("qf_v3", Some(L("qf0.65")), Some(L("qf0.80")), 10_000, "question_frac 0.80 (vs 0.65)", "rejected", "eval 0-4"),
        ("qf_v3", Some(L("qf0.65")), Some(L("qf1.00")), 10_000, "question_frac 1.00 (vs 0.65)", "rejected", "eval 0-4"),
        (
            "rx_base065|rx_reextract",
            Some(P(1)),
            Some(P(1)),
            10_000,
            "local re-extraction at the user (vs qf 0.65 batched)",
            "rejected",
            "eval 0-4",
        ),
        (
            "rk2_qf0.65|rx_base065",
            Some(P(1)),
            Some(P(1)),
            10_000,
            "batched descent metering alone (ranker v2, qf 0.65; same decisions)",
            "ACCEPTED",
            "eval 0-4",
        ),
        (
            "refit_mod",
            Some(L("v3")),
            Some(L("mod_refit")),
            10_000,
            "modal link timing (+ refitted ranker)",
            "rejected",
            "eval 0-4",
        ),
        (
            "refit_hyb",
            Some(L("v3")),
            Some(L("hyb_refit")),
            10_000,
            "hybrid link timing (+ refitted ranker)",
            "ACCEPTED",
            "eval 0-4",
        ),
        (
            "refit_hyb",
            Some(L("v3")),
            Some(L("hyb_refit_lean")),
            10_000,
            "hybrid + chain-scoped questions (H_mycelic_lean)",
            "Pareto arm",
            "eval 0-4",
        ),
        (
            "v50_v3",
            Some(L("H_old")),
            Some(L("H_v3_qf0.65")),
            50_000,
            "ranker v3 + qf 0.65 + batched, 50k",
            "ACCEPTED",
            "eval 0-2",
        ),
        (
            "v50_v3",
            Some(L("H_v3_qf0.65")),
            Some(L("H_v3_qf0.80")),
            50_000,
            "question_frac 0.80 at 50k",
            "rejected",
            "eval 0-2",
        ),
        (
            "v50_v3",
            Some(L("H_v3_qf0.65")),
            Some(L("H_v3_qf0.65_rx")),
            50_000,
            "local re-extraction at 50k",
            "rejected",
            "eval 0-2",
        ),
        (
            "v50_hyb",
            Some(L("v3")),
            Some(L("hyb")),
            50_000,
            "hybrid link timing at 50k (+ refitted ranker)",
            "validation",
            "eval 0-2",
        ),
        (
            "v50_hyb",
            Some(L("v3")),
            Some(L("hyb_lean")),
            50_000,
            "hybrid + chain-scoped questions at 50k",
            "Pareto arm",
            "eval 0-2",
        ),
        (
            "prev_batched",
            Some(L("prev")),
            Some(L("prev_batched")),
            10_000,
            "v1 hierarchy re-metered with batched descent (identical decisions)",
            "metering control",
            "eval 0-4",
        ),
        (
            "prev_batched50",
            Some(L("prev")),
            Some(L("prev_batched")),
            50_000,
            "v1 hierarchy re-metered with batched descent at 50k",
            "metering control",
            "eval 0-2",
        ),
        ("diag_reg", None, None, 10_000, "unbounded register (DIAGNOSTIC ONLY, not a fix)", "diagnostic", "eval 0-4"),
        (
            "diag_both",
            None,
            None,
            10_000,
            "unbounded register + full question budget (DIAGNOSTIC)",
            "diagnostic",
            "eval 0-4",
        ),
        (
            "cal_screen",
            Some(L("base")),
            Some(L("span2")),
            10_000,
            "chain-scoped questions alone (calibration seeds)",
            "screen",
            "cal 500-502",
        ),
        (
            "stale_cal",
            Some(L("hyb")),
            Some(L("hyb_strong")),
            10_000,
            "cluster-aware staleness gate (strong positives only), calibration seeds",
            "screen",
            "cal 500-502",
        ),
        (
            "strong_fresh",
            Some(L("hyb")),
            Some(L("strong_hybrk")),
            10_000,
            "strong staleness gate, frozen ranker, FRESH panel",
            "fresh-panel read",
            "eval 10-14",
        ),
        (
            "strong_fresh",
            Some(L("hyb")),
            Some(L("strong_refit")),
            10_000,
            "strong staleness gate + refitted ranker, FRESH panel",
            "fresh-panel read",
            "eval 10-14",
        ),
    ]
};

/// The change ledger: every paired experiment, its deltas, its cost and its status.
pub fn ledger_table() -> String {
    let mut lines = vec![
        "| change | scale | seeds | Δ found | Δ rare | Δ cov. | Δ decoy all | Δ D1 | Δ D2 | Δ D3 | Δ D5 stale | compute \
         | calls (metering) | status |"
            .to_string(),
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|".to_string(),
    ];
    for (tag, base, variant, scale, what, status, seeds) in LEDGER {
        let (a, b) = arms(tag, base, variant);
        if common_seeds(&a, &b).is_empty() {
            lines.push(format!(
                "| {what} | {} | {seeds} | — | — | — | — | — | — | — | — | — | — | {status} (no rows) |",
                fmt::i(scale, ",")
            ));
            continue;
        }
        let mut keys = vec![
            "found_anywhere_in_register",
            "rare_signal_recall",
            "evidence_coverage_2links",
            "decoy_acceptance_all",
            "compute_units",
            "inference_calls",
        ];
        keys.extend(DECOYS);
        let pr = paired_rows(&borrow_map(&a), &borrow_map(&b), &keys);
        let f = &pr["found_anywhere_in_register"];
        let (k, n, _) = sign_test(&f.2);
        let cu = &pr["compute_units"];
        let ca = &pr["inference_calls"];
        let dmean = |key: &str| fmt::f(math::mean(&pr[key].2), "+.3f");
        let decoys: Vec<String> = DECOYS.iter().map(|d| dmean(d)).collect();
        lines.push(format!(
            "| {what} | {} | {seeds} | {} ({k}/{n} better) | {} | {} | {} | {} | {}% | {}% | {status} |",
            fmt::i(scale, ","),
            fmt::f(math::mean(&f.2), "+.3f"),
            dmean("rare_signal_recall"),
            dmean("evidence_coverage_2links"),
            dmean("decoy_acceptance_all"),
            decoys.join(" | "),
            fmt::f((cu.1 / cu.0 - 1.0) * 100.0, "+.0f"),
            fmt::f((ca.1 / ca.0 - 1.0) * 100.0, "+.0f")
        ));
    }
    lines.join("\n")
}

/// Rows of a multi-arm file grouped by arm label, then keyed by seed (`by.setdefault(r["arch"], {})[seed] = r`).
pub fn by_arm(rows: &[Row]) -> IndexMap<String, BySeed<'_>> {
    let mut by: IndexMap<String, BySeed> = IndexMap::new();
    for r in rows {
        by.entry(text(r, "arch").to_string()).or_default().insert(int(r, "seed"), r);
    }
    by
}

fn sorted_ints(it: impl IntoIterator<Item = i64>) -> Vec<i64> {
    it.into_iter().collect::<BTreeSet<_>>().into_iter().collect()
}

fn comma_ints(v: &[i64]) -> String {
    v.iter().map(|x| fmt::i(*x, ",")).collect::<Vec<_>>().join(", ")
}

/// A multi-arm paired file as one row per arm (mean over its seeds, with the paired delta against the base arm),
/// used for the budget sweeps.
pub fn sweep_table(tag: &str, base_label: &str) -> String {
    let rows = quick(tag);
    if rows.is_empty() {
        return "_(not run)_".to_string();
    }
    let by = by_arm(&rows);
    let empty = BySeed::new();
    let base = by.get(base_label).unwrap_or(&empty);
    let seeds = sorted_ints(rows.iter().map(|r| int(r, "seed")));
    let scale = sorted_ints(rows.iter().map(|r| int(r, "scale")));
    let mut lines = vec![
        format!(
            "seeds {}–{} at {} ({} panel), paired Δ against {base_label}:",
            seeds[0],
            seeds[seeds.len() - 1],
            comma_ints(&scale),
            if seeds[0] >= 500 { "calibration" } else { "evaluation" }
        ),
        String::new(),
        "| arm | seeds | found | Δ found | rare recall | evidence cov. | AP | decoy acc. | compute | calls |"
            .to_string(),
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|".to_string(),
    ];
    for (label, d) in &by {
        let mut s: Vec<i64> = d.keys().copied().collect();
        s.sort();
        let m = |k: &str| mean_of(s.iter().map(|x| d[x]), k);
        let common = common_seeds(d, base);
        let dl = if !common.is_empty() && label != base_label {
            let v: Vec<f64> = common
                .iter()
                .map(|x| metric(d[x], "found_anywhere_in_register") - metric(base[x], "found_anywhere_in_register"))
                .collect();
            math::mean(&v)
        } else {
            0.0
        };
        lines.push(format!(
            "| {label} | {} | {} | {} | {} | {} | {} | {} | {} | {} |",
            s.len(),
            fmt::f(m("found_anywhere_in_register"), ".3f"),
            fmt::f(dl, "+.3f"),
            fmt::f(m("rare_signal_recall"), ".3f"),
            fmt::f(m("evidence_coverage_2links"), ".3f"),
            fmt::f(m("average_precision"), ".3f"),
            fmt::f(m("decoy_acceptance_all"), ".3f"),
            fmt::f(m("compute_units"), ".2e"),
            fmt::f(m("inference_calls"), ".2e")
        ));
    }
    lines.join("\n")
}

/// The levels `fresh_panel_numbers` reports.
#[derive(Clone, Copy, Debug)]
pub struct FreshPanel {
    pub found: f64,
    pub rare: f64,
    pub cov: f64,
    pub decoy: f64,
    pub d5: f64,
    pub cu: f64,
    pub n: usize,
}

/// The frozen configuration's one read on seeds 10-14 (the base arm of the staleness-gate test), as levels.
/// `None` where Python returned an empty dict.
pub fn fresh_panel_numbers(tag: &str, arm: &str) -> Option<FreshPanel> {
    let all = quick(tag);
    let rows: Vec<&Row> = all.iter().filter(|r| text(r, "arch") == arm).collect();
    if rows.is_empty() {
        return None;
    }
    let m = |k: &str| mean_of(rows.iter().copied(), k);
    Some(FreshPanel {
        found: m("found_anywhere_in_register"),
        rare: m("rare_signal_recall"),
        cov: m("evidence_coverage_2links"),
        decoy: m("decoy_acceptance_all"),
        d5: m("decoy_D5_stale_chain"),
        cu: m("compute_units"),
        n: rows.len(),
    })
}

/// Compute cost per accepted change at 10k, in the order they were stacked.
pub fn compute_per_accepted_change() -> String {
    // (name, tag, arm label, row parity when unlabelled, chained to the previous row)
    let steps: [(&str, &str, Option<&str>, usize, bool); 4] = [
        ("v1 hierarchy (hand ranker, qf 0.25), metered unbatched as benchmarked", "v3_H", None, 0, false),
        (
            "v1 hierarchy re-metered with batched descent (identical decisions) — the like-for-like base",
            "prev_batched",
            Some("prev_batched"),
            0,
            true,
        ),
        ("+ ranker v3, qf 0.65, batched descent", "v3_H", None, 1, true),
        ("+ hybrid link timing (refitted ranker)", "refit_hyb", Some("hyb_refit"), 0, true),
    ];
    let mut lines = vec![
        "| configuration | found | rare recall | compute | calls (metering) | kernel prompt tokens | Δ found per +10% \
         compute (vs previous row) |"
            .to_string(),
        "|---|---:|---:|---:|---:|---:|---:|".to_string(),
    ];
    let mut prev: Option<(f64, f64)> = None;
    for (name, tag, label, parity, chain) in steps {
        let rows = quick(tag);
        let sel: Vec<&Row> = match label {
            None => rows.iter().enumerate().filter(|(i, _)| i % 2 == parity).map(|(_, r)| r).collect(),
            Some(l) => rows.iter().filter(|r| text(r, "arch") == l).collect(),
        };
        if sel.is_empty() {
            lines.push(format!("| {name} | — | — | — | — | — | — |"));
            continue;
        }
        let m = |k: &str| mean_of(sel.iter().copied(), k);
        let mut eff = "—".to_string();
        if let (true, Some(p)) = (chain, prev) {
            if m("compute_units") > p.1 * 1.02 {
                eff =
                    fmt::f((m("found_anywhere_in_register") - p.0) / ((m("compute_units") / p.1 - 1.0) * 10.0), "+.3f");
            } else {
                eff = format!(
                    "{} at {}% compute",
                    fmt::f(m("found_anywhere_in_register") - p.0, "+.3f"),
                    fmt::f((m("compute_units") / p.1 - 1.0) * 100.0, "+.1f")
                );
            }
        }
        lines.push(format!(
            "| {name} | {} | {} | {} | {} | {} | {eff} |",
            fmt::f(m("found_anywhere_in_register"), ".3f"),
            fmt::f(m("rare_signal_recall"), ".3f"),
            fmt::f(m("compute_units"), ".2e"),
            fmt::f(m("inference_calls"), ".2e"),
            fmt::f(m("max_context_tokens"), ".2e")
        ));
        prev = Some((m("found_anywhere_in_register"), m("compute_units")));
    }
    lines.push(String::new());
    lines.push(
        "The first row is the v1 base as it was benchmarked (unbatched metering); the second is the same decisions \
         re-metered with batched descent, which is the base every later row should be read against. Batching is a \
         metering convention: it changes no decision and no per-record token."
            .to_string(),
    );
    lines.join("\n")
}

// ---------------------------------------------------------------------------
// funnels, old and new
// ---------------------------------------------------------------------------

/// `_funnel(base)`: `loss_funnel.jsonl` under `base`.
pub fn funnel(base: &Path) -> Vec<Row> {
    rows("loss_funnel.jsonl", base)
}

fn funnel_arch(base: &Path, scale: i64, arch: &str) -> Vec<Row> {
    funnel(base).into_iter().filter(|r| !is_summary(r) && int(r, "scale") == scale && text(r, "arch") == arch).collect()
}

fn rare_eq(r: &Row, rare: bool) -> bool {
    value::py_eq(&r["rare"], &Value::Bool(rare))
}

/// Stage survival of the hierarchy, OLD funnel against NEW, all / rare / common.
pub fn funnel_compare(scale: i64, arch: &str) -> String {
    let old = funnel_arch(&v1(), scale, arch);
    let new = funnel_arch(&art(), scale, arch);
    if old.is_empty() || new.is_empty() {
        return "_(funnel rows missing for one side)_".to_string();
    }
    let stages = stages_for(arch, Some(&new));
    let surv = |rows: &[Row], s: &str, rare: Option<bool>| -> f64 {
        let sub: Vec<&Row> = rows.iter().filter(|r| has(r, s) && rare.is_none_or(|x| rare_eq(r, x))).collect();
        if sub.is_empty() { f64::NAN } else { sub.iter().filter(|r| flag(r, s)).count() as f64 / sub.len() as f64 }
    };
    let mut lines = vec![
        "| stage | OLD all | NEW all | OLD rare | NEW rare | OLD common | NEW common |".to_string(),
        "|---|---:|---:|---:|---:|---:|---:|".to_string(),
    ];
    for s in stages {
        lines.push(format!(
            "| {} | {} | {} | {} | {} | {} | {} |",
            stage_label(s).unwrap(),
            fmt::f(surv(&old, s, None), ".3f"),
            fmt::f(surv(&new, s, None), ".3f"),
            fmt::f(surv(&old, s, Some(true)), ".3f"),
            fmt::f(surv(&new, s, Some(true)), ".3f"),
            fmt::f(surv(&old, s, Some(false)), ".3f"),
            fmt::f(surv(&new, s, Some(false)), ".3f")
        ));
    }
    let rare = |rows: &[Row]| rows.iter().filter(|r| flag(r, "rare")).count();
    lines.push(format!(
        "| patterns | {} | {} | {} | {} | {} | {} |",
        old.len(),
        new.len(),
        rare(&old),
        rare(&new),
        old.len() - rare(&old),
        new.len() - rare(&new)
    ));
    lines.join("\n")
}

/// Where the hierarchy's patterns died (terminal stage), OLD funnel against NEW.
pub fn terminal_loss_compare(scale: i64, arch: &str) -> String {
    let old = funnel_arch(&v1(), scale, arch);
    let new = funnel_arch(&art(), scale, arch);
    if old.is_empty() || new.is_empty() {
        return "_(funnel rows missing for one side)_".to_string();
    }
    let mut stages = stages_for(arch, Some(&new));
    stages.push("reported");
    let mut lines = vec![
        "| the pattern died at | OLD | NEW | OLD rare | NEW rare |".to_string(),
        "|---|---:|---:|---:|---:|".into(),
    ];
    for s in stages {
        let key = if s == "reported" { "" } else { s };
        let lab = loss_label(s).or_else(|| stage_label(s)).unwrap_or("**reported**");
        let count = |rows: &[Row], rare_only: bool| {
            rows.iter().filter(|r| text(r, "loss_stage") == key && (!rare_only || flag(r, "rare"))).count()
        };
        let (oc, nc) = (count(&old, false), count(&new, false));
        lines.push(format!(
            "| {lab} | {oc} ({}) | {nc} ({}) | {} | {} |",
            fmt::f(oc as f64 / old.len() as f64, ".0%"),
            fmt::f(nc as f64 / new.len() as f64, ".0%"),
            count(&old, true),
            count(&new, true)
        ));
    }
    lines.join("\n")
}

/// (stage label, count, total) of the most common terminal loss in the NEW funnel.
pub fn largest_remaining_loss(scale: i64, arch: &str) -> (String, usize, usize) {
    let new = funnel_arch(&art(), scale, arch);
    if new.is_empty() {
        return ("—".into(), 0, 0);
    }
    let mut counts: IndexMap<String, usize> = IndexMap::new();
    for r in &new {
        let s = text(r, "loss_stage");
        if !s.is_empty() {
            *counts.entry(s.to_string()).or_default() += 1;
        }
    }
    if counts.is_empty() {
        return ("none".into(), 0, new.len());
    }
    let (s, c) = value::max_by(counts.iter(), |e| *e.1 as f64).unwrap();
    let lab = loss_label(s).or_else(|| stage_label(s)).map(str::to_string).unwrap_or_else(|| s.clone());
    (lab, *c, new.len())
}

// ---------------------------------------------------------------------------
// ranker evaluation and the frozen configuration
// ---------------------------------------------------------------------------

fn read_json(p: &Path) -> Value {
    let text = std::fs::read_to_string(p).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("{}: {e}", p.display()))
}

/// The ranker's out-of-sample evaluation per dump and system (`ranker_eval_final.json`).
pub fn ranker_eval_table() -> String {
    let p = art().join("ranker_eval_final.json");
    if !p.exists() {
        return "_(ranker_eval_final.json missing: run final_rerun.sh)_".to_string();
    }
    let ev = read_json(&p);
    let mut lines = vec![
        "| dump | system | gold patterns | matchable | found, hand ranker | found, learned (same count) | AUC hand | AUC \
         learned |"
            .to_string(),
        "|---|---|---:|---:|---:|---:|---:|---:|".to_string(),
    ];
    for (tag, r) in ev.as_object().expect("ranker_eval_final.json is an object") {
        let Some(per_arch) = r.get("per_arch").and_then(Value::as_object) else { continue };
        for (arch, pa) in per_arch {
            lines.push(format!(
                "| {tag} | {arch} | {} | {} | {} | {} | {} | {} |",
                py_str(&pa["gold_patterns"]),
                py_str(&pa["matchable_patterns"]),
                value::format(&pa["found_baseline"], ".3f"),
                value::format(&pa["found_learned_rank_only"], ".3f"),
                value::format(&pa["auc_conf"], ".3f"),
                value::format(&pa["auc_learned"], ".3f")
            ));
        }
    }
    lines.push(String::new());
    lines.push(
        "Evaluation rows are the held-out seeds 0–4 only; the logistic is refitted on the calibration seeds of the \
         same dump with the stored l2 and interaction choice, so the AUC and found columns are out-of-sample."
            .to_string(),
    );
    lines.join("\n")
}

/// `d.get(k, default)` of a JSON object, as `str()` prints it.
fn get_str(d: &Value, k: &str, default: &str) -> String {
    d.get(k).map(py_str).unwrap_or_else(|| default.to_string())
}

/// The frozen configuration, each knob with the file that justified it and the panel that file was read on.
pub fn frozen_config_table() -> String {
    let cal = read_json(&art().join("calibration.json"));
    let empty = Value::Object(Default::default());
    let prov = cal.get("vnext").unwrap_or(&empty);
    let r = cal.get("ranker").unwrap_or(&empty);
    let mut lines = vec![
        "| knob | frozen value | seeds the cited evidence was read on | evidence |".to_string(),
        "|---|---|---|---|".to_string(),
    ];
    for (k, v) in prov.as_object().expect("vnext provenance is an object") {
        // the "chosen on" column is read from the evidence files themselves: a knob whose only evidence is on the
        // evaluation panel says so
        let mut where_ = Vec::new();
        let evidence = get_str(v, "evidence", "");
        for ev in evidence.split(',').map(str::trim).filter(|e| !e.is_empty()) {
            let rs = rows(ev, &art());
            if rs.is_empty() {
                where_.push(format!("{ev}: (file missing)"));
                continue;
            }
            let sd = sorted_ints(rs.iter().map(|r| int(r, "seed")));
            let sc = sorted_ints(rs.iter().map(|r| int(r, "scale")));
            let panel = if sd.iter().all(|&x| x >= 500) {
                "calibration"
            } else if sd.iter().all(|&x| x < 5) {
                "evaluation (development panel)"
            } else {
                "mixed"
            };
            where_.push(format!("{ev}: seeds {}–{} at {} — {panel}", sd[0], sd[sd.len() - 1], comma_ints(&sc)));
        }
        let joined = if where_.is_empty() { "—".to_string() } else { where_.join("; ") };
        lines.push(format!("| `{k}` | `{}` | {joined} | `{evidence}` |", py_str(&v["value"])));
    }
    if truthy(r) {
        lines.push(format!(
            "| ranker | {}, l2 {}, interactions {}, {} candidates | seeds {} | `{}` |",
            get_str(r, "kind", "logistic"),
            get_str(r, "l2", "None"),
            get_str(r, "interactions", "None"),
            get_str(r, "n_train", "None"),
            get_str(r, "seeds", "None"),
            get_str(r, "source_file", "calibration.json")
        ));
    }
    if let Some(sel) = cal.get("ranker_archs").filter(|s| truthy(s)) {
        let names: Vec<String> = sel.as_array().expect("ranker_archs is a list").iter().map(py_str).collect();
        lines.push(format!("| ranker adopted by | {} | calibration seeds | ranker_arch_table |", names.join(", ")));
    }
    lines.join("\n")
}

/// The handful of numbers the executive text quotes, keyed `<side>_<metric>_<scale>`.  The `<side>_n_<scale>`
/// entries are seed counts (ints in the Python); no document prints them.
pub fn headline_numbers() -> IndexMap<String, f64> {
    let mut out: IndexMap<String, f64> = IndexMap::new();
    let old = e1_rows(&v1());
    let new = e1_rows(&art());
    for (tag, f) in [("prevb", "prev_batched"), ("prevb50", "prev_batched50")] {
        let all = quick(f);
        let rows: Vec<&Row> = all.iter().filter(|r| text(r, "arch") == "prev_batched").collect();
        let sc = if tag.ends_with("50") { 50_000 } else { 10_000 };
        for k in CORE {
            let v = if rows.is_empty() { f64::NAN } else { mean_of(rows.iter().copied(), k) };
            out.insert(format!("{tag}_{k}_{sc}"), v);
        }
    }
    let conf = 5..10;
    for (tag, rows, arch) in [
        ("old_conf", &old, "H_mycelic_full"),
        ("new_conf", &new, "H_mycelic_full"),
        ("a2_conf", &new, "A2_chunked_ctx"),
        ("a2_old_conf", &old, "A2_chunked_ctx"),
    ] {
        let d: Vec<&Row> =
            by_seed(rows, arch, 10_000).into_iter().filter(|(s, _)| conf.contains(s)).map(|e| e.1).collect();
        for k in CORE.iter().chain(&["decoy_acceptance_all"]) {
            let v = if d.is_empty() { f64::NAN } else { mean_of(d.iter().copied(), k) };
            out.insert(format!("{tag}_{k}_10000"), v);
        }
        out.insert(format!("{tag}_n_10000"), d.len() as f64);
    }
    for scale in [10_000, 50_000] {
        for (tag, rows, arch) in [
            ("old", &old, "H_mycelic_full"),
            ("new", &new, "H_mycelic_full"),
            ("a2", &new, "A2_chunked_ctx"),
            ("a2_old", &old, "A2_chunked_ctx"),
            ("lean", &new, "H_mycelic_lean"),
            ("prev", &new, "H_mycelic_prev"),
        ] {
            let d: Vec<&Row> = by_seed(rows, arch, scale).into_iter().filter(|(s, _)| *s < 5).map(|e| e.1).collect();
            for k in CORE.iter().chain(&["decoy_acceptance_all", "common_signal_recall"]) {
                let v = if d.is_empty() { f64::NAN } else { mean_of(d.iter().copied(), k) };
                out.insert(format!("{tag}_{k}_{scale}"), v);
            }
            out.insert(format!("{tag}_n_{scale}"), d.len() as f64);
        }
    }
    out
}

/// `python -m research.mycelic.vnext_data`: print every table.
pub fn main(_args: &[String]) -> anyhow::Result<()> {
    for sc in [10_000, 50_000] {
        println!("\n## OLD vs NEW at {}\n", fmt::i(sc, ","));
        println!("{}", old_new_table(sc, None));
        println!("\n## gap at {}\n", fmt::i(sc, ","));
        println!("{}", gap_table(sc));
        println!("\n## v1 reproduction at {}\n", fmt::i(sc, ","));
        println!("{}", prev_reproduction_table(sc));
        println!("\n## funnel {}\n", fmt::i(sc, ","));
        println!("{}", funnel_compare(sc, "H_mycelic_full"));
        println!("{}", terminal_loss_compare(sc, "H_mycelic_full"));
    }
    println!("\n## ledger\n");
    println!("{}", ledger_table());
    println!("\n## compute per accepted change\n");
    println!("{}", compute_per_accepted_change());
    println!("\n## ranker\n");
    println!("{}", ranker_eval_table());
    println!("\n## frozen\n");
    println!("{}", frozen_config_table());
    Ok(())
}
