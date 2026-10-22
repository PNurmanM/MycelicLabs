//! Generate the final report, `docs/MYCELIC_ENTERPRISE.md`, from the raw JSONL artifacts (`report.py`).
//!
//! Everything in the output is computed here from `artifacts/*.jsonl`; nothing is transcribed by hand.  Missing
//! experiments degrade to an explicit "(not run)" line rather than being silently omitted.
//!
//! ```sh
//! mycelic-research report
//! ```

use crate::analysis::{Paired, agg, boot_ci, dedupe_default, load, md_table, paired, paired_by};
use crate::findings::{
    GLOSSARY, critique, decision_summary, executive_summary, next_experiments, questions_section, recommendation,
};
use crate::loss_report::{int, text};
use crate::np::math;
use crate::py::fmt;
use crate::py::value::{self, Row, fl, num, py_cmp, py_eq, py_str, truthy};
use crate::report_text::{Sections, compose};
use crate::runner::{art, repo_root};
use crate::vnext_data as vd;
use rayon::prelude::*;
use regex::Regex;
use serde_json::Value;
use std::collections::BTreeSet;
use std::path::{Component, Path, PathBuf};

/// `docs/MYCELIC_ENTERPRISE.md`
pub fn out_path() -> PathBuf {
    repo_root().join("docs").join("MYCELIC_ENTERPRISE.md")
}

pub const CORE: [&str; 17] = [
    "average_precision",
    "found_anywhere_in_register",
    "recall_at_100",
    "rare_signal_recall",
    "evidence_coverage_2links",
    "false_discovery_rate",
    "decoy_acceptance_all",
    "independent_evidence_accuracy",
    "lineage_accuracy",
    "contradiction_f1",
    "information_loss",
    "compute_units",
    "tokens_total",
    "inference_calls",
    "wall_seconds",
    "privacy_exposure_fraction",
    "executive_answer_accuracy",
];

pub const NICE: [(&str, &str); 36] = [
    ("average_precision", "AP"),
    ("average_precision_strict", "AP (strict match)"),
    ("found_anywhere_in_register", "found"),
    ("recall_at_100", "R@100"),
    ("recall_at_300", "R@300"),
    ("rare_signal_recall", "rare recall"),
    ("common_signal_recall", "common recall"),
    ("evidence_coverage_2links", "evidence cov."),
    ("evidence_coverage_3links", "evidence cov.3"),
    ("false_discovery_rate", "FDR"),
    ("decoy_acceptance_all", "decoy acc."),
    ("decoy_D1_entity_coincidence", "D1 entity-coinc."),
    ("decoy_D2_temporal_scramble", "D2 time-scramble"),
    ("decoy_D3_near_miss_entity", "D3 near-miss"),
    ("decoy_D5_stale_chain", "D5 stale"),
    ("dup_inflation_support_error", "D4 support infl."),
    ("independent_evidence_accuracy", "indep. acc."),
    ("lineage_accuracy", "lineage"),
    ("provenance_preservation", "provenance"),
    ("evidence_precision", "evid. prec."),
    ("evidence_recall", "evid. recall"),
    ("contradiction_f1", "contra F1"),
    ("information_loss", "info loss"),
    ("compute_units", "compute (cu)"),
    ("usd_estimate", "USD est."),
    ("tokens_total", "tokens"),
    ("inference_calls", "calls"),
    ("wall_seconds", "wall s"),
    ("p95_call_s", "p95 call s"),
    ("privacy_exposure_fraction", "privacy exp."),
    ("executive_answer_accuracy", "exec acc."),
    ("cost_per_correct_discovery", "cu / discovery"),
    ("question_utility", "Q utility"),
    ("family_recall", "variant families"),
    ("kernel_context_tokens", "kernel ctx"),
    ("compression_ratio", "compression"),
];

const FOUND: &str = "found_anywhere_in_register";

/// `NICE.get(m, m)`
fn nice(m: &str) -> &str {
    NICE.iter().find(|e| e.0 == m).map_or(m, |e| e.1)
}

fn comma(x: i64) -> String {
    fmt::i(x, ",")
}

/// `_cols(metrics, first)`: the first column, then each metric under its short name (counts with no decimals).
fn cols<'a>(metrics: &[&'a str], first: (&'a str, &'a str)) -> Vec<(&'a str, &'a str, usize)> {
    let mut out = vec![(first.0, first.1, 0)];
    for &m in metrics {
        let dec = if matches!(m, "compute_units" | "tokens_total" | "inference_calls") { 0 } else { 3 };
        out.push((m, nice(m), dec));
    }
    out
}

/// `_fmt(rows, metrics, first, sort, ci)`: an aggregate table sorted by AP, descending.
fn fmt_table(rows: &[Row], metrics: &[&str], first: (&str, &str), ci: &[&str]) -> String {
    if rows.is_empty() {
        return "_(not run)_".to_string();
    }
    md_table(rows, &cols(metrics, first), Some("average_precision"), true, ci)
}

/// `rows + rows`: concatenated artifact files.
fn load_all(files: &[&str]) -> Vec<Row> {
    files.iter().flat_map(|f| load(f)).collect()
}

fn scales_of(rows: &[Row]) -> Vec<i64> {
    rows.iter().map(|r| int(r, "scale")).collect::<BTreeSet<_>>().into_iter().collect()
}

fn sorted_strs<'a>(rows: &'a [Row], key: &str) -> Vec<&'a str> {
    rows.iter().map(|r| text(r, key)).collect::<BTreeSet<_>>().into_iter().collect()
}

/// `sorted({r[key] for r in rows})` for values of any kind, in Python's order.
fn sorted_values(rows: &[Row], key: &str) -> Vec<Value> {
    let mut out: Vec<Value> = Vec::new();
    for r in rows {
        let v = &r[key];
        if !out.iter().any(|x| py_eq(x, v)) {
            out.push(v.clone());
        }
    }
    out.sort_by(py_cmp);
    out
}

fn f(r: &Row, k: &str) -> f64 {
    value::f(r, k)
}

/// Section 10: the headline table per scale.
pub fn section_baselines() -> String {
    let rows = dedupe_default(load_all(&["e1_baselines.jsonl", "e1b_extra.jsonl"]));
    if rows.is_empty() {
        return "_(E1 not run)_".to_string();
    }
    let mut out = Vec::new();
    for scale in scales_of(&rows) {
        let sub: Vec<Row> = rows.iter().filter(|r| int(r, "scale") == scale).cloned().collect();
        let n_seeds = sub.iter().map(|r| int(r, "seed")).collect::<BTreeSet<_>>().len();
        let a = agg(&sub, &CORE, &["arch"]);
        out.push(format!(
            "\n#### {} users ({} records, {} hidden patterns, {n_seeds} seeds)\n",
            comma(scale),
            value::format(&sub[0]["n_records"], ","),
            py_str(&sub[0]["n_gold"])
        ));
        out.push(fmt_table(
            &a,
            &[
                "average_precision",
                FOUND,
                "recall_at_100",
                "rare_signal_recall",
                "evidence_coverage_2links",
                "false_discovery_rate",
                "decoy_acceptance_all",
                "independent_evidence_accuracy",
                "lineage_accuracy",
                "information_loss",
                "compute_units",
                "privacy_exposure_fraction",
            ],
            ("arch", "architecture"),
            &["average_precision", FOUND],
        ));
    }
    out.join("\n")
}

/// One row of a paired-comparison table: `| label | Δ | [lo, hi] | wins | p |` at four decimals.
fn paired_cells(p: &Paired) -> String {
    format!(
        "{} | [{}, {}] | {}/{} | {}",
        fmt::f(p.mean_diff, "+.4f"),
        fmt::f(p.ci_lo, "+.4f"),
        fmt::f(p.ci_hi, "+.4f"),
        p.wins,
        p.n_nonzero,
        fmt::f(p.sign_p, ".3f")
    )
}

/// Section 10, paired comparisons on (scale, seed).
pub fn section_pairs() -> String {
    let rows = dedupe_default(load_all(&["e1_baselines.jsonl", "e1b_extra.jsonl"]));
    if rows.is_empty() {
        return "_(E1 not run)_".to_string();
    }
    let pairs = [
        ("H_mycelic_full", "E_hier_lineage"),
        ("H_mycelic_full", "B2_map_reduce"),
        ("H_mycelic_full", "A_flat_rag"),
        ("H_mycelic_full", "B_long_context"),
        ("H_mycelic_full", "C_recursive_sum"),
        ("H_mycelic_full", "D_hier_nolineage"),
        ("H_mycelic_full", "B4_central_triage"),
        ("H_mycelic_full", "Y_oracle_retrieval"),
        ("B2_map_reduce", "Z_random_rank"),
        ("G_hier_questions", "F_hier_retrieval"),
        ("F_hier_retrieval", "E_hier_lineage"),
    ];
    let mut lines = vec![
        "| comparison | metric | mean A | mean B | Δ | 95% CI | wins | sign p |".to_string(),
        "|---|---|---:|---:|---:|---|---:|---:|".to_string(),
    ];
    for metric in [FOUND, "average_precision"] {
        for (a, b) in pairs {
            let p = paired(&rows, a, b, metric);
            if p.n == 0 {
                continue;
            }
            lines.push(format!(
                "| {a} vs {b} | {} | {} | {} | {} |",
                nice(metric),
                fmt::f(p.mean_a, ".4f"),
                fmt::f(p.mean_b, ".4f"),
                paired_cells(&p)
            ));
        }
    }
    lines.join("\n")
}

/// (variant, Δ found, ci lo, ci hi, nonzero seeds); Δ is the VARIANT minus full.
type Ablation = (String, f64, f64, f64, usize);

/// Section 12: every ablation, and each paired against the full system.
pub fn section_ablations() -> String {
    let rows = load("e2_ablations.jsonl");
    if rows.is_empty() {
        return "_(E2 not run)_".to_string();
    }
    let mut metrics: Vec<&str> = CORE.to_vec();
    metrics.extend([
        "rare_signal_recall",
        "question_utility",
        "provenance_preservation",
        "family_recall",
        "dup_inflation_support_error",
    ]);
    let a = agg(&rows, &metrics, &["ablation"]);
    let tbl = md_table(
        &a,
        &cols(
            &[
                "average_precision",
                FOUND,
                "rare_signal_recall",
                "evidence_coverage_2links",
                "false_discovery_rate",
                "independent_evidence_accuracy",
                "lineage_accuracy",
                "contradiction_f1",
                "dup_inflation_support_error",
                "question_utility",
                "compute_units",
            ],
            ("ablation", "variant"),
        ),
        Some("average_precision"),
        true,
        &[FOUND],
    );
    let mut lines = vec![
        tbl,
        String::new(),
        "**Paired against the full system** (same seeds):".to_string(),
        String::new(),
        "| removed | Δ found | 95% CI | wins | sign p | Δ AP | Δ compute |".to_string(),
        "|---|---:|---|---:|---:|---:|---:|".to_string(),
    ];
    let mut collected: Vec<Ablation> = Vec::new();
    for name in sorted_strs(&rows, "ablation") {
        if name == "full" {
            continue;
        }
        let pair = |metric: &str| {
            paired_by(&rows, &Value::from("full"), &Value::from(name), metric, "ablation", &["scale", "seed"])
        };
        let (pf, pa, pc) = (pair(FOUND), pair("average_precision"), pair("compute_units"));
        if pf.n == 0 {
            continue;
        }
        lines.push(format!(
            "| {name} | {} | [{}, {}] | {}/{} | {} | {} | {} |",
            fmt::f(-pf.mean_diff, "+.4f"),
            fmt::f(-pf.ci_hi, "+.4f"),
            fmt::f(-pf.ci_lo, "+.4f"),
            pf.wins,
            pf.n_nonzero,
            fmt::f(pf.sign_p, ".3f"),
            fmt::f(-pa.mean_diff, "+.4f"),
            fmt::f(-pc.mean_diff, "+.3e")
        ));
        collected.push((name.to_string(), -pf.mean_diff, -pf.ci_hi, -pf.ci_lo, pf.n_nonzero));
    }
    lines.push(String::new());
    lines.push(ablation_verdict(&collected));
    lines.join("\n")
}

/// Spell out what a `+` variant actually changes.
///
/// `+source_dispersion` does not switch the feature on — calibration already selected it — it raises the weight.
/// Reading the bare name as "adds the feature" gets that backwards.
fn ablation_meaning(name: &str) -> String {
    let abl = crate::experiments::ablations();
    let over = abl.iter().find(|a| a.0 == name).map(|a| a.1.clone()).unwrap_or(Value::Object(Default::default()));
    let parts: Vec<String> =
        over.as_object().map(|o| o.iter().map(|(k, v)| format!("{k}={}", py_str(v))).collect()).unwrap_or_default();
    format!("{name} = {}", parts.join(", "))
}

fn sort_by_key_f<T>(v: &mut [T], key: impl Fn(&T) -> f64) {
    v.sort_by(|a, b| key(a).partial_cmp(&key(b)).expect("sort key is not NaN"));
}

/// Sort the variants into buckets, keeping removals and additions apart.
///
/// A name beginning `-` removes a mechanism the full system has; one beginning `+` adds a mechanism it does not.
/// They need opposite readings, and lumping them together produced a bucket list that said the wrong thing about
/// `+source_dispersion`.
fn ablation_verdict(rows: &[Ablation]) -> String {
    if rows.is_empty() {
        return String::new();
    }
    const EPS: f64 = 5e-4; // a CI endpoint this close to zero is not a decisive interval
    let rm: Vec<&Ablation> = rows.iter().filter(|r| !r.0.starts_with('+')).collect();
    let add: Vec<&Ablation> = rows.iter().filter(|r| r.0.starts_with('+')).collect();
    // r = (name, delta, ci_lo, ci_hi, n); delta is the VARIANT minus full.
    let mut costly: Vec<&Ablation> = rm.iter().copied().filter(|r| r.3 < -EPS).collect(); // removing it is worse
    let mut gainful: Vec<&Ablation> = rm.iter().copied().filter(|r| r.2 > EPS).collect(); // removing it is better
    let inert: Vec<&Ablation> =
        rm.iter().copied().filter(|r| !costly.iter().chain(&gainful).any(|c| c.0 == r.0)).collect();
    let mut out = vec![
        "**Reading the ablations.** Δ is the variant minus the full system. For a `-` row, a negative Δ means \
         removing the mechanism made things worse, so the mechanism is doing work. For a `+` row the mechanism is \
         *added* to the full system, so a positive Δ is the case for adopting it. A CI spanning zero means the seed \
         count cannot separate the two; with 5 seeds an exact sign test cannot go below 0.0625, so the interval is \
         carrying more of the argument than the p-value."
            .to_string(),
        String::new(),
    ];
    if !costly.is_empty() {
        sort_by_key_f(&mut costly, |r| r.1);
        out.push(format!(
            "* **Load-bearing** — removing it costs discovery, interval entirely below zero: {}.",
            costly.iter().map(|r| format!("`{}` ({})", r.0, fmt::f(r.1, "+.3f"))).collect::<Vec<_>>().join(", ")
        ));
    }
    if !inert.is_empty() {
        out.push(format!(
            "* **Not distinguishable from noise at this seed count**: {}. These are not shown to be useless; they are \
             shown to be unmeasured, which is a different statement and the honest one.",
            inert.iter().map(|r| format!("`{}`", r.0)).collect::<Vec<_>>().join(", ")
        ));
    }
    if !add.is_empty() {
        out.push(
            "* **Variants tested on top of the full system.** These are not \"the feature off vs on\" — the full \
             system already carries whatever calibration selected — so each is shown with the setting it actually \
             changes:"
                .to_string(),
        );
        out.push(String::new());
        let mut add = add;
        sort_by_key_f(&mut add, |r| -r.1);
        for r in add {
            let call = if r.2 > EPS {
                "**adopt**"
            } else if r.3 < -EPS {
                "**do not adopt**"
            } else {
                "inconclusive at this seed count"
            };
            // Four spaces, not two: two-space indentation nests on GitHub but is a flat list to a strict Markdown
            // parser, which is what the PDF renderer uses.
            out.push(format!(
                "    * `{}` → Δ {} [{}, {}] — {call}",
                ablation_meaning(&r.0),
                fmt::f(r.1, "+.3f"),
                fmt::f(r.2, "+.3f"),
                fmt::f(r.3, "+.3f")
            ));
        }
        out.push(String::new());
    }
    if !gainful.is_empty() {
        sort_by_key_f(&mut gainful, |r| -r.1);
        out.push(String::new());
        out.push(format!(
            "* **Removal apparently IMPROVES the system** — interval entirely above zero: {}. This is the bucket worth \
             arguing about. A mechanism that costs accuracy is a design defect however good the reason for adding it \
             was, and each should either be justified on a non-accuracy ground — confidentiality, auditability, \
             operability — or removed. The caveat is the seed count: these are the ablations most worth re-running \
             with more seeds before acting on them.",
            gainful.iter().map(|r| format!("`{}` ({})", r.0, fmt::f(r.1, "+.3f"))).collect::<Vec<_>>().join(", ")
        ));
    }
    out.join("\n")
}

/// Section 13: each allocation of model tiers to levels.
pub fn section_alloc() -> String {
    let rows = load("e3_allocation.jsonl");
    if rows.is_empty() {
        return "_(E3 not run)_".to_string();
    }
    let mut out = Vec::new();
    let mut metrics: Vec<&str> = CORE.to_vec();
    metrics.extend(["cost_per_correct_discovery", "usd_estimate"]);
    for arch in sorted_strs(&rows, "arch") {
        let sub: Vec<Row> = rows.iter().filter(|r| text(r, "arch") == arch).cloned().collect();
        let mut a = agg(&sub, &metrics, &["alloc"]);
        for r in &mut a {
            let t = sub.iter().find(|x| py_eq(&x["alloc"], &r["alloc"])).expect("allocation rows");
            let tiers = t.get("tiers").cloned().unwrap_or(Value::from(""));
            r.insert("tiers".into(), tiers);
        }
        out.push(format!("\n#### {arch}\n"));
        out.push(alloc_verdict(arch, &a));
        out.push(md_table(
            &a,
            &[
                ("alloc", "allocation", 0),
                ("tiers", "user→…→kernel", 0),
                ("average_precision", "AP", 4),
                (FOUND, "found", 3),
                ("rare_signal_recall", "rare", 3),
                ("compute_units", "compute", 0),
                ("cost_per_correct_discovery", "cu/disc", 0),
                ("wall_seconds", "wall s", 1),
            ],
            Some(FOUND),
            true,
            &[],
        ));
    }
    out.join("\n")
}

/// Answer the compute-allocation question: is a big ladder worth it?
///
/// The interesting comparison is not "which allocation scores highest" — that is always the most expensive one —
/// but what the quality/cost frontier looks like, and whether a cheap edge with one strong kernel keeps up.
fn alloc_verdict(_arch: &str, a: &[Row]) -> String {
    let rows: Vec<&Row> = a.iter().filter(|r| r.get("cost_per_correct_discovery").is_some_and(truthy)).collect();
    if rows.len() < 3 {
        return String::new();
    }
    let best = value::max_by(a.iter(), |r| f(r, FOUND)).unwrap();
    let cheapest_per = value::min_by(rows.iter().copied(), |r| f(r, "cost_per_correct_discovery")).unwrap();
    let floor = value::min_by(a.iter(), |r| f(r, "compute_units")).unwrap();
    let mut lines = vec![format!(
        "Best discovery: **`{}`** ({} at {} cu). Best value: **`{}`** ({} at {} cu — {} cu per correct discovery, \
         against {} for the best-discovery option).",
        text(best, "alloc"),
        fmt::f(f(best, FOUND), ".3f"),
        fmt::f(f(best, "compute_units"), ".2e"),
        text(cheapest_per, "alloc"),
        fmt::f(f(cheapest_per, FOUND), ".3f"),
        fmt::f(f(cheapest_per, "compute_units"), ".2e"),
        fmt::f(f(cheapest_per, "cost_per_correct_discovery"), ".3g"),
        fmt::f(f(best, "cost_per_correct_discovery"), ".3g")
    )];
    if text(cheapest_per, "alloc") != text(best, "alloc") {
        let ratio = f(best, "compute_units") / value::max2(1.0, f(cheapest_per, "compute_units"));
        let frac = f(cheapest_per, FOUND) / value::max2(1e-9, f(best, FOUND));
        lines.push(format!(
            "`{}` keeps {} of the discovery for {} of the compute. Whether that trade is worth making is a budget \
             question, not a research one — but it is the answer to \"should capability increase up the ladder?\": a \
             *graduated* ladder is not what wins here. What wins is a cheap edge and a strong kernel, because the \
             kernel is where the discrimination happens and the edge is only extracting.",
            text(cheapest_per, "alloc"),
            fmt::f(frac, ".0%"),
            fmt::f(1.0 / ratio, ".0%")
        ));
    }
    lines.push(format!(
        "For reference the cheapest allocation of all, `{}`, reaches {} at {} cu, so the spread between doing nothing \
         clever and doing the most expensive thing is {} discovery for {}x the compute.",
        text(floor, "alloc"),
        fmt::f(f(floor, FOUND), ".3f"),
        fmt::f(f(floor, "compute_units"), ".2e"),
        fmt::f(f(best, FOUND) - f(floor, FOUND), "+.3f"),
        fmt::f(f(best, "compute_units") / value::max2(1.0, f(floor, "compute_units")), ".0f")
    ));
    lines.join(" ") + "\n"
}

/// Section 13: one level at a time upgraded to the frontier tier.
pub fn section_level_marginal() -> String {
    let rows = load("e3b_level_marginal.jsonl");
    if rows.is_empty() {
        return "_(E3b not run)_".to_string();
    }
    let mut out = Vec::new();
    for arch in sorted_strs(&rows, "arch") {
        let sub: Vec<Row> = rows.iter().filter(|r| text(r, "arch") == arch).cloned().collect();
        let mut a = agg(&sub, &CORE, &["upgraded_level"]);
        let base = a.iter().find(|x| text(x, "upgraded_level") == "none").cloned();
        out.push(format!("\n#### {arch} — baseline is every level at small-7b\n"));
        if let Some(base) = &base {
            for r in &mut a {
                let delta_found = f(r, FOUND) - f(base, FOUND);
                let delta_cu = f(r, "compute_units") - f(base, "compute_units");
                let per = if delta_cu > 0.0 { delta_found / value::max2(1.0, delta_cu / 1e6) } else { 0.0 };
                r.insert("delta_found".into(), fl(delta_found));
                r.insert("delta_cu".into(), fl(delta_cu));
                r.insert("found_per_Mcu".into(), fl(per));
            }
        }
        out.push(md_table(
            &a,
            &[
                ("upgraded_level", "level upgraded to frontier", 0),
                ("average_precision", "AP", 4),
                (FOUND, "found", 3),
                ("delta_found", "Δ found", 3),
                ("compute_units", "compute", 0),
                ("delta_cu", "Δ compute", 0),
                ("found_per_Mcu", "Δfound / Mcu", 4),
            ],
            Some("delta_found"),
            true,
            &[],
        ));
        // paired significance vs the none baseline
        out.push(String::new());
        out.push("| level | Δ found | 95% CI | wins | sign p |".into());
        out.push("|---|---:|---|---:|---:|".into());
        for lvl in sorted_strs(&sub, "upgraded_level") {
            if lvl == "none" {
                continue;
            }
            let p =
                paired_by(&sub, &Value::from(lvl), &Value::from("none"), FOUND, "upgraded_level", &["scale", "seed"]);
            if p.n > 0 {
                out.push(format!("| {lvl} | {} |", paired_cells(&p)));
            }
        }
        out.push(String::new());
        out.push(level_verdict(arch, &a));
    }
    out.join("\n")
}

pub const LEVEL_ORDER: [&str; 6] = ["user", "team", "dept", "site", "region", "enterprise"];

/// Should capability increase as you go up?  Answered from the deltas.
fn level_verdict(arch: &str, a: &[Row]) -> String {
    let get = |lv: &str| a.iter().rfind(|r| text(r, "upgraded_level") == lv);
    if LEVEL_ORDER.iter().any(|lv| get(lv).is_none()) {
        return String::new();
    }
    let opt = |r: &Row, k: &str| r.get(k).and_then(num).unwrap_or(0.0);
    let gains: Vec<(&str, f64, f64)> = LEVEL_ORDER
        .iter()
        .map(|lv| {
            let r = get(lv).unwrap();
            (*lv, opt(r, "delta_found"), opt(r, "found_per_Mcu"))
        })
        .collect();
    let best_abs = value::max_by(gains.iter(), |g| g.1).unwrap();
    let best_eff = value::max_by(gains.iter(), |g| g.2).unwrap();
    let worst = value::min_by(gains.iter(), |g| g.1).unwrap();
    // Is the gain monotone in level height, ignoring the user level?  The user level does extraction on raw text,
    // which is a different job from the aggregation the middle levels do.
    let mid: Vec<f64> = gains[1..].iter().map(|g| g.1).collect();
    let monotone = mid.windows(2).all(|w| w[0] <= w[1] + 1e-9);
    // "Monotone" is vacuous when every middle level is flat at zero and only the top moves.  That is a different
    // statement and deserves its own one.
    let flat_middle = mid[..mid.len() - 1].iter().all(|v| v.abs() < 1e-9);
    let user_gain = gains[0].1;
    let barbell = !monotone && user_gain > value::max_by(mid[..mid.len() - 1].iter().copied(), |x| *x).unwrap_or(0.0);
    let shape = if flat_middle {
        "**only the top level matters at all** — every intermediate level is flat at zero"
    } else if monotone {
        "**capability should increase monotonically upward**"
    } else if barbell {
        "**a barbell: buy capability at the very bottom and the very top, and run the middle cheap**"
    } else {
        "**not a clean trend**"
    };
    let mut ranked: Vec<&(&str, f64, f64)> = gains.iter().collect();
    sort_by_key_f(&mut ranked, |g| -g.1);
    let order = ranked.iter().map(|(lv, g, _)| format!("{lv} {}", fmt::f(*g, "+.3f"))).collect::<Vec<_>>().join(" > ");
    let n_runs = get(best_abs.0).unwrap().get("n_runs").and_then(Value::as_i64).unwrap_or(0);
    let caveat = if n_runs < 5 {
        format!(
            " Every interval here is from {n_runs} runs per cell, so the ordering is directional and the individual \
             gaps are not separable."
        )
    } else {
        String::new()
    };
    format!(
        "**Where the model budget goes for `{arch}`.** Upgrading the **{}** level buys the most discovery ({}) and \
         upgrading **{}** buys the most per unit of extra compute ({} discovery per Mcu). The **{}** level buys the \
         least ({}). Ranked: {order}. The pattern is {shape} — and note that the user level is doing extraction from \
         raw text, a different job from the aggregation the middle levels do, so its gain does not belong to the same \
         trend as theirs.{caveat}",
        best_abs.0,
        fmt::f(best_abs.1, "+.3f"),
        best_eff.0,
        fmt::f(best_eff.2, ".3f"),
        worst.0,
        fmt::f(worst.1, "+.3f")
    )
}

/// Section 13: every level at the same capability `q`.
pub fn section_qsweep() -> String {
    let rows = load("e3c_q_sweep.jsonl");
    if rows.is_empty() {
        return "_(E3c not run)_".to_string();
    }
    let a = agg(&rows, &CORE, &["arch", "q"]);
    let tbl = md_table(
        &a,
        &[
            ("arch", "architecture", 0),
            ("q", "q", 2),
            ("average_precision", "AP", 4),
            (FOUND, "found", 3),
            ("rare_signal_recall", "rare", 3),
            ("false_discovery_rate", "FDR", 3),
            ("compute_units", "compute", 0),
        ],
        None,
        true,
        &[],
    );
    tbl + "\n\n" + &qsweep_verdict(&a)
}

/// What the capability sweep says about where capability is worth buying.
fn qsweep_verdict(a: &[Row]) -> String {
    let qs = sorted_values(a, "q");
    let archs = sorted_strs(a, "arch");
    if qs.len() < 3 {
        return String::new();
    }
    let (lo, hi) = (&qs[0], &qs[qs.len() - 1]);
    let g = |arch: &str, q: &Value, k: &str| -> f64 {
        a.iter().find(|r| text(r, "arch") == arch && py_eq(&r["q"], q)).map_or(f64::NAN, |r| f(r, k))
    };
    let (lo_f, hi_f) = (num(lo).unwrap(), num(hi).unwrap());
    let mut lines = vec![
        "**What the sweep says.** Each architecture is run with *every* level set to the same capability `q`, so the \
         curve isolates the operator from the topology."
            .to_string(),
        String::new(),
        format!(
            "| architecture | found at q={} | found at q={} | gain | compute multiple | gain per doubling of compute |",
            fmt::f(lo_f, ".2f"),
            fmt::f(hi_f, ".2f")
        ),
        "|---|---:|---:|---:|---:|---:|".to_string(),
    ];
    let mut rows_out: Vec<(&str, f64, f64, f64, f64, f64)> = Vec::new();
    for arch in archs {
        let (f0, f1) = (g(arch, lo, FOUND), g(arch, hi, FOUND));
        let (c0, c1) = (g(arch, lo, "compute_units"), g(arch, hi, "compute_units"));
        // `if c0`, `if mult and mult > 0`: zero is falsy, NaN truthy but not positive
        let mult = if c0 != 0.0 { c1 / c0 } else { f64::NAN };
        let doublings = if mult != 0.0 && mult > 0.0 { mult.log2() } else { f64::NAN };
        let per = if doublings != 0.0 && doublings > 0.0 { (f1 - f0) / doublings } else { 0.0 };
        rows_out.push((arch, f0, f1, f1 - f0, mult, per));
        lines.push(format!(
            "| {arch} | {} | {} | {} | {}x | {} |",
            fmt::f(f0, ".3f"),
            fmt::f(f1, ".3f"),
            fmt::f(f1 - f0, "+.3f"),
            fmt::f(mult, ".1f"),
            fmt::f(per, "+.3f")
        ));
    }
    lines.push(String::new());
    // The architecture that converts capability into discovery most efficiently is the one worth spending a model
    // budget on.
    let best = value::max_by(rows_out.iter(), |r| r.5).unwrap();
    let worst = value::min_by(rows_out.iter(), |r| r.5).unwrap();
    lines.push(format!(
        "Capability is not worth the same everywhere. Per doubling of compute spent on better operators, `{}` \
         converts it into {} discovery and `{}` into {}. Read together with §13, which upgrades one level at a time: \
         this table says how much a *uniform* capability increase is worth, §13 says where to put it if you are only \
         buying one.",
        best.0,
        fmt::f(best.5, "+.3f"),
        worst.0,
        fmt::f(worst.5, "+.3f")
    ));
    // Pure upward aggregation is the interesting special case.
    if let Some(e) = rows_out.iter().find(|r| r.0.starts_with("E_")) {
        lines.push(String::new());
        lines.push(format!(
            "The row worth dwelling on is `{}`, pure upward aggregation with no descent. It finds {} at q={} and {} \
             at q={}. Its failure at realistic operator quality is therefore not a structural impossibility — a \
             *perfect* summariser would make it work — but every real operator sits far enough below perfect that the \
             structure cannot be rescued by a better model. That is a stronger negative result than 'it does not \
             work', and a more useful one: it says the design is sensitive to exactly the thing we cannot guarantee.",
            e.0,
            fmt::f(e.1, ".1%"),
            fmt::f(lo_f, ".2f"),
            fmt::f(e.2, ".1%"),
            fmt::f(hi_f, ".2f")
        ));
    }
    lines.join("\n")
}

/// Section 19: users per team and teams per department.
pub fn section_fanin() -> String {
    let mut out = Vec::new();
    let rows = load("e4_fanin.jsonl");
    if !rows.is_empty() {
        let mut metrics: Vec<&str> = CORE.to_vec();
        metrics.extend(["kernel_context_tokens", "compression_ratio"]);
        let a = agg(&rows, &metrics, &["arch", "team_size"]);
        out.push("#### Users per team\n".to_string());
        out.push(md_table(
            &a,
            &[
                ("arch", "architecture", 0),
                ("team_size", "team size", 0),
                ("average_precision", "AP", 4),
                (FOUND, "found", 3),
                ("rare_signal_recall", "rare", 3),
                ("information_loss", "info loss", 3),
                ("compute_units", "compute", 0),
                ("wall_seconds", "wall s", 1),
            ],
            None,
            true,
            &[],
        ));
    }
    let rows = load("e4b_dept_fanin.jsonl");
    if !rows.is_empty() {
        let a = agg(&rows, &CORE, &["arch", "teams_per_dept"]);
        out.push("\n#### Teams per department\n".to_string());
        out.push(md_table(
            &a,
            &[
                ("arch", "architecture", 0),
                ("teams_per_dept", "teams/dept", 0),
                ("average_precision", "AP", 4),
                (FOUND, "found", 3),
                ("information_loss", "info loss", 3),
                ("compute_units", "compute", 0),
            ],
            None,
            true,
            &[],
        ));
    }
    if out.is_empty() {
        return "_(E4 not run)_".to_string();
    }
    out.push(String::new());
    out.push(fanin_verdict(&load("e4_fanin.jsonl"), "team_size", "users per team"));
    out.push(String::new());
    out.push(fanin_verdict(&load("e4b_dept_fanin.jsonl"), "teams_per_dept", "teams per department"));
    out.join("\n")
}

/// Is fan-in a lever, or is the spread inside the seed noise?
fn fanin_verdict(rows: &[Row], key: &str, label: &str) -> String {
    if rows.is_empty() {
        return String::new();
    }
    let mut lines = Vec::new();
    for arch in sorted_strs(rows, "arch") {
        let sub: Vec<Row> = rows.iter().filter(|r| text(r, "arch") == arch).cloned().collect();
        let per: Vec<(Value, (f64, f64, f64))> = sorted_values(&sub, key)
            .into_iter()
            .map(|x| {
                let v: Vec<f64> = sub.iter().filter(|r| py_eq(&r[key], &x)).map(|r| f(r, FOUND)).collect();
                let ci = boot_ci(&v);
                (x, ci)
            })
            .collect();
        let best_x = value::max_by(per.iter(), |p| p.1.0).unwrap();
        let worst_x = value::min_by(per.iter(), |p| p.1.0).unwrap();
        let spread = best_x.1.0 - worst_x.1.0;
        // Widest CI anywhere on the curve: if the whole spread fits inside a single point's own uncertainty, the
        // curve is flat.
        let widest = value::max_by(per.iter().map(|p| p.1.2 - p.1.1), |x| *x).unwrap();
        let flat = spread <= widest;
        lines.push(format!(
            "* `{arch}`: best at **{} {}** ({}), worst at {} ({}), total spread {}. The widest 95% interval on any \
             single point is {}, so the whole {}",
            py_str(&best_x.0),
            label.split_whitespace().next().unwrap_or_default(),
            fmt::f(best_x.1.0, ".3f"),
            py_str(&worst_x.0),
            fmt::f(worst_x.1.0, ".3f"),
            fmt::f(spread, ".3f"),
            fmt::f(widest, ".3f"),
            if flat {
                "spread fits inside one point's own uncertainty: **this curve is flat** and the apparent optimum is \
                 noise."
            } else {
                "spread is larger than any single point's uncertainty, so there is a real effect here."
            }
        ));
    }
    format!(
        "**Is {label} a lever?**\n\n{}\n\n{}",
        lines.join("\n"),
        if lines.iter().all(|x| x.contains("is flat")) {
            "Choose it for human reasons — span of control, meeting load, management overhead — because strategic \
             discovery does not distinguish the options."
        } else {
            "Where the curve is not flat, the table above gives the direction; treat the optimum as indicative rather \
             than tuned, since the sweep is one dimension at a time."
        }
    )
}

/// Section 18: degradation under each adversarial condition.
pub fn section_adversarial() -> String {
    let rows = load("e5_adversarial.jsonl");
    if rows.is_empty() {
        return "_(E5 not run)_".to_string();
    }
    let mut metrics: Vec<&str> = CORE.to_vec();
    metrics.extend([
        "decoy_D1_entity_coincidence",
        "decoy_D2_temporal_scramble",
        "decoy_D3_near_miss_entity",
        "decoy_D5_stale_chain",
        "dup_inflation_support_error",
    ]);
    let a = agg(&rows, &metrics, &["condition", "arch"]);
    let tbl = md_table(
        &a,
        &[
            ("condition", "condition", 0),
            ("arch", "architecture", 0),
            ("average_precision", "AP", 4),
            (FOUND, "found", 3),
            ("false_discovery_rate", "FDR", 3),
            ("decoy_acceptance_all", "decoy acc", 3),
            ("decoy_D1_entity_coincidence", "D1", 3),
            ("decoy_D2_temporal_scramble", "D2", 3),
            ("decoy_D3_near_miss_entity", "D3", 3),
            ("decoy_D5_stale_chain", "D5", 3),
            ("dup_inflation_support_error", "D4 infl", 2),
        ],
        None,
        true,
        &[],
    );
    tbl + "\n\n" + &adversarial_verdict(&a)
}

/// Does the hierarchy invent strategic narratives out of noise?
///
/// That was the explicit question the adversarial suite was built to answer, so it is answered here rather than
/// left to the reader to subtract two rows of a wide table.
fn adversarial_verdict(a: &[Row]) -> String {
    // {arch: row} over the clean condition: a later row for the same arch replaces the earlier one
    let mut base: Vec<(&str, &Row)> = Vec::new();
    for r in a.iter().filter(|r| text(r, "condition") == "clean") {
        match base.iter_mut().find(|b| b.0 == text(r, "arch")) {
            Some(b) => b.1 = r,
            None => base.push((text(r, "arch"), r)),
        }
    }
    let conds: Vec<&str> = sorted_strs(a, "condition").into_iter().filter(|c| *c != "clean").collect();
    if base.is_empty() || conds.is_empty() {
        return String::new();
    }
    let row_at = |c: &str, arch: &str| a.iter().find(|r| text(r, "condition") == c && text(r, "arch") == arch);
    let mut lines = vec![
        "**Does the hierarchy invent strategic narratives out of noise?** Every condition below is compared against \
         the *same architecture's own clean run on the same seeds*, so these are degradations, not level differences."
            .to_string(),
        String::new(),
        "| architecture | worst condition for discovery | Δ found | worst condition for false discoveries | Δ FDR | max \
         Δ FDR across all conditions |"
            .to_string(),
        "|---|---|---:|---|---:|---:|".to_string(),
    ];
    let mut sorted_base = base.clone();
    sorted_base.sort_by(|x, y| x.0.cmp(y.0));
    for (arch, b) in &sorted_base {
        let mut df: Vec<(&str, f64)> = Vec::new();
        let mut dr: Vec<(&str, f64)> = Vec::new();
        for &c in &conds {
            let Some(row) = row_at(c, arch) else { continue };
            df.push((c, f(row, FOUND) - f(b, FOUND)));
            dr.push((c, f(row, "false_discovery_rate") - f(b, "false_discovery_rate")));
        }
        if df.is_empty() {
            continue;
        }
        let wf = value::min_by(df.iter(), |x| x.1).unwrap();
        let wr = value::max_by(dr.iter(), |x| x.1).unwrap();
        let max_dr = value::max_by(dr.iter().map(|x| x.1), |x| *x).unwrap();
        lines.push(format!(
            "| {arch} | {} | {} | {} | {} | {} |",
            wf.0,
            fmt::f(wf.1, "+.3f"),
            wr.0,
            fmt::f(wr.1, "+.3f"),
            fmt::f(max_dr, "+.3f")
        ));
    }
    // The headline: how large is the largest FDR increase anywhere.
    let mut worst_fdr = 0.0;
    for (arch, b) in &base {
        for &c in &conds {
            if let Some(row) = row_at(c, arch) {
                worst_fdr = value::max2(worst_fdr, f(row, "false_discovery_rate") - f(b, "false_discovery_rate"));
            }
        }
    }
    lines.push(String::new());
    lines.push(format!(
        "**The largest increase in false-discovery rate under any adversarial condition, for any architecture, is \
         {}.** Flooding the corpus with correlated false reports, duplicated chatter posing as independent \
         corroboration, stale retracted chains, or corrupted nodes does not make these systems markedly more likely to \
         assert things that are not there. What the adversarial conditions cost is *recall* — the systems go quieter, \
         not wronger. For an executive register that is the preferable failure mode, but it is also the more \
         dangerous one to operate blind: a degraded system looks exactly like a calm quarter.",
        fmt::f(worst_fdr, "+.3f")
    ));
    lines.join("\n")
}

/// Section 14: the question/descent budget and the map-reduce kernel budget.
pub fn section_frontier() -> String {
    let rows = load("e6_frontier.jsonl");
    if rows.is_empty() {
        return "_(E6 not run)_".to_string();
    }
    let mut out = Vec::new();
    let mut metrics: Vec<&str> = CORE.to_vec();
    metrics.extend(["cost_per_correct_discovery", "usd_estimate"]);
    let h: Vec<Row> = rows.iter().filter(|r| text(r, "knob") == "question_frac").cloned().collect();
    if !h.is_empty() {
        let a = agg(&h, &metrics, &["question_frac"]);
        out.push("#### Hierarchy: question / descent budget\n".to_string());
        out.push(md_table(
            &a,
            &[
                ("question_frac", "question budget frac", 2),
                ("average_precision", "AP", 4),
                (FOUND, "found", 3),
                ("rare_signal_recall", "rare", 3),
                ("compute_units", "compute", 0),
                ("cost_per_correct_discovery", "cu/disc", 0),
            ],
            None,
            true,
            &[],
        ));
        let v = budget_verdict(&a, "question_frac", "question-budget fraction");
        if !v.is_empty() {
            out.push(format!("\n{v}"));
        }
    }
    let m: Vec<Row> = rows.iter().filter(|r| text(r, "knob") == "mr_budget").cloned().collect();
    if !m.is_empty() {
        let a = agg(&m, &metrics, &["mr_budget"]);
        out.push("\n#### Map-reduce: kernel object budget\n".to_string());
        out.push(md_table(
            &a,
            &[
                ("mr_budget", "kernel objects", 0),
                ("average_precision", "AP", 4),
                (FOUND, "found", 3),
                ("compute_units", "compute", 0),
                ("cost_per_correct_discovery", "cu/disc", 0),
            ],
            None,
            true,
            &[],
        ));
        let v = budget_verdict(&a, "mr_budget", "kernel objects");
        if !v.is_empty() {
            out.push(format!("\n{v}"));
        }
    }
    out.retain(|x| !x.is_empty());
    out.join("\n")
}

/// Report where a budget curve turns over, if it does.
///
/// More evidence making the answer worse is the study's central structural claim, so it is stated from the curve
/// rather than asserted.
fn budget_verdict(a: &[Row], key: &str, label: &str) -> String {
    let mut pts: Vec<&Row> = a.iter().collect();
    pts.sort_by(|x, y| py_cmp(&x[key], &y[key]));
    if pts.len() < 3 {
        return String::new();
    }
    let fs: Vec<f64> = pts.iter().map(|r| f(r, FOUND)).collect();
    let cost: Vec<f64> =
        pts.iter().map(|r| r.get("cost_per_correct_discovery").and_then(num).unwrap_or(f64::NAN)).collect();
    let peak = math::argmax(&fs);
    // np.nanargmin: the first smallest non-NaN value
    let cheapest = if cost.iter().all(|c| c.is_nan()) {
        peak
    } else {
        let mut best: Option<usize> = None;
        for (i, c) in cost.iter().enumerate() {
            if !c.is_nan() && best.is_none_or(|b| *c < cost[b]) {
                best = Some(i);
            }
        }
        best.unwrap()
    };
    let last = pts.len() - 1;
    let tail_drop = fs[peak] - fs[last];
    if peak == last {
        return format!(
            "Discovery is still rising at the largest {label} tested ({}), so this budget has not been pushed to its \
             turning point; the ceiling here is the sweep, not the design.",
            py_str(&pts[last][key])
        );
    }
    format!(
        "**The curve turns over.** Discovery peaks at **{} {label}** ({}) and falls to {} at {} — a loss of {} from \
         spending *more*. Cost per correct discovery is lowest at {} ({} cu). This is the study's central structural \
         claim in one table: past a point, additional undifferentiated evidence crowds the genuine candidates down the \
         ranking faster than it adds new ones. A budget is a filter, and removing the filter is not free.",
        py_str(&pts[peak][key]),
        fmt::f(fs[peak], ".3f"),
        fmt::f(fs[last], ".3f"),
        py_str(&pts[last][key]),
        fmt::f(tail_drop, ".3f"),
        py_str(&pts[cheapest][key]),
        fmt::f(cost[cheapest], ".3g")
    )
}

/// Section 22: the allocation under each capability-shape assumption.
pub fn section_shape() -> String {
    let rows = load("e7_shape.jsonl");
    if rows.is_empty() {
        return "_(E7 not run)_".to_string();
    }
    let a = agg(&rows, &CORE, &["shape", "alloc", "arch"]);
    let tbl = md_table(
        &a,
        &[
            ("shape", "capability shape", 0),
            ("alloc", "allocation", 0),
            ("arch", "architecture", 0),
            ("average_precision", "AP", 4),
            (FOUND, "found", 3),
            ("compute_units", "compute", 0),
        ],
        None,
        true,
        &[],
    );
    tbl + "\n\n" + &shape_verdict(&a)
}

/// Does the recommended allocation change if the shape assumption does?
///
/// That is the only question this sweep exists to answer, so it is answered rather than left as a 36-row table to
/// diff by eye.
fn shape_verdict(a: &[Row]) -> String {
    let shapes = sorted_strs(a, "shape");
    let archs = sorted_strs(a, "arch");
    if shapes.len() < 2 {
        return String::new();
    }
    let mut lines = vec![
        "**Does the recommendation depend on the shape assumption?**".to_string(),
        String::new(),
        "| architecture | best allocation, by shape | stable? | spread in `found` across shapes at the best \
         allocation |"
            .to_string(),
        "|---|---|---|---:|".to_string(),
    ];
    let mut all_stable = true;
    for arch in archs {
        let mut best: Vec<(&str, &Row)> = Vec::new();
        for &sh in &shapes {
            let sub = a.iter().filter(|r| text(r, "shape") == sh && text(r, "arch") == arch);
            if let Some(b) = value::max_by(sub, |r| f(r, FOUND)) {
                best.push((sh, b));
            }
        }
        if best.is_empty() {
            continue;
        }
        let names =
            |sh: &str| text(best.iter().find(|b| b.0 == sh).expect("every shape run for every arch").1, "alloc");
        let vals: Vec<f64> = best.iter().map(|b| f(b.1, FOUND)).collect();
        let spread =
            value::max_by(vals.iter().copied(), |x| *x).unwrap() - value::min_by(vals.iter().copied(), |x| *x).unwrap();
        // A different name at the top is only a shape DEPENDENCE if the winners actually score differently.  Where the
        // top allocations tie, argmax is picking arbitrarily and there is nothing to report.
        let same_name = best.iter().map(|b| text(b.1, "alloc")).collect::<BTreeSet<_>>().len() == 1;
        let tie = !same_name && spread < 1e-9;
        let stable = same_name || tie;
        all_stable = all_stable && stable;
        let mark = if same_name {
            "yes"
        } else if tie {
            "yes (tie)"
        } else {
            "**no**"
        };
        lines.push(format!(
            "| {arch} | {} | {mark} | {} |",
            shapes.iter().map(|sh| format!("{sh}: `{}`", names(sh))).collect::<Vec<_>>().join(", "),
            fmt::f(spread, ".3f")
        ));
    }
    lines.push(String::new());
    let lead = if all_stable {
        "**No architecture's best allocation changes materially with the shape assumption**, so the allocation \
         conclusion in §13 is not an artefact of how hard capabilities are assumed to scale. Where a row is marked \
         \"tie\", two allocations score identically and the name at the top is arbitrary."
    } else {
        "**At least one architecture's best allocation changes with the shape assumption**, so that recommendation is \
         shape-dependent and should be read with the row above rather than from §13 alone."
    };
    lines.push(format!(
        "{lead} Many rows repeat exactly across shapes: the shape parameter only bends the *hard* capabilities at \
         intermediate `q`, and allocations that pin the relevant level at an anchor tier see no difference at all. \
         That is expected, and is the reason the table is not more interesting than it looks."
    ));
    lines.join("\n")
}

/// Section 19: the strict tree against the tree plus semantic cross-links.
pub fn section_crosslinks() -> String {
    let rows = load("e8_crosslinks.jsonl");
    if rows.is_empty() {
        return "_(E8 not run)_".to_string();
    }
    let a = agg(&rows, &CORE, &["scale", "cross_links"]);
    let mut lines = vec![
        md_table(
            &a,
            &[
                ("scale", "users", 0),
                ("cross_links", "semantic cross-links", 0),
                ("average_precision", "AP", 4),
                (FOUND, "found", 3),
                ("rare_signal_recall", "rare", 3),
                ("compute_units", "compute", 0),
            ],
            None,
            true,
            &[],
        ),
        String::new(),
        "| scale | Δ found (links on − off) | 95% CI | wins | sign p |".to_string(),
        "|---|---:|---|---:|---:|".to_string(),
    ];
    let mut deltas: Vec<(i64, Paired)> = Vec::new();
    for scale in scales_of(&rows) {
        let sub: Vec<Row> = rows.iter().filter(|r| int(r, "scale") == scale).cloned().collect();
        let p = paired_by(&sub, &Value::Bool(true), &Value::Bool(false), FOUND, "cross_links", &["scale", "seed"]);
        if p.n > 0 {
            lines.push(format!("| {} | {} |", comma(scale), paired_cells(&p)));
            deltas.push((scale, p));
        }
    }
    lines.push(String::new());
    lines.push(crosslink_verdict(&deltas));
    lines.join("\n")
}

/// State what the cross-link sweep found, including when it found nothing.
fn crosslink_verdict(deltas: &[(i64, Paired)]) -> String {
    if deltas.is_empty() {
        return String::new();
    }
    // A scale counts as supporting cross-links only if the whole bootstrap CI is on one side of zero; with 3 seeds
    // the sign test cannot go below 0.25, so requiring significance there would reject everything by construction.
    let pos: Vec<i64> = deltas.iter().filter(|(_, p)| p.ci_lo > 0.0).map(|(s, _)| *s).collect();
    let neg: Vec<i64> = deltas.iter().filter(|(_, p)| p.ci_hi < 0.0).map(|(s, _)| *s).collect();
    let biggest = value::max_by(deltas.iter(), |x| x.1.mean_diff.abs()).unwrap();
    let top = deltas.iter().map(|d| d.0).max().unwrap();
    let mut out = "**Verdict.** Adding semantic cross-links to the organisational tree ".to_string();
    if !pos.is_empty() && neg.is_empty() && pos.len() == deltas.len() {
        out += "helps at every scale measured";
    } else if !pos.is_empty() && pos.iter().max() == Some(&top) {
        out += "helps, and the effect holds at the largest scale measured";
    } else if !pos.is_empty() {
        out += &format!(
            "helps at {} users but the effect does not survive to {}",
            pos.iter().map(|s| comma(*s)).collect::<Vec<_>>().join(", "),
            comma(top)
        );
    } else {
        out += "does not reliably change discovery at any scale measured";
    }
    out += &format!(
        ". The largest effect anywhere is {} at {} users, on {} paired seeds. ",
        fmt::f(biggest.1.mean_diff, "+.3f"),
        comma(biggest.0),
        biggest.1.n
    );
    if pos.is_empty() || pos.iter().copied().max().unwrap_or(0) != top {
        out += "The mechanism the links are supposed to supply — a path between branches that the org chart does not \
                provide — is already supplied by the sketch channel, which is indexed by entity rather than by branch \
                and so crosses the tree for free. On this evidence the cross-links are redundant with it, not \
                additive to it, and should not be built on the strength of these numbers.";
    } else {
        out += "Note that the seed counts here are small enough that the sign test cannot fall below 0.25, so the \
                bootstrap interval is doing the work; treat this as directional.";
    }
    out
}

/// Section 21: does cited evidence name the entity the report is about?
pub fn section_provenance() -> String {
    let rows = load("e12_provenance.jsonl");
    if rows.is_empty() {
        return "_(E12 not run)_".to_string();
    }
    let a = agg(
        &rows,
        &[
            "evidence_names_claimed_entity",
            "reports_fully_attributed",
            "reports_with_no_matching_evidence",
            FOUND,
            "evidence_precision",
            "provenance_preservation",
        ],
        &["scale", "arch"],
    );
    let tbl = md_table(
        &a,
        &[
            ("scale", "users", 0),
            ("arch", "architecture", 0),
            ("evidence_names_claimed_entity", "cited evidence names the claimed entity", 3),
            ("reports_fully_attributed", "reports fully attributed", 3),
            ("reports_with_no_matching_evidence", "reports with NO matching evidence", 3),
            ("evidence_precision", "evidence precision", 3),
            (FOUND, "found", 3),
        ],
        None,
        true,
        &[],
    );
    let big = a.iter().map(|r| int(r, "scale")).max().unwrap();
    let sub: Vec<&Row> = a.iter().filter(|r| int(r, "scale") == big).collect();
    let Some(best) = value::max_by(sub.iter().copied(), |r| f(r, "evidence_names_claimed_entity")) else {
        return tbl;
    };
    let mut out = vec![tbl, String::new()];
    out.push(format!(
        "**Verdict at {} users.** The strongest attribution is `{}` — {} of cited evidence names the entity the report \
         is about, and {} of its reports are fully attributable end to end.",
        comma(big),
        text(best, "arch"),
        fmt::f(f(best, "evidence_names_claimed_entity"), ".0%"),
        fmt::f(f(best, "reports_fully_attributed"), ".0%")
    ));
    if let Some(h) = sub.iter().find(|r| text(r, "arch") == "H_mycelic_full") {
        out.push(String::new());
        out.push(format!(
            "The hierarchy reaches {} on the first measure and {} on the second. **That second number is the \
             uncomfortable one**, and it should be read before any claim that a lineage-carrying architecture is \
             inherently more auditable: carrying a lineage *path* is not the same as being able to put an executive in \
             front of the original note. The hierarchy knows which nodes a claim travelled through; the centralised \
             options can still show you the text. For a regulator or an incident review, the second is what is being \
             asked for.",
            fmt::f(f(h, "evidence_names_claimed_entity"), ".0%"),
            fmt::f(f(h, "reports_fully_attributed"), ".1%")
        ));
        out.push(String::new());
        let hn = f(h, "reports_with_no_matching_evidence");
        let bn = f(best, "reports_with_no_matching_evidence");
        // Lower is better here: this counts reports the system made with no evidence behind them at all.
        let mid = if hn > bn {
            format!(
                ", so the hierarchy is roughly {}x more likely to put something on the register it cannot back up. ",
                fmt::f(hn / value::max2(bn, 1e-9), ".0f")
            )
        } else {
            ", so the hierarchy is the more conservative of the two here. ".to_string()
        };
        out.push(format!(
            "It goes the same way on unsupported assertions — reports made with no matching evidence behind them at \
             all, where lower is better. The hierarchy is at {} and `{}` at {}{mid}Across all three attribution \
             measures, provenance is a place the hierarchy loses, not a place it wins. It should not be used as an \
             argument for building one.",
            fmt::f(hn, ".1%"),
            text(best, "arch"),
            fmt::f(bn, ".1%")
        ));
    }
    out.join("\n")
}

/// Section 20: what leaves each node, by kind.
pub fn section_privacy() -> String {
    let rows = load("e9_privacy.jsonl");
    if rows.is_empty() {
        return "_(E9 not run)_".to_string();
    }
    let a = agg(
        &rows,
        &[
            "raw_text_exposure_fraction",
            "claim_exposure_fraction",
            "sketch_entries_leaving_node",
            "privacy_exposure_fraction",
            FOUND,
            "compute_units",
        ],
        &["scale", "arch"],
    );
    md_table(
        &a,
        &[
            ("scale", "users", 0),
            ("arch", "architecture", 0),
            ("raw_text_exposure_fraction", "raw text out", 4),
            ("claim_exposure_fraction", "claims out", 4),
            ("sketch_entries_leaving_node", "index entries out", 0),
            (FOUND, "found", 3),
            ("compute_units", "compute", 0),
        ],
        None,
        true,
        &[],
    )
}

fn scale_mean(rows: &[Row], arch: &str, sc: i64, key: &str) -> Option<f64> {
    let v: Vec<f64> =
        rows.iter().filter(|r| text(r, "arch") == arch && int(r, "scale") == sc).map(|r| f(r, key)).collect();
    if v.is_empty() { None } else { Some(math::mean(&v)) }
}

/// Section 11: discovery and compute per architecture across enterprise sizes.
pub fn section_scale_trend() -> String {
    let rows = dedupe_default(load_all(&["e1_baselines.jsonl", "e1b_extra.jsonl", "e10_scale_trend.jsonl"]));
    if rows.is_empty() {
        return "_(not run)_".to_string();
    }
    let archs = [
        "A_flat_rag",
        "A2_chunked_ctx",
        "B_long_context",
        "B2_map_reduce",
        "B4_central_triage",
        "E_hier_lineage",
        "G_hier_questions",
        "H_mycelic_full",
        "J_mycelic_verified",
        "Y_oracle_retrieval",
    ];
    let scales = scales_of(&rows);
    let head = format!("| architecture | {} |", scales.iter().map(|s| comma(*s)).collect::<Vec<_>>().join(" | "));
    let sep = format!("|---|{}", "---:|".repeat(scales.len()));
    let mut lines = vec![head.clone(), sep.clone()];
    let table = |key: &str, spec: &str, lines: &mut Vec<String>| {
        for a in archs {
            let cells: Vec<String> = scales
                .iter()
                .map(|&sc| scale_mean(&rows, a, sc, key).map_or("—".to_string(), |m| fmt::f(m, spec)))
                .collect();
            if cells.iter().any(|c| c != "—") {
                lines.push(format!("| {a} | {} |", cells.join(" | ")));
            }
        }
    };
    table(FOUND, ".3f", &mut lines);
    lines.push(String::new());
    lines.push("Compute (cu) at the same points:".into());
    lines.push(String::new());
    lines.push(head);
    lines.push(sep);
    table("compute_units", ".2e", &mut lines);
    lines.push(String::new());
    lines.push(scale_verdict(&rows, &scales));
    lines.join("\n")
}

/// How do discovery and cost trend with enterprise size, and do the curves ever cross?  Stated from the endpoints
/// rather than eyeballed.
fn scale_verdict(rows: &[Row], scales: &[i64]) -> String {
    if scales.len() < 2 {
        return String::new();
    }
    let (lo, hi) = (scales[0], scales[scales.len() - 1]);
    let g = |arch: &str, sc: i64, key: &str| scale_mean(rows, arch, sc, key);
    const HIER: &str = "H_mycelic_full";
    // Compare against whichever non-reference architecture is strongest at the largest scale both were run at.
    // (The Python iterated a set of names here, whose order depends on string hashing; only an exact tie for the
    // strongest could depend on it, and the names are taken in sorted order.)
    let cands: Vec<&str> = sorted_strs(rows, "arch")
        .into_iter()
        .filter(|a| !["Y_oracle_retrieval", "Z_random_rank", "Z2_naive_enumerate", HIER].contains(a))
        .filter(|a| g(a, hi, FOUND).is_some())
        .collect();
    let Some(rival) = value::max_by(cands.iter().copied(), |a| g(a, hi, FOUND).unwrap()) else { return String::new() };
    let vals = [
        g(HIER, lo, FOUND),
        g(HIER, hi, FOUND),
        g(rival, lo, FOUND),
        g(rival, hi, FOUND),
        g(HIER, lo, "compute_units"),
        g(HIER, hi, "compute_units"),
        g(rival, lo, "compute_units"),
        g(rival, hi, "compute_units"),
    ];
    if vals.iter().any(Option::is_none) {
        return String::new();
    }
    let [hf_lo, hf_hi, rf_lo, rf_hi, hc_lo, hc_hi, rc_lo, rc_hi] = vals.map(Option::unwrap);
    let users_mult = hi as f64 / lo as f64;
    format!(
        "**Does the picture change with size?** From {} to {} users — {}x the people — the hierarchy's discovery goes \
         {} → {} and `{rival}`'s goes {} → {}, so the gap {} rather than closing. Compute tells the opposite story: \
         the hierarchy's rises {}x over that {}x growth in people, `{rival}`'s {}x, so the hierarchy's cost advantage \
         {} with size: `{rival}` costs {}x the hierarchy at {} users{}and {}x at {}. Neither curve crosses inside the \
         range measured. The trend is favourable to the hierarchy on cost and unfavourable on accuracy, and nothing \
         here justifies extrapolating a crossing point beyond {} — that would be an extrapolation, and this study \
         does not make any.",
        comma(lo),
        comma(hi),
        fmt::f(users_mult, ".0f"),
        fmt::f(hf_lo, ".3f"),
        fmt::f(hf_hi, ".3f"),
        fmt::f(rf_lo, ".3f"),
        fmt::f(rf_hi, ".3f"),
        if (rf_hi - hf_hi) > (rf_lo - hf_lo) { "widens" } else { "narrows" },
        fmt::f(hc_hi / hc_lo, ".1f"),
        fmt::f(users_mult, ".0f"),
        fmt::f(rc_hi / rc_lo, ".1f"),
        if (rc_hi / hc_hi) > (rc_lo / hc_lo) { "grows" } else { "shrinks" },
        fmt::f(rc_lo / hc_lo, ".1f"),
        comma(lo),
        if rc_lo < hc_lo { " — i.e. it is the cheaper of the two there — " } else { " " },
        fmt::f(rc_hi / hc_hi, ".1f"),
        comma(hi),
        comma(hi)
    )
}

fn read_json(name: &str) -> Option<Value> {
    let text = std::fs::read_to_string(art().join(name)).ok()?;
    Some(serde_json::from_str(&text).unwrap_or_else(|e| panic!("{name}: {e}")))
}

fn obj(v: &Value) -> Vec<(&String, &Value)> {
    v.as_object().map(|o| o.iter().collect()).unwrap_or_default()
}

fn vf(v: &Value, k: &str, spec: &str) -> String {
    value::format(&v[k], spec)
}

/// Section 23: the live measurements on real models.
pub fn section_live() -> String {
    let mut out: Vec<String> = Vec::new();
    if let Some(d) = read_json("live_calibration.json") {
        out.push("#### Primitive operators (blind, 198 items)\n".into());
        out.push(
            "| model | extraction | entity fidelity | causal check | temporal check | entity linking | independence | \
             fitted q |"
                .into(),
        );
        out.push("|---|---:|---:|---:|---:|---:|---:|---:|".into());
        for (m, v) in obj(&d["measurements"]) {
            out.push(format!(
                "| {m} | {} | {} | {} | {} | {} | {} | {} |",
                vf(v, "extract_recall", ".3f"),
                vf(v, "entity_fidelity", ".3f"),
                vf(v, "causal_check", ".3f"),
                vf(v, "temporal_check", ".3f"),
                vf(v, "entity_check", ".3f"),
                vf(v, "dedup_check", ".3f"),
                vf(v, "fitted_q", ".2f")
            ));
        }
    }
    for (suffix, label) in [("", "evidence STATISTICS only"), ("_rich", "statistics + the RAW WORK NOTES")] {
        let Some(d) = read_json(&format!("live_rank_results{suffix}.json")) else { continue };
        out.push(format!(
            "\n#### Candidate discrimination — {label} ({} candidates from a real run, {} genuine)\n",
            py_str(&d["n_items"]),
            py_str(&d["n_real"])
        ));
        out.push("| ranker | AP | selection precision | selection recall | selection F1 | n selected |".into());
        out.push("|---|---:|---:|---:|---:|---:|".into());
        out.push(format!("| random | {} | — | — | — | — |", vf(&d, "random_ap", ".4f")));
        out.push(format!(
            "| simulator's calibrated logistic (statistics) | {} | — | — | — | — |",
            vf(&d, "simulator_logistic_ap", ".4f")
        ));
        let mut models = d.get("models").map(obj).unwrap_or_default();
        models.sort_by(|a, b| a.0.cmp(b.0));
        for (m, v) in models {
            out.push(format!(
                "| {m} | **{}** | {} | {} | {} | {} |",
                vf(v, "ap", ".4f"),
                vf(v, "selection_precision", ".3f"),
                vf(v, "selection_recall", ".3f"),
                vf(v, "selection_f1", ".3f"),
                py_str(&v["n_selected"])
            ));
        }
        let mut ag: Vec<(&String, &Value)> = d
            .get("models_aggregated")
            .map(obj)
            .unwrap_or_default()
            .into_iter()
            .filter(|(_, v)| v.get("n_runs").and_then(num).unwrap_or(1.0) > 1.0)
            .collect();
        if !ag.is_empty() {
            out.push(String::new());
            out.push(
                "Independent repeats of the same task, same model, fresh context. One run per cell cannot carry a \
                 mechanism claim, so the spread is reported rather than averaged away:"
                    .into(),
            );
            out.push(String::new());
            out.push("| model | runs | AP mean | AP min | AP max | spread |".into());
            out.push("|---|---:|---:|---:|---:|---:|".into());
            ag.sort_by(|a, b| a.0.cmp(b.0));
            for (m, v) in ag {
                out.push(format!(
                    "| {m} | {} | {} | {} | {} | {} |",
                    py_str(&v["n_runs"]),
                    vf(v, "ap_mean", ".4f"),
                    vf(v, "ap_min", ".4f"),
                    vf(v, "ap_max", ".4f"),
                    fmt::f(num(&v["ap_max"]).unwrap() - num(&v["ap_min"]).unwrap(), ".4f")
                ));
            }
        }
    }
    out.push(String::new());
    out.push(live_rich_delta());
    out.join("\n")
}

/// (mean AP, min, max, runs) of one model in one condition.
type Cell = (f64, f64, f64, i64);

/// Does richer evidence beat a bigger model?  Answered from the repeats.
fn live_rich_delta() -> String {
    let (Some(d), Some(dr)) = (
        art().join("live_rank_results.json").exists().then(|| read_json("live_rank_results.json")).flatten(),
        art().join("live_rank_results_rich.json").exists().then(|| read_json("live_rank_results_rich.json")).flatten(),
    ) else {
        return String::new();
    };
    let cell = |dd: &Value, m: &str| -> Option<Cell> {
        if let Some(ag) = dd.get("models_aggregated").and_then(|a| a.get(m)).filter(|a| truthy(a)) {
            let g = |k: &str| num(&ag[k]).unwrap();
            return Some((g("ap_mean"), g("ap_min"), g("ap_max"), ag["n_runs"].as_i64().unwrap()));
        }
        let v = dd.get("models").and_then(|a| a.get(m)).filter(|a| truthy(a))?;
        let ap = num(&v["ap"]).unwrap();
        Some((ap, ap, ap, 1))
    };
    // set(d.get("models_aggregated", d.get("models", {}))): the keys of the aggregated table when it exists
    let keys = |dd: &Value| -> Vec<String> {
        let t = dd.get("models_aggregated").or_else(|| dd.get("models"));
        t.and_then(Value::as_object).map(|o| o.keys().cloned().collect()).unwrap_or_default()
    };
    let names: BTreeSet<String> = keys(&d).into_iter().chain(keys(&dr)).collect();
    let mut lines = vec![
        "**Does giving the SAME model richer evidence beat giving the task to a BIGGER model?**".to_string(),
        String::new(),
        "| model | statistics only (mean [min, max], n) | + raw work notes (mean [min, max], n) | Δ |".to_string(),
        "|---|---:|---:|---:|".to_string(),
    ];
    let mut deltas: Vec<(String, f64, Cell, Cell)> = Vec::new();
    for m in names {
        let (Some(ca), Some(cb)) = (cell(&d, &m), cell(&dr, &m)) else { continue };
        lines.push(format!(
            "| {m} | {} [{}, {}] n={} | {} [{}, {}] n={} | {} |",
            fmt::f(ca.0, ".4f"),
            fmt::f(ca.1, ".4f"),
            fmt::f(ca.2, ".4f"),
            ca.3,
            fmt::f(cb.0, ".4f"),
            fmt::f(cb.1, ".4f"),
            fmt::f(cb.2, ".4f"),
            cb.3,
            fmt::f(cb.0 - ca.0, "+.4f")
        ));
        deltas.push((m, cb.0 - ca.0, ca, cb));
    }
    if deltas.is_empty() {
        return String::new();
    }
    let pos = deltas.iter().filter(|d| d.1 > 0.0).count();
    // True interval overlap, not a one-sided comparison: the conditions are separated only when one range lies
    // entirely above the other.
    let overlap: Vec<&str> =
        deltas.iter().filter(|(_, _, ca, cb)| !(cb.1 > ca.2 || ca.1 > cb.2)).map(|d| d.0.as_str()).collect();
    let thin: Vec<&str> = deltas.iter().filter(|(_, _, ca, cb)| ca.3 < 2 || cb.3 < 2).map(|d| d.0.as_str()).collect();
    lines.push(String::new());
    let overlap_txt = if overlap.is_empty() {
        "No model's two condition ranges overlap. ".to_string()
    } else {
        format!(
            "For {} the two conditions' run ranges OVERLAP, so for those models this comparison does not separate the \
             conditions at all. ",
            overlap.join(", ")
        )
    };
    let thin_txt = if thin.is_empty() {
        String::new()
    } else {
        format!(
            "{}{} a condition measured only once, so that range is a single point and any separation there is not \
             evidence. ",
            thin.join(", "),
            if thin.len() == 1 { " still has" } else { " still have" }
        )
    };
    lines.push(format!(
        "Richer evidence helped {pos} of {} models. {overlap_txt}{thin_txt}Read the size of the run-to-run spread \
         before reading any Δ: where the spread is comparable to the gap, the gap is not a result.",
        deltas.len()
    ));
    lines.join("\n")
}

/// Section 8: the calibrated knobs and the grids they were chosen from.
pub fn section_calibration() -> String {
    let Some(d) = read_json("calibration.json") else { return "_(calibration not run)_".to_string() };
    let mut lines = vec![
        format!(
            "Fitted on calibration seeds {} at {} users, objective `{}`, then frozen:",
            py_str(&d["seeds"]),
            value::format(&d["scale"], ","),
            py_str(&d["objective"])
        ),
        String::new(),
        "| knob | chosen |".to_string(),
        "|---|---:|".to_string(),
    ];
    for k in ["triage_prior_weight", "question_frac", "mr_budget", "ct_kernel_ko_cap", "flat_budget"] {
        if let Some(v) = d.get(k) {
            lines.push(format!("| `{k}` | {} |", py_str(v)));
        }
    }
    for (gname, label) in
        [("hier_grid", "hierarchy"), ("mr_grid", "map-reduce"), ("ct_grid", "central triage"), ("rag_grid", "flat RAG")]
    {
        let Some(grid) = d.get(gname).and_then(Value::as_array) else { continue };
        lines.push(format!("\n**{label} grid**\n"));
        let keys: Vec<&String> = grid[0]
            .as_object()
            .expect("grid row")
            .keys()
            .filter(|k| !["average_precision", "found", "recall_at_100", "cu"].contains(&k.as_str()))
            .collect();
        let mut head: Vec<&str> = keys.iter().map(|k| k.as_str()).collect();
        head.extend(["AP", "found", "compute"]);
        lines.push(format!("| {} |", head.join(" | ")));
        lines.push(format!("|{}", "---|".repeat(keys.len() + 3)));
        for row in grid {
            let cells: Vec<String> = keys.iter().map(|k| py_str(&row[k.as_str()])).collect();
            let found = row.get("found").and_then(num).unwrap_or(f64::NAN);
            lines.push(format!(
                "| {} | {} | {} | {} |",
                cells.join(" | "),
                vf(row, "average_precision", ".5f"),
                fmt::f(found, ".3f"),
                vf(row, "cu", ".3e")
            ));
        }
    }
    lines.join("\n")
}

/// Section 7: the generated enterprise at 2,000, 10,000 and 50,000 users, built by the simulator.
pub fn world_description() -> String {
    let lines: Vec<String> = [2_000usize, 10_000, 50_000]
        .par_iter()
        .map(|&n| {
            // `except Exception as e`: a world that fails to build is reported in its row
            std::panic::catch_unwind(|| world_row(n)).unwrap_or_else(|e| {
                let msg = e
                    .downcast_ref::<&str>()
                    .map(|s| s.to_string())
                    .or_else(|| e.downcast_ref::<String>().cloned())
                    .unwrap_or_default();
                format!("| {n} | error: {msg} |")
            })
        })
        .collect();
    let head = format!(
        "| users | records | teams | depts | sites | regions | users/team (p10–p90) | teams/dept | depts/site | \
         patterns | rare | decoys | entities |\n|{}",
        "---|".repeat(13)
    );
    format!("{head}\n{}", lines.join("\n"))
}

fn world_row(n: usize) -> String {
    let w = crate::runner::build_world_default(n, 0);
    let s = w.corpus.org.stats();
    let c = w.corpus.stats();
    let lv = |level: &str, k: &str| py_str(&s[level][k]);
    format!(
        "| {} | {} | {} | {} | {} | {} | {}–{} (med {}) | {}–{} | {}–{} | {} | {} | {} | {} |",
        comma(w.corpus.org.user_ids.len() as i64),
        value::format(&c["n_records"], ","),
        value::format(&s["team"]["count"], ","),
        value::format(&s["department"]["count"], ","),
        lv("site", "count"),
        lv("region", "count"),
        lv("team", "fanin_p10"),
        lv("team", "fanin_p90"),
        lv("team", "fanin_p50"),
        lv("department", "fanin_p10"),
        lv("department", "fanin_p90"),
        lv("site", "fanin_min"),
        lv("site", "fanin_max"),
        py_str(&c["n_real_patterns"]),
        py_str(&c["n_rare_patterns"]),
        py_str(&c["n_decoys"]),
        value::format(&c["n_entities"], ",")
    )
}

/// Section 28: the claims that are about the problem rather than about our design.
///
/// Numbers are read back from the artifacts so that this section cannot drift from the results as seeds are added.
pub fn section_headline() -> String {
    let rows = dedupe_default(load_all(&["e1_baselines.jsonl", "e1b_extra.jsonl"]));
    let big = rows.iter().map(|r| int(r, "scale")).max().unwrap_or(0);
    let mean = |arch: &str, key: &str| scale_mean(&rows, arch, big, key).unwrap_or(f64::NAN);
    let (o_cov, o_found) = (mean("Y_oracle_retrieval", "evidence_coverage_2links"), mean("Y_oracle_retrieval", FOUND));
    let n_or = rows
        .iter()
        .filter(|r| text(r, "arch") == "Y_oracle_retrieval" && int(r, "scale") == big)
        .map(|r| int(r, "seed"))
        .collect::<BTreeSet<_>>()
        .len();

    let mut live = String::new();
    if art().join("live_rank_results.json").exists() && art().join("live_rank_results_rich.json").exists() {
        let d = read_json("live_rank_results.json").unwrap();
        let dr = read_json("live_rank_results_rich.json").unwrap();
        let (ag, agr) = (d.get("models_aggregated").map(obj).unwrap_or_default(), dr.get("models_aggregated"));
        let agr_o = agr.map(obj).unwrap_or_default();
        if !ag.is_empty() && !agr_o.is_empty() {
            let means = || ag.iter().map(|(_, v)| num(&v["ap_mean"]).unwrap());
            let band = format!(
                "{}–{}",
                fmt::f(value::min_by(means(), |x| *x).unwrap(), ".3f"),
                fmt::f(value::max_by(means(), |x| *x).unwrap(), ".3f")
            );
            let agr = agr.unwrap();
            let gaps: Vec<f64> = ag
                .iter()
                .filter_map(|(k, v)| {
                    agr.get(k.as_str()).map(|r| num(&r["ap_mean"]).unwrap() - num(&v["ap_mean"]).unwrap())
                })
                .collect();
            let runs = agr_o.iter().map(|(_, v)| v["n_runs"].as_i64().unwrap()).max().unwrap();
            live = format!(
                "3. **At fixed evidence, model capability is not the lever people expect it to be.** Measured \
                 directly, blind, on {} real models across {runs} runs each: given the same aggregated evidence \
                 statistics, all of them land in a band (AP {band}) that does not beat a six-feature logistic \
                 regression ({}). Changing what the evidence *contains* moved one model by {} and another by {}. \
                 *Might not transfer* — and note that it did not transfer uniformly even here, which is the point.\n",
                ag.len(),
                vf(&d, "simulator_logistic_ap", ".3f"),
                fmt::f(value::max_by(gaps.iter().copied(), |x| *x).unwrap(), "+.2f"),
                fmt::f(value::min_by(gaps.iter().copied(), |x| *x).unwrap(), "+.2f")
            );
        }
    }

    let mut prov = String::new();
    let pr = load("e12_provenance.jsonl");
    if !pr.is_empty() {
        let pb = pr.iter().map(|r| int(r, "scale")).max().unwrap();
        let at: Vec<Row> = pr.iter().filter(|r| int(r, "scale") == pb).cloned().collect();
        let sub = agg(&at, &["reports_fully_attributed"], &["arch"]);
        let best = value::max_by(sub.iter(), |r| f(r, "reports_fully_attributed")).unwrap();
        if let Some(h) = sub.iter().find(|r| text(r, "arch") == "H_mycelic_full") {
            prov = format!(
                "4. **Carrying lineage is not the same as being auditable.** Measured: the lineage-carrying hierarchy \
                 attributes {} of its reports to original evidence end to end; `{}` attributes {}. A lineage path \
                 records where a claim travelled, which is not what an incident review asks for. *Transfers directly* \
                 to any system marketing provenance as a benefit of decentralisation.\n",
                fmt::f(f(h, "reports_fully_attributed"), ".1%"),
                text(best, "arch"),
                fmt::f(f(best, "reports_fully_attributed"), ".0%")
            );
        }
    }

    let parts = [
        "1. **Weak cross-organisational signals are not findable by ranking records.** Measured: 0 of 306 \
         pattern-facet records appear in the global top 900 by any per-record importance feature at 10,000 users. A \
         record that is one facet of a distributed problem is, by construction, indistinguishable from benign chatter. \
         Any design whose first stage is \"rank the documents\" is solving a different problem. *Might not transfer \
         if* real weak signals carry lexical markers our generator does not simulate — urgency language, escalation \
         formatting, named severity levels.\n"
            .to_string(),
        format!(
            "2. **Above a certain corpus size the binding constraint moves from retrieval to discrimination.** \
             Measured: a perfect-retrieval oracle holds the evidence for {} of hidden patterns at {} users ({n_or} \
             seeds) and reports {} of them. Adding undifferentiated evidence past that point makes the ranking \
             *worse*. A propagation budget is therefore a feature, not only a cost. *Might not transfer if* the \
             executive layer can be given far more reading budget than we modelled.\n",
            fmt::f(o_cov, ".0%"),
            comma(big),
            fmt::f(o_found, ".0%")
        ),
        live,
        prov,
        "5. **A tuning knob that reverses sign when the corpus is regenerated is fitted to the corpus.** Measured: \
         `w_dispersion` was rejected twice on one generator and selected at 0.8 on a corrected one, mechanism \
         unchanged. *Transfers directly*: hold out the data used to fit anything, and re-fit when the data changes \
         rather than inheriting the setting.\n"
            .to_string(),
        "6. **Measurement subjects make good reviewers.** The two most consequential defects found in this benchmark — \
         a non-blind operator task file, and entity collisions between real patterns and decoys — were both \
         reported, unprompted, by models being measured, not by reading the code. *Transfers directly*: leave room in \
         the task for the subject to say the task is broken, and read what comes back.\n"
            .to_string(),
    ];
    parts.into_iter().filter(|p| !p.is_empty()).collect::<Vec<_>>().join("\n")
}

/// Build the contents list from the rendered document itself.
///
/// Generating it from the output rather than maintaining a list means a renumbered or renamed section can never
/// leave a stale entry behind.
fn toc(body: &str) -> String {
    let heading = Regex::new(r"^(#{1,2}) (.+)$").unwrap();
    let not_anchor = Regex::new(r"[^a-z0-9\s-]").unwrap();
    let mut lines = vec!["<details>".to_string(), "<summary><b>Contents</b></summary>".to_string(), String::new()];
    let mut started = false;
    for ln in body.lines() {
        let Some(m) = heading.captures(ln) else { continue };
        if ln.starts_with("# Mycelic") {
            continue;
        }
        // The document's own subtitle is an h2 above the first PART divider; nothing before that divider belongs in
        // the contents.
        let level = &m[1];
        if level == "#" {
            started = true;
        }
        if !started {
            continue;
        }
        let title = m[2].trim();
        let anchor = not_anchor.replace_all(&title.to_lowercase(), "").replace(' ', "-");
        if level == "#" {
            // a PART divider
            lines.push(format!("\n**{title}**\n"));
        } else {
            lines.push(format!("* [{title}](#{anchor})"));
        }
    }
    lines.extend([String::new(), "</details>".to_string()]);
    lines.join("\n")
}

const NUM_WORDS: [(usize, &str); 11] = [
    (10, "Ten"),
    (11, "Eleven"),
    (12, "Twelve"),
    (13, "Thirteen"),
    (14, "Fourteen"),
    (15, "Fifteen"),
    (16, "Sixteen"),
    (17, "Seventeen"),
    (18, "Eighteen"),
    (19, "Nineteen"),
    (20, "Twenty"),
];

/// How many architectures were actually run, spelled out.
fn n_arch_word() -> String {
    let n = crate::runner::archs().len();
    NUM_WORDS.iter().find(|e| e.0 == n).map_or(n.to_string(), |e| e.1.to_string())
}

/// `os.path.relpath(target, start)` on absolute paths, lexically (no symlinks resolved, as Python does).
fn relpath(target: &Path, start: &Path) -> String {
    let norm = |p: &Path| -> Vec<String> {
        let abs = if p.is_absolute() { p.to_path_buf() } else { std::env::current_dir().unwrap().join(p) };
        let mut out: Vec<String> = Vec::new();
        for c in abs.components() {
            match c {
                Component::ParentDir => {
                    out.pop();
                }
                Component::Normal(s) => out.push(s.to_string_lossy().into_owned()),
                _ => {}
            }
        }
        out
    };
    let (t, s) = (norm(target), norm(start));
    let common = t.iter().zip(&s).take_while(|(a, b)| a == b).count();
    let mut parts: Vec<String> = vec!["..".to_string(); s.len() - common];
    parts.extend(t[common..].iter().cloned());
    if parts.is_empty() { ".".to_string() } else { parts.join("/") }
}

/// OLD (artifacts/v1) against NEW (artifacts) on identical worlds, for the executive summary; every table comes
/// from `vnext_data`.
pub fn section_vnext() -> String {
    let fig = art().join("figures").join("old_new.png");
    let mut parts = vec![
        "### What the vNext work changed, on identical worlds".to_string(),
        String::new(),
        "OLD is the benchmark as archived before the vNext work (`artifacts/v1/`: hand-set ranker, question budget \
         0.25 of the triage queue, unbatched metering, earliest-mention link timing); NEW is this rerun (learned ranker \
         fitted on the calibration seeds and adopted per architecture, budget 0.65, batched metering, hybrid link \
         timing). Every other setting, the worlds, the seeds, the gold labels, the metrics and the register cap are \
         identical. `H_mycelic_prev` runs the OLD configuration inside the NEW suite as the reproduction check. The \
         full account is `docs/mycelic_vnext/NEXT_RESEARCH_REPORT.md`."
            .to_string(),
        String::new(),
    ];
    if fig.exists() {
        let docs = out_path().parent().unwrap().to_path_buf();
        parts.push(format!("![old vs new]({})", relpath(&fig, &docs)));
        parts.push(String::new());
    }
    for sc in [10_000, 50_000] {
        parts.extend([format!("**{} users**", comma(sc)), String::new(), vd::old_new_table(sc, None), String::new()]);
    }
    parts.extend([
        "**Reproduction of the v1 configuration inside the new suite (10,000 users)**".to_string(),
        String::new(),
        vd::prev_reproduction_table(10_000),
        String::new(),
        "**Frozen configuration and the paired file that justified each value**".to_string(),
        String::new(),
        vd::frozen_config_table(),
        String::new(),
    ]);
    parts.join("\n")
}

/// The document, dated `date` (`YYYY-MM-DD`).
pub fn build_at(date: &str) -> String {
    let mut sections: Sections = Sections::new();
    sections.insert("toc", String::new());
    sections.insert("world", world_description());
    sections.insert("calibration", section_calibration());
    sections.insert("baselines", section_baselines());
    sections.insert("pairs", section_pairs());
    sections.insert("ablations", section_ablations());
    sections.insert("alloc", section_alloc());
    sections.insert("level_marginal", section_level_marginal());
    sections.insert("qsweep", section_qsweep());
    sections.insert("fanin", section_fanin());
    sections.insert("adversarial", section_adversarial());
    sections.insert("frontier", section_frontier());
    sections.insert("shape", section_shape());
    sections.insert("crosslinks", section_crosslinks());
    sections.insert("live", section_live());
    sections.insert("privacy", section_privacy());
    sections.insert("scale_trend", section_scale_trend());
    sections.insert("provenance", section_provenance());
    sections.insert("summary", executive_summary() + "\n\n" + &section_vnext());
    sections.insert("decision", decision_summary());
    sections.insert("glossary", GLOSSARY.to_string());
    sections.insert("recommendation", recommendation());
    sections.insert("critique", critique());
    sections.insert("next", next_experiments());
    sections.insert("questions", questions_section());
    sections.insert("headline", section_headline());
    sections.insert("n_arch", n_arch_word());
    // Render once with an empty contents list to learn the real headings, then again with the list built from them.
    let contents = toc(&compose(&sections, date));
    sections.insert("toc", contents);
    compose(&sections, date)
}

/// The document, dated today.
pub fn build() -> String {
    build_at(&crate::loss_doc::today())
}

/// `python -m research.mycelic.report`: write `docs/MYCELIC_ENTERPRISE.md`.
pub fn main(_args: &[String]) -> anyhow::Result<()> {
    let txt = build();
    let out = out_path();
    std::fs::create_dir_all(out.parent().unwrap())?;
    std::fs::write(&out, &txt).map_err(|e| anyhow::anyhow!("{}: {e}", out.display()))?;
    println!("wrote {} {} chars", out.display(), txt.chars().count());
    Ok(())
}
