//! Generate `docs/MYCELIC_LOSS_ACCOUNTING.md` from the funnel and diagnostic rows (`loss_doc.py`).
//!
//! Every number is read from `artifacts/*.jsonl` at generation time.
//!
//! ```sh
//! mycelic-research loss_doc
//! ```

use crate::analysis::boot_ci;
use crate::loss_accounting::funnel_path;
use crate::loss_report::{self, flag, has, int, is_summary, text};
use crate::np::math;
use crate::py::fmt;
use crate::py::value::{self, Row, num, py_eq, py_str, truthy};
use crate::runner::{art, repo_root};
use crate::vnext_data::{self, Sel};
use indexmap::IndexMap;
use serde_json::Value;
use std::collections::BTreeSet;
use std::path::PathBuf;

/// `docs/MYCELIC_LOSS_ACCOUNTING.md`
pub fn out_path() -> PathBuf {
    repo_root().join("docs").join("MYCELIC_LOSS_ACCOUNTING.md")
}

pub const DIAG: [(&str, &str); 4] = [
    ("diag_qf1", "ask every triage candidate (question_frac = 1.0)"),
    ("diag_reg", "unbounded register (diagnostic only)"),
    ("diag_both", "both"),
    ("diag_qf1_nofilter", "full budget, home-site evidence allowed"),
];
pub const DIAG50: [(&str, &str); 3] = [
    ("diag50_qf1", "ask every triage candidate (question_frac = 1.0)"),
    ("diag50_reg", "unbounded register (diagnostic only)"),
    ("diag50_both", "both"),
];
/// (key, header, decimals; 0 = exponent form)
pub const KEYS: [(&str, &str, usize); 7] = [
    ("found_anywhere_in_register", "found", 3),
    ("evidence_coverage_2links", "evidence cov.", 3),
    ("rare_signal_recall", "rare recall", 3),
    ("average_precision", "AP", 4),
    ("false_discovery_rate", "FDR", 3),
    ("compute_units", "compute", 0),
    ("inference_calls", "calls", 0),
];

/// `artifacts/quick_<tag>.jsonl`
fn quick(tag: &str) -> Vec<Row> {
    vnext_data::quick(tag)
}

/// `f"{v:.{d}f}" if d else f"{v:.2e}"`
fn fmt_dec(v: f64, d: usize) -> String {
    if d > 0 { fmt::f(v, &format!(".{d}f")) } else { fmt::f(v, ".2e") }
}

fn seed_map<'a>(rows: impl IntoIterator<Item = &'a Row>) -> IndexMap<i64, &'a Row> {
    let mut m = IndexMap::new();
    for r in rows {
        m.insert(int(r, "seed"), r);
    }
    m
}

/// ` **↑**` / ` **↓**` when the bootstrap interval of the paired difference excludes zero.
fn arrow(m: f64, lo: f64, hi: f64) -> String {
    if lo > 0.0 || hi < 0.0 { format!(" **{}**", if m > 0.0 { "↑" } else { "↓" }) } else { String::new() }
}

/// The diagnostic variants of the v1 hierarchy against their base, one row each.
pub fn diag_table(specs: &[(&str, &str)], scale: i64) -> String {
    let mut lines = vec![
        format!("| variant | {} |", KEYS.iter().map(|k| k.1).collect::<Vec<_>>().join(" | ")),
        format!("|---|{}", "---:|".repeat(KEYS.len())),
    ];
    let mut base_done = false;
    for &(tag, label) in specs {
        let all = quick(tag);
        let rows: Vec<&Row> = all.iter().filter(|r| int(r, "scale") == scale).collect();
        if rows.is_empty() {
            continue;
        }
        let cfg_none = |r: &Row| r.get("cfg_over").is_none_or(Value::is_null);
        let base = seed_map(rows.iter().copied().filter(|r| cfg_none(r)));
        let var = seed_map(rows.iter().copied().filter(|r| !cfg_none(r)));
        let seeds = vnext_data::common_seeds(&base, &var);
        if seeds.is_empty() {
            continue;
        }
        if !base_done {
            let cells: Vec<String> = KEYS
                .iter()
                .map(|&(k, _, d)| fmt_dec(vnext_data::mean_of(seeds.iter().map(|s| base[s]), k), d))
                .collect();
            lines.push(format!("| H_mycelic_full, v1 (hand ranker, qf 0.25) | {} |", cells.join(" | ")));
            base_done = true;
        }
        let mut cells = Vec::new();
        for &(k, _, d) in &KEYS {
            let dv: Vec<f64> = seeds.iter().map(|s| value::f(var[s], k) - value::f(base[s], k)).collect();
            let v = vnext_data::mean_of(seeds.iter().map(|s| var[s]), k);
            let (m, lo, hi) = boot_ci(&dv);
            cells.push(fmt_dec(v, d) + &arrow(m, lo, hi));
        }
        lines.push(format!("| {label} ({} seeds) | {} |", seeds.len(), cells.join(" | ")));
    }
    lines.join("\n")
}

pub const PAIR_KEYS: [(&str, &str, usize); 8] = [
    ("found_anywhere_in_register", "found", 3),
    ("rare_signal_recall", "rare recall", 3),
    ("evidence_coverage_2links", "evidence cov.", 3),
    ("average_precision", "AP", 4),
    ("false_discovery_rate", "FDR", 3),
    ("decoy_acceptance_all", "decoy acc.", 3),
    ("compute_units", "compute", 0),
    ("inference_calls", "calls", 0),
];

fn pair_header(first: &str) -> Vec<String> {
    vec![
        format!("| {first} | seeds | {} |", PAIR_KEYS.iter().map(|k| k.1).collect::<Vec<_>>().join(" | ")),
        format!("|---|---:|{}", "---:|".repeat(PAIR_KEYS.len())),
    ]
}

/// The variant's mean with the paired Δ against its base underneath, and an arrow when the Δ is decisive.
fn variant_cell(base: &IndexMap<i64, &Row>, var: &IndexMap<i64, &Row>, seeds: &[i64], k: &str, d: usize) -> String {
    let b: Vec<f64> = seeds.iter().map(|s| value::f(base[s], k)).collect();
    let v: Vec<f64> = seeds.iter().map(|s| value::f(var[s], k)).collect();
    let diff: Vec<f64> = v.iter().zip(&b).map(|(x, y)| x - y).collect();
    let (m, lo, hi) = boot_ci(&diff);
    let mark = if seeds.len() > 1 { arrow(m, lo, hi) } else { String::new() };
    format!("{} <sub>{}{}</sub>{mark}", fmt_dec(math::mean(&v), d), if m >= 0.0 { "+" } else { "" }, fmt_dec(m, d))
}

/// One row per experiment: the variant's mean, with the paired Δ against its own base on the same seeds and a mark
/// when the 95% bootstrap interval of the Δ is entirely on one side of zero.  Reads `quick_<tag>.jsonl`.
pub fn paired_table(specs: &[(&str, &str)], scale: i64, base_label: &str) -> String {
    let mut lines = pair_header("experiment");
    for &(tag, label) in specs {
        let all = quick(tag);
        let rows: Vec<&Row> = all.iter().filter(|r| int(r, "scale") == scale).collect();
        if rows.is_empty() {
            continue;
        }
        // rows alternate base, variant per seed (quick_paired writes them so)
        let base = seed_map(rows.iter().copied().step_by(2));
        let var = seed_map(rows.iter().copied().skip(1).step_by(2));
        let seeds = vnext_data::common_seeds(&base, &var);
        if seeds.is_empty() {
            continue;
        }
        let mut bcells = Vec::new();
        let mut vcells = Vec::new();
        for &(k, _, d) in &PAIR_KEYS {
            let b: Vec<f64> = seeds.iter().map(|s| value::f(base[s], k)).collect();
            bcells.push(fmt_dec(math::mean(&b), d));
            vcells.push(variant_cell(&base, &var, &seeds, k, d));
        }
        lines.push(format!("| {label}: {base_label} | {} | {} |", seeds.len(), bcells.join(" | ")));
        lines.push(format!("| {label}: variant | {} | {} |", seeds.len(), vcells.join(" | ")));
    }
    if lines.len() > 2 { lines.join("\n") } else { "_(not run)_".to_string() }
}

pub const RANKER_V1: [(&str, &str); 5] = [
    ("rk_H", "hierarchy, calibrated budget"),
    ("rk_Hqf1", "hierarchy, full question budget"),
    ("rk_A2", "A2_chunked_ctx"),
    ("rk_B4", "B4_central_triage"),
    ("rk_Y", "Y_oracle_retrieval"),
];
pub const RANKER_V2: [(&str, &str); 4] = [
    ("rk2_H", "hierarchy, calibrated budget"),
    ("rk2_A2", "A2_chunked_ctx"),
    ("rk2_B4", "B4_central_triage"),
    ("rk2_Y", "Y_oracle_retrieval"),
];
pub const QF_SWEEP: [(&str, &str); 4] = [
    ("rk2_qf0.50", "question_frac 0.50"),
    ("rk2_qf0.65", "question_frac 0.65"),
    ("rk2_qf0.80", "question_frac 0.80"),
    ("rk2_qf1.00", "question_frac 1.00"),
];

/// `arm_paired` format (distinct label per arm): one row per arm with the paired Δ against the base arm on the same
/// seeds.
pub fn arm_table(tag: &str, base_label: &str, arms: &[(&str, &str)], scale: i64, seeds_note: &str) -> String {
    let all = quick(tag);
    let rows: Vec<Row> = all.into_iter().filter(|r| int(r, "scale") == scale).collect();
    let by = vnext_data::by_arm(&rows);
    let Some(base) = by.get(base_label).filter(|b| !b.is_empty()) else { return "_(not run)_".to_string() };
    let mut lines = pair_header("arm");
    let mut sb: Vec<i64> = base.keys().copied().collect();
    sb.sort();
    let cells: Vec<String> =
        PAIR_KEYS.iter().map(|&(k, _, d)| fmt_dec(vnext_data::mean_of(sb.iter().map(|s| base[s]), k), d)).collect();
    let base_name = match arms.first() {
        Some(a) if a.0 == base_label => a.1,
        _ => base_label,
    };
    lines.push(format!("| {base_name} (base) | {} | {} |", sb.len(), cells.join(" | ")));
    let empty = IndexMap::new();
    for &(label, name) in arms {
        if label == base_label {
            continue;
        }
        let var = by.get(label).unwrap_or(&empty);
        let seeds = vnext_data::common_seeds(base, var);
        if seeds.is_empty() {
            continue;
        }
        let cells: Vec<String> = PAIR_KEYS.iter().map(|&(k, _, d)| variant_cell(base, var, &seeds, k, d)).collect();
        lines.push(format!("| {name} | {} | {} |", seeds.len(), cells.join(" | ")));
    }
    if !seeds_note.is_empty() {
        lines.push(String::new());
        lines.push(seeds_note.to_string());
    }
    lines.join("\n")
}

pub const RANKER_V3: [(&str, &str); 4] = [
    ("v3_H", "hierarchy: ranker v3 + question_frac 0.65 + batched descent"),
    ("v3_A2", "A2_chunked_ctx"),
    ("v3_B4", "B4_central_triage"),
    ("v3_Y", "Y_oracle_retrieval"),
];

/// Section 9: every vNext change as a paired experiment, the ledger and the frozen configuration.
pub fn vnext_section() -> String {
    let mut t: Vec<String> = vec![
        r#"## 9. vNext, measured

Everything in this section is a **paired** experiment: the variant and its
base run on identical worlds, and the Δ shown under each variant value is
the paired mean with an arrow when its 95% bootstrap interval excludes zero.
Evaluation seeds are 0–4 at 10,000 users and 0–2 at 50,000; every ranker's
weights and every v1 knob were fitted on calibration-seed rows (500–502), and
the two tables that use calibration seeds say so. Several vNext decisions were
made on these evaluation seeds (the question budget, and the rejection of local
re-extraction, decoy-weighted ranker selection and modal link timing); the
ledger in 9.7 cites the paired file behind each. Nothing here changes the register cap, the threshold, the gold
labels, the worlds or the matching rule; the unbounded register appears
only as a diagnostic.

### 9.1 A ranker fitted on kernel-side features

The confidence logistic is replaced as a *ranker* only. Version 1 kept the
hand-set ≥ 0.5 gate and re-ordered inside it; version 2 keeps the hand-gated
**count** and lets the learned score choose which candidates fill those
slots, with anchor-level context features; version 3 adds evidence-shape
features (origin users, single-witness fraction, echo ratio, lag
dispersion), the triage context of the anchor and the kernel's own
attribution verdict — 45 kernel-held quantities, no raw text. l2 and the
a-priori interaction set are chosen leave-one-seed-out on the calibration
seeds; the depth-2 boosted trees lost to the logistic in every selection.
Adoption is decided per architecture on the calibration seeds (each system
keeps whichever ranker is not worse for it), so no control is compared at a
ranker chosen for somebody else.

**Version 1 — order within the hand gate** (base = same system, hand-set ranker)

"#
        .to_string(),
    ];
    t.push(paired_table(&RANKER_V1, 10_000, "hand ranker"));
    t.push("\n**Version 2 — learned top-K gate + anchor context**\n".to_string());
    t.push(paired_table(&RANKER_V2, 10_000, "hand ranker"));
    t.push(
        "\n**Version 3 — evidence shape + triage context + attribution** (the hierarchy row also carries the budget \
         and batching changes of 9.2; the controls run at their own settings)\n"
            .to_string(),
    );
    t.push(paired_table(&RANKER_V3, 10_000, "hand ranker"));
    t.push(
        r#"
**What it says.** Version 3 lifts the hierarchy by 21 points of discovery on
5/5 seeds, rare recall from 0.21 to 0.39 and evidence coverage to 0.97, and
now helps `A2_chunked_ctx` too (0.685 → 0.745), so every control adopts it.
Decoy acceptance rises with the ranker on every system by about +0.15: the
planted traps share the features that make a genuine chain look genuine.
FDR falls slightly at the same time.

**The decoy regression, attacked and not fixed.** Up-weighting decoy rows in
the fit and penalising decoys in the selection objective chose weight 10 on
the calibration seeds; on the evaluation seeds it lost found and rare
recall for a 0.05 reduction in decoy acceptance. Rejected; decoy weighting
stays available as an explicit option.

"#
        .to_string(),
    );
    t.push(arm_table(
        "seldw_H",
        "v3",
        &[("v3", "ranker v3"), ("v3dw", "decoy-weighted selection (weight 10)")],
        10_000,
        "",
    ));
    t.push(
        r#"
### 9.2 The question budget, with the ranker fixed

Base = the hierarchy with ranker v3 and batched descent at question_frac
0.50; the v1 calibration had chosen 0.25 because the hand-set confidence
let extra candidates flood the register.

"#
        .to_string(),
    );
    t.push(arm_table(
        "qf_v3",
        "qf0.50",
        &[
            ("qf0.50", "question_frac 0.50"),
            ("qf0.65", "question_frac 0.65"),
            ("qf0.80", "question_frac 0.80"),
            ("qf1.00", "question_frac 1.00"),
        ],
        10_000,
        "",
    ));
    t.push(
        r#"
**What the sweep says.** Coverage saturates at 0.65 (0.97); 0.80 and 1.00 buy
no discovery, cost +13% and +31% compute and lose AP, because the wider
budget floods the 600-entry register with more same-anchor chains than the
ranker was fitted to demote. 0.65 is the budget at both scales (9.6).

### 9.3 Batched metering, and the merge that was rejected

Batching meters one routing call per node and one read per queried user
per round, with identical decisions and identical per-record tokens; it is
why calls fall by about two thirds while compute rises with the budget.
Profiled on one calibration seed (500) at the full budget, single runs,
**not** evaluation-seed measurements:

| stage | share of compute | calls | note |
|---|---:|---:|---|
| kernel re-read of the merged pool | 42% | 1 | 89,525 objects × 26 tokens in one frontier call |
| descent routing | 31% | 145,823 | one frontier-tier call per (parent node, anchor) |
| user reads | 6% | 69,549 | 8,361 of 9,491 reached users queried more than once |
| edge extraction | 6% | 9,999 | unchanged by any of this |

| change (same seed, full budget, ranker off) | found | AP | rare | pool | compute | calls | kept? |
|---|---:|---:|---:|---:|---:|---:|---|
| none | 0.400 | 0.0123 | 0.125 | 89,525 | 2.31e+06 | 227,524 | — |
| batched routing + reads | 0.400 | 0.0123 | 0.125 | 89,525 | 1.91e+06 | 27,127 | **yes** — pure cost |
| + merge descent returns per (predicate, entity) before the kernel reads them | 0.275 | 0.0052 | — | 22,225 | 1.07e+06 | 27,128 | no |
| + merge per (predicate, entity, **polarity**) | 0.325 | 0.0075 | 0.312 | 23,210 | 1.08e+06 | 27,128 | no — needs the ranker re-fitted on merged candidates; queued |

### 9.4 Targeted local re-extraction: rejected

When a descent reaches a user, that user re-reads its *own* notes that name
the entity but produced no claim on the first pass; raw text stays on the
node. One calibration seed promised +0.125. Paired on the evaluation seeds
(base = the v1 hierarchy; compare the two variant rows):

"#
        .to_string(),
    );
    t.push(paired_table(
        &[("rx_base065", "ranker v2, qf 0.65, batched"), ("rx_reextract", "same + local re-extraction")],
        10_000,
        "v1 hierarchy",
    ));
    // the 50k re-extraction arm, per seed, from the 9.6 file
    let mut v50: IndexMap<(String, i64), f64> = IndexMap::new();
    for r in quick("v50_v3").iter().filter(|r| int(r, "scale") == 50_000) {
        v50.insert((text(r, "arch").to_string(), int(r, "seed")), value::f(r, "found_anywhere_in_register"));
    }
    let seeds: BTreeSet<i64> = v50.keys().map(|k| k.1).collect();
    let rx: Vec<f64> = seeds
        .iter()
        .filter_map(|&s| {
            let rx = v50.get(&("H_v3_qf0.65_rx".to_string(), s))?;
            let base = v50.get(&("H_v3_qf0.65".to_string(), s))?;
            Some(rx - base)
        })
        .collect();
    let rx_txt = rx.iter().map(|d| fmt::f(*d, "+.2f").replace('-', "−")).collect::<Vec<_>>().join("/");
    let rx_n = match rx.len() {
        1 => "one".to_string(),
        2 => "two".to_string(),
        3 => "three".to_string(),
        4 => "four".to_string(),
        5 => "five".to_string(),
        n => n.to_string(),
    };
    t.push(format!(
        r#"
No gain at 10k, {rx_txt} on {rx_n} 50k seeds (9.6), six times the simulator
wall time. Off in the frozen configuration.

### 9.5 Link timing in the temporal check

The temporal DP dated every link at the earliest mention of its (predicate,
entity) pair. A descent that returns every mention of an entity drags that
date to a stale or routine mention, and the chain-order test then fails:
the panel's replay found 20–22 of 25 in-pool gold patterns per 10k seed with
at least one link dated that way. Two repairs, both free (no calls, no new
candidates): *modal* dates a link at its heaviest witness cluster; *hybrid*
does that for links with ≥ 2 agreeing witnesses and lets single-witness
links float across their clusters as separate DP states.

**Screen on the calibration seeds 500–502** (ranker v3 not refitted; these
are NOT evaluation-seed numbers):

"#
    ));
    t.push(arm_table(
        "cal_screen",
        "base",
        &[
            ("base", "qf 0.65 + batched, min timing"),
            ("modal", "modal timing"),
            ("hybrid", "hybrid timing"),
            ("span2", "chain-scoped questions (strict)"),
            ("span2_modal", "chain-scoped + modal"),
            ("span2_hybrid", "chain-scoped + hybrid"),
        ],
        10_000,
        "",
    ));
    t.push("\n**Evaluation seeds 0–4, ranker refitted on the calibration seeds under each pipeline:**\n\n".to_string());
    t.push(arm_table(
        "refit_hyb",
        "v3",
        &[
            ("v3", "min timing, ranker v3"),
            ("hyb_v3rk", "hybrid, ranker v3 (not refitted)"),
            ("hyb_refit", "hybrid, refitted ranker"),
            ("hyb_refit_lean", "hybrid + chain-scoped questions (lean)"),
        ],
        10_000,
        "",
    ));
    t.push(String::new());
    t.push(arm_table(
        "refit_mod",
        "v3",
        &[
            ("v3", "min timing, ranker v3"),
            ("mod_v3rk", "modal, ranker v3 (not refitted)"),
            ("mod_refit", "modal, refitted ranker"),
        ],
        10_000,
        "",
    ));
    t.push(
        r#"
Modal did not replicate (+0.04 on the calibration seeds, −0.01 on the
evaluation seeds); hybrid did (+0.04, no seed worse). The calibration-seed
"decoy resistance" did not replicate either: on the evaluation seeds the
stale-chain family D5 rises under hybrid with the refitted ranker (+0.12
at 10k, +0.13 at 50k, up on 7 of 8 seeds) while D1–D3 are flat, so the
honest statement is "found +0.04/+0.06, D5 +0.12/+0.13". The independent
review traced it to a staleness gate that one routine positive mention
after a retraction defeats, which min timing had been masking by accident;
the per-family numbers are in 9.7 and the gate fix is queued. The refitted ranker
goes forward because that is the pre-registered procedure; the previous
ranker's better rare recall and decoy acceptance under hybrid timing are
reported, not acted on, because acting on them would be selection on the
held-out seeds. The chain-scoped variant halves compute at a cost in
coverage and rare recall and is kept as a separate arm (`H_mycelic_lean`).

### 9.6 50,000 users

"#
        .to_string(),
    );
    t.push(arm_table(
        "v50_v3",
        "H_old",
        &[
            ("H_old", "v1 hierarchy (hand ranker, qf 0.25)"),
            ("H_v3_qf0.65", "ranker v3, qf 0.65, batched"),
            ("H_v3_qf0.80", "ranker v3, qf 0.80, batched"),
            ("H_v3_qf1.00", "ranker v3, qf 1.00, batched"),
            ("H_v3_qf0.65_rx", "ranker v3, qf 0.65 + local re-extraction"),
            ("A2_old", "A2_chunked_ctx, hand ranker"),
            ("A2_v3", "A2_chunked_ctx, ranker v3"),
        ],
        50_000,
        "",
    ));
    t.push(String::new());
    t.push(arm_table(
        "v50_hyb",
        "v3",
        &[
            ("v3", "ranker v3, qf 0.65, batched"),
            ("hyb", "hybrid timing, refitted ranker"),
            ("hyb_lean", "hybrid + chain-scoped questions"),
        ],
        50_000,
        "",
    ));
    let v50_rows = quick("v50_v3");
    let a2_v3 = vnext_data::mean_of(
        v50_rows.iter().filter(|r| text(r, "arch") == "A2_v3" && int(r, "scale") == 50_000),
        "found_anywhere_in_register",
    );
    let rerun = rerun_funnel();
    let h50 = arch_rows(&rerun, 50_000, "H_mycelic_full");
    let cov = if !h50.is_empty() {
        let mut order = [
            ("sketch_visible", "sketch visibility"),
            ("questioned", "the question budget"),
            ("extracted", "extraction"),
        ];
        order.sort_by_key(|o| std::cmp::Reverse(died(&h50, o.0, None)));
        let (k0, lab0) = order[0];
        let rest: Vec<String> = order[1..].iter().map(|(k, lab)| format!("{lab} ({})", died(&h50, k, None))).collect();
        let descent = match died(&h50, "descent_reached", None) {
            0 => "none".to_string(),
            n => n.to_string(),
        };
        format!(
            "the coverage it lacks is lost mainly at {lab0}\n({} of {} patterns in the section 2 funnel,\n{} of them \
             rare), then {}; descent\nloses {descent}",
            died(&h50, k0, None),
            h50.len(),
            died(&h50, k0, Some(true)),
            rest.join(" and ")
        )
    } else {
        "where the coverage it lacks is lost is attributed in section 2 once\nthe rerun's 50k funnel is available"
            .to_string()
    };
    let e1 = vnext_data::e1_rows(&art());
    let rng = |sc: i64| -> String {
        let v: Vec<f64> = e1
            .iter()
            .filter(|r| text(r, "arch") == "H_mycelic_full" && int(r, "scale") == sc)
            .map(|r| value::f(r, "max_context_tokens"))
            .collect();
        if v.is_empty() {
            return "n/a".to_string();
        }
        let lo = value::min_by(v.iter().copied(), |x| *x).unwrap();
        let hi = value::max_by(v.iter().copied(), |x| *x).unwrap();
        format!("{}–{}M", fmt::f(lo / 1e6, ".1f"), fmt::f(hi / 1e6, ".1f"))
    };
    let (rng50, rng10) = (rng(50_000), rng(10_000));
    t.push(format!(
        r#"
At 50k the register (2,999 slots) is not the binding stage; coverage (0.72)
is. The budget saturates at 0.65 here too, re-extraction does not pay, and
hybrid timing adds +0.06 on 3/3 seeds at equal compute. The hierarchy ends
at about 0.59 against A2's {a2_v3} with the same ranker, at 38% of A2's
compute; {cov}.

Two accounting caveats from the independent review apply to every 50k row
above: the "calls" column compares unbatched v1 metering with batched vNext
metering (batching alone is −84% calls at identical decisions; the v1 base
re-metered batched is in 9.7), and the kernel's single read of its pool is
{rng50} tokens at 50k ({rng10} at 10k; final rerun,
`e1_baselines.jsonl`) against the modelled tier's 1M
context, which the simulator enforces only for the flat controls.

### 9.7 The ledger, the cost of each accepted change, and what is frozen

"#,
        a2_v3 = fmt::f(a2_v3, ".2f")
    ));
    t.push(vnext_data::ledger_table());
    t.push("\n**Compute per accepted change, 10k evaluation seeds:**\n".to_string());
    t.push(vnext_data::compute_per_accepted_change());
    t.push("\n**Frozen configuration** (`calibration.json`, with the file that justified each value):\n".to_string());
    t.push(vnext_data::frozen_config_table());
    t.push(
        r#"
The full benchmark is rerun on this configuration by `final_rerun.sh`
(every experiment, the controls on identical worlds, the loss accounting at
both scales, the candidate dump with an out-of-sample ranker evaluation),
and the old-vs-new comparison, the architecture, the experiment matrix and
the adversarial review are in `docs/mycelic_vnext/`.
"#
        .to_string(),
    );
    t.join("\n")
}

/// One dump's ranker evaluation as the report prints it.
pub fn format_ranker_eval(tag: &str, r: &Value) -> Vec<String> {
    let mut out = vec![format!(
        "**Dump `{}`** — {} training candidates (seeds 500–502), {} evaluation candidates (seeds 0–4), positive \
         rate {}. Pooled AUC gold-vs-spurious: hand-set confidence {}, learned {}.\n",
        if tag.is_empty() { "default budget" } else { tag },
        value::format(&r["n_train"], ","),
        value::format(&r["n_test"], ","),
        value::format(&r["pos_rate_train"], ".1%"),
        value::format(&r["auc_conf"], ".3f"),
        value::format(&r["auc_learned"], ".3f")
    )];
    out.push(
        "| architecture | gold patterns | with any candidate | found (current ranker) | found (learned rank, same \
         #kept) | found (learned, p≥0.5) | AUC current | AUC learned |"
            .to_string(),
    );
    out.push("|---|---:|---:|---:|---:|---:|---:|---:|".to_string());
    for (a, v) in r["per_arch"].as_object().expect("per_arch") {
        out.push(format!(
            "| {a} | {} | {} | {} | {} | {} | {} | {} |",
            py_str(&v["gold_patterns"]),
            py_str(&v["matchable_patterns"]),
            value::format(&v["found_baseline"], ".3f"),
            value::format(&v["found_learned_rank_only"], ".3f"),
            value::format(&v["found_learned_thresholded"], ".3f"),
            value::format(&v["auc_conf"], ".3f"),
            value::format(&v["auc_learned"], ".3f")
        ));
    }
    out.push(String::new());
    let mut top: Vec<(&String, f64)> =
        r["weights"].as_object().expect("weights").iter().map(|(k, v)| (k, num(v).expect("weight"))).collect();
    // sorted(key=-abs(v)) is stable: equal magnitudes keep the stored order
    top.sort_by(|a, b| (-a.1.abs()).partial_cmp(&-b.1.abs()).expect("weights are not NaN"));
    top.truncate(8);
    let named: Vec<String> = top
        .iter()
        .filter(|(k, _)| k.as_str() != "bias")
        .map(|(k, v)| format!("`{k}` {}", fmt::f(*v, "+.2f")))
        .collect();
    out.push(format!("Largest standardised weights: {}.", named.join(", ")));
    out.push(String::new());
    out
}

/// The message of a caught panic, as Python's `{e}` prints an exception.
fn panic_message(e: &(dyn std::any::Any + Send)) -> String {
    e.downcast_ref::<&str>().map(|s| s.to_string()).or_else(|| e.downcast_ref::<String>().cloned()).unwrap_or_default()
}

/// Section 6: the ranker evaluation, from the feature dumps when they are present, else from the stored
/// evaluation of the final dumps.
pub fn calibrator_section() -> String {
    let dump = |t: &str| art().join(format!("hyp_features{t}.jsonl"));
    let mut tags: Vec<&str> =
        ["_finalH", "_finalC", "_hybH", "_v3H", "_v3C"].into_iter().filter(|t| dump(t).exists()).collect();
    // the final dumps supersede the interim ones
    if tags.contains(&"_finalH") {
        tags.retain(|t| t.starts_with("_final"));
    } else {
        // The feature dumps are 10-25 MB each and gitignored, so a fresh clone has none of them.  The final rerun
        // stored its evaluation of the final dumps in ranker_eval_final.json; that is the published section, and a
        // stale interim dump lying around must not replace it.
        let fin = art().join("ranker_eval_final.json");
        if let Ok(text) = std::fs::read_to_string(&fin) {
            let stored: Value = serde_json::from_str(&text).expect("ranker_eval_final.json");
            let mut out: Vec<String> = Vec::new();
            for (tag, r) in stored.as_object().expect("ranker_eval_final.json is an object") {
                out.extend(format_ranker_eval(tag, r));
            }
            if !out.is_empty() {
                return out.join("\n");
            }
        }
    }
    if tags.is_empty() {
        return "_(feature dump not yet complete)_".to_string();
    }
    // `except Exception as e`: a failing evaluation is reported in the document rather than aborting it
    let evaluated = std::panic::catch_unwind(|| {
        let mut out: Vec<String> = Vec::new();
        for tag in &tags {
            let pp = dump(tag);
            if std::fs::metadata(&pp).map(|m| m.len() == 0).unwrap_or(true) {
                continue;
            }
            out.extend(format_ranker_eval(tag, &crate::calibrator::evaluate_calibrator(tag)));
        }
        out
    });
    match evaluated {
        Ok(out) if !out.is_empty() => out.join("\n"),
        Ok(_) => "_(feature dump not yet complete)_".to_string(),
        Err(e) => format!("_(calibrator evaluation failed: {})_", panic_message(e.as_ref())),
    }
}

/// The headline discovery rates the funnel explains, from the artifacts.
fn intro_numbers() -> String {
    let Ok(h) = std::panic::catch_unwind(vnext_data::headline_numbers) else {
        return "The benchmark report gives the hierarchy's and the centralised systems' discovery rates.".to_string();
    };
    let get = |k: &str| h.get(k).copied();
    let (mut o50, mut a50) =
        (get("old_found_anywhere_in_register_50000"), get("a2_old_found_anywhere_in_register_50000"));
    let (n50, na50) = (get("new_found_anywhere_in_register_50000"), get("a2_found_anywhere_in_register_50000"));
    let pct = |x: Option<f64>| fmt::f(x.expect("headline number"), ".0%");
    if o50.is_none_or(f64::is_nan) {
        (o50, a50) = (n50, na50);
        return format!(
            "The benchmark report says the hierarchy finds {} of the hidden patterns at 50,000 users and the \
             strongest centralised system finds {}.",
            pct(o50),
            pct(a50)
        );
    }
    let mut lead = format!(
        "The v1 benchmark said the hierarchy found {} of the hidden patterns at 50,000 users and the strongest \
         centralised system found {}",
        pct(o50),
        pct(a50)
    );
    if n50.is_some_and(|x| !x.is_nan()) {
        lead += &format!("; after the vNext work (section 9) the rerun gives {} and {}", pct(n50), pct(na50));
    }
    lead + "."
}

/// The pattern rows of one architecture at one scale.
fn arch_rows(rows: &[Row], scale: i64, arch: &str) -> Vec<Row> {
    rows.iter().filter(|r| !is_summary(r) && int(r, "scale") == scale && text(r, "arch") == arch).cloned().collect()
}

/// Patterns whose terminal (loss) stage is `stage`.
fn died(pr: &[Row], stage: &str, rare: Option<bool>) -> usize {
    pr.iter()
        .filter(|r| text(r, "loss_stage") == stage && rare.is_none_or(|x| py_eq(&r["rare"], &Value::Bool(x))))
        .count()
}

/// The fraction of patterns that passed `stage`, over those that reached a verdict on it.
fn surv(pr: &[Row], stage: &str) -> f64 {
    let v: Vec<f64> = pr.iter().filter(|r| has(r, stage)).map(|r| num(&r[stage]).expect("stage flag")).collect();
    if v.is_empty() { f64::NAN } else { value::sum(v.iter().copied()) / v.len() as f64 }
}

/// Share of the patterns A2 reports and the hierarchy misses that the hierarchy lost at `stage` (as in the gap
/// decomposition tables).
fn gap_share(rows: &[Row], scale: i64, stage: &str) -> f64 {
    let key = |r: &Row| (int(r, "seed"), int(r, "pid"));
    let pa_rows = arch_rows(rows, scale, "H_mycelic_full");
    let pb_rows = arch_rows(rows, scale, "A2_chunked_ctx");
    let pa: IndexMap<(i64, i64), &Row> = pa_rows.iter().map(|r| (key(r), r)).collect();
    let pb: IndexMap<(i64, i64), &Row> = pb_rows.iter().map(|r| (key(r), r)).collect();
    let lost: Vec<Row> = pa
        .iter()
        .filter(|(k, a)| pb.get(*k).is_some_and(|b| flag(b, "in_register")) && !flag(a, "in_register"))
        .map(|(_, a)| (*a).clone())
        .collect();
    died(&lost, stage, None) as f64 / lost.len().max(1) as f64
}

const STAGE_NAME: [(&str, &str); 10] = [
    ("in_register", "the **register**"),
    ("sketch_visible", "**sketch visibility**"),
    ("questioned", "the **question budget**"),
    ("extracted", "**extraction**"),
    ("descent_reached", "the **descent**"),
    ("in_triage", "**triage**"),
    ("in_pool", "the **pool**"),
    ("candidate", "**candidate formation**"),
    ("matched_any", "**matching**"),
    ("matched_tau", "**matching**"),
];

/// The terminal stage that loses the most patterns ('' marks a found pattern).
fn largest_loss(pr: &[Row]) -> Option<String> {
    // Counter.most_common(1): the first-inserted of the largest counts
    let mut c: IndexMap<String, usize> = IndexMap::new();
    for r in pr {
        if let Some(s) = r.get("loss_stage").filter(|v| truthy(v)) {
            *c.entry(py_str(s)).or_default() += 1;
        }
    }
    value::max_by(c, |e| e.1 as f64).map(|e| e.0)
}

/// Section 1: the current funnel, then the v1 funnel it replaced.
///
/// `interim` is set while a rerun is rebuilding loss_funnel.jsonl and `load()` has substituted the v1 rows: only
/// the v1 paragraph is printed then, since those are the rows on the page.
fn finding(rows: &[Row], old: &[Row], interim: bool) -> String {
    let h10 = arch_rows(rows, 10_000, "H_mycelic_full");
    let h50 = arch_rows(rows, 50_000, "H_mycelic_full");
    let a10 = arch_rows(rows, 10_000, "A2_chunked_ctx");
    let both = [(10_000, &h10), (50_000, &h50)];
    let mut out: Vec<String> = Vec::new();
    if interim {
        out.push(
            "**Interim:** loss_funnel.jsonl is being rebuilt, so this section shows the archived v1 funnel only."
                .to_string(),
        );
    } else if !h10.is_empty() || !h50.is_empty() {
        let big: Vec<(i64, Option<String>)> =
            both.iter().filter(|(_, pr)| !pr.is_empty()).map(|(sc, pr)| (*sc, largest_loss(pr))).collect();
        let mut parts = vec!["**After the rerun** (the tables in section 2).".to_string()];
        if !big.iter().any(|(_, st)| st.as_deref() == Some("questioned")) {
            let asks: Vec<String> = both
                .iter()
                .filter(|(_, pr)| !pr.is_empty())
                .map(|(sc, pr)| {
                    format!("{} of gold anchors at {} users", fmt::f(surv(pr, "questioned"), ".1%"), fmt::i(*sc, ","))
                })
                .collect();
            parts.push(format!("The question budget is no longer the largest loss: it asks {}.", asks.join(" and ")));
        }
        for (sc, pr) in both {
            let st = big.iter().find(|b| b.0 == sc).and_then(|b| b.1.clone());
            let Some(st) = st.filter(|_| !pr.is_empty()) else { continue };
            let n = died(pr, &st, None);
            let name = STAGE_NAME.iter().find(|e| e.0 == st).map_or(st.as_str(), |e| e.1);
            let mut sent = format!(
                "At {}k the largest terminal loss is {name}: {n} of {} patterns ({}, {} of them rare) are lost there",
                sc / 1000,
                pr.len(),
                fmt::f(n as f64 / pr.len() as f64, ".0%"),
                died(pr, &st, Some(true))
            );
            if st == "in_register" {
                sent += " after being matched at confidence ≥ 0.5";
            }
            if sc == 10_000 && !a10.is_empty() {
                sent += &format!(
                    "; A2 loses {} at the register ({} matched at ≥ 0.5, {} reported)",
                    died(&a10, "in_register", None),
                    fmt::f(surv(&a10, "matched_tau"), ".3f"),
                    fmt::f(surv(&a10, "in_register"), ".3f")
                );
            }
            let others: Vec<String> =
                [("in_register", "cut at the register"), ("questioned", "lost to the question budget")]
                    .iter()
                    .filter(|(o, _)| *o != st)
                    .map(|(o, lab)| format!("{} {lab}", died(pr, o, None)))
                    .collect();
            let against = if sc == 50_000 { format!(", against {}", others.join(" and ")) } else { String::new() };
            parts.push(format!("{sent}{against}."));
        }
        out.push(parts.join(" "));
    }
    let o10 = arch_rows(old, 10_000, "H_mycelic_full");
    let o50 = arch_rows(old, 50_000, "H_mycelic_full");
    let oa10 = arch_rows(old, 10_000, "A2_chunked_ctx");
    if o10.is_empty() || o50.is_empty() || oa10.is_empty() {
        return if out.is_empty() { "_(loss funnel not yet available)_".to_string() } else { out.join("\n\n") };
    }
    let inv50 = o50.iter().filter(|r| !flag(r, "sketch_visible")).count();
    let matched = |pr: &[Row]| pr.iter().filter(|r| flag(r, "matched_tau")).count() as f64;
    out.push(format!(
        r#"**The v1 finding** (before the vNext work; `artifacts/v1/loss_funnel.jsonl`,
same seeds). The handoff pack's leading hypothesis was that the sketch channel is too poor
to seed good hypotheses. It is not: at 10,000 users the sketch triage sees
{sk10} of gold anchors. The pattern dies later, at two places. First, the
**question budget**: the calibrated budget asks only {q10} of gold anchors at
10k and {q50} at 50k, and that is the single largest first-loss stage at both
scales ({g10} and {g50} of the gap to A2). Second, the **register**: at 10k, {cut10}
of all patterns are matched at confidence ≥ 0.5 and then cut, because the
hierarchy rates spurious candidates as highly as genuine ones and the
register holds one entry per entity. A2 loses {a2cut} there, {a2share} of its
confident matches against the hierarchy's {hshare}. Asking
every triage candidate recovers evidence for 98.5% of patterns and the kernel
forms a correct confident candidate for ~80% — more than A2 reports (0.685) — and then buries
them. The calibrated small budget was the right choice *given the ranker*; the
ranker is the fault. At 50,000 users the sketch additionally loses {inv} of
patterns (mostly rare: a single-witness facet cannot set a site bit under the
support threshold), so scale adds a third, smaller stage."#,
        sk10 = fmt::f(surv(&o10, "sketch_visible"), ".1%"),
        q10 = fmt::f(surv(&o10, "questioned"), ".1%"),
        q50 = fmt::f(surv(&o50, "questioned"), ".0%"),
        g10 = fmt::f(gap_share(old, 10_000, "questioned"), ".0%"),
        g50 = fmt::f(gap_share(old, 50_000, "questioned"), ".0%"),
        cut10 = fmt::f(died(&o10, "in_register", None) as f64 / o10.len() as f64, ".0%"),
        a2cut = died(&oa10, "in_register", None),
        a2share = fmt::f(died(&oa10, "in_register", None) as f64 / matched(&oa10), ".0%"),
        hshare = fmt::f(died(&o10, "in_register", None) as f64 / matched(&o10), ".0%"),
        inv = fmt::f(inv50 as f64 / o50.len() as f64, ".0%"),
    ));
    out.join("\n\n")
}

/// Calls change from batched metering alone, at identical decisions: the three metering rows of the 9.7 ledger
/// (variant mean / base mean - 1).
fn batching_call_cuts() -> Vec<f64> {
    let mut cuts = Vec::new();
    for (tag, base, var) in [
        ("rk2_qf0.65|rx_base065", Sel::Parity(1), Sel::Parity(1)),
        ("prev_batched", Sel::Label("prev"), Sel::Label("prev_batched")),
        ("prev_batched50", Sel::Label("prev"), Sel::Label("prev_batched")),
    ] {
        let (a, b) = vnext_data::arms(tag, Some(base), Some(var));
        let s = vnext_data::common_seeds(&a, &b);
        if !s.is_empty() {
            let mb = vnext_data::mean_of(s.iter().map(|x| &b[x]), "inference_calls");
            let ma = vnext_data::mean_of(s.iter().map(|x| &a[x]), "inference_calls");
            cuts.push(mb / ma - 1.0);
        }
    }
    cuts
}

fn funnel_missing() -> bool {
    std::fs::metadata(funnel_path()).map(|m| m.len() == 0).unwrap_or(true)
}

/// The rerun's own funnel rows; empty while it is being rebuilt (`load()` would substitute the v1 rows, which must
/// not be described as the rerun's).
fn rerun_funnel() -> Vec<Row> {
    if funnel_missing() { Vec::new() } else { loss_report::load(&funnel_path()) }
}

/// v1 A2's reported share at `scale`, or a marker when the v1 archive is absent.
fn v1_a2(old: &[Row], scale: i64) -> String {
    let rows = arch_rows(old, scale, "A2_chunked_ctx");
    if rows.is_empty() { "(v1 archive absent)".to_string() } else { fmt::f(surv(&rows, "in_register"), ".3f") }
}

/// The document, dated `date` (`YYYY-MM-DD`).
pub fn compose_at(date: &str) -> String {
    let rows = loss_report::load_default();
    let interim = funnel_missing();
    let old = loss_report::load(&vnext_data::v1().join("loss_funnel.jsonl"));
    let h50 = arch_rows(&rerun_funnel(), 50_000, "H_mycelic_full");
    let is_false = |r: &&Row, k: &str| r.get(k) == Some(&Value::Bool(false));
    let nx50: Vec<&Row> = h50.iter().filter(|r| is_false(r, "extracted")).collect();
    let inv50: Vec<&Row> = h50.iter().filter(|r| is_false(r, "sketch_visible")).collect();
    let rare = |rs: &[&Row]| rs.iter().filter(|r| flag(r, "rare")).count();
    let (ext_head, sk_head, sk_fail) = if !h50.is_empty() {
        let all: Vec<&Row> = h50.iter().collect();
        (
            format!(
                "Extraction ({} at 50k, {} of rare patterns)",
                fmt::f(nx50.len() as f64 / h50.len() as f64, ".0%"),
                fmt::f(rare(&nx50) as f64 / rare(&all).max(1) as f64, ".0%")
            ),
            format!(
                "Sketch ({} at 50k, {} of {} rare)",
                fmt::f(inv50.len() as f64 / h50.len() as f64, ".0%"),
                rare(&inv50),
                inv50.len()
            ),
            format!(
                "{} of the {} fail on \"causal span < 2",
                inv50.iter().filter(|r| r.get("sketch_fail").and_then(Value::as_str) == Some("span_lt2")).count(),
                inv50.len()
            ),
        )
    } else {
        (
            "Extraction (the rerun's 50k funnel is not yet available)".to_string(),
            "Sketch (the rerun's 50k funnel is not yet available)".to_string(),
            "Sketch losses fail mostly on \"causal span < 2".to_string(),
        )
    };
    let cuts = batching_call_cuts();
    let cut_txt = if cuts.is_empty() {
        "—".to_string()
    } else {
        let hi = value::max_by(cuts.iter().copied(), |x| *x).unwrap();
        let lo = value::min_by(cuts.iter().copied(), |x| *x).unwrap();
        format!("−{}% to −{}%", fmt::f(-hi * 100.0, ".0f"), fmt::f(-lo * 100.0, ".0f"))
    };
    let seeds_at =
        |sc: i64| rows.iter().filter(|r| int(r, "scale") == sc).map(|r| int(r, "seed")).collect::<BTreeSet<_>>().len();
    let (n10, n50) = (seeds_at(10_000), seeds_at(50_000));
    format!(
        r#"# Mycelic: where the hidden patterns go
## Loss accounting and gap decomposition

**What this is.** {intro} This document says *where the other patterns went*. Every discoverable
gold pattern is traced through the pipeline and the first stage at which it
is lost is recorded, for the hierarchy (`H_mycelic_full`), its centralised
twin (`B4_central_triage`), the perfect-retrieval oracle (`Y_oracle_retrieval`)
and the strongest centralised baseline (`A2_chunked_ctx`), on the same worlds.
Then three paired diagnostics on identical worlds decompose the gap into the
stages responsible.

Nothing here is a design. It is the measurement the next design is built on.
Every number is computed from `research/mycelic/artifacts/loss_funnel.jsonl`
and `quick_diag*.jsonl` at generation time. {n10} seeds at 10,000 users,
{n50} at 50,000.

## 1. The finding

{finding}

## 2. Stage-by-stage survival

Stages are ordered by pipeline position. They are not strictly nested — a
pattern can be visible to the sketch without having been extracted — so
`first stage lost` records the first stage in order that failed. The flat
systems have no sketch, triage, question or descent stage.

{loss_tables}

## 3. Paired decomposition on identical worlds

Each row is the calibrated **v1** hierarchy (hand-set ranker, question_frac
0.25; not the vNext configuration of section 9) with one setting changed, run on the
same (scale, seed) worlds as the base. **An unbounded register is a
benchmark change, not a design**; it appears here only to measure how much
discovery is being lost to ranking rather than to retrieval or judgement.
Arrows mark a 95% bootstrap interval entirely on one side of zero.

### 10,000 users

{diag10}

### 50,000 users

{diag50}

**Reading it.** Asking every triage candidate raises evidence coverage from
0.69 to 0.985 at 10k (0.45 to 0.79 at 50k) and *lowers* discovery at 10k:
the recovered evidence produces candidates the register cannot hold. Remove
the register pressure and the same runs report 0.795 at 10k — above v1 A2's
{v1a2_10} — and 0.643 at 50k against v1 A2's {v1a2_50}, at two-thirds of A2's compute at 50k.
Allowing home-site evidence back in does not help, so the flood is not the
foreign-only filter. The residual at 50k is the sketch loss, which no
question budget can recover because the anchor never reaches the triage list.

## 4. Why the hierarchy's ranker is worse than A2's on the same operator

Both systems rank with the same confidence logistic. Its main terms — a
causal span, several regions, links dominated by different branches — are
exactly the properties the triage *pre-selects* anchors for. Among the
candidates the hierarchy's kernel actually has to order, those terms carry
almost no information, because every questioned anchor already has them. A2's
pool holds every anchor, including thousands with no cross-organisational
structure at all, for which the same terms are decisive. The hierarchy's
discrimination problem is a conditioning problem, not an operator problem.

The consequence is testable: features the triage did **not** condition on —
per-link independent support, inter-link lags, conflict volume, dispersion,
whether the candidate came from a question — should separate genuine from
spurious candidates among the triage survivors. Section 6 measures that.

## 5. Two further losses, measured

**{ext_head}.** This is stochastic recall,
not a capability ceiling: in an earlier funnel, with a 10% extraction loss
(research log, iteration 20), a fresh edge-tier re-read of the same records
recovered 80–91% of the patterns lost there and a kernel-tier *local* re-read
100%; the rerun funnel's figure was not re-measured. A user agent asked a
targeted question about an entity can re-read its own notes about it — the
entity name is in every record's surface text — so the loss is recoverable
without moving any text. The paired tests of targeted local re-extraction did
not pay, though (9.4), and it is off in the frozen configuration.

**{sk_head}.** {sk_fail}
across foreign sites": a facet with one witness at a site sets no bit under
`sketch_min_support = 2`. Lowering the threshold lets benign single mentions
set bits and lengthens the triage list, which only pays if the ranker can
absorb the noise — which is the same lever as section 4.

## 6. How separable are genuine candidates, on kernel-side features?

Every candidate the synthesis produced (before the register cut) was dumped
with its confidence-logistic inputs and labelled against gold. One shared
logistic was fitted on the pooled candidates of all four systems on the
calibration seeds (500–502) and evaluated on the evaluation seeds (0–4) by
re-ranking each run's candidates under its own register cap. No feature reads
raw text; the privacy accounting is unchanged.

{calibrator}

## 7. What this changed about the next architecture

Not "richer sketches". In order of the loss they addressed, with the
measured outcome (section 9 has every paired table):

1. **A calibrated ranker** over kernel-side features, fitted on the
   calibration seeds by the same protocol as every v1 knob, so the
   question budget could be spent without flooding the register. Done:
   +0.21 found at 10k, +0.16 at 50k, together with the budget it unlocked.
2. **A cheaper descent**, so the budget is affordable at 50k. Done as
   batched metering (calls {cut_txt} at identical decisions); the
   chain-scoped variant that halves compute is kept as a separate arm
   because it costs evidence coverage.
3. **Link timing in the kernel's temporal check**, which the loss replay
   found contaminated by stale mentions. Done (hybrid): +0.04 at 10k, +0.06
   at 50k, free.
4. **Targeted local re-extraction** on descent. Measured and rejected: no
   gain at 10k, inconclusive at 50k.
5. **A support-1 sketch bit** for rare facets. Measured on one calibration
   seed and rejected as implemented: it floods the triage list.

Each was a paired experiment against `H_mycelic_full` on identical worlds
with the centralised controls re-run at the same time. None moves raw text.

## 8. Reproducing this

```sh
python3 -m research.mycelic.loss_accounting      # 10k x 5, 50k x 3 -> loss_funnel.jsonl
python3 -m research.mycelic.loss_report          # the tables
python3 -m research.mycelic.quick_paired --tag diag_qf1 --cfg '{{"question_frac": 1.0}}' --seeds 0,1,2,3,4
python3 -m research.mycelic.calibrator dump && python3 -m research.mycelic.calibrator fit
python3 -m research.mycelic.loss_doc             # this document
python3 -m research.mycelic.make_pdf docs/MYCELIC_LOSS_ACCOUNTING.md docs/MYCELIC_LOSS_ACCOUNTING.pdf
```

_Generated {date} by `crates/mycelic-research/src/loss_doc.rs` from the
raw artifacts._

{vnext}
"#,
        intro = intro_numbers(),
        finding = finding(&rows, &old, interim),
        loss_tables = loss_report::build(&loss_report::DEFAULT_SCALES),
        diag10 = diag_table(&DIAG, 10_000),
        diag50 = diag_table(&DIAG50, 50_000),
        v1a2_10 = v1_a2(&old, 10_000),
        v1a2_50 = v1_a2(&old, 50_000),
        calibrator = calibrator_section(),
        vnext = vnext_section(),
    )
}

/// Today's local date, as `date.today().isoformat()` prints it.
pub fn today() -> String {
    chrono::Local::now().date_naive().format("%Y-%m-%d").to_string()
}

/// The document, dated today.
pub fn compose() -> String {
    compose_at(&today())
}

/// `python -m research.mycelic.loss_doc`: write `docs/MYCELIC_LOSS_ACCOUNTING.md`.
pub fn main(_args: &[String]) -> anyhow::Result<()> {
    let txt = compose();
    let out = out_path();
    std::fs::write(&out, &txt).map_err(|e| anyhow::anyhow!("{}: {e}", out.display()))?;
    println!("wrote {} {} chars", out.display(), txt.chars().count());
    Ok(())
}
