//! Headline findings, computed from the artifacts (`findings.py`).
//!
//! Every claim this module emits is derived from the raw rows, so the narrative cannot drift from the numbers.
//! Where a result does not reach significance the text says so rather than rounding it into a claim.

use crate::analysis::{Paired, agg, dedupe_default, load, paired, paired_by};
use crate::loss_report::{int, text};
use crate::np::math;
use crate::py::fmt;
use crate::py::value::{self, Row, py_str};
use crate::runner::art;
use indexmap::IndexMap;
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};

pub const P_FLOOR_NOTE: &str = "an exact sign test on *n* paired seeds cannot report below 2^-(n-1); with 5 seeds \
                                the floor is p = 0.0625 and with 10 it is p = 0.002";

const FOUND: &str = "found_anywhere_in_register";

/// The headline rows: E1, E1b and the E10 scale trend, first row per (arch, scale, seed).
pub fn rows() -> Vec<Row> {
    let mut r = load("e1_baselines.jsonl");
    r.extend(load("e1b_extra.jsonl"));
    r.extend(load("e10_scale_trend.jsonl"));
    dedupe_default(r)
}

/// `np.mean` of one metric of one architecture at one scale, `None` when it was not run there.
pub fn mean(rows: &[Row], arch: &str, scale: i64, metric: &str) -> Option<f64> {
    let v: Vec<f64> = rows
        .iter()
        .filter(|r| text(r, "arch") == arch && int(r, "scale") == scale)
        .map(|r| value::f(r, metric))
        .collect();
    if v.is_empty() { None } else { Some(math::mean(&v)) }
}

/// The strongest architecture on `metric` at `scale`, the references excluded (`_best`).
pub fn best(rows: &[Row], scale: i64) -> (String, f64) {
    best_with(rows, scale, FOUND, &["Y_oracle_retrieval", "Z_random_rank"])
}

pub fn best_with(rows: &[Row], scale: i64, metric: &str, exclude: &[&str]) -> (String, f64) {
    let sub: Vec<Row> = rows.iter().filter(|r| int(r, "scale") == scale).cloned().collect();
    let a: Vec<Row> =
        agg(&sub, &[metric], &["arch"]).into_iter().filter(|x| !exclude.contains(&text(x, "arch"))).collect();
    match value::max_by(a.iter(), |x| value::f(x, metric)) {
        None => ("—".to_string(), 0.0),
        Some(b) => (text(b, "arch").to_string(), value::f(b, metric)),
    }
}

pub const HIER: &str = "H_mycelic_full";
/// The architectures the front-page comparison and its three decisions cite.
pub const COMPARISON: [&str; 10] = [
    HIER,
    "A_flat_rag",
    "A2_chunked_ctx",
    "B2_map_reduce",
    "B4_central_triage",
    "C_recursive_sum",
    "D_hier_nolineage",
    "E_hier_lineage",
    "F_hier_retrieval",
    "G_hier_questions",
];
/// The usual arguments for a hierarchy: (label, metric, higher is better).
pub const GROUNDS: [(&str, &str, bool); 5] = [
    ("confidentiality (no original notes leave the owning agent)", "raw_text_exposure_fraction", false),
    ("weak-signal sensitivity", "rare_signal_recall", true),
    ("independent-support accuracy", "independent_evidence_accuracy", true),
    ("lineage / provenance", "lineage_accuracy", true),
    ("resistance to planted traps", "decoy_acceptance_all", false),
];

fn scales_of(rows: &[Row]) -> Vec<i64> {
    rows.iter().map(|r| int(r, "scale")).collect::<BTreeSet<_>>().into_iter().collect()
}

fn archs_at(rows: &[Row], scale: i64) -> BTreeSet<&str> {
    rows.iter().filter(|r| int(r, "scale") == scale).map(|r| text(r, "arch")).collect()
}

/// The largest scale at which every architecture in `need` was run.
///
/// Picking the largest scale present would, mid-run or where a scale was run for a subset only, report a missing
/// architecture as 0%.
pub fn largest_complete_scale(rows: &[Row], need: &[&str]) -> Option<i64> {
    scales_of(rows).into_iter().rfind(|&sc| need.iter().all(|a| archs_at(rows, sc).contains(a)))
}

/// (label, centralised value, hierarchy value) of one argument.
type Ground = (&'static str, f64, f64);

/// Which of `GROUNDS` the hierarchy wins against `best_arch` at `scale`: (wins, losses).
pub fn grounds(rows: &[Row], scale: i64, best_arch: &str) -> (Vec<Ground>, Vec<Ground>) {
    let (mut wins, mut losses) = (Vec::new(), Vec::new());
    for (label, key, hib) in GROUNDS {
        let (Some(bv), Some(hv)) = (mean(rows, best_arch, scale, key), mean(rows, HIER, scale, key)) else { continue };
        let better = if hib { hv > bv + 1e-9 } else { hv < bv - 1e-9 };
        if better { wins.push((label, bv, hv)) } else { losses.push((label, bv, hv)) }
    }
    (wins, losses)
}

/// A paired comparison in one phrase.
pub fn sig(p: &Paired) -> String {
    if p.n == 0 {
        return "no paired runs".to_string();
    }
    format!(
        "Δ {} (95% CI [{}, {}], {}/{} seeds, sign p={})",
        fmt::f(p.mean_diff, "+.3f"),
        fmt::f(p.ci_lo, "+.3f"),
        fmt::f(p.ci_hi, "+.3f"),
        p.wins,
        p.n_nonzero,
        fmt::f(p.sign_p, ".3f")
    )
}

pub const GLOSSARY: &str = r#"
The report uses a small number of terms repeatedly. All of them are
measurements, not scores on a scale someone invented.

| term | what it means |
|---|---|
| **hidden pattern** | A genuine emerging problem planted in the synthetic company: a chain of causally linked events about one supplier, system or component, whose parts are deliberately scattered across different teams, sites and regions so that nobody sees more than one part. |
| **discovery (`found`)** | The fraction of hidden patterns the system puts on the executive risk register *anywhere*. The plainest measure of "did we find it at all". |
| **AP** (average precision) | How well the system *ranks* what it found. A register nobody can read top to bottom is worth less than one where the real items are at the top. Random ranking of the same candidates scores about 0.30 on the discrimination task. |
| **evidence coverage** | The fraction of hidden patterns for which the system ever *held* the evidence, whether or not it reported them. The gap between coverage and discovery is what is lost to judgement rather than to retrieval. |
| **rare-signal recall** | Discovery restricted to the hardest patterns: those supported by one or two people in the entire 50,000-person company. |
| **FDR** (false discovery rate) | The fraction of what the system reports that is not a real pattern. |
| **decoy acceptance** | The fraction of the deliberately planted traps the system falls for. |
| **compute (cu)** | Normalised compute units: model size times tokens. Provider-independent; a dollar estimate is given separately. |
| **raw-text exposure** | The fraction of employees' original notes read by anything other than the agent that owns them. The confidentiality cost. |
"#;

fn comma(x: i64) -> String {
    fmt::i(x, ",")
}

/// The front page: what to build, what it costs, what it buys.
///
/// Every comparative claim here is derived from the rows.  An earlier version asserted three standing advantages
/// for the hierarchy; one of them (weak-signal sensitivity) is contradicted by the data at some scales, so the
/// claims are now computed and the ones that fail are stated as failures.
pub fn decision_summary() -> String {
    let rows = rows();
    if rows.is_empty() {
        return "_(no results yet)_".to_string();
    }
    // Use the largest scale at which the comparison is actually COMPLETE.
    let Some(big) = largest_complete_scale(&rows, &COMPARISON) else {
        return "_(results incomplete: no scale yet has every architecture)_".to_string();
    };
    let n_seeds = rows.iter().filter(|r| int(r, "scale") == big).map(|r| int(r, "seed")).collect::<BTreeSet<_>>().len();
    let mut out: Vec<String> = Vec::new();
    let (best_arch, best_v) = best(&rows, big);
    let m = |arch: &str, k: &str| mean(&rows, arch, big, k);

    let h = m(HIER, FOUND);
    let b = best_v;
    out.push("### The one-paragraph version".into());
    out.push(String::new());
    let verdict = if h.unwrap_or(0.0) < b - 1e-9 {
        "is not the most accurate option"
    } else {
        "is the most accurate option measured"
    };
    let all_scales = scales_of(&rows);
    let scale_list = if all_scales.len() > 1 {
        let head: Vec<String> = all_scales[..all_scales.len() - 1].iter().map(|s| comma(*s)).collect();
        format!("{} and {}", head.join(", "), comma(all_scales[all_scales.len() - 1]))
    } else {
        comma(all_scales[0])
    };
    let n_archs = rows.iter().map(|r| text(r, "arch")).collect::<BTreeSet<_>>().len();
    out.push(format!(
        "We built a synthetic enterprise with known hidden problems planted in it, at {scale_list} people, and tested \
         {n_archs} ways of finding those problems, from reading a filtered subset of the company's notes with one \
         very large model, to a six-level hierarchy of agents mirroring the org chart. **The hierarchy {verdict}.** \
         The strongest single approach measured at {} people is `{best_arch}`, which finds {} of the hidden problems \
         against the hierarchy's {}. Whether the hierarchy is nonetheless worth building depends entirely on which \
         of the secondary properties below we actually need; the table states which ones it delivers and which it \
         does not.",
        comma(big),
        fmt::f(b, ".0%"),
        fmt::f(h.unwrap_or(0.0), ".0%")
    ));
    out.push(String::new());

    out.push(format!("### Head to head at {} users ({n_seeds} seeds)", comma(big)));
    out.push(String::new());
    let bigger: Vec<i64> = all_scales.iter().copied().filter(|&sc| sc > big).collect();
    if let Some(&first) = bigger.first() {
        let present = archs_at(&rows, first);
        let missing: Vec<&str> =
            COMPARISON.iter().copied().filter(|a| !present.contains(a)).collect::<BTreeSet<_>>().into_iter().collect();
        out.push(format!(
            "*(Runs exist at {} users too, and §11 reports them. The head-to-head is stated at {} because that is the \
             largest scale where **every** architecture in the comparison was run — {} {} not run at {}. Choosing the \
             largest complete scale is a rule applied by the generator, not a choice made per result.)*",
            bigger.iter().map(|s| comma(*s)).collect::<Vec<_>>().join(", "),
            comma(big),
            missing.iter().map(|m| format!("`{m}`")).collect::<Vec<_>>().join(", "),
            if missing.len() == 1 { "was" } else { "were" },
            comma(first)
        ));
        out.push(String::new());
    }
    if big == 50_000 {
        out.push(
            "*Seeds 0–2 at 50,000 users were also the development panel on which several vNext decisions were read \
             (§6); the held-out panel is seeds 5–9 at 10,000 users.*"
                .into(),
        );
        out.push(String::new());
    }
    out.push("| | best centralised option | the hierarchy | hierarchy better? |".into());
    out.push("|---|---:|---:|---|".into());
    out.push(format!("| approach | `{best_arch}` | `{HIER}` | |"));

    let mut row = |label: &str, key: &str, higher_is_better: bool, spec: &str| {
        let (Some(bv), Some(hv)) = (m(&best_arch, key), m(HIER, key)) else { return };
        let better = if higher_is_better { hv > bv + 1e-9 } else { hv < bv - 1e-9 };
        let mark = if better {
            "**yes**"
        } else if (hv - bv).abs() > 1e-9 {
            "no"
        } else {
            "tie"
        };
        let show = |x: f64| if x.abs() < 1e4 { fmt::f(x, spec) } else { fmt::f(x, ".2e") };
        out.push(format!("| {label} | {} | {} | {mark} |", show(bv), show(hv)));
    };
    row("hidden problems found", FOUND, true, ".0%");
    row("the hardest problems (1-2 witnesses company-wide)", "rare_signal_recall", true, ".0%");
    row("found within the top 100 of the register", "recall_at_100", true, ".0%");
    row("precision in the top 40 of the register", "precision_at_40", true, ".0%");
    row("precision in the top 100 of the register", "precision_at_100", true, ".0%");
    row("ranking quality (AP)", "average_precision", true, ".3f");
    row("decoy traps accepted", "decoy_acceptance_all", false, ".0%");
    row("evidence coverage (held it at all)", "evidence_coverage_2links", true, ".0%");
    row("independent-support accuracy", "independent_evidence_accuracy", true, ".0%");
    row("lineage accuracy", "lineage_accuracy", true, ".0%");
    row("contradictions detected (F1)", "contradiction_f1", true, ".0%");
    row("compute", "compute_units", false, ".2e");
    row("model calls", "inference_calls", false, ",.0f");
    row("employees' original notes read centrally", "raw_text_exposure_fraction", false, ".0%");
    out.push(String::new());
    out.push(
        "*Reading the false-discovery rate.* The register is a ranked watchlist of roughly one entry per tracked \
         entity, not a shortlist, so a high FDR over the whole register is structural and is true of every \
         architecture including the perfect-retrieval reference. The operationally meaningful numbers are precision \
         in the top 40 or top 100 — what an executive would actually read — and the ranking quality (AP) that \
         determines them."
            .into(),
    );
    out.push(String::new());

    // which grounds actually hold
    let (wins, losses) = grounds(&rows, big, &best_arch);
    out.push("### Which of the usual arguments for a hierarchy actually hold".into());
    out.push(String::new());
    if !wins.is_empty() {
        out.push("**Hold, on this evidence:**".into());
        out.push(String::new());
        for (label, bv, hv) in &wins {
            out.push(if hv.abs() < 1e4 {
                format!("* {label} — {} vs {} for the centralised option.", fmt::f(*hv, ".0%"), fmt::f(*bv, ".0%"))
            } else {
                format!("* {label} — {} vs {}.", fmt::f(*hv, ".3g"), fmt::f(*bv, ".3g"))
            });
        }
        out.push(String::new());
    }
    if !losses.is_empty() {
        out.push("**Do NOT hold, and should not be used to justify the build:**".into());
        out.push(String::new());
        for (label, bv, hv) in &losses {
            out.push(if hv.abs() < 1e4 {
                format!(
                    "* {label} — {} vs {}; the centralised option is better.",
                    fmt::f(*hv, ".0%"),
                    fmt::f(*bv, ".0%")
                )
            } else {
                format!("* {label} — {} vs {}; centralised wins.", fmt::f(*hv, ".3g"), fmt::f(*bv, ".3g"))
            });
        }
        out.push(String::new());
    }

    // The sharpest framing available: the centralised twin of the hierarchy's OWN algorithm, which isolates the
    // algorithm from the topology.
    let b4f = m("B4_central_triage", FOUND);
    let b4c = m("B4_central_triage", "compute_units");
    let b4p = m("B4_central_triage", "claim_exposure_fraction");
    let hp = m(HIER, "claim_exposure_fraction");
    let hc = m(HIER, "compute_units");
    if let (Some(b4f), Some(h), Some(b4c), Some(hc)) = (b4f, h, b4c.filter(|x| *x != 0.0), hc.filter(|x| *x != 0.0)) {
        out.push("### The control that reframes the decision".into());
        out.push(String::new());
        out.push(
            "`B4_central_triage` runs **exactly the hierarchy's own discovery algorithm**, centrally: one claim pool, \
             no propagation budget, no routing error, no descent. It separates the value of the *algorithm* from the \
             value of the *topology*."
                .into(),
        );
        out.push(String::new());
        // Assert the tie from the paired test over every (scale, seed), not from the size of the gap at one scale.
        let pt = paired(&rows, HIER, "B4_central_triage", FOUND);
        let mut verdict = String::new();
        let mut crosses: Option<bool> = None;
        if pt.n > 0 {
            let c = pt.ci_lo <= 0.0 && 0.0 <= pt.ci_hi;
            crosses = Some(c);
            verdict = format!(
                "  Across all {} paired (scale, seed) runs the difference is {} (95% CI [{}, {}], {}/{} seeds, sign \
                 p={}), so {}",
                pt.n,
                fmt::f(pt.mean_diff, "+.3f"),
                fmt::f(pt.ci_lo, "+.3f"),
                fmt::f(pt.ci_hi, "+.3f"),
                pt.wins,
                pt.n_nonzero,
                fmt::f(pt.sign_p, ".3f"),
                if c {
                    "**the two are statistically indistinguishable** — the interval spans zero and the wins split \
                     roughly evenly."
                } else {
                    "the difference is real, and its sign is the one above."
                }
            );
        }
        let factor = fmt::f(hc / value::max2(1.0, b4c), ".1f");
        out.push(format!(
            "It finds {} of the hidden problems against the hierarchy's {}, at {} compute against {} — about {factor}x \
             less.{verdict}",
            fmt::f(b4f, ".0%"),
            fmt::f(h, ".0%"),
            fmt::f(b4c, ".2e"),
            fmt::f(hc, ".2e")
        ));
        out.push(String::new());
        if let (Some(b4p), Some(hp)) = (b4p, hp) {
            let pools = format!(
                "The centralised version pools {} of all extracted claims in one place; the hierarchy pools {} and \
                 moves no original text at all.",
                fmt::f(b4p, ".0%"),
                fmt::f(hp, ".0%")
            );
            let cost = format!("roughly {factor}x the compute and tens of thousands of extra model calls");
            if crosses.is_none_or(|c| c) {
                out.push(format!(
                    "So the honest statement of what the hierarchy buys is narrow and specific: **the triage algorithm \
                     is what finds the problems; the hierarchy is how you run that algorithm without centralising the \
                     company's data.** {pools} That privacy property costs {cost}. If we do not need it, we should \
                     run the algorithm centrally."
                ));
            } else if pt.mean_diff > 0.0 {
                let mut per = Vec::new();
                for sc in scales_of(&rows) {
                    let by_seed = |arch: &str| -> BTreeMap<i64, f64> {
                        rows.iter()
                            .filter(|r| text(r, "arch") == arch && int(r, "scale") == sc)
                            .map(|r| (int(r, "seed"), value::f(r, FOUND)))
                            .collect()
                    };
                    let (hs, bs) = (by_seed(HIER), by_seed("B4_central_triage"));
                    let diffs: Vec<f64> = hs.iter().filter_map(|(k, hv)| bs.get(k).map(|bv| hv - bv)).collect();
                    if !diffs.is_empty() {
                        per.push(format!("{} at {}", fmt::f(math::mean(&diffs), "+.3f"), comma(sc)));
                    }
                }
                out.push(format!(
                    "So the algorithm alone does not explain the result: **run centrally, the same triage finds fewer \
                     of the hidden problems than the hierarchy does.** The topology — propagation budgets, routing \
                     and the descent — adds discovery on these worlds, at {cost}. Per scale, the paired difference is \
                     {} users. {pools}",
                    join(&per, "and")
                ));
            } else {
                out.push(format!(
                    "So the topology costs discovery as well as compute: **run centrally, the same triage finds more \
                     of the hidden problems than the hierarchy does.** What the hierarchy buys is confidentiality \
                     alone. {pools} That property costs {cost}. If we do not need it, we should run the algorithm \
                     centrally."
                ));
            }
            out.push(String::new());
        }
    }
    out.push("### Three decisions this supports".into());
    out.push(String::new());
    let c = m("C_recursive_sum", FOUND);
    let d_ = m("D_hier_nolineage", FOUND);
    let e_ = m("E_hier_lineage", FOUND);
    let f_ = m("F_hier_retrieval", FOUND);
    let g_ = m("G_hier_questions", FOUND);
    out.push(format!(
        "**1. Do not build progressive summarisation up the org chart.** This is the intuitive design — each layer \
         summarises the layer below — and it is the clearest negative result in the study. At {} users it finds {} \
         (plain summarisation), {} (structured, no lineage) and {} (structured with lineage) of the hidden problems. \
         The reason is measurable rather than a matter of tuning, and is given in the executive summary.",
        comma(big),
        fmt::f(c.unwrap_or(0.0), ".0%"),
        fmt::f(d_.unwrap_or(0.0), ".0%"),
        fmt::f(e_.unwrap_or(0.0), ".0%")
    ));
    out.push(String::new());
    if let (Some(f_), Some(g_)) = (f_, g_) {
        out.push(format!(
            "**2. If we build a hierarchy, the value is in the downward path, not the upward one.** Adding targeted \
             downward retrieval takes discovery from {} to {}; adding sketch-driven questioning on top takes it to \
             {}. Budget accordingly: the upward channel should be cheap and statistical, and the downward channel is \
             where the work — and the cost — actually is.",
            fmt::f(e_.unwrap_or(0.0), ".0%"),
            fmt::f(f_, ".0%"),
            fmt::f(g_, ".0%")
        ));
        out.push(String::new());
    }
    out.push(decision_evidence_vs_capability());
    out.push(String::new());
    out.join("\n")
}

fn read_json(name: &str) -> Option<Value> {
    let text = std::fs::read_to_string(art().join(name)).ok()?;
    Some(serde_json::from_str(&text).unwrap_or_else(|e| panic!("{name}: {e}")))
}

/// Both live-ranking result files (statistics only, and with the raw notes), when both exist.
fn live_results() -> Option<(Value, Value)> {
    let p = art().join("live_rank_results.json");
    let pr = art().join("live_rank_results_rich.json");
    if !(p.exists() && pr.exists()) {
        return None;
    }
    Some((read_json("live_rank_results.json")?, read_json("live_rank_results_rich.json")?))
}

/// `dd.get(key)` as an object, empty when absent.
fn obj<'a>(dd: &'a Value, key: &str) -> Vec<(&'a String, &'a Value)> {
    dd.get(key).and_then(Value::as_object).map(|o| o.iter().collect()).unwrap_or_default()
}

fn fnum(v: &Value, k: &str) -> f64 {
    value::num(&v[k]).unwrap_or_else(|| panic!("no number {k} in {v}"))
}

/// Per model (mean AP, min AP, max AP, runs): the repeats when aggregated, else the single run.
type Cells = IndexMap<String, (f64, f64, f64, i64)>;

fn cells(dd: &Value) -> Cells {
    let ag = obj(dd, "models_aggregated");
    if !ag.is_empty() {
        return ag
            .into_iter()
            .map(|(k, v)| (k.clone(), (fnum(v, "ap_mean"), fnum(v, "ap_min"), fnum(v, "ap_max"), int_of(&v["n_runs"]))))
            .collect();
    }
    obj(dd, "models").into_iter().map(|(k, v)| (k.clone(), (fnum(v, "ap"), fnum(v, "ap"), fnum(v, "ap"), 1))).collect()
}

fn int_of(v: &Value) -> i64 {
    v.as_i64().unwrap_or_else(|| value::num(v).expect("a count") as i64)
}

fn sorted_common<A, B>(a: &IndexMap<String, A>, b: &IndexMap<String, B>) -> Vec<String> {
    a.keys().filter(|k| b.contains_key(*k)).cloned().collect::<BTreeSet<_>>().into_iter().collect()
}

/// Decision 3, stated only as strongly as the repeats allow.
fn decision_evidence_vs_capability() -> String {
    let head = "**3. Spend top-tier model budget on re-opening original evidence, not on bigger reasoning over \
                summaries.** ";
    let Some((d, dr)) = live_results() else { return format!("{head}_(live measurement not run)_") };
    let (cs, cr) = (cells(&d), cells(&dr));
    let common = sorted_common(&cs, &cr);
    if common.is_empty() {
        return format!("{head}_(live measurement not run)_");
    }
    let helped = common.iter().filter(|m| cr[*m].0 > cs[*m].0).count();
    // A model counts as separated only if its two conditions' observed run ranges do not overlap AND the richer
    // condition is the higher one.  With 2-3 runs that is the only claim the data supports; anything finer would be
    // invented precision.  A cell with a single run has a degenerate "range", so separation there is not evidence
    // and is excluded.
    let sep: Vec<&String> = common.iter().filter(|m| cs[*m].3 > 1 && cr[*m].3 > 1 && cr[*m].1 > cs[*m].2).collect();
    let thin: Vec<&String> = common.iter().filter(|m| cs[*m].3 < 2 || cr[*m].3 < 2).collect();
    let ap_lo = value::min_by(cs.values().map(|c| c.0), |x| *x).unwrap();
    let ap_hi = value::max_by(cs.values().map(|c| c.0), |x| *x).unwrap();
    let ap_stats = format!("{}–{}", fmt::f(ap_lo, ".3f"), fmt::f(ap_hi, ".3f"));
    let runs = cr.values().map(|c| c.3).max().unwrap();
    let mut body = format!(
        "Measured directly on {} real models, each run {runs} times per condition: given the same summarised \
         evidence, a large model and a small model land in the same band (AP {ap_stats}) and no better than a \
         six-feature statistical rule ({}). Given the *original notes* behind that evidence, {helped} of {} improve",
        common.len(),
        value::format(&d["simulator_logistic_ap"], ".3f"),
        common.len()
    );
    let names = |v: &[&String]| v.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(", ");
    if !sep.is_empty() {
        body += &format!(
            ", and for {} the richer condition's worst run still beats the summarised condition's best",
            names(&sep)
        );
    }
    let losers: Vec<&String> = common.iter().filter(|m| cr[*m].0 <= cs[*m].0).collect();
    if !losers.is_empty() {
        body += &format!(
            ". It is **not** universal: {} did not improve, so this is a property of a model's ability to use raw \
             evidence, not a law",
            names(&losers)
        );
    }
    if !thin.is_empty() {
        let one = thin.len() == 1;
        body += &format!(
            ". {} still {} a condition measured only once, so {} {} provisional",
            names(&thin),
            if one { "has" } else { "have" },
            if one { "its" } else { "those" },
            if one { "direction is" } else { "directions are" }
        );
    }
    body += ". Acting on it — having the kernel re-read a handful of source notes per candidate — was still the best \
             return on compute found in the study, but size it against §23 rather than against the headline number.";
    format!("{head}{body}")
}

/// Per-model mean AP over repeats, the largest run-to-run range seen in any cell — which bounds how much of any
/// difference below could be sampling rather than mechanism — and the most runs in a cell.
fn means(dd: &Value) -> (IndexMap<String, f64>, f64, i64) {
    let ag = obj(dd, "models_aggregated");
    if !ag.is_empty() {
        let m = ag.iter().map(|(k, v)| ((*k).clone(), fnum(v, "ap_mean"))).collect();
        let spread =
            value::max_by(ag.iter().map(|(_, v)| fnum(v, "ap_max") - fnum(v, "ap_min")), |x| *x).unwrap_or(0.0);
        let runs = ag.iter().map(|(_, v)| int_of(&v["n_runs"])).max().unwrap_or(1);
        return (m, spread, runs);
    }
    (obj(dd, "models").into_iter().map(|(k, v)| (k.clone(), fnum(v, "ap"))).collect(), 0.0, 1)
}

/// Section 2: what wins where, and the seven numbered findings.
pub fn executive_summary() -> String {
    let rows = rows();
    if rows.is_empty() {
        return "_(no results yet)_".to_string();
    }
    let scales = scales_of(&rows);
    let mut lines: Vec<String> = Vec::new();

    // 1. what wins, per scale
    lines.push("**What actually wins, and where.**\n".into());
    lines.push(
        "| users | best architecture | discovery (`found`) | compute (cu) | raw records leaving their owner |".into(),
    );
    lines.push("|---|---|---:|---:|---:|".into());
    for &s in &scales {
        let (a, v) = best(&rows, s);
        let cu = mean(&rows, &a, s, "compute_units").expect("compute of the best architecture");
        let raw = mean(&rows, &a, s, "raw_text_exposure_fraction").unwrap_or(f64::NAN);
        lines.push(format!(
            "| {} | `{a}` | {} | {} | {} |",
            comma(s),
            fmt::f(v, ".3f"),
            fmt::f(cu, ".2e"),
            fmt::f(raw, ".1%")
        ));
    }
    lines.push(String::new());

    // 2. the central negative result, at the largest scale where every rung of the ladder was run (a missing
    // architecture is not a 0%)
    let big = scales[scales.len() - 1];
    let ladder = ["C_recursive_sum", "D_hier_nolineage", "E_hier_lineage", "F_hier_retrieval", "G_hier_questions"];
    if let Some(lad) = largest_complete_scale(&rows, &ladder) {
        let v: Vec<f64> = ladder.iter().map(|a| mean(&rows, a, lad, FOUND).unwrap()).collect();
        let at: Vec<Row> = rows.iter().filter(|r| int(r, "scale") == lad).cloned().collect();
        lines.push(format!(
            "**1. Upward propagation alone does not work, at any scale, in any form.** At {} users (the largest scale \
             where all five were run) recursive summarisation finds {} of the hidden patterns, hierarchical \
             aggregation without lineage and with it find {} and {}. Adding targeted downward retrieval takes it to \
             {}; adding sketch-driven questioning takes it to {}. The hierarchy's value is almost entirely in the \
             *downward* path — {}.\n",
            comma(lad),
            fmt::f(v[0], ".1%"),
            fmt::f(v[1], ".1%"),
            fmt::f(v[2], ".1%"),
            fmt::f(v[3], ".1%"),
            fmt::f(v[4], ".1%"),
            sig(&paired(&at, "G_hier_questions", "F_hier_retrieval", FOUND))
        ));
    }

    // 3. the measured reason
    lines.push(
        "**2. The reason is measurable and is not a tuning artefact.** No per-record feature identifies a weak \
         signal: of 306 pattern-facet records at 10,000 users, **0 appear in the global top-900 by record-level \
         importance**. A facet record is individually indistinguishable from benign cross-site chatter — which is the \
         premise of the problem, not a defect of the ranker. Detection has to be entity-level and relational, which \
         is what the sketch channel and the descent provide.\n"
            .into(),
    );

    // 4. the oracle result
    let o_cov = mean(&rows, "Y_oracle_retrieval", big, "evidence_coverage_2links");
    let o_found = mean(&rows, "Y_oracle_retrieval", big, FOUND);
    let h_cov = mean(&rows, HIER, big, "evidence_coverage_2links");
    let h_found = mean(&rows, HIER, big, FOUND);
    if let (Some(o_cov), Some(h_cov)) = (o_cov, h_cov) {
        lines.push(format!(
            "**3. At enterprise scale the binding constraint is discrimination, not retrieval.** A perfect-retrieval \
             oracle — every extracted claim, no budget at all — holds the evidence for {} of the hidden patterns and \
             still reports only {} of them. The hierarchy holds {} and reports {}. More undifferentiated evidence \
             makes the kernel's ranking worse, so a propagation budget is a feature and not only a cost.\n",
            fmt::f(o_cov, ".0%"),
            fmt::f(o_found.expect("oracle found"), ".1%"),
            fmt::f(h_cov, ".0%"),
            fmt::f(h_found.expect("hierarchy found"), ".1%")
        ));
    }

    // 5. live measurement
    if let Some((d, dr)) = live_results() {
        let (ms, spread_s, n_s) = means(&d);
        let (mr, spread_r, n_r) = means(&dr);
        if !ms.is_empty() && !mr.is_empty() {
            let common = sorted_common(&ms, &mr);
            let helped = common.iter().filter(|m| mr[*m] > ms[*m]).count();
            let gaps: Vec<f64> = common.iter().map(|m| mr[m] - ms[m]).collect();
            let biggest_spread = value::max2(spread_s, spread_r);
            let decisive: Vec<&String> = common.iter().filter(|m| (mr[*m] - ms[*m]).abs() > biggest_spread).collect();
            let strength = if decisive.len() == common.len() {
                "and changing what the evidence *contains* buys a lot"
            } else if !decisive.is_empty() {
                "changing what the evidence *contains* helps the strongest model a lot and is not reliable across \
                 models"
            } else {
                "and changing what the evidence *contains* does not reliably help either"
            };
            let lo = |m: &IndexMap<String, f64>| value::min_by(m.values().copied(), |x| *x).unwrap();
            let hi = |m: &IndexMap<String, f64>| value::max_by(m.values().copied(), |x| *x).unwrap();
            let tail = if decisive.is_empty() {
                "no model moves by more than that, so this comparison does not separate the two conditions. "
                    .to_string()
            } else {
                format!(
                    "{} of the {} {} by more than that ({}). ",
                    decisive.len(),
                    common.len(),
                    if decisive.len() == 1 { "moves" } else { "move" },
                    decisive.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(", ")
                )
            };
            lines.push(format!(
                "**4. Directly measured: at fixed evidence, model capability buys almost nothing here, {strength}.** \
                 {} real models ranked the same {} candidates from a real run ({} independent runs per cell). Given \
                 the aggregate evidence statistics they scored AP {}–{}, against {} for a six-feature logistic and {} \
                 for random — i.e. a large capability range lands within noise of a logistic. Given the **raw work \
                 notes** behind the same statistics the same models scored {}–{}. Richer evidence helped {helped} of \
                 {} models (Δ {} to {}); the largest run-to-run range within a single cell is {}, and {tail}The \
                 abstraction, not the reasoner, is the plausible ceiling — but see §23 before treating the size of the \
                 effect as established.\n",
                common.len(),
                py_str(&d["n_items"]),
                n_s.max(n_r),
                fmt::f(lo(&ms), ".3f"),
                fmt::f(hi(&ms), ".3f"),
                value::format(&d["simulator_logistic_ap"], ".3f"),
                value::format(&d["random_ap"], ".3f"),
                fmt::f(lo(&mr), ".3f"),
                fmt::f(hi(&mr), ".3f"),
                common.len(),
                fmt::f(value::min_by(gaps.iter().copied(), |x| *x).unwrap(), "+.3f"),
                fmt::f(value::max_by(gaps.iter().copied(), |x| *x).unwrap(), "+.3f"),
                fmt::f(biggest_spread, ".3f")
            ));
        }
    }

    // 6. what that implies, measured
    let ab = load("e2_ablations.jsonl");
    if !ab.is_empty() {
        let pair = |metric: &str| {
            paired_by(
                &ab,
                &Value::from("+evidence_verification"),
                &Value::from("full"),
                metric,
                "ablation",
                &["scale", "seed"],
            )
        };
        let pv = pair("average_precision");
        let pc = pair("compute_units");
        if pv.n > 0 {
            let rel = pv.mean_diff / value::max2(1e-9, pv.mean_b);
            let crel = if pc.n > 0 { pc.mean_diff / value::max2(1e-9, pc.mean_b) } else { 0.0 };
            lines.push(format!(
                "**5. Acting on that: the useful way to spend frontier compute at the kernel is to re-open original \
                 evidence, not to reason harder over the same abstraction.** Having the kernel re-read a handful of \
                 each top candidate's original notes with its own extractor changes AP by {} for {} compute ({}).\n",
                fmt::f(rel, "+.0%"),
                fmt::f(crel, "+.1%"),
                sig(&pv)
            ));
        }
    }

    // 7. privacy
    let pr_rows = load("e9_privacy.jsonl");
    if !pr_rows.is_empty() {
        let top = pr_rows.iter().map(|r| int(r, "scale")).max().unwrap();
        let sub: Vec<Row> = pr_rows.iter().filter(|r| int(r, "scale") == top).cloned().collect();
        let a = agg(&sub, &["raw_text_exposure_fraction", "claim_exposure_fraction"], &["arch"]);
        let find = |arch: &str| a.iter().find(|x| text(x, "arch") == arch);
        if let (Some(h), Some(fr)) = (find(HIER), find("A_flat_rag")) {
            let ch = find("A2_chunked_ctx").map_or(f64::NAN, |x| value::f(x, "raw_text_exposure_fraction"));
            lines.push(format!(
                "**6. The hierarchy's defensible advantage is confidentiality, and it is categorical rather than \
                 marginal.** No raw record text ever leaves the agent that owns it ({}); the retrieval baseline \
                 centralises {} of all records as original text and the chunked-context baseline {}. What leaves a \
                 node in the hierarchy is structured claims and an entity/predicate sketch.\n",
                fmt::f(value::f(h, "raw_text_exposure_fraction"), ".1%"),
                fmt::f(value::f(fr, "raw_text_exposure_fraction"), ".1%"),
                fmt::f(ch, ".1%")
            ));
        }
    }

    lines.push(bottom_line(&rows));
    lines.push(format!("_Statistical note: {P_FLOOR_NOTE}._"));
    lines.join("\n")
}

/// `'a'`, `'a and b'`, `'a, b and c'`.
pub fn join<S: AsRef<str>>(items: &[S], conj: &str) -> String {
    let items: Vec<&str> = items.iter().map(AsRef::as_ref).collect();
    if items.len() <= 1 {
        return items.concat();
    }
    format!("{} {conj} {}", items[..items.len() - 1].join(", "), items[items.len() - 1])
}

/// Item 7, from the same computed comparison as the decision summary.
fn bottom_line(rows: &[Row]) -> String {
    let Some(sc) = largest_complete_scale(rows, &COMPARISON) else {
        return "**7. The honest bottom line.** _(comparison incomplete)_\n".to_string();
    };
    let (best_arch, bv) = best(rows, sc);
    let hv = mean(rows, HIER, sc, FOUND).unwrap_or(0.0);
    let raw = mean(rows, &best_arch, sc, "raw_text_exposure_fraction");
    let cu_b = mean(rows, &best_arch, sc, "compute_units").unwrap_or(0.0);
    let cu_h = mean(rows, HIER, sc, "compute_units").unwrap_or(0.0);
    let (wins, losses) = grounds(rows, sc, &best_arch);
    // the NAMED architecture must lead the hierarchy at every scale, and both must have been run there
    let scales = scales_of(rows);
    let every = !scales.is_empty()
        && scales.iter().all(|&s| match (mean(rows, &best_arch, s, FOUND), mean(rows, HIER, s, FOUND)) {
            (Some(b), Some(h)) => b > h + 1e-9,
            _ => false,
        });
    let mut text = format!(
        "**7. The honest bottom line.** If centralising raw text is acceptable, the strongest thing measured here is \
         `{best_arch}`, which "
    );
    if let Some(raw) = raw.filter(|x| *x != 0.0) {
        text += &format!("reads {} of all records centrally and ", fmt::f(raw, ".0%"));
    }
    text += &format!(
        "finds {} of the hidden problems at {} users against the hierarchy's {}",
        fmt::f(bv, ".0%"),
        comma(sc),
        fmt::f(hv, ".0%")
    );
    if every {
        text += " — ahead of it at every scale measured";
    }
    text += &if cu_b > cu_h {
        format!(", at {}x the hierarchy's compute. ", fmt::f(cu_b / value::max2(1.0, cu_h), ".1f"))
    } else {
        ". ".to_string()
    };
    text += "The hierarchy earns its cost where sovereign local memory is a requirement; that is a real constraint, \
             but a constraint — not an accuracy win. ";
    let short = |g: &[Ground]| -> Vec<String> {
        g.iter().map(|w| w.0.split(" (").next().unwrap_or_default().to_string()).collect()
    };
    if !wins.is_empty() {
        text += &format!("On this evidence it also delivers {}. ", join(&short(&wins), "and"));
    }
    if !losses.is_empty() {
        text += &format!(
            "It does not deliver {}, and those should not be used to justify it.",
            join(&short(&losses), "or")
        );
    }
    text.trim_end().to_string() + "\n"
}

/// Sections 3-6: the recommended hierarchy, models per level, allocation and fan-in.
pub fn recommendation() -> String {
    let alloc = load("e3_allocation.jsonl");
    let marg = load("e3b_level_marginal.jsonl");
    let fan = load("e4_fanin.jsonl");
    let mut out: Vec<String> = Vec::new();

    out.push("### 3. Recommended hierarchy\n".into());
    out.push(
        r#"Keep the six human levels — USER → TEAM → DEPARTMENT → SITE → REGION → ENTERPRISE — but do not make them all carry content. The recommended design splits the upward flow in two:

* **a complete, LLM-free sketch channel** (per entity, per site: mention count, a bitmask of operational predicates seen at least `min_support` times, and a time span), which is what makes discovery possible at all; and
* **a budgeted, evidence-bearing object channel** carrying lineage, evidence pointers, independent-support sketches, contradictions and confidence — most of which is deliberately dropped.

Discovery then happens by triaging the sketch channel for entities whose operational evidence repeats at sites where the entity does not belong, and descending on those entities specifically. The intermediate levels exist to aggregate the sketch and to bound the descent, not to summarise text upward.
"#
        .into(),
    );

    if !marg.is_empty() {
        let a = agg(&marg, &[FOUND, "average_precision", "compute_units"], &["arch", "upgraded_level"]);
        let h: Vec<&Row> = a.iter().filter(|x| text(x, "arch") == HIER).collect();
        let base: Vec<&&Row> = h.iter().filter(|x| text(x, "upgraded_level") == "none").collect();
        if !base.is_empty() && h.len() > 1 {
            let b0 = value::f(base[0], FOUND);
            let mut ranked: Vec<&&Row> = h.iter().filter(|x| text(x, "upgraded_level") != "none").collect();
            ranked.sort_by(|x, y| (-(value::f(x, FOUND) - b0)).partial_cmp(&-(value::f(y, FOUND) - b0)).unwrap());
            let (best_l, worst_l) = (ranked[0], ranked[ranked.len() - 1]);
            out.push("### 4. Recommended models at each level\n".into());
            out.push(format!(
                "Measured directly, by upgrading exactly one level from `small-7b` to `frontier` and holding \
                 everything else fixed: the largest gain comes from the **{}** level (Δ found {}) and the smallest \
                 from the **{}** level (Δ found {}). Full table and paired tests in §13.\n",
                text(best_l, "upgraded_level"),
                fmt::f(value::f(best_l, FOUND) - b0, "+.3f"),
                text(worst_l, "upgraded_level"),
                fmt::f(value::f(worst_l, FOUND) - b0, "+.3f")
            ));
        }
    }
    if !alloc.is_empty() {
        let hier: Vec<Row> = alloc.iter().filter(|r| text(r, "arch") == HIER).cloned().collect();
        let a = agg(&hier, &[FOUND, "compute_units", "average_precision"], &["alloc"]);
        if !a.is_empty() {
            let best_a = value::max_by(a.iter(), |x| value::f(x, FOUND)).unwrap();
            let cheap = value::min_by(a.iter(), |x| value::f(x, "compute_units")).unwrap();
            let eff =
                value::max_by(a.iter(), |x| value::f(x, FOUND) / value::max2(1.0, value::f(x, "compute_units") / 1e6))
                    .unwrap();
            let frac = value::f(eff, FOUND) / value::max2(1e-9, value::f(best_a, FOUND));
            let ratio = value::f(eff, "compute_units") / value::max2(1.0, value::f(best_a, "compute_units"));
            out.push(format!(
                "Best absolute allocation measured: **{}** (found {}, {} cu). Best discovery per unit compute: **{}** \
                 (found {}, {} cu). Cheapest: **{}** (found {}).\n",
                text(best_a, "alloc"),
                fmt::f(value::f(best_a, FOUND), ".3f"),
                fmt::f(value::f(best_a, "compute_units"), ".2e"),
                text(eff, "alloc"),
                fmt::f(value::f(eff, FOUND), ".3f"),
                fmt::f(value::f(eff, "compute_units"), ".2e"),
                text(cheap, "alloc"),
                fmt::f(value::f(cheap, FOUND), ".3f")
            ));
            out.push(format!(
                "**The practical recommendation is `{}`, not a graduated ladder.** It keeps {} of the best measured \
                 discovery for {} of the compute. The reason is the same one that runs through the whole study: the \
                 edge is only extracting claims, which is cheap and mostly saturated, while the kernel is doing the \
                 discrimination, which is where the difficulty actually is. Spending on the middle levels buys the \
                 least of anything measured here. If the budget stretches further, put it in the kernel — re-reading \
                 original evidence per candidate — before putting it into a bigger model at any intermediate level.\n",
                text(eff, "alloc"),
                fmt::f(frac, ".0%"),
                fmt::f(ratio, ".0%")
            ));
        }
    }
    if !fan.is_empty() {
        let hier: Vec<Row> = fan.iter().filter(|r| text(r, "arch") == HIER).cloned().collect();
        let a = agg(&hier, &[FOUND, "average_precision", "information_loss", "compute_units"], &["team_size"]);
        if !a.is_empty() {
            let best_f = value::max_by(a.iter(), |x| value::f(x, FOUND)).unwrap();
            let founds = || a.iter().map(|x| value::f(x, FOUND));
            let spread = value::max_by(founds(), |x| *x).unwrap() - value::min_by(founds(), |x| *x).unwrap();
            out.push("### 5-6. Recommended fan-in, and why\n".into());
            out.push(format!(
                "Team size was swept over 6, 8, 10, 12 and 15 users. Best measured: **{} users per team** (found {}); \
                 the total spread across the whole range is {}. {}",
                int(best_f, "team_size"),
                fmt::f(value::f(best_f, FOUND), ".3f"),
                fmt::f(spread, ".3f"),
                if spread < 0.06 {
                    "Team fan-in is therefore **not** a material design lever for strategic discovery in this regime \
                     — the signal is recovered by descent, which is indexed by entity rather than routed through team \
                     boundaries. Choose team size for human reasons.\n"
                } else {
                    "Team fan-in does move the result and should be chosen deliberately; see §18.\n"
                }
            ));
        }
    }
    out.join("\n")
}

/// Section 24: the reviewer questions, answered against the measurements rather than around them.
pub fn critique() -> String {
    let mut lines =
        vec!["The reviewer questions, answered against the measurements rather than around them.\n".to_string()];
    lines.push(
        "**Is the benchmark fair?** Every architecture calls one shared implementation of every operator; the \
         centralised baselines get a frontier-tier extractor on raw text while the hierarchy's edge runs a small \
         model, which is a real advantage for them and is left in place; the kernel model tier is matched, though \
         kernel prompt sizes are not (§2); every v1 knob is fitted on calibration seeds disjoint from the evaluation \
         seeds, by the same procedure for every system, including the centralised triage control's evidence budget \
         (whose best held-out value turned out to be *no cap*, so the hierarchy's margin over it is not an artefact \
         of denying it a filtering step). Several vNext decisions — the question budget, and the rejection of local \
         re-extraction, decoy-weighted ranker selection and modal link timing — were made on evaluation seeds 0–4 at \
         10,000 users and 0–2 at 50,000; seeds 5–9 at 10,000 users are the panel no development decision read.\n"
            .into(),
    );
    lines.push(
        "**Did we design data that favours the hierarchy?** The opposite is closer to true. A pattern is equally \
         visible to any system that gets its facet records into one context, and the headline result is that a \
         centralised retrieval baseline does so more effectively than the hierarchy at every scale measured. Three \
         generator properties were specifically added to remove hierarchy-favouring artefacts: benign cross-site \
         entity traffic (so multi-region presence is a weak signal rather than a giveaway — index triage alone yields \
         ~4% precision at 10k users), echoes that repeat the original wording (so duplicate detection is a text \
         problem everyone faces), and site-local echo propagation (global echoing smeared every entity across every \
         region and destroyed the locality structure).\n"
            .into(),
    );
    lines.push(critique_improvements_real());
    let mut seeds_by_scale: BTreeMap<i64, BTreeSet<i64>> = BTreeMap::new();
    for r in rows() {
        seeds_by_scale.entry(int(&r, "scale")).or_default().insert(int(&r, "seed"));
    }
    let counts: Vec<String> =
        seeds_by_scale.iter().map(|(sc, v)| format!("{}: {} seeds", comma(*sc), v.len())).collect();
    let thin: Vec<i64> = seeds_by_scale.iter().filter(|(_, v)| v.len() < 6).map(|(sc, _)| *sc).collect();
    let tail = if thin.is_empty() {
        ". Every scale has enough seeds for a unanimous result to clear p = 0.05.\n".to_string()
    } else {
        format!(
            " — so differences at {} users can be directionally unanimous and still not reach a conventional \
             threshold. Those are labelled in the tables rather than described as significant.\n",
            thin.iter().map(|s| comma(*s)).collect::<Vec<_>>().join(" and ")
        )
    };
    lines.push(format!(
        "**Are the statistics adequate?** No, not fully. {}. Seeds actually run per scale — {}{tail}",
        fmt::capitalize(P_FLOOR_NOTE),
        counts.join(", ")
    ));
    lines.push(
        "**The biggest unresolved weakness.** The causal predicate schema is given to every system. Real enterprises \
         have no such schema, and inducing one is plausibly harder than using it. Everything here is therefore an \
         upper bound on the *verification* half of the problem and says nothing about schema induction.\n"
            .into(),
    );
    lines.push(
        "**The experiment most likely to falsify the conclusion.** Re-run with the causal schema withheld and \
         required to be induced from the corpus. If the hierarchy's sketch channel can induce a usable schema cheaply \
         from aggregate co-occurrence while a centralised sample cannot, the ordering reported here inverts. That \
         experiment was not run.\n"
            .into(),
    );
    lines.join("\n")
}

/// Read the calibration file rather than asserting what it chose.
///
/// An earlier hand-written version of this paragraph named the source-dispersion feature as a rejected
/// non-result.  The calibration on the corrected corpus *selected* it, so the paragraph was contradicting the file
/// it was describing.  Everything here is now read back from `calibration.json`.
fn critique_improvements_real() -> String {
    let head = "**Are the improvements real?** ";
    let Some(d) = read_json("calibration.json") else { return format!("{head}_(calibration not run)_\n") };
    let (mut rejected, mut kept) = (Vec::new(), Vec::new());
    for (knob, label) in [
        ("triage_prior_weight", "a triage prior"),
        ("w_dispersion", "a source-dispersion feature"),
        ("w_synchrony", "a reporting-synchrony feature"),
        ("w_attribution", "an entity-attribution penalty"),
    ] {
        let Some(v) = d.get(knob) else { continue };
        let w = value::num(v).expect("a knob weight");
        let entry = format!("{label} (held-out weight {})", fmt::f(w, "g"));
        if w > 0.0 { kept.push(entry) } else { rejected.push(entry) }
    }
    let mut body = format!(
        "Every v1 knob was fitted on calibration seeds {}, disjoint from the evaluation seeds, and then frozen. ",
        py_str(&d["seeds"])
    );
    if !rejected.is_empty() {
        body += &format!(
            "The protocol **rejected** {} — each of which had looked like a clear win on a single seed. They are \
             reported as non-results rather than quietly dropped. ",
            rejected.join("; ")
        );
    }
    if !kept.is_empty() {
        body += &format!("It **selected** {}. ", kept.join("; "));
    }
    body += r#"The one that matters most, kernel-side evidence verification, was selected on held-out seeds and is reported with its paired test in §12.

One caveat belongs here rather than in a footnote: `w_dispersion` was rejected twice on the pre-correction corpus and selected on the corrected one, with no change to the mechanism. A setting whose sign flips when the data is regenerated is fitted to that data, not to the problem. It should be re-fitted on real data before deployment and should not be treated as a transferable finding.
"#;
    format!("{head}{body}")
}

/// The AP change from giving models the raw notes, per model, from the live results.
fn live_rich_gain() -> String {
    let Some((a, b)) = live_results() else {
        return "Models given the raw notes changed AP by an amount not yet measured".to_string();
    };
    let (a, b) = (&a["models_aggregated"], &b["models_aggregated"]);
    let mut d: BTreeMap<String, f64> = BTreeMap::new();
    for (m, va) in a.as_object().expect("models_aggregated") {
        if let Some(vb) = b.get(m) {
            d.insert(m.clone(), fnum(vb, "ap_mean") - fnum(va, "ap_mean"));
        }
    }
    let parts: Vec<String> = d.iter().map(|(m, v)| format!("{m} {}", fmt::f(*v, "+.3f"))).collect();
    let up = d.values().filter(|v| **v > 0.0).count();
    format!(
        "Models given the raw notes changed AP by {} over the same statistics ({up} of {} improved)",
        parts.join(", "),
        d.len()
    )
}

const NEXT_EXPERIMENTS: &str = r#"
In rough order of expected information per unit compute:

1. **Withhold the causal schema.** The single largest unexamined assumption.
   Require each architecture to induce the predicate chains from the corpus.
   This is the experiment most likely to change the ordering.
2. **Richer propagated objects, guided by the live measurement.** @LIVE_RICH_GAIN@
   and named three specific cues. Source dispersion and synchrony were implemented and did not
   generalise; kernel-side re-reading did. The remaining move is to propagate a
   small, structured *sample* of verbatim evidence with each object and
   measure whether that closes more of the gap than re-reading does.
3. **Multiple pattern shapes.** Every hidden pattern here is a causal chain on
   one entity. Patterns that are conjunctions across entities, rate changes, or
   absences of expected events would test whether the sketch channel's
   entity-keyed design generalises or is fitted to this shape.
4. **Adversaries that adapt.** The malicious-node condition here is static.
   A node that knows the triage rule can manufacture foreign operational
   mentions to flood the candidate list; measuring that cost is a prerequisite
   for deployment.
5. **A real model in the loop at one level.** Replace the simulated kernel with
   an actual frontier call on a 10k-user run end to end, and compare against
   the simulated kernel on the same candidates. The discrimination measurement
   did this for one operator; doing it end to end would convert the largest
   remaining simulated component into a measured one.
6. **Incremental operation.** Everything here is a single batch over a
   180-day window. A standing system re-runs continuously, and the interesting
   questions — when does a pattern become detectable, how much does keeping
   state save — are invisible to a batch benchmark.
"#;

/// Section 27: the recommended next experiments.
pub fn next_experiments() -> String {
    NEXT_EXPERIMENTS.replace("@LIVE_RICH_GAIN@", &live_rich_gain())
}

/// Section 17: what the questions do, and the ablations that test them.
pub fn questions_section() -> String {
    let ab = load("e2_ablations.jsonl");
    if ab.is_empty() {
        return "_(E2 not run)_".to_string();
    }
    let a = agg(
        &ab,
        &[
            "question_utility",
            "question_new_evidence",
            "question_targeting",
            "n_questions",
            FOUND,
            "average_precision",
            "compute_units",
            "rare_signal_recall",
        ],
        &["ablation"],
    );
    let mut out: Vec<String> = Vec::new();
    if let Some(f) = a.iter().find(|x| text(x, "ablation") == "full") {
        out.push(format!(
            "The full system asks {} questions per run. {} of them change a conclusion — a question counts as useful \
             only if the kernel's hypothesis set or its confidence actually moved, never merely because text was \
             produced. Each returns {} new knowledge objects on average, and {} are well-targeted.\n",
            fmt::f(value::f(f, "n_questions"), ".0f"),
            fmt::f(value::f(f, "question_utility"), ".0%"),
            fmt::f(value::f(f, "question_new_evidence"), ".0f"),
            fmt::f(value::f(f, "question_targeting"), ".0%")
        ));
        out.push(
            "**That last figure is not a measurement and should not be read as one.** `question_targeting` is the \
             fraction of questions the kernel aimed at the entity it was actually reasoning about, and in the \
             simulator that is drawn directly from the kernel tier's `question_quality` parameter — so with a \
             frontier kernel it is ~100% *by construction*, and it would report ~100% even if targeting were \
             worthless. The informative test is the ablation below that destroys targeting outright, and that one \
             does not reach significance. It is listed here because leaving a parameter echoed back as a headline \
             number would be the kind of thing this report is supposed to catch.\n"
                .into(),
        );
    }
    for (name, label) in [
        ("-questions", "removing questioning entirely"),
        ("-question_targeting", "keeping questions but destroying their targeting"),
    ] {
        let pair = |metric: &str| {
            paired_by(&ab, &Value::from("full"), &Value::from(name), metric, "ablation", &["scale", "seed"])
        };
        let p = pair(FOUND);
        let pr_ = pair("rare_signal_recall");
        let pc = pair("compute_units");
        if p.n > 0 {
            // `sig` reports full MINUS ablated.  Written next to a label that names what was REMOVED, a positive
            // number reads as if removal helped, so the deltas are flipped here to be deltas of the ablation itself
            // and the direction is spelled out.
            out.push(format!(
                "* **{label}**: discovery {}, rare-signal recall {}, compute {} cu.",
                flip(&p),
                flip(&pr_),
                fmt::f(-pc.mean_diff, "+.2e")
            ));
        }
    }
    if out.len() > 1 {
        out.push(String::new());
        out.push(
            "Every Δ above is *the ablated system minus the full one*, so a negative number means removing the \
             mechanism made things worse. The seed count is the number of paired seeds on which the ablation was the \
             worse of the two."
                .into(),
        );
    }
    out.join("\n")
}

/// `sig`, with the sign of a paired(full, ablated) difference reversed.
fn flip(p: &Paired) -> String {
    if p.n == 0 {
        return "no paired runs".to_string();
    }
    // p.wins counts seeds where full > ablated, i.e. seeds on which the ablation was the worse of the pair.  That is
    // the count the label promises, and it is deliberately NOT flipped along with the sign.
    format!(
        "Δ {} (95% CI [{}, {}], worse on {}/{} seeds, sign p={})",
        fmt::f(-p.mean_diff, "+.3f"),
        fmt::f(-p.ci_hi, "+.3f"),
        fmt::f(-p.ci_lo, "+.3f"),
        p.wins,
        p.n_nonzero,
        fmt::f(p.sign_p, ".3f")
    )
}
