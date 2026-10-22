//! Summarise `artifacts/loss_funnel.jsonl` into the tables `LOSS_ACCOUNTING.md` is built from (`loss_report.py`).
//!
//! Everything here is a count over per-pattern rows; nothing is modelled.
//!
//! ```sh
//! mycelic-research loss_report            # print markdown
//! ```

use crate::analysis::{boot_ci, load_path};
use crate::loss_accounting::{FLAT_STAGES, HIER_STAGES, funnel_path};
use crate::np::math;
use crate::py::fmt;
use crate::py::value::{Row, num, truthy};
use serde_json::Value;
use std::borrow::Borrow;
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

pub const STAGE_LABEL: [(&str, &str); 10] = [
    ("extracted", "extracted (>=2 facets, right entity+predicate)"),
    ("sketch_visible", "visible to sketch triage (>=2 foreign sites, >=2 regions, span>=2)"),
    ("in_triage", "on the triage candidate list"),
    ("questioned", "inside the question budget"),
    ("descent_reached", "descent reached a facet holder"),
    ("in_pool", ">=2 gold links in kernel pool (= evidence coverage)"),
    ("candidate", "candidate formed on right entity + chain"),
    ("matched_any", "matched primary rule at ANY confidence"),
    ("matched_tau", "matched with confidence >= 0.5"),
    ("in_register", "survived the register cut (= reported)"),
];

/// Labels for the loss tables, where the row means "died HERE".
pub const LOSS_LABEL: [(&str, &str); 3] = [
    ("in_register", "cut from the register (matched at >= 0.5, out-ranked)"),
    ("matched_tau", "matched, but confidence < 0.5"),
    ("reported", "**reported**"),
];

pub const SKETCH_FAIL_LABEL: [(&str, &str); 4] = [
    ("no_site_bit", "no site set a predicate bit (support threshold)"),
    ("foreign_lt2", "fewer than 2 foreign sites"),
    ("regions_lt2", "foreign sites in fewer than 2 regions"),
    ("span_lt2", "causal span < 2 across foreign sites"),
];

fn lookup(table: &[(&'static str, &'static str)], key: &str) -> Option<&'static str> {
    table.iter().find(|e| e.0 == key).map(|e| e.1)
}

/// `STAGE_LABEL.get(stage)`
pub fn stage_label(stage: &str) -> Option<&'static str> {
    lookup(&STAGE_LABEL, stage)
}

/// `LOSS_LABEL.get(stage)`
pub fn loss_label(stage: &str) -> Option<&'static str> {
    lookup(&LOSS_LABEL, stage)
}

/// `r.get("summary")` is truthy: a per-(arch, scale, seed) summary row rather than a pattern row.
pub fn is_summary(r: &Row) -> bool {
    r.get("summary").is_some_and(truthy)
}

/// `r.get(key) is not None`
pub(crate) fn has(r: &Row, key: &str) -> bool {
    r.get(key).is_some_and(|v| !v.is_null())
}

/// `bool(r[key])`
pub(crate) fn flag(r: &Row, key: &str) -> bool {
    truthy(r.get(key).unwrap_or_else(|| panic!("funnel row has no {key:?}")))
}

pub(crate) fn int(r: &Row, key: &str) -> i64 {
    let v = &r[key];
    v.as_i64().unwrap_or_else(|| num(v).unwrap_or_else(|| panic!("{key} is not a number: {v}")) as i64)
}

pub(crate) fn text<'a>(r: &'a Row, key: &str) -> &'a str {
    r.get(key).and_then(Value::as_str).unwrap_or_else(|| panic!("row has no string {key:?}"))
}

/// `load(path)`: the funnel rows.  While a rerun is rebuilding the funnel the archived v1 funnel stands in, so an
/// interim document never shows empty tables (loss_accounting creates the file as soon as it starts, so an empty
/// file counts as missing).
pub fn load(path: &Path) -> Vec<Row> {
    let missing = std::fs::metadata(path).map(|m| m.len() == 0).unwrap_or(true);
    let mut path = path.to_path_buf();
    if missing {
        let alt = path.parent().unwrap_or(Path::new("")).join("v1").join(path.file_name().unwrap_or_default());
        if alt.exists() {
            path = alt;
        } else {
            return Vec::new();
        }
    }
    load_path(&path)
}

/// `load()`: the default funnel, `artifacts/loss_funnel.jsonl`.
pub fn load_default() -> Vec<Row> {
    load(&funnel_path())
}

/// The stages a system's funnel has: read from its first pattern row when there is one, else from its name.
pub fn stages_for<R: Borrow<Row>>(arch: &str, rows: Option<&[R]>) -> Vec<&'static str> {
    if let Some(r0) = rows
        .filter(|r| !r.is_empty())
        .and_then(|rows| rows.iter().map(Borrow::borrow).find(|r| text(r, "arch") == arch && !is_summary(r)))
    {
        if r0.contains_key("sketch_visible") {
            return HIER_STAGES.to_vec();
        }
        return if has(r0, "extracted") { FLAT_STAGES.to_vec() } else { FLAT_STAGES[1..].to_vec() };
    }
    if ["H_", "G_", "F_", "E_", "J_", "I_", "V"].iter().any(|p| arch.starts_with(p)) {
        HIER_STAGES.to_vec()
    } else {
        FLAT_STAGES.to_vec()
    }
}

/// Pattern rows (no summaries) at one scale.
fn pattern_rows(rows: &[Row], scale: i64) -> Vec<&Row> {
    rows.iter().filter(|r| !is_summary(r) && int(r, "scale") == scale).collect()
}

fn sorted_archs(pr: &[&Row]) -> Vec<String> {
    pr.iter().map(|r| text(r, "arch").to_string()).collect::<BTreeSet<_>>().into_iter().collect()
}

/// Stage-by-stage survival per architecture, mean over seeds with a bootstrap interval, for all / rare / common
/// patterns.
pub fn survival_table(rows: &[Row], scale: i64, subset: &str) -> String {
    let mut pr = pattern_rows(rows, scale);
    if subset == "rare" {
        pr.retain(|r| flag(r, "rare"));
    } else if subset == "common" {
        pr.retain(|r| !flag(r, "rare"));
    }
    let archs = sorted_archs(&pr);
    if archs.is_empty() {
        return "_(no rows)_".to_string();
    }
    let head = format!("| stage | {} |", archs.join(" | "));
    let sep = format!("|---|{}", "---:|".repeat(archs.len()));
    let mut lines = vec![head, sep];
    for s in HIER_STAGES {
        let mut cells = Vec::new();
        for a in &archs {
            if !stages_for(a, Some(&pr)).contains(&s) {
                cells.push("—".to_string());
                continue;
            }
            // per-seed fraction (unweighted mean over seeds); with fewer than 5 seeds a bootstrap interval is not
            // meaningful, so the seed range is printed instead
            let seeds: BTreeSet<i64> = pr.iter().filter(|r| text(r, "arch") == a).map(|r| int(r, "seed")).collect();
            let mut fr = Vec::new();
            for sd in seeds {
                let sub: Vec<&&Row> =
                    pr.iter().filter(|r| text(r, "arch") == a && int(r, "seed") == sd && has(r, s)).collect();
                if !sub.is_empty() {
                    fr.push(sub.iter().filter(|r| flag(r, s)).count() as f64 / sub.len() as f64);
                }
            }
            if fr.is_empty() {
                cells.push("—".to_string());
                continue;
            }
            let (m, mut lo, mut hi) = boot_ci(&fr);
            if fr.len() < 5 {
                lo = crate::py::value::min_by(fr.iter().copied(), |x| *x).unwrap();
                hi = crate::py::value::max_by(fr.iter().copied(), |x| *x).unwrap();
            }
            cells.push(format!("{} <sub>[{}, {}]</sub>", fmt::f(m, ".3f"), fmt::f(lo, ".2f"), fmt::f(hi, ".2f")));
        }
        lines.push(format!("| {} | {} |", stage_label(s).unwrap(), cells.join(" | ")));
    }
    let n: Vec<String> = archs
        .iter()
        .map(|a| {
            let k: BTreeSet<(i64, i64)> =
                pr.iter().filter(|r| text(r, "arch") == a).map(|r| (int(r, "seed"), int(r, "pid"))).collect();
            format!("{a}: {}", k.len())
        })
        .collect();
    lines.push(String::new());
    lines.push(format!("patterns counted: {}", n.join(", ")));
    lines.join("\n")
}

/// Where patterns die FIRST, all / rare / common.
pub fn first_loss_table(rows: &[Row], scale: i64, arch: &str) -> String {
    let pr: Vec<&Row> = pattern_rows(rows, scale).into_iter().filter(|r| text(r, "arch") == arch).collect();
    if pr.is_empty() {
        return "_(no rows)_".to_string();
    }
    let mut stages = stages_for(arch, Some(&pr));
    stages.push("reported");
    let mut lines =
        vec!["| stage the pattern died at | all | rare | common |".to_string(), "|---|---:|---:|---:|".into()];
    for s in stages {
        let key = if s == "reported" { "" } else { s };
        let allc = pr.iter().filter(|r| text(r, "loss_stage") == key).count();
        let rc = pr.iter().filter(|r| text(r, "loss_stage") == key && flag(r, "rare")).count();
        let cc = allc - rc;
        let lab = loss_label(s).or_else(|| stage_label(s)).unwrap_or("**reported**");
        lines.push(format!("| {lab} | {allc} ({}) | {rc} | {cc} |", fmt::f(allc as f64 / pr.len() as f64, ".0%")));
    }
    let rare = pr.iter().filter(|r| flag(r, "rare")).count();
    lines.push(format!("| total | {} | {rare} | {} |", pr.len(), pr.len() - rare));
    lines.join("\n")
}

/// Why the sketch could not see the patterns it missed.
pub fn sketch_fail_table(rows: &[Row], scale: i64, arch: &str) -> String {
    let pr: Vec<&Row> = pattern_rows(rows, scale)
        .into_iter()
        .filter(|r| text(r, "arch") == arch && r.contains_key("sketch_fail") && !flag(r, "sketch_visible"))
        .collect();
    if pr.is_empty() {
        return "_(no sketch-invisible patterns)_".to_string();
    }
    let mut lines = vec![
        "| why the sketch could not see it | all | rare | common | median witnesses |".to_string(),
        "|---|---:|---:|---:|---:|".into(),
    ];
    for (k, lab) in SKETCH_FAIL_LABEL {
        let sub: Vec<&&Row> = pr.iter().filter(|r| text(r, "sketch_fail") == k).collect();
        if sub.is_empty() {
            continue;
        }
        let rare = sub.iter().filter(|r| flag(r, "rare")).count();
        let witnesses: Vec<i64> = sub.iter().map(|r| int(r, "n_witnesses")).collect();
        lines.push(format!(
            "| {lab} | {} | {rare} | {} | {} |",
            sub.len(),
            sub.len() - rare,
            math::median_int(&witnesses) as i64
        ));
    }
    lines.join("\n")
}

/// Discovery as a function of how many people witnessed the pattern.
pub fn witness_curve(rows: &[Row], scale: i64, arch: &str) -> String {
    let pr: Vec<&Row> = pattern_rows(rows, scale).into_iter().filter(|r| text(r, "arch") == arch).collect();
    if pr.is_empty() {
        return String::new();
    }
    let bins: [(i64, i64); 6] = [(0, 4), (5, 7), (8, 11), (12, 16), (17, 25), (26, 1_000_000)];
    let mut lines = vec![
        "| witnesses (records) | n | visible to sketch | in pool | reported |".to_string(),
        "|---|---:|---:|---:|---:|".into(),
    ];
    for (lo, hi) in bins {
        let sub: Vec<&&Row> = pr.iter().filter(|r| (lo..=hi).contains(&int(r, "n_witnesses"))).collect();
        if sub.is_empty() {
            continue;
        }
        let n = sub.len() as f64;
        let frac = |k: &str| sub.iter().filter(|r| flag(r, k)).count() as f64 / n;
        let sv =
            if sub[0].contains_key("sketch_visible") { fmt::f(frac("sketch_visible"), ".2f") } else { "—".into() };
        let top = if hi > 100_000 { "∞".to_string() } else { hi.to_string() };
        lines.push(format!(
            "| {lo}–{top} | {} | {sv} | {} | {} |",
            sub.len(),
            fmt::f(frac("in_pool"), ".2f"),
            fmt::f(frac("in_register"), ".2f")
        ));
    }
    lines.join("\n")
}

/// Of the patterns that ARE reported, where in the register do they sit?
pub fn rank_table(rows: &[Row], scale: i64) -> String {
    let pr: Vec<&Row> = pattern_rows(rows, scale).into_iter().filter(|r| flag(r, "in_register")).collect();
    let archs = sorted_archs(&pr);
    let mut lines = vec![
        "| architecture | reported | median rank (0-based, confidence-sorted register) | within top 40 | within top \
         100 | median confidence |"
            .to_string(),
        "|---|---:|---:|---:|---:|---:|".into(),
    ];
    for a in &archs {
        let sub: Vec<&&Row> = pr.iter().filter(|r| text(r, "arch") == a).collect();
        let rk: Vec<i64> = sub.iter().map(|r| int(r, "rank")).filter(|&x| x >= 0).collect();
        let med = if rk.is_empty() { "—".to_string() } else { (math::median_int(&rk) as i64).to_string() };
        let conf: Vec<f64> = sub.iter().map(|r| num(&r["match_conf"]).expect("match_conf")).collect();
        lines.push(format!(
            "| {a} | {} | {med} | {} | {} | {} |",
            sub.len(),
            rk.iter().filter(|&&x| x < 40).count(),
            rk.iter().filter(|&&x| x < 100).count(),
            fmt::f(math::median(&conf), ".2f")
        ));
    }
    lines.join("\n")
}

/// Decompose the discovery gap between two systems into the stages that account for it, on the SAME patterns
/// (paired by seed and pid).
pub fn gap_decomposition(rows: &[Row], scale: i64, a: &str, b: &str) -> String {
    // dict comprehensions keyed (seed, pid): a later row replaces an earlier one; iteration is over sorted keys
    let side = |arch: &str| -> BTreeMap<(i64, i64), &Row> {
        pattern_rows(rows, scale)
            .into_iter()
            .filter(|r| text(r, "arch") == arch)
            .map(|r| ((int(r, "seed"), int(r, "pid")), r))
            .collect()
    };
    let pa = side(a);
    let pb = side(b);
    let keys: Vec<(i64, i64)> = pa.keys().filter(|k| pb.contains_key(k)).copied().collect();
    if keys.is_empty() {
        return "_(no paired rows)_".to_string();
    }
    let n = keys.len();
    let a_rep = keys.iter().filter(|k| flag(pa[k], "in_register")).count();
    let b_rep = keys.iter().filter(|k| flag(pb[k], "in_register")).count();
    // patterns B reports and A does not: where did A lose them?
    let lost: Vec<&Row> =
        keys.iter().filter(|k| flag(pb[k], "in_register") && !flag(pa[k], "in_register")).map(|k| pa[k]).collect();
    let won = keys.iter().filter(|k| flag(pa[k], "in_register") && !flag(pb[k], "in_register")).count();
    let mut st: Vec<(String, usize)> = Vec::new();
    for r in &lost {
        let s = text(r, "loss_stage");
        match st.iter_mut().find(|e| e.0 == s) {
            Some(e) => e.1 += 1,
            None => st.push((s.to_string(), 1)),
        }
    }
    let mut lines = vec![
        format!(
            "Paired on {n} (seed, pattern) pairs at {} users: `{b}` reports {b_rep} ({}), `{a}` reports {a_rep} ({}). \
             `{b}` finds {} that `{a}` misses; `{a}` finds {won} that `{b}` misses.",
            fmt::i(scale, ","),
            fmt::f(b_rep as f64 / n as f64, ".1%"),
            fmt::f(a_rep as f64 / n as f64, ".1%"),
            lost.len()
        ),
        String::new(),
        format!(
            "**Of the {} patterns `{b}` reports and `{a}` does not, the first stage `{a}` lost them at:**",
            lost.len()
        ),
        String::new(),
        "| stage it died at | patterns | share of the gap | of which rare |".to_string(),
        "|---|---:|---:|---:|".to_string(),
    ];
    let paired: Vec<&Row> = keys.iter().map(|k| pa[k]).collect();
    for s in stages_for(a, Some(&paired)) {
        let c = st.iter().find(|e| e.0 == s).map_or(0, |e| e.1);
        if c == 0 {
            continue;
        }
        let rc = lost.iter().filter(|r| text(r, "loss_stage") == s && flag(r, "rare")).count();
        let lab = loss_label(s).or_else(|| stage_label(s)).unwrap();
        lines.push(format!("| {lab} | {c} | {} | {rc} |", fmt::f(c as f64 / lost.len().max(1) as f64, ".0%")));
    }
    lines.join("\n")
}

/// Every table, for each scale that has rows.
pub fn build(scales: &[i64]) -> String {
    let rows = load_default();
    if rows.is_empty() {
        return "_(loss_funnel.jsonl not found — run loss_accounting first)_".to_string();
    }
    let mut out: Vec<String> = Vec::new();
    for &sc in scales {
        if !rows.iter().any(|r| int(r, "scale") == sc) {
            continue;
        }
        let n_seeds =
            rows.iter().filter(|r| int(r, "scale") == sc).map(|r| int(r, "seed")).collect::<BTreeSet<_>>().len();
        out.push(format!("## {} users ({n_seeds} seeds)\n", fmt::i(sc, ",")));
        out.push("### Stage survival, all patterns\n".into());
        out.push(survival_table(&rows, sc, "all"));
        out.push("\n### Stage survival, rare-signal patterns only\n".into());
        out.push(survival_table(&rows, sc, "rare"));
        out.push("\n### Stage survival, common patterns only\n".into());
        out.push(survival_table(&rows, sc, "common"));
        out.push("\n### Where the hierarchy loses each pattern (terminal stage)\n".into());
        out.push(first_loss_table(&rows, sc, "H_mycelic_full"));
        out.push("\n### Why the sketch could not see the invisible ones\n".into());
        out.push(sketch_fail_table(&rows, sc, "H_mycelic_full"));
        out.push("\n### Discovery by number of witnesses (hierarchy)\n".into());
        out.push(witness_curve(&rows, sc, "H_mycelic_full"));
        out.push("\n### Discovery by number of witnesses (A2_chunked_ctx)\n".into());
        out.push(witness_curve(&rows, sc, "A2_chunked_ctx"));
        out.push("\n### Where the reported patterns sit in the register\n".into());
        out.push(rank_table(&rows, sc));
        out.push("\n### Decomposing the gap to the strongest centralised system\n".into());
        out.push(gap_decomposition(&rows, sc, "H_mycelic_full", "A2_chunked_ctx"));
        out.push("\n### Decomposing the gap to the centralised twin (same algorithm)\n".into());
        out.push(gap_decomposition(&rows, sc, "H_mycelic_full", "B4_central_triage"));
        out.push("\n### Decomposing the gap to the perfect-retrieval oracle\n".into());
        out.push(gap_decomposition(&rows, sc, "H_mycelic_full", "Y_oracle_retrieval"));
        out.push(String::new());
    }
    out.join("\n")
}

/// The scales `build()` covers by default.
pub const DEFAULT_SCALES: [i64; 2] = [10_000, 50_000];

/// `python -m research.mycelic.loss_report`: print the markdown.
pub fn main(_args: &[String]) -> anyhow::Result<()> {
    println!("{}", build(&DEFAULT_SCALES));
    Ok(())
}
