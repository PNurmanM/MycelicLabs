//! Figures, generated from the raw rows (`plots.py`).
//!
//! Deliberately plain: no styling that could make a difference look bigger than it is, error bars everywhere there
//! are seeds to compute them from, and log axes labelled as such.  Each figure takes the same rows, series,
//! titles, axis labels, legends, sizes (inches) and dpi as its matplotlib original; the drawing is
//! [`crate::plotting`], so the PNGs are equivalent rather than pixel-identical.

use crate::analysis::{agg, boot_ci, dedupe_default, load_path};
use crate::np::math;
use crate::plotting::{Artist, Axes, Color, Dash, Figure, Fonts, Legend, Loc};
use crate::py::value::{Row, num, py_cmp, py_str};
use crate::runner::art;
use anyhow::Result;
use serde_json::Value;
use std::path::{Path, PathBuf};

pub const ORDER: [&str; 18] = [
    "A_flat_rag",
    "A2_chunked_ctx",
    "B_long_context",
    "B2_map_reduce",
    "B4_central_triage",
    "C_recursive_sum",
    "D_hier_nolineage",
    "E_hier_lineage",
    "F_hier_retrieval",
    "G_hier_questions",
    "H_mycelic_full",
    "I_mycelic_completion",
    "J_mycelic_verified",
    "Y_oracle_retrieval",
    "Z_random_rank",
    "Z2_naive_enumerate",
    "H_mycelic_prev",
    "H_mycelic_lean",
];

const DPI: f64 = 150.0;

/// What every figure reads and writes: the rows' directory, the figures' directory and the fonts.
pub struct Ctx<'f> {
    pub art: PathBuf,
    pub out: PathBuf,
    pub fonts: &'f Fonts,
}

impl Ctx<'_> {
    fn load(&self, name: &str) -> Vec<Row> {
        load_path(&self.art.join(name))
    }

    fn save(&self, fig: &Figure, name: &str) -> Result<Option<PathBuf>> {
        let p = self.out.join(name);
        fig.save_png(self.fonts, &p, DPI)?;
        Ok(Some(p))
    }
}

/// `_color(a)`: the architecture's tab20 slot, gray for anything outside `ORDER`.
fn color(arch: &str) -> Color {
    ORDER.iter().position(|a| *a == arch).map_or(Color::GRAY, Color::tab20)
}

/// `_style(ax, title, xlabel, ylabel)`
fn style(ax: &mut Axes, title: &str, xlabel: &str, ylabel: &str) {
    ax.title = title.to_string();
    ax.title_size = 11.0;
    ax.xlabel = xlabel.to_string();
    ax.ylabel = ylabel.to_string();
    ax.label_size = 9.0;
    ax.grid = true;
    ax.xtick_size = 8.0;
    ax.ytick_size = 8.0;
}

fn f(r: &Row, key: &str) -> f64 {
    r.get(key).and_then(num).unwrap_or_else(|| panic!("row has no numeric {key:?}"))
}

fn s(r: &Row, key: &str) -> String {
    r.get(key).map(py_str).unwrap_or_default()
}

/// `sorted({r[key] for r in rows})`
fn distinct(rows: &[&Row], key: &str) -> Vec<Value> {
    let mut out: Vec<Value> = Vec::new();
    for r in rows {
        let v = r.get(key).cloned().unwrap_or(Value::Null);
        if !out.iter().any(|x| crate::py::value::py_eq(x, &v)) {
            out.push(v);
        }
    }
    out.sort_by(py_cmp);
    out
}

fn same(r: &Row, key: &str, v: &Value) -> bool {
    r.get(key).is_some_and(|x| crate::py::value::py_eq(x, v))
}

/// `f"{n:,}"`
fn thousands(n: i64) -> String {
    let digits = n.unsigned_abs().to_string();
    let mut out = String::new();
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(c);
    }
    if n < 0 { format!("-{out}") } else { out }
}

/// (mean, minus, plus) of a bootstrap CI, the error-bar form plots.py passes as `yerr`.
fn ci_bar(v: &[f64]) -> (f64, f64, f64) {
    let (m, lo, hi) = boot_ci(v);
    (m, m - lo, hi - m)
}

fn errorbar(
    xs: Vec<f64>,
    cis: Vec<(f64, f64, f64)>,
    color: Color,
    ms: f64,
    lw: f64,
    capsize: f64,
    label: &str,
) -> Artist {
    Artist::ErrorBar {
        xs,
        ys: cis.iter().map(|c| c.0).collect(),
        lo: cis.iter().map(|c| c.1).collect(),
        hi: cis.iter().map(|c| c.2).collect(),
        color,
        lw,
        ms,
        capsize,
        label: Some(label.to_string()),
    }
}

fn legend(fontsize: f64) -> Option<Legend> {
    Some(Legend { fontsize, ncol: 1, loc: Loc::Best })
}

pub fn fig_scale(cx: &Ctx) -> Result<Option<PathBuf>> {
    let mut rows = cx.load("e1_baselines.jsonl");
    rows.extend(cx.load("e1b_extra.jsonl"));
    rows.extend(cx.load("e10_scale_trend.jsonl"));
    let rows = dedupe_default(rows);
    if rows.is_empty() {
        return Ok(None);
    }
    let all: Vec<&Row> = rows.iter().collect();
    let scales = distinct(&all, "scale");
    let mut fig = Figure::new(11.0, 4.2, 2);
    for (ax, (metric, lab)) in fig
        .axes
        .iter_mut()
        .zip([("found_anywhere_in_register", "discovery (found)"), ("evidence_coverage_2links", "evidence coverage")])
    {
        for a in ORDER {
            let (mut xs, mut cis) = (Vec::new(), Vec::new());
            for sc in &scales {
                let v: Vec<f64> =
                    rows.iter().filter(|r| s(r, "arch") == a && same(r, "scale", sc)).map(|r| f(r, metric)).collect();
                if v.is_empty() {
                    continue;
                }
                xs.push(num(sc).unwrap_or(f64::NAN));
                cis.push(ci_bar(&v));
            }
            if xs.is_empty() {
                continue;
            }
            ax.add(errorbar(xs, cis, color(a), 3.5, 1.2, 2.0, a));
        }
        ax.xlog = true;
        style(ax, &format!("{lab} vs enterprise size"), "users (log)", lab);
    }
    fig.axes[0].legend = Some(Legend { fontsize: 6.0, ncol: 2, loc: Loc::LowerLeft });
    cx.save(&fig, "scale.png")
}

pub fn fig_frontier(cx: &Ctx) -> Result<Option<PathBuf>> {
    let mut rows = cx.load("e1_baselines.jsonl");
    rows.extend(cx.load("e1b_extra.jsonl"));
    let rows = dedupe_default(rows);
    let fr = cx.load("e6_frontier.jsonl");
    if rows.is_empty() {
        return Ok(None);
    }
    let all: Vec<&Row> = rows.iter().collect();
    let scales = distinct(&all, "scale");
    let mut fig = Figure::new(4.2 * scales.len() as f64, 4.2, scales.len());
    for (ax, sc) in fig.axes.iter_mut().zip(&scales) {
        for a in ORDER {
            let v: Vec<(f64, f64)> = rows
                .iter()
                .filter(|r| s(r, "arch") == a && same(r, "scale", sc))
                .map(|r| (f(r, "compute_units"), f(r, "found_anywhere_in_register")))
                .collect();
            if v.is_empty() {
                continue;
            }
            let x = math::mean(&v.iter().map(|q| q.0).collect::<Vec<_>>());
            let y = math::mean(&v.iter().map(|q| q.1).collect::<Vec<_>>());
            ax.add(Artist::Scatter { xs: vec![x], ys: vec![y], area: 26.0, color: color(a) });
            let short = a.split('_').next().unwrap_or(a).to_string();
            ax.add(Artist::Annotate { x, y, text: short, size: 6.0, offset: (3.0, 3.0) });
        }
        let sub: Vec<Row> = fr
            .iter()
            .filter(|r| same(r, "scale", sc) && r.get("knob").and_then(Value::as_str) == Some("question_frac"))
            .cloned()
            .collect();
        if !sub.is_empty() {
            let mut a2 = agg(&sub, &["compute_units", "found_anywhere_in_register"], &["question_frac"]);
            a2.sort_by(|p, q| {
                f(p, "compute_units").partial_cmp(&f(q, "compute_units")).unwrap_or(std::cmp::Ordering::Equal)
            });
            ax.add(Artist::Line {
                xs: a2.iter().map(|r| f(r, "compute_units")).collect(),
                ys: a2.iter().map(|r| f(r, "found_anywhere_in_register")).collect(),
                color: Color::BLACK,
                lw: 1.0,
                dash: Dash::Solid,
                alpha: 0.6,
                label: Some("hierarchy question budget".into()),
            });
            ax.legend = legend(6.0);
        }
        ax.xlog = true;
        let users = thousands(num(sc).unwrap_or(0.0) as i64);
        style(ax, &format!("{users} users"), "compute units (log)", "discovery (found)");
    }
    cx.save(&fig, "cost_quality.png")
}

pub fn fig_level_marginal(cx: &Ctx) -> Result<Option<PathBuf>> {
    let rows = cx.load("e3b_level_marginal.jsonl");
    if rows.is_empty() {
        return Ok(None);
    }
    let levels = ["user", "team", "dept", "site", "region", "enterprise"];
    let mut fig = Figure::new(7.2, 4.2, 1);
    let ax = &mut fig.axes[0];
    let w = 0.38;
    let all: Vec<&Row> = rows.iter().collect();
    for (j, a) in distinct(&all, "arch").iter().enumerate() {
        let sub: Vec<&Row> = rows.iter().filter(|r| same(r, "arch", a)).collect();
        let found = |lv: &str| -> Vec<f64> {
            sub.iter().filter(|r| s(r, "upgraded_level") == lv).map(|r| f(r, "found_anywhere_in_register")).collect()
        };
        let base = found("none");
        let b = if base.is_empty() { 0.0 } else { math::mean(&base) };
        let cis: Vec<(f64, f64, f64)> = levels
            .iter()
            .map(|lv| {
                let v: Vec<f64> = found(lv).into_iter().map(|x| x - b).collect();
                if v.is_empty() { (0.0, 0.0, 0.0) } else { ci_bar(&v) }
            })
            .collect();
        ax.add(Artist::Bars {
            horizontal: false,
            pos: (0..levels.len()).map(|i| i as f64 + j as f64 * w).collect(),
            len: cis.iter().map(|c| c.0).collect(),
            thickness: w,
            colors: vec![Color::tab20(j * 4)],
            alpha: 1.0,
            err: Some((cis.iter().map(|c| c.1).collect(), cis.iter().map(|c| c.2).collect())),
            capsize: 2.0,
            label: Some(py_str(a)),
        });
    }
    ax.add(Artist::AxLine { horizontal: true, at: 0.0, color: Color::BLACK, lw: 0.8, dash: Dash::Solid, label: None });
    style(
        ax,
        "Marginal value of upgrading ONE level to a frontier model\n(baseline: every level small-7b)",
        "level upgraded",
        "\u{394} discovery vs baseline",
    );
    ax.xticks = Some(levels.iter().enumerate().map(|(i, l)| (i as f64 + w / 2.0, l.to_string())).collect());
    ax.legend = legend(7.0);
    cx.save(&fig, "level_marginal.png")
}

pub fn fig_qsweep(cx: &Ctx) -> Result<Option<PathBuf>> {
    let rows = cx.load("e3c_q_sweep.jsonl");
    if rows.is_empty() {
        return Ok(None);
    }
    let mut fig = Figure::new(6.4, 4.2, 1);
    let ax = &mut fig.axes[0];
    let all: Vec<&Row> = rows.iter().collect();
    for a in distinct(&all, "arch") {
        let sub: Vec<&Row> = rows.iter().filter(|r| same(r, "arch", &a)).collect();
        let qs = distinct(&sub, "q");
        let cis: Vec<(f64, f64, f64)> = qs
            .iter()
            .map(|q| {
                ci_bar(
                    &sub.iter()
                        .filter(|r| same(r, "q", q))
                        .map(|r| f(r, "found_anywhere_in_register"))
                        .collect::<Vec<_>>(),
                )
            })
            .collect();
        let name = py_str(&a);
        ax.add(errorbar(qs.iter().filter_map(num).collect(), cis, color(&name), 3.5, 1.2, 2.0, &name));
    }
    style(
        ax,
        "Operator quality \u{2192} architecture quality\n(every level at the same capability q)",
        "capability q (all levels)",
        "discovery (found)",
    );
    ax.legend = legend(7.0);
    cx.save(&fig, "q_sweep.png")
}

pub fn fig_discrimination(cx: &Ctx) -> Result<Option<PathBuf>> {
    let (pa, pb) = (cx.art.join("live_rank_results.json"), cx.art.join("live_rank_results_rich.json"));
    if !(pa.exists() && pb.exists()) {
        return Ok(None);
    }
    let d: Value = serde_json::from_str(&std::fs::read_to_string(&pa)?)?;
    let dr: Value = serde_json::from_str(&std::fs::read_to_string(&pb)?)?;
    // (mean, lower err, upper err) across repeats, NaN if absent
    let cell = |dd: &Value, m: &str| -> (f64, f64, f64) {
        if let Some(ag) = dd.get("models_aggregated").and_then(|x| x.get(m)).filter(|x| crate::py::value::truthy(x)) {
            let (mean, lo, hi) = (
                ag["ap_mean"].as_f64().unwrap_or(0.0),
                ag["ap_min"].as_f64().unwrap_or(0.0),
                ag["ap_max"].as_f64().unwrap_or(0.0),
            );
            return (mean, mean - lo, hi - mean);
        }
        match dd.get("models").and_then(|x| x.get(m)).filter(|x| crate::py::value::truthy(x)) {
            Some(v) => (v["ap"].as_f64().unwrap_or(0.0), 0.0, 0.0),
            None => (f64::NAN, 0.0, 0.0),
        }
    };
    let model_names = |dd: &Value| -> Vec<String> {
        dd.get("models_aggregated")
            .or_else(|| dd.get("models"))
            .and_then(Value::as_object)
            .map(|o| o.keys().cloned().collect())
            .unwrap_or_default()
    };
    let mut names: Vec<String> = model_names(&d);
    names.extend(model_names(&dr));
    names.sort();
    names.dedup();
    let stats: Vec<(f64, f64, f64)> = names.iter().map(|m| cell(&d, m)).collect();
    let rich: Vec<(f64, f64, f64)> = names.iter().map(|m| cell(&dr, m)).collect();
    let mut fig = Figure::new(7.4, 4.4, 1);
    let ax = &mut fig.axes[0];
    // The error bar is the FULL RANGE over independent repeats of the same task with the same model, not a CI:
    // with 2-3 runs a CI would be meaningless, and the range is the honest statement of the spread.
    for (cells, offset, label, slot) in
        [(&stats, -0.2, "evidence statistics only", 0), (&rich, 0.2, "statistics + raw work notes", 4)]
    {
        ax.add(Artist::Bars {
            horizontal: false,
            pos: (0..names.len()).map(|i| i as f64 + offset).collect(),
            len: cells.iter().map(|c| c.0).collect(),
            thickness: 0.4,
            colors: vec![Color::tab20(slot)],
            alpha: 1.0,
            err: Some((cells.iter().map(|c| c.1).collect(), cells.iter().map(|c| c.2).collect())),
            capsize: 3.0,
            label: Some(label.into()),
        });
    }
    let random_ap = d["random_ap"].as_f64().unwrap_or(f64::NAN);
    let logistic_ap = d["simulator_logistic_ap"].as_f64().unwrap_or(f64::NAN);
    ax.add(Artist::AxLine {
        horizontal: true,
        at: random_ap,
        color: Color::GRAY,
        lw: 1.2,
        dash: Dash::Dotted,
        label: Some(format!("random ({random_ap:.3})")),
    });
    ax.add(Artist::AxLine {
        horizontal: true,
        at: logistic_ap,
        color: Color::BLACK,
        lw: 1.2,
        dash: Dash::Dashed,
        label: Some(format!("simulator logistic ({logistic_ap:.3})")),
    });
    ax.xticks = Some(names.iter().enumerate().map(|(i, n)| (i as f64, n.clone())).collect());
    style(
        ax,
        &format!(
            "Directly measured: candidate discrimination\n({} candidates from a real run, {} genuine; bars = mean, \
             whiskers = full range over repeats)",
            py_str(&d["n_items"]),
            py_str(&d["n_real"])
        ),
        "model",
        "average precision",
    );
    ax.legend = legend(7.0);
    cx.save(&fig, "discrimination.png")
}

pub fn fig_ablation(cx: &Ctx) -> Result<Option<PathBuf>> {
    let rows = cx.load("e2_ablations.jsonl");
    if rows.is_empty() {
        return Ok(None);
    }
    let a = agg(&rows, &["found_anywhere_in_register", "average_precision"], &["ablation"]);
    let Some(base) = a.iter().find(|r| s(r, "ablation") == "full") else { return Ok(None) };
    let b = f(base, "found_anywhere_in_register");
    let mut a: Vec<&Row> = a.iter().filter(|r| s(r, "ablation") != "full").collect();
    a.sort_by(|p, q| {
        (f(p, "found_anywhere_in_register") - b)
            .partial_cmp(&(f(q, "found_anywhere_in_register") - b))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut fig = Figure::new(7.6, (0.30 * a.len() as f64 + 1.4).max(3.4), 1);
    let ax = &mut fig.axes[0];
    let ys: Vec<f64> = a.iter().map(|r| f(r, "found_anywhere_in_register") - b).collect();
    let colors = ys.iter().map(|&y| Color::parse(if y < 0.0 { "#b2182b" } else { "#2166ac" })).collect();
    ax.add(Artist::Bars {
        horizontal: true,
        pos: (0..a.len()).map(|i| i as f64).collect(),
        len: ys,
        thickness: 0.8,
        colors,
        alpha: 1.0,
        err: None,
        capsize: 0.0,
        label: None,
    });
    ax.add(Artist::AxLine { horizontal: false, at: 0.0, color: Color::BLACK, lw: 0.8, dash: Dash::Solid, label: None });
    style(ax, "Ablations: change in discovery vs the full system", "\u{394} discovery (found)", "");
    ax.yticks = Some(a.iter().enumerate().map(|(i, r)| (i as f64, s(r, "ablation"))).collect());
    cx.save(&fig, "ablations.png")
}

pub fn fig_privacy(cx: &Ctx) -> Result<Option<PathBuf>> {
    let rows = cx.load("e9_privacy.jsonl");
    if rows.is_empty() {
        return Ok(None);
    }
    let all: Vec<&Row> = rows.iter().collect();
    let big = distinct(&all, "scale").pop().unwrap_or(Value::Null);
    let at_big: Vec<Row> = rows.iter().filter(|r| same(r, "scale", &big)).cloned().collect();
    let mut a = agg(
        &at_big,
        &["raw_text_exposure_fraction", "claim_exposure_fraction", "found_anywhere_in_register"],
        &["arch"],
    );
    a.sort_by(|p, q| {
        (-f(p, "found_anywhere_in_register"))
            .partial_cmp(&-f(q, "found_anywhere_in_register"))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut fig = Figure::new(7.2, 4.4, 1);
    let ax = &mut fig.axes[0];
    for r in &a {
        let x = f(r, "raw_text_exposure_fraction").max(1e-4);
        let y = f(r, "found_anywhere_in_register");
        let arch = s(r, "arch");
        ax.add(Artist::Scatter { xs: vec![x], ys: vec![y], area: 40.0, color: color(&arch) });
        let short = arch.split('_').next().unwrap_or(&arch).to_string();
        ax.add(Artist::Annotate { x, y, text: short, size: 6.5, offset: (4.0, 3.0) });
    }
    ax.xlog = true;
    let users = thousands(num(&big).unwrap_or(0.0) as i64);
    style(
        ax,
        &format!(
            "Discovery vs confidentiality cost at {users} users\n(x: fraction of records whose ORIGINAL TEXT left its \
             owner; 1e-4 = none)"
        ),
        "raw-text exposure (log)",
        "discovery (found)",
    );
    cx.save(&fig, "privacy.png")
}

pub fn fig_adversarial(cx: &Ctx) -> Result<Option<PathBuf>> {
    let rows = cx.load("e5_adversarial.jsonl");
    if rows.is_empty() {
        return Ok(None);
    }
    let all: Vec<&Row> = rows.iter().collect();
    let archs: Vec<String> = distinct(&all, "arch").iter().map(py_str).collect();
    let a = agg(
        &rows,
        &["found_anywhere_in_register", "false_discovery_rate", "decoy_acceptance_all"],
        &["condition", "arch"],
    );
    let base = |arch: &str| a.iter().rev().find(|r| s(r, "condition") == "clean" && s(r, "arch") == arch);
    let agg_refs: Vec<&Row> = a.iter().collect();
    let conds: Vec<String> = distinct(&agg_refs, "condition").iter().map(py_str).filter(|c| c != "clean").collect();
    if conds.is_empty() {
        return Ok(None);
    }
    let mut fig = Figure::new(13.0, (0.32 * conds.len() as f64 + 1.8).max(3.6), 2);
    let h = 0.8 / archs.len().max(1) as f64;
    for (j, arch) in archs.iter().enumerate() {
        let (mut d_found, mut d_fdr) = (Vec::new(), Vec::new());
        for c in &conds {
            let row = a.iter().find(|r| s(r, "condition") == *c && s(r, "arch") == *arch);
            match (row, base(arch)) {
                (Some(row), Some(b)) => {
                    d_found.push(f(row, "found_anywhere_in_register") - f(b, "found_anywhere_in_register"));
                    d_fdr.push(f(row, "false_discovery_rate") - f(b, "false_discovery_rate"));
                }
                _ => {
                    d_found.push(0.0);
                    d_fdr.push(0.0);
                }
            }
        }
        for (ax, values) in fig.axes.iter_mut().zip([d_found, d_fdr]) {
            ax.add(Artist::Bars {
                horizontal: true,
                pos: (0..conds.len()).map(|i| i as f64 + j as f64 * h).collect(),
                len: values,
                thickness: h,
                colors: vec![color(arch)],
                alpha: 1.0,
                err: None,
                capsize: 0.0,
                label: Some(arch.clone()),
            });
        }
    }
    for (ax, lab) in fig.axes.iter_mut().zip([
        "\u{394} discovery (found) vs that architecture's own clean baseline",
        "\u{394} false-discovery rate vs own clean baseline",
    ]) {
        ax.add(Artist::AxLine {
            horizontal: false,
            at: 0.0,
            color: Color::BLACK,
            lw: 0.8,
            dash: Dash::Solid,
            label: None,
        });
        style(ax, lab, "\u{394}", "");
        ax.yticks = Some(conds.iter().enumerate().map(|(i, c)| (i as f64 + 0.4 - h / 2.0, c.clone())).collect());
        ax.ytick_size = 7.5;
    }
    fig.axes[0].legend = Some(Legend { fontsize: 6.5, ncol: 1, loc: Loc::LowerLeft });
    cx.save(&fig, "adversarial.png")
}

pub fn fig_fanin(cx: &Ctx) -> Result<Option<PathBuf>> {
    let rows = cx.load("e4_fanin.jsonl");
    let rows_d = cx.load("e4b_dept_fanin.jsonl");
    if rows.is_empty() && rows_d.is_empty() {
        return Ok(None);
    }
    let mut fig = Figure::new(11.0, 4.2, 2);
    for (ax, (rs, xkey, xlab)) in fig
        .axes
        .iter_mut()
        .zip([(&rows, "team_size", "users per team"), (&rows_d, "teams_per_dept", "teams per department")])
    {
        if rs.is_empty() {
            ax.add(Artist::AxesText { fx: 0.5, fy: 0.5, text: "(not run)".into(), size: 10.0 });
            style(ax, "", xlab, "discovery (found)");
            continue;
        }
        let all: Vec<&Row> = rs.iter().collect();
        for arch in distinct(&all, "arch") {
            let sub: Vec<&Row> = rs.iter().filter(|r| same(r, "arch", &arch)).collect();
            let xs = distinct(&sub, xkey);
            let cis = xs
                .iter()
                .map(|x| {
                    ci_bar(
                        &sub.iter()
                            .filter(|r| same(r, xkey, x))
                            .map(|r| f(r, "found_anywhere_in_register"))
                            .collect::<Vec<_>>(),
                    )
                })
                .collect();
            let name = py_str(&arch);
            ax.add(errorbar(xs.iter().filter_map(num).collect(), cis, color(&name), 4.0, 1.3, 3.0, &name));
        }
        ax.ylim = (Some(0.0), None);
        style(ax, "Fan-in vs strategic discovery", xlab, "discovery (found)");
        ax.legend = legend(7.0);
    }
    cx.save(&fig, "fanin.png")
}

/// OLD (`artifacts/v1`) vs NEW (`artifacts`) for the hierarchy and A2, at every scale both hold: found, evidence
/// coverage, rare recall, and compute.
pub fn fig_old_new(cx: &Ctx) -> Result<Option<PathBuf>> {
    let (old, new) = (crate::vnext_data::e1_rows(&cx.art.join("v1")), crate::vnext_data::e1_rows(&cx.art));
    if old.is_empty() || new.is_empty() {
        return Ok(None);
    }
    let (old_refs, new_refs): (Vec<&Row>, Vec<&Row>) = (old.iter().collect(), new.iter().collect());
    let old_scales = distinct(&old_refs, "scale");
    let scales: Vec<Value> = distinct(&new_refs, "scale")
        .into_iter()
        .filter(|s| old_scales.iter().any(|o| crate::py::value::py_eq(o, s)))
        .collect();
    if scales.is_empty() {
        return Ok(None);
    }
    let metrics = [
        ("found_anywhere_in_register", "found"),
        ("evidence_coverage_2links", "evidence coverage"),
        ("rare_signal_recall", "rare recall"),
        ("compute_units", "compute units"),
    ];
    let mut fig = Figure::new(3.6 * metrics.len() as f64, 3.8, metrics.len());
    let systems = [("H_mycelic_full", "Mycelic"), ("A2_chunked_ctx", "A2 chunked ctx")];
    let width = 0.18;
    for (ax, (key, lab)) in fig.axes.iter_mut().zip(metrics) {
        for (j, (arch, name)) in systems.iter().enumerate() {
            for (k, (rows, tag, alpha)) in [(&old, "old", 0.45), (&new, "new", 1.0)].into_iter().enumerate() {
                let cis: Vec<(f64, f64, f64)> = scales
                    .iter()
                    .map(|sc| {
                        let v: Vec<f64> = crate::vnext_data::by_seed(rows, arch, sc.as_i64().expect("scale"))
                            .values()
                            .map(|r| f(r, key))
                            .collect();
                        if v.is_empty() { (f64::NAN, 0.0, 0.0) } else { ci_bar(&v) }
                    })
                    .collect();
                let off = (j as f64 * 2.0 + k as f64 - 1.5) * width;
                ax.add(Artist::Bars {
                    horizontal: false,
                    pos: (0..scales.len()).map(|i| i as f64 + off).collect(),
                    len: cis.iter().map(|c| c.0).collect(),
                    thickness: width,
                    colors: vec![color(arch)],
                    alpha,
                    err: Some((cis.iter().map(|c| c.1).collect(), cis.iter().map(|c| c.2).collect())),
                    capsize: 2.0,
                    label: Some(format!("{name} {tag}")),
                });
            }
        }
        ax.ylog = key == "compute_units";
        style(ax, lab, "users", lab);
        ax.xticks = Some(
            scales.iter().enumerate().map(|(i, sc)| (i as f64, thousands(num(sc).unwrap_or(0.0) as i64))).collect(),
        );
    }
    fig.axes[0].legend = legend(6.5);
    cx.save(&fig, "old_new.png")
}

type FigFn = fn(&Ctx) -> Result<Option<PathBuf>>;

/// `ALL`, in order.
pub const ALL: [(&str, FigFn); 10] = [
    ("fig_scale", fig_scale),
    ("fig_frontier", fig_frontier),
    ("fig_level_marginal", fig_level_marginal),
    ("fig_qsweep", fig_qsweep),
    ("fig_discrimination", fig_discrimination),
    ("fig_ablation", fig_ablation),
    ("fig_privacy", fig_privacy),
    ("fig_adversarial", fig_adversarial),
    ("fig_fanin", fig_fanin),
    ("fig_old_new", fig_old_new),
];

/// The figures directory: `MYCELIC_FIGURES`, else `<artifacts>/figures` (`FIG`).
pub fn figures_dir() -> PathBuf {
    std::env::var_os("MYCELIC_FIGURES").map(PathBuf::from).unwrap_or_else(|| art().join("figures"))
}

/// `build()`: every figure from the rows in `art_dir` into `out_dir`; a figure that fails is reported and
/// skipped, as in the Python.
pub fn build(art_dir: &Path, out_dir: &Path, fonts: &Fonts) -> Result<Vec<PathBuf>> {
    std::fs::create_dir_all(out_dir)?;
    let cx = Ctx { art: art_dir.to_path_buf(), out: out_dir.to_path_buf(), fonts };
    let cwd = std::env::current_dir().unwrap_or_default();
    let mut out = Vec::new();
    for (name, fig) in ALL {
        // a panic on a malformed row is that figure's failure, as an exception was in the Python
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| fig(&cx))) {
            Ok(Ok(Some(p))) => {
                println!("  {name}: {}", p.strip_prefix(&cwd).unwrap_or(&p).display());
                out.push(p);
            }
            Ok(Ok(None)) => println!("  {name}: (no data)"),
            Ok(Err(e)) => println!("  {name}: FAILED {e:#}"),
            Err(_) => println!("  {name}: FAILED (see the panic message above)"),
        }
    }
    Ok(out)
}

/// `python -m research.mycelic.plots [out_dir]`: the figures from the artifacts' rows into `out_dir`
/// (default `MYCELIC_FIGURES` or `<artifacts>/figures`).
pub fn main(args: &[String]) -> Result<()> {
    let out = args.first().map(PathBuf::from).unwrap_or_else(figures_dir);
    let fonts = Fonts::load()?;
    build(&art(), &out, &fonts)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn thousands_separators() {
        assert_eq!(thousands(2000), "2,000");
        assert_eq!(thousands(100_000), "100,000");
        assert_eq!(thousands(999), "999");
        assert_eq!(thousands(-1_234_567), "-1,234,567");
    }
}
