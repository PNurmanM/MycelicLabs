//! A small matplotlib look-alike for the report figures: a one-row grid of axes with error bars, bars, scatter
//! points, reference lines, annotations and legends, laid out like `tight_layout` and rendered to PNG.
//!
//! Why not a plotting crate: the figures are compared side by side with the matplotlib originals, so the details
//! that carry meaning there are reproduced deliberately - matplotlib's tick choice (`AutoLocator` steps
//! 1/2/2.5/5/10, decade ticks and `10^k` labels on log axes), its 5% margins with sticky bar baselines,
//! tab20 colours, `legend(loc="best")`, figure sizes in inches at a given dpi and font sizes in points.
//!
//! The figure is written as SVG in points (1/72 inch) and rasterised with resvg at `dpi / 72` pixels per point,
//! so a 11 x 4.2 inch figure at dpi 150 is 1650 x 630 pixels as in matplotlib.  resvg, its font database and
//! skrifa (text measurement for the layout) are pure Rust: nothing is needed from the system at build time.
//! Fonts are found at RUN time: `MYCELIC_FONT` (a .ttf/.otf/.ttc file), else the system fonts, preferring
//! DejaVu Sans (matplotlib's default face) and then common sans-serif faces; with none, rendering fails with
//! an error that says so.

use anyhow::{Context, Result, bail};
use resvg::tiny_skia;
use resvg::usvg::{self, fontdb};
use skrifa::MetadataProvider;
use skrifa::instance::{LocationRef, Size};
use std::fmt::Write as _;
use std::path::Path;
use std::sync::Arc;

// -------------------------------------------------------------------------------------------------------------
// Fonts
// -------------------------------------------------------------------------------------------------------------

/// Sans-serif faces to use, in order of preference: matplotlib's own face first.
const PREFERRED_FAMILIES: [&str; 8] =
    ["DejaVu Sans", "Bitstream Vera Sans", "Helvetica", "Arial", "Liberation Sans", "Noto Sans", "FreeSans", "Verdana"];

/// The font database the renderer uses and the face the layout measures text with.
pub struct Fonts {
    db: Arc<fontdb::Database>,
    family: String,
    data: Vec<u8>,
    index: u32,
}

impl Fonts {
    /// `MYCELIC_FONT`, else the preferred system faces, else any sans-serif face on the system.
    pub fn load() -> Result<Fonts> {
        let mut db = fontdb::Database::new();
        let mut chosen = None;
        if let Some(path) = std::env::var_os("MYCELIC_FONT") {
            let before: Vec<fontdb::ID> = db.faces().map(|f| f.id).collect();
            db.load_font_file(&path)
                .with_context(|| format!("MYCELIC_FONT={}: cannot load the font", Path::new(&path).display()))?;
            chosen = db.faces().map(|f| f.id).find(|id| !before.contains(id));
            if chosen.is_none() {
                bail!("MYCELIC_FONT={}: no usable face in the file", Path::new(&path).display());
            }
        }
        // the rest of the system's fonts serve as fallback for glyphs the chosen face lacks (Δ, →)
        db.load_system_fonts();
        let query = |family: fontdb::Family| {
            let families = [family];
            db.query(&fontdb::Query { families: &families, ..fontdb::Query::default() })
        };
        let chosen = chosen
            .or_else(|| PREFERRED_FAMILIES.iter().find_map(|name| query(fontdb::Family::Name(name))))
            .or_else(|| query(fontdb::Family::SansSerif))
            .or_else(|| db.faces().next().map(|f| f.id));
        let Some(id) = chosen else {
            bail!(
                "no font found for the figures: install a TrueType font (e.g. DejaVu Sans) or set MYCELIC_FONT \
                 to a .ttf/.otf file"
            );
        };
        let face = db.face(id).expect("face of a known id");
        let family = face.families.first().map(|f| f.0.clone()).unwrap_or_default();
        let (data, index) =
            db.with_face_data(id, |data, index| (data.to_vec(), index)).context("reading the font file")?;
        skrifa::FontRef::from_index(&data, index).map_err(|e| anyhow::anyhow!("{family}: unreadable font: {e}"))?;
        db.set_sans_serif_family(family.clone());
        Ok(Fonts { db: Arc::new(db), family, data, index })
    }

    /// The family name written into the SVG.
    pub fn family(&self) -> &str {
        &self.family
    }

    fn font(&self) -> skrifa::FontRef<'_> {
        skrifa::FontRef::from_index(&self.data, self.index).expect("checked in load")
    }

    /// Advance width of a single-line text at `size` points, in points.  Characters the face lacks (rendered from
    /// a fallback face) count as 0.6 em.
    pub fn width(&self, text: &str, size: f64) -> f64 {
        let font = self.font();
        let charmap = font.charmap();
        let metrics = font.glyph_metrics(Size::new(size as f32), LocationRef::default());
        text.chars().map(|c| charmap.map(c).and_then(|g| metrics.advance_width(g)).map_or(0.6 * size, f64::from)).sum()
    }

    /// (ascent, descent) at `size` points, both positive.
    pub fn extents(&self, size: f64) -> (f64, f64) {
        let m = self.font().metrics(Size::new(size as f32), LocationRef::default());
        (f64::from(m.ascent), -f64::from(m.descent))
    }
}

// -------------------------------------------------------------------------------------------------------------
// Figure model
// -------------------------------------------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Color(pub u8, pub u8, pub u8);

impl Color {
    pub const BLACK: Color = Color(0, 0, 0);
    pub const GRAY: Color = Color(0x80, 0x80, 0x80);

    pub fn hex(self) -> String {
        format!("#{:02x}{:02x}{:02x}", self.0, self.1, self.2)
    }

    /// `#rrggbb`
    pub fn parse(hex: &str) -> Color {
        let v = u32::from_str_radix(hex.trim_start_matches('#'), 16).unwrap_or(0);
        Color((v >> 16) as u8, (v >> 8) as u8, v as u8)
    }

    /// `plt.get_cmap("tab20")(i)` for an integer i (the colour list repeats every 20).
    pub fn tab20(i: usize) -> Color {
        const TAB20: [&str; 20] = [
            "#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c", "#98df8a", "#d62728", "#ff9896", "#9467bd",
            "#c5b0d5", "#8c564b", "#c49c94", "#e377c2", "#f7b6d2", "#7f7f7f", "#c7c7c7", "#bcbd22", "#dbdb8d",
            "#17becf", "#9edae5",
        ];
        Color::parse(TAB20[i % 20])
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Dash {
    /// `"-"`
    Solid,
    /// `"--"`
    Dashed,
    /// `":"`
    Dotted,
}

/// Where a legend goes (`loc=`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Loc {
    Best,
    UpperRight,
    UpperLeft,
    LowerLeft,
    LowerRight,
    Right,
    CenterLeft,
    CenterRight,
    LowerCenter,
    UpperCenter,
    Center,
}

/// `loc="best"` tries the fixed positions in matplotlib's code order and keeps the first least-covering one.
const BEST_ORDER: [Loc; 10] = [
    Loc::UpperRight,
    Loc::UpperLeft,
    Loc::LowerLeft,
    Loc::LowerRight,
    Loc::Right,
    Loc::CenterLeft,
    Loc::CenterRight,
    Loc::LowerCenter,
    Loc::UpperCenter,
    Loc::Center,
];

#[derive(Clone, Debug)]
pub struct Legend {
    pub fontsize: f64,
    pub ncol: usize,
    pub loc: Loc,
}

/// One drawn element of an axes, in data coordinates.
#[derive(Clone, Debug)]
pub enum Artist {
    /// `ax.plot` (optionally with circle markers of the given diameter)
    Line { xs: Vec<f64>, ys: Vec<f64>, color: Color, lw: f64, dash: Dash, alpha: f64, label: Option<String> },
    /// `ax.errorbar(xs, ys, yerr=[lo, hi], marker="o", ms, lw, capsize)`
    ErrorBar {
        xs: Vec<f64>,
        ys: Vec<f64>,
        lo: Vec<f64>,
        hi: Vec<f64>,
        color: Color,
        lw: f64,
        ms: f64,
        capsize: f64,
        label: Option<String>,
    },
    /// `ax.bar` (vertical) / `ax.barh` (horizontal): bars centred on `pos` with `thickness`, from 0 to `len`; NaN
    /// lengths are not drawn.  `err` = (minus, plus) error-bar extents, drawn in black.
    Bars {
        horizontal: bool,
        pos: Vec<f64>,
        len: Vec<f64>,
        thickness: f64,
        colors: Vec<Color>,
        alpha: f64,
        err: Option<(Vec<f64>, Vec<f64>)>,
        capsize: f64,
        label: Option<String>,
    },
    /// `ax.scatter([x], [y], s=area_pt2, color)`
    Scatter { xs: Vec<f64>, ys: Vec<f64>, area: f64, color: Color },
    /// `ax.annotate(text, (x, y), xytext=offset, textcoords="offset points")`
    Annotate { x: f64, y: f64, text: String, size: f64, offset: (f64, f64) },
    /// `ax.axhline(at)` / `ax.axvline(at)`
    AxLine { horizontal: bool, at: f64, color: Color, lw: f64, dash: Dash, label: Option<String> },
    /// `ax.text(fx, fy, text, ha="center", transform=ax.transAxes)`
    AxesText { fx: f64, fy: f64, text: String, size: f64 },
}

/// Fixed tick positions and labels (`set_xticks` + `set_xticklabels`).
pub type Ticks = Vec<(f64, String)>;

#[derive(Clone, Debug)]
pub struct Axes {
    /// may contain `\n`
    pub title: String,
    pub xlabel: String,
    pub ylabel: String,
    pub title_size: f64,
    pub label_size: f64,
    pub xtick_size: f64,
    pub ytick_size: f64,
    pub xlog: bool,
    pub ylog: bool,
    pub xlim: (Option<f64>, Option<f64>),
    pub ylim: (Option<f64>, Option<f64>),
    pub xticks: Option<Ticks>,
    pub yticks: Option<Ticks>,
    /// `ax.grid(alpha=0.25, linewidth=0.5)`
    pub grid: bool,
    pub artists: Vec<Artist>,
    pub legend: Option<Legend>,
}

impl Default for Axes {
    fn default() -> Axes {
        Axes {
            title: String::new(),
            xlabel: String::new(),
            ylabel: String::new(),
            title_size: 12.0,
            label_size: 10.0,
            xtick_size: 10.0,
            ytick_size: 10.0,
            xlog: false,
            ylog: false,
            xlim: (None, None),
            ylim: (None, None),
            xticks: None,
            yticks: None,
            grid: false,
            artists: Vec::new(),
            legend: None,
        }
    }
}

impl Axes {
    pub fn add(&mut self, a: Artist) {
        self.artists.push(a);
    }
}

/// A one-row figure (`plt.subplots(1, n, figsize=(w, h))`).
#[derive(Clone, Debug)]
pub struct Figure {
    pub width_in: f64,
    pub height_in: f64,
    pub axes: Vec<Axes>,
}

impl Figure {
    pub fn new(width_in: f64, height_in: f64, n_axes: usize) -> Figure {
        Figure { width_in, height_in, axes: vec![Axes::default(); n_axes] }
    }

    /// The figure as SVG, in points.
    pub fn to_svg(&self, fonts: &Fonts) -> String {
        render_svg(self, fonts)
    }

    /// `fig.savefig(path, dpi=dpi)`
    pub fn save_png(&self, fonts: &Fonts, path: &Path, dpi: f64) -> Result<()> {
        let svg = self.to_svg(fonts);
        let png = svg_to_png(&svg, fonts, dpi)?;
        std::fs::write(path, png).with_context(|| format!("writing {}", path.display()))
    }
}

/// Rasterise an SVG drawn in points at `dpi` onto white.
pub fn svg_to_png(svg: &str, fonts: &Fonts, dpi: f64) -> Result<Vec<u8>> {
    let opt = usvg::Options { fontdb: fonts.db.clone(), font_family: fonts.family.clone(), ..Default::default() };
    let tree = usvg::Tree::from_str(svg, &opt).context("parsing the generated SVG")?;
    let scale = (dpi / 72.0) as f32;
    let size = tree.size();
    let (w, h) = ((size.width() * scale).round() as u32, (size.height() * scale).round() as u32);
    let mut pixmap = tiny_skia::Pixmap::new(w, h).context("figure size")?;
    pixmap.fill(tiny_skia::Color::WHITE);
    resvg::render(&tree, tiny_skia::Transform::from_scale(scale, scale), &mut pixmap.as_mut());
    pixmap.encode_png().context("encoding the PNG")
}

// -------------------------------------------------------------------------------------------------------------
// Scales and ticks
// -------------------------------------------------------------------------------------------------------------

/// A tick label: plain text, or a power `base^exp` as matplotlib's mathtext writes log ticks.
#[derive(Clone, Debug, PartialEq)]
enum Label {
    Plain(String),
    Power { base: String, exp: String },
}

impl Label {
    fn width(&self, fonts: &Fonts, size: f64) -> f64 {
        match self {
            Label::Plain(s) => fonts.width(s, size),
            Label::Power { base, exp } => fonts.width(base, size) + fonts.width(exp, size * SUPERSCRIPT),
        }
    }
}

const SUPERSCRIPT: f64 = 0.7;

#[derive(Clone, Debug)]
struct AxisTicks {
    major: Vec<(f64, Label)>,
    minor: Vec<f64>,
}

#[derive(Clone, Copy, Debug)]
struct Scale {
    log: bool,
    lo: f64,
    hi: f64,
}

impl Scale {
    fn t(&self, v: f64) -> f64 {
        if self.log { v.log10() } else { v }
    }

    /// Fraction of the axis (0 at `lo`, 1 at `hi`).
    fn frac(&self, v: f64) -> f64 {
        (self.t(v) - self.t(self.lo)) / (self.t(self.hi) - self.t(self.lo))
    }

    fn contains(&self, v: f64) -> bool {
        let (a, b) = (self.lo.min(self.hi), self.lo.max(self.hi));
        let eps = (b - a).abs() * 1e-9;
        v >= a - eps && v <= b + eps
    }
}

/// `MaxNLocator(nbins, steps=[1, 2, 2.5, 5, 10])` as `AutoLocator` uses it: the smallest "nice" step not below
/// range / nbins, refined downwards until at least two ticks fall inside the view.
fn linear_ticks(lo: f64, hi: f64, nbins: usize) -> Vec<f64> {
    let (lo, hi) = (lo.min(hi), lo.max(hi));
    let span = hi - lo;
    if span <= 0.0 || !span.is_finite() {
        return vec![lo];
    }
    let raw = span / nbins.max(1) as f64;
    let scale = 10f64.powf(raw.log10().floor());
    let steps: Vec<f64> = [1.0, 2.0, 2.5, 5.0, 10.0].iter().map(|s| s * scale).collect();
    let first = steps.iter().position(|&s| s >= raw * (1.0 - 1e-10)).unwrap_or(steps.len() - 1);
    let mut best = Vec::new();
    for &step in steps[..=first].iter().rev() {
        let k0 = (lo / step - 1e-9).ceil() as i64;
        let k1 = (hi / step + 1e-9).floor() as i64;
        best = (k0..=k1).map(|k| if k == 0 { 0.0 } else { k as f64 * step }).collect();
        if best.len() >= 2 {
            break;
        }
    }
    best
}

/// `ScalarFormatter` without offset: the fewest decimals that represent every tick, shared by all ticks, with a
/// unicode minus.
fn format_linear(ticks: &[f64]) -> Vec<Label> {
    let decimals = (0..10)
        .find(|&d| {
            ticks.iter().all(|&t| {
                let s = format!("{t:.d$}");
                (s.parse::<f64>().unwrap_or(f64::NAN) - t).abs() <= 1e-9 * t.abs().max(1e-12)
            })
        })
        .unwrap_or(10);
    ticks
        .iter()
        .map(|&t| {
            let s = format!("{t:.decimals$}");
            let s = if s.trim_start_matches('-').chars().all(|c| c == '0' || c == '.') {
                s.trim_start_matches('-').to_string()
            } else {
                s
            };
            Label::Plain(s.replace('-', "\u{2212}"))
        })
        .collect()
}

/// `LogLocator` + `LogFormatterSciNotation`: decades labelled `10^k`; the 2..9 multiples as minor ticks, labelled
/// too when at most one decade tick is in view (`minor_thresholds=(1, 0.4)`): the multiples 2, 3, 4, 6 when the
/// view spans more than 0.4 decades, all of them otherwise.
fn log_ticks(lo: f64, hi: f64) -> AxisTicks {
    let (lo, hi) = (lo.min(hi), lo.max(hi));
    let (d0, d1) = (lo.log10().floor() as i32, hi.log10().ceil() as i32);
    let inside = |v: f64| v >= lo * (1.0 - 1e-9) && v <= hi * (1.0 + 1e-9);
    let numdec = hi.log10() - lo.log10();
    let decades_in_view = (d0..=d1).filter(|&d| inside(10f64.powi(d))).count();
    let label_minor: &[u32] = if decades_in_view > 1 {
        &[]
    } else if numdec > 0.4 {
        &[2, 3, 4, 6]
    } else {
        &[2, 3, 4, 5, 6, 7, 8, 9]
    };
    let mut major = Vec::new();
    let mut minor = Vec::new();
    for d in d0..=d1 {
        let base = 10f64.powi(d);
        if inside(base) {
            major.push((base, Label::Power { base: "10".into(), exp: d.to_string().replace('-', "\u{2212}") }));
        }
        for m in 2..10u32 {
            let v = base * f64::from(m);
            if !inside(v) {
                continue;
            }
            if label_minor.contains(&m) {
                major.push((
                    v,
                    Label::Power { base: format!("{m}\u{d7}10"), exp: d.to_string().replace('-', "\u{2212}") },
                ));
            } else {
                minor.push(v);
            }
        }
    }
    major.sort_by(|a, b| a.0.total_cmp(&b.0));
    AxisTicks { major, minor }
}

// -------------------------------------------------------------------------------------------------------------
// Data limits
// -------------------------------------------------------------------------------------------------------------

#[derive(Default)]
struct Extent {
    lo: f64,
    hi: f64,
    any: bool,
    /// values a margin may not cross (bar baselines)
    sticky: Vec<f64>,
}

impl Extent {
    fn new() -> Extent {
        Extent { lo: f64::INFINITY, hi: f64::NEG_INFINITY, any: false, sticky: Vec::new() }
    }

    fn add(&mut self, v: f64, log: bool) {
        if !v.is_finite() || (log && v <= 0.0) {
            return;
        }
        self.lo = self.lo.min(v);
        self.hi = self.hi.max(v);
        self.any = true;
    }
}

fn extents(ax: &Axes) -> (Extent, Extent) {
    let (mut ex, mut ey) = (Extent::new(), Extent::new());
    for a in &ax.artists {
        match a {
            Artist::Line { xs, ys, .. } | Artist::Scatter { xs, ys, .. } => {
                for (&x, &y) in xs.iter().zip(ys) {
                    if x.is_finite() && y.is_finite() {
                        ex.add(x, ax.xlog);
                        ey.add(y, ax.ylog);
                    }
                }
            }
            Artist::ErrorBar { xs, ys, lo, hi, .. } => {
                for i in 0..xs.len() {
                    if xs[i].is_finite() && ys[i].is_finite() {
                        ex.add(xs[i], ax.xlog);
                        ey.add(ys[i] - lo[i], ax.ylog);
                        ey.add(ys[i] + hi[i], ax.ylog);
                    }
                }
            }
            Artist::Bars { horizontal, pos, len, thickness, err, .. } => {
                let (ep, el, plog, llog) = if *horizontal {
                    (&mut ey, &mut ex, ax.ylog, ax.xlog)
                } else {
                    (&mut ex, &mut ey, ax.xlog, ax.ylog)
                };
                for (i, (&p, &l)) in pos.iter().zip(len).enumerate() {
                    if !l.is_finite() {
                        continue;
                    }
                    ep.add(p - thickness / 2.0, plog);
                    ep.add(p + thickness / 2.0, plog);
                    el.add(0.0, llog);
                    el.add(l, llog);
                    if let Some((minus, plus)) = err {
                        el.add(l - minus[i], llog);
                        el.add(l + plus[i], llog);
                    }
                }
                el.sticky.push(0.0);
            }
            Artist::AxLine { horizontal, at, .. } => {
                if *horizontal {
                    ey.add(*at, ax.ylog);
                } else {
                    ex.add(*at, ax.xlog);
                }
            }
            Artist::Annotate { .. } | Artist::AxesText { .. } => {}
        }
    }
    (ex, ey)
}

/// The view limits: data range plus 5% margins (in log space on a log axis) that stop at sticky edges; explicit
/// limits win.
fn view_limits(e: &Extent, log: bool, fixed: (Option<f64>, Option<f64>)) -> (f64, f64) {
    let (mut lo, mut hi) = if e.any {
        (e.lo, e.hi)
    } else if log {
        (1.0, 10.0)
    } else {
        (0.0, 1.0)
    };
    let t = |v: f64| if log { v.log10() } else { v };
    let inv = |v: f64| if log { 10f64.powf(v) } else { v };
    if t(hi) - t(lo) <= 0.0 {
        // matplotlib's nonsingular expansion
        let d = if log || lo == 0.0 { 1.0 } else { 0.05 * lo.abs() };
        lo = inv(t(lo) - d);
        hi = inv(t(hi) + d);
    }
    let (tl, th) = (t(lo), t(hi));
    let margin = 0.05 * (th - tl);
    let (mut mlo, mut mhi) = (inv(tl - margin), inv(th + margin));
    for &s in &e.sticky {
        if s <= lo && s > mlo {
            mlo = s;
        }
        if s >= hi && s < mhi {
            mhi = s;
        }
    }
    (fixed.0.unwrap_or(mlo), fixed.1.unwrap_or(mhi))
}

fn ticks_for(fixed: &Option<Ticks>, scale: Scale, axis_len: f64, label_size: f64, per_label: f64) -> AxisTicks {
    if let Some(t) = fixed {
        return AxisTicks { major: t.iter().map(|(v, s)| (*v, Label::Plain(s.clone()))).collect(), minor: vec![] };
    }
    if scale.log {
        return log_ticks(scale.lo, scale.hi);
    }
    // `get_tick_space`: one bin per 3 (x) or 2 (y) label heights, clipped to 1..9
    let nbins = ((axis_len / (label_size * per_label)).floor() as usize).clamp(1, 9);
    let values: Vec<f64> = linear_ticks(scale.lo, scale.hi, nbins).into_iter().filter(|&v| scale.contains(v)).collect();
    let labels = format_linear(&values);
    AxisTicks { major: values.into_iter().zip(labels).collect(), minor: vec![] }
}

// -------------------------------------------------------------------------------------------------------------
// Layout and drawing
// -------------------------------------------------------------------------------------------------------------

/// matplotlib's defaults, in points.
const TICK_LEN: f64 = 3.5;
const MINOR_TICK_LEN: f64 = 2.0;
const TICK_PAD: f64 = 3.5;
const LABEL_PAD: f64 = 4.0;
const TITLE_PAD: f64 = 6.0;
/// `tight_layout(pad=1.08)` in units of the 10 pt default font
const FIG_PAD: f64 = 10.8;
/// error bars on bars use `lines.linewidth`
const BAR_ERR_LW: f64 = 1.5;

struct Placed {
    x0: f64,
    y0: f64,
    w: f64,
    h: f64,
    xs: Scale,
    ys: Scale,
    xt: AxisTicks,
    yt: AxisTicks,
}

impl Placed {
    fn px(&self, x: f64) -> f64 {
        self.x0 + self.xs.frac(x) * self.w
    }

    fn py(&self, y: f64) -> f64 {
        self.y0 + self.h - self.ys.frac(y) * self.h
    }
}

fn line_height(fonts: &Fonts, size: f64) -> f64 {
    let (a, d) = fonts.extents(size);
    a + d
}

/// Space the decorations of an axes need around its frame: (left, right, top, bottom).
fn decorations(ax: &Axes, xt: &AxisTicks, yt: &AxisTicks, fonts: &Fonts) -> (f64, f64, f64, f64) {
    let ytick_w = yt.major.iter().map(|(_, l)| l.width(fonts, ax.ytick_size)).fold(0.0, f64::max);
    let mut left = TICK_LEN + TICK_PAD + ytick_w;
    if !ax.ylabel.is_empty() {
        left += LABEL_PAD + line_height(fonts, ax.label_size);
    }
    let mut bottom = TICK_LEN + TICK_PAD + line_height(fonts, ax.xtick_size) * if ax.xlog { 1.25 } else { 1.0 };
    if !ax.xlabel.is_empty() {
        bottom += LABEL_PAD + line_height(fonts, ax.label_size);
    }
    let n_title = if ax.title.is_empty() { 0 } else { ax.title.split('\n').count() };
    let top = if n_title > 0 { TITLE_PAD + n_title as f64 * line_height(fonts, ax.title_size) } else { 0.0 };
    // the last x tick label may stick out on the right
    let right = xt.major.last().map_or(0.0, |(_, l)| l.width(fonts, ax.xtick_size) / 2.0);
    (left, right, top, bottom)
}

fn place(fig: &Figure, fonts: &Fonts) -> Vec<Placed> {
    let (fw, fh) = (fig.width_in * 72.0, fig.height_in * 72.0);
    let n = fig.axes.len().max(1);
    let cell_w = (fw - 2.0 * FIG_PAD - (n - 1) as f64 * FIG_PAD) / n as f64;
    let mut out = Vec::new();
    for (i, ax) in fig.axes.iter().enumerate() {
        let (ex, ey) = extents(ax);
        let (xl, xh) = view_limits(&ex, ax.xlog, ax.xlim);
        let (yl, yh) = view_limits(&ey, ax.ylog, ax.ylim);
        let xs = Scale { log: ax.xlog, lo: xl, hi: xh };
        let ys = Scale { log: ax.ylog, lo: yl, hi: yh };
        // two passes: tick density depends on the axis length, which depends on the tick labels
        let (mut w, mut h) = (cell_w * 0.8, (fh - 2.0 * FIG_PAD) * 0.8);
        let mut placed = None;
        for _ in 0..2 {
            let xt = ticks_for(&ax.xticks, xs, w, ax.xtick_size, 3.0);
            let yt = ticks_for(&ax.yticks, ys, h, ax.ytick_size, 2.0);
            let (l, r, t, b) = decorations(ax, &xt, &yt, fonts);
            let x0 = FIG_PAD + i as f64 * (cell_w + FIG_PAD) + l;
            w = (cell_w - l - r).max(10.0);
            h = (fh - 2.0 * FIG_PAD - t - b).max(10.0);
            placed = Some(Placed { x0, y0: FIG_PAD + t, w, h, xs, ys, xt, yt });
        }
        out.push(placed.expect("two passes"));
    }
    out
}

fn esc(s: &str) -> String {
    s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;").replace('"', "&quot;")
}

fn dash_attr(dash: Dash, lw: f64) -> String {
    match dash {
        Dash::Solid => String::new(),
        Dash::Dashed => format!(" stroke-dasharray=\"{:.2} {:.2}\"", 3.7 * lw, 1.6 * lw),
        Dash::Dotted => format!(" stroke-dasharray=\"{:.2} {:.2}\"", 1.0 * lw, 1.65 * lw),
    }
}

struct Svg<'f> {
    out: String,
    fonts: &'f Fonts,
}

#[derive(Clone, Copy)]
enum Anchor {
    Start,
    Middle,
    End,
}

#[derive(Clone, Copy)]
enum VAlign {
    Top,
    Center,
    Baseline,
    Bottom,
}

impl Svg<'_> {
    fn line(&mut self, (x1, y1): (f64, f64), (x2, y2): (f64, f64), color: Color, lw: f64, extra: &str) {
        let _ = writeln!(
            self.out,
            "<line x1=\"{x1:.3}\" y1=\"{y1:.3}\" x2=\"{x2:.3}\" y2=\"{y2:.3}\" stroke=\"{}\" stroke-width=\"{lw:.3}\"{extra}/>",
            color.hex()
        );
    }

    fn polyline(&mut self, pts: &[(f64, f64)], color: Color, lw: f64, dash: Dash, alpha: f64) {
        if pts.len() < 2 {
            return;
        }
        let p: Vec<String> = pts.iter().map(|(x, y)| format!("{x:.3},{y:.3}")).collect();
        let _ = writeln!(
            self.out,
            "<polyline points=\"{}\" fill=\"none\" stroke=\"{}\" stroke-width=\"{lw:.3}\" stroke-opacity=\"{alpha}\" \
             stroke-linejoin=\"round\" stroke-linecap=\"{}\"{}/>",
            p.join(" "),
            color.hex(),
            if dash == Dash::Solid { "square" } else { "butt" },
            dash_attr(dash, lw)
        );
    }

    fn circle(&mut self, x: f64, y: f64, diameter: f64, color: Color) {
        let _ = writeln!(
            self.out,
            "<circle cx=\"{x:.3}\" cy=\"{y:.3}\" r=\"{:.3}\" fill=\"{c}\" stroke=\"{c}\" stroke-width=\"1\"/>",
            diameter / 2.0,
            c = color.hex()
        );
    }

    fn rect(&mut self, x: f64, y: f64, w: f64, h: f64, fill: Color, alpha: f64) {
        let (x, w) = if w < 0.0 { (x + w, -w) } else { (x, w) };
        let (y, h) = if h < 0.0 { (y + h, -h) } else { (y, h) };
        let _ = writeln!(
            self.out,
            "<rect x=\"{x:.3}\" y=\"{y:.3}\" width=\"{w:.3}\" height=\"{h:.3}\" fill=\"{}\" fill-opacity=\"{alpha}\"/>",
            fill.hex()
        );
    }

    /// Text with its anchor point at (x, y); `rotate` turns it 90 degrees counter-clockwise about that point.
    fn text(&mut self, (x, y): (f64, f64), label: &Label, size: f64, (anchor, valign): (Anchor, VAlign), rotate: bool) {
        let (asc, desc) = self.fonts.extents(size);
        let base = match valign {
            VAlign::Top => y + asc,
            VAlign::Center => y + (asc - desc) / 2.0,
            VAlign::Baseline => y,
            VAlign::Bottom => y - desc,
        };
        let width = label.width(self.fonts, size);
        let start = match anchor {
            Anchor::Start => x,
            Anchor::Middle => x - width / 2.0,
            Anchor::End => x - width,
        };
        let transform = if rotate { format!(" transform=\"rotate(-90 {x:.3} {y:.3})\"") } else { String::new() };
        let body = match label {
            Label::Plain(s) => esc(s),
            Label::Power { base, exp } => format!(
                "{}<tspan dy=\"{:.3}\" font-size=\"{:.3}\">{}</tspan>",
                esc(base),
                -0.45 * size,
                size * SUPERSCRIPT,
                esc(exp)
            ),
        };
        let _ = writeln!(
            self.out,
            "<text x=\"{start:.3}\" y=\"{base:.3}\" font-family=\"{}\" font-size=\"{size:.3}\" fill=\"#000000\"{transform}>{body}</text>",
            esc(&self.fonts.family)
        );
    }

    fn plain(&mut self, x: f64, y: f64, s: &str, size: f64, anchor: Anchor, valign: VAlign) {
        self.text((x, y), &Label::Plain(s.to_string()), size, (anchor, valign), false);
    }
}

/// The marker/line sample of one legend entry.
#[derive(Clone)]
enum Handle {
    Line { color: Color, lw: f64, dash: Dash, marker: Option<f64> },
    ErrorBar { color: Color, lw: f64, ms: f64 },
    Patch { color: Color, alpha: f64 },
}

/// Legend entries in matplotlib's order: labelled lines (incl. `axhline`) first, then the containers that
/// `bar` and `errorbar` create, each group in drawing order.
fn handles(ax: &Axes) -> Vec<(Handle, String)> {
    let mut lines = Vec::new();
    let mut containers = Vec::new();
    for a in &ax.artists {
        match a {
            Artist::Line { color, lw, dash, label: Some(l), .. }
            | Artist::AxLine { color, lw, dash, label: Some(l), .. } => {
                lines.push((Handle::Line { color: *color, lw: *lw, dash: *dash, marker: None }, l.clone()))
            }
            Artist::ErrorBar { color, lw, ms, label: Some(l), .. } => {
                containers.push((Handle::ErrorBar { color: *color, lw: *lw, ms: *ms }, l.clone()))
            }
            Artist::Bars { colors, alpha, label: Some(l), .. } => containers.push((
                Handle::Patch { color: colors.first().copied().unwrap_or(Color::GRAY), alpha: *alpha },
                l.clone(),
            )),
            _ => {}
        }
    }
    lines.extend(containers);
    lines
}

/// Points (in figure coordinates) and boxes an axes draws, for `loc="best"`.
fn occupancy(ax: &Axes, p: &Placed) -> (Vec<(f64, f64)>, Vec<(f64, f64, f64, f64)>) {
    let mut pts = Vec::new();
    let mut boxes = Vec::new();
    for a in &ax.artists {
        match a {
            Artist::Line { xs, ys, .. } | Artist::Scatter { xs, ys, .. } | Artist::ErrorBar { xs, ys, .. } => {
                for (&x, &y) in xs.iter().zip(ys) {
                    if x.is_finite() && y.is_finite() {
                        pts.push((p.px(x), p.py(y)));
                    }
                }
            }
            Artist::Bars { horizontal, pos, len, thickness, .. } => {
                for (&q, &l) in pos.iter().zip(len) {
                    if !l.is_finite() {
                        continue;
                    }
                    let (a0, a1, b0, b1) = if *horizontal {
                        (bar_base(p.xs, &|v| p.px(v)), p.px(l), p.py(q - thickness / 2.0), p.py(q + thickness / 2.0))
                    } else {
                        (p.px(q - thickness / 2.0), p.px(q + thickness / 2.0), bar_base(p.ys, &|v| p.py(v)), p.py(l))
                    };
                    boxes.push((a0.min(a1), b0.min(b1), a0.max(a1), b0.max(b1)));
                }
            }
            _ => {}
        }
    }
    (pts, boxes)
}

/// Where a bar starts: 0, or the axis edge on a log axis.
fn bar_base(scale: Scale, to_px: &dyn Fn(f64) -> f64) -> f64 {
    if scale.log { to_px(scale.lo.min(scale.hi)) } else { to_px(0.0) }
}

fn draw_legend(svg: &mut Svg, ax: &Axes, p: &Placed, legend: &Legend) {
    let entries = handles(ax);
    if entries.is_empty() {
        return;
    }
    let fs = legend.fontsize;
    let fonts = svg.fonts;
    let ncol = legend.ncol.max(1);
    let nrow = entries.len().div_ceil(ncol);
    let text_h = line_height(fonts, fs);
    let (handle_len, handle_pad, border_pad, row_gap, col_gap, axes_pad) =
        (2.0 * fs, 0.8 * fs, 0.4 * fs, 0.5 * fs, 2.0 * fs, 0.5 * fs);
    // column-major fill, as matplotlib's legend does
    let columns: Vec<&[(Handle, String)]> = entries.chunks(nrow).collect();
    let col_w: Vec<f64> = columns
        .iter()
        .map(|c| handle_len + handle_pad + c.iter().map(|(_, t)| fonts.width(t, fs)).fold(0.0, f64::max))
        .collect();
    let box_w = 2.0 * border_pad + col_w.iter().sum::<f64>() + col_gap * (columns.len() - 1) as f64;
    let box_h = 2.0 * border_pad + nrow as f64 * text_h + (nrow - 1) as f64 * row_gap;
    let origin = |loc: Loc| -> (f64, f64) {
        let left = p.x0 + axes_pad;
        let right = p.x0 + p.w - axes_pad - box_w;
        let hcenter = p.x0 + (p.w - box_w) / 2.0;
        let top = p.y0 + axes_pad;
        let bottom = p.y0 + p.h - axes_pad - box_h;
        let vcenter = p.y0 + (p.h - box_h) / 2.0;
        match loc {
            Loc::UpperRight | Loc::Best => (right, top),
            Loc::UpperLeft => (left, top),
            Loc::LowerLeft => (left, bottom),
            Loc::LowerRight => (right, bottom),
            Loc::Right | Loc::CenterRight => (right, vcenter),
            Loc::CenterLeft => (left, vcenter),
            Loc::LowerCenter => (hcenter, bottom),
            Loc::UpperCenter => (hcenter, top),
            Loc::Center => (hcenter, vcenter),
        }
    };
    let loc = if legend.loc == Loc::Best {
        let (pts, boxes) = occupancy(ax, p);
        let badness = |loc: Loc| {
            let (x, y) = origin(loc);
            let inside = pts.iter().filter(|(px, py)| *px >= x && *px <= x + box_w && *py >= y && *py <= y + box_h);
            let overlaps =
                boxes.iter().filter(|(a0, b0, a1, b1)| *a0 < x + box_w && *a1 > x && *b0 < y + box_h && *b1 > y);
            inside.count() + overlaps.count()
        };
        let mut best = (usize::MAX, Loc::UpperRight);
        for loc in BEST_ORDER {
            let b = badness(loc);
            if b < best.0 {
                best = (b, loc);
            }
        }
        best.1
    } else {
        legend.loc
    };
    let (bx, by) = origin(loc);
    let _ = writeln!(
        svg.out,
        "<rect x=\"{bx:.3}\" y=\"{by:.3}\" width=\"{box_w:.3}\" height=\"{box_h:.3}\" rx=\"{r:.3}\" fill=\"#ffffff\" \
         fill-opacity=\"0.8\" stroke=\"#cccccc\" stroke-opacity=\"0.8\" stroke-width=\"1\"/>",
        r = 0.2 * fs
    );
    let mut cx = bx + border_pad;
    for (c, column) in columns.iter().enumerate() {
        for (r, (handle, text)) in column.iter().enumerate() {
            let cy = by + border_pad + r as f64 * (text_h + row_gap) + text_h / 2.0;
            let (h0, h1) = (cx, cx + handle_len);
            match handle {
                Handle::Line { color, lw, dash, marker } => {
                    svg.polyline(&[(h0, cy), (h1, cy)], *color, *lw, *dash, 1.0);
                    if let Some(ms) = marker {
                        svg.circle((h0 + h1) / 2.0, cy, *ms, *color);
                    }
                }
                Handle::ErrorBar { color, lw, ms } => {
                    svg.polyline(&[(h0, cy), (h1, cy)], *color, *lw, Dash::Solid, 1.0);
                    let mid = (h0 + h1) / 2.0;
                    let half = 0.35 * fs;
                    svg.line((mid, cy - half), (mid, cy + half), *color, *lw, "");
                    svg.line((mid - 0.15 * fs, cy - half), (mid + 0.15 * fs, cy - half), *color, *lw, "");
                    svg.line((mid - 0.15 * fs, cy + half), (mid + 0.15 * fs, cy + half), *color, *lw, "");
                    svg.circle(mid, cy, *ms, *color);
                }
                Handle::Patch { color, alpha } => svg.rect(h0, cy - 0.35 * fs, handle_len, 0.7 * fs, *color, *alpha),
            }
            svg.plain(h1 + handle_pad, cy, text, fs, Anchor::Start, VAlign::Center);
        }
        cx += col_w[c] + col_gap;
    }
}

fn draw_axes(svg: &mut Svg, ax: &Axes, p: &Placed, clip_id: &str) {
    let (x0, y0, x1, y1) = (p.x0, p.y0, p.x0 + p.w, p.y0 + p.h);
    let _ = writeln!(
        svg.out,
        "<clipPath id=\"{clip_id}\"><rect x=\"{x0:.3}\" y=\"{y0:.3}\" width=\"{:.3}\" height=\"{:.3}\"/></clipPath>",
        p.w, p.h
    );
    let _ = writeln!(svg.out, "<g clip-path=\"url(#{clip_id})\">");
    // patches first, then the grid, then lines and points (matplotlib's `axisbelow="line"` z-order)
    for a in &ax.artists {
        if let Artist::Bars { horizontal, pos, len, thickness, colors, alpha, .. } = a {
            for (i, (&q, &l)) in pos.iter().zip(len).enumerate() {
                if !l.is_finite() {
                    continue;
                }
                let color = colors[i % colors.len()];
                if *horizontal {
                    let base = bar_base(p.xs, &|v| p.px(v));
                    let (ya, yb) = (p.py(q + thickness / 2.0), p.py(q - thickness / 2.0));
                    svg.rect(base, ya, p.px(l) - base, yb - ya, color, *alpha);
                } else {
                    let base = bar_base(p.ys, &|v| p.py(v));
                    let (xa, xb) = (p.px(q - thickness / 2.0), p.px(q + thickness / 2.0));
                    svg.rect(xa, p.py(l), xb - xa, base - p.py(l), color, *alpha);
                }
            }
        }
    }
    if ax.grid {
        let grid = " stroke-opacity=\"0.25\"";
        let gray = Color::parse("#b0b0b0");
        for (v, _) in &p.xt.major {
            let x = p.px(*v);
            svg.line((x, y0), (x, y1), gray, 0.5, grid);
        }
        for (v, _) in &p.yt.major {
            let y = p.py(*v);
            svg.line((x0, y), (x1, y), gray, 0.5, grid);
        }
    }
    for a in &ax.artists {
        match a {
            Artist::Bars { horizontal, pos, len, err: Some((minus, plus)), capsize, .. } => {
                for i in 0..pos.len() {
                    if !len[i].is_finite() {
                        continue;
                    }
                    let (lo, hi) = (len[i] - minus[i], len[i] + plus[i]);
                    if *horizontal {
                        let y = p.py(pos[i]);
                        svg.line((p.px(lo), y), (p.px(hi), y), Color::BLACK, BAR_ERR_LW, "");
                        for v in [lo, hi] {
                            svg.line((p.px(v), y - capsize), (p.px(v), y + capsize), Color::BLACK, BAR_ERR_LW, "");
                        }
                    } else {
                        let x = p.px(pos[i]);
                        svg.line((x, p.py(lo)), (x, p.py(hi)), Color::BLACK, BAR_ERR_LW, "");
                        for v in [lo, hi] {
                            svg.line((x - capsize, p.py(v)), (x + capsize, p.py(v)), Color::BLACK, BAR_ERR_LW, "");
                        }
                    }
                }
            }
            Artist::Line { xs, ys, color, lw, dash, alpha, .. } => {
                let pts: Vec<(f64, f64)> = xs.iter().zip(ys).map(|(&x, &y)| (p.px(x), p.py(y))).collect();
                svg.polyline(&pts, *color, *lw, *dash, *alpha);
            }
            Artist::AxLine { horizontal, at, color, lw, dash, .. } => {
                if *horizontal {
                    let y = p.py(*at);
                    svg.polyline(&[(x0, y), (x1, y)], *color, *lw, *dash, 1.0);
                } else {
                    let x = p.px(*at);
                    svg.polyline(&[(x, y0), (x, y1)], *color, *lw, *dash, 1.0);
                }
            }
            Artist::ErrorBar { xs, ys, lo, hi, color, lw, ms, capsize, .. } => {
                let pts: Vec<(f64, f64)> = xs.iter().zip(ys).map(|(&x, &y)| (p.px(x), p.py(y))).collect();
                svg.polyline(&pts, *color, *lw, Dash::Solid, 1.0);
                for i in 0..xs.len() {
                    let x = p.px(xs[i]);
                    let (ya, yb) = (p.py(ys[i] - lo[i]), p.py(ys[i] + hi[i]));
                    svg.line((x, ya), (x, yb), *color, *lw, "");
                    for y in [ya, yb] {
                        svg.line((x - capsize, y), (x + capsize, y), *color, *lw, "");
                    }
                    svg.circle(x, p.py(ys[i]), *ms, *color);
                }
            }
            Artist::Scatter { xs, ys, area, color } => {
                for (&x, &y) in xs.iter().zip(ys) {
                    svg.circle(p.px(x), p.py(y), area.sqrt(), *color);
                }
            }
            _ => {}
        }
    }
    let _ = writeln!(svg.out, "</g>");
    // annotations are drawn unclipped, but only when the annotated point is inside the axes
    for a in &ax.artists {
        match a {
            Artist::Annotate { x, y, text, size, offset } if p.xs.contains(*x) && p.ys.contains(*y) => {
                svg.plain(p.px(*x) + offset.0, p.py(*y) - offset.1, text, *size, Anchor::Start, VAlign::Baseline);
            }
            Artist::AxesText { fx, fy, text, size } => {
                svg.plain(x0 + fx * p.w, y1 - fy * p.h, text, *size, Anchor::Middle, VAlign::Baseline);
            }
            _ => {}
        }
    }
    // frame, ticks and tick labels
    let _ = writeln!(
        svg.out,
        "<rect x=\"{x0:.3}\" y=\"{y0:.3}\" width=\"{:.3}\" height=\"{:.3}\" fill=\"none\" stroke=\"#000000\" stroke-width=\"0.8\"/>",
        p.w, p.h
    );
    for (v, label) in &p.xt.major {
        let x = p.px(*v);
        svg.line((x, y1), (x, y1 + TICK_LEN), Color::BLACK, 0.8, "");
        svg.text((x, y1 + TICK_LEN + TICK_PAD), label, ax.xtick_size, (Anchor::Middle, VAlign::Top), false);
    }
    for v in &p.xt.minor {
        let x = p.px(*v);
        svg.line((x, y1), (x, y1 + MINOR_TICK_LEN), Color::BLACK, 0.6, "");
    }
    let mut ytick_w: f64 = 0.0;
    for (v, label) in &p.yt.major {
        let y = p.py(*v);
        svg.line((x0 - TICK_LEN, y), (x0, y), Color::BLACK, 0.8, "");
        svg.text((x0 - TICK_LEN - TICK_PAD, y), label, ax.ytick_size, (Anchor::End, VAlign::Center), false);
        ytick_w = ytick_w.max(label.width(svg.fonts, ax.ytick_size));
    }
    for v in &p.yt.minor {
        let y = p.py(*v);
        svg.line((x0 - MINOR_TICK_LEN, y), (x0, y), Color::BLACK, 0.6, "");
    }
    // axis labels and title
    if !ax.xlabel.is_empty() {
        let xtick_h = line_height(svg.fonts, ax.xtick_size) * if ax.xlog { 1.25 } else { 1.0 };
        let y = y1 + TICK_LEN + TICK_PAD + xtick_h + LABEL_PAD;
        svg.plain((x0 + x1) / 2.0, y, &ax.xlabel, ax.label_size, Anchor::Middle, VAlign::Top);
    }
    if !ax.ylabel.is_empty() {
        let x = x0 - TICK_LEN - TICK_PAD - ytick_w - LABEL_PAD;
        let label = Label::Plain(ax.ylabel.clone());
        svg.text((x, (y0 + y1) / 2.0), &label, ax.label_size, (Anchor::Middle, VAlign::Bottom), true);
    }
    if !ax.title.is_empty() {
        let lines: Vec<&str> = ax.title.split('\n').collect();
        let lh = line_height(svg.fonts, ax.title_size);
        for (i, line) in lines.iter().enumerate() {
            let bottom = y0 - TITLE_PAD - (lines.len() - 1 - i) as f64 * lh;
            svg.plain((x0 + x1) / 2.0, bottom, line, ax.title_size, Anchor::Middle, VAlign::Bottom);
        }
    }
    if let Some(legend) = &ax.legend {
        draw_legend(svg, ax, p, legend);
    }
}

fn render_svg(fig: &Figure, fonts: &Fonts) -> String {
    let (w, h) = (fig.width_in * 72.0, fig.height_in * 72.0);
    let mut svg = Svg { out: String::new(), fonts };
    let _ = writeln!(
        svg.out,
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{w:.3}\" height=\"{h:.3}\" viewBox=\"0 0 {w:.3} {h:.3}\">"
    );
    let _ = writeln!(svg.out, "<rect width=\"100%\" height=\"100%\" fill=\"#ffffff\"/>");
    for (i, (ax, p)) in fig.axes.iter().zip(place(fig, fonts)).enumerate() {
        draw_axes(&mut svg, ax, &p, &format!("axes{i}"));
    }
    svg.out.push_str("</svg>\n");
    svg.out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_ticks_follow_matplotlib() {
        assert_eq!(linear_ticks(-0.04, 0.93, 9), vec![0.0, 0.2, 0.4, 0.6000000000000001, 0.8]);
        assert_eq!(linear_ticks(0.0, 10.0, 9), vec![0.0, 2.0, 4.0, 6.0, 8.0, 10.0]);
        let t = linear_ticks(-0.25, 0.12, 9);
        assert_eq!(t.len(), 8);
        assert_eq!(
            format_linear(&t),
            ["\u{2212}0.25", "\u{2212}0.20", "\u{2212}0.15", "\u{2212}0.10", "\u{2212}0.05", "0.00", "0.05", "0.10"]
                .map(|s| Label::Plain(s.into()))
        );
        assert_eq!(format_linear(&[0.0, 0.5, 1.0]), ["0.0", "0.5", "1.0"].map(|s| Label::Plain(s.into())));
    }

    #[test]
    fn log_ticks_label_decades() {
        let t = log_ticks(1600.0, 125_000.0);
        let majors: Vec<f64> = t.major.iter().map(|m| m.0).collect();
        assert_eq!(majors, vec![1e4, 1e5]);
        assert!(t.minor.contains(&2000.0) && t.minor.contains(&90_000.0));
    }

    #[test]
    fn margins_stop_at_bar_baselines() {
        let mut e = Extent::new();
        e.add(0.0, false);
        e.add(1.0, false);
        e.sticky.push(0.0);
        assert_eq!(view_limits(&e, false, (None, None)), (0.0, 1.05));
    }
}
