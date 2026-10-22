//! Model tiers as explicit, sweepable capability vectors (`models.py`).
//!
//! The numbers in the anchors are *assumed placements* of named model classes on a capability axis; nothing here
//! is a measurement of a specific checkpoint.  What the benchmark establishes is a function from operator quality
//! `q` to end-to-end architecture quality, and the named models are labels on that axis.
//!
//! Cost is reported in normalised compute units, `cu = active_params_B * (tok_in + 4 tok_out) / 1e3`; a dollar
//! estimate is a clearly-labelled secondary column using [`usd_per_mtok`].

/// How "hard" capabilities (multi-step verification, near-miss entity discrimination, question generation) scale
/// with q.  The single most load-bearing modelling assumption, so it is swept (E7) rather than fixed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HardShape {
    Convex,
    Linear,
    Concave,
}

impl HardShape {
    pub fn name(self) -> &'static str {
        match self {
            HardShape::Convex => "convex",
            HardShape::Linear => "linear",
            HardShape::Concave => "concave",
        }
    }

    pub fn parse(s: &str) -> HardShape {
        match s {
            "linear" => HardShape::Linear,
            "concave" => HardShape::Concave,
            _ => HardShape::Convex,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Tier {
    pub name: String,
    pub q: f64,
    /// nominal *active* params, billions (cost proxy)
    pub params_b: f64,
    /// usable context tokens
    pub ctx: i64,
    // extraction (text -> structured claim)
    pub extract_recall: f64,
    pub extract_precision: f64,
    pub pred_confusion: f64,
    pub entity_fidelity: f64,
    // abstraction (claims -> knowledge objects for the parent)
    pub salience_noise: f64,
    pub abstract_retention: f64,
    pub abstract_distortion: f64,
    // synthesis / verification
    pub synth_depth: i64,
    pub causal_check: f64,
    pub temporal_check: f64,
    pub entity_check: f64,
    pub dedup_check: f64,
    pub contradiction_acc: f64,
    pub hallucination: f64,
    // questioning / routing
    pub question_quality: f64,
    pub route_quality: f64,
    // runtime
    pub prefill_tok_s: f64,
    pub decode_tok_s: f64,
    pub concurrency: i64,
}

impl Tier {
    /// normalised compute units
    pub fn cu(&self, tok_in: i64, tok_out: i64) -> f64 {
        self.params_b * (tok_in as f64 + 4.0 * tok_out as f64) / 1e3
    }

    pub fn seconds(&self, tok_in: i64, tok_out: i64) -> f64 {
        tok_in as f64 / self.prefill_tok_s + tok_out as f64 / self.decode_tok_s
    }
}

/// Price assumptions (USD per million tokens) for the secondary dollar column; 1.0 for an unnamed tier.
pub fn usd_per_mtok(name: &str) -> f64 {
    match name {
        "lexical" => 0.001,
        "edge-3b" => 0.04,
        "small-7b" => 0.08,
        "mid-14b" => 0.16,
        "mid-32b" => 0.35,
        "large-70b" => 0.75,
        "frontier" => 6.00,
        "frontier-plus" => 15.00,
        _ => 1.0,
    }
}

fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + (b - a) * t
}

/// Python's `round(x)` for a float: nearest integer, ties to even.
pub fn py_round(x: f64) -> i64 {
    x.round_ties_even() as i64
}

fn hard(q: f64, a: f64, shape: HardShape) -> f64 {
    match shape {
        HardShape::Convex => crate::py::pow(q, a),
        HardShape::Linear => q,
        HardShape::Concave => 1.0 - crate::py::pow(1.0 - q, a),
    }
}

/// Monotone capability vector as a function of the scalar q in [0, 1] (`tier_from_q`).
pub fn tier_from_q(q: f64, name: Option<&str>, params_b: Option<f64>, ctx: Option<i64>, shape: HardShape) -> Tier {
    let q = q.clamp(0.0, 1.0);
    let easy = |a: f64| 1.0 - crate::py::pow(1.0 - q, a);
    let hard = |a: f64| hard(q, a, shape);
    let dec = |hi: f64, lo: f64| lerp(hi, lo, q);
    Tier {
        name: name.map(str::to_string).unwrap_or_else(|| format!("q{q:.2}")),
        q,
        params_b: params_b.unwrap_or_else(|| 3.0 * crate::py::pow(120.0 / 3.0, q)),
        ctx: ctx.unwrap_or_else(|| (8_000.0 * crate::py::pow(128.0, q)) as i64),
        extract_recall: lerp(0.55, 0.96, easy(1.4)),
        extract_precision: lerp(0.70, 0.985, easy(1.6)),
        pred_confusion: dec(0.30, 0.03),
        entity_fidelity: lerp(0.72, 0.995, easy(1.8)),
        salience_noise: dec(0.42, 0.06),
        abstract_retention: lerp(0.62, 0.97, easy(1.5)),
        abstract_distortion: dec(0.22, 0.015),
        synth_depth: py_round(lerp(2.0, 7.0, hard(1.0))),
        causal_check: lerp(0.50, 0.97, hard(1.3)),
        temporal_check: lerp(0.58, 0.99, easy(1.2)),
        entity_check: lerp(0.48, 0.96, hard(1.2)),
        dedup_check: lerp(0.55, 0.97, hard(1.0)),
        contradiction_acc: lerp(0.50, 0.95, hard(1.2)),
        hallucination: lerp(0.34, 0.025, easy(1.2)),
        question_quality: lerp(0.30, 0.94, hard(1.4)),
        route_quality: lerp(0.42, 0.96, hard(1.0)),
        prefill_tok_s: lerp(9000.0, 1600.0, q),
        decode_tok_s: lerp(220.0, 42.0, q),
        concurrency: py_round(lerp(4096.0, 48.0, q)),
    }
}

/// Named anchors, in `ANCHOR_Q` order: (name, q, active params B, context).
pub const ANCHOR_SPECS: [(&str, f64, f64, i64); 8] = [
    // Not a language model at all: a regex / gazetteer / NER pipeline.
    ("lexical", 0.0, 0.02, 4_000),
    ("edge-3b", 0.10, 3.0, 16_000),
    ("small-7b", 0.28, 7.0, 32_000),
    ("mid-14b", 0.44, 14.0, 64_000),
    ("mid-32b", 0.58, 32.0, 128_000),
    ("large-70b", 0.72, 70.0, 200_000),
    ("frontier", 0.88, 200.0, 400_000),
    ("frontier-plus", 1.00, 400.0, 1_000_000),
];

/// `anchor(name)` under a given hard-capability shape.
pub fn anchor(name: &str, shape: HardShape) -> Tier {
    let (_, q, params, ctx) =
        *ANCHOR_SPECS.iter().find(|a| a.0 == name).unwrap_or_else(|| panic!("unknown model anchor {name:?}"));
    let mut t = tier_from_q(q, Some(name), Some(params), Some(ctx), shape);
    if name == "lexical" {
        // measured-style overrides for a non-LLM pipeline: good at copying strings, blind to meaning
        t.extract_recall = 0.84;
        t.extract_precision = 0.80;
        t.pred_confusion = 0.12;
        t.entity_fidelity = 0.97;
        t.entity_check = 0.30;
        t.causal_check = 0.50;
        t.temporal_check = 0.60;
        t.dedup_check = 0.55;
        t.contradiction_acc = 0.50;
        t.hallucination = 0.0;
        t.synth_depth = 2;
        t.salience_noise = 0.30;
        t.abstract_retention = 0.95;
        t.abstract_distortion = 0.02;
        t.question_quality = 0.10;
        t.route_quality = 0.40;
        t.prefill_tok_s = 2_000_000.0;
        t.decode_tok_s = 50_000.0;
        t.concurrency = 100_000;
    }
    t
}

/// Level -> tier assignments (the compute-allocation designs under test), in `ALLOCATIONS` order.
/// Index 0..5 == USER, TEAM, DEPT, SITE, REGION, ENTERPRISE.
pub const ALLOCATIONS: [(&str, [&str; 6]); 11] = [
    ("flat-small", ["small-7b"; 6]),
    ("flat-mid", ["mid-32b"; 6]),
    ("flat-frontier", ["frontier"; 6]),
    ("naive-ladder", ["small-7b", "mid-14b", "mid-32b", "large-70b", "frontier", "frontier-plus"]),
    ("back-loaded", ["small-7b", "small-7b", "mid-14b", "mid-32b", "frontier", "frontier-plus"]),
    ("front-loaded", ["mid-14b", "mid-32b", "mid-32b", "mid-14b", "mid-14b", "large-70b"]),
    ("kernel-only", ["edge-3b", "edge-3b", "edge-3b", "edge-3b", "edge-3b", "frontier-plus"]),
    ("two-step", ["small-7b", "small-7b", "mid-32b", "mid-32b", "mid-32b", "frontier-plus"]),
    ("edge-bottom", ["edge-3b", "mid-14b", "mid-32b", "large-70b", "frontier", "frontier-plus"]),
    ("lexical-bottom", ["lexical", "small-7b", "mid-14b", "mid-32b", "frontier", "frontier-plus"]),
    ("lexical-to-kernel", ["lexical", "lexical", "lexical", "lexical", "lexical", "frontier-plus"]),
];

pub fn allocation_tiers(name: &str) -> [&'static str; 6] {
    ALLOCATIONS.iter().find(|a| a.0 == name).unwrap_or_else(|| panic!("unknown allocation {name:?}")).1
}

/// `allocation(name)` under a hard-capability shape.
pub fn allocation(name: &str, shape: HardShape) -> Vec<Tier> {
    allocation_tiers(name).iter().map(|t| anchor(t, shape)).collect()
}

/// `allocation(name)` with the default (convex) shape.
pub fn allocation_default(name: &str) -> Vec<Tier> {
    allocation(name, HardShape::Convex)
}

pub fn describe(alloc: &[Tier]) -> String {
    alloc.iter().map(|t| t.name.as_str()).collect::<Vec<_>>().join(" -> ")
}

/// `python3 -m research.mycelic.models`: the named anchors on the capability axis.
pub fn main(_args: &[String]) -> anyhow::Result<()> {
    use crate::py::fmt::f;
    for (name, ..) in ANCHOR_SPECS {
        let t = anchor(name, HardShape::Convex);
        println!(
            "{} q={} recall={} causal={} depth={} halluc={} ctx={}",
            crate::py::fmt::s(name, "14s"),
            f(t.q, ".2f"),
            f(t.extract_recall, ".2f"),
            f(t.causal_check, ".2f"),
            t.synth_depth,
            f(t.hallucination, ".3f"),
            t.ctx
        );
    }
    Ok(())
}
