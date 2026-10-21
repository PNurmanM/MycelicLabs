//! Per-pattern loss accounting (`loss_accounting.py`): the first stage of the pipeline at which each hidden
//! pattern is lost.

use crate::runner::art;
use std::path::PathBuf;

pub const HIER_STAGES: [&str; 10] = [
    "extracted",
    "sketch_visible",
    "in_triage",
    "questioned",
    "descent_reached",
    "in_pool",
    "candidate",
    "matched_any",
    "matched_tau",
    "in_register",
];
pub const FLAT_STAGES: [&str; 6] = ["extracted", "in_pool", "candidate", "matched_any", "matched_tau", "in_register"];

pub const TAU: f64 = 0.5;

/// `artifacts/loss_funnel.jsonl`
pub fn funnel_path() -> PathBuf {
    art().join("loss_funnel.jsonl")
}
