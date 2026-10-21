//! The learned candidate ranker (`calibrator.py`): feature dumps, logistic and GBDT fitting, leave-one-seed-out
//! selection, per-architecture adoption and the held-out evaluation.

pub const CAL_SEEDS: [u64; 3] = [500, 501, 502];
pub const EVAL_SEEDS: [u64; 5] = [0, 1, 2, 3, 4];
pub const TAU: f64 = 0.5;

/// `evaluate_calibrator(tag)`: fit on the calibration seeds of a feature dump, evaluate on the evaluation seeds.
pub fn evaluate_calibrator(tag: &str) -> serde_json::Value {
    let _ = tag;
    unimplemented!("calibrator::evaluate_calibrator is being ported")
}
