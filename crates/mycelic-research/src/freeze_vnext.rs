//! Freeze the vNext knobs into calibration.json with their provenance (`freeze_vnext.py`):
//! `freeze_vnext question_frac=0.65:qf_v3 batched_descent=true:rk2_batch link_time=hybrid:refit_hyb`.
//! A value is `<literal>:<quick tag(s)>`; literals are parsed as JSON where they can be and kept as strings
//! otherwise.

use crate::calibrator::write_json;
use crate::py::value::py_repr;
use crate::runner::art;
use anyhow::{Result, bail};
use serde_json::{Map, Value, json};

pub const KNOBS: [&str; 8] = [
    "batched_descent",
    "link_time",
    "local_reextract",
    "merge_descent_evidence",
    "question_frac",
    "sketch_weak_bits",
    "strict_targeting",
    "triage_target_chains",
];

pub fn main(args: &[String]) -> Result<()> {
    let path = art().join("calibration.json");
    let mut cal: Map<String, Value> = serde_json::from_str(&std::fs::read_to_string(&path)?)?;
    let mut prov: Map<String, Value> = cal.get("vnext").and_then(Value::as_object).cloned().unwrap_or_default();
    for a in args {
        let (k, rest) = a.split_once('=').unwrap_or((a.as_str(), ""));
        if !KNOBS.contains(&k) {
            bail!("unknown knob {}; known: {}", crate::py::json::repr_str(k), py_repr(&json!(KNOBS)));
        }
        let (lit, tag) = rest.split_once(':').unwrap_or((rest, ""));
        let v: Value = serde_json::from_str(lit).unwrap_or_else(|_| Value::String(lit.to_string()));
        cal.insert(k.into(), v.clone());
        // several evidence files may be cited, comma-separated
        let ev: Vec<String> =
            tag.split(',').map(str::trim).filter(|t| !t.is_empty()).map(|t| format!("quick_{t}.jsonl")).collect();
        let ev = ev.join(", ");
        let frozen_at = chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string();
        println!("  {k} = {}   ({})", py_repr(&v), if ev.is_empty() { "no tag" } else { &ev });
        prov.insert(k.into(), json!({"value": v, "evidence": ev, "frozen_at": frozen_at}));
    }
    cal.insert("vnext".into(), Value::Object(prov));
    write_json(&path, &Value::Object(cal), 1)?;
    println!("wrote {}", path.display());
    Ok(())
}
