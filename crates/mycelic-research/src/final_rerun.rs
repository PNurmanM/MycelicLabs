//! The two calibration-file steps of `research/mycelic/final_rerun.sh` that were inline Python:
//! `final_rerun archive-v1-calibration` and `final_rerun install-ranker <calibration file>`.

use crate::calibrator::write_json;
use crate::runner::{art, repo_root};
use anyhow::{Result, bail};
use serde_json::{Map, Value};

fn read(p: &std::path::Path) -> Result<Map<String, Value>> {
    Ok(serde_json::from_str(&std::fs::read_to_string(p)?)?)
}

/// Archive the v1 calibration (hand ranker, v1 knobs) as `artifacts/v1/calibration.json`.
pub fn archive_v1_calibration() -> Result<()> {
    let mut cal = read(&art().join("calibration.json"))?;
    for k in ["ranker", "ranker_archs", "ranker_arch_table", "vnext"] {
        cal.shift_remove(k);
    }
    write_json(&art().join("v1").join("calibration.json"), &Value::Object(cal), 1)?;
    println!("v1 calibration archived (hand ranker, v1 knobs)");
    Ok(())
}

/// Install the ranker of another calibration file into calibration.json, recording where it came from.
pub fn install_ranker(src: &str) -> Result<()> {
    let p = if std::path::Path::new(src).is_absolute() { src.into() } else { repo_root().join(src) };
    let from = read(&p)?;
    let path = art().join("calibration.json");
    let mut cal = read(&path)?;
    let mut ranker = from.get("ranker").and_then(Value::as_object).cloned().unwrap_or_default();
    ranker.insert("source_file".into(), Value::from(src));
    println!(
        "ranker installed from {src} l2 {} interactions {} n_train {}",
        crate::py::value::py_str(&ranker["l2"]),
        crate::py::value::py_str(&ranker["interactions"]),
        crate::py::value::py_str(&ranker["n_train"])
    );
    cal.insert("ranker".into(), Value::Object(ranker));
    write_json(&path, &Value::Object(cal), 1)
}

pub fn main(args: &[String]) -> Result<()> {
    match args.first().map(String::as_str) {
        Some("archive-v1-calibration") => archive_v1_calibration(),
        Some("install-ranker") => match args.get(1) {
            Some(src) => install_ranker(src),
            None => bail!("install-ranker needs a calibration file"),
        },
        _ => bail!("usage: final_rerun archive-v1-calibration | install-ranker <calibration file>"),
    }
}
