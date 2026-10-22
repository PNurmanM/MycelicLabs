//! The calibrator against the Python on a slice of the committed feature dump
//! (`tests/golden/gen/gen_calibrator.py`): standardisation, the gradient-boosted trees and their predictions
//! and the AUC are bit-exact; the logistic's weights, which the Python solved through BLAS/LAPACK, agree to
//! rounding.

use mycelic_research::calibrator::{self, Mat};
use serde_json::{Map, Value};

fn load() -> Value {
    let path = format!("{}/tests/golden/calibrator.json", env!("CARGO_MANIFEST_DIR"));
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn rows(v: &Value) -> Vec<Map<String, Value>> {
    v.as_array().unwrap().iter().map(|r| r.as_object().unwrap().clone()).collect()
}

fn bits(v: &Value) -> Vec<u64> {
    v.as_array().unwrap().iter().map(|x| x.as_u64().unwrap()).collect()
}

fn exact_platform() -> bool {
    cfg!(all(target_os = "macos", target_arch = "aarch64"))
}

#[test]
fn calibrator_numerics_match_python() {
    let fx = load();
    let tr = rows(&fx["train"]);
    let te = rows(&fx["test"]);
    let y: Vec<f64> = tr.iter().map(|r| if r["gold"] == Value::Bool(true) { 1.0 } else { 0.0 }).collect();
    let yt: Vec<f64> = te.iter().map(|r| if r["gold"] == Value::Bool(true) { 1.0 } else { 0.0 }).collect();
    for inter in [false, true] {
        let want = &fx[&format!("inter_{}", if inter { "True" } else { "False" })];
        let (xs, mu, sd) = calibrator::standardise(&calibrator::design(&tr, inter), None);
        assert_eq!(mu.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), bits(&want["mu"]), "mu inter={inter}");
        assert_eq!(sd.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), bits(&want["sd"]), "sd inter={inter}");
        let (xt, _, _): (Mat, _, _) = calibrator::standardise(&calibrator::design(&te, inter), Some((&mu, &sd)));
        let w = calibrator::fit_logistic(&xs, &y, 1.0, None);
        for (a, b) in w.iter().zip(want["logistic"].as_array().unwrap()) {
            let b = b.as_f64().unwrap();
            assert!((a - b).abs() <= 1e-8 * b.abs().max(1.0), "logistic weight {a} vs {b}");
        }
        let rw: Vec<f64> =
            tr.iter().map(|r| if r.get("decoy") == Some(&Value::Bool(true)) { 3.0 } else { 1.0 }).collect();
        let trees = calibrator::fit_gbdt(&xs, &y, 12, 0.1, Some(&rw));
        let got: Vec<Value> = trees.iter().map(|t| t.to_json()).collect();
        if exact_platform() {
            assert_eq!(Value::Array(got), want["gbdt_trees"], "trees inter={inter}");
            let pred = calibrator::gbdt_predict(&trees, 0.1, &xt);
            assert_eq!(
                pred.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                bits(&want["gbdt_pred"]),
                "gbdt predictions"
            );
        }
        let conf: Vec<f64> = te.iter().map(|r| r["conf"].as_f64().unwrap()).collect();
        assert_eq!(calibrator::auc(&conf, &yt).to_bits(), want["auc_conf"].as_u64().unwrap(), "auc");
    }
}
