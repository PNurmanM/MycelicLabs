//! Rerun experiments on the committed calibration and compare every row with the committed artifact, all fields
//! but `runtime_s`.  Ignored by default (release-mode work); CI runs it with `--ignored`:
//!
//!     cargo test --release -p mycelic-research --test artifacts_reproduce -- --ignored
//!
//! `MYCELIC_REPRODUCE` (comma-separated experiment names) widens the set; the default is the light experiments.
//! The full suite was rerun this way when the port was made: all 16 result files reproduced.

use mycelic_research::experiments;
use mycelic_research::runner::{Cal, repo_root};
use serde_json::{Map, Value};

fn rows(text: &str) -> Vec<Map<String, Value>> {
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            let mut m: Map<String, Value> = serde_json::from_str(l).unwrap();
            m.shift_remove("runtime_s");
            m
        })
        .collect()
}

#[test]
#[ignore]
fn experiments_reproduce_the_committed_rows() {
    let committed = repo_root().join("research/mycelic/artifacts");
    let tmp = std::env::temp_dir().join(format!("mycelic-reproduce-{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    std::fs::copy(committed.join("calibration.json"), tmp.join("calibration.json")).unwrap();
    // SAFETY: the only test in this binary; nothing else reads the environment concurrently
    unsafe { std::env::set_var("MYCELIC_ART", &tmp) };
    let which = std::env::var("MYCELIC_REPRODUCE").unwrap_or_else(|_| "e1d,e3b,e3c,e4b,e6,e8".into());
    let file_of = |e: &str| match e {
        "e1" | "e1c" => "e1_baselines",
        "e1b" | "e11" => "e1b_extra",
        "e1d" => "e1d_b4_capped",
        "e2" => "e2_ablations",
        "e3" => "e3_allocation",
        "e3b" => "e3b_level_marginal",
        "e3c" => "e3c_q_sweep",
        "e4" => "e4_fanin",
        "e4b" => "e4b_dept_fanin",
        "e5" => "e5_adversarial",
        "e6" => "e6_frontier",
        "e7" => "e7_shape",
        "e8" => "e8_crosslinks",
        "e9" => "e9_privacy",
        "e10" => "e10_scale_trend",
        "e12" => "e12_provenance",
        other => panic!("unknown experiment {other}"),
    };
    let mut checked = Vec::new();
    for e in which.split(',').map(str::trim) {
        experiments::run_one(&Cal::load(), e).unwrap();
        checked.push(file_of(e));
    }
    checked.sort();
    checked.dedup();
    for f in checked {
        let mut got = rows(&std::fs::read_to_string(tmp.join(format!("{f}.jsonl"))).unwrap());
        let mut want = rows(&std::fs::read_to_string(committed.join(format!("{f}.jsonl"))).unwrap());
        // e1b_extra's rows were interleaved by two parallel streams in the original run
        let key = |m: &Map<String, Value>| serde_json::to_string(m).unwrap();
        got.sort_by_key(key);
        want.sort_by_key(key);
        assert_eq!(got.len(), want.len(), "{f}: row count");
        for (g, w) in got.iter().zip(&want) {
            assert_eq!(g, w, "{f}: a rerun row differs from the committed one");
        }
        eprintln!("{f}: {} rows identical", got.len());
    }
    let _ = std::fs::remove_dir_all(&tmp);
}
