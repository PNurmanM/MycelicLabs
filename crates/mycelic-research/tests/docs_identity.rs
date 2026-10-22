//! The committed documents regenerate from the committed artifacts byte for byte.
//!
//! Every generated document is compared line by line with the file in `docs/`; the only lines allowed to differ
//! are the `_Generated ...` lines, which carry the generation date and name the generator (the Python module then,
//! the Rust one now).  Also the report-generator tests of `test_mycelic.py` (`TestReportGenerators`).

use mycelic_research::loss_doc;
use mycelic_research::runner::{art, repo_root};

const DATE: &str = "2026-10-21";

/// Assert `generated` equals the committed file except on `_Generated` lines, which must exist in both, line for
/// line.  Reports the first differing line.
fn assert_matches_committed(rel: &str, generated: &str) {
    let path = repo_root().join(rel);
    let committed = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    // split on '\n' (not lines()) so a missing or extra trailing newline is a difference too
    let a: Vec<&str> = committed.split('\n').collect();
    let b: Vec<&str> = generated.split('\n').collect();
    let mut stamped = 0;
    for (i, (x, y)) in a.iter().zip(&b).enumerate() {
        if x.starts_with("_Generated") || y.starts_with("_Generated") {
            assert!(
                x.starts_with("_Generated") && y.starts_with("_Generated"),
                "{rel}:{}: only one side has the _Generated line\ncommitted: {x}\ngenerated: {y}",
                i + 1
            );
            stamped += 1;
            continue;
        }
        assert_eq!(x, y, "{rel}:{}: first differing line", i + 1);
    }
    assert_eq!(a.len(), b.len(), "{rel}: {} lines committed, {} generated", a.len(), b.len());
    assert!(stamped > 0 || !committed.contains("\n_Generated"), "{rel}: no _Generated line compared");
}

#[test]
fn loss_accounting_doc_regenerates() {
    let text = loss_doc::compose_at(DATE);
    assert_matches_committed("docs/MYCELIC_LOSS_ACCOUNTING.md", &text);
    assert!(text.contains(&format!("_Generated {DATE} by `crates/mycelic-research/src/loss_doc.rs` from the\n")));
}

#[test]
fn loss_doc_ranker_section_survives_a_fresh_clone() {
    let dumps = std::fs::read_dir(art())
        .unwrap()
        .filter_map(Result::ok)
        .any(|e| e.file_name().to_string_lossy().starts_with("hyp_features_final"));
    if dumps {
        eprintln!("final feature dumps present; the fallback is not exercised");
        return;
    }
    let sec = loss_doc::calibrator_section();
    assert!(!sec.contains("not yet complete"), "{sec}");
    assert!(sec.contains("_finalH"));
    assert!(sec.contains("_finalC"));
}
