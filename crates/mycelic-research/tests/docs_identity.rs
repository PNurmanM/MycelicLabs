//! The committed documents regenerate from the committed artifacts byte for byte.
//!
//! Every generated document is compared line by line with the file in `docs/`; the only lines allowed to differ
//! are the `_Generated ...` lines, which carry the generation date and name the generator (the Python module then,
//! the Rust one now).  Also the report-generator tests of `test_mycelic.py` (`TestReportGenerators`).

use mycelic_research::loss_report::{int, text};
use mycelic_research::py::fmt;
use mycelic_research::runner::{art, repo_root};
use mycelic_research::{findings, loss_doc, report, vnext_docs};
use std::collections::BTreeSet;

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

#[test]
fn vnext_docs_regenerate() {
    let stamp = format!("{DATE} 12:00 UTC");
    let files = vnext_docs::build_at(&stamp);
    let names: Vec<&str> = files.iter().map(|f| f.0).collect();
    assert_eq!(
        names,
        [
            "fig_topology.svg",
            "fig_loop.svg",
            "NEXT_RESEARCH_REPORT.md",
            "VNEXT_ARCHITECTURE.md",
            "EXPERIMENT_MATRIX.md",
            "LOSS_ACCOUNTING.md",
            "REVIEWER_ATTACK.md",
            "ALL.md"
        ]
    );
    for (name, text) in &files {
        assert_matches_committed(&format!("docs/mycelic_vnext/{name}"), text);
    }
    let report = &files[2].1;
    assert!(report.contains(&format!(
        "_Generated {stamp} from `research/mycelic/artifacts/` (new) and `artifacts/v1/` (old) by `vnext_docs.rs`."
    )));
}

#[test]
fn enterprise_report_regenerates() {
    let text = report::build_at(DATE);
    assert_matches_committed("docs/MYCELIC_ENTERPRISE.md", &text);
    assert!(text.contains(&format!("_Generated {DATE} by `crates/mycelic-research/src/report.rs` from the\n")));
}

#[test]
fn missing_architecture_is_never_reported_as_zero() {
    let rows = findings::rows();
    let ladder = ["C_recursive_sum", "D_hier_nolineage", "E_hier_lineage", "F_hier_retrieval", "G_hier_questions"];
    let sc = findings::largest_complete_scale(&rows, &ladder).expect("a scale with the whole ladder");
    let ran = |a: &str, s: i64| rows.iter().any(|r| text(r, "arch") == a && int(r, "scale") == s);
    for a in ladder {
        assert!(ran(a, sc), "{a}");
    }
    let summary = findings::executive_summary();
    let before_2 = summary.split("**2.").next().unwrap();
    for s in rows.iter().map(|r| int(r, "scale")).collect::<BTreeSet<_>>() {
        if ladder.iter().any(|a| !ran(a, s)) {
            assert!(
                !before_2.contains(&format!("At {} users", fmt::i(s, ","))),
                "scale {s} reported with a rung missing"
            );
        }
    }
}
