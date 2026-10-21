//! Port of tests/mycelic/test_strategic_demo_process.py: the strategic-synthesis demo end to end with the process
//! driver; every printed verdict is a check and the exit code is 0 only when all held.

mod support;

#[test]
fn test_strategic_demo_passes() {
    if !support::ready_for_processes() {
        return;
    }
    let out_dir = tempfile::tempdir().unwrap();
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_mycelic-strategic-demo"))
        .args(["--driver", "process", "--pace", "0", "--out"])
        .arg(out_dir.path())
        .output()
        .unwrap();
    let text = String::from_utf8_lossy(&out.stdout);
    assert_eq!(out.status.code(), Some(0), "{text}\n{}", String::from_utf8_lossy(&out.stderr));
    assert!(text.contains("10/10 checks passed"), "{text}");
    for f in ["lineage.json", "lineage.mmd", "summary.json"] {
        assert!(out_dir.path().join(f).is_file(), "{f} written");
    }
}
