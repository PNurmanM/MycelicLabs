//! Port of tests/mycelic/test_smoke_process.py: the production smoke test with the process driver (real
//! nats-server, real service process, real agent processes).  The compose driver runs in CI with Docker.

mod support;

#[test]
fn test_smoke_scenario_passes() {
    if !support::ready_for_processes() {
        return;
    }
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_mycelic-smoke"))
        .args(["--driver", "process", "--agents-per-team", "2"])
        .output()
        .unwrap();
    let text = String::from_utf8_lossy(&out.stdout);
    assert_eq!(out.status.code(), Some(0), "{text}\n{}", String::from_utf8_lossy(&out.stderr));
    assert!(text.contains("PASS in"));
    assert_eq!(text.matches("  ok  ").count(), 9, "all nine steps reported");
}
