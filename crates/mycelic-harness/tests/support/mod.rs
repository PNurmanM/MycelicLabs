//! Shared by the harness's process tests: building the service and agent binaries the drivers start, and finding
//! nats-server.
#![allow(dead_code)]

use std::path::PathBuf;
use std::process::Command;
use std::sync::OnceLock;

/// Build `mycelic` and `mycelic-agent` (same profile as this test) once; false when the build failed.
pub fn binaries_built() -> bool {
    static BUILT: OnceLock<bool> = OnceLock::new();
    *BUILT.get_or_init(|| {
        let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".into());
        let release = std::env::current_exe().map(|p| p.to_string_lossy().contains("/release/")).unwrap_or(false);
        let mut cmd = Command::new(cargo);
        cmd.args(["build", "-p", "mycelic", "-p", "mycelic-sdk", "--bins"]).current_dir(repo_root());
        if release {
            cmd.arg("--release");
        }
        cmd.status().map(|s| s.success()).unwrap_or(false)
    })
}

pub fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize().unwrap()
}

pub fn nats_available() -> bool {
    mycelic_harness::nats_server_bin().is_some_and(|p| p.is_file())
}

/// `true` when a process test can run here; otherwise it says why and the test returns (a skip).
pub fn ready_for_processes() -> bool {
    if !nats_available() {
        eprintln!("skipped: nats-server binary not available (set MYCELIC_NATS_SERVER_BIN)");
        return false;
    }
    assert!(binaries_built(), "building the mycelic and mycelic-agent binaries failed");
    true
}
