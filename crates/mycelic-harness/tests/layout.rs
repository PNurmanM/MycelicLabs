//! Port of tests/mycelic/test_packaging.py: the agent side needs none of the server's dependencies.
//!
//! Agents, the SDK, the smoke test and the demos talk to a deployed Mycelic over HTTP; they must not link the
//! service or its stack (axum, async-nats, prometheus).  The resolved dependency graph (`cargo metadata`) is the
//! proof: neither `mycelic-sdk` nor `mycelic-harness` reaches the `mycelic` package or those crates, through normal
//! dependencies, at any depth.

use serde_json::Value;
use std::collections::{BTreeSet, HashMap};
use std::process::Command;

fn metadata() -> Value {
    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".into());
    let out = Command::new(cargo)
        .args(["metadata", "--format-version", "1"])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .expect("cargo metadata runs");
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    serde_json::from_slice(&out.stdout).unwrap()
}

/// Package names reachable from `root` through normal (non-dev, non-build) dependencies.
fn runtime_closure(meta: &Value, root: &str) -> BTreeSet<String> {
    let names: HashMap<&str, &str> = meta["packages"]
        .as_array()
        .unwrap()
        .iter()
        .map(|p| (p["id"].as_str().unwrap(), p["name"].as_str().unwrap()))
        .collect();
    let nodes: HashMap<&str, &Value> =
        meta["resolve"]["nodes"].as_array().unwrap().iter().map(|n| (n["id"].as_str().unwrap(), n)).collect();
    let start = names.iter().find(|(_, n)| **n == root).map(|(id, _)| *id).expect("package in the workspace");
    let mut seen = BTreeSet::new();
    let mut stack = vec![start];
    while let Some(id) = stack.pop() {
        for dep in nodes[id]["deps"].as_array().unwrap() {
            let normal = dep["dep_kinds"].as_array().unwrap().iter().any(|k| k["kind"].is_null());
            let dep_id = dep["pkg"].as_str().unwrap();
            if normal && seen.insert(names[dep_id].to_string()) {
                stack.push(dep_id);
            }
        }
    }
    seen
}

const SERVER_STACK: [&str; 5] = ["mycelic", "axum", "async-nats", "prometheus", "axum-server"];

#[test]
fn test_sdk_does_not_depend_on_the_server() {
    let deps = runtime_closure(&metadata(), "mycelic-sdk");
    for name in SERVER_STACK {
        assert!(!deps.contains(name), "mycelic-sdk reaches {name}");
    }
}

#[test]
fn test_harness_smoke_and_demos_do_not_depend_on_the_server() {
    let deps = runtime_closure(&metadata(), "mycelic-harness");
    assert!(deps.contains("mycelic-sdk"));
    for name in SERVER_STACK {
        assert!(!deps.contains(name), "mycelic-harness reaches {name}");
    }
}

#[test]
fn test_server_stack_is_really_in_the_server_closure() {
    // guards the tests above: if the closure walk stopped finding dependencies they would pass vacuously
    let deps = runtime_closure(&metadata(), "mycelic");
    for name in ["mycelic-sdk", "axum", "async-nats", "prometheus"] {
        assert!(deps.contains(name), "the server closure lacks {name}");
    }
}

#[test]
fn test_version_is_single_sourced() {
    let meta = metadata();
    let versions: BTreeSet<&str> = meta["packages"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|p| p["name"].as_str().is_some_and(|n| n.starts_with("mycelic")))
        .map(|p| p["version"].as_str().unwrap())
        .collect();
    assert_eq!(
        versions,
        BTreeSet::from([mycelic_sdk::VERSION]),
        "every crate carries the workspace version, the one the SDK reports"
    );
}
