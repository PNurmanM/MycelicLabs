//! Port of tests/mycelic/test_hierarchy_and_auth.py: the hierarchy helpers, the key format and the visibility
//! rule in code (the SQL twin is checked through the API and against the golden fixture).

use mycelic::auth::{RateLimiter, generate_api_key, memory_visible, parse_agent_id};
use mycelic::hierarchy::{AgentPath, ancestors, child_unit_of, is_ancestor_or_self, layer_of_path, unit_at_layer};
use mycelic::models::Memory;
use serde_json::Map;

fn mem(layer: &str, scope: &str, visibility: &str) -> Memory {
    Memory {
        memory_id: "m".into(),
        org_id: "acme".into(),
        layer: layer.into(),
        scope: scope.into(),
        text: "t".into(),
        topic: None,
        slot: None,
        entity: None,
        kind: "fact".into(),
        confidence: 0.5,
        support: 1,
        independent_teams: 1,
        producer_id: "a".into(),
        operator: "agent_observation".into(),
        rule_id: None,
        event_id: None,
        visibility: visibility.into(),
        status: "active".into(),
        superseded_by: None,
        created_at: String::new(),
        applied_at: None,
        source_event_ids: vec![],
        local_ref: None,
        metadata: Map::new(),
    }
}

#[test]
fn test_layers_follow_path_depth() {
    let p = AgentPath::from_parts("acme", Some("eu"), Some("acme-de"), Some("ops"), Some("log"), "a1").unwrap();
    assert_eq!(layer_of_path(&p.path()).unwrap(), "agent");
    assert_eq!(layer_of_path(&p.team_path()).unwrap(), "team");
    assert_eq!(layer_of_path("acme").unwrap(), "enterprise");
    assert_eq!(unit_at_layer(&p.path(), "department").unwrap().unwrap(), "acme/eu/acme-de/ops");
    assert_eq!(
        ancestors(&p.team_path(), true).unwrap(),
        ["acme", "acme/eu", "acme/eu/acme-de", "acme/eu/acme-de/ops", "acme/eu/acme-de/ops/log"]
    );
    assert_eq!(child_unit_of(&p.path(), "acme/eu").unwrap(), "acme/eu/acme-de");
    assert!(is_ancestor_or_self("acme/eu", &p.path()));
    assert!(!is_ancestor_or_self("acme/eu/other", &p.path()));
}

#[test]
fn test_small_org_defaults_and_validation() {
    let p = AgentPath::from_parts("acme", None, None, None, None, "a1").unwrap();
    assert_eq!(p.path(), "acme/acme/acme/acme/acme/a1");
    assert!(AgentPath::from_parts("Acme Inc", None, None, None, None, "a1").is_err());
    assert!(AgentPath::parse("acme/eu").is_err());
    assert!(layer_of_path("a/b/c/d/e/f/g").is_err());
}

#[test]
fn test_key_format() {
    let (key, key_hash, _prefix) = generate_api_key("agent-7");
    assert!(key.starts_with("mk_agent-7."));
    assert_eq!(parse_agent_id(&key), Some("agent-7"));
    assert_eq!(key_hash.len(), 64);
    assert_eq!(parse_agent_id("nope"), None);
}

#[test]
fn test_visibility_rules() {
    let (me, team) = ("acme/eu/de/ops/log/a1", "acme/eu/de/ops/log");
    let v = |m: &Memory| memory_visible(m, Some(me), Some(team));
    assert!(v(&mem("agent", "acme/eu/de/ops/log/a2", "team")));
    assert!(!v(&mem("agent", "acme/eu/de/ops/proc/b1", "team")));
    assert!(v(&mem("agent", "acme/eu/de/ops/proc/b1", "org")));
    assert!(v(&mem("department", "acme/eu/de/ops", "team")));
    assert!(v(&mem("enterprise", "acme", "team")));
    assert!(!v(&mem("department", "acme/eu/de/sales", "team")));
    assert!(!v(&mem("team", "acme/eu/de/ops/proc", "team")));
    assert!(memory_visible(&mem("team", "acme/eu/de/ops/proc", "team"), None, None));
}

#[test]
fn test_rate_limiter() {
    let mut r = RateLimiter::new(10.0, 3);
    let got: Vec<bool> = (0..4).map(|_| r.allow_at("k", 0.0, 1)).collect();
    assert_eq!(got, [true, true, true, false]);
    assert!(r.allow_at("k", 0.2, 1));
    assert!(RateLimiter::new(0.0, 100).allow("k", 1));
}
