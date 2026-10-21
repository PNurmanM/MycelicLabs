//! Port of tests/mycelic/test_store_migration.py: a version-1 database (rules without the composition columns)
//! opens as version 2 with the documented defaults, re-opens as a no-op, and a database from a newer version
//! refuses to open.

use mycelic::models::{random_ids, system_clock};
use mycelic::store::{SCHEMA_VERSION, Store};

const V1_DDL: &str = r#"
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE rules (
    rule_id TEXT PRIMARY KEY, org_id TEXT, target_layer TEXT NOT NULL, required_slots TEXT NOT NULL,
    conclusion TEXT NOT NULL, topic_prefix TEXT, min_agents INTEGER NOT NULL DEFAULT 2, min_teams INTEGER NOT NULL DEFAULT 1,
    kind TEXT NOT NULL DEFAULT 'risk', enabled INTEGER NOT NULL DEFAULT 1, metadata TEXT NOT NULL DEFAULT '{}', updated_at TEXT NOT NULL
);
INSERT INTO meta(key, value) VALUES ('schema_version', '1');
INSERT INTO rules(rule_id, org_id, target_layer, required_slots, conclusion, topic_prefix, min_agents, min_teams, kind, enabled, metadata, updated_at)
VALUES ('legacy', NULL, 'team', '["a","b"]', 'x', 'ops', 2, 1, 'risk', 1, '{}', '2026-01-01T00:00:00+00:00');
"#;

fn open(path: &std::path::Path) -> anyhow::Result<Store> {
    Store::open(path.to_str().unwrap(), system_clock(), random_ids())
}

#[test]
fn test_v1_database_migrates_to_current_schema() {
    let tmp = tempfile::tempdir().unwrap();
    let db = tmp.path().join("v1.db");
    rusqlite::Connection::open(&db).unwrap().execute_batch(V1_DDL).unwrap();
    let store = open(&db).unwrap();
    let cols: Vec<String> = store
        .read(|d| {
            let mut stmt = d.c.prepare("PRAGMA table_info(rules)")?;
            Ok(stmt.query_map([], |r| r.get::<_, String>("name"))?.collect::<rusqlite::Result<Vec<_>>>()?)
        })
        .unwrap();
    for c in ["sources", "emits_slot", "emits_topic", "min_units", "corroborate"] {
        assert!(cols.iter().any(|x| x == c), "column {c} added");
    }
    assert_eq!(store.get_meta("schema_version").unwrap().unwrap(), SCHEMA_VERSION.to_string());
    assert_eq!(SCHEMA_VERSION, 2);
    let rule = store.get_rule("legacy").unwrap().unwrap();
    assert_eq!(rule.required_slots, ["a", "b"]);
    assert_eq!(rule.sources, ["agent_observation"]);
    assert_eq!(rule.emits_slot, None);
    assert_eq!(rule.emits_topic, None);
    assert!(rule.min_units.is_empty());
    assert!(!rule.corroborate);
    store.close();
    let again = open(&db).unwrap();
    assert_eq!(again.get_meta("schema_version").unwrap().unwrap(), "2");
    assert_eq!(again.get_rule("legacy").unwrap().unwrap().sources, ["agent_observation"]);
}

#[test]
fn test_newer_schema_refuses_to_open() {
    let tmp = tempfile::tempdir().unwrap();
    let db = tmp.path().join("v99.db");
    rusqlite::Connection::open(&db)
        .unwrap()
        .execute_batch("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL); INSERT INTO meta VALUES ('schema_version', '99');")
        .unwrap();
    let err = open(&db).err().expect("a newer schema must refuse to open");
    assert_eq!(err.to_string(), "database schema 99 is newer than this code (2)");
}
