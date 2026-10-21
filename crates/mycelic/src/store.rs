//! SQLite persistence for Mycelic: agents, memories, lineage, the event log (which doubles as the transport
//! outbox), rules and the audit log.  The schema (version 2) and every encoding are the Python service's, so a
//! database written by either implementation opens in the other.
//!
//! Design:
//!
//! * One writer connection in WAL mode with `synchronous=FULL`.  Every write goes through
//!   [`Store::transaction`] (`BEGIN IMMEDIATE` ... `COMMIT`) under a mutex, so a multi-row write is atomic and
//!   writes never interleave.  FULL, not NORMAL: the consumer acks a JetStream message right after the commit that
//!   applied it, so the commit must survive an OS crash or power loss too, not only a process crash.
//! * Reads outside a transaction use a small pool of read-only connections (WAL readers never block the writer).
//!   Reads that must see a transaction's own uncommitted rows (the aggregator inside `apply`) go through the
//!   transaction's [`Tx`], which derefs to the same read API ([`Db`]).
//! * Writes that must be atomic together are composed by the service inside ONE transaction: "store the memory +
//!   append its event to the outbox", "mark the event applied + write the derived memories + lineage edges +
//!   append the derived events".  A crash between two of those steps therefore leaves nothing half-done.
//! * `events` is both the durable log of what this node accepted and the outbox: rows start `pending`, the
//!   publisher marks them `published` with their JetStream sequence, and the consumer marks them `applied`.
//!   Duplicate deliveries are rejected by primary key (`event_id`), which is also the JetStream `Nats-Msg-Id`.
//! * `revision` increases with every memory write, after the commit that made it, so the retrieval index never
//!   caches a view of rows that a concurrent reader could not yet see.

use crate::hierarchy::LAYERS;
use crate::models::{Agent, Clock, EventRecord, IdGen, LineageEdge, Memory, Rule};
use crate::pyjson::{dumps_sorted, float_repr};
use anyhow::{Context, Result, anyhow, bail};
use rusqlite::types::Value as SqlValue;
use rusqlite::{Connection, OptionalExtension, Row, params, params_from_iter};
use serde_json::{Map, Value, json};
use std::cell::Cell;
use std::collections::HashMap;
use std::ops::Deref;
use std::path::Path;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Mutex, MutexGuard};

pub const SCHEMA_VERSION: i64 = 2;
const READERS: usize = 4;

const DDL: &str = r#"
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS orgs (
    org_id     TEXT PRIMARY KEY,
    name       TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS agents (
    agent_id     TEXT PRIMARY KEY,
    org_id       TEXT NOT NULL REFERENCES orgs(org_id),
    display_name TEXT NOT NULL,
    path         TEXT NOT NULL UNIQUE,
    scopes       TEXT NOT NULL DEFAULT '[]',
    key_hash     TEXT NOT NULL,
    key_prefix   TEXT NOT NULL,
    status       TEXT NOT NULL DEFAULT 'active',
    created_at   TEXT NOT NULL,
    last_seen_at TEXT,
    metadata     TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_agents_org ON agents(org_id, status);

CREATE TABLE IF NOT EXISTS memories (
    rid               INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_id         TEXT NOT NULL UNIQUE,
    org_id            TEXT NOT NULL,
    layer             TEXT NOT NULL,
    scope             TEXT NOT NULL,
    text              TEXT NOT NULL,
    topic             TEXT,
    slot              TEXT,
    entity            TEXT,
    kind              TEXT NOT NULL,
    confidence        REAL NOT NULL,
    support           INTEGER NOT NULL DEFAULT 1,
    independent_teams INTEGER NOT NULL DEFAULT 1,
    producer_id       TEXT NOT NULL,
    operator          TEXT NOT NULL,
    rule_id           TEXT,
    agg_key           TEXT,
    event_id          TEXT,
    visibility        TEXT NOT NULL DEFAULT 'team',
    status            TEXT NOT NULL DEFAULT 'active',
    superseded_by     TEXT,
    created_at        TEXT NOT NULL,
    applied_at        TEXT,
    source_event_ids  TEXT NOT NULL DEFAULT '[]',
    local_ref         TEXT,
    metadata          TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_memories_org_status ON memories(org_id, status);
CREATE INDEX IF NOT EXISTS idx_memories_scope ON memories(org_id, scope);
CREATE INDEX IF NOT EXISTS idx_memories_topic ON memories(org_id, topic, status);
CREATE INDEX IF NOT EXISTS idx_memories_entity ON memories(org_id, entity, status);
CREATE INDEX IF NOT EXISTS idx_memories_agg ON memories(org_id, operator, scope, agg_key, status);
-- at most one *active* derived memory per (unit, operator, key): supersession is the only way to replace it
CREATE UNIQUE INDEX IF NOT EXISTS uq_memories_active_agg ON memories(org_id, operator, scope, agg_key)
    WHERE status='active' AND agg_key IS NOT NULL;

CREATE TABLE IF NOT EXISTS lineage_edges (
    child_id       TEXT NOT NULL,
    parent_id      TEXT NOT NULL,
    contributed_by TEXT NOT NULL,
    parent_layer   TEXT NOT NULL,
    created_at     TEXT NOT NULL,
    PRIMARY KEY (child_id, parent_id)
);
CREATE INDEX IF NOT EXISTS idx_lineage_parent ON lineage_edges(parent_id);

CREATE TABLE IF NOT EXISTS events (
    rid          INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id     TEXT NOT NULL UNIQUE,
    kind         TEXT NOT NULL,
    org_id       TEXT NOT NULL,
    agent_id     TEXT,
    subject      TEXT NOT NULL,
    payload      TEXT NOT NULL,
    status       TEXT NOT NULL DEFAULT 'pending',
    js_seq       INTEGER,
    attempts     INTEGER NOT NULL DEFAULT 0,
    last_error   TEXT,
    created_at   TEXT NOT NULL,
    published_at TEXT,
    applied_at   TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_status ON events(status, rid);
CREATE INDEX IF NOT EXISTS idx_events_org ON events(org_id, kind, rid);

CREATE TABLE IF NOT EXISTS rules (
    rule_id        TEXT PRIMARY KEY,
    org_id         TEXT,
    target_layer   TEXT NOT NULL,
    required_slots TEXT NOT NULL,
    conclusion     TEXT NOT NULL,
    topic_prefix   TEXT,
    min_agents     INTEGER NOT NULL DEFAULT 2,
    min_teams      INTEGER NOT NULL DEFAULT 1,
    kind           TEXT NOT NULL DEFAULT 'risk',
    enabled        INTEGER NOT NULL DEFAULT 1,
    metadata       TEXT NOT NULL DEFAULT '{}',
    updated_at     TEXT NOT NULL,
    sources        TEXT NOT NULL DEFAULT '["agent_observation"]',
    emits_slot     TEXT,
    emits_topic    TEXT,
    min_units      TEXT NOT NULL DEFAULT '{}',
    corroborate    INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS audit_log (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    at        TEXT NOT NULL,
    principal TEXT NOT NULL,
    action    TEXT NOT NULL,
    target    TEXT,
    detail    TEXT NOT NULL DEFAULT '{}',
    remote    TEXT
);
CREATE INDEX IF NOT EXISTS idx_audit_at ON audit_log(at);
"#;

/// `json.dumps(v if v is not None else {}, ensure_ascii=False, sort_keys=True)`: how JSON columns are written.
pub fn j(v: &Value) -> String {
    if v.is_null() { "{}".to_string() } else { dumps_sorted(v) }
}

/// `_jl`: a JSON column read back, or `default` when it is empty or not JSON.
fn jl(s: Option<String>, default: Value) -> Value {
    match s {
        Some(s) if !s.is_empty() => serde_json::from_str(&s).unwrap_or(default),
        _ => default,
    }
}

/// A column as text whatever SQLite stored in it (a database written by another implementation may hold an
/// integer where text was meant).
fn text(r: &Row, name: &str) -> rusqlite::Result<Option<String>> {
    Ok(match r.get::<_, SqlValue>(name)? {
        SqlValue::Null => None,
        SqlValue::Text(s) => Some(s),
        SqlValue::Integer(i) => Some(i.to_string()),
        SqlValue::Real(f) => Some(float_repr(f)),
        SqlValue::Blob(b) => Some(String::from_utf8_lossy(&b).into_owned()),
    })
}

fn text_req(r: &Row, name: &str) -> rusqlite::Result<String> {
    Ok(text(r, name)?.unwrap_or_default())
}

fn real(r: &Row, name: &str) -> rusqlite::Result<f64> {
    Ok(match r.get::<_, SqlValue>(name)? {
        SqlValue::Real(f) => f,
        SqlValue::Integer(i) => i as f64,
        SqlValue::Text(s) => s.trim().parse().unwrap_or(0.0),
        _ => 0.0,
    })
}

fn int(r: &Row, name: &str) -> rusqlite::Result<i64> {
    Ok(match r.get::<_, SqlValue>(name)? {
        SqlValue::Integer(i) => i,
        SqlValue::Real(f) => f as i64,
        SqlValue::Text(s) => s.trim().parse().unwrap_or(0),
        _ => 0,
    })
}

fn str_list(v: Value) -> Vec<String> {
    match v {
        Value::Array(items) => items
            .into_iter()
            .map(|x| match x {
                Value::String(s) => s,
                other => crate::pyjson::dumps(&other),
            })
            .collect(),
        _ => Vec::new(),
    }
}

fn obj(v: Value) -> Map<String, Value> {
    match v {
        Value::Object(m) => m,
        _ => Map::new(),
    }
}

pub fn row_memory(r: &Row) -> rusqlite::Result<Memory> {
    Ok(Memory {
        memory_id: text_req(r, "memory_id")?,
        org_id: text_req(r, "org_id")?,
        layer: text_req(r, "layer")?,
        scope: text_req(r, "scope")?,
        text: text_req(r, "text")?,
        topic: text(r, "topic")?,
        slot: text(r, "slot")?,
        entity: text(r, "entity")?,
        kind: text_req(r, "kind")?,
        confidence: real(r, "confidence")?,
        support: int(r, "support")?,
        independent_teams: int(r, "independent_teams")?,
        producer_id: text_req(r, "producer_id")?,
        operator: text_req(r, "operator")?,
        rule_id: text(r, "rule_id")?,
        event_id: text(r, "event_id")?,
        visibility: text_req(r, "visibility")?,
        status: text_req(r, "status")?,
        superseded_by: text(r, "superseded_by")?,
        created_at: text_req(r, "created_at")?,
        applied_at: text(r, "applied_at")?,
        source_event_ids: str_list(jl(text(r, "source_event_ids")?, json!([]))),
        local_ref: text(r, "local_ref")?,
        metadata: obj(jl(text(r, "metadata")?, json!({}))),
    })
}

pub fn row_agent(r: &Row) -> rusqlite::Result<Agent> {
    Ok(Agent {
        agent_id: text_req(r, "agent_id")?,
        org_id: text_req(r, "org_id")?,
        display_name: text_req(r, "display_name")?,
        path: text_req(r, "path")?,
        scopes: str_list(jl(text(r, "scopes")?, json!([]))),
        key_prefix: text_req(r, "key_prefix")?,
        status: text_req(r, "status")?,
        created_at: text_req(r, "created_at")?,
        last_seen_at: text(r, "last_seen_at")?,
        metadata: obj(jl(text(r, "metadata")?, json!({}))),
    })
}

pub fn row_event(r: &Row) -> rusqlite::Result<EventRecord> {
    Ok(EventRecord {
        event_id: text_req(r, "event_id")?,
        kind: text_req(r, "kind")?,
        org_id: text_req(r, "org_id")?,
        agent_id: text(r, "agent_id")?,
        subject: text_req(r, "subject")?,
        payload: jl(text(r, "payload")?, json!({})),
        status: text_req(r, "status")?,
        js_seq: r.get::<_, Option<i64>>("js_seq").ok().flatten(),
        attempts: int(r, "attempts")?,
        last_error: text(r, "last_error")?,
        created_at: text_req(r, "created_at")?,
        published_at: text(r, "published_at")?,
        applied_at: text(r, "applied_at")?,
    })
}

/// `int(n)` the way the Python row reader coerced counts in `min_units`.
fn coerce_int(v: &Value) -> Option<i64> {
    match v {
        Value::Number(n) => n.as_i64().or_else(|| n.as_f64().map(|f| f.trunc() as i64)),
        Value::Bool(b) => Some(*b as i64),
        Value::String(s) => s.trim().parse().ok(),
        _ => None,
    }
}

pub fn parse_min_units(v: &Value) -> Vec<(String, Vec<(String, i64)>)> {
    let mut out = Vec::new();
    if let Value::Object(m) = v {
        for (slot, per) in m {
            if let Value::Object(per) = per {
                out.push((
                    slot.clone(),
                    per.iter().map(|(layer, n)| (layer.clone(), coerce_int(n).unwrap_or(0))).collect(),
                ));
            }
        }
    }
    out
}

pub fn row_rule(r: &Row) -> rusqlite::Result<Rule> {
    Ok(Rule {
        rule_id: text_req(r, "rule_id")?,
        org_id: text(r, "org_id")?,
        target_layer: text_req(r, "target_layer")?,
        required_slots: str_list(jl(text(r, "required_slots")?, json!([]))),
        conclusion: text_req(r, "conclusion")?,
        topic_prefix: text(r, "topic_prefix")?,
        min_agents: int(r, "min_agents")?,
        min_teams: int(r, "min_teams")?,
        kind: text_req(r, "kind")?,
        enabled: int(r, "enabled")? != 0,
        metadata: obj(jl(text(r, "metadata")?, json!({}))),
        sources: str_list(jl(text(r, "sources")?, json!(["agent_observation"]))),
        emits_slot: text(r, "emits_slot")?,
        emits_topic: text(r, "emits_topic")?,
        min_units: parse_min_units(&jl(text(r, "min_units")?, json!({}))),
        corroborate: int(r, "corroborate")? != 0,
    })
}

pub fn row_edge(r: &Row) -> rusqlite::Result<LineageEdge> {
    Ok(LineageEdge {
        child_id: text_req(r, "child_id")?,
        parent_id: text_req(r, "parent_id")?,
        contributed_by: text_req(r, "contributed_by")?,
        parent_layer: text_req(r, "parent_layer")?,
        created_at: text_req(r, "created_at")?,
    })
}

fn chars(s: &str) -> i64 {
    s.chars().count() as i64
}

fn placeholders(n: usize) -> String {
    vec!["?"; n].join(",")
}

/// Filters for [`Db::list_memories`]; the defaults are the Python keyword defaults.
#[derive(Clone, Debug)]
pub struct MemoryFilter<'a> {
    pub scope: Option<&'a str>,
    pub layers: Option<&'a [String]>,
    pub status: Option<&'a str>,
    pub topic: Option<&'a str>,
    pub entity: Option<&'a str>,
    pub slot: Option<&'a str>,
    pub operator: Option<&'a str>,
    pub producer_id: Option<&'a str>,
    pub since: Option<&'a str>,
    pub limit: i64,
    pub newest_first: bool,
    /// only what the consumer has applied: aggregation must not see ahead
    pub applied_only: bool,
    pub operators: Option<&'a [String]>,
}

impl Default for MemoryFilter<'_> {
    fn default() -> Self {
        MemoryFilter {
            scope: None,
            layers: None,
            status: Some("active"),
            topic: None,
            entity: None,
            slot: None,
            operator: None,
            producer_id: None,
            since: None,
            limit: 200,
            newest_first: true,
            applied_only: false,
            operators: None,
        }
    }
}

/// The read API, over any connection: a pooled reader, or the writer inside a transaction.
pub struct Db<'c> {
    pub c: &'c Connection,
}

impl Db<'_> {
    fn query_rows<T>(&self, sql: &str, args: Vec<SqlValue>, f: fn(&Row) -> rusqlite::Result<T>) -> Result<Vec<T>> {
        let mut stmt = self.c.prepare_cached(sql)?;
        let rows = stmt.query_map(params_from_iter(args), f)?;
        Ok(rows.collect::<rusqlite::Result<Vec<T>>>()?)
    }

    pub fn get_meta(&self, key: &str) -> Result<Option<String>> {
        Ok(self.c.prepare_cached("SELECT value FROM meta WHERE key=?")?.query_row([key], |r| r.get(0)).optional()?)
    }

    pub fn get_agent(&self, agent_id: &str) -> Result<Option<Agent>> {
        Ok(self
            .c
            .prepare_cached("SELECT * FROM agents WHERE agent_id=?")?
            .query_row([agent_id], row_agent)
            .optional()?)
    }

    pub fn get_agent_credentials(&self, agent_id: &str) -> Result<Option<(Agent, String)>> {
        Ok(self
            .c
            .prepare_cached("SELECT * FROM agents WHERE agent_id=?")?
            .query_row([agent_id], |r| Ok((row_agent(r)?, text_req(r, "key_hash")?)))
            .optional()?)
    }

    pub fn list_agents(&self, org_id: Option<&str>, include_revoked: bool) -> Result<Vec<Agent>> {
        let mut sql = "SELECT * FROM agents WHERE 1=1".to_string();
        let mut args = Vec::new();
        if let Some(o) = org_id.filter(|o| !o.is_empty()) {
            sql.push_str(" AND org_id=?");
            args.push(SqlValue::Text(o.into()));
        }
        if !include_revoked {
            sql.push_str(" AND status='active'");
        }
        sql.push_str(" ORDER BY path");
        self.query_rows(&sql, args, row_agent)
    }

    /// Direct child units of `unit` according to the registered (active) agents' paths.
    pub fn child_units(&self, org_id: &str, unit: &str) -> Result<Vec<String>> {
        let depth = unit.matches('/').count() + 2;
        let mut stmt = self.c.prepare_cached(
            "SELECT path FROM agents WHERE org_id=? AND status='active' AND (path=? OR substr(path, 1, ?)=?)",
        )?;
        let prefix = format!("{unit}/");
        let paths = stmt
            .query_map(params![org_id, unit, chars(unit) + 1, prefix], |r| text_req(r, "path"))?
            .collect::<rusqlite::Result<Vec<String>>>()?;
        let mut out: Vec<String> = paths
            .iter()
            .filter(|p| p.matches('/').count() + 1 >= depth)
            .map(|p| p.split('/').take(depth).collect::<Vec<_>>().join("/"))
            .collect();
        out.sort();
        out.dedup();
        Ok(out)
    }

    pub fn list_orgs(&self) -> Result<Vec<Value>> {
        self.query_rows("SELECT * FROM orgs ORDER BY org_id", vec![], |r| {
            Ok(json!({"org_id": text_req(r, "org_id")?, "name": text_req(r, "name")?, "created_at": text_req(r, "created_at")?}))
        })
    }

    pub fn get_memory(&self, memory_id: &str) -> Result<Option<Memory>> {
        Ok(self
            .c
            .prepare_cached("SELECT * FROM memories WHERE memory_id=?")?
            .query_row([memory_id], row_memory)
            .optional()?)
    }

    pub fn memory_exists(&self, memory_id: &str) -> Result<bool> {
        Ok(self.c.prepare_cached("SELECT 1 FROM memories WHERE memory_id=?")?.exists([memory_id])?)
    }

    pub fn list_memories(&self, org_id: &str, f: &MemoryFilter) -> Result<Vec<Memory>> {
        let mut sql = "SELECT * FROM memories WHERE org_id=?".to_string();
        let mut args = vec![SqlValue::Text(org_id.into())];
        if let Some(ops) = f.operators.filter(|o| !o.is_empty()) {
            sql.push_str(&format!(" AND operator IN ({})", placeholders(ops.len())));
            args.extend(ops.iter().map(|o| SqlValue::Text(o.clone())));
        }
        if f.applied_only {
            sql.push_str(" AND applied_at IS NOT NULL");
        }
        if let Some(scope) = f.scope.filter(|s| !s.is_empty()) {
            sql.push_str(" AND (scope=? OR substr(scope, 1, ?)=?)");
            args.extend([
                SqlValue::Text(scope.into()),
                SqlValue::Integer(chars(scope) + 1),
                SqlValue::Text(format!("{scope}/")),
            ]);
        }
        if let Some(layers) = f.layers.filter(|l| !l.is_empty()) {
            sql.push_str(&format!(" AND layer IN ({})", placeholders(layers.len())));
            args.extend(layers.iter().map(|l| SqlValue::Text(l.clone())));
        }
        for (col, v) in [
            ("status", f.status),
            ("topic", f.topic),
            ("entity", f.entity),
            ("slot", f.slot),
            ("operator", f.operator),
            ("producer_id", f.producer_id),
        ] {
            if let Some(v) = v.filter(|v| !v.is_empty()) {
                sql.push_str(&format!(" AND {col}=?"));
                args.push(SqlValue::Text(v.into()));
            }
        }
        if let Some(since) = f.since.filter(|s| !s.is_empty()) {
            sql.push_str(" AND created_at>=?");
            args.push(SqlValue::Text(since.into()));
        }
        sql.push_str(if f.newest_first { " ORDER BY rid DESC LIMIT ?" } else { " ORDER BY rid ASC LIMIT ?" });
        args.push(SqlValue::Integer(f.limit));
        self.query_rows(&sql, args, row_memory)
    }

    pub fn current_derived(&self, org_id: &str, operator: &str, scope: &str, agg_key: &str) -> Result<Option<Memory>> {
        Ok(self
            .c
            .prepare_cached(
                "SELECT * FROM memories WHERE org_id=? AND operator=? AND scope=? AND agg_key=? AND status='active' ORDER BY rid DESC LIMIT 1",
            )?
            .query_row(params![org_id, operator, scope, agg_key], row_memory)
            .optional()?)
    }

    /// Memories a principal may read, optionally restricted to a unit subtree.
    ///
    /// `agent_path`/`team_path` None means an administrator (everything in the org).  For an agent: agent-layer
    /// memories of its own team (or any with `visibility='org'`), and derived memories of every unit the agent
    /// belongs to (the memory's scope is an ancestor-or-self of the agent's path).  Prefix tests use `substr`
    /// rather than `LIKE` because `_` is a LIKE wildcard and a legal path character.  This is the SQL twin of
    /// `auth::memory_visible`.
    pub fn visible_rows(
        &self,
        org_id: &str,
        agent_path: Option<&str>,
        team_path: Option<&str>,
        scope: Option<&str>,
        min_layer_index: usize,
        statuses: &[&str],
    ) -> Result<Vec<Memory>> {
        let mut sql = format!("SELECT * FROM memories WHERE org_id=? AND status IN ({})", placeholders(statuses.len()));
        let mut args = vec![SqlValue::Text(org_id.into())];
        args.extend(statuses.iter().map(|s| SqlValue::Text((*s).into())));
        if let (Some(agent_path), Some(team_path)) = (agent_path, team_path) {
            sql.push_str(" AND ((layer='agent' AND (visibility='org' OR substr(scope, 1, ?)=?))");
            sql.push_str(" OR (layer!='agent' AND (scope=? OR substr(?, 1, length(scope)+1)=scope||'/')))");
            args.extend([
                SqlValue::Integer(chars(team_path) + 1),
                SqlValue::Text(format!("{team_path}/")),
                SqlValue::Text(agent_path.into()),
                SqlValue::Text(agent_path.into()),
            ]);
        }
        if let Some(scope) = scope.filter(|s| !s.is_empty()) {
            sql.push_str(" AND (scope=? OR substr(scope, 1, ?)=?)");
            args.extend([
                SqlValue::Text(scope.into()),
                SqlValue::Integer(chars(scope) + 1),
                SqlValue::Text(format!("{scope}/")),
            ]);
        }
        if min_layer_index > 0 {
            let ls = &LAYERS[min_layer_index.min(LAYERS.len())..];
            sql.push_str(&format!(" AND layer IN ({})", placeholders(ls.len())));
            args.extend(ls.iter().map(|l| SqlValue::Text((*l).into())));
        }
        sql.push_str(" ORDER BY rid");
        self.query_rows(&sql, args, row_memory)
    }

    pub fn parents_of(&self, child_id: &str) -> Result<Vec<LineageEdge>> {
        self.query_rows(
            "SELECT * FROM lineage_edges WHERE child_id=? ORDER BY parent_id",
            vec![SqlValue::Text(child_id.into())],
            row_edge,
        )
    }

    pub fn children_of(&self, parent_id: &str) -> Result<Vec<LineageEdge>> {
        self.query_rows(
            "SELECT * FROM lineage_edges WHERE parent_id=? ORDER BY child_id",
            vec![SqlValue::Text(parent_id.into())],
            row_edge,
        )
    }

    /// Every memory that (transitively) derives from `memory_id`, nearest first.
    pub fn dependents_of(&self, memory_id: &str, max_nodes: usize) -> Result<Vec<String>> {
        let mut seen = Vec::new();
        let mut frontier = vec![memory_id.to_string()];
        let mut visited = std::collections::HashSet::from([memory_id.to_string()]);
        while !frontier.is_empty() && seen.len() < max_nodes {
            let mut next = Vec::new();
            for pid in &frontier {
                for e in self.children_of(pid)? {
                    if visited.insert(e.child_id.clone()) {
                        seen.push(e.child_id.clone());
                        next.push(e.child_id);
                    }
                }
            }
            frontier = next;
        }
        Ok(seen)
    }

    pub fn get_event(&self, event_id: &str) -> Result<Option<EventRecord>> {
        Ok(self
            .c
            .prepare_cached("SELECT * FROM events WHERE event_id=?")?
            .query_row([event_id], row_event)
            .optional()?)
    }

    pub fn get_events(&self, ids: &[String]) -> Result<HashMap<String, EventRecord>> {
        let mut unique: Vec<&String> = Vec::new();
        for id in ids {
            if !unique.contains(&id) {
                unique.push(id);
            }
        }
        let mut out = HashMap::new();
        for chunk in unique.chunks(500) {
            let sql = format!("SELECT * FROM events WHERE event_id IN ({})", placeholders(chunk.len()));
            for e in self.query_rows(&sql, chunk.iter().map(|s| SqlValue::Text((*s).clone())).collect(), row_event)? {
                out.insert(e.event_id.clone(), e);
            }
        }
        Ok(out)
    }

    pub fn event_status(&self, event_id: &str) -> Result<Option<String>> {
        Ok(self
            .c
            .prepare_cached("SELECT status FROM events WHERE event_id=?")?
            .query_row([event_id], |r| text_req(r, "status"))
            .optional()?)
    }

    pub fn pending_events(&self, limit: i64) -> Result<Vec<EventRecord>> {
        self.query_rows(
            "SELECT * FROM events WHERE status='pending' ORDER BY rid LIMIT ?",
            vec![SqlValue::Integer(limit)],
            row_event,
        )
    }

    pub fn list_events(
        &self,
        org_id: &str,
        agent_id: Option<&str>,
        kind: Option<&str>,
        status: Option<&str>,
        limit: i64,
    ) -> Result<Vec<EventRecord>> {
        let mut sql = "SELECT * FROM events WHERE org_id=?".to_string();
        let mut args = vec![SqlValue::Text(org_id.into())];
        for (col, v) in [("agent_id", agent_id), ("kind", kind), ("status", status)] {
            if let Some(v) = v.filter(|v| !v.is_empty()) {
                sql.push_str(&format!(" AND {col}=?"));
                args.push(SqlValue::Text(v.into()));
            }
        }
        sql.push_str(" ORDER BY rid DESC LIMIT ?");
        args.push(SqlValue::Integer(limit));
        self.query_rows(&sql, args, row_event)
    }

    /// Events by status, in SQLite's group order (status name).
    pub fn event_counts(&self) -> Result<Vec<(String, i64)>> {
        self.query_rows("SELECT status, COUNT(*) AS n FROM events GROUP BY status", vec![], |r| {
            Ok((text_req(r, "status")?, int(r, "n")?))
        })
    }

    pub fn max_applied_seq(&self) -> Result<i64> {
        Ok(self.c.query_row("SELECT COALESCE(MAX(js_seq), 0) AS s FROM events WHERE status='applied'", [], |r| {
            int(r, "s")
        })?)
    }

    pub fn list_rules(&self, org_id: Option<&str>, enabled_only: bool) -> Result<Vec<Rule>> {
        let mut sql = "SELECT * FROM rules WHERE 1=1".to_string();
        let mut args = Vec::new();
        if let Some(o) = org_id {
            sql.push_str(" AND (org_id IS NULL OR org_id=?)");
            args.push(SqlValue::Text(o.into()));
        }
        if enabled_only {
            sql.push_str(" AND enabled=1");
        }
        sql.push_str(" ORDER BY rule_id");
        self.query_rows(&sql, args, row_rule)
    }

    pub fn get_rule(&self, rule_id: &str) -> Result<Option<Rule>> {
        Ok(self.c.prepare_cached("SELECT * FROM rules WHERE rule_id=?")?.query_row([rule_id], row_rule).optional()?)
    }

    pub fn recent_audit(&self, limit: i64, org_id: Option<&str>) -> Result<Vec<Value>> {
        let rows =
            self.query_rows("SELECT * FROM audit_log ORDER BY id DESC LIMIT ?", vec![SqlValue::Integer(limit)], |r| {
                Ok(json!({
                    "id": int(r, "id")?, "at": text_req(r, "at")?, "principal": text_req(r, "principal")?,
                    "action": text_req(r, "action")?, "target": text(r, "target")?,
                    "detail": jl(text(r, "detail")?, json!({})), "remote": text(r, "remote")?,
                }))
            })?;
        Ok(rows
            .into_iter()
            .filter(|d| org_id.is_none() || d["detail"].get("org_id").and_then(Value::as_str) == org_id)
            .collect())
    }

    pub fn memories_by_layer(&self, org_id: Option<&str>) -> Result<Map<String, Value>> {
        let mut by_layer = Map::new();
        for l in LAYERS {
            by_layer.insert(l.into(), json!(0));
        }
        let mut sql = "SELECT layer, COUNT(*) AS n FROM memories WHERE status='active'".to_string();
        let mut args = Vec::new();
        if let Some(o) = org_id {
            sql.push_str(" AND org_id=?");
            args.push(SqlValue::Text(o.into()));
        }
        sql.push_str(" GROUP BY layer");
        for (layer, n) in self.query_rows(&sql, args, |r| Ok((text_req(r, "layer")?, int(r, "n")?)))? {
            by_layer.insert(layer, json!(n));
        }
        Ok(by_layer)
    }

    fn count(&self, sql: &str, args: Vec<SqlValue>) -> Result<i64> {
        Ok(self.c.prepare_cached(sql)?.query_row(params_from_iter(args), |r| int(r, "n"))?)
    }

    /// The counters `/health`, `/admin/status` and the metrics gauges report.  "Active" agents are those seen in
    /// the last 15 minutes of wall-clock time.
    pub fn stats(&self, revision: u64) -> Result<Value> {
        let by_layer = self.memories_by_layer(None)?;
        let mut by_status = Map::new();
        for (s, n) in self.query_rows("SELECT status, COUNT(*) AS n FROM memories GROUP BY status", vec![], |r| {
            Ok((text_req(r, "status")?, int(r, "n")?))
        })? {
            by_status.insert(s, json!(n));
        }
        let cutoff = (chrono::Utc::now() - chrono::Duration::minutes(15)).format("%Y-%m-%dT%H:%M:%S+00:00").to_string();
        let active = self.count(
            "SELECT COUNT(*) AS n FROM agents WHERE status='active' AND last_seen_at>=?",
            vec![SqlValue::Text(cutoff)],
        )?;
        let registered = self.count("SELECT COUNT(*) AS n FROM agents WHERE status='active'", vec![])?;
        let mut events = Map::new();
        let mut pending = 0;
        for (s, n) in self.event_counts()? {
            if s == "pending" {
                pending = n;
            }
            events.insert(s, json!(n));
        }
        Ok(json!({
            "memories_by_layer": by_layer,
            "memories_by_status": by_status,
            "lineage_edges": self.count("SELECT COUNT(*) AS n FROM lineage_edges", vec![])?,
            "events": events,
            "outbox_pending": pending,
            "registered_agents": registered,
            "active_agents": active,
            "orgs": self.count("SELECT COUNT(*) AS n FROM orgs", vec![])?,
            "rules": self.count("SELECT COUNT(*) AS n FROM rules WHERE enabled=1", vec![])?,
            "last_applied_seq": self.max_applied_seq()?,
            "revision": revision,
        }))
    }
}

/// Write handle valid inside [`Store::transaction`]; derefs to the read API over the same connection, so the
/// aggregator reads what the transaction has written so far.
pub struct Tx<'a> {
    db: Db<'a>,
    store: &'a Store,
    bumps: Cell<u64>,
}

impl<'a> Deref for Tx<'a> {
    type Target = Db<'a>;
    fn deref(&self) -> &Db<'a> {
        &self.db
    }
}

impl Tx<'_> {
    pub fn now(&self) -> String {
        (self.store.clock)()
    }

    pub fn new_id(&self, prefix: &str) -> String {
        (self.store.ids)(prefix)
    }

    fn bump(&self) {
        self.bumps.set(self.bumps.get() + 1);
    }

    pub fn execute(&self, sql: &str, args: &[&dyn rusqlite::ToSql]) -> Result<usize> {
        Ok(self.db.c.prepare_cached(sql)?.execute(args)?)
    }

    // ---- agents
    pub fn ensure_org(&self, org_id: &str, name: Option<&str>) -> Result<()> {
        self.execute(
            "INSERT OR IGNORE INTO orgs(org_id, name, created_at) VALUES (?, ?, ?)",
            &[&org_id, &name.unwrap_or(org_id), &self.now()],
        )?;
        Ok(())
    }

    pub fn insert_agent(&self, agent: &Agent, key_hash: &str) -> Result<()> {
        self.ensure_org(&agent.org_id, None)?;
        let created = if agent.created_at.is_empty() { self.now() } else { agent.created_at.clone() };
        self.execute(
            "INSERT INTO agents(agent_id, org_id, display_name, path, scopes, key_hash, key_prefix, status, created_at, last_seen_at, metadata)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            &[
                &agent.agent_id,
                &agent.org_id,
                &agent.display_name,
                &agent.path,
                &j(&json!(agent.scopes)),
                &key_hash,
                &agent.key_prefix,
                &agent.status,
                &created,
                &agent.last_seen_at,
                &j(&Value::Object(agent.metadata.clone())),
            ],
        )?;
        Ok(())
    }

    pub fn set_agent_status(&self, agent_id: &str, status: &str) -> Result<bool> {
        Ok(self.execute("UPDATE agents SET status=? WHERE agent_id=?", &[&status, &agent_id])? > 0)
    }

    pub fn rotate_agent_key(&self, agent_id: &str, key_hash: &str, key_prefix: &str) -> Result<bool> {
        Ok(self.execute(
            "UPDATE agents SET key_hash=?, key_prefix=? WHERE agent_id=?",
            &[&key_hash, &key_prefix, &agent_id],
        )? > 0)
    }

    pub fn touch_agent(&self, agent_id: &str, at: &str) -> Result<()> {
        self.execute("UPDATE agents SET last_seen_at=? WHERE agent_id=?", &[&at, &agent_id])?;
        Ok(())
    }

    // ---- memories
    /// Insert if absent.  Returns false when the id already exists (idempotent replay).
    pub fn insert_memory(&self, m: &Memory) -> Result<bool> {
        if self.memory_exists(&m.memory_id)? {
            return Ok(false);
        }
        let agg_key: Option<String> = if m.operator != "agent_observation" {
            m.metadata.get("agg_key").and_then(|v| match v {
                Value::Null => None,
                Value::String(s) => Some(s.clone()),
                other => Some(crate::pyjson::dumps(other)),
            })
        } else {
            None
        };
        self.execute(
            "INSERT INTO memories(memory_id, org_id, layer, scope, text, topic, slot, entity, kind, confidence,
                                  support, independent_teams, producer_id, operator, rule_id, agg_key, event_id,
                                  visibility, status, superseded_by, created_at, applied_at, source_event_ids,
                                  local_ref, metadata)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            &[
                &m.memory_id,
                &m.org_id,
                &m.layer,
                &m.scope,
                &m.text,
                &m.topic,
                &m.slot,
                &m.entity,
                &m.kind,
                &m.confidence,
                &m.support,
                &m.independent_teams,
                &m.producer_id,
                &m.operator,
                &m.rule_id,
                &agg_key,
                &m.event_id,
                &m.visibility,
                &m.status,
                &m.superseded_by,
                &m.created_at,
                &m.applied_at,
                &j(&json!(m.source_event_ids)),
                &m.local_ref,
                &j(&Value::Object(m.metadata.clone())),
            ],
        )?;
        self.bump();
        Ok(true)
    }

    pub fn set_memory_status(
        &self,
        memory_id: &str,
        status: &str,
        superseded_by: Option<&str>,
        reason: Option<&str>,
    ) -> Result<bool> {
        let row: Option<Option<String>> = self
            .db
            .c
            .prepare_cached("SELECT metadata FROM memories WHERE memory_id=?")?
            .query_row([memory_id], |r| text(r, "metadata"))
            .optional()?;
        let Some(meta_text) = row else { return Ok(false) };
        let mut meta = obj(jl(meta_text, json!({})));
        if let Some(reason) = reason.filter(|r| !r.is_empty()) {
            meta.insert("status_reason".into(), json!(reason));
        }
        self.execute(
            "UPDATE memories SET status=?, superseded_by=COALESCE(?, superseded_by), metadata=? WHERE memory_id=?",
            &[&status, &superseded_by, &j(&Value::Object(meta)), &memory_id],
        )?;
        self.bump();
        Ok(true)
    }

    /// A derived memory whose exact coalition returns (after a retraction) becomes the active version again.
    pub fn reactivate_memory(&self, memory_id: &str, applied_at: &str, metadata: &Map<String, Value>) -> Result<()> {
        self.execute(
            "UPDATE memories SET status='active', superseded_by=NULL, applied_at=?, metadata=? WHERE memory_id=?",
            &[&applied_at, &j(&Value::Object(metadata.clone())), &memory_id],
        )?;
        self.bump();
        Ok(())
    }

    pub fn set_applied(&self, memory_id: &str, applied_at: &str) -> Result<()> {
        self.execute(
            "UPDATE memories SET applied_at=COALESCE(applied_at, ?) WHERE memory_id=?",
            &[&applied_at, &memory_id],
        )?;
        Ok(())
    }

    pub fn set_memory_event(&self, memory_id: &str, event_id: &str) -> Result<()> {
        self.execute("UPDATE memories SET event_id=? WHERE memory_id=?", &[&event_id, &memory_id])?;
        Ok(())
    }

    pub fn add_lineage_edges(&self, edges: &[LineageEdge]) -> Result<()> {
        let mut stmt = self.db.c.prepare_cached(
            "INSERT OR IGNORE INTO lineage_edges(child_id, parent_id, contributed_by, parent_layer, created_at) VALUES (?, ?, ?, ?, ?)",
        )?;
        for e in edges {
            stmt.execute(params![e.child_id, e.parent_id, e.contributed_by, e.parent_layer, e.created_at])?;
        }
        Ok(())
    }

    // ---- events / outbox
    /// Append to the log/outbox.  Returns false if the event id is already known.
    pub fn insert_event(&self, e: &EventRecord) -> Result<bool> {
        if self.event_status(&e.event_id)?.is_some() {
            return Ok(false);
        }
        let created = if e.created_at.is_empty() { self.now() } else { e.created_at.clone() };
        self.execute(
            "INSERT INTO events(event_id, kind, org_id, agent_id, subject, payload, status, js_seq, attempts,
                                last_error, created_at, published_at, applied_at)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            &[
                &e.event_id,
                &e.kind,
                &e.org_id,
                &e.agent_id,
                &e.subject,
                &j(&e.payload),
                &e.status,
                &e.js_seq,
                &e.attempts,
                &e.last_error,
                &created,
                &e.published_at,
                &e.applied_at,
            ],
        )?;
        Ok(true)
    }

    /// An event delivered from the stream is by definition published, whatever the publisher has recorded so far.
    pub fn mark_applied(&self, event_id: &str, js_seq: Option<i64>, applied_at: &str) -> Result<()> {
        self.execute(
            "UPDATE events SET status='applied', applied_at=?, js_seq=COALESCE(?, js_seq),
                    published_at=COALESCE(published_at, CASE WHEN ? IS NOT NULL THEN ? ELSE NULL END) WHERE event_id=?",
            &[&applied_at, &js_seq, &js_seq, &applied_at, &event_id],
        )?;
        Ok(())
    }

    pub fn mark_published(&self, event_id: &str, js_seq: Option<i64>) -> Result<()> {
        self.execute(
            "UPDATE events SET status=CASE WHEN status='pending' THEN 'published' ELSE status END,
                    published_at=COALESCE(published_at, ?), js_seq=COALESCE(js_seq, ?) WHERE event_id=?",
            &[&self.now(), &js_seq, &event_id],
        )?;
        Ok(())
    }

    pub fn mark_publish_failed(&self, event_id: &str, error: &str) -> Result<()> {
        let e: String = error.chars().take(500).collect();
        self.execute("UPDATE events SET attempts=attempts+1, last_error=? WHERE event_id=?", &[&e, &event_id])?;
        Ok(())
    }

    pub fn mark_failed(&self, event_id: &str, error: &str) -> Result<()> {
        let e: String = error.chars().take(500).collect();
        self.execute(
            "UPDATE events SET status='failed', attempts=attempts+1, last_error=? WHERE event_id=?",
            &[&e, &event_id],
        )?;
        Ok(())
    }

    // ---- rules
    pub fn upsert_rule(&self, rule: &Rule) -> Result<()> {
        self.execute(
            "INSERT INTO rules(rule_id, org_id, target_layer, required_slots, conclusion, topic_prefix, min_agents,
                               min_teams, kind, enabled, metadata, updated_at, sources, emits_slot, emits_topic,
                               min_units, corroborate)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
             ON CONFLICT(rule_id) DO UPDATE SET org_id=excluded.org_id, target_layer=excluded.target_layer,
               required_slots=excluded.required_slots, conclusion=excluded.conclusion, topic_prefix=excluded.topic_prefix,
               min_agents=excluded.min_agents, min_teams=excluded.min_teams, kind=excluded.kind, enabled=excluded.enabled,
               metadata=excluded.metadata, updated_at=excluded.updated_at, sources=excluded.sources,
               emits_slot=excluded.emits_slot, emits_topic=excluded.emits_topic, min_units=excluded.min_units,
               corroborate=excluded.corroborate",
            &[
                &rule.rule_id,
                &rule.org_id,
                &rule.target_layer,
                &j(&json!(rule.required_slots)),
                &rule.conclusion,
                &rule.topic_prefix,
                &rule.min_agents,
                &rule.min_teams,
                &rule.kind,
                &(rule.enabled as i64),
                &j(&Value::Object(rule.metadata.clone())),
                &self.now(),
                &j(&json!(rule.sources)),
                &rule.emits_slot,
                &rule.emits_topic,
                &j(&rule.min_units_json()),
                &(rule.corroborate as i64),
            ],
        )?;
        Ok(())
    }

    pub fn delete_rule(&self, rule_id: &str) -> Result<bool> {
        Ok(self.execute("DELETE FROM rules WHERE rule_id=?", &[&rule_id])? > 0)
    }

    // ---- meta / audit
    pub fn set_meta(&self, key: &str, value: &str) -> Result<()> {
        self.execute(
            "INSERT INTO meta(key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            &[&key, &value],
        )?;
        Ok(())
    }

    pub fn delete_meta(&self, key: &str) -> Result<()> {
        self.execute("DELETE FROM meta WHERE key=?", &[&key])?;
        Ok(())
    }

    pub fn audit(
        &self,
        principal: &str,
        action: &str,
        target: Option<&str>,
        detail: Value,
        remote: Option<&str>,
    ) -> Result<()> {
        let detail = if detail.is_null() { json!({}) } else { detail };
        self.execute(
            "INSERT INTO audit_log(at, principal, action, target, detail, remote) VALUES (?, ?, ?, ?, ?, ?)",
            &[&self.now(), &principal, &action, &target, &j(&detail), &remote],
        )?;
        Ok(())
    }
}

pub struct Store {
    pub db_path: String,
    writer: Mutex<Connection>,
    readers: Vec<Mutex<Connection>>,
    next_reader: AtomicUsize,
    revision: AtomicU64,
    pub clock: Clock,
    pub ids: IdGen,
}

fn configure(c: &Connection, wal: bool) -> Result<()> {
    if wal {
        // a database on a filesystem that refuses WAL still works (rollback journal), as before
        let _ = c.query_row("PRAGMA journal_mode=WAL", [], |r| r.get::<_, String>(0));
    }
    c.execute_batch("PRAGMA synchronous=FULL; PRAGMA foreign_keys=ON;")?;
    c.busy_timeout(std::time::Duration::from_millis(5000))?;
    Ok(())
}

impl Store {
    pub fn open(db_path: &str, clock: Clock, ids: IdGen) -> Result<Store> {
        let memory = db_path == ":memory:";
        let path = if memory { db_path.to_string() } else { expand_user(db_path) };
        if !memory
            && let Some(parent) = Path::new(&path).parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
        }
        let writer = Connection::open(&path).with_context(|| format!("opening {path}"))?;
        configure(&writer, !memory)?;
        writer.execute_batch(DDL)?;
        let version: Option<String> =
            writer.query_row("SELECT value FROM meta WHERE key='schema_version'", [], |r| r.get(0)).optional()?;
        match version {
            None => {
                writer.execute(
                    "INSERT INTO meta(key, value) VALUES ('schema_version', ?)",
                    [SCHEMA_VERSION.to_string()],
                )?;
            }
            Some(v) => {
                let v: i64 = v.trim().parse().map_err(|_| anyhow!("database schema version {v:?} is not a number"))?;
                if v > SCHEMA_VERSION {
                    bail!("database schema {v} is newer than this code ({SCHEMA_VERSION})");
                }
                if v < SCHEMA_VERSION {
                    migrate(&writer, v)?;
                }
            }
        }
        let mut readers = Vec::new();
        if !memory {
            for _ in 0..READERS {
                let r = Connection::open_with_flags(
                    &path,
                    rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY
                        | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX
                        | rusqlite::OpenFlags::SQLITE_OPEN_URI,
                )?;
                r.busy_timeout(std::time::Duration::from_millis(5000))?;
                readers.push(Mutex::new(r));
            }
        }
        Ok(Store {
            db_path: path,
            writer: Mutex::new(writer),
            readers,
            next_reader: AtomicUsize::new(0),
            revision: AtomicU64::new(0),
            clock,
            ids,
        })
    }

    pub fn now(&self) -> String {
        (self.clock)()
    }

    pub fn revision(&self) -> u64 {
        self.revision.load(Ordering::SeqCst)
    }

    fn lock_writer(&self) -> MutexGuard<'_, Connection> {
        self.writer.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Run `f` against a read connection.  Never call this from inside a [`Store::transaction`] closure: use the
    /// [`Tx`] there (an in-memory database has no reader pool and reads through the writer).
    pub fn read<T>(&self, f: impl FnOnce(&Db) -> Result<T>) -> Result<T> {
        if self.readers.is_empty() {
            let c = self.lock_writer();
            return f(&Db { c: &c });
        }
        let n = self.readers.len();
        let start = self.next_reader.fetch_add(1, Ordering::Relaxed);
        for i in 0..n {
            if let Ok(c) = self.readers[(start + i) % n].try_lock() {
                return f(&Db { c: &c });
            }
        }
        let c = self.readers[start % n].lock().unwrap_or_else(|e| e.into_inner());
        f(&Db { c: &c })
    }

    /// `BEGIN IMMEDIATE` ... `COMMIT`; rolled back when `f` fails (or panics).  The revision moves after the commit.
    pub fn transaction<T>(&self, f: impl FnOnce(&Tx) -> Result<T>) -> Result<T> {
        let c = self.lock_writer();
        c.execute_batch("BEGIN IMMEDIATE")?;
        let tx = Tx { db: Db { c: &c }, store: self, bumps: Cell::new(0) };
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| f(&tx)));
        let bumps = tx.bumps.get();
        match result {
            Ok(Ok(v)) => {
                if let Err(e) = c.execute_batch("COMMIT") {
                    let _ = c.execute_batch("ROLLBACK");
                    return Err(e.into());
                }
                self.revision.fetch_add(bumps, Ordering::SeqCst);
                Ok(v)
            }
            Ok(Err(e)) => {
                let _ = c.execute_batch("ROLLBACK");
                Err(e)
            }
            Err(panic) => {
                let _ = c.execute_batch("ROLLBACK");
                drop(c);
                std::panic::resume_unwind(panic)
            }
        }
    }

    pub fn get_meta(&self, key: &str) -> Result<Option<String>> {
        self.read(|db| db.get_meta(key))
    }

    pub fn set_meta(&self, key: &str, value: &str) -> Result<()> {
        self.transaction(|tx| tx.set_meta(key, value))
    }

    pub fn get_memory(&self, id: &str) -> Result<Option<Memory>> {
        self.read(|db| db.get_memory(id))
    }

    pub fn get_agent(&self, id: &str) -> Result<Option<Agent>> {
        self.read(|db| db.get_agent(id))
    }

    pub fn get_rule(&self, id: &str) -> Result<Option<Rule>> {
        self.read(|db| db.get_rule(id))
    }

    pub fn list_rules(&self, org_id: Option<&str>, enabled_only: bool) -> Result<Vec<Rule>> {
        self.read(|db| db.list_rules(org_id, enabled_only))
    }

    pub fn list_memories(&self, org_id: &str, f: &MemoryFilter) -> Result<Vec<Memory>> {
        self.read(|db| db.list_memories(org_id, f))
    }

    pub fn list_agents(&self, org_id: Option<&str>, include_revoked: bool) -> Result<Vec<Agent>> {
        self.read(|db| db.list_agents(org_id, include_revoked))
    }

    pub fn stats(&self) -> Result<Value> {
        let rev = self.revision();
        self.read(|db| db.stats(rev))
    }

    pub fn audit(
        &self,
        principal: &str,
        action: &str,
        target: Option<&str>,
        detail: Value,
        remote: Option<&str>,
    ) -> Result<()> {
        self.transaction(|tx| tx.audit(principal, action, target, detail, remote))
    }

    /// Delete audit rows older than `older_than_days` of wall-clock time; returns how many went.
    pub fn prune_audit(&self, older_than_days: i64) -> Result<usize> {
        let cutoff = (chrono::Utc::now() - chrono::Duration::days(older_than_days))
            .format("%Y-%m-%dT%H:%M:%S+00:00")
            .to_string();
        self.transaction(|tx| tx.execute("DELETE FROM audit_log WHERE at < ?", &[&cutoff]))
    }

    /// Close every connection (tests that reopen the file, the CLI on shutdown).  The writer checkpoints the WAL.
    pub fn close(self) {
        drop(self.readers);
        let _ = self.writer.into_inner().map(|c| c.close());
    }
}

/// Forward-only, additive migrations (columns with defaults), applied in one transaction.
fn migrate(c: &Connection, from_version: i64) -> Result<()> {
    c.execute_batch("BEGIN IMMEDIATE")?;
    let run = || -> Result<()> {
        if from_version < 2 {
            let have: Vec<String> = {
                let mut stmt = c.prepare("PRAGMA table_info(rules)")?;
                stmt.query_map([], |r| r.get::<_, String>("name"))?.collect::<rusqlite::Result<_>>()?
            };
            for (name, ddl) in [
                ("sources", r#"TEXT NOT NULL DEFAULT '["agent_observation"]'"#),
                ("emits_slot", "TEXT"),
                ("emits_topic", "TEXT"),
                ("min_units", "TEXT NOT NULL DEFAULT '{}'"),
                ("corroborate", "INTEGER NOT NULL DEFAULT 0"),
            ] {
                if !have.iter().any(|h| h == name) {
                    c.execute_batch(&format!("ALTER TABLE rules ADD COLUMN {name} {ddl}"))?;
                }
            }
        }
        c.execute("UPDATE meta SET value=? WHERE key='schema_version'", [SCHEMA_VERSION.to_string()])?;
        Ok(())
    };
    match run() {
        Ok(()) => {
            c.execute_batch("COMMIT")?;
            tracing::info!("migrated database schema {from_version} -> {SCHEMA_VERSION}");
            Ok(())
        }
        Err(e) => {
            let _ = c.execute_batch("ROLLBACK");
            Err(e)
        }
    }
}

/// `Path(p).expanduser()`.
pub fn expand_user(p: &str) -> String {
    if let Some(rest) = p.strip_prefix("~/").or_else(|| (p == "~").then_some(""))
        && let Some(home) = std::env::var_os("HOME")
    {
        return Path::new(&home).join(rest).to_string_lossy().into_owned();
    }
    p.to_string()
}
