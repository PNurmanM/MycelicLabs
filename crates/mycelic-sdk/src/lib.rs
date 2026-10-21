//! Mycelic SDK: a client for the HTTP API plus an agent-local memory store, and nothing of the server.
//!
//! ```no_run
//! use mycelic_sdk::{LocalMemory, MycelicClient, Note, QueryOptions};
//!
//! let local = LocalMemory::open("~/.mycelic/agent-7.db").unwrap();      // what this agent knows, privately
//! let client = MycelicClient::new("https://mycelic.example.com", "mk_agent-7....");
//!
//! let note_id = local
//!     .note(&Note { text: "Port of Rotterdam terminal 3 strike announced for weeks 41-43".into(),
//!                   topic: Some("supply:sd-9/transport".into()), slot: Some("transport_disruption".into()),
//!                   entity: Some("sd-9".into()), confidence: 0.9, ..Note::default() })
//!     .unwrap();
//! local.share(&client, &note_id, "team").unwrap();                        // publish it; idempotent on retry
//! let answer = client.query("delivery risk for RX-4 in Q4", &QueryOptions::scoped("northwind")).unwrap();
//! let lineage = client.lineage(answer["answer"]["memory_id"].as_str().unwrap()).unwrap();
//! ```
//!
//! The client is synchronous (it drops into any agent runtime) and verifies server certificates, against the
//! bundled web roots or, with [`MycelicClient::with_ca_file`], against a private CA only.  Retries cover
//! connection errors and 429/502/503/504 with exponential backoff; other 4xx/5xx raise [`MycelicError`]
//! immediately.  This crate must never depend on the server crate: agents carry it, not the service's stack
//! (`crates/mycelic-harness/tests/layout.rs` enforces that).

use serde_json::{Map, Value, json};
use std::fmt;
use std::path::Path;
use std::time::Duration;

/// The Mycelic release version, single-sourced here: the server crate re-exports it.
pub const VERSION: &str = "0.1.0";

#[derive(Debug, Clone)]
pub struct MycelicError {
    /// HTTP status, or 0 when the server could not be reached
    pub status: u16,
    pub message: String,
    pub body: Value,
}

impl fmt::Display for MycelicError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "HTTP {}: {}", self.status, self.message)
    }
}

impl std::error::Error for MycelicError {}

pub type Result<T> = std::result::Result<T, MycelicError>;

fn now() -> String {
    chrono::Utc::now().format("%Y-%m-%dT%H:%M:%S+00:00").to_string()
}

/// `urllib.parse.quote` for a path segment: everything but unreserved characters and `/` percent-encoded.
pub fn quote(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        if b.is_ascii_alphanumeric() || b"_.-~/".contains(&b) {
            out.push(b as char);
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

/// Form-encode query parameters (`urllib.parse.urlencode`).
fn urlencode(params: &[(&str, String)]) -> String {
    let enc = |s: &str| {
        let mut out = String::new();
        for b in s.bytes() {
            if b.is_ascii_alphanumeric() || b"_.-~".contains(&b) {
                out.push(b as char);
            } else if b == b' ' {
                out.push('+');
            } else {
                out.push_str(&format!("%{b:02X}"));
            }
        }
        out
    };
    params.iter().map(|(k, v)| format!("{}={}", enc(k), enc(v))).collect::<Vec<_>>().join("&")
}

/// What `query` asks for besides the text (`POST /query` fields).
#[derive(Clone, Debug)]
pub struct QueryOptions {
    pub scope: Option<String>,
    pub min_layer: String,
    pub k: i64,
    pub include_lineage: bool,
    pub topic: Option<String>,
    pub entity: Option<String>,
}

impl Default for QueryOptions {
    fn default() -> Self {
        QueryOptions { scope: None, min_layer: "agent".into(), k: 5, include_lineage: true, topic: None, entity: None }
    }
}

impl QueryOptions {
    pub fn scoped(scope: &str) -> QueryOptions {
        QueryOptions { scope: Some(scope.to_string()), ..QueryOptions::default() }
    }
}

/// The fields of one observation an agent shares (`POST /memory`).
#[derive(Clone, Debug, PartialEq)]
pub struct Note {
    pub text: String,
    pub topic: Option<String>,
    pub slot: Option<String>,
    pub entity: Option<String>,
    pub kind: String,
    pub confidence: f64,
    pub tags: Vec<String>,
    pub local_id: Option<String>,
}

impl Default for Note {
    fn default() -> Self {
        Note {
            text: String::new(),
            topic: None,
            slot: None,
            entity: None,
            kind: "observation".into(),
            confidence: 0.8,
            tags: Vec::new(),
            local_id: None,
        }
    }
}

/// An agent registration for the administrator surface.
#[derive(Clone, Debug, Default)]
pub struct Registration {
    pub agent_id: String,
    pub enterprise: String,
    pub region: Option<String>,
    pub subsidiary: Option<String>,
    pub department: Option<String>,
    pub team: Option<String>,
    pub display_name: Option<String>,
    pub scopes: Option<Vec<String>>,
}

/// A client of one deployment with one credential (an agent key or the administrator token).
#[derive(Clone)]
pub struct MycelicClient {
    pub base_url: String,
    api_key: String,
    pub timeout: Duration,
    pub retries: u32,
    pub backoff: f64,
    pub user_agent: String,
    agent: ureq::Agent,
    ca_certs: Option<Vec<ureq::tls::Certificate<'static>>>,
}

impl fmt::Debug for MycelicClient {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MycelicClient").field("base_url", &self.base_url).finish_non_exhaustive()
    }
}

impl MycelicClient {
    pub fn new(base_url: &str, api_key: &str) -> MycelicClient {
        let mut c = MycelicClient {
            base_url: base_url.trim_end_matches('/').to_string(),
            api_key: api_key.to_string(),
            timeout: Duration::from_secs(15),
            retries: 4,
            backoff: 0.5,
            user_agent: "mycelic-sdk/0.1".into(),
            agent: ureq::Agent::new_with_defaults(),
            ca_certs: None,
        };
        c.rebuild_agent();
        c
    }

    fn rebuild_agent(&mut self) {
        let mut tls = ureq::tls::TlsConfig::builder();
        if let Some(certs) = &self.ca_certs {
            tls = tls.root_certs(ureq::tls::RootCerts::new_with_certs(certs));
        }
        let config = ureq::Agent::config_builder()
            .timeout_global(Some(self.timeout))
            .http_status_as_error(false)
            .user_agent(self.user_agent.as_str())
            .tls_config(tls.build())
            .build();
        self.agent = ureq::Agent::new_with_config(config);
    }

    /// Trust only this CA bundle (PEM) for the server's certificate, like `ssl.create_default_context(cafile=...)`.
    pub fn with_ca_file(mut self, ca_file: Option<&str>) -> Result<MycelicClient> {
        let Some(path) = ca_file.filter(|p| !p.is_empty()) else { return Ok(self) };
        let pem = std::fs::read(path).map_err(|e| MycelicError {
            status: 0,
            message: format!("cannot read {path}: {e}"),
            body: Value::Null,
        })?;
        let certs: Vec<ureq::tls::Certificate<'static>> = ureq::tls::parse_pem(&pem)
            .filter_map(|item| match item {
                Ok(ureq::tls::PemItem::Certificate(c)) => Some(c),
                _ => None,
            })
            .collect();
        if certs.is_empty() {
            return Err(MycelicError { status: 0, message: format!("no certificate in {path}"), body: Value::Null });
        }
        self.ca_certs = Some(certs);
        self.rebuild_agent();
        Ok(self)
    }

    pub fn with_timeout(mut self, timeout: Duration) -> MycelicClient {
        self.timeout = timeout;
        self.rebuild_agent();
        self
    }

    pub fn with_retries(mut self, retries: u32) -> MycelicClient {
        self.retries = retries;
        self
    }

    // ---------------------------------------------------------------- transport
    pub fn request(
        &self,
        method: &str,
        path: &str,
        body: Option<&Value>,
        params: &[(&str, Option<String>)],
    ) -> Result<Value> {
        let mut url = format!("{}{path}", self.base_url);
        let present: Vec<(&str, String)> = params.iter().filter_map(|(k, v)| v.clone().map(|v| (*k, v))).collect();
        if !present.is_empty() {
            url.push('?');
            url.push_str(&urlencode(&present));
        }
        let data = body.map(|b| serde_json::to_vec(b).unwrap_or_default());
        let mut attempt = 0u32;
        loop {
            let auth = format!("Bearer {}", self.api_key);
            let result = match (method, &data) {
                ("GET", _) => {
                    self.agent.get(&url).header("Authorization", &auth).header("Accept", "application/json").call()
                }
                ("DELETE", _) => {
                    self.agent.delete(&url).header("Authorization", &auth).header("Accept", "application/json").call()
                }
                (_, Some(d)) => self
                    .agent
                    .post(&url)
                    .header("Authorization", &auth)
                    .header("Accept", "application/json")
                    .header("Content-Type", "application/json")
                    .send(d.as_slice()),
                (_, None) => self
                    .agent
                    .post(&url)
                    .header("Authorization", &auth)
                    .header("Accept", "application/json")
                    .send_empty(),
            };
            match result {
                Ok(mut resp) => {
                    let status = resp.status().as_u16();
                    let raw = resp.body_mut().with_config().limit(512 * 1024 * 1024).read_to_vec().unwrap_or_default();
                    if (200..300).contains(&status) {
                        return Ok(if raw.is_empty() {
                            json!({})
                        } else {
                            serde_json::from_slice(&raw).unwrap_or(Value::Null)
                        });
                    }
                    let payload: Value = if raw.is_empty() {
                        json!({})
                    } else {
                        serde_json::from_slice(&raw).unwrap_or_else(|_| json!({"error": String::from_utf8_lossy(&raw)}))
                    };
                    let retryable = matches!(status, 429 | 502 | 503 | 504);
                    if !retryable || attempt >= self.retries {
                        let reason = resp.status().canonical_reason().unwrap_or("").to_string();
                        let message = payload.get("error").filter(|e| !e.is_null()).map(value_text).unwrap_or(reason);
                        return Err(MycelicError { status, message, body: payload });
                    }
                }
                Err(e) => {
                    if attempt >= self.retries {
                        return Err(MycelicError {
                            status: 0,
                            message: format!("connection failed: {e}"),
                            body: Value::Null,
                        });
                    }
                }
            }
            attempt += 1;
            std::thread::sleep(Duration::from_secs_f64(self.backoff * 2f64.powi(attempt as i32 - 1)));
        }
    }

    // ---------------------------------------------------------------- agent surface
    pub fn health(&self) -> Result<Value> {
        self.request("GET", "/health", None, &[])
    }

    pub fn whoami(&self) -> Result<Value> {
        self.request("GET", "/whoami", None, &[])
    }

    /// Share one note as the calling agent (`POST /memory`); `local_ref` and `idempotency_key` make re-sends safe.
    pub fn remember(
        &self,
        note: &Note,
        visibility: &str,
        idempotency_key: Option<&str>,
        observed_at: Option<&str>,
        local_ref: Option<&str>,
    ) -> Result<Value> {
        let mut body = Map::new();
        body.insert("text".into(), json!(note.text));
        for (k, v) in [("topic", &note.topic), ("slot", &note.slot), ("entity", &note.entity)] {
            if let Some(v) = v {
                body.insert(k.into(), json!(v));
            }
        }
        body.insert("kind".into(), json!(note.kind));
        body.insert("confidence".into(), json!(note.confidence));
        body.insert("visibility".into(), json!(visibility));
        for (k, v) in [("idempotency_key", idempotency_key), ("observed_at", observed_at), ("local_ref", local_ref)] {
            if let Some(v) = v {
                body.insert(k.into(), json!(v));
            }
        }
        self.request("POST", "/memory", Some(&Value::Object(body)), &[])
    }

    /// `POST /memory` with a body built by the caller (null fields dropped).
    pub fn remember_body(&self, body: &Value) -> Result<Value> {
        let filtered: Map<String, Value> = body
            .as_object()
            .map(|m| m.iter().filter(|(_, v)| !v.is_null()).map(|(k, v)| (k.clone(), v.clone())).collect())
            .unwrap_or_default();
        self.request("POST", "/memory", Some(&Value::Object(filtered)), &[])
    }

    pub fn publish_events(&self, events: &[Value]) -> Result<Value> {
        self.request("POST", "/events", Some(&json!({"events": events})), &[])
    }

    pub fn query(&self, text: &str, opts: &QueryOptions) -> Result<Value> {
        let mut body =
            json!({"query": text, "min_layer": opts.min_layer, "k": opts.k, "include_lineage": opts.include_lineage});
        for (k, v) in [("scope", &opts.scope), ("topic", &opts.topic), ("entity", &opts.entity)] {
            if let Some(v) = v {
                body[k] = json!(v);
            }
        }
        self.request("POST", "/query", Some(&body), &[])
    }

    pub fn get_memory(&self, memory_id: &str) -> Result<Value> {
        Ok(self.request("GET", &format!("/memory/{}", quote(memory_id)), None, &[])?["memory"].clone())
    }

    pub fn lineage(&self, memory_id: &str) -> Result<Value> {
        self.request("GET", &format!("/lineage/{}", quote(memory_id)), None, &[])
    }

    pub fn retract(&self, memory_id: &str, reason: &str) -> Result<Value> {
        self.request("POST", &format!("/memory/{}/retract", quote(memory_id)), Some(&json!({"reason": reason})), &[])
    }

    pub fn list_memories(&self, scope: Option<&str>, layer: Option<&str>, limit: i64) -> Result<Vec<Value>> {
        self.list_memories_with_status(scope, layer, limit, None)
    }

    pub fn list_memories_with_status(
        &self,
        scope: Option<&str>,
        layer: Option<&str>,
        limit: i64,
        status: Option<&str>,
    ) -> Result<Vec<Value>> {
        let params = [
            ("scope", scope.map(String::from)),
            ("layer", layer.map(String::from)),
            ("limit", Some(limit.to_string())),
            ("status", status.map(String::from)),
        ];
        Ok(self.request("GET", "/memories", None, &params)?["memories"].as_array().cloned().unwrap_or_default())
    }

    // ---------------------------------------------------------------- admin surface (admin token)
    pub fn register_agent(&self, r: &Registration) -> Result<Value> {
        let mut body = Map::new();
        body.insert("agent_id".into(), json!(r.agent_id));
        body.insert("enterprise".into(), json!(r.enterprise));
        for (k, v) in [
            ("region", &r.region),
            ("subsidiary", &r.subsidiary),
            ("department", &r.department),
            ("team", &r.team),
            ("display_name", &r.display_name),
        ] {
            if let Some(v) = v {
                body.insert(k.into(), json!(v));
            }
        }
        if let Some(s) = &r.scopes {
            body.insert("scopes".into(), json!(s));
        }
        self.request("POST", "/admin/agents", Some(&Value::Object(body)), &[])
    }

    pub fn list_agents(&self, org: Option<&str>) -> Result<Vec<Value>> {
        Ok(self.request("GET", "/admin/agents", None, &[("org", org.map(String::from))])?["agents"]
            .as_array()
            .cloned()
            .unwrap_or_default())
    }

    pub fn revoke_agent(&self, agent_id: &str) -> Result<Value> {
        self.request("DELETE", &format!("/admin/agents/{}", quote(agent_id)), None, &[])
    }

    pub fn rotate_key(&self, agent_id: &str) -> Result<Value> {
        self.request("POST", &format!("/admin/agents/{}/rotate", quote(agent_id)), None, &[])
    }

    pub fn upsert_rule(&self, rule: &Value) -> Result<Value> {
        self.request("POST", "/admin/rules", Some(rule), &[])
    }

    pub fn list_rules(&self) -> Result<Vec<Value>> {
        Ok(self.request("GET", "/admin/rules", None, &[])?["rules"].as_array().cloned().unwrap_or_default())
    }

    pub fn replay(&self) -> Result<Value> {
        self.request("POST", "/admin/replay", Some(&json!({})), &[])
    }

    pub fn status(&self) -> Result<Value> {
        self.request("GET", "/admin/status", None, &[])
    }

    pub fn events(&self, org: &str, params: &[(&str, Option<String>)]) -> Result<Vec<Value>> {
        let mut all = vec![("org", Some(org.to_string()))];
        all.extend(params.iter().cloned());
        Ok(self.request("GET", "/admin/events", None, &all)?["events"].as_array().cloned().unwrap_or_default())
    }
}

fn value_text(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

/// `Path(p).expanduser()`.
fn expand_user(p: &str) -> String {
    if let Some(rest) = p.strip_prefix("~/")
        && let Some(home) = std::env::var_os("HOME")
    {
        return Path::new(&home).join(rest).to_string_lossy().into_owned();
    }
    p.to_string()
}

/// The agent's own notes, kept on its own disk.  Only what the agent chooses to share leaves the machine.
///
/// Every note gets a local id; sharing sends that id as the idempotency key, so a crash between the request and the
/// acknowledgement cannot duplicate a memory on the server, and `mark_shared` records the server ids so the agent
/// can later ask for the lineage of what it contributed to.  The file format is the Python SDK's, so an agent keeps
/// its notes across the switch.
pub struct LocalMemory {
    pub path: String,
    c: rusqlite::Connection,
}

impl LocalMemory {
    pub fn open(path: &str) -> rusqlite::Result<LocalMemory> {
        let path = if path == ":memory:" { path.to_string() } else { expand_user(path) };
        if path != ":memory:"
            && let Some(parent) = Path::new(&path).parent()
            && !parent.as_os_str().is_empty()
        {
            let _ = std::fs::create_dir_all(parent);
        }
        let c = rusqlite::Connection::open(&path)?;
        let _ = c.query_row("PRAGMA journal_mode=WAL", [], |r| r.get::<_, String>(0));
        c.execute_batch(
            "CREATE TABLE IF NOT EXISTS notes (
                local_id   TEXT PRIMARY KEY,
                text       TEXT NOT NULL,
                topic      TEXT, slot TEXT, entity TEXT,
                kind       TEXT NOT NULL DEFAULT 'observation',
                confidence REAL NOT NULL DEFAULT 0.8,
                tags       TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL,
                shared_memory_id TEXT, shared_event_id TEXT, shared_at TEXT
            );",
        )?;
        Ok(LocalMemory { path, c })
    }

    /// Record a note; returns its local id (`note_<16 hex>` unless one is given).  An existing id is kept as it was.
    pub fn note(&self, n: &Note) -> rusqlite::Result<String> {
        let local_id =
            n.local_id.clone().unwrap_or_else(|| format!("note_{}", &uuid::Uuid::new_v4().simple().to_string()[..16]));
        self.c.execute(
            "INSERT OR IGNORE INTO notes(local_id, text, topic, slot, entity, kind, confidence, tags, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rusqlite::params![local_id, n.text, n.topic, n.slot, n.entity, n.kind, n.confidence, serde_json::to_string(&n.tags).unwrap_or("[]".into()), now()],
        )?;
        Ok(local_id)
    }

    fn row(r: &rusqlite::Row) -> rusqlite::Result<Value> {
        let tags: String = r.get("tags")?;
        let shared_memory_id: Option<String> = r.get("shared_memory_id")?;
        Ok(json!({
            "local_id": r.get::<_, String>("local_id")?, "text": r.get::<_, String>("text")?,
            "topic": r.get::<_, Option<String>>("topic")?, "slot": r.get::<_, Option<String>>("slot")?,
            "entity": r.get::<_, Option<String>>("entity")?, "kind": r.get::<_, String>("kind")?,
            "confidence": r.get::<_, f64>("confidence")?,
            "tags": serde_json::from_str::<Value>(&tags).unwrap_or(json!([])),
            "created_at": r.get::<_, String>("created_at")?, "shared_memory_id": shared_memory_id,
            "shared_event_id": r.get::<_, Option<String>>("shared_event_id")?, "shared_at": r.get::<_, Option<String>>("shared_at")?,
            "shared": shared_memory_id.is_some(),
        }))
    }

    pub fn get(&self, local_id: &str) -> rusqlite::Result<Option<Value>> {
        use rusqlite::OptionalExtension;
        self.c.query_row("SELECT * FROM notes WHERE local_id=?", [local_id], Self::row).optional()
    }

    /// Every note, oldest first; `shared` filters to the shared or the never-shared ones.
    pub fn all(&self, shared: Option<bool>) -> rusqlite::Result<Vec<Value>> {
        let filter = match shared {
            Some(true) => " WHERE shared_memory_id IS NOT NULL",
            Some(false) => " WHERE shared_memory_id IS NULL",
            None => "",
        };
        let mut stmt = self.c.prepare(&format!("SELECT * FROM notes{filter} ORDER BY created_at, local_id"))?;
        stmt.query_map([], Self::row)?.collect()
    }

    pub fn search(&self, text: &str, limit: i64) -> rusqlite::Result<Vec<Value>> {
        let like = format!("%{text}%");
        let mut stmt = self.c.prepare(
            "SELECT * FROM notes WHERE text LIKE ? OR topic LIKE ? OR entity LIKE ? ORDER BY created_at DESC LIMIT ?",
        )?;
        stmt.query_map(rusqlite::params![like, like, like, limit], Self::row)?.collect()
    }

    pub fn counts(&self) -> rusqlite::Result<Value> {
        let total: i64 = self.c.query_row("SELECT COUNT(*) FROM notes", [], |r| r.get(0))?;
        let shared: i64 =
            self.c.query_row("SELECT COUNT(*) FROM notes WHERE shared_memory_id IS NOT NULL", [], |r| r.get(0))?;
        Ok(json!({"local": total, "shared": shared, "local_only": total - shared}))
    }

    pub fn mark_shared(&self, local_id: &str, memory_id: &str, event_id: Option<&str>) -> rusqlite::Result<()> {
        self.c.execute(
            "UPDATE notes SET shared_memory_id=?, shared_event_id=?, shared_at=? WHERE local_id=?",
            rusqlite::params![memory_id, event_id, now(), local_id],
        )?;
        Ok(())
    }

    /// Share a stored note, with its local id as idempotency key and local reference.
    pub fn share(&self, client: &MycelicClient, local_id: &str, visibility: &str) -> Result<Value> {
        let local_err =
            |e: rusqlite::Error| MycelicError { status: 0, message: format!("local memory: {e}"), body: Value::Null };
        let n = self.get(local_id).map_err(local_err)?.ok_or_else(|| MycelicError {
            status: 0,
            message: format!("KeyError: {local_id:?}"),
            body: Value::Null,
        })?;
        let s = |k: &str| n[k].as_str().map(String::from);
        let note = Note {
            text: s("text").unwrap_or_default(),
            topic: s("topic"),
            slot: s("slot"),
            entity: s("entity"),
            kind: s("kind").unwrap_or_else(|| "observation".into()),
            confidence: n["confidence"].as_f64().unwrap_or(0.8),
            tags: Vec::new(),
            local_id: None,
        };
        let created_at = s("created_at");
        let res = client.remember(&note, visibility, Some(local_id), created_at.as_deref(), Some(local_id))?;
        let memory_id = res["memory_id"].as_str().unwrap_or_default().to_string();
        self.mark_shared(local_id, &memory_id, res["event_id"].as_str()).map_err(local_err)?;
        Ok(res)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_memory_keeps_private_notes_and_counts() {
        let m = LocalMemory::open(":memory:").unwrap();
        let a = m.note(&Note { text: "shared".into(), ..Note::default() }).unwrap();
        m.note(&Note {
            text: "private".into(),
            tags: vec!["private".into()],
            local_id: Some("p1".into()),
            ..Note::default()
        })
        .unwrap();
        m.note(&Note { text: "again".into(), local_id: Some("p1".into()), ..Note::default() }).unwrap();
        assert!(a.starts_with("note_") && a.len() == 21);
        m.mark_shared(&a, "mem_1", Some("evt_1")).unwrap();
        assert_eq!(m.counts().unwrap(), json!({"local": 2, "shared": 1, "local_only": 1}));
        let private = m.all(Some(false)).unwrap();
        assert_eq!(private[0]["text"], "private", "INSERT OR IGNORE keeps the first note with an id");
        assert_eq!(private[0]["tags"], json!(["private"]));
    }

    #[test]
    fn quoting_and_query_strings() {
        assert_eq!(quote("mem_a/b c"), "mem_a/b%20c");
        assert_eq!(urlencode(&[("scope", "a b/c".into()), ("limit", "5".into())]), "scope=a+b%2Fc&limit=5");
    }
}
