//! `MycelicService`: the one object the API, MCP tools and CLI talk to.
//!
//! It owns the store, the transport and the aggregator and runs the background loops that make the data path
//! real:
//!
//! * **publisher**: drains the outbox (`events.status='pending'`) into JetStream, marking rows published with their
//!   stream sequence; a broker outage leaves rows pending and the loop retries with backoff;
//! * **consumer**: pulls from the durable consumer, applies each event inside one transaction (record the event,
//!   upsert the memory it carries, run aggregation, enqueue derived events) and acks only after the commit.
//!
//! The complete path is therefore:
//!
//! ```text
//! agent -> POST /memory -> [txn: memory + event(pending)] -> publisher -> JetStream (file store)
//!       -> consumer -> apply [txn: applied + derived memories + lineage + derived events(pending)]
//!       -> publisher -> JetStream -> consumer -> apply (idempotent no-op) ...
//! POST /query -> retrieval over what the caller may see -> answer + lineage
//! ```
//!
//! Recovery: on start, a database that has never applied anything while the stream already holds events is treated
//! as lost state and the durable consumer is reset so the whole log replays; `replay()` does the same on demand.
//! Because derived ids are deterministic, a replay converges on exactly the state it rebuilds.
//!
//! Validation messages, the order in which inputs are checked and the order in which ids are drawn are the Python
//! service's, so a client sees the same answers and a pinned-clock run produces the same rows (the golden tests
//! rely on that).

use crate::aggregation::{Aggregator, Derivation, MYCELIC_PRODUCER};
use crate::auth::{AuthError, Principal, RateLimiter, authenticate, constant_time_eq, generate_api_key};
use crate::config::Settings;
use crate::hierarchy::{AgentPath, HierarchyError, LAYERS, layer_index, split_path};
use crate::lineage::{LineageNotFound, MAX_NODES, reconstruct};
use crate::metrics::Metrics;
use crate::models::{
    ALL_SCOPES, Agent, Clock, DEFAULT_AGENT_SCOPES, EventRecord, IdGen, LineageEdge, MEMORY_KINDS, Memory, OPERATORS,
    Rule, VISIBILITY, content_hash, parse_iso, random_ids, system_clock,
};
use crate::pyjson::{self, char_len, repr_list, repr_tuple, truthy};
use crate::retrieval::{Retriever, SearchParams, search_index};
use crate::store::{Store, Tx, parse_min_units};
use crate::transport::{PublishError, Transport, build_transport, info_i64, subject_for};
use anyhow::anyhow;
use hmac::{Hmac, Mac};
use serde_json::{Map, Value, json};
use sha2::Sha256;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::sync::Notify;
use tokio_util::sync::CancellationToken;

/// Metadata keys the aggregator owns; an agent may not set them on a raw observation.
pub const RESERVED_METADATA_KEYS: [&str; 20] = [
    "agg_key",
    "promoted_from",
    "version_of",
    "contributing_agents",
    "contributing_teams",
    "children",
    "child_layer",
    "parent_count",
    "fragility",
    "slots",
    "candidates",
    "effective_min_support",
    "registered_child_units",
    "status_reason",
    "reactivated_at",
    "roots",
    "evidence",
    "corroborated_units",
    "rule_chain",
    "fragility_scored_candidates",
];
const SLOT_PATTERN: &str = "^[A-Za-z0-9_.:-]{1,100}$";
const ID_PATTERN: &str = "^[A-Za-z0-9_.:-]{1,200}$";

/// What can go wrong in a service call, in the categories the API maps to status codes.
#[derive(Debug)]
pub enum ServiceError {
    /// bad client input (400)
    Validation(String),
    /// a path or layer that does not fit the hierarchy (400; MCP reports it as `HierarchyError: ...`)
    Hierarchy(String),
    /// authenticated but not allowed (403)
    Forbidden(String),
    /// not there, or not visible to the caller (404; the API never says which)
    NotFound(String),
    Internal(anyhow::Error),
}

impl std::fmt::Display for ServiceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ServiceError::Validation(m)
            | ServiceError::Hierarchy(m)
            | ServiceError::Forbidden(m)
            | ServiceError::NotFound(m) => f.write_str(m),
            ServiceError::Internal(e) => write!(f, "{e}"),
        }
    }
}

impl From<anyhow::Error> for ServiceError {
    fn from(e: anyhow::Error) -> Self {
        match e.downcast::<ServiceError>() {
            Ok(se) => se,
            Err(e) => match e.downcast::<HierarchyError>() {
                Ok(h) => ServiceError::Hierarchy(h.0),
                Err(e) => ServiceError::Internal(e),
            },
        }
    }
}

impl From<HierarchyError> for ServiceError {
    fn from(e: HierarchyError) -> Self {
        ServiceError::Hierarchy(e.0)
    }
}

impl std::error::Error for ServiceError {}

pub type SResult<T> = Result<T, ServiceError>;

fn validation(msg: impl Into<String>) -> ServiceError {
    ServiceError::Validation(msg.into())
}

/// The name Python would have printed for the error's type (`ValidationError: ...` in audit rows and errors).
pub fn error_type_name(e: &anyhow::Error) -> &'static str {
    if let Some(se) = e.downcast_ref::<ServiceError>() {
        return match se {
            ServiceError::Validation(_) => "ValidationError",
            ServiceError::Hierarchy(_) => "HierarchyError",
            ServiceError::Forbidden(_) => "Forbidden",
            ServiceError::NotFound(_) => "NotFound",
            ServiceError::Internal(_) => "Error",
        };
    }
    if e.downcast_ref::<HierarchyError>().is_some() {
        return "HierarchyError";
    }
    if e.downcast_ref::<serde_json::Error>().is_some() {
        return "JSONDecodeError";
    }
    if let Some(sq) = e.downcast_ref::<rusqlite::Error>() {
        return match sq {
            rusqlite::Error::SqliteFailure(f, _) if f.code == rusqlite::ErrorCode::ConstraintViolation => {
                "IntegrityError"
            }
            _ => "DatabaseError",
        };
    }
    "Error"
}

fn error_text(e: &anyhow::Error) -> String {
    let msg = match e.downcast_ref::<ServiceError>() {
        Some(se) => se.to_string(),
        None => e.to_string(),
    };
    // errors raised from Python-style code paths already carry their type ("ValueError: ...")
    if msg.starts_with("ValueError: ") || msg.starts_with("KeyError: ") || msg.starts_with("RuntimeError: ") {
        msg
    } else {
        format!("{}: {msg}", error_type_name(e))
    }
}

// ---------------------------------------------------------------------------------------------------- body fields
fn char_class_ok(v: &str, max: usize) -> bool {
    let n = v.chars().count();
    (1..=max).contains(&n) && v.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '.' | ':' | '-'))
}

fn slot_ok(v: &str) -> bool {
    char_class_ok(v, 100)
}

fn id_ok(v: &str) -> bool {
    char_class_ok(v, 200)
}

/// A field's format check: its description for the error message, and the predicate.
pub type FieldPattern<'a> = (&'a str, fn(&str) -> bool);

/// `_s`: an optional (or required) string field, stripped, length-checked and optionally pattern-checked.
pub fn field_str(
    body: &Map<String, Value>,
    key: &str,
    required: bool,
    max_len: usize,
    pattern: Option<FieldPattern>,
) -> SResult<Option<String>> {
    let v = body.get(key);
    let s = match v {
        None | Some(Value::Null) => None,
        Some(Value::String(s)) if s.is_empty() => None,
        Some(Value::String(s)) => Some(s.as_str()),
        Some(_) => return Err(validation(format!("'{key}' must be a string"))),
    };
    let Some(s) = s else {
        return if required { Err(validation(format!("'{key}' is required"))) } else { Ok(None) };
    };
    let v = crate::textutil::py_strip(s);
    if v.is_empty() && required {
        return Err(validation(format!("'{key}' is required")));
    }
    if char_len(v) > max_len {
        return Err(validation(format!("'{key}' is longer than {max_len} characters")));
    }
    if let Some((pat, ok)) = pattern
        && !v.is_empty()
        && !ok(v)
    {
        return Err(validation(format!("'{key}' contains characters outside {pat}")));
    }
    Ok((!v.is_empty()).then(|| v.to_string()))
}

pub fn field_f(body: &Map<String, Value>, key: &str, default: f64) -> SResult<f64> {
    let v = match body.get(key) {
        None => return Ok(default),
        Some(v) => v,
    };
    let f = match v {
        Value::Number(n) => n.as_f64().ok_or_else(|| validation(format!("'{key}' must be a number")))?,
        _ => return Err(validation(format!("'{key}' must be a number"))),
    };
    if !(0.0..=1.0).contains(&f) {
        return Err(validation(format!("'{key}' must be between 0 and 1")));
    }
    Ok(f)
}

pub fn field_iso(body: &Map<String, Value>, key: &str) -> SResult<Option<String>> {
    let Some(v) = field_str(body, key, false, 40, None)? else { return Ok(None) };
    match parse_iso(&v) {
        Some(dt) => Ok(Some(dt.isoformat_seconds())),
        None => Err(validation(format!("'{key}' must be an ISO-8601 timestamp"))),
    }
}

pub fn field_small_dict(body: &Map<String, Value>, key: &str, max_bytes: usize) -> SResult<Map<String, Value>> {
    match body.get(key) {
        None | Some(Value::Null) => Ok(Map::new()),
        Some(Value::Object(m)) => {
            if char_len(&pyjson::dumps(&Value::Object(m.clone()))) > max_bytes {
                return Err(validation(format!("'{key}' is larger than {max_bytes} bytes")));
            }
            Ok(m.clone())
        }
        Some(_) => Err(validation(format!("'{key}' must be an object"))),
    }
}

/// Python's `str(value)` for a JSON value (reasons and ids that arrive as other types).
pub fn py_str(v: &Value) -> String {
    match v {
        Value::Null => "None".into(),
        Value::Bool(true) => "True".into(),
        Value::Bool(false) => "False".into(),
        Value::String(s) => s.clone(),
        Value::Number(n) => match n.as_i64().map(|i| i.to_string()).or_else(|| n.as_u64().map(|u| u.to_string())) {
            Some(s) => s,
            None => pyjson::float_repr(n.as_f64().unwrap_or(0.0)),
        },
        other => pyjson::dumps(other),
    }
}

fn opt_str(v: Option<&Value>) -> Option<String> {
    v.and_then(Value::as_str).map(String::from)
}

/// `x or default` for an optional JSON string.
fn str_or(v: Option<&Value>, default: &str) -> String {
    match v {
        Some(Value::String(s)) if !s.is_empty() => s.clone(),
        Some(v) if truthy(v) => py_str(v),
        _ => default.to_string(),
    }
}

fn py_float(v: &Value) -> SResult<f64> {
    match v {
        Value::Number(n) => n.as_f64().ok_or_else(|| validation("could not convert to float")),
        Value::Bool(b) => Ok(*b as i64 as f64),
        Value::String(s) => crate::config::py_float(s)
            .ok_or_else(|| validation(format!("could not convert string to float: {}", pyjson::repr_str(s)))),
        _ => Err(validation("float() argument must be a string or a real number")),
    }
}

fn py_int(v: &Value) -> SResult<i64> {
    match v {
        Value::Number(n) => n
            .as_i64()
            .or_else(|| n.as_f64().map(|f| f.trunc() as i64))
            .ok_or_else(|| validation("could not convert to int")),
        Value::Bool(b) => Ok(*b as i64),
        Value::String(s) => crate::config::py_int(s)
            .ok_or_else(|| validation(format!("invalid literal for int() with base 10: {}", pyjson::repr_str(s)))),
        _ => Err(validation("int() argument must be a string, a bytes-like object or a real number")),
    }
}

fn as_object(v: &Value) -> Map<String, Value> {
    match v {
        Value::Object(m) => m.clone(),
        _ => Map::new(),
    }
}

// ---------------------------------------------------------------------------------------------------- service
pub struct MycelicService {
    pub settings: Settings,
    pub metrics: Arc<Metrics>,
    pub store: Arc<Store>,
    pub transport: Arc<dyn Transport>,
    pub aggregator: Aggregator,
    pub retriever: Retriever,
    pub limiter: Mutex<RateLimiter>,
    started_at: Mutex<Option<String>>,
    tasks: Mutex<Vec<tokio::task::JoinHandle<()>>>,
    stop: Mutex<CancellationToken>,
    outbox_wake: Arc<Notify>,
    idle: AtomicBool,
    consumer_running: AtomicBool,
    publisher_running: AtomicBool,
    replay_target: Mutex<Option<i64>>,
    last_seen_touch: Mutex<HashMap<String, Instant>>,
}

/// Optional parts a caller may supply instead of the defaults (tests: a shared transport, a pinned clock).
#[derive(Default)]
pub struct ServiceParts {
    pub store: Option<Store>,
    pub transport: Option<Arc<dyn Transport>>,
    pub metrics: Option<Arc<Metrics>>,
    pub clock: Option<Clock>,
    pub ids: Option<IdGen>,
}

impl MycelicService {
    pub fn new(settings: Settings) -> anyhow::Result<Arc<MycelicService>> {
        Self::with_parts(settings, ServiceParts::default())
    }

    pub fn with_parts(settings: Settings, parts: ServiceParts) -> anyhow::Result<Arc<MycelicService>> {
        let metrics = parts.metrics.unwrap_or_else(|| Arc::new(Metrics::new()));
        let clock = parts.clock.unwrap_or_else(system_clock);
        let ids = parts.ids.unwrap_or_else(random_ids);
        let store = match parts.store {
            Some(s) => s,
            None => Store::open(&settings.db_path, clock, ids)?,
        };
        let outbox_wake = Arc::new(Notify::new());
        let transport = match parts.transport {
            Some(t) => t,
            None => {
                let (m, wake) = (metrics.clone(), outbox_wake.clone());
                build_transport(
                    &settings,
                    Arc::new(move |connected| {
                        m.transport_connected.set(connected as i64);
                        if connected {
                            wake.notify_one();
                        }
                    }),
                )
            }
        };
        metrics.info.with_label_values(&[crate::VERSION]).set(1);
        Ok(Arc::new(MycelicService {
            aggregator: Aggregator::new(settings.min_support),
            retriever: Retriever::new(),
            limiter: Mutex::new(RateLimiter::new(settings.rate_limit_rps, settings.rate_limit_burst)),
            settings,
            metrics,
            store: Arc::new(store),
            transport,
            started_at: Mutex::new(None),
            tasks: Mutex::new(Vec::new()),
            stop: Mutex::new(CancellationToken::new()),
            outbox_wake,
            idle: AtomicBool::new(true),
            consumer_running: AtomicBool::new(false),
            publisher_running: AtomicBool::new(false),
            replay_target: Mutex::new(None),
            last_seen_touch: Mutex::new(HashMap::new()),
        }))
    }

    fn now(&self) -> String {
        self.store.now()
    }

    pub fn replay_target(&self) -> Option<i64> {
        *self.replay_target.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn set_replay_target_mem(&self, v: Option<i64>) {
        *self.replay_target.lock().unwrap_or_else(|e| e.into_inner()) = v;
    }

    pub fn started_at(&self) -> Option<String> {
        self.started_at.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }

    fn stop_token(&self) -> CancellationToken {
        self.stop.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }

    pub fn wake_outbox(&self) {
        self.outbox_wake.notify_one();
    }

    // ------------------------------------------------------------------ lifecycle
    pub async fn start(self: &Arc<Self>, background: bool) -> anyhow::Result<()> {
        *self.started_at.lock().unwrap_or_else(|e| e.into_inner()) = Some(self.now());
        *self.stop.lock().unwrap_or_else(|e| e.into_inner()) = CancellationToken::new();
        let svc = self.clone();
        tokio::task::spawn_blocking(move || svc.load_rules_file()).await??;
        let svc = self.clone();
        match tokio::task::spawn_blocking(move || svc.store.prune_audit(svc.settings.audit_retention_days)).await? {
            Ok(n) if n > 0 => {
                tracing::info!("pruned {n} audit rows older than {} days", self.settings.audit_retention_days)
            }
            Ok(_) => {}
            Err(e) => tracing::error!("audit pruning failed: {e}"),
        }
        self.connect_with_retry().await;
        if background {
            let mut tasks = self.tasks.lock().unwrap_or_else(|e| e.into_inner());
            let s = self.clone();
            tasks.push(tokio::spawn(async move { s.transport_keeper().await }));
            let s = self.clone();
            tasks.push(tokio::spawn(async move { s.publisher_loop().await }));
            let s = self.clone();
            tasks.push(tokio::spawn(async move { s.consumer_loop().await }));
        }
        tracing::info!(
            "mycelic {} started (db={}, transport={})",
            crate::VERSION,
            self.store.db_path,
            self.transport.name()
        );
        Ok(())
    }

    pub async fn stop(&self) {
        self.stop_token().cancel();
        self.outbox_wake.notify_one();
        let tasks: Vec<_> = std::mem::take(&mut *self.tasks.lock().unwrap_or_else(|e| e.into_inner()));
        for t in &tasks {
            t.abort();
        }
        for t in tasks {
            let _ = t.await;
        }
        self.publisher_running.store(false, Ordering::SeqCst);
        self.consumer_running.store(false, Ordering::SeqCst);
        self.transport.close().await;
    }

    pub async fn close(&self) {
        self.stop().await;
    }

    async fn connect_with_retry(self: &Arc<Self>) -> bool {
        if let Err(e) = self.transport.connect().await {
            tracing::warn!("transport connect failed: {e}");
            self.metrics.transport_connected.set(0);
            return false;
        }
        self.metrics.transport_connected.set(1);
        if let Err(e) = self.recover_if_needed().await {
            tracing::error!("recovery check failed: {e}");
        }
        true
    }

    /// Remember how far a replay must go, in memory and in the database, so a crash mid-rebuild resumes it.
    fn set_replay_target(&self, target: Option<i64>) -> anyhow::Result<()> {
        self.set_replay_target_mem(target);
        self.store.transaction(|tx| match target {
            None => tx.delete_meta("replay_target_seq"),
            Some(t) => tx.set_meta("replay_target_seq", &t.to_string()),
        })
    }

    /// Bring a database that is out of step with the stream back in sync.
    ///
    /// * never applied anything while the stream has events (fresh volume, lost database): replay the whole log;
    /// * applied less than the durable consumer has acknowledged (database restored from a backup), or the durable
    ///   consumer is brand new while the database is not (consumer lost): recreate the consumer at
    ///   `last_applied_seq + 1` so the events since then are delivered again (applying is idempotent);
    /// * the stream is shorter than what the database applied (stream purged or recreated): keep serving from the
    ///   database, start counting from the new stream, and say so loudly;
    /// * a replay recorded in the database that did not finish (crash mid-rebuild): keep reporting not-ready until
    ///   it completes.
    async fn recover_if_needed(&self) -> anyhow::Result<()> {
        let last = self.store.get_meta("last_applied_seq")?;
        let info = self.transport.info().await;
        let count = info_i64(&info, "stream_messages").unwrap_or(0);
        let last_seq = info_i64(&info, "last_seq").filter(|v| *v != 0).unwrap_or(count);
        let Some(last) = last else {
            if count != 0 {
                tracing::warn!(
                    "fresh database but the stream holds {count} events: replaying the full log to rebuild state"
                );
                self.transport.reset_consumer(None).await?;
                self.set_replay_target(Some(last_seq))?;
                self.metrics.recoveries.with_label_values(&["replay_fresh_db"]).inc();
            }
            return Ok(());
        };
        let applied: i64 = last.trim().parse().unwrap_or(0);
        let connected = info.get("connected").and_then(Value::as_bool).unwrap_or(false);
        if last_seq < applied && connected {
            tracing::error!(
                "the stream ends at seq {last_seq} but this database applied up to {applied}: the stream was purged or recreated; \
                 serving from the database, replay is impossible until events accumulate again"
            );
            self.metrics.recoveries.with_label_values(&["stream_behind_database"]).inc();
            self.store.transaction(|tx| {
                tx.set_meta("last_applied_seq", "0")?;
                tx.delete_meta("replay_target_seq")?;
                tx.audit(
                    "mycelic",
                    "recovery.stream_behind_database",
                    None,
                    json!({"stream_last_seq": last_seq, "applied": applied}),
                    None,
                )
            })?;
            self.set_replay_target_mem(None);
            return Ok(());
        }
        let floor = self.transport.consumer_ack_floor().await;
        let delivered = info_i64(&info, "consumer_delivered_seq").unwrap_or(0);
        let behind = floor.is_some_and(|f| applied < f);
        let fresh_consumer = floor == Some(0) && delivered == 0 && applied > 0;
        if behind || fresh_consumer {
            let kind = if behind { "replay_restored_backup" } else { "consumer_recreated" };
            let why = if behind {
                format!("acknowledged up to {} (restored backup?)", floor.unwrap_or(0))
            } else {
                "is new (consumer lost?)".to_string()
            };
            tracing::warn!(
                "database applied up to seq {applied} but the consumer {why}: re-delivering from {}",
                applied + 1
            );
            self.transport.reset_consumer(Some(applied + 1)).await?;
            self.set_replay_target(if last_seq > applied { Some(last_seq) } else { None })?;
            self.metrics.recoveries.with_label_values(&[kind]).inc();
            return Ok(());
        }
        if let Some(pending) = self.store.get_meta("replay_target_seq")? {
            let pending: i64 = pending.trim().parse().unwrap_or(0);
            if applied < pending {
                tracing::warn!("resuming an unfinished replay to seq {pending} (applied {applied})");
                self.set_replay_target_mem(Some(pending));
                self.metrics.recoveries.with_label_values(&["replay_resumed"]).inc();
            } else {
                self.set_replay_target(None)?;
            }
        }
        Ok(())
    }

    /// Apply the rules file at start.  A rule that differs from what is stored is upserted *and* appended to the
    /// event log, so a rebuild from the stream ends with the same rules as this node; identical rules are skipped.
    pub fn load_rules_file(&self) -> anyhow::Result<usize> {
        let Some(path) = self.settings.rules_file.clone().filter(|p| !p.is_empty()) else { return Ok(0) };
        let data: Value = serde_json::from_str(&std::fs::read_to_string(&path)?)?;
        // `data.get("rules", data)`, then iterated the way Python iterates it (a mapping yields its keys)
        let rules = match &data {
            Value::Object(m) => m.get("rules").cloned().unwrap_or_else(|| data.clone()),
            other => other.clone(),
        };
        let items: Vec<Value> = match rules {
            Value::Array(a) => a,
            Value::Object(m) => m.keys().map(|k| json!(k)).collect(),
            Value::String(st) => st.chars().map(|c| json!(c.to_string())).collect(),
            other => anyhow::bail!("{path}: rules must be a list, got {}", pyjson::dumps(&other)),
        };
        let mut n = 0;
        for item in items {
            let rule = self.rule_from_body(&item).map_err(|e| anyhow!("{path}: {e}"))?;
            self.check_rule_cycles(&rule).map_err(|e| anyhow!("{path}: {e}"))?;
            if let Some(existing) = self.store.get_rule(&rule.rule_id)?
                && existing.to_json() == rule.to_json()
            {
                continue;
            }
            self.store.transaction(|tx| {
                tx.upsert_rule(&rule)?;
                tx.insert_event(&self.admin_event(
                    tx,
                    "rule.upserted",
                    rule.org_id.as_deref().unwrap_or("_"),
                    rule.to_json(),
                ))?;
                tx.audit(
                    "rules_file",
                    "rule.upsert",
                    Some(&rule.rule_id),
                    json!({"source": path, "target_layer": rule.target_layer}),
                    None,
                )
            })?;
            n += 1;
        }
        if n > 0 {
            tracing::info!("loaded {n} changed rules from {path}");
            self.wake_outbox();
        }
        Ok(n)
    }

    // ------------------------------------------------------------------ background loops
    async fn sleep(&self, seconds: f64) {
        let token = self.stop_token();
        tokio::select! {
            _ = token.cancelled() => {}
            _ = tokio::time::sleep(Duration::from_secs_f64(seconds.max(0.0))) => {}
        }
    }

    fn stopped(&self) -> bool {
        self.stop_token().is_cancelled()
    }

    /// Establish the transport when there is no live client; a client that is reconnecting is left alone.
    async fn transport_keeper(self: Arc<Self>) {
        let mut delay = 1.0f64;
        while !self.stopped() {
            if self.transport.needs_connect() {
                let ok = self.connect_with_retry().await;
                delay = if ok { 1.0 } else { f64::min(15.0, delay * 2.0) };
                if ok {
                    self.metrics.recoveries.with_label_values(&["transport_reconnect"]).inc();
                }
            }
            self.sleep(if !self.transport.connected() { delay } else { 2.0 }).await;
        }
    }

    async fn publisher_loop(self: Arc<Self>) {
        self.publisher_running.store(true, Ordering::SeqCst);
        let interval = self.settings.publish_interval_seconds;
        let mut backoff = interval;
        while !self.stopped() {
            if !self.transport.connected() {
                self.sleep(1.0).await;
                continue;
            }
            let svc = self.clone();
            let rows = match tokio::task::spawn_blocking(move || {
                svc.store.read(|db| db.pending_events(svc.settings.publish_batch))
            })
            .await
            {
                Ok(Ok(rows)) => rows,
                Ok(Err(e)) => {
                    tracing::error!("reading the outbox failed: {e}");
                    self.sleep(1.0).await;
                    continue;
                }
                Err(_) => break,
            };
            if rows.is_empty() {
                let token = self.stop_token();
                tokio::select! {
                    _ = token.cancelled() => {}
                    _ = self.outbox_wake.notified() => {}
                    _ = tokio::time::sleep(Duration::from_secs_f64(interval)) => {}
                }
                continue;
            }
            let mut failed = false;
            for ev in rows {
                let wire = pyjson::dumps(&Self::event_wire(&ev)).into_bytes();
                let size = wire.len();
                let headers = self.sign(&wire);
                match self.transport.publish(&ev.subject, wire.into(), &ev.event_id, &headers).await {
                    Ok(seq) => {
                        let (svc, id) = (self.clone(), ev.event_id.clone());
                        let _ = tokio::task::spawn_blocking(move || {
                            svc.store.transaction(|tx| tx.mark_published(&id, seq))
                        })
                        .await;
                        self.metrics.events_published.with_label_values(&[&ev.kind]).inc();
                    }
                    Err(e)
                        if matches!(e, PublishError::Permanent(_)) || size as i64 > self.settings.max_event_bytes =>
                    {
                        // permanent: the broker will never take this event; record it and move on so the log continues
                        let (svc, id, text) = (self.clone(), ev.event_id.clone(), e.to_string());
                        let _ = tokio::task::spawn_blocking(move || {
                            svc.store.transaction(|tx| {
                                tx.mark_failed(&id, &text)?;
                                tx.audit(
                                    "mycelic",
                                    "event.unpublishable",
                                    Some(&id),
                                    json!({"error": text, "bytes": size}),
                                    None,
                                )
                            })
                        })
                        .await;
                        self.metrics.events_failed.with_label_values(&["publish_permanent"]).inc();
                        tracing::error!("event {} cannot be published ({e}); marked failed", ev.event_id);
                    }
                    Err(e) => {
                        failed = true;
                        let (svc, id, text) = (self.clone(), ev.event_id.clone(), e.to_string());
                        let _ = tokio::task::spawn_blocking(move || {
                            svc.store.transaction(|tx| tx.mark_publish_failed(&id, &text))
                        })
                        .await;
                        self.metrics.events_failed.with_label_values(&["publish"]).inc();
                        tracing::warn!("publish of {} failed: {e}", ev.event_id);
                        break;
                    }
                }
            }
            if failed {
                self.sleep(f64::min(10.0, backoff)).await;
                backoff = f64::min(10.0, backoff * 2.0);
            } else {
                backoff = interval;
            }
        }
        self.publisher_running.store(false, Ordering::SeqCst);
    }

    async fn consumer_loop(self: Arc<Self>) {
        self.consumer_running.store(true, Ordering::SeqCst);
        let mut backoff = 0.5f64;
        while !self.stopped() {
            if !self.transport.connected() {
                self.sleep(0.5).await;
                continue;
            }
            let deliveries =
                match self.transport.fetch(self.settings.consume_batch.max(1) as usize, Duration::from_secs(1)).await {
                    Ok(d) => d,
                    Err(e) => {
                        tracing::warn!("fetch failed: {e}");
                        self.sleep(backoff).await;
                        backoff = f64::min(10.0, backoff * 2.0);
                        continue;
                    }
                };
            backoff = 0.5;
            if deliveries.is_empty() {
                self.idle.store(true, Ordering::SeqCst);
                continue;
            }
            self.idle.store(false, Ordering::SeqCst);
            for d in deliveries {
                self.handle_delivery(d).await;
            }
            self.idle.store(true, Ordering::SeqCst);
        }
        self.consumer_running.store(false, Ordering::SeqCst);
    }

    pub async fn handle_delivery(self: &Arc<Self>, d: Box<dyn crate::transport::Delivery>) {
        let t0 = Instant::now();
        let seq = d.seq();
        if self.settings.event_signing_key.as_deref().is_some_and(|k| !k.is_empty())
            && !self.verify(d.data(), d.header("Mycelic-Signature").as_deref())
        {
            // not ours: whoever published it did not hold the signing key.  Drop it loudly and permanently.
            self.metrics.events_failed.with_label_values(&["signature"]).inc();
            tracing::error!("event at seq {seq:?} has a missing or invalid signature; terminated");
            let _ = d.term().await;
            let (svc, subject) = (self.clone(), d.subject().to_string());
            let _ = tokio::task::spawn_blocking(move || {
                svc.store.transaction(|tx| {
                    tx.audit(
                        "mycelic",
                        "event.rejected",
                        None,
                        json!({"reason": "invalid signature", "seq": seq, "subject": subject}),
                        None,
                    )?;
                    svc.progress_in_tx(tx, seq)
                })
            })
            .await;
            self.note_progress(seq);
            return;
        }
        let parsed: anyhow::Result<Map<String, Value>> = (|| {
            let v: Value = serde_json::from_slice(d.data())?;
            match v {
                Value::Object(m) if m.get("event_id").is_some_and(Value::is_string) => Ok(m),
                _ => Err(ServiceError::Validation("event without event_id".into()).into()),
            }
        })();
        let outcome: anyhow::Result<(String, String)> = match parsed {
            Err(e) => Err(e),
            Ok(event) => {
                let svc = self.clone();
                let kind = event.get("kind").and_then(Value::as_str).unwrap_or("?").to_string();
                match tokio::task::spawn_blocking(move || svc.apply_event(&event, seq)).await {
                    Ok(Ok(result)) => Ok((kind, result)),
                    Ok(Err(e)) => Err(e),
                    Err(join) => Err(anyhow!("apply panicked: {join}")),
                }
            }
        };
        match outcome {
            Ok((kind, result)) => {
                let _ = d.ack().await;
                self.metrics.aggregation_latency.observe(t0.elapsed().as_secs_f64());
                self.metrics.events_applied.with_label_values(&[&kind, &result]).inc();
                self.note_progress(seq);
            }
            Err(exc) => {
                self.metrics.events_failed.with_label_values(&["apply"]).inc();
                let text = error_text(&exc);
                let n = d.num_delivered();
                if n >= self.settings.nats_max_deliver {
                    tracing::error!("event at seq {seq:?} failed {n} times, terminating: {text}");
                    let _ = d.term().await;
                    let eid: Option<String> = serde_json::from_slice::<Value>(d.data())
                        .ok()
                        .and_then(|v| v.get("event_id").and_then(Value::as_str).map(String::from));
                    let svc = self.clone();
                    let _ = tokio::task::spawn_blocking(move || {
                        svc.store.transaction(|tx| {
                            if let Some(eid) = &eid
                                && tx.event_status(eid)?.is_some()
                            {
                                tx.mark_failed(eid, &text)?;
                            }
                            tx.audit(
                                "mycelic",
                                "event.failed",
                                eid.as_deref(),
                                json!({"error": text, "seq": seq}),
                                None,
                            )?;
                            svc.progress_in_tx(tx, seq)
                        })
                    })
                    .await;
                    self.note_progress(seq);
                } else {
                    tracing::warn!("event at seq {seq:?} failed (attempt {n}): {text}");
                    // back off *before* the nak so the message stays in flight and nothing overtakes it
                    self.sleep(f64::min(10.0, 0.25 * 2f64.powi(n.clamp(0, 30) as i32))).await;
                    let _ = d.nak().await;
                }
            }
        }
    }

    /// Record the stream position inside the transaction that consumed it (applied, rejected or terminated).
    fn progress_in_tx(&self, tx: &Tx, seq: Option<i64>) -> anyhow::Result<()> {
        let Some(seq) = seq else { return Ok(()) };
        tx.set_meta("last_applied_seq", &seq.to_string())?;
        if self.replay_target().is_some_and(|t| seq >= t) {
            tx.delete_meta("replay_target_seq")?;
        }
        Ok(())
    }

    fn note_progress(&self, seq: Option<i64>) {
        let mut target = self.replay_target.lock().unwrap_or_else(|e| e.into_inner());
        let Some(t) = *target else { return };
        self.metrics.replay_events.inc();
        if let Some(s) = seq
            && s >= t
        {
            tracing::info!("replay complete at seq {s}");
            *target = None;
        }
    }

    /// Wait until the outbox is empty and the consumer has nothing pending (tests, demos, CLI).
    pub async fn wait_idle(&self, timeout: Duration) -> bool {
        let deadline = Instant::now() + timeout;
        let pending = |svc: &Self| {
            svc.store.stats().ok().and_then(|s| s.get("outbox_pending").and_then(Value::as_i64)).unwrap_or(0)
        };
        while Instant::now() < deadline {
            let info = self.transport.info().await;
            if pending(self) == 0
                && self.idle.load(Ordering::SeqCst)
                && info_i64(&info, "consumer_pending").unwrap_or(0) == 0
                && self.replay_target().is_none()
            {
                // one more fetch round so a just-published event is applied
                tokio::time::sleep(Duration::from_millis(250)).await;
                let info = self.transport.info().await;
                if pending(self) == 0
                    && self.idle.load(Ordering::SeqCst)
                    && info_i64(&info, "consumer_pending").unwrap_or(0) == 0
                {
                    return true;
                }
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        false
    }

    // ------------------------------------------------------------------ wire format
    pub fn event_wire(ev: &EventRecord) -> Value {
        json!({"event_id": ev.event_id, "kind": ev.kind, "org_id": ev.org_id, "agent_id": ev.agent_id,
               "created_at": ev.created_at, "payload": ev.payload, "schema": 1})
    }

    fn mac(&self, wire: &[u8]) -> Option<String> {
        let key = self.settings.event_signing_key.as_deref().filter(|k| !k.is_empty())?;
        let mut mac = Hmac::<Sha256>::new_from_slice(key.as_bytes()).expect("HMAC accepts any key length");
        mac.update(wire);
        Some(format!("v1={}", hex::encode(mac.finalize().into_bytes())))
    }

    /// HMAC-SHA256 over the exact bytes published, so the consumer can tell its own events from injected ones.
    pub fn sign(&self, wire: &[u8]) -> Vec<(String, String)> {
        self.mac(wire).map(|sig| vec![("Mycelic-Signature".to_string(), sig)]).unwrap_or_default()
    }

    pub fn verify(&self, wire: &[u8], header: Option<&str>) -> bool {
        let Some(expected) = self.mac(wire) else { return true };
        constant_time_eq(header.unwrap_or("").as_bytes(), expected.as_bytes())
    }

    /// What a caller may see of a memory: metadata that names other teams' agents is reserved for admins.
    pub fn public_view(&self, m: &Value, principal: &Principal) -> Value {
        let mut d = as_object(m);
        let own = d.get("layer").and_then(Value::as_str) == Some("agent")
            && d.get("producer_id").and_then(Value::as_str) == Some(principal.id.as_str());
        if principal.is_admin() || own {
            return Value::Object(d);
        }
        let mut allowed = vec![
            "agg_key",
            "child_layer",
            "parent_count",
            "version_of",
            "fragility",
            "contributing_teams",
            "children",
            "promoted_from",
            "effective_min_support",
            "corroborated_units",
            "rule_chain",
        ];
        if principal.has("lineage:read") {
            // the root ids are exactly what GET /lineage/{id} shows this caller
            allowed.push("roots");
        }
        let meta = as_object(d.get("metadata").unwrap_or(&Value::Null));
        let filtered: Map<String, Value> = meta.into_iter().filter(|(k, _)| allowed.contains(&k.as_str())).collect();
        d.insert("metadata".into(), Value::Object(filtered));
        d.insert("local_ref".into(), Value::Null);
        Value::Object(d)
    }

    // ------------------------------------------------------------------ authentication
    pub fn authenticate(&self, authorization: Option<&str>) -> SResult<Result<Principal, AuthError>> {
        let p = match authenticate(&self.store, self.settings.admin_token.as_deref(), authorization)? {
            Ok(p) => p,
            Err(e) => return Ok(Err(e)),
        };
        if p.kind == "agent" {
            let due = {
                let mut touched = self.last_seen_touch.lock().unwrap_or_else(|e| e.into_inner());
                let due = touched.get(&p.id).is_none_or(|t| t.elapsed() > Duration::from_secs(30));
                if due {
                    touched.insert(p.id.clone(), Instant::now());
                }
                due
            };
            if due {
                // synchronous, so the next request sees it (Python's touch task always ran before the next one)
                let at = self.store.now();
                if let Err(e) = self.store.transaction(|tx| tx.touch_agent(&p.id, &at)) {
                    tracing::warn!("recording last_seen_at for {} failed: {e}", p.id);
                }
            }
        }
        Ok(Ok(p))
    }

    // ------------------------------------------------------------------ ingest
    fn validate_memory_body(
        &self,
        principal: &Principal,
        body: &Map<String, Value>,
        extra_source_events: &[String],
    ) -> SResult<Memory> {
        if principal.kind != "agent" || principal.agent.is_none() {
            return Err(ServiceError::Forbidden(
                "only agents can write memories (administrators register agents)".into(),
            ));
        }
        if !principal.has("memory:write") {
            return Err(ServiceError::Forbidden("missing scope memory:write".into()));
        }
        if let Some(claimed) = field_str(body, "agent_id", false, 200, None)?
            && claimed != principal.id
        {
            return Err(ServiceError::Forbidden("agents may only write as themselves".into()));
        }
        let text =
            field_str(body, "text", true, self.settings.max_text_chars.max(0) as usize, None)?.unwrap_or_default();
        let kind = field_str(body, "kind", false, 40, None)?.unwrap_or_else(|| "observation".into());
        if !MEMORY_KINDS.contains(&kind.as_str()) {
            return Err(validation(format!("'kind' must be one of {}", repr_tuple(&MEMORY_KINDS))));
        }
        let visibility = field_str(body, "visibility", false, 10, None)?.unwrap_or_else(|| "team".into());
        if !VISIBILITY.contains(&visibility.as_str()) {
            return Err(validation(format!("'visibility' must be one of {}", repr_tuple(&VISIBILITY))));
        }
        let srcs: Vec<String> = match body.get("source_event_ids") {
            None => Vec::new(),
            Some(v) if !truthy(v) => Vec::new(),
            Some(Value::Array(items)) if items.len() <= 50 && items.iter().all(|x| x.as_str().is_some_and(id_ok)) => {
                items.iter().filter_map(|x| x.as_str().map(String::from)).collect()
            }
            Some(_) => return Err(validation("'source_event_ids' must be a list of at most 50 ids")),
        };
        if !srcs.is_empty() {
            let found = self.store.read(|db| db.get_events(&srcs))?;
            // one message for unknown and foreign ids alike, so the check is not an oracle for other organizations
            if srcs.iter().any(|eid| found.get(eid).is_none_or(|e| Some(&e.org_id) != principal.org_id.as_ref())) {
                return Err(validation("'source_event_ids' must reference events recorded by your organization"));
            }
        }
        let mut all_srcs: Vec<String> = Vec::new();
        for s in srcs.iter().chain(extra_source_events) {
            if !all_srcs.contains(s) {
                all_srcs.push(s.clone());
            }
        }
        let meta: Map<String, Value> = field_small_dict(body, "metadata", 4096)?
            .into_iter()
            .filter(|(k, _)| !RESERVED_METADATA_KEYS.contains(&k.as_str()))
            .collect();
        let observed_at = match field_iso(body, "observed_at")? {
            Some(t) => t,
            None => self.now(),
        };
        let idem = field_str(body, "idempotency_key", false, 200, None)?;
        let memory_id = match &idem {
            Some(k) => format!("mem_{}", &content_hash(&[&principal.id, k])[..22]),
            None => (self.store.ids)("mem"),
        };
        let topic = field_str(body, "topic", false, 200, None)?;
        let slot = field_str(body, "slot", false, 100, Some((SLOT_PATTERN, slot_ok)))?;
        let entity = field_str(body, "entity", false, 200, None)?;
        let confidence = field_f(body, "confidence", 0.8)?;
        let local_ref = field_str(body, "local_ref", false, 200, None)?;
        Ok(Memory {
            memory_id,
            org_id: principal.org_id.clone().unwrap_or_default(),
            layer: "agent".into(),
            scope: principal.path.clone().unwrap_or_default(),
            text,
            topic,
            slot,
            entity,
            kind,
            confidence,
            support: 1,
            independent_teams: 1,
            producer_id: principal.id.clone(),
            operator: "agent_observation".into(),
            rule_id: None,
            event_id: None,
            visibility,
            status: "active".into(),
            superseded_by: None,
            created_at: observed_at,
            applied_at: None,
            source_event_ids: all_srcs,
            local_ref,
            metadata: meta,
        })
    }

    fn check_event_size(&self, ev: &EventRecord) -> SResult<()> {
        let size = pyjson::dumps(&Self::event_wire(ev)).len();
        if size as i64 > self.settings.max_event_bytes {
            return Err(validation(format!("event is {size} bytes; the limit is {}", self.settings.max_event_bytes)));
        }
        Ok(())
    }

    /// Store a memory and append its event to the outbox in one transaction.  Returns (memory, created).
    pub fn ingest_memory(&self, principal: &Principal, body: &Value, remote: Option<&str>) -> SResult<(Memory, bool)> {
        let Value::Object(body) = body else { return Err(validation("body must be a JSON object")) };
        let mut m = self.validate_memory_body(principal, body, &[])?;
        let mut event = EventRecord::new(
            (self.store.ids)("evt"),
            "memory.observed",
            &m.org_id,
            Some(principal.id.clone()),
            subject_for(&m.org_id, "memory.observed"),
            json!({}),
            self.now(),
        );
        m.event_id = Some(event.event_id.clone());
        event.payload = m.to_json();
        self.check_event_size(&event)?;
        let result = self.store.transaction(|tx| {
            if let Some(existing) = tx.get_memory(&m.memory_id)? {
                return Ok((existing, false));
            }
            tx.insert_memory(&m)?;
            tx.insert_event(&event)?;
            tx.audit(
                &principal.id,
                "memory.ingest",
                Some(&m.memory_id),
                json!({"org_id": m.org_id, "topic": m.topic, "slot": m.slot, "event_id": event.event_id}),
                remote,
            )?;
            Ok((m.clone(), true))
        })?;
        if result.1 {
            self.metrics.memories_ingested.with_label_values(&["agent"]).inc();
            self.wake_outbox();
        }
        Ok(result)
    }

    pub fn ingest_events(
        &self,
        principal: &Principal,
        items: Option<&Value>,
        remote: Option<&str>,
    ) -> SResult<Vec<Value>> {
        if principal.kind != "agent" {
            return Err(ServiceError::Forbidden("only agents can publish events".into()));
        }
        if !principal.has("events:write") {
            return Err(ServiceError::Forbidden("missing scope events:write".into()));
        }
        let items = match items {
            Some(Value::Array(a)) if !a.is_empty() => a,
            _ => return Err(validation("'events' must be a non-empty list")),
        };
        if items.len() as i64 > self.settings.max_batch {
            return Err(validation(format!("at most {} events per request", self.settings.max_batch)));
        }
        let org = principal.org_id.clone().unwrap_or_default();
        let mut prepared: Vec<(EventRecord, Option<Memory>)> = Vec::new();
        for (i, item) in items.iter().enumerate() {
            let Value::Object(item) = item else { return Err(validation(format!("events[{i}] must be an object"))) };
            let etype = field_str(item, "type", true, 100, Some((ID_PATTERN, id_ok)))?.unwrap_or_default();
            let occurred_at = match field_iso(item, "occurred_at")? {
                Some(t) => t,
                None => self.now(),
            };
            let payload = field_small_dict(item, "payload", 16 * 1024)?;
            let idem = field_str(item, "idempotency_key", false, 200, None)?;
            let event_id = match &idem {
                Some(k) => format!("evt_{}", &content_hash(&[&principal.id, "event", k])[..22]),
                None => (self.store.ids)("evt"),
            };
            let record = EventRecord::new(
                event_id.clone(),
                "agent.event",
                &org,
                Some(principal.id.clone()),
                subject_for(&org, "agent.event"),
                json!({"type": etype, "occurred_at": occurred_at, "agent_id": principal.id, "path": principal.path, "payload": payload}),
                self.now(),
            );
            let mut mem = None;
            if let Some(mv) = item.get("memory").filter(|v| !v.is_null()) {
                let Value::Object(mbody) = mv else {
                    return Err(validation(format!("events[{i}].memory must be an object")));
                };
                let mut m = self.validate_memory_body(principal, mbody, std::slice::from_ref(&event_id))?;
                let own_key = mbody.get("idempotency_key").is_some_and(truthy);
                if let Some(k) = &idem
                    && !own_key
                {
                    // the event's idempotency key also makes its embedded memory idempotent (domain-separated)
                    m.memory_id = format!("mem_{}", &content_hash(&[&principal.id, "event-memory", k])[..22]);
                }
                mem = Some(m);
            }
            self.check_event_size(&record)?;
            prepared.push((record, mem));
        }
        let count = prepared.len();
        let (results, created_memories) = self.store.transaction(|tx| {
            let mut results = Vec::new();
            let mut created = 0;
            for (record, mem) in &prepared {
                let was_new = tx.insert_event(record)?;
                let mut entry = Map::new();
                entry.insert("event_id".into(), json!(record.event_id));
                entry.insert("created".into(), json!(was_new));
                if let Some(mem) = mem {
                    let mut mem = mem.clone();
                    let mut mev = EventRecord::new(
                        tx.new_id("evt"),
                        "memory.observed",
                        &mem.org_id,
                        Some(principal.id.clone()),
                        subject_for(&mem.org_id, "memory.observed"),
                        json!({}),
                        tx.now(),
                    );
                    mem.event_id = Some(mev.event_id.clone());
                    mev.payload = mem.to_json();
                    if tx.get_memory(&mem.memory_id)?.is_none() {
                        tx.insert_memory(&mem)?;
                        tx.insert_event(&mev)?;
                        created += 1;
                        entry.insert("memory_id".into(), json!(mem.memory_id));
                        entry.insert("memory_event_id".into(), json!(mev.event_id));
                    } else {
                        entry.insert("memory_id".into(), json!(mem.memory_id));
                        entry.insert("memory_created".into(), json!(false));
                    }
                }
                results.push(Value::Object(entry));
            }
            tx.audit(
                &principal.id,
                "events.ingest",
                None,
                json!({"org_id": principal.org_id, "count": count}),
                remote,
            )?;
            Ok((results, created))
        })?;
        for _ in 0..created_memories {
            self.metrics.memories_ingested.with_label_values(&["agent"]).inc();
        }
        self.metrics.events_received.inc_by(count as u64);
        self.wake_outbox();
        Ok(results)
    }

    pub fn retract(
        &self,
        principal: &Principal,
        memory_id: &str,
        reason: &str,
        remote: Option<&str>,
    ) -> SResult<EventRecord> {
        let m = self.store.get_memory(memory_id)?;
        let Some(m) = m.filter(|m| principal.can_read(m)) else { return Err(ServiceError::NotFound(memory_id.into())) };
        if !principal.is_admin() && m.producer_id != principal.id {
            return Err(ServiceError::Forbidden(
                "only the producing agent or an administrator can retract a memory".into(),
            ));
        }
        if m.operator != "agent_observation" {
            return Err(validation(
                "derived memories are recomputed from their evidence and cannot be retracted directly; \
                 retract the contributing observations (the roots in GET /lineage/{id}) instead",
            ));
        }
        let reason: String = reason.chars().take(200).collect();
        let event = EventRecord::new(
            (self.store.ids)("evt"),
            "memory.retracted",
            &m.org_id,
            Some(principal.id.clone()),
            subject_for(&m.org_id, "memory.retracted"),
            json!({"memory_id": memory_id, "reason": reason, "by": principal.id}),
            self.now(),
        );
        self.store.transaction(|tx| {
            tx.insert_event(&event)?;
            tx.audit(
                &principal.id,
                "memory.retract",
                Some(memory_id),
                json!({"org_id": m.org_id, "reason": reason}),
                remote,
            )
        })?;
        self.wake_outbox();
        Ok(event)
    }

    // ------------------------------------------------------------------ apply (consumer side)
    /// Apply one event from the log.  Idempotent: an already-applied event id is a no-op ('duplicate').
    pub fn apply_event(&self, event: &Map<String, Value>, seq: Option<i64>) -> anyhow::Result<String> {
        let event_id = event
            .get("event_id")
            .and_then(Value::as_str)
            .ok_or_else(|| ServiceError::Validation("event without event_id".into()))?;
        let kind: Option<&str> = match event.get("kind") {
            None | Some(Value::Null) => None,
            Some(Value::String(s)) => Some(s.as_str()),
            Some(_) => return Err(ServiceError::Validation("event kind must be a string".into()).into()),
        };
        let payload: Map<String, Value> = match event.get("payload") {
            Some(v) if truthy(v) => match v {
                Value::Object(m) => m.clone(),
                _ => return Err(ServiceError::Validation("event payload must be an object".into()).into()),
            },
            _ => Map::new(),
        };
        let org_id = match event.get("org_id").filter(|v| truthy(v)) {
            Some(v) => py_str(v),
            None => str_or(payload.get("org_id"), ""),
        };
        let replaying = self.replay_target().is_some();
        let (result, derived) = self.store.transaction(|tx| {
            let now = tx.now();
            let status = tx.event_status(event_id)?;
            if status.as_deref() == Some("applied") {
                if seq.is_some() {
                    tx.mark_applied(event_id, seq, &now)?;
                    self.progress_in_tx(tx, seq)?;
                }
                return Ok(("duplicate".to_string(), false));
            }
            if status.is_none() {
                // Not in this database: the event came from the stream (rebuild after data loss, or another writer).
                let k = kind.filter(|k| !k.is_empty()).unwrap_or("unknown");
                let mut rec = EventRecord::new(
                    event_id.to_string(),
                    k,
                    &org_id,
                    opt_str(event.get("agent_id")),
                    subject_for(if org_id.is_empty() { "unknown" } else { &org_id }, k),
                    Value::Object(payload.clone()),
                    str_or(event.get("created_at"), &now),
                );
                rec.status = "published".into();
                rec.js_seq = seq;
                tx.insert_event(&rec)?;
            }
            let mut derivations: Vec<Derivation> = Vec::new();
            let mut result = "applied".to_string();
            match kind {
                Some("memory.observed") => {
                    let mut m = memory_from_payload(&payload, &now)?;
                    m.applied_at = Some(now.clone());
                    let agent = tx.get_agent(&m.producer_id)?;
                    let registered = agent.as_ref().is_some_and(|a| a.org_id == m.org_id && a.path == m.scope) && m.layer == "agent";
                    if !registered {
                        // Anything that can publish to the stream could claim to be an agent; the registry is the
                        // authority.  (Registrations travel through the same log ahead of the agent's first memory.)
                        tx.audit(
                            "mycelic",
                            "event.rejected",
                            Some(event_id),
                            json!({"reason": "producer not registered for this scope", "producer_id": m.producer_id, "scope": m.scope}),
                            None,
                        )?;
                        tx.mark_applied(event_id, seq, &now)?;
                        self.progress_in_tx(tx, seq)?;
                        self.metrics.events_failed.with_label_values(&["validate"]).inc();
                        return Ok(("rejected".to_string(), false));
                    }
                    if tx.insert_memory(&m)? {
                        // replayed memories were counted when first ingested
                        self.metrics.memories_ingested.with_label_values(&["agent"]);
                    } else {
                        tx.set_applied(&m.memory_id, &now)?;
                        if let Some(stored) = tx.get_memory(&m.memory_id)? {
                            m = stored;
                        }
                    }
                    if m.status == "active" {
                        derivations = self.aggregator.derive_for(tx, &m)?;
                    }
                }
                Some("memory.derived") => {
                    let mut m = memory_from_payload(&payload, &now)?;
                    if m.applied_at.as_deref().is_none_or(str::is_empty) {
                        m.applied_at = Some(now.clone());
                    }
                    if tx.insert_memory(&m)? {
                        let mut edges = Vec::new();
                        if let Some(Value::Array(parents)) = payload.get("parents") {
                            for p in parents {
                                let Value::Object(p) = p else { continue };
                                let Some(pid) = p.get("parent_id").filter(|v| truthy(v)) else { continue };
                                let need = |k: &str| p.get(k).map(py_str).ok_or_else(|| anyhow!("KeyError: {}", pyjson::repr_str(k)));
                                edges.push(LineageEdge {
                                    child_id: m.memory_id.clone(),
                                    parent_id: py_str(pid),
                                    contributed_by: need("contributed_by")?,
                                    parent_layer: need("parent_layer")?,
                                    created_at: str_or(p.get("created_at"), &now),
                                });
                            }
                        }
                        tx.add_lineage_edges(&edges)?;
                        if let Some(Value::String(old)) = payload.get("supersedes")
                            && let Some(prev) = tx.get_memory(old)?
                            && prev.status == "active"
                        {
                            tx.set_memory_status(old, "superseded", Some(&m.memory_id), Some("coalition changed"))?;
                        }
                        self.metrics.derived.with_label_values(&[&m.layer, &m.operator]).inc();
                    } else {
                        result = "duplicate".into();
                    }
                }
                Some("memory.retracted") => {
                    let target = match payload.get("memory_id") {
                        Some(Value::String(mid)) => tx.get_memory(mid)?,
                        _ => None,
                    };
                    match target {
                        Some(m) if m.status == "active" && m.operator == "agent_observation" => {
                            let reason = str_or(payload.get("reason"), "retracted");
                            tx.set_memory_status(&m.memory_id, "retracted", None, Some(&reason))?;
                            let retired = self.aggregator.retire_dependents(tx, &m.memory_id, "evidence retracted")?;
                            derivations = self.aggregator.derive_for(tx, &m)?;
                            // a conclusion that lost one piece of evidence may still hold on the rest: re-evaluate it
                            derivations.extend(self.aggregator.reevaluate(tx, &retired, 1)?);
                        }
                        _ => {
                            // derived memories are a function of their evidence: they can only go away with it
                            tx.audit("mycelic", "event.ignored", Some(event_id), json!({"reason": "retraction target is not an active raw observation"}), None)?;
                            result = "ignored".into();
                        }
                    }
                }
                Some("agent.event") => {}
                Some("agent.registered") => {
                    let agent = agent_from_payload(&payload, &now)?;
                    if tx.get_agent(&agent.agent_id)?.is_none() {
                        tx.insert_agent(&agent, &str_or(payload.get("key_hash"), ""))?;
                    } else {
                        result = "duplicate".into();
                    }
                }
                Some("agent.revoked") => {
                    tx.set_agent_status(&py_str(payload.get("agent_id").unwrap_or(&Value::Null)), "revoked")?;
                }
                Some("agent.key_rotated") => {
                    tx.rotate_agent_key(
                        &py_str(payload.get("agent_id").unwrap_or(&Value::Null)),
                        &str_or(payload.get("key_hash"), ""),
                        &str_or(payload.get("key_prefix"), ""),
                    )?;
                }
                Some("rule.upserted") => {
                    let rule = self.rule_from_body(&Value::Object(payload.clone()))?;
                    tx.upsert_rule(&rule)?;
                }
                Some("rule.deleted") => {
                    tx.delete_rule(&py_str(payload.get("rule_id").unwrap_or(&Value::Null)))?;
                }
                _ => result = "ignored".into(),
            }
            for d in &mut derivations {
                let mut dev = EventRecord::new(
                    format!("evt_d{}", &content_hash(&[&d.memory.memory_id])[..22]),
                    "memory.derived",
                    &d.memory.org_id,
                    None,
                    subject_for(&d.memory.org_id, "memory.derived"),
                    json!({}),
                    now.clone(),
                );
                if replaying {
                    // During a replay the stream already carries the original derived events (they follow the
                    // observations that produced them), so re-deriving must not append duplicates to the log.
                    dev.status = "published".into();
                    dev.published_at = Some(now.clone());
                }
                d.memory.event_id = Some(dev.event_id.clone());
                dev.payload = d.event_payload();
                tx.set_memory_event(&d.memory.memory_id, &dev.event_id)?;
                tx.insert_event(&dev)?;
                self.metrics.derived.with_label_values(&[&d.memory.layer, &d.memory.operator]).inc();
            }
            tx.mark_applied(event_id, seq, &now)?;
            self.progress_in_tx(tx, seq)?;
            Ok((result, !derivations.is_empty()))
        })?;
        if derived {
            self.wake_outbox();
        }
        Ok(result)
    }

    // ------------------------------------------------------------------ reads
    pub fn get_memory(&self, principal: &Principal, memory_id: &str) -> SResult<Memory> {
        if !principal.has("memory:read") {
            return Err(ServiceError::Forbidden("missing scope memory:read".into()));
        }
        match self.store.get_memory(memory_id)? {
            Some(m) if principal.can_read(&m) => Ok(m),
            _ => Err(ServiceError::NotFound(memory_id.into())),
        }
    }

    pub fn query(&self, principal: &Principal, body: &Value) -> SResult<Value> {
        if !principal.has("memory:read") {
            return Err(ServiceError::Forbidden("missing scope memory:read".into()));
        }
        let Value::Object(body) = body else { return Err(validation("body must be a JSON object")) };
        let t0 = Instant::now();
        let text = match field_str(body, "query", false, 2000, None)? {
            Some(t) => t,
            None => field_str(body, "text", false, 2000, None)?.unwrap_or_default(),
        };
        let mut scope = field_str(body, "scope", false, 400, None)?;
        let mut org_id = principal.org_id.clone();
        if let Some(s) = &scope {
            split_path(s).map_err(|e| validation(e.0))?;
            if principal.is_admin() {
                org_id = Some(s.split('/').next().unwrap_or("").to_string());
            } else if !principal.can_query_scope(s) {
                return Err(ServiceError::Forbidden("scope is outside the units this agent belongs to".into()));
            }
        } else {
            scope = principal.org_id.clone();
            if principal.is_admin() {
                org_id = field_str(body, "org_id", false, 64, None)?;
                if org_id.is_none() {
                    return Err(validation("administrators must pass 'scope' or 'org_id'"));
                }
                scope = org_id.clone();
            }
        }
        let min_layer = field_str(body, "min_layer", false, 20, None)?.unwrap_or_else(|| "agent".into());
        layer_index(&min_layer)?;
        let k = match body.get("k") {
            None => 10,
            Some(Value::Number(n)) if n.is_i64() || n.is_u64() => match n.as_i64() {
                Some(k) if (1..=100).contains(&k) => k,
                _ => return Err(validation("'k' must be an integer between 1 and 100")),
            },
            Some(_) => return Err(validation("'k' must be an integer between 1 and 100")),
        };
        let include_lineage = body.get("include_lineage").is_none_or(truthy) && principal.has("lineage:read");
        let topic = field_str(body, "topic", false, 200, None)?;
        let entity = field_str(body, "entity", false, 200, None)?;
        let org = org_id.clone().unwrap_or_default();
        let revision = self.store.revision();
        let index = self
            .retriever
            .index(&org, revision, || self.store.read(|db| db.visible_rows(&org, None, None, None, 0, &["active"])))?;
        let params = SearchParams {
            scope: scope.as_deref(),
            min_layer: &min_layer,
            k: k as usize,
            topic: topic.as_deref(),
            entity: entity.as_deref(),
        };
        let hits = search_index(&index, &text, &|m| principal.can_read(m), &params);
        let mut answer = Value::Null;
        let mut lineage = Value::Null;
        if let Some(top) = hits.first().map(|h| &h.memory) {
            let summary = self.lineage_summary(principal, top)?;
            answer = json!({"memory_id": top.memory_id, "text": top.text, "layer": top.layer, "scope": top.scope,
                            "confidence": top.confidence, "support": top.support, "independent_teams": top.independent_teams,
                            "operator": top.operator, "rule_id": top.rule_id, "created_at": top.created_at, "lineage": summary});
            if include_lineage {
                lineage = self.lineage(principal, &top.memory_id)?;
            }
        }
        let latency = t0.elapsed().as_secs_f64();
        self.metrics.retrieval_latency.observe(latency);
        Ok(json!({"query": text, "scope": scope, "min_layer": min_layer, "answer": answer,
                  "results": hits.iter().map(|h| h.to_json()).collect::<Vec<_>>(), "lineage": lineage,
                  "latency_ms": pyjson::round(latency * 1000.0, 2)}))
    }

    fn lineage_summary(&self, principal: &Principal, m: &Memory) -> SResult<Value> {
        let g = self
            .store
            .read(|db| reconstruct(db, &m.memory_id, &|x| principal.can_read(x), &|x| principal.owns(x), MAX_NODES))?;
        let Ok(g) = g else { return Ok(json!({"available": false})) };
        let len = |k: &str| g.get(k).and_then(Value::as_array).map(Vec::len).unwrap_or(0);
        let redacted = g.get("redacted_contributions").and_then(Value::as_u64).unwrap_or(0) as usize;
        Ok(json!({"available": true, "contributing_agents": len("contributing_agents") + redacted,
                  "contributing_teams": len("contributing_teams"), "roots": len("roots"), "layers": g["layers"],
                  "reconstructable": g["evidence"]["reconstructable"], "first_observed_at": g["timeline"]["first_observed_at"]}))
    }

    pub fn lineage(&self, principal: &Principal, memory_id: &str) -> SResult<Value> {
        if !principal.has("lineage:read") {
            return Err(ServiceError::Forbidden("missing scope lineage:read".into()));
        }
        match self.store.get_memory(memory_id)? {
            Some(m) if principal.can_read(&m) => {}
            _ => {
                self.metrics.lineage_results.with_label_values(&["not_found"]).inc();
                return Err(ServiceError::NotFound(memory_id.into()));
            }
        }
        let t0 = Instant::now();
        let g = match self
            .store
            .read(|db| reconstruct(db, memory_id, &|x| principal.can_read(x), &|x| principal.owns(x), MAX_NODES))
        {
            Ok(Ok(g)) => g,
            Ok(Err(LineageNotFound(id))) => {
                self.metrics.lineage_results.with_label_values(&["failure"]).inc();
                return Err(ServiceError::NotFound(id));
            }
            Err(e) => {
                self.metrics.lineage_results.with_label_values(&["failure"]).inc();
                return Err(e.into());
            }
        };
        self.metrics.lineage_latency.observe(t0.elapsed().as_secs_f64());
        let ok = g["evidence"]["reconstructable"].as_bool().unwrap_or(false);
        self.metrics.lineage_results.with_label_values(&[if ok { "success" } else { "incomplete" }]).inc();
        Ok(g)
    }

    pub fn list_memories(
        &self,
        principal: &Principal,
        scope: Option<&str>,
        layers: Option<&[String]>,
        limit: i64,
        status: &str,
    ) -> SResult<Vec<Memory>> {
        if !principal.has("memory:read") {
            return Err(ServiceError::Forbidden("missing scope memory:read".into()));
        }
        let mut org_id = principal.org_id.clone();
        if let Some(s) = scope.filter(|s| !s.is_empty()) {
            split_path(s)?;
            if principal.is_admin() {
                org_id = Some(s.split('/').next().unwrap_or("").to_string());
            } else if !principal.can_query_scope(s) {
                return Err(ServiceError::Forbidden("scope is outside the units this agent belongs to".into()));
            }
        }
        let Some(org_id) = org_id else { return Err(validation("administrators must pass 'scope'")) };
        let mut rows = self.store.read(|db| {
            db.visible_rows(
                &org_id,
                principal.path.as_deref(),
                principal.team_path.as_deref(),
                scope.filter(|s| !s.is_empty()),
                0,
                &[status],
            )
        })?;
        if let Some(layers) = layers.filter(|l| !l.is_empty()) {
            rows.retain(|m| layers.contains(&m.layer));
        }
        rows.sort_by(|a, b| b.created_at.cmp(&a.created_at));
        rows.truncate(limit.clamp(1, 500) as usize);
        Ok(rows)
    }

    // ------------------------------------------------------------------ admin
    // Registrations, revocations, key rotations and rules are written to the database *and* appended to the event
    // log in the same transaction, so a database rebuilt from the stream has its agents and rules back (the stream
    // carries key hashes, never keys).  `apply` treats these events as idempotent upserts.
    fn admin_event(&self, tx: &Tx, kind: &str, org_id: &str, payload: Value) -> EventRecord {
        EventRecord::new(tx.new_id("evt"), kind, org_id, None, subject_for(org_id, kind), payload, tx.now())
    }

    pub fn register_agent(&self, body: &Value, remote: Option<&str>) -> SResult<(Agent, String)> {
        let Value::Object(body) = body else { return Err(validation("body must be a JSON object")) };
        let enterprise = match field_str(body, "enterprise", false, 64, None)? {
            Some(e) => e,
            None => field_str(body, "org_id", false, 64, None)?.unwrap_or_default(),
        };
        let region = field_str(body, "region", false, 64, None)?;
        let subsidiary = field_str(body, "subsidiary", false, 64, None)?;
        let department = field_str(body, "department", false, 64, None)?;
        let team = field_str(body, "team", false, 64, None)?;
        let agent_id = field_str(body, "agent_id", true, 64, None)?.unwrap_or_default();
        let ap = AgentPath::from_parts(
            &enterprise,
            region.as_deref(),
            subsidiary.as_deref(),
            department.as_deref(),
            team.as_deref(),
            &agent_id,
        )
        .map_err(|e| validation(e.0))?;
        if ap.agent_id == MYCELIC_PRODUCER {
            return Err(validation(format!("agent_id '{MYCELIC_PRODUCER}' is reserved for derived memories")));
        }
        let grantable: Vec<&str> = ALL_SCOPES.iter().copied().filter(|s| *s != "admin").collect();
        let scopes: Vec<String> = match body.get("scopes") {
            None | Some(Value::Null) => DEFAULT_AGENT_SCOPES.iter().map(|s| s.to_string()).collect(),
            Some(Value::Array(items)) if items.iter().all(|s| s.as_str().is_some_and(|s| grantable.contains(&s))) => {
                items.iter().filter_map(|s| s.as_str().map(String::from)).collect()
            }
            Some(_) => return Err(validation(format!("'scopes' must be a subset of {}", repr_list(&grantable)))),
        };
        let display = field_str(body, "display_name", false, 120, None)?.unwrap_or_else(|| ap.agent_id.clone());
        let (key, key_hash, prefix) = generate_api_key(&ap.agent_id);
        let metadata = field_small_dict(body, "metadata", 2048)?;
        let agent = Agent {
            agent_id: ap.agent_id.clone(),
            org_id: ap.org_id().to_string(),
            display_name: display,
            path: ap.path(),
            scopes: scopes.clone(),
            key_prefix: prefix,
            status: "active".into(),
            created_at: self.now(),
            last_seen_at: None,
            metadata,
        };
        let mut payload = as_object(&agent.to_json());
        payload.insert("key_hash".into(), json!(key_hash));
        let event = EventRecord::new(
            (self.store.ids)("evt"),
            "agent.registered",
            &agent.org_id,
            None,
            subject_for(&agent.org_id, "agent.registered"),
            Value::Object(payload),
            self.now(),
        );
        self.store.transaction(|tx| {
            if tx.get_agent(&ap.agent_id)?.is_some() {
                return Err(ServiceError::Validation(format!("agent '{}' already exists", ap.agent_id)).into());
            }
            tx.insert_agent(&agent, &key_hash)?;
            tx.insert_event(&event)?;
            tx.audit(
                "admin",
                "agent.register",
                Some(&agent.agent_id),
                json!({"org_id": agent.org_id, "path": agent.path, "scopes": scopes}),
                remote,
            )
        })?;
        self.wake_outbox();
        Ok((agent, key))
    }

    pub fn revoke_agent(&self, agent_id: &str, remote: Option<&str>) -> SResult<bool> {
        let Some(agent) = self.store.get_agent(agent_id)? else { return Ok(false) };
        let ok = self.store.transaction(|tx| {
            let ok = tx.set_agent_status(agent_id, "revoked")?;
            if ok {
                tx.insert_event(&self.admin_event(tx, "agent.revoked", &agent.org_id, json!({"agent_id": agent_id})))?;
                tx.audit("admin", "agent.revoke", Some(agent_id), json!({"org_id": agent.org_id}), remote)?;
            }
            Ok(ok)
        })?;
        self.wake_outbox();
        Ok(ok)
    }

    pub fn rotate_agent_key(&self, agent_id: &str, remote: Option<&str>) -> SResult<Option<String>> {
        let Some(agent) = self.store.get_agent(agent_id)? else { return Ok(None) };
        let (key, key_hash, prefix) = generate_api_key(agent_id);
        self.store.transaction(|tx| {
            tx.rotate_agent_key(agent_id, &key_hash, &prefix)?;
            tx.insert_event(&self.admin_event(
                tx,
                "agent.key_rotated",
                &agent.org_id,
                json!({"agent_id": agent_id, "key_hash": key_hash, "key_prefix": prefix}),
            ))?;
            tx.audit("admin", "agent.rotate_key", Some(agent_id), json!({"org_id": agent.org_id}), remote)
        })?;
        self.wake_outbox();
        Ok(Some(key))
    }

    pub fn rule_from_body(&self, body: &Value) -> SResult<Rule> {
        Self::parse_rule(body)
    }

    /// Validate a rule body (`POST /admin/rules`, the rules file, a `rule.upserted` event).
    pub fn parse_rule(body: &Value) -> SResult<Rule> {
        let Value::Object(body) = body else { return Err(validation("rule must be an object")) };
        let rule_id = field_str(body, "rule_id", true, 100, Some((ID_PATTERN, id_ok)))?.unwrap_or_default();
        let target = field_str(body, "target_layer", true, 20, None)?.unwrap_or_default();
        if !LAYERS[1..].contains(&target.as_str()) {
            return Err(validation(format!("'target_layer' must be one of {}", repr_tuple(&LAYERS[1..]))));
        }
        let slots: Vec<String> = match body.get("required_slots") {
            Some(Value::Array(items)) if !items.is_empty() && items.iter().all(|s| s.as_str().is_some_and(slot_ok)) => {
                items.iter().filter_map(|s| s.as_str().map(String::from)).collect()
            }
            _ => return Err(validation("'required_slots' must be a non-empty list of slot names")),
        };
        let mut unique: Vec<String> = Vec::new();
        for s in &slots {
            if !unique.contains(s) {
                unique.push(s.clone());
            }
        }
        if unique.len() > 6 {
            return Err(validation("'required_slots' may name at most 6 slots"));
        }
        let conclusion = field_str(body, "conclusion", true, 2000, None)?.unwrap_or_default();
        let kind = field_str(body, "kind", false, 40, None)?.unwrap_or_else(|| "risk".into());
        if !MEMORY_KINDS.contains(&kind.as_str()) {
            return Err(validation(format!("'kind' must be one of {}", repr_tuple(&MEMORY_KINDS))));
        }
        let positive = |name: &str, default: i64| -> SResult<i64> {
            match body.get(name) {
                None => Ok(default),
                Some(Value::Number(n)) if n.is_i64() || n.is_u64() => match n.as_i64() {
                    Some(v) if v >= 1 => Ok(v),
                    _ => Err(validation(format!("'{name}' must be a positive integer"))),
                },
                Some(_) => Err(validation(format!("'{name}' must be a positive integer"))),
            }
        };
        let min_agents = positive("min_agents", 2)?;
        let min_teams = positive("min_teams", 1)?;
        let sources: Vec<String> = match body.get("sources") {
            None => vec!["agent_observation".into()],
            Some(Value::Array(items))
                if !items.is_empty() && items.iter().all(|s| s.as_str().is_some_and(|s| OPERATORS.contains(&s))) =>
            {
                items.iter().filter_map(|s| s.as_str().map(String::from)).collect()
            }
            Some(_) => {
                return Err(validation(format!("'sources' must be a non-empty subset of {}", repr_tuple(&OPERATORS))));
            }
        };
        let emits_slot = field_str(body, "emits_slot", false, 100, Some((SLOT_PATTERN, slot_ok)))?;
        if let Some(e) = &emits_slot
            && slots.contains(e)
        {
            return Err(validation("'emits_slot' must not be one of the rule's own required slots"));
        }
        let min_units_raw = match body.get("min_units") {
            Some(v) if truthy(v) => v.clone(),
            _ => json!({}),
        };
        let ok = match &min_units_raw {
            Value::Object(m) => m.iter().all(|(slot, per)| {
                slots.contains(slot)
                    && match per {
                        Value::Object(per) if !per.is_empty() => per.iter().all(|(layer, n)| {
                            LAYERS[1..].contains(&layer.as_str())
                                && n.as_i64().is_some_and(|n| n >= 1)
                                && (n.is_i64() || n.is_u64())
                        }),
                        _ => false,
                    }
            }),
            _ => false,
        };
        if !ok {
            return Err(validation(format!(
                "'min_units' must map required slots to {{layer: positive integer}} with layers in {}",
                repr_tuple(&LAYERS[1..])
            )));
        }
        let topic_prefix = field_str(body, "topic_prefix", false, 200, None)?;
        let org_id = field_str(body, "org_id", false, 64, None)?;
        let enabled = body.get("enabled").is_none_or(truthy);
        let metadata = field_small_dict(body, "metadata", 2048)?;
        let mut unique_sources: Vec<String> = Vec::new();
        for s in sources {
            if !unique_sources.contains(&s) {
                unique_sources.push(s);
            }
        }
        let emits_topic = field_str(body, "emits_topic", false, 200, None)?;
        Ok(Rule {
            rule_id,
            target_layer: target,
            required_slots: unique,
            conclusion,
            topic_prefix,
            min_agents,
            min_teams,
            kind,
            org_id,
            enabled,
            metadata,
            sources: unique_sources,
            emits_slot,
            emits_topic,
            min_units: parse_min_units(&min_units_raw),
            corroborate: body.get("corroborate").is_some_and(truthy),
        })
    }

    /// Could a conclusion of `p` be evidence for `c`?
    fn rule_feeds(p: &Rule, c: &Rule) -> bool {
        let Some(emits) = p.emits_slot.as_ref().filter(|e| !e.is_empty()) else { return false };
        if !c.required_slots.contains(emits) {
            return false;
        }
        if !c.sources.iter().any(|s| s == "slot_composition" || s == "topic_consolidation") {
            return false;
        }
        if crate::hierarchy::layer_pos(&p.target_layer) > crate::hierarchy::layer_pos(&c.target_layer) {
            return false;
        }
        if let (Some(po), Some(co)) =
            (p.org_id.as_ref().filter(|o| !o.is_empty()), c.org_id.as_ref().filter(|o| !o.is_empty()))
            && po != co
        {
            return false;
        }
        let topic = p
            .emits_topic
            .as_deref()
            .filter(|t| !t.is_empty())
            .or(p.topic_prefix.as_deref().filter(|t| !t.is_empty()))
            .unwrap_or(&p.rule_id);
        c.topic_prefix.as_deref().filter(|t| !t.is_empty()).is_none_or(|prefix| topic.starts_with(prefix))
    }

    /// Refuse a rule that would (transitively) feed on its own conclusions; the aggregator also guards at apply time.
    pub fn check_rule_cycles(&self, new: &Rule) -> SResult<()> {
        if !new.enabled {
            return Ok(());
        }
        let mut graph: Vec<Rule> = self
            .store
            .list_rules(new.org_id.as_deref(), false)?
            .into_iter()
            .filter(|r| r.enabled && r.rule_id != new.rule_id)
            .collect();
        graph.push(new.clone());
        let find = |id: &str| graph.iter().find(|r| r.rule_id == id).expect("rule in graph");
        let mut stack = vec![new.rule_id.clone()];
        let mut seen = std::collections::HashSet::new();
        while let Some(rid) = stack.pop() {
            let from = find(&rid);
            for other in &graph {
                if Self::rule_feeds(from, other) {
                    if other.rule_id == new.rule_id {
                        return Err(validation(format!(
                            "rule '{}' would form a cycle through slot '{}' with '{rid}'",
                            new.rule_id,
                            from.emits_slot.as_deref().unwrap_or("None")
                        )));
                    }
                    if seen.insert(other.rule_id.clone()) {
                        stack.push(other.rule_id.clone());
                    }
                }
            }
        }
        Ok(())
    }

    pub fn upsert_rule(&self, body: &Value, remote: Option<&str>) -> SResult<Rule> {
        let rule = self.rule_from_body(body)?;
        self.check_rule_cycles(&rule)?;
        self.store.transaction(|tx| {
            tx.upsert_rule(&rule)?;
            tx.insert_event(&self.admin_event(
                tx,
                "rule.upserted",
                rule.org_id.as_deref().unwrap_or("_"),
                rule.to_json(),
            ))?;
            tx.audit(
                "admin",
                "rule.upsert",
                Some(&rule.rule_id),
                json!({"target_layer": rule.target_layer, "slots": rule.required_slots}),
                remote,
            )
        })?;
        self.wake_outbox();
        Ok(rule)
    }

    pub fn delete_rule(&self, rule_id: &str, remote: Option<&str>) -> SResult<bool> {
        let Some(rule) = self.store.get_rule(rule_id)? else { return Ok(false) };
        let ok = self.store.transaction(|tx| {
            let ok = tx.delete_rule(rule_id)?;
            if ok {
                tx.insert_event(&self.admin_event(
                    tx,
                    "rule.deleted",
                    rule.org_id.as_deref().unwrap_or("_"),
                    json!({"rule_id": rule_id}),
                ))?;
                tx.audit("admin", "rule.delete", Some(rule_id), json!({}), remote)?;
            }
            Ok(ok)
        })?;
        self.wake_outbox();
        Ok(ok)
    }

    /// Re-deliver the whole log to this instance.  Applying is idempotent; missing derived state is rebuilt.
    pub async fn replay(self: &Arc<Self>, remote: Option<&str>) -> SResult<Value> {
        let info = self.transport.info().await;
        self.transport.reset_consumer(None).await?;
        let target =
            info_i64(&info, "last_seq").filter(|v| *v != 0).or(info_i64(&info, "stream_messages").filter(|v| *v != 0));
        let svc = self.clone();
        let remote = remote.map(String::from);
        tokio::task::spawn_blocking(move || -> anyhow::Result<()> {
            svc.set_replay_target(target)?;
            svc.metrics.recoveries.with_label_values(&["replay_requested"]).inc();
            svc.store.audit("admin", "replay", None, json!({"target_seq": target}), remote.as_deref())
        })
        .await
        .map_err(|e| anyhow!("{e}"))??;
        Ok(json!({"replaying": true, "target_seq": target}))
    }

    // ------------------------------------------------------------------ health
    pub async fn health(&self) -> Value {
        let mut checks = Map::new();
        let mut status = "ok";
        let stats = match self.store.stats() {
            Ok(s) => {
                checks.insert("db".into(), json!({"ok": true, "path": self.store.db_path}));
                s
            }
            Err(e) => {
                checks.insert("db".into(), json!({"ok": false, "error": format!("{}: {e}", error_type_name(&e))}));
                status = "failing";
                json!({})
            }
        };
        let tinfo = self.transport.info().await;
        let connected = tinfo.get("connected").and_then(Value::as_bool).unwrap_or(false);
        let pending = tinfo.get("consumer_pending").cloned().unwrap_or(Value::Null);
        checks.insert("transport".into(), Value::Object(tinfo));
        let publisher = self.publisher_running.load(Ordering::SeqCst);
        let consumer = self.consumer_running.load(Ordering::SeqCst);
        checks.insert("publisher".into(), json!({"running": publisher, "outbox_pending": stats.get("outbox_pending")}));
        checks.insert(
            "consumer".into(),
            json!({"running": consumer, "last_applied_seq": stats.get("last_applied_seq"), "pending": pending,
                   "replaying_to_seq": self.replay_target()}),
        );
        if status == "ok" && (!connected || !consumer || !publisher) {
            status = "degraded";
        }
        self.metrics.refresh_from_stats(&stats);
        json!({"status": status, "version": crate::VERSION, "instance": self.settings.instance_id, "started_at": self.started_at(),
               "checks": checks, "stats": stats})
    }

    /// Readiness: database open, background loops running, no replay in progress.
    ///
    /// A broker outage does *not* make the service unready by default: the outbox exists precisely so agents can
    /// keep writing through one.  Set `MYCELIC_READY_REQUIRES_NATS=true` to change that.
    pub async fn ready(&self) -> (bool, Value) {
        let h = self.health().await;
        let connected = h["checks"]["transport"]["connected"].as_bool().unwrap_or(false);
        let mut ok = h["checks"]["db"]["ok"].as_bool().unwrap_or(false)
            && self.consumer_running.load(Ordering::SeqCst)
            && self.publisher_running.load(Ordering::SeqCst)
            && self.replay_target().is_none();
        if self.settings.ready_requires_nats && !connected {
            ok = false;
        }
        (
            ok,
            json!({"ready": ok, "status": h["status"], "replaying_to_seq": self.replay_target(), "transport_connected": connected}),
        )
    }
}

fn memory_from_payload(p: &Map<String, Value>, now: &str) -> anyhow::Result<Memory> {
    for k in ["memory_id", "org_id", "layer", "scope", "text"] {
        if !p.get(k).and_then(Value::as_str).is_some_and(|s| !s.is_empty()) {
            return Err(ServiceError::Validation(format!("memory payload lacks '{k}'")).into());
        }
    }
    let s = |k: &str| p[k].as_str().unwrap_or_default().to_string();
    if !LAYERS.contains(&s("layer").as_str()) {
        return Err(ServiceError::Validation("memory payload has an unknown layer".into()).into());
    }
    split_path(&s("scope"))?;
    let opt = |k: &str| -> anyhow::Result<Option<String>> {
        match p.get(k) {
            None | Some(Value::Null) => Ok(None),
            Some(Value::String(v)) => Ok(Some(v.clone())),
            Some(_) => Err(ServiceError::Validation(format!("memory payload field '{k}' must be a string")).into()),
        }
    };
    let source_event_ids = match p.get("source_event_ids") {
        Some(Value::Array(items)) => items.iter().map(py_str).collect(),
        Some(v) if truthy(v) => {
            return Err(
                ServiceError::Validation("memory payload field 'source_event_ids' must be a list".into()).into()
            );
        }
        _ => Vec::new(),
    };
    let metadata = match p.get("metadata") {
        Some(Value::Object(m)) => m.clone(),
        Some(v) if truthy(v) => {
            return Err(ServiceError::Validation("memory payload field 'metadata' must be an object".into()).into());
        }
        _ => Map::new(),
    };
    Ok(Memory {
        memory_id: s("memory_id"),
        org_id: s("org_id"),
        layer: s("layer"),
        scope: s("scope"),
        text: s("text"),
        topic: opt("topic")?,
        slot: opt("slot")?,
        entity: opt("entity")?,
        kind: str_or(p.get("kind"), "observation"),
        confidence: py_float(p.get("confidence").unwrap_or(&json!(0.5)))?,
        support: py_int(p.get("support").unwrap_or(&json!(1)))?,
        independent_teams: py_int(p.get("independent_teams").unwrap_or(&json!(1)))?,
        producer_id: str_or(p.get("producer_id"), "unknown"),
        operator: str_or(p.get("operator"), "agent_observation"),
        rule_id: opt("rule_id")?,
        event_id: opt("event_id")?,
        visibility: str_or(p.get("visibility"), "team"),
        status: str_or(p.get("status"), "active"),
        superseded_by: None,
        created_at: str_or(p.get("created_at"), now),
        applied_at: opt("applied_at")?,
        source_event_ids,
        local_ref: opt("local_ref")?,
        metadata,
    })
}

fn agent_from_payload(p: &Map<String, Value>, now: &str) -> anyhow::Result<Agent> {
    for k in ["agent_id", "org_id", "path"] {
        if !p.get(k).and_then(Value::as_str).is_some_and(|s| !s.is_empty()) {
            return Err(ServiceError::Validation(format!("agent payload lacks '{k}'")).into());
        }
    }
    let s = |k: &str| p[k].as_str().unwrap_or_default().to_string();
    AgentPath::parse(&s("path"))?;
    if s("agent_id") == MYCELIC_PRODUCER {
        return Err(ServiceError::Validation(format!(
            "agent_id '{MYCELIC_PRODUCER}' is reserved for derived memories"
        ))
        .into());
    }
    let scopes = match p.get("scopes") {
        None | Some(Value::Null) => DEFAULT_AGENT_SCOPES.iter().map(|s| s.to_string()).collect(),
        Some(Value::Array(items)) => items.iter().map(py_str).collect(),
        Some(Value::String(st)) => st.chars().map(String::from).collect(),
        Some(_) => return Err(ServiceError::Validation("agent payload scopes must be a list".into()).into()),
    };
    Ok(Agent {
        agent_id: s("agent_id"),
        org_id: s("org_id"),
        display_name: str_or(p.get("display_name"), &s("agent_id")),
        path: s("path"),
        scopes,
        key_prefix: str_or(p.get("key_prefix"), ""),
        status: str_or(p.get("status"), "active"),
        created_at: str_or(p.get("created_at"), now),
        last_seen_at: None,
        metadata: match p.get("metadata") {
            Some(Value::Object(m)) => m.clone(),
            _ => Map::new(),
        },
    })
}
