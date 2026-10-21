//! Watching the deployment: the admin read API polled into trace events.

use super::scenario::{ENTERPRISE, LiveScenario};
use super::trace::TraceRecorder;
use anyhow::{Result, anyhow};
use mycelic_sdk::MycelicClient;
use serde_json::{Map, Value, json};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

pub const STATUSES: [&str; 3] = ["active", "superseded", "retracted"];
/// The organizational layers, bottom up (the service's `hierarchy::LAYERS`).
pub const LAYERS: [&str; 6] = ["agent", "team", "department", "subsidiary", "region", "enterprise"];
const PAGE: i64 = 500;

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// One page of the enterprise's memories with this status (`GET /memories?status=`).
pub fn list_memories(client: &MycelicClient, status: &str) -> Result<Vec<Value>> {
    Ok(client.list_memories_with_status(Some(ENTERPRISE), None, PAGE, Some(status))?)
}

/// Every memory of the enterprise, whatever its status, by id.
pub fn all_memories(client: &MycelicClient) -> Result<Map<String, Value>> {
    let mut out = Map::new();
    for status in STATUSES {
        let rows = list_memories(client, status)?;
        if rows.len() >= PAGE as usize {
            eprintln!("warning: {status} memories hit the {PAGE}-row page limit; the watcher may miss some");
        }
        for m in rows {
            out.insert(str_of(&m["memory_id"]), m);
        }
    }
    Ok(out)
}

/// The `metric` event's payload from `GET /admin/status`.
pub fn metric_payload(st: &Value) -> Value {
    let s = &st["stats"];
    json!({"memories_by_layer": s["memories_by_layer"], "memories_by_status": s["memories_by_status"],
           "lineage_edges": s["lineage_edges"], "registered_agents": s["registered_agents"],
           "stream_messages": st["checks"]["transport"]["stream_messages"]})
}

fn str_of(v: &Value) -> String {
    v.as_str().unwrap_or_default().to_string()
}

/// Python truthiness of a JSON value (`x or y` chains in the original).
fn truthy(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64() != Some(0.0),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

/// An agent as the watcher labels its notes.
#[derive(Clone)]
struct AgentInfo {
    team_path: String,
    region: String,
    role: &'static str,
}

/// A flag a sleeping thread can be woken by (Python's `threading.Event` used as a wake-up).
#[derive(Default)]
struct Wake {
    set: Mutex<bool>,
    cv: Condvar,
}

impl Wake {
    fn set(&self) {
        *lock(&self.set) = true;
        self.cv.notify_all();
    }

    /// Sleep up to `timeout` or until set, then clear.
    fn wait(&self, timeout: Duration) {
        let g = lock(&self.set);
        let (mut g, _) = self.cv.wait_timeout_while(g, timeout, |set| !*set).unwrap_or_else(|e| e.into_inner());
        *g = false;
    }
}

#[derive(Default)]
struct PollState {
    /// memory_id -> (status, layer) as last seen
    known: HashMap<String, (String, String)>,
    /// memory_id -> time of the first read that returned it
    first_seen: HashMap<String, f64>,
    /// memory_id -> when the last read without it was sent
    absent_at: HashMap<String, Option<f64>>,
    last_sent: Option<f64>,
}

struct MetricState {
    last: Option<Value>,
    at: Option<Instant>,
}

struct Inner {
    client: MycelicClient,
    rec: Arc<TraceRecorder>,
    agents: HashMap<String, AgentInfo>,
    interval: Duration,
    fast_interval: Duration,
    fast: AtomicBool,
    stop: AtomicBool,
    paused: AtomicBool,
    wake: Wake,
    poll: Mutex<PollState>,
    metric: Mutex<MetricState>,
}

/// Polls the admin read API and turns every change into a trace event, stamped with the read that saw it.
///
/// Each poll reads the active memories; every [`MemoryWatcher::FULL_EVERY`]-th poll while agents share (every
/// poll otherwise) also reads the superseded and retracted ones, so a version that lived between two polls is
/// still recorded.  For each memory the watcher keeps when it was first seen and when it was last known to be
/// absent, which is what [`MemoryWatcher::latency`] turns into an upper bound on "note accepted -> conclusion
/// readable".
pub struct MemoryWatcher {
    inner: Arc<Inner>,
    thread: Option<JoinHandle<()>>,
}

impl MemoryWatcher {
    pub const FULL_EVERY: u64 = 10;

    pub fn new(
        base_url: &str,
        admin_token: &str,
        rec: Arc<TraceRecorder>,
        sc: &LiveScenario,
        interval: f64,
        fast_interval: f64,
    ) -> MemoryWatcher {
        let agents = sc
            .agents
            .iter()
            .map(|a| {
                (a.agent_id.clone(), AgentInfo { team_path: a.team_path(), region: a.region.clone(), role: a.role })
            })
            .collect();
        let inner = Inner {
            client: MycelicClient::new(base_url, admin_token).with_retries(0).with_timeout(Duration::from_secs(10)),
            rec,
            agents,
            interval: Duration::from_secs_f64(interval.max(0.0)),
            fast_interval: Duration::from_secs_f64(fast_interval.max(0.0)),
            fast: AtomicBool::new(false),
            stop: AtomicBool::new(false),
            paused: AtomicBool::new(false),
            wake: Wake::default(),
            poll: Mutex::new(PollState::default()),
            metric: Mutex::new(MetricState { last: None, at: None }),
        };
        MemoryWatcher { inner: Arc::new(inner), thread: None }
    }

    pub fn start(&mut self) {
        if self.thread.is_some() {
            return;
        }
        let inner = Arc::clone(&self.inner);
        self.thread = Some(
            std::thread::Builder::new()
                .name("mycelic-live-watcher".into())
                .spawn(move || inner.run())
                .expect("spawning the watcher thread"),
        );
    }

    /// Stop polling; waits up to 10 s for a poll in flight.
    pub fn stop(&mut self) {
        self.inner.stop.store(true, Ordering::SeqCst);
        self.inner.wake.set();
        if let Some(t) = self.thread.take() {
            let deadline = Instant::now() + Duration::from_secs(10);
            while !t.is_finished() && Instant::now() < deadline {
                std::thread::sleep(Duration::from_millis(10));
            }
            if t.is_finished() {
                let _ = t.join();
            }
        }
    }

    /// No polling until [`MemoryWatcher::resume`]; returns once a poll in flight is over.
    pub fn pause(&self) {
        self.inner.paused.store(true, Ordering::SeqCst);
        drop(lock(&self.inner.poll));
    }

    pub fn resume(&self) {
        self.inner.paused.store(false, Ordering::SeqCst);
    }

    /// Poll every `fast_interval` (agents are sharing) or every `interval`; switching wakes the poller.
    pub fn set_fast(&self, fast: bool) {
        self.inner.fast.store(fast, Ordering::SeqCst);
        self.inner.wake.set();
    }

    pub fn poll_once(&self, full: bool) -> Result<()> {
        self.inner.poll_once(full)
    }

    pub fn metric(&self, force: bool) -> Result<()> {
        self.inner.metric(force)
    }

    /// Upper bound, in seconds, from the last note a conclusion rests on being accepted to the conclusion being
    /// readable: from sending the last watcher read that did not yet return that note to receiving the first read
    /// that returned the conclusion (both reads by the same poller, so no clock skew).
    pub fn latency(&self, memory: &Value) -> Option<f64> {
        let st = lock(&self.inner.poll);
        let roots: Vec<String> = memory["metadata"]["roots"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|r| r.as_str().map(String::from))
            .collect();
        let seen = *st.first_seen.get(memory["memory_id"].as_str()?)?;
        if roots.is_empty() || roots.iter().any(|r| !st.first_seen.contains_key(r)) {
            return None;
        }
        // the first root with the latest first sighting, like Python's max()
        let mut last = &roots[0];
        for r in &roots[1..] {
            if st.first_seen[r] > st.first_seen[last] {
                last = r;
            }
        }
        let start = st.absent_at.get(last).copied().flatten().unwrap_or(st.first_seen[last]);
        Some(((seen - start) * 1000.0).round() / 1000.0)
    }
}

impl Drop for MemoryWatcher {
    fn drop(&mut self) {
        self.stop();
    }
}

impl Inner {
    fn run(&self) {
        let mut n: u64 = 0;
        while !self.stop.load(Ordering::SeqCst) {
            if !self.paused.load(Ordering::SeqCst) {
                let fast = self.fast.load(Ordering::SeqCst);
                // an error is a restart in progress; the next poll catches up
                if self.poll_once(!fast || n % MemoryWatcher::FULL_EVERY == 0).is_ok() {
                    n += 1;
                    let stale = lock(&self.metric).at.is_none_or(|at| at.elapsed() > Duration::from_secs(1));
                    if !fast && stale {
                        let _ = self.metric(false);
                    }
                }
            }
            self.wake.wait(if self.fast.load(Ordering::SeqCst) { self.fast_interval } else { self.interval });
        }
    }

    fn metric(&self, force: bool) -> Result<()> {
        let payload = metric_payload(&self.client.status()?);
        let mut m = lock(&self.metric);
        m.at = Some(Instant::now());
        if force || m.last.as_ref() != Some(&payload) {
            m.last = Some(payload.clone());
            self.rec.emit("metric", payload);
        }
        Ok(())
    }

    fn poll_once(&self, full: bool) -> Result<()> {
        let mut st = lock(&self.poll);
        if self.paused.load(Ordering::SeqCst) {
            return Ok(());
        }
        let sent_at = self.rec.now(); // before the request: the snapshot is at or after this
        let mut rows = Map::new();
        for m in list_memories(&self.client, "active")? {
            rows.insert(str_of(&m["memory_id"]), m);
        }
        let read_at = self.rec.now(); // after the response: the snapshot is at or before this
        if full {
            for status in &STATUSES[1..] {
                for m in list_memories(&self.client, status)? {
                    rows.insert(str_of(&m["memory_id"]), m);
                }
            }
        }
        let before = st.last_sent.replace(sent_at);
        let mut new: Vec<&Value> = rows.iter().filter(|(id, _)| !st.known.contains_key(*id)).map(|(_, m)| m).collect();
        let layer_index = |m: &Value| LAYERS.iter().position(|l| m["layer"] == *l).unwrap_or(LAYERS.len());
        let when = |m: &Value| {
            [&m["applied_at"], &m["created_at"]].into_iter().find(|v| truthy(v)).map(str_of).unwrap_or_default()
        };
        new.sort_by_cached_key(|m| (layer_index(m), when(m), str_of(&m["memory_id"])));
        for m in &new {
            let id = str_of(&m["memory_id"]);
            st.first_seen.insert(id.clone(), read_at);
            st.absent_at.insert(id, before);
        }
        for m in &new {
            if m["layer"] == "agent" {
                self.note(m, read_at)?;
            } else {
                self.derived(m, read_at)?;
            }
            st.known.insert(str_of(&m["memory_id"]), ("active".into(), str_of(&m["layer"])));
            if m["status"] != "active" {
                // first seen already superseded/retracted: never observed active
                self.status(&mut st, m, None, read_at);
            }
        }
        for (id, m) in &rows {
            let known = st.known.get(id).map(|(status, _)| status.clone());
            if let Some(known) = known.filter(|k| m["status"] != *k.as_str()) {
                self.status(&mut st, m, Some(known), read_at);
            }
        }
        Ok(())
    }

    fn status(&self, st: &mut PollState, m: &Value, before: Option<String>, at: f64) {
        self.rec.emit_at(
            "status_change",
            Some(at),
            json!({"memory_id": m["memory_id"], "layer": m["layer"], "from": before, "to": m["status"],
                   "reason": m["metadata"]["status_reason"]}),
        );
        if let Some(k) = st.known.get_mut(m["memory_id"].as_str().unwrap_or_default()) {
            k.0 = str_of(&m["status"]);
        }
    }

    fn note(&self, m: &Value, at: f64) -> Result<()> {
        let producer = str_of(&m["producer_id"]);
        let a =
            self.agents.get(&producer).ok_or_else(|| anyhow!("a note from {producer}, who is not in the scenario"))?;
        self.rec.emit_at(
            "note_shared",
            Some(at),
            json!({"agent_id": producer, "memory_id": m["memory_id"], "team_path": a.team_path, "region": a.region,
                   "role": a.role, "topic": m["topic"], "slot": m["slot"], "entity": m["entity"], "text": m["text"],
                   "confidence": m["confidence"], "status": m["status"]}),
        );
        Ok(())
    }

    fn derived(&self, m: &Value, at: f64) -> Result<()> {
        let id = str_of(&m["memory_id"]);
        let meta = &m["metadata"];
        let g = self.client.lineage(&id)?;
        let mut parents: Vec<String> = g["edges"]
            .as_array()
            .into_iter()
            .flatten()
            .filter(|e| e["child"] == id.as_str())
            .map(|e| str_of(&e["parent"]))
            .collect();
        parents.sort();
        let corroborated =
            if truthy(&meta["corroborated_units"]) { meta["corroborated_units"].clone() } else { Value::Null };
        self.rec.emit_at(
            "derived",
            Some(at),
            json!({"memory_id": id, "layer": m["layer"], "scope": m["scope"], "operator": m["operator"],
                   "rule_id": m["rule_id"], "topic": m["topic"], "slot": m["slot"], "entity": m["entity"], "kind": m["kind"],
                   "text": m["text"], "confidence": m["confidence"], "support": m["support"],
                   "independent_teams": m["independent_teams"], "version_of": meta["version_of"], "status": m["status"],
                   "parents": parents, "roots": g["roots"], "corroborated_units": corroborated}),
        );
        Ok(())
    }
}
