//! Shared test fixtures: services over the in-process transport with a temporary database, optionally with the
//! clock and the id generator pinned the way the Python golden-fixture generator pinned them.
#![allow(dead_code)]

use mycelic::auth::Principal;
use mycelic::config::Settings;
use mycelic::metrics::Metrics;
use mycelic::models::{Clock, IdGen};
use mycelic::service::{MycelicService, ServiceParts};
use mycelic::store::Store;
use mycelic::transport::{InProcessTransport, Transport};
use serde_json::{Map, Value, json};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

pub const ADMIN_TOKEN: &str = "test-admin-token-0123456789abcdef0123456789";
pub const FIXED_NOW: &str = "2026-10-01T12:00:00+00:00";

pub fn golden(name: &str) -> Value {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/golden").join(name);
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    serde_json::from_str(&text.replace(": Infinity", ": \"Infinity\"")).expect("fixture JSON")
}

pub fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize().expect("repo root")
}

pub fn demo_rule() -> Value {
    json!({
        "rule_id": "component_supply_risk", "target_layer": "enterprise",
        "required_slots": ["transport_disruption", "supplier_buffer_low", "demand_commitment"],
        "conclusion": "Supply risk for {entity}: inbound transport is disrupted ({slot:transport_disruption}); the supplier buffer is thin ({slot:supplier_buffer_low}); demand is committed ({slot:demand_commitment}).",
        "topic_prefix": "supply:", "min_agents": 3, "min_teams": 2, "kind": "risk",
    })
}

/// `tests/mycelic/helpers.settings`: loopback, in-process transport, no rate limiting, fast outbox.
pub fn settings(dir: &Path) -> Settings {
    let s = Settings {
        host: "127.0.0.1".into(),
        port: 0,
        db_path: dir.join("mycelic.db").to_string_lossy().into_owned(),
        nats_url: None,
        admin_token: Some(ADMIN_TOKEN.into()),
        rate_limit_rps: 0.0,
        min_support: 2,
        publish_interval_seconds: 0.05,
        ..Settings::default()
    };
    s.validate().expect("valid test settings");
    s
}

pub fn fixed_clock() -> Clock {
    Arc::new(|| FIXED_NOW.to_string())
}

/// `prefix_<20-digit counter>`, the generator's pinned `new_id`.
pub fn counter_ids(counter: Arc<AtomicU64>) -> IdGen {
    Arc::new(move |prefix: &str| format!("{prefix}_{:020}", counter.fetch_add(1, Ordering::SeqCst) + 1))
}

/// One service over the in-process transport, plus helpers to register agents and act as them.
pub struct Harness {
    pub dir: tempfile::TempDir,
    pub settings: Settings,
    pub service: Arc<MycelicService>,
    pub transport: InProcessTransport,
    pub keys: HashMap<String, String>,
    pub ids: Arc<AtomicU64>,
    pub pinned: bool,
}

impl Harness {
    pub async fn start(pinned: bool, tweak: impl FnOnce(&mut Settings)) -> Harness {
        let dir = tempfile::tempdir().unwrap();
        let mut settings = settings(dir.path());
        tweak(&mut settings);
        settings.validate().unwrap();
        let transport = InProcessTransport::new();
        let ids = Arc::new(AtomicU64::new(0));
        let service = Self::service(&settings, &transport, &ids, pinned, None);
        service.start(true).await.unwrap();
        Harness { dir, settings, service, transport, keys: HashMap::new(), ids, pinned }
    }

    pub fn service(
        settings: &Settings,
        transport: &InProcessTransport,
        ids: &Arc<AtomicU64>,
        pinned: bool,
        store_path: Option<&str>,
    ) -> Arc<MycelicService> {
        let (clock, idgen) = if pinned {
            (fixed_clock(), counter_ids(ids.clone()))
        } else {
            (mycelic::models::system_clock(), mycelic::models::random_ids())
        };
        let store = Store::open(store_path.unwrap_or(&settings.db_path), clock.clone(), idgen.clone()).unwrap();
        let t: Arc<dyn Transport> = Arc::new(transport.clone());
        MycelicService::with_parts(
            settings.clone(),
            ServiceParts {
                store: Some(store),
                transport: Some(t),
                metrics: Some(Arc::new(Metrics::new())),
                clock: Some(clock),
                ids: Some(idgen),
            },
        )
        .unwrap()
    }

    pub async fn register(
        &mut self,
        agent_id: &str,
        team: &str,
        department: &str,
        subsidiary: &str,
        region: &str,
        enterprise: &str,
        scopes: Option<Vec<&str>>,
    ) -> Principal {
        let mut body = json!({"enterprise": enterprise, "region": region, "subsidiary": subsidiary, "department": department,
                              "team": team, "agent_id": agent_id});
        if let Some(s) = scopes {
            body["scopes"] = json!(s);
        }
        let (_agent, key) = self.service.register_agent(&body, None).unwrap();
        self.keys.insert(agent_id.to_string(), key);
        self.principal(agent_id)
    }

    /// An agent in Northwind's EMEA subsidiary (the tests' usual organization).
    pub async fn reg(&mut self, agent_id: &str, team: &str) -> Principal {
        self.register(agent_id, team, "ops", "nw-gmbh", "emea", "northwind", None).await
    }

    pub fn principal(&self, who: &str) -> Principal {
        let token = if who == "admin" { ADMIN_TOKEN.to_string() } else { self.keys[who].clone() };
        self.service.authenticate(Some(&format!("Bearer {token}"))).unwrap().unwrap()
    }

    pub fn admin(&self) -> Principal {
        self.principal("admin")
    }

    pub async fn observe(&self, agent: &str, text: &str, fields: Value) -> String {
        let mut body = fields.as_object().cloned().unwrap_or_default();
        body.insert("text".into(), json!(text));
        let (m, _) = self.service.ingest_memory(&self.principal(agent), &Value::Object(body), None).unwrap();
        m.memory_id
    }

    pub async fn settle(&self) {
        assert!(self.service.wait_idle(Duration::from_secs(60)).await, "service did not become idle");
    }

    pub async fn close(self) {
        self.service.close().await;
    }
}

/// The first path at which two JSON values differ, for readable failures.
pub fn first_difference(got: &Value, want: &Value, path: &str) -> Option<String> {
    match (got, want) {
        (Value::Object(a), Value::Object(b)) => {
            for (k, v) in b {
                match a.get(k) {
                    None => return Some(format!("{path}.{k}: missing (want {})", short(v))),
                    Some(x) => {
                        if let Some(d) = first_difference(x, v, &format!("{path}.{k}")) {
                            return Some(d);
                        }
                    }
                }
            }
            a.keys().find(|k| !b.contains_key(*k)).map(|k| format!("{path}.{k}: unexpected (got {})", short(&a[k])))
        }
        (Value::Array(a), Value::Array(b)) => {
            for (i, (x, y)) in a.iter().zip(b).enumerate() {
                if let Some(d) = first_difference(x, y, &format!("{path}[{i}]")) {
                    return Some(d);
                }
            }
            (a.len() != b.len()).then(|| format!("{path}: length {} != {}", a.len(), b.len()))
        }
        _ => (got != want).then(|| format!("{path}: got {} want {}", short(got), short(want))),
    }
}

fn short(v: &Value) -> String {
    let s = v.to_string();
    if s.len() > 300 { format!("{}…", &s[..300]) } else { s }
}

pub fn assert_json_eq(got: &Value, want: &Value, ctx: &str) {
    if let Some(d) = first_difference(got, want, "$") {
        panic!("{ctx}: {d}");
    }
}

pub fn obj(v: &Value) -> Map<String, Value> {
    v.as_object().cloned().unwrap_or_default()
}
