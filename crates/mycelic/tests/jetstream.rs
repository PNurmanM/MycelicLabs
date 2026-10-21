//! Port of tests/mycelic/test_jetstream.py: integration tests against a real nats-server with JetStream (file store).
//!
//! They prove what the in-process transport cannot: the service publishes to and consumes from a real broker;
//! state survives a service restart (durable consumer resumes), a broker restart (file store keeps the stream), an
//! outage in between (the outbox flushes afterwards), and a lost database is rebuilt from the stream with identical
//! derived ids.
//!
//! The binary is found via `MYCELIC_NATS_SERVER_BIN` or `nats-server` on PATH.  Without it every test here prints
//! that it was skipped and passes; CI installs the binary, so there they always run.

mod common;

use common::demo_rule;
use mycelic::config::Settings;
use mycelic::service::MycelicService;
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::Arc;
use std::time::{Duration, Instant};

const NATS_USER: &str = "mycelic";
const NATS_PASSWORD: &str = "nats-test-password-0123456789abcdef";
const SIGNING_KEY: &str = "test-signing-key-0123456789abcdef0123";

pub fn nats_bin() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("MYCELIC_NATS_SERVER_BIN")
        && !p.is_empty()
    {
        return Some(PathBuf::from(p));
    }
    std::env::var_os("PATH")
        .and_then(|paths| std::env::split_paths(&paths).map(|d| d.join("nats-server")).find(|p| p.is_file()))
}

macro_rules! require_nats {
    () => {
        if nats_bin().is_none() {
            eprintln!("skipped: nats-server binary not available (set MYCELIC_NATS_SERVER_BIN)");
            return;
        }
    };
}

fn free_port() -> u16 {
    TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port()
}

/// A nats-server subprocess with JetStream on a temp dir and user/password auth, like the compose broker.
struct NatsServer {
    conf: PathBuf,
    port: u16,
    proc: Option<Child>,
}

impl NatsServer {
    fn new(dir: &Path, port: u16) -> NatsServer {
        let conf = dir.join("nats.conf");
        std::fs::write(
            &conf,
            format!(
                "port: {port}\nhttp_port: -1\njetstream {{ store_dir: \"{}\" }}\nauthorization {{ user: \"{NATS_USER}\", password: \"{NATS_PASSWORD}\" }}\n",
                dir.join("js").display()
            ),
        )
        .unwrap();
        NatsServer { conf, port, proc: None }
    }

    fn start(&mut self) {
        let child = Command::new(nats_bin().unwrap())
            .arg("-c")
            .arg(&self.conf)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .unwrap();
        self.proc = Some(child);
        let deadline = Instant::now() + Duration::from_secs(10);
        while Instant::now() < deadline {
            if std::net::TcpStream::connect(("127.0.0.1", self.port)).is_ok() {
                return;
            }
            std::thread::sleep(Duration::from_millis(50));
        }
        panic!("nats-server did not start");
    }

    fn kill(&mut self) {
        if let Some(mut p) = self.proc.take() {
            let _ = p.kill();
            let _ = p.wait();
        }
    }

    fn url(&self) -> String {
        format!("nats://127.0.0.1:{}", self.port)
    }
}

impl Drop for NatsServer {
    fn drop(&mut self) {
        self.kill();
    }
}

struct Env {
    root: tempfile::TempDir,
    nats: NatsServer,
}

impl Env {
    fn new() -> Env {
        let root = tempfile::tempdir().unwrap();
        let mut nats = NatsServer::new(root.path(), free_port());
        nats.start();
        Env { root, nats }
    }

    fn db(&self) -> PathBuf {
        self.root.path().join("mycelic.db")
    }

    fn settings(&self, max_deliver: Option<i64>) -> Settings {
        let mut s = common::settings(self.root.path());
        s.db_path = self.db().to_string_lossy().into_owned();
        s.nats_url = Some(self.nats.url());
        s.nats_user = Some(NATS_USER.into());
        s.nats_password = Some(NATS_PASSWORD.into());
        s.nats_stream = "MYCELIC_TEST".into();
        s.nats_consumer = "mycelic-test".into();
        s.event_signing_key = Some(SIGNING_KEY.into());
        if let Some(m) = max_deliver {
            s.nats_max_deliver = m;
        }
        s
    }

    async fn service(&self, max_deliver: Option<i64>) -> Arc<MycelicService> {
        let s = MycelicService::new(self.settings(max_deliver)).unwrap();
        s.start(true).await.unwrap();
        s
    }

    fn wipe_db(&self) {
        for entry in std::fs::read_dir(self.root.path()).unwrap() {
            let p = entry.unwrap().path();
            if p.file_name().unwrap().to_string_lossy().starts_with("mycelic.db") {
                std::fs::remove_file(p).unwrap();
            }
        }
    }
}

async fn seed(s: &MycelicService) -> BTreeMap<String, String> {
    s.upsert_rule(&demo_rule(), None).unwrap();
    let mut keys = BTreeMap::new();
    for (agent, dept, team) in [
        ("log-1", "ops", "logistics"),
        ("log-2", "ops", "logistics"),
        ("proc-1", "ops", "procurement"),
        ("sales-1", "commercial", "field-sales"),
    ] {
        let (_, key) = s
            .register_agent(&json!({"enterprise": "northwind", "region": "emea", "subsidiary": "nw-gmbh", "department": dept, "team": team, "agent_id": agent}), None)
            .unwrap();
        keys.insert(agent.to_string(), key);
    }
    let obs = [
        (
            "log-1",
            "Port of Rotterdam terminal 3 strike announced for weeks 41-43.",
            "supply:sd-9/transport",
            "transport_disruption",
            0.9,
        ),
        (
            "log-2",
            "Carrier ETA for the SD-9 container slipped by 12 days.",
            "supply:sd-9/transport",
            "transport_disruption",
            0.7,
        ),
        (
            "proc-1",
            "Kessler Antriebe has two weeks of SD-9 inventory left.",
            "supply:sd-9/supplier",
            "supplier_buffer_low",
            0.85,
        ),
        (
            "sales-1",
            "Helios Automation committed to 40 RX-4 arms for November.",
            "supply:sd-9/demand",
            "demand_commitment",
            0.8,
        ),
    ];
    for (agent, text, topic, slot, conf) in obs {
        let p = principal(s, &keys[agent]);
        s.ingest_memory(
            &p,
            &json!({"text": text, "topic": topic, "slot": slot, "entity": "sd-9", "confidence": conf}),
            None,
        )
        .unwrap();
    }
    assert!(s.wait_idle(Duration::from_secs(20)).await);
    keys
}

fn principal(s: &MycelicService, key: &str) -> mycelic::auth::Principal {
    s.authenticate(Some(&format!("Bearer {key}"))).unwrap().unwrap()
}

fn snapshot(s: &MycelicService) -> BTreeMap<String, Value> {
    s.store
        .list_memories("northwind", &mycelic::store::MemoryFilter { status: None, limit: 1000, ..Default::default() })
        .unwrap()
        .into_iter()
        .map(|m| (m.memory_id.clone(), json!([m.layer, m.status, m.support, m.confidence])))
        .collect()
}

fn active(snap: &BTreeMap<String, Value>) -> BTreeMap<String, Value> {
    snap.iter().filter(|(_, v)| v[1] == "active").map(|(k, v)| (k.clone(), v.clone())).collect()
}

async fn info(s: &MycelicService) -> serde_json::Map<String, Value> {
    s.transport.info().await
}

fn stat(s: &MycelicService, key: &str) -> Value {
    s.store.stats().unwrap()[key].clone()
}

#[tokio::test(flavor = "multi_thread")]
async fn test_publish_consume_and_survive_service_restart() {
    require_nats!();
    let env = Env::new();
    let s1 = env.service(None).await;
    let keys = seed(&s1).await;
    let i = info(&s1).await;
    assert_eq!(i["connected"], true);
    assert!(
        i["stream_messages"].as_i64().unwrap() >= 9,
        "registrations, rule, observations and derived events are in the stream"
    );
    assert_eq!(i["consumer_pending"], 0);
    assert_eq!(stat(&s1, "memories_by_layer")["enterprise"], 1);
    assert_eq!(stat(&s1, "outbox_pending"), 0);
    let before = snapshot(&s1);
    let seq_before = stat(&s1, "last_applied_seq");
    s1.close().await;
    drop(s1);

    let s2 = env.service(None).await;
    assert!(s2.wait_idle(Duration::from_secs(10)).await);
    assert_eq!(snapshot(&s2), before, "restart changes nothing");
    assert_eq!(stat(&s2, "last_applied_seq"), seq_before);
    assert_eq!(s2.replay_target(), None, "an up-to-date database does not replay");
    let res = s2
        .query(&principal(&s2, &keys["sales-1"]), &json!({"query": "supply risk sd-9", "scope": "northwind"}))
        .unwrap();
    assert_eq!(res["answer"]["layer"], "enterprise");
    assert_eq!(res["answer"]["lineage"]["reconstructable"], true);
    s2.close().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn test_broker_outage_is_absorbed_by_the_outbox() {
    require_nats!();
    let mut env = Env::new();
    let s = env.service(None).await;
    let keys = seed(&s).await;
    let (_, key3) = s
        .register_agent(&json!({"enterprise": "northwind", "region": "emea", "subsidiary": "nw-gmbh", "department": "ops", "team": "logistics", "agent_id": "log-3"}), None)
        .unwrap();
    assert!(s.wait_idle(Duration::from_secs(10)).await);
    env.nats.kill();
    tokio::time::sleep(Duration::from_millis(500)).await;
    let (m, _) = s
        .ingest_memory(&principal(&s, &key3), &json!({"text": "Customs backlog of two weeks reported at Rotterdam.", "topic": "supply:sd-9/transport",
                                                     "slot": "transport_disruption", "entity": "sd-9", "confidence": 0.6}), None)
        .unwrap();
    tokio::time::sleep(Duration::from_secs(1)).await;
    assert!(stat(&s, "outbox_pending").as_i64().unwrap() >= 1, "accepted while the broker is down");
    assert_eq!(s.health().await["status"], "degraded");
    assert!(s.ready().await.0, "readiness does not depend on the broker by default");
    env.nats.start();
    assert!(s.wait_idle(Duration::from_secs(30)).await, "outbox flushed and applied after the broker returned");
    assert_eq!(stat(&s, "outbox_pending"), 0);
    assert_eq!(s.transport.reconnects(), 1, "the existing client reconnected; no second client was opened");
    assert_eq!(s.metrics.recoveries.with_label_values(&["transport_reconnect"]).get(), 0);
    assert!(s.store.get_memory(&m.memory_id).unwrap().unwrap().applied_at.is_some());
    let team = s
        .store
        .list_memories(
            "northwind",
            &mycelic::store::MemoryFilter { layers: Some(&["team".to_string()]), ..Default::default() },
        )
        .unwrap();
    assert_eq!(team[0].support, 3);
    let _ = keys;
    s.close().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn test_lost_database_is_rebuilt_from_the_stream() {
    require_nats!();
    let env = Env::new();
    let s1 = env.service(None).await;
    let keys = seed(&s1).await;
    let before = snapshot(&s1);
    let stream_before = info(&s1).await["stream_messages"].clone();
    s1.close().await;
    drop(s1);
    env.wipe_db();

    let s2 = env.service(None).await;
    assert!(s2.metrics.recoveries.with_label_values(&["replay_fresh_db"]).get() > 0);
    assert!(s2.wait_idle(Duration::from_secs(30)).await);
    let after = snapshot(&s2);
    assert_eq!(active(&after), active(&before), "same derived ids after the rebuild");
    assert_eq!(info(&s2).await["stream_messages"], stream_before, "replay appended nothing");
    assert_eq!(s2.store.list_agents(Some("northwind"), false).unwrap().len(), 4);
    // the old key still works: registrations replayed
    let res = s2
        .query(&principal(&s2, &keys["sales-1"]), &json!({"query": "supply risk sd-9", "scope": "northwind"}))
        .unwrap();
    assert_eq!(res["answer"]["layer"], "enterprise");
    assert_eq!(res["answer"]["lineage"]["reconstructable"], true);
    assert_eq!(s2.store.list_rules(Some("northwind"), true).unwrap().len(), 1);
    s2.close().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn test_restored_backup_receives_the_events_it_missed() {
    require_nats!();
    let env = Env::new();
    let s1 = env.service(None).await;
    seed(&s1).await;
    s1.close().await;
    drop(s1);
    // a quiescent copy, like a volume snapshot
    let backup = env.root.path().join("backup.db");
    rusqlite::Connection::open(env.db()).unwrap().execute("VACUUM INTO ?", [backup.to_str().unwrap()]).unwrap();

    // life goes on after the backup: a new agent and a new observation
    let s2 = env.service(None).await;
    let (_, key3) = s2
        .register_agent(&json!({"enterprise": "northwind", "region": "emea", "subsidiary": "nw-gmbh", "department": "ops", "team": "logistics", "agent_id": "log-3"}), None)
        .unwrap();
    let (m, _) = s2
        .ingest_memory(&principal(&s2, &key3), &json!({"text": "Customs backlog of two weeks reported at Rotterdam.", "topic": "supply:sd-9/transport",
                                                      "slot": "transport_disruption", "entity": "sd-9", "confidence": 0.6}), None)
        .unwrap();
    assert!(s2.wait_idle(Duration::from_secs(20)).await);
    s2.close().await;
    drop(s2);

    // disaster: restore the older database
    env.wipe_db();
    std::fs::copy(&backup, env.db()).unwrap();
    let s3 = env.service(None).await;
    assert!(s3.wait_idle(Duration::from_secs(20)).await);
    assert!(s3.metrics.recoveries.with_label_values(&["replay_restored_backup"]).get() > 0);
    assert!(s3.store.get_memory(&m.memory_id).unwrap().is_some(), "the event after the backup was re-delivered");
    assert!(s3.store.get_agent("log-3").unwrap().is_some(), "so was the registration");
    let team = s3
        .store
        .list_memories(
            "northwind",
            &mycelic::store::MemoryFilter { layers: Some(&["team".to_string()]), ..Default::default() },
        )
        .unwrap();
    assert_eq!(team[0].support, 3);
    assert_eq!(principal(&s3, &key3).id, "log-3");
    s3.close().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn test_lost_consumer_is_recreated_after_the_last_applied_event() {
    require_nats!();
    let env = Env::new();
    let s1 = env.service(None).await;
    seed(&s1).await;
    let before = snapshot(&s1);
    let stream_before = info(&s1).await["stream_messages"].clone();
    let client = async_nats::ConnectOptions::new()
        .user_and_password(NATS_USER.into(), NATS_PASSWORD.into())
        .connect(env.nats.url())
        .await
        .unwrap();
    let js = async_nats::jetstream::new(client);
    js.get_stream("MYCELIC_TEST").await.unwrap().delete_consumer("mycelic-test").await.unwrap();
    s1.close().await;
    drop(s1);
    let s2 = env.service(None).await;
    assert!(s2.wait_idle(Duration::from_secs(20)).await);
    assert!(s2.metrics.recoveries.with_label_values(&["consumer_recreated"]).get() > 0);
    assert_eq!(snapshot(&s2), before);
    assert_eq!(info(&s2).await["stream_messages"], stream_before, "nothing was re-published");
    assert_eq!(stat(&s2, "outbox_pending"), 0);
    s2.close().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn test_forced_replay_is_idempotent() {
    require_nats!();
    let env = Env::new();
    let s = env.service(None).await;
    seed(&s).await;
    let before = snapshot(&s);
    let stream_before = info(&s).await["stream_messages"].clone();
    let res = s.replay(None).await.unwrap();
    assert_eq!(res["replaying"], true);
    assert!(!s.ready().await.0, "not ready while replaying");
    assert!(s.wait_idle(Duration::from_secs(30)).await);
    assert!(s.ready().await.0);
    assert_eq!(snapshot(&s), before);
    assert_eq!(info(&s).await["stream_messages"], stream_before);
    assert!(s.metrics.recoveries.with_label_values(&["replay_requested"]).get() > 0);
    assert_eq!(s.store.get_meta("replay_target_seq").unwrap(), None);
    s.close().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn test_poison_event_is_terminated_and_replay_completes() {
    require_nats!();
    let env = Env::new();
    let s = env.service(Some(2)).await;
    seed(&s).await;
    let bad = json!({"event_id": "evt_poison", "kind": "memory.observed", "org_id": "northwind", "agent_id": "log-1",
                     "created_at": "2026-01-01T00:00:00+00:00", "payload": {"memory_id": "mem_poison"}, "schema": 1});
    let wire = serde_json::to_vec(&bad).unwrap();
    s.transport
        .publish("mycelic.northwind.memory-observed", wire.clone().into(), "evt_poison", &s.sign(&wire))
        .await
        .unwrap();
    assert!(s.wait_idle(Duration::from_secs(30)).await);
    assert!(s.metrics.events_failed.with_label_values(&["apply"]).get() > 0);
    assert!(s.store.get_memory("mem_poison").unwrap().is_none());
    let actions: Vec<String> = s
        .store
        .read(|db| db.recent_audit(20, None))
        .unwrap()
        .iter()
        .map(|r| r["action"].as_str().unwrap().to_string())
        .collect();
    assert!(actions.contains(&"event.failed".to_string()));
    // an unsigned event as the very last stream entry must not leave a rebuild stuck in 'replaying'
    let unsigned = br#"{"event_id": "evt_unsigned", "kind": "agent.event", "payload": {}}"#.to_vec();
    s.transport.publish("mycelic.northwind.memory-observed", unsigned.into(), "evt_unsigned", &[]).await.unwrap();
    assert!(s.wait_idle(Duration::from_secs(20)).await);
    s.close().await;
    drop(s);
    env.wipe_db();
    let s2 = env.service(Some(2)).await;
    assert!(s2.wait_idle(Duration::from_secs(30)).await);
    assert!(s2.ready().await.0, "terminated events count as consumed for replay completion");
    assert_eq!(stat(&s2, "memories_by_layer")["enterprise"], 1);
    s2.close().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn test_unfinished_replay_resumes_after_a_crash() {
    require_nats!();
    let env = Env::new();
    let s1 = env.service(None).await;
    seed(&s1).await;
    let before = snapshot(&s1);
    let stream_before = info(&s1).await["stream_messages"].clone();
    s1.close().await;
    drop(s1);
    env.wipe_db();
    // a rebuild that dies after a few events: no background loops, three deliveries applied by hand
    let s2 = MycelicService::new(env.settings(None)).unwrap();
    s2.start(false).await.unwrap();
    assert!(s2.replay_target().is_some());
    for _ in 0..3 {
        for d in s2.transport.fetch(1, Duration::from_secs(2)).await.unwrap() {
            s2.handle_delivery(d).await;
        }
    }
    assert_eq!(s2.store.get_meta("last_applied_seq").unwrap().as_deref(), Some("3"));
    assert!(s2.store.get_meta("replay_target_seq").unwrap().is_some());
    s2.transport.close().await;
    drop(s2);
    let s3 = env.service(None).await;
    assert!(!s3.ready().await.0, "still replaying after the crash");
    assert!(s3.metrics.recoveries.with_label_values(&["replay_resumed"]).get() > 0);
    assert!(s3.wait_idle(Duration::from_secs(30)).await);
    assert!(s3.ready().await.0);
    assert_eq!(snapshot(&s3), before);
    assert_eq!(info(&s3).await["stream_messages"], stream_before, "no duplicates were appended");
    s3.close().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn test_unsigned_events_are_rejected() {
    require_nats!();
    let env = Env::new();
    let s = env.service(None).await;
    seed(&s).await;
    let forged = json!({"event_id": "evt_forged", "kind": "memory.observed", "org_id": "northwind", "agent_id": "log-1", "created_at": "2026-01-01T00:00:00+00:00",
                        "payload": {"memory_id": "mem_forged", "org_id": "northwind", "layer": "agent", "scope": "northwind/emea/nw-gmbh/ops/logistics/log-1",
                                    "text": "forged", "producer_id": "log-1", "operator": "agent_observation"}, "schema": 1});
    s.transport
        .publish("mycelic.northwind.memory-observed", serde_json::to_vec(&forged).unwrap().into(), "evt_forged", &[])
        .await
        .unwrap();
    tokio::time::sleep(Duration::from_millis(1500)).await;
    assert!(s.store.get_memory("mem_forged").unwrap().is_none());
    assert!(s.metrics.events_failed.with_label_values(&["signature"]).get() > 0);
    s.close().await;
}
