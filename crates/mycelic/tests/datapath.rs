//! Port of tests/mycelic/test_datapath.py: the complete data path over the in-process transport (observe -> event
//! -> apply -> aggregate -> query -> lineage), visibility and redaction, idempotency, supersession, reactivation,
//! retraction and a rebuild from the event log.

mod common;

use common::{Harness, demo_rule};
use mycelic::service::ServiceError;
use mycelic::store::MemoryFilter;
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet};
use std::time::Duration;

const TRANSPORT: &str = "supply:sd-9/transport";

async fn setup() -> Harness {
    let mut h = Harness::start(false, |_| {}).await;
    h.service.upsert_rule(&demo_rule(), None).unwrap();
    for a in ["log-1", "log-2"] {
        h.reg(a, "logistics").await;
    }
    h.reg("proc-1", "procurement").await;
    for a in ["sales-1", "sales-2"] {
        h.register(a, "field-sales", "commercial", "nw-gmbh", "emea", "northwind", None).await;
    }
    h
}

async fn seed(h: &Harness) -> BTreeMap<&'static str, String> {
    let mut ids = BTreeMap::new();
    ids.insert("log-1", h.observe("log-1", "Port of Rotterdam terminal 3 strike announced for weeks 41-43; SD-9 servo drive shipments route through it.",
        json!({"topic": TRANSPORT, "slot": "transport_disruption", "entity": "sd-9", "confidence": 0.9})).await);
    ids.insert(
        "log-2",
        h.observe(
            "log-2",
            "Carrier ETA for the SD-9 servo drive container slipped by 12 days.",
            json!({"topic": TRANSPORT, "slot": "transport_disruption", "entity": "sd-9", "confidence": 0.7}),
        )
        .await,
    );
    ids.insert("proc-1", h.observe("proc-1", "Kessler Antriebe confirms roughly two weeks of SD-9 inventory left.",
        json!({"topic": "supply:sd-9/supplier", "slot": "supplier_buffer_low", "entity": "sd-9", "confidence": 0.85})).await);
    ids.insert(
        "sales-1",
        h.observe(
            "sales-1",
            "Helios Automation committed to 40 RX-4 arms for November; every RX-4 uses an SD-9 drive.",
            json!({"topic": "supply:sd-9/demand", "slot": "demand_commitment", "entity": "sd-9", "confidence": 0.8}),
        )
        .await,
    );
    h.settle().await;
    ids
}

fn strs(v: &Value) -> Vec<String> {
    v.as_array().unwrap().iter().map(|x| x.as_str().unwrap().to_string()).collect()
}

#[tokio::test(flavor = "multi_thread")]
async fn test_observations_become_team_and_enterprise_memories_with_lineage() {
    let h = setup().await;
    let ids = seed(&h).await;
    let s = &h.service;
    let st = s.store.stats().unwrap();
    assert_eq!(st["memories_by_layer"]["agent"], 4);
    assert_eq!(st["memories_by_layer"]["team"], 1, "logistics consolidates its two observations");
    assert_eq!(st["memories_by_layer"]["enterprise"], 1, "the rule fires once all three slots are covered");
    assert_eq!(st["outbox_pending"], 0);
    assert_eq!(st["events"].get("pending").and_then(Value::as_i64).unwrap_or(0), 0);

    // query from a different team's agent, at enterprise scope
    let res = s
        .query(
            &h.principal("sales-2"),
            &json!({"query": "delivery risk for RX-4 and the SD-9 servo drive", "scope": "northwind"}),
        )
        .unwrap();
    let top = &res["answer"];
    assert_eq!(top["layer"], "enterprise");
    assert!(top["text"].as_str().unwrap().contains("Supply risk for sd-9"));
    assert_eq!(top["support"], 3);
    assert_eq!(top["independent_teams"], 3);
    assert_eq!(top["lineage"]["contributing_agents"], 3);
    assert_eq!(top["lineage"]["reconstructable"], true);
    assert_eq!(top["lineage"]["layers"], json!(["agent", "enterprise"]));

    let g = &res["lineage"];
    let roots: BTreeSet<String> = strs(&g["roots"]).into_iter().collect();
    assert_eq!(
        roots,
        BTreeSet::from([ids["log-1"].clone(), ids["proc-1"].clone(), ids["sales-1"].clone()]),
        "the strongest transport observation is selected"
    );
    assert_eq!(g["redacted_contributions"], 2, "sales-2 may not read logistics' or procurement's raw notes");
    assert_eq!(g["contributing_agents"], json!(["sales-1"]));
    assert_eq!(g["contributing_teams"].as_array().unwrap().len(), 3);
    assert_eq!(g["nodes"][&ids["log-1"]]["text"], Value::Null);
    assert_eq!(g["nodes"][&ids["log-1"]]["redacted"], true);
    assert_eq!(g["nodes"][&ids["log-1"]]["scope"], "northwind/emea/nw-gmbh/ops/logistics");
    assert!(g["nodes"][&ids["sales-1"]]["text"].is_string());
    assert_eq!(g["evidence"]["reconstructable"], true);
    assert_eq!(g["support"]["fragility"]["unique_failure_domain_count"], 3);
    assert!(g["timeline"]["first_observed_at"].is_string());

    // the same lineage seen by an administrator has nothing redacted
    let ga = s.lineage(&h.admin(), top["memory_id"].as_str().unwrap()).unwrap();
    assert_eq!(ga["redacted_contributions"], 0);
    assert_eq!(ga["contributing_agents"], json!(["log-1", "proc-1", "sales-1"]));

    // team consolidation is visible to its own team only
    let team = h.layer("team").remove(0);
    assert_eq!(team.support, 2);
    assert_eq!(team.scope, "northwind/emea/nw-gmbh/ops/logistics");
    assert!((team.confidence - 0.97).abs() < 0.005);
    assert_eq!(s.get_memory(&h.principal("log-2"), &team.memory_id).unwrap().memory_id, team.memory_id);
    assert!(matches!(s.get_memory(&h.principal("sales-2"), &team.memory_id), Err(ServiceError::NotFound(_))));
    assert!(matches!(s.get_memory(&h.principal("sales-2"), &ids["log-1"]), Err(ServiceError::NotFound(_))));
    assert!(matches!(
        s.query(&h.principal("sales-2"), &json!({"query": "x", "scope": "northwind/emea/nw-gmbh/ops"})),
        Err(ServiceError::Forbidden(_))
    ));
    let tl = s.lineage(&h.principal("log-1"), &team.memory_id).unwrap();
    assert_eq!(tl["layers"], json!(["agent", "team"]));
    assert_eq!(tl["contributing_agents"], json!(["log-1", "log-2"]));
    h.close().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn test_idempotent_resend_and_write_as_self() {
    let h = setup().await;
    let s = &h.service;
    let p = h.principal("log-1");
    let (m1, c1) = s.ingest_memory(&p, &json!({"text": "same note", "idempotency_key": "note-1"}), None).unwrap();
    let (m2, c2) = s.ingest_memory(&p, &json!({"text": "same note", "idempotency_key": "note-1"}), None).unwrap();
    assert_eq!(m1.memory_id, m2.memory_id);
    assert_eq!((c1, c2), (true, false));
    assert!(matches!(
        s.ingest_memory(&p, &json!({"text": "spoof", "agent_id": "log-2"}), None),
        Err(ServiceError::Forbidden(_))
    ));
    assert!(matches!(
        s.ingest_memory(&h.admin(), &json!({"text": "admins do not observe"}), None),
        Err(ServiceError::Forbidden(_))
    ));
    h.close().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn test_growing_coalition_supersedes_and_keeps_history() {
    let mut h = setup().await;
    seed(&h).await;
    let old = h.layer("team").remove(0);
    h.reg("log-3", "logistics").await;
    h.observe(
        "log-3",
        "Rotterdam customs broker warns of a two-week backlog for SD-9 containers.",
        json!({"topic": TRANSPORT, "slot": "transport_disruption", "entity": "sd-9", "confidence": 0.6}),
    )
    .await;
    h.settle().await;
    let active = h.layer("team");
    assert_eq!(active.len(), 1);
    let new = &active[0];
    assert_ne!(new.memory_id, old.memory_id);
    assert_eq!(new.support, 3);
    let stale = h.memory(&old.memory_id);
    assert_eq!(stale.status, "superseded");
    assert_eq!(stale.superseded_by.as_deref(), Some(new.memory_id.as_str()));
    let g = h.service.lineage(&h.principal("log-1"), &new.memory_id).unwrap();
    assert_eq!(g["previous_versions"], json!([old.memory_id]));
    assert_eq!(g["roots"].as_array().unwrap().len(), 3);
    h.close().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn test_retraction_withdraws_dependent_conclusions() {
    let h = setup().await;
    let ids = seed(&h).await;
    let s = &h.service;
    let conclusion = h.layer("enterprise").remove(0);
    s.retract(&h.principal("sales-1"), &ids["sales-1"], "customer cancelled the order", None).unwrap();
    h.settle().await;
    assert_eq!(h.memory(&ids["sales-1"]).status, "retracted");
    assert_eq!(h.memory(&conclusion.memory_id).status, "retracted");
    assert!(h.layer("enterprise").is_empty());
    let res = s
        .query(
            &h.principal("sales-2"),
            &json!({"query": "supply risk sd-9", "scope": "northwind", "min_layer": "enterprise"}),
        )
        .unwrap();
    assert!(res["answer"].is_null());
    // derived memories cannot be retracted directly: they follow their evidence
    let team = h.layer("team").remove(0);
    assert!(matches!(s.retract(&h.admin(), &team.memory_id, "x", None), Err(ServiceError::Validation(_))));
    for agent in ["log-1", "log-2"] {
        s.retract(&h.principal(agent), &ids[agent], "withdrawn", None).unwrap();
    }
    h.settle().await;
    assert!(h.layer("team").is_empty());
    h.observe("proc-1", "Unrelated note about pallets.", json!({"topic": "ops:pallets"})).await;
    h.settle().await;
    assert!(h.layer("team").is_empty(), "retracted evidence does not come back");
    // another team's raw note: not even its existence is revealed
    assert!(matches!(s.retract(&h.principal("sales-2"), &ids["log-1"], "x", None), Err(ServiceError::NotFound(_))));
    // visible (enterprise-level) but not the producer
    assert!(matches!(
        s.retract(&h.principal("sales-2"), &conclusion.memory_id, "x", None),
        Err(ServiceError::Forbidden(_))
    ));
    h.close().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn test_coalition_that_shrinks_back_reactivates_the_earlier_version() {
    let mut h = setup().await;
    seed(&h).await;
    let first = h.layer("team").remove(0);
    h.reg("log-3", "logistics").await;
    let third = h
        .observe(
            "log-3",
            "Customs broker warns of a backlog for SD-9 containers.",
            json!({"topic": TRANSPORT, "slot": "transport_disruption", "entity": "sd-9", "confidence": 0.6}),
        )
        .await;
    h.settle().await;
    let second = h.layer("team").remove(0);
    assert_ne!(first.memory_id, second.memory_id);
    h.service.retract(&h.principal("log-3"), &third, "false alarm", None).unwrap();
    h.settle().await;
    let active: Vec<String> = h.layer("team").into_iter().map(|m| m.memory_id).collect();
    assert_eq!(active, std::slice::from_ref(&first.memory_id), "the earlier coalition is current again");
    assert_eq!(h.memory(&second.memory_id).status, "retracted");
    let g = h.service.lineage(&h.principal("log-1"), &first.memory_id).unwrap();
    assert_eq!(
        strs(&g["contributing_agents"]).into_iter().collect::<BTreeSet<_>>(),
        BTreeSet::from(["log-1".into(), "log-2".into()])
    );
    assert_eq!(g["evidence"]["reconstructable"], true);
    h.close().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn test_reactivated_version_keeps_its_history() {
    let mut h = setup().await;
    seed(&h).await;
    let t1 = h.layer("team").remove(0);
    h.reg("log-3", "logistics").await;
    h.observe(
        "log-3",
        "Customs backlog for SD-9 containers.",
        json!({"topic": TRANSPORT, "slot": "transport_disruption", "entity": "sd-9", "confidence": 0.6}),
    )
    .await;
    h.settle().await;
    let t2 = h.layer("team").remove(0);
    h.reg("log-4", "logistics").await;
    let third = h
        .observe(
            "log-4",
            "Rail alternative for SD-9 is fully booked.",
            json!({"topic": TRANSPORT, "slot": "transport_disruption", "entity": "sd-9", "confidence": 0.5}),
        )
        .await;
    h.settle().await;
    let t3 = h.layer("team").remove(0);
    assert_eq!(BTreeSet::from([&t1.memory_id, &t2.memory_id, &t3.memory_id]).len(), 3);
    h.service.retract(&h.principal("log-4"), &third, "mistaken", None).unwrap();
    h.settle().await;
    let active: Vec<String> = h.layer("team").into_iter().map(|m| m.memory_id).collect();
    assert_eq!(active, std::slice::from_ref(&t2.memory_id));
    assert_eq!(h.service.lineage(&h.admin(), &t2.memory_id).unwrap()["previous_versions"], json!([t1.memory_id]));
    h.close().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn test_agent_input_cannot_impersonate_aggregator_metadata() {
    let h = setup().await;
    seed(&h).await;
    let conclusion = h.layer("enterprise").remove(0);
    h.observe("sales-2", "Nothing to see here.",
        json!({"topic": "misc", "visibility": "org", "metadata": {"promoted_from": conclusion.memory_id, "agg_key": "k", "version_of": "x", "note": "kept"}})).await;
    h.observe("sales-2", "Second note.", json!({"topic": "misc", "metadata": {"agg_key": "k"}})).await;
    h.settle().await;
    let layers = ["agent".to_string()];
    let mine = h
        .service
        .store
        .list_memories(
            "northwind",
            &MemoryFilter { layers: Some(&layers), producer_id: Some("sales-2"), ..Default::default() },
        )
        .unwrap();
    assert_eq!(mine.len(), 2);
    assert!(!mine[0].metadata.contains_key("promoted_from"));
    assert!(mine.iter().any(|m| m.metadata.contains_key("note")));
    let res = h
        .service
        .query(
            &h.principal("log-1"),
            &json!({"query": "supply risk sd-9", "scope": "northwind", "min_layer": "enterprise"}),
        )
        .unwrap();
    assert_eq!(res["answer"]["memory_id"], json!(conclusion.memory_id), "an org-visible note cannot hide a conclusion");
    h.close().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn test_source_events_must_belong_to_the_organization() {
    let mut h = setup().await;
    let s = h.service.clone();
    assert!(matches!(
        s.ingest_memory(&h.principal("log-1"), &json!({"text": "x", "source_event_ids": ["evt_doesnotexist"]}), None),
        Err(ServiceError::Validation(_))
    ));
    h.register("acme-1", "acme", "acme", "acme", "acme", "acme", None).await;
    let out = s.ingest_events(&h.principal("acme-1"), Some(&json!([{"type": "x"}])), None).unwrap();
    let foreign = out[0]["event_id"].as_str().unwrap();
    assert!(matches!(
        s.ingest_memory(&h.principal("log-1"), &json!({"text": "x", "source_event_ids": [foreign]}), None),
        Err(ServiceError::Validation(_))
    ));
    h.close().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn test_reserved_producer_id_and_metadata_visibility() {
    let mut h = setup().await;
    let s = h.service.clone();
    assert!(matches!(
        s.register_agent(&json!({"enterprise": "northwind", "agent_id": "mycelic"}), None),
        Err(ServiceError::Validation(_))
    ));
    seed(&h).await;
    let conclusion = h.layer("enterprise").remove(0).to_json();
    let default = h.principal("sales-2");
    assert!(s.public_view(&conclusion, &default)["metadata"].get("roots").is_some());
    assert!(s.public_view(&conclusion, &default)["metadata"].get("evidence").is_none());
    h.register("reader", "field-sales", "commercial", "nw-gmbh", "emea", "northwind", Some(vec!["memory:read"])).await;
    let reader = h.principal("reader");
    assert!(s.public_view(&conclusion, &reader)["metadata"].get("roots").is_none());
    let m = mycelic::models::Memory::from_json(&conclusion).unwrap();
    assert!(!reader.owns(&m));
    assert!(!h.principal("log-1").owns(&m), "derived memories have no owner but the organization");
    h.close().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn test_local_reference_is_private_to_the_producer() {
    let h = setup().await;
    let s = &h.service;
    let mid = h.observe("log-1", "Terminal 3 strike.", json!({"topic": TRANSPORT, "local_ref": "note-42"})).await;
    h.settle().await;
    assert_eq!(s.lineage(&h.principal("log-1"), &mid).unwrap()["nodes"][&mid]["local_ref"], "note-42");
    assert_eq!(s.lineage(&h.admin(), &mid).unwrap()["nodes"][&mid]["local_ref"], "note-42");
    assert!(s.lineage(&h.principal("log-2"), &mid).unwrap()["nodes"][&mid]["local_ref"].is_null());
    let p2 = h.principal("log-2");
    assert!(s.public_view(&s.get_memory(&p2, &mid).unwrap().to_json(), &p2)["local_ref"].is_null());
    h.close().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn test_events_with_embedded_memories_and_evidence() {
    let h = setup().await;
    let s = &h.service;
    let p = h.principal("proc-1");
    let item = json!({"type": "supplier.call", "payload": {"supplier": "Kessler"}, "idempotency_key": "call-1",
                      "memory": {"text": "Kessler has two weeks of SD-9 stock.", "topic": "supply:sd-9/supplier", "slot": "supplier_buffer_low", "entity": "sd-9"}});
    let out = s.ingest_events(&p, Some(&json!([item, {"type": "heartbeat"}])), None).unwrap();
    assert_eq!(out.len(), 2);
    assert_eq!(out[0]["created"], true);
    let mem = h.memory(out[0]["memory_id"].as_str().unwrap());
    assert_eq!(mem.source_event_ids, [out[0]["event_id"].as_str().unwrap()]);
    let again =
        s.ingest_events(&p, Some(&json!([{"type": "supplier.call", "idempotency_key": "call-1"}])), None).unwrap();
    assert_eq!(again[0]["created"], false);
    // re-sending the full item (memory included) is idempotent for the embedded memory too
    for _ in 0..2 {
        let rep = s.ingest_events(&p, Some(&json!([item])), None).unwrap();
        assert_eq!(rep[0]["memory_id"], out[0]["memory_id"]);
        assert_eq!(rep[0]["memory_created"], false);
    }
    let layers = ["agent".to_string()];
    let mine = s
        .store
        .list_memories(
            "northwind",
            &MemoryFilter { layers: Some(&layers), producer_id: Some("proc-1"), ..Default::default() },
        )
        .unwrap();
    assert_eq!(mine.len(), 1);
    h.settle().await;
    let g = s.lineage(&p, &mem.memory_id).unwrap();
    assert!(g["evidence"]["source_event_ids"].as_array().unwrap().contains(&out[0]["event_id"]));
    assert_eq!(g["evidence"]["reconstructable"], true);
    h.close().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn test_rebuild_from_the_event_log_reproduces_state_agents_and_rules() {
    let h = setup().await;
    seed(&h).await;
    let s1 = h.service.clone();
    // a burst: the API writes land before the consumer applies them; aggregation must only see applied ones so
    // that a rebuild derives exactly the same history (every version, not only the active set)
    for (i, agent) in ["log-1", "log-2", "proc-1"].iter().enumerate() {
        h.observe(
            agent,
            &format!("Burst note {i} about the SD-9 servo drive delay."),
            json!({"topic": "supply:sd-9/burst", "confidence": 0.5}),
        )
        .await;
    }
    h.settle().await;
    let full = |svc: &mycelic::service::MycelicService| -> BTreeMap<String, Value> {
        svc.store
            .list_memories("northwind", &MemoryFilter { status: None, limit: 1000, ..Default::default() })
            .unwrap()
            .into_iter()
            .map(|m| (m.memory_id.clone(), json!([m.layer, m.status, m.support, m.metadata.get("version_of")])))
            .collect()
    };
    let full_before = full(&s1);
    let events_before = s1.store.read(|db| db.event_counts()).unwrap();
    s1.stop().await;

    // a brand-new database over the same log: state, agents (their keys) and rules come back from the stream
    let s2 = Harness::service(&h.settings, &h.transport, &h.ids, false, Some(":memory:"));
    s2.start(true).await.unwrap();
    assert!(s2.wait_idle(Duration::from_secs(10)).await);
    assert_eq!(full(&s2), full_before, "the whole history, superseded versions included, is reproduced");
    assert_eq!(s2.store.list_rules(Some("northwind"), true).unwrap().len(), 1);
    let p = s2.authenticate(Some(&format!("Bearer {}", h.keys["sales-2"]))).unwrap().unwrap();
    let res = s2.query(&p, &json!({"query": "supply risk sd-9", "scope": "northwind"})).unwrap();
    assert_eq!(res["answer"]["layer"], "enterprise");
    assert_eq!(res["answer"]["lineage"]["reconstructable"], true);
    assert_eq!(s2.store.stats().unwrap()["outbox_pending"], 0, "replay must not re-publish derived events");
    assert!(s2.metrics.replay_events.get() > 0);
    assert_eq!(s2.store.read(|db| db.event_counts()).unwrap(), events_before);
    let unsequenced = s2
        .store
        .read(|db| db.list_events("northwind", None, Some("memory.derived"), None, 1000))
        .unwrap()
        .into_iter()
        .filter(|e| e.js_seq.is_none())
        .count();
    assert_eq!(unsequenced, 0);
    s2.close().await;
}
