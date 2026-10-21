//! Port of tests/mycelic/test_strategic.py: strategic synthesis.  Conclusions built from other units'
//! conclusions, across regions, with multi-hop lineage, corroboration, retraction cascades with re-evaluation,
//! and a rebuild from the log.

mod common;

use common::{Harness, repo_root};
use mycelic::metrics::Metrics;
use mycelic::models::Memory;
use mycelic::service::ServiceError;
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet};
use std::time::Duration;

const ENTITY: &str = "sd-9";

fn rules() -> Vec<Value> {
    let text = std::fs::read_to_string(repo_root().join("deploy/mycelic/rules.json")).unwrap();
    serde_json::from_str::<Value>(&text).unwrap()["rules"].as_array().unwrap().clone()
}

fn rule(id: &str) -> Value {
    rules().into_iter().find(|r| r["rule_id"] == id).unwrap()
}

async fn setup() -> Harness {
    let mut h = Harness::start(false, |_| {}).await;
    h.service.upsert_rule(&rule("regional_supply_risk"), None).unwrap();
    h.service.upsert_rule(&rule("strategic_second_source"), None).unwrap();
    for (region, sub) in [("emea", "nw-gmbh"), ("apac", "nw-kk")] {
        for (team, dept) in [("logistics", "ops"), ("procurement", "ops"), ("field-sales", "commercial")] {
            for i in 1..=2 {
                h.register(&format!("{region}-{team}-{i}"), team, dept, sub, region, "northwind", None).await;
            }
        }
    }
    h.register("hq-sourcing-1", "sourcing", "strategy", "northwind-hq", "hq", "northwind", None).await;
    h.register("hq-analytics-1", "analytics", "strategy", "northwind-hq", "hq", "northwind", None).await;
    h
}

async fn observe_region(h: &Harness, region: &str) -> BTreeMap<&'static str, String> {
    let port = if region == "emea" { "Rotterdam" } else { "Busan" };
    let mut ids = BTreeMap::new();
    ids.insert("t1", h.observe(&format!("{region}-logistics-1"), &format!("Port of {port} strike announced; SD-9 shipments route through it."),
        json!({"topic": "supply:sd-9/transport", "slot": "transport_disruption", "entity": ENTITY, "confidence": 0.9})).await);
    ids.insert("t2", h.observe(&format!("{region}-logistics-2"), &format!("Carrier ETA for SD-9 containers via {port} slipped by 12 days."),
        json!({"topic": "supply:sd-9/transport", "slot": "transport_disruption", "entity": ENTITY, "confidence": 0.7})).await);
    ids.insert("s1", h.observe(&format!("{region}-procurement-1"), "Kessler Antriebe confirms two weeks of SD-9 inventory left.",
        json!({"topic": "supply:sd-9/supplier", "slot": "supplier_buffer_low", "entity": ENTITY, "confidence": 0.85})).await);
    ids.insert(
        "d1",
        h.observe(
            &format!("{region}-field-sales-1"),
            "A customer committed to 40 RX-4 arms for November; every RX-4 uses an SD-9.",
            json!({"topic": "supply:sd-9/demand", "slot": "demand_commitment", "entity": ENTITY, "confidence": 0.8}),
        )
        .await,
    );
    ids
}

async fn observe_hq(h: &Harness) {
    h.observe(
        "hq-analytics-1",
        "RX-4 order intake is up 40% year over year across all regions.",
        json!({"topic": "strategy:demand", "slot": "demand_growth", "entity": ENTITY, "confidence": 0.8}),
    )
    .await;
    h.observe("hq-sourcing-1", "Kessler Antriebe is the only qualified SD-9 source for every subsidiary.",
        json!({"topic": "strategy:supply-base", "slot": "supplier_concentration", "entity": ENTITY, "confidence": 0.95})).await;
}

fn strategic(h: &Harness) -> Vec<Memory> {
    h.layer("enterprise").into_iter().filter(|m| m.rule_id.as_deref() == Some("strategic_second_source")).collect()
}

fn regional(h: &Harness) -> BTreeMap<String, Memory> {
    h.layer("region")
        .into_iter()
        .filter(|m| m.rule_id.as_deref() == Some("regional_supply_risk"))
        .map(|m| (m.scope.clone(), m))
        .collect()
}

fn scopes(h: &Harness) -> BTreeSet<String> {
    regional(h).into_keys().collect()
}

fn set(items: &[&str]) -> BTreeSet<String> {
    items.iter().map(|s| s.to_string()).collect()
}

#[tokio::test(flavor = "multi_thread")]
async fn test_strategic_conclusion_rests_on_regional_conclusions() {
    let h = setup().await;
    let s = &h.service;
    observe_region(&h, "emea").await;
    observe_hq(&h).await;
    h.settle().await;
    assert_eq!(scopes(&h), set(&["northwind/emea"]));
    assert!(strategic(&h).is_empty(), "one region is not corroboration");
    observe_region(&h, "apac").await;
    h.settle().await;
    assert_eq!(scopes(&h), set(&["northwind/emea", "northwind/apac"]));
    let strat = strategic(&h);
    assert_eq!(strat.len(), 1);
    let m = &strat[0];
    assert_eq!(m.slot.as_deref(), Some("sourcing_decision"));
    assert_eq!(m.topic.as_deref(), Some("strategy:sourcing"));
    assert!(m.text.contains("qualify a second source"));
    assert_eq!(m.metadata["corroborated_units"]["supply_risk"]["region"], json!(["northwind/apac", "northwind/emea"]));
    assert_eq!(m.metadata["contributing_teams"][0], "northwind/apac/nw-kk/commercial/field-sales");
    assert!(m.support >= 8);
    assert!(m.independent_teams >= 7);
    // lineage: enterprise <- two regional conclusions <- raw observations
    let g = s.lineage(&h.admin(), &m.memory_id).unwrap();
    assert_eq!(g["layers"], json!(["agent", "region", "enterprise"]));
    let parents: BTreeSet<String> = g["edges"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|e| e["child"] == json!(m.memory_id))
        .map(|e| e["parent"].as_str().unwrap().to_string())
        .collect();
    assert!(regional(&h).values().all(|r| parents.contains(&r.memory_id)));
    assert_eq!(
        g["roots"].as_array().unwrap().len(),
        8,
        "3 selected observations per region (strongest per slot) + 2 hq observations"
    );
    assert_eq!(g["evidence"]["reconstructable"], true);
    let hops: BTreeSet<(String, String)> = g["transformations"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| (t["to_layer"].as_str().unwrap().to_string(), t["rule_id"].as_str().unwrap_or("").to_string()))
        .collect();
    let parent_ops: BTreeSet<String> =
        parents.iter().map(|p| g["nodes"][p]["operator"].as_str().unwrap().to_string()).collect();
    assert_eq!(
        parent_ops,
        set(&["agent_observation", "slot_composition"]),
        "no consolidation restating its own parents"
    );
    assert!(hops.contains(&("region".into(), "regional_supply_risk".into())));
    assert!(hops.contains(&("enterprise".into(), "strategic_second_source".into())));
    // an EMEA agent sees the strategy and its own region's conclusion, not APAC's
    let emea = h.principal("emea-field-sales-2");
    let res = s
        .query(
            &emea,
            &json!({"query": "second source sourcing decision sd-9", "scope": "northwind", "min_layer": "region"}),
        )
        .unwrap();
    assert_eq!(res["answer"]["memory_id"], json!(m.memory_id));
    let visible_regions: BTreeSet<String> = res["results"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|hit| hit["memory"]["layer"] == "region")
        .map(|hit| hit["memory"]["scope"].as_str().unwrap().to_string())
        .collect();
    assert_eq!(visible_regions, set(&["northwind/emea"]));
    let gl = s.lineage(&emea, &m.memory_id).unwrap();
    assert!(gl["redacted_contributions"].as_u64().unwrap() > 0);
    assert!(gl["nodes"][&regional(&h)["northwind/apac"].memory_id]["text"].is_null());
    h.close().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn test_strategy_follows_evidence() {
    let mut h = setup().await;
    let s = h.service.clone();
    observe_region(&h, "emea").await;
    let apac = observe_region(&h, "apac").await;
    observe_hq(&h).await;
    h.settle().await;
    let strat = strategic(&h).remove(0);
    s.retract(&h.principal("apac-procurement-1"), &apac["s1"], "counted wrong", None).unwrap();
    h.settle().await;
    assert_eq!(scopes(&h), set(&["northwind/emea"]), "APAC lost its supplier evidence");
    assert!(strategic(&h).is_empty(), "a single region no longer corroborates the strategy");
    assert_eq!(h.memory(&strat.memory_id).status, "retracted");
    // the evidence returns as a new observation: the regional conclusion and the strategy come back
    h.observe(
        "apac-procurement-2",
        "Kessler Antriebe confirms two weeks of SD-9 inventory left.",
        json!({"topic": "supply:sd-9/supplier", "slot": "supplier_buffer_low", "entity": ENTITY, "confidence": 0.85}),
    )
    .await;
    h.settle().await;
    assert_eq!(scopes(&h), set(&["northwind/emea", "northwind/apac"]));
    let again = strategic(&h);
    assert_eq!(again.len(), 1);
    assert_ne!(again[0].memory_id, strat.memory_id, "different evidence, different version");
    assert_eq!(s.lineage(&h.admin(), &again[0].memory_id).unwrap()["layers"], json!(["agent", "region", "enterprise"]));
    // losing one of three corroborating pieces keeps a conclusion that still has enough support
    for (team, dept, i) in
        [("logistics", "ops", 1), ("logistics", "ops", 2), ("procurement", "ops", 1), ("field-sales", "commercial", 1)]
    {
        h.register(&format!("amer-{team}-{i}"), team, dept, "nw-inc", "amer", "northwind", None).await;
    }
    let amer = observe_region(&h, "amer").await;
    h.settle().await;
    let three = strategic(&h).remove(0);
    assert_eq!(
        three.metadata["corroborated_units"]["supply_risk"]["region"],
        json!(["northwind/amer", "northwind/apac", "northwind/emea"])
    );
    s.retract(&h.principal("amer-procurement-1"), &amer["s1"], "duplicate entry", None).unwrap();
    h.settle().await;
    let two = strategic(&h);
    assert_eq!(two.len(), 1, "still corroborated by two regions after re-evaluation");
    assert_eq!(two[0].memory_id, again[0].memory_id, "the exact earlier coalition is reactivated");
    assert!(!two[0].metadata.contains_key("status_reason"));
    assert_eq!(h.memory(&three.memory_id).status, "retracted");
    h.close().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn test_corroboration_needs_corroborate_unless_one_memory_spans_the_units() {
    let h = setup().await;
    let s = &h.service;
    let mut no_corroborate = rule("strategic_second_source");
    no_corroborate["corroborate"] = json!(false);
    s.upsert_rule(&no_corroborate, None).unwrap();
    observe_region(&h, "emea").await;
    observe_region(&h, "apac").await;
    observe_hq(&h).await;
    h.settle().await;
    assert!(strategic(&h).is_empty(), "without corroborate only the strongest regional conclusion is a parent");
    s.upsert_rule(&rule("strategic_second_source"), None).unwrap();
    h.observe(
        "hq-analytics-1",
        "Second demand datapoint: backlog doubled.",
        json!({"topic": "strategy:demand", "slot": "demand_growth", "entity": ENTITY, "confidence": 0.7}),
    )
    .await;
    h.settle().await;
    assert_eq!(strategic(&h).len(), 1);
    h.close().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn test_withdrawal_for_lost_support_cascades() {
    // A stronger note can change the per-slot selection so that distinct agents drop below the threshold; the
    // regional conclusion is withdrawn, and so is everything built on it.
    let h = setup().await;
    let s = &h.service;
    observe_region(&h, "emea").await;
    observe_region(&h, "apac").await;
    observe_hq(&h).await;
    h.settle().await;
    assert_eq!(strategic(&h).len(), 1);
    let emea_before = regional(&h)["northwind/emea"].clone();
    // emea-logistics-1 already supplies the strongest transport note; a supplier note from the same agent becomes
    // the strongest supplier claim, so the selection rests on only 2 distinct agents (< min_agents 3)
    h.observe(
        "emea-logistics-1",
        "Kessler told us there are two weeks of SD-9 inventory left.",
        json!({"topic": "supply:sd-9/supplier", "slot": "supplier_buffer_low", "entity": ENTITY, "confidence": 0.95}),
    )
    .await;
    h.settle().await;
    assert_eq!(scopes(&h), set(&["northwind/apac"]));
    assert_eq!(h.memory(&emea_before.memory_id).status, "retracted");
    assert!(strategic(&h).is_empty(), "the strategy rested on the withdrawn regional conclusion");
    let res = s
        .query(
            &h.principal("hq-sourcing-1"),
            &json!({"query": "second source sd-9", "scope": "northwind", "min_layer": "enterprise"}),
        )
        .unwrap();
    assert!(res["answer"].is_null() || res["answer"]["rule_id"] != "strategic_second_source");
    // an even stronger note from another agent restores three distinct agents: both come back
    h.observe(
        "emea-procurement-2",
        "Recount: two weeks of SD-9 inventory, no second source.",
        json!({"topic": "supply:sd-9/supplier", "slot": "supplier_buffer_low", "entity": ENTITY, "confidence": 0.97}),
    )
    .await;
    h.settle().await;
    assert_eq!(scopes(&h), set(&["northwind/emea", "northwind/apac"]));
    assert_eq!(strategic(&h).len(), 1);
    assert_active_rests_on_active(&h);
    h.close().await;
}

fn assert_active_rests_on_active(h: &Harness) {
    for m in h.all("northwind").into_iter().filter(|m| m.status == "active") {
        for e in h.service.store.read(|db| db.parents_of(&m.memory_id)).unwrap() {
            let parent = h.memory(&e.parent_id);
            assert_eq!(parent.status, "active", "active {} rests on {} {}", m.memory_id, parent.status, e.parent_id);
        }
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn test_stale_dependents_of_a_superseded_consolidation_are_retired() {
    let h = setup().await;
    let s = &h.service;
    s.upsert_rule(&json!({"rule_id": "r", "target_layer": "department", "required_slots": ["transport_disruption", "supplier_buffer_low"],
                          "sources": ["topic_consolidation"], "min_agents": 2, "min_teams": 2, "conclusion": "R for {entity}"}), None).unwrap();
    for (a, slot, topic) in [
        ("emea-logistics-1", "transport_disruption", "supply:sd-9/transport"),
        ("emea-logistics-2", "transport_disruption", "supply:sd-9/transport"),
        ("emea-procurement-1", "supplier_buffer_low", "supply:sd-9/supplier"),
        ("emea-procurement-2", "supplier_buffer_low", "supply:sd-9/supplier"),
    ] {
        h.observe(
            a,
            &format!("note by {a}"),
            json!({"topic": topic, "slot": slot, "entity": "sd-9", "confidence": 0.8}),
        )
        .await;
    }
    h.settle().await;
    let conclusions: Vec<Memory> =
        h.layer("department").into_iter().filter(|m| m.rule_id.as_deref() == Some("r")).collect();
    assert_eq!(conclusions.iter().map(|m| m.entity.clone()).collect::<Vec<_>>(), [Some("sd-9".to_string())]);
    // a transport note about another entity joins the logistics consolidation: its entity becomes None, the old
    // 'r:sd-9' conclusion rests on a superseded parent and must go
    h.observe(
        "emea-logistics-1",
        "Rail slot for RX-2 frames also affected by the strike.",
        json!({"topic": "supply:sd-9/transport", "slot": "transport_disruption", "entity": "rx-2", "confidence": 0.6}),
    )
    .await;
    h.settle().await;
    let active: Vec<Option<String>> =
        h.layer("department").into_iter().filter(|m| m.rule_id.as_deref() == Some("r")).map(|m| m.entity).collect();
    assert_eq!(active, [None]);
    assert_eq!(h.memory(&conclusions[0].memory_id).status, "retracted");
    assert_active_rests_on_active(&h);
    h.close().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn test_consolidation_requires_unanimous_slot_and_entity() {
    let h = setup().await;
    h.service
        .upsert_rule(&json!({"rule_id": "r2", "target_layer": "region", "required_slots": ["transport_disruption", "supplier_buffer_low"],
                             "sources": ["agent_observation", "topic_consolidation"], "min_agents": 3, "conclusion": "R2 {entity}"}), None)
        .unwrap();
    h.observe(
        "emea-logistics-1",
        "strike",
        json!({"topic": "supply:sd-9/transport", "slot": "transport_disruption", "entity": "sd-9", "confidence": 0.9}),
    )
    .await;
    h.observe(
        "emea-procurement-1",
        "low stock",
        json!({"topic": "supply:sd-9/supplier", "slot": "supplier_buffer_low", "entity": "sd-9", "confidence": 0.8}),
    )
    .await;
    h.observe("emea-logistics-2", "Port congestion note without a slot.", json!({"topic": "supply:sd-9/transport"}))
        .await;
    h.settle().await;
    let team: Vec<Memory> =
        h.layer("team").into_iter().filter(|m| m.topic.as_deref() == Some("supply:sd-9/transport")).collect();
    assert_eq!(team.len(), 1);
    assert_eq!(team[0].slot, None);
    assert_eq!(team[0].entity, None);
    assert!(
        h.layer("region").iter().all(|m| m.rule_id.as_deref() != Some("r2")),
        "a slotless note must not lend its agent to a rule's support"
    );
    h.close().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn test_cyclic_rules_are_refused_but_cross_layer_chains_are_not() {
    let h = setup().await;
    let s = &h.service;
    s.upsert_rule(
        &json!({"rule_id": "a", "target_layer": "team", "required_slots": ["x"], "emits_slot": "y", "conclusion": "A",
                          "sources": ["agent_observation", "slot_composition"]}),
        None,
    )
    .unwrap();
    assert!(matches!(
        s.upsert_rule(&json!({"rule_id": "b", "target_layer": "team", "required_slots": ["y"], "emits_slot": "x", "conclusion": "B",
                              "sources": ["slot_composition", "agent_observation"]}), None),
        Err(ServiceError::Validation(_))
    ));
    s.upsert_rule(&json!({"rule_id": "c", "target_layer": "enterprise", "required_slots": ["y"], "emits_slot": "z", "conclusion": "C",
                          "sources": ["slot_composition"]}), None).unwrap();
    assert!(matches!(
        s.upsert_rule(&json!({"rule_id": "too-many", "target_layer": "team", "required_slots": ["a", "b", "c", "d", "e", "f", "g"], "conclusion": "x"}), None),
        Err(ServiceError::Validation(_))
    ));
    assert_eq!(s.store.list_rules(Some("northwind"), true).unwrap().len(), 2 + 2); // regional, strategic, a, c
    h.close().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn test_fragility_scoring_is_bounded() {
    let h = setup().await;
    // every note is strictly stronger than the one before, so the last one always changes the coalition and the
    // surviving conclusion was derived with all 72 candidates in view
    for i in 0..12 {
        for j in 1..=2 {
            let c = 0.5 + (2 * i + j) as f64 / 100.0;
            h.observe(&format!("emea-logistics-{j}"), &format!("transport note {i}-{j}"),
                json!({"topic": "supply:sd-9/transport", "slot": "transport_disruption", "entity": ENTITY, "confidence": c})).await;
            h.observe(&format!("emea-procurement-{j}"), &format!("supplier note {i}-{j}"),
                json!({"topic": "supply:sd-9/supplier", "slot": "supplier_buffer_low", "entity": ENTITY, "confidence": c})).await;
            h.observe(
                &format!("emea-field-sales-{j}"),
                &format!("demand note {i}-{j}"),
                json!({"topic": "supply:sd-9/demand", "slot": "demand_commitment", "entity": ENTITY, "confidence": c}),
            )
            .await;
        }
    }
    assert!(h.service.wait_idle(Duration::from_secs(60)).await);
    let m = regional(&h)["northwind/emea"].clone();
    assert_eq!(m.metadata["candidates"], 72);
    assert_eq!(m.metadata["fragility_scored_candidates"], 18);
    assert!(Metrics::histogram_mean(&h.service.metrics.aggregation_latency) < 0.2);
    h.close().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn test_rebuild_reproduces_the_strategic_lineage() {
    let h = setup().await;
    let s1 = h.service.clone();
    observe_region(&h, "emea").await;
    observe_region(&h, "apac").await;
    observe_hq(&h).await;
    h.settle().await;
    let snap = |hh: &mycelic::service::MycelicService| -> BTreeMap<String, Value> {
        hh.store
            .list_memories(
                "northwind",
                &mycelic::store::MemoryFilter { status: None, limit: 1000, ..Default::default() },
            )
            .unwrap()
            .into_iter()
            .map(|m| (m.memory_id.clone(), json!([m.layer, m.status, m.support, m.slot])))
            .collect()
    };
    let before = snap(&s1);
    let strat = strategic(&h).remove(0);
    let lineage_before = s1.lineage(&h.admin(), &strat.memory_id).unwrap();
    s1.stop().await;
    let s2 = Harness::service(&h.settings, &h.transport, &h.ids, false, Some(":memory:"));
    s2.start(true).await.unwrap();
    assert!(s2.wait_idle(Duration::from_secs(15)).await);
    assert_eq!(snap(&s2), before);
    let p = s2.authenticate(Some(&format!("Bearer {}", h.keys["hq-sourcing-1"]))).unwrap().unwrap();
    let g = s2.lineage(&p, &strat.memory_id).unwrap();
    let sorted = |v: &Value| -> Vec<String> {
        let mut r: Vec<String> = v.as_array().unwrap().iter().map(|x| x.as_str().unwrap().to_string()).collect();
        r.sort();
        r
    };
    assert_eq!(sorted(&g["roots"]), sorted(&lineage_before["roots"]));
    assert_eq!(g["layers"], lineage_before["layers"]);
    s2.close().await;
}
