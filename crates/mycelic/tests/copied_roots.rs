//! Copied evidence is correlated support, not independent corroboration (research/mycelic/ingestion_tesseract_20261024,
//! reviewer finding 2, reproduced against the real service and broker before this test existed).
//!
//! A byte-identical copy of an observation (same slot, same text after case and whitespace normalisation) is one source:
//! the earliest observation stands, the copies add no agent, team or root. Paraphrases are not detected (stated limit).
//! A verbatim re-post of a retracted note by another agent still counts: the golden scenario follows_evidence (steps
//! 36-41) specifies that recount as evidence, so blocking it is an owners' decision, not part of this fix.

mod common;

use common::{Harness, demo_rule};
use serde_json::json;

const BUFFER: &str = "Kessler Antriebe confirms roughly two weeks of SD-9 inventory left.";
const TRANSPORT: &str = "Port of Rotterdam terminal 3 strike announced; SD-9 servo drive shipments route through it.";

fn obs(slot: &str, topic: &str) -> serde_json::Value {
    obs_c(slot, topic, 0.8)
}

fn obs_c(slot: &str, topic: &str, confidence: f64) -> serde_json::Value {
    json!({"topic": topic, "slot": slot, "entity": "sd-9", "confidence": confidence})
}

async fn setup() -> Harness {
    let mut h = Harness::start(false, |_| {}).await;
    h.service.upsert_rule(&demo_rule(), None).unwrap();
    for (a, team) in [("log-1", "logistics"), ("log-2", "logistics"), ("proc-1", "procurement")] {
        h.reg(a, team).await;
    }
    h.register("sales-1", "field-sales", "commercial", "nw-gmbh", "emea", "northwind", None).await;
    h
}

#[tokio::test(flavor = "multi_thread")]
async fn test_a_copy_from_another_agent_is_not_an_independent_source() {
    let h = setup().await;
    // two genuine sources hold all three slots: log-1 (transport and buffer) and sales-1 (demand)
    h.observe("log-1", TRANSPORT, obs("transport_disruption", "supply:sd-9/transport")).await;
    h.observe("log-1", BUFFER, obs("supplier_buffer_low", "supply:sd-9/supplier")).await;
    h.observe("sales-1", "Helios committed to 40 RX-4 arms; every RX-4 uses an SD-9 drive.",
              obs("demand_commitment", "supply:sd-9/demand")).await;
    h.settle().await;
    assert!(h.layer("enterprise").is_empty(), "two agents cannot meet min_agents = 3");
    // proc-1 re-posts log-1's buffer note, differing only in case and spacing, with a higher confidence so that it
    // wins the slot: still one source, so the coalition is still log-1 and sales-1
    h.observe("proc-1", &format!("  {}  ", BUFFER.to_uppercase()),
              obs_c("supplier_buffer_low", "supply:sd-9/supplier", 0.95)).await;
    h.settle().await;
    assert!(h.layer("enterprise").is_empty(), "a copy must not stand in for a third independent agent");
    // control: proc-1's own, different observation is independent evidence and completes the conclusion
    h.observe("proc-1", "Our own count: 9 SD-9 drives on the shelf, reorder lead time 6 weeks.",
              obs_c("supplier_buffer_low", "supply:sd-9/supplier", 0.97)).await;
    h.settle().await;
    let concl = h.layer("enterprise");
    assert_eq!(concl.len(), 1, "an independent observation still composes");
    let agents = concl[0].metadata["contributing_agents"].clone();
    assert_eq!(agents, json!(["log-1", "proc-1", "sales-1"]));
    h.close().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn test_a_backdated_or_offset_copy_cannot_displace_the_original() {
    // reviewer probe: the client chooses observed_at, so "earliest" must not trust it
    for observed_at in ["2020-01-01T00:00:00Z", "2026-10-24T00:00:01-23:59"] {
        let h = setup().await;
        h.observe("log-1", TRANSPORT, obs("transport_disruption", "supply:sd-9/transport")).await;
        h.observe("log-1", BUFFER, obs("supplier_buffer_low", "supply:sd-9/supplier")).await;
        h.observe("sales-1", "Helios committed to 40 RX-4 arms; every RX-4 uses an SD-9 drive.",
                  obs("demand_commitment", "supply:sd-9/demand")).await;
        h.settle().await;
        let mut fields = obs_c("supplier_buffer_low", "supply:sd-9/supplier", 0.95);
        fields["observed_at"] = json!(observed_at);
        h.observe("proc-1", BUFFER, fields).await;
        h.settle().await;
        assert!(h.layer("enterprise").is_empty(), "a copy sent with observed_at {observed_at} displaced the original");
        h.close().await;
    }
}
