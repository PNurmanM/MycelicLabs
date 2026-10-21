//! Prometheus metrics for a live deployment.  One registry per service; `render()` serves `GET /metrics`.
//!
//! The names, help texts, labels and histogram buckets are the Python service's, so dashboards and alerts keep
//! working: ingestion rate, retrieval and aggregation latency, event throughput, failed events, replay/recovery
//! events, active agents, memory counts by layer, lineage reconstruction success/failure.  Gauges that describe
//! stored state (memories by layer, outbox depth, active agents) are refreshed from the store before every scrape.
//!
//! One exposition difference: the Python client also emitted `*_created` timestamp series for counters and wrote
//! integral values as `1.0`; the Rust client writes `1` and no `_created` series.  Prometheus reads both the same.

use crate::hierarchy::LAYERS;
use prometheus::{
    Encoder, Histogram, HistogramOpts, IntCounter, IntCounterVec, IntGauge, IntGaugeVec, Opts, Registry, TextEncoder,
};
use serde_json::Value;

const LATENCY_BUCKETS: [f64; 13] = [0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0];

pub struct Metrics {
    pub registry: Registry,
    pub memories_ingested: IntCounterVec,
    pub events_received: IntCounter,
    pub events_published: IntCounterVec,
    pub events_applied: IntCounterVec,
    pub events_failed: IntCounterVec,
    pub replay_events: IntCounter,
    pub recoveries: IntCounterVec,
    pub derived: IntCounterVec,
    pub retrieval_latency: Histogram,
    pub aggregation_latency: Histogram,
    pub lineage_latency: Histogram,
    pub lineage_results: IntCounterVec,
    pub http_requests: IntCounterVec,
    pub auth_failures: IntCounterVec,
    pub active_agents: IntGauge,
    pub registered_agents: IntGauge,
    pub memories_by_layer: IntGaugeVec,
    pub outbox_depth: IntGauge,
    pub transport_connected: IntGauge,
    pub consumer_pending: IntGauge,
    pub info: IntGaugeVec,
}

fn counter_vec(r: &Registry, name: &str, help: &str, labels: &[&str]) -> IntCounterVec {
    let c = IntCounterVec::new(Opts::new(name, help), labels).expect("metric definition");
    r.register(Box::new(c.clone())).expect("metric registration");
    c
}

fn counter(r: &Registry, name: &str, help: &str) -> IntCounter {
    let c = IntCounter::with_opts(Opts::new(name, help)).expect("metric definition");
    r.register(Box::new(c.clone())).expect("metric registration");
    c
}

fn gauge(r: &Registry, name: &str, help: &str) -> IntGauge {
    let g = IntGauge::with_opts(Opts::new(name, help)).expect("metric definition");
    r.register(Box::new(g.clone())).expect("metric registration");
    g
}

fn histogram(r: &Registry, name: &str, help: &str) -> Histogram {
    let h = Histogram::with_opts(HistogramOpts::new(name, help).buckets(LATENCY_BUCKETS.to_vec()))
        .expect("metric definition");
    r.register(Box::new(h.clone())).expect("metric registration");
    h
}

impl Default for Metrics {
    fn default() -> Self {
        Self::new()
    }
}

impl Metrics {
    pub fn new() -> Metrics {
        let r = Registry::new();
        let m = Metrics {
            memories_ingested: counter_vec(
                &r,
                "mycelic_memories_ingested_total",
                "Memories accepted by the API",
                &["layer"],
            ),
            events_received: counter(&r, "mycelic_events_received_total", "Events accepted by POST /events"),
            events_published: counter_vec(
                &r,
                "mycelic_events_published_total",
                "Events published to the transport",
                &["kind"],
            ),
            events_applied: counter_vec(
                &r,
                "mycelic_events_applied_total",
                "Events applied by the consumer",
                &["kind", "result"],
            ),
            events_failed: counter_vec(
                &r,
                "mycelic_events_failed_total",
                "Events that failed to publish or apply",
                &["stage"],
            ),
            replay_events: counter(&r, "mycelic_replay_events_total", "Events re-applied during a replay"),
            recoveries: counter_vec(&r, "mycelic_recovery_total", "Recovery actions taken", &["kind"]),
            derived: counter_vec(
                &r,
                "mycelic_memories_derived_total",
                "Higher-layer memories derived by aggregation",
                &["layer", "operator"],
            ),
            retrieval_latency: histogram(&r, "mycelic_retrieval_latency_seconds", "POST /query latency"),
            aggregation_latency: histogram(
                &r,
                "mycelic_aggregation_latency_seconds",
                "Time to apply one event including aggregation",
            ),
            lineage_latency: histogram(&r, "mycelic_lineage_latency_seconds", "Lineage reconstruction latency"),
            lineage_results: counter_vec(
                &r,
                "mycelic_lineage_reconstruction_total",
                "Lineage reconstructions",
                &["result"],
            ),
            http_requests: counter_vec(&r, "mycelic_http_requests_total", "HTTP requests", &["route", "status"]),
            auth_failures: counter_vec(&r, "mycelic_auth_failures_total", "Rejected requests", &["reason"]),
            active_agents: gauge(&r, "mycelic_active_agents", "Agents seen in the last 15 minutes"),
            registered_agents: gauge(&r, "mycelic_registered_agents", "Agents registered and not revoked"),
            memories_by_layer: {
                let g = IntGaugeVec::new(Opts::new("mycelic_memories", "Active memories by layer"), &["layer"])
                    .expect("metric definition");
                r.register(Box::new(g.clone())).expect("metric registration");
                g
            },
            outbox_depth: gauge(&r, "mycelic_outbox_pending", "Events waiting to be published"),
            transport_connected: gauge(&r, "mycelic_transport_connected", "1 when the transport is connected"),
            consumer_pending: gauge(
                &r,
                "mycelic_consumer_pending",
                "Stream messages not yet delivered to the consumer",
            ),
            info: {
                let g = IntGaugeVec::new(Opts::new("mycelic_build_info", "Build information"), &["version"])
                    .expect("metric definition");
                r.register(Box::new(g.clone())).expect("metric registration");
                g
            },
            registry: r,
        };
        for layer in LAYERS {
            m.memories_by_layer.with_label_values(&[layer]).set(0);
        }
        for kind in ["memory.observed", "memory.derived", "memory.retracted", "agent.event"] {
            m.events_published.with_label_values(&[kind]);
        }
        m
    }

    pub fn refresh_from_stats(&self, stats: &Value) {
        for layer in LAYERS {
            let n = stats.get("memories_by_layer").and_then(|b| b.get(layer)).and_then(Value::as_i64).unwrap_or(0);
            self.memories_by_layer.with_label_values(&[layer]).set(n);
        }
        let get = |k: &str| stats.get(k).and_then(Value::as_i64).unwrap_or(0);
        self.outbox_depth.set(get("outbox_pending"));
        self.active_agents.set(get("active_agents"));
        self.registered_agents.set(get("registered_agents"));
    }

    /// The Prometheus text exposition (`text/plain; version=0.0.4`).
    pub fn render(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        let _ = TextEncoder::new().encode(&self.registry.gather(), &mut buf);
        buf
    }

    /// Mean observed value of a histogram (tests: aggregation stays fast).
    pub fn histogram_mean(h: &Histogram) -> f64 {
        h.get_sample_sum() / (h.get_sample_count() as f64).max(1.0)
    }
}
