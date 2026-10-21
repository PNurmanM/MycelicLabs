//! Time the per-query BM25 path (index build + `search_index`) on synthetic organizations, store bypassed: the
//! Rust twin of the Python baseline script (Retriever.search over a fake store).
//!
//! ```text
//! cargo run --release -p mycelic --example retrieval_bench
//! ```
//!
//! Same organization shape as the Python script: 20 teams of 7 agents, 14 random words per memory from a 32-word
//! vocabulary, every third memory org-visible, 30 topics and entities; the caller is one agent of team 3; five
//! queries.  "first" includes tokenizing every memory into the index (the Python script's "first query").

use mycelic::auth::memory_visible;
use mycelic::models::Memory;
use mycelic::retrieval::{Retriever, SearchParams, search_index};
use serde_json::Map;
use std::time::Instant;

const W: &str = "port strike rotterdam terminal carrier eta supplier stock servo drive order customer november shipment \
                 delay forecast region emea apac warehouse inventory quality defect recall audit invoice contract price";

/// xorshift64*: deterministic, no dependency.
struct Rng(u64);

impl Rng {
    fn below(&mut self, n: usize) -> usize {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        (self.0.wrapping_mul(0x2545F4914F6CDD1D) >> 33) as usize % n
    }
}

fn memories(n: usize) -> Vec<Memory> {
    let words: Vec<&str> = W.split_whitespace().collect();
    let teams: Vec<String> = (0..20).map(|i| format!("acme/emea/gmbh/ops/t{i}")).collect();
    let mut rng = Rng(1);
    (0..n)
        .map(|i| Memory {
            memory_id: format!("m{i}"),
            org_id: "acme".into(),
            layer: "agent".into(),
            scope: format!("{}/a{}", teams[i % 20], i % 7),
            text: (0..14).map(|_| words[rng.below(words.len())]).collect::<Vec<_>>().join(" "),
            topic: Some(format!("supply:sd-{}", i % 30)),
            slot: None,
            entity: Some(format!("sd-{}", i % 30)),
            kind: "agent_observation".into(),
            confidence: 0.8,
            support: 1,
            independent_teams: 1,
            producer_id: format!("a{i}"),
            operator: "observe".into(),
            rule_id: None,
            event_id: None,
            visibility: if i % 3 == 0 { "org".into() } else { "team".into() },
            status: "active".into(),
            superseded_by: None,
            created_at: "2026-10-01T00:00:00Z".into(),
            applied_at: None,
            source_event_ids: vec![],
            local_ref: None,
            metadata: Map::new(),
        })
        .collect()
}

fn main() {
    let (agent, team) = ("acme/emea/gmbh/ops/t3/a3", "acme/emea/gmbh/ops/t3");
    let visible = |m: &Memory| memory_visible(m, Some(agent), Some(team));
    let params = SearchParams::default();
    for n in [1_000, 10_000, 100_000] {
        let rows = memories(n);
        let retriever = Retriever::new();
        let t0 = Instant::now();
        let index = retriever.index("acme", 1, || Ok(rows.clone())).unwrap();
        search_index(&index, "port strike servo", &visible, &params);
        let cold = t0.elapsed().as_secs_f64();
        let mut warm: Vec<f64> = Vec::new();
        for q in [
            "port strike servo",
            "supplier stock november",
            "carrier eta delay emea",
            "quality recall audit",
            "warehouse inventory",
        ] {
            let t = Instant::now();
            let index = retriever.index("acme", 1, || unreachable!("cached")).unwrap();
            search_index(&index, q, &visible, &params);
            warm.push(t.elapsed().as_secs_f64());
        }
        warm.sort_by(f64::total_cmp);
        println!(
            "N={n:>6}: first query (tokenize index) {:.1} ms | warm query median {:.2} ms",
            cold * 1e3,
            warm[2] * 1e3
        );
    }
}
