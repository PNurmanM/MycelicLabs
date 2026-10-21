//! The experiment suite E1-E12 (`experiments.py`): each experiment appends one JSON row per run to its own
//! JSONL file in the artifacts directory.

use serde_json::{Value, json};

/// E2: one feature removed (or added) at a time from the full system, in `ABLATIONS` order.
pub fn ablations() -> Vec<(&'static str, Value)> {
    vec![
        ("full", json!({})),
        ("-lineage", json!({"lineage": false})),
        ("-independence", json!({"independence": false})),
        ("-contradiction", json!({"contradiction": false})),
        ("-temporal", json!({"temporal": false})),
        ("-adaptive_routing", json!({"adaptive_routing": false})),
        ("-questions", json!({"questions": false})),
        ("-question_targeting", json!({"question_targeting": false})),
        ("-downward_retrieval", json!({"downward_retrieval": false, "questions": false})),
        ("-cross_links", json!({"cross_links": false})),
        ("-sketch_channel", json!({"sketch_channel": false})),
        ("-adaptive_abstraction", json!({"adaptive_abstraction": false})),
        ("-triage_prior", json!({"triage_prior_weight": 0.0})),
        ("+evidence_verification", json!({"verify_evidence": true})),
        ("+source_dispersion", json!({"w_dispersion": 1.5})),
        ("+chain_completion", json!({"chain_completion": true})),
        ("-foreign_filter", json!({"foreign_only_evidence": false})),
        ("-synthesis_restriction", json!({"restrict_synthesis_to_triage": false})),
    ]
}
