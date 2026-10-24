# Pipeline as implemented (main 64027cb), stage by stage

Read from source and exercised by tests on 2026-10-24. "Implemented" means a production caller reaches it.

| # | Stage the brief asks for | Status | Where / evidence |
|---|---|---|---|
| 1 | Source bytes → owner-local ingestion | **Absent server-side** | `POST /memory` takes caller-supplied `text` plus `topic`, `slot`, `entity`, `confidence` (`service.rs` `validate_memory_body`). The SDK agent (`mycelic-sdk/src/bin/mycelic-agent.rs`) reads `observations.jsonl` that is already labelled. `LocalMemory` is a SQLite note table searched with `LIKE`. |
| 2 | Parsing / chunking / normalisation | **Absent** | No parser, chunker or normaliser in `crates/mycelic`. The `extract` functions in `mycelic-research` are simulation. |
| 3 | Evidence extraction / entity / time / source identity | **Absent (caller supplies labels)** | Entity and slot are inputs. Source identity is limited to `idempotency_key`, `local_ref` and the producing agent; no byte hash of a source document exists. |
| 4 | Durable store + authenticated outbox | **Implemented** | One transaction writes the memory and a `memory.observed` event (`ingest_memory`, service.rs:1106); HMAC-signed events. |
| 5 | Real JetStream delivery | **Implemented, tested** | `transport.rs`; 9 broker tests in `tests/jetstream.rs` really run with nats-server 2.10.29 (verified with `--nocapture`: no "skipped"). |
| 6 | Application / indexing | **Implemented** | `apply_event` (idempotent); BM25 index per org (`retrieval.rs`: K1 1.5, B 0.75, layer boost 0.15). No embeddings ("deliberately not part of this slice"). |
| 7 | Retrieval / routing | **BM25 only** | `query` (service.rs:1511) ranks the caller's visible memories; the answer is `hits.first()`. An `entity` filter exists and is optional. No graph traversal, no routing between holders. |
| 8 | Cross-owner composition | **Implemented, rule-based** | `aggregation.rs` `compose_rule` + `coordination.rs` `RuleBasedSynthesizer`: one claim per required slot, `min_agents`, `min_teams`; derived memories per layer (team → … → enterprise) with lineage roots. |
| 9 | Answer with evidence references | Implemented | `answer` + `lineage` (roots, contributing agents/teams, redaction for non-visible nodes). |
| 10 | Authorised lineage drill-down | Implemented | `GET /lineage/{id}`; non-visible nodes redacted; a foreign org's ids return the same 404 body as made-up ids (checked). |
| 11 | Correction / withdrawal / rebuild | Implemented | `retract` (agent notes only) withdraws dependants; rebuild from JetStream into a wiped store reproduces the active set (process test and `smoke.rs`). |
| – | Active inquiry / coordinator / registry / router | **Absent in Rust** | `coordination.rs` ports claim synthesis, `LineageAnalyzer`, coalitions and min-cut only; called from `aggregation.rs` and golden tests. No "tesseract" anywhere in this repository. |

## Ownership

Nurman: NeuralGraph memory and retrieval. Anar: coordination, lineage, experiments. The Rust Mycelic service is
the production path; the research crate is simulation.

## Missing links that matter most

1. No server-side extraction: every slot/entity is asserted by the caller, so "emergence" is composition over
   pre-classified observations.
2. Support is counted by agent id and roots by memory id; copies were independent sources (fixed in 3daed66 for
   byte-identical text; paraphrases remain).
3. Slot selection is not independence-aware (one agent holding two slots can block composition).
4. Retrieval is BM25 with an optional entity filter; NeuralGraph's Tesseract is not on this path (see
   TESSERACT_TRACE.md).
