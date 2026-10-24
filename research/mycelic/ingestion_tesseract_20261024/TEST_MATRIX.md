# Test matrix (2026-10-24)

Broker: nats-server v2.10.29 (sha256 e314da74…), `MYCELIC_NATS_SERVER_BIN` set for every run below.
Outcome: PASS / FAIL / BLOCKED / NOT-RUN. "Wrong-reason" = passes but the reviewer showed the assertion is vacuous.

| Fixture / invariant | Layer | Command | Outcome |
|---|---|---|---|
| Baseline suite on main 64027cb | 1+2 | `cargo test --workspace` | PASS: 184 passed, 0 failed, 1 ignored; 9 JetStream tests executed (`--nocapture`, no "skipped") |
| 3-owner fixture: composition with exact roots | 2 (real broker + service process + SDK) | `cargo test -p mycelic-harness --test ingestion_fixture_process -- --nocapture` | PASS (roots = exactly the three fragments) |
| Abstention with denied owner holding the missing slot | 2 | same | PASS, **wrong-reason**: query not entity-bound; null because no enterprise memory exists at all |
| Similar entity (sd-90) not merged | 2 | same | PASS, **wrong-reason**: sd-90 never complete, check cannot fire |
| Cross-enterprise denial (memory, lineage) | 2 | same | PASS; 404 bodies identical for real and made-up ids (reviewer probe) |
| Canary not in serialized payloads | 2 | same | PASS, narrow: query + admin lineage only; `/admin/events`, agent-layer results not scanned |
| Broker + service SIGKILL and restart | 2 | same | PASS (same answer, same active set) |
| Retraction withdraws the enterprise conclusion | 2 | same | PASS |
| Casual unslotted mention does not restore | 2 | same | PASS, weak: unslotted notes never compose |
| Rebuild from JetStream into a wiped store | 2 | same | PASS (same active set; withdrawn conclusion stays withdrawn) |
| Resent root (same idempotency key) is one memory | 2 | same | PASS |
| **Copy by another agent counted as an independent source** | 1 | `cargo test -p mycelic --test copied_roots` | **FAIL on 64027cb** (`copied_roots.rs:49`), PASS after 3daed66 |
| Copy of a retracted note revives the conclusion | 2 (reviewer probe) | probe, deleted | Reproduced; **not fixed**: contradicts golden `scenario_follows_evidence` steps 36–41 (owners' decision) |
| Free-text query answers another entity's conclusion | 2 (reviewer probe) | probe, deleted | Reproduced; quality defect; the `entity` filter exists |
| wait_idle ignores JetStream ack-pending | – | code read | SUSPECTED (timing) |
| Unknown-parent / stale `memory.derived` admitted from the stream | – | code read (`service.rs:1386-1418`) | SUSPECTED; needs a direct NATS publish to test; NOT-RUN |
| Full suite after 3daed66 | 1+2 | `cargo test --workspace --no-fail-fast` | goldens, datapath, golden_scenarios, copied_roots PASS; one `api.rs:273` 503-vs-200 health race under heavy CPU load (see RESULTS.md) |
| Raw-text blind extraction on a local model | 3 | – | **BLOCKED**: no local model endpoint in this container; cloud inference forbidden |
| Tesseract retrieval / coordination on the production path | – | source trace | Absent (TESSERACT_TRACE.md) |
