# Results: ingestion fixture, adversarial review, copied-source fix

Branch `claude/mycelic-gate-0-recovery-tjl82x` from main `64027cb`. Broker nats-server 2.10.29. Planner → engineer →
adversarial reviewer, two review rounds, each reviewer an independent agent.

## What was executed

| Step | Command | Outcome |
|---|---|---|
| Baseline | `MYCELIC_NATS_SERVER_BIN=… cargo test --workspace` on 64027cb | 184 passed, 0 failed, 1 ignored; 9 JetStream tests really executed |
| 3-owner process fixture | `cargo test -p mycelic-harness --test ingestion_fixture_process -- --nocapture` | passed first run (27 s): abstention, exact roots, cross-org 404s, broker+service SIGKILL restart, retraction, rebuild from JetStream |
| Review round 1 | probes against the real service and broker (deleted after) | 2 production defects reproduced; 4 assertions shown vacuous or narrow (TEST_MATRIX.md) |
| Fix 1 (3daed66) | `cargo test -p mycelic --test copied_roots` | fail before (`copied_roots.rs:49`), pass after |
| Review round 2 | probes | fix 1 bypassable with a backdated or offset `observed_at` (client-controlled) |
| Fix 2 (8e17b46) | same, + backdated/offset test | fail before, pass after (3 consecutive runs) |

## The defect, before and after

A conclusion of rule `component_supply_risk` needs 3 agents in 2 teams. Two genuine sources (log-1 holds transport and
buffer; sales-1 holds demand) cannot meet it.

- **Before (64027cb):** proc-1 re-posts log-1's buffer note (case/whitespace changed, confidence 0.95). The copy wins the
  slot and the conclusion forms with agents {log-1, proc-1, sales-1}.
- **After fix 1:** the copy is collapsed into one source, earliest `(created_at, arrival)`. Review round 2 showed the
  client sets `created_at` (via `observed_at`): a copy dated 2020, or dated with offset `-23:59` (later in time, earlier
  as a string), was "earliest" and the conclusion formed again.
- **After fix 2:** a copy group is credited to the member whose producer already contributes other evidence; ties to
  store arrival. Copying can no longer add an agent regardless of timestamps or order. The independent-observation
  control still composes.

## Deliberately not changed

- A verbatim re-post of a **retracted** note by another agent still restores the conclusion. The golden scenario
  `scenario_follows_evidence` (steps 36–41) specifies exactly that as a legitimate recount ("counted wrong", then a
  colleague's recount). My first version blocked it and broke that golden (22 memories vs 25); the golden was not
  edited. Whether verbatim recounts should count is an owners' decision.
- Free-text query binding: the API's `entity` filter exists; the vacuous abstention is a test-design defect, recorded
  rather than fixed by editing the reviewed test.
- Slot selection is not independence-aware (found while writing the regression test): an agent holding two slots can
  block composition when an independent alternative exists. Recall, not safety. Not changed.

## Regressions

goldens (`golden_api`, `golden_scenarios`, `golden_units`), `datapath`, `copied_roots`: pass. Health-race flakes
(`api.rs:273` 503 vs 200; `golden_api` `/health` degraded; `/admin/replay target_seq 23 vs 24`) reproduce on unmodified
64027cb under load (reviewer: 5 runs each) and predate this work.

## Limitations

- Paraphrased copies are not detected (exact normalised text only).
- The reviewed fixture test was reformatted by rustfmt in a10e2cb (whitespace only).
- Layer 3 (blind raw-text extraction on a local model) is blocked: no local model endpoint in this container, and cloud
  inference is forbidden by the brief.
- No extractor exists server-side; every slot and entity is caller-asserted (PIPELINE.md).
