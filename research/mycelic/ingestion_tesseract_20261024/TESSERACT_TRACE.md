NeuralGraph origin/main 038fb63 (Rust port; local branch is Python):
- retrieval::tesseract::Tesseract (crates/neuralgraph-research/src/retrieval/tesseract.rs). Algorithm: four stores
  (Temporal/Entity/Reasoning/Adversarial), each scans ALL nodes of the session (nodes_or_session) and computes a
  "charge" (embedding cosine + keyword/entity/temporal heuristics), keeps charge>=min_charge, ranks top-limit;
  results fused. Exhaustive scoring, no graph traversal / no routing.
  Runtime callers: benchmarks only (runner.rs:1442,1513; call_center.rs:262; retrieval_eval.rs:292;
  two_agent_eval.rs:326,697) + tests. Production crate neuralgraph-memory: 0 references. service.rs:45 imports
  only extract_topic_keywords_universal.
- coordination::TesseractCoordinator + Router + Registry: constructed only in coordination/benchmark.rs:411,
  experiment.rs:44 and tests (coordination*.rs). No service caller.
MycelicLabs main 64027cb: "tesseract" appears nowhere. coordination.rs ports only ClaimEnvelope,
RuleBasedSynthesizer, LineageAnalyzer, min-cut; called from aggregation.rs. No router/registry/coordinator.
=> Neither Tesseract executes on any production request path in either repo.
