//! Claim contracts, rule-based synthesis and lineage/fragility scoring used by the aggregator.
//!
//! A port of the parts of the coordination package (copied from NeuralGraph's `research.coordination` at
//! e7ef73ea) that Mycelic calls: `ClaimEnvelope`, `RuleBasedSynthesizer` and `LineageAnalyzer`.  The rest of that
//! package (router, capability registry, failure injector, mock adapters, trace logger) was research scaffolding
//! with no caller in the service and was not ported.
//!
//! Every number keeps Python's arithmetic and rounding (`round(x, 6)`), so a conclusion's stored fragility metrics
//! are identical whichever implementation derived it.

use crate::models::parse_iso;
use crate::pyjson::round;
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet};

/// One piece of evidence offered to a rule: a memory, seen as a claim about one slot.
#[derive(Clone, Debug, PartialEq)]
pub struct ClaimEnvelope {
    pub claim_id: String,
    pub producer_node_id: String,
    pub slot: Option<String>,
    pub value: String,
    pub confidence: f64,
    pub lineage_root_ids: Vec<String>,
    pub failure_domains: Vec<String>,
}

impl ClaimEnvelope {
    /// The validation the Python dataclass ran in `__post_init__` (ids non-empty, confidence in [0, 1], an
    /// ISO-8601 creation time); a claim that fails it makes the whole apply step fail, as before.
    pub fn validated(
        claim_id: &str,
        query_id: &str,
        producer_node_id: &str,
        slot: Option<&str>,
        value: &str,
        confidence: f64,
        lineage_root_ids: Vec<String>,
        failure_domains: Vec<String>,
        created_at: &str,
    ) -> Result<ClaimEnvelope, String> {
        for (v, name) in [(claim_id, "claim_id"), (query_id, "query_id"), (producer_node_id, "producer_node_id")] {
            if crate::textutil::py_strip(v).is_empty() {
                return Err(format!("{name} must be a non-empty string"));
            }
        }
        if !(0.0..=1.0).contains(&confidence) {
            return Err("claim confidence must be between 0 and 1".into());
        }
        if crate::textutil::py_strip(created_at).is_empty() {
            return Err("created_at must be a non-empty string".into());
        }
        if parse_iso(created_at).is_none() {
            return Err("created_at must be an ISO-8601 timestamp".into());
        }
        Ok(ClaimEnvelope {
            claim_id: claim_id.into(),
            producer_node_id: producer_node_id.into(),
            slot: slot.map(str::to_string),
            value: value.into(),
            confidence,
            lineage_root_ids,
            failure_domains,
        })
    }

    fn sort_key(&self) -> (OrdF64, &str, &str) {
        (OrdF64(-self.confidence), self.producer_node_id.as_str(), self.claim_id.as_str())
    }
}

/// Total order on f64 for sort keys (the values are finite confidences).
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct OrdF64(pub f64);

impl Eq for OrdF64 {}

impl Ord for OrdF64 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&other.0)
    }
}

/// Python sorts `(-confidence, producer, claim_id)`; `-0.0` and `0.0` compare equal there, so normalise.
pub fn claim_order(a: &ClaimEnvelope, b: &ClaimEnvelope) -> std::cmp::Ordering {
    let (ca, pa, ia) = a.sort_key();
    let (cb, pb, ib) = b.sort_key();
    ca.0.partial_cmp(&cb.0).unwrap_or(std::cmp::Ordering::Equal).then_with(|| pa.cmp(pb)).then_with(|| ia.cmp(ib))
}

#[derive(Clone, Debug, PartialEq)]
pub struct SynthesisResult {
    pub success: bool,
    pub answer: Option<String>,
    pub confidence: f64,
    pub selected_claim_ids: Vec<String>,
    pub selected_node_ids: Vec<String>,
    pub covered_slots: Vec<String>,
}

impl SynthesisResult {
    pub fn to_json(&self) -> Value {
        json!({"success": self.success, "answer": self.answer, "confidence": self.confidence,
               "selected_claim_ids": self.selected_claim_ids, "selected_node_ids": self.selected_node_ids,
               "covered_slots": self.covered_slots})
    }
}

/// Fixture-neutral slot composition; no model output is involved.
pub struct RuleBasedSynthesizer<'a> {
    pub required_slots: &'a [String],
}

impl RuleBasedSynthesizer<'_> {
    pub fn synthesize(&self, claims: &[ClaimEnvelope]) -> SynthesisResult {
        let mut selected: Vec<&ClaimEnvelope> = Vec::new();
        let mut covered = Vec::new();
        for slot in self.required_slots {
            let mut candidates: Vec<&ClaimEnvelope> =
                claims.iter().filter(|c| c.slot.as_deref() == Some(slot.as_str()) && c.value != "[REDACTED]").collect();
            if candidates.is_empty() {
                continue;
            }
            candidates.sort_by(|a, b| claim_order(a, b));
            selected.push(candidates[0]);
            covered.push(slot.clone());
        }
        let success = covered.len() == self.required_slots.len();
        let mut answer = None;
        let mut confidence = claims.iter().map(|c| c.confidence).fold(None, |acc: Option<f64>, c| match acc {
            Some(a) if a >= c => Some(a),
            _ => Some(c),
        });
        if success {
            answer = Some(selected.iter().map(|c| c.value.clone()).collect::<Vec<_>>().join(":"));
            confidence = selected.iter().map(|c| c.confidence).fold(None, |acc: Option<f64>, c| match acc {
                Some(a) if a <= c => Some(a),
                _ => Some(c),
            });
        }
        SynthesisResult {
            success,
            answer,
            confidence: round(confidence.unwrap_or(0.0), 6),
            selected_claim_ids: selected.iter().map(|c| c.claim_id.clone()).collect(),
            selected_node_ids: selected.iter().map(|c| c.producer_node_id.clone()).collect(),
            covered_slots: covered,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct ReconstructionCoalition {
    pub lineage_root_ids: Vec<String>,
    pub failure_domains: Vec<String>,
    pub claim_ids: Vec<String>,
}

impl ReconstructionCoalition {
    pub fn to_json(&self) -> Value {
        json!({"lineage_root_ids": self.lineage_root_ids, "failure_domains": self.failure_domains,
               "claim_ids": self.claim_ids})
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct FragilityMetrics {
    pub apparent_replica_count: usize,
    pub unique_lineage_root_count: usize,
    pub unique_failure_domain_count: usize,
    pub reconstruction_coalition_count: usize,
    pub minimal_support_coalition_size: usize,
    pub minimal_failure_domain_cut: usize,
    pub current_task_success: bool,
    pub current_confidence: f64,
    pub valid_support_coverage: f64,
    pub confidence_valid_support_gap: f64,
}

impl FragilityMetrics {
    pub fn to_json(&self) -> Value {
        json!({
            "apparent_replica_count": self.apparent_replica_count,
            "unique_lineage_root_count": self.unique_lineage_root_count,
            "unique_failure_domain_count": self.unique_failure_domain_count,
            "reconstruction_coalition_count": self.reconstruction_coalition_count,
            "minimal_support_coalition_size": self.minimal_support_coalition_size,
            "minimal_failure_domain_cut": self.minimal_failure_domain_cut,
            "current_task_success": self.current_task_success,
            "current_confidence": self.current_confidence,
            "valid_support_coverage": self.valid_support_coverage,
            "confidence_valid_support_gap": self.confidence_valid_support_gap,
        })
    }
}

pub struct LineageAnalyzer;

impl LineageAnalyzer {
    /// Every way of filling each required slot with one claim, reduced to its distinct (roots, domains) pairs;
    /// the lexicographically smallest claim set represents each pair.  Ordered by (roots, domains).
    pub fn reconstruction_coalitions(
        claims: &[ClaimEnvelope],
        required_slots: &[String],
    ) -> Vec<ReconstructionCoalition> {
        let per_slot: Vec<Vec<&ClaimEnvelope>> = required_slots
            .iter()
            .map(|slot| claims.iter().filter(|c| c.slot.as_deref() == Some(slot.as_str())).collect())
            .collect();
        if per_slot.iter().any(Vec::is_empty) {
            return Vec::new();
        }
        let mut canonical: BTreeMap<(Vec<String>, Vec<String>), ReconstructionCoalition> = BTreeMap::new();
        let mut index = vec![0usize; per_slot.len()];
        loop {
            let selected: Vec<&ClaimEnvelope> = index.iter().enumerate().map(|(s, &i)| per_slot[s][i]).collect();
            let roots: Vec<String> = selected
                .iter()
                .flat_map(|c| c.lineage_root_ids.iter().cloned())
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect();
            let domains: Vec<String> = selected
                .iter()
                .flat_map(|c| c.failure_domains.iter().cloned())
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect();
            if !roots.is_empty() && !domains.is_empty() {
                let mut claim_ids: Vec<String> = selected.iter().map(|c| c.claim_id.clone()).collect();
                claim_ids.sort();
                let key = (roots.clone(), domains.clone());
                let replace = match canonical.get(&key) {
                    None => true,
                    Some(existing) => claim_ids < existing.claim_ids,
                };
                if replace {
                    canonical.insert(
                        key,
                        ReconstructionCoalition { lineage_root_ids: roots, failure_domains: domains, claim_ids },
                    );
                }
            }
            // advance the mixed-radix counter in itertools.product order (last slot fastest)
            let mut pos = per_slot.len();
            loop {
                if pos == 0 {
                    return canonical.into_values().collect();
                }
                pos -= 1;
                index[pos] += 1;
                if index[pos] < per_slot[pos].len() {
                    break;
                }
                index[pos] = 0;
            }
        }
    }

    pub fn score(claims: &[ClaimEnvelope], required_slots: &[String], synthesis: &SynthesisResult) -> FragilityMetrics {
        let coalitions = Self::reconstruction_coalitions(claims, required_slots);
        let roots: BTreeSet<&String> = claims.iter().flat_map(|c| c.lineage_root_ids.iter()).collect();
        let domains: BTreeSet<&String> = claims.iter().flat_map(|c| c.failure_domains.iter()).collect();
        let min_size = coalitions.iter().map(|c| c.lineage_root_ids.len()).min().unwrap_or(0);
        let min_cut = Self::minimum_domain_cut(&coalitions);
        let coverage = if required_slots.is_empty() {
            1.0
        } else {
            synthesis.covered_slots.len() as f64 / required_slots.len() as f64
        };
        let gap = f64::max(0.0, synthesis.confidence - coverage);
        FragilityMetrics {
            apparent_replica_count: claims.len(),
            unique_lineage_root_count: roots.len(),
            unique_failure_domain_count: domains.len(),
            reconstruction_coalition_count: coalitions.len(),
            minimal_support_coalition_size: min_size,
            minimal_failure_domain_cut: min_cut,
            current_task_success: synthesis.success,
            current_confidence: round(synthesis.confidence, 6),
            valid_support_coverage: round(coverage, 6),
            confidence_valid_support_gap: round(gap, 6),
        }
    }

    /// The smallest number of failure domains whose loss breaks every coalition.
    fn minimum_domain_cut(coalitions: &[ReconstructionCoalition]) -> usize {
        if coalitions.is_empty() {
            return 0;
        }
        let domains: Vec<&String> =
            coalitions.iter().flat_map(|c| c.failure_domains.iter()).collect::<BTreeSet<_>>().into_iter().collect();
        let sets: Vec<BTreeSet<&String>> = coalitions.iter().map(|c| c.failure_domains.iter().collect()).collect();
        for size in 1..=domains.len() {
            let mut combo: Vec<usize> = (0..size).collect();
            loop {
                if sets.iter().all(|s| combo.iter().any(|&i| s.contains(domains[i]))) {
                    return size;
                }
                // next combination in lexicographic order
                let mut i = size;
                let mut advanced = false;
                while i > 0 {
                    i -= 1;
                    if combo[i] < domains.len() - size + i {
                        combo[i] += 1;
                        for j in i + 1..size {
                            combo[j] = combo[j - 1] + 1;
                        }
                        advanced = true;
                        break;
                    }
                }
                if !advanced {
                    break;
                }
            }
        }
        0
    }
}
