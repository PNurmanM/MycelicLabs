//! Aggregation: how agent observations become team, department, subsidiary, region and enterprise memories.
//!
//! Two deterministic operators run inside the consumer's apply transaction.  No model is involved, so every derived
//! memory can be explained from its lineage alone and a replay of the event log reproduces it exactly.
//!
//! **Topic consolidation** (`operator='topic_consolidation'`).  A unit U at layer L gets a memory on topic T when
//! at least `min_support` of its *direct children* contribute something on T.  A child's contribution is its own
//! consolidation on T if it has one, otherwise the best material in its subtree (recursively down to the raw agent
//! observations).  Parents of the derived memory are exactly those contributions, so the lineage records the chain
//! of units the knowledge passed through, and `support`/`independent_teams` count the distinct agents and teams
//! underneath.  When more evidence arrives the coalition grows, a new derived memory (new deterministic id)
//! supersedes the old one, and the old one stays readable as a previous version.
//!
//! **Slot composition** (`operator='slot_composition'`).  A [`Rule`] names the slots a conclusion needs.  The
//! conclusion exists only once every slot is covered for the same entity by observations from at least
//! `min_agents` agents in at least `min_teams` teams inside the rule's target unit.  Slot selection is
//! [`RuleBasedSynthesizer`] and the support metrics are [`LineageAnalyzer`]'s, with each raw observation as its
//! own lineage root and its team as its failure domain.
//!
//! **Composition and strategic synthesis.**  A rule may name `sources` other than raw observations, so the
//! conclusions of one rule (carrying `emits_slot`) or the consolidations of a unit become evidence for a higher
//! rule; `min_units` demands the evidence span distinct units of a layer and `corroborate` keeps every memory that
//! fills a slot as a parent.  Derivations cascade: a new conclusion is itself offered to the rules and
//! consolidations above it.  When evidence is retracted, every dependent conclusion is retired and then
//! re-evaluated on what remains.
//!
//! Confidence of a consolidation is the noisy-OR of the strongest contribution per child (`1 - prod(1 - c_i)`).
//! Confidence of a rule conclusion is the *minimum* over the selected slots: a conclusion is only as certain as its
//! weakest required piece.  Every float is computed in the Python implementation's order and rounded with its
//! `round`, so derived ids, confidences and stored metadata match it exactly.

use crate::coordination::{ClaimEnvelope, LineageAnalyzer, RuleBasedSynthesizer, claim_order};
use crate::hierarchy::{LAYERS, ancestors, child_unit_of, is_below, layer_of_path, layer_pos, unit_at_layer};
use crate::models::{LineageEdge, Memory, Rule, derived_memory_id};
use crate::pyjson::round;
use crate::store::{MemoryFilter, Tx};
use crate::textutil::{clip_collapsed, collapse_ws};
use anyhow::{Result, anyhow, bail};
use regex::Regex;
use serde_json::{Map, Value, json};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::LazyLock;

/// The units behind one slot of a rule, per layer named in its `min_units`: (slot, [(layer, unit paths)]).
type SlotUnits = (String, Vec<(String, Vec<String>)>);

pub const MYCELIC_PRODUCER: &str = "mycelic";
pub const CONSOLIDATABLE: [&str; 3] = ["agent_observation", "topic_consolidation", "slot_composition"];
pub const MAX_CASCADE: usize = 8;
/// per-slot claims scored for fragility (the analyzer enumerates their product)
pub const FRAGILITY_TOP_K: usize = 6;
static SLOT_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\{slot:([a-zA-Z0-9_.-]+)\}").expect("slot regex"));

/// One derived memory produced by an apply step, with everything the caller must persist and publish.
#[derive(Clone, Debug)]
pub struct Derivation {
    pub memory: Memory,
    pub edges: Vec<LineageEdge>,
    pub supersedes: Option<String>,
}

impl Derivation {
    pub fn event_payload(&self) -> Value {
        let mut payload = match self.memory.to_json() {
            Value::Object(m) => m,
            _ => Map::new(),
        };
        payload.insert("parents".into(), Value::Array(self.edges.iter().map(LineageEdge::to_json).collect()));
        payload.insert("supersedes".into(), json!(self.supersedes));
        Value::Object(payload)
    }
}

pub fn noisy_or(confidences: &[f64]) -> f64 {
    let mut p = 1.0;
    for c in confidences {
        p *= 1.0 - c.clamp(0.0, 1.0);
    }
    round(f64::min(0.99, 1.0 - p), 4)
}

fn meta_strings(m: &Memory, key: &str) -> Vec<String> {
    match m.metadata.get(key) {
        Some(Value::Array(items)) => {
            items.iter().map(|v| v.as_str().map(String::from).unwrap_or_else(|| v.to_string())).collect()
        }
        _ => Vec::new(),
    }
}

pub fn contributing_agents(m: &Memory) -> Vec<String> {
    if m.layer == "agent" { vec![m.producer_id.clone()] } else { meta_strings(m, "contributing_agents") }
}

pub fn contributing_teams(m: &Memory) -> Result<Vec<String>> {
    if m.layer == "agent" {
        return Ok(vec![unit_at_layer(&m.scope, "team")?.unwrap_or_else(|| m.scope.clone())]);
    }
    Ok(meta_strings(m, "contributing_teams"))
}

/// Units at `layer` that stand behind a memory: its own unit for a raw note, its evidence's units otherwise.
pub fn contributing_units(m: &Memory, layer: &str) -> Result<BTreeSet<String>> {
    if m.layer == "agent" {
        return Ok(unit_at_layer(&m.scope, layer)?.into_iter().collect());
    }
    let mut units = BTreeSet::new();
    for t in contributing_teams(m)? {
        if let Some(u) = unit_at_layer(&t, layer)? {
            units.insert(u);
        }
    }
    if let Some(own) = unit_at_layer(&m.scope, layer)? {
        units.insert(own);
    }
    Ok(units)
}

/// The raw observations a memory ultimately rests on (itself, for a raw note).
pub fn lineage_roots(m: &Memory) -> Vec<String> {
    if m.layer == "agent" {
        return vec![m.memory_id.clone()];
    }
    match m.metadata.get("roots") {
        Some(Value::Array(_)) => meta_strings(m, "roots"),
        Some(_) => Vec::new(),
        None => vec![m.memory_id.clone()],
    }
}

/// Rules whose conclusions a memory (transitively) rests on; empty for a raw observation.
pub fn rule_chain(m: &Memory) -> BTreeSet<String> {
    if m.layer == "agent" {
        return BTreeSet::new();
    }
    let mut chain: BTreeSet<String> = meta_strings(m, "rule_chain").into_iter().collect();
    if let Some(r) = m.rule_id.as_ref().filter(|r| !r.is_empty()) {
        chain.insert(r.clone());
    }
    chain
}

fn leaf(path: &str) -> &str {
    path.rsplit('/').next().unwrap_or(path)
}

pub fn render_consolidation(
    unit: &str,
    topic: &str,
    contributions: &BTreeMap<String, Vec<Memory>>,
    support: usize,
) -> Result<String> {
    let layer = layer_of_path(unit)?;
    let child_layer = LAYERS[layer_pos(layer).saturating_sub(1)];
    let mut statements = Vec::new();
    let mut seen = BTreeSet::new();
    for (child, group) in contributions {
        for m in group {
            let key = collapse_ws(&m.text.to_lowercase());
            if !seen.insert(key) {
                continue;
            }
            statements.push(format!("[{}] {}", leaf(child), clip_collapsed(&m.text, 220)));
        }
    }
    let plural = if support != 1 { "s" } else { "" };
    let head = format!(
        "{topic} — consolidated at {layer} '{}' from {} {child_layer} sources ({support} agent{plural}): ",
        leaf(unit),
        contributions.len()
    );
    Ok(head + &statements.join(" "))
}

pub fn render_conclusion(template: &str, entity: Option<&str>, slot_texts: &BTreeMap<String, String>) -> String {
    let out = template.replace("{entity}", entity.filter(|e| !e.is_empty()).unwrap_or("unknown entity"));
    SLOT_RE
        .replace_all(&out, |caps: &regex::Captures| {
            let name = &caps[1];
            let text = slot_texts.get(name).cloned().unwrap_or_else(|| format!("<{name}: missing>"));
            clip_collapsed(&text, 200)
        })
        .into_owned()
}

/// The one value every parent carries, or None: a parent without it (None) blocks it, so a consolidation never
/// claims a slot or entity that only some of its evidence supports.
fn common(mems: &[Memory], attr: fn(&Memory) -> Option<&str>) -> Option<String> {
    let values: BTreeSet<Option<&str>> = mems.iter().map(attr).collect();
    if values.len() == 1 { values.into_iter().next().flatten().map(String::from) } else { None }
}

fn strs(v: &[String]) -> Value {
    json!(v)
}

pub struct Aggregator {
    pub min_support: usize,
    pub max_candidates: i64,
}

impl Aggregator {
    pub fn new(min_support: i64) -> Aggregator {
        Aggregator { min_support: min_support.max(1) as usize, max_candidates: 5000 }
    }

    /// Run every operator that the given (already inserted) memory could have changed.
    ///
    /// Called inside the apply transaction, so reads see the just-inserted rows.  Returns the derivations in the
    /// order they were persisted (each later consolidation may already include an earlier one as a parent).
    pub fn derive_for(&self, tx: &Tx, memory: &Memory) -> Result<Vec<Derivation>> {
        self.derive(tx, memory, 0)
    }

    fn derive(&self, tx: &Tx, memory: &Memory, depth: usize) -> Result<Vec<Derivation>> {
        let mut out: Vec<Derivation> = Vec::new();
        if let Some(topic) = memory.topic.as_deref().filter(|t| !t.is_empty())
            && CONSOLIDATABLE.contains(&memory.operator.as_str())
        {
            out.extend(self.consolidate_topic(tx, &memory.org_id, &memory.scope, topic)?);
        }
        if memory.slot.as_deref().is_some_and(|s| !s.is_empty()) {
            out.extend(self.compose_rules(tx, memory)?);
        }
        if depth < MAX_CASCADE {
            // a derived memory is evidence for whatever sits above it: rules that take conclusions, consolidations
            // of conclusions across sibling units, and so on up to the enterprise
            let first = out.clone();
            for d in &first {
                out.extend(self.derive(tx, &d.memory, depth + 1)?);
                if let Some(old) = &d.supersedes {
                    // whatever still rests on the superseded version and was not itself replaced by the cascade (a
                    // conclusion keyed by an entity the new version no longer carries) is stale: retire it and
                    // re-evaluate it on active evidence
                    let stale = self.retire_dependents(tx, old, "evidence superseded")?;
                    out.extend(self.reevaluate(tx, &stale, depth + 1)?);
                }
            }
        } else if !out.is_empty() {
            tracing::warn!(
                "cascade truncated at depth {depth} after {}; not re-offered: {:?} (rules may form a cycle)",
                memory.memory_id,
                out.iter().map(|d| &d.memory.memory_id).collect::<Vec<_>>()
            );
        }
        Ok(out)
    }

    /// Retract every memory derived (transitively) from `memory_id`; the caller re-derives afterwards.
    pub fn retire_dependents(&self, tx: &Tx, memory_id: &str, reason: &str) -> Result<Vec<String>> {
        let mut retired = Vec::new();
        for dep in tx.dependents_of(memory_id, 10_000)? {
            if let Some(m) = tx.get_memory(&dep)?
                && m.status == "active"
            {
                tx.set_memory_status(&dep, "retracted", None, Some(reason))?;
                retired.push(dep);
            }
        }
        Ok(retired)
    }

    /// Support fell below the threshold (a stronger note changed the selection, a child unit appeared, evidence
    /// went away): retract it, retire everything built on it and re-evaluate those on what remains.
    fn withdraw(&self, tx: &Tx, current: &Memory) -> Result<Vec<Derivation>> {
        tx.set_memory_status(&current.memory_id, "retracted", None, Some("support below threshold"))?;
        let retired = self.retire_dependents(tx, &current.memory_id, "evidence withdrawn")?;
        if retired.is_empty() { Ok(Vec::new()) } else { self.reevaluate(tx, &retired, 1) }
    }

    /// Re-run the operator of each retired derived memory on the evidence that is still active.
    ///
    /// A conclusion that lost one piece of evidence may still hold on the rest (three regions minus one is still
    /// two); it then comes back as a new version, or as the reactivated earlier version whose coalition returned.
    pub fn reevaluate(&self, tx: &Tx, retired_ids: &[String], depth: usize) -> Result<Vec<Derivation>> {
        let mut out: Vec<Derivation> = Vec::new();
        for mid in retired_ids {
            let Some(m) = tx.get_memory(mid)? else { continue };
            if m.operator == "topic_consolidation"
                && let Some(topic) = m.topic.as_deref().filter(|t| !t.is_empty())
            {
                // consolidations are recomputed for the unit itself and everything above it
                let child_scope = format!("{}/x", m.scope); // any path directly below the unit climbs through it
                out.extend(self.consolidate_topic(tx, &m.org_id, &child_scope, topic)?);
            } else if m.operator == "slot_composition"
                && let Some(rule_id) = m.rule_id.as_deref().filter(|r| !r.is_empty())
                && let Some(rule) = tx.get_rule(rule_id)?
                && rule.enabled
            {
                out.extend(self.compose_rule(tx, &rule, &m.org_id, &m.scope, m.entity.as_deref())?);
            }
        }
        let first = out.clone();
        for d in &first {
            out.extend(self.derive(tx, &d.memory, depth)?);
        }
        Ok(out)
    }

    // ------------------------------------------------------------------ topic consolidation
    fn consolidate_topic(&self, tx: &Tx, org_id: &str, scope: &str, topic: &str) -> Result<Vec<Derivation>> {
        let mut results = Vec::new();
        let units = ancestors(scope, false)?;
        for unit in units.iter().rev() {
            // team first, enterprise last
            let filter = MemoryFilter {
                scope: Some(unit),
                topic: Some(topic),
                status: Some("active"),
                limit: self.max_candidates,
                newest_first: false,
                applied_only: true,
                ..Default::default()
            };
            let mems: Vec<Memory> = tx
                .list_memories(org_id, &filter)?
                .into_iter()
                .filter(|m| CONSOLIDATABLE.contains(&m.operator.as_str()) && m.scope != *unit)
                .collect();
            let contributions = self.contributions(unit, &mems)?;
            // A unit with fewer registered child units than min_support (a subsidiary with one department, an
            // enterprise with one region) would otherwise never get a memory.  Such a unit *promotes* its child's
            // own consolidation unchanged, which keeps the chain of transformations explicit in the lineage instead
            // of leaving the top layers empty.  A raw observation is never promoted: a team memory always means at
            // least `min_support` agents agreed, and a solo agent's note stays discoverable through subtree search.
            let registered_children = tx.child_units(org_id, unit)?;
            let promotion = !contributions.is_empty()
                && contributions.len() < self.min_support
                && registered_children.len() < self.min_support
                && contributions.iter().all(|(child, group)| {
                    group.len() == 1 && group[0].operator == "topic_consolidation" && group[0].scope == *child
                });
            let current = tx.current_derived(org_id, "topic_consolidation", unit, topic)?;
            if contributions.len() < self.min_support && !promotion {
                if let Some(cur) = current {
                    // support fell below the threshold (retraction): the memory no longer holds
                    results.extend(self.withdraw(tx, &cur)?);
                }
                continue;
            }
            let effective = if promotion { 1 } else { self.min_support };
            let parents: Vec<Memory> = contributions.values().flatten().cloned().collect();
            let parent_ids: Vec<String> = {
                let mut ids: Vec<String> = parents.iter().map(|m| m.memory_id.clone()).collect();
                ids.sort();
                ids
            };
            let new_id = derived_memory_id("topic_consolidation", unit, topic, &parent_ids);
            if current.as_ref().is_some_and(|c| c.memory_id == new_id) {
                continue;
            }
            let agents: Vec<String> =
                parents.iter().flat_map(contributing_agents).collect::<BTreeSet<_>>().into_iter().collect();
            let mut teams_set = BTreeSet::new();
            for m in &parents {
                teams_set.extend(contributing_teams(m)?);
            }
            let teams: Vec<String> = teams_set.into_iter().collect();
            let strongest: Vec<f64> = contributions
                .values()
                .map(|group| group.iter().map(|m| m.confidence).fold(f64::NEG_INFINITY, f64::max))
                .collect();
            let confidence = noisy_or(&strongest);
            let now = tx.now();
            let layer = layer_of_path(unit)?;
            let child_layer = LAYERS[layer_pos(layer).saturating_sub(1)];
            let roots: Vec<String> =
                parents.iter().flat_map(lineage_roots).collect::<BTreeSet<_>>().into_iter().collect();
            let chain: Vec<String> = parents.iter().flat_map(rule_chain).collect::<BTreeSet<_>>().into_iter().collect();
            let mut meta = Map::new();
            meta.insert("agg_key".into(), json!(topic));
            meta.insert("contributing_agents".into(), strs(&agents));
            meta.insert("contributing_teams".into(), strs(&teams));
            meta.insert("children".into(), json!(contributions.keys().collect::<Vec<_>>()));
            meta.insert("child_layer".into(), json!(child_layer));
            meta.insert("parent_count".into(), json!(parents.len()));
            meta.insert("version_of".into(), json!(current.as_ref().map(|c| &c.memory_id)));
            meta.insert("effective_min_support".into(), json!(effective));
            meta.insert("registered_child_units".into(), json!(registered_children.len()));
            meta.insert("promoted_from".into(), if promotion { json!(parents[0].memory_id) } else { Value::Null });
            meta.insert("roots".into(), strs(&roots));
            meta.insert("rule_chain".into(), strs(&chain));
            let memory = Memory {
                memory_id: new_id,
                org_id: org_id.to_string(),
                layer: layer.to_string(),
                scope: unit.clone(),
                text: render_consolidation(unit, topic, &contributions, agents.len())?,
                topic: Some(topic.to_string()),
                // a consolidation of same-slot evidence is itself evidence
                slot: common(&parents, |m| m.slot.as_deref()),
                entity: common(&parents, |m| m.entity.as_deref()),
                kind: common(&parents, |m| Some(m.kind.as_str()))
                    .filter(|k| !k.is_empty())
                    .unwrap_or_else(|| "fact".into()),
                confidence,
                support: agents.len() as i64,
                independent_teams: teams.len() as i64,
                producer_id: MYCELIC_PRODUCER.into(),
                operator: "topic_consolidation".into(),
                rule_id: None,
                event_id: None,
                visibility: "org".into(),
                status: "active".into(),
                superseded_by: None,
                created_at: now.clone(),
                applied_at: Some(now),
                source_event_ids: Vec::new(),
                local_ref: None,
                metadata: meta,
            };
            results.push(self.persist(tx, memory, &parents, current.as_ref())?);
        }
        Ok(results)
    }

    /// What each direct child of `unit` contributes on the topic: its own consolidation, or its subtree's best.
    ///
    /// Leaves are raw observations and rule conclusions (anything that is not itself a consolidation); a
    /// conclusion sitting exactly at a child unit represents that unit.
    fn contributions(&self, unit: &str, mems: &[Memory]) -> Result<BTreeMap<String, Vec<Memory>>> {
        // later rows win, like the Python dict comprehension
        let mut derived_by_scope: BTreeMap<String, Memory> = BTreeMap::new();
        for m in mems.iter().filter(|m| m.operator == "topic_consolidation") {
            derived_by_scope.insert(m.scope.clone(), m.clone());
        }
        let leaves: Vec<&Memory> = mems.iter().filter(|m| m.operator != "topic_consolidation").collect();
        let mut by_child: BTreeSet<String> = BTreeSet::new();
        for m in &leaves {
            by_child.insert(child_unit_of(&m.scope, unit)?);
        }
        for scope in derived_by_scope.keys() {
            if scope != unit && is_below(scope, unit) {
                by_child.insert(child_unit_of(scope, unit)?);
            }
        }
        let mut out = BTreeMap::new();
        for child in by_child {
            let best = self.best_in_subtree(&child, &derived_by_scope, &leaves)?;
            if !best.is_empty() {
                out.insert(child, best);
            }
        }
        Ok(out)
    }

    fn best_in_subtree(
        &self,
        unit: &str,
        derived_by_scope: &BTreeMap<String, Memory>,
        leaves: &[&Memory],
    ) -> Result<Vec<Memory>> {
        if let Some(d) = derived_by_scope.get(unit) {
            return Ok(vec![d.clone()]);
        }
        let here: Vec<Memory> = leaves.iter().filter(|m| m.scope == unit).map(|m| (*m).clone()).collect();
        if !here.is_empty() || layer_of_path(unit)? == "agent" {
            return Ok(here);
        }
        let mut children = BTreeSet::new();
        for m in leaves.iter().filter(|m| is_below(&m.scope, unit)) {
            children.insert(child_unit_of(&m.scope, unit)?);
        }
        let mut out = Vec::new();
        for child in children {
            out.extend(self.best_in_subtree(&child, derived_by_scope, leaves)?);
        }
        Ok(out)
    }

    // ------------------------------------------------------------------ slot composition (rules)
    fn compose_rules(&self, tx: &Tx, memory: &Memory) -> Result<Vec<Derivation>> {
        let mut results = Vec::new();
        let slot = memory.slot.as_deref().unwrap_or("");
        for rule in tx.list_rules(Some(&memory.org_id), true)? {
            if !rule.required_slots.iter().any(|s| s == slot) || !rule.sources.contains(&memory.operator) {
                continue;
            }
            if rule_chain(memory).contains(&rule.rule_id) {
                continue; // a rule never feeds on its own conclusions, however indirectly
            }
            if let Some(prefix) = rule.topic_prefix.as_deref().filter(|p| !p.is_empty())
                && !memory.topic.as_deref().unwrap_or("").starts_with(prefix)
            {
                continue;
            }
            let Some(target_unit) = unit_at_layer(&memory.scope, &rule.target_layer)? else { continue };
            results.extend(self.compose_rule(tx, &rule, &memory.org_id, &target_unit, memory.entity.as_deref())?);
        }
        Ok(results)
    }

    fn claim(m: &Memory, rule: &Rule) -> Result<ClaimEnvelope> {
        let roots = lineage_roots(m);
        let teams = contributing_teams(m)?;
        let now = crate::models::now_iso();
        let created = if m.created_at.is_empty() { now.as_str() } else { m.created_at.as_str() };
        ClaimEnvelope::validated(
            &m.memory_id,
            &rule.rule_id,
            &m.producer_id,
            m.slot.as_deref(),
            &m.text,
            m.confidence.clamp(0.0, 1.0),
            if roots.is_empty() { vec![m.memory_id.clone()] } else { roots },
            if teams.is_empty() { vec![m.scope.clone()] } else { teams },
            created,
        )
        .map_err(|e| anyhow!("ValueError: {e}"))
    }

    /// Evaluate one rule at one unit: the new conclusion (possibly with what its withdrawal re-derived), or [].
    fn compose_rule(
        &self,
        tx: &Tx,
        rule: &Rule,
        org_id: &str,
        target_unit: &str,
        entity: Option<&str>,
    ) -> Result<Vec<Derivation>> {
        let filter = MemoryFilter {
            scope: Some(target_unit),
            status: Some("active"),
            entity,
            operators: Some(&rule.sources),
            limit: self.max_candidates,
            newest_first: false,
            applied_only: true,
            ..Default::default()
        };
        let candidates: Vec<Memory> = tx
            .list_memories(org_id, &filter)?
            .into_iter()
            .filter(|m| {
                m.slot.as_ref().is_some_and(|s| rule.required_slots.contains(s))
                    && !rule_chain(m).contains(&rule.rule_id)
                    && rule
                        .topic_prefix
                        .as_deref()
                        .filter(|p| !p.is_empty())
                        .is_none_or(|p| m.topic.as_deref().unwrap_or("").starts_with(p))
                    && entity.is_none_or(|e| m.entity.as_deref() == Some(e))
            })
            .collect();
        // finer-grained evidence wins: a consolidation whose own parents are already candidates would only
        // restate them (and count them twice), so it is used only when its parents are not available here
        let ids: BTreeSet<&String> = candidates.iter().map(|m| &m.memory_id).collect();
        let mut kept = Vec::with_capacity(candidates.len());
        for m in &candidates {
            if m.operator == "topic_consolidation"
                && tx.parents_of(&m.memory_id)?.iter().all(|e| ids.contains(&e.parent_id))
            {
                continue;
            }
            kept.push(m.clone());
        }
        // a copy is not an independent source: active raw notes with the same slot and the same text (case and
        // whitespace normalised) are one source, the earliest observation standing for it. Paraphrases are not
        // detected. A verbatim re-post of a retracted note by another agent is NOT blocked: the golden scenario
        // follows_evidence specifies such a recount as evidence (an open decision, not settled here).
        let norm = |t: &str| t.split_whitespace().collect::<Vec<_>>().join(" ").to_lowercase();
        // earliest by observation time, then by arrival (the list is in store insertion order, which replay keeps)
        let mut first: BTreeMap<(String, String), (String, usize, String)> = BTreeMap::new();
        for (arrival, m) in kept.iter().enumerate().filter(|(_, m)| m.operator == "agent_observation") {
            if let Some(slot) = &m.slot {
                let rank = (m.created_at.clone(), arrival, m.memory_id.clone());
                let at = first.entry((slot.clone(), norm(&m.text))).or_insert_with(|| rank.clone());
                if rank < *at {
                    *at = rank;
                }
            }
        }
        let candidates: Vec<Memory> = kept
            .into_iter()
            .filter(|m| {
                if m.operator != "agent_observation" {
                    return true;
                }
                let Some(slot) = &m.slot else { return true };
                first.get(&(slot.clone(), norm(&m.text))).is_some_and(|(_, _, id)| *id == m.memory_id)
            })
            .collect();
        let key = format!("{}:{}", rule.rule_id, entity.filter(|e| !e.is_empty()).unwrap_or("*"));
        let current = tx.current_derived(org_id, "slot_composition", target_unit, &key)?;
        let claims: Vec<ClaimEnvelope> = candidates.iter().map(|m| Self::claim(m, rule)).collect::<Result<_>>()?;
        let slots = &rule.required_slots;
        let synthesis = RuleBasedSynthesizer { required_slots: slots }.synthesize(&claims);
        // fragility enumerates the product of per-slot candidates: score only the K strongest per slot, in the
        // synthesizer's own order so the selection is always inside the scored set
        let mut scored: Vec<ClaimEnvelope> = Vec::new();
        for slot in slots {
            let mut per: Vec<&ClaimEnvelope> =
                claims.iter().filter(|c| c.slot.as_deref() == Some(slot.as_str())).collect();
            per.sort_by(|a, b| claim_order(a, b));
            scored.extend(per.into_iter().take(FRAGILITY_TOP_K).cloned());
        }
        let by_id: BTreeMap<&String, &Memory> = candidates.iter().map(|m| (&m.memory_id, m)).collect();
        let selected: Vec<Memory> = if synthesis.success {
            synthesis
                .selected_claim_ids
                .iter()
                .map(|cid| by_id.get(cid).map(|m| (*m).clone()).ok_or_else(|| anyhow!("KeyError: {cid}")))
                .collect::<Result<_>>()?
        } else {
            Vec::new()
        };
        // with corroboration every memory that fills a required slot is evidence, not only the strongest per slot
        let evidence: Vec<Memory> = if rule.corroborate && synthesis.success {
            let mut e = candidates.clone();
            e.sort_by(|a, b| a.memory_id.cmp(&b.memory_id));
            e
        } else {
            selected.clone()
        };
        let agents: Vec<String> =
            evidence.iter().flat_map(contributing_agents).collect::<BTreeSet<_>>().into_iter().collect();
        let mut teams_set = BTreeSet::new();
        for m in &evidence {
            teams_set.extend(contributing_teams(m)?);
        }
        let teams: Vec<String> = teams_set.into_iter().collect();
        // corroboration is per slot: "supply_risk reported by two regions" counts the units behind that slot only
        let mut units: Vec<SlotUnits> = Vec::new();
        for (slot, per) in &rule.min_units {
            let mut per_layer = Vec::new();
            for (layer, _) in per {
                let mut set = BTreeSet::new();
                for m in evidence.iter().filter(|m| m.slot.as_deref() == Some(slot.as_str())) {
                    set.extend(contributing_units(m, layer)?);
                }
                per_layer.push((layer.clone(), set.into_iter().collect::<Vec<_>>()));
            }
            units.push((slot.clone(), per_layer));
        }
        let enough_units = rule
            .min_units
            .iter()
            .zip(&units)
            .all(|((_, per), (_, got))| per.iter().zip(got).all(|((_, n), (_, us))| us.len() as i64 >= *n));
        if !synthesis.success
            || (agents.len() as i64) < rule.min_agents
            || (teams.len() as i64) < rule.min_teams
            || !enough_units
        {
            return match current {
                Some(cur) => self.withdraw(tx, &cur),
                None => Ok(Vec::new()),
            };
        }
        let parent_ids: Vec<String> = {
            let mut ids: Vec<String> = evidence.iter().map(|m| m.memory_id.clone()).collect();
            ids.sort();
            ids
        };
        let new_id = derived_memory_id("slot_composition", target_unit, &key, &parent_ids);
        if current.as_ref().is_some_and(|c| c.memory_id == new_id) {
            return Ok(Vec::new());
        }
        let metrics = LineageAnalyzer::score(&scored, slots, &synthesis);
        let mut slot_texts = BTreeMap::new();
        for m in &selected {
            if let Some(s) = m.slot.as_ref().filter(|s| !s.is_empty()) {
                slot_texts.insert(s.clone(), m.text.clone());
            }
        }
        let confidence = if rule.corroborate {
            // confidence per slot rises with independent corroboration (noisy-OR over the units filling it); the
            // conclusion is as certain as its weakest slot
            let mut per_slot = Vec::new();
            for slot in slots {
                let mut best_by_unit: Vec<(String, f64)> = Vec::new();
                for m in evidence.iter().filter(|m| m.slot.as_deref() == Some(slot.as_str())) {
                    let unit = if m.layer != "agent" {
                        m.scope.clone()
                    } else {
                        unit_at_layer(&m.scope, "team")?.unwrap_or_else(|| m.scope.clone())
                    };
                    match best_by_unit.iter_mut().find(|(u, _)| *u == unit) {
                        Some((_, c)) => *c = f64::max(*c, m.confidence),
                        None => best_by_unit.push((unit, f64::max(0.0, m.confidence))),
                    }
                }
                per_slot.push(noisy_or(&best_by_unit.iter().map(|(_, c)| *c).collect::<Vec<_>>()));
            }
            per_slot.iter().copied().reduce(f64::min).map(|c| round(c, 4)).unwrap_or(0.0)
        } else {
            round(synthesis.confidence, 4)
        };
        let now = tx.now();
        let mut slots_meta = Map::new();
        for m in &selected {
            if let Some(s) = m.slot.as_ref().filter(|s| !s.is_empty()) {
                slots_meta.insert(s.clone(), json!(m.memory_id));
            }
        }
        let mut evidence_meta = Map::new();
        for m in &evidence {
            evidence_meta.insert(
                m.memory_id.clone(),
                json!({"slot": m.slot, "layer": m.layer, "scope": m.scope, "operator": m.operator}),
            );
        }
        let mut units_meta = Map::new();
        for (slot, per) in &units {
            let mut inner = Map::new();
            for (layer, us) in per {
                inner.insert(layer.clone(), json!(us));
            }
            units_meta.insert(slot.clone(), Value::Object(inner));
        }
        let roots: Vec<String> = evidence.iter().flat_map(lineage_roots).collect::<BTreeSet<_>>().into_iter().collect();
        let mut chain: BTreeSet<String> = BTreeSet::from([rule.rule_id.clone()]);
        for m in &evidence {
            chain.extend(rule_chain(m));
        }
        let mut meta = Map::new();
        meta.insert("agg_key".into(), json!(key));
        meta.insert("contributing_agents".into(), strs(&agents));
        meta.insert("contributing_teams".into(), strs(&teams));
        meta.insert("slots".into(), Value::Object(slots_meta));
        meta.insert("candidates".into(), json!(candidates.len()));
        meta.insert("fragility_scored_candidates".into(), json!(scored.len()));
        meta.insert("evidence".into(), Value::Object(evidence_meta));
        meta.insert("corroborated_units".into(), Value::Object(units_meta));
        meta.insert("roots".into(), strs(&roots));
        meta.insert("rule_chain".into(), json!(chain));
        meta.insert("fragility".into(), metrics.to_json());
        meta.insert("version_of".into(), json!(current.as_ref().map(|c| &c.memory_id)));
        let topic = rule
            .emits_topic
            .clone()
            .filter(|t| !t.is_empty())
            .or_else(|| rule.topic_prefix.clone().filter(|t| !t.is_empty()))
            .unwrap_or_else(|| rule.rule_id.clone());
        let memory = Memory {
            memory_id: new_id,
            org_id: org_id.to_string(),
            layer: rule.target_layer.clone(),
            scope: target_unit.to_string(),
            text: render_conclusion(&rule.conclusion, entity, &slot_texts),
            topic: Some(topic),
            slot: rule.emits_slot.clone(),
            entity: entity.map(String::from),
            kind: rule.kind.clone(),
            confidence,
            support: agents.len() as i64,
            independent_teams: teams.len() as i64,
            producer_id: MYCELIC_PRODUCER.into(),
            operator: "slot_composition".into(),
            rule_id: Some(rule.rule_id.clone()),
            event_id: None,
            visibility: "org".into(),
            status: "active".into(),
            superseded_by: None,
            created_at: now.clone(),
            applied_at: Some(now),
            source_event_ids: Vec::new(),
            local_ref: None,
            metadata: meta,
        };
        Ok(vec![self.persist(tx, memory, &evidence, current.as_ref())?])
    }

    // ------------------------------------------------------------------ persistence
    fn persist(&self, tx: &Tx, mut memory: Memory, parents: &[Memory], current: Option<&Memory>) -> Result<Derivation> {
        let now = memory.created_at.clone();
        let edges: Vec<LineageEdge> = parents
            .iter()
            .map(|p| LineageEdge {
                child_id: memory.memory_id.clone(),
                parent_id: p.memory_id.clone(),
                contributed_by: if p.layer == "agent" { p.producer_id.clone() } else { MYCELIC_PRODUCER.into() },
                parent_layer: p.layer.clone(),
                created_at: now.clone(),
            })
            .collect();
        let mut supersedes = None;
        if let Some(cur) = current {
            // before the insert: only one active memory per (unit, operator, key)
            tx.set_memory_status(&cur.memory_id, "superseded", Some(&memory.memory_id), Some("coalition changed"))?;
            supersedes = Some(cur.memory_id.clone());
        }
        match tx.get_memory(&memory.memory_id)? {
            None => {
                tx.insert_memory(&memory)?;
            }
            Some(existing)
                if existing.status != "active"
                    && existing.operator == memory.operator
                    && existing.scope == memory.scope =>
            {
                // the exact earlier coalition is back (evidence was retracted, or a retraction was undone by new
                // evidence): the earlier derived memory becomes current again, keeping its id and its lineage edges
                let version_of = match current {
                    Some(cur) => json!(cur.memory_id),
                    None => existing.metadata.get("version_of").cloned().unwrap_or(Value::Null),
                };
                let mut meta = existing.metadata.clone();
                for (k, v) in &memory.metadata {
                    meta.insert(k.clone(), v.clone());
                }
                meta.insert("version_of".into(), version_of);
                meta.insert("reactivated_at".into(), json!(now));
                // the reason it was retired does not describe an active memory
                meta.shift_remove("status_reason");
                tx.reactivate_memory(&memory.memory_id, &now, &meta)?;
                if let Some(m) = tx.get_memory(&memory.memory_id)? {
                    memory = m;
                }
            }
            Some(_) => {
                bail!("RuntimeError: derived memory id collision for {}; refusing to attach lineage", memory.memory_id)
            }
        }
        tx.add_lineage_edges(&edges)?;
        tracing::info!(
            "derived {} at {} ({}) from {} parents{}",
            memory.memory_id,
            memory.scope,
            memory.operator,
            parents.len(),
            supersedes.as_ref().map(|s| format!(", supersedes {s}")).unwrap_or_default()
        );
        Ok(Derivation { memory, edges, supersedes })
    }
}
