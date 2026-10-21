//! Lineage reconstruction: the answer to "where did this come from?" for any memory.
//!
//! For a memory id the graph is walked upward through `lineage_edges` to the raw observations at its roots and
//! returned as a plain JSON structure that answers, explicitly:
//!
//! * where it came from (`roots` and their agents, teams and source events),
//! * which memories contributed (`nodes`/`edges`),
//! * which agents and teams contributed (`contributing_agents`/`contributing_teams`),
//! * which organizational layers transformed it (`layers` and `transformations`),
//! * when it was produced (`timeline`),
//! * what confidence/support exists (`support`, including the fragility metrics of rule conclusions),
//! * whether the underlying evidence can still be reconstructed (`evidence.reconstructable`: every root is still
//!   present and not retracted, and every source event it cites is still in the log).
//!
//! Nodes the caller may not read are *redacted*, not dropped: the shape of the lineage stays truthful (the count
//! of contributions, their layer and unit) while text, agent ids and evidence references are withheld.

use crate::hierarchy::{LAYERS, layer_pos, unit_at_layer};
use crate::models::Memory;
use crate::store::Db;
use anyhow::Result;
use serde_json::{Map, Value, json};
use std::collections::{BTreeSet, HashSet};

pub const MAX_NODES: usize = 2000;

/// The memory is not in the store at all.
#[derive(Debug)]
pub struct LineageNotFound(pub String);

fn team_of(scope: &str) -> Option<String> {
    unit_at_layer(scope, "team").ok().flatten()
}

fn node_view(m: &Memory, redacted: bool, shared_entity: Option<&str>, full: bool) -> Value {
    if redacted {
        let scope =
            if m.layer == "agent" { team_of(&m.scope).map(Value::from).unwrap_or(Value::Null) } else { json!(m.scope) };
        let entity = match (&m.entity, shared_entity) {
            (Some(e), Some(s)) if !e.is_empty() && e == s => json!(e),
            _ => Value::Null,
        };
        return json!({
            "memory_id": m.memory_id, "layer": m.layer, "scope": scope, "text": null, "topic": m.topic, "slot": m.slot,
            "entity": entity, "kind": m.kind, "confidence": m.confidence, "support": m.support,
            "independent_teams": m.independent_teams, "producer_id": null, "operator": m.operator, "rule_id": m.rule_id,
            "status": m.status, "created_at": m.created_at, "applied_at": m.applied_at, "event_id": null,
            "source_event_ids": [], "local_ref": null, "redacted": true,
        });
    }
    json!({
        "memory_id": m.memory_id, "layer": m.layer, "scope": m.scope, "text": m.text, "topic": m.topic, "slot": m.slot,
        "entity": m.entity, "kind": m.kind, "confidence": m.confidence, "support": m.support,
        "independent_teams": m.independent_teams, "producer_id": m.producer_id, "operator": m.operator,
        "rule_id": m.rule_id, "status": m.status, "superseded_by": m.superseded_by, "created_at": m.created_at,
        "applied_at": m.applied_at, "event_id": m.event_id, "source_event_ids": m.source_event_ids,
        "local_ref": if full { json!(m.local_ref) } else { Value::Null },
        "fragility": m.metadata.get("fragility").cloned().unwrap_or(Value::Null), "redacted": false,
    })
}

/// The event ids a root cites: its source events and the event that carried it.
fn cited(m: &Memory) -> BTreeSet<String> {
    let mut s: BTreeSet<String> = m.source_event_ids.iter().cloned().collect();
    if let Some(e) = m.event_id.as_ref().filter(|e| !e.is_empty()) {
        s.insert(e.clone());
    }
    s
}

/// `visible` decides redaction; `full` (owner or administrator) additionally reveals the producer's local
/// reference.
pub fn reconstruct(
    db: &Db,
    memory_id: &str,
    visible: &dyn Fn(&Memory) -> bool,
    full: &dyn Fn(&Memory) -> bool,
    max_nodes: usize,
) -> Result<std::result::Result<Value, LineageNotFound>> {
    let Some(root_memory) = db.get_memory(memory_id)? else {
        return Ok(Err(LineageNotFound(memory_id.to_string())));
    };
    // insertion-ordered, like the Python dict: transformation ties keep discovery order
    let mut memories: Vec<(String, Memory)> = vec![(memory_id.to_string(), root_memory.clone())];
    let mut index: std::collections::HashMap<String, usize> =
        std::collections::HashMap::from([(memory_id.to_string(), 0)]);
    let mut edges: Vec<Value> = Vec::new();
    let mut edge_children: Vec<(String, String)> = Vec::new();
    let mut missing_parents: Vec<String> = Vec::new();
    let mut frontier = vec![memory_id.to_string()];
    let mut visited: HashSet<String> = HashSet::from([memory_id.to_string()]);
    let mut complete = true;
    while !frontier.is_empty() {
        if visited.len() > max_nodes {
            complete = false;
            break;
        }
        let mut next = Vec::new();
        for child_id in &frontier {
            if !index.contains_key(child_id) {
                continue;
            }
            for e in db.parents_of(child_id)? {
                let parent = match index.get(&e.parent_id) {
                    Some(&i) => Some(memories[i].1.clone()),
                    None => db.get_memory(&e.parent_id)?,
                };
                let parent_visible = parent.as_ref().is_some_and(visible);
                edges.push(json!({"child": child_id, "parent": e.parent_id, "parent_layer": e.parent_layer,
                                   "contributed_by": if parent_visible { json!(e.contributed_by) } else { Value::Null },
                                   "redacted": !parent_visible}));
                edge_children.push((child_id.clone(), e.parent_layer.clone()));
                let Some(parent) = parent else {
                    missing_parents.push(e.parent_id.clone());
                    continue;
                };
                if let Some(&i) = index.get(&e.parent_id) {
                    memories[i].1 = parent;
                } else {
                    index.insert(e.parent_id.clone(), memories.len());
                    memories.push((e.parent_id.clone(), parent));
                }
                if visited.insert(e.parent_id.clone()) {
                    next.push(e.parent_id.clone());
                }
            }
        }
        frontier = next;
    }

    let shared_entity = root_memory.entity.as_deref();
    let mut nodes = Map::new();
    for (mid, m) in &memories {
        nodes.insert(mid.clone(), node_view(m, !visible(m), shared_entity, full(m)));
    }
    let has_parent: HashSet<&String> = edge_children.iter().map(|(c, _)| c).collect();
    let mut roots: Vec<String> =
        memories.iter().map(|(id, _)| id.clone()).filter(|id| !has_parent.contains(id)).collect();
    roots.sort();
    let visible_mems: Vec<&Memory> = memories.iter().map(|(_, m)| m).filter(|m| visible(m)).collect();
    let redacted_count = memories.len() - visible_mems.len();
    let agents: BTreeSet<&String> =
        visible_mems.iter().filter(|m| m.layer == "agent").map(|m| &m.producer_id).collect();
    let teams: BTreeSet<String> = memories
        .iter()
        .map(|(_, m)| m)
        .filter(|m| m.layer == "agent")
        .map(|m| team_of(&m.scope).unwrap_or_else(|| m.scope.clone()))
        .collect();
    let layers_present: Vec<&str> =
        LAYERS.iter().copied().filter(|l| memories.iter().any(|(_, m)| m.layer == *l)).collect();
    let mut transformations: Vec<(usize, String, Value)> = Vec::new();
    for (_, m) in &memories {
        if m.layer == "agent" {
            continue;
        }
        let mut parent_layers: Vec<&String> = edge_children
            .iter()
            .filter(|(c, _)| *c == m.memory_id)
            .map(|(_, l)| l)
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        parent_layers.sort_by_key(|l| layer_pos(l));
        let at = m.applied_at.clone().filter(|s| !s.is_empty()).unwrap_or_else(|| m.created_at.clone());
        transformations.push((
            layer_pos(&m.layer),
            at.clone(),
            json!({"memory_id": m.memory_id, "operator": m.operator, "rule_id": m.rule_id, "from_layers": parent_layers,
                   "to_layer": m.layer, "scope": m.scope, "at": at}),
        ));
    }
    transformations.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));

    // evidence: roots must be present and active, and their source events still in the log
    let root_mems: Vec<&Memory> = roots.iter().map(|r| &memories[index[r]].1).collect();
    let all_cited: BTreeSet<String> = root_mems.iter().flat_map(|m| cited(m)).collect();
    let cited_list: Vec<String> = all_cited.into_iter().collect();
    let present = if cited_list.is_empty() { Default::default() } else { db.get_events(&cited_list)? };
    let visible_cited: BTreeSet<String> = root_mems.iter().filter(|m| visible(m)).flat_map(|m| cited(m)).collect();
    let missing_all: Vec<&String> = cited_list.iter().filter(|e| !present.contains_key(*e)).collect();
    // never name another team's event ids
    let missing_events: Vec<&String> = missing_all.iter().copied().filter(|e| visible_cited.contains(*e)).collect();
    let retracted_roots: Vec<&String> =
        root_mems.iter().filter(|m| m.status == "retracted").map(|m| &m.memory_id).collect();
    let reconstructable =
        missing_parents.is_empty() && missing_all.is_empty() && retracted_roots.is_empty() && complete;
    let observed: Vec<&String> = root_mems.iter().map(|m| &m.created_at).filter(|c| !c.is_empty()).collect();

    // version history of the queried memory
    let mut previous: Vec<String> = Vec::new();
    let mut cursor = root_memory.meta_str("version_of").filter(|s| !s.is_empty()).map(String::from);
    while let Some(c) = cursor {
        if previous.len() >= 50 {
            break;
        }
        previous.push(c.clone());
        cursor = db.get_memory(&c)?.and_then(|p| p.meta_str("version_of").filter(|s| !s.is_empty()).map(String::from));
    }

    let memory_node = nodes.get(memory_id).cloned().unwrap_or(Value::Null);
    Ok(Ok(json!({
        "memory_id": memory_id,
        "memory": memory_node,
        "nodes": nodes,
        "edges": edges,
        "roots": roots,
        "contributing_agents": agents,
        "contributing_teams": teams,
        "redacted_contributions": redacted_count,
        "layers": layers_present,
        "transformations": transformations.into_iter().map(|t| t.2).collect::<Vec<_>>(),
        "timeline": {
            "first_observed_at": observed.iter().min(),
            "last_observed_at": observed.iter().max(),
            "produced_at": root_memory.applied_at.clone().filter(|s| !s.is_empty()).unwrap_or_else(|| root_memory.created_at.clone()),
        },
        "support": {
            "agents": root_memory.support, "teams": root_memory.independent_teams,
            "confidence": root_memory.confidence, "roots": roots.len(),
            "fragility": root_memory.metadata.get("fragility").cloned().unwrap_or(Value::Null),
        },
        "evidence": {
            "source_event_ids": visible_cited,
            "events_present": present.len(), "events_missing": missing_events,
            "events_missing_total": missing_all.len(),
            "missing_parents": missing_parents, "retracted_roots": retracted_roots,
            "reconstructable": reconstructable,
        },
        "previous_versions": previous,
        "superseded_by": root_memory.superseded_by,
        "complete": complete,
    })))
}
