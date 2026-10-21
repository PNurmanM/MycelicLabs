//! Plain records for Mycelic.  The store is the only writer; every record serialises to the JSON shape the
//! Python service produced (`dataclasses.asdict` field order), which is also its event payload on the wire.
//!
//! Timestamps are ISO-8601 UTC strings with second precision (`2026-09-24T10:00:00+00:00`) so SQLite string
//! comparison orders them.

use crate::hierarchy::{LAYERS, layer_pos};
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};
use std::sync::Arc;

pub const MEMORY_KINDS: [&str; 8] = ["fact", "observation", "event", "risk", "decision", "plan", "preference", "other"];
pub const VISIBILITY: [&str; 2] = ["team", "org"];
pub const OPERATORS: [&str; 3] = ["agent_observation", "topic_consolidation", "slot_composition"];
pub const DEFAULT_AGENT_SCOPES: [&str; 4] = ["memory:read", "memory:write", "events:write", "lineage:read"];
pub const ALL_SCOPES: [&str; 5] = ["memory:read", "memory:write", "events:write", "lineage:read", "admin"];

/// Where "now" comes from.  The service reads time only through this, so tests can pin it.
pub type Clock = Arc<dyn Fn() -> String + Send + Sync>;
/// Where random ids (`evt_…`, `mem_…`) come from; pinned to a counter by the golden tests.
pub type IdGen = Arc<dyn Fn(&str) -> String + Send + Sync>;

pub fn system_clock() -> Clock {
    Arc::new(now_iso)
}

pub fn random_ids() -> IdGen {
    Arc::new(new_id)
}

pub fn now_iso() -> String {
    chrono::Utc::now().format("%Y-%m-%dT%H:%M:%S+00:00").to_string()
}

/// `prefix_<20 hex chars>`, the first 20 hex digits of a random UUID like the Python `uuid4().hex[:20]`.
pub fn new_id(prefix: &str) -> String {
    let hex = uuid::Uuid::new_v4().simple().to_string();
    format!("{prefix}_{}", &hex[..20])
}

fn sha256_hex(data: &[u8]) -> String {
    hex::encode(Sha256::digest(data))
}

/// Deterministic id for an aggregated memory.
///
/// The id is a function of *what was aggregated* (operator, target unit, topic/rule key, sorted parents), never
/// of time or process identity.  Replaying the event log therefore reproduces exactly the same derived ids, which
/// is what makes `apply` idempotent and a rebuild from JetStream byte-comparable with the state it replaces.
pub fn derived_memory_id(operator: &str, scope: &str, key: &str, parent_ids: &[String]) -> String {
    let mut sorted: Vec<&str> = parent_ids.iter().map(String::as_str).collect();
    sorted.sort_unstable();
    let mut parts: Vec<&str> = vec![operator, scope, key];
    parts.extend(sorted);
    let digest = sha256_hex(parts.join("|").as_bytes());
    format!("mem_d{}", &digest[..22])
}

/// SHA-256 over the parts joined by the unit separator, first 32 hex characters (idempotency keys, event ids).
pub fn content_hash(parts: &[&str]) -> String {
    sha256_hex(parts.join("\u{1f}").as_bytes())[..32].to_string()
}

// ---------------------------------------------------------------------------------------------------- records
#[derive(Clone, Debug, PartialEq)]
pub struct Agent {
    pub agent_id: String,
    pub org_id: String,
    pub display_name: String,
    /// full six-segment path, see `hierarchy::AgentPath`
    pub path: String,
    pub scopes: Vec<String>,
    /// first characters of the key, for display/rotation only
    pub key_prefix: String,
    pub status: String,
    pub created_at: String,
    pub last_seen_at: Option<String>,
    pub metadata: Map<String, Value>,
}

impl Agent {
    pub fn to_json(&self) -> Value {
        json!({
            "agent_id": self.agent_id, "org_id": self.org_id, "display_name": self.display_name, "path": self.path,
            "scopes": self.scopes, "key_prefix": self.key_prefix, "status": self.status, "created_at": self.created_at,
            "last_seen_at": self.last_seen_at, "metadata": self.metadata,
        })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Memory {
    pub memory_id: String,
    pub org_id: String,
    /// one of `hierarchy::LAYERS`
    pub layer: String,
    /// unit path the memory belongs to (agent path for layer 'agent')
    pub scope: String,
    pub text: String,
    pub topic: Option<String>,
    pub slot: Option<String>,
    pub entity: Option<String>,
    pub kind: String,
    pub confidence: f64,
    /// distinct agents whose observations are in the lineage
    pub support: i64,
    /// distinct teams in the lineage (independent failure domains)
    pub independent_teams: i64,
    /// agent id, or 'mycelic' for derived memories
    pub producer_id: String,
    pub operator: String,
    pub rule_id: Option<String>,
    /// event that created/published this memory
    pub event_id: Option<String>,
    pub visibility: String,
    pub status: String,
    pub superseded_by: Option<String>,
    /// producer time (the agent's observed_at for raw memories)
    pub created_at: String,
    /// when the consumer applied the event that carries it
    pub applied_at: Option<String>,
    pub source_event_ids: Vec<String>,
    /// the agent's own local memory id (opaque to Mycelic)
    pub local_ref: Option<String>,
    pub metadata: Map<String, Value>,
}

impl Memory {
    pub fn layer_index(&self) -> usize {
        layer_pos(&self.layer)
    }

    pub fn meta_str(&self, key: &str) -> Option<&str> {
        self.metadata.get(key).and_then(Value::as_str)
    }

    /// A memory from its JSON form (`to_json`'s inverse), for fixtures and tools.
    pub fn from_json(v: &Value) -> Option<Memory> {
        let s = |k: &str| v.get(k).and_then(Value::as_str).map(String::from);
        let strings = |k: &str| {
            v.get(k)
                .and_then(Value::as_array)
                .map(|a| a.iter().filter_map(|x| x.as_str().map(String::from)).collect())
                .unwrap_or_default()
        };
        Some(Memory {
            memory_id: s("memory_id")?,
            org_id: s("org_id")?,
            layer: s("layer")?,
            scope: s("scope")?,
            text: s("text").unwrap_or_default(),
            topic: s("topic"),
            slot: s("slot"),
            entity: s("entity"),
            kind: s("kind").unwrap_or_default(),
            confidence: v.get("confidence").and_then(Value::as_f64).unwrap_or(0.0),
            support: v.get("support").and_then(Value::as_i64).unwrap_or(1),
            independent_teams: v.get("independent_teams").and_then(Value::as_i64).unwrap_or(1),
            producer_id: s("producer_id").unwrap_or_default(),
            operator: s("operator").unwrap_or_default(),
            rule_id: s("rule_id"),
            event_id: s("event_id"),
            visibility: s("visibility").unwrap_or_else(|| "team".into()),
            status: s("status").unwrap_or_else(|| "active".into()),
            superseded_by: s("superseded_by"),
            created_at: s("created_at").unwrap_or_default(),
            applied_at: s("applied_at"),
            source_event_ids: strings("source_event_ids"),
            local_ref: s("local_ref"),
            metadata: v.get("metadata").and_then(Value::as_object).cloned().unwrap_or_default(),
        })
    }

    pub fn to_json(&self) -> Value {
        let mut m = Map::new();
        m.insert("memory_id".into(), json!(self.memory_id));
        m.insert("org_id".into(), json!(self.org_id));
        m.insert("layer".into(), json!(self.layer));
        m.insert("scope".into(), json!(self.scope));
        m.insert("text".into(), json!(self.text));
        m.insert("topic".into(), json!(self.topic));
        m.insert("slot".into(), json!(self.slot));
        m.insert("entity".into(), json!(self.entity));
        m.insert("kind".into(), json!(self.kind));
        m.insert("confidence".into(), json!(self.confidence));
        m.insert("support".into(), json!(self.support));
        m.insert("independent_teams".into(), json!(self.independent_teams));
        m.insert("producer_id".into(), json!(self.producer_id));
        m.insert("operator".into(), json!(self.operator));
        m.insert("rule_id".into(), json!(self.rule_id));
        m.insert("event_id".into(), json!(self.event_id));
        m.insert("visibility".into(), json!(self.visibility));
        m.insert("status".into(), json!(self.status));
        m.insert("superseded_by".into(), json!(self.superseded_by));
        m.insert("created_at".into(), json!(self.created_at));
        m.insert("applied_at".into(), json!(self.applied_at));
        m.insert("source_event_ids".into(), json!(self.source_event_ids));
        m.insert("local_ref".into(), json!(self.local_ref));
        m.insert("metadata".into(), Value::Object(self.metadata.clone()));
        Value::Object(m)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct LineageEdge {
    pub child_id: String,
    pub parent_id: String,
    /// agent id (raw parent) or 'mycelic'
    pub contributed_by: String,
    pub parent_layer: String,
    pub created_at: String,
}

impl LineageEdge {
    pub fn to_json(&self) -> Value {
        json!({"child_id": self.child_id, "parent_id": self.parent_id, "contributed_by": self.contributed_by,
               "parent_layer": self.parent_layer, "created_at": self.created_at})
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct EventRecord {
    pub event_id: String,
    pub kind: String,
    pub org_id: String,
    pub agent_id: Option<String>,
    /// NATS subject
    pub subject: String,
    pub payload: Value,
    pub status: String,
    pub js_seq: Option<i64>,
    pub attempts: i64,
    pub last_error: Option<String>,
    pub created_at: String,
    pub published_at: Option<String>,
    pub applied_at: Option<String>,
}

impl EventRecord {
    pub fn new(
        event_id: String,
        kind: &str,
        org_id: &str,
        agent_id: Option<String>,
        subject: String,
        payload: Value,
        created_at: String,
    ) -> EventRecord {
        EventRecord {
            event_id,
            kind: kind.to_string(),
            org_id: org_id.to_string(),
            agent_id,
            subject,
            payload,
            status: "pending".into(),
            js_seq: None,
            attempts: 0,
            last_error: None,
            created_at,
            published_at: None,
            applied_at: None,
        }
    }

    pub fn to_json(&self) -> Value {
        json!({
            "event_id": self.event_id, "kind": self.kind, "org_id": self.org_id, "agent_id": self.agent_id,
            "subject": self.subject, "payload": self.payload, "status": self.status, "js_seq": self.js_seq,
            "attempts": self.attempts, "last_error": self.last_error, "created_at": self.created_at,
            "published_at": self.published_at, "applied_at": self.applied_at,
        })
    }
}

/// A slot-composition rule: a conclusion that only exists once every required slot is covered.
///
/// Rules compose.  A rule whose `emits_slot` is set produces conclusions that carry that slot, so a higher rule
/// can list it among its `required_slots` and take conclusions as evidence (`sources` names the operators whose
/// memories may fill a slot).  `min_units` demands corroboration of a slot across organizational units
/// (`{"supply_risk": {"region": 2}}`: supply risk reported by at least two regions); the units are counted over
/// the memories that become parents, which without `corroborate` is the single strongest memory per slot, so a
/// count above one needs `corroborate` unless that one memory itself spans the units.  With `corroborate` every
/// memory that fills a required slot becomes evidence, not only the strongest one per slot.
#[derive(Clone, Debug, PartialEq)]
pub struct Rule {
    pub rule_id: String,
    pub target_layer: String,
    pub required_slots: Vec<String>,
    /// template with `{entity}` and `{slot:<name>}` placeholders
    pub conclusion: String,
    /// only memories whose topic starts with this feed the rule
    pub topic_prefix: Option<String>,
    pub min_agents: i64,
    pub min_teams: i64,
    pub kind: String,
    /// None = every organization on this deployment
    pub org_id: Option<String>,
    pub enabled: bool,
    pub metadata: Map<String, Value>,
    /// operators that may fill a slot
    pub sources: Vec<String>,
    /// the slot the conclusion carries, so higher rules can consume it
    pub emits_slot: Option<String>,
    /// the topic the conclusion carries (default: topic_prefix or rule_id)
    pub emits_topic: Option<String>,
    /// slot -> layer -> distinct units, in the order given
    pub min_units: Vec<(String, Vec<(String, i64)>)>,
    /// every memory filling a required slot is evidence, not just the strongest
    pub corroborate: bool,
}

impl Rule {
    pub fn min_units_json(&self) -> Value {
        let mut out = Map::new();
        for (slot, per) in &self.min_units {
            let mut inner = Map::new();
            for (layer, n) in per {
                inner.insert(layer.clone(), json!(n));
            }
            out.insert(slot.clone(), Value::Object(inner));
        }
        Value::Object(out)
    }

    pub fn to_json(&self) -> Value {
        json!({
            "rule_id": self.rule_id, "target_layer": self.target_layer, "required_slots": self.required_slots,
            "conclusion": self.conclusion, "topic_prefix": self.topic_prefix, "min_agents": self.min_agents,
            "min_teams": self.min_teams, "kind": self.kind, "org_id": self.org_id, "enabled": self.enabled,
            "metadata": self.metadata, "sources": self.sources, "emits_slot": self.emits_slot,
            "emits_topic": self.emits_topic, "min_units": self.min_units_json(), "corroborate": self.corroborate,
        })
    }
}

// ---------------------------------------------------------------------------------------------------- ISO-8601
/// A parsed timestamp as Python 3.11's `datetime.fromisoformat` reads one, with the offset it was given.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IsoDateTime {
    pub date: chrono::NaiveDate,
    pub hour: u32,
    pub minute: u32,
    pub second: u32,
    /// UTC offset in whole seconds; naive input counts as UTC
    pub offset_seconds: i32,
}

impl IsoDateTime {
    /// `isoformat(timespec="seconds")` of the aware datetime (naive input was assumed to be UTC).
    pub fn isoformat_seconds(&self) -> String {
        let off = self.offset_seconds;
        let sign = if off < 0 { '-' } else { '+' };
        let a = off.unsigned_abs();
        let (h, m, s) = (a / 3600, (a % 3600) / 60, a % 60);
        let offset = if s != 0 { format!("{sign}{h:02}:{m:02}:{s:02}") } else { format!("{sign}{h:02}:{m:02}") };
        format!("{}T{:02}:{:02}:{:02}{offset}", self.date.format("%Y-%m-%d"), self.hour, self.minute, self.second)
    }
}

fn digits(cs: &[char], from: usize, n: usize) -> Option<u32> {
    if from + n > cs.len() {
        return None;
    }
    let mut v = 0u32;
    for c in &cs[from..from + n] {
        v = v * 10 + c.to_digit(10).filter(|_| c.is_ascii_digit())?;
    }
    Some(v)
}

fn find_separator(cs: &[char]) -> Option<usize> {
    let n = cs.len();
    if n == 7 {
        return Some(7);
    }
    if n < 8 {
        return None;
    }
    if cs[4] == '-' {
        if cs[5] == 'W' {
            if n > 8 && cs[8] == '-' {
                if n == 9 {
                    return None;
                }
                if n > 10 && cs[10].is_ascii_digit() {
                    return Some(8);
                }
                return Some(10);
            }
            return Some(8);
        }
        return Some(10);
    }
    if cs[4] == 'W' {
        let mut idx = 7;
        while idx < n && cs[idx].is_ascii_digit() {
            idx += 1;
        }
        if idx < 9 {
            return Some(idx);
        }
        return Some(if idx % 2 == 0 { 7 } else { 8 });
    }
    Some(8)
}

fn parse_date(cs: &[char]) -> Option<chrono::NaiveDate> {
    if !matches!(cs.len(), 7 | 8 | 10) {
        return None;
    }
    let year = digits(cs, 0, 4)? as i32;
    let has_sep = cs[4] == '-';
    let mut pos = 4 + has_sep as usize;
    if cs.get(pos) == Some(&'W') {
        pos += 1;
        let week = digits(cs, pos, 2)?;
        pos += 2;
        let mut day = 1;
        if cs.len() > pos {
            if (cs[pos] == '-') != has_sep {
                return None;
            }
            pos += has_sep as usize;
            day = digits(cs, pos, 1)?;
            if pos + 1 != cs.len() {
                return None;
            }
        }
        if !(1..=7).contains(&day) || week == 0 {
            return None;
        }
        let weekday = chrono::Weekday::try_from((day - 1) as u8).ok()?;
        return chrono::NaiveDate::from_isoywd_opt(year, week, weekday);
    }
    let month = digits(cs, pos, 2)?;
    pos += 2;
    if (cs.get(pos) == Some(&'-')) != has_sep {
        return None;
    }
    pos += has_sep as usize;
    let day = digits(cs, pos, 2)?;
    if pos + 2 != cs.len() {
        return None;
    }
    chrono::NaiveDate::from_ymd_opt(year, month, day)
}

/// `HH[:?MM[:?SS[{.,}fff[fff]]]]` -> (h, m, s); the fraction is validated and dropped (seconds precision).
fn parse_hms(cs: &[char]) -> Option<(u32, u32, u32)> {
    let n = cs.len();
    let mut comps = [0u32; 3];
    let mut pos = 0;
    let mut has_sep = false;
    for (comp, slot) in comps.iter_mut().enumerate() {
        if n.saturating_sub(pos) < 2 {
            return None;
        }
        *slot = digits(cs, pos, 2)?;
        pos += 2;
        let next = cs.get(pos).copied();
        if comp == 0 {
            has_sep = next == Some(':');
        }
        if next.is_none() || comp >= 2 {
            break;
        }
        if has_sep && next != Some(':') {
            return None;
        }
        pos += has_sep as usize;
    }
    if pos < n {
        if cs[pos] != '.' && cs[pos] != ',' {
            return None;
        }
        pos += 1;
        let remainder = n - pos;
        if remainder == 0 || !cs[pos..].iter().all(|c| c.is_ascii_digit()) {
            return None;
        }
    }
    Some((comps[0], comps[1], comps[2]))
}

/// Python 3.11 `datetime.fromisoformat(s.replace("Z", "+00:00"))`, naive results taken as UTC.
pub fn parse_iso(s: &str) -> Option<IsoDateTime> {
    let replaced = s.replace('Z', "+00:00");
    let cs: Vec<char> = replaced.chars().collect();
    if cs.len() < 7 {
        return None;
    }
    let sep = find_separator(&cs)?;
    if sep > cs.len() {
        return None;
    }
    let date = parse_date(&cs[..sep])?;
    let tstr: &[char] = if sep + 1 <= cs.len() { &cs[sep + 1..] } else { &[] };
    if tstr.is_empty() {
        return Some(IsoDateTime { date, hour: 0, minute: 0, second: 0, offset_seconds: 0 });
    }
    if tstr.len() < 2 {
        return None;
    }
    let tz_pos = tstr.iter().position(|&c| c == '-').or_else(|| tstr.iter().position(|&c| c == '+'));
    let timestr = match tz_pos {
        Some(p) => &tstr[..p],
        None => tstr,
    };
    let (h, m, sec) = parse_hms(timestr)?;
    let mut offset = 0i32;
    if let Some(p) = tz_pos {
        let tz = &tstr[p + 1..];
        if matches!(tz.len(), 0 | 1 | 3) {
            return None;
        }
        let (th, tm, ts) = parse_hms(tz)?;
        let total = (th * 3600 + tm * 60 + ts) as i32;
        if total >= 86400 {
            return None;
        }
        offset = if tstr[p] == '-' { -total } else { total };
    }
    if h > 23 || m > 59 || sec > 59 {
        return None;
    }
    Some(IsoDateTime { date, hour: h, minute: m, second: sec, offset_seconds: offset })
}

/// Layers above `agent`, the ones a unit (and a rule's target) can have.
pub fn unit_layers() -> &'static [&'static str] {
    &LAYERS[1..]
}
