//! Agent identity, organization identity and the authorization rules Mycelic enforces.
//!
//! * **Agent identity** is a per-agent API key `mk_<agent_id>.<secret>`.  Only its SHA-256 is stored; the key is
//!   shown once, at registration.  The agent id inside the key is a routing hint, never trusted: the secret is
//!   compared in constant time against the stored hash of that agent.  The format and the hash are the Python
//!   service's, so keys issued by it keep working.
//! * **Organization identity** is the enterprise root of the agent's path.  Everything an agent reads or writes is
//!   confined to that organization.
//! * **Administrator** is the holder of `MYCELIC_ADMIN_TOKEN` (bootstrap, agent registration, rules, replay).
//!
//! Authorization, in full:
//!
//! * an agent writes memories and events only as itself (its own path; a body naming another agent is rejected);
//! * an agent reads agent-layer memories of its own team (or any marked `visibility='org'`) and derived memories
//!   of every unit it belongs to (the memory's unit is an ancestor-or-self of the agent's path);
//! * an agent queries at its own team or any of its ancestor units; results are filtered by the rule above, so a
//!   query cannot reveal even the existence of memories outside the caller's view;
//! * lineage of a readable memory is returned with unreadable contributions redacted (shape kept, content withheld);
//! * an administrator can read everything on the deployment.

use crate::hierarchy::{AgentPath, is_ancestor_or_self, unit_at_layer};
use crate::models::{Agent, Memory};
use crate::store::Store;
use base64::Engine;
use rand::RngCore;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::{BTreeSet, HashMap};
use std::time::Instant;

pub const API_KEY_PREFIX: &str = "mk_";

/// `(key, key_hash, key_prefix)` for a new agent key: 256 random bits, URL-safe base64 without padding (Python's
/// `secrets.token_urlsafe(32)`); the prefix is `mk_<agent_id>.` plus the first four characters of the secret.
pub fn generate_api_key(agent_id: &str) -> (String, String, String) {
    let mut secret = [0u8; 32];
    rand::rng().fill_bytes(&mut secret);
    let key = format!("{API_KEY_PREFIX}{agent_id}.{}", base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(secret));
    let prefix: String = key.chars().take(API_KEY_PREFIX.len() + agent_id.chars().count() + 5).collect();
    let hash = hash_key(&key);
    (key, hash, prefix)
}

pub fn hash_key(key: &str) -> String {
    hex::encode(Sha256::digest(key.as_bytes()))
}

pub fn parse_agent_id(key: &str) -> Option<&str> {
    let rest = key.strip_prefix(API_KEY_PREFIX)?;
    if !key.contains('.') {
        return None;
    }
    let agent_id = rest.split('.').next().unwrap_or("");
    (!agent_id.is_empty()).then_some(agent_id)
}

/// Constant-time byte comparison (`hmac.compare_digest`): the time depends on the length, never on the content.
pub fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        // still walk the input so a wrong length costs about what a wrong value costs
        let mut acc = 1u8;
        for x in a {
            acc |= x ^ 0xff;
        }
        std::hint::black_box(acc);
        return false;
    }
    let mut diff = 0u8;
    for (x, y) in a.iter().zip(b) {
        diff |= x ^ y;
    }
    diff == 0
}

/// The read rule written once in code (the SQL twin is `Db::visible_rows`).
pub fn memory_visible(m: &Memory, agent_path: Option<&str>, team_path: Option<&str>) -> bool {
    let (Some(agent_path), Some(team_path)) = (agent_path, team_path) else { return true };
    if m.layer == "agent" {
        return m.visibility == "org"
            || (m.scope.len() > team_path.len()
                && m.scope.starts_with(team_path)
                && m.scope.as_bytes()[team_path.len()] == b'/');
    }
    is_ancestor_or_self(&m.scope, agent_path)
}

#[derive(Clone, Debug, PartialEq)]
pub struct Principal {
    /// "admin" | "agent"
    pub kind: String,
    pub id: String,
    pub org_id: Option<String>,
    pub scopes: BTreeSet<String>,
    pub agent: Option<Agent>,
    pub path: Option<String>,
    pub team_path: Option<String>,
}

impl Principal {
    pub fn admin() -> Principal {
        Principal {
            kind: "admin".into(),
            id: "admin".into(),
            org_id: None,
            scopes: BTreeSet::from(["admin".to_string()]),
            agent: None,
            path: None,
            team_path: None,
        }
    }

    pub fn for_agent(agent: Agent) -> Principal {
        let team = match AgentPath::parse(&agent.path) {
            Ok(ap) => ap.team_path(),
            Err(_) => unit_at_layer(&agent.path, "team").ok().flatten().unwrap_or_else(|| agent.path.clone()),
        };
        Principal {
            kind: "agent".into(),
            id: agent.agent_id.clone(),
            org_id: Some(agent.org_id.clone()),
            scopes: agent.scopes.iter().cloned().collect(),
            path: Some(agent.path.clone()),
            team_path: Some(team),
            agent: Some(agent),
        }
    }

    pub fn is_admin(&self) -> bool {
        self.kind == "admin"
    }

    pub fn has(&self, scope: &str) -> bool {
        self.is_admin() || self.scopes.contains(scope)
    }

    pub fn can_read(&self, m: &Memory) -> bool {
        if self.is_admin() {
            return true;
        }
        if Some(&m.org_id) != self.org_id.as_ref() {
            return false;
        }
        memory_visible(m, self.path.as_deref(), self.team_path.as_deref())
    }

    /// Only a raw observation has an owner; derived memories belong to the organization.
    pub fn owns(&self, m: &Memory) -> bool {
        self.is_admin() || (m.layer == "agent" && m.producer_id == self.id)
    }

    pub fn can_query_scope(&self, scope: &str) -> bool {
        if self.is_admin() {
            return true;
        }
        let Some(path) = &self.path else { return false };
        let team = self.team_path.as_deref().unwrap_or("");
        is_ancestor_or_self(scope, path) || scope.starts_with(&format!("{team}/"))
    }

    pub fn to_json(&self) -> Value {
        json!({"kind": self.kind, "id": self.id, "org_id": self.org_id, "scopes": self.scopes,
               "path": self.path, "team_path": self.team_path,
               "agent": self.agent.as_ref().map(Agent::to_json)})
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AuthError {
    pub status: u16,
    pub reason: &'static str,
}

/// Resolve an `Authorization: Bearer ...` header to a principal.
pub fn authenticate(
    store: &Store,
    admin_token: Option<&str>,
    authorization: Option<&str>,
) -> anyhow::Result<Result<Principal, AuthError>> {
    let missing = AuthError { status: 401, reason: "missing bearer token" };
    let invalid = AuthError { status: 401, reason: "invalid token" };
    let Some(header) = authorization.filter(|h| h.starts_with("Bearer ")) else { return Ok(Err(missing)) };
    let token = crate::textutil::py_strip(&header[7..]);
    if token.is_empty() {
        return Ok(Err(missing));
    }
    if let Some(admin) = admin_token.filter(|t| !t.is_empty())
        && constant_time_eq(token.as_bytes(), admin.as_bytes())
    {
        return Ok(Ok(Principal::admin()));
    }
    let Some(agent_id) = parse_agent_id(token) else { return Ok(Err(invalid)) };
    let Some((agent, key_hash)) = store.read(|db| db.get_agent_credentials(agent_id))? else {
        // keep timing similar to the found case
        let decoy = hash_key(&"x".repeat(token.chars().count()));
        std::hint::black_box(constant_time_eq(hash_key(token).as_bytes(), decoy.as_bytes()));
        return Ok(Err(invalid));
    };
    if !constant_time_eq(hash_key(token).as_bytes(), key_hash.as_bytes()) {
        return Ok(Err(invalid));
    }
    if agent.status != "active" {
        return Ok(Err(AuthError { status: 403, reason: "agent revoked" }));
    }
    Ok(Ok(Principal::for_agent(agent)))
}

struct Bucket {
    tokens: f64,
    updated: f64,
}

/// Token bucket per key (peer address before authentication, principal after it).  `rps <= 0` disables it.
pub struct RateLimiter {
    pub rps: f64,
    pub burst: i64,
    pub max_keys: usize,
    buckets: HashMap<String, Bucket>,
    epoch: Instant,
}

impl RateLimiter {
    pub fn new(rps: f64, burst: i64) -> RateLimiter {
        RateLimiter { rps, burst, max_keys: 20_000, buckets: HashMap::new(), epoch: Instant::now() }
    }

    pub fn now(&self) -> f64 {
        self.epoch.elapsed().as_secs_f64()
    }

    /// Take `cost` tokens (a batch request costs as many as the requests it carries).
    pub fn allow(&mut self, key: &str, cost: i64) -> bool {
        let now = self.now();
        self.allow_at(key, now, cost)
    }

    pub fn allow_at(&mut self, key: &str, now: f64, cost: i64) -> bool {
        // `not rps > 0` rather than `rps <= 0`: a NaN rate (possible from the environment) disables limiting too
        if !(self.rps > 0.0) || cost <= 0 {
            return true;
        }
        if !self.buckets.contains_key(key) && self.buckets.len() >= self.max_keys {
            self.prune(now);
        }
        let burst = self.burst as f64;
        let rps = self.rps;
        let b = self.buckets.entry(key.to_string()).or_insert(Bucket { tokens: burst, updated: now });
        b.tokens = f64::min(burst, b.tokens + (now - b.updated) * rps);
        b.updated = now;
        if b.tokens >= cost as f64 {
            b.tokens -= cost as f64;
            return true;
        }
        false
    }

    /// Drop idle buckets; if the map is still full, evict the least recently used tenth (never everyone).
    fn prune(&mut self, now: f64) {
        self.buckets.retain(|_, b| now - b.updated <= 600.0);
        if self.buckets.len() >= self.max_keys {
            let mut by_age: Vec<(String, f64)> = self.buckets.iter().map(|(k, b)| (k.clone(), b.updated)).collect();
            by_age.sort_by(|a, b| a.1.total_cmp(&b.1));
            for (k, _) in by_age.into_iter().take((self.max_keys / 10).max(1)) {
                self.buckets.remove(&k);
            }
        }
    }

    pub fn len(&self) -> usize {
        self.buckets.len()
    }

    pub fn is_empty(&self) -> bool {
        self.buckets.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn key_format() {
        let (key, hash, prefix) = generate_api_key("agent-7");
        assert!(key.starts_with("mk_agent-7."));
        assert_eq!(parse_agent_id(&key), Some("agent-7"));
        assert_eq!(hash.len(), 64);
        assert_eq!(prefix.len(), "mk_agent-7.".len() + 4);
        assert_eq!(parse_agent_id("nope"), None);
    }

    #[test]
    fn rate_limiter() {
        let mut r = RateLimiter::new(10.0, 3);
        let got: Vec<bool> = (0..4).map(|_| r.allow_at("k", 0.0, 1)).collect();
        assert_eq!(got, vec![true, true, true, false]);
        assert!(r.allow_at("k", 0.2, 1));
        assert!(RateLimiter::new(0.0, 100).allow("k", 1));
    }

    #[test]
    fn eviction_keeps_most_buckets() {
        let mut r = RateLimiter::new(1.0, 1);
        r.max_keys = 100;
        for i in 0..100 {
            r.allow_at(&format!("k{i}"), i as f64, 1);
        }
        r.allow_at("new", 101.0, 1);
        assert_eq!(r.len(), 91, "the least recently used tenth goes, the rest stay");
    }
}
