//! Organizational hierarchy: layers, unit paths and the subtree/ancestor relations everything else uses.
//!
//! Mycelic propagates knowledge through six layers:
//!
//! ```text
//! agent -> team -> department -> subsidiary -> region -> enterprise
//! ```
//!
//! A *unit path* names one organizational unit from the enterprise root downwards, joined with `/`:
//!
//! ```text
//! northwind                                                   enterprise   (layer index 5)
//! northwind/emea                                              region       (4)
//! northwind/emea/northwind-gmbh                               subsidiary   (3)
//! northwind/emea/northwind-gmbh/operations                    department   (2)
//! northwind/emea/northwind-gmbh/operations/logistics          team         (1)
//! northwind/emea/northwind-gmbh/operations/logistics/agent-7  agent        (0)
//! ```
//!
//! The enterprise segment doubles as the organization id: two agents share an organization exactly when their
//! paths share a root.  A memory's `scope` is the path of the unit it belongs to, so the layer of a memory is a
//! function of its scope and the subtree of a unit is a prefix match.  Both facts are relied upon by the store
//! (SQL prefix tests on `scope`) and by authorization.

use crate::pyjson::{repr_str, repr_tuple};
use serde_json::{Map, Value, json};
use std::fmt;

pub const LAYERS: [&str; 6] = ["agent", "team", "department", "subsidiary", "region", "enterprise"];
pub const ENTERPRISE_INDEX: usize = 5;
/// Segments in a full agent path.
pub const DEPTH: usize = LAYERS.len();
pub const SEGMENT_PATTERN: &str = "^[a-z0-9][a-z0-9_-]{0,63}$";

/// A path or layer name that does not fit the hierarchy.  The message is the Python one, verbatim.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HierarchyError(pub String);

impl fmt::Display for HierarchyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for HierarchyError {}

pub type Result<T> = std::result::Result<T, HierarchyError>;

fn segment_ok(s: &str) -> bool {
    let b = s.as_bytes();
    if b.is_empty() || b.len() > 64 {
        return false;
    }
    let first_ok = b[0].is_ascii_lowercase() || b[0].is_ascii_digit();
    first_ok && b[1..].iter().all(|&c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == b'_' || c == b'-')
}

pub fn validate_segment<'a>(segment: &'a str, what: &str) -> Result<&'a str> {
    if !segment_ok(segment) {
        return Err(HierarchyError(format!(
            "{what} must match {} (lowercase, no '/'): {}",
            repr_str(SEGMENT_PATTERN),
            repr_str(segment)
        )));
    }
    Ok(segment)
}

pub fn split_path(path: &str) -> Result<Vec<&str>> {
    if path.is_empty() {
        return Err(HierarchyError("path must be a non-empty string".into()));
    }
    let parts: Vec<&str> = path.split('/').collect();
    if parts.len() > DEPTH {
        return Err(HierarchyError(format!(
            "path has {} segments; at most {DEPTH} are allowed: {}",
            parts.len(),
            repr_str(path)
        )));
    }
    for p in &parts {
        validate_segment(p, "path segment")?;
    }
    Ok(parts)
}

/// The layer a unit path denotes: 1 segment is the enterprise, 6 segments is an agent.
pub fn layer_of_path(path: &str) -> Result<&'static str> {
    let n = split_path(path)?.len();
    Ok(LAYERS[DEPTH - n])
}

pub fn layer_index(layer: &str) -> Result<usize> {
    LAYERS.iter().position(|l| *l == layer).ok_or_else(|| {
        HierarchyError(format!("unknown layer {}; expected one of {}", repr_str(layer), repr_tuple(&LAYERS)))
    })
}

/// Index of a layer known to be valid (stored memories); unknown layers sort first.
pub fn layer_pos(layer: &str) -> usize {
    LAYERS.iter().position(|l| *l == layer).unwrap_or(0)
}

pub fn parent_path(path: &str) -> Result<Option<String>> {
    let parts = split_path(path)?;
    Ok(if parts.len() > 1 { Some(parts[..parts.len() - 1].join("/")) } else { None })
}

/// Unit paths from the enterprise root down to `path` (inclusive when `include_self`).
pub fn ancestors(path: &str, include_self: bool) -> Result<Vec<String>> {
    let parts = split_path(path)?;
    let mut out: Vec<String> = (1..=parts.len()).map(|i| parts[..i].join("/")).collect();
    if !include_self {
        out.pop();
    }
    Ok(out)
}

pub fn org_of(path: &str) -> Result<String> {
    Ok(split_path(path)?[0].to_string())
}

/// True when `unit` is `path` itself or one of its ancestors (`path` lies in `unit`'s subtree).
pub fn is_ancestor_or_self(unit: &str, path: &str) -> bool {
    path == unit || (path.len() > unit.len() && path.starts_with(unit) && path.as_bytes()[unit.len()] == b'/')
}

/// `s` lies strictly below `unit` (`s.startswith(unit + "/")`).
pub fn is_below(s: &str, unit: &str) -> bool {
    s.len() > unit.len() && s.starts_with(unit) && s.as_bytes()[unit.len()] == b'/'
}

/// The ancestor of `path` at `layer` (None when `path` is above that layer).
pub fn unit_at_layer(path: &str, layer: &str) -> Result<Option<String>> {
    let parts = split_path(path)?;
    let want = DEPTH - layer_index(layer)?;
    if parts.len() < want {
        return Ok(None);
    }
    Ok(Some(parts[..want].join("/")))
}

/// The direct child of `unit` that contains `descendant` (used to count independent children).
pub fn child_unit_of(descendant: &str, unit: &str) -> Result<String> {
    if !is_ancestor_or_self(unit, descendant) || descendant == unit {
        return Err(HierarchyError(format!("{} is not below {}", repr_str(descendant), repr_str(unit))));
    }
    let unit_parts = split_path(unit)?;
    let parts = split_path(descendant)?;
    Ok(parts[..unit_parts.len() + 1].join("/"))
}

/// The five organizational units an agent belongs to plus its own id, as one validated path.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AgentPath {
    pub enterprise: String,
    pub region: String,
    pub subsidiary: String,
    pub department: String,
    pub team: String,
    pub agent_id: String,
}

impl AgentPath {
    /// Missing intermediate units default to the enterprise name so small organizations need not invent them.
    pub fn from_parts(
        enterprise: &str,
        region: Option<&str>,
        subsidiary: Option<&str>,
        department: Option<&str>,
        team: Option<&str>,
        agent_id: &str,
    ) -> Result<AgentPath> {
        let e = validate_segment(enterprise, "enterprise")?;
        let pick = |v: Option<&str>| match v {
            Some(s) if !s.is_empty() => s.to_string(),
            _ => e.to_string(),
        };
        let region = pick(region);
        validate_segment(&region, "region")?;
        let subsidiary = pick(subsidiary);
        validate_segment(&subsidiary, "subsidiary")?;
        let department = pick(department);
        validate_segment(&department, "department")?;
        let team = pick(team);
        validate_segment(&team, "team")?;
        validate_segment(agent_id, "agent_id")?;
        Ok(AgentPath {
            enterprise: e.to_string(),
            region,
            subsidiary,
            department,
            team,
            agent_id: agent_id.to_string(),
        })
    }

    pub fn parse(path: &str) -> Result<AgentPath> {
        let parts = split_path(path)?;
        if parts.len() != DEPTH {
            return Err(HierarchyError(format!(
                "an agent path needs {DEPTH} segments, got {}: {}",
                parts.len(),
                repr_str(path)
            )));
        }
        Ok(AgentPath {
            enterprise: parts[0].into(),
            region: parts[1].into(),
            subsidiary: parts[2].into(),
            department: parts[3].into(),
            team: parts[4].into(),
            agent_id: parts[5].into(),
        })
    }

    pub fn path(&self) -> String {
        format!("{}/{}", self.team_path(), self.agent_id)
    }

    pub fn team_path(&self) -> String {
        [&self.enterprise, &self.region, &self.subsidiary, &self.department, &self.team].map(|s| s.as_str()).join("/")
    }

    pub fn org_id(&self) -> &str {
        &self.enterprise
    }

    pub fn to_json(&self) -> Value {
        let mut m = Map::new();
        for (k, v) in [
            ("enterprise", &self.enterprise),
            ("region", &self.region),
            ("subsidiary", &self.subsidiary),
            ("department", &self.department),
            ("team", &self.team),
            ("agent_id", &self.agent_id),
        ] {
            m.insert(k.into(), json!(v));
        }
        m.insert("path".into(), json!(self.path()));
        Value::Object(m)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layers_follow_path_depth() {
        let p = AgentPath::from_parts("acme", Some("eu"), Some("acme-de"), Some("ops"), Some("log"), "a1").unwrap();
        assert_eq!(layer_of_path(&p.path()).unwrap(), "agent");
        assert_eq!(layer_of_path(&p.team_path()).unwrap(), "team");
        assert_eq!(layer_of_path("acme").unwrap(), "enterprise");
        assert_eq!(unit_at_layer(&p.path(), "department").unwrap().unwrap(), "acme/eu/acme-de/ops");
        assert_eq!(
            ancestors(&p.team_path(), true).unwrap(),
            vec!["acme", "acme/eu", "acme/eu/acme-de", "acme/eu/acme-de/ops", "acme/eu/acme-de/ops/log"]
        );
        assert_eq!(child_unit_of(&p.path(), "acme/eu").unwrap(), "acme/eu/acme-de");
        assert!(is_ancestor_or_self("acme/eu", &p.path()));
        assert!(!is_ancestor_or_self("acme/eu/other", &p.path()));
    }

    #[test]
    fn small_org_defaults_and_validation() {
        let p = AgentPath::from_parts("acme", None, None, None, None, "a1").unwrap();
        assert_eq!(p.path(), "acme/acme/acme/acme/acme/a1");
        assert!(AgentPath::from_parts("Acme Inc", None, None, None, None, "a1").is_err());
        assert!(AgentPath::parse("acme/eu").is_err());
        assert!(layer_of_path("a/b/c/d/e/f/g").is_err());
    }
}
