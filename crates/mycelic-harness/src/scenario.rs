//! The Northwind scenario: a small organization whose agents each see a piece of a supply problem.
//!
//! Used by the production smoke test and the demo.  Nothing here is a mock: agents are real processes
//! (`mycelic-agent`) with their own local memory files, talking to a real deployment over HTTP.
//!
//! Organization (enterprise `northwind` -> region `emea` -> subsidiary `nw-gmbh`):
//!
//! ```text
//! ops / logistics      port, carrier and customs observations   -> slot transport_disruption
//! ops / procurement    supplier stock levels                     -> slot supplier_buffer_low
//! commercial / sales   customer commitments                      -> slot demand_commitment
//! ```
//!
//! No single agent, team or department can see the conclusion.  The rule `component_supply_risk`
//! (deploy/mycelic/rules.json) needs all three slots for the same entity from at least three agents in at least two
//! teams; Mycelic derives it at the enterprise layer with the full lineage.  Every agent also keeps notes that are
//! never shared (`share: false`), so the demo can show what stayed local.

use anyhow::{Result, bail};
use mycelic_sdk::{MycelicClient, Registration};
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::fs::OpenOptions;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

pub const ENTITY: &str = "sd-9";
pub const ENTERPRISE: &str = "northwind";
pub const REGION: &str = "emea";
pub const SUBSIDIARY: &str = "nw-gmbh";
/// team -> department
pub const TEAMS: [(&str, &str); 3] = [("logistics", "ops"), ("procurement", "ops"), ("field-sales", "commercial")];
pub const QUERY: &str = "Is there a delivery risk for the RX-4 arm and its SD-9 servo drive this quarter?";

fn obs(text: &str, topic: &str, slot: &str, confidence: f64) -> Value {
    json!({"text": text, "topic": topic, "slot": slot, "entity": ENTITY, "confidence": confidence})
}

/// Each team's shared observations (cycled over the team's agents).
pub fn shared(team: &str) -> Vec<Value> {
    match team {
        "logistics" => vec![
            obs(
                "Port of Rotterdam terminal 3 strike announced for weeks 41-43; SD-9 servo drive shipments route through it.",
                "supply:sd-9/transport",
                "transport_disruption",
                0.9,
            ),
            obs(
                "Carrier ETA for the SD-9 servo drive container slipped by 12 days.",
                "supply:sd-9/transport",
                "transport_disruption",
                0.7,
            ),
            obs(
                "Customs broker warns of a two-week backlog for SD-9 containers arriving via Rotterdam.",
                "supply:sd-9/transport",
                "transport_disruption",
                0.6,
            ),
        ],
        "procurement" => vec![
            obs(
                "Kessler Antriebe confirms roughly two weeks of SD-9 inventory left and no second source qualified.",
                "supply:sd-9/supplier",
                "supplier_buffer_low",
                0.85,
            ),
            obs(
                "Kessler Antriebe quoted 9 weeks lead time for additional SD-9 units.",
                "supply:sd-9/supplier",
                "supplier_buffer_low",
                0.75,
            ),
        ],
        _ => vec![
            obs(
                "Helios Automation committed to 40 RX-4 arms for November; every RX-4 uses an SD-9 drive.",
                "supply:sd-9/demand",
                "demand_commitment",
                0.8,
            ),
            obs(
                "Borealis Packaging asked for delivery dates on 12 RX-4 arms in Q4.",
                "supply:sd-9/demand",
                "demand_commitment",
                0.65,
            ),
        ],
    }
}

/// Each team's private notes.
pub fn private(team: &str) -> Vec<&'static str> {
    match team {
        "logistics" => vec![
            "Reminder: renew the forklift certification before October.",
            "Draft note: the Antwerp fallback route was slower last year, verify before proposing it.",
        ],
        "procurement" => vec![
            "Personal note: Kessler's account manager prefers calls on Tuesdays.",
            "Unverified rumour that Kessler is being acquired; do not share until confirmed.",
        ],
        _ => {
            vec!["Helios wants a demo of the new gripper next month.", "Internal: discount ceiling for Borealis is 8%."]
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct AgentSpec {
    pub agent_id: String,
    pub team: String,
    pub department: String,
    pub api_key: String,
    pub local_db: PathBuf,
    pub observations: PathBuf,
    pub state_file: PathBuf,
    pub path: String,
}

impl AgentSpec {
    pub fn team_path(&self) -> String {
        format!("{ENTERPRISE}/{REGION}/{SUBSIDIARY}/{}/{}", self.department, self.team)
    }
}

pub struct Scenario {
    pub workdir: PathBuf,
    pub agents_per_team: usize,
    pub agents: Vec<AgentSpec>,
}

impl Scenario {
    pub fn build(workdir: &Path, agents_per_team: usize) -> Result<Scenario> {
        std::fs::create_dir_all(workdir)?;
        let mut agents = Vec::new();
        for (team, department) in TEAMS {
            for i in 1..=agents_per_team {
                agents.push(AgentSpec {
                    agent_id: format!("{team}-{i}"),
                    team: team.into(),
                    department: department.into(),
                    ..AgentSpec::default()
                });
            }
        }
        Ok(Scenario { workdir: workdir.to_path_buf(), agents_per_team, agents })
    }

    pub fn register_all(&mut self, admin: &MycelicClient) -> Result<()> {
        for a in &mut self.agents {
            let res = admin.register_agent(&Registration {
                agent_id: a.agent_id.clone(),
                enterprise: ENTERPRISE.into(),
                region: Some(REGION.into()),
                subsidiary: Some(SUBSIDIARY.into()),
                department: Some(a.department.clone()),
                team: Some(a.team.clone()),
                display_name: Some(title(&a.agent_id)),
                scopes: None,
            })?;
            a.api_key = res["api_key"].as_str().unwrap_or_default().to_string();
            a.path = res["agent"]["path"].as_str().unwrap_or_default().to_string();
        }
        Ok(())
    }

    /// Each agent shares one observation (cycling through the team's list) so pieces are spread across agents, and
    /// the first two agents of a team overlap on the strongest one to show partial overlap; each keeps one note
    /// private.
    pub fn write_observations(&mut self) -> Result<()> {
        for a in &mut self.agents {
            let idx: usize = a.agent_id.rsplit('-').next().and_then(|n| n.parse::<usize>().ok()).unwrap_or(1) - 1;
            let shared = shared(&a.team);
            let private = private(&a.team);
            let mut mine = vec![shared[idx % shared.len()].clone()];
            if idx == 1 && shared.len() > 1 {
                mine.push(shared[0].clone());
            }
            let mut lines = Vec::new();
            for (n, o) in mine.iter().enumerate() {
                let mut line = o.clone();
                line["share"] = json!(true);
                line["local_id"] = json!(format!("{}-obs-{n}", a.agent_id));
                lines.push(line.to_string());
            }
            lines.push(json!({"text": private[idx % private.len()], "share": false, "local_id": format!("{}-private", a.agent_id)}).to_string());
            a.observations = self.workdir.join(format!("{}.observations.jsonl", a.agent_id));
            std::fs::write(&a.observations, lines.join("\n") + "\n")?;
            a.local_db = self.workdir.join(format!("{}.local.db", a.agent_id));
            a.state_file = self.workdir.join(format!("{}.state.json", a.agent_id));
        }
        Ok(())
    }

    pub fn by_team(&self, team: &str) -> Vec<&AgentSpec> {
        self.agents.iter().filter(|a| a.team == team).collect()
    }

    pub fn agent(&self, agent_id: &str) -> Option<&AgentSpec> {
        self.agents.iter().find(|a| a.agent_id == agent_id)
    }

    /// Run every agent as its own process, concurrently; each agent's state summary.
    pub fn run_agents(&self, base_url: &str, agents: Option<&[&AgentSpec]>) -> Result<BTreeMap<String, Value>> {
        let all: Vec<&AgentSpec> = self.agents.iter().collect();
        run_agents(&self.workdir, base_url, agents.unwrap_or(&all), Duration::from_secs(120), &|| false)
    }
}

/// "logistics-1" -> "Logistics 1" (Python's `str.title` on the id with dashes as spaces).
pub fn title(agent_id: &str) -> String {
    agent_id
        .split('-')
        .map(|w| {
            let mut c = w.chars();
            match c.next() {
                Some(f) => f.to_uppercase().collect::<String>() + &c.as_str().to_lowercase(),
                None => String::new(),
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// Kills every child still running when dropped, so a failure or an interrupt leaves no agent behind.
struct Children(Vec<(String, Child)>);

impl Drop for Children {
    fn drop(&mut self) {
        for (_, c) in &mut self.0 {
            if c.try_wait().ok().flatten().is_none() {
                let _ = c.kill();
                let _ = c.wait();
            }
        }
    }
}

/// Run agents as processes (`mycelic-agent`), concurrently; their state summaries by agent id.
///
/// Each key reaches its agent through the environment (`MYCELIC_API_KEY`), not the command line, so `ps` never
/// shows it.  `interrupted` is polled while waiting; when it turns true the agents are killed and an error returned.
pub fn run_agents(
    workdir: &Path,
    base_url: &str,
    agents: &[&AgentSpec],
    timeout: Duration,
    interrupted: &dyn Fn() -> bool,
) -> Result<BTreeMap<String, Value>> {
    let bin = crate::binary("mycelic-agent");
    let mut procs = Children(Vec::new());
    for a in agents {
        let log = OpenOptions::new().create(true).append(true).open(workdir.join(format!("{}.log", a.agent_id)))?;
        let child = Command::new(&bin)
            .args(["--url", base_url, "--local-db"])
            .arg(&a.local_db)
            .arg("--observations")
            .arg(&a.observations)
            .arg("--state-file")
            .arg(&a.state_file)
            .env("MYCELIC_API_KEY", &a.api_key)
            .current_dir(crate::repo_root())
            .stdout(Stdio::from(log.try_clone()?))
            .stderr(Stdio::from(log))
            .spawn()
            .map_err(|e| anyhow::anyhow!("starting {}: {e}", bin.display()))?;
        procs.0.push((a.agent_id.clone(), child));
    }
    let deadline = Instant::now() + timeout;
    let mut states = BTreeMap::new();
    for (i, a) in agents.iter().enumerate() {
        let status = loop {
            if let Some(s) = procs.0[i].1.try_wait()? {
                break s;
            }
            if interrupted() {
                bail!("interrupted");
            }
            if Instant::now() > deadline {
                bail!("agent {} did not finish within {}s", a.agent_id, timeout.as_secs());
            }
            std::thread::sleep(Duration::from_millis(20));
        };
        if !status.success() {
            let log = std::fs::read_to_string(workdir.join(format!("{}.log", a.agent_id))).unwrap_or_default();
            let tail: Vec<&str> = log.lines().collect();
            eprintln!("--- agent {} log tail ---\n{}", a.agent_id, tail[tail.len().saturating_sub(20)..].join("\n"));
            bail!("agent {} exited with {status}: {}", a.agent_id, tail.last().copied().unwrap_or("(no output)"));
        }
        states.insert(a.agent_id.clone(), serde_json::from_str(&std::fs::read_to_string(&a.state_file)?)?);
    }
    Ok(states)
}
