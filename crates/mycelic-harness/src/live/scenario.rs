//! The live demo's own scenario: Orrery Robotics, a fictional company with 40 agents in three regions.
//!
//! Fictional company, synthetic data.  Every name below (the company, its subsidiaries, the supplier "Mirelund
//! Drives", the customers "Tolvenka Packaging" and "Sorano Assembly") is invented; the port cities are real places
//! used only as scenery.
//!
//! The core of the story is the SD-9 servo-drive scenario of [`crate::strategic`] (same topics, slots,
//! confidences, the same entity and the same rules in `deploy/mycelic/rules.json`), retold with invented names and
//! staged in three waves so a presenter can watch each layer conclude:
//!
//! ```text
//! orrery (enterprise)
//! ├─ emea / orrery-gmbh   ops: logistics, procurement, quality, maintenance   commercial: field-sales, pricing,
//! │                       customer-success   people: hiring
//! ├─ apac / orrery-kk     (same teams as EMEA)
//! └─ hq   / orrery-hq     strategy: sourcing, analytics   finance: fp-and-a   it: security
//! ```
//!
//! * 14 **core** agents (two per core team in EMEA and APAC, one each in HQ sourcing and analytics) hold the
//!   pieces of the SD-9 story.  Stage `observe`: logistics and field-sales share; stage `conclude`: procurement
//!   shares, which completes `regional_supply_risk` in each region; stage `synthesise`: HQ shares demand growth
//!   and supplier concentration, which completes `strategic_second_source` at the enterprise.
//! * 26 **background** agents in ten more teams share ordinary notes (weld quality, CNC maintenance, pricing,
//!   support tickets, hiring, FX, patching).  Their topics consolidate up the org chart but use no slot any rule
//!   needs, so they can never complete a rule: the signal has to come out of real, unrelated traffic.
//!
//! Every agent also keeps at least one private note (`share: false`) that stays in its own local SQLite file.
//! Agents are real processes (`mycelic-agent`) with their own keys, run by [`crate::scenario::run_agents`].

use crate::scenario::AgentSpec;
pub use crate::scenario::ENTITY;
pub use crate::strategic::STRATEGIC_QUERY;
use anyhow::{Context, Result};
use mycelic_sdk::{MycelicClient, Registration};
use serde_json::{Value, json};
use std::collections::HashSet;
use std::path::{Path, PathBuf};

pub const COMPANY: &str = "Orrery Robotics";
/// What the story is about (the console's labels).
pub const SUBJECT: &str = "SD-9 servo drive";
pub const ENTERPRISE: &str = "orrery";
pub const SUPPLIER: &str = "Mirelund Drives";
/// The act in which an agent shares (acts 2, 3, 4).
pub const STAGES: [&str; 3] = ["observe", "conclude", "synthesise"];
/// region -> (label, subsidiary id, subsidiary label)
pub const REGIONS: [(&str, (&str, &str, &str)); 3] = [
    ("emea", ("EMEA", "orrery-gmbh", "Orrery GmbH")),
    ("apac", ("APAC", "orrery-kk", "Orrery K.K.")),
    ("hq", ("Head office", "orrery-hq", "Orrery HQ")),
];

/// team -> (department, stage) for the story's teams in EMEA and APAC.
const CORE_TEAMS: [(&str, (&str, &str)); 3] = [
    ("logistics", ("ops", "observe")),
    ("procurement", ("ops", "conclude")),
    ("field-sales", ("commercial", "observe")),
];
/// team -> department for head office's story teams.
const HQ_CORE_TEAMS: [(&str, &str); 2] = [("sourcing", "strategy"), ("analytics", "strategy")];
const BACKGROUND_TEAMS: [(&str, &str); 5] = [
    ("quality", "ops"),
    ("maintenance", "ops"),
    ("pricing", "commercial"),
    ("customer-success", "commercial"),
    ("hiring", "people"),
];
const HQ_BACKGROUND_TEAMS: [(&str, &str); 3] = [("fp-and-a", "finance"), ("security", "it"), ("hiring", "people")];

fn region(region: &str) -> (&'static str, &'static str, &'static str) {
    REGIONS.iter().find(|(r, _)| *r == region).map(|(_, info)| *info).unwrap_or(("", "", ""))
}

fn port(region: &str) -> &'static str {
    if region == "emea" { "Rotterdam" } else { "Busan" }
}

fn customer(region: &str) -> &'static str {
    if region == "emea" { "Tolvenka Packaging" } else { "Sorano Assembly" }
}

fn plant(region: &str) -> &'static str {
    match region {
        "emea" => "Eindhoven",
        "apac" => "Nagoya",
        _ => "head office",
    }
}

pub fn department_label(department: &str) -> &'static str {
    match department {
        "ops" => "Operations",
        "commercial" => "Commercial",
        "people" => "People",
        "strategy" => "Strategy",
        "finance" => "Finance",
        "it" => "IT",
        _ => "",
    }
}

pub fn team_label(team: &str) -> &'static str {
    match team {
        "logistics" => "Logistics",
        "procurement" => "Procurement",
        "field-sales" => "Field sales",
        "quality" => "Quality",
        "maintenance" => "Maintenance",
        "pricing" => "Pricing",
        "customer-success" => "Customer success",
        "hiring" => "Hiring",
        "sourcing" => "Sourcing",
        "analytics" => "Analytics",
        "fp-and-a" => "FP&A",
        "security" => "Security",
        _ => "",
    }
}

/// The number at the end of an agent id: "emea-logistics-1" -> "1".
fn suffix(agent_id: &str) -> &str {
    agent_id.rsplit('-').next().unwrap_or(agent_id)
}

fn obs(text: String, topic: &str, slot: &str, entity: &str, confidence: f64) -> Value {
    json!({"text": text, "topic": topic, "slot": slot, "entity": entity, "confidence": confidence})
}

/// The SD-9 pieces one region's core teams hold; slots, topics and confidences as in the strategic scenario.
pub fn core_observations(region: &str, team: &str) -> Vec<Value> {
    let (port, customer) = (port(region), customer(region));
    match team {
        "logistics" => vec![
            obs(
                format!(
                    "Port of {port} strike announced for weeks 41-43; SD-9 servo drive shipments route through it."
                ),
                "supply:sd-9/transport",
                "transport_disruption",
                ENTITY,
                0.9,
            ),
            obs(
                format!("Carrier ETA for the SD-9 servo drive container via {port} slipped by 12 days."),
                "supply:sd-9/transport",
                "transport_disruption",
                ENTITY,
                0.7,
            ),
        ],
        "procurement" => vec![
            obs(
                format!(
                    "{SUPPLIER} confirms roughly two weeks of SD-9 inventory for this region and no second source qualified."
                ),
                "supply:sd-9/supplier",
                "supplier_buffer_low",
                ENTITY,
                0.85,
            ),
            obs(
                format!("{SUPPLIER} quoted 9 weeks lead time for additional SD-9 units."),
                "supply:sd-9/supplier",
                "supplier_buffer_low",
                ENTITY,
                0.75,
            ),
        ],
        "field-sales" => vec![
            obs(
                format!("{customer} committed to 40 RX-4 arms for November; every RX-4 uses an SD-9 drive."),
                "supply:sd-9/demand",
                "demand_commitment",
                ENTITY,
                0.8,
            ),
            obs(
                format!("{customer} asked for delivery dates on 12 more RX-4 arms in Q4."),
                "supply:sd-9/demand",
                "demand_commitment",
                ENTITY,
                0.65,
            ),
        ],
        _ => Vec::new(),
    }
}

pub fn hq_observations(team: &str) -> Vec<Value> {
    match team {
        "sourcing" => vec![obs(
            format!(
                "{SUPPLIER} is the only qualified SD-9 source for every subsidiary; no second source is qualified."
            ),
            "strategy:supply-base",
            "supplier_concentration",
            ENTITY,
            0.95,
        )],
        "analytics" => vec![obs(
            "RX-4 order intake is up 40% year over year across all regions.".into(),
            "strategy:demand",
            "demand_growth",
            ENTITY,
            0.8,
        )],
        _ => Vec::new(),
    }
}

/// The note APAC procurement posts after the retraction (act 6).
pub fn recount() -> Value {
    obs(
        "Physical count confirms roughly two weeks of SD-9 inventory in the APAC warehouse.".into(),
        "supply:sd-9/supplier",
        "supplier_buffer_low",
        ENTITY,
        0.9,
    )
}

/// Ordinary notes on other topics.  Their slots feed no rule, so they consolidate but never conclude.
pub fn background_observations(region: &str, team: &str) -> Vec<Value> {
    let plant = plant(region);
    let pair = |a: String, b: String, topic: &str, slot: &str, entity: &str, ca: f64, cb: f64| {
        vec![obs(a, topic, slot, entity, ca), obs(b, topic, slot, entity, cb)]
    };
    match team {
        "quality" => pair(
            format!("Weld seam rejects on RX-4 base frames in {plant} at 1.8% this week, inside the 2% limit."),
            format!("Second-shift inspection in {plant} found porosity on 3 of 160 RX-4 frames; rework scheduled."),
            "quality:rx-4/welds",
            "defect_rate",
            "rx-4",
            0.8,
            0.7,
        ),
        "maintenance" => pair(
            format!("CNC line 2 in {plant}: spindle bearing replacement planned for Saturday, 6 hours of downtime."),
            format!("Vibration on CNC line 2 in {plant} is back within tolerance after the coolant pump swap."),
            "maintenance:cnc-line-2",
            "planned_downtime",
            "cnc-line-2",
            0.85,
            0.7,
        ),
        "pricing" => pair(
            "Competitor list prices for comparable six-axis arms rose about 4% this quarter.".into(),
            "Two distributors asked for volume tiers above 50 RX-4 units.".into(),
            "pricing:rx-4/list",
            "price_signal",
            "rx-4",
            0.7,
            0.65,
        ),
        "customer-success" => pair(
            "RX-4 gripper calibration tickets are down 20% since the firmware update.".into(),
            "Most open RX-4 tickets are first-week setup questions, not faults.".into(),
            "service:rx-4/tickets",
            "ticket_trend",
            "rx-4",
            0.8,
            0.7,
        ),
        "hiring" => pair(
            "Four field-service engineer roles are open; median time to hire is 51 days.".into(),
            "Referral candidates accept field-engineer offers faster than agency candidates.".into(),
            "hiring:field-engineers",
            "open_roles",
            "field-engineers",
            0.8,
            0.6,
        ),
        "fp-and-a" => pair(
            "EUR/JPY swings moved the Q3 APAC margin by about half a point.".into(),
            "Hedges cover 60% of next quarter's yen receivables.".into(),
            "finance:fx-exposure",
            "fx_exposure",
            "fx",
            0.75,
            0.8,
        ),
        "security" => pair(
            "Plant-floor HMI patch backlog is down to 11 devices.".into(),
            "Two plant gateways still run an end-of-life TLS library; replacements are ordered.".into(),
            "it:patching",
            "patch_backlog",
            "hmi",
            0.8,
            0.75,
        ),
        _ => Vec::new(),
    }
}

/// Each team's private notes (never shared).
pub fn private(team: &str) -> Vec<String> {
    let notes: &[&str] = match team {
        "logistics" => &[
            "Reminder: renew the forklift certification before October.",
            "Draft: the Antwerp fallback route was slower last year, verify before proposing it.",
        ],
        "procurement" => {
            return vec![
                format!("Personal note: the {SUPPLIER} account manager prefers calls on Tuesdays."),
                format!("Unverified rumour that {SUPPLIER} is up for sale; do not share until confirmed."),
            ];
        }
        "field-sales" => &[
            "Internal: discount ceiling for this customer is 8%.",
            "The buyer hinted at a competing bid; keep this between us for now.",
        ],
        "sourcing" => &["Draft: shortlist of alternative drive suppliers, not yet validated."],
        "analytics" => &["Working note: the Q3 forecast model still double-counts returns."],
        "quality" => {
            &["Suspect the new weld wire batch; checking before I raise it.", "Audit prep checklist, personal copy."]
        }
        "maintenance" => &[
            "Spare bearings are in locker 14, the key is with the night shift.",
            "Idea: move the lubrication round to Mondays; not discussed yet.",
        ],
        "pricing" => &[
            "Scratch model: a 3% list increase would cost us two distributors.",
            "Do not quote the old tier table again.",
        ],
        "customer-success" => &[
            "Customer contact prefers WhatsApp; personal number on file.",
            "Escalation draft for the Q4 renewals, not sent.",
        ],
        "hiring" => &["Candidate salary expectations, confidential.", "Interview feedback notes, not for circulation."],
        "fp-and-a" => &["Board pack draft numbers, unreviewed."],
        "security" => &["Pen-test finding under embargo until the patch ships."],
        _ => &[],
    };
    notes.iter().map(|s| s.to_string()).collect()
}

/// One agent: where it sits in the org chart, what it will share and what it keeps to itself.
///
/// `api_key`/`local_db`/`observations`/`state_file` have the same names as [`AgentSpec`] so the agent runner
/// treats both alike ([`LiveAgent::spec`]).  The key lives only in memory and is never written to a trace.
#[derive(Clone, Debug, Default)]
pub struct LiveAgent {
    pub agent_id: String,
    pub region: String,
    pub department: String,
    pub team: String,
    /// "core" | "background"
    pub role: &'static str,
    /// [`STAGES`]: the act in which it shares
    pub stage: &'static str,
    pub shared: Vec<Value>,
    pub private: Vec<String>,
    /// seconds before its first note, so a busy network trickles in
    pub delay: f64,
    pub api_key: String,
    pub local_db: PathBuf,
    pub observations: PathBuf,
    pub state_file: PathBuf,
}

impl LiveAgent {
    pub fn subsidiary(&self) -> &'static str {
        region(&self.region).1
    }

    pub fn team_path(&self) -> String {
        format!("{ENTERPRISE}/{}/{}/{}/{}", self.region, self.subsidiary(), self.department, self.team)
    }

    pub fn path(&self) -> String {
        format!("{}/{}", self.team_path(), self.agent_id)
    }

    /// "emea-logistics-1" -> "Logistics 1"
    pub fn label(&self) -> String {
        format!("{} {}", team_label(&self.team), suffix(&self.agent_id))
    }

    /// The agent runner's view of this agent.
    pub fn spec(&self) -> AgentSpec {
        AgentSpec {
            agent_id: self.agent_id.clone(),
            team: self.team.clone(),
            department: self.department.clone(),
            api_key: self.api_key.clone(),
            local_db: self.local_db.clone(),
            observations: self.observations.clone(),
            state_file: self.state_file.clone(),
            path: self.path(),
        }
    }
}

#[derive(Debug)]
pub struct LiveScenario {
    pub workdir: PathBuf,
    /// seconds over which the stage-"observe" agents start sharing
    pub spread: f64,
    pub agents: Vec<LiveAgent>,
}

impl LiveScenario {
    pub fn build(workdir: &Path, spread: f64) -> Result<LiveScenario> {
        std::fs::create_dir_all(workdir).with_context(|| format!("creating {}", workdir.display()))?;
        let mut agents = Vec::new();
        let agent = |region: &str, dept: &str, team: &str, i: usize, role, stage, shared, private| LiveAgent {
            agent_id: format!("{region}-{team}-{i}"),
            region: region.into(),
            department: dept.into(),
            team: team.into(),
            role,
            stage,
            shared,
            private,
            ..LiveAgent::default()
        };
        for region in ["emea", "apac"] {
            for (team, (dept, stage)) in CORE_TEAMS {
                let (shared, private) = (core_observations(region, team), private(team));
                for i in 1..=2 {
                    agents.push(agent(
                        region,
                        dept,
                        team,
                        i,
                        "core",
                        stage,
                        vec![shared[i - 1].clone()],
                        vec![private[i - 1].clone()],
                    ));
                }
            }
        }
        for (team, dept) in HQ_CORE_TEAMS {
            agents.push(agent("hq", dept, team, 1, "core", "synthesise", hq_observations(team), private(team)));
        }
        for region in ["emea", "apac", "hq"] {
            let teams: &[(&str, &str)] = if region == "hq" { &HQ_BACKGROUND_TEAMS } else { &BACKGROUND_TEAMS };
            for &(team, dept) in teams {
                let (notes, private) = (background_observations(region, team), private(team));
                for i in 1..=2 {
                    agents.push(agent(
                        region,
                        dept,
                        team,
                        i,
                        "background",
                        "observe",
                        vec![notes[i - 1].clone()],
                        vec![private[(i - 1) % private.len()].clone()],
                    ));
                }
            }
        }
        // a deterministic trickle: background and core notes interleave over `spread` seconds
        let mut observe: Vec<usize> = (0..agents.len()).filter(|&i| agents[i].stage == "observe").collect();
        observe.sort_by(|&a, &b| {
            let key = |a: &LiveAgent| (suffix(&a.agent_id).to_string(), a.team.clone(), a.region.clone());
            key(&agents[a]).cmp(&key(&agents[b]))
        });
        let steps = observe.len().saturating_sub(1).max(1) as f64;
        for (n, &i) in observe.iter().enumerate() {
            agents[i].delay = (spread * n as f64 / steps * 100.0).round() / 100.0;
        }
        Ok(LiveScenario { workdir: workdir.to_path_buf(), spread, agents })
    }

    // ------------------------------------------------------------------ queries over the scenario
    pub fn stage(&self, stage: &str) -> Vec<&LiveAgent> {
        self.agents.iter().filter(|a| a.stage == stage).collect()
    }

    pub fn by_role(&self, role: &str) -> Vec<&LiveAgent> {
        self.agents.iter().filter(|a| a.role == role).collect()
    }

    pub fn by_region(&self, region: &str) -> Vec<&LiveAgent> {
        self.agents.iter().filter(|a| a.region == region).collect()
    }

    pub fn agent(&self, agent_id: &str) -> Option<&LiveAgent> {
        self.agents.iter().find(|a| a.agent_id == agent_id)
    }

    /// Every organizational unit above the agents, parents before children (the trace's `org.units`).
    pub fn units(&self) -> Vec<Value> {
        let mut seen = HashSet::new();
        let mut units = vec![json!({"path": ENTERPRISE, "layer": "enterprise", "label": COMPANY, "parent": null})];
        seen.insert(ENTERPRISE.to_string());
        let mut add = |path: String, layer: &str, label: &str, parent: &str| {
            if seen.insert(path.clone()) {
                units.push(json!({"path": path, "layer": layer, "label": label, "parent": parent}));
            }
        };
        for a in &self.agents {
            let (region_label, _, sub_label) = region(&a.region);
            let region = format!("{ENTERPRISE}/{}", a.region);
            let sub = format!("{region}/{}", a.subsidiary());
            let dept = format!("{sub}/{}", a.department);
            add(region.clone(), "region", region_label, ENTERPRISE);
            add(sub.clone(), "subsidiary", sub_label, &region);
            add(dept.clone(), "department", department_label(&a.department), &sub);
            add(a.team_path(), "team", team_label(&a.team), &dept);
        }
        units
    }

    // ------------------------------------------------------------------ actions
    /// Register one agent; keeps its key in memory and returns the service's view of the agent.
    pub fn register(&mut self, admin: &MycelicClient, index: usize) -> Result<Value> {
        let a = &mut self.agents[index];
        let res = admin.register_agent(&Registration {
            agent_id: a.agent_id.clone(),
            enterprise: ENTERPRISE.into(),
            region: Some(a.region.clone()),
            subsidiary: Some(a.subsidiary().into()),
            department: Some(a.department.clone()),
            team: Some(a.team.clone()),
            display_name: Some(a.label()),
            scopes: None,
        })?;
        a.api_key = res["api_key"].as_str().unwrap_or_default().to_string();
        Ok(res["agent"].clone())
    }

    /// Write the JSONL file the agent process consumes: its shared notes (`a.shared` unless `shared` is given), then
    /// its private ones.  Only the first file of an agent carries its start delay.
    pub fn write_observations(
        &mut self,
        index: usize,
        shared: Option<&[Value]>,
        tag: &str,
        include_private: bool,
    ) -> Result<()> {
        let workdir = self.workdir.clone();
        let a = &mut self.agents[index];
        let notes = shared.unwrap_or(&a.shared);
        let mut lines = Vec::new();
        for (n, obs) in notes.iter().enumerate() {
            let mut line = obs.clone();
            line["share"] = json!(true);
            line["local_id"] = json!(format!("{}-{tag}-{n}", a.agent_id));
            if n == 0 && shared.is_none() && a.delay != 0.0 {
                line["delay"] = json!(a.delay);
            }
            lines.push(line.to_string());
        }
        if include_private {
            for (n, text) in a.private.iter().enumerate() {
                lines.push(
                    json!({"text": text, "share": false, "local_id": format!("{}-private-{n}", a.agent_id)})
                        .to_string(),
                );
            }
        }
        a.observations = workdir.join(format!("{}.{tag}.jsonl", a.agent_id));
        std::fs::write(&a.observations, lines.join("\n") + "\n")
            .with_context(|| format!("writing {}", a.observations.display()))?;
        a.local_db = workdir.join(format!("{}.local.db", a.agent_id));
        a.state_file = workdir.join(format!("{}.state.json", a.agent_id));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forty_agents_in_three_stages_with_a_trickle() {
        let tmp = tempfile::tempdir().unwrap();
        let sc = LiveScenario::build(tmp.path(), 6.0).unwrap();
        assert_eq!((sc.agents.len(), sc.by_role("core").len(), sc.by_role("background").len()), (40, 14, 26));
        assert_eq!(STAGES.map(|s| sc.stage(s).len()), [34, 4, 2]);
        let delays: Vec<f64> = sc.stage("observe").iter().map(|a| a.delay).collect();
        assert_eq!(delays.iter().cloned().fold(f64::MIN, f64::max), 6.0);
        assert_eq!(delays.iter().filter(|d| **d == 0.0).count(), 1);
        assert_eq!(sc.agent("emea-logistics-1").unwrap().label(), "Logistics 1");
        assert_eq!(sc.agent("hq-fp-and-a-2").unwrap().team_path(), "orrery/hq/orrery-hq/finance/fp-and-a");
        let units = sc.units();
        assert_eq!(units[0]["path"], ENTERPRISE);
        assert_eq!(units.iter().filter(|u| u["layer"] == "team").count(), 21);
    }
}
