//! The Northwind strategic scenario: two regions discover the same supply problem independently; HQ knows the supply
//! base is concentrated and demand is growing.  No region, department or team can see the strategic conclusion;
//! Mycelic composes it from the regional conclusions (which are themselves composed from agents' notes) and keeps
//! every hop in the lineage.
//!
//! ```text
//! northwind (enterprise)
//! ├─ emea / nw-gmbh        ops: logistics, procurement   commercial: field-sales
//! ├─ apac / nw-kk          ops: logistics, procurement   commercial: field-sales
//! └─ hq   / northwind-hq   strategy: sourcing, analytics
//! ```
//!
//! Rules (deploy/mycelic/rules.json): `regional_supply_risk` fires at the region layer and emits the slot
//! `supply_risk`; `strategic_second_source` fires at the enterprise once `supply_risk` is corroborated in at least
//! two regions and HQ has observed `demand_growth` and `supplier_concentration`.

use crate::scenario::{AgentSpec, ENTERPRISE, ENTITY, title};
use anyhow::Result;
use mycelic_sdk::{MycelicClient, Registration};
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Duration;

pub const REGIONS: [(&str, &str); 2] = [("emea", "nw-gmbh"), ("apac", "nw-kk")];
pub const TEAMS: [(&str, &str); 3] = [("logistics", "ops"), ("procurement", "ops"), ("field-sales", "commercial")];
pub const HQ_REGION: &str = "hq";
pub const HQ_SUBSIDIARY: &str = "northwind-hq";
pub const HQ_DEPARTMENT: &str = "strategy";
pub const STRATEGIC_QUERY: &str = "Should we qualify a second source for the SD-9 servo drive?";

fn obs(text: String, topic: &str, slot: &str, confidence: f64) -> Value {
    json!({"text": text, "topic": topic, "slot": slot, "entity": ENTITY, "confidence": confidence})
}

pub fn regional_observations(region: &str, team: &str) -> Vec<Value> {
    let (port, customer) =
        if region == "emea" { ("Rotterdam", "Helios Automation") } else { ("Busan", "Daichi Robotics") };
    match team {
        "logistics" => vec![
            obs(format!("Port of {port} strike announced for weeks 41-43; SD-9 servo drive shipments route through it."), "supply:sd-9/transport", "transport_disruption", 0.9),
            obs(format!("Carrier ETA for the SD-9 servo drive container via {port} slipped by 12 days."), "supply:sd-9/transport", "transport_disruption", 0.7),
        ],
        "procurement" => vec![
            obs("Kessler Antriebe confirms roughly two weeks of SD-9 inventory for this region and no second source qualified.".into(), "supply:sd-9/supplier", "supplier_buffer_low", 0.85),
            obs("Kessler Antriebe quoted 9 weeks lead time for additional SD-9 units.".into(), "supply:sd-9/supplier", "supplier_buffer_low", 0.75),
        ],
        _ => vec![
            obs(format!("{customer} committed to 40 RX-4 arms for November; every RX-4 uses an SD-9 drive."), "supply:sd-9/demand", "demand_commitment", 0.8),
            obs(format!("{customer} asked for delivery dates on 12 more RX-4 arms in Q4."), "supply:sd-9/demand", "demand_commitment", 0.65),
        ],
    }
}

pub fn hq_observations(team: &str) -> Vec<Value> {
    match team {
        "sourcing" => vec![obs(
            "Kessler Antriebe is the only qualified SD-9 source for every subsidiary; no second source is qualified."
                .into(),
            "strategy:supply-base",
            "supplier_concentration",
            0.95,
        )],
        _ => vec![obs(
            "RX-4 order intake is up 40% year over year across all regions.".into(),
            "strategy:demand",
            "demand_growth",
            0.8,
        )],
    }
}

pub fn private(team: &str) -> &'static str {
    match team {
        "logistics" => "Reminder: renew the forklift certification before October.",
        "procurement" => "Personal note: Kessler's account manager prefers calls on Tuesdays.",
        "field-sales" => "Internal: discount ceiling for this customer is 8%.",
        "sourcing" => "Draft: shortlist of alternative drive suppliers, not yet validated.",
        _ => "Working note: the Q3 forecast model still double-counts returns.",
    }
}

pub struct StrategicScenario {
    pub workdir: PathBuf,
    pub agents: Vec<AgentSpec>,
}

impl StrategicScenario {
    pub fn build(workdir: &Path, agents_per_team: usize) -> Result<StrategicScenario> {
        std::fs::create_dir_all(workdir)?;
        let mut agents = Vec::new();
        for (region, sub) in REGIONS {
            for (team, dept) in TEAMS {
                for i in 1..=agents_per_team {
                    let id = format!("{region}-{team}-{i}");
                    agents.push(AgentSpec {
                        path: format!("{ENTERPRISE}/{region}/{sub}/{dept}/{team}/{id}"),
                        agent_id: id,
                        team: team.into(),
                        department: dept.into(),
                        ..AgentSpec::default()
                    });
                }
            }
        }
        for team in ["sourcing", "analytics"] {
            let id = format!("hq-{team}-1");
            agents.push(AgentSpec {
                path: format!("{ENTERPRISE}/{HQ_REGION}/{HQ_SUBSIDIARY}/{HQ_DEPARTMENT}/{team}/{id}"),
                agent_id: id,
                team: team.into(),
                department: HQ_DEPARTMENT.into(),
                ..AgentSpec::default()
            });
        }
        Ok(StrategicScenario { workdir: workdir.to_path_buf(), agents })
    }

    pub fn region_of(a: &AgentSpec) -> &str {
        a.path.split('/').nth(1).unwrap_or("")
    }

    pub fn register_all(&mut self, admin: &MycelicClient) -> Result<()> {
        for a in &mut self.agents {
            let parts: Vec<&str> = a.path.split('/').collect();
            let res = admin.register_agent(&Registration {
                agent_id: a.agent_id.clone(),
                enterprise: parts[0].into(),
                region: Some(parts[1].into()),
                subsidiary: Some(parts[2].into()),
                department: Some(parts[3].into()),
                team: Some(parts[4].into()),
                display_name: Some(title(&a.agent_id)),
                scopes: None,
            })?;
            a.api_key = res["api_key"].as_str().unwrap_or_default().to_string();
        }
        Ok(())
    }

    pub fn write_observations(&mut self) -> Result<()> {
        for a in &mut self.agents {
            let region = Self::region_of(a).to_string();
            let idx: usize = a.agent_id.rsplit('-').next().and_then(|n| n.parse::<usize>().ok()).unwrap_or(1) - 1;
            let shared =
                if region == HQ_REGION { hq_observations(&a.team) } else { regional_observations(&region, &a.team) };
            let mut line = shared[idx % shared.len()].clone();
            line["share"] = json!(true);
            line["local_id"] = json!(format!("{}-obs-0", a.agent_id));
            let lines = [
                line.to_string(),
                json!({"text": private(&a.team), "share": false, "local_id": format!("{}-private", a.agent_id)})
                    .to_string(),
            ];
            a.observations = self.workdir.join(format!("{}.observations.jsonl", a.agent_id));
            std::fs::write(&a.observations, lines.join("\n") + "\n")?;
            a.local_db = self.workdir.join(format!("{}.local.db", a.agent_id));
            a.state_file = self.workdir.join(format!("{}.state.json", a.agent_id));
        }
        Ok(())
    }

    pub fn run_agents(&self, base_url: &str) -> Result<BTreeMap<String, Value>> {
        let all: Vec<&AgentSpec> = self.agents.iter().collect();
        crate::scenario::run_agents(&self.workdir, base_url, &all, Duration::from_secs(120), &|| false)
    }

    pub fn by_region(&self, region: &str) -> Vec<&AgentSpec> {
        self.agents.iter().filter(|a| Self::region_of(a) == region).collect()
    }

    pub fn agent(&self, agent_id: &str) -> &AgentSpec {
        self.agents.iter().find(|a| a.agent_id == agent_id).expect("agent in the scenario")
    }
}
