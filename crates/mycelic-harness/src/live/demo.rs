//! The story: nine acts run against a real deployment, every claim checked and recorded.

use super::scenario::{ENTERPRISE, LiveAgent, LiveScenario, STRATEGIC_QUERY, recount};
use super::signals;
use super::trace::TraceRecorder;
use super::watcher::{MemoryWatcher, all_memories, metric_payload};
use crate::scenario::{AgentSpec, run_agents};
use crate::smoke::wait_idle;
use crate::{Driver, leaf};
use anyhow::{Context, Result, anyhow};
use mycelic_sdk::{MycelicClient, QueryOptions};
use serde_json::{Map, Value, json};
use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

/// The two rules of the story, copied verbatim from `deploy/mycelic/rules.json` into the process driver's rules file.
pub const DEMO_RULES: [&str; 2] = ["regional_supply_risk", "strategic_second_source"];
pub const ACT_COUNT: usize = 9;
const AGENT_TIMEOUT: Duration = Duration::from_secs(120);

/// Act titles and captions; every number in a caption is counted from the scenario.
pub fn acts_for(sc: &LiveScenario) -> Vec<Value> {
    let (n, core, bg) = (sc.agents.len(), sc.by_role("core").len(), sc.by_role("background").len());
    let observe = sc.stage("observe").len();
    let act = |id: &str, title: &str, caption: String| json!({"id": id, "title": title, "caption": caption});
    vec![
        act(
            "cold-open",
            "Every agent sees a sliver",
            "A company runs dozens of AI agents. Each sees a sliver. Who knows what they know?".into(),
        ),
        act(
            "organisation",
            "The organisation",
            format!("{n} agents across EMEA, APAC and head office. Each has its own key and its own local memory."),
        ),
        act(
            "observe",
            "Agents observe",
            format!(
                "{observe} agents share notes; every agent keeps private notes on its own disk. \
                 Mycelic rolls the shared ones up the org chart."
            ),
        ),
        act(
            "regions",
            "Each region concludes",
            "Procurement in each region shares its stock level. The operator-written rule is evaluated in EMEA \
             and in APAC, separately."
                .into(),
        ),
        act(
            "enterprise",
            "The enterprise puts together what no single team held",
            "Head office shares demand growth and supplier concentration. The strategy rule is evaluated at \
             the enterprise."
                .into(),
        ),
        act(
            "receipts",
            "Every conclusion has a receipt",
            "Ask for the strategy's lineage: who said what, when, how sure. Then the same lineage as another \
             agent sees it."
                .into(),
        ),
        act(
            "evidence",
            "Strategy follows evidence",
            "APAC finds its stock was counted twice and retracts. Watch what rested on those notes, then a \
             recount arrives."
                .into(),
        ),
        act(
            "plug",
            "Pull the plug",
            "Kill the service. Delete its database. Rebuild its memories, lineage and agents from the event log."
                .into(),
        ),
        act(
            "close",
            "Shared memory for organisations of AI agents",
            format!(
                "Verified checks and real totals from this run: {core} story agents, {bg} background agents, \
                 no LLM in the data path."
            ),
        ),
    ]
}

/// A JSON value as Python's `str()` would print it inside a check sentence.
pub fn py_str(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Null => "None".into(),
        Value::Bool(true) => "True".into(),
        Value::Bool(false) => "False".into(),
        other => other.to_string(),
    }
}

/// A float as Python prints it (`1.0`, not `1`).
pub fn py_float(x: f64) -> String {
    json!(x).to_string()
}

fn str_of(v: &Value) -> String {
    v.as_str().unwrap_or_default().to_string()
}

/// The regions a search returned conclusions from, as a check sentence reads them: "EMEA only", "none".
pub fn region_hits(scopes: &BTreeSet<String>) -> String {
    let mut names: Vec<String> = scopes.iter().map(|s| leaf(s).to_uppercase()).collect();
    names.sort();
    match names.len() {
        0 => "none".into(),
        1 => format!("{} only", names[0]),
        _ => names.join(" and "),
    }
}

fn query_payload(res: &Value, viewer: &str, question: &str, scope: &str, min_layer: &str) -> Value {
    let results: Vec<Value> = res["results"]
        .as_array()
        .into_iter()
        .flatten()
        .map(|h| {
            let m = &h["memory"];
            json!({"memory_id": m["memory_id"], "layer": m["layer"], "scope": m["scope"], "text": m["text"],
                   "score": h["score"], "rule_id": m["rule_id"], "operator": m["operator"]})
        })
        .collect();
    json!({"viewer": viewer, "question": question, "scope": scope, "min_layer": min_layer, "results": results})
}

fn edges(g: &Value) -> BTreeSet<(String, String)> {
    g["edges"].as_array().into_iter().flatten().map(|e| (str_of(&e["child"]), str_of(&e["parent"]))).collect()
}

fn strings(v: &Value) -> Vec<String> {
    v.as_array().into_iter().flatten().map(str_of).collect()
}

fn hits(res: &Value) -> impl Iterator<Item = &Value> {
    res["results"].as_array().into_iter().flatten().map(|h| &h["memory"])
}

fn opts(scope: &str, min_layer: &str, k: i64) -> QueryOptions {
    QueryOptions {
        scope: Some(scope.into()),
        min_layer: min_layer.into(),
        k,
        include_lineage: false,
        ..QueryOptions::default()
    }
}

/// Runs the acts against a real deployment and records what happened.
pub struct LiveDemo {
    pub driver: Box<dyn Driver>,
    pub rec: Arc<TraceRecorder>,
    pub sc: LiveScenario,
    poll: f64,
    fast_poll: f64,
    log: Box<dyn Fn(&str)>,
    admin: Option<MycelicClient>,
    watcher: Option<MemoryWatcher>,
    /// each agent's local-state summary after it last ran, in the order the agents first ran
    states: Map<String, Value>,
    regional: BTreeMap<String, Value>,
    strategy: Option<Value>,
    numbers: Map<String, Value>,
}

impl LiveDemo {
    pub fn new(
        driver: Box<dyn Driver>,
        rec: Arc<TraceRecorder>,
        workdir: &Path,
        spread: f64,
        poll: f64,
        fast_poll: f64,
    ) -> Result<LiveDemo> {
        let sc = LiveScenario::build(&workdir.join("agents"), spread)?;
        let acts: Vec<Value> = acts_for(&sc)
            .into_iter()
            .enumerate()
            .map(|(i, mut a)| {
                a["index"] = json!(i);
                a["t0"] = Value::Null;
                a["t1"] = Value::Null;
                a
            })
            .collect();
        let agents: Vec<Value> = sc
            .agents
            .iter()
            .map(|a| {
                json!({"agent_id": a.agent_id, "path": a.path(), "team_path": a.team_path(), "region": a.region,
                       "label": a.label(), "role": a.role, "stage": a.stage})
            })
            .collect();
        let units = sc.units();
        rec.update(|tr| {
            tr["acts"] = json!(acts);
            tr["org"] = json!({"enterprise": ENTERPRISE, "units": units, "agents": agents});
        });
        let mut numbers = Map::new();
        numbers.insert("time_to_region_s".into(), json!({}));
        Ok(LiveDemo {
            driver,
            rec,
            sc,
            poll,
            fast_poll,
            log: Box::new(|m| println!("{m}")),
            admin: None,
            watcher: None,
            states: Map::new(),
            regional: BTreeMap::new(),
            strategy: None,
            numbers,
        })
    }

    // ------------------------------------------------------------------ plumbing
    /// Everything a trace must never contain: the deployment's secrets and every agent key.
    pub fn secrets(&self) -> Vec<String> {
        let mut s = self.driver.secrets();
        s.extend(self.sc.agents.iter().map(|a| a.api_key.clone()));
        s
    }

    pub fn run_act(&mut self, index: usize) -> Result<()> {
        signals::checkpoint()?;
        let title = self.rec.read(|tr| str_of(&tr["acts"][index]["title"]));
        (self.log)(&format!("\n=== act {index}: {title}"));
        self.rec.act_start(index);
        match index {
            0 => self.cold_open(),
            1 => self.organisation(),
            2 => self.observe(),
            3 => self.regions(),
            4 => self.enterprise(),
            5 => self.receipts(),
            6 => self.evidence(),
            7 => self.plug(),
            8 => self.close(),
            _ => Err(anyhow!("there is no act {index}")),
        }?;
        signals::checkpoint()?;
        self.rec.act_end(index);
        Ok(())
    }

    fn check(&self, check_id: &str, text: &str, ok: bool) -> bool {
        (self.log)(&format!("  {} {text}", if ok { "ok " } else { "FAIL" }));
        self.rec.check(check_id, text, ok)
    }

    fn admin(&self) -> Result<&MycelicClient> {
        self.admin.as_ref().ok_or_else(|| anyhow!("the service is not up"))
    }

    fn watcher(&self) -> Result<&MemoryWatcher> {
        self.watcher.as_ref().ok_or_else(|| anyhow!("the watcher is not running"))
    }

    fn index(&self, agent_id: &str) -> Result<usize> {
        self.sc.agents.iter().position(|a| a.agent_id == agent_id).ok_or_else(|| anyhow!("no agent {agent_id}"))
    }

    fn client(&self, agent_id: &str) -> Result<MycelicClient> {
        let a = self.sc.agent(agent_id).ok_or_else(|| anyhow!("no agent {agent_id}"))?;
        Ok(MycelicClient::new(&self.driver.base_url(), &a.api_key))
    }

    /// Wait until the service has applied everything, then record the final state of this step.
    fn settle(&self, timeout: Duration) -> Result<Value> {
        signals::checkpoint()?;
        let st = wait_idle(self.admin()?, timeout)?;
        signals::checkpoint()?;
        if let Some(w) = &self.watcher {
            w.poll_once(true)?;
            w.metric(true)?;
        }
        Ok(st)
    }

    fn run_stage(&mut self, agent_ids: &[String]) -> Result<Map<String, Value>> {
        let specs: Vec<AgentSpec> = agent_ids.iter().filter_map(|id| self.sc.agent(id)).map(LiveAgent::spec).collect();
        let refs: Vec<&AgentSpec> = specs.iter().collect();
        self.watcher()?.set_fast(true);
        let ran = run_agents(&self.sc.workdir, &self.driver.base_url(), &refs, AGENT_TIMEOUT, &signals::interrupted)
            .and_then(|states| self.settle(Duration::from_secs(90)).map(|_| states));
        self.watcher()?.set_fast(false);
        let states = ran?;
        for id in agent_ids {
            let known: HashSet<String> =
                self.states.get(id).map(|s| strings(&s["local_only"])).unwrap_or_default().into_iter().collect();
            for text in strings(&states.get(id).cloned().unwrap_or_default()["local_only"]) {
                if !known.contains(&text) {
                    self.rec.emit("note_private", json!({"agent_id": id, "text": text, "local_only": true}));
                }
            }
        }
        let states: Map<String, Value> = states.into_iter().collect();
        for (k, v) in &states {
            self.states.insert(k.clone(), v.clone());
        }
        Ok(states)
    }

    fn latency(&self, memory: &Value) -> Option<f64> {
        self.watcher.as_ref().and_then(|w| w.latency(memory))
    }

    fn regional_conclusions(&self) -> Result<BTreeMap<String, Value>> {
        Ok(self
            .admin()?
            .list_memories(Some(ENTERPRISE), Some("region"), 200)?
            .into_iter()
            .filter(|m| m["rule_id"] == "regional_supply_risk")
            .map(|m| (str_of(&m["scope"]), m))
            .collect())
    }

    /// Head office asks the strategic question; the strategy conclusion it gets, as the administrator reads it.
    fn ask_strategy(&self) -> Result<Option<Value>> {
        signals::checkpoint()?;
        let viewer = "hq-sourcing-1";
        let res = self.client(viewer)?.query(STRATEGIC_QUERY, &opts(ENTERPRISE, "enterprise", 5))?;
        self.rec.emit("query", query_payload(&res, viewer, STRATEGIC_QUERY, ENTERPRISE, "enterprise"));
        match hits(&res).find(|m| m["rule_id"] == "strategic_second_source") {
            Some(m) => Ok(Some(self.admin()?.get_memory(m["memory_id"].as_str().unwrap_or_default())?)),
            None => Ok(None),
        }
    }

    /// Private notes read from the agents' own local state, and those whose text the service holds anywhere.
    fn private_leaks(&self, agent_ids: &[String]) -> Result<(usize, Vec<String>)> {
        let admin = self.admin()?;
        let texts: Vec<String> = agent_ids
            .iter()
            .flat_map(|id| strings(&self.states.get(id).cloned().unwrap_or_default()["local_only"]))
            .collect();
        let held: Vec<String> = all_memories(admin)?.values().map(|m| str_of(&m["text"])).collect();
        let mut leaks: BTreeSet<String> =
            texts.iter().filter(|t| held.iter().any(|h| h.contains(t.as_str()))).cloned().collect();
        for t in &texts {
            // and the keyword index, as an administrator sees it
            signals::checkpoint()?;
            let res = admin.query(t, &opts(ENTERPRISE, "agent", 5))?;
            if hits(&res).any(|m| str_of(&m["text"]).contains(t.as_str())) {
                leaks.insert(t.clone());
            }
        }
        Ok((texts.len(), leaks.into_iter().collect()))
    }

    fn ids(agents: Vec<&LiveAgent>) -> Vec<String> {
        agents.into_iter().map(|a| a.agent_id.clone()).collect()
    }

    // ------------------------------------------------------------------ acts
    fn cold_open(&mut self) -> Result<()> {
        let name = self.driver.name();
        self.rec.emit(
            "service",
            json!({"state": "starting", "detail": format!("nats-server + Mycelic service ({name} driver)")}),
        );
        self.driver.up()?;
        let admin = MycelicClient::new(&self.driver.base_url(), &self.driver.admin_token());
        let h = admin.health()?;
        self.admin = Some(admin);
        let version = h["version"].clone();
        self.rec.update(|tr| tr["meta"]["mycelic_version"] = version);
        let (version, status) = (py_str(&h["version"]), py_str(&h["status"]));
        self.rec.emit(
            "service",
            json!({"state": "healthy", "detail": format!("Mycelic {version}: {status}, event log on NATS JetStream")}),
        );
        let rules: Vec<Value> =
            self.admin()?.list_rules()?.into_iter().filter(|r| r.get("enabled").is_none_or(|e| e != false)).collect();
        let listed: Vec<Value> = rules
            .iter()
            .map(|r| {
                let mut out = Map::new();
                for k in [
                    "rule_id",
                    "target_layer",
                    "required_slots",
                    "emits_slot",
                    "min_units",
                    "min_agents",
                    "min_teams",
                    "conclusion",
                    "topic_prefix",
                    "sources",
                    "corroborate",
                    "kind",
                ] {
                    out.insert(k.into(), r.get(k).cloned().unwrap_or(Value::Null));
                }
                out.insert("owner".into(), r["metadata"]["owner"].clone());
                Value::Object(out)
            })
            .collect();
        self.rec.update(|tr| tr["rules"] = json!(listed));
        let have: BTreeSet<String> = rules.iter().map(|r| str_of(&r["rule_id"])).collect();
        self.check(
            "service_ready",
            &format!(
                "the service is {status} and runs the operator-written rules {}",
                have.iter().cloned().collect::<Vec<_>>().join(", ")
            ),
            h["status"] == "ok" && DEMO_RULES.iter().all(|r| have.contains(*r)),
        );
        let mut watcher = MemoryWatcher::new(
            &self.driver.base_url(),
            &self.driver.admin_token(),
            Arc::clone(&self.rec),
            &self.sc,
            self.poll,
            self.fast_poll,
        );
        watcher.metric(true)?;
        watcher.start();
        self.watcher = Some(watcher);
        Ok(())
    }

    fn organisation(&mut self) -> Result<()> {
        let admin = self.admin()?.clone();
        for i in 0..self.sc.agents.len() {
            signals::checkpoint()?;
            let agent = self.sc.register(&admin, i)?;
            let a = &self.sc.agents[i];
            self.rec.emit(
                "agent_registered",
                json!({"agent_id": a.agent_id, "path": agent["path"], "team_path": a.team_path(), "region": a.region,
                       "role": a.role, "label": a.label()}),
            );
            self.sc.write_observations(i, None, "obs", true)?;
        }
        let registered = admin.list_agents(Some(ENTERPRISE))?.into_iter().filter(|x| x["status"] == "active").count();
        let keys: HashSet<&str> = self.sc.agents.iter().map(|a| a.api_key.as_str()).collect();
        let teams: HashSet<String> = self.sc.agents.iter().map(LiveAgent::team_path).collect();
        let n = self.sc.agents.len();
        self.check(
            "agents_registered",
            &format!(
                "{registered} agents in {} teams registered, each with its own key ({} in the SD-9 story, {} on other work)",
                teams.len(),
                self.sc.by_role("core").len(),
                self.sc.by_role("background").len()
            ),
            registered == n && n == keys.len() && keys.iter().all(|k| k.starts_with("mk_")),
        );
        self.watcher()?.metric(true)
    }

    fn observe(&mut self) -> Result<()> {
        let agents = Self::ids(self.sc.stage("observe"));
        let states = self.run_stage(&agents)?;
        let shared: i64 = states.values().map(|s| s["counts"]["shared"].as_i64().unwrap_or(0)).sum();
        let (n, leaks) = self.private_leaks(&agents)?;
        (self.log)(&format!("  {} agents shared {shared} notes and kept {n} private", agents.len()));
        self.check(
            "private_stayed_local",
            &format!(
                "{n} private notes stayed on {} agents' own disks: every memory and the keyword index were searched \
                 and none of their text is in the service",
                agents.len()
            ),
            n >= agents.len() && leaks.is_empty(),
        );
        Ok(())
    }

    fn regions(&mut self) -> Result<()> {
        self.run_stage(&Self::ids(self.sc.stage("conclude")))?;
        self.regional = self.regional_conclusions()?;
        let mut times = Map::new();
        for (scope, m) in &self.regional {
            times.insert(leaf(scope).to_string(), json!(self.latency(m)));
            let text: String = str_of(&m["text"]).chars().take(100).collect();
            (self.log)(&format!(
                "  [{scope}] {} agents / {} teams: {text}",
                py_str(&m["support"]),
                py_str(&m["independent_teams"])
            ));
        }
        if let Some(Value::Object(t)) = self.numbers.get_mut("time_to_region_s") {
            t.extend(times);
        }
        let both: BTreeSet<String> = [format!("{ENTERPRISE}/emea"), format!("{ENTERPRISE}/apac")].into();
        self.check(
            "regions_concluded",
            "EMEA and APAC each reached a regional supply-risk conclusion, independently",
            self.regional.keys().cloned().collect::<BTreeSet<_>>() == both,
        );
        let (viewer, question) = ("emea-field-sales-1", "supply risk sd-9");
        let res = self.client(viewer)?.query(question, &opts(ENTERPRISE, "region", 10))?;
        self.rec.emit("query", query_payload(&res, viewer, question, ENTERPRISE, "region"));
        let seen: BTreeSet<String> =
            hits(&res).filter(|m| m["layer"] == "region").map(|m| str_of(&m["scope"])).collect();
        let others: BTreeSet<String> = self
            .sc
            .units()
            .iter()
            .filter(|u| u["layer"] == "region")
            .map(|u| leaf(u["path"].as_str().unwrap_or_default()).to_string())
            .filter(|r| r != "emea")
            .collect();
        let quoting = hits(&res).any(|m| {
            m["layer"] == "enterprise" && others.iter().any(|o| str_of(&m["text"]).contains(&format!("[{o}]")))
        });
        self.check(
            "region_visibility",
            &(format!(
                "access follows the org chart: {viewer}'s search returns EMEA's regional conclusion, not APAC's \
                 (region-layer hits: {})",
                region_hits(&seen)
            ) + if quoting { "; the enterprise summary it can read quotes both regions' conclusions" } else { "" }),
            seen == BTreeSet::from([format!("{ENTERPRISE}/emea")])
                && hits(&res).any(|m| m["rule_id"] == "regional_supply_risk"),
        );
        Ok(())
    }

    fn enterprise(&mut self) -> Result<()> {
        self.run_stage(&Self::ids(self.sc.stage("synthesise")))?;
        let strat = self.ask_strategy()?;
        self.check(
            "strategy_exists",
            "the enterprise holds a strategic recommendation: qualify a second source",
            strat.is_some(),
        );
        let Some(strat) = strat else { return Ok(()) };
        self.strategy = Some(strat.clone());
        let regions = strings(&strat["metadata"]["corroborated_units"]["supply_risk"]["region"]);
        let mut names: Vec<String> = regions.iter().map(|r| leaf(r).to_uppercase()).collect();
        names.sort();
        self.check(
            "corroborated_regions",
            &format!(
                "supply risk corroborated in {} regions ({}); support {} agents in {} teams",
                regions.len(),
                names.join(" and "),
                py_str(&strat["support"]),
                py_str(&strat["independent_teams"])
            ),
            regions.len() >= 2,
        );
        let g = self.admin()?.lineage(strat["memory_id"].as_str().unwrap_or_default())?;
        let layers = strings(&g["layers"]);
        self.check(
            "lineage_layers",
            &format!("its lineage crosses {}: two derivation steps from notes to strategy", layers.join(" -> ")),
            layers == ["agent", "region", "enterprise"],
        );
        let tte = self.latency(&strat);
        self.numbers.insert("time_to_enterprise_s".into(), json!(tte));
        let text = match tte {
            Some(t) => format!(
                "head office's last note accepted -> enterprise conclusion readable within {:.0} ms (upper bound, \
                 measured by polling the API)",
                t * 1000.0
            ),
            None => "time to the enterprise conclusion could not be measured".into(),
        };
        self.check("time_to_enterprise", &text, tte.is_some_and(|t| t >= 0.0));
        // background traffic: consolidated, but no rule conclusion rests on it
        let bg: HashSet<String> = self.sc.by_role("background").iter().map(|a| a.agent_id.clone()).collect();
        let mems = all_memories(self.admin()?)?;
        let bg_notes: HashSet<String> = mems
            .iter()
            .filter(|(_, m)| m["layer"] == "agent" && bg.contains(m["producer_id"].as_str().unwrap_or_default()))
            .map(|(id, _)| id.clone())
            .collect();
        let roots = |m: &Value| strings(&m["metadata"]["roots"]);
        let rule_hits = mems
            .values()
            .filter(|m| m["operator"] == "slot_composition" && roots(m).iter().any(|r| bg_notes.contains(r)))
            .count();
        let bg_cons = mems
            .values()
            .filter(|m| {
                let r = roots(m);
                m["operator"] == "topic_consolidation" && !r.is_empty() && r.iter().all(|x| bg_notes.contains(x))
            })
            .count();
        self.numbers.insert("background_notes".into(), json!(bg_notes.len()));
        self.numbers.insert("background_consolidations".into(), json!(bg_cons));
        self.numbers.insert("background_rule_conclusions".into(), json!(rule_hits));
        self.check(
            "background_no_rule",
            &format!(
                "{} background agents' {} notes consolidated into {bg_cons} memories and completed no rule",
                bg.len(),
                bg_notes.len()
            ),
            !bg_notes.is_empty() && bg_cons > 0 && rule_hits == 0,
        );
        Ok(())
    }

    fn receipts(&mut self) -> Result<()> {
        let Some(strat) = self.strategy.clone() else {
            self.check("redacted_view", "no strategy to trace", false);
            return Ok(());
        };
        let id = str_of(&strat["memory_id"]);
        let g = self.admin()?.lineage(&id)?;
        self.rec.emit("lineage", json!({"viewer": "admin", "memory_id": id, "graph": g}));
        let viewer = "emea-field-sales-1";
        let view = self.client(viewer)?.lineage(&id)?;
        self.rec.emit("lineage", json!({"viewer": viewer, "memory_id": id, "graph": view}));
        let empty = Map::new();
        let nodes = view["nodes"].as_object().unwrap_or(&empty);
        let mut redacted_by_region: BTreeMap<String, i64> = BTreeMap::new();
        for n in nodes.values().filter(|n| n["redacted"] == true) {
            let scope = str_of(&n["scope"]);
            let r = if scope.contains('/') { scope.split('/').nth(1).unwrap_or_default().to_string() } else { scope };
            *redacted_by_region.entry(r).or_default() += 1;
        }
        let g_nodes = g["nodes"].as_object().unwrap_or(&empty);
        let hq_notes: Vec<&String> = g_nodes
            .iter()
            .filter(|(_, n)| {
                n["layer"] == "agent"
                    && n["scope"].as_str().unwrap_or_default().starts_with(&format!("{ENTERPRISE}/hq/"))
            })
            .map(|(id, _)| id)
            .collect();
        let regional_id =
            |region: &str| self.regional.get(&format!("{ENTERPRISE}/{region}")).map(|m| str_of(&m["memory_id"]));
        let (apac, emea) = (regional_id("apac"), regional_id("emea"));
        let total: i64 = redacted_by_region.values().sum();
        let shown: Vec<String> = redacted_by_region.iter().map(|(k, v)| format!("{v} {k}")).collect();
        let node = |id: &Option<String>| id.as_ref().and_then(|id| nodes.get(id));
        let ok = view["redacted_contributions"].as_i64() == Some(total)
            && total > 0
            && node(&apac).is_some_and(|n| n["text"].is_null())
            && node(&emea).is_some_and(|n| n["redacted"] != true)
            && hq_notes.iter().all(|id| nodes.get(*id).is_some_and(|n| n["redacted"] == true))
            && nodes.keys().collect::<BTreeSet<_>>() == g_nodes.keys().collect::<BTreeSet<_>>()
            && edges(&view) == edges(&g);
        self.check(
            "redacted_view",
            &format!(
                "the same lineage for {viewer}: {} of {} contributions redacted ({}), its own region and note \
                 readable, shape intact",
                py_str(&view["redacted_contributions"]),
                nodes.len(),
                shown.join(", ")
            ),
            ok,
        );
        Ok(())
    }

    fn evidence(&mut self) -> Result<()> {
        let Some(strat) = self.strategy.clone() else {
            self.check("strategy_withdrawn", "no strategy to withdraw", false);
            return Ok(());
        };
        let strat_id = str_of(&strat["memory_id"]);
        let reason = "inventory was counted twice";
        let mut supplier_notes: Vec<Value> = self
            .admin()?
            .list_memories(Some(&format!("{ENTERPRISE}/apac")), Some("agent"), 200)?
            .into_iter()
            .filter(|m| m["slot"] == "supplier_buffer_low")
            .collect();
        supplier_notes.sort_by_key(|m| str_of(&m["producer_id"]));
        for m in &supplier_notes {
            signals::checkpoint()?;
            let producer = str_of(&m["producer_id"]);
            self.client(&producer)?.retract(m["memory_id"].as_str().unwrap_or_default(), reason)?;
            self.rec.emit("retracted", json!({"agent_id": producer, "memory_id": m["memory_id"], "reason": reason}));
        }
        self.settle(Duration::from_secs(90))?;
        let regional_now: BTreeSet<String> = self.regional_conclusions()?.into_keys().collect();
        let gone = self.ask_strategy()?.is_none();
        self.check(
            "strategy_withdrawn",
            &format!(
                "APAC's regional conclusion and the strategy are withdrawn (regions still concluding: {})",
                region_hits(&regional_now)
            ),
            regional_now == BTreeSet::from([format!("{ENTERPRISE}/emea")]) && gone,
        );
        let old = self.admin()?.get_memory(&strat_id)?;
        self.check(
            "history_kept",
            &format!("the withdrawn strategy is kept as history (status: {})", py_str(&old["status"])),
            old["status"] == "retracted",
        );
        let agent_id = "apac-procurement-2";
        let i = self.index(agent_id)?;
        let retracted_by: HashSet<String> = supplier_notes.iter().map(|m| str_of(&m["producer_id"])).collect();
        self.sc.write_observations(i, Some(&[recount()]), "recount", false)?;
        self.run_stage(&[agent_id.to_string()])?;
        let back = self.ask_strategy()?;
        let path = self.sc.agents[i].path();
        let recount_ids: Vec<String> = self
            .admin()?
            .list_memories(Some(&path), Some("agent"), 20)?
            .into_iter()
            .filter(|m| m["local_ref"] == format!("{agent_id}-recount-0").as_str())
            .map(|m| str_of(&m["memory_id"]))
            .collect();
        let both: BTreeSet<String> = [format!("{ENTERPRISE}/emea"), format!("{ENTERPRISE}/apac")].into();
        // evaluated in order and only as far as needed, like the Python `and` chain
        let ok = match &back {
            Some(b) => {
                b["memory_id"] != strat["memory_id"]
                    && !recount_ids.is_empty()
                    && strings(&b["metadata"]["roots"]).contains(&recount_ids[0])
                    && self.admin()?.get_memory(&strat_id)?["status"] == "retracted"
                    && self.regional_conclusions()?.into_keys().collect::<BTreeSet<_>>() == both
            }
            None => false,
        };
        let one_of = if retracted_by.contains(agent_id) { ", one of the agents that retracted" } else { "" };
        self.check(
            "strategy_restored",
            &format!(
                "a physical recount posted by {agent_id} (APAC procurement{one_of}) restores the regional conclusion \
                 and the strategy as new conclusions resting on the new note; the withdrawn ones stay in history"
            ),
            ok,
        );
        if let Some(back) = back {
            self.numbers.insert("time_to_restore_s".into(), json!(self.latency(&back)));
            let id = str_of(&back["memory_id"]);
            self.strategy = Some(back);
            self.regional = self.regional_conclusions()?;
            let g = self.admin()?.lineage(&id)?;
            self.rec.emit("lineage", json!({"viewer": "admin", "memory_id": id, "graph": g}));
        }
        Ok(())
    }

    fn plug(&mut self) -> Result<()> {
        let st = self.settle(Duration::from_secs(90))?; // nothing in flight: the log is complete before the kill
        self.watcher()?.pause();
        let result = self.pull_the_plug(&st);
        if let Some(w) = &self.watcher {
            w.resume();
        }
        result?;
        let w = self.watcher()?;
        w.poll_once(true)?;
        w.metric(true)
    }

    fn pull_the_plug(&mut self, st: &Value) -> Result<()> {
        let admin = self.admin()?.clone();
        let strat_id = self.strategy.as_ref().map(|s| str_of(&s["memory_id"]));
        let before: BTreeSet<String> = all_memories(&admin)?.keys().cloned().collect();
        let active = |admin: &MycelicClient| -> Result<BTreeSet<String>> {
            Ok(admin.list_memories(Some(ENTERPRISE), None, 500)?.iter().map(|m| str_of(&m["memory_id"])).collect())
        };
        let active_before = active(&admin)?;
        let g_before = strat_id.as_deref().map(|id| admin.lineage(id)).transpose()?;
        let transport = &st["checks"]["transport"];
        let total = if transport["last_seq"].as_i64().unwrap_or(0) != 0 {
            &transport["last_seq"]
        } else {
            &transport["stream_messages"]
        };
        let total_n = total.as_i64();
        let total = total.clone();
        self.driver.kill("mycelic")?;
        self.rec.emit("service", json!({"state": "killed", "detail": "SIGKILL: no graceful shutdown"}));
        self.driver.wipe_database()?;
        self.rec.emit(
            "service",
            json!({"state": "db_wiped", "detail": "the service's SQLite database files were deleted"}),
        );
        self.rec.emit(
            "service",
            json!({"state": "rebuilding", "detail": format!("replaying {} events from the JetStream log", py_str(&total))}),
        );
        let probe = MycelicClient::new(&self.driver.base_url(), &self.driver.admin_token())
            .with_retries(0)
            .with_timeout(Duration::from_secs(2));
        let stop = AtomicBool::new(false);
        let t0 = Instant::now();
        let rec = &self.rec;
        let driver = &mut self.driver;
        let (started, rebuild_s) = std::thread::scope(|s| {
            s.spawn(|| {
                let mut last = None;
                while !stop.load(Ordering::SeqCst) {
                    if let Ok(status) = probe.status() {
                        let applied = status["checks"]["consumer"]["last_applied_seq"].as_i64();
                        if let (Some(applied), Some(total)) = (applied, total_n)
                            && last != Some(applied)
                            && applied <= total
                        {
                            last = Some(applied);
                            rec.emit("rebuild_progress", json!({"applied": applied, "total": total}));
                        }
                    }
                    std::thread::sleep(Duration::from_millis(100));
                }
            });
            // returns once /ready answers 200: the replay is done
            let started = driver.start("mycelic");
            let rebuild_s = (t0.elapsed().as_secs_f64() * 100.0).round() / 100.0;
            stop.store(true, Ordering::SeqCst);
            (started, rebuild_s)
        });
        started?;
        self.rec.emit("rebuild_progress", json!({"applied": total, "total": total}));
        self.rec.emit(
            "service",
            json!({"state": "ready",
                   "detail": format!("ready {} s after the restart: {} logged events replayed into an empty database",
                                     py_float(rebuild_s), py_str(&total)),
                   "rebuild_s": rebuild_s, "rebuild_events": total}),
        );
        signals::checkpoint()?;
        let st = wait_idle(&admin, Duration::from_secs(180))?;
        let applied = st["checks"]["consumer"]["last_applied_seq"].as_i64().unwrap_or(0);
        // a fresh database does not know the rules yet, so the restarted service appends its rules file to the log
        // again (one rule.upserted event per rule) right after the replay: the log grows, the replay does not
        let appended = (applied - total_n.unwrap_or(0)).max(0);
        self.numbers.insert("rebuild_s".into(), json!(rebuild_s));
        self.numbers.insert("rebuild_events".into(), total.clone());
        self.numbers.insert("events_appended_on_restart".into(), json!(appended));
        let after: BTreeSet<String> = all_memories(&admin)?.keys().cloned().collect();
        let active_after = active(&admin)?;
        let g_after = strat_id.as_deref().map(|id| admin.lineage(id)).transpose()?;
        if let (Some(g), Some(id)) = (&g_after, &strat_id) {
            self.rec.emit("lineage", json!({"viewer": "admin", "memory_id": id, "graph": g}));
        }
        let same = match (&g_before, &g_after) {
            (Some(b), Some(a)) => a["roots"] == b["roots"] && edges(a) == edges(b) && a["layers"] == b["layers"],
            _ => false,
        };
        self.check(
            "rebuild_identical",
            &format!(
                "rebuilt {} active memories from {} events in {} s; the strategy's lineage is identical (same roots, \
                 same {} edges)",
                active_after.len(),
                py_str(&total),
                py_float(rebuild_s),
                g_after.as_ref().map_or(0, |g| edges(g).len())
            ),
            same && active_after == active_before && after == before,
        );
        let back = self.ask_strategy()?;
        self.check(
            "keys_survive_rebuild",
            "head office's agent asks again with the same key and gets the same strategy",
            back.as_ref().zip(strat_id.as_ref()).is_some_and(|(b, id)| b["memory_id"] == id.as_str()),
        );
        Ok(())
    }

    fn close(&mut self) -> Result<()> {
        let ran: Vec<String> = self.states.keys().cloned().collect();
        let (n, leaks) = self.private_leaks(&ran)?;
        self.check(
            "private_never_reached",
            &format!(
                "after everything, none of the {n} private notes of {} agents was ever sent to the service (the demo \
                 read them from each agent's own disk to show them)",
                ran.len()
            ),
            n >= self.sc.agents.len() && leaks.is_empty(),
        );
        let st = self.admin()?.status()?;
        self.rec.emit("metric", metric_payload(&st));
        let (passed, total) = self.rec.read(|tr| {
            let checks = tr["checks"].as_array().map(Vec::as_slice).unwrap_or_default();
            (checks.iter().filter(|c| c["ok"] == true).count(), checks.len())
        });
        let count = |k: &str| self.states.values().map(|s| s["counts"][k].as_i64().unwrap_or(0)).sum::<i64>();
        let teams: HashSet<String> = self.sc.agents.iter().map(LiveAgent::team_path).collect();
        let num = |k: &str| self.numbers.get(k).cloned().unwrap_or(Value::Null);
        let strategy = self.strategy.as_ref().map(|s| {
            let regions: Vec<String> = strings(&s["metadata"]["corroborated_units"]["supply_risk"]["region"])
                .iter()
                .map(|r| leaf(r).to_string())
                .collect();
            json!({"memory_id": s["memory_id"], "support": s["support"], "independent_teams": s["independent_teams"],
                   "confidence": s["confidence"], "regions": regions})
        });
        let summary = json!({
            "agents": self.sc.agents.len(), "agents_core": self.sc.by_role("core").len(),
            "agents_background": self.sc.by_role("background").len(), "teams": teams.len(),
            "notes_shared": count("shared"), "notes_private": count("local_only"),
            "memories_by_layer": st["stats"]["memories_by_layer"],
            "memories_by_status": st["stats"]["memories_by_status"],
            "lineage_edges": st["stats"]["lineage_edges"],
            "stream_messages": st["checks"]["transport"]["stream_messages"],
            "rebuild_s": num("rebuild_s"), "rebuild_events": num("rebuild_events"),
            "events_appended_on_restart": num("events_appended_on_restart"),
            "time_to_region_s": num("time_to_region_s"),
            "time_to_enterprise_s": num("time_to_enterprise_s"),
            "time_to_restore_s": num("time_to_restore_s"),
            "background_notes": num("background_notes"),
            "background_consolidations": num("background_consolidations"),
            "background_rule_conclusions": num("background_rule_conclusions"),
            "strategy": strategy,
            "checks_passed": passed, "checks_total": total,
        });
        self.rec.update(|tr| tr["summary"] = summary);
        Ok(())
    }

    /// Stop watching (the stack itself goes down with [`LiveDemo::down`]).
    pub fn shutdown(&mut self) {
        if let Some(mut w) = self.watcher.take() {
            w.stop();
        }
    }

    /// Tear the stack down (idempotent).
    pub fn down(&mut self) {
        self.driver.down();
    }
}

/// The demo's rules file for the process driver: the two rules of the story, verbatim from `rules.json`.
pub fn write_demo_rules(workdir: &Path) -> Result<std::path::PathBuf> {
    let source = crate::rules_file();
    let text = std::fs::read_to_string(&source).with_context(|| format!("reading {}", source.display()))?;
    let all: Value = serde_json::from_str(&text).with_context(|| format!("parsing {}", source.display()))?;
    let rules: Vec<Value> = all["rules"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|r| DEMO_RULES.iter().any(|id| r["rule_id"] == *id))
        .cloned()
        .collect();
    let path = workdir.join("demo-rules.json");
    std::fs::write(&path, serde_json::to_string_pretty(&json!({"rules": rules}))?)
        .with_context(|| format!("writing {}", path.display()))?;
    Ok(path)
}
