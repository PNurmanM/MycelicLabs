//! The production smoke test: the canonical deployment demonstration, every step asserted.
//!
//! 1. the deployment is healthy and ready, the rule file is loaded;
//! 2. agents from three teams in two departments connect as separate processes, keep private notes locally and
//!    share partially overlapping observations;
//! 3. Mycelic consolidates the logistics observations into a team memory and composes the enterprise-level
//!    conclusion from three teams' evidence;
//! 4. a query from a higher organizational layer returns that conclusion, with lineage that names the contributing
//!    teams and layers, is reconstructable, and redacts other teams' raw notes for a non-admin caller;
//! 5. SIGKILL of the service, restart: the same answer and the same lineage;
//! 6. the broker is killed, an agent keeps writing (outbox), the broker returns, the write is applied and changes
//!    the team memory as expected;
//! 7. the service is killed and its database deleted; on restart it rebuilds itself from the JetStream log: the
//!    same active memory ids, the agents' keys still work, the rule is back;
//! 8. an agent process restarts with its local memory: nothing is duplicated, its private note is still local;
//! 9. metrics report what happened (memories by layer, replayed events, recoveries).

use crate::Driver;
use crate::scenario::{ENTERPRISE, ENTITY, QUERY, Scenario};
use anyhow::{Result, bail};
use mycelic_sdk::{MycelicClient, Note, QueryOptions};
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::time::{Duration, Instant};

/// A failed smoke step (exit code 1), as opposed to an error running the test (exit code 2).
#[derive(Debug)]
pub struct SmokeFailure(pub String);

impl std::fmt::Display for SmokeFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for SmokeFailure {}

pub fn check(cond: bool, msg: impl Into<String>) -> Result<()> {
    if cond { Ok(()) } else { Err(SmokeFailure(msg.into()).into()) }
}

fn pending(st: &Value) -> i64 {
    let c = &st["checks"];
    c["publisher"]["outbox_pending"].as_i64().unwrap_or(0) + c["consumer"]["pending"].as_i64().unwrap_or(0)
}

/// Poll `GET /admin/status` until the outbox is empty, the consumer has nothing pending, no replay runs and the
/// broker is connected; checked twice, half a second apart, so a just-published event is not missed.
pub fn wait_idle(admin: &MycelicClient, timeout: Duration) -> Result<Value> {
    let deadline = Instant::now() + timeout;
    let mut last = Value::Null;
    while Instant::now() < deadline {
        let Ok(st) = admin.status() else {
            std::thread::sleep(Duration::from_millis(500));
            continue;
        };
        last = st.clone();
        let c = &st["checks"];
        if pending(&st) == 0
            && c["consumer"]["replaying_to_seq"].is_null()
            && c["transport"]["connected"].as_bool().unwrap_or(false)
        {
            std::thread::sleep(Duration::from_millis(500));
            let st = admin.status()?;
            if pending(&st) == 0 {
                return Ok(st);
            }
        }
        std::thread::sleep(Duration::from_millis(500));
    }
    let checks: String = last["checks"].to_string().chars().take(800).collect();
    Err(SmokeFailure(format!("deployment did not become idle: {checks}")).into())
}

pub fn active_snapshot(admin: &MycelicClient) -> Result<BTreeMap<String, Value>> {
    Ok(admin
        .list_memories(Some(ENTERPRISE), None, 500)?
        .into_iter()
        .map(|m| {
            (
                m["memory_id"].as_str().unwrap_or_default().to_string(),
                json!([m["layer"], m["status"], m["support"], m["independent_teams"], m["confidence"]]),
            )
        })
        .collect())
}

fn roots(lineage: &Value) -> Vec<String> {
    let mut r: Vec<String> =
        lineage["roots"].as_array().into_iter().flatten().filter_map(|v| v.as_str().map(String::from)).collect();
    r.sort();
    r
}

fn transport_team(admin: &MycelicClient) -> Result<Value> {
    admin
        .list_memories(Some(ENTERPRISE), Some("team"), 100)?
        .into_iter()
        .find(|m| m["topic"] == "supply:sd-9/transport")
        .ok_or_else(|| SmokeFailure("no logistics team memory".into()).into())
}

/// Run the nine steps; the JSON report on success.
pub fn run(driver: &mut dyn Driver, agents_per_team: usize, workdir: &Path, log: &dyn Fn(&str)) -> Result<Value> {
    let t_start = Instant::now();
    let mut steps: Vec<Value> = Vec::new();
    let mut step = |name: &str, detail: Value, log: &dyn Fn(&str)| {
        let mut entry = json!({"step": name, "t": (t_start.elapsed().as_secs_f64() * 100.0).round() / 100.0});
        if let Value::Object(d) = &detail {
            for (k, v) in d {
                entry[k] = v.clone();
            }
        }
        let text: String = if detail.as_object().is_some_and(|d| !d.is_empty()) {
            format!("  {}", detail).chars().take(302).collect()
        } else {
            String::new()
        };
        log(&format!("  ok  {name}{text}"));
        steps.push(entry);
    };

    log(&format!("== 1. deployment up ({}) -> {}", driver.name(), driver.base_url()));
    driver.up()?;
    let admin = MycelicClient::new(&driver.base_url(), &driver.admin_token());
    let health = admin.health()?;
    check(health["status"] == "ok" || health["status"] == "degraded", format!("unexpected health {health}"))?;
    let rules = admin.list_rules()?;
    check(rules.iter().any(|r| r["rule_id"] == "component_supply_risk"), "rule file not loaded")?;
    step(
        "deployment ready",
        json!({"health": health["status"], "rules": rules.iter().map(|r| r["rule_id"].clone()).collect::<Vec<_>>()}),
        log,
    );

    log(&format!("== 2. register {} agents in 3 teams / 2 departments and run them as processes", 3 * agents_per_team));
    let mut sc = Scenario::build(workdir, agents_per_team)?;
    sc.register_all(&admin)?;
    sc.write_observations()?;
    let states = sc.run_agents(&driver.base_url(), None)?;
    let count = |k: &str| states.values().map(|s| s["counts"][k].as_i64().unwrap_or(0)).sum::<i64>();
    let (shared, local_only) = (count("shared"), count("local_only"));
    check(local_only == sc.agents.len() as i64, "every agent should have kept exactly one private note")?;
    check(shared >= sc.agents.len() as i64, "every agent should have shared at least one observation")?;
    step("agents ran", json!({"agents": sc.agents.len(), "shared": shared, "local_only": local_only}), log);

    log("== 3. aggregation");
    let st = wait_idle(&admin, Duration::from_secs(90))?;
    let by_layer = st["stats"]["memories_by_layer"].clone();
    check(by_layer["team"].as_i64().unwrap_or(0) >= 1, format!("no team memory: {by_layer}"))?;
    check(by_layer["enterprise"].as_i64().unwrap_or(0) >= 1, format!("no enterprise conclusion: {by_layer}"))?;
    step(
        "aggregated",
        json!({"memories_by_layer": by_layer, "stream": st["checks"]["transport"]["stream_messages"]}),
        log,
    );

    log(&format!("== 4. query from the enterprise layer: {QUERY:?}"));
    let sales = sc.by_team("field-sales").last().copied().cloned().expect("a sales agent");
    let agent_client = MycelicClient::new(&driver.base_url(), &sales.api_key);
    let enterprise =
        QueryOptions { scope: Some(ENTERPRISE.into()), min_layer: "enterprise".into(), ..QueryOptions::default() };
    let res = agent_client.query(QUERY, &enterprise)?;
    let ans = res["answer"].clone();
    check(!ans.is_null(), "no answer at enterprise scope")?;
    check(
        ans["layer"] == "enterprise" && ans["rule_id"] == "component_supply_risk",
        format!("unexpected answer {ans}"),
    )?;
    check(
        ans["text"].as_str().unwrap_or("").to_lowercase().contains(ENTITY),
        "conclusion does not mention the entity",
    )?;
    let lin = res["lineage"].clone();
    check(lin["evidence"]["reconstructable"] == true, format!("lineage not reconstructable: {}", lin["evidence"]))?;
    check(
        lin["contributing_teams"].as_array().map(Vec::len) == Some(3),
        format!("expected 3 teams, got {}", lin["contributing_teams"]),
    )?;
    let layers = lin["layers"].as_array().cloned().unwrap_or_default();
    check(
        layers.first() == Some(&json!("agent")) && layers.last() == Some(&json!("enterprise")),
        format!("layers {}", lin["layers"]),
    )?;
    check(
        lin["redacted_contributions"].as_i64().unwrap_or(0) >= 2,
        "other teams' raw observations must be redacted for a sales agent",
    )?;
    check(
        lin["nodes"].as_object().into_iter().flatten().all(|(_, n)| n["redacted"] != true || n["text"].is_null()),
        "redacted nodes must carry no text",
    )?;
    let memory_id = ans["memory_id"].as_str().unwrap_or_default().to_string();
    let admin_lin = admin.lineage(&memory_id)?;
    check(
        admin_lin["redacted_contributions"] == 0
            && admin_lin["contributing_agents"].as_array().map(Vec::len).unwrap_or(0) >= 3,
        "admin sees every contributor",
    )?;
    // a raw note of another team is invisible to the sales agent
    let log_root = roots(&admin_lin)
        .into_iter()
        .find(|r| {
            let scope = admin_lin["nodes"][r]["scope"].as_str().unwrap_or("");
            scope.ends_with("logistics") || scope.contains("/logistics/")
        })
        .ok_or_else(|| SmokeFailure("no logistics root in the lineage".into()))?;
    match agent_client.get_memory(&log_root) {
        Ok(_) => check(false, "sales agent could read a logistics raw note")?,
        Err(e) => check(e.status == 404, format!("expected 404, got {}", e.status))?,
    }
    let answer_before = json!({"memory_id": memory_id, "roots": roots(&lin), "support": ans["support"], "teams": ans["independent_teams"], "confidence": ans["confidence"]});
    let snap_before = active_snapshot(&admin)?;
    let text: String = ans["text"].as_str().unwrap_or("").chars().take(200).collect();
    step(
        "query + lineage",
        json!({"answer": answer_before, "redacted": lin["redacted_contributions"], "layers": lin["layers"],
               "agents": admin_lin["contributing_agents"].as_array().map(Vec::len), "text": text}),
        log,
    );

    log("== 5. kill the service (SIGKILL) and restart it");
    driver.kill("mycelic")?;
    driver.start("mycelic")?;
    let st = wait_idle(&admin, Duration::from_secs(90))?;
    let res2 = agent_client.query(QUERY, &enterprise)?;
    check(res2["answer"]["memory_id"] == answer_before["memory_id"], "answer changed after a service restart")?;
    check(json!(roots(&res2["lineage"])) == answer_before["roots"], "lineage changed after a service restart")?;
    check(active_snapshot(&admin)? == snap_before, "active memories changed after a service restart")?;
    step("service restart survived", json!({"recoveries": st["checks"]["consumer"]}), log);

    log("== 6. kill the broker, keep writing, bring it back");
    let team_before = transport_team(&admin)?;
    driver.kill("nats")?;
    std::thread::sleep(Duration::from_secs(1));
    let writer = sc.by_team("logistics").last().copied().cloned().expect("a logistics agent");
    let wclient = MycelicClient::new(&driver.base_url(), &writer.api_key).with_retries(0);
    let late = wclient.remember(
        &Note {
            text: "Antwerp reroute for SD-9 containers adds 9 days and is not yet approved.".into(),
            topic: Some("supply:sd-9/transport".into()),
            slot: Some("transport_disruption".into()),
            entity: Some(ENTITY.into()),
            confidence: 0.55,
            ..Note::default()
        },
        "team",
        Some("late-note-1"),
        None,
        None,
    )?;
    // unauthenticated: the minimal public view
    let public = MycelicClient::new(&driver.base_url(), "").health()?;
    let st_out = admin.status()?;
    check(
        st_out["checks"]["publisher"]["outbox_pending"].as_i64().unwrap_or(0) >= 1,
        "write during the outage must sit in the outbox",
    )?;
    check(
        public["status"] == "degraded" && public["transport_connected"] == false,
        format!("expected degraded health, got {public}"),
    )?;
    driver.start("nats")?;
    let st = wait_idle(&admin, Duration::from_secs(120))?;
    let late_mem = admin.get_memory(late["memory_id"].as_str().unwrap_or_default())?;
    check(!late_mem["applied_at"].is_null(), "the outage-time write was never applied")?;
    let team_after = transport_team(&admin)?;
    check(
        team_after["memory_id"] != team_before["memory_id"],
        "the team memory must be re-derived with the new evidence",
    )?;
    check(
        team_after["metadata"]["parent_count"].as_i64()
            == team_before["metadata"]["parent_count"].as_i64().map(|n| n + 1),
        "new evidence must appear in the lineage",
    )?;
    step(
        "broker outage absorbed",
        json!({"outbox_during_outage": st_out["checks"]["publisher"]["outbox_pending"], "team_support_before": team_before["support"],
               "team_support_after": team_after["support"], "reconnects": st["checks"]["transport"]["reconnects"]}),
        log,
    );
    let snap_before_wipe = active_snapshot(&admin)?;
    let answer_before_wipe = agent_client.query(QUERY, &enterprise)?["answer"]["memory_id"].clone();

    log("== 7. kill the service, delete its database, restart: rebuild from the event log");
    driver.kill("mycelic")?;
    driver.wipe_database()?;
    driver.start("mycelic")?;
    let st = wait_idle(&admin, Duration::from_secs(180))?;
    check(st["checks"]["consumer"]["replaying_to_seq"].is_null(), "replay did not finish")?;
    let snap_after = active_snapshot(&admin)?;
    if snap_after != snap_before_wipe {
        let a: BTreeSet<&String> = snap_after.keys().collect();
        let b: BTreeSet<&String> = snap_before_wipe.keys().collect();
        bail!(SmokeFailure(format!("rebuilt state differs: {:?}", a.symmetric_difference(&b).collect::<Vec<_>>())));
    }
    check(admin.list_agents(Some(ENTERPRISE))?.len() == sc.agents.len(), "agents were not restored from the log")?;
    check(admin.list_rules()?.iter().any(|r| r["rule_id"] == "component_supply_risk"), "rules were not restored")?;
    // same agent key as before the wipe
    let res3 = agent_client.query(QUERY, &enterprise)?;
    check(res3["answer"]["memory_id"] == answer_before_wipe, "answer differs after the rebuild")?;
    check(res3["lineage"]["evidence"]["reconstructable"] == true, "lineage not reconstructable after the rebuild")?;
    step(
        "database rebuilt from JetStream",
        json!({"active_memories": snap_after.len(), "agents": sc.agents.len()}),
        log,
    );

    log("== 8. restart an agent process with its local memory");
    let one = sc.by_team("procurement")[0].clone();
    let before_state = states[&one.agent_id].clone();
    let after_state = sc.run_agents(&driver.base_url(), Some(&[&one]))?[&one.agent_id].clone();
    check(
        after_state["counts"] == before_state["counts"],
        format!("agent restart changed local counts {} -> {}", before_state["counts"], after_state["counts"]),
    )?;
    check(after_state["shared"] == before_state["shared"], "agent restart re-sent or duplicated memories")?;
    check(
        after_state["local_only"].as_array().is_some_and(|a| !a.is_empty())
            && after_state["local_only"] == before_state["local_only"],
        "private note lost or shared",
    )?;
    wait_idle(&admin, Duration::from_secs(90))?;
    check(active_snapshot(&admin)? == snap_after, "an agent restart must not change organizational memory")?;
    step("agent restart idempotent", json!({"counts": after_state["counts"]}), log);

    log("== 9. metrics");
    let text = metrics_text(&driver.base_url(), &driver.admin_token())?;
    check(text.contains("mycelic_memories{layer=\"enterprise\"}"), "metrics lack memories by layer")?;
    check(
        text.contains("mycelic_replay_events_total") && text.contains("mycelic_recovery_total"),
        "metrics lack replay/recovery counters",
    )?;
    let replayed = text.lines().find(|l| l.starts_with("mycelic_replay_events_total")).map(String::from);
    step("metrics present", json!({"replay_counter": replayed}), log);

    Ok(json!({
        "driver": driver.name(), "agents_per_team": agents_per_team, "steps": steps, "result": "PASS",
        "seconds": (t_start.elapsed().as_secs_f64() * 10.0).round() / 10.0,
        "final_answer": res3["answer"], "memories_by_layer": st["stats"]["memories_by_layer"],
    }))
}

pub fn metrics_text(base_url: &str, token: &str) -> Result<String> {
    let agent: ureq::Agent = ureq::Agent::config_builder().timeout_global(Some(Duration::from_secs(5))).build().into();
    Ok(agent
        .get(format!("{base_url}/metrics"))
        .header("Authorization", format!("Bearer {token}"))
        .call()?
        .body_mut()
        .read_to_string()?)
}
