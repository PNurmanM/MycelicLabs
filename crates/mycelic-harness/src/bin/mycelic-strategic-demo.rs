//! Mycelic strategic-synthesis demo: a conclusion nobody in the organization could reach alone.
//!
//! ```text
//! mycelic-strategic-demo                     # nats-server + service as local processes
//! mycelic-strategic-demo --driver compose    # against `docker compose`
//! mycelic-strategic-demo --keep              # leave the deployment running at the end
//! ```
//!
//! Two regions of Northwind Robotics, EMEA and APAC, each independently run into the same trouble with the SD-9
//! servo drive: a port strike, slipping carrier ETAs, a supplier with two weeks of stock, committed customer orders.
//! Each region's agents share their notes; Mycelic composes a *regional* supply-risk conclusion in each region.  Head
//! office knows two other things: order intake is up 40% and Kessler is the only qualified source.  Nobody holds all
//! of it.  The rule `strategic_second_source` fires at the enterprise once supply risk is corroborated in at least
//! two regions: "qualify a second source".  Its lineage spans three layers in two derivation steps: enterprise
//! strategy <- regional conclusions <- agents' observations, resting on 8 of the 14 agents (the strongest note per
//! slot in each region plus the two head-office notes) in 8 teams across 3 regions.
//!
//! The demo then shows that strategy follows evidence (retract APAC's supplier notes: the regional conclusion and
//! the strategy are withdrawn; new evidence brings them back as a new version), that an agent sees its own region's
//! conclusion and its own team's notes while everything else in the lineage is redacted rather than dropped, and
//! that a rebuild from the event log reproduces the whole chain.  Every number printed is read from the live
//! deployment; every verdict is a check, and the exit code is 0 only when all of them held.

use anyhow::Result;
use clap::Parser;
use mycelic_harness::scenario::{ENTERPRISE, ENTITY};
use mycelic_harness::smoke::wait_idle;
use mycelic_harness::strategic::{
    HQ_DEPARTMENT, HQ_REGION, HQ_SUBSIDIARY, REGIONS, STRATEGIC_QUERY, StrategicScenario, TEAMS,
};
use mycelic_harness::{Driver, clip, leaf, make_driver, repo_root};
use mycelic_sdk::{MycelicClient, Note, QueryOptions};
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Duration;

const H: &str = "\x1b[1;36m";
const B: &str = "\x1b[1m";
const D: &str = "\x1b[2m";
const G: &str = "\x1b[32m";
const Y: &str = "\x1b[33m";
const R: &str = "\x1b[31m";
const X: &str = "\x1b[0m";

#[derive(Parser)]
#[command(
    name = "mycelic-strategic-demo",
    about = "Mycelic strategic synthesis: a conclusion nobody in the organization could reach alone"
)]
struct Args {
    #[arg(long, default_value = "process", value_parser = ["process", "compose"])]
    driver: String,
    #[arg(long, default_value_t = 2)]
    agents_per_team: usize,
    #[arg(long, default_value_t = 0.6)]
    pace: f64,
    #[arg(long)]
    keep: bool,
    #[arg(long)]
    no_build: bool,
    #[arg(long)]
    out: Option<PathBuf>,
}

fn head(text: &str) {
    println!("\n{H}=== {text}{X}");
}

fn say(text: &str, kind: &str) {
    let prefix = match kind {
        "ok" => format!("{G}  ✓{X}"),
        "warn" => format!("{Y}  !{X}"),
        "bad" => format!("{R}  ✗{X}"),
        _ => "   ".to_string(),
    };
    println!("{prefix} {text}");
}

fn s(v: &Value) -> String {
    v.as_str().map(String::from).unwrap_or_else(|| v.to_string())
}

fn tree(g: &Value, root: &str, indent: &str, seen: &mut BTreeSet<String>) -> Vec<String> {
    let n = &g["nodes"][root];
    let redacted = n["redacted"] == true;
    let who = if redacted {
        "redacted".to_string()
    } else if n["layer"] == "agent" {
        s(&n["producer_id"])
    } else {
        format!("{}{}", s(&n["operator"]), n["rule_id"].as_str().map(|r| format!(" {r}")).unwrap_or_default())
    };
    let text = if redacted { "[text withheld: outside your visibility]".to_string() } else { clip(&s(&n["text"]), 96) };
    let mut lines = vec![
        format!(
            "{indent}[{}] {}  ({who}, confidence {}, support {})",
            s(&n["layer"]),
            leaf(&s(&n["scope"])),
            n["confidence"],
            n["support"]
        ),
        format!("{indent}   {D}{text}{X}"),
    ];
    if !seen.insert(root.to_string()) {
        return vec![lines.remove(0), format!("{indent}   {D}(shown above){X}")];
    }
    let mut parents: Vec<String> =
        g["edges"].as_array().into_iter().flatten().filter(|e| e["child"] == root).map(|e| s(&e["parent"])).collect();
    parents.sort();
    for (i, pid) in parents.iter().enumerate() {
        let next = if i + 1 < parents.len() { "   │ " } else { "     " };
        lines.extend(tree(g, pid, &format!("{indent}{next}"), seen));
    }
    lines
}

fn mermaid(g: &Value) -> String {
    let mut out = vec!["graph BT".to_string()];
    for (mid, n) in g["nodes"].as_object().into_iter().flatten() {
        let text =
            if n["redacted"] == true { "redacted".to_string() } else { clip(&s(&n["text"]), 60).replace('"', "'") };
        out.push(format!("  {mid}[\"{}: {text}\"]", s(&n["layer"])));
    }
    for e in g["edges"].as_array().into_iter().flatten() {
        out.push(format!("  {} --> {}", s(&e["parent"]), s(&e["child"])));
    }
    out.join("\n") + "\n"
}

/// The strategic recommendation among the enterprise-level hits for the strategic question, if any.
fn strategic_answer(client: &MycelicClient) -> Result<Option<Value>> {
    let res = client.query(
        STRATEGIC_QUERY,
        &QueryOptions {
            scope: Some(ENTERPRISE.into()),
            min_layer: "enterprise".into(),
            k: 5,
            ..QueryOptions::default()
        },
    )?;
    Ok(res["results"]
        .as_array()
        .into_iter()
        .flatten()
        .map(|h| h["memory"].clone())
        .find(|m| m["rule_id"] == "strategic_second_source"))
}

fn regional(admin: &MycelicClient) -> Result<BTreeMap<String, Value>> {
    Ok(admin
        .list_memories(Some(ENTERPRISE), Some("region"), 50)?
        .into_iter()
        .filter(|m| m["rule_id"] == "regional_supply_risk")
        .map(|m| (s(&m["scope"]), m))
        .collect())
}

fn run(driver: &mut dyn Driver, args: &Args, out: &Path) -> Result<u8> {
    let pause = || std::thread::sleep(Duration::from_secs_f64(args.pace));
    let workdir = tempfile::Builder::new().prefix("mycelic-strategic-").tempdir()?;
    let mut checks: Vec<bool> = Vec::new();
    let mut verdict = |cond: bool, text: &str| {
        checks.push(cond);
        say(text, if cond { "ok" } else { "bad" });
    };

    head(&format!("Mycelic strategic synthesis — starting the deployment ({})", driver.name()));
    driver.up()?;
    let admin = MycelicClient::new(&driver.base_url(), &driver.admin_token());
    let h = admin.health()?;
    say(&format!("service {} at {}: {}", s(&h["version"]), driver.base_url(), s(&h["status"])), "ok");
    let rules: BTreeMap<String, Value> = admin.list_rules()?.into_iter().map(|r| (s(&r["rule_id"]), r)).collect();
    for rid in ["regional_supply_risk", "strategic_second_source"] {
        let r = &rules[rid];
        let mut line = format!("rule {rid}: at {}, needs {}", s(&r["target_layer"]), r["required_slots"]);
        if let Some(e) = r["emits_slot"].as_str() {
            line.push_str(&format!(", emits '{e}'"));
        }
        if r["min_units"].as_object().is_some_and(|m| !m.is_empty()) {
            line.push_str(&format!(", corroboration {}", r["min_units"]));
        }
        say(&line, "ok");
    }
    pause();

    head("The organization: two regions and head office");
    let mut sc = StrategicScenario::build(workdir.path(), args.agents_per_team)?;
    sc.register_all(&admin)?;
    sc.write_observations()?;
    println!("   {ENTERPRISE} (enterprise)");
    for (region, sub) in REGIONS {
        println!("   ├─ {region} / {sub}");
        for (team, dept) in TEAMS {
            let ids: Vec<&str> =
                sc.by_region(region).iter().filter(|a| a.team == team).map(|a| a.agent_id.as_str()).collect();
            println!("   │   {dept}/{team}: {}", ids.join(", "));
        }
    }
    println!(
        "   └─ {HQ_REGION} / {HQ_SUBSIDIARY}: {HQ_DEPARTMENT}/sourcing: hq-sourcing-1, {HQ_DEPARTMENT}/analytics: hq-analytics-1"
    );
    say(&format!("{} agents registered, each with its own key and its own local memory file", sc.agents.len()), "ok");
    pause();

    head("Agents observe (each shares one note and keeps one private)");
    let states = sc.run_agents(&driver.base_url())?;
    for a in &sc.agents {
        for line in
            std::fs::read_to_string(workdir.path().join(format!("{}.log", a.agent_id))).unwrap_or_default().lines()
        {
            if line.contains("shared") {
                println!("   {D}{line}{X}");
            }
        }
    }
    wait_idle(&admin, Duration::from_secs(90))?;
    pause();

    head("What each region concluded on its own");
    let regional_now = regional(&admin)?;
    for (scope, m) in &regional_now {
        println!(
            "   [{scope}] confidence {}, {} agents / {} teams",
            m["confidence"], m["support"], m["independent_teams"]
        );
        println!("      {}", clip(&s(&m["text"]), 150));
    }
    let want: BTreeSet<String> = REGIONS.iter().map(|(r, _)| format!("{ENTERPRISE}/{r}")).collect();
    verdict(
        regional_now.keys().cloned().collect::<BTreeSet<_>>() == want,
        "both regions reached a regional supply-risk conclusion, independently",
    );
    let emea_agent = MycelicClient::new(&driver.base_url(), &sc.agent("emea-field-sales-1").api_key);
    let res = emea_agent.query(
        "supply risk sd-9",
        &QueryOptions {
            scope: Some(ENTERPRISE.into()),
            min_layer: "region".into(),
            k: 10,
            include_lineage: false,
            ..QueryOptions::default()
        },
    )?;
    let seen: BTreeSet<String> = res["results"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|h| h["memory"]["layer"] == "region")
        .map(|h| s(&h["memory"]["scope"]))
        .collect();
    verdict(
        seen == BTreeSet::from([format!("{ENTERPRISE}/emea")]),
        &format!("an EMEA agent sees only its own region's conclusion: {seen:?}"),
    );
    pause();

    head("What head office knew");
    for a in sc.by_region(HQ_REGION) {
        for m in admin.list_memories(Some(&a.path), Some("agent"), 5)? {
            println!("   {:<16} slot {:<24} {}", a.agent_id, s(&m["slot"]), clip(&s(&m["text"]), 100));
        }
    }
    pause();

    head(&format!("Strategic synthesis: {STRATEGIC_QUERY:?} (asked at the enterprise by hq-sourcing-1)"));
    let hq = MycelicClient::new(&driver.base_url(), &sc.agent("hq-sourcing-1").api_key);
    let Some(mut strat) = strategic_answer(&hq)? else {
        verdict(false, "the enterprise holds a strategic recommendation");
        return Ok(1);
    };
    verdict(true, "the enterprise holds a strategic recommendation");
    println!(
        "   {B}[{} {}] confidence {}, support {} agents / {} teams{X}",
        s(&strat["layer"]),
        s(&strat["scope"]),
        strat["confidence"],
        strat["support"],
        strat["independent_teams"]
    );
    println!("   {}", s(&strat["text"]));
    let strat_id = s(&strat["memory_id"]);
    let g = admin.lineage(&strat_id)?;
    let regions: Vec<String> = strat["metadata"]["corroborated_units"]["supply_risk"]["region"]
        .as_array()
        .into_iter()
        .flatten()
        .map(|r| leaf(&s(r)).to_string())
        .collect();
    verdict(regions.len() >= 2, &format!("corroborated across regions: {regions:?}"));
    verdict(
        g["layers"] == json!(["agent", "region", "enterprise"]),
        &format!("lineage crosses layers {}: two derivation steps from agents' notes to strategy", g["layers"]),
    );
    let mut parents: Vec<String> = g["edges"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|e| e["child"] == json!(strat_id))
        .map(|e| s(&e["parent"]))
        .collect();
    parents.sort();
    let mut by_layer: BTreeMap<String, usize> = BTreeMap::new();
    for pid in &parents {
        *by_layer.entry(s(&g["nodes"][pid]["layer"])).or_default() += 1;
    }
    say(
        &format!(
            "direct evidence: {by_layer:?} ({} memories); raw observations at the roots: {}; agents: {}; teams: {}; reconstructable: {}",
            parents.len(),
            g["roots"].as_array().map(Vec::len).unwrap_or(0),
            g["contributing_agents"].as_array().map(Vec::len).unwrap_or(0),
            g["contributing_teams"].as_array().map(Vec::len).unwrap_or(0),
            g["evidence"]["reconstructable"]
        ),
        "ok",
    );
    for t in g["transformations"].as_array().into_iter().flatten() {
        let from: Vec<String> = t["from_layers"].as_array().into_iter().flatten().map(s).collect();
        let rule = t["rule_id"].as_str().map(|r| format!(" rule {r}")).unwrap_or_default();
        println!(
            "   {} → {:<10} {}{rule}  [{}]",
            from.join(" + "),
            s(&t["to_layer"]),
            s(&t["operator"]),
            leaf(&s(&t["scope"]))
        );
    }
    pause();

    head("Lineage graph (administrator view)");
    for line in tree(&g, &strat_id, "   ", &mut BTreeSet::new()) {
        println!("{line}");
    }
    std::fs::write(out.join("lineage.json"), serde_json::to_string_pretty(&g)?)?;
    std::fs::write(out.join("lineage.mmd"), mermaid(&g))?;
    say(&format!("written {} and {}", out.join("lineage.json").display(), out.join("lineage.mmd").display()), "ok");
    let emea_view = emea_agent.lineage(&strat_id)?;
    let mut redacted_by_region: BTreeMap<String, usize> = BTreeMap::new();
    for n in emea_view["nodes"].as_object().into_iter().flatten().map(|(_, n)| n) {
        if n["redacted"] == true {
            let scope = s(&n["scope"]);
            let r = scope.split('/').nth(1).map(String::from).unwrap_or(scope);
            *redacted_by_region.entry(r).or_default() += 1;
        }
    }
    let hq_note_ids: Vec<String> = sc
        .by_region(HQ_REGION)
        .iter()
        .flat_map(|a| admin.list_memories(Some(&a.path), Some("agent"), 5).unwrap_or_default())
        .map(|m| s(&m["memory_id"]))
        .collect();
    let apac_id = s(&regional_now[&format!("{ENTERPRISE}/apac")]["memory_id"]);
    let emea_id = s(&regional_now[&format!("{ENTERPRISE}/emea")]["memory_id"]);
    let total_redacted: usize = redacted_by_region.values().sum();
    let breakdown: Vec<String> = redacted_by_region.iter().map(|(k, v)| format!("{v} {k}")).collect();
    verdict(
        emea_view["redacted_contributions"].as_u64() == Some(total_redacted as u64)
            && total_redacted > 0
            && emea_view["nodes"][&apac_id]["text"].is_null()
            && emea_view["nodes"][&emea_id]["redacted"] != true
            && hq_note_ids.iter().all(|mid| emea_view["nodes"].get(mid).is_none_or(|n| n["redacted"] == true)),
        &format!(
            "the same lineage for an EMEA field-sales agent: {} of {} contributions redacted ({}: APAC's conclusion and notes, HQ's notes, other EMEA teams' notes); EMEA's conclusion and its own note readable, shape intact",
            emea_view["redacted_contributions"],
            emea_view["nodes"].as_object().map(|n| n.len()).unwrap_or(0),
            breakdown.join(", ")
        ),
    );
    pause();

    head("Strategy follows evidence: APAC retracts its supplier notes");
    let apac_supplier: Vec<Value> = admin
        .list_memories(Some(&format!("{ENTERPRISE}/apac")), Some("agent"), 200)?
        .into_iter()
        .filter(|m| m["slot"] == "supplier_buffer_low")
        .collect();
    for m in &apac_supplier {
        let producer = s(&m["producer_id"]);
        MycelicClient::new(&driver.base_url(), &sc.agent(&producer).api_key)
            .retract(&s(&m["memory_id"]), "inventory was counted twice")?;
        say(&format!("{producer} retracted: {}", clip(&s(&m["text"]), 80)), "warn");
    }
    wait_idle(&admin, Duration::from_secs(90))?;
    let regional_after: BTreeSet<String> = regional(&admin)?.into_keys().collect();
    let active_strategies = admin
        .list_memories(Some(ENTERPRISE), Some("enterprise"), 50)?
        .into_iter()
        .filter(|m| m["rule_id"] == "strategic_second_source")
        .count();
    let gone = strategic_answer(&hq)?.is_none() && active_strategies == 0;
    let left: Vec<&str> = regional_after.iter().map(|r| leaf(r)).collect();
    verdict(
        regional_after == BTreeSet::from([format!("{ENTERPRISE}/emea")]) && gone,
        &format!(
            "APAC's regional conclusion is withdrawn and so is the strategy (regions with a conclusion now: {left:?})"
        ),
    );
    let strat_now = admin.get_memory(&strat_id)?;
    verdict(
        strat_now["status"] == "retracted",
        &format!("the withdrawn strategy is kept as history: status {}", s(&strat_now["status"])),
    );
    MycelicClient::new(&driver.base_url(), &sc.agent("apac-procurement-2").api_key).remember(
        &Note {
            text: "Physical count confirms roughly two weeks of SD-9 inventory in the APAC warehouse.".into(),
            topic: Some("supply:sd-9/supplier".into()),
            slot: Some("supplier_buffer_low".into()),
            entity: Some(ENTITY.into()),
            confidence: 0.9,
            ..Note::default()
        },
        "team",
        Some("apac-recount-1"),
        None,
        None,
    )?;
    wait_idle(&admin, Duration::from_secs(90))?;
    let back = strategic_answer(&hq)?;
    verdict(
        back.as_ref().is_some_and(|b| b["memory_id"] != json!(strat_id)),
        "a fresh count from another APAC agent restores the regional conclusion and the strategy, as a new version with new evidence",
    );
    if let Some(b) = back {
        strat = b;
    }
    let strat_id = s(&strat["memory_id"]);
    pause();

    head("Failure: the service is killed and its database deleted — rebuild from the event log");
    let ids = |admin: &MycelicClient| -> Result<BTreeSet<String>> {
        Ok(admin.list_memories(Some(ENTERPRISE), None, 500)?.iter().map(|m| s(&m["memory_id"])).collect())
    };
    let before = ids(&admin)?;
    let lineage_before = admin.lineage(&strat_id)?;
    driver.kill("mycelic")?;
    driver.wipe_database()?;
    driver.start("mycelic")?;
    let st = wait_idle(&admin, Duration::from_secs(180))?;
    let after = ids(&admin)?;
    let g2 = admin.lineage(&strat_id)?;
    let edge_set = |g: &Value| -> BTreeSet<(String, String)> {
        g["edges"].as_array().into_iter().flatten().map(|e| (s(&e["child"]), s(&e["parent"]))).collect()
    };
    verdict(
        after == before
            && g2["roots"] == lineage_before["roots"]
            && edge_set(&g2) == edge_set(&lineage_before)
            && g2["layers"] == lineage_before["layers"],
        &format!(
            "rebuilt {} active memories from {} events; the strategic lineage is identical",
            after.len(),
            st["checks"]["transport"]["stream_messages"]
        ),
    );
    pause();

    head("Summary");
    let by_layer_counts = st["stats"]["memories_by_layer"].clone();
    let local_total: i64 = states.values().map(|v| v["counts"]["local"].as_i64().unwrap_or(0)).sum();
    let private_total: i64 = states.values().map(|v| v["counts"]["local_only"].as_i64().unwrap_or(0)).sum();
    println!(
        "   agents: {} in {} regions; notes on agents' disks: {local_total}, private: {private_total}",
        sc.agents.len(),
        REGIONS.len() + 1
    );
    let layers: Vec<String> =
        by_layer_counts.as_object().into_iter().flatten().map(|(k, v)| format!("{k}={v}")).collect();
    println!("   organizational memory by layer: {}", layers.join(", "));
    let agents: serde_json::Map<String, Value> = states.iter().map(|(k, v)| (k.clone(), v["counts"].clone())).collect();
    std::fs::write(
        out.join("summary.json"),
        serde_json::to_string_pretty(
            &json!({"strategy": strat, "memories_by_layer": by_layer_counts, "checks": checks, "agents": agents}),
        )?,
    )?;
    let ok = checks.iter().all(|c| *c);
    say(
        &format!(
            "{}/{} checks passed; summary in {}",
            checks.iter().filter(|c| **c).count(),
            checks.len(),
            out.join("summary.json").display()
        ),
        if ok { "ok" } else { "bad" },
    );
    if args.keep && ok {
        println!(
            "\n   Deployment left running at {}\n   MYCELIC_ADMIN_TOKEN={}\n   MYCELIC_API_KEY={}   # hq-sourcing-1",
            driver.base_url(),
            driver.admin_token(),
            sc.agent("hq-sourcing-1").api_key
        );
        println!(
            "   try:  mycelic query \"{STRATEGIC_QUERY}\" --scope northwind --min-layer enterprise --lineage\n   press Ctrl-C to stop"
        );
        loop {
            std::thread::sleep(Duration::from_secs(3600));
        }
    }
    Ok(if ok { 0 } else { 1 })
}

fn main() -> ExitCode {
    let args = Args::parse();
    let out = args.out.clone().unwrap_or_else(|| repo_root().join("demo/results/mycelic-strategic"));
    if let Err(e) = std::fs::create_dir_all(&out) {
        eprintln!("error: cannot create {}: {e}", out.display());
        return ExitCode::from(2);
    }
    let mut driver = match make_driver(&args.driver, "mycelic-strategic", None, !args.no_build) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("error: {e:#}");
            return ExitCode::from(2);
        }
    };
    let code = match run(driver.as_mut(), &args, &out) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("\nERROR: {e:#}\n--- mycelic log tail ---\n{}", driver.logs("mycelic", 40));
            2
        }
    };
    driver.down();
    ExitCode::from(code)
}
