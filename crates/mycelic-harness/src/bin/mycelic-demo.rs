//! Mycelic end-to-end demo: a small organization discovers a supply risk that no single agent could see.
//!
//! ```text
//! mycelic-demo                       # nats-server + service as local processes (needs nats-server)
//! mycelic-demo --driver compose      # against `docker compose` (builds the image)
//! mycelic-demo --keep                # leave everything running at the end and print how to connect
//! ```
//!
//! The story: Northwind Robotics builds the RX-4 arm around the SD-9 servo drive.  Logistics agents notice a port
//! strike and slipping carrier ETAs.  A procurement agent hears the supplier has two weeks of stock left.  A sales
//! agent books a large November order.  Each note alone is routine.  Mycelic propagates the shared ones through the
//! organization, consolidates the logistics team's observations, and composes an enterprise-level supply-risk
//! conclusion whose lineage names every contributing observation, agent, team and layer.  Then the deployment is
//! broken on purpose: the service is killed, the broker is killed, the service's database is deleted.  Everything
//! comes back from the JetStream event log.  Every number printed is read from the live deployment.

use anyhow::Result;
use clap::Parser;
use mycelic_harness::scenario::{ENTERPRISE, ENTITY, QUERY, REGION, SUBSIDIARY, Scenario, TEAMS};
use mycelic_harness::smoke::wait_idle;
use mycelic_harness::{Driver, clip, leaf, make_driver, repo_root};
use mycelic_sdk::{MycelicClient, Note, QueryOptions};
use serde_json::{Value, json};
use std::path::PathBuf;
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
    name = "mycelic-demo",
    about = "Mycelic end-to-end demo: a supply risk no single agent could see, then the deployment broken on purpose"
)]
struct Args {
    #[arg(long, default_value = "process", value_parser = ["process", "compose"])]
    driver: String,
    #[arg(long, default_value_t = 2)]
    agents_per_team: usize,
    /// seconds to pause between narrated steps
    #[arg(long, default_value_t = 0.6)]
    pace: f64,
    /// leave the deployment running at the end
    #[arg(long)]
    keep: bool,
    /// compose: reuse the existing image
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

/// The lineage graph as an indented tree, parents under children.
fn lineage_tree(g: &Value, root: &str, indent: &str) -> Vec<String> {
    let n = &g["nodes"][root];
    let redacted = n["redacted"] == true;
    let who = if redacted {
        "redacted".to_string()
    } else if n["layer"] == "agent" {
        s(&n["producer_id"])
    } else {
        s(&n["operator"])
    };
    let label = format!(
        "[{}] {}  ({who}, confidence {}, support {})",
        s(&n["layer"]),
        leaf(&s(&n["scope"])),
        n["confidence"],
        n["support"]
    );
    let text = if redacted { "[text withheld: outside your visibility]".to_string() } else { clip(&s(&n["text"]), 90) };
    let mut lines = vec![format!("{indent}{label}"), format!("{indent}   {D}{text}{X}")];
    let mut parents: Vec<String> =
        g["edges"].as_array().into_iter().flatten().filter(|e| e["child"] == root).map(|e| s(&e["parent"])).collect();
    parents.sort();
    for (i, pid) in parents.iter().enumerate() {
        let next = if i + 1 < parents.len() { "   │ " } else { "     " };
        lines.extend(lineage_tree(g, pid, &format!("{indent}{next}")));
    }
    lines
}

/// The lineage graph as Mermaid (`graph BT`).
pub fn mermaid(g: &Value) -> String {
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

fn team_memory(admin: &MycelicClient) -> Result<Value> {
    admin
        .list_memories(Some(ENTERPRISE), Some("team"), 50)?
        .into_iter()
        .find(|m| m["topic"] == "supply:sd-9/transport")
        .ok_or_else(|| anyhow::anyhow!("no logistics team memory"))
}

fn run(driver: &mut dyn Driver, args: &Args, out: &std::path::Path) -> Result<u8> {
    let pause = || std::thread::sleep(Duration::from_secs_f64(args.pace));
    let workdir = tempfile::Builder::new().prefix("mycelic-demo-agents-").tempdir()?;
    head(&format!("Mycelic demo — starting the deployment ({})", driver.name()));
    driver.up()?;
    let admin = MycelicClient::new(&driver.base_url(), &driver.admin_token());
    let h = admin.status()?;
    say(
        &format!(
            "service {} at {}: {}, transport connected={}",
            s(&h["version"]),
            driver.base_url(),
            s(&h["status"]),
            h["checks"]["transport"]["connected"]
        ),
        "ok",
    );
    let rules: Vec<String> = admin.list_rules()?.iter().map(|r| s(&r["rule_id"])).collect();
    say(&format!("rules loaded: {rules:?}"), "ok");
    pause();

    head("The organization");
    let mut sc = Scenario::build(workdir.path(), args.agents_per_team)?;
    sc.register_all(&admin)?;
    sc.write_observations()?;
    println!("   {ENTERPRISE} (enterprise)\n   └─ {REGION} (region)\n      └─ {SUBSIDIARY} (subsidiary)");
    let mut depts: Vec<&str> = TEAMS.iter().map(|(_, d)| *d).collect();
    depts.sort();
    depts.dedup();
    for dept in depts {
        println!("         ├─ {dept} (department)");
        for (team, d) in TEAMS {
            if d == dept {
                let ids: Vec<&str> = sc.by_team(team).iter().map(|a| a.agent_id.as_str()).collect();
                println!("         │   └─ {team} (team): {}", ids.join(", "));
            }
        }
    }
    say(
        &format!("{} agents registered, each with its own API key and its own local memory file", sc.agents.len()),
        "ok",
    );
    pause();

    head("Agents observe — each keeps notes locally and shares only some of them");
    let states = sc.run_agents(&driver.base_url(), None)?;
    for a in &sc.agents {
        for line in
            std::fs::read_to_string(workdir.path().join(format!("{}.log", a.agent_id))).unwrap_or_default().lines()
        {
            if line.contains("shared") || line.contains("locally only") {
                println!("   {D}{line}{X}");
            }
        }
    }
    let st = wait_idle(&admin, Duration::from_secs(90))?;
    pause();

    head("What each agent knows on its own (nothing here is a conclusion)");
    for a in &sc.agents {
        let c = &states[&a.agent_id]["counts"];
        println!(
            "   {:<14} local notes: {}   shared: {}   stayed private: {}",
            a.agent_id, c["local"], c["shared"], c["local_only"]
        );
    }
    pause();

    head("What Mycelic concluded");
    let asker = sc.by_team("field-sales").last().copied().cloned().expect("a sales agent");
    let client = MycelicClient::new(&driver.base_url(), &asker.api_key);
    let enterprise =
        QueryOptions { scope: Some(ENTERPRISE.into()), min_layer: "enterprise".into(), ..QueryOptions::default() };
    let res = client.query(QUERY, &enterprise)?;
    let ans = res["answer"].clone();
    if ans.is_null() {
        say("no enterprise-level answer — aggregation did not produce the conclusion", "bad");
        return Ok(1);
    }
    println!("   {B}Q ({}, asking at the enterprise layer):{X} {QUERY}", asker.agent_id);
    println!(
        "   {B}A [{} {}] confidence {}, support {} agents / {} teams{X}",
        s(&ans["layer"]),
        s(&ans["scope"]),
        ans["confidence"],
        ans["support"],
        ans["independent_teams"]
    );
    println!("   {}", s(&ans["text"]));
    let g = res["lineage"].clone();
    say(
        &format!(
            "lineage: reconstructable={}, roots={}, layers={}, first observed {}, produced {}",
            g["evidence"]["reconstructable"],
            g["roots"].as_array().map(Vec::len).unwrap_or(0),
            g["layers"],
            s(&g["timeline"]["first_observed_at"]),
            s(&g["timeline"]["produced_at"])
        ),
        "ok",
    );
    pause();

    head("Supporting observations and independent sources (as the sales agent sees them)");
    for rid in g["roots"].as_array().into_iter().flatten() {
        let n = &g["nodes"][s(rid)];
        let redacted = n["redacted"] == true;
        let scope = s(&n["scope"]);
        let src = if redacted {
            format!("[redacted agent in {}]", leaf(&scope))
        } else {
            let team = scope.rsplit('/').nth(1).unwrap_or("");
            format!("{} ({team})", s(&n["producer_id"]))
        };
        let text =
            if redacted { "[text withheld: another team's raw note]".to_string() } else { clip(&s(&n["text"]), 100) };
        println!("   • slot {:<22} {src:<38} conf {}  {D}{text}{X}", s(&n["slot"]), n["confidence"]);
    }
    let fr = &g["support"]["fragility"];
    say(
        &format!(
            "independent failure domains (teams): {}, candidate observations: {}, minimal cut: {} team(s) would have to be wrong for the conclusion to lose support",
            fr["unique_failure_domain_count"], fr["unique_lineage_root_count"], fr["minimal_failure_domain_cut"]
        ),
        "ok",
    );
    say(
        &format!(
            "{} contributions are redacted for {}; an administrator sees all of them:",
            g["redacted_contributions"], asker.agent_id
        ),
        "warn",
    );
    let memory_id = s(&ans["memory_id"]);
    let ga = admin.lineage(&memory_id)?;
    let teams: Vec<String> =
        ga["contributing_teams"].as_array().into_iter().flatten().map(|t| leaf(&s(t)).to_string()).collect();
    println!("      agents: {}   teams: {teams:?}", ga["contributing_agents"]);
    pause();

    head("Organizational propagation: which layers transformed what");
    for m in admin.list_memories(Some(ENTERPRISE), Some("team"), 50)? {
        println!(
            "   [team {:<12}] {:<26} support {}  ← {} observations  {D}{}{X}",
            leaf(&s(&m["scope"])),
            s(&m["topic"]),
            m["support"],
            m["metadata"]["parent_count"],
            clip(&s(&m["text"]), 70)
        );
    }
    for m in admin.list_memories(Some(ENTERPRISE), Some("department"), 50)? {
        let how = match m["metadata"]["promoted_from"].as_str() {
            Some(pf) => format!("promoted unchanged from team memory {pf} (single registered team)"),
            None => format!("consolidated from {} team memories", m["metadata"]["parent_count"]),
        };
        println!("   [dept {:<12}] {:<26} {how}", leaf(&s(&m["scope"])), s(&m["topic"]));
    }
    for t in ga["transformations"].as_array().into_iter().flatten() {
        let from: Vec<String> = t["from_layers"].as_array().into_iter().flatten().map(s).collect();
        let rule = t["rule_id"].as_str().map(|r| format!(" rule {r}")).unwrap_or_default();
        println!(
            "   {} → {:<11} via {}{rule}  at {}",
            from.join(" + "),
            s(&t["to_layer"]),
            s(&t["operator"]),
            s(&t["at"])
        );
    }
    pause();

    head("Lineage graph (administrator view)");
    for line in lineage_tree(&ga, &memory_id, "   ") {
        println!("{line}");
    }
    std::fs::write(out.join("lineage.json"), serde_json::to_string_pretty(&ga)?)?;
    std::fs::write(out.join("lineage.mmd"), mermaid(&ga))?;
    say(
        &format!("written {} and {} (Mermaid)", out.join("lineage.json").display(), out.join("lineage.mmd").display()),
        "ok",
    );
    pause();

    head("What stayed local, what was aggregated");
    let by_layer = st["stats"]["memories_by_layer"].clone();
    let local_total: i64 = states.values().map(|v| v["counts"]["local"].as_i64().unwrap_or(0)).sum();
    let private_total: i64 = states.values().map(|v| v["counts"]["local_only"].as_i64().unwrap_or(0)).sum();
    println!(
        "   notes on agents' own disks: {local_total}   never left the agent: {private_total}   shared: {}",
        local_total - private_total
    );
    let layers: Vec<String> = by_layer.as_object().into_iter().flatten().map(|(k, v)| format!("{k}={v}")).collect();
    println!("   organizational memory by layer: {}", layers.join(", "));
    for a in &sc.agents {
        for text in states[&a.agent_id]["local_only"].as_array().into_iter().flatten() {
            println!("   {D}kept private by {}: {}{X}", a.agent_id, clip(&s(text), 80));
        }
    }
    pause();

    head("Failure 1: the service is killed (SIGKILL) and restarted");
    driver.kill("mycelic")?;
    match MycelicClient::new(&driver.base_url(), &asker.api_key).with_retries(0).health() {
        Ok(_) => say("service still answering after kill?!", "bad"),
        Err(e) => say(&format!("service down: {}", e.message), "warn"),
    }
    driver.start("mycelic")?;
    wait_idle(&admin, Duration::from_secs(90))?;
    let res2 = client.query(QUERY, &enterprise)?;
    let roots = |v: &Value| -> Vec<String> {
        let mut r: Vec<String> = v["roots"].as_array().into_iter().flatten().map(s).collect();
        r.sort();
        r
    };
    let same = res2["answer"]["memory_id"] == ans["memory_id"] && roots(&res2["lineage"]) == roots(&g);
    if same {
        say(&format!("back: same conclusion {} with the same lineage roots", s(&res2["answer"]["memory_id"])), "ok");
    } else {
        say("conclusion changed after restart", "bad");
    }
    pause();

    head("Failure 2: the message broker is killed while an agent keeps working");
    let team_before = team_memory(&admin)?;
    driver.kill("nats")?;
    let writer = sc.by_team("logistics")[0].clone();
    let late = MycelicClient::new(&driver.base_url(), &writer.api_key).with_retries(0).remember(
        &Note {
            text: "Antwerp reroute for SD-9 containers adds 9 days and is not yet approved.".into(),
            topic: Some("supply:sd-9/transport".into()),
            slot: Some("transport_disruption".into()),
            entity: Some(ENTITY.into()),
            confidence: 0.55,
            ..Note::default()
        },
        "team",
        Some("antwerp-1"),
        None,
        None,
    )?;
    let public = MycelicClient::new(&driver.base_url(), "").health()?;
    say(
        &format!(
            "{} shared a new observation during the outage: accepted ({}), health={}, outbox pending={}",
            writer.agent_id,
            s(&late["memory_id"]),
            s(&public["status"]),
            admin.status()?["checks"]["publisher"]["outbox_pending"]
        ),
        "warn",
    );
    driver.start("nats")?;
    wait_idle(&admin, Duration::from_secs(120))?;
    let applied_at = admin.get_memory(&s(&late["memory_id"]))?["applied_at"].clone();
    let team = team_memory(&admin)?;
    let absorbed = !applied_at.is_null() && team["memory_id"] != team_before["memory_id"];
    if absorbed {
        say(
            &format!(
                "broker back: outbox flushed, observation applied at {}, logistics team memory re-derived ({} -> {} observations)",
                s(&applied_at),
                team_before["metadata"]["parent_count"],
                team["metadata"]["parent_count"]
            ),
            "ok",
        );
    } else {
        let changed = if team["memory_id"] == team_before["memory_id"] { "unchanged" } else { "changed" };
        say(
            &format!(
                "broker back but the outage-time observation was not absorbed (applied_at={applied_at}, team memory {changed})"
            ),
            "bad",
        );
    }
    pause();

    head("Failure 3: the service is killed and its database deleted — rebuild from the event log");
    let ids = |admin: &MycelicClient| -> Result<std::collections::BTreeSet<String>> {
        Ok(admin.list_memories(Some(ENTERPRISE), None, 500)?.iter().map(|m| s(&m["memory_id"])).collect())
    };
    let before = ids(&admin)?;
    driver.kill("mycelic")?;
    driver.wipe_database()?;
    driver.start("mycelic")?;
    let st = wait_idle(&admin, Duration::from_secs(180))?;
    let after = ids(&admin)?;
    let res3 = client.query(QUERY, &enterprise)?;
    let ok = after == before
        && res3["answer"]["memory_id"] == ans["memory_id"]
        && res3["lineage"]["evidence"]["reconstructable"] == true;
    say(
        &format!(
            "rebuilt {} active memories from JetStream (stream holds {} events); same conclusion id, lineage reconstructable, {} agents and their keys restored",
            after.len(),
            st["checks"]["transport"]["stream_messages"],
            admin.list_agents(Some(ENTERPRISE))?.len()
        ),
        if ok { "ok" } else { "bad" },
    );
    pause();

    let agents: serde_json::Map<String, Value> = states.iter().map(|(k, v)| (k.clone(), v["counts"].clone())).collect();
    let results = json!({"driver": driver.name(), "answer": res3["answer"], "lineage_roots": res3["lineage"]["roots"],
                         "memories_by_layer": st["stats"]["memories_by_layer"], "agents": agents,
                         "stream_messages": st["checks"]["transport"]["stream_messages"]});
    std::fs::write(out.join("summary.json"), serde_json::to_string_pretty(&results)?)?;
    head("Done");
    say(&format!("summary written to {}", out.join("summary.json").display()), "ok");
    if args.keep {
        println!("\n   Deployment left running at {}", driver.base_url());
        println!(
            "   MYCELIC_URL={}\n   MYCELIC_ADMIN_TOKEN={}\n   MYCELIC_API_KEY={}   # {}",
            driver.base_url(),
            driver.admin_token(),
            asker.api_key,
            asker.agent_id
        );
        println!("   try:  mycelic query \"delivery risk sd-9\" --scope northwind --lineage");
        println!(
            "   MCP:  claude mcp add --transport http mycelic $MYCELIC_URL/mcp --header \"Authorization: Bearer $MYCELIC_API_KEY\""
        );
        println!("   press Ctrl-C to stop");
        loop {
            std::thread::sleep(Duration::from_secs(3600));
        }
    }
    Ok(if ok && absorbed { 0 } else { 1 })
}

fn main() -> ExitCode {
    let args = Args::parse();
    let out = args.out.clone().unwrap_or_else(|| repo_root().join("demo/results/mycelic"));
    if let Err(e) = std::fs::create_dir_all(&out) {
        eprintln!("error: cannot create {}: {e}", out.display());
        return ExitCode::from(2);
    }
    let mut driver = match make_driver(&args.driver, "mycelic-demo", None, !args.no_build) {
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
