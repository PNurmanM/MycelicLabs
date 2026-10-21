//! Reference agent process: keep notes locally, share the ones worth propagating, watch for conclusions.
//!
//! ```text
//! mycelic-agent --url http://127.0.0.1:8080 --api-key mk_... --local-db /tmp/agent-7.db \
//!     --observations observations.jsonl --watch "delivery risk sd-9" --watch-scope northwind
//! ```
//!
//! `observations.jsonl` holds one JSON object per line: `{"text": ..., "topic"?, "slot"?, "entity"?,
//! "confidence"?, "share": true|false, "delay"?: seconds, "local_id"?, "visibility"?}`.  Notes with
//! `share: false` stay in the local store only (the demos use them to show what never left the agent).  The process
//! exits when the file is consumed unless `--stay` keeps it alive polling the watched query, printing every new
//! conclusion it sees.

use clap::Parser;
use mycelic_sdk::{LocalMemory, MycelicClient, Note, QueryOptions};
use serde_json::{Value, json};
use std::collections::BTreeSet;
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::{Duration, Instant};

#[derive(Parser)]
#[command(
    name = "mycelic-agent",
    about = "Reference agent process: keep notes locally, share the ones worth propagating, watch for conclusions."
)]
struct Args {
    #[arg(long, env = "MYCELIC_URL", default_value = "http://127.0.0.1:8080")]
    url: String,
    #[arg(long, env = "MYCELIC_API_KEY", hide_env_values = true)]
    api_key: Option<String>,
    #[arg(long, env = "MYCELIC_CA_FILE")]
    ca_file: Option<String>,
    /// sqlite file for this agent's private notes
    #[arg(long)]
    local_db: String,
    /// JSONL file of observations to make
    #[arg(long)]
    observations: Option<PathBuf>,
    /// query to poll for conclusions
    #[arg(long)]
    watch: Option<String>,
    #[arg(long)]
    watch_scope: Option<String>,
    #[arg(long, default_value = "team")]
    watch_min_layer: String,
    /// keep polling after the observations are consumed
    #[arg(long)]
    stay: bool,
    #[arg(long, default_value_t = 2.0)]
    poll_seconds: f64,
    /// write a JSON summary (counts, shared ids) here on exit
    #[arg(long)]
    state_file: Option<PathBuf>,
}

fn log(agent: &str, msg: &str) {
    println!("[{agent}] {msg}");
}

fn clip(s: &str, n: usize) -> String {
    s.chars().take(n).collect()
}

fn main() -> ExitCode {
    let args = Args::parse();
    let Some(key) = args.api_key.clone().filter(|k| !k.is_empty()) else {
        eprintln!("an API key is required (--api-key or MYCELIC_API_KEY)");
        return ExitCode::from(2);
    };
    match run(&args, &key) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::from(1)
        }
    }
}

fn run(args: &Args, key: &str) -> Result<(), Box<dyn std::error::Error>> {
    let client = MycelicClient::new(&args.url, key).with_ca_file(args.ca_file.as_deref())?;
    let local = LocalMemory::open(&args.local_db)?;
    let me = client.whoami()?;
    let agent_id = me["id"].as_str().unwrap_or("?").to_string();
    let counts = local.counts()?;
    log(
        &agent_id,
        &format!(
            "connected to {} as {} (local notes: {})",
            args.url,
            me["path"].as_str().unwrap_or("?"),
            counts["local"]
        ),
    );

    // 1. resume: anything noted but never acknowledged by the server is re-sent (idempotent by local id)
    for n in local.all(Some(false))? {
        if n["tags"].as_array().is_some_and(|t| t.iter().any(|x| x == "private")) {
            continue;
        }
        let id = n["local_id"].as_str().unwrap_or_default();
        match local.share(&client, id, "team") {
            Ok(res) => log(
                &agent_id,
                &format!("re-sent unacknowledged note {id} -> {}", res["memory_id"].as_str().unwrap_or("?")),
            ),
            Err(e) => log(&agent_id, &format!("could not re-send {id}: {e}")),
        }
    }

    // 2. observe
    if let Some(path) = &args.observations {
        for line in std::fs::read_to_string(path)?.lines() {
            if line.trim().is_empty() {
                continue;
            }
            let obs: Value = serde_json::from_str(line)?;
            if let Some(d) = obs.get("delay").and_then(Value::as_f64).filter(|d| *d > 0.0) {
                std::thread::sleep(Duration::from_secs_f64(d));
            }
            let share = obs.get("share").is_none_or(|v| v.as_bool().unwrap_or(true));
            let s = |k: &str| obs.get(k).and_then(Value::as_str).map(String::from);
            let note = Note {
                text: s("text").unwrap_or_default(),
                topic: s("topic"),
                slot: s("slot"),
                entity: s("entity"),
                kind: s("kind").unwrap_or_else(|| "observation".into()),
                confidence: obs.get("confidence").and_then(Value::as_f64).unwrap_or(0.8),
                tags: if share { vec![] } else { vec!["private".into()] },
                local_id: s("local_id"),
            };
            let local_id = local.note(&note)?;
            if !share {
                log(&agent_id, &format!("noted locally only: {}", clip(&note.text, 70)));
                continue;
            }
            let visibility = s("visibility").unwrap_or_else(|| "team".into());
            match local.share(&client, &local_id, &visibility) {
                Ok(res) => {
                    let fresh = res.get("created").and_then(Value::as_bool).unwrap_or(true);
                    log(
                        &agent_id,
                        &format!(
                            "shared {local_id} -> {} ({})",
                            res["memory_id"].as_str().unwrap_or("?"),
                            if fresh { "new" } else { "already known" }
                        ),
                    );
                }
                Err(e) => log(
                    &agent_id,
                    &format!("share failed for {local_id}: {e} (kept locally, will retry on next start)"),
                ),
            }
        }
    }

    // 3. watch for conclusions
    let mut seen: BTreeSet<String> = BTreeSet::new();
    let deadline = if args.stay { None } else { Some(Instant::now()) };
    if let Some(watch) = &args.watch {
        loop {
            let opts = QueryOptions {
                scope: args.watch_scope.clone(),
                min_layer: args.watch_min_layer.clone(),
                k: 3,
                include_lineage: false,
                ..QueryOptions::default()
            };
            match client.query(watch, &opts) {
                Ok(res) => {
                    let ans = &res["answer"];
                    if let Some(mid) = ans["memory_id"].as_str()
                        && seen.insert(mid.to_string())
                    {
                        log(
                            &agent_id,
                            &format!(
                                "conclusion at {} '{}' (support {} agents / {} teams, confidence {}): {}",
                                ans["layer"].as_str().unwrap_or("?"),
                                ans["scope"].as_str().unwrap_or("?"),
                                ans["support"],
                                ans["independent_teams"],
                                ans["confidence"],
                                clip(ans["text"].as_str().unwrap_or(""), 160)
                            ),
                        );
                    }
                }
                Err(e) => log(&agent_id, &format!("query failed: {e}")),
            }
            if deadline.is_some_and(|d| Instant::now() >= d) {
                break;
            }
            std::thread::sleep(Duration::from_secs_f64(args.poll_seconds));
        }
    }

    let summary = json!({
        "agent_id": agent_id, "path": me["path"], "counts": local.counts()?,
        "shared": local.all(Some(true))?.iter().map(|n| json!({"local_id": n["local_id"], "memory_id": n["shared_memory_id"]})).collect::<Vec<_>>(),
        "local_only": local.all(Some(false))?.iter().map(|n| n["text"].clone()).collect::<Vec<_>>(),
        "conclusions_seen": seen,
    });
    if let Some(path) = &args.state_file {
        std::fs::write(path, serde_json::to_string_pretty(&summary)?)?;
    }
    log(&agent_id, &format!("done: {}", summary["counts"]));
    Ok(())
}
