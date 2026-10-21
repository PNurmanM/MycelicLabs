//! Command line for Mycelic.
//!
//! ```text
//! mycelic serve                       # API + publisher + consumer (configuration from MYCELIC_* env vars)
//! mycelic mcp --url URL --api-key KEY # MCP over stdio, proxied to a running server (Claude Desktop / Code)
//! mycelic register-agent --enterprise northwind --team logistics --agent-id agent-7
//! mycelic agents | rules | status | replay | query "delivery risk"
//! mycelic healthcheck                 # the container probe: exit 0 when /health answers 200
//! ```
//!
//! Admin commands are HTTP clients of a running server: MYCELIC_URL (default http://127.0.0.1:8080) and
//! MYCELIC_ADMIN_TOKEN.  `query` uses MYCELIC_API_KEY (an agent key) or the admin token.

use clap::{Args, Parser, Subcommand};
use mycelic::config::Settings;
use mycelic::service::MycelicService;
use mycelic_sdk::{MycelicClient, MycelicError, QueryOptions, Registration};
use serde_json::{Value, json};
use std::process::ExitCode;

#[derive(Parser)]
#[command(name = "mycelic", version = mycelic::VERSION, about = "Mycelic: shared organizational memory for AI agents",
          long_about = None, after_help = "Admin commands are HTTP clients of a running server: MYCELIC_URL (default http://127.0.0.1:8080) and MYCELIC_ADMIN_TOKEN. query uses MYCELIC_API_KEY (an agent key) or the admin token.")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Args, Clone)]
struct Common {
    #[arg(long, env = "MYCELIC_URL", default_value = "http://127.0.0.1:8080")]
    url: String,
    #[arg(long)]
    admin_token: Option<String>,
    #[arg(long, env = "MYCELIC_CA_FILE")]
    ca_file: Option<String>,
    #[arg(long)]
    json: bool,
}

#[derive(Subcommand)]
enum Cmd {
    /// run the service
    Serve {
        #[arg(long)]
        host: Option<String>,
        #[arg(long)]
        port: Option<i64>,
        #[arg(long)]
        db: Option<String>,
        /// '' disables NATS (in-process transport; development only)
        #[arg(long)]
        nats_url: Option<String>,
        #[arg(long)]
        rules_file: Option<String>,
    },
    /// MCP over stdio, proxied to a running server
    Mcp {
        #[command(flatten)]
        common: Common,
        #[arg(long)]
        api_key: Option<String>,
    },
    /// register an agent and print its key once
    RegisterAgent {
        #[command(flatten)]
        common: Common,
        #[arg(long)]
        agent_id: String,
        #[arg(long)]
        enterprise: String,
        #[arg(long)]
        region: Option<String>,
        #[arg(long)]
        subsidiary: Option<String>,
        #[arg(long)]
        department: Option<String>,
        #[arg(long)]
        team: Option<String>,
        #[arg(long)]
        display_name: Option<String>,
        /// comma-separated (default: read/write/events/lineage)
        #[arg(long)]
        scopes: Option<String>,
    },
    /// list agents
    Agents {
        #[command(flatten)]
        common: Common,
        #[arg(long)]
        org: Option<String>,
    },
    /// list rules, optionally loading a JSON file first
    Rules {
        #[command(flatten)]
        common: Common,
        #[arg(long)]
        file: Option<String>,
    },
    /// full service status (admin)
    Status {
        #[command(flatten)]
        common: Common,
    },
    /// re-deliver the whole event log to the service (admin)
    Replay {
        #[command(flatten)]
        common: Common,
    },
    /// ask the organization
    Query {
        #[command(flatten)]
        common: Common,
        text: String,
        #[arg(long)]
        api_key: Option<String>,
        #[arg(long)]
        scope: Option<String>,
        #[arg(long, default_value = "agent")]
        min_layer: String,
        #[arg(short = 'k', default_value_t = 5)]
        k: i64,
        #[arg(long)]
        lineage: bool,
    },
    /// probe GET /health on this host (the container healthcheck): exit 0 on 200
    Healthcheck {
        #[arg(long, env = "MYCELIC_PORT", default_value_t = 8080)]
        port: u16,
    },
}

enum Fail {
    Exit(u8, String),
    Api(MycelicError),
}

impl From<MycelicError> for Fail {
    fn from(e: MycelicError) -> Self {
        Fail::Api(e)
    }
}

type CmdResult = Result<u8, Fail>;

fn admin_client(c: &Common) -> Result<MycelicClient, Fail> {
    let token = c.admin_token.clone().or_else(|| std::env::var("MYCELIC_ADMIN_TOKEN").ok()).filter(|t| !t.is_empty());
    let Some(token) = token else {
        return Err(Fail::Exit(1, "MYCELIC_ADMIN_TOKEN (or --admin-token) is required".into()));
    };
    Ok(MycelicClient::new(&c.url, &token).with_ca_file(c.ca_file.as_deref())?)
}

fn indent1(v: &Value) -> String {
    mycelic::pyjson::dumps_with(v, mycelic::pyjson::Style::INDENT1)
}

async fn cmd_serve(
    host: Option<String>,
    port: Option<i64>,
    db: Option<String>,
    nats_url: Option<String>,
    rules_file: Option<String>,
) -> CmdResult {
    let mut settings = match Settings::from_env() {
        Ok(s) => s,
        Err(e) => return Err(Fail::Exit(2, format!("configuration error: {e}"))),
    };
    if let Some(h) = host.filter(|h| !h.is_empty()) {
        settings.host = h;
    }
    if let Some(p) = port.filter(|p| *p != 0) {
        settings.port = p;
    }
    if let Some(d) = db.filter(|d| !d.is_empty()) {
        settings.db_path = d;
    }
    if let Some(n) = nats_url {
        settings.nats_url = Some(n).filter(|n| !n.is_empty());
    }
    if let Some(r) = rules_file.filter(|r| !r.is_empty()) {
        settings.rules_file = Some(r);
    }
    if let Err(e) = settings.validate() {
        return Err(Fail::Exit(2, format!("configuration error: {e}")));
    }
    let level = match settings.log_level.as_str() {
        "DEBUG" => "debug",
        "WARNING" | "WARN" => "warn",
        "ERROR" | "CRITICAL" => "error",
        _ => "info",
    };
    let filter = tracing_subscriber::EnvFilter::new(format!("{level},async_nats=warn,hyper=warn"));
    use std::io::IsTerminal;
    let _ = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(true)
        .with_ansi(std::io::stdout().is_terminal())
        .with_writer(std::io::stdout)
        .try_init();
    let service =
        MycelicService::new(settings.clone()).map_err(|e| Fail::Exit(1, format!("cannot open the database: {e:#}")))?;
    // bind first so /health answers while the broker comes up
    let server =
        mycelic::api::run_server(service.clone()).await.map_err(|e| Fail::Exit(1, format!("cannot bind: {e:#}")))?;
    if let Err(e) = service.start(true).await {
        return Err(Fail::Exit(1, format!("start failed: {e:#}")));
    }
    let scheme = if settings.tls_enabled() { "https" } else { "http" };
    println!(
        "{}",
        mycelic::pyjson::dumps(
            &json!({"mycelic": "started", "url": format!("{scheme}://{}:{}", settings.host, server.local_addr.port()),
                                         "db": settings.db_path, "transport": service.transport.name(), "nats_url": settings.nats_url})
        )
    );
    let mut term = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
        .map_err(|e| Fail::Exit(1, e.to_string()))?;
    tokio::select! {
        _ = tokio::signal::ctrl_c() => {}
        _ = term.recv() => {}
    }
    println!("shutting down");
    server.shutdown().await;
    service.close().await;
    Ok(0)
}

async fn cmd_mcp(common: Common, api_key: Option<String>) -> CmdResult {
    let _ = tracing_subscriber::fmt().with_env_filter("warn").with_writer(std::io::stderr).try_init();
    let key = api_key.or_else(|| std::env::var("MYCELIC_API_KEY").ok()).filter(|k| !k.is_empty());
    let Some(key) = key else { return Err(Fail::Exit(2, "MYCELIC_API_KEY (or --api-key) is required".into())) };
    let client = MycelicClient::new(&common.url, &key).with_ca_file(common.ca_file.as_deref())?;
    mycelic::mcp::serve_stdio_proxy(client).await.map_err(|e| Fail::Exit(1, e.to_string()))?;
    Ok(0)
}

fn cmd_query(
    common: Common,
    text: String,
    api_key: Option<String>,
    scope: Option<String>,
    min_layer: String,
    k: i64,
    lineage: bool,
) -> CmdResult {
    let key = api_key
        .or_else(|| std::env::var("MYCELIC_API_KEY").ok().filter(|k| !k.is_empty()))
        .or_else(|| std::env::var("MYCELIC_ADMIN_TOKEN").ok().filter(|k| !k.is_empty()));
    let Some(key) = key else { return Err(Fail::Exit(1, "MYCELIC_API_KEY or MYCELIC_ADMIN_TOKEN is required".into())) };
    let client = MycelicClient::new(&common.url, &key).with_ca_file(common.ca_file.as_deref())?;
    let res = client
        .query(&text, &QueryOptions { scope, min_layer, k, include_lineage: lineage, topic: None, entity: None })?;
    if common.json {
        println!("{}", indent1(&res));
        return Ok(0);
    }
    let ans = &res["answer"];
    if ans.is_null() {
        println!("(nothing relevant visible to this principal)");
        return Ok(1);
    }
    let s = |v: &Value| v.as_str().map(String::from).unwrap_or_else(|| v.to_string());
    println!(
        "[{} {}] confidence {} support {} agents / {} teams",
        s(&ans["layer"]),
        s(&ans["scope"]),
        ans["confidence"],
        ans["support"],
        ans["independent_teams"]
    );
    println!("{}", s(&ans["text"]));
    let lin = &ans["lineage"];
    println!(
        "lineage: {} agents, {} teams, layers {}, reconstructable={}  (memory {})",
        lin["contributing_agents"],
        lin["contributing_teams"],
        lin["layers"],
        lin["reconstructable"],
        s(&ans["memory_id"])
    );
    for h in res["results"].as_array().into_iter().flatten().skip(1) {
        let text: String = s(&h["memory"]["text"]).chars().take(120).collect();
        println!("  - [{}] {text}", s(&h["memory"]["layer"]));
    }
    Ok(0)
}

fn healthcheck(port: u16) -> CmdResult {
    // the container image has no curl/wget: the binary probes itself.  With MYCELIC_TLS_CERT_FILE set the service
    // speaks HTTPS with a certificate issued for its public name, so the probe only checks that TLS answers.
    let tls = std::env::var("MYCELIC_TLS_CERT_FILE").is_ok_and(|v| !v.is_empty());
    let scheme = if tls { "https" } else { "http" };
    let url = format!("{scheme}://127.0.0.1:{port}/health");
    let mut tls_config = ureq::tls::TlsConfig::builder();
    if tls {
        tls_config = tls_config.disable_verification(true);
    }
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .timeout_global(Some(std::time::Duration::from_secs(4)))
        .http_status_as_error(false)
        .tls_config(tls_config.build())
        .build()
        .into();
    match agent.get(&url).call() {
        Ok(r) if r.status().as_u16() == 200 => Ok(0),
        Ok(r) => Err(Fail::Exit(1, format!("{url}: HTTP {}", r.status().as_u16()))),
        Err(e) => Err(Fail::Exit(1, format!("{url}: {e}"))),
    }
}

async fn dispatch(cli: Cli) -> CmdResult {
    match cli.cmd {
        Cmd::Serve { host, port, db, nats_url, rules_file } => cmd_serve(host, port, db, nats_url, rules_file).await,
        Cmd::Mcp { common, api_key } => cmd_mcp(common, api_key).await,
        Cmd::Healthcheck { port } => healthcheck(port),
        other => {
            tokio::task::spawn_blocking(move || blocking_cmd(other)).await.map_err(|e| Fail::Exit(1, e.to_string()))?
        }
    }
}

fn blocking_cmd(cmd: Cmd) -> CmdResult {
    match cmd {
        Cmd::RegisterAgent {
            common,
            agent_id,
            enterprise,
            region,
            subsidiary,
            department,
            team,
            display_name,
            scopes,
        } => {
            let client = admin_client(&common)?;
            let res = client.register_agent(&Registration {
                agent_id,
                enterprise,
                region,
                subsidiary,
                department,
                team,
                display_name,
                scopes: scopes.map(|s| s.split(',').map(String::from).collect()),
            })?;
            if common.json {
                println!("{}", indent1(&res));
            } else {
                let a = &res["agent"];
                println!(
                    "registered {} at {}",
                    a["agent_id"].as_str().unwrap_or("?"),
                    a["path"].as_str().unwrap_or("?")
                );
                println!("MYCELIC_API_KEY={}", res["api_key"].as_str().unwrap_or("?"));
                println!("(the key is shown once; store it in the agent's secret store)");
            }
            Ok(0)
        }
        Cmd::Agents { common, org } => {
            for a in admin_client(&common)?.list_agents(org.as_deref())? {
                println!(
                    "{:<24} {:<8} {}  last seen {}",
                    a["agent_id"].as_str().unwrap_or(""),
                    a["status"].as_str().unwrap_or(""),
                    a["path"].as_str().unwrap_or(""),
                    a["last_seen_at"].as_str().unwrap_or("-")
                );
            }
            Ok(0)
        }
        Cmd::Rules { common, file } => {
            let client = admin_client(&common)?;
            if let Some(f) = file {
                let text = std::fs::read_to_string(&f).map_err(|e| Fail::Exit(1, format!("{f}: {e}")))?;
                let data: Value = serde_json::from_str(&text).map_err(|e| Fail::Exit(1, format!("{f}: {e}")))?;
                let rules = match &data {
                    Value::Object(m) => m.get("rules").cloned().unwrap_or(data.clone()),
                    other => other.clone(),
                };
                for rule in rules.as_array().cloned().unwrap_or_default() {
                    let r = client.upsert_rule(&rule)?;
                    println!("upserted {}", r["rule"]["rule_id"].as_str().unwrap_or("?"));
                }
            }
            for r in client.list_rules()? {
                let slots: Vec<&str> =
                    r["required_slots"].as_array().into_iter().flatten().filter_map(Value::as_str).collect();
                let enabled = if r["enabled"].as_bool().unwrap_or(false) { "True" } else { "False" };
                println!(
                    "{:<32} -> {:<12} slots={} enabled={enabled}",
                    r["rule_id"].as_str().unwrap_or(""),
                    r["target_layer"].as_str().unwrap_or(""),
                    slots.join(",")
                );
            }
            Ok(0)
        }
        Cmd::Status { common } => {
            println!("{}", indent1(&admin_client(&common)?.status()?));
            Ok(0)
        }
        Cmd::Replay { common } => {
            println!("{}", mycelic::pyjson::dumps(&admin_client(&common)?.replay()?));
            Ok(0)
        }
        Cmd::Query { common, text, api_key, scope, min_layer, k, lineage } => {
            cmd_query(common, text, api_key, scope, min_layer, k, lineage)
        }
        _ => unreachable!("async commands are dispatched before"),
    }
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let rt = match tokio::runtime::Builder::new_multi_thread().enable_all().build() {
        Ok(rt) => rt,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::from(1);
        }
    };
    match rt.block_on(dispatch(cli)) {
        Ok(code) => ExitCode::from(code),
        Err(Fail::Exit(code, msg)) => {
            eprintln!("{msg}");
            ExitCode::from(code)
        }
        Err(Fail::Api(e)) => {
            eprintln!("error: {e}");
            ExitCode::from(1)
        }
    }
}
