//! Mycelic live demo: a visual console that drives the real stack, and a replay of a recorded run.
//!
//! ```text
//! mycelic-live                                             # start the stack, serve the console, open the browser
//! mycelic-live --auto --dwell 8                            # advance the acts by themselves, 8 s apart
//! mycelic-live --record demo/live/trace.json               # headless: every act, write the trace
//! mycelic-live --replay [demo/live/trace.json]             # serve a recorded run, no stack needed
//! mycelic-live --export page.html [--trace demo/live/trace.json]      # standalone replay page
//! mycelic-live --export-fragment page.fragment.html                    # same, no skeleton
//! ```
//!
//! Orrery Robotics, a fictional company (synthetic data), runs 40 AI agents in EMEA, APAC and head office
//! ([`scenario`]).  The default mode (`--present`) checks the prerequisites, starts nats-server and the Mycelic
//! service (`--driver process`, or `--driver compose`), serves the console at `http://127.0.0.1:8765/` and runs
//! act 0; the presenter advances the next act from the console (`POST /control {"action": "next"}`) or `--auto`
//! does it every `--dwell` seconds.  Presenter notes open in a second window at `http://127.0.0.1:8765/?presenter`
//! (it follows the same run), never on the projected page unless asked for (N).  Every act does real work against
//! the stack: agents are processes (`mycelic-agent`) with their own keys and local memory files; the conclusions
//! are Mycelic's.  Nothing shown is hard-coded: the engine polls the admin read API (memories by status, lineage,
//! status) and records what changed, with the time it saw it, plus the actions it takes itself.  If the stack
//! cannot run, `--replay` (default: `demo/live/trace.json`) serves a recorded run instead.  After the close act the
//! engine stops the stack and says so (a `service` event, state `stopped`); the console stays up showing it, until
//! Ctrl-C.  If a live run fails, the console stays up with the error and the replay command, until Ctrl-C.
//!
//! Acts: 0 cold open (service comes up) · 1 the organisation (agents register) · 2 agents observe (34 agents
//! share, everyone keeps private notes) · 3 each region concludes (procurement shares; `regional_supply_risk` is
//! evaluated in EMEA and APAC) · 4 the enterprise puts together what no single team held (HQ shares;
//! `strategic_second_source` is evaluated) · 5 every conclusion has a receipt (lineage, admin and redacted views) ·
//! 6 strategy follows evidence (retraction, recount) · 7 pull the plug (SIGKILL, delete the database, rebuild the
//! memories, lineage and agents from JetStream; the audit log is not in the event log and is not rebuilt) · 8 close.
//!
//! The demo's operator runs two rules, copied verbatim from `deploy/mycelic/rules.json`: `regional_supply_risk` and
//! `strategic_second_source` (with the compose driver the deployment's own rules file applies).
//!
//! Exit codes: 0 every check passed, 1 a check failed, 2 the prerequisites are missing or the live run failed, 130
//! interrupted (Ctrl-C, SIGTERM, SIGHUP; a finished run stopped that way keeps its result).
//!
//! # HTTP endpoints (live and replay)
//!
//! ```text
//! GET  /         the console (demo/live/console.html wrapped in a <!doctype html> skeleton; in replay mode with
//!                the trace embedded)
//! GET  /trace    the trace so far (JSON, format below)
//! GET  /events   Server-Sent Events.  On connect: one `event: snapshot` message (the trace without `events`),
//!                then every event so far, then new ones as they happen; each default message is one trace event
//!                as JSON with `id: <index>` (a reconnect with `Last-Event-ID` resumes after it).  Another
//!                `snapshot` is sent whenever org, rules, checks or summary change.  `: ping` every 15 s.
//!                A client whose `Last-Event-ID` is past the end of this run's events (a tab left open across a
//!                restart) gets every event from the start; the console also compares `meta.run_id` and reloads.
//! POST /control  {"action": "next"} runs the next act (live mode; `busy` while an act runs; 409 in replay, with
//!                --auto, and once the run is complete; 400 for a body that is not a JSON object)
//! ```
//!
//! # Console contract
//!
//! `console.html` is a *fragment*: `<title>` and `<style>` first, then markup and scripts, and no
//! doctype/html/head/body tags.  The server and `--export` wrap it in a skeleton with charset and viewport meta;
//! `--export-fragment` does not (for publishers that add their own); the skeleton puts the fragment's leading
//! `<title>` and `<style>` in `<head>`.  The trace is injected as
//! `<script id="mycelic-trace" type="application/json">...</script>` in place of the marker comment
//! `<!-- mycelic:trace -->` (or before the first `<script` if the marker is missing), with every `<` in the JSON
//! escaped as `\u003c`.  With the tag present the console replays it; without it the console is live and reads
//! `/trace` and `/events`.
//!
//! # Trace format (`schema: "mycelic-demo-trace/1"`)
//!
//! ```text
//! {"schema": "mycelic-demo-trace/1",
//!  "meta": {"recorded_at": ISO-8601 UTC, "mycelic_version", "driver", "nats_server_version", "python" (null: the
//!           recorder is this crate), "rust" (the recorder's crate version), "implementation": "rust", "platform",
//!           "duration_s", "fictional": true, "company", "mode": "record"|"present", "poll_interval_s",
//!           "fast_poll_interval_s", "rules_source", "run_id": random per run,
//!           "story": {"entity", "label", "question"} (the scenario's subject, for the console's labels)},
//!  "org": {"enterprise": id,
//!          "units": [{"path", "layer", "label", "parent"}],          # enterprise, region, subsidiary, department, team
//!          "agents": [{"agent_id", "path", "team_path", "region", "label", "role": "core"|"background",
//!                      "stage": "observe"|"conclude"|"synthesise"}]},
//!  "rules": [{"rule_id", "target_layer", "required_slots", "emits_slot", "min_units", "min_agents", "min_teams",
//!             "conclusion", "topic_prefix", "sources", "corroborate", "kind", "owner"}],
//!  "acts": [{"id", "index", "title", "caption", "t0", "t1"}],
//!  "events": [{"t": seconds since run start, "act": act id, "type", ...payload}],   # ordered, t non-decreasing
//!  "checks": [{"id", "act", "text", "ok"}],
//!  "summary": {...}}
//! ```
//!
//! Events are in the order they were recorded and `t` never decreases.  Events the watcher produces carry the
//! time of the API read that first returned the change (or the previous event's time if that is later), so `t` of
//! a `note_shared` or `derived` event is when it became readable, not when the engine got round to writing it.
//!
//! Event types and payloads:
//!
//! * `service` {state: starting|healthy|killed|db_wiped|rebuilding|ready|stopped, detail} (`ready` in act 7 also
//!   carries rebuild_s and rebuild_events; it is stamped when the restarted service answered `/ready`, i.e. replay
//!   done; `stopped` is emitted once, after the close act, when the engine has torn the stack down, in every mode)
//! * `agent_registered` {agent_id, path, team_path, region, role, label}
//! * `note_shared` {agent_id, memory_id, team_path, region, role, topic, slot, entity, text, confidence, status}
//!   -- seen when the service accepted it (before it is applied)
//! * `note_private` {agent_id, text, local_only: true} -- read from the agent's own local state after it ran
//! * `derived` {memory_id, layer, scope, operator, rule_id, topic, slot, entity, kind, text, confidence, support,
//!   independent_teams, version_of, status, parents: [memory ids], roots: [agent note ids], corroborated_units}
//! * `status_change` {memory_id, layer, from, to, reason} -- `from` is null when the watcher first saw the memory
//!   already superseded or retracted (it never observed it active)
//! * `retracted` {agent_id, memory_id, reason} -- the agent withdrew its own note (the engine did it with its key)
//! * `query` {viewer, question, scope, min_layer, results: [{memory_id, layer, scope, text, score, rule_id, operator}]}
//! * `lineage` {viewer, memory_id, graph} -- `graph` exactly as `GET /lineage/{id}` returns it for that viewer
//!   (`viewer` is `"admin"` or an agent id)
//! * `rebuild_progress` {applied, total}
//! * `metric` {memories_by_layer, memories_by_status, lineage_edges, registered_agents, stream_messages}
//! * `check` {id, text, ok}
//! * `act_start` / `act_end` (`act` names it), `done` {} at the end, and `error` {message, hint} if a live run
//!   failed (one line; `hint` is the replay command)
//!
//! A completed run ends `act_end` (close), `service` (stopped), `done`: the stopped event carries the close act's
//! id and `done` is always the last event, so a live console sees the stack go down before the run is marked
//! complete, and a replay shows the same end state.  `meta.duration_s` runs to `done` (teardown included).  A
//! failed run ends with `error` instead, and an interrupted one where it was cut off (neither has `done`).
//!
//! `summary` holds real counts from the run: agents, agents_core, agents_background, teams, notes_shared,
//! notes_private, memories_by_layer, memories_by_status, lineage_edges, stream_messages (at the end), rebuild_s
//! (restart to ready, including process start), rebuild_events (the log replayed), events_appended_on_restart (the
//! restarted service logs its rules file again, one event per rule, because a fresh database does not know them),
//! time_to_region_s {region: s}, time_to_enterprise_s, time_to_restore_s (each an upper bound on "the last note a
//! conclusion rests on accepted -> the conclusion readable": from the moment the last watcher read that did not
//! return that note was sent to the arrival of the first read that returned the conclusion, so the poller's own
//! cadence, `fast_poll_interval_s` plus the time a poll takes, is included), background_notes,
//! background_consolidations, background_rule_conclusions, strategy {memory_id, support, independent_teams,
//! confidence, regions}, checks_passed, checks_total.  API keys, the admin token, the NATS password and the
//! signing key never enter the trace; a guard refuses to write one that contains any of them.

pub mod demo;
pub mod scenario;
pub mod server;
pub mod signals;
pub mod trace;
pub mod watcher;

pub use demo::{DEMO_RULES, LiveDemo};
pub use server::{BusyFn, ConsoleServer, ControlQueue, RefuseFn, Source};
pub use signals::Interrupted;
pub use trace::{SCHEMA, TraceRecorder, build_page, console_path, default_trace_path, html_json, load_trace};

use crate::{ComposeDriver, Driver, ProcessDriver};
use anyhow::Result;
use clap::Parser;
use demo::{ACT_COUNT, py_float, py_str, write_demo_rules};
use serde_json::{Value, json};
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

pub const REPLAY_COMMAND: &str = "mycelic-live --replay --port 8766";
pub const REPLAY_HINT: &str =
    "or, if you cannot run the stack, serve the recorded run: mycelic-live --replay --port 8766";
pub const STOPPED_DETAIL: &str = "demo finished; stack stopped (nats-server and the Mycelic service)";

/// The command line (`mycelic-live --help`).
#[derive(Parser, Debug, Clone)]
#[command(
    name = "mycelic-live",
    about = "Mycelic live demo: a visual console that drives the real stack, and a replay of a recorded run",
    long_about = LONG_ABOUT,
)]
pub struct Args {
    /// start the stack and serve the live console (the default)
    #[arg(long, conflicts_with_all = ["record", "replay", "export"])]
    pub present: bool,
    /// headless: run every act back to back and write the trace
    #[arg(long, value_name = "PATH", conflicts_with_all = ["replay", "export"])]
    pub record: Option<PathBuf>,
    /// serve the console over a recorded trace (default: demo/live/trace.json); no stack needed
    #[arg(long, value_name = "PATH", num_args = 0..=1, conflicts_with = "export")]
    pub replay: Option<Option<PathBuf>>,
    /// write a standalone replay page with the trace embedded
    #[arg(long, value_name = "PATH")]
    pub export: Option<PathBuf>,
    /// the replay page without doctype/html/head/body
    #[arg(long, value_name = "PATH")]
    pub export_fragment: Option<PathBuf>,
    /// the trace to export (default: demo/live/trace.json)
    #[arg(long, value_name = "TRACE")]
    pub trace: Option<PathBuf>,
    #[arg(long, default_value = "process", value_parser = ["process", "compose"])]
    pub driver: String,
    /// compose: do not rebuild the image
    #[arg(long)]
    pub no_build: bool,
    #[arg(long, default_value = "127.0.0.1")]
    pub host: String,
    #[arg(long, default_value_t = 8765)]
    pub port: u16,
    #[arg(long)]
    pub no_browser: bool,
    /// advance the acts by themselves
    #[arg(long)]
    pub auto: bool,
    /// --auto: seconds between acts
    #[arg(long, default_value_t = 8.0)]
    pub dwell: f64,
    /// live/replay: stop this many seconds after the last act (default: wait for Ctrl-C)
    #[arg(long, value_name = "SECONDS")]
    pub exit_after: Option<f64>,
    /// seconds over which act 2's agents start sharing
    #[arg(long, default_value_t = 6.0)]
    pub spread: f64,
    /// watcher interval between agent waves
    #[arg(long, default_value_t = 0.25)]
    pub poll: f64,
    /// watcher interval while agents share
    #[arg(long, default_value_t = 0.02)]
    pub fast_poll: f64,
}

const LONG_ABOUT: &str = "\
Mycelic live demo: a visual console that drives the real stack, and a replay of a recorded run.

    mycelic-live                                  # start the stack, serve the console, open the browser
    mycelic-live --auto --dwell 8                 # advance the acts by themselves, 8 s apart
    mycelic-live --record demo/live/trace.json    # headless: every act, write the trace
    mycelic-live --replay [demo/live/trace.json]  # serve a recorded run, no stack needed
    mycelic-live --export page.html [--trace demo/live/trace.json]   # standalone replay page
    mycelic-live --export-fragment page.fragment.html                 # same, no skeleton

Orrery Robotics, a fictional company (synthetic data), runs 40 AI agents in EMEA, APAC and head office.  The
default mode (--present) checks the prerequisites, starts nats-server and the Mycelic service (--driver process,
or --driver compose), serves the console at http://127.0.0.1:8765/ and runs act 0; the presenter advances the
next act from the console or --auto does it every --dwell seconds.  Presenter notes: http://127.0.0.1:8765/?presenter.
If the stack cannot run, --replay serves a recorded run instead.

Exit codes: 0 every check passed, 1 a check failed, 2 prerequisites missing or the live run failed, 130 interrupted.";

/// Run the demo; the process exit code.
pub fn main(args: Args) -> u8 {
    if args.export.is_some() || args.export_fragment.is_some() {
        if args.record.is_some() || args.replay.is_some() || args.present {
            use clap::CommandFactory;
            Args::command()
                .error(
                    clap::error::ErrorKind::ArgumentConflict,
                    "--export/--export-fragment cannot be combined with --present, --record or --replay",
                )
                .exit();
        }
        return export(&args);
    }
    bypass_proxies_for_localhost();
    if let Some(replay) = &args.replay {
        let path = replay.clone().unwrap_or_else(default_trace_path);
        return serve_replay(&args, &path);
    }
    let problems = preflight(&args, args.record.is_none());
    if !problems.is_empty() {
        for p in problems {
            eprintln!("error: {p}");
        }
        eprintln!("fix the above, {REPLAY_HINT}");
        return 2;
    }
    run_live(&args)
}

/// Every request the demo makes goes to 127.0.0.1; an `http_proxy` without a `no_proxy` entry would route it to the
/// proxy.  The agent processes and the service inherit this.
fn bypass_proxies_for_localhost() {
    for var in ["NO_PROXY", "no_proxy"] {
        let current = std::env::var(var).unwrap_or_default();
        let mut have: Vec<String> =
            current.split(',').map(str::trim).filter(|h| !h.is_empty()).map(String::from).collect();
        for h in ["127.0.0.1", "localhost"] {
            if !have.iter().any(|x| x == h) {
                have.push(h.into());
            }
        }
        // SAFETY: called from `main` before any thread of this program is started, so nothing reads the
        // environment concurrently.
        unsafe { std::env::set_var(var, have.join(",")) };
    }
}

fn port_free(host: &str, port: u16) -> bool {
    TcpListener::bind((host, port)).is_ok()
}

fn on_path(name: &str) -> bool {
    std::env::var_os("PATH").is_some_and(|paths| std::env::split_paths(&paths).any(|d| d.join(name).is_file()))
}

#[cfg(unix)]
fn executable(path: &Path) -> bool {
    use std::os::unix::fs::PermissionsExt;
    std::fs::metadata(path).is_ok_and(|m| m.permissions().mode() & 0o111 != 0)
}

#[cfg(not(unix))]
fn executable(_: &Path) -> bool {
    true
}

/// Everything that would make the live run fail, each with what to do about it.
pub fn preflight(args: &Args, serve: bool) -> Vec<String> {
    let mut problems = Vec::new();
    if args.driver == "process" {
        match crate::nats_server_bin() {
            None => problems.push(
                "nats-server not found: install it (https://github.com/nats-io/nats-server/releases) and put it on \
                 PATH, or set MYCELIC_NATS_SERVER_BIN=/path/to/nats-server"
                    .to_string(),
            ),
            Some(bin) if !bin.is_file() => problems.push(format!(
                "nats-server not found at {} (MYCELIC_NATS_SERVER_BIN): point it at the binary itself",
                bin.display()
            )),
            Some(bin) if !executable(&bin) => {
                problems.push(format!("nats-server at {} is not executable: chmod +x it", bin.display()))
            }
            Some(_) => {}
        }
        let missing: Vec<&str> =
            ["mycelic", "mycelic-agent"].into_iter().filter(|b| !crate::binary(b).is_file()).collect();
        if !missing.is_empty() {
            problems.push(format!(
                "the Mycelic binaries were not found next to this program or on PATH ({}): build them with \
                 `cargo build --release` (or set MYCELIC_BIN_DIR)",
                missing.join(", ")
            ));
        }
    } else if !on_path("docker") {
        problems.push("docker not found: install Docker, or use --driver process".into());
    } else if !Command::new("docker")
        .arg("info")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|s| s.success())
    {
        problems.push("the Docker daemon is not reachable: start it, or use --driver process".into());
    }
    if serve && !port_free(&args.host, args.port) {
        problems.push(format!(
            "port {} on {} is in use: pass --port <another>, or stop what is using it",
            args.port, args.host
        ));
    }
    if serve && !console_path().exists() {
        problems.push(format!("the console page {} is missing", console_path().display()));
    }
    problems
}

/// The deployment: the process driver with the demo's own rules file, or the compose driver with the deployment's.
fn make_driver(args: &Args) -> Result<(Box<dyn Driver>, Option<PathBuf>)> {
    if args.driver == "process" {
        let mut d = ProcessDriver::new()?;
        d.rules_file = write_demo_rules(&d.workdir)?;
        let nats = d.nats_bin.clone();
        return Ok((Box::new(d), Some(nats)));
    }
    let extra: Vec<(String, String)> = std::env::var("MYCELIC_BUILD_CA_BUNDLE")
        .ok()
        .filter(|v| !v.is_empty())
        .map(|v| vec![("CA_BUNDLE_FILE".to_string(), v)])
        .unwrap_or_default();
    Ok((Box::new(ComposeDriver::new(None, "mycelic-live", !args.no_build, &extra)?), None))
}

fn nats_version(nats_bin: Option<&Path>) -> Option<String> {
    let out = Command::new(nats_bin?).arg("--version").stdin(Stdio::null()).output().ok()?;
    String::from_utf8_lossy(&out.stdout).split_whitespace().last().map(String::from)
}

/// Open the console in the default browser; a machine without one just does not get a tab.
fn open_browser(url: &str) {
    let opener = if cfg!(target_os = "macos") { "open" } else { "xdg-open" };
    if let Ok(mut child) =
        Command::new(opener).arg(url).stdin(Stdio::null()).stdout(Stdio::null()).stderr(Stdio::null()).spawn()
    {
        // reap it, so it does not linger as a zombie child of the demo
        std::thread::spawn(move || child.wait());
    }
}

/// The first line of an error, at most 200 characters.
fn one_line(e: &anyhow::Error) -> String {
    let text = format!("{e:#}");
    let first = text.trim().lines().next().unwrap_or("").to_string();
    let first = if first.is_empty() { "error".to_string() } else { first };
    if first.chars().count() <= 200 { first } else { format!("{}…", first.chars().take(199).collect::<String>()) }
}

fn export(args: &Args) -> u8 {
    let trace_path = args.trace.clone().unwrap_or_else(default_trace_path);
    let trace = match load_trace(&trace_path) {
        Ok(t) => t,
        Err(msg) => {
            eprintln!("{msg}");
            return 1;
        }
    };
    for (path, fragment) in [(&args.export, false), (&args.export_fragment, true)] {
        let Some(path) = path else { continue };
        let written = build_page(Some(&trace), fragment, &console_path()).and_then(|page| {
            if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
                std::fs::create_dir_all(dir)?;
            }
            std::fs::write(path, page)?;
            Ok(std::fs::metadata(path)?.len())
        });
        match written {
            Ok(size) => println!(
                "wrote {} ({} KiB, {}, {} events)",
                path.display(),
                size / 1024,
                if fragment { "fragment" } else { "standalone page" },
                trace["events"].as_array().map_or(0, Vec::len)
            ),
            Err(e) => {
                eprintln!("error: cannot write {}: {e:#}", path.display());
                return 1;
            }
        }
    }
    0
}

fn serve_replay(args: &Args, path: &Path) -> u8 {
    let trace = match load_trace(path) {
        Ok(t) => t,
        Err(msg) => {
            eprintln!("{msg}");
            return 1;
        }
    };
    if !port_free(&args.host, args.port) {
        eprintln!("error: port {} on {} is in use: pass --port <another>", args.port, args.host);
        return 2;
    }
    if let Err(e) = signals::install() {
        eprintln!("warning: cannot handle stop signals: {e}");
    }
    let n_events = trace["events"].as_array().map_or(0, Vec::len);
    let recorded_at = py_str(&trace["meta"]["recorded_at"]);
    let mut server = match ConsoleServer::bind(&args.host, args.port, Source::Replay { trace: Arc::new(trace) }) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: {e:#}");
            return 2;
        }
    };
    server.start();
    let url = server.url().to_string();
    println!("replaying {} ({n_events} events, recorded {recorded_at}) at {url}", path.display());
    println!("  presenter notes: {url}?presenter   ·   manual advance: {url}#manual");
    if !args.no_browser {
        open_browser(&url);
    }
    let _ = signals::wait_or_interrupt(args.exit_after);
    signals::ignoring(|| server.stop());
    0
}

/// The engine's shared flags, read by the console server's `busy`/`refuse` callbacks.
#[derive(Default)]
struct RunFlags {
    busy: AtomicBool,
    failed: AtomicBool,
    acts_over: AtomicBool,
}

fn run_live(args: &Args) -> u8 {
    if let Err(e) = signals::install() {
        eprintln!("warning: cannot handle stop signals: {e}");
    }
    let recording = args.record.is_some();
    let (driver, nats_bin) = match make_driver(args) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("error: {e:#}");
            eprintln!("fix the above, {REPLAY_HINT}");
            return 2;
        }
    };
    let rules_source = format!(
        "deploy/mycelic/rules.json: {}",
        if args.driver == "process" { DEMO_RULES.join(", ") } else { "all rules".into() }
    );
    let mode = if recording { "record" } else { "present" };
    let rec = Arc::new(TraceRecorder::new(mode, driver.name(), args.poll, args.fast_poll, &rules_source));
    let version = nats_version(nats_bin.as_deref());
    rec.update(|tr| tr["meta"]["nats_server_version"] = json!(version));
    let workdir = match tempfile::Builder::new().prefix("mycelic-live-").tempdir() {
        Ok(w) => w,
        Err(e) => {
            eprintln!("error: cannot create a working directory: {e}");
            return 2;
        }
    };
    let mut demo = match LiveDemo::new(driver, Arc::clone(&rec), workdir.path(), args.spread, args.poll, args.fast_poll)
    {
        Ok(d) => d,
        Err(e) => {
            eprintln!("error: {e:#}");
            return 2;
        }
    };
    let control = Arc::new(ControlQueue::default());
    let flags = Arc::new(RunFlags::default());
    let mut server: Option<ConsoleServer> = None;
    let mut code = 2;
    let outcome = live_session(args, &mut demo, &rec, &control, &flags, &mut server, &mut code);
    let code = match outcome {
        Ok(c) => c,
        // a finished run stopped with Ctrl-C keeps its result
        Err(Interrupted) if rec.done() => code,
        Err(Interrupted) => {
            println!("\ninterrupted: stopping the stack");
            130
        }
    };
    signals::ignoring(|| {
        demo.shutdown();
        demo.down();
        // the agents' local files hold the private notes
        let _ = workdir.close();
        if let Some(mut s) = server.take() {
            s.stop();
        }
    });
    code
}

/// The live run proper; `Err` when a stop signal ended it.  `code` is the result once the run is complete.
fn live_session(
    args: &Args,
    demo: &mut LiveDemo,
    rec: &Arc<TraceRecorder>,
    control: &Arc<ControlQueue>,
    flags: &Arc<RunFlags>,
    server: &mut Option<ConsoleServer>,
    code: &mut u8,
) -> Result<u8, Interrupted> {
    let recording = args.record.is_some();
    if !recording {
        let busy: BusyFn = {
            let flags = Arc::clone(flags);
            Arc::new(move || flags.busy.load(Ordering::SeqCst))
        };
        let refuse: RefuseFn = {
            let (flags, rec, auto) = (Arc::clone(flags), Arc::clone(rec), args.auto);
            Arc::new(move || {
                if auto {
                    Some("auto mode: the acts advance by themselves".into())
                } else if rec.done() || flags.acts_over.load(Ordering::SeqCst) {
                    Some("the run is complete".into())
                } else if flags.failed.load(Ordering::SeqCst) {
                    Some("the run stopped".into())
                } else {
                    None
                }
            })
        };
        let source = Source::Live { recorder: Arc::clone(rec), control: Arc::clone(control), busy, refuse };
        let mut s = match ConsoleServer::bind(&args.host, args.port, source) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("error: {e:#}");
                return Ok(2);
            }
        };
        s.start();
        let how = if args.auto {
            format!("auto, {} s per act", py_float(args.dwell))
        } else {
            "advance from the console".into()
        };
        println!("Mycelic live console: {}  ({how})", s.url());
        println!("  presenter notes: {}?presenter", s.url());
        if !args.no_browser {
            open_browser(s.url());
        }
        *server = Some(s);
    }

    let acts = (|| -> Result<()> {
        for index in 0..ACT_COUNT {
            if index > 0 && !recording {
                if args.auto {
                    signals::sleep(Duration::from_secs_f64(args.dwell.max(0.0)))?;
                } else {
                    while control.get_timeout(Duration::from_millis(500)).is_none() {
                        signals::checkpoint()?;
                    }
                }
            }
            flags.busy.store(true, Ordering::SeqCst);
            let ran = demo.run_act(index);
            flags.busy.store(false, Ordering::SeqCst);
            ran?;
        }
        Ok(())
    })();
    if let Err(e) = acts {
        // an error caused by the stop (the agents killed, the stack gone) is the interrupt, not a failure
        if signals::interrupted() || e.is::<Interrupted>() {
            return Err(Interrupted);
        }
        flags.failed.store(true, Ordering::SeqCst);
        let message = one_line(&e);
        rec.emit("error", json!({"message": message, "hint": REPLAY_COMMAND}));
        eprintln!("\nERROR: {message}");
        eprintln!("--- mycelic log tail ---\n{}", demo.driver.logs("mycelic", 40));
        eprintln!("\nthe live run stopped; {REPLAY_HINT}");
        if server.is_some() {
            // keep the console up: it shows the error and the replay command
            signals::ignoring(|| {
                demo.shutdown();
                demo.down();
            });
            eprintln!("the console stays up with the error: Ctrl-C to stop");
            signals::wait_or_interrupt(args.exit_after)?;
        }
        return Ok(2);
    }
    flags.acts_over.store(true, Ordering::SeqCst);
    // the stack is not needed any more; say so once it is down
    signals::ignoring(|| {
        demo.shutdown();
        demo.down();
    });
    rec.emit("service", json!({"state": "stopped", "detail": STOPPED_DETAIL}));
    rec.finish();
    let (passed, total, summary) = rec.read(|tr| {
        let checks = tr["checks"].as_array().map(Vec::as_slice).unwrap_or_default();
        (checks.iter().filter(|c| c["ok"] == true).count(), checks.len(), tr["summary"].clone())
    });
    println!("\n{passed}/{total} checks passed");
    let keys = ["agents", "notes_shared", "notes_private", "time_to_enterprise_s", "rebuild_s", "stream_messages"];
    let shown: Vec<String> =
        keys.iter().map(|k| format!("\"{k}\": {}", summary.get(*k).unwrap_or(&Value::Null))).collect();
    println!("summary: {{{}}}", shown.join(", "));
    *code = if total > 0 && passed == total { 0 } else { 1 };
    if let Some(path) = &args.record {
        let snap = rec.snapshot(true);
        let text = snap.to_string();
        let written = trace::assert_no_secrets(&text, &demo.secrets()).and_then(|()| {
            if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
                std::fs::create_dir_all(dir)?;
            }
            std::fs::write(path, format!("{text}\n"))?;
            Ok(())
        });
        if let Err(e) = written {
            eprintln!("error: {e:#}");
            return Ok(2);
        }
        let n_events = snap["events"].as_array().map_or(0, Vec::len);
        println!("trace written to {} ({} KiB, {n_events} events)", path.display(), text.chars().count() / 1024);
    } else {
        // the console stays up, showing the stopped stack
        let until = match args.exit_after {
            Some(s) => format!(" for {} s", py_float(s)),
            None => ": Ctrl-C to stop".into(),
        };
        println!("the run is complete and the stack is stopped; the console stays up{until}");
        signals::wait_or_interrupt(args.exit_after)?;
    }
    Ok(*code)
}
