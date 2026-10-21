//! The live demo engine: a headless recording against the real stack, the replay pages built from a trace, and
//! the console server (SSE, control, preflight, cleanup).
//!
//! The recording and interrupt tests run `mycelic-live` with the process driver (skipped without nats-server, like
//! the other process tests); the page and server tests need no stack.  Child processes are listed with
//! `ps -A -o pid=,ppid=,stat=`, so the cleanup tests run on macOS and Linux alike.

use mycelic_harness::live::{BusyFn, ConsoleServer, ControlQueue, RefuseFn, Source, TraceRecorder};
use regex::Regex;
use serde_json::{Value, json};
use std::collections::{HashMap, HashSet};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Output, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock, mpsc};
use std::time::{Duration, Instant};

const BIN: &str = env!("CARGO_BIN_EXE_mycelic-live");
const EVENT_TYPES: [&str; 15] = [
    "service",
    "agent_registered",
    "note_shared",
    "note_private",
    "derived",
    "status_change",
    "retracted",
    "query",
    "lineage",
    "rebuild_progress",
    "metric",
    "check",
    "act_start",
    "act_end",
    "done",
];

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn committed_trace() -> PathBuf {
    root().join("demo/live/trace.json")
}

fn read_json(path: &Path) -> Value {
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn tmpdir(prefix: &str) -> tempfile::TempDir {
    tempfile::Builder::new().prefix(prefix).tempdir().unwrap()
}

/// A completed run: the close act ends, the engine stops the stack and says so once, and `done` comes last.
fn assert_ends_stopped_then_done(events: &[Value]) {
    let tail: Vec<(Value, Value, Value)> = events[events.len().saturating_sub(3)..]
        .iter()
        .map(|e| (e["type"].clone(), e["act"].clone(), e.get("state").cloned().unwrap_or(Value::Null)))
        .collect();
    assert_eq!(
        tail,
        [
            (json!("act_end"), json!("close"), Value::Null),
            (json!("service"), json!("close"), json!("stopped")),
            (json!("done"), json!("close"), Value::Null)
        ]
    );
    let stopped: Vec<&Value> = events.iter().filter(|e| e["type"] == "service" && e["state"] == "stopped").collect();
    assert_eq!(stopped.len(), 1, "the stack is stopped once, after the run");
    assert!(stopped[0]["detail"].as_str().is_some_and(|d| !d.is_empty()));
    assert_eq!(events.iter().filter(|e| e["type"] == "done").count(), 1);
    assert!(!events.iter().any(|e| e["type"] == "error"));
}

fn embedded_trace(page: &str) -> Value {
    let re = Regex::new(r#"(?s)<script id="mycelic-trace" type="application/json">(.*?)</script>"#).unwrap();
    let m = re.captures(page).expect("no embedded trace");
    serde_json::from_str(&m[1]).unwrap()
}

/// A maximal run of exactly 64 lowercase hex characters (the admin token, the signing key).
fn has_hex64(text: &str) -> bool {
    let mut run = 0;
    for b in text.bytes().chain(std::iter::once(b' ')) {
        if b.is_ascii_digit() || (b'a'..=b'f').contains(&b) {
            run += 1;
        } else {
            if run == 64 {
                return true;
            }
            run = 0;
        }
    }
    false
}

/// Anything in a trace that looks like a credential: an agent key, a 64-hex token, or a credential-named field.
fn secret_findings(text: &str) -> Vec<String> {
    let mut found = Vec::new();
    if text.contains("mk_") {
        found.push("agent api key prefix mk_".to_string());
    }
    if has_hex64(text) {
        found.push("64-hex token (admin token / signing key)".into());
    }
    for key in ["api_key", "admin_token", "password", "signing_key", "Authorization"] {
        if text.contains(&format!("\"{key}\"")) {
            found.push(format!("field {key}"));
        }
    }
    found
}

fn run_demo(args: &[&str], env: &[(&str, &str)]) -> Output {
    let mut cmd = Command::new(BIN);
    cmd.args(args).current_dir(root()).env_remove("MYCELIC_NATS_SERVER_BIN");
    for (k, v) in env {
        cmd.env(k, v);
    }
    cmd.output().unwrap()
}

// ---------------------------------------------------------------------------------------------------- HTTP
/// One request on its own connection; status and body.
fn http(port: u16, method: &str, path: &str, body: &[u8], headers: &[(&str, &str)]) -> (u16, Vec<u8>) {
    let mut s = TcpStream::connect(("127.0.0.1", port)).unwrap();
    s.set_read_timeout(Some(Duration::from_secs(30))).unwrap();
    let mut req = format!(
        "{method} {path} HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\nContent-Length: {}\r\n",
        body.len()
    );
    for (k, v) in headers {
        req.push_str(&format!("{k}: {v}\r\n"));
    }
    req.push_str("\r\n");
    s.write_all(req.as_bytes()).unwrap();
    s.write_all(body).unwrap();
    let mut raw = Vec::new();
    s.read_to_end(&mut raw).unwrap();
    let split = raw.windows(4).position(|w| w == b"\r\n\r\n").expect("a complete response");
    let head = String::from_utf8_lossy(&raw[..split]).to_string();
    let status = head.split_whitespace().nth(1).unwrap().parse().unwrap();
    (status, raw[split + 4..].to_vec())
}

fn get_json(port: u16, path: &str) -> Value {
    serde_json::from_slice(&http(port, "GET", path, b"", &[]).1).unwrap()
}

fn post_control(port: u16, body: &[u8]) -> (u16, Value) {
    let (status, body) = http(port, "POST", "/control", body, &[("Content-Type", "application/json")]);
    (status, serde_json::from_slice(&body).unwrap())
}

/// A minimal Server-Sent Events reader.
struct Sse {
    reader: BufReader<TcpStream>,
}

impl Sse {
    fn open(port: u16, last_event_id: Option<i64>, timeout: Duration) -> Sse {
        let mut s = TcpStream::connect(("127.0.0.1", port)).unwrap();
        s.set_read_timeout(Some(timeout)).unwrap();
        let last = last_event_id.map(|id| format!("Last-Event-ID: {id}\r\n")).unwrap_or_default();
        write!(s, "GET /events HTTP/1.1\r\nHost: 127.0.0.1\r\nAccept: text/event-stream\r\n{last}\r\n").unwrap();
        let mut reader = BufReader::new(s);
        let mut line = String::new();
        reader.read_line(&mut line).unwrap();
        assert!(line.starts_with("HTTP/1.1 200"), "{line}");
        loop {
            line.clear();
            reader.read_line(&mut line).unwrap();
            if line.trim_end().is_empty() {
                break;
            }
        }
        Sse { reader }
    }

    /// The next `n` messages (comments skipped): {"event", "id", "data"}.
    fn messages(&mut self, n: usize) -> Vec<HashMap<String, Value>> {
        let mut out = Vec::new();
        let mut msg = HashMap::new();
        while out.len() < n {
            let mut line = String::new();
            assert!(self.reader.read_line(&mut line).unwrap() > 0, "the stream ended");
            let line = line.trim_end_matches('\n');
            if line.starts_with(':') {
                continue;
            }
            if line.is_empty() {
                if !msg.is_empty() {
                    out.push(std::mem::take(&mut msg));
                }
                continue;
            }
            let (key, value) = line.split_once(": ").unwrap_or((line, ""));
            let value = if key == "data" { serde_json::from_str(value).unwrap() } else { json!(value) };
            msg.insert(key.to_string(), value);
        }
        out
    }
}

// ---------------------------------------------------------------------------------------------------- processes
/// Build the service and agent binaries (same profile as this test) once; the directory they are in.
fn service_binaries() -> PathBuf {
    static DIR: OnceLock<PathBuf> = OnceLock::new();
    DIR.get_or_init(|| {
        let cargo = std::env::var("CARGO").unwrap_or_else(|_| env!("CARGO").to_string());
        let mut cmd = Command::new(cargo);
        cmd.args(["build", "-p", "mycelic", "-p", "mycelic-sdk", "--bins"]).current_dir(root());
        if BIN.contains("/release/") {
            cmd.arg("--release");
        }
        assert!(cmd.status().unwrap().success(), "building the mycelic binaries failed");
        Path::new(BIN).parent().unwrap().to_path_buf()
    })
    .clone()
}

/// The nats-server the process tests use, or `None` to skip them.
fn nats_available() -> bool {
    let env = std::env::var("MYCELIC_NATS_SERVER_BIN").ok().filter(|v| !v.is_empty());
    env.is_some()
        || std::env::var_os("PATH").is_some_and(|p| std::env::split_paths(&p).any(|d| d.join("nats-server").is_file()))
}

macro_rules! needs_nats {
    () => {
        if !nats_available() {
            eprintln!("skipped: nats-server binary not available (set MYCELIC_NATS_SERVER_BIN)");
            return;
        }
    };
}

/// pid -> (ppid, state) for every process.
fn process_table() -> HashMap<u32, (u32, String)> {
    let out = Command::new("ps").args(["-A", "-o", "pid=,ppid=,stat="]).output().unwrap();
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter_map(|l| {
            let mut f = l.split_whitespace();
            Some((f.next()?.parse().ok()?, (f.next()?.parse().ok()?, f.next().unwrap_or("").to_string())))
        })
        .collect()
}

/// Every live descendant of `pid`.
fn children(pid: u32) -> HashSet<u32> {
    let table = process_table();
    let mut out = HashSet::new();
    let mut parents = HashSet::from([pid]);
    while !parents.is_empty() {
        let found: HashSet<u32> = table
            .iter()
            .filter(|(p, (ppid, stat))| parents.contains(ppid) && !out.contains(*p) && !stat.starts_with('Z'))
            .map(|(p, _)| *p)
            .collect();
        out.extend(&found);
        parents = found;
    }
    out
}

fn alive(pid: u32) -> bool {
    process_table().get(&pid).is_some_and(|(_, stat)| !stat.starts_with('Z'))
}

fn sigterm(pid: u32) {
    assert!(Command::new("kill").args(["-TERM", &pid.to_string()]).status().unwrap().success());
}

fn wait_exit(child: &mut Child, timeout: Duration) -> Option<i32> {
    let deadline = Instant::now() + timeout;
    while Instant::now() < deadline {
        if let Some(status) = child.try_wait().unwrap() {
            return status.code();
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    panic!("the demo did not exit within {timeout:?}");
}

/// A running `mycelic-live` that serves its console; killed if a test fails.
struct Demo {
    child: Child,
    port: u16,
}

impl Demo {
    fn start(args: &[&str]) -> Demo {
        let mut child = Command::new(BIN)
            .args(args)
            .current_dir(root())
            .env("MYCELIC_BIN_DIR", service_binaries())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .unwrap();
        // keep draining stdout so the demo never blocks on a full pipe; hand over the console URL
        let (tx, rx) = mpsc::channel();
        let stdout = child.stdout.take().unwrap();
        std::thread::spawn(move || {
            let re = Regex::new(r"http://127\.0\.0\.1:(\d+)/").unwrap();
            for line in BufReader::new(stdout).lines().map_while(Result::ok) {
                if let Some(m) = re.captures(&line) {
                    let _ = tx.send(m[1].parse::<u16>().unwrap());
                }
            }
        });
        let port = rx.recv_timeout(Duration::from_secs(60));
        let mut demo = Demo { child, port: 0 };
        demo.port = port.expect("the console URL was not printed");
        demo
    }

    fn pid(&self) -> u32 {
        self.child.id()
    }
}

impl Drop for Demo {
    /// A failed test still lets the demo clean up (SIGTERM), so no stack is left behind; SIGKILL only as a last resort.
    fn drop(&mut self) {
        if self.child.try_wait().ok().flatten().is_none() {
            let _ = Command::new("kill").args(["-TERM", &self.child.id().to_string()]).status();
            let deadline = Instant::now() + Duration::from_secs(30);
            while Instant::now() < deadline && self.child.try_wait().ok().flatten().is_none() {
                std::thread::sleep(Duration::from_millis(100));
            }
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
    }
}

// ---------------------------------------------------------------------------------------------------- tests
/// No stack needed: pages built from the committed trace.
mod trace_page_tests {
    use super::*;

    fn export(args: &[&str]) -> Output {
        let trace = committed_trace();
        let mut all: Vec<&str> = args.to_vec();
        all.extend(["--trace", trace.to_str().unwrap()]);
        run_demo(&all, &[])
    }

    #[test]
    fn test_committed_trace_is_complete_and_clean() {
        let text = std::fs::read_to_string(committed_trace()).unwrap();
        assert!(text.len() < 2_000_000);
        assert_eq!(secret_findings(&text), Vec::<String>::new());
        let tr: Value = serde_json::from_str(&text).unwrap();
        assert_eq!(tr["schema"], "mycelic-demo-trace/1");
        assert_eq!(tr["meta"]["fictional"], true);
        let checks = tr["checks"].as_array().unwrap();
        let failed: Vec<&Value> = checks.iter().filter(|c| c["ok"] != true).collect();
        assert!(!checks.is_empty() && failed.is_empty(), "{failed:?}");
        assert_eq!(tr["summary"]["checks_passed"], tr["summary"]["checks_total"]);
        let events = tr["events"].as_array().unwrap();
        assert!(events.iter().all(|e| EVENT_TYPES.contains(&e["type"].as_str().unwrap())));
        let times: Vec<f64> = events.iter().map(|e| e["t"].as_f64().unwrap()).collect();
        assert!(times.windows(2).all(|w| w[0] <= w[1]));
        assert_ends_stopped_then_done(events);
    }

    #[test]
    fn test_fragment_has_no_skeleton_and_starts_with_the_title() {
        let tmp = tmpdir("mycelic-live-page-");
        let out = tmp.path().join("fragment.html");
        let r = export(&["--export-fragment", out.to_str().unwrap()]);
        assert!(r.status.success(), "{}", String::from_utf8_lossy(&r.stderr));
        let page = std::fs::read_to_string(&out).unwrap();
        let skeleton = Regex::new(r"(?i)<!doctype|<html[\s>]|</html>|<head[\s>]|</head>|<body[\s>]|</body>").unwrap();
        assert!(skeleton.find(&page).is_none(), "{:?}", skeleton.find(&page));
        assert!(page.trim_start().starts_with("<title>"), "{}", &page[..80]);
        assert!(page.contains("</title>"));
        assert_eq!(embedded_trace(&page)["summary"], read_json(&committed_trace())["summary"]);
    }

    #[test]
    fn test_standalone_page_has_skeleton_and_trace() {
        let tmp = tmpdir("mycelic-live-page-");
        let out = tmp.path().join("page.html");
        let r = export(&["--export", out.to_str().unwrap()]);
        assert!(r.status.success(), "{}", String::from_utf8_lossy(&r.stderr));
        let page = std::fs::read_to_string(&out).unwrap();
        assert!(page.starts_with("<!doctype html>"));
        assert!(page.contains("<meta charset=\"utf-8\">"));
        assert!(page.contains("name=\"viewport\""));
        let head_end = page.find("</head>").unwrap();
        assert!(page.find("<title>").unwrap() < head_end, "the fragment's title belongs in <head>");
        assert!(page.find("<style>").unwrap() < head_end);
        assert_eq!(
            embedded_trace(&page)["events"].as_array().unwrap().len(),
            read_json(&committed_trace())["events"].as_array().unwrap().len()
        );
    }
}

/// No stack needed: the HTTP side of the console against a recorder with a few events.
mod console_server_tests {
    use super::*;

    struct Fixture {
        rec: Arc<TraceRecorder>,
        busy: Arc<AtomicBool>,
        refusal: Arc<Mutex<Option<String>>>,
        control: Arc<ControlQueue>,
        server: ConsoleServer,
    }

    fn fixture() -> Fixture {
        let rec = Arc::new(TraceRecorder::new("present", "process", 0.25, 0.02, "test"));
        for i in 0..3 {
            rec.emit("metric", json!({"n": i}));
        }
        let busy = Arc::new(AtomicBool::new(false));
        let refusal = Arc::new(Mutex::new(None::<String>));
        let control = Arc::new(ControlQueue::default());
        let busy_fn: BusyFn = {
            let busy = Arc::clone(&busy);
            Arc::new(move || busy.load(Ordering::SeqCst))
        };
        let refuse_fn: RefuseFn = {
            let refusal = Arc::clone(&refusal);
            Arc::new(move || refusal.lock().unwrap().clone())
        };
        let source = Source::Live {
            recorder: Arc::clone(&rec),
            control: Arc::clone(&control),
            busy: busy_fn,
            refuse: refuse_fn,
        };
        let mut server = ConsoleServer::bind("127.0.0.1", 0, source).unwrap();
        server.start();
        Fixture { rec, busy, refusal, control, server }
    }

    #[test]
    fn test_snapshot_first_then_events_with_their_index() {
        let f = fixture();
        let mut sse = Sse::open(f.server.port(), None, Duration::from_secs(10));
        let msgs = sse.messages(4);
        let (snap, events) = (&msgs[0], &msgs[1..]);
        assert_eq!(snap["event"], "snapshot");
        assert!(snap["data"].get("events").is_none());
        assert_eq!(snap["data"]["meta"]["run_id"], f.rec.snapshot(false)["meta"]["run_id"]);
        assert_eq!(events.iter().map(|e| e["id"].clone()).collect::<Vec<_>>(), [json!("0"), json!("1"), json!("2")]);
        assert_eq!(events.iter().map(|e| e["data"]["n"].clone()).collect::<Vec<_>>(), [json!(0), json!(1), json!(2)]);
        f.rec.emit("metric", json!({"n": 3})); // new events follow as they happen
        let live = &sse.messages(1)[0];
        assert_eq!((live["id"].clone(), live["data"]["n"].clone()), (json!("3"), json!(3)));
    }

    #[test]
    fn test_last_event_id_resumes_and_an_id_past_the_end_starts_over() {
        let f = fixture();
        let mut resumed = Sse::open(f.server.port(), Some(1), Duration::from_secs(10));
        assert_eq!(resumed.messages(2)[1]["id"], "2");
        // a tab left open across a restart
        let mut stale = Sse::open(f.server.port(), Some(200), Duration::from_secs(10));
        assert_eq!(stale.messages(2)[1]["id"], "0");
    }

    #[test]
    fn test_control_replies() {
        let f = fixture();
        let port = f.server.port();
        assert_eq!(post_control(port, b"not json").0, 400);
        assert_eq!(post_control(port, b"[1, 2]").0, 400);
        assert_eq!(post_control(port, br#"{"action": "jump"}"#).0, 400);
        f.busy.store(true, Ordering::SeqCst);
        let (status, body) = post_control(port, br#"{"action": "next"}"#);
        assert_eq!((status, body["busy"].clone()), (200, json!(true)));
        f.busy.store(false, Ordering::SeqCst);
        *f.refusal.lock().unwrap() = Some("the run is complete".into());
        let (status, body) = post_control(port, br#"{"action": "next"}"#);
        assert_eq!((status, body["error"].clone()), (409, json!("the run is complete")));
        *f.refusal.lock().unwrap() = None;
        let (status, body) = post_control(port, br#"{"action": "next"}"#);
        assert_eq!((status, body["ok"].clone()), (200, json!(true)));
        assert_eq!(f.control.try_get().as_deref(), Some("next"));
    }

    #[test]
    fn test_replay_server_refuses_control_and_serves_the_trace() {
        let trace = read_json(&committed_trace());
        let n = trace["events"].as_array().unwrap().len();
        let mut server = ConsoleServer::bind("127.0.0.1", 0, Source::Replay { trace: Arc::new(trace) }).unwrap();
        server.start();
        let port = server.port();
        assert_eq!(http(port, "POST", "/control", br#"{"action": "next"}"#, &[]).0, 409);
        let (status, page) = http(port, "GET", "/", b"", &[]);
        assert_eq!(status, 200);
        assert_eq!(embedded_trace(&String::from_utf8(page).unwrap())["events"].as_array().unwrap().len(), n);
    }
}

mod preflight_tests {
    use super::*;

    #[test]
    fn test_missing_nats_server_is_reported_with_the_replay_command() {
        let r = run_demo(&["--record", "/dev/null"], &[("PATH", "")]);
        let stderr = String::from_utf8_lossy(&r.stderr);
        assert_eq!(r.status.code(), Some(2), "{stderr}");
        assert!(stderr.contains("nats-server not found"), "{stderr}");
        assert!(stderr.contains("--replay"), "{stderr}");
    }

    #[test]
    fn test_a_directory_is_not_a_nats_server() {
        let dir = std::env::temp_dir();
        let r = run_demo(&["--record", "/dev/null"], &[("MYCELIC_NATS_SERVER_BIN", dir.to_str().unwrap())]);
        let stderr = String::from_utf8_lossy(&r.stderr);
        assert_eq!(r.status.code(), Some(2), "{stderr}");
        assert!(stderr.contains("nats-server not found at"), "{stderr}");
    }

    #[test]
    fn test_a_bad_trace_is_one_line_not_a_traceback() {
        let r = run_demo(&["--replay", "/nonexistent/trace.json", "--no-browser"], &[]);
        let stderr = String::from_utf8_lossy(&r.stderr);
        assert!(!r.status.success());
        assert!(!stderr.contains("panicked"), "{stderr}");
        assert!(stderr.contains("cannot read the trace"), "{stderr}");
        assert_eq!(stderr.trim().lines().count(), 1, "{stderr}");
    }
}

mod live_record_process_tests {
    use super::*;

    #[test]
    fn test_record_passes_every_check_and_exports() {
        needs_nats!();
        let tmp = tmpdir("mycelic-live-record-");
        let trace_path = tmp.path().join("trace.json");
        let r = Command::new(BIN)
            .args(["--record", trace_path.to_str().unwrap(), "--spread", "1"])
            .current_dir(root())
            .env("MYCELIC_BIN_DIR", service_binaries())
            .output()
            .unwrap();
        assert_eq!(
            r.status.code(),
            Some(0),
            "{}\n{}",
            String::from_utf8_lossy(&r.stdout),
            String::from_utf8_lossy(&r.stderr)
        );
        let text = std::fs::read_to_string(&trace_path).unwrap();
        assert_eq!(secret_findings(&text), Vec::<String>::new());
        let tr: Value = serde_json::from_str(&text).unwrap();
        assert_eq!(tr["schema"], "mycelic-demo-trace/1");
        for key in ["meta", "org", "rules", "acts", "events", "checks", "summary"] {
            assert!(tr.get(key).is_some(), "{key}");
        }
        let acts = tr["acts"].as_array().unwrap();
        assert_eq!(acts.iter().map(|a| a["index"].as_u64().unwrap()).collect::<Vec<_>>(), (0..9).collect::<Vec<_>>());
        assert!(acts.iter().all(|a| !a["t0"].is_null() && !a["t1"].is_null()));
        let rules: HashSet<&str> =
            tr["rules"].as_array().unwrap().iter().map(|r| r["rule_id"].as_str().unwrap()).collect();
        assert_eq!(rules, HashSet::from(["regional_supply_risk", "strategic_second_source"]));
        assert_eq!(json!(tr["org"]["agents"].as_array().unwrap().len()), tr["summary"]["agents"]);
        let checks: HashMap<&str, &Value> =
            tr["checks"].as_array().unwrap().iter().map(|c| (c["id"].as_str().unwrap(), c)).collect();
        let failed: Vec<&&Value> = checks.values().filter(|c| c["ok"] != true).collect();
        assert!(failed.is_empty(), "{failed:?}");
        for id in
            ["private_stayed_local", "time_to_enterprise", "background_no_rule", "redacted_view", "rebuild_identical"]
        {
            assert!(checks.contains_key(id), "{id}");
        }
        let events = tr["events"].as_array().unwrap();
        let types: HashSet<&str> = events.iter().map(|e| e["type"].as_str().unwrap()).collect();
        assert!(types.iter().all(|t| EVENT_TYPES.contains(t)), "{types:?}");
        for t in ["note_shared", "note_private", "derived", "lineage", "retracted", "rebuild_progress", "done"] {
            assert!(types.contains(t), "{t}");
        }
        assert_ends_stopped_then_done(events);
        let s = &tr["summary"];
        assert_eq!(s["background_rule_conclusions"], 0);
        assert!(s["notes_private"].as_i64().unwrap() > 0);
        assert!(!s["time_to_enterprise_s"].is_null());
        // the private notes the agents kept are nowhere in what the service handed out
        let private: HashSet<&str> =
            events.iter().filter(|e| e["type"] == "note_private").map(|e| e["text"].as_str().unwrap()).collect();
        let shared: Vec<&str> = events
            .iter()
            .filter(|e| e["type"] == "note_shared" || e["type"] == "derived")
            .map(|e| e["text"].as_str().unwrap())
            .collect();
        let leaked: Vec<&&str> = private.iter().filter(|p| shared.iter().any(|t| t.contains(**p))).collect();
        assert!(leaked.is_empty(), "{leaked:?}");

        let page = tmp.path().join("page.html");
        let r = run_demo(&["--export", page.to_str().unwrap(), "--trace", trace_path.to_str().unwrap()], &[]);
        assert!(r.status.success(), "{}", String::from_utf8_lossy(&r.stderr));
        assert_eq!(&embedded_trace(&std::fs::read_to_string(&page).unwrap())["summary"], s);
    }

    /// --auto: an SSE client sees the stack stop after the close act, then `done`; the stack is really gone.
    #[test]
    fn test_auto_run_tells_the_console_the_stack_stopped() {
        needs_nats!();
        let mut demo = Demo::start(&[
            "--auto",
            "--dwell",
            "0",
            "--spread",
            "1",
            "--no-browser",
            "--port",
            "0",
            "--exit-after",
            "20",
        ]);
        let mut sse = Sse::open(demo.port, None, Duration::from_secs(60));
        let mut events: Vec<Value> = Vec::new();
        let mut kids = HashSet::new();
        let deadline = Instant::now() + Duration::from_secs(240);
        while events.last().is_none_or(|e| e["type"] != "done") {
            assert!(Instant::now() < deadline, "the run never finished");
            let msg = sse.messages(1).remove(0);
            if msg.get("event").is_some_and(|e| e == "snapshot") {
                continue;
            }
            let ev = msg["data"].clone();
            if ev["type"] == "act_start" && ev["act"] == "close" {
                kids = children(demo.pid()); // the stack is still up during the close act
            }
            events.push(ev);
        }
        assert!(!kids.is_empty(), "no nats-server/service processes found during the close act");
        assert_ends_stopped_then_done(&events);
        let left: Vec<&u32> = kids.iter().filter(|k| alive(**k)).collect();
        assert!(left.is_empty(), "the stack still runs after the stopped event: {left:?}");
        // the console is still up, with the same end
        assert_ends_stopped_then_done(get_json(demo.port, "/trace")["events"].as_array().unwrap());
        sigterm(demo.pid()); // a finished run stopped early keeps its result
        assert_eq!(wait_exit(&mut demo.child, Duration::from_secs(60)), Some(0));
    }

    #[test]
    fn test_sigterm_mid_act_leaves_no_process_behind() {
        needs_nats!();
        let mut demo = Demo::start(&["--no-browser", "--port", "0", "--spread", "4"]);
        let port = demo.port;
        let act_running = |act: &str| {
            get_json(port, "/trace")["events"]
                .as_array()
                .unwrap()
                .iter()
                .any(|e| e["type"] == "act_start" && e["act"] == act)
        };
        // act 0 runs by itself; queue acts 1 and 2
        for act in ["organisation", "observe"] {
            let deadline = Instant::now() + Duration::from_secs(90);
            while post_control(port, br#"{"action": "next"}"#).1["ok"] != true {
                assert!(Instant::now() < deadline, "the engine never accepted the next act");
                std::thread::sleep(Duration::from_millis(300));
            }
            while !act_running(act) {
                assert!(Instant::now() < deadline, "act {act} never started");
                std::thread::sleep(Duration::from_millis(200));
            }
        }
        std::thread::sleep(Duration::from_millis(1500)); // agents are sharing now
        let kids = children(demo.pid());
        assert!(!kids.is_empty(), "no nats-server/service/agent processes found");
        sigterm(demo.pid());
        assert_eq!(wait_exit(&mut demo.child, Duration::from_secs(60)), Some(130));
        std::thread::sleep(Duration::from_millis(500));
        let left: Vec<&u32> = kids.iter().filter(|k| alive(**k)).collect();
        assert!(left.is_empty(), "processes left behind after SIGTERM: {left:?}");
    }
}
