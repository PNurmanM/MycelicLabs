//! The console's HTTP side: `GET /` (console), `GET /trace`, `GET /events` (SSE), `POST /control`.
//!
//! One `tiny_http` server; every request is handled on its own thread because an SSE stream lasts as long as the
//! tab stays open.  The SSE responses are written straight to the socket (`Request::into_writer`) and flushed per
//! message, so a console sees each event as it is recorded.

use super::trace::{TraceRecorder, build_page, console_path};
use anyhow::{Result, anyhow};
use serde_json::{Value, json};
use std::collections::VecDeque;
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};
use tiny_http::{Header, Method, Request, Response};

/// How often an idle SSE stream gets a `: ping` comment (keeps proxies and the browser from timing it out).
const PING_EVERY: Duration = Duration::from_secs(15);
/// How long an SSE thread waits at most before it looks at the stop flag again.
const STOP_CHECK: Duration = Duration::from_millis(500);

/// The presenter's "next act" requests, from `POST /control` to the engine (Python's `queue.Queue`).
#[derive(Default)]
pub struct ControlQueue {
    items: Mutex<VecDeque<String>>,
    cv: Condvar,
}

impl ControlQueue {
    pub fn put(&self, item: &str) {
        self.items.lock().unwrap_or_else(|e| e.into_inner()).push_back(item.to_string());
        self.cv.notify_all();
    }

    pub fn try_get(&self) -> Option<String> {
        self.items.lock().unwrap_or_else(|e| e.into_inner()).pop_front()
    }

    /// The next item, waiting up to `timeout` for one.
    pub fn get_timeout(&self, timeout: Duration) -> Option<String> {
        let items = self.items.lock().unwrap_or_else(|e| e.into_inner());
        let (mut items, _) =
            self.cv.wait_timeout_while(items, timeout, |q| q.is_empty()).unwrap_or_else(|e| e.into_inner());
        items.pop_front()
    }

    pub fn is_empty(&self) -> bool {
        self.items.lock().unwrap_or_else(|e| e.into_inner()).is_empty()
    }
}

/// Whether an act is running (`POST /control` then answers `busy`).
pub type BusyFn = Arc<dyn Fn() -> bool + Send + Sync>;
/// Why `POST /control` cannot queue an act right now (`--auto`, run complete), or `None`.
pub type RefuseFn = Arc<dyn Fn() -> Option<String> + Send + Sync>;

/// What the console shows: a run being recorded, or a recorded trace.
pub enum Source {
    Live { recorder: Arc<TraceRecorder>, control: Arc<ControlQueue>, busy: BusyFn, refuse: RefuseFn },
    Replay { trace: Arc<Value> },
}

struct Shared {
    source: Source,
    console: PathBuf,
    stopping: AtomicBool,
}

pub struct ConsoleServer {
    http: Option<Arc<tiny_http::Server>>,
    shared: Arc<Shared>,
    thread: Option<JoinHandle<()>>,
    port: u16,
    url: String,
}

impl ConsoleServer {
    /// Bind `host:port` (port 0 picks a free one); requests wait until [`ConsoleServer::start`].
    pub fn bind(host: &str, port: u16, source: Source) -> Result<ConsoleServer> {
        let http = tiny_http::Server::http((host, port)).map_err(|e| anyhow!("cannot listen on {host}:{port}: {e}"))?;
        let port = http.server_addr().to_ip().map_or(port, |a| a.port());
        Ok(ConsoleServer {
            http: Some(Arc::new(http)),
            shared: Arc::new(Shared { source, console: console_path(), stopping: AtomicBool::new(false) }),
            thread: None,
            port,
            url: format!("http://{host}:{port}/"),
        })
    }

    pub fn url(&self) -> &str {
        &self.url
    }

    pub fn port(&self) -> u16 {
        self.port
    }

    pub fn start(&mut self) {
        let (Some(http), None) = (self.http.clone(), &self.thread) else { return };
        let shared = Arc::clone(&self.shared);
        self.thread = Some(
            std::thread::Builder::new()
                .name("mycelic-live-http".into())
                .spawn(move || {
                    while !shared.stopping.load(Ordering::SeqCst) {
                        if let Ok(Some(req)) = http.recv_timeout(Duration::from_millis(200)) {
                            let shared = Arc::clone(&shared);
                            let _ = std::thread::Builder::new()
                                .name("mycelic-live-request".into())
                                .spawn(move || handle(&shared, req));
                        }
                    }
                })
                .expect("spawning the console server thread"),
        );
    }

    /// Stop accepting, close the listener and end every SSE stream.
    pub fn stop(&mut self) {
        self.shared.stopping.store(true, Ordering::SeqCst);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
        self.http = None;
    }
}

impl Drop for ConsoleServer {
    fn drop(&mut self) {
        self.stop();
    }
}

fn header(name: &str, value: &str) -> Header {
    Header::from_bytes(name.as_bytes(), value.as_bytes()).expect("a valid header")
}

fn send(req: Request, status: u16, body: Vec<u8>, ctype: &str) {
    // always a Content-Length (tiny_http would switch large bodies to chunked encoding), as the Python server sent
    let resp = Response::from_data(body)
        .with_chunked_threshold(usize::MAX)
        .with_status_code(status)
        .with_header(header("Content-Type", ctype))
        .with_header(header("Cache-Control", "no-store"));
    // a client that went away is not an error
    let _ = req.respond(resp);
}

fn send_json(req: Request, status: u16, data: &Value) {
    send(req, status, data.to_string().into_bytes(), "application/json");
}

fn handle(shared: &Shared, req: Request) {
    let path = req.url().split('?').next().unwrap_or("").to_string();
    match req.method() {
        Method::Get => get(shared, req, &path),
        Method::Post => post(shared, req, &path),
        _ => send_json(req, 501, &json!({"error": "unsupported method"})),
    }
}

fn get(shared: &Shared, req: Request, path: &str) {
    match path {
        "/" | "/index.html" => {
            let trace = match &shared.source {
                Source::Replay { trace } => Some(trace.as_ref()),
                Source::Live { .. } => None,
            };
            match build_page(trace, false, &shared.console) {
                Ok(page) => send(req, 200, page.into_bytes(), "text/html; charset=utf-8"),
                Err(e) => send_json(req, 500, &json!({"error": format!("{e:#}")})),
            }
        }
        "/trace" => match &shared.source {
            Source::Replay { trace } => send_json(req, 200, trace),
            Source::Live { recorder, .. } => send_json(req, 200, &recorder.snapshot(true)),
        },
        "/events" => events(shared, req),
        "/favicon.ico" => send(req, 204, Vec::new(), "image/x-icon"),
        _ => send_json(req, 404, &json!({"error": "not found"})),
    }
}

fn post(shared: &Shared, mut req: Request, path: &str) {
    if path != "/control" {
        return send_json(req, 404, &json!({"error": "not found"}));
    }
    let mut raw = Vec::new();
    if req.as_reader().read_to_end(&mut raw).is_err() {
        return send_json(req, 400, &json!({"ok": false, "error": "body must be JSON"}));
    }
    let body: Value = if raw.is_empty() {
        json!({})
    } else {
        match serde_json::from_slice(&raw) {
            Ok(v) => v,
            Err(_) => return send_json(req, 400, &json!({"ok": false, "error": "body must be JSON"})),
        }
    };
    if !body.is_object() {
        return send_json(req, 400, &json!({"ok": false, "error": "body must be a JSON object"}));
    }
    let Source::Live { recorder, control, busy, refuse } = &shared.source else {
        return send_json(req, 409, &json!({"ok": false, "error": "replay mode: the console advances by itself"}));
    };
    if let Some(reason) = refuse().filter(|r| !r.is_empty()) {
        return send_json(req, 409, &json!({"ok": false, "error": reason}));
    }
    if body.get("action").and_then(Value::as_str) != Some("next") {
        return send_json(req, 400, &json!({"ok": false, "error": "unknown action; expected {\"action\": \"next\"}"}));
    }
    if busy() || !control.is_empty() {
        return send_json(req, 200, &json!({"ok": false, "busy": true, "act": recorder.act()}));
    }
    control.put("next");
    send_json(req, 200, &json!({"ok": true, "queued": "next", "act": recorder.act()}))
}

/// One SSE stream: a `snapshot`, every event after `Last-Event-ID`, then new ones as they are recorded.
fn events(shared: &Shared, req: Request) {
    let mut last: i64 = req
        .headers()
        .iter()
        .find(|h| h.field.equiv("Last-Event-ID"))
        .and_then(|h| h.value.as_str().trim().parse().ok())
        .unwrap_or(-1);
    let n_events = match &shared.source {
        Source::Replay { trace } => trace["events"].as_array().map_or(0, Vec::len),
        Source::Live { recorder, .. } => recorder.event_count(),
    };
    if last >= n_events as i64 {
        last = -1; // an id from another run (a tab left open across a restart)
    }
    let mut w = req.into_writer();
    // a write error means the tab went away: the stream just ends
    let _ = stream(shared, &mut *w, last);
}

fn write_snapshot(w: &mut dyn Write, snap: &Value) -> std::io::Result<()> {
    write!(w, "event: snapshot\ndata: {snap}\n\n")?;
    w.flush()
}

fn write_message(w: &mut dyn Write, index: i64, ev: &Value) -> std::io::Result<()> {
    write!(w, "id: {index}\ndata: {ev}\n\n")?;
    w.flush()
}

fn stream(shared: &Shared, w: &mut dyn Write, mut last: i64) -> std::io::Result<()> {
    w.write_all(
        b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-store\r\nConnection: keep-alive\r\n\r\n",
    )?;
    w.flush()?;
    let stopping = || shared.stopping.load(Ordering::SeqCst);
    let mut last_write = Instant::now();
    match &shared.source {
        Source::Replay { trace } => {
            let snap: serde_json::Map<String, Value> = trace
                .as_object()
                .map(|m| {
                    m.iter().filter(|(k, _)| k.as_str() != "events").map(|(k, v)| (k.clone(), v.clone())).collect()
                })
                .unwrap_or_default();
            write_snapshot(w, &Value::Object(snap))?;
            for (i, ev) in trace["events"].as_array().into_iter().flatten().enumerate() {
                if i as i64 > last {
                    write_message(w, i as i64, ev)?;
                }
            }
            while !stopping() {
                std::thread::sleep(STOP_CHECK);
                if last_write.elapsed() >= PING_EVERY {
                    w.write_all(b": ping\n\n")?;
                    w.flush()?;
                    last_write = Instant::now();
                }
            }
        }
        Source::Live { recorder, .. } => {
            let mut revision = None;
            while !stopping() {
                let wait = PING_EVERY.saturating_sub(last_write.elapsed()).min(STOP_CHECK);
                let (events, rev) = recorder.wait_events(last, revision, wait);
                let mut wrote = false;
                if revision != Some(rev) {
                    revision = Some(rev);
                    write_snapshot(w, &recorder.snapshot(false))?;
                    wrote = true;
                }
                for ev in &events {
                    last += 1;
                    write_message(w, last, ev)?;
                    wrote = true;
                }
                if wrote {
                    last_write = Instant::now();
                } else if last_write.elapsed() >= PING_EVERY {
                    w.write_all(b": ping\n\n")?;
                    w.flush()?;
                    last_write = Instant::now();
                }
            }
        }
    }
    Ok(())
}
