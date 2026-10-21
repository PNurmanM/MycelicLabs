//! The trace being recorded, the replay pages built from a trace, and the secret guard.

use super::scenario::{COMPANY, ENTERPRISE, ENTITY, STRATEGIC_QUERY, SUBJECT};
use anyhow::{Context, Result, bail};
use regex::Regex;
use serde_json::{Map, Value, json};
use std::path::{Path, PathBuf};
use std::sync::{Condvar, Mutex, MutexGuard, OnceLock};
use std::time::{Duration, Instant};

pub const SCHEMA: &str = "mycelic-demo-trace/1";
pub const TRACE_MARKER: &str = "<!-- mycelic:trace -->";

/// `demo/live/console.html`: the console fragment the server and the exports wrap.
pub fn console_path() -> PathBuf {
    crate::repo_root().join("demo/live/console.html")
}

/// `demo/live/trace.json`: the committed recording `--replay` and `--export` use by default.
pub fn default_trace_path() -> PathBuf {
    crate::repo_root().join("demo/live/trace.json")
}

/// Seconds rounded to milliseconds, the precision of every `t` in a trace.
fn round3(x: f64) -> f64 {
    (x * 1000.0).round() / 1000.0
}

struct State {
    act: Option<String>,
    done: bool,
    revision: u64,
    trace: Value,
}

/// The trace being recorded, plus a condition variable the SSE handlers wait on.
pub struct TraceRecorder {
    t0: Instant,
    state: Mutex<State>,
    cv: Condvar,
}

impl TraceRecorder {
    pub fn new(mode: &str, driver: &str, poll: f64, fast_poll: f64, rules_source: &str) -> TraceRecorder {
        let trace = json!({
            "schema": SCHEMA,
            "meta": {
                "recorded_at": chrono::Utc::now().format("%Y-%m-%dT%H:%M:%S+00:00").to_string(),
                "mycelic_version": null, "driver": driver, "nats_server_version": null,
                // the recording engine is this crate now, not a Python interpreter
                "python": null, "rust": env!("CARGO_PKG_VERSION"), "implementation": "rust",
                "platform": format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH),
                "duration_s": null, "fictional": true, "company": COMPANY, "mode": mode, "poll_interval_s": poll,
                "fast_poll_interval_s": fast_poll, "rules_source": rules_source, "run_id": crate::token_hex(6),
                "story": {"entity": ENTITY, "label": SUBJECT, "question": STRATEGIC_QUERY},
            },
            "org": {"enterprise": ENTERPRISE, "units": [], "agents": []},
            "rules": [], "acts": [], "events": [], "checks": [], "summary": {},
        });
        TraceRecorder {
            t0: Instant::now(),
            state: Mutex::new(State { act: None, done: false, revision: 0, trace }),
            cv: Condvar::new(),
        }
    }

    fn lock(&self) -> MutexGuard<'_, State> {
        // a panicking SSE thread must not take the recording down with it
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Seconds since the run started, to the millisecond.
    pub fn now(&self) -> f64 {
        round3(self.t0.elapsed().as_secs_f64())
    }

    /// The act now running, if any.
    pub fn act(&self) -> Option<String> {
        self.lock().act.clone()
    }

    /// The run completed (`done` was emitted).
    pub fn done(&self) -> bool {
        self.lock().done
    }

    /// Append one event stamped now; returns it.
    pub fn emit(&self, type_: &str, payload: Value) -> Value {
        self.emit_at(type_, None, payload)
    }

    /// Append one event; `at` backdates it to when it was observed (the watcher's read), else it is now.
    pub fn emit_at(&self, type_: &str, at: Option<f64>, payload: Value) -> Value {
        let mut st = self.lock();
        let mut t = at.unwrap_or_else(|| self.now());
        let events = st.trace["events"].as_array().map(Vec::as_slice).unwrap_or_default();
        if let Some(prev) = events.last().and_then(|e| e["t"].as_f64()) {
            t = t.max(prev); // keep the event list ordered by time
        }
        let mut ev = Map::new();
        ev.insert("t".into(), json!(t));
        ev.insert("act".into(), json!(st.act));
        ev.insert("type".into(), json!(type_));
        if let Value::Object(fields) = payload {
            ev.extend(fields);
        }
        let ev = Value::Object(ev);
        if let Some(events) = st.trace["events"].as_array_mut() {
            events.push(ev.clone());
        }
        self.cv.notify_all();
        ev
    }

    /// Change a non-event part of the trace (org, rules, meta, summary); SSE clients get a new snapshot.
    pub fn update(&self, f: impl FnOnce(&mut Value)) {
        let mut st = self.lock();
        f(&mut st.trace);
        st.revision += 1;
        self.cv.notify_all();
    }

    /// Read the trace without copying it.
    pub fn read<T>(&self, f: impl FnOnce(&Value) -> T) -> T {
        f(&self.lock().trace)
    }

    pub fn act_start(&self, index: usize) {
        {
            let mut st = self.lock();
            let now = self.now();
            let id = st.trace["acts"][index]["id"].as_str().map(String::from);
            st.act = id;
            st.trace["acts"][index]["t0"] = json!(now);
            st.revision += 1;
        }
        self.emit("act_start", json!({}));
    }

    pub fn act_end(&self, index: usize) {
        self.emit("act_end", json!({}));
        let now = self.now();
        self.update(|tr| tr["acts"][index]["t1"] = json!(now));
    }

    /// Record a verified claim of the run (the trace's `checks` and a `check` event); returns `ok`.
    pub fn check(&self, check_id: &str, text: &str, ok: bool) -> bool {
        let act = self.act();
        self.update(|tr| {
            if let Some(checks) = tr["checks"].as_array_mut() {
                checks.push(json!({"id": check_id, "act": act, "text": text, "ok": ok}));
            }
        });
        self.emit("check", json!({"id": check_id, "text": text, "ok": ok}));
        ok
    }

    /// Stamp the duration, emit `done` and mark the run complete.
    pub fn finish(&self) {
        let now = self.now();
        self.update(|tr| tr["meta"]["duration_s"] = json!(now));
        self.emit("done", json!({}));
        self.lock().done = true;
        self.cv.notify_all();
    }

    /// A copy of the trace, with or without its events.
    pub fn snapshot(&self, events: bool) -> Value {
        let st = self.lock();
        match &st.trace {
            Value::Object(m) => Value::Object(
                m.iter()
                    .filter(|(k, _)| events || k.as_str() != "events")
                    .map(|(k, v)| (k.clone(), v.clone()))
                    .collect(),
            ),
            other => other.clone(),
        }
    }

    pub fn event_count(&self) -> usize {
        self.lock().trace["events"].as_array().map_or(0, Vec::len)
    }

    /// Events with index > `after` and the current revision, waiting up to `timeout` for either to move
    /// (`revision: None` never matches, so the first call returns at once).
    pub fn wait_events(&self, after: i64, revision: Option<u64>, timeout: Duration) -> (Vec<Value>, u64) {
        let start = usize::try_from(after + 1).unwrap_or(0);
        let st = self.lock();
        let (st, _) = self
            .cv
            .wait_timeout_while(st, timeout, |st| {
                st.trace["events"].as_array().map_or(0, Vec::len) <= start && Some(st.revision) == revision
            })
            .unwrap_or_else(|e| e.into_inner());
        let events = st.trace["events"].as_array().map(|e| e.get(start..).unwrap_or_default().to_vec());
        (events.unwrap_or_default(), st.revision)
    }
}

/// JSON that is safe inside a `<script>` element: no `<` survives, so neither `</script>` nor `<!--` can.
pub fn html_json(data: &Value) -> String {
    data.to_string().replace('<', "\\u003c")
}

/// Refuse a trace that contains any of `secrets` or an agent key (`mk_` prefix).
pub fn assert_no_secrets(text: &str, secrets: &[String]) -> Result<()> {
    let leaked = secrets.iter().filter(|s| !s.is_empty() && text.contains(s.as_str())).count();
    let api_key = text.contains("mk_");
    if leaked > 0 || api_key {
        let prefix = if api_key { "True" } else { "False" };
        bail!("refusing to write a trace that contains a secret ({leaked} found, api-key prefix: {prefix})");
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------------------- page
/// The fragment with the trace embedded at the marker (or before the first `<script`, or at the end).
pub fn inject_trace(fragment: &str, trace: &Value) -> String {
    let tag = format!(r#"<script id="mycelic-trace" type="application/json">{}</script>"#, html_json(trace));
    if fragment.contains(TRACE_MARKER) {
        return fragment.replacen(TRACE_MARKER, &tag, 1);
    }
    match fragment.find("<script") {
        Some(i) => format!("{}{tag}\n{}", &fragment[..i], &fragment[i..]),
        None => format!("{fragment}\n{tag}"),
    }
}

/// The console page: the fragment, with the trace embedded when given, wrapped in a skeleton unless `fragment`.
pub fn build_page(trace: Option<&Value>, fragment: bool, console: &Path) -> Result<String> {
    let mut body =
        std::fs::read_to_string(console).with_context(|| format!("reading the console page {}", console.display()))?;
    if let Some(trace) = trace {
        body = inject_trace(&body, trace);
    }
    if fragment {
        return Ok(body);
    }
    // the fragment's leading title + style belong in <head>
    static HEAD: OnceLock<Regex> = OnceLock::new();
    let head_re =
        HEAD.get_or_init(|| Regex::new(r"(?s)\A\s*<title>.*?</title>\s*<style>.*?</style>\s*").expect("regex"));
    let mut head = String::new();
    if let Some(m) = head_re.find(&body) {
        head = format!("{}\n", m.as_str().trim());
        body = body[m.end()..].to_string();
    }
    Ok(format!(
        "<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n\
         <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n{head}</head>\n<body>\n{body}\n</body>\n</html>\n"
    ))
}

/// Python's `repr` of what a non-trace file holds, for the one-line error.
fn schema_repr(trace: &Value) -> String {
    let quoted = |s: &str| format!("'{s}'");
    match trace {
        Value::Object(m) => match m.get("schema") {
            None | Some(Value::Null) => "None".into(),
            Some(Value::String(s)) => quoted(s),
            Some(other) => other.to_string(),
        },
        Value::Array(_) => quoted("list"),
        Value::String(_) => quoted("str"),
        Value::Bool(_) => quoted("bool"),
        Value::Null => quoted("NoneType"),
        Value::Number(n) => quoted(if n.is_f64() { "float" } else { "int" }),
    }
}

/// Read a trace, or the one line saying what is wrong with it (`error: ...`).
pub fn load_trace(path: &Path) -> std::result::Result<Value, String> {
    let shown = path.display();
    let text = std::fs::read_to_string(path).map_err(|e| {
        // the OS message without Rust's " (os error N)" suffix, like Python's strerror
        let msg = e.to_string();
        let msg = msg.split(" (os error").next().unwrap_or(&msg).to_string();
        format!("error: cannot read the trace {shown}: {msg}")
    })?;
    let trace: Value = serde_json::from_str(&text).map_err(|e| {
        let msg = e.to_string();
        let msg = msg.split(" at line ").next().unwrap_or(&msg).to_string();
        format!("error: {shown} is not JSON ({msg}, line {})", e.line())
    })?;
    if trace.get("schema").and_then(Value::as_str) != Some(SCHEMA) {
        return Err(format!("error: {shown}: not a {SCHEMA} trace (schema: {})", schema_repr(&trace)));
    }
    Ok(trace)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn events_stay_ordered_and_carry_the_act() {
        let rec = TraceRecorder::new("present", "process", 0.25, 0.02, "test");
        rec.update(|tr| tr["acts"] = json!([{"id": "cold-open", "t0": null, "t1": null}]));
        rec.act_start(0);
        rec.emit_at("metric", Some(100.0), json!({"n": 1}));
        let late = rec.emit_at("metric", Some(0.0), json!({"n": 2}));
        assert_eq!(late["t"], json!(100.0), "a backdated event never goes before the previous one");
        assert_eq!(late["act"], "cold-open");
        assert!(rec.check("c", "text", true));
        let snap = rec.snapshot(false);
        assert!(snap.get("events").is_none());
        assert_eq!(snap["checks"][0], json!({"id": "c", "act": "cold-open", "text": "text", "ok": true}));
        let full = rec.snapshot(true);
        let keys: Vec<&String> = full["events"][1].as_object().unwrap().keys().collect();
        assert_eq!(keys, ["t", "act", "type", "n"]);
    }

    #[test]
    fn injected_trace_cannot_close_its_script_tag() {
        let page = inject_trace("<title>x</title><script>1</script>", &json!({"text": "</script><!--"}));
        assert!(page.contains(r#"{"text":"\u003c/script>\u003c!--"}</script>"#));
        assert!(page.starts_with("<title>x</title><script id=\"mycelic-trace\""));
        assert_eq!(
            inject_trace("a <!-- mycelic:trace --> b", &json!(1)),
            "a <script id=\"mycelic-trace\" type=\"application/json\">1</script> b"
        );
    }

    #[test]
    fn the_guard_refuses_keys_and_secrets() {
        assert!(assert_no_secrets("{\"a\": 1}", &["s3cret".into(), String::new()]).is_ok());
        let err = assert_no_secrets("x s3cret mk_abc", &["s3cret".into()]).unwrap_err().to_string();
        assert_eq!(err, "refusing to write a trace that contains a secret (1 found, api-key prefix: True)");
    }
}
