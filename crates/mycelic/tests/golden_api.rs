//! The HTTP API and the MCP endpoint against the Python service: every request recorded in
//! `tests/golden/api.json` (aiohttp test client, pinned clock and ids) is sent to the Rust router and the status,
//! the headers that matter (content type, `WWW-Authenticate`, `Cache-Control`, `Allow`, MCP headers) and the
//! body are compared.  API keys, key prefixes, latencies and the database path are random or machine-specific
//! and were normalised on both sides.

mod common;

use axum::body::Body;
use axum::extract::ConnectInfo;
use axum::http::{HeaderName, HeaderValue, Request};
use common::{ADMIN_TOKEN, Harness, assert_json_eq, golden};
use mycelic::models::content_hash;
use serde_json::{Map, Value, json};
use std::collections::HashMap;
use std::net::SocketAddr;
use std::time::Duration;
use tower::ServiceExt;

const IGNORE_KEYS: [&str; 3] = ["api_key", "key_prefix", "latency_ms"];

/// The generator's `normalize`: ignored keys and paths replaced, MCP text content parsed.
fn normalize(v: &Value, path: &[String]) -> Value {
    let ignored_path = |p: &[String]| {
        let p: Vec<&str> = p.iter().map(String::as_str).collect();
        p == ["checks", "db", "path"] || p == ["settings", "db_path"]
    };
    match v {
        Value::Object(m) => {
            let mut out = Map::new();
            for (k, x) in m {
                let mut sub = path.to_vec();
                sub.push(k.clone());
                if IGNORE_KEYS.contains(&k.as_str()) || ignored_path(&sub) {
                    out.insert(k.clone(), json!("<ignored>"));
                } else if k == "text"
                    && path.last().map(String::as_str) == Some("content[]")
                    && x.as_str().is_some_and(|s| s.starts_with('{'))
                    && let Ok(parsed) = serde_json::from_str::<Value>(x.as_str().unwrap())
                {
                    out.insert(k.clone(), json!({"json": normalize(&parsed, &[])}));
                } else {
                    out.insert(k.clone(), normalize(x, &sub));
                }
            }
            Value::Object(out)
        }
        Value::Array(items) => {
            let sub: Vec<String> = match path.split_last() {
                Some((last, rest)) => {
                    let mut p = rest.to_vec();
                    p.push(format!("{last}[]"));
                    p
                }
                None => vec!["[]".into()],
            };
            Value::Array(items.iter().map(|x| normalize(x, &sub)).collect())
        }
        other => other.clone(),
    }
}

struct Runner {
    keys: HashMap<String, String>,
}

impl Runner {
    fn sub(&self, s: &str, h: &Harness) -> String {
        let mut s = s.replace("{admin}", ADMIN_TOKEN);
        for (who, key) in &self.keys {
            s = s.replace(&format!("{{key:{who}}}"), key);
        }
        while let Some(i) = s.find("{mem:") {
            let j = i + s[i..].find('}').unwrap();
            let (agent, idem) = s[i + 5..j].rsplit_once(':').unwrap();
            let id = format!("mem_{}", &content_hash(&[agent, idem])[..22]);
            s = format!("{}{id}{}", &s[..i], &s[j + 1..]);
        }
        while let Some(i) = s.find("{select:") {
            let j = i + s[i..].find('}').unwrap();
            let layer = s[i + 8..j].to_string();
            let id = h
                .service
                .store
                .read(|db| {
                    use rusqlite::OptionalExtension;
                    Ok(db
                        .c
                        .query_row(
                            "SELECT memory_id FROM memories WHERE layer=? AND status='active' ORDER BY rid LIMIT 1",
                            [layer],
                            |r| r.get::<_, String>(0),
                        )
                        .optional()?)
                })
                .unwrap()
                .unwrap_or_else(|| "none".into());
            s = format!("{}{id}{}", &s[..i], &s[j + 1..]);
        }
        s
    }

    fn walk(&self, v: &Value, h: &Harness) -> Value {
        match v {
            Value::String(s) => json!(self.sub(s, h)),
            Value::Array(a) => Value::Array(a.iter().map(|x| self.walk(x, h)).collect()),
            Value::Object(m) => Value::Object(m.iter().map(|(k, x)| (k.clone(), self.walk(x, h))).collect()),
            other => other.clone(),
        }
    }
}

async fn run_config(name: &str, cfg: &Value, header_names: &[String]) {
    let o = cfg["overrides"].clone();
    let h = Harness::start(true, |s| {
        if let Some(v) = o.get("rate_limit_rps").and_then(Value::as_f64) {
            s.rate_limit_rps = v;
        }
        if let Some(v) = o.get("rate_limit_burst").and_then(Value::as_i64) {
            s.rate_limit_burst = v;
        }
        if let Some(v) = o.get("trust_proxy_headers").and_then(Value::as_bool) {
            s.trust_proxy_headers = v;
        }
        if let Some(v) = o.get("trusted_proxy_hops").and_then(Value::as_i64) {
            s.trusted_proxy_hops = v;
        }
        if let Some(v) = o.get("allowed_hosts").and_then(Value::as_array) {
            s.allowed_hosts = v.iter().map(|x| x.as_str().unwrap().to_string()).collect();
        }
        if let Some(v) = o.get("cors_origins").and_then(Value::as_array) {
            s.cors_origins = v.iter().map(|x| x.as_str().unwrap().to_string()).collect();
        }
        if let Some(v) = o.get("metrics_token").and_then(Value::as_str) {
            s.metrics_token = Some(v.to_string());
        }
    })
    .await;
    let app = mycelic::api::create_app(h.service.clone());
    let mut runner = Runner { keys: HashMap::new() };
    let requests = cfg["requests"].as_array().unwrap();
    let responses = cfg["responses"].as_array().unwrap();
    for (i, (rq, want)) in requests.iter().zip(responses).enumerate() {
        let path = runner.sub(rq["path"].as_str().unwrap(), &h);
        let mut builder = Request::builder().method(rq["method"].as_str().unwrap()).uri(&path);
        let mut has_host = false;
        for (k, v) in rq.get("headers").and_then(Value::as_object).into_iter().flatten() {
            let v = runner.sub(v.as_str().unwrap(), &h);
            has_host |= k.eq_ignore_ascii_case("host");
            builder = builder
                .header(HeaderName::from_bytes(k.as_bytes()).unwrap(), HeaderValue::from_bytes(v.as_bytes()).unwrap());
        }
        if !has_host {
            builder = builder.header("host", "127.0.0.1:8080");
        }
        let body = if let Some(j) = rq.get("json") {
            if !rq["headers"].as_object().is_some_and(|m| m.keys().any(|k| k.eq_ignore_ascii_case("content-type"))) {
                builder = builder.header("content-type", "application/json");
            }
            Body::from(serde_json::to_vec(&runner.walk(j, &h)).unwrap())
        } else if let Some(d) = rq.get("data").and_then(Value::as_str) {
            Body::from(d.to_string())
        } else if let Some(rep) = rq.get("data_repeat") {
            Body::from(rep[0].as_str().unwrap().repeat(rep[1].as_u64().unwrap() as usize))
        } else {
            Body::empty()
        };
        let mut req = builder.body(body).unwrap();
        req.extensions_mut().insert(ConnectInfo(SocketAddr::from(([127, 0, 0, 1], 50000))));
        let resp = app.clone().oneshot(req).await.unwrap();
        let status = resp.status().as_u16();
        let headers = resp.headers().clone();
        let raw = axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        let ctx = format!("config {name}, request {i}: {} {path}", rq["method"]);
        assert_eq!(json!(status), want["status"], "{ctx}: status; body {}", String::from_utf8_lossy(&raw));
        if !rq.get("status_only").and_then(Value::as_bool).unwrap_or(false) {
            let mut got_headers = Map::new();
            for hn in header_names {
                if let Some(v) = headers.get(hn.as_str()) {
                    let v =
                        if hn == "mcp-session-id" { "<present>".to_string() } else { v.to_str().unwrap().to_string() };
                    got_headers.insert(hn.clone(), json!(v));
                }
            }
            assert_json_eq(&Value::Object(got_headers), &want["headers"], &format!("{ctx}: headers"));
            match serde_json::from_slice::<Value>(&raw) {
                Ok(v) if !raw.is_empty() => assert_json_eq(&normalize(&v, &[]), &want["json"], &format!("{ctx}: body")),
                _ if raw.is_empty() => assert_eq!(want["json"], Value::Null, "{ctx}: empty body"),
                _ => assert_eq!(json!(String::from_utf8_lossy(&raw)), want["text"], "{ctx}: text body"),
            }
        }
        if let Some(who) = rq.get("capture_key").and_then(Value::as_str)
            && (status == 200 || status == 201)
        {
            let v: Value = serde_json::from_slice(&raw).unwrap();
            runner.keys.insert(who.to_string(), v["api_key"].as_str().unwrap().to_string());
        }
        if rq.get("settle").and_then(Value::as_bool).unwrap_or(false) {
            assert!(h.service.wait_idle(Duration::from_secs(30)).await, "{ctx}: did not settle");
        }
    }
    h.close().await;
}

async fn run(name: &str) {
    let g = golden("api.json");
    let header_names: Vec<String> =
        g["header_names"].as_array().unwrap().iter().map(|v| v.as_str().unwrap().to_string()).collect();
    run_config(name, &g["configs"][name], &header_names).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn main_routes_and_mcp() {
    run("main").await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn rate_limit() {
    run("rate_limit").await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn rate_limit_mcp_batches() {
    run("rate_limit_mcp").await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn proxy_hops() {
    run("proxy_hops").await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn host_allow_list_cors_and_origins() {
    run("hosts").await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn metrics_token() {
    run("metrics_token").await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn metrics_on_loopback() {
    run("metrics_open").await;
}
