//! Port of tests/mycelic/test_api.py: authentication, authorization, limits, health, metrics, memory/query/lineage
//! routes, admin routes and the per-request MCP identity, through the HTTP router.

mod common;

use axum::body::Body;
use axum::extract::ConnectInfo;
use axum::http::{HeaderMap, Request};
use common::{ADMIN_TOKEN, Harness, demo_rule};
use mycelic::config::Settings;
use serde_json::{Value, json};
use std::collections::BTreeSet;
use std::net::SocketAddr;
use tower::ServiceExt;

struct Api {
    h: Harness,
    app: axum::Router,
}

struct Resp {
    status: u16,
    headers: HeaderMap,
    raw: Vec<u8>,
}

impl Resp {
    fn json(&self) -> Value {
        serde_json::from_slice(&self.raw).unwrap_or(Value::Null)
    }
    fn text(&self) -> String {
        String::from_utf8_lossy(&self.raw).into_owned()
    }
}

fn bearer(token: &str) -> Vec<(&'static str, String)> {
    vec![("authorization", format!("Bearer {token}"))]
}

impl Api {
    async fn new(tweak: impl FnOnce(&mut Settings)) -> Api {
        let h = Harness::start(false, tweak).await;
        let app = mycelic::api::create_app(h.service.clone());
        Api { h, app }
    }

    async fn call(
        &self,
        method: &str,
        path: &str,
        headers: &[(&str, String)],
        body: Option<Body>,
        content_type: Option<&str>,
    ) -> Resp {
        let mut b = Request::builder().method(method).uri(path);
        if !headers.iter().any(|(k, _)| k.eq_ignore_ascii_case("host")) {
            b = b.header("host", "127.0.0.1:8080");
        }
        for (k, v) in headers {
            b = b.header(*k, axum::http::HeaderValue::from_bytes(v.as_bytes()).unwrap());
        }
        if let Some(ct) = content_type {
            b = b.header("content-type", ct);
        }
        let mut req = b.body(body.unwrap_or_else(Body::empty)).unwrap();
        req.extensions_mut().insert(ConnectInfo(SocketAddr::from(([127, 0, 0, 1], 40000))));
        let resp = self.app.clone().oneshot(req).await.unwrap();
        let status = resp.status().as_u16();
        let headers = resp.headers().clone();
        let raw = axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap().to_vec();
        Resp { status, headers, raw }
    }

    async fn get(&self, path: &str, headers: &[(&str, String)]) -> Resp {
        self.call("GET", path, headers, None, None).await
    }

    async fn post(&self, path: &str, body: Value, headers: &[(&str, String)]) -> Resp {
        self.call("POST", path, headers, Some(Body::from(serde_json::to_vec(&body).unwrap())), Some("application/json"))
            .await
    }

    async fn delete(&self, path: &str, headers: &[(&str, String)]) -> Resp {
        self.call("DELETE", path, headers, None, None).await
    }

    fn admin(&self) -> Vec<(&'static str, String)> {
        bearer(ADMIN_TOKEN)
    }

    async fn register(&self, agent_id: &str, units: Value) -> String {
        let mut body = json!({"enterprise": "northwind", "region": "emea", "subsidiary": "nw-gmbh", "department": "ops",
                              "team": "logistics", "agent_id": agent_id});
        for (k, v) in units.as_object().unwrap() {
            body[k] = v.clone();
        }
        let r = self.post("/admin/agents", body, &self.admin()).await;
        assert_eq!(r.status, 201, "{}", r.text());
        let key = r.json()["api_key"].as_str().unwrap().to_string();
        assert!(key.starts_with(&format!("mk_{agent_id}.")));
        key
    }

    async fn rpc(&self, key: Option<&str>, method: &str, params: Option<Value>) -> Resp {
        let mut headers = vec![("accept", "application/json".to_string())];
        if let Some(k) = key {
            headers.extend(bearer(k));
        }
        let mut payload = json!({"jsonrpc": "2.0", "id": 1, "method": method});
        if let Some(p) = params {
            payload["params"] = p;
        }
        self.post("/mcp", payload, &headers).await
    }
}

// ------------------------------------------------------------------------------------------------ auth and limits
#[tokio::test(flavor = "multi_thread")]
async fn test_unauthenticated_and_malformed_requests() {
    let a = Api::new(|_| {}).await;
    let r = a.post("/memory", json!({"text": "x"}), &[]).await;
    assert_eq!(r.status, 401);
    assert!(r.headers["www-authenticate"].to_str().unwrap().contains("Bearer"));
    let r = a.call("POST", "/memory", &bearer("mk_a.b"), Some(Body::from("text=1")), Some("text/plain")).await;
    assert_eq!(r.status, 415);
    assert_eq!(a.get("/whoami", &bearer("mk_nobody.secret")).await.status, 401);
    assert_eq!(a.get("/whoami", &bearer("é")).await.status, 401, "non-ASCII tokens are rejected, not crashed on");
    assert_eq!(a.get("/admin/agents", &a.admin()).await.status, 200);
    let big = "x".repeat(a.h.settings.max_body_bytes as usize + 10);
    let r = a.call("POST", "/memory", &a.admin(), Some(Body::from(big)), Some("application/json")).await;
    assert_eq!(r.status, 413);
    a.h.close().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn test_admin_cannot_write_and_agents_cannot_administer() {
    let a = Api::new(|_| {}).await;
    let key = a.register("log-1", json!({})).await;
    assert_eq!(a.post("/memory", json!({"text": "admin note"}), &a.admin()).await.status, 403);
    assert_eq!(
        a.post("/admin/agents", json!({"enterprise": "northwind", "agent_id": "evil"}), &bearer(&key)).await.status,
        403
    );
    assert_eq!(
        a.post("/admin/agents", json!({"enterprise": "northwind", "agent_id": "x", "scopes": ["admin"]}), &a.admin())
            .await
            .status,
        400
    );
    assert_eq!(
        a.post("/admin/agents", json!({"enterprise": "northwind", "agent_id": "log-1"}), &a.admin()).await.status,
        400
    );
    a.h.close().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn test_health_ready_metrics() {
    let a = Api::new(|_| {}).await;
    let r = a.get("/health", &[]).await;
    assert_eq!(r.status, 200);
    let keys: BTreeSet<String> = r.json().as_object().unwrap().keys().cloned().collect();
    let public: BTreeSet<String> =
        ["status", "version", "transport_connected", "consumer_running"].iter().map(|s| s.to_string()).collect();
    assert_eq!(keys, public);
    let full = a.get("/health", &a.admin()).await.json();
    assert!(full.get("checks").is_some() && full.get("stats").is_some());
    let r = a.get("/ready", &[]).await;
    assert_eq!(r.status, 200);
    assert_eq!(r.json()["ready"], true);
    assert_eq!(a.get("/admin/status", &a.admin()).await.json()["settings"]["admin_token"], "set");
    let r = a.get("/metrics", &[]).await;
    assert_eq!(r.status, 200, "loopback bind: metrics scrapable without a token");
    let text = r.text();
    assert!(text.contains("mycelic_memories{layer=\"agent\"}"));
    assert!(text.contains("mycelic_http_requests_total"));
    assert!(text.contains("mycelic_consumer_pending 0"));
    // a wrong token on the public /health still yields the public view and is counted
    let failures = || a.h.service.metrics.auth_failures.with_label_values(&["invalid_token"]).get();
    let before = failures();
    let r = a.get("/health", &bearer("mk_nobody.wrong")).await;
    let keys: BTreeSet<String> = r.json().as_object().unwrap().keys().cloned().collect();
    assert_eq!(keys, public);
    assert_eq!(failures(), before + 1);
    a.h.close().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn test_rotate_and_revoke() {
    let a = Api::new(|_| {}).await;
    let key = a.register("log-1", json!({})).await;
    let new_key = a.post("/admin/agents/log-1/rotate", json!({}), &a.admin()).await.json()["api_key"]
        .as_str()
        .unwrap()
        .to_string();
    assert_eq!(a.get("/whoami", &bearer(&key)).await.status, 401);
    assert_eq!(a.get("/whoami", &bearer(&new_key)).await.status, 200);
    assert_eq!(a.delete("/admin/agents/log-1", &a.admin()).await.status, 200);
    assert_eq!(a.get("/whoami", &bearer(&new_key)).await.status, 403);
    a.h.close().await;
}

// ------------------------------------------------------------------------------------------------ rate limits
fn limited(s: &mut Settings) {
    s.rate_limit_rps = 1.0;
    s.rate_limit_burst = 3;
    s.trust_proxy_headers = true;
}

#[tokio::test(flavor = "multi_thread")]
async fn test_per_principal_and_per_peer_buckets() {
    let a = Api::new(limited).await;
    let mut statuses = Vec::new();
    for _ in 0..5 {
        statuses.push(a.get("/whoami", &a.admin()).await.status);
    }
    assert!(statuses.contains(&429));
    a.h.close().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn test_forwarded_for_uses_the_proxy_observed_entry() {
    let a = Api::new(limited).await;
    // the leftmost entry is client-chosen; rotating it must not buy a fresh bucket
    let mut statuses = Vec::new();
    for i in 0..6 {
        let mut h = bearer("mk_nobody.x");
        h.push(("x-forwarded-for", format!("10.0.0.{i}, 203.0.113.9")));
        statuses.push(a.get("/whoami", &h).await.status);
    }
    assert!(statuses.contains(&429));
    assert_eq!(statuses[..3], [401, 401, 401]);
    a.h.close().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn test_mcp_batch_is_capped_and_charged() {
    let a = Api::new(limited).await;
    let key = a.register("log-1", json!({})).await;
    let big: Vec<Value> =
        (0..=a.h.settings.max_batch).map(|i| json!({"jsonrpc": "2.0", "id": i, "method": "ping"})).collect();
    let mut h = bearer(&key);
    h.push(("accept", "application/json".into()));
    let r = a.post("/mcp", json!(big), &h).await;
    assert_eq!(r.status, 400);
    assert_eq!(r.json()["error"]["code"], -32600);
    // a fresh principal (each request from a different proxy-observed address so the address buckets do not
    // interfere): burst 3 covers a 2-message batch plus one request, the next request is refused
    let mut admin_from = a.admin();
    admin_from.push(("x-forwarded-for", "198.51.100.9".into()));
    let r = a
        .post("/admin/agents", json!({"enterprise": "northwind", "region": "emea", "subsidiary": "nw-gmbh", "department": "ops", "team": "logistics", "agent_id": "log-2"}), &admin_from)
        .await;
    let key2 = r.json()["api_key"].as_str().unwrap().to_string();
    let xff = |n: u8| {
        let mut h = bearer(&key2);
        h.push(("accept", "application/json".into()));
        h.push(("x-forwarded-for", format!("198.51.100.{n}")));
        h
    };
    let batch = json!([{"jsonrpc": "2.0", "id": 1, "method": "ping"}, {"jsonrpc": "2.0", "id": 2, "method": "ping"}]);
    assert_eq!(a.post("/mcp", batch, &xff(1)).await.status, 200);
    assert_eq!(a.get("/whoami", &xff(2)).await.status, 200);
    assert_eq!(a.get("/whoami", &xff(3)).await.status, 429);
    a.h.close().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn test_probes_bypass_the_host_allow_list_but_api_routes_do_not() {
    let a = Api::new(|s| s.allowed_hosts = vec!["mycelic.example.com".into()]).await;
    let probe = || vec![("host", "127.0.0.1:8080".to_string())];
    assert_eq!(a.get("/health", &probe()).await.status, 200);
    assert_eq!(a.get("/ready", &probe()).await.status, 200);
    let mut h = probe();
    h.extend(a.admin());
    assert_eq!(a.get("/whoami", &h).await.status, 421);
    let mut h = vec![("host", "mycelic.example.com".to_string())];
    h.extend(a.admin());
    assert_eq!(a.get("/whoami", &h).await.status, 200);
    a.h.close().await;
}

// ------------------------------------------------------------------------------------------------ scopes
#[tokio::test(flavor = "multi_thread")]
async fn test_scopes_are_enforced_per_route() {
    let a = Api::new(|_| {}).await;
    let writer = a.register("w-1", json!({})).await;
    let mid = a.post("/memory", json!({"text": "note", "topic": "t"}), &bearer(&writer)).await.json()["memory_id"]
        .as_str()
        .unwrap()
        .to_string();
    let eo = a.register("events-only", json!({"scopes": ["events:write"]})).await;
    assert_eq!(a.get(&format!("/memory/{mid}"), &bearer(&eo)).await.status, 403);
    assert_eq!(a.get("/memories?scope=northwind", &bearer(&eo)).await.status, 403);
    assert_eq!(a.post("/query", json!({"query": "note"}), &bearer(&eo)).await.status, 403);
    let mut h = bearer(&eo);
    h.push(("accept", "application/json".into()));
    let r = a
        .post("/mcp", json!({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "mycelic_get_memory", "arguments": {"memory_id": mid}}}), &h)
        .await;
    assert_eq!(r.json()["result"]["isError"], true);
    let ro = a.register("read-only", json!({"scopes": ["memory:read"]})).await;
    assert_eq!(a.get(&format!("/lineage/{mid}"), &bearer(&ro)).await.status, 403);
    let r = a.post("/query", json!({"query": "note"}), &bearer(&ro)).await;
    assert_eq!(r.status, 200);
    assert!(r.json()["lineage"].is_null(), "no lineage graph without lineage:read");
    let r = a
        .post("/admin/agents", json!({"enterprise": "northwind", "agent_id": "no-scopes", "scopes": []}), &a.admin())
        .await;
    assert_eq!(r.status, 201);
    assert_eq!(r.json()["agent"]["scopes"], json!([]));
    let k = r.json()["api_key"].as_str().unwrap().to_string();
    assert_eq!(a.post("/memory", json!({"text": "x"}), &bearer(&k)).await.status, 403);
    a.h.close().await;
}

// ------------------------------------------------------------------------------------------------ memory routes
struct Org {
    a: Api,
    log1: String,
    log2: String,
    proc1: String,
    sales1: String,
    sales2: String,
}

async fn org() -> Org {
    let a = Api::new(|_| {}).await;
    let r = a.post("/admin/rules", demo_rule(), &a.admin()).await;
    assert_eq!(r.status, 201, "{}", r.text());
    let log1 = a.register("log-1", json!({})).await;
    let log2 = a.register("log-2", json!({})).await;
    let proc1 = a.register("proc-1", json!({"team": "procurement"})).await;
    let sales1 = a.register("sales-1", json!({"department": "commercial", "team": "field-sales"})).await;
    let sales2 = a.register("sales-2", json!({"department": "commercial", "team": "field-sales"})).await;
    Org { a, log1, log2, proc1, sales1, sales2 }
}

impl Org {
    async fn observe(&self, key: &str, text: &str, fields: Value) -> Value {
        let mut body = fields;
        body["text"] = json!(text);
        let r = self.a.post("/memory", body, &bearer(key)).await;
        assert_eq!(r.status, 202, "{}", r.text());
        r.json()
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn test_memory_query_lineage_and_visibility() {
    let o = org().await;
    let a = &o.a;
    let first = o
        .observe(&o.log1, "Port of Rotterdam terminal 3 strike announced for weeks 41-43.",
            json!({"topic": "supply:sd-9/transport", "slot": "transport_disruption", "entity": "sd-9", "confidence": 0.9}))
        .await;
    o.observe(
        &o.log2,
        "Carrier ETA for the SD-9 container slipped by 12 days.",
        json!({"topic": "supply:sd-9/transport", "slot": "transport_disruption", "entity": "sd-9", "confidence": 0.7}),
    )
    .await;
    o.observe(
        &o.proc1,
        "Kessler Antriebe has two weeks of SD-9 inventory left.",
        json!({"topic": "supply:sd-9/supplier", "slot": "supplier_buffer_low", "entity": "sd-9"}),
    )
    .await;
    o.observe(
        &o.sales1,
        "Helios Automation committed to 40 RX-4 arms for November.",
        json!({"topic": "supply:sd-9/demand", "slot": "demand_commitment", "entity": "sd-9"}),
    )
    .await;
    a.h.settle().await;
    let mid = first["memory_id"].as_str().unwrap();

    // own memory is readable; another team's is not even acknowledged
    assert_eq!(a.get(&format!("/memory/{mid}"), &bearer(&o.log1)).await.status, 200);
    assert_eq!(a.get(&format!("/memory/{mid}"), &bearer(&o.sales2)).await.status, 404);
    assert_eq!(a.get(&format!("/memory/{mid}"), &a.admin()).await.status, 200);

    let r =
        a.post("/query", json!({"query": "delivery risk RX-4 SD-9", "scope": "northwind"}), &bearer(&o.sales2)).await;
    assert_eq!(r.status, 200, "{}", r.text());
    let res = r.json();
    assert_eq!(res["answer"]["layer"], "enterprise");
    assert_eq!(res["lineage"]["redacted_contributions"], 2);
    let top = &res["results"][0]["memory"];
    assert!(top["metadata"].get("contributing_agents").is_none(), "other teams' agent ids never leave through /query");
    assert!(top["metadata"].get("contributing_teams").is_some());
    assert_eq!(
        a.post("/query", json!({"query": "delivery risk", "scope": "northwind/emea/nw-gmbh/ops"}), &bearer(&o.sales2))
            .await
            .status,
        403
    );
    assert_eq!(
        a.post("/query", json!({"query": "delivery risk", "scope": "northwind/emea/nw-gmbh/ops"}), &a.admin())
            .await
            .status,
        200
    );

    let g =
        a.get(&format!("/lineage/{}", res["answer"]["memory_id"].as_str().unwrap()), &bearer(&o.sales2)).await.json();
    assert_eq!(g["evidence"]["reconstructable"], true);
    assert_eq!(g["layers"], json!(["agent", "enterprise"]));
    assert_eq!(a.get("/lineage/mem_doesnotexist", &bearer(&o.sales2)).await.status, 404);
    let r = a.get("/memories?scope=northwind&layer=enterprise", &bearer(&o.sales2)).await;
    assert_eq!(r.json()["memories"].as_array().unwrap().len(), 1);
    let scopes = |r: &Resp| -> BTreeSet<String> {
        r.json()["memories"].as_array().unwrap().iter().map(|m| m["scope"].as_str().unwrap().to_string()).collect()
    };
    let seen = scopes(&a.get("/memories?scope=northwind&limit=500", &bearer(&o.sales2)).await);
    assert!(seen.contains("northwind"));
    assert!(seen.contains("northwind/emea/nw-gmbh/commercial/field-sales/sales-1"));
    assert!(
        !seen.iter().any(|s| s.contains("/ops/")),
        "the SQL visibility rule must hide other teams' notes and consolidations"
    );
    let all = scopes(&a.get("/memories?scope=northwind&limit=500", &a.admin()).await);
    assert!(all.iter().any(|s| s.contains("/ops/")));

    // retract through the API: only the producer (or admin)
    assert_eq!(
        a.post(&format!("/memory/{mid}/retract"), json!({"reason": "strike called off"}), &bearer(&o.log2))
            .await
            .status,
        403
    );
    assert_eq!(
        a.post(&format!("/memory/{mid}/retract"), json!({"reason": "strike called off"}), &bearer(&o.log1))
            .await
            .status,
        202
    );
    a.h.settle().await;
    assert_eq!(a.get(&format!("/memory/{mid}"), &bearer(&o.log1)).await.json()["memory"]["status"], "retracted");
    o.a.h.close().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn test_events_validation_and_size() {
    let o = org().await;
    let a = &o.a;
    let r = a
        .post("/events", json!({"events": [{"type": "call", "memory": {"text": "Kessler stock is low", "slot": "supplier_buffer_low"}}]}), &bearer(&o.proc1))
        .await;
    assert_eq!(r.status, 202, "{}", r.text());
    assert!(r.json()["results"][0].get("memory_id").is_some());
    assert_eq!(a.post("/events", json!({"events": []}), &bearer(&o.proc1)).await.status, 400);
    assert_eq!(
        a.post("/memory", json!({"text": "x", "metadata": {"blob": "y".repeat(5000)}}), &bearer(&o.proc1)).await.status,
        400
    );
    assert_eq!(a.post("/memory", json!({"text": "x", "slot": "bad slot!"}), &bearer(&o.proc1)).await.status, 400);
    assert_eq!(a.post("/memory", json!({"text": "x", "agent_id": "log-1"}), &bearer(&o.proc1)).await.status, 403);
    o.a.h.close().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn test_admin_audit_and_events() {
    let o = org().await;
    let a = &o.a;
    o.observe(&o.log1, "note", json!({"topic": "t"})).await;
    let actions: BTreeSet<String> = a.get("/admin/audit", &a.admin()).await.json()["audit"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| r["action"].as_str().unwrap().to_string())
        .collect();
    assert!(actions.contains("memory.ingest"));
    assert!(actions.contains("agent.register"));
    let r = a.get("/admin/events?org=northwind&kind=memory.observed", &a.admin()).await;
    assert_eq!(r.json()["events"].as_array().unwrap().len(), 1);
    assert_eq!(a.get("/admin/audit", &bearer(&o.log1)).await.status, 403);
    o.a.h.close().await;
}

// ------------------------------------------------------------------------------------------------ MCP
#[tokio::test(flavor = "multi_thread")]
async fn test_mcp_identity_is_the_bearer_agent() {
    let a = Api::new(|_| {}).await;
    assert_eq!(a.rpc(None, "tools/list", None).await.status, 401);
    let key = a.register("log-1", json!({})).await;
    let other = a.register("sales-9", json!({"department": "commercial", "team": "field-sales"})).await;
    let r = a.rpc(Some(&key), "initialize", Some(json!({"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "t", "version": "1"}}))).await;
    assert_eq!(r.status, 200);
    assert_eq!(r.json()["result"]["serverInfo"]["name"], "mycelic");
    assert!(r.headers.contains_key("mcp-session-id"));
    let names: Vec<String> = a.rpc(Some(&key), "tools/list", None).await.json()["result"]["tools"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t["name"].as_str().unwrap().to_string())
        .collect();
    assert_eq!(names, ["mycelic_query", "mycelic_remember", "mycelic_lineage", "mycelic_get_memory", "mycelic_status"]);
    let out = a
        .rpc(Some(&key), "tools/call", Some(json!({"name": "mycelic_remember", "arguments": {"text": "Terminal 3 strike announced.", "topic": "supply:sd-9/transport"}})))
        .await
        .json()["result"]
        .clone();
    assert_eq!(out["isError"], false, "{out}");
    let mid = out["structuredContent"]["memory_id"].as_str().unwrap().to_string();
    assert_eq!(out["structuredContent"]["scope"], "northwind/emea/nw-gmbh/ops/logistics/log-1");
    a.h.settle().await;
    let r = a
        .rpc(
            Some(&key),
            "tools/call",
            Some(json!({"name": "mycelic_query", "arguments": {"query": "terminal strike"}})),
        )
        .await;
    assert_eq!(r.json()["result"]["structuredContent"]["answer"]["memory_id"], json!(mid));
    let r = a
        .rpc(Some(&other), "tools/call", Some(json!({"name": "mycelic_get_memory", "arguments": {"memory_id": mid}})))
        .await;
    assert_eq!(r.json()["result"]["isError"], true, "another team's agent cannot read it through MCP either");
    let r = a
        .rpc(Some(&key), "tools/call", Some(json!({"name": "mycelic_lineage", "arguments": {"memory_id": mid}})))
        .await;
    assert_eq!(r.json()["result"]["structuredContent"]["roots"], json!([mid]));
    let r = a.rpc(Some(&key), "tools/call", Some(json!({"name": "mycelic_status", "arguments": {}}))).await;
    assert_eq!(r.json()["result"]["structuredContent"]["memories_by_layer"]["agent"], 1);
    // another organization sees its own counts only
    let acme = a.post("/admin/agents", json!({"enterprise": "acme", "agent_id": "acme-1"}), &a.admin()).await.json()["api_key"].as_str().unwrap().to_string();
    let r = a.rpc(Some(&acme), "tools/call", Some(json!({"name": "mycelic_status", "arguments": {}}))).await;
    assert_eq!(r.json()["result"]["structuredContent"]["memories_by_layer"]["agent"], 0);
    a.h.close().await;
}
