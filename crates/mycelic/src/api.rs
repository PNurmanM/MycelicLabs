//! HTTP API (axum).  Every route is thin: validation and authorization live in [`crate::service`].
//!
//! Routes and the scope each requires:
//!
//! ```text
//! GET  /                         service index (no auth)
//! GET  /health                   liveness: 200 while the database is open (public: status only; details for admins)
//! GET  /ready                    readiness: database open, loops running, no replay in progress
//! GET  /metrics                  Prometheus text format (token: MYCELIC_METRICS_TOKEN, else admin token or agent key)
//! GET  /whoami                   the authenticated principal
//! POST /memory                   memory:write   store one memory (202 Accepted; aggregation is asynchronous)
//! GET  /memory/{id}              memory:read
//! POST /memory/{id}/retract      producer or admin, raw observations only (derived memories follow their evidence)
//! GET  /memories                 memory:read    ?scope=&layer=&limit=&status=
//! POST /events                   events:write   {"events": [...]} (each may embed a "memory")
//! POST /query                    memory:read    {"query", "scope"?, "min_layer"?, "k"?, "include_lineage"?}
//! GET  /lineage/{id}             lineage:read   (POST /query embeds the lineage only for callers holding it)
//! POST /admin/agents             admin          register an agent (the key is returned once)
//! GET  /admin/agents             admin
//! DELETE /admin/agents/{id}      admin          revoke
//! POST /admin/agents/{id}/rotate admin          new key (returned once)
//! GET/POST /admin/rules, DELETE /admin/rules/{id}   admin
//! POST /admin/replay             admin          re-deliver the whole event log to this instance
//! GET  /admin/status             admin          full health, settings (secrets masked), transport state
//! GET  /admin/audit, GET /admin/events            admin
//! *    /mcp                      MCP Streamable HTTP; identity = the Bearer agent key, same as the REST routes
//! ```
//!
//! Status codes, bodies and headers are the aiohttp service's, including its framework responses (`404: Not
//! Found`, `405: Method Not Allowed` with `Allow`, the plain-text 413), and the middleware checks run in the same
//! order: host allow-list (421), CORS preflight, per-address rate limit (429), content type (415), authentication
//! (401/403), per-principal rate limit (429).

// Handlers return `Result<_, Response>`: the error *is* the HTTP answer, built where the refusal is decided.  It
// is large, but boxing it would only add indirection on the error path.
#![allow(clippy::result_large_err)]

use crate::auth::{AuthError, Principal, constant_time_eq};
use crate::mcp::McpHttp;
use crate::pyjson::{self, Style};
use crate::service::{MycelicService, SResult, ServiceError, error_type_name, py_str};
use axum::Router;
use axum::body::{Body, Bytes};
use axum::extract::{ConnectInfo, MatchedPath, Path, Request, State};
use axum::http::{HeaderMap, HeaderValue, Method, StatusCode, header};
use axum::middleware::{self, Next};
use axum::response::Response;
use axum::routing::{any, delete, get, post};
use serde_json::{Value, json};
use std::collections::{HashMap, HashSet};
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use std::time::Instant;

const PUBLIC_PATHS: [&str; 3] = ["/", "/health", "/ready"];

#[derive(Clone)]
pub struct AppState {
    pub service: Arc<MycelicService>,
    pub mcp: Arc<McpHttp>,
    allowed_hosts: Option<Arc<HashSet<String>>>,
    last_log: Arc<Mutex<HashMap<(String, String), Instant>>>,
}

/// What the middleware learned about the request, for the handlers.
#[derive(Clone)]
pub struct RequestContext {
    pub principal: Option<Principal>,
    pub remote: String,
}

/// A response that aiohttp would have produced itself (404, 405, 413) and therefore never counted in
/// `mycelic_http_requests_total`.
#[derive(Clone, Copy)]
struct FrameworkResponse;

pub fn json_response(data: &Value, status: u16) -> Response {
    let mut r = Response::new(Body::from(pyjson::dumps(data)));
    *r.status_mut() = StatusCode::from_u16(status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
    let h = r.headers_mut();
    h.insert(header::CONTENT_TYPE, HeaderValue::from_static("application/json; charset=utf-8"));
    h.insert(header::CACHE_CONTROL, HeaderValue::from_static("no-store"));
    r
}

pub fn error_response(message: &str, status: u16) -> Response {
    json_response(&json!({"error": message}), status)
}

fn text_response(text: &str, status: StatusCode) -> Response {
    let mut r = Response::new(Body::from(text.to_string()));
    *r.status_mut() = status;
    r.headers_mut().insert(header::CONTENT_TYPE, HeaderValue::from_static("text/plain; charset=utf-8"));
    r.extensions_mut().insert(FrameworkResponse);
    r
}

fn service_error(e: ServiceError) -> Response {
    match e {
        ServiceError::Validation(m) | ServiceError::Hierarchy(m) => error_response(&m, 400),
        ServiceError::Forbidden(m) => error_response(&m, 403),
        ServiceError::NotFound(id) => error_response(&format!("no visible memory with id {id}"), 404),
        ServiceError::Internal(e) => {
            tracing::error!("unhandled error: {e:?}");
            error_response(&format!("internal error: {}", error_type_name(&e)), 500)
        }
    }
}

/// The client address as the proxy saw it: the Nth entry from the right of X-Forwarded-For, never the leftmost
/// (which the client itself can set).  Only used when the deployment says the proxy is the only path in.
fn remote_of(headers: &HeaderMap, peer: Option<SocketAddr>, trust_proxy: bool, hops: i64) -> String {
    if trust_proxy && let Some(fwd) = headers.get("x-forwarded-for").and_then(|v| v.to_str().ok()) {
        let parts: Vec<&str> = fwd.split(',').map(str::trim).filter(|p| !p.is_empty()).collect();
        let hops = hops.max(1) as usize;
        if parts.len() >= hops {
            return parts[parts.len() - hops].to_string();
        }
    }
    peer.map(|p| p.ip().to_string()).unwrap_or_else(|| "unknown".into())
}

/// A header as text whatever bytes the client sent (aiohttp decoded headers leniently too).
pub fn header_text(headers: &HeaderMap, name: &str) -> Option<String> {
    headers.get(name).map(|v| String::from_utf8_lossy(v.as_bytes()).into_owned())
}

fn has_body(req: &Request) -> bool {
    use axum::body::HttpBody;
    let hint = req.body().size_hint();
    !(hint.exact() == Some(0) || req.body().is_end_stream())
}

fn content_type_is_json(headers: &HeaderMap) -> bool {
    let ct = header_text(headers, "content-type").unwrap_or_else(|| "application/octet-stream".into());
    ct.split(';').next().unwrap_or("").trim().to_ascii_lowercase().starts_with("application/json")
}

impl AppState {
    fn host_ok(&self, req: &Request) -> bool {
        let Some(allowed) = &self.allowed_hosts else { return true };
        let host = header_text(req.headers(), "host")
            .or_else(|| req.uri().authority().map(|a| a.to_string()))
            .unwrap_or_default()
            .trim()
            .to_lowercase();
        let name = if host.starts_with('[') {
            format!("{}]", host.split(']').next().unwrap_or(""))
        } else {
            host.split(':').next().unwrap_or("").to_string()
        };
        allowed.contains(&name)
    }

    /// Log an authentication failure at most once a minute per reason and address.
    fn suppressed_log(&self, reason: &str, remote: &str, message: &str) {
        let mut seen = self.last_log.lock().unwrap_or_else(|e| e.into_inner());
        let key = (reason.to_string(), remote.to_string());
        if seen.get(&key).is_none_or(|t| t.elapsed().as_secs_f64() > 60.0) {
            if seen.len() > 10_000 {
                seen.clear();
            }
            seen.insert(key, Instant::now());
            tracing::warn!("{reason} from {remote}: {message}");
        }
    }

    pub fn allow(&self, key: &str, cost: i64) -> bool {
        self.service.limiter.lock().unwrap_or_else(|e| e.into_inner()).allow(key, cost)
    }
}

fn auth_failure_response(e: &AuthError) -> Response {
    if e.status == 401 {
        // aiohttp built this one by hand: json.dumps defaults, no Cache-Control
        let mut r = Response::new(Body::from(pyjson::dumps_with(&json!({"error": e.reason}), Style::ASCII)));
        *r.status_mut() = StatusCode::UNAUTHORIZED;
        r.headers_mut().insert(header::CONTENT_TYPE, HeaderValue::from_static("application/json; charset=utf-8"));
        r.headers_mut().insert(header::WWW_AUTHENTICATE, HeaderValue::from_static("Bearer realm=\"mycelic\""));
        r
    } else {
        error_response(e.reason, e.status)
    }
}

async fn guard(State(app): State<AppState>, mut req: Request, next: Next) -> Response {
    let s = &app.service.settings;
    let m = &app.service.metrics;
    let route =
        req.extensions().get::<MatchedPath>().map(|p| p.as_str().to_string()).unwrap_or_else(|| "unmatched".into());
    let peer = req.extensions().get::<ConnectInfo<SocketAddr>>().map(|c| c.0);
    let remote = remote_of(req.headers(), peer, s.trust_proxy_headers, s.trusted_proxy_hops);
    let path = req.uri().path().to_string();
    let public = PUBLIC_PATHS.contains(&path.as_str());
    let count = |status: u16| m.http_requests.with_label_values(&[&route, &status.to_string()]).inc();
    // probes (kubelet, Docker HEALTHCHECK) send the pod/container address as Host: only the public paths skip the check
    if !public && !app.host_ok(&req) {
        count(421);
        return error_response("host not allowed", 421);
    }
    if req.method() == Method::OPTIONS {
        let mut r = Response::new(Body::empty());
        *r.status_mut() = if s.cors_origins.is_empty() { StatusCode::FORBIDDEN } else { StatusCode::NO_CONTENT };
        return r;
    }
    // rate limit by peer before authentication (a bad token must never spend the claimed agent's budget)
    if !app.allow(&format!("ip:{remote}"), 1) {
        count(429);
        m.auth_failures.with_label_values(&["rate_limited"]).inc();
        return error_response("rate limit exceeded", 429);
    }
    if matches!(*req.method(), Method::POST | Method::PUT | Method::PATCH | Method::DELETE)
        && has_body(&req)
        && !content_type_is_json(req.headers())
    {
        count(415);
        return error_response("Content-Type must be application/json", 415);
    }
    let mut principal = None;
    if !public {
        let auth = header_text(req.headers(), "authorization");
        if path == "/metrics" && s.metrics_token.as_deref().is_some_and(|t| !t.is_empty()) {
            let token = auth
                .as_deref()
                .and_then(|a| a.strip_prefix("Bearer "))
                .map(|t| crate::textutil::py_strip(t).to_string())
                .unwrap_or_default();
            if !constant_time_eq(token.as_bytes(), s.metrics_token.as_deref().unwrap_or("").as_bytes()) {
                m.auth_failures.with_label_values(&["metrics_token"]).inc();
                return error_response("unauthorized", 401);
            }
        } else if path == "/metrics" && s.is_loopback() && auth.is_none() {
            // loopback development: scrape without a token
        } else {
            let svc = app.service.clone();
            let a = auth.clone();
            let result = tokio::task::spawn_blocking(move || svc.authenticate(a.as_deref())).await;
            match result {
                Ok(Ok(Ok(p))) => {
                    if !app.allow(&format!("principal:{}:{}", p.kind, p.id), 1) {
                        count(429);
                        m.auth_failures.with_label_values(&["rate_limited"]).inc();
                        return error_response("rate limit exceeded", 429);
                    }
                    principal = Some(p);
                }
                Ok(Ok(Err(e))) => {
                    m.auth_failures.with_label_values(&[&e.reason.replace(' ', "_")]).inc();
                    count(e.status);
                    app.suppressed_log("auth failure", &remote, e.reason);
                    return auth_failure_response(&e);
                }
                Ok(Err(e)) => {
                    count(500);
                    return service_error(e);
                }
                Err(_) => return error_response("internal error: JoinError", 500),
            }
        }
    }
    req.extensions_mut().insert(RequestContext { principal, remote });
    let mut response = next.run(req).await;
    if response.status() == StatusCode::METHOD_NOT_ALLOWED && !response.headers().contains_key(header::CONTENT_TYPE) {
        // axum's own 405 (wrong method on a known route): the body aiohttp sent, keeping axum's Allow header
        let allow = response.headers().get(header::ALLOW).cloned();
        response = text_response("405: Method Not Allowed", StatusCode::METHOD_NOT_ALLOWED);
        if let Some(a) = allow {
            response.headers_mut().insert(header::ALLOW, a);
        }
    }
    if response.extensions().get::<FrameworkResponse>().is_none() {
        count(response.status().as_u16());
    }
    response
}

/// Read a JSON object body: 413 above the limit (plain text, as aiohttp answered), 400 when it is not JSON or
/// not an object.
async fn json_body(app: &AppState, body: Body) -> Result<Value, Response> {
    let v = read_json(app, body).await?;
    if !v.is_object() {
        return Err(error_response("JSON body must be an object", 400));
    }
    Ok(v)
}

async fn read_json(app: &AppState, body: Body) -> Result<Value, Response> {
    let bytes = read_limited(app, body).await?;
    serde_json::from_slice::<Value>(&bytes).map_err(|_| error_response("invalid JSON body", 400))
}

pub async fn read_limited(app: &AppState, body: Body) -> Result<Bytes, Response> {
    let limit = app.service.settings.max_body_bytes.max(0) as usize;
    match axum::body::to_bytes(body, limit).await {
        Ok(b) => Ok(b),
        Err(_) => {
            Err(text_response(&format!("Maximum request body size {limit} exceeded."), StatusCode::PAYLOAD_TOO_LARGE))
        }
    }
}

fn ctx(req: &Request) -> RequestContext {
    req.extensions()
        .get::<RequestContext>()
        .cloned()
        .unwrap_or(RequestContext { principal: None, remote: "unknown".into() })
}

fn principal(c: &RequestContext) -> Result<Principal, Response> {
    c.principal.clone().ok_or_else(|| error_response("internal error: AuthError", 500))
}

fn admin(c: &RequestContext) -> Result<Principal, Response> {
    let p = principal(c)?;
    if !p.is_admin() {
        return Err(error_response("administrator token required", 403));
    }
    Ok(p)
}

/// Run a blocking service call off the async workers.
async fn blocking<T: Send + 'static>(f: impl FnOnce() -> SResult<T> + Send + 'static) -> Result<T, Response> {
    match tokio::task::spawn_blocking(f).await {
        Ok(Ok(v)) => Ok(v),
        Ok(Err(e)) => Err(service_error(e)),
        Err(e) => {
            Err(error_response(&format!("internal error: {}", if e.is_panic() { "panic" } else { "cancelled" }), 500))
        }
    }
}

type Handled = Result<Response, Response>;

fn done(r: Handled) -> Response {
    r.unwrap_or_else(|e| e)
}

/// Query string values (first occurrence of each key), decoded like a form.
fn query_params(req: &Request) -> HashMap<String, String> {
    let mut out = HashMap::new();
    if let Some(q) = req.uri().query() {
        for (k, v) in url_decode_pairs(q) {
            out.entry(k).or_insert(v);
        }
    }
    out
}

fn url_decode_pairs(q: &str) -> Vec<(String, String)> {
    q.split('&')
        .filter(|p| !p.is_empty())
        .map(|p| {
            let (k, v) = p.split_once('=').unwrap_or((p, ""));
            (percent_decode(k), percent_decode(v))
        })
        .collect()
}

/// `application/x-www-form-urlencoded` decoding: `+` is a space, `%XX` a byte; a malformed escape stays as it is.
fn percent_decode(s: &str) -> String {
    let hex = |b: u8| (b as char).to_digit(16).map(|d| d as u8);
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        let b = bytes[i];
        if b == b'+' {
            out.push(b' ');
            i += 1;
            continue;
        }
        if b == b'%'
            && i + 3 <= bytes.len()
            && let (Some(h), Some(l)) = (hex(bytes[i + 1]), hex(bytes[i + 2]))
        {
            out.push(h * 16 + l);
            i += 3;
            continue;
        }
        out.push(b);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn limit_param(q: &HashMap<String, String>, default: &str, max: i64) -> Result<i64, Response> {
    let raw = q.get("limit").map(String::as_str).unwrap_or(default);
    let n = crate::config::py_int(raw).ok_or_else(|| error_response("limit must be an integer", 400))?;
    Ok(n.clamp(1, max))
}

// ---------------------------------------------------------------------------------------------------- handlers
async fn index() -> Response {
    json_response(
        &json!({"service": "mycelic", "version": crate::VERSION,
                "docs": "https://github.com/PNurmanM/MycelicLabs/blob/main/DEPLOYMENT.md",
                "endpoints": ["/health", "/ready", "/metrics", "/whoami", "/memory", "/memories", "/events", "/query",
                              "/lineage/{id}", "/admin/*", "/mcp"]}),
        200,
    )
}

async fn health(State(app): State<AppState>, req: Request) -> Response {
    let h = app.service.health().await;
    let status = if h["status"] == "failing" { 503 } else { 200 };
    if let Some(auth) = header_text(req.headers(), "authorization") {
        let svc = app.service.clone();
        let res = tokio::task::spawn_blocking(move || svc.authenticate(Some(&auth))).await;
        match res {
            Ok(Ok(Ok(p))) if p.is_admin() => return json_response(&h, status),
            Ok(Ok(Err(e))) => {
                app.service.metrics.auth_failures.with_label_values(&[&e.reason.replace(' ', "_")]).inc();
                app.suppressed_log("auth failure", &ctx(&req).remote, e.reason);
            }
            _ => {}
        }
    }
    json_response(
        &json!({"status": h["status"], "version": h["version"],
                "transport_connected": h["checks"]["transport"]["connected"].as_bool().unwrap_or(false),
                "consumer_running": h["checks"]["consumer"]["running"]}),
        status,
    )
}

async fn ready(State(app): State<AppState>) -> Response {
    let (ok, detail) = app.service.ready().await;
    json_response(&detail, if ok { 200 } else { 503 })
}

async fn metrics(State(app): State<AppState>) -> Response {
    let svc = app.service.clone();
    if let Ok(Ok(stats)) = tokio::task::spawn_blocking(move || svc.store.stats()).await {
        app.service.metrics.refresh_from_stats(&stats);
    }
    let info = app.service.transport.info().await;
    if let Some(n) = info.get("consumer_pending").and_then(Value::as_i64) {
        app.service.metrics.consumer_pending.set(n);
    }
    let mut r = Response::new(Body::from(app.service.metrics.render()));
    r.headers_mut().insert(header::CONTENT_TYPE, HeaderValue::from_static("text/plain; charset=utf-8"));
    r
}

async fn whoami(req: Request) -> Response {
    done(principal(&ctx(&req)).map(|p| json_response(&p.to_json(), 200)))
}

async fn post_memory(State(app): State<AppState>, req: Request) -> Response {
    done(async move {
        let c = ctx(&req);
        let p = principal(&c)?;
        let body = json_body(&app, req.into_body()).await?;
        let svc = app.service.clone();
        let (mem, created) = blocking(move || svc.ingest_memory(&p, &body, Some(&c.remote))).await?;
        Ok(json_response(
            &json!({"memory_id": mem.memory_id, "event_id": mem.event_id, "created": created, "scope": mem.scope, "status": "accepted"}),
            if created { 202 } else { 200 },
        ))
    }
    .await)
}

async fn get_memory(State(app): State<AppState>, Path(id): Path<String>, req: Request) -> Response {
    done(
        async move {
            let p = principal(&ctx(&req))?;
            let svc = app.service.clone();
            let p2 = p.clone();
            let mem = blocking(move || svc.get_memory(&p2, &id)).await?;
            Ok(json_response(&json!({"memory": app.service.public_view(&mem.to_json(), &p)}), 200))
        }
        .await,
    )
}

async fn retract_memory(State(app): State<AppState>, Path(id): Path<String>, req: Request) -> Response {
    done(
        async move {
            let c = ctx(&req);
            let p = principal(&c)?;
            let body = if has_body(&req) { json_body(&app, req.into_body()).await? } else { json!({}) };
            let reason = match body.get("reason") {
                Some(v) if pyjson::truthy(v) => py_str(v),
                _ => "retracted by producer".to_string(),
            };
            let svc = app.service.clone();
            let id2 = id.clone();
            let ev = blocking(move || svc.retract(&p, &id2, &reason, Some(&c.remote))).await?;
            Ok(json_response(&json!({"memory_id": id, "event_id": ev.event_id, "status": "accepted"}), 202))
        }
        .await,
    )
}

async fn list_memories(State(app): State<AppState>, req: Request) -> Response {
    done(
        async move {
            let p = principal(&ctx(&req))?;
            let q = query_params(&req);
            let limit = limit_param(&q, "50", 500)?;
            let layers: Vec<String> = q
                .get("layer")
                .map(|l| l.split(',').filter(|x| !x.is_empty()).map(String::from).collect())
                .unwrap_or_default();
            let scope = q.get("scope").filter(|s| !s.is_empty()).cloned();
            let status = q.get("status").filter(|s| !s.is_empty()).cloned().unwrap_or_else(|| "active".into());
            let svc = app.service.clone();
            let p2 = p.clone();
            let rows = blocking(move || {
                svc.list_memories(
                    &p2,
                    scope.as_deref(),
                    (!layers.is_empty()).then_some(layers.as_slice()),
                    limit,
                    &status,
                )
            })
            .await?;
            let out: Vec<Value> = rows.iter().map(|m| app.service.public_view(&m.to_json(), &p)).collect();
            Ok(json_response(&json!({"memories": out}), 200))
        }
        .await,
    )
}

async fn post_events(State(app): State<AppState>, req: Request) -> Response {
    done(
        async move {
            let c = ctx(&req);
            let p = principal(&c)?;
            let body = json_body(&app, req.into_body()).await?;
            let svc = app.service.clone();
            let results = blocking(move || svc.ingest_events(&p, body.get("events"), Some(&c.remote))).await?;
            Ok(json_response(&json!({"results": results, "status": "accepted"}), 202))
        }
        .await,
    )
}

async fn post_query(State(app): State<AppState>, req: Request) -> Response {
    done(
        async move {
            let p = principal(&ctx(&req))?;
            let body = json_body(&app, req.into_body()).await?;
            let svc = app.service.clone();
            let p2 = p.clone();
            let mut res = blocking(move || svc.query(&p2, &body)).await?;
            if let Some(results) = res.get_mut("results").and_then(Value::as_array_mut) {
                for hit in results.iter_mut() {
                    let view = app.service.public_view(&hit["memory"], &p);
                    hit["memory"] = view;
                }
            }
            Ok(json_response(&res, 200))
        }
        .await,
    )
}

async fn get_lineage(State(app): State<AppState>, Path(id): Path<String>, req: Request) -> Response {
    done(
        async move {
            let p = principal(&ctx(&req))?;
            let svc = app.service.clone();
            Ok(json_response(&blocking(move || svc.lineage(&p, &id)).await?, 200))
        }
        .await,
    )
}

async fn admin_register(State(app): State<AppState>, req: Request) -> Response {
    done(
        async move {
            let c = ctx(&req);
            admin(&c)?;
            let body = json_body(&app, req.into_body()).await?;
            let svc = app.service.clone();
            let (agent, key) = blocking(move || svc.register_agent(&body, Some(&c.remote))).await?;
            Ok(json_response(
                &json!({"agent": agent.to_json(), "api_key": key, "note": "store the key now; it is not shown again"}),
                201,
            ))
        }
        .await,
    )
}

async fn admin_list_agents(State(app): State<AppState>, req: Request) -> Response {
    done(
        async move {
            admin(&ctx(&req))?;
            let q = query_params(&req);
            let org = q.get("org").filter(|o| !o.is_empty()).cloned();
            let all = q.get("all").map(String::as_str) == Some("1");
            let svc = app.service.clone();
            let agents = blocking(move || Ok(svc.store.list_agents(org.as_deref(), all)?)).await?;
            Ok(json_response(&json!({"agents": agents.iter().map(|a| a.to_json()).collect::<Vec<_>>()}), 200))
        }
        .await,
    )
}

async fn admin_revoke(State(app): State<AppState>, Path(id): Path<String>, req: Request) -> Response {
    done(
        async move {
            let c = ctx(&req);
            admin(&c)?;
            let svc = app.service.clone();
            let id2 = id.clone();
            let ok = blocking(move || svc.revoke_agent(&id2, Some(&c.remote))).await?;
            Ok(json_response(&json!({"agent_id": id, "revoked": ok}), if ok { 200 } else { 404 }))
        }
        .await,
    )
}

async fn admin_rotate(State(app): State<AppState>, Path(id): Path<String>, req: Request) -> Response {
    done(
        async move {
            let c = ctx(&req);
            admin(&c)?;
            let svc = app.service.clone();
            let id2 = id.clone();
            match blocking(move || svc.rotate_agent_key(&id2, Some(&c.remote))).await? {
                None => Ok(error_response("no such agent", 404)),
                Some(key) => Ok(json_response(
                    &json!({"agent_id": id, "api_key": key, "note": "store the key now; it is not shown again"}),
                    200,
                )),
            }
        }
        .await,
    )
}

async fn admin_rules(State(app): State<AppState>, req: Request) -> Response {
    done(
        async move {
            admin(&ctx(&req))?;
            let svc = app.service.clone();
            let rules = blocking(move || Ok(svc.store.list_rules(None, false)?)).await?;
            Ok(json_response(&json!({"rules": rules.iter().map(|r| r.to_json()).collect::<Vec<_>>()}), 200))
        }
        .await,
    )
}

async fn admin_put_rule(State(app): State<AppState>, req: Request) -> Response {
    done(
        async move {
            let c = ctx(&req);
            admin(&c)?;
            let body = json_body(&app, req.into_body()).await?;
            let svc = app.service.clone();
            let rule = blocking(move || svc.upsert_rule(&body, Some(&c.remote))).await?;
            Ok(json_response(&json!({"rule": rule.to_json()}), 201))
        }
        .await,
    )
}

async fn admin_delete_rule(State(app): State<AppState>, Path(id): Path<String>, req: Request) -> Response {
    done(
        async move {
            let c = ctx(&req);
            admin(&c)?;
            let svc = app.service.clone();
            let id2 = id.clone();
            let ok = blocking(move || svc.delete_rule(&id2, Some(&c.remote))).await?;
            Ok(json_response(&json!({"rule_id": id, "deleted": ok}), if ok { 200 } else { 404 }))
        }
        .await,
    )
}

async fn admin_replay(State(app): State<AppState>, req: Request) -> Response {
    done(
        async move {
            let c = ctx(&req);
            admin(&c)?;
            let res = app.service.replay(Some(&c.remote)).await.map_err(service_error)?;
            Ok(json_response(&res, 202))
        }
        .await,
    )
}

async fn admin_status(State(app): State<AppState>, req: Request) -> Response {
    done(
        async move {
            admin(&ctx(&req))?;
            let mut h = app.service.health().await;
            h["settings"] = app.service.settings.redacted();
            Ok(json_response(&h, 200))
        }
        .await,
    )
}

async fn admin_audit(State(app): State<AppState>, req: Request) -> Response {
    done(
        async move {
            admin(&ctx(&req))?;
            let limit = limit_param(&query_params(&req), "100", 1000)?;
            let svc = app.service.clone();
            let rows = blocking(move || Ok(svc.store.read(|db| db.recent_audit(limit, None))?)).await?;
            Ok(json_response(&json!({"audit": rows}), 200))
        }
        .await,
    )
}

async fn admin_events(State(app): State<AppState>, req: Request) -> Response {
    done(
        async move {
            admin(&ctx(&req))?;
            let q = query_params(&req);
            let Some(org) = q.get("org").filter(|o| !o.is_empty()).cloned() else {
                return Err(error_response("org is required", 400));
            };
            let limit = limit_param(&q, "100", 1000)?;
            let get = |k: &str| q.get(k).filter(|v| !v.is_empty()).cloned();
            let (agent, kind, status) = (get("agent"), get("kind"), get("status"));
            let svc = app.service.clone();
            let rows = blocking(move || {
                Ok(svc
                    .store
                    .read(|db| db.list_events(&org, agent.as_deref(), kind.as_deref(), status.as_deref(), limit))?)
            })
            .await?;
            Ok(json_response(&json!({"events": rows.iter().map(|e| e.to_json()).collect::<Vec<_>>()}), 200))
        }
        .await,
    )
}

async fn mcp_route(State(app): State<AppState>, req: Request) -> Response {
    let c = ctx(&req);
    app.mcp.clone().handle(app.clone(), c, req).await
}

async fn not_found() -> Response {
    text_response("404: Not Found", StatusCode::NOT_FOUND)
}

pub fn create_app(service: Arc<MycelicService>) -> Router {
    let s = &service.settings;
    let allowed_hosts =
        (!s.allowed_hosts.is_empty()).then(|| Arc::new(s.allowed_hosts.iter().map(|h| h.to_lowercase()).collect()));
    let mcp = Arc::new(McpHttp::new(service.clone(), (!s.cors_origins.is_empty()).then(|| s.cors_origins.clone())));
    let state = AppState { service, mcp, allowed_hosts, last_log: Arc::new(Mutex::new(HashMap::new())) };
    let routes = Router::new()
        .route("/", get(index))
        .route("/health", get(health))
        .route("/ready", get(ready))
        .route("/metrics", get(metrics))
        .route("/whoami", get(whoami))
        .route("/memory", post(post_memory))
        .route("/memory/{id}", get(get_memory).delete(retract_memory))
        .route("/memory/{id}/retract", post(retract_memory))
        .route("/memories", get(list_memories))
        .route("/events", post(post_events))
        .route("/query", post(post_query))
        .route("/lineage/{id}", get(get_lineage))
        .route("/admin/agents", post(admin_register).get(admin_list_agents))
        .route("/admin/agents/{id}", delete(admin_revoke))
        .route("/admin/agents/{id}/rotate", post(admin_rotate))
        .route("/admin/rules", get(admin_rules).post(admin_put_rule))
        .route("/admin/rules/{id}", delete(admin_delete_rule))
        .route("/admin/replay", post(admin_replay))
        .route("/admin/status", get(admin_status))
        .route("/admin/audit", get(admin_audit))
        .route("/admin/events", get(admin_events))
        .route("/mcp", any(mcp_route))
        .fallback(not_found)
        .layer(middleware::from_fn_with_state(state.clone(), guard))
        .with_state(state);
    // axum adds `Allow` to every response of a route's method fallback, including the middleware's early answers
    // (a 403 to OPTIONS, a 401 to an unauthenticated wrong-method request); aiohttp sent it with a 405 only
    Router::new().fallback_service(routes).layer(middleware::map_response(|mut r: Response| async move {
        if r.status() != StatusCode::METHOD_NOT_ALLOWED {
            r.headers_mut().remove(header::ALLOW);
        }
        r
    }))
}

/// A bound HTTP(S) server; `shutdown()` stops accepting and waits for in-flight requests.
pub struct ServerHandle {
    pub local_addr: SocketAddr,
    handle: axum_server::Handle<SocketAddr>,
    task: tokio::task::JoinHandle<std::io::Result<()>>,
}

impl ServerHandle {
    pub async fn shutdown(self) {
        self.handle.graceful_shutdown(Some(std::time::Duration::from_secs(10)));
        let _ = self.task.await;
    }
}

/// Bind the HTTP server (before the transport connects, so probes answer during a slow broker start).
pub async fn run_server(service: Arc<MycelicService>) -> anyhow::Result<ServerHandle> {
    let s = service.settings.clone();
    let addr: SocketAddr = tokio::net::lookup_host((s.host.as_str(), s.port as u16))
        .await?
        .next()
        .ok_or_else(|| anyhow::anyhow!("cannot resolve {}:{}", s.host, s.port))?;
    let listener = std::net::TcpListener::bind(addr)?;
    listener.set_nonblocking(true)?;
    let local_addr = listener.local_addr()?;
    let app = create_app(service).into_make_service_with_connect_info::<SocketAddr>();
    let handle = axum_server::Handle::new();
    let task = if s.tls_enabled() {
        let _ = rustls::crypto::ring::default_provider().install_default();
        let config = axum_server::tls_rustls::RustlsConfig::from_pem_file(
            s.tls_cert_file.clone().unwrap_or_default(),
            s.tls_key_file.clone().unwrap_or_default(),
        )
        .await?;
        let server = axum_server::from_tcp_rustls(listener, config)?.handle(handle.clone());
        tokio::spawn(async move { server.serve(app).await })
    } else {
        let server = axum_server::from_tcp(listener)?.handle(handle.clone());
        tokio::spawn(async move { server.serve(app).await })
    };
    let scheme = if s.tls_enabled() { "https" } else { "http" };
    tracing::info!("mycelic API on {scheme}://{}:{}  (MCP at /mcp)", s.host, local_addr.port());
    Ok(ServerHandle { local_addr, handle, task })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn query_strings_decode_like_a_form() {
        assert_eq!(
            url_decode_pairs("scope=bad%20scope&layer=a,b&x=1+2"),
            vec![("scope".into(), "bad scope".into()), ("layer".into(), "a,b".into()), ("x".into(), "1 2".into())]
        );
        assert_eq!(percent_decode("%zz%4"), "%zz%4");
    }
}
