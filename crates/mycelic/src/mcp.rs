//! MCP (Model Context Protocol) surface for Mycelic: Claude Code or any MCP client can use the organization's
//! memory as a tool set.
//!
//! [`McpProtocol`] is the JSON-RPC 2.0 / MCP core (`initialize`, `notifications/*`, `ping`, `tools/list`,
//! `tools/call`, empty `resources/*` and `prompts/*`); a [`ToolSet`] supplies the tools.  Identity is per request:
//! the API middleware authenticates the `Authorization: Bearer <agent key>` header exactly as for the REST routes
//! and the principal is handed to the tools with the call, so `mycelic_remember` writes as the calling agent and
//! `mycelic_query` sees only what that agent may see.  There is no shared MCP token.
//!
//! Two transports:
//!
//! * Streamable HTTP at `/mcp` on the service itself ([`McpHttp`]);
//! * stdio via `mycelic mcp --url https://mycelic.example --api-key mk_...` ([`serve_stdio_proxy`]), which
//!   proxies to the HTTP API for desktop clients that only speak stdio.
//!
//! Tool definitions, error codes and texts (including the `TypeError` wording Python produced for bad tool
//! arguments) are the Python service's.

use crate::api::{AppState, RequestContext, header_text, read_limited};
use crate::auth::Principal;
use crate::pyjson::{self, Style, repr_str};
use crate::service::{MycelicService, ServiceError, error_type_name, py_str};
use async_trait::async_trait;
use axum::body::Body;
use axum::extract::Request;
use axum::http::{HeaderValue, Method, StatusCode, header};
use axum::response::Response;
use serde_json::{Map, Value, json};
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

pub const SUPPORTED_PROTOCOL_VERSIONS: [&str; 3] = ["2025-06-18", "2025-03-26", "2024-11-05"];
pub const LATEST_PROTOCOL_VERSION: &str = SUPPORTED_PROTOCOL_VERSIONS[0];
pub const MAX_HTTP_SESSIONS: usize = 10_000;

pub const JSONRPC_PARSE_ERROR: i64 = -32700;
pub const JSONRPC_INVALID_REQUEST: i64 = -32600;
pub const JSONRPC_METHOD_NOT_FOUND: i64 = -32601;
pub const JSONRPC_INVALID_PARAMS: i64 = -32602;
pub const JSONRPC_INTERNAL_ERROR: i64 = -32603;

pub const INSTRUCTIONS: &str = "Organizational memory shared across agents. Call mycelic_query before answering questions about the \
organization, its customers, suppliers, projects or risks; the answer carries lineage you can inspect with mycelic_lineage. \
Call mycelic_remember to share an observation worth propagating; give it a topic (and a slot/entity when it is evidence for a \
known pattern) so it can be aggregated with other agents' notes.";

fn schema(props: Value, required: &[&str]) -> Value {
    json!({"type": "object", "properties": props, "required": required, "additionalProperties": false})
}

/// The `tools/list` payload.
pub fn tool_defs() -> Value {
    json!([
        {
            "name": "mycelic_query",
            "title": "Query organizational memory",
            "description": "Retrieve what the organization knows about a question, ranked with a preference for higher organizational layers (team, department, ... enterprise). Returns the best answer with a lineage summary (contributing agents/teams, layers, whether the evidence is still reconstructable) and the supporting memories. Only memories visible to the calling agent are considered.",
            "inputSchema": schema(json!({
                "query": {"type": "string"},
                "scope": {"type": "string", "description": "Unit path to search under (default: the whole enterprise)."},
                "min_layer": {"type": "string", "enum": ["agent", "team", "department", "subsidiary", "region", "enterprise"], "default": "agent"},
                "k": {"type": "integer", "minimum": 1, "maximum": 50, "default": 5},
                "topic": {"type": "string"},
                "entity": {"type": "string"},
            }), &["query"]),
            "annotations": {"readOnlyHint": true, "openWorldHint": false},
        },
        {
            "name": "mycelic_remember",
            "title": "Share an observation",
            "description": "Publish one memory to the organization as the calling agent. It stays attributed to you, is visible to your team (or the whole organization with visibility='org'), and is aggregated with other agents' memories on the same topic. Set slot/entity when the observation is evidence for a known pattern.",
            "inputSchema": schema(json!({
                "text": {"type": "string", "description": "One self-contained statement."},
                "topic": {"type": "string", "description": "Aggregation key, e.g. 'supply:sd-9/transport'."},
                "slot": {"type": "string", "description": "Which piece of evidence this is, e.g. 'transport_disruption'."},
                "entity": {"type": "string", "description": "What it is about, e.g. 'sd-9'."},
                "kind": {"type": "string", "default": "observation"},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1, "default": 0.8},
                "visibility": {"type": "string", "enum": ["team", "org"], "default": "team"},
                "idempotency_key": {"type": "string", "description": "Stable id for safe re-sends."},
                "observed_at": {"type": "string", "description": "ISO-8601 time of the observation."},
            }), &["text"]),
            "annotations": {"readOnlyHint": false, "destructiveHint": false, "idempotentHint": true, "openWorldHint": false},
        },
        {
            "name": "mycelic_lineage",
            "title": "Lineage of a memory",
            "description": "Where a memory came from: the graph of contributing memories down to the raw observations, the agents and teams behind them, the organizational layers it passed through, timestamps, confidence and support, and whether the underlying evidence can still be reconstructed. Contributions outside your visibility are redacted, not hidden.",
            "inputSchema": schema(json!({"memory_id": {"type": "string"}}), &["memory_id"]),
            "annotations": {"readOnlyHint": true, "openWorldHint": false},
        },
        {
            "name": "mycelic_get_memory",
            "title": "Read one memory",
            "description": "Fetch a memory by id (if you are allowed to see it).",
            "inputSchema": schema(json!({"memory_id": {"type": "string"}}), &["memory_id"]),
            "annotations": {"readOnlyHint": true, "openWorldHint": false},
        },
        {
            "name": "mycelic_status",
            "title": "Service status",
            "description": "Health of the memory service and counts of memories by organizational layer.",
            "inputSchema": schema(json!({}), &[]),
            "annotations": {"readOnlyHint": true, "openWorldHint": false},
        },
    ])
}

pub fn server_info() -> Value {
    json!({"name": "mycelic", "version": "0.1.0"})
}

/// How a tool call failed, in the three shapes the protocol reports differently.
#[derive(Debug)]
pub enum ToolFailure {
    /// a message for the model (`ToolError`): the call's own error
    Tool(String),
    /// the arguments did not fit the tool's signature (Python's `TypeError`)
    Arguments(String),
    /// anything else, already prefixed with its type name
    Other(String),
}

#[async_trait]
pub trait ToolSet: Send + Sync {
    async fn call(&self, name: &str, args: &Map<String, Value>) -> Result<Value, ToolFailure>;
}

pub fn result(id: &Value, result: Value) -> Value {
    json!({"jsonrpc": "2.0", "id": id, "result": result})
}

pub fn error(id: &Value, code: i64, message: &str) -> Value {
    json!({"jsonrpc": "2.0", "id": id, "error": {"code": code, "message": message}})
}

/// JSON-RPC/MCP core over a tool set.
pub struct McpProtocol<T: ToolSet> {
    pub tools: T,
}

impl<T: ToolSet> McpProtocol<T> {
    fn initialize(&self, params: &Map<String, Value>) -> Value {
        let requested = params.get("protocolVersion").filter(|v| pyjson::truthy(v)).map(py_str).unwrap_or_default();
        let version = if SUPPORTED_PROTOCOL_VERSIONS.contains(&requested.as_str()) {
            requested
        } else {
            LATEST_PROTOCOL_VERSION.into()
        };
        json!({
            "protocolVersion": version,
            "capabilities": {"tools": {"listChanged": false}, "resources": {"subscribe": false, "listChanged": false},
                             "prompts": {"listChanged": false}, "logging": {}},
            "serverInfo": server_info(),
            "instructions": INSTRUCTIONS,
        })
    }

    async fn tools_call(&self, params: &Map<String, Value>) -> Result<Value, String> {
        let name = params.get("name");
        let empty = Value::Object(Map::new());
        let args = match params.get("arguments") {
            Some(v) if pyjson::truthy(v) => v,
            _ => &empty,
        };
        let defs = tool_defs();
        let known =
            name.and_then(Value::as_str).filter(|n| defs.as_array().is_some_and(|d| d.iter().any(|t| t["name"] == *n)));
        let Some(name) = known else {
            return Err(format!("unknown tool: {}", name.map(py_str).unwrap_or_else(|| "None".into())));
        };
        let Value::Object(args) = args else { return Err("arguments must be an object".into()) };
        let text_result = |text: String| json!({"content": [{"type": "text", "text": text}], "isError": true});
        match self.tools.call(name, args).await {
            Ok(out) => {
                let text = pyjson::dumps_with(&out, Style::INDENT1);
                Ok(json!({"content": [{"type": "text", "text": text}], "structuredContent": out, "isError": false}))
            }
            Err(ToolFailure::Tool(m)) => Ok(text_result(m)),
            Err(ToolFailure::Arguments(m)) => Ok(text_result(format!("invalid arguments: {m}"))),
            Err(ToolFailure::Other(m)) => {
                tracing::error!("tool {name} failed: {m}");
                Ok(text_result(m))
            }
        }
    }

    /// Process one JSON-RPC message (request or notification).
    pub async fn handle(&self, message: &Value) -> Option<Value> {
        let Some(msg) = message
            .as_object()
            .filter(|m| m.get("jsonrpc") == Some(&json!("2.0")) && m.get("method").is_some_and(Value::is_string))
        else {
            let id = message.as_object().and_then(|m| m.get("id")).cloned().unwrap_or(Value::Null);
            return Some(error(&id, JSONRPC_INVALID_REQUEST, "invalid request"));
        };
        let method = msg["method"].as_str().unwrap_or_default();
        let params = match msg.get("params") {
            None | Some(Value::Null) => Map::new(),
            Some(Value::Object(p)) => p.clone(),
            Some(_) => {
                return Some(error(
                    msg.get("id").unwrap_or(&Value::Null),
                    JSONRPC_INVALID_PARAMS,
                    "params must be an object",
                ));
            }
        };
        let id = msg.get("id").cloned().unwrap_or(Value::Null);
        // Python accepted str and int ids, and bool (an int subclass); a float id is refused
        let id_ok = match &id {
            Value::Null | Value::String(_) | Value::Bool(_) => true,
            Value::Number(n) => n.is_i64() || n.is_u64(),
            _ => false,
        };
        if !id_ok {
            return Some(error(&Value::Null, JSONRPC_INVALID_REQUEST, "id must be a string or number"));
        }
        let is_notification = !msg.contains_key("id");
        let outcome: Result<Value, String> = match method {
            "initialize" => Ok(self.initialize(&params)),
            m if m == "notifications/initialized" || m.starts_with("notifications/") => return None,
            "ping" | "logging/setLevel" => Ok(json!({})),
            "tools/list" => Ok(json!({"tools": tool_defs()})),
            "tools/call" => self.tools_call(&params).await,
            "resources/list" => Ok(json!({"resources": []})),
            "resources/read" => Err(format!(
                "unknown resource: {}",
                params.get("uri").filter(|v| pyjson::truthy(v)).map(py_str).unwrap_or_default()
            )),
            "resources/templates/list" => Ok(json!({"resourceTemplates": []})),
            "prompts/list" => Ok(json!({"prompts": []})),
            "prompts/get" => Err("no prompts".into()),
            _ => {
                if is_notification {
                    return None;
                }
                return Some(error(&id, JSONRPC_METHOD_NOT_FOUND, &format!("method not found: {method}")));
            }
        };
        match outcome {
            Err(m) => Some(error(&id, JSONRPC_INVALID_PARAMS, &m)),
            Ok(_) if is_notification => None,
            Ok(r) => Some(result(&id, r)),
        }
    }

    /// A decoded payload that may be a single message or a batch.
    pub async fn handle_payload(&self, payload: &Value) -> Option<Value> {
        if let Value::Array(items) = payload {
            if items.is_empty() {
                return Some(error(&Value::Null, JSONRPC_INVALID_REQUEST, "empty batch"));
            }
            let mut out = Vec::new();
            for m in items {
                if let Some(r) = self.handle(m).await {
                    out.push(r);
                }
            }
            return (!out.is_empty()).then_some(Value::Array(out));
        }
        self.handle(payload).await
    }
}

// ---------------------------------------------------------------------------------------------------- tool binding
/// Bind JSON arguments to a tool signature the way Python bound `**kwargs`: an unknown keyword or a missing
/// required argument is a `TypeError` with Python's wording.
fn bind<'a>(
    tool: &str,
    args: &'a Map<String, Value>,
    params: &[&str],
    required: &[&str],
) -> Result<Map<String, Value>, ToolFailure> {
    let qual = format!("MycelicTools.tool_{tool}()");
    for k in args.keys() {
        if !params.contains(&k.as_str()) {
            return Err(ToolFailure::Arguments(format!("{qual} got an unexpected keyword argument {}", repr_str(k))));
        }
    }
    let missing: Vec<&str> = required.iter().copied().filter(|r| !args.contains_key(*r)).collect();
    if let [one] = missing.as_slice() {
        return Err(ToolFailure::Arguments(format!(
            "{qual} missing 1 required positional argument: {}",
            repr_str(one)
        )));
    }
    Ok(args.clone())
}

/// Python's `int(x)` on a tool argument.
fn py_int_arg(v: &Value) -> Result<i64, ToolFailure> {
    match v {
        Value::Number(n) => Ok(n.as_i64().or_else(|| n.as_f64().map(|f| f.trunc() as i64)).unwrap_or(0)),
        Value::Bool(b) => Ok(*b as i64),
        Value::String(s) => crate::config::py_int(s).ok_or_else(|| {
            ToolFailure::Other(format!("ValueError: invalid literal for int() with base 10: {}", repr_str(s)))
        }),
        Value::Null => Err(ToolFailure::Arguments(
            "int() argument must be a string, a bytes-like object or a real number, not 'NoneType'".into(),
        )),
        Value::Array(_) => Err(ToolFailure::Arguments(
            "int() argument must be a string, a bytes-like object or a real number, not 'list'".into(),
        )),
        Value::Object(_) => Err(ToolFailure::Arguments(
            "int() argument must be a string, a bytes-like object or a real number, not 'dict'".into(),
        )),
    }
}

fn service_failure(e: ServiceError) -> ToolFailure {
    match e {
        ServiceError::Validation(m) | ServiceError::Forbidden(m) => ToolFailure::Tool(m),
        // str(KeyError(id)) quotes the id
        ServiceError::NotFound(id) => ToolFailure::Tool(format!("no visible memory with id {}", repr_str(&id))),
        ServiceError::Hierarchy(m) => ToolFailure::Other(format!("HierarchyError: {m}")),
        ServiceError::Internal(e) => ToolFailure::Other(format!("{}: {e}", error_type_name(&e))),
    }
}

/// The service's tools, acting as the principal the HTTP request authenticated.
pub struct MycelicTools {
    pub service: Arc<MycelicService>,
    pub principal: Option<Principal>,
}

impl MycelicTools {
    fn principal(&self) -> Result<Principal, ToolFailure> {
        self.principal
            .clone()
            .ok_or_else(|| ToolFailure::Tool("unauthenticated: send an agent API key as the Bearer token".into()))
    }

    async fn run<T: Send + 'static>(
        &self,
        f: impl FnOnce(Arc<MycelicService>) -> Result<T, ServiceError> + Send + 'static,
    ) -> Result<T, ToolFailure> {
        let svc = self.service.clone();
        match tokio::task::spawn_blocking(move || f(svc)).await {
            Ok(r) => r.map_err(service_failure),
            Err(e) => Err(ToolFailure::Other(format!("JoinError: {e}"))),
        }
    }
}

#[async_trait]
impl ToolSet for MycelicTools {
    async fn call(&self, name: &str, args: &Map<String, Value>) -> Result<Value, ToolFailure> {
        match name {
            "mycelic_query" => {
                let a = bind(name, args, &["query", "scope", "min_layer", "k", "topic", "entity"], &["query"])?;
                let p = self.principal()?;
                let k = py_int_arg(a.get("k").unwrap_or(&json!(5)))?;
                let mut body = json!({"query": a["query"], "min_layer": a.get("min_layer").cloned().unwrap_or(json!("agent")),
                                      "k": k, "include_lineage": false});
                for key in ["scope", "topic", "entity"] {
                    if let Some(v) = a.get(key).filter(|v| pyjson::truthy(v)) {
                        body[key] = v.clone();
                    }
                }
                let p2 = p.clone();
                let mut res = self.run(move |svc| svc.query(&p2, &body)).await?;
                if let Some(results) = res.get_mut("results").and_then(Value::as_array_mut) {
                    for hit in results.iter_mut() {
                        let mut view =
                            self.service.public_view(&hit["memory"], &p).as_object().cloned().unwrap_or_default();
                        view.insert("score".into(), hit["score"].clone());
                        view.insert("explanation".into(), hit["explanation"].clone());
                        *hit = Value::Object(view);
                    }
                }
                Ok(res)
            }
            "mycelic_remember" => {
                let params = [
                    "text",
                    "topic",
                    "slot",
                    "entity",
                    "kind",
                    "confidence",
                    "visibility",
                    "idempotency_key",
                    "observed_at",
                ];
                let a = bind(name, args, &params, &["text"])?;
                let p = self.principal()?;
                let defaults =
                    [("kind", json!("observation")), ("confidence", json!(0.8)), ("visibility", json!("team"))];
                let mut body = Map::new();
                for k in params {
                    let v =
                        a.get(k).cloned().or_else(|| defaults.iter().find(|(d, _)| *d == k).map(|(_, v)| v.clone()));
                    if let Some(v) = v.filter(|v| !v.is_null()) {
                        body.insert(k.into(), v);
                    }
                }
                let (m, created) = self.run(move |svc| svc.ingest_memory(&p, &Value::Object(body), None)).await?;
                Ok(json!({"memory_id": m.memory_id, "event_id": m.event_id, "created": created, "scope": m.scope,
                          "note": "accepted; aggregation happens asynchronously through the event log"}))
            }
            "mycelic_lineage" => {
                let a = bind(name, args, &["memory_id"], &["memory_id"])?;
                let p = self.principal()?;
                let id = memory_id_arg(&a["memory_id"]);
                self.run(move |svc| svc.lineage(&p, &id)).await
            }
            "mycelic_get_memory" => {
                let a = bind(name, args, &["memory_id"], &["memory_id"])?;
                let p = self.principal()?;
                let id = memory_id_arg(&a["memory_id"]);
                let p2 = p.clone();
                let m = self.run(move |svc| svc.get_memory(&p2, &id)).await?;
                Ok(self.service.public_view(&m.to_json(), &p))
            }
            "mycelic_status" => {
                bind(name, args, &[], &[])?;
                let p = self.principal()?;
                let h = self.service.health().await;
                let by_layer = if p.is_admin() {
                    h["stats"].get("memories_by_layer").cloned().unwrap_or(Value::Null)
                } else {
                    let org = p.org_id.clone();
                    Value::Object(
                        self.run(move |svc| Ok(svc.store.read(|db| db.memories_by_layer(org.as_deref()))?)).await?,
                    )
                };
                Ok(json!({"status": h["status"], "version": h["version"], "memories_by_layer": by_layer,
                          "transport_connected": h["checks"]["transport"]["connected"].as_bool().unwrap_or(false)}))
            }
            other => Err(ToolFailure::Tool(format!("unknown tool: {other}"))),
        }
    }
}

fn memory_id_arg(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        other => py_str(other),
    }
}

// ---------------------------------------------------------------------------------------------------- HTTP transport
/// MCP Streamable HTTP (2025-03-26+): one endpoint, POST for client->server messages.
///
/// * Responses are `application/json` (single response) or, when the client only accepts `text/event-stream`, one
///   SSE `message` event followed by stream end.
/// * A session id is issued on `initialize` (`Mcp-Session-Id`); sessions are lightweight here (no
///   server-initiated messages), so an unknown or missing session is tolerated for stateless clients.
/// * `GET` opens an SSE stream that only carries keep-alives; `DELETE` ends a session.
/// * `Origin` is validated against the allowed origins when set (DNS-rebinding protection).  Authentication and
///   the per-message rate charge for batches are the API middleware's and this transport's job respectively.
pub struct McpHttp {
    service: Arc<MycelicService>,
    allowed_origins: Option<Vec<String>>,
    sessions: Mutex<VecDeque<String>>,
}

fn json_rpc_response(data: &Value, status: StatusCode, headers: &[(&'static str, String)]) -> Response {
    // aiohttp's web.json_response: json.dumps defaults (ASCII), no Cache-Control
    let mut r = Response::new(Body::from(pyjson::dumps_with(data, Style::ASCII)));
    *r.status_mut() = status;
    r.headers_mut().insert(header::CONTENT_TYPE, HeaderValue::from_static("application/json; charset=utf-8"));
    for (k, v) in headers {
        if let Ok(v) = HeaderValue::from_str(v) {
            r.headers_mut().insert(*k, v);
        }
    }
    r
}

impl McpHttp {
    pub fn new(service: Arc<MycelicService>, allowed_origins: Option<Vec<String>>) -> McpHttp {
        McpHttp { service, allowed_origins, sessions: Mutex::new(VecDeque::new()) }
    }

    fn origin_ok(&self, req: &Request) -> bool {
        match (header_text(req.headers(), "origin"), &self.allowed_origins) {
            (Some(o), Some(allowed)) if !o.is_empty() => allowed.contains(&o),
            _ => true,
        }
    }

    pub async fn handle(self: Arc<Self>, app: AppState, ctx: RequestContext, req: Request) -> Response {
        let method = req.method().clone();
        let (parts, body) = req.into_parts();
        let req_headers = parts.headers.clone();
        let req = Request::from_parts(parts, Body::empty());
        let payload: Option<Value> = if method == Method::POST {
            match read_limited(&app, body).await {
                Ok(bytes) => serde_json::from_slice(&bytes).ok(),
                Err(_) => None,
            }
        } else {
            None
        };
        if let Some(Value::Array(batch)) = &payload {
            // a JSON-RPC batch is as many requests as it carries: cap it like POST /events and charge the limiter
            let max = self.service.settings.max_batch;
            if batch.len() as i64 > max {
                return json_rpc_response(
                    &error(&Value::Null, JSONRPC_INVALID_REQUEST, &format!("batch too large (max {max})")),
                    StatusCode::BAD_REQUEST,
                    &[],
                );
            }
            let extra = batch.len() as i64 - 1;
            if extra > 0
                && (!app.allow(&format!("ip:{}", ctx.remote), extra)
                    || ctx
                        .principal
                        .as_ref()
                        .is_some_and(|p| !app.allow(&format!("principal:{}:{}", p.kind, p.id), extra)))
            {
                return json_rpc_response(&json!({"error": "rate limit exceeded"}), StatusCode::TOO_MANY_REQUESTS, &[]);
            }
        }
        if !self.origin_ok(&req) {
            return json_rpc_response(&json!({"error": "origin not allowed"}), StatusCode::FORBIDDEN, &[]);
        }
        if method == Method::GET {
            return self.handle_get(&req);
        }
        if method == Method::DELETE {
            if let Some(sid) = header_text(&req_headers, "mcp-session-id") {
                self.sessions.lock().unwrap_or_else(|e| e.into_inner()).retain(|s| *s != sid);
            }
            let mut r = Response::new(Body::empty());
            *r.status_mut() = StatusCode::NO_CONTENT;
            return r;
        }
        if method != Method::POST {
            return json_rpc_response(
                &json!({"error": "method not allowed"}),
                StatusCode::METHOD_NOT_ALLOWED,
                &[("allow", "GET, POST, DELETE".into())],
            );
        }
        if let Some(pv) = header_text(&req_headers, "mcp-protocol-version").filter(|v| !v.is_empty())
            && !SUPPORTED_PROTOCOL_VERSIONS.contains(&pv.as_str())
        {
            return json_rpc_response(
                &json!({"error": format!("unsupported MCP-Protocol-Version {pv}"), "supported": SUPPORTED_PROTOCOL_VERSIONS}),
                StatusCode::BAD_REQUEST,
                &[],
            );
        }
        let Some(payload) = payload else {
            return json_rpc_response(
                &error(&Value::Null, JSONRPC_PARSE_ERROR, "parse error"),
                StatusCode::BAD_REQUEST,
                &[],
            );
        };
        let first = match &payload {
            Value::Array(a) => a.first().cloned().unwrap_or(Value::Null),
            other => other.clone(),
        };
        let is_init = first.get("method").and_then(Value::as_str) == Some("initialize");
        let proto =
            McpProtocol { tools: MycelicTools { service: self.service.clone(), principal: ctx.principal.clone() } };
        let resp = proto.handle_payload(&payload).await;
        let mut headers: Vec<(&'static str, String)> = vec![("mcp-protocol-version", LATEST_PROTOCOL_VERSION.into())];
        if is_init {
            let sid = uuid::Uuid::new_v4().simple().to_string();
            let mut sessions = self.sessions.lock().unwrap_or_else(|e| e.into_inner());
            sessions.push_back(sid.clone());
            while sessions.len() > MAX_HTTP_SESSIONS {
                sessions.pop_front();
            }
            headers.push(("mcp-session-id", sid));
            if let Some(v) = resp
                .as_ref()
                .and_then(|r| r.get("result"))
                .and_then(|r| r.get("protocolVersion"))
                .and_then(Value::as_str)
            {
                headers[0].1 = v.to_string();
            }
        } else if let Some(sid) = header_text(&req_headers, "mcp-session-id").filter(|s| !s.is_empty()) {
            headers.push(("mcp-session-id", sid));
        }
        let Some(resp) = resp else {
            // notifications / responses only
            let mut r = Response::new(Body::empty());
            *r.status_mut() = StatusCode::ACCEPTED;
            for (k, v) in &headers {
                if let Ok(v) = HeaderValue::from_str(v) {
                    r.headers_mut().insert(*k, v);
                }
            }
            return r;
        };
        let accept = header_text(&req_headers, "accept").unwrap_or_default();
        if accept.contains("text/event-stream") && !accept.contains("application/json") {
            let body = format!("event: message\ndata: {}\n\n", pyjson::dumps(&resp));
            let mut r = Response::new(Body::from(body));
            r.headers_mut().insert(header::CONTENT_TYPE, HeaderValue::from_static("text/event-stream; charset=utf-8"));
            r.headers_mut().insert(header::CACHE_CONTROL, HeaderValue::from_static("no-cache"));
            for (k, v) in &headers {
                if let Ok(v) = HeaderValue::from_str(v) {
                    r.headers_mut().insert(*k, v);
                }
            }
            return r;
        }
        json_rpc_response(&resp, StatusCode::OK, &headers)
    }

    fn handle_get(&self, req: &Request) -> Response {
        let accept = header_text(req.headers(), "accept").unwrap_or_default();
        if !accept.contains("text/event-stream") {
            return json_rpc_response(
                &json!({"error": "GET requires Accept: text/event-stream"}),
                StatusCode::METHOD_NOT_ALLOWED,
                &[("allow", "POST".into())],
            );
        }
        let stream = futures::stream::unfold(true, |first| async move {
            if !first {
                tokio::time::sleep(std::time::Duration::from_secs(15)).await;
            }
            Some((Ok::<_, std::io::Error>(bytes::Bytes::from_static(b": keep-alive\n\n")), false))
        });
        let mut r = Response::new(Body::from_stream(stream));
        r.headers_mut().insert(header::CONTENT_TYPE, HeaderValue::from_static("text/event-stream"));
        r.headers_mut().insert(header::CACHE_CONTROL, HeaderValue::from_static("no-cache"));
        r.headers_mut().insert("mcp-protocol-version", HeaderValue::from_static(LATEST_PROTOCOL_VERSION));
        r
    }
}

// ---------------------------------------------------------------------------------------------------- stdio proxy
/// The tools of a stdio MCP server that forwards every call to a running Mycelic over HTTP.
pub struct ProxyTools {
    pub client: mycelic_sdk::MycelicClient,
}

#[async_trait]
impl ToolSet for ProxyTools {
    async fn call(&self, name: &str, args: &Map<String, Value>) -> Result<Value, ToolFailure> {
        let client = self.client.clone();
        let (name, args) = (name.to_string(), args.clone());
        let r = tokio::task::spawn_blocking(move || -> Result<Value, String> {
            let err = |e: mycelic_sdk::MycelicError| format!("MycelicError: {e}");
            match name.as_str() {
                "mycelic_query" => {
                    let q = args.get("query").and_then(Value::as_str).ok_or("KeyError: 'query'")?;
                    let k = args.get("k").and_then(Value::as_i64).unwrap_or(5);
                    let opts = mycelic_sdk::QueryOptions {
                        scope: args.get("scope").and_then(Value::as_str).map(String::from),
                        min_layer: args.get("min_layer").and_then(Value::as_str).unwrap_or("agent").to_string(),
                        k,
                        include_lineage: false,
                        topic: args.get("topic").and_then(Value::as_str).map(String::from),
                        entity: args.get("entity").and_then(Value::as_str).map(String::from),
                    };
                    client.query(q, &opts).map_err(err)
                }
                "mycelic_remember" => client.remember_body(&Value::Object(args)).map_err(err),
                "mycelic_lineage" => client
                    .lineage(args.get("memory_id").and_then(Value::as_str).ok_or("KeyError: 'memory_id'")?)
                    .map_err(err),
                "mycelic_get_memory" => client
                    .get_memory(args.get("memory_id").and_then(Value::as_str).ok_or("KeyError: 'memory_id'")?)
                    .map_err(err),
                "mycelic_status" => client.health().map_err(err),
                other => Err(format!("unknown tool: {other}")),
            }
        })
        .await;
        match r {
            Ok(Ok(v)) => Ok(v),
            Ok(Err(m)) => Err(ToolFailure::Tool(m)),
            Err(e) => Err(ToolFailure::Other(format!("JoinError: {e}"))),
        }
    }
}

/// stdio MCP server proxying to a running Mycelic (for desktop clients): one JSON-RPC message per line in, one
/// compact JSON response per line out.
pub async fn serve_stdio_proxy(client: mycelic_sdk::MycelicClient) -> anyhow::Result<()> {
    use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
    let proto = McpProtocol { tools: ProxyTools { client } };
    let mut lines = BufReader::new(tokio::io::stdin()).lines();
    let mut out = tokio::io::stdout();
    while let Some(line) = lines.next_line().await? {
        if line.trim().is_empty() {
            continue;
        }
        let resp = match serde_json::from_str::<Value>(&line) {
            Ok(payload) => proto.handle_payload(&payload).await,
            Err(_) => Some(error(&Value::Null, JSONRPC_PARSE_ERROR, "parse error")),
        };
        if let Some(resp) = resp {
            out.write_all(format!("{}\n", pyjson::dumps_with(&resp, Style::COMPACT)).as_bytes()).await?;
            out.flush().await?;
        }
    }
    Ok(())
}
