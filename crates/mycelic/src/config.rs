//! Configuration from environment variables.  Nothing secret lives in source or in the image.
//!
//! Every knob has a `MYCELIC_*` variable; [`Settings::from_env`] validates them once at start-up and fails fast
//! with a message naming the variable, which is what an operator wants from `docker compose up`.  Parsing follows
//! the Python service (`int()`/`float()` accept surrounding whitespace and digit-group underscores), so a
//! deployment's `.env` means the same thing to both implementations.

use crate::pyjson::repr_str;
use serde_json::{Map, Value, json};
use std::fmt;
use std::path::Path;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConfigError(pub String);

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for ConfigError {}

#[derive(Clone, Debug, PartialEq)]
pub struct Settings {
    // service
    pub host: String,
    pub port: i64,
    pub db_path: String,
    pub instance_id: String,
    pub log_level: String,
    // transport: None disables NATS (in-process transport; tests/dev only)
    pub nats_url: Option<String>,
    pub nats_user: Option<String>,
    pub nats_password: Option<String>,
    pub nats_token: Option<String>,
    pub nats_ca_file: Option<String>,
    pub nats_stream: String,
    pub nats_consumer: String,
    /// 0 = keep events forever (replay needs them)
    pub nats_max_age_seconds: i64,
    /// -1 = unlimited
    pub nats_max_bytes: i64,
    pub nats_duplicate_window_seconds: i64,
    pub nats_ack_wait_seconds: i64,
    pub nats_max_deliver: i64,
    pub publish_batch: i64,
    pub publish_interval_seconds: f64,
    /// 1 = strict stream order (see transport)
    pub consume_batch: i64,
    // security
    pub admin_token: Option<String>,
    /// None = /metrics needs an admin token or agent key (open only on a loopback bind)
    pub metrics_token: Option<String>,
    /// /ready fails while NATS is down (default: the outbox absorbs outages)
    pub ready_requires_nats: bool,
    pub tls_cert_file: Option<String>,
    pub tls_key_file: Option<String>,
    pub allowed_hosts: Vec<String>,
    pub cors_origins: Vec<String>,
    /// per peer address before authentication and per principal after it
    pub rate_limit_rps: f64,
    pub rate_limit_burst: i64,
    pub max_body_bytes: i64,
    pub max_text_chars: i64,
    pub max_batch: i64,
    /// serialized event; must stay below the broker's max_payload
    pub max_event_bytes: i64,
    /// use X-Forwarded-For for rate limiting (behind a trusted proxy only)
    pub trust_proxy_headers: bool,
    /// how many proxies append to X-Forwarded-For (take the Nth from the right)
    pub trusted_proxy_hops: i64,
    pub audit_retention_days: i64,
    /// HMAC key: the consumer rejects events the publisher did not sign
    pub event_signing_key: Option<String>,
    // aggregation
    /// distinct child units needed for topic consolidation
    pub min_support: i64,
    pub rules_file: Option<String>,
    // SDK/demo conveniences
    pub public_url: Option<String>,
}

impl Default for Settings {
    fn default() -> Self {
        Settings {
            host: "0.0.0.0".into(),
            port: 8080,
            db_path: "/data/mycelic.db".into(),
            instance_id: "mycelic-main".into(),
            log_level: "INFO".into(),
            nats_url: Some("nats://nats:4222".into()),
            nats_user: None,
            nats_password: None,
            nats_token: None,
            nats_ca_file: None,
            nats_stream: "MYCELIC".into(),
            nats_consumer: "mycelic-main".into(),
            nats_max_age_seconds: 0,
            nats_max_bytes: -1,
            nats_duplicate_window_seconds: 7200,
            nats_ack_wait_seconds: 30,
            nats_max_deliver: 8,
            publish_batch: 100,
            publish_interval_seconds: 0.2,
            consume_batch: 1,
            admin_token: None,
            metrics_token: None,
            ready_requires_nats: false,
            tls_cert_file: None,
            tls_key_file: None,
            allowed_hosts: Vec::new(),
            cors_origins: Vec::new(),
            rate_limit_rps: 50.0,
            rate_limit_burst: 100,
            max_body_bytes: 1024 * 1024,
            max_text_chars: 4000,
            max_batch: 100,
            max_event_bytes: 256 * 1024,
            trust_proxy_headers: false,
            trusted_proxy_hops: 1,
            audit_retention_days: 90,
            event_signing_key: None,
            min_support: 2,
            rules_file: None,
            public_url: None,
        }
    }
}

/// Python's `int(s)` for a decimal string: surrounding whitespace, a sign, `_` between digits.
pub fn py_int(raw: &str) -> Option<i64> {
    let s = crate::textutil::py_strip(raw);
    let (sign, body) = match s.strip_prefix('-') {
        Some(rest) => (-1, rest),
        None => (1, s.strip_prefix('+').unwrap_or(s)),
    };
    if body.is_empty() || body.starts_with('_') || body.ends_with('_') || body.contains("__") {
        return None;
    }
    let digits: String = body.chars().filter(|c| *c != '_').collect();
    if !digits.chars().all(|c| c.is_ascii_digit()) {
        return None;
    }
    digits.parse::<i64>().ok().map(|v| sign * v)
}

/// Python's `float(s)`: whitespace, `_` between digits, `inf`/`infinity`/`nan` in any case.
pub fn py_float(raw: &str) -> Option<f64> {
    let s = crate::textutil::py_strip(raw);
    if s.is_empty() {
        return None;
    }
    let lower = s.to_ascii_lowercase();
    let unsigned = lower.trim_start_matches(['+', '-']);
    if matches!(unsigned, "inf" | "infinity" | "nan") {
        return lower.parse::<f64>().ok();
    }
    let bytes = s.as_bytes();
    for (i, b) in bytes.iter().enumerate() {
        if *b == b'_' {
            let ok = i > 0 && i + 1 < bytes.len() && bytes[i - 1].is_ascii_digit() && bytes[i + 1].is_ascii_digit();
            if !ok {
                return None;
            }
        } else if !(b.is_ascii_digit() || matches!(b, b'.' | b'e' | b'E' | b'+' | b'-')) {
            return None;
        }
    }
    s.replace('_', "").parse::<f64>().ok()
}

/// Reads `MYCELIC_*` values through a lookup function (the process environment, or a map in tests).
struct Env<'a> {
    get: &'a dyn Fn(&str) -> Option<String>,
}

impl Env<'_> {
    fn raw(&self, name: &str) -> Option<String> {
        (self.get)(name)
    }

    fn int(&self, name: &str, default: i64, minimum: Option<i64>) -> Result<i64, ConfigError> {
        let Some(raw) = self.raw(name).filter(|r| !r.is_empty()) else { return Ok(default) };
        let value =
            py_int(&raw).ok_or_else(|| ConfigError(format!("{name} must be an integer, got {}", repr_str(&raw))))?;
        if let Some(m) = minimum
            && value < m
        {
            return Err(ConfigError(format!("{name} must be >= {m}, got {value}")));
        }
        Ok(value)
    }

    fn float(&self, name: &str, default: f64, minimum: Option<f64>) -> Result<f64, ConfigError> {
        let Some(raw) = self.raw(name).filter(|r| !r.is_empty()) else { return Ok(default) };
        let value =
            py_float(&raw).ok_or_else(|| ConfigError(format!("{name} must be a number, got {}", repr_str(&raw))))?;
        if let Some(m) = minimum
            && value < m
        {
            return Err(ConfigError(format!(
                "{name} must be >= {}, got {}",
                crate::pyjson::float_repr(m),
                crate::pyjson::float_json(value)
            )));
        }
        Ok(value)
    }

    fn boolean(&self, name: &str, default: bool) -> bool {
        match self.raw(name).filter(|r| !r.is_empty()) {
            None => default,
            Some(raw) => matches!(crate::textutil::py_strip(&raw).to_lowercase().as_str(), "1" | "true" | "yes" | "on"),
        }
    }

    fn string(&self, name: &str, default: Option<&str>) -> Option<String> {
        match self.raw(name) {
            Some(raw) if !raw.is_empty() => Some(raw),
            _ => default.map(str::to_string),
        }
    }

    fn list(&self, name: &str) -> Vec<String> {
        self.raw(name)
            .unwrap_or_default()
            .split(',')
            .map(|x| crate::textutil::py_strip(x).to_string())
            .filter(|x| !x.is_empty())
            .collect()
    }
}

impl Settings {
    pub fn tls_enabled(&self) -> bool {
        self.tls_cert_file.as_deref().is_some_and(|s| !s.is_empty())
            && self.tls_key_file.as_deref().is_some_and(|s| !s.is_empty())
    }

    pub fn nats_enabled(&self) -> bool {
        self.nats_url.as_deref().is_some_and(|s| !s.is_empty())
    }

    pub fn is_loopback(&self) -> bool {
        matches!(self.host.as_str(), "127.0.0.1" | "localhost" | "::1")
    }

    pub fn from_env() -> Result<Settings, ConfigError> {
        Self::from_lookup(&|name: &str| std::env::var(name).ok())
    }

    pub fn from_lookup(get: &dyn Fn(&str) -> Option<String>) -> Result<Settings, ConfigError> {
        let e = Env { get };
        let s = Settings {
            host: e.string("MYCELIC_HOST", Some("0.0.0.0")).unwrap_or_default(),
            port: e.int("MYCELIC_PORT", 8080, Some(1))?,
            db_path: e.string("MYCELIC_DB_PATH", Some("/data/mycelic.db")).unwrap_or_default(),
            instance_id: e.string("MYCELIC_INSTANCE_ID", Some("mycelic-main")).unwrap_or_default(),
            log_level: e.string("MYCELIC_LOG_LEVEL", Some("INFO")).unwrap_or_default().to_uppercase(),
            nats_url: e.string("MYCELIC_NATS_URL", Some("nats://nats:4222")),
            nats_user: e.string("MYCELIC_NATS_USER", None),
            nats_password: e.string("MYCELIC_NATS_PASSWORD", None),
            nats_token: e.string("MYCELIC_NATS_TOKEN", None),
            nats_ca_file: e.string("MYCELIC_NATS_CA_FILE", None),
            nats_stream: e.string("MYCELIC_NATS_STREAM", Some("MYCELIC")).unwrap_or_default(),
            nats_consumer: e.string("MYCELIC_NATS_CONSUMER", Some("mycelic-main")).unwrap_or_default(),
            nats_max_age_seconds: e.int("MYCELIC_NATS_MAX_AGE_SECONDS", 0, Some(0))?,
            nats_max_bytes: e.int("MYCELIC_NATS_MAX_BYTES", -1, None)?,
            nats_duplicate_window_seconds: e.int("MYCELIC_NATS_DUPLICATE_WINDOW_SECONDS", 7200, Some(1))?,
            nats_ack_wait_seconds: e.int("MYCELIC_NATS_ACK_WAIT_SECONDS", 30, Some(1))?,
            nats_max_deliver: e.int("MYCELIC_NATS_MAX_DELIVER", 8, Some(1))?,
            publish_batch: e.int("MYCELIC_PUBLISH_BATCH", 100, Some(1))?,
            publish_interval_seconds: e.float("MYCELIC_PUBLISH_INTERVAL_SECONDS", 0.2, Some(0.01))?,
            consume_batch: e.int("MYCELIC_CONSUME_BATCH", 1, Some(1))?,
            admin_token: e.string("MYCELIC_ADMIN_TOKEN", None),
            metrics_token: e.string("MYCELIC_METRICS_TOKEN", None),
            ready_requires_nats: e.boolean("MYCELIC_READY_REQUIRES_NATS", false),
            tls_cert_file: e.string("MYCELIC_TLS_CERT_FILE", None),
            tls_key_file: e.string("MYCELIC_TLS_KEY_FILE", None),
            allowed_hosts: e.list("MYCELIC_ALLOWED_HOSTS"),
            cors_origins: e.list("MYCELIC_CORS_ORIGINS"),
            rate_limit_rps: e.float("MYCELIC_RATE_LIMIT_RPS", 50.0, Some(0.0))?,
            rate_limit_burst: e.int("MYCELIC_RATE_LIMIT_BURST", 100, Some(1))?,
            max_body_bytes: e.int("MYCELIC_MAX_BODY_BYTES", 1024 * 1024, Some(1024))?,
            max_text_chars: e.int("MYCELIC_MAX_TEXT_CHARS", 4000, Some(16))?,
            max_batch: e.int("MYCELIC_MAX_BATCH", 100, Some(1))?,
            max_event_bytes: e.int("MYCELIC_MAX_EVENT_BYTES", 256 * 1024, Some(4096))?,
            trust_proxy_headers: e.boolean("MYCELIC_TRUST_PROXY_HEADERS", false),
            trusted_proxy_hops: e.int("MYCELIC_TRUSTED_PROXY_HOPS", 1, Some(1))?,
            audit_retention_days: e.int("MYCELIC_AUDIT_RETENTION_DAYS", 90, Some(1))?,
            event_signing_key: e.string("MYCELIC_EVENT_SIGNING_KEY", None),
            min_support: e.int("MYCELIC_MIN_SUPPORT", 2, Some(1))?,
            rules_file: e.string("MYCELIC_RULES_FILE", None),
            public_url: e.string("MYCELIC_PUBLIC_URL", None),
        };
        s.validate()?;
        Ok(s)
    }

    pub fn validate(&self) -> Result<(), ConfigError> {
        if let Some(url) = self.nats_url.as_deref().filter(|u| !u.is_empty()) {
            if !["nats://", "tls://", "ws://", "wss://"].iter().any(|p| url.starts_with(p)) {
                return Err(ConfigError("MYCELIC_NATS_URL must start with nats://, tls://, ws:// or wss://".into()));
            }
            if url_has_credentials(url) {
                return Err(ConfigError(
                    "do not put credentials in MYCELIC_NATS_URL; use MYCELIC_NATS_USER / MYCELIC_NATS_PASSWORD".into(),
                ));
            }
        }
        for (name, value) in [
            ("MYCELIC_ADMIN_TOKEN", &self.admin_token),
            ("MYCELIC_EVENT_SIGNING_KEY", &self.event_signing_key),
            ("MYCELIC_NATS_PASSWORD", &self.nats_password),
            ("MYCELIC_METRICS_TOKEN", &self.metrics_token),
        ] {
            if let Some(v) = value.as_deref().filter(|v| !v.is_empty()) {
                let lower = v.to_lowercase();
                if ["change", "example", "replace", "placeholder"].iter().any(|w| lower.contains(w)) {
                    return Err(ConfigError(format!(
                        "{name} looks like a placeholder; generate one with: openssl rand -hex 32"
                    )));
                }
            }
        }
        if let Some(k) = &self.event_signing_key
            && k.chars().count() < 32
        {
            return Err(ConfigError(
                "MYCELIC_EVENT_SIGNING_KEY must be at least 32 characters (openssl rand -hex 32)".into(),
            ));
        }
        let set = |v: &Option<String>| v.as_deref().is_some_and(|s| !s.is_empty());
        if set(&self.tls_cert_file) != set(&self.tls_key_file) {
            return Err(ConfigError("MYCELIC_TLS_CERT_FILE and MYCELIC_TLS_KEY_FILE must be set together".into()));
        }
        for (name, path) in [
            ("MYCELIC_TLS_CERT_FILE", &self.tls_cert_file),
            ("MYCELIC_TLS_KEY_FILE", &self.tls_key_file),
            ("MYCELIC_NATS_CA_FILE", &self.nats_ca_file),
            ("MYCELIC_RULES_FILE", &self.rules_file),
        ] {
            if let Some(p) = path.as_deref().filter(|p| !p.is_empty())
                && !Path::new(p).is_file()
            {
                return Err(ConfigError(format!("{name} points to a file that does not exist: {p}")));
            }
        }
        if let Some(t) = &self.admin_token
            && t.chars().count() < 32
        {
            return Err(ConfigError(
                "MYCELIC_ADMIN_TOKEN must be at least 32 characters (openssl rand -hex 32)".into(),
            ));
        }
        if !self.is_loopback() && !set(&self.admin_token) {
            return Err(ConfigError("MYCELIC_ADMIN_TOKEN is required when binding to a non-loopback address".into()));
        }
        Ok(())
    }

    /// For logs and `/admin/status`: every field, each secret replaced by its presence.
    pub fn redacted(&self) -> Value {
        let secret =
            |v: &Option<String>| if v.as_deref().is_some_and(|s| !s.is_empty()) { json!("set") } else { Value::Null };
        let mut m = Map::new();
        let mut put = |k: &str, v: Value| {
            m.insert(k.to_string(), v);
        };
        put("host", json!(self.host));
        put("port", json!(self.port));
        put("db_path", json!(self.db_path));
        put("instance_id", json!(self.instance_id));
        put("log_level", json!(self.log_level));
        put("nats_url", json!(self.nats_url));
        put("nats_user", json!(self.nats_user));
        put("nats_password", secret(&self.nats_password));
        put("nats_token", secret(&self.nats_token));
        put("nats_ca_file", json!(self.nats_ca_file));
        put("nats_stream", json!(self.nats_stream));
        put("nats_consumer", json!(self.nats_consumer));
        put("nats_max_age_seconds", json!(self.nats_max_age_seconds));
        put("nats_max_bytes", json!(self.nats_max_bytes));
        put("nats_duplicate_window_seconds", json!(self.nats_duplicate_window_seconds));
        put("nats_ack_wait_seconds", json!(self.nats_ack_wait_seconds));
        put("nats_max_deliver", json!(self.nats_max_deliver));
        put("publish_batch", json!(self.publish_batch));
        put("publish_interval_seconds", float_value(self.publish_interval_seconds));
        put("consume_batch", json!(self.consume_batch));
        put("admin_token", secret(&self.admin_token));
        put("metrics_token", secret(&self.metrics_token));
        put("ready_requires_nats", json!(self.ready_requires_nats));
        put("tls_cert_file", json!(self.tls_cert_file));
        put("tls_key_file", json!(self.tls_key_file));
        put("allowed_hosts", json!(self.allowed_hosts));
        put("cors_origins", json!(self.cors_origins));
        put("rate_limit_rps", float_value(self.rate_limit_rps));
        put("rate_limit_burst", json!(self.rate_limit_burst));
        put("max_body_bytes", json!(self.max_body_bytes));
        put("max_text_chars", json!(self.max_text_chars));
        put("max_batch", json!(self.max_batch));
        put("max_event_bytes", json!(self.max_event_bytes));
        put("trust_proxy_headers", json!(self.trust_proxy_headers));
        put("trusted_proxy_hops", json!(self.trusted_proxy_hops));
        put("audit_retention_days", json!(self.audit_retention_days));
        put("event_signing_key", secret(&self.event_signing_key));
        put("min_support", json!(self.min_support));
        put("rules_file", json!(self.rules_file));
        put("public_url", json!(self.public_url));
        Value::Object(m)
    }
}

/// A float field as JSON; infinities (accepted from the environment as Python accepted them) have no JSON form,
/// so they are shown as the strings Python's encoder would have produced.
fn float_value(f: f64) -> Value {
    serde_json::Number::from_f64(f).map(Value::Number).unwrap_or_else(|| json!(crate::pyjson::float_json(f)))
}

/// `urlsplit(url).username or urlsplit(url).password`.
fn url_has_credentials(url: &str) -> bool {
    let Some((_, rest)) = url.split_once("://") else { return false };
    let netloc = rest.split(['/', '?', '#']).next().unwrap_or("");
    let Some((userinfo, _)) = netloc.rsplit_once('@') else { return false };
    let (user, password) = match userinfo.split_once(':') {
        Some((u, p)) => (u, p),
        None => (userinfo, ""),
    };
    !user.is_empty() || !password.is_empty()
}
