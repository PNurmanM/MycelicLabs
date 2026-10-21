//! Message transport: NATS JetStream in production, an in-process log for unit tests and development.
//!
//! Both implement the same small surface the service needs:
//!
//! ```text
//! t.connect()                               idempotent; JetStream: ensure stream + durable consumer
//! t.publish(subject, bytes, msg_id, hdrs)   at-least-once; msg_id is the dedup key
//! t.fetch(batch, timeout)                   deliveries; each is ack()-ed, nak()-ed or term()-ed
//! t.reset_consumer(start_seq)               replay: deliver everything from the first (or a given) sequence again
//! t.info()                                  connected?, stream/consumer positions for /health
//! ```
//!
//! JetStream specifics that matter for correctness:
//!
//! * the stream is file-backed with no age limit by default, because a rebuild after losing the Mycelic database
//!   replays it from sequence 1;
//! * every publish carries `Nats-Msg-Id` = the event id, so a retry after a lost ack is deduplicated inside the
//!   stream's duplicate window; beyond the window the consumer's idempotent apply covers it;
//! * the consumer is a *durable pull* consumer with explicit acks: a message is acked only after the apply
//!   transaction committed, so a crash in between redelivers it and the second apply is a no-op;
//! * a message that keeps failing is `term`-ed after `max_deliver` attempts and recorded as failed, so one poison
//!   event cannot stall the log.
//!
//! Subjects, payload bytes, headers (`Nats-Msg-Id`, `Mycelic-Signature`), stream and consumer settings are the
//! Python service's, so either implementation consumes a stream the other wrote.

use crate::config::Settings;
use anyhow::{Result, anyhow};
use async_nats::jetstream::{self, consumer::pull, stream};
use async_trait::async_trait;
use bytes::Bytes;
use futures::StreamExt;
use serde_json::{Map, Value, json};
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::Duration;
use tokio::sync::Notify;

pub const SUBJECT_PREFIX: &str = "mycelic";

/// `mycelic.<org>.<kind-with-dashes>` (a NATS subject token cannot contain dots).
pub fn subject_for(org_id: &str, kind: &str) -> String {
    format!("{SUBJECT_PREFIX}.{org_id}.{}", kind.replace('.', "-"))
}

#[async_trait]
pub trait Delivery: Send + Sync {
    fn data(&self) -> &[u8];
    fn subject(&self) -> &str;
    fn seq(&self) -> Option<i64>;
    fn num_delivered(&self) -> i64;
    /// A header value by exact name (`Mycelic-Signature`, `Nats-Msg-Id`).
    fn header(&self, name: &str) -> Option<String>;
    async fn ack(&self) -> Result<()>;
    async fn nak(&self) -> Result<()>;
    async fn term(&self) -> Result<()>;
}

/// Why a publish failed: `Permanent` means the broker will never take this event (too large, rejected as a bad
/// request), so it is recorded as failed and the log moves on; `Transient` is retried with backoff.
#[derive(Debug)]
pub enum PublishError {
    Permanent(String),
    Transient(String),
}

impl std::fmt::Display for PublishError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PublishError::Permanent(s) | PublishError::Transient(s) => f.write_str(s),
        }
    }
}

#[async_trait]
pub trait Transport: Send + Sync {
    fn name(&self) -> &'static str;
    fn connected(&self) -> bool;
    /// True only when there is no live client.  A client that is reconnecting on its own is left alone.
    fn needs_connect(&self) -> bool;
    async fn connect(&self) -> Result<()>;
    async fn close(&self);
    async fn publish(
        &self,
        subject: &str,
        payload: Bytes,
        msg_id: &str,
        headers: &[(String, String)],
    ) -> Result<Option<i64>, PublishError>;
    async fn fetch(&self, batch: usize, timeout: Duration) -> Result<Vec<Box<dyn Delivery>>>;
    async fn reset_consumer(&self, start_seq: Option<i64>) -> Result<()>;
    async fn info(&self) -> Map<String, Value>;
    /// Stream sequence up to which the durable consumer acknowledged everything; None when unknown (or when the
    /// transport has no such notion, like the in-process log).
    async fn consumer_ack_floor(&self) -> Option<i64> {
        None
    }
    fn reconnects(&self) -> u64 {
        0
    }
}

pub fn info_i64(info: &Map<String, Value>, key: &str) -> Option<i64> {
    info.get(key).and_then(Value::as_i64)
}

// ---------------------------------------------------------------------------------------------------- in-process
#[derive(Default)]
struct MemLog {
    log: Vec<(String, Bytes, String, Vec<(String, String)>)>,
    seen_ids: HashMap<String, i64>,
    cursor: usize,
    inflight: HashMap<i64, i64>,
    redeliver: VecDeque<i64>,
    delivered: HashMap<i64, i64>,
    terminated: Vec<i64>,
    connected: bool,
}

/// An append-only log with one cursor.  Survives nothing; exists so the whole service loop runs in tests and in
/// a broker-less development setup (`MYCELIC_NATS_URL` empty on the command line).
#[derive(Clone, Default)]
pub struct InProcessTransport {
    state: Arc<Mutex<MemLog>>,
    wake: Arc<Notify>,
}

impl InProcessTransport {
    pub fn new() -> InProcessTransport {
        InProcessTransport::default()
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, MemLog> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    pub fn terminated(&self) -> Vec<i64> {
        self.lock().terminated.clone()
    }

    fn deliver(&self, st: &mut MemLog, seq: i64) -> Box<dyn Delivery> {
        let (subject, payload, _msg_id, headers) = st.log[(seq - 1) as usize].clone();
        let n = st.delivered.get(&seq).copied().unwrap_or(0) + 1;
        st.delivered.insert(seq, n);
        st.inflight.insert(seq, n);
        Box::new(MemDelivery { data: payload, subject, seq, num_delivered: n, headers, log: self.clone() })
    }
}

struct MemDelivery {
    data: Bytes,
    subject: String,
    seq: i64,
    num_delivered: i64,
    headers: Vec<(String, String)>,
    log: InProcessTransport,
}

#[async_trait]
impl Delivery for MemDelivery {
    fn data(&self) -> &[u8] {
        &self.data
    }
    fn subject(&self) -> &str {
        &self.subject
    }
    fn seq(&self) -> Option<i64> {
        Some(self.seq)
    }
    fn num_delivered(&self) -> i64 {
        self.num_delivered
    }
    fn header(&self, name: &str) -> Option<String> {
        self.headers.iter().find(|(k, _)| k == name).map(|(_, v)| v.clone())
    }
    async fn ack(&self) -> Result<()> {
        self.log.lock().inflight.remove(&self.seq);
        Ok(())
    }
    async fn nak(&self) -> Result<()> {
        let mut st = self.log.lock();
        st.inflight.remove(&self.seq);
        st.redeliver.push_back(self.seq);
        Ok(())
    }
    async fn term(&self) -> Result<()> {
        let mut st = self.log.lock();
        st.inflight.remove(&self.seq);
        st.terminated.push(self.seq);
        Ok(())
    }
}

#[async_trait]
impl Transport for InProcessTransport {
    fn name(&self) -> &'static str {
        "in-process"
    }

    fn connected(&self) -> bool {
        self.lock().connected
    }

    fn needs_connect(&self) -> bool {
        !self.lock().connected
    }

    async fn connect(&self) -> Result<()> {
        self.lock().connected = true;
        Ok(())
    }

    async fn close(&self) {
        self.lock().connected = false;
    }

    async fn publish(
        &self,
        subject: &str,
        payload: Bytes,
        msg_id: &str,
        headers: &[(String, String)],
    ) -> Result<Option<i64>, PublishError> {
        let seq = {
            let mut st = self.lock();
            if !st.connected {
                return Err(PublishError::Transient("ConnectionError: in-process transport is closed".into()));
            }
            if let Some(seq) = st.seen_ids.get(msg_id) {
                return Ok(Some(*seq));
            }
            st.log.push((subject.to_string(), payload, msg_id.to_string(), headers.to_vec()));
            let seq = st.log.len() as i64;
            st.seen_ids.insert(msg_id.to_string(), seq);
            seq
        };
        self.wake.notify_one();
        Ok(Some(seq))
    }

    async fn fetch(&self, batch: usize, timeout: Duration) -> Result<Vec<Box<dyn Delivery>>> {
        let mut out: Vec<Box<dyn Delivery>> = Vec::new();
        let must_wait = {
            let mut st = self.lock();
            if !st.connected {
                return Err(anyhow!("ConnectionError: in-process transport is closed"));
            }
            while out.len() < batch {
                let Some(seq) = st.redeliver.pop_front() else { break };
                let d = self.deliver(&mut st, seq);
                out.push(d);
            }
            out.is_empty() && st.cursor >= st.log.len()
        };
        if must_wait && tokio::time::timeout(timeout, self.wake.notified()).await.is_err() {
            return Ok(out);
        }
        let mut st = self.lock();
        while st.cursor < st.log.len() && out.len() < batch {
            st.cursor += 1;
            let seq = st.cursor as i64;
            let d = self.deliver(&mut st, seq);
            out.push(d);
        }
        Ok(out)
    }

    async fn reset_consumer(&self, start_seq: Option<i64>) -> Result<()> {
        let mut st = self.lock();
        st.cursor = match start_seq {
            Some(s) if s != 0 => (s - 1).max(0) as usize,
            _ => 0,
        };
        st.redeliver.clear();
        st.inflight.clear();
        st.delivered.clear();
        Ok(())
    }

    async fn info(&self) -> Map<String, Value> {
        let st = self.lock();
        let n = st.log.len();
        match json!({"transport": "in-process", "connected": st.connected, "stream_messages": n,
                     "consumer_pending": n.saturating_sub(st.cursor), "last_seq": n})
        {
            Value::Object(m) => m,
            _ => Map::new(),
        }
    }
}

// ---------------------------------------------------------------------------------------------------- JetStream
struct JsDelivery {
    msg: jetstream::Message,
    seq: Option<i64>,
    num_delivered: i64,
}

#[async_trait]
impl Delivery for JsDelivery {
    fn data(&self) -> &[u8] {
        &self.msg.payload
    }
    fn subject(&self) -> &str {
        self.msg.subject.as_str()
    }
    fn seq(&self) -> Option<i64> {
        self.seq
    }
    fn num_delivered(&self) -> i64 {
        self.num_delivered
    }
    fn header(&self, name: &str) -> Option<String> {
        let headers = self.msg.headers.as_ref()?;
        headers.iter().find(|(k, _)| k.to_string() == name).and_then(|(_, v)| v.first().map(|v| v.as_str().to_string()))
    }
    async fn ack(&self) -> Result<()> {
        self.msg.ack().await.map_err(|e| anyhow!("{e}"))
    }
    async fn nak(&self) -> Result<()> {
        self.msg.ack_with(jetstream::AckKind::Nak(None)).await.map_err(|e| anyhow!("{e}"))
    }
    async fn term(&self) -> Result<()> {
        self.msg.ack_with(jetstream::AckKind::Term).await.map_err(|e| anyhow!("{e}"))
    }
}

#[derive(Clone)]
struct Handles {
    client: async_nats::Client,
    js: jetstream::Context,
    consumer: Option<jetstream::consumer::Consumer<pull::Config>>,
}

pub type StateCallback = Arc<dyn Fn(bool) + Send + Sync>;

pub struct JetStreamTransport {
    s: Settings,
    handles: RwLock<Option<Handles>>,
    connected: Arc<AtomicBool>,
    reconnects: Arc<AtomicU64>,
    last_error: Arc<Mutex<Option<String>>>,
    stream_bounded: AtomicBool,
    on_state: StateCallback,
    filter_subject: String,
}

fn type_name<E: std::fmt::Debug>(e: &E) -> String {
    let dbg = format!("{e:?}");
    dbg.split(['(', ' ', '{']).next().unwrap_or("Error").to_string()
}

impl JetStreamTransport {
    pub fn new(settings: Settings, on_state: StateCallback) -> JetStreamTransport {
        JetStreamTransport {
            s: settings,
            handles: RwLock::new(None),
            connected: Arc::new(AtomicBool::new(false)),
            reconnects: Arc::new(AtomicU64::new(0)),
            last_error: Arc::new(Mutex::new(None)),
            stream_bounded: AtomicBool::new(false),
            on_state,
            filter_subject: format!("{SUBJECT_PREFIX}.>"),
        }
    }

    fn handles(&self) -> Option<Handles> {
        self.handles.read().unwrap_or_else(|e| e.into_inner()).clone()
    }

    fn set_connected(connected: &AtomicBool, on_state: &StateCallback, value: bool) {
        connected.store(value, Ordering::SeqCst);
        on_state(value);
    }

    fn stream_config(&self) -> stream::Config {
        stream::Config {
            name: self.s.nats_stream.clone(),
            subjects: vec![self.filter_subject.clone()],
            storage: stream::StorageType::File,
            retention: stream::RetentionPolicy::Limits,
            // when a limit is hit, refuse new events loudly; never drop history
            discard: stream::DiscardPolicy::New,
            max_age: Duration::from_secs(self.s.nats_max_age_seconds.max(0) as u64),
            max_bytes: self.s.nats_max_bytes,
            max_messages: -1,
            num_replicas: 1,
            duplicate_window: Duration::from_secs(self.s.nats_duplicate_window_seconds.max(1) as u64),
            description: Some("Mycelic event log: agent observations and derived organizational memories".into()),
            ..Default::default()
        }
    }

    async fn ensure_stream(&self, js: &jetstream::Context) -> Result<stream::Stream> {
        let config = self.stream_config();
        match js.get_stream(&self.s.nats_stream).await {
            Ok(existing) => {
                let c = &existing.cached_info().config;
                if c.max_age > Duration::ZERO || c.max_bytes > 0 || c.max_messages > 0 {
                    tracing::error!(
                        "stream {} is bounded (max_age={:?} max_bytes={} max_msgs={}): a rebuild from replay may be partial",
                        self.s.nats_stream,
                        c.max_age,
                        c.max_bytes,
                        c.max_messages
                    );
                    self.stream_bounded.store(true, Ordering::SeqCst);
                }
                if let Err(e) = js.update_stream(config).await {
                    tracing::warn!("stream update rejected ({e}); keeping the existing configuration");
                }
                Ok(js.get_stream(&self.s.nats_stream).await?)
            }
            Err(e) => {
                let not_found = matches!(e.kind(), jetstream::context::GetStreamErrorKind::JetStream(ref je)
                    if je.error_code() == jetstream::ErrorCode::STREAM_NOT_FOUND || je.code() == 404);
                if !not_found {
                    return Err(anyhow!("stream info failed: {e}"));
                }
                Ok(js.create_stream(config).await?)
            }
        }
    }

    /// Bind the durable pull consumer; `reset` recreates it from the first sequence (or `start_seq`).
    ///
    /// `max_ack_pending` equals the fetch batch, so with the default batch of 1 the stream is applied strictly in
    /// order: a failed message is redelivered in place (the consumer sleeps, then naks) and nothing overtakes it.
    async fn ensure_consumer(
        &self,
        stream: &stream::Stream,
        reset: bool,
        start_seq: Option<i64>,
    ) -> Result<jetstream::consumer::Consumer<pull::Config>> {
        if reset {
            match stream.delete_consumer(&self.s.nats_consumer).await {
                Ok(_) => tracing::info!("deleted durable consumer {} for replay", self.s.nats_consumer),
                Err(e) => tracing::debug!("delete consumer {}: {e}", self.s.nats_consumer),
            }
        }
        let deliver_policy = match start_seq {
            Some(s) if s > 0 => jetstream::consumer::DeliverPolicy::ByStartSequence { start_sequence: s as u64 },
            _ => jetstream::consumer::DeliverPolicy::All,
        };
        let config = pull::Config {
            durable_name: Some(self.s.nats_consumer.clone()),
            deliver_policy,
            ack_policy: jetstream::consumer::AckPolicy::Explicit,
            ack_wait: Duration::from_secs(self.s.nats_ack_wait_seconds.max(1) as u64),
            max_deliver: self.s.nats_max_deliver,
            filter_subject: self.filter_subject.clone(),
            max_ack_pending: self.s.consume_batch.max(1),
            ..Default::default()
        };
        Ok(stream.get_or_create_consumer(&self.s.nats_consumer, config).await?)
    }
}

#[async_trait]
impl Transport for JetStreamTransport {
    fn name(&self) -> &'static str {
        "nats-jetstream"
    }

    fn connected(&self) -> bool {
        self.connected.load(Ordering::SeqCst)
            && self.handles().is_some_and(|h| h.client.connection_state() == async_nats::connection::State::Connected)
    }

    fn needs_connect(&self) -> bool {
        self.handles().is_none()
    }

    async fn connect(&self) -> Result<()> {
        if self.handles().is_some() {
            return Ok(()); // connected, or reconnecting in the background: the events flip `connected`
        }
        let url = self.s.nats_url.clone().unwrap_or_default();
        let connected = self.connected.clone();
        let reconnects = self.reconnects.clone();
        let last_error = self.last_error.clone();
        let on_state = self.on_state.clone();
        let was_disconnected = Arc::new(AtomicBool::new(false));
        let mut options = async_nats::ConnectOptions::new()
            .name(&self.s.instance_id)
            .max_reconnects(None)
            // the callback is consulted before the very first attempt too: no delay then, a second between reconnects
            .reconnect_delay_callback(|attempts| if attempts == 0 { Duration::ZERO } else { Duration::from_secs(1) })
            .connection_timeout(Duration::from_secs(5))
            .event_callback(move |event| {
                let (connected, reconnects, last_error, on_state, was_disconnected) = (
                    connected.clone(),
                    reconnects.clone(),
                    last_error.clone(),
                    on_state.clone(),
                    was_disconnected.clone(),
                );
                async move {
                    match event {
                        async_nats::Event::Disconnected => {
                            tracing::warn!("NATS disconnected");
                            was_disconnected.store(true, Ordering::SeqCst);
                            JetStreamTransport::set_connected(&connected, &on_state, false);
                        }
                        async_nats::Event::Connected => {
                            if was_disconnected.swap(false, Ordering::SeqCst) {
                                let n = reconnects.fetch_add(1, Ordering::SeqCst) + 1;
                                tracing::info!("NATS reconnected ({n})");
                                JetStreamTransport::set_connected(&connected, &on_state, true);
                            }
                        }
                        async_nats::Event::Closed => {
                            tracing::warn!("NATS connection closed");
                            JetStreamTransport::set_connected(&connected, &on_state, false);
                        }
                        async_nats::Event::ServerError(e) => {
                            let msg = format!("ServerError: {e}");
                            tracing::warn!("NATS error: {msg}");
                            *last_error.lock().unwrap_or_else(|e| e.into_inner()) = Some(msg);
                        }
                        async_nats::Event::ClientError(e) => {
                            let msg = format!("ClientError: {e}");
                            tracing::warn!("NATS error: {msg}");
                            *last_error.lock().unwrap_or_else(|e| e.into_inner()) = Some(msg);
                        }
                        other => tracing::info!("NATS event: {other}"),
                    }
                }
            });
        if let (Some(u), Some(p)) =
            (self.s.nats_user.clone().filter(|u| !u.is_empty()), self.s.nats_password.clone().filter(|p| !p.is_empty()))
        {
            options = options.user_and_password(u, p);
        }
        if let Some(t) = self.s.nats_token.clone().filter(|t| !t.is_empty()) {
            options = options.token(t);
        }
        if let Some(ca) = self.s.nats_ca_file.clone().filter(|c| !c.is_empty()) {
            options = options.add_root_certificates(ca.into()).require_tls(true);
        } else if url.starts_with("tls://") {
            options = options.require_tls(true);
        }
        let client = options.connect(url.as_str()).await.map_err(|e| anyhow!("{}: {e}", type_name(&e.kind())))?;
        let mut js = jetstream::new(client.clone());
        js.set_timeout(Duration::from_secs(5));
        let stream = self.ensure_stream(&js).await?;
        let consumer = self.ensure_consumer(&stream, false, None).await?;
        *self.handles.write().unwrap_or_else(|e| e.into_inner()) =
            Some(Handles { client, js, consumer: Some(consumer) });
        Self::set_connected(&self.connected, &self.on_state, true);
        tracing::info!("connected to NATS {url}, stream {}, consumer {}", self.s.nats_stream, self.s.nats_consumer);
        Ok(())
    }

    async fn close(&self) {
        let h = self.handles.write().unwrap_or_else(|e| e.into_inner()).take();
        if let Some(h) = h {
            let _ = tokio::time::timeout(Duration::from_secs(5), h.client.drain()).await;
        }
        Self::set_connected(&self.connected, &self.on_state, false);
    }

    async fn publish(
        &self,
        subject: &str,
        payload: Bytes,
        msg_id: &str,
        headers: &[(String, String)],
    ) -> Result<Option<i64>, PublishError> {
        let Some(h) = self.handles() else {
            return Err(PublishError::Transient("ConnectionError: transport not connected".into()));
        };
        let mut hm = async_nats::HeaderMap::new();
        for (k, v) in headers {
            hm.insert(k.as_str(), v.as_str());
        }
        hm.insert("Nats-Msg-Id", msg_id);
        let classify = |e: async_nats::jetstream::context::PublishError| {
            use async_nats::jetstream::context::PublishErrorKind as K;
            let api_400 = std::error::Error::source(&e)
                .and_then(|s| s.downcast_ref::<jetstream::Error>())
                .is_some_and(|je| je.code() == 400);
            let name = match e.kind() {
                K::MaxPayloadExceeded => "MaxPayloadError",
                K::TimedOut => "TimeoutError",
                K::StreamNotFound => "NoStreamResponseError",
                _ if api_400 => "BadRequestError",
                _ => "PublishError",
            };
            let text = format!("{name}: {e}");
            if matches!(e.kind(), K::MaxPayloadExceeded) || api_400 {
                PublishError::Permanent(text)
            } else {
                PublishError::Transient(text)
            }
        };
        let ack_future =
            tokio::time::timeout(Duration::from_secs(5), h.js.publish_with_headers(subject.to_string(), hm, payload))
                .await
                .map_err(|_| PublishError::Transient("TimeoutError: publish timed out".into()))?
                .map_err(classify)?;
        let ack = ack_future.await.map_err(classify)?;
        Ok(Some(ack.sequence as i64))
    }

    async fn fetch(&self, batch: usize, timeout: Duration) -> Result<Vec<Box<dyn Delivery>>> {
        let Some(consumer) = self.handles().and_then(|h| h.consumer) else {
            return Err(anyhow!("ConnectionError: transport not connected"));
        };
        let work = async {
            let mut msgs =
                consumer.batch().max_messages(batch).expires(timeout).messages().await.map_err(|e| anyhow!("{e}"))?;
            let mut out: Vec<Box<dyn Delivery>> = Vec::new();
            while let Some(m) = msgs.next().await {
                let m = m.map_err(|e| anyhow!("{e}"))?;
                let (seq, n) = match m.info() {
                    Ok(i) => (Some(i.stream_sequence as i64), i.delivered.max(1)),
                    Err(_) => (None, 1),
                };
                out.push(Box::new(JsDelivery { msg: m, seq, num_delivered: n }));
                if out.len() >= batch {
                    break;
                }
            }
            Ok::<_, anyhow::Error>(out)
        };
        match tokio::time::timeout(timeout + Duration::from_secs(5), work).await {
            Ok(r) => r,
            Err(_) => Ok(Vec::new()),
        }
    }

    async fn reset_consumer(&self, start_seq: Option<i64>) -> Result<()> {
        let Some(h) = self.handles() else { return Err(anyhow!("ConnectionError: transport not connected")) };
        let stream = h.js.get_stream(&self.s.nats_stream).await?;
        let consumer = self.ensure_consumer(&stream, true, start_seq).await?;
        if let Some(cur) = self.handles.write().unwrap_or_else(|e| e.into_inner()).as_mut() {
            cur.consumer = Some(consumer);
        }
        Ok(())
    }

    async fn info(&self) -> Map<String, Value> {
        let mut out = Map::new();
        out.insert("transport".into(), json!(self.name()));
        out.insert("connected".into(), json!(self.connected()));
        out.insert("url".into(), json!(self.s.nats_url));
        out.insert("stream".into(), json!(self.s.nats_stream));
        out.insert("consumer".into(), json!(self.s.nats_consumer));
        out.insert("reconnects".into(), json!(self.reconnects.load(Ordering::SeqCst)));
        out.insert("last_error".into(), json!(*self.last_error.lock().unwrap_or_else(|e| e.into_inner())));
        out.insert("stream_bounded".into(), json!(self.stream_bounded.load(Ordering::SeqCst)));
        let Some(h) = self.handles().filter(|_| self.connected()) else { return out };
        let result: Result<()> = async {
            let mut stream = h.js.get_stream(&self.s.nats_stream).await?;
            let si = stream.info().await?;
            out.insert("stream_messages".into(), json!(si.state.messages));
            out.insert("last_seq".into(), json!(si.state.last_sequence));
            out.insert("first_seq".into(), json!(si.state.first_sequence));
            out.insert("stream_bytes".into(), json!(si.state.bytes));
            let ci = stream.consumer_info(&self.s.nats_consumer).await?;
            out.insert("consumer_pending".into(), json!(ci.num_pending));
            out.insert("consumer_ack_pending".into(), json!(ci.num_ack_pending));
            out.insert("consumer_delivered_seq".into(), json!(ci.delivered.stream_sequence));
            out.insert("consumer_ack_floor_seq".into(), json!(ci.ack_floor.stream_sequence));
            Ok(())
        }
        .await;
        if let Err(e) = result {
            // broker mid-restart: report, do not raise
            out.insert("error".into(), json!(format!("{e}")));
        }
        out
    }

    async fn consumer_ack_floor(&self) -> Option<i64> {
        let h = self.handles()?;
        let stream = h.js.get_stream(&self.s.nats_stream).await.ok()?;
        let ci = stream.consumer_info(&self.s.nats_consumer).await.ok()?;
        Some(ci.ack_floor.stream_sequence as i64)
    }

    fn reconnects(&self) -> u64 {
        self.reconnects.load(Ordering::SeqCst)
    }
}

pub fn build_transport(settings: &Settings, on_state: StateCallback) -> Arc<dyn Transport> {
    if settings.nats_enabled() {
        return Arc::new(JetStreamTransport::new(settings.clone(), on_state));
    }
    tracing::warn!("MYCELIC_NATS_URL is empty: using the in-process transport (no durability across restarts)");
    Arc::new(InProcessTransport::new())
}
