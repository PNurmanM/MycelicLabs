//! HTTP throughput under concurrent load, against any running Mycelic (Rust or the old Python one): the same client
//! for both, so the numbers compare the services.
//!
//! ```text
//! cargo run --release -p mycelic --example http_load -- --url http://127.0.0.1:8080 --admin-token $TOKEN \
//!     [--threads 16] [--seconds 10] [--write-seconds 2] [--seed 2000] [--org bench]
//! ```
//!
//! Setup: 20 agents in 4 teams of one organization (rate limiting must be off on the service:
//! MYCELIC_RATE_LIMIT_RPS=0), `--seed` observations shared and applied.  Then two phases of `--seconds` each with
//! `--threads` keep-alive clients:
//!
//! * query: `POST /query` with three random vocabulary words at the organization scope (the default body, so the
//!   answer's lineage is reconstructed as well);
//! * write (`--write-seconds`, default `--seconds`): `POST /memory` with fresh idempotency keys, then the time until
//!   the service has applied every event (outbox empty, consumer caught up), which is the end-to-end apply rate
//!   including aggregation.

use serde_json::{Value, json};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

const W: &str = "port strike rotterdam terminal carrier eta supplier stock servo drive order customer november shipment \
                 delay forecast region emea apac warehouse inventory quality defect recall audit invoice contract price";

fn arg(name: &str, default: &str) -> String {
    let args: Vec<String> = std::env::args().collect();
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned().unwrap_or_else(|| default.to_string())
}

fn agent() -> ureq::Agent {
    ureq::Agent::config_builder()
        .timeout_global(Some(Duration::from_secs(30)))
        .http_status_as_error(false)
        .build()
        .into()
}

fn post(a: &ureq::Agent, url: &str, token: &str, body: &Value) -> (u16, Value) {
    match a
        .post(url)
        .header("Authorization", format!("Bearer {token}"))
        .header("Content-Type", "application/json")
        .send(body.to_string())
    {
        Ok(mut r) => {
            let status = r.status().as_u16();
            let v = r.body_mut().read_json::<Value>().unwrap_or(Value::Null);
            (status, v)
        }
        Err(_) => (0, Value::Null),
    }
}

fn status(a: &ureq::Agent, base: &str, token: &str) -> Value {
    a.get(format!("{base}/admin/status"))
        .header("Authorization", format!("Bearer {token}"))
        .call()
        .ok()
        .and_then(|mut r| r.body_mut().read_json::<Value>().ok())
        .unwrap_or(Value::Null)
}

fn wait_idle(a: &ureq::Agent, base: &str, token: &str) -> Duration {
    let t0 = Instant::now();
    loop {
        let st = status(a, base, token);
        let c = &st["checks"];
        let pending =
            c["publisher"]["outbox_pending"].as_i64().unwrap_or(1) + c["consumer"]["pending"].as_i64().unwrap_or(0);
        if pending == 0 && c["consumer"]["replaying_to_seq"].is_null() {
            return t0.elapsed();
        }
        std::thread::sleep(Duration::from_millis(50));
    }
}

struct Rng(u64);
impl Rng {
    fn below(&mut self, n: usize) -> usize {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        (self.0.wrapping_mul(0x2545F4914F6CDD1D) >> 33) as usize % n
    }
}

fn words(rng: &mut Rng, n: usize) -> String {
    let w: Vec<&str> = W.split_whitespace().collect();
    (0..n).map(|_| w[rng.below(w.len())]).collect::<Vec<_>>().join(" ")
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() { 0.0 } else { sorted[((sorted.len() - 1) as f64 * p).round() as usize] }
}

/// Run `threads` workers for `seconds`; each calls `work(thread, n)` and reports its status.
fn phase(
    threads: usize,
    seconds: f64,
    ok_status: u16,
    work: Arc<dyn Fn(usize, u64, &ureq::Agent) -> u16 + Send + Sync>,
) -> Value {
    let stop = Arc::new(AtomicBool::new(false));
    let done = Arc::new(AtomicU64::new(0));
    let failed = Arc::new(AtomicU64::new(0));
    let t0 = Instant::now();
    let handles: Vec<_> = (0..threads)
        .map(|t| {
            let (stop, done, failed, work) = (stop.clone(), done.clone(), failed.clone(), work.clone());
            std::thread::spawn(move || {
                let a = agent();
                let mut lat = Vec::new();
                let mut n = 0u64;
                while !stop.load(Ordering::Relaxed) {
                    let started = Instant::now();
                    let s = work(t, n, &a);
                    lat.push(started.elapsed().as_secs_f64() * 1e3);
                    n += 1;
                    if s == ok_status {
                        done.fetch_add(1, Ordering::Relaxed)
                    } else {
                        failed.fetch_add(1, Ordering::Relaxed)
                    };
                }
                lat
            })
        })
        .collect();
    std::thread::sleep(Duration::from_secs_f64(seconds));
    stop.store(true, Ordering::Relaxed);
    let mut lat: Vec<f64> = handles.into_iter().flat_map(|h| h.join().unwrap()).collect();
    let elapsed = t0.elapsed().as_secs_f64();
    lat.sort_by(f64::total_cmp);
    let ok = done.load(Ordering::Relaxed);
    json!({"requests_ok": ok, "requests_failed": failed.load(Ordering::Relaxed), "seconds": (elapsed * 100.0).round() / 100.0,
           "rps": (ok as f64 / elapsed).round(), "p50_ms": (percentile(&lat, 0.5) * 100.0).round() / 100.0,
           "p99_ms": (percentile(&lat, 0.99) * 100.0).round() / 100.0})
}

fn main() {
    let base = arg("--url", "http://127.0.0.1:8080");
    let token = arg("--admin-token", "");
    let threads: usize = arg("--threads", "16").parse().unwrap();
    let seconds: f64 = arg("--seconds", "10").parse().unwrap();
    let seed: usize = arg("--seed", "2000").parse().unwrap();
    // the write burst is kept short by default: its drain grows with what it wrote (consolidations re-render
    // every contribution on their topic), and the drain is what the apply rate is measured over
    let write_seconds: f64 = arg("--write-seconds", &seconds.to_string()).parse().unwrap();
    let org = arg("--org", "bench");
    let a = agent();

    // 20 agents in 4 teams
    let mut keys: Vec<String> = Vec::new();
    for i in 0..20 {
        let (s, v) = post(
            &a,
            &format!("{base}/admin/agents"),
            &token,
            &json!({"enterprise": org, "region": "r", "subsidiary": "s", "department": "d", "team": format!("t{}", i % 4), "agent_id": format!("bench-{i}")}),
        );
        assert_eq!(s, 201, "register: {v}");
        keys.push(v["api_key"].as_str().unwrap().to_string());
    }
    let keys = Arc::new(keys);
    let t0 = Instant::now();
    let accepted = seed_observations(&format!("{base}/memory"), &keys, seed, threads);
    wait_idle(&a, &base, &token);
    eprintln!("seeded {accepted} observations, applied in {:.1}s", t0.elapsed().as_secs_f64());

    let q_keys = keys.clone();
    let q_url = format!("{base}/query");
    let q_org = org.clone();
    let query = phase(
        threads,
        seconds,
        200,
        Arc::new(move |t, n, a| {
            let mut rng = Rng(0x9E3779B97F4A7C15 ^ ((t as u64) << 32 | n));
            let key = &q_keys[(t + n as usize) % q_keys.len()];
            post(a, &q_url, key, &json!({"query": words(&mut rng, 3), "scope": q_org})).0
        }),
    );
    let before = status(&a, &base, &token);
    let w_keys = keys.clone();
    let w_url = format!("{base}/memory");
    let write = phase(
        threads,
        write_seconds,
        202,
        Arc::new(move |t, n, a| {
            let mut rng = Rng(0xD1B54A32D192ED03 ^ ((t as u64) << 32 | n));
            let key = &w_keys[t % w_keys.len()];
            let body = json!({"text": words(&mut rng, 14), "topic": format!("bench:t{}", rng.below(30)), "confidence": 0.7,
                              "idempotency_key": format!("load-{t}-{n}")});
            post(a, &w_url, key, &body).0
        }),
    );
    let drained = wait_idle(&a, &base, &token);
    let after = status(&a, &base, &token);
    let applied = after["stats"]["last_applied_seq"].as_i64().unwrap_or(0)
        - before["stats"]["last_applied_seq"].as_i64().unwrap_or(0);
    let accepted = write["requests_ok"].as_u64().unwrap_or(0);
    let total = write["seconds"].as_f64().unwrap_or(1.0) + drained.as_secs_f64();
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "threads": threads, "seed": seed, "query": query, "write": write,
            "write_drain_seconds": (drained.as_secs_f64() * 100.0).round() / 100.0,
            "events_applied_during_write_phase": applied,
            "accepted_then_applied_per_second": (accepted as f64 / total).round(),
            "events_applied_per_second": (applied as f64 / total).round(),
            "memories_by_layer": after["stats"]["memories_by_layer"],
        }))
        .unwrap()
    );
}

/// Seed observations with `threads` parallel writers (unique idempotency keys), returning how many were accepted.
fn seed_observations(url: &str, keys: &[String], seed: usize, threads: usize) -> usize {
    let per = seed.div_ceil(threads);
    let handles: Vec<_> = (0..threads)
        .map(|t| {
            let (url, keys) = (url.to_string(), keys.to_vec());
            std::thread::spawn(move || {
                let a = agent();
                let mut rng = Rng(0xA5A5_5A5A_1234_5678 ^ t as u64);
                let mut ok = 0;
                for n in 0..per {
                    let key = &keys[(t * per + n) % keys.len()];
                    let body = json!({"text": words(&mut rng, 14), "topic": format!("bench:t{}", rng.below(30)), "confidence": 0.8,
                                      "idempotency_key": format!("seed-{t}-{n}")});
                    if post(&a, &url, key, &body).0 == 202 {
                        ok += 1;
                    }
                }
                ok
            })
        })
        .collect();
    handles.into_iter().map(|h| h.join().unwrap()).sum()
}
