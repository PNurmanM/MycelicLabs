//! Whole service runs against the Python implementation: the recorded step lists of
//! `tests/golden/scenario_*.json` replayed on the Rust service with the same pinned clock and id counter.  Every
//! output is compared exactly, including full database snapshots (memories, lineage edges, the event log, audit
//! rows, rules, agents) after the steps that took one: derived ids, confidences, texts, metadata, versions,
//! statuses and event ids all have to be the ones Python produced.

mod common;

use common::{ADMIN_TOKEN, Harness, assert_json_eq, golden};
use mycelic::service::{MycelicService, ServiceError};
use mycelic::store::{row_agent, row_edge, row_event, row_memory, row_rule};
use serde_json::{Map, Value, json};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

fn snapshot(svc: &MycelicService) -> Value {
    svc.store
        .read(|db| {
            let c = db.c;
            let rows = |sql: &str| -> anyhow::Result<Vec<Value>> {
                let mut stmt = c.prepare(sql)?;
                let cols: Vec<String> = stmt.column_names().iter().map(|s| s.to_string()).collect();
                let out = stmt
                    .query_map([], |r| {
                        let mut m = Map::new();
                        for (i, name) in cols.iter().enumerate() {
                            let v = match r.get::<_, rusqlite::types::Value>(i)? {
                                rusqlite::types::Value::Null => Value::Null,
                                rusqlite::types::Value::Integer(n) => json!(n),
                                rusqlite::types::Value::Real(f) => json!(f),
                                rusqlite::types::Value::Text(s) => json!(s),
                                rusqlite::types::Value::Blob(_) => Value::Null,
                            };
                            m.insert(name.clone(), v);
                        }
                        Ok(Value::Object(m))
                    })?
                    .collect::<rusqlite::Result<Vec<_>>>()?;
                Ok(out)
            };
            let mapped = |sql: &str, f: fn(&rusqlite::Row) -> rusqlite::Result<Value>| -> anyhow::Result<Vec<Value>> {
                let mut stmt = c.prepare(sql)?;
                Ok(stmt.query_map([], f)?.collect::<rusqlite::Result<Vec<_>>>()?)
            };
            let memories = mapped("SELECT * FROM memories ORDER BY rid", |r| Ok(row_memory(r)?.to_json()))?;
            let agg_keys: Vec<Value> =
                rows("SELECT agg_key FROM memories ORDER BY rid")?.into_iter().map(|r| r["agg_key"].clone()).collect();
            let edges =
                mapped("SELECT * FROM lineage_edges ORDER BY child_id, parent_id", |r| Ok(row_edge(r)?.to_json()))?;
            let events = mapped("SELECT * FROM events ORDER BY rid", |r| {
                let mut e = row_event(r)?.to_json();
                if let Some(p) = e["payload"].as_object_mut() {
                    p.remove("key_hash");
                    p.remove("key_prefix");
                }
                Ok(e)
            })?;
            let audit: Vec<Value> = rows("SELECT * FROM audit_log ORDER BY id")?
                .into_iter()
                .map(|mut a| {
                    let detail: Value = serde_json::from_str(a["detail"].as_str().unwrap_or("{}")).unwrap_or(json!({}));
                    a["detail"] = detail;
                    a
                })
                .collect();
            let rules = mapped("SELECT * FROM rules ORDER BY rule_id", |r| {
                let mut v = row_rule(r)?.to_json();
                v["updated_at"] = json!(r.get::<_, String>("updated_at")?);
                Ok(v)
            })?;
            let agents = mapped("SELECT * FROM agents ORDER BY path", |r| {
                let mut v = row_agent(r)?.to_json();
                let m = v.as_object_mut().unwrap();
                m.remove("key_prefix");
                m.remove("last_seen_at");
                Ok(v)
            })?;
            let orgs = rows("SELECT * FROM orgs ORDER BY org_id")?;
            let mut meta = Map::new();
            for r in rows("SELECT * FROM meta ORDER BY key")? {
                meta.insert(r["key"].as_str().unwrap().to_string(), r["value"].clone());
            }
            Ok(json!({"memories": memories, "agg_keys": agg_keys, "edges": edges, "events": events, "audit": audit,
                      "rules": rules, "agents": agents, "orgs": orgs, "meta": meta}))
        })
        .unwrap()
}

fn err(e: ServiceError) -> Value {
    let (kind, msg) = match e {
        ServiceError::Validation(m) => ("ValidationError", m),
        ServiceError::Forbidden(m) => ("Forbidden", m),
        ServiceError::NotFound(m) => ("NotFound", m),
        ServiceError::Hierarchy(m) => ("HierarchyError", m),
        ServiceError::Internal(e) => panic!("unexpected internal error: {e:?}"),
    };
    json!({"error": {"type": kind, "message": msg}})
}

fn resolve(svc: &MycelicService, r: &Value, aliases: &HashMap<String, String>) -> String {
    if let Some(s) = r.as_str() {
        return s.to_string();
    }
    if let Some(a) = r.get("alias") {
        return aliases[a.as_str().unwrap()].clone();
    }
    let sel = r["select"].as_object().unwrap();
    let all = svc.store.read(|db| {
        let mut stmt = db.c.prepare("SELECT * FROM memories ORDER BY rid")?;
        Ok(stmt.query_map([], |r| Ok(row_memory(r)?.to_json()))?.collect::<rusqlite::Result<Vec<_>>>()?)
    });
    all.unwrap()
        .into_iter()
        .find(|m| sel.iter().all(|(k, v)| m.get(k) == Some(v)))
        .and_then(|m| m["memory_id"].as_str().map(String::from))
        .unwrap_or_else(|| panic!("no memory matches {sel:?}"))
}

fn without_latency(mut v: Value) -> Value {
    if let Some(m) = v.as_object_mut() {
        m.remove("latency_ms");
    }
    v
}

async fn run(name: &str) {
    let g = golden(&format!("scenario_{name}.json"));
    assert_eq!(g["fixed_now"], common::FIXED_NOW);
    assert_eq!(g["admin_token"], ADMIN_TOKEN);
    let mut h = Harness::start(true, |_| {}).await;
    let mut aliases: HashMap<String, String> = HashMap::new();
    let steps = g["steps"].as_array().unwrap();
    let outputs = g["outputs"].as_array().unwrap();
    for (i, (step, want)) in steps.iter().zip(outputs).enumerate() {
        let svc = h.service.clone();
        let op = step["op"].as_str().unwrap();
        let p = |who: &Value| h.principal(who.as_str().unwrap());
        let got: Value = match op {
            "rule" => svc.upsert_rule(&step["body"], None).map(|r| r.to_json()).unwrap_or_else(err),
            "delete_rule" => {
                svc.delete_rule(step["rule_id"].as_str().unwrap(), None).map(|b| json!(b)).unwrap_or_else(err)
            }
            "register" => match svc.register_agent(&step["body"], None) {
                Ok((agent, key)) => {
                    h.keys.insert(agent.agent_id.clone(), key);
                    let mut v = agent.to_json();
                    v.as_object_mut().unwrap().remove("key_prefix");
                    v
                }
                Err(e) => err(e),
            },
            "revoke" => {
                svc.revoke_agent(step["agent_id"].as_str().unwrap(), None).map(|b| json!(b)).unwrap_or_else(err)
            }
            "observe" => match svc.ingest_memory(&p(&step["agent"]), &step["body"], None) {
                Ok((m, created)) => {
                    if let Some(a) = step["as"].as_str() {
                        aliases.insert(a.to_string(), m.memory_id.clone());
                    }
                    json!({"memory_id": m.memory_id, "event_id": m.event_id, "created": created})
                }
                Err(e) => err(e),
            },
            "events" => {
                svc.ingest_events(&p(&step["agent"]), Some(&step["items"]), None).map(Value::Array).unwrap_or_else(err)
            }
            "retract" => {
                let id = resolve(&svc, &step["memory"], &aliases);
                let reason = step["reason"].as_str().unwrap_or("retracted by producer");
                svc.retract(&p(&step["agent"]), &id, reason, None).map(|e| e.to_json()).unwrap_or_else(err)
            }
            "settle" => {
                h.settle().await;
                Value::Null
            }
            "query" => {
                let pr = p(&step["viewer"]);
                match svc.query(&pr, &step["body"]) {
                    Ok(mut res) => {
                        let results: Vec<Value> = res["results"]
                            .as_array()
                            .unwrap()
                            .iter()
                            .map(|hit| {
                                let mut hit = hit.clone();
                                hit["memory"] = svc.public_view(&hit["memory"], &pr);
                                hit
                            })
                            .collect();
                        res["results"] = Value::Array(results);
                        without_latency(res)
                    }
                    Err(e) => err(e),
                }
            }
            "lineage" => {
                let id = resolve(&svc, &step["memory"], &aliases);
                svc.lineage(&p(&step["viewer"]), &id).unwrap_or_else(err)
            }
            "get_memory" => {
                let id = resolve(&svc, &step["memory"], &aliases);
                let pr = p(&step["viewer"]);
                svc.get_memory(&pr, &id).map(|m| svc.public_view(&m.to_json(), &pr)).unwrap_or_else(err)
            }
            "list_memories" => {
                let pr = p(&step["viewer"]);
                let layers: Option<Vec<String>> =
                    step["layers"].as_array().map(|a| a.iter().map(|x| x.as_str().unwrap().to_string()).collect());
                svc.list_memories(
                    &pr,
                    step["scope"].as_str(),
                    layers.as_deref(),
                    step["limit"].as_i64().unwrap_or(50),
                    step["status"].as_str().unwrap_or("active"),
                )
                .map(|rows| Value::Array(rows.iter().map(|m| svc.public_view(&m.to_json(), &pr)).collect()))
                .unwrap_or_else(err)
            }
            "snapshot" => snapshot(&svc),
            "stats" => {
                let mut st = svc.store.stats().unwrap();
                st.as_object_mut().unwrap().remove("revision");
                st
            }
            "rebuild" => {
                svc.stop().await;
                drop(svc);
                let fresh = Harness::service(&h.settings, &h.transport, &h.ids, true, Some(":memory:"));
                let old = std::mem::replace(&mut h.service, fresh);
                drop(old);
                h.service.start(true).await.unwrap();
                assert!(h.service.wait_idle(Duration::from_secs(60)).await, "rebuild did not settle");
                snapshot(&h.service)
            }
            other => panic!("unknown op {other}"),
        };
        assert_json_eq(&got, want, &format!("scenario {name}, step {i} ({op}) {step}"));
    }
    let svc: Arc<MycelicService> = h.service.clone();
    svc.close().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn datapath() {
    run("datapath").await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn supersede_reactivate() {
    run("supersede_reactivate").await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn retraction() {
    run("retraction").await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn events() {
    run("events").await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn strategic() {
    run("strategic").await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn follows_evidence() {
    run("follows_evidence").await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn corroborate_toggle() {
    run("corroborate_toggle").await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn withdrawal_cascade() {
    run("withdrawal_cascade").await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn stale_dependents() {
    run("stale_dependents").await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn unanimous_and_cycles() {
    run("unanimous_and_cycles").await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn fragility_bounded() {
    run("fragility_bounded").await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn promotion() {
    run("promotion").await;
}
