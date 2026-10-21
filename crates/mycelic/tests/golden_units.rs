//! The deterministic building blocks against the golden fixtures recorded from the Python implementation
//! (`tests/golden/*.json`, written by `tests/golden/generate_from_python.py` at ace3efb).  Every comparison is exact:
//! floats to the last bit, error messages character for character.

use mycelic::auth::{Principal, hash_key, memory_visible, parse_agent_id};
use mycelic::config::Settings;
use mycelic::coordination::{ClaimEnvelope, LineageAnalyzer, RuleBasedSynthesizer};
use mycelic::hierarchy::{self, AgentPath, LAYERS};
use mycelic::models::{Agent, Memory, content_hash, derived_memory_id, random_ids, system_clock};
use mycelic::pyjson::{self, Style};
use mycelic::retrieval::{OrgIndex, SearchParams, bm25_scores, search_index};
use mycelic::service::{MycelicService, field_iso};
use mycelic::store::Store;
use mycelic::textutil;
use serde_json::{Map, Value, json};
use std::path::{Path, PathBuf};

fn golden(name: &str) -> Value {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/golden").join(name);
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    // Python's encoder writes infinities as a bare `Infinity` token, which is not JSON; the Rust side reports the
    // same value as the string "Infinity"
    let text = text.replace(": Infinity", ": \"Infinity\"");
    serde_json::from_str(&text).expect("fixture JSON")
}

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize().expect("repo root")
}

fn strings(v: &Value) -> Vec<String> {
    v.as_array().expect("array").iter().map(|x| x.as_str().expect("string").to_string()).collect()
}

#[test]
fn tokenizer_matches_python() {
    let g = golden("textutil.json");
    for case in g["cases"].as_array().unwrap() {
        let text = case["text"].as_str().unwrap();
        assert_eq!(textutil::tokenize(text), strings(&case["tokens"]), "tokens of {text:?}");
        assert_eq!(
            textutil::tokenize_with(text, false, true),
            strings(&case["tokens_nostem"]),
            "unstemmed tokens of {text:?}"
        );
        assert_eq!(
            textutil::tokenize_with(text, true, false),
            strings(&case["tokens_nonum"]),
            "tokens without numbers of {text:?}"
        );
        assert_eq!(textutil::fold(text), case["fold"].as_str().unwrap(), "fold of {text:?}");
        assert_eq!(textutil::normalize_ws(text), case["normalize_ws"].as_str().unwrap(), "normalize_ws of {text:?}");
    }
    for pair in g["stem"].as_array().unwrap() {
        assert_eq!(textutil::stem(pair[0].as_str().unwrap()), pair[1].as_str().unwrap(), "stem of {}", pair[0]);
    }
}

#[test]
fn bm25_scores_match_python_to_the_bit() {
    let g = golden("bm25.json");
    for case in g["cases"].as_array().unwrap() {
        let docs: Vec<Vec<String>> = case["docs"].as_array().unwrap().iter().map(strings).collect();
        let query = strings(&case["query"]);
        let got = bm25_scores(&docs, &query, case["k1"].as_f64().unwrap(), case["b"].as_f64().unwrap());
        let want: Vec<f64> = case["scores"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
        assert_eq!(got, want, "scores for query {query:?}");
    }
}

fn principal_visible(p: &Value) -> Box<dyn Fn(&Memory) -> bool> {
    let agent = p["agent_path"].as_str().map(String::from);
    let team = p["team_path"].as_str().map(String::from);
    Box::new(move |m| memory_visible(m, agent.as_deref(), team.as_deref()))
}

#[test]
fn retrieval_ranks_like_python() {
    let g = golden("retrieval.json");
    let rows: Vec<Memory> = g["memories"].as_array().unwrap().iter().map(|m| Memory::from_json(m).unwrap()).collect();
    let index = OrgIndex::new(rows);
    let principals: Map<String, Value> = g["principals"]
        .as_array()
        .unwrap()
        .iter()
        .map(|p| (p["name"].as_str().unwrap().to_string(), p.clone()))
        .collect();
    for case in g["cases"].as_array().unwrap() {
        let p = &principals[case["principal"].as_str().unwrap()];
        let params = &case["params"];
        let sp = SearchParams {
            scope: params["scope"].as_str(),
            min_layer: params["min_layer"].as_str().unwrap_or("agent"),
            k: params["k"].as_u64().unwrap_or(10) as usize,
            topic: params["topic"].as_str(),
            entity: params["entity"].as_str(),
        };
        let hits = search_index(&index, params["query"].as_str().unwrap(), &*principal_visible(p), &sp);
        let got: Vec<Value> = hits
            .iter()
            .map(|h| {
                let mut d = h.to_json().as_object().unwrap().clone();
                d.remove("memory");
                let mut out = Map::new();
                out.insert("memory_id".into(), json!(h.memory.memory_id));
                out.extend(d);
                Value::Object(out)
            })
            .collect();
        assert_eq!(Value::Array(got), case["hits"], "{} / {}", case["principal"], params);
    }
}

fn principal_from(p: &Value) -> Principal {
    match p["agent_path"].as_str() {
        None => Principal::admin(),
        Some(path) => Principal::for_agent(Agent {
            agent_id: path.rsplit('/').next().unwrap().to_string(),
            org_id: "acme".into(),
            display_name: "x".into(),
            path: path.into(),
            scopes: vec!["memory:read".into()],
            key_prefix: "mk_".into(),
            status: "active".into(),
            created_at: String::new(),
            last_seen_at: None,
            metadata: Map::new(),
        }),
    }
}

#[test]
fn visibility_rule_in_code_and_in_sql_matches_python() {
    let g = golden("visibility.json");
    let rows: Vec<Memory> = g["memories"].as_array().unwrap().iter().map(|m| Memory::from_json(m).unwrap()).collect();
    let tmp = tempfile::tempdir().unwrap();
    let store = Store::open(tmp.path().join("v.db").to_str().unwrap(), system_clock(), random_ids()).unwrap();
    store
        .transaction(|tx| {
            for m in &rows {
                tx.insert_memory(m)?;
            }
            Ok(())
        })
        .unwrap();
    for pj in g["principals"].as_array().unwrap() {
        let name = pj["name"].as_str().unwrap();
        let p = principal_from(pj);
        let readable: Vec<&str> = rows.iter().filter(|m| p.can_read(m)).map(|m| m.memory_id.as_str()).collect();
        assert_eq!(json!(readable), g["can_read"][name], "can_read for {name}");
        let owns: Vec<&str> = rows.iter().filter(|m| p.owns(m)).map(|m| m.memory_id.as_str()).collect();
        assert_eq!(json!(owns), g["owns"][name], "owns for {name}");
        for (scope, want) in g["can_query_scope"][name].as_object().unwrap() {
            assert_eq!(json!(p.can_query_scope(scope)), *want, "can_query_scope({scope}) for {name}");
        }
    }
    for case in g["visible_rows"].as_array().unwrap() {
        let p = principal_from(
            g["principals"].as_array().unwrap().iter().find(|x| x["name"] == case["principal"]).unwrap(),
        );
        let got = store
            .read(|db| {
                db.visible_rows(
                    "acme",
                    p.path.as_deref(),
                    p.team_path.as_deref(),
                    case["scope"].as_str(),
                    case["min_layer_index"].as_u64().unwrap() as usize,
                    &["active"],
                )
            })
            .unwrap();
        let ids: Vec<&str> = got.iter().map(|m| m.memory_id.as_str()).collect();
        assert_eq!(json!(ids), case["ids"], "visible_rows {case}");
    }
}

#[test]
fn derived_ids_hashes_and_keys_match_python() {
    let g = golden("ids.json");
    for c in g["derived_memory_id"].as_array().unwrap() {
        let id = derived_memory_id(
            c["operator"].as_str().unwrap(),
            c["scope"].as_str().unwrap(),
            c["key"].as_str().unwrap(),
            &strings(&c["parent_ids"]),
        );
        assert_eq!(id, c["id"].as_str().unwrap());
    }
    for c in g["content_hash"].as_array().unwrap() {
        let parts = strings(&c["parts"]);
        let refs: Vec<&str> = parts.iter().map(String::as_str).collect();
        assert_eq!(content_hash(&refs), c["hash"].as_str().unwrap());
    }
    for c in g["api_keys"].as_array().unwrap() {
        // the recorded key was issued by Python: Rust hashes and parses it the same way, so it keeps working
        let key = c["key"].as_str().unwrap();
        assert_eq!(hash_key(key), c["hash"].as_str().unwrap());
        assert_eq!(parse_agent_id(key), c["parsed"].as_str());
        let (rk, rh, rp) = mycelic::auth::generate_api_key(c["agent_id"].as_str().unwrap());
        assert_eq!(rk.len() as u64, c["key_len"].as_u64().unwrap(), "same key length");
        assert_eq!(rp.len() as u64, c["prefix_len"].as_u64().unwrap(), "same prefix length");
        assert_eq!(hash_key(&rk), rh);
        assert!(rk.starts_with(&rp));
    }
    for (k, want) in g["parse_agent_id"].as_object().unwrap() {
        assert_eq!(json!(parse_agent_id(k)), *want, "parse_agent_id({k:?})");
    }
    for (k, want) in g["hash_key"].as_object().unwrap() {
        assert_eq!(hash_key(k), want.as_str().unwrap());
    }
    for c in g["idempotency"].as_array().unwrap() {
        let (a, k) = (c["agent"].as_str().unwrap(), c["key"].as_str().unwrap());
        assert_eq!(format!("mem_{}", &content_hash(&[a, k])[..22]), c["memory_id"].as_str().unwrap());
        assert_eq!(format!("evt_{}", &content_hash(&[a, "event", k])[..22]), c["event_id"].as_str().unwrap());
        assert_eq!(
            format!("mem_{}", &content_hash(&[a, "event-memory", k])[..22]),
            c["event_memory_id"].as_str().unwrap()
        );
    }
    for (mid, want) in g["derived_event_id"].as_object().unwrap() {
        assert_eq!(format!("evt_d{}", &content_hash(&[mid])[..22]), want.as_str().unwrap());
    }
}

fn catch<T: serde::Serialize>(r: Result<T, hierarchy::HierarchyError>) -> Value {
    match r {
        Ok(v) => json!({"ok": v}),
        Err(e) => json!({"error": e.0}),
    }
}

#[test]
fn hierarchy_matches_python() {
    let g = golden("hierarchy.json");
    for row in g["paths"].as_array().unwrap() {
        let p = row["path"].as_str().unwrap();
        assert_eq!(catch(hierarchy::split_path(p)), row["split"], "split {p:?}");
        assert_eq!(catch(hierarchy::layer_of_path(p)), row["layer_of_path"], "layer_of_path {p:?}");
        assert_eq!(catch(hierarchy::parent_path(p)), row["parent_path"], "parent_path {p:?}");
        assert_eq!(catch(hierarchy::ancestors(p, true)), row["ancestors"], "ancestors {p:?}");
        assert_eq!(catch(hierarchy::ancestors(p, false)), row["ancestors_strict"], "ancestors_strict {p:?}");
        assert_eq!(catch(hierarchy::org_of(p)), row["org_of"], "org_of {p:?}");
        for layer in LAYERS {
            assert_eq!(
                catch(hierarchy::unit_at_layer(p, layer)),
                row["unit_at_layer"][layer],
                "unit_at_layer {p:?} {layer}"
            );
        }
        assert_eq!(catch(AgentPath::parse(p).map(|a| a.to_json())), row["agent_path"], "AgentPath.parse {p:?}");
    }
    for c in g["child_unit_of"].as_array().unwrap() {
        let (d, u) = (c["descendant"].as_str().unwrap(), c["unit"].as_str().unwrap());
        assert_eq!(catch(hierarchy::child_unit_of(d, u)), c["result"]);
        assert_eq!(json!(hierarchy::is_ancestor_or_self(u, d)), c["is_ancestor_or_self"]);
    }
    for c in g["from_parts"].as_array().unwrap() {
        let pp = &c["parts"];
        let r = AgentPath::from_parts(
            pp["enterprise"].as_str().unwrap(),
            pp["region"].as_str(),
            pp["subsidiary"].as_str(),
            pp["department"].as_str(),
            pp["team"].as_str(),
            pp["agent_id"].as_str().unwrap(),
        );
        assert_eq!(catch(r.map(|a| a.to_json())), c["result"], "from_parts {pp}");
    }
    for (name, want) in g["layer_index"].as_object().unwrap() {
        assert_eq!(catch(hierarchy::layer_index(name)), *want, "layer_index {name:?}");
    }
}

#[test]
fn timestamps_parse_like_python_fromisoformat() {
    let g = golden("iso.json");
    for c in g["cases"].as_array().unwrap() {
        let mut body = Map::new();
        body.insert("k".into(), c["input"].clone());
        let got = match field_iso(&body, "k") {
            Ok(v) => json!({"input": c["input"], "output": v}),
            Err(e) => json!({"input": c["input"], "error": e.to_string()}),
        };
        assert_eq!(got, *c, "{}", c["input"]);
    }
}

#[test]
fn configuration_matches_python() {
    let g = golden("config.json");
    let rules = repo_root().join("deploy/mycelic/rules.json").to_string_lossy().into_owned();
    for c in g["cases"].as_array().unwrap() {
        let env: Map<String, Value> = c["env"].as_object().unwrap().clone();
        let lookup = |name: &str| env.get(name).and_then(Value::as_str).map(|v| v.replace("{rules}", &rules));
        let got = match Settings::from_lookup(&lookup) {
            Ok(s) => {
                let text = pyjson::dumps(&s.redacted()).replace(&rules, "{rules}");
                json!({"env": c["env"], "settings": serde_json::from_str::<Value>(&text).unwrap()})
            }
            Err(e) => json!({"env": c["env"], "error": e.0.replace(&rules, "{rules}")}),
        };
        assert_eq!(got, *c, "env {}", c["env"]);
    }
}

#[test]
fn synthesis_and_fragility_match_python() {
    let g = golden("coordination.json");
    for c in g["cases"].as_array().unwrap() {
        let slots = strings(&c["required_slots"]);
        let claims: Vec<ClaimEnvelope> = c["claims"]
            .as_array()
            .unwrap()
            .iter()
            .map(|cl| ClaimEnvelope {
                claim_id: cl["claim_id"].as_str().unwrap().into(),
                producer_node_id: cl["producer_node_id"].as_str().unwrap().into(),
                slot: cl["content"]["slot"].as_str().map(String::from),
                value: cl["content"]["value"].as_str().unwrap().into(),
                confidence: cl["confidence"].as_f64().unwrap(),
                lineage_root_ids: strings(&cl["lineage_root_ids"]),
                failure_domains: strings(&cl["failure_domains"]),
            })
            .collect();
        let synthesis = RuleBasedSynthesizer { required_slots: &slots }.synthesize(&claims);
        assert_eq!(synthesis.to_json(), c["synthesis"], "synthesis {c}");
        assert_eq!(LineageAnalyzer::score(&claims, &slots, &synthesis).to_json(), c["fragility"], "fragility {c}");
        let coalitions: Vec<Value> =
            LineageAnalyzer::reconstruction_coalitions(&claims, &slots).iter().map(|x| x.to_json()).collect();
        assert_eq!(Value::Array(coalitions), c["coalitions"], "coalitions {c}");
    }
}

#[test]
fn json_text_floats_and_rounding_match_python() {
    let g = golden("pyjson.json");
    for f in g["floats"].as_array().unwrap() {
        let x = f["value"].as_f64().unwrap();
        assert_eq!(pyjson::float_repr(x), f["repr"].as_str().unwrap(), "repr of {x:e}");
        assert_eq!(pyjson::float_json(x), f["json"].as_str().unwrap());
    }
    for d in g["dumps"].as_array().unwrap() {
        let v = &d["value"];
        assert_eq!(pyjson::dumps(v), d["default"].as_str().unwrap());
        assert_eq!(pyjson::dumps_sorted(v), d["sorted"].as_str().unwrap());
        assert_eq!(pyjson::dumps_with(v, Style::INDENT1), d["indent1"].as_str().unwrap());
        assert_eq!(pyjson::dumps_with(v, Style::COMPACT), d["compact"].as_str().unwrap());
        assert_eq!(pyjson::dumps_with(v, Style::ASCII), d["ascii"].as_str().unwrap());
    }
    for r in g["round"].as_array().unwrap() {
        let x = r["x"].as_f64().unwrap();
        assert_eq!(pyjson::round(x, 4), r["r4"].as_f64().unwrap(), "round({x}, 4)");
        assert_eq!(pyjson::round(x, 6), r["r6"].as_f64().unwrap(), "round({x}, 6)");
        assert_eq!(pyjson::fmt_fixed(x, 3), r["f3"].as_str().unwrap());
        assert_eq!(pyjson::fmt_fixed(x, 2), r["f2"].as_str().unwrap());
    }
}

#[test]
fn rule_validation_matches_python() {
    let g = golden("rules_validation.json");
    for c in g["cases"].as_array().unwrap() {
        let got = match MycelicService::parse_rule(&c["body"]) {
            Ok(r) => json!({"body": c["body"], "rule": r.to_json()}),
            Err(e) => json!({"body": c["body"], "error": e.to_string()}),
        };
        assert_eq!(got, *c, "rule body {}", c["body"]);
    }
}
