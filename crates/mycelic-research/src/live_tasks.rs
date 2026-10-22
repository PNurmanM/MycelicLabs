//! A deterministic operator test-set built from the real corpus, for DIRECT measurement on real models
//! (`live_tasks.py`).
//!
//! The simulator parameterises five operator abilities.  This module turns each of them into a concrete, gradeable
//! task over text produced by the same generator the benchmark uses, so that the mapping "model class -> capability
//! vector" has at least a few measured anchors instead of being asserted end to end.
//!
//! * T1 extraction        text -> (predicate, entity)
//! * T2 causal check      claim set -> is this one causal chain?
//! * T3 temporal check    claim set -> is it in causal order / has it been retracted?
//! * T4 entity linking    two mention strings -> same entity or not?
//! * T5 independence      claim set with echoes -> how many independent sources?
//!
//! What this measures is the *operator*, not the architecture: a calibration of the simulator's inputs; no
//! end-to-end enterprise result is claimed to be directly measured.
//!
//! The task file goes to the artifacts directory and the answer key to a separate `keys` directory, both written
//! as `json.dump(obj, fh, indent=1)` wrote them.

use crate::corpus::{CAUSAL_CHAINS, SCHEMA, build_corpus};
use crate::np::Generator;
use crate::org::build_org_default;
use crate::py::json::{self as pyjson, Style};
use crate::runner::{art, repo_root};
use anyhow::{Context, Result};
use indexmap::IndexMap;
use serde_json::{Map, Value, json};
use std::collections::HashSet;
use std::path::{Path, PathBuf};

pub const N_T1: usize = 48;
pub const N_T2: usize = 48;
pub const N_T3: usize = 36;
pub const N_T4: usize = 36;
pub const N_T5: usize = 30;

/// `research/mycelic/keys` (`KEYDIR`), where the answer keys live: OUTSIDE the directory the measured model is
/// pointed at.  `MYCELIC_KEYS` overrides it, so a rerun can write its keys elsewhere.
pub fn key_dir() -> PathBuf {
    match std::env::var_os("MYCELIC_KEYS") {
        Some(p) => PathBuf::from(p),
        None => repo_root().join("research").join("mycelic").join("keys"),
    }
}

/// Write `json.dump(value, fh, indent=indent)`, creating the directory first.
pub(crate) fn write_json(dir: &Path, name: &str, value: &Value, indent: usize) -> Result<PathBuf> {
    std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    let path = dir.join(name);
    std::fs::write(&path, pyjson::dumps_with(value, Style::indent(indent)))
        .with_context(|| format!("writing {}", path.display()))?;
    Ok(path)
}

/// `json.load(open(path))`
pub(crate) fn read_json(path: &Path) -> Result<Value> {
    let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))
}

/// `CAUSAL_CHAINS` as the JSON lists `json.dump` makes of the tuples.
pub(crate) fn causal_chains_json() -> Value {
    Value::Array(CAUSAL_CHAINS.iter().map(|ch| json!(ch)).collect())
}

fn surface(pred: &str) -> &'static [String] {
    &SCHEMA.surface[SCHEMA.pred_id[pred]]
}

/// A uniformly drawn element of a non-empty slice (`seq[int(rng.integers(0, len(seq)))]`).
fn pick<'a, T>(rng: &mut Generator, seq: &'a [T]) -> &'a T {
    &seq[rng.integers(0, seq.len() as i64) as usize]
}

/// `build(seed, scale)`: the task file into `art_dir/live_tasks.json`, the key into `key_dir/live_key.json`;
/// returns the task spec.
pub fn build(seed: u64, scale: usize, art_dir: &Path, key_dir: &Path) -> Result<Value> {
    let org = build_org_default(scale, seed);
    let cp = build_corpus(org, seed, None);
    let mut rng = Generator::new(seed);
    let recs = &cp.recs;
    let predicates = &SCHEMA.predicates;
    let entities = &cp.entities;
    let n_chains = CAUSAL_CHAINS.len() as i64;

    let mut tasks: IndexMap<&str, Vec<Value>> =
        ["T1", "T2", "T3", "T4", "T5"].iter().map(|&g| (g, Vec::new())).collect();
    let mut key = Map::new();

    // ---- T1 extraction ----
    // Only records with a single entity mention: in a two-mention note nothing grammatically attaches the predicate
    // to either token, so half the items would be decided by a positional convention rather than by reading.
    // (Flagged by the first blind measurement run.)
    let single: Vec<usize> = (0..recs.len()).filter(|&i| recs.aux[i] < 0).collect();
    for pos in rng.choice_without_replace(single.len(), N_T1) {
        let i = single[pos as usize];
        let id = format!("T1-{i}");
        tasks["T1"].push(json!({"id": id, "text": cp.text(i)}));
        key.insert(
            id,
            json!({"predicate": predicates[recs.pred[i] as usize], "entity": entities[recs.anchor[i] as usize]}),
        );
    }

    // ---- T2 causal chain membership ----
    // The yes/no label is drawn at random, NOT alternated by index: an alternating label lets a model score well by
    // noticing the pattern instead of by reasoning.  (Found by the first measurement run.)
    let t2_lab: Vec<bool> = rng.random_n(N_T2).into_iter().map(|x| x < 0.5).collect();
    for (k, &label) in t2_lab.iter().enumerate() {
        let mut real = label;
        let ch = CAUSAL_CHAINS[rng.integers(0, n_chains) as usize];
        let len = rng.integers(3, 5.min(ch.len() as i64) + 1) as usize;
        let start = rng.integers(0, (ch.len() - len + 1) as i64) as usize;
        let ent = pick(&mut rng, entities).clone();
        let mut preds: Vec<&str> = if real {
            ch[start..start + len].to_vec()
        } else {
            let mut preds = Vec::with_capacity(len);
            for _ in 0..len {
                let other = CAUSAL_CHAINS[rng.integers(0, n_chains) as usize];
                preds.push(*pick(&mut rng, other));
            }
            preds
        };
        if !real {
            // a "no" item must genuinely span more than one chain
            let chains_used: HashSet<usize> = preds
                .iter()
                .flat_map(|p| CAUSAL_CHAINS.iter().enumerate().filter(|(_, ch2)| ch2.contains(p)).map(|(ci, _)| ci))
                .collect();
            let distinct: HashSet<&str> = preds.iter().copied().collect();
            if chains_used.len() < 2 || distinct.len() < 2 {
                preds = ch[start..start + len].to_vec();
                real = true;
            }
        }
        let mut claims = Vec::with_capacity(preds.len());
        for (j, p) in preds.iter().enumerate() {
            let surf = pick(&mut rng, surface(p));
            claims.push(format!("day {:3}: {surf} ({ent})", 10 + 12 * j));
        }
        let id = format!("T2-{k}");
        tasks["T2"].push(json!({"id": id, "claims": claims, "entity": ent}));
        key.insert(id, json!({"answer": if real { "yes" } else { "no" }}));
    }

    // ---- T3 temporal order / retraction ----
    let t3_mode = rng.integers_n(0, 3, N_T3);
    for (k, &mode) in t3_mode.iter().enumerate() {
        // 0 ordered, 1 scrambled, 2 retracted
        let ch = CAUSAL_CHAINS[rng.integers(0, n_chains) as usize];
        let len = 4.min(ch.len());
        let start = rng.integers(0, (ch.len() - len + 1) as i64) as usize;
        let ent = pick(&mut rng, entities).clone();
        let mut times: Vec<usize> = (0..len).map(|j| 10 + 12 * j).collect();
        if mode == 1 {
            times.reverse();
        }
        let mut claims = Vec::new();
        for j in 0..len {
            let surf = pick(&mut rng, surface(ch[start + j]));
            claims.push(format!("day {:3}: {surf} ({ent})", times[j]));
        }
        if mode == 2 {
            let last = times[len - 1];
            for j in 0..len {
                let surf = &surface(ch[start + j])[0];
                claims.push(format!("day {:3}: NOT {surf} - earlier report withdrawn ({ent})", last + 40 + j));
            }
        }
        let gold = ["current", "out_of_order", "retracted"][mode as usize];
        let id = format!("T3-{k}");
        tasks["T3"].push(json!({"id": id, "claims": claims, "entity": ent}));
        key.insert(id, json!({"answer": gold}));
    }

    // ---- T4 entity linking ----
    // entities grouped by their first five characters, in first-seen order (a dict of lists in the Python)
    let mut stems: IndexMap<&str, Vec<&str>> = IndexMap::new();
    for e in entities {
        stems.entry(crate::corpus::prefix5(e)).or_default().push(e);
    }
    let sibs: Vec<&Vec<&str>> = stems.values().filter(|v| v.len() >= 2).collect();
    let t4_lab: Vec<bool> = rng.random_n(N_T4).into_iter().map(|x| x < 0.5).collect();
    let mut used_pairs: HashSet<(String, String)> = HashSet::new();
    // same referent, different surface wrapping
    let wrap = |rng: &mut Generator, e: &str| -> String {
        if rng.random() < 0.5 { e.to_uppercase() } else { format!("the {e} platform") }
    };
    for (k, &same) in t4_lab.iter().enumerate() {
        let (mut a, mut b) = if same {
            let e = pick(&mut rng, entities).clone();
            let b = wrap(&mut rng, &e);
            (e, b)
        } else if !sibs.is_empty() {
            let grp = pick(&mut rng, &sibs);
            (grp[0].to_string(), grp[1].to_string())
        } else {
            (entities[0].clone(), entities[1].clone())
        };
        let mut tries = 0;
        while used_pairs.contains(&(a.clone(), b.clone())) && tries < 40 {
            tries += 1;
            if same {
                let e = pick(&mut rng, entities).clone();
                b = wrap(&mut rng, &e);
                a = e;
            } else if !sibs.is_empty() {
                let grp = pick(&mut rng, &sibs);
                let i2 = rng.integers(0, grp.len() as i64) as usize;
                let j2 = (i2 + 1 + rng.integers(0, grp.len() as i64 - 1) as usize) % grp.len();
                a = grp[i2].to_string();
                b = grp[j2].to_string();
            }
        }
        used_pairs.insert((a.clone(), b.clone()));
        let id = format!("T4-{k}");
        tasks["T4"].push(json!({"id": id, "a": a, "b": b}));
        key.insert(id, json!({"answer": if same { "same" } else { "different" }}));
    }

    // ---- T5 independent support ----
    for k in 0..N_T5 {
        let n_indep = rng.integers(1, 5);
        let ent = pick(&mut rng, entities).clone();
        let surfs = surface(predicates[rng.integers(0, 34) as usize]);
        let mut lines = Vec::new();
        for j in 0..n_indep as usize {
            let base = &surfs[j % surfs.len()];
            lines.push(format!("[team-{j}] day {}: {base} affecting {ent}; logged by owner {j}", 20 + j));
            let n_echo = rng.integers(0, 4);
            for echo in 0..n_echo {
                lines.push(format!(
                    "[team-{j}-relay-{echo}] day {}: {base} affecting {ent}; forwarded from owner {j}",
                    21 + j
                ));
            }
        }
        let reports: Vec<&String> = rng.permutation(lines.len()).iter().map(|&i| &lines[i as usize]).collect();
        let id = format!("T5-{k}");
        tasks["T5"].push(json!({"id": id, "reports": reports, "entity": ent}));
        key.insert(id, json!({"answer": n_indep}));
    }

    // shuffle item order within each group so no positional regularity remains
    for items in tasks.values_mut() {
        let order = rng.permutation(items.len());
        *items = order.iter().map(|&i| items[i as usize].clone()).collect();
    }

    let tasks_json: Map<String, Value> = tasks.into_iter().map(|(g, v)| (g.to_string(), Value::Array(v))).collect();
    let spec = json!({
        "seed": seed, "scale": scale,
        // the FULL predicate vocabulary, including the routine/background predicates: the first version listed only
        // the 34 operational ones, so 30 of the 48 extraction items had an out-of-vocabulary gold label
        "predicate_vocabulary": predicates,
        "operational_predicates": &predicates[..34],
        "causal_chains": causal_chains_json(),
        "tasks": tasks_json,
    });
    write_json(art_dir, "live_tasks.json", &spec, 1)?;
    // The answer key lives OUTSIDE the directory the measured model is pointed at.  The first version inlined gold
    // labels next to every item; the second put the key in the same folder, which a measurement run correctly
    // called out as a leakage hazard even though it did not open it.
    write_json(key_dir, "live_key.json", &Value::Object(key), 1)?;
    Ok(spec)
}

/// `python -m research.mycelic.live_tasks`: build with the defaults (seed 900, 2,000 users) and show one example
/// per group.
pub fn main(_args: &[String]) -> Result<()> {
    let art_dir = art();
    let spec = build(900, 2000, &art_dir, &key_dir())?;
    println!("wrote {}", art_dir.join("live_tasks.json").display());
    for (group, items) in spec["tasks"].as_object().into_iter().flatten() {
        let items = items.as_array().map(Vec::as_slice).unwrap_or_default();
        let example = items.first().map(|v| pyjson::dumps_with(v, Style::ASCII)).unwrap_or_default();
        println!("  {group} {} items; example: {}", items.len(), py_prefix(&example, 180));
    }
    Ok(())
}

/// `text[:n]`: the first n code points.
pub(crate) fn py_prefix(text: &str, n: usize) -> &str {
    match text.char_indices().nth(n) {
        Some((i, _)) => &text[..i],
        None => text,
    }
}
