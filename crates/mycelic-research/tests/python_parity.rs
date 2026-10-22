//! End-to-end parity with the Python simulator (last Python commit f7e954b): the generated worlds (org, corpus,
//! gold, near-miss map, user layer) and every metric of every architecture on small worlds, recorded by
//! `tests/golden/gen/gen_worlds.py` into `worlds.json` and `runs.json`.

use mycelic_research::corpus::CorpusCfg;
use mycelic_research::models::{self, HardShape, tier_from_q};
use mycelic_research::org::{TEAM, default_fanin};
use mycelic_research::py::pow;
use mycelic_research::runner::{self, Cal, World, WorldOpts};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::HashMap;

/// The fixtures were recorded on macOS arm64.  Bit-exactness holds against Python on the same platform: the
/// results route through the platform libm (`exp`, `log`, `pow`), whose last-bit rounding differs between
/// Apple's libm and glibc - the Python itself gives ULP-different confidences on the two (see
/// research/mycelic/README.md).  Elsewhere integers and rankings are still compared exactly and floats to 1e-9.
fn exact_platform() -> bool {
    cfg!(all(target_os = "macos", target_arch = "aarch64"))
}

fn close(a: f64, b: f64) -> bool {
    a == b || (!exact_platform() && (a - b).abs() <= 1e-9 * a.abs().max(b.abs()).max(1e-12))
}

fn load(name: &str) -> Value {
    let path = format!("{}/tests/golden/{name}", env!("CARGO_MANIFEST_DIR"));
    serde_json::from_str(&std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"))).unwrap()
}

fn sha_ints<I: IntoIterator<Item = i64>>(v: I) -> String {
    let text: Vec<String> = v.into_iter().map(|x| x.to_string()).collect();
    hex::encode(Sha256::digest(text.join(",").as_bytes()))
}

fn sha_floats<I: IntoIterator<Item = f64>>(v: I) -> String {
    let text: Vec<String> = v.into_iter().map(|x| x.to_bits().to_string()).collect();
    hex::encode(Sha256::digest(text.join(",").as_bytes()))
}

fn sha_text(s: &str) -> String {
    hex::encode(Sha256::digest(s.as_bytes()))
}

fn opts_of(o: &Value) -> WorldOpts {
    let mut w = WorldOpts::default();
    if o.is_null() {
        return w;
    }
    if let Some(ts) = o.get("team_size") {
        let mut f = default_fanin();
        f[TEAM].0 = ts.as_f64().unwrap();
        w.fanin = Some(f);
    }
    if let Some(n) = o.get("n_regions") {
        w.n_regions = Some(n.as_u64().unwrap() as usize);
    }
    if let Some(c) = o.get("corpus") {
        let mut cfg = CorpusCfg::default();
        for (k, v) in c.as_object().unwrap() {
            cfg.set(k, v);
        }
        w.corpus = Some(cfg);
    }
    w
}

fn key_of(scale: u64, seed: u64, opts: &Value) -> String {
    format!("{scale}/{seed}/{opts}")
}

#[test]
fn worlds_and_runs_match_python() {
    let worlds = load("worlds.json");
    let mut built: HashMap<String, World> = HashMap::new();
    for wd in worlds.as_array().unwrap() {
        let (scale, seed) = (wd["scale"].as_u64().unwrap(), wd["seed"].as_u64().unwrap());
        let w = runner::build_world(scale as usize, seed, &opts_of(&wd["opts"]));
        let ctx = format!("world {scale}/{seed} {}", wd["opts"]);
        let org = &w.corpus.org;
        let o = &wd["org"];
        assert_eq!(org.nodes.len() as u64, o["n_nodes"].as_u64().unwrap(), "{ctx}");
        for (l, ids) in org.levels.iter().enumerate() {
            assert_eq!(sha_ints(ids.iter().map(|&x| x as i64)), o["level_ids"][l].as_str().unwrap(), "{ctx} level {l}");
        }
        assert_eq!(
            sha_ints(org.nodes.iter().map(|n| n.parent.map_or(-1, |p| p as i64))),
            o["parents"].as_str().unwrap(),
            "{ctx} parents"
        );
        assert_eq!(sha_ints(org.nodes.iter().map(|n| n.function)), o["functions"].as_str().unwrap(), "{ctx}");
        assert_eq!(sha_ints(org.nodes.iter().map(|n| n.n_users as i64)), o["n_users"].as_str().unwrap(), "{ctx}");
        let names: Vec<&str> = org.nodes.iter().map(|n| n.name.as_str()).collect();
        assert_eq!(sha_text(&names.join("\n")), o["names"].as_str().unwrap(), "{ctx} names");
        assert_eq!(org.stats(), o["stats"], "{ctx} org stats");

        let c = &w.corpus;
        let cd = &wd["corpus"];
        let r = &c.recs;
        assert_eq!(r.len() as u64, cd["n_recs"].as_u64().unwrap(), "{ctx}");
        let f = &cd["fields"];
        let i = |v: &[i64]| sha_ints(v.iter().copied());
        assert_eq!(i(&r.uid.iter().map(|&x| x as i64).collect::<Vec<_>>()), f["uid"].as_str().unwrap(), "{ctx} uid");
        assert_eq!(i(&r.t.iter().map(|&x| x as i64).collect::<Vec<_>>()), f["t"].as_str().unwrap(), "{ctx} t");
        assert_eq!(i(&r.pred.iter().map(|&x| x as i64).collect::<Vec<_>>()), f["pred"].as_str().unwrap(), "{ctx} pred");
        assert_eq!(
            i(&r.anchor.iter().map(|&x| x as i64).collect::<Vec<_>>()),
            f["anchor"].as_str().unwrap(),
            "{ctx} anchor"
        );
        assert_eq!(i(&r.aux.iter().map(|&x| x as i64).collect::<Vec<_>>()), f["aux"].as_str().unwrap(), "{ctx} aux");
        assert_eq!(
            i(&r.event.iter().map(|&x| x as i64).collect::<Vec<_>>()),
            f["event"].as_str().unwrap(),
            "{ctx} event"
        );
        assert_eq!(i(&r.kind.iter().map(|&x| x as i64).collect::<Vec<_>>()), f["kind"].as_str().unwrap(), "{ctx} kind");
        assert_eq!(
            i(&r.veracity.iter().map(|&x| x as i64).collect::<Vec<_>>()),
            f["veracity"].as_str().unwrap(),
            "{ctx}"
        );
        assert_eq!(
            i(&r.polarity.iter().map(|&x| x as i64).collect::<Vec<_>>()),
            f["polarity"].as_str().unwrap(),
            "{ctx}"
        );
        assert_eq!(
            i(&r.group.iter().map(|&x| x as i64).collect::<Vec<_>>()),
            f["group"].as_str().unwrap(),
            "{ctx} group"
        );
        assert_eq!(
            i(&r.facet.iter().map(|&x| x as i64).collect::<Vec<_>>()),
            f["facet"].as_str().unwrap(),
            "{ctx} facet"
        );
        assert_eq!(
            i(&r.superseded.iter().map(|&x| x as i64).collect::<Vec<_>>()),
            f["superseded"].as_str().unwrap(),
            "{ctx}"
        );
        assert_eq!(sha_floats(r.salience.iter().map(|&x| x as f64)), f["salience"].as_str().unwrap(), "{ctx} salience");
        assert_eq!(i(&c.user_slices), cd["user_slices"].as_str().unwrap(), "{ctx} slices");
        assert_eq!(sha_text(&c.entities.join("\n")), cd["entities"].as_str().unwrap(), "{ctx} entities");
        for (p, want) in c.patterns.iter().zip(cd["patterns"].as_array().unwrap()) {
            let mut flat: Vec<i64> = p.facet_records.iter().flatten().map(|&x| x as i64).collect();
            flat.extend(p.facet_records.iter().map(|fr| -(fr.len() as i64)));
            let got = json!({
                "pid": p.pid, "chain": p.chain, "start_pos": p.start_pos, "preds": p.preds, "anchor": p.anchor,
                "facet_records": sha_ints(flat),
                "facet_users": sha_ints(p.facet_users.iter().flatten().map(|&u| u as i64)),
                "facet_times": p.facet_times, "real": p.real, "decoy_type": p.decoy_type, "rare": p.rare,
                "n_regions": p.n_regions, "n_sites": p.n_sites, "contradicted": p.contradicted, "family": p.family,
            });
            assert_eq!(&got, want, "{ctx} pattern {}", p.pid);
        }
        assert_eq!(c.patterns.len(), cd["patterns"].as_array().unwrap().len(), "{ctx}");
        assert_eq!(json!(c.families), cd["families"], "{ctx} families");
        assert_eq!(
            sha_ints(c.contradiction_pairs.iter().flat_map(|&(a, b)| [a as i64, b as i64])),
            cd["contradiction_pairs"].as_str().unwrap(),
            "{ctx}"
        );
        assert_eq!(sha_ints(c.notable_records.iter().map(|&x| x as i64)), cd["notable"].as_str().unwrap(), "{ctx}");
        assert_eq!(c.stats(), cd["stats"], "{ctx} corpus stats");
        for (rid, text) in cd["texts"].as_object().unwrap() {
            assert_eq!(c.text(rid.parse().unwrap()), text.as_str().unwrap(), "{ctx} text {rid}");
        }
        let g = &wd["gold"];
        assert_eq!(json!(w.gold.discoverable), g["discoverable"], "{ctx}");
        assert_eq!(json!(w.gold.rare), g["rare"], "{ctx}");
        assert_eq!(json!(w.gold.contradicted), g["contradicted"], "{ctx}");
        assert_eq!(json!(w.gold.families), g["families"], "{ctx}");
        assert_eq!(json!(w.gold.decoys), g["decoys"], "{ctx}");
        assert_eq!(json!(w.gold.top_supported), g["top_supported"], "{ctx}");
        for pair in g["independent_sources"].as_array().unwrap() {
            let pid = pair[0].as_u64().unwrap() as usize;
            assert_eq!(w.gold.independent_sources[&pid] as u64, pair[1].as_u64().unwrap(), "{ctx}");
        }
        assert_eq!(sha_ints(w.near_miss.iter().map(|&x| x as i64)), wd["near_miss"].as_str().unwrap(), "{ctx}");
        assert_eq!(sha_ints(w.stem.iter().map(|&x| x as i64)), wd["stem"].as_str().unwrap(), "{ctx}");
        let ul = w.user_layer(&models::anchor("small-7b", HardShape::Convex), seed);
        let u = &wd["user_layer"];
        let ex = &ul.ex;
        assert_eq!(ex.len() as u64, u["n"].as_u64().unwrap(), "{ctx} ul n");
        assert_eq!(sha_ints(ex.rid.iter().copied()), u["rid"].as_str().unwrap(), "{ctx} ul rid");
        assert_eq!(sha_ints(ex.uid.iter().copied()), u["uid"].as_str().unwrap(), "{ctx} ul uid");
        assert_eq!(sha_ints(ex.pred.iter().map(|&x| x as i64)), u["pred"].as_str().unwrap(), "{ctx} ul pred");
        assert_eq!(sha_ints(ex.anchor.iter().map(|&x| x as i64)), u["anchor"].as_str().unwrap(), "{ctx} ul anchor");
        assert_eq!(sha_ints(ex.t.iter().map(|&x| x as i64)), u["t"].as_str().unwrap(), "{ctx} ul t");
        assert_eq!(sha_ints(ex.sig.iter().copied()), u["sig"].as_str().unwrap(), "{ctx} ul sig");
        assert_eq!(sha_ints(ex.spurious.iter().map(|&x| x as i64)), u["spurious"].as_str().unwrap(), "{ctx}");
        if exact_platform() {
            assert_eq!(sha_floats(ul.imp.iter().copied()), u["imp"].as_str().unwrap(), "{ctx} ul imp");
            assert_eq!(sha_floats(ul.nov.iter().copied()), u["nov"].as_str().unwrap(), "{ctx} ul nov");
        }
        assert_eq!(ul.sketch_tokens as u64, u["sketch_tokens"].as_u64().unwrap(), "{ctx}");
        built.insert(key_of(scale, seed, &wd["opts"]), w);
    }

    let runs = load("runs.json");
    let base_cal = Cal::load();
    let mut failures: Vec<String> = Vec::new();
    let mut n_metrics = 0usize;
    for run in runs.as_array().unwrap() {
        let arch = run["arch"].as_str().unwrap();
        let (scale, seed) = (run["scale"].as_u64().unwrap(), run["seed"].as_u64().unwrap());
        // which world: find one with this scale and seed whose user layer/opts match the run's metrics
        let opts_key = built.keys().filter(|k| k.starts_with(&format!("{scale}/{seed}/"))).cloned().collect::<Vec<_>>();
        let mut cal = base_cal.clone();
        if run.get("ranker").and_then(Value::as_str) == Some("off") {
            cal.force_ranker = Some(false);
        }
        let shape = HardShape::parse(run.get("shape").and_then(Value::as_str).unwrap_or("convex"));
        let alloc = if let Some(q) = run.get("q").and_then(Value::as_f64) {
            let t = tier_from_q(
                q,
                Some(&format!("q{q:.2}")),
                Some(3.0 * pow(40.0, q)),
                Some((8000.0 * pow(128.0, q)) as i64),
                HardShape::Convex,
            );
            vec![t; 6]
        } else {
            models::allocation(run.get("alloc").and_then(Value::as_str).unwrap_or("back-loaded"), shape)
        };
        let cfg_over = run["cfg_over"].as_object().cloned();
        let mut matched = false;
        let mut last_err = String::new();
        for k in &opts_key {
            let w = &built[k];
            if shape != HardShape::Convex {
                w.clear_cache();
            }
            let r = runner::run_arch(&cal, arch, w, &alloc, seed, cfg_over.as_ref());
            if shape != HardShape::Convex {
                w.clear_cache();
            }
            let m = runner::evaluate(w, &r.res);
            let mut errs = Vec::new();
            for (mk, mv) in run["metrics"].as_object().unwrap() {
                let got = m.get(mk.as_str()).copied();
                let same = matches!((got, mv.as_f64()), (Some(a), Some(b)) if close(a, b));
                if !same {
                    errs.push(format!("{mk}: rust {got:?} python {mv}"));
                }
            }
            let confs = sha_floats(r.res.hypotheses.iter().map(|&i| r.res.store[i].conf));
            if exact_platform() && confs != run["confs"].as_str().unwrap() {
                errs.push(format!("ranking confidences differ ({} vs {} hyps)", r.res.hypotheses.len(), run["n_hyps"]));
            }
            if errs.is_empty() {
                n_metrics += run["metrics"].as_object().unwrap().len();
                matched = true;
                break;
            }
            last_err = format!("{k}: {}", errs.iter().take(6).cloned().collect::<Vec<_>>().join("; "));
        }
        if !matched {
            failures.push(format!(
                "{arch} {scale}/{seed} alloc={:?} shape={:?} q={:?} ranker={:?} over={}: {last_err}",
                run.get("alloc"),
                run.get("shape"),
                run.get("q"),
                run.get("ranker"),
                run["cfg_over"]
            ));
        }
    }
    eprintln!("{} runs, {n_metrics} metric values identical", runs.as_array().unwrap().len());
    assert!(n_metrics > 117 * 70, "too few metrics compared: {n_metrics}");
    assert!(
        failures.is_empty(),
        "{} of {} runs differ:\n{}",
        failures.len(),
        runs.as_array().unwrap().len(),
        failures.join("\n")
    );
}
