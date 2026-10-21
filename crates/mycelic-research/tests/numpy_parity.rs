//! The numpy and CPython behaviours the port reproduces, checked against values recorded from numpy 2.4.6 and
//! CPython 3.11 (`tests/golden/gen/gen_numpy.py`): every Generator method the simulator calls, the float
//! reductions, set iteration order and format specs.

use mycelic_research::np::Generator;
use mycelic_research::np::math;
use mycelic_research::py::{PySet, fmt, json};
use serde_json::Value;
use sha2::{Digest, Sha256};

fn load(name: &str) -> Value {
    let path = format!("{}/tests/golden/{name}", env!("CARGO_MANIFEST_DIR"));
    serde_json::from_str(&std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"))).unwrap()
}

/// Summaries of one array: sum, mean, std, std with ddof=1, median and the percentile function.
type Reductions = (f64, f64, f64, Option<f64>, f64, Box<dyn Fn(f64) -> f64>);

fn f(bits: &Value) -> f64 {
    f64::from_bits(bits.as_u64().unwrap())
}

fn ints(v: &Value) -> Vec<i64> {
    v.as_array().unwrap().iter().map(|x| x.as_i64().unwrap()).collect()
}

/// Compare an integer output with its fixture, which is either the full list or (len, head, sha256).
fn check_ints(got: &[i64], want: &Value, ctx: &str) {
    if want.is_array() {
        assert_eq!(got, ints(want).as_slice(), "{ctx}");
        return;
    }
    assert_eq!(got.len() as u64, want["len"].as_u64().unwrap(), "{ctx} len");
    assert_eq!(&got[..20], ints(&want["head"]).as_slice(), "{ctx} head");
    let text: Vec<String> = got.iter().map(|v| v.to_string()).collect();
    let digest = hex::encode(Sha256::digest(text.join(",").as_bytes()));
    assert_eq!(digest, want["sha256"].as_str().unwrap(), "{ctx} sha256");
}

#[test]
fn generator_methods_match_numpy() {
    let fx = load("numpy_rng.json");
    let mut n_ops = 0;
    let mut n_values = 0;
    for case in fx["cases"].as_array().unwrap() {
        let seed = case["seed"].as_u64().unwrap();
        let mut g = Generator::new(seed);
        for (k, op) in case["ops"].as_array().unwrap().iter().enumerate() {
            let ctx = format!("seed {seed} op {k} {op}");
            let name = op["op"].as_str().unwrap();
            n_ops += 1;
            match name {
                "random" => {
                    assert_eq!(g.random(), f(&op["out"]), "{ctx}");
                    n_values += 1;
                }
                "random_n" => {
                    let got = g.random_n(op["n"].as_u64().unwrap() as usize);
                    let want: Vec<f64> = op["out"].as_array().unwrap().iter().map(f).collect();
                    n_values += want.len();
                    assert_eq!(got, want, "{ctx}");
                }
                "integers" => {
                    let got = g.integers(op["lo"].as_i64().unwrap(), op["hi"].as_i64().unwrap());
                    assert_eq!(got, op["out"].as_i64().unwrap(), "{ctx}");
                    n_values += 1;
                }
                "integers_closed" => {
                    let got = g.integers_closed(op["lo"].as_i64().unwrap(), op["hi"].as_i64().unwrap());
                    assert_eq!(got, op["out"].as_i64().unwrap(), "{ctx}");
                    n_values += 1;
                }
                "integers_n" => {
                    let n = op["n"].as_u64().unwrap() as usize;
                    let got = g.integers_n(op["lo"].as_i64().unwrap(), op["hi"].as_i64().unwrap(), n);
                    n_values += n;
                    check_ints(&got, &op["out"], &ctx);
                }
                "normal_n" => {
                    let n = op["n"].as_u64().unwrap() as usize;
                    let got = g.normal_n(op["loc"].as_f64().unwrap(), op["scale"].as_f64().unwrap(), n);
                    let want: Vec<f64> = op["out"].as_array().unwrap().iter().map(f).collect();
                    n_values += n;
                    assert_eq!(got, want, "{ctx}");
                }
                "lognormal_n" => {
                    let n = op["n"].as_u64().unwrap() as usize;
                    let got = g.lognormal_n(op["mean"].as_f64().unwrap(), op["sigma"].as_f64().unwrap(), n);
                    let want: Vec<f64> = op["out"].as_array().unwrap().iter().map(f).collect();
                    n_values += n;
                    assert_eq!(got, want, "{ctx}");
                }
                "binomial" => {
                    let got = g.binomial(op["n"].as_i64().unwrap(), op["p"].as_f64().unwrap());
                    assert_eq!(got, op["out"].as_i64().unwrap(), "{ctx}");
                    n_values += 1;
                }
                "shuffle" | "shuffle_arr" => {
                    let n = op["n"].as_u64().unwrap() as i64;
                    let mut x: Vec<i64> = (0..n).collect();
                    g.shuffle(&mut x);
                    n_values += x.len();
                    check_ints(&x, &op["out"], &ctx);
                }
                "permutation" => {
                    let got = g.permutation(op["n"].as_u64().unwrap() as usize);
                    n_values += got.len();
                    check_ints(&got, &op["out"], &ctx);
                }
                "choice_nr" => {
                    let got = g.choice_without_replace(
                        op["pop"].as_u64().unwrap() as usize,
                        op["size"].as_u64().unwrap() as usize,
                    );
                    n_values += got.len();
                    check_ints(&got, &op["out"], &ctx);
                }
                "choice_r" => {
                    let pop = op["pop"].as_i64().unwrap();
                    let got = g.integers_n(0, pop, op["size"].as_u64().unwrap() as usize);
                    n_values += got.len();
                    check_ints(&got, &op["out"], &ctx);
                }
                "choice_scalar" => {
                    let values = ints(&op["values"]);
                    let got = values[g.choice_index(values.len())];
                    assert_eq!(got, op["out"].as_i64().unwrap(), "{ctx}");
                    n_values += 1;
                }
                "choice_p" => {
                    let p: Vec<f64> = op["p"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect();
                    let got: Vec<i64> =
                        g.choice_p(&p, op["n"].as_u64().unwrap() as usize).into_iter().map(|v| v as i64).collect();
                    n_values += got.len();
                    check_ints(&got, &op["out"], &ctx);
                }
                other => panic!("unknown op {other}"),
            }
        }
    }
    assert!(n_ops > 3000 && n_values > 20000, "fixture too small: {n_ops} ops, {n_values} values");
}

#[test]
fn reductions_match_numpy() {
    let fx = load("numpy_math.json");
    for rec in fx["arrays"].as_array().unwrap() {
        let n = rec["n"].as_u64().unwrap();
        let kind = rec["kind"].as_str().unwrap();
        let ctx = format!("n={n} kind={kind}");
        let (sum, mean, std, std1, median, pct): Reductions;
        if kind == "int" {
            let v = ints(&rec["values"]);
            sum = math::sum_int_as_f64(&v);
            mean = math::mean_int(&v);
            std = math::std_int(&v, 0);
            std1 = (v.len() > 1).then(|| math::std_int(&v, 1));
            median = math::median_int(&v);
            let vv = v.clone();
            pct = Box::new(move |q| math::percentile_int(&vv, q));
        } else {
            let v: Vec<f64> = rec["values"].as_array().unwrap().iter().map(f).collect();
            sum = math::sum(&v);
            mean = math::mean(&v);
            std = math::std(&v, 0);
            std1 = (v.len() > 1).then(|| math::std(&v, 1));
            median = math::median(&v);
            let vv = v.clone();
            pct = Box::new(move |q| math::percentile(&vv, q));
        }
        assert_eq!(sum.to_bits(), f(&rec["sum"]).to_bits(), "{ctx} sum");
        assert_eq!(mean.to_bits(), f(&rec["mean"]).to_bits(), "{ctx} mean");
        assert_eq!(std.to_bits(), f(&rec["std"]).to_bits(), "{ctx} std");
        if let Some(s1) = std1 {
            assert_eq!(s1.to_bits(), f(&rec["std1"]).to_bits(), "{ctx} std ddof=1");
        }
        assert_eq!(median.to_bits(), f(&rec["median"]).to_bits(), "{ctx} median");
        for (q, want) in rec["percentiles"].as_object().unwrap() {
            let q: f64 = q.parse().unwrap();
            assert_eq!(pct(q).to_bits(), f(want).to_bits(), "{ctx} percentile {q}");
        }
    }
    for rec in fx["lists"].as_array().unwrap() {
        let v: Vec<f64> = rec["values"].as_array().unwrap().iter().map(f).collect();
        assert_eq!(math::mean(&v).to_bits(), f(&rec["mean"]).to_bits());
        assert_eq!(math::std(&v, 0).to_bits(), f(&rec["std"]).to_bits());
        assert_eq!(math::median(&v).to_bits(), f(&rec["median"]).to_bits());
    }
    for rec in fx["linspace"].as_array().unwrap() {
        let got = math::linspace(
            rec["start"].as_f64().unwrap(),
            rec["stop"].as_f64().unwrap(),
            rec["num"].as_u64().unwrap() as usize,
        );
        let want: Vec<f64> = rec["out"].as_array().unwrap().iter().map(f).collect();
        assert_eq!(got, want, "linspace {rec}");
    }
}

#[test]
fn set_iteration_order_matches_cpython() {
    let fx = load("pyset.json");
    for (ci, case) in fx["cases"].as_array().unwrap().iter().enumerate() {
        let mut sets = vec![PySet::new()];
        for op in case.as_array().unwrap() {
            let xs = || ints(&op["xs"]);
            match op["op"].as_str().unwrap() {
                "add" => sets.last_mut().unwrap().add(op["x"].as_i64().unwrap()),
                "add_many" => {
                    for x in xs() {
                        sets.last_mut().unwrap().add(x);
                    }
                }
                "from_list" => sets.push(PySet::from_iter_seq(xs())),
                "update" => {
                    let other = PySet::from_iter_seq(xs());
                    sets.last_mut().unwrap().update_set(&other);
                }
                "union" => {
                    let other = PySet::from_iter_seq(xs());
                    let u = sets.last().unwrap().union(&other);
                    sets.push(u);
                }
                "copy" => {
                    let c = PySet::copy_of(sets.last().unwrap());
                    sets.push(c);
                }
                "dict_keys" => {
                    let mut keys: Vec<i64> = Vec::new();
                    for x in xs() {
                        if !keys.contains(&x) {
                            keys.push(x);
                        }
                    }
                    sets.push(PySet::from_dict_keys(&keys));
                }
                other => panic!("unknown op {other}"),
            }
            assert_eq!(sets.last().unwrap().to_vec(), ints(&op["state"]), "case {ci} after {op}");
        }
    }
}

#[test]
fn format_specs_match_cpython() {
    let fx = load("pyfmt.json");
    for rec in fx["float"].as_array().unwrap() {
        let v = f(&rec["v"]);
        let spec = rec["spec"].as_str().unwrap();
        assert_eq!(fmt::f(v, spec), rec["out"].as_str().unwrap(), "format({v:?}, {spec:?})");
    }
    for rec in fx["int"].as_array().unwrap() {
        let v = rec["v"].as_i64().unwrap();
        let spec = rec["spec"].as_str().unwrap();
        assert_eq!(fmt::i(v, spec), rec["out"].as_str().unwrap(), "format({v}, {spec:?})");
    }
    for rec in fx["repr"].as_array().unwrap() {
        let v = f(&rec["v"]);
        assert_eq!(fmt::repr(v), rec["out"].as_str().unwrap(), "repr({v:?})");
    }
    for rec in fx["round"].as_array().unwrap() {
        let v = f(&rec["v"]);
        let n = rec["n"].as_u64().unwrap() as usize;
        assert_eq!(json::round(v, n).to_bits(), f(&rec["out"]).to_bits(), "round({v:?}, {n})");
    }
}

#[test]
fn long_sampler_runs_match_numpy() {
    let fx = load("numpy_stress.json");
    let mut draws = 0;
    // the fixture draws every (n, p) of the grid from one generator per seed, in order
    let recs = fx["binomial"].as_array().unwrap();
    let mut i = 0;
    while i < recs.len() {
        let seed = recs[i]["seed"].as_u64().unwrap();
        let mut g = Generator::new(seed);
        while i < recs.len() && recs[i]["seed"].as_u64().unwrap() == seed {
            let (n, p) = (recs[i]["n"].as_i64().unwrap(), recs[i]["p"].as_f64().unwrap());
            let got: Vec<i64> = (0..40).map(|_| g.binomial(n, p)).collect();
            assert_eq!(got, ints(&recs[i]["out"]), "binomial seed {seed} n={n} p={p}");
            draws += got.len();
            i += 1;
        }
    }
    for (k, rec) in fx["normal"].as_array().unwrap().iter().enumerate() {
        let mut g = Generator::new(rec["seed"].as_u64().unwrap());
        let z = g.normal_n(0.0, 1.0, rec["n"].as_u64().unwrap() as usize);
        let text: Vec<String> = z.iter().map(|v| v.to_bits().to_string()).collect();
        assert_eq!(hex::encode(Sha256::digest(text.join(",").as_bytes())), rec["sha256"].as_str().unwrap());
        let ln = &fx["lognormal"][k];
        let z = g.lognormal_n(28f64.ln(), 0.55, ln["n"].as_u64().unwrap() as usize);
        let text: Vec<String> = z.iter().map(|v| v.to_bits().to_string()).collect();
        assert_eq!(hex::encode(Sha256::digest(text.join(",").as_bytes())), ln["sha256"].as_str().unwrap());
        draws += 70_000;
    }
    assert!(draws > 400_000);
}
