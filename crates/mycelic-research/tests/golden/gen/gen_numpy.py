"""Record numpy / CPython behaviour the Rust port must reproduce exactly.

Run with the Python research environment (numpy 2.4.6, CPython 3.11):
    python crates/mycelic-research/tests/golden/gen/gen_numpy.py
Writes numpy_rng.json, numpy_math.json, pyset.json and pyfmt.json next to this directory.
"""
import hashlib
import json
import os
import random
import struct
import sys

import numpy as np

OUT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def bits(x):
    return struct.unpack("<Q", struct.pack("<d", float(x)))[0]


def compact(values):
    """Long integer outputs are stored as length, head and a SHA-256 of their decimal text."""
    values = [int(v) for v in values]
    if len(values) <= 120:
        return values
    text = ",".join(str(v) for v in values)
    return {"len": len(values), "head": values[:20],
            "sha256": hashlib.sha256(text.encode()).hexdigest()}


def rng_cases():
    py = random.Random(20261021)
    seeds = list(range(0, 12)) + [42, 99, 500, 501, 502, 900, 901, 7777, 31_000, 41_000, 50_000 + 13 * 3 + 280,
                                  55_000, 61_000, 66_000, 77_000, 88_000, 90_000, 1_000_003, 1_000_503,
                                  2**31 - 1, 2**32, 2**32 + 5, 2**40 + 17, 2**63 + 3, 2**64 - 1]
    int_ranges = [(0, 1), (0, 2), (0, 3), (0, 7), (5, 6), (-5, 5), (0, 100), (40, 220), (900, 4200), (0, 180),
                  (0, 2**31), (0, 2**32 - 1), (0, 2**32), (0, 2**32 + 1), (-2**31, 2**31), (0, 2**40),
                  (0, 2**62), (-2**62, 2**62), (3, 9), (12, 41), (1, 3)]
    binoms = [(0, 0.3), (10, 0.0), (10, 1.0), (5, 0.3), (40, 0.5), (100, 0.5), (1000, 0.035), (1000, 0.9),
              (3000, 0.015), (12000, 0.2), (7, 0.95), (200, 0.75), (50000, 0.02), (1, 0.5), (60, 0.5)]
    cases = []
    for seed in seeds:
        g = np.random.default_rng(seed)
        ops = []
        for _ in range(90):
            kind = py.choice(["random", "random_n", "integers", "integers_n", "integers_closed", "normal_n",
                              "lognormal_n", "binomial", "shuffle", "shuffle_arr", "permutation", "choice_nr",
                              "choice_r", "choice_scalar", "choice_p", "integers"])
            if kind == "random":
                ops.append({"op": kind, "out": bits(g.random())})
            elif kind == "random_n":
                n = py.randint(0, 12)
                ops.append({"op": kind, "n": n, "out": [bits(x) for x in g.random(n)]})
            elif kind == "integers":
                lo, hi = py.choice(int_ranges)
                ops.append({"op": kind, "lo": lo, "hi": hi, "out": int(g.integers(lo, hi))})
            elif kind == "integers_n":
                lo, hi = py.choice(int_ranges)
                n = py.randint(0, 15)
                ops.append({"op": kind, "lo": lo, "hi": hi, "n": n, "out": [int(x) for x in g.integers(lo, hi, n)]})
            elif kind == "integers_closed":
                lo, hi = py.choice(int_ranges)
                ops.append({"op": kind, "lo": lo, "hi": hi - 1,
                            "out": int(g.integers(lo, hi - 1, endpoint=True))})
            elif kind == "normal_n":
                loc, sc = py.choice([(0.0, 1.0), (0.0, 0.42), (3.3, 0.55), (-1.0, 2.5), (0.0, 0.06)])
                n = py.randint(0, 20)
                ops.append({"op": kind, "loc": loc, "scale": sc, "n": n,
                            "out": [bits(x) for x in g.normal(loc, sc, n)]})
            elif kind == "lognormal_n":
                m, s = py.choice([(np.log(28.0), 0.55), (np.log(9.0), 0.32), (np.log(5.0), 0.6), (0.0, 1.0)])
                n = py.randint(0, 20)
                ops.append({"op": kind, "mean": float(m), "sigma": s, "n": n,
                            "out": [bits(x) for x in g.lognormal(m, s, n)]})
            elif kind == "binomial":
                n, p = py.choice(binoms)
                ops.append({"op": kind, "n": n, "p": p, "out": int(g.binomial(n, p))})
            elif kind == "shuffle":
                n = py.randint(0, 30)
                x = list(range(n))
                g.shuffle(x)
                ops.append({"op": kind, "n": n, "out": x})
            elif kind == "shuffle_arr":
                n = py.choice([0, 1, 2, 5, 40, 300])
                x = np.arange(n, dtype=np.int64)
                g.shuffle(x)
                ops.append({"op": kind, "n": n, "out": compact(x)})
            elif kind == "permutation":
                n = py.randint(0, 25)
                ops.append({"op": kind, "n": n, "out": [int(v) for v in g.permutation(n)]})
            elif kind == "choice_nr":
                pop, size = py.choice([(10, 3), (7, 7), (5, 0), (200, 40), (12000, 100), (12000, 300),
                                       (20000, 20000), (15000, 9000), (50, 50), (3, 1), (1000, 900)])
                ops.append({"op": kind, "pop": pop, "size": size,
                            "out": compact(g.choice(pop, size=size, replace=False))})
            elif kind == "choice_r":
                pop, size = py.choice([(10, 3), (3, 5), (100, 20)])
                ops.append({"op": kind, "pop": pop, "size": size,
                            "out": [int(v) for v in g.choice(pop, size=size)]})
            elif kind == "choice_scalar":
                arr = py.choice([[3, 4, 5], [7], [1, 2, 3, 4, 5, 6, 7]])
                ops.append({"op": kind, "values": arr, "out": int(g.choice(np.array(arr)))})
            elif kind == "choice_p":
                k = py.randint(1, 6)
                w = [py.random() for _ in range(k)]
                s = sum(w)
                p = [x / s for x in w]
                n = py.randint(0, 6)
                ops.append({"op": kind, "p": p, "n": n,
                            "out": [int(v) for v in g.choice(k, size=n, p=p)]})
        cases.append({"seed": seed, "ops": ops})
    return cases


def stress_cases():
    """Long runs of the samplers with rare branches: BTPE's Step52 and the ziggurat's tail and wedge."""
    out = {"binomial": [], "normal": [], "lognormal": []}
    grid = [(31, 0.99), (33, 0.5), (100, 0.31), (250, 0.15), (1200, 0.034), (5000, 0.3), (30000, 0.015),
            (30000, 0.30), (250000, 0.04), (1_400_000, 0.015), (3_000_000, 0.27), (777, 0.6), (64, 0.47),
            (2000, 0.034), (12, 0.34), (90, 0.025), (400, 0.2), (15000, 0.02), (120, 0.97), (1000, 0.5)]
    for seed in range(12):
        g = np.random.default_rng(10_000 + seed)
        for n, p in grid:
            out["binomial"].append({"seed": 10_000 + seed, "n": n, "p": p,
                                    "out": [int(g.binomial(n, p)) for _ in range(40)]})
    for seed in range(6):
        g = np.random.default_rng(20_000 + seed)
        z = g.normal(0.0, 1.0, 50_000)
        text = ",".join(str(bits(v)) for v in z)
        out["normal"].append({"seed": 20_000 + seed, "n": 50_000, "head": [bits(v) for v in z[:5]],
                              "sha256": hashlib.sha256(text.encode()).hexdigest()})
        z = g.lognormal(np.log(28.0), 0.55, 20_000)
        text = ",".join(str(bits(v)) for v in z)
        out["lognormal"].append({"seed": 20_000 + seed, "n": 20_000,
                                 "sha256": hashlib.sha256(text.encode()).hexdigest()})
    return out


def math_cases():
    py = random.Random(7)
    g = np.random.default_rng(123)
    out = []
    lengths = list(range(1, 260)) + [300, 511, 1000, 1023, 1500, 5000, 8191, 8192, 8193, 20000]
    for n in lengths:
        kind = py.choice(["normal", "uniform", "mixed", "int"])
        if kind == "normal":
            a = g.normal(0.0, 1.0, n)
        elif kind == "uniform":
            a = g.random(n) * 1e3
        elif kind == "mixed":
            a = np.where(g.random(n) < 0.5, g.random(n) * 1e-3, g.normal(5e4, 1e4, n))
        else:
            a = g.integers(-1000, 1000000, n)
        rec = {"n": n, "kind": kind}
        if kind == "int":
            rec["values"] = [int(v) for v in a]
        else:
            rec["values"] = [bits(v) for v in a]
        rec["sum"] = bits(np.sum(a.astype(np.float64)) if kind != "int" else np.sum(a, dtype=np.float64))
        rec["mean"] = bits(np.mean(a))
        rec["std"] = bits(np.std(a))
        if n > 1:
            rec["std1"] = bits(np.std(a, ddof=1))
        rec["median"] = bits(np.median(a))
        rec["percentiles"] = {str(q): bits(np.percentile(a, q)) for q in (2.5, 10, 50, 90, 97.5, 100, 0)}
        if n <= 1500:
            out.append(rec)
        else:
            rec["values"] = None
            rec["regen_seed"] = None
            out.append({"n": n, "kind": kind, "skip_values": True})
    # python-list means (as np.mean receives them from the report code)
    lists = []
    for n in (1, 2, 3, 5, 9, 17, 40, 130, 300):
        vals = [py.random() * 10 ** py.randint(-3, 4) for _ in range(n)]
        lists.append({"values": [bits(v) for v in vals], "mean": bits(np.mean(vals)),
                      "std": bits(np.std(vals)), "median": bits(np.median(vals))})
    lin = []
    for (a, b, n) in ((0.05, 1.0, 12), (0.01, 1.0, 199), (0.05, 0.95, 19), (0, 59, 45), (0, 1000, 7), (2.0, 2.0, 5)):
        lin.append({"start": a, "stop": b, "num": n, "out": [bits(v) for v in np.linspace(a, b, n)]})
    return {"arrays": [r for r in out if not r.get("skip_values")], "lists": lists, "linspace": lin}


def pyset_cases():
    py = random.Random(99)
    cases = []
    pools = [lambda: py.randint(0, 40), lambda: py.randint(0, 5000), lambda: py.randint(-3, 3),
             lambda: py.randint(-2_000_000_100, -1_999_999_900), lambda: py.randint(0, 2**33),
             lambda: py.choice([-1, -2, 0, 1, 2, 8, 16, 32, 64, 1024, 4096])]
    for _ in range(400):
        pool = py.choice(pools)
        ops = []
        sets = [set()]
        for _ in range(py.randint(1, 12)):
            kind = py.choice(["add", "add", "from_list", "update", "union", "copy", "dict_keys", "add_many"])
            if kind == "add":
                x = pool()
                sets[-1].add(x)
                ops.append({"op": "add", "x": x})
            elif kind == "add_many":
                xs = [pool() for _ in range(py.randint(0, 40))]
                for x in xs:
                    sets[-1].add(x)
                ops.append({"op": "add_many", "xs": xs})
            elif kind == "from_list":
                xs = [pool() for _ in range(py.randint(0, 60))]
                sets.append(set(xs))
                ops.append({"op": "from_list", "xs": xs})
            elif kind == "update":
                xs = [pool() for _ in range(py.randint(0, 50))]
                other = set(xs)
                sets[-1] |= other
                ops.append({"op": "update", "xs": xs})
            elif kind == "union":
                xs = [pool() for _ in range(py.randint(0, 50))]
                other = set(xs)
                sets.append(sets[-1] | other)
                ops.append({"op": "union", "xs": xs})
            elif kind == "copy":
                sets.append(set(sets[-1]))
                ops.append({"op": "copy"})
            elif kind == "dict_keys":
                xs = [pool() for _ in range(py.randint(0, 60))]
                d = dict.fromkeys(xs)
                sets.append(set(d))
                ops.append({"op": "dict_keys", "xs": xs})
            ops[-1]["state"] = list(sets[-1])
        cases.append(ops)
    return cases


def fmt_cases():
    py = random.Random(5)
    specs = [".0f", ".1f", ".2f", ".3f", ".4f", ".5f", "+.3f", "+.4f", "+.2f", ".2e", ".3e", "+.2e", "+.3e", ".0%",
             ".1%", "+.0%", "+.1%", ".3g", "g", ".2g", ",", ",.0f", ".1f", "3d", "+.0f", "6.2f", ""]
    vals = [0.0, -0.0, 0.5, 1.5, 2.5, 0.125, 0.375, 1e-7, 123456.789, 1e16, 1e15, 0.1, 0.2, 1 / 3, 2 / 3, -1 / 3,
            0.04999999999999999, 0.005, 0.0005, 999.9995, 1e5, 99999.99, 12.5, 0.995, 0.9995, 2.675, 1.005,
            float("nan"), float("inf"), -float("inf"), 4.5e-05, 7.7e-310]
    for _ in range(400):
        vals.append(py.choice([py.random(), py.random() * 1e6, -py.random(), py.random() * 1e-4,
                               py.randint(0, 10**7) / 1000, py.randint(0, 999) / 1000.0, py.gauss(0, 0.1)]))
    out = []
    for v in vals:
        for s in specs:
            if s in (",", "3d"):
                continue
            try:
                out.append({"v": bits(v), "spec": s, "out": format(v, s)})
            except ValueError:
                pass
    ints = [0, 7, -7, 999, 1000, 1234567, -1234567, 50000, 100000]
    iout = []
    for n in ints:
        for s in (",", "3d", "", ",.0f", ".2f", ".0%", ".2e"):
            iout.append({"v": n, "spec": s, "out": format(n, s)})
    reprs = [{"v": bits(v), "out": repr(v)} for v in vals if v == v and v not in (float("inf"), -float("inf"))]
    rounds = []
    for v in vals[:200]:
        if v == v and abs(v) != float("inf"):
            for nd in (1, 2, 3, 4, 5):
                rounds.append({"v": bits(v), "n": nd, "out": bits(round(v, nd))})
    return {"float": out, "int": iout, "repr": reprs, "round": rounds}


def dump(name, obj):
    with open(os.path.join(OUT, name), "w") as fh:
        json.dump(obj, fh, separators=(",", ":"))
    print("wrote", name, os.path.getsize(os.path.join(OUT, name)), "bytes")


if __name__ == "__main__":
    assert np.__version__ == "2.4.6", np.__version__
    assert sys.version_info[:2] == (3, 11), sys.version
    dump("numpy_rng.json", {"numpy": np.__version__, "cases": rng_cases()})
    dump("numpy_stress.json", stress_cases())
    dump("numpy_math.json", math_cases())
    dump("pyset.json", {"python": sys.version.split()[0], "cases": pyset_cases()})
    dump("pyfmt.json", fmt_cases())
