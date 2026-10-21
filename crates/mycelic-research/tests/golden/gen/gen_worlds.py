"""Record worlds and runs from the Python simulator for the Rust port's end-to-end parity tests.

    PYTHONPATH=. python crates/mycelic-research/tests/golden/gen/gen_worlds.py
Writes worlds.json (generator digests) and runs.json (every metric of every architecture on small worlds).
Uses the committed research/mycelic/artifacts/calibration.json, as the experiments do.
"""
import hashlib
import json
import os
import struct
import sys

import numpy as np

from research.mycelic import experiments as E
from research.mycelic import models as M
from research.mycelic import ops as _ops
from research.mycelic import runner as R
from research.mycelic.evalm import evaluate
from research.mycelic.models import allocation, ANCHORS
from research.mycelic.org import DEFAULT_FANIN, TEAM, USER
from research.mycelic.runner import build_world

OUT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def bits(x):
    return struct.unpack("<Q", struct.pack("<d", float(x)))[0]


def sha_ints(vals):
    return hashlib.sha256(",".join(str(int(v)) for v in vals).encode()).hexdigest()


def sha_floats(vals):
    return hashlib.sha256(",".join(str(bits(v)) for v in vals).encode()).hexdigest()


def world_digest(w):
    org, c = w.org, w.corpus
    recs = c.recs
    d = {
        "org": {
            "n_nodes": len(org.nodes),
            "levels": [len(l) for l in org.levels],
            "level_ids": [sha_ints(l) for l in org.levels],
            "parents": sha_ints([-1 if n.parent is None else n.parent for n in org.nodes]),
            "functions": sha_ints([n.function for n in org.nodes]),
            "n_users": sha_ints([n.n_users for n in org.nodes]),
            "names": hashlib.sha256("\n".join(n.name for n in org.nodes).encode()).hexdigest(),
            "stats": org.stats(),
        },
        "corpus": {
            "n_recs": int(len(recs)),
            "fields": {f: (sha_floats(recs[f]) if f == "salience" else sha_ints(recs[f])) for f in recs.dtype.names},
            "user_slices": sha_ints(c.user_slices),
            "entities": hashlib.sha256("\n".join(c.entities).encode()).hexdigest(),
            "entity_kind": sha_ints(c.entity_kind),
            "patterns": [{"pid": p.pid, "chain": p.chain, "start_pos": p.start_pos, "preds": [int(x) for x in p.preds],
                          "anchor": int(p.anchor), "facet_records": sha_ints([r for fr in p.facet_records for r in fr] + [-len(fr) for fr in p.facet_records]),
                          "facet_users": sha_ints([u for fu in p.facet_users for u in fu]), "facet_times": [int(t) for t in p.facet_times],
                          "real": p.real, "decoy_type": p.decoy_type, "rare": bool(p.rare), "n_regions": p.n_regions,
                          "n_sites": p.n_sites, "contradicted": bool(p.contradicted), "family": p.family} for p in c.patterns],
            "families": [[f, ps] for f, ps in c.families.items()],
            "contradiction_pairs": sha_ints([x for pr in c.contradiction_pairs for x in pr]),
            "notable": sha_ints(c.notable_records),
            "stats": c.stats(),
            "texts": {str(r): c.text(r) for r in sorted({0, 17, 100, len(recs) // 2, len(recs) - 1}
                                                         | {fr[0] for p in c.patterns[:6] for fr in p.facet_records if fr})},
        },
        "gold": {"discoverable": w.gold.discoverable, "rare": w.gold.rare, "contradicted": w.gold.contradicted,
                 "families": [[f, ps] for f, ps in w.gold.families.items()], "decoys": w.gold.decoys,
                 "top_supported": w.gold.top_supported,
                 "independent_sources": [[k, v] for k, v in w.gold.independent_sources.items()]},
        "near_miss": sha_ints(w.near_miss),
        "stem": sha_ints(w.stem),
    }
    ul = w.user_layer(ANCHORS["small-7b"], w.seed)
    ex = ul.ex
    d["user_layer"] = {"n": int(len(ex)), "rid": sha_ints(ex.rid), "uid": sha_ints(ex.uid), "pred": sha_ints(ex.pred),
                       "anchor": sha_ints(ex.anchor), "t": sha_ints(ex.t), "sig": sha_ints(ex.sig),
                       "spurious": sha_ints(ex.spurious), "imp": sha_floats(ul.imp), "nov": sha_floats(ul.nov),
                       "sketch_tokens": int(ul.sketch_tokens)}
    return d


def run_digest(name, w, alloc, seed, cfg_over=None):
    res = E._run_named(name, w, alloc, seed, cfg_over=cfg_over)
    m = evaluate(w.corpus, w.gold, res)
    hy = res.hypotheses
    return {"arch": name, "scale": w.scale, "seed": seed, "cfg_over": cfg_over,
            "metrics": m, "n_hyps": len(hy),
            "top": [[int(h.anchor), [int(p) for p in h.preds], bits(h.conf)] for h in hy[:40]],
            "confs": sha_floats([h.conf for h in hy]),
            "anchors": sha_ints([h.anchor for h in hy]),
            "by_stage": {k: [v.calls, v.tok_in, v.tok_out, bits(v.cu)] for k, v in res.meter.levels.items()},
            "n_kernel_kos": len(res.kernel_kos), "n_questions": len(res.questions)}


if __name__ == "__main__":
    worlds = []
    specs = [(1200, 0, None), (2000, 0, None), (2000, 1, None), (5000, 2, None), (3000, 4, None),
             (2000, 0, {"team_size": 6}), (2000, 1, {"n_regions": 3}), (2000, 2, {"corpus": {"p_dup": 0.30}}),
             (2000, 3, {"corpus": {"decoy_ratio": 3.0}})]
    built = {}
    for scale, seed, opt in specs:
        kw = {}
        if opt and "team_size" in opt:
            fan = dict(DEFAULT_FANIN)
            m_, s_, lo, hi = fan[TEAM]
            fan[TEAM] = (float(opt["team_size"]), s_, lo, hi)
            kw["org_kwargs"] = {"fanin": fan}
        if opt and "n_regions" in opt:
            kw["org_kwargs"] = {"n_regions": opt["n_regions"]}
        if opt and "corpus" in opt:
            kw["corpus_cfg"] = opt["corpus"]
        w = build_world(scale, seed, **kw)
        d = world_digest(w)
        d.update({"scale": scale, "seed": seed, "opts": opt})
        worlds.append(d)
        built[(scale, seed, json.dumps(opt))] = w
        print("world", scale, seed, opt, flush=True)
    with open(os.path.join(OUT, "worlds.json"), "w") as fh:
        json.dump(worlds, fh, separators=(",", ":"))

    runs = []
    alloc = allocation("back-loaded")
    w = built[(2000, 0, "null")]
    for a in R.ARCHS:
        runs.append(run_digest(a, w, alloc, 0))
        print("run", a, flush=True)
    for key, archs in (((1200, 0, "null"), ["H_mycelic_full", "A2_chunked_ctx", "B4_central_triage", "Y_oracle_retrieval",
                                           "J_mycelic_verified", "I_mycelic_completion", "H_mycelic_lean"]),
                       ((2000, 1, "null"), list(R.ARCHS)),
                       ((5000, 2, "null"), ["H_mycelic_full", "H_mycelic_prev", "B2_map_reduce", "A_flat_rag", "B4_central_triage",
                                           "A2_chunked_ctx", "J_mycelic_verified", "G_hier_questions"]),
                       ((3000, 4, "null"), ["H_mycelic_full", "Y_oracle_retrieval", "B_long_context"]),
                       ((2000, 1, json.dumps({"n_regions": 3})), ["H_mycelic_full", "B2_map_reduce"]),
                       ((2000, 2, json.dumps({"corpus": {"p_dup": 0.30}})), ["H_mycelic_full", "E_hier_lineage"]),
                       ((2000, 0, json.dumps({"team_size": 6})), ["H_mycelic_full"])):
        wv = built[key]
        for a in archs:
            runs.append(run_digest(a, wv, alloc, wv.seed))
            print("run", key, a, flush=True)
    # hierarchy configuration variants (E2 ablations, E5 adversarial knobs, vNext knobs)
    for over in ({"malicious_frac": 0.05}, {"unavailable_frac": 0.2}, {"bloom_fp": 0.02}, {"lineage": False},
                 {"independence": False}, {"temporal": False}, {"adaptive_routing": False}, {"question_targeting": False},
                 {"sketch_channel": False}, {"local_reextract": True}, {"sketch_weak_bits": True},
                 {"merge_descent_evidence": True}, {"link_time": "modal"}, {"link_time": "min", "stale_rule": "strong"},
                 {"strict_targeting": True, "triage_target_chains": "span2"}, {"triage_prior_weight": 1.2},
                 {"chain_completion": True}, {"question_frac": 1.0}, {"cross_links": False, "batched_descent": False},
                 {"foreign_only_evidence": False}, {"restrict_synthesis_to_triage": False}, {"w_dispersion": 1.5},
                 {"verify_evidence": True}):
        runs.append(run_digest("H_mycelic_full", w, alloc, 0, cfg_over=over))
        print("variant", over, flush=True)
    # every allocation and a uniform-q tier, and the three capability shapes
    for aname in M.ALLOCATIONS:
        r = run_digest("H_mycelic_full", w, allocation(aname), 0)
        r["alloc"] = aname
        runs.append(r)
        r = run_digest("B2_map_reduce", w, allocation(aname), 0)
        r["alloc"] = aname
        runs.append(r)
    w.clear_cache()
    for shape in ("linear", "concave"):
        M.HARD_SHAPE = shape
        M.ANCHORS.update({k: M.anchor(k) for k in M.ANCHOR_Q})
        for aname in ("naive-ladder", "two-step"):
            r = run_digest("H_mycelic_full", w, allocation(aname), 0)
            r["alloc"] = aname
            r["shape"] = shape
            runs.append(r)
        w.clear_cache()
    M.HARD_SHAPE = "convex"
    M.ANCHORS.update({k: M.anchor(k) for k in M.ANCHOR_Q})
    for q in (0.10, 0.58, 1.00):
        alloc_q = [M.tier_from_q(q, name=f"q{q:.2f}", params_b=float(3.0 * 40.0 ** q), ctx=int(8000 * 128 ** q))] * 6
        for a in ("H_mycelic_full", "E_hier_lineage"):
            r = run_digest(a, w, alloc_q, 0)
            r["q"] = q
            runs.append(r)
    # the ranker forced off (the hand-set logistic everywhere)
    R.apply_ranker(False)
    for a in ("H_mycelic_full", "B4_central_triage", "A2_chunked_ctx"):
        r = run_digest(a, w, alloc, 0)
        r["ranker"] = "off"
        runs.append(r)
    R.apply_ranker(None)
    with open(os.path.join(OUT, "runs.json"), "w") as fh:
        json.dump(runs, fh, separators=(",", ":"))
    print("wrote", len(worlds), "worlds,", len(runs), "runs")
