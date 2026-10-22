"""Record the calibrator's numerics on a slice of the committed feature dump for the Rust parity test.

    PYTHONPATH=. python crates/mycelic-research/tests/golden/gen/gen_calibrator.py
"""
import json
import os
import struct

import numpy as np

from research.mycelic import calibrator as C

OUT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def bits(x):
    return struct.unpack("<Q", struct.pack("<d", float(x)))[0]


rows = [json.loads(l) for l in open("research/mycelic/artifacts/hyp_features.jsonl")]
tr = [r for r in rows if r["seed"] in (500, 501)][::8]
te = [r for r in rows if r["seed"] == 502][::8]
out = {"train": tr, "test": te}
for inter in (False, True):
    X = C._design(tr, inter)
    y = np.array([1.0 if r["gold"] else 0.0 for r in tr])
    Xs, mu, sd = C._standardise(X)
    Xt, _, _ = C._standardise(C._design(te, inter), mu, sd)
    w = C.fit_logistic(Xs, y, l2=1.0)
    rw = C._row_weight(tr, 3.0)
    model = C.fit_gbdt(Xs, y, rounds=12, row_weight=rw)
    out[f"inter_{inter}"] = {
        "mu": [bits(v) for v in mu], "sd": [bits(v) for v in sd],
        "logistic": [float(v) for v in w],
        "gbdt_trees": model["trees"],
        "gbdt_pred": [bits(v) for v in C.gbdt_predict(model, Xt)],
        "auc_conf": bits(C.auc(np.array([r["conf"] for r in te]), np.array([1.0 if r["gold"] else 0.0 for r in te]))),
    }
with open(os.path.join(OUT, "calibrator.json"), "w") as fh:
    json.dump(out, fh, separators=(",", ":"))
print("wrote calibrator.json", len(tr), "train rows", len(te), "test rows")
