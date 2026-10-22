# Golden fixtures

Values recorded from the Python research implementation (Python 3.11.15, numpy 2.4.6, macOS arm64) that the Rust
port reproduces. They were written by the scripts `gen/gen_numpy.py`, `gen/gen_worlds.py` and
`gen/gen_calibrator.py`, run at the repository root against the Python at `f7e954b`; the scripts were removed with the
rest of the Python and are in the history at `96cb1dd` (`git show 96cb1ddrates/mycelic-research/tests/golden/gen/gen_worlds.py`).

| file | what it pins | test |
|---|---|---|
| `numpy_rng.json`, `numpy_stress.json` | every `Generator` method the simulator calls, 41 seeds, >400k draws; long BTPE/ziggurat runs | `numpy_parity.rs` |
| `numpy_math.json`, `boot.json`, `argsort.json` | pairwise sums, mean/std/median/percentile/linspace, the bootstrap CI and sign test, numpy's default argsort on ties | `numpy_parity.rs` |
| `pyset.json`, `pyfmt.json` | CPython set iteration order; format specs, float repr, `round` | `numpy_parity.rs` |
| `worlds.json`, `runs.json` | 9 generated worlds (digests of every record field, patterns, gold, user layer) and every metric of 117 runs | `python_parity.rs` |
| `calibrator.json` | standardisation, gradient-boosted trees, AUC on a slice of the committed feature dump | `calibrator_parity.rs` |

A change that moves a recorded value changes the published numbers: regenerate the documents, say why in the commit,
and update the fixture deliberately in the same commit (the generators can be restored from the history and run against
`f7e954b` with numpy 2.4.6).
