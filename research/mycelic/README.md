# Mycelic — hierarchical enterprise intelligence benchmark

A mechanism study: how should thousands of private user-level agents abstract,
route, combine, question, verify and propagate knowledge upward so that an
enterprise kernel discovers things no single part of the organisation could?

The findings, with every table generated from the raw rows, are in
[`docs/MYCELIC_ENTERPRISE.md`](../../docs/MYCELIC_ENTERPRISE.md).
The chronological research log — including every design that was measured and
thrown away — is in [`logs/research_log.md`](logs/research_log.md).

The code is the Rust crate [`crates/mycelic-research`](../../crates/mycelic-research)
(the study was first written in Python; that implementation was ported and
removed — its last commit is `f7e954b`, see [Reproducibility](#reproducibility)).
This directory holds the data: `artifacts/` (raw per-run rows, figures, the
live-measurement files), `keys/` (answer keys for the live measurements, kept
out of the served directory), `logs/` (the history of the Python-era runs) and
the two driver scripts.

## Layout

Every module of the Python is a module of the crate and a subcommand of the
`mycelic-research` binary with the same arguments
(`python3 -m research.mycelic.<module> args` became `mycelic-research <module> args`).

| module | what it is |
|---|---|
| `org` | heterogeneous six-level enterprise generator |
| `corpus` | records, hidden causal-chain patterns, five decoy classes, ground truth |
| `models` | model tiers as a sweepable capability vector; cost and latency model |
| `ops` | the shared cognitive operators every architecture calls |
| `systems` | all architectures and controls |
| `evalm` | metrics (raw measurements; composites are secondary) |
| `runner` | world construction, architecture registry, calibrated knobs |
| `calibrate` | knob fitting on held-out seeds 500-502 |
| `experiments` | E1-E12, each streaming rows to its own JSONL |
| `quick_paired` / `arm_paired` | paired A/B runs on identical worlds (the `artifacts/quick_*.jsonl` evidence) |
| `analysis` | bootstrap CIs, paired exact sign tests |
| `calibrator` | the learned candidate ranker: feature dumps, fitting, per-architecture adoption, the final evaluation |
| `freeze_vnext` | writes the frozen vNext knobs into `calibration.json` with the file that justified each |
| `final_rerun` | the calibration-file steps of `final_rerun.sh` (archive the v1 calibration, install a ranker) |
| `loss_accounting` / `loss_report` | per-pattern loss funnel: where each hidden pattern is lost |
| `findings` | headline claims, computed from the rows |
| `report` / `report_text` | generates `docs/MYCELIC_ENTERPRISE.md` |
| `loss_doc` | generates `docs/MYCELIC_LOSS_ACCOUNTING.md` |
| `vnext_data` / `vnext_docs` | generate `docs/mycelic_vnext/*` |
| `plots` | the figures in `artifacts/figures/` |
| `make_pdf` | renders a generated document to a paginated PDF through headless Chrome/Chromium |
| `live_tasks` / `score_live` | blind primitive-operator measurement on real models |
| `live_rank` | blind candidate-discrimination measurement on real models |
| `final_rerun.sh` | the full rerun on the frozen vNext configuration: E1-E12, the loss funnel and the ranker evaluation (the paired `quick_*.jsonl` evidence and the live `live_*.json` measurements are not rerun) |
| `run_suite.sh` | every experiment after E1, in sequence, on whatever `calibration.json` holds |
| `artifacts/` | raw per-run metrics, one JSON object per run; `artifacts/v1/` is the archived pre-vNext benchmark |
| `keys/` | answer keys for the live measurements, kept OUT of the served directory |

The research tests (the port of the Python suite, numpy/CPython parity, the
recorded-Python parity of the simulator, document byte-identity) are under
`crates/mycelic-research/tests/`.

## Running it

Everything needs only a Rust toolchain (1.88+). `make_pdf` additionally needs
Google Chrome or Chromium (found on the usual paths, or set `MYCELIC_CHROME`).

```sh
cargo build --release -p mycelic-research
R=target/release/mycelic-research
```

**Regenerate every document from the committed artifacts** (seconds, changes
nothing but the "Generated" lines):

```sh
cargo test --release -p mycelic-research    # the research tests
$R report                                   # docs/MYCELIC_ENTERPRISE.md
$R loss_doc                                 # docs/MYCELIC_LOSS_ACCOUNTING.md
$R vnext_docs                               # docs/mycelic_vnext/*
$R plots                                    # artifacts/figures/*.png
$R make_pdf                                 # docs/MYCELIC_ENTERPRISE.pdf
```

The last lines of `final_rerun.sh` render the other two PDFs.

**Rerun the published benchmark from scratch** (minutes; the frozen vNext
configuration, with the arguments the final rerun used):

```sh
sh research/mycelic/final_rerun.sh question_frac=0.65:qf_v3 batched_descent=true:v3_H \
   link_time=hybrid:refit_hyb local_reextract=false:rx_reextract \
   strict_targeting=false:cal_screen triage_target_chains=none:cal_screen
```

Experiments append to their JSONL files. The headline E1/E1b/E10 tables
keep the first row per architecture, scale and seed, but every other table,
figure and finding (E2-E9, E12, the loss funnel) pools every row, so a rerun
has to start from empty files. `final_rerun.sh` does that: on a checkout where
`artifacts/v1/` already exists it moves the current rows to
`artifacts/previous/` (gitignored). Running experiments by hand on top of the
committed files changes published numbers; `git checkout
research/mycelic/artifacts` restores them. `MYCELIC_ART=<dir>` points every
subcommand at another artifacts directory (with a `calibration.json` in it),
which is the safe way to rerun anything.

The rerun installs the learned ranker from the committed
`artifacts/calibration.hyb.json`, fitted on the interim candidate dumps
`_hybH`, `_v3C` and `_J` (10-25 MB each, now gitignored). `_v3C` and `_J` were
committed during development and can be restored with
`git show 9713789^:research/mycelic/artifacts/hyp_features_v3C.jsonl` (and
`_J`); `_hybH` was only ever committed as an in-flight partial file, so the
weights can be reused but not refitted exactly. With complete dumps in place,
`$R calibrator select _hybH,_v3C,_J` refits the ranker
into `artifacts/calibration.json`; pass
`RANKER=research/mycelic/artifacts/calibration.json` to `final_rerun.sh` to
rerun with it, since by default the rerun reinstalls `calibration.hyb.json`.

**The archived v1 benchmark** (`artifacts/v1/`): `$R calibrate` (with `ct` and
`evidence`) fits the v1 knobs on seeds 500-502. It rewrites
`calibration.json` from scratch, dropping the frozen vNext knobs and the
learned ranker, so run it only to reproduce v1.

**Live measurements.** To re-score the committed model answers, run only the
scorers:

```sh
$R score_live
$R live_rank score           # add `rich` for the notes variant
```

`live_tasks` and `live_rank` build task files and answer keys for a *new*
measurement: they overwrite the committed ones, after which the committed
answers no longer match their keys. For a new measurement: build the task
file, hand it to the model, drop its answers in `artifacts/`, then score.

## Reproducibility

The Rust port reproduces the Python implementation bit for bit: numpy's
random stream (SeedSequence, PCG64 and every Generator method used, including
the fused multiply-adds numpy's C gets on arm64), numpy's float reductions and
default argsort, CPython's set iteration order, `json` text and format specs.
The parity tests replay values recorded from Python 3.11.15 and numpy 2.4.6 on
macOS arm64 (`crates/mycelic-research/tests/golden/`, generated by the
`tests/golden/gen/*.py` scripts in the commit history).

`final_rerun.sh`, run end to end on a fresh clone (no Python anywhere), takes
4 minutes on an 18-core Mac (the Python run it replaces took 1 h 37 min) and
leaves the committed tree unchanged except for the `runtime_s` fields, the
`frozen_at` stamps in `calibration.json`, the documents' "Generated" lines, the
re-rendered figures and PDFs, and the loss-funnel and ranker-evaluation drift
described below.

Against the committed artifacts, rerun on the committed `calibration.json`:

* **E1-E12** — all 16 result files, 1,371 rows: every field identical except
  `runtime_s` (`e1b_extra.jsonl`'s rows were interleaved by two parallel
  streams in the original run, so only its order differs).
* **`calibration.json`** — `calibrator archs` rewrites it byte for byte.
* **`loss_funnel.jsonl`** — 1,964 of 2,032 rows identical; the other 68 differ
  only in `candidate_conf`/`match_conf`, by 1-2 ULP (relative 2e-15).
* **`ranker_eval_final.json`** — every discovery and kept count and both
  learned AUCs identical; logistic weights within 6e-12 (the Python solved
  them through BLAS/LAPACK); `_finalH`'s hand-confidence AUC is 0.6904 against
  the committed 0.6903.
* **Documents** — `docs/MYCELIC_ENTERPRISE.md`, `docs/MYCELIC_LOSS_ACCOUNTING.md`
  and `docs/mycelic_vnext/*` regenerate byte for byte apart from their
  "Generated" lines.
* **Live measurements** — `score_live` and `live_rank score [rich]` rewrite
  `live_calibration.json` and `live_rank_results{,_rich}.json` byte for byte
  from the committed tasks, keys and answers. The committed task files are
  frozen (the recorded answers refer to their ids) and came from older code
  (`live_tasks.json`/`live_key.json` from `c6ba738`, the `live_rank_*` files
  from `36f0c30`); the builders reproduce what the last Python writes.
* **Paired evidence (`quick_*.jsonl`)** — these were written during
  development, each under the calibration of its day, which the rows do not
  record. Rerun under the calibration they evidently used (the archived v1
  calibration, plus the ranker file of the time where one was active), 36 of
  the 43 files reproduce in every recorded field. `quick_qf_v3.jsonl`: 19 of
  20 rows (the 20th differs in its AP values only, and the Python on macOS
  gives the Rust's value - the platform effect below); `quick_strong_fresh`:
  10 of 15 (its third arm read a scratch calibration file outside the
  repository); `quick_rk_*` (5 files) used a first ranker that was never
  committed.

The loss-funnel, ranker-evaluation and `quick_qf_v3` differences are not the
port's: the committed rows were produced by the Python on Linux, and the Python
on macOS gives exactly the Rust's values (checked row by row). Two platform
effects are behind them — glibc's and Apple's `exp` round some logistic
confidences differently in the last bit, and Linux numpy's SIMD argsort breaks
ties between equal confidences in another order than numpy's portable
introsort. Apart from that one development row's AP, no count or ranking in
any committed file depends on either. The parity tests therefore compare float bits on macOS
arm64 and to 1e-9 elsewhere.

The published numbers in `docs/` and the committed figures and PDFs are the
ones the Python implementation produced (last Python commit `f7e954b`).
Figures and PDFs regenerated by the Rust (`plots`, `make_pdf`) carry the same
data but are not pixel- or byte-identical to the committed ones (different
plotting library and Markdown renderer); the committed files are kept as they
are.

Speed: the whole suite (E1-E12) takes 108 s on an 18-core Mac against 8,368 s
for the Python run that produced the artifacts; on the same Mac, single-threaded,
E1 takes 57 s against the Python's 624 s and E3b 4.2 s against 42 s.

## Three things to know before reading the numbers

1. **Model tiers are a capability axis, not checkpoints.** No GPU or local
   inference was available, so named model classes are *labelled positions* on
   a swept scalar. What is established is a function from operator quality to
   architecture quality. See §5 and §23 (Direct model measurement) of the
   report for exactly what was measured on real models.
2. **Not every decision was made on held-out seeds.** The v1 knobs, the
   learned ranker's weights and its per-architecture adoption were fitted on
   calibration-seed rows (500-502). Several vNext decisions were made on
   evaluation seeds, the development panel (0-4 at 10,000 users, 0-2 at
   50,000): the question budget (`question_frac`), and the rejection of local
   re-extraction, of decoy-weighted ranker selection and of modal link timing.
   The 50,000-user headline (seeds 0-4) therefore includes three development
   seeds; seeds 5-9 at 10,000 users are the panel no development decision
   read. The report's frozen-configuration table names the file and seeds
   behind every value.
3. **The headline result is not favourable to the hierarchy on accuracy.**
   `A2_chunked_ctx`, a chunked long-context pass over every causal-predicate
   record (about 22% of all records), finds more of the hidden problems at
   every scale measured. §1 of the report lists which of the usual arguments
   for a hierarchy hold on this evidence (confidentiality, independent-support
   accuracy, resistance to planted decoys) and which do not (weak-signal
   sensitivity, lineage).
