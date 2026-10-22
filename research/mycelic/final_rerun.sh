#!/bin/sh
# Final rerun of the whole Mycelic benchmark on the frozen vNext configuration.
#
#   research/mycelic/final_rerun.sh <freeze args for `mycelic-research freeze_vnext`...>
#
# 1. archives every v1 artifact under artifacts/v1/ (the report reads the
#    FIRST row per world, so a rerun has to start from empty files); when the
#    v1 archive already exists, the current rows move to artifacts/previous/;
# 2. freezes the vNext knobs into calibration.json with their provenance;
# 3. reruns every experiment in three parallel streams, then the 100k scale
#    trend, the loss accounting and the candidate dump + ranker evaluation;
# 4. regenerates every table, figure, document and PDF.
# Each stage appends to its own log under logs/final_*.log and leaves a
# marker line so an interrupted run can be resumed by hand.  Every step is the
# Rust binary `mycelic-research` (crates/mycelic-research), built first.
set -u
cd "$(dirname "$0")/../.." || exit 1   # the repository root, wherever it is checked out
cargo build --release -q -p mycelic-research || exit 1
BIN=target/release/mycelic-research
A=research/mycelic/artifacts
L=research/mycelic/logs
mkdir -p "$A/v1" "$L"
now() { date -u +%Y-%m-%dT%H:%M:%S+00:00; }
# GNU timeout where it exists (Linux); macOS has none
T=""; command -v timeout >/dev/null 2>&1 && T="timeout 7200"

# v1/calibration.json is written right after this loop, so it marks an archive
# that already exists (a fresh clone, or a second run): then the current rows
# are set aside in previous/ and v1/ is never touched again.
if [ -f "$A/v1/calibration.json" ]; then ARCHIVE="$A/previous"; else ARCHIVE="$A/v1"; fi
mkdir -p "$ARCHIVE"
echo "=== archive to $ARCHIVE $(now)"
for f in e1_baselines e1b_extra e2_ablations e3_allocation e3b_level_marginal \
         e3c_q_sweep e4_fanin e4b_dept_fanin e5_adversarial e6_frontier e7_shape \
         e8_crosslinks e9_privacy e10_scale_trend e12_provenance e1d_b4_capped \
         loss_funnel; do
  if [ -f "$A/$f.jsonl" ]; then mv "$A/$f.jsonl" "$ARCHIVE/$f.jsonl"; fi
done
if [ ! -f "$A/v1/calibration.json" ]; then
  $BIN final_rerun archive-v1-calibration
fi

echo "=== freeze $(now)"
# the ranker fitted on the calibration seeds under the frozen pipeline
RANKER=${RANKER:-research/mycelic/artifacts/calibration.hyb.json}
$BIN final_rerun install-ranker "$RANKER" || exit 1
$BIN freeze_vnext "$@" || exit 1
# per-architecture ranker adoption, decided on the calibration seeds under
# the frozen knobs (every system keeps whichever ranker is not worse for it)
echo "=== ranker adoption $(now)"
$BIN calibrator archs > "$L/final_ranker_archs.log" 2>&1
tail -12 "$L/final_ranker_archs.log"

run() {  # run <log> <experiment...>
  log=$1; shift
  for e in "$@"; do
    echo "=== $e start $(now)" >> "$L/$log"
    $T $BIN experiments "$e" >> "$L/$log" 2>&1
    rc=$?   # before any command substitution: under bash $(date) would reset $?
    echo "=== $e end $(now) rc=$rc" >> "$L/$log"
  done
}

echo "=== experiments $(now)"
: > "$L/final_s1.log"; : > "$L/final_s2.log"; : > "$L/final_s3.log"
run final_s1.log e1 e11 e1c &
P1=$!
run final_s2.log e1b e2 e3 e3b e3c e6 e1d &
P2=$!
run final_s3.log e5 e4 e4b e7 e8 e9 e12 &
P3=$!
wait $P1 $P2 $P3
echo "=== experiments done $(now)"

echo "=== scale trend + loss accounting $(now)"
: > "$L/final_s4.log"
run final_s4.log e10 &
P4=$!
$BIN loss_accounting 10000 0,1,2,3,4 > "$L/final_loss10k.log" 2>&1 &
P5=$!
$BIN loss_accounting 50000 0,1,2 > "$L/final_loss50k.log" 2>&1 &
P6=$!
wait $P4 $P5 $P6
echo "=== scale trend + loss accounting done $(now)"

echo "=== candidate dump + ranker evaluation $(now)"
# The previous run's evaluation and dumps go first (`calibrator final` removes
# them): if this step fails, the loss document then says the dump is
# incomplete instead of printing stale numbers.  The hierarchy is dumped in its
# frozen configuration, the controls at theirs.
$BIN calibrator final > "$L/final_ranker_eval.log" 2>&1

echo "=== reports $(now)"
$BIN plots > "$L/final_reports.log" 2>&1
$BIN report >> "$L/final_reports.log" 2>&1
$BIN loss_doc >> "$L/final_reports.log" 2>&1
$BIN vnext_docs >> "$L/final_reports.log" 2>&1
$BIN make_pdf docs/MYCELIC_ENTERPRISE.md docs/MYCELIC_ENTERPRISE.pdf >> "$L/final_reports.log" 2>&1
$BIN make_pdf docs/MYCELIC_LOSS_ACCOUNTING.md docs/MYCELIC_LOSS_ACCOUNTING.pdf \
  "Mycelic: where the hidden patterns go" \
  "Loss accounting, gap decomposition and the vNext results on the frozen configuration" >> "$L/final_reports.log" 2>&1
$BIN make_pdf docs/mycelic_vnext/ALL.md docs/MYCELIC_VNEXT.pdf \
  "Mycelic vNext" \
  "Research report, architecture, experiment matrix, loss accounting and adversarial review" >> "$L/final_reports.log" 2>&1
echo "=== FINAL_RERUN_DONE $(now)"
