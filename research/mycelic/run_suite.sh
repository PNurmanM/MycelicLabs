#!/bin/sh
# Remaining experiment suite, run in sequence. Each appends to its own JSONL,
# so an interrupted suite still leaves usable rows.
cd "$(dirname "$0")/../.." || exit 1   # the repository root, wherever it is checked out
cargo build --release -q -p mycelic-research || exit 1
BIN=target/release/mycelic-research
# GNU timeout where it exists (Linux); macOS has none
T=""; command -v timeout >/dev/null 2>&1 && T="timeout 5400"
for e in e1b e2 e3b e9 e3 e3c e6 e5 e4 e4b e7 e8 e11 e1c e1d e12 e10; do
  echo "=== $e start $(date -u +%Y-%m-%dT%H:%M:%S+00:00) ==="
  $T $BIN experiments $e
  echo "=== $e end $(date -u +%Y-%m-%dT%H:%M:%S+00:00) ==="
done
echo "=== suite complete $(date -u +%Y-%m-%dT%H:%M:%S+00:00) ==="
