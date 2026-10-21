#!/usr/bin/env bash
# HTTP throughput of one service implementation under concurrent load: query and write requests per second, and the
# end-to-end apply rate after a write burst.  Same client (crates/mycelic/examples/http_load.rs), same deployment
# shape (fresh database, nats-server with JetStream, rate limiting off, signing on, WARNING logs) for both:
#
#   scripts/bench-http.sh rust   [threads] [query-seconds] [seed] [write-seconds]
#   PYTHON=/path/to/python3.11-with-the-old-requirements scripts/bench-http.sh python [threads] [query-seconds] [seed] [write-seconds]
#
# "python" runs the service from ace3efb (the last commit with Python under mycelic/) in a temporary worktree.
set -euo pipefail

IMPL="${1:?rust or python}"; THREADS="${2:-16}"; SECONDS_PER_PHASE="${3:-10}"; SEED="${4:-2000}"; WRITE_SECONDS="${5:-2}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NATS_BIN="${MYCELIC_NATS_SERVER_BIN:-$(command -v nats-server)}"
(cd "$ROOT" && cargo build --release -q -p mycelic --bin mycelic --example http_load)

WORK="$(mktemp -d "${TMPDIR:-/tmp}/mycelic-bench.XXXXXX")"
PIDS=()
cleanup() {
  for p in "${PIDS[@]:-}"; do { kill -9 "$p" && wait "$p"; } 2>/dev/null || true; done
  git -C "$ROOT" worktree remove --force "$WORK/python-tree" >/dev/null 2>&1 || true
  rm -rf "$WORK"
}
trap cleanup EXIT

port() { python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1])'; }
NATS_PORT="$(port)"; HTTP_PORT="$(port)"; PW="n$(openssl rand -hex 16)"
mkdir -p "$WORK/js"
printf 'port: %s\nhttp_port: -1\njetstream { store_dir: "%s" }\nauthorization { user: "mycelic", password: "%s" }\n' "$NATS_PORT" "$WORK/js" "$PW" > "$WORK/nats.conf"
"$NATS_BIN" -c "$WORK/nats.conf" > "$WORK/nats.log" 2>&1 & PIDS+=($!)

export MYCELIC_HOST=127.0.0.1 MYCELIC_PORT="$HTTP_PORT" MYCELIC_DB_PATH="$WORK/mycelic.db" \
       MYCELIC_NATS_URL="nats://127.0.0.1:$NATS_PORT" MYCELIC_NATS_USER=mycelic MYCELIC_NATS_PASSWORD="$PW" \
       MYCELIC_ADMIN_TOKEN="$(openssl rand -hex 32)" MYCELIC_EVENT_SIGNING_KEY="$(openssl rand -hex 32)" \
       MYCELIC_RULES_FILE="$ROOT/deploy/mycelic/rules.json" MYCELIC_RATE_LIMIT_RPS=0 MYCELIC_LOG_LEVEL=WARNING
case "$IMPL" in
  rust) "$ROOT/target/release/mycelic" serve > "$WORK/service.log" 2>&1 & PIDS+=($!) ;;
  python)
    git -C "$ROOT" worktree add --detach -q "$WORK/python-tree" "${PY_COMMIT:-ace3efb}"
    (cd "$WORK/python-tree" && PYTHONPATH="$WORK/python-tree" exec "${PYTHON:-python3}" -m mycelic serve) > "$WORK/service.log" 2>&1 & PIDS+=($!) ;;
  *) echo "rust or python" >&2; exit 2 ;;
esac
for _ in $(seq 1 240); do curl -sf "http://127.0.0.1:$HTTP_PORT/ready" >/dev/null 2>&1 && break; sleep 0.25; done
echo "== $IMPL: $THREADS clients, ${SECONDS_PER_PHASE}s of queries, ${WRITE_SECONDS}s of writes, $SEED seeded observations" >&2
"$ROOT/target/release/examples/http_load" --url "http://127.0.0.1:$HTTP_PORT" --admin-token "$MYCELIC_ADMIN_TOKEN" \
  --threads "$THREADS" --seconds "$SECONDS_PER_PHASE" --seed "$SEED" --write-seconds "$WRITE_SECONDS"
