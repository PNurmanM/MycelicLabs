#!/usr/bin/env bash
# Python -> Rust compatibility check: a deployment written by the Python service keeps working on the Rust one.
#
#   PYTHON=/path/to/python3.11-with-the-old-requirements scripts/compat-python-to-rust.sh [report.json]
#
# 1. the Python service (checked out from ace3efb, the last commit with Python under mycelic/, into a temporary
#    worktree) runs against a fresh nats-server: it registers the Northwind agents and the rules, ingests their
#    observations, derives the team and enterprise memories and publishes everything, signed, to JetStream;
# 2. it is stopped and the Rust service starts on the SAME database file, broker, stream, durable consumer and
#    signing key: the Python-issued agent keys authenticate, the answer, its lineage and every active memory are
#    the same, and a new observation is applied and aggregated on top of the Python state;
# 3. the Rust service is stopped, the database deleted and the Rust service started again: it replays the whole
#    stream (Python-signed events first, then its own) and rebuilds exactly the state it had.
#
# Needs: nats-server on PATH (or MYCELIC_NATS_SERVER_BIN), a Python 3.11 with aiohttp, nats-py, prometheus-client,
# numpy and rank-bm25 (the old requirements-runtime.txt), and the Rust binary (built here if missing).
# Exit code 0 and "COMPAT PASS" when every check held.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY_COMMIT="${PY_COMMIT:-ace3efb}"
PYTHON="${PYTHON:-python3}"
NATS_BIN="${MYCELIC_NATS_SERVER_BIN:-$(command -v nats-server || true)}"
REPORT="${1:-}"
[ -n "$NATS_BIN" ] || { echo "nats-server not found (PATH or MYCELIC_NATS_SERVER_BIN)" >&2; exit 2; }
"$PYTHON" -c "import aiohttp, nats, prometheus_client" 2>/dev/null || { echo "PYTHON=$PYTHON lacks the old service requirements" >&2; exit 2; }

if [ ! -x "$ROOT/target/release/mycelic" ]; then
  (cd "$ROOT" && cargo build --release -p mycelic --bin mycelic)
fi
RUST_BIN="$ROOT/target/release/mycelic"

WORK="$(mktemp -d "${TMPDIR:-/tmp}/mycelic-compat.XXXXXX")"
PYTREE="$WORK/python-tree"
PIDS=()
cleanup() {
  for p in "${PIDS[@]:-}"; do { kill -9 "$p" && wait "$p"; } 2>/dev/null || true; done
  git -C "$ROOT" worktree remove --force "$PYTREE" >/dev/null 2>&1 || true
  rm -rf "$WORK"
}
trap cleanup EXIT
git -C "$ROOT" worktree add --detach -q "$PYTREE" "$PY_COMMIT"

free_port() { "$PYTHON" -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1])'; }
NATS_PORT="$(free_port)"; HTTP_PORT="$(free_port)"
NATS_PASSWORD="n$(openssl rand -hex 16)"
mkdir -p "$WORK/js"
cat > "$WORK/nats.conf" <<EOF
port: $NATS_PORT
http_port: -1
jetstream { store_dir: "$WORK/js" }
authorization { user: "mycelic", password: "$NATS_PASSWORD" }
EOF
"$NATS_BIN" -c "$WORK/nats.conf" > "$WORK/nats.log" 2>&1 &
PIDS+=($!)

export MYCELIC_HOST=127.0.0.1 MYCELIC_PORT="$HTTP_PORT" MYCELIC_DB_PATH="$WORK/mycelic.db" \
       MYCELIC_NATS_URL="nats://127.0.0.1:$NATS_PORT" MYCELIC_NATS_USER=mycelic MYCELIC_NATS_PASSWORD="$NATS_PASSWORD" \
       MYCELIC_ADMIN_TOKEN="$(openssl rand -hex 32)" MYCELIC_EVENT_SIGNING_KEY="$(openssl rand -hex 32)" \
       MYCELIC_RULES_FILE="$ROOT/deploy/mycelic/rules.json" MYCELIC_LOG_LEVEL=INFO
URL="http://127.0.0.1:$HTTP_PORT"

wait_ready() {
  for _ in $(seq 1 240); do
    if curl -sf "$URL/ready" >/dev/null 2>&1; then return 0; fi
    sleep 0.25
  done
  echo "service did not become ready" >&2; tail -30 "$1" >&2; return 1
}

start_python() {
  (cd "$PYTREE" && PYTHONPATH="$PYTREE" exec "$PYTHON" -m mycelic serve) > "$WORK/python.log" 2>&1 &
  SERVICE_PID=$!; PIDS+=($SERVICE_PID); wait_ready "$WORK/python.log"
}
start_rust() {
  "$RUST_BIN" serve > "$WORK/rust.$1.log" 2>&1 &
  SERVICE_PID=$!; PIDS+=($SERVICE_PID); wait_ready "$WORK/rust.$1.log"
}
stop_service() { kill -TERM "$SERVICE_PID"; wait "$SERVICE_PID" 2>/dev/null || true; }

# The checks speak HTTP only, through the old Python SDK (urllib), so the same client reads both services.
CHECK="$WORK/check.py"
cat > "$CHECK" <<'PYEOF'
import json, os, sys, time
sys.path.insert(0, os.environ["PYTREE"])
from mycelic.sdk import MycelicClient

url, token, state_file, phase = os.environ["URL"], os.environ["MYCELIC_ADMIN_TOKEN"], os.environ["STATE"], sys.argv[1]
admin = MycelicClient(url, token)
QUERY = "Is there a delivery risk for the RX-4 arm and its SD-9 servo drive this quarter?"

def idle(timeout=120):
    deadline = time.time() + timeout
    while time.time() < deadline:
        c = admin.status()["checks"]
        if (c["publisher"]["outbox_pending"] or 0) + (c["consumer"].get("pending") or 0) == 0 and c["consumer"]["replaying_to_seq"] is None \
                and c["transport"].get("connected"):
            time.sleep(0.5)
            c = admin.status()["checks"]
            if (c["publisher"]["outbox_pending"] or 0) + (c["consumer"].get("pending") or 0) == 0:
                return admin.status()
        time.sleep(0.25)
    raise SystemExit(f"{phase}: not idle")

def snapshot():
    rows = admin._request("GET", "/memories", params={"scope": "northwind", "status": "active", "limit": 500})["memories"]
    every = {s: admin._request("GET", "/memories", params={"scope": "northwind", "status": s, "limit": 500})["memories"]
             for s in ("active", "superseded", "retracted")}
    return {"active": {m["memory_id"]: [m["layer"], m["support"], m["confidence"], m["text"]] for m in rows},
            "all": sorted(m["memory_id"] for ms in every.values() for m in ms)}

def answer(key):
    res = MycelicClient(url, key).query(QUERY, scope="northwind", min_layer="enterprise")
    a = res["answer"]
    return {"memory_id": a["memory_id"], "confidence": a["confidence"], "support": a["support"], "text": a["text"],
            "roots": sorted(res["lineage"]["roots"]), "layers": res["lineage"]["layers"],
            "reconstructable": res["lineage"]["evidence"]["reconstructable"]}

checks = []
def check(ok, text):
    checks.append({"ok": bool(ok), "text": text})
    print(("  ok   " if ok else "  FAIL ") + text, flush=True)

if phase == "python":
    keys = {}
    for agent, dept, team in (("log-1", "ops", "logistics"), ("log-2", "ops", "logistics"), ("proc-1", "ops", "procurement"),
                              ("sales-1", "commercial", "field-sales"), ("sales-2", "commercial", "field-sales")):
        keys[agent] = admin.register_agent(agent_id=agent, enterprise="northwind", region="emea", subsidiary="nw-gmbh",
                                           department=dept, team=team)["api_key"]
    obs = [("log-1", "Port of Rotterdam terminal 3 strike announced for weeks 41-43; SD-9 servo drive shipments route through it.", "supply:sd-9/transport", "transport_disruption", 0.9),
           ("log-2", "Carrier ETA for the SD-9 servo drive container slipped by 12 days.", "supply:sd-9/transport", "transport_disruption", 0.7),
           ("proc-1", "Kessler Antriebe confirms roughly two weeks of SD-9 inventory left and no second source qualified.", "supply:sd-9/supplier", "supplier_buffer_low", 0.85),
           ("sales-1", "Helios Automation committed to 40 RX-4 arms for November; every RX-4 uses an SD-9 drive.", "supply:sd-9/demand", "demand_commitment", 0.8)]
    for agent, text, topic, slot, conf in obs:
        MycelicClient(url, keys[agent]).remember(text, topic=topic, slot=slot, entity="sd-9", confidence=conf, idempotency_key=f"{agent}-1",
                                                 local_ref=f"{agent}-note")
    st = idle()
    ans = answer(keys["sales-2"])
    check(st["stats"]["memories_by_layer"]["enterprise"] == 1, f"python: the enterprise conclusion exists ({st['stats']['memories_by_layer']})")
    state = {"keys": keys, "answer": ans, "snapshot": snapshot(), "stream": st["checks"]["transport"]["stream_messages"],
             "last_applied_seq": st["stats"]["last_applied_seq"], "service": admin.health().get("version"),
             "lineage_admin": admin.lineage(ans["memory_id"])}
    json.dump(state, open(state_file, "w"))
    print(json.dumps({"python_stream_messages": state["stream"], "answer": ans["memory_id"]}))

elif phase == "rust-takeover":
    state = json.load(open(state_file))
    st = idle()
    check(st["checks"]["transport"]["connected"] and st["checks"]["consumer"]["replaying_to_seq"] is None,
          "rust opened the python database and bound the python durable consumer without a replay")
    check(st["stats"]["last_applied_seq"] == state["last_applied_seq"], f"same last applied sequence ({st['stats']['last_applied_seq']})")
    who = MycelicClient(url, state["keys"]["sales-2"]).whoami()
    check(who["id"] == "sales-2", "a python-issued agent key authenticates against the rust service")
    ans = answer(state["keys"]["sales-2"])
    check(ans == state["answer"], "same answer, confidence, support, text and lineage roots as python gave")
    check(snapshot() == state["snapshot"], f"same {len(state['snapshot']['active'])} active memories and the same history")
    lin = admin.lineage(ans["memory_id"])
    strip = lambda g: {k: v for k, v in g.items() if k not in ("timeline",)}
    check(strip(lin) == strip(state["lineage_admin"]), "the administrator's lineage graph is identical (nodes, edges, support, fragility, evidence)")
    MycelicClient(url, state["keys"]["log-1"]).remember("Customs broker warns of a two-week backlog for SD-9 containers arriving via Rotterdam.",
                                                         topic="supply:sd-9/transport", slot="transport_disruption", entity="sd-9",
                                                         confidence=0.6, idempotency_key="log-1-2")
    st = idle()
    team = [m for m in admin.list_memories(scope="northwind", layer="team") if m["topic"] == "supply:sd-9/transport"][0]
    check(team["metadata"]["parent_count"] == 3 and team["metadata"]["version_of"] in state["snapshot"]["all"],
          "a new observation through rust supersedes the python-derived team memory (3 parents, version_of the python one)")
    check(st["checks"]["transport"]["stream_messages"] > state["stream"], "rust appended its events to the stream python wrote")
    state["rust_snapshot"] = snapshot()
    state["rust_answer"] = answer(state["keys"]["sales-2"])
    state["rust_stream"] = st["checks"]["transport"]["stream_messages"]
    json.dump(state, open(state_file, "w"))

elif phase == "rust-rebuild":
    state = json.load(open(state_file))
    st = idle(180)
    check(st["checks"]["consumer"]["replaying_to_seq"] is None, "the rebuild from the stream finished")
    check(st["checks"]["transport"]["stream_messages"] >= state["rust_stream"], "nothing was lost from the stream")
    check(snapshot() == state["rust_snapshot"], "replaying python-signed and rust-signed events rebuilt the exact state (active set and history)")
    check(answer(state["keys"]["sales-2"]) == state["rust_answer"], "same answer and lineage after the rebuild; python-issued keys still work")
    check(len(admin.list_agents("northwind")) == 5 and any(r["rule_id"] == "component_supply_risk" for r in admin.list_rules()),
          "agents and rules came back from the log")

out = os.environ.get("CHECKS")
if out:
    prev = json.load(open(out)) if os.path.exists(out) else []
    json.dump(prev + [{"phase": phase, **c} for c in checks], open(out, "w"))
sys.exit(0 if all(c["ok"] for c in checks) else 1)
PYEOF

export PYTREE URL STATE="$WORK/state.json" CHECKS="$WORK/checks.json"
echo "== 1. python service (mycelic @ $PY_COMMIT) writes the database and the stream"
start_python
"$PYTHON" "$CHECK" python
stop_service

echo "== 2. rust service takes over the same database, broker, stream and consumer"
start_rust takeover
"$PYTHON" "$CHECK" rust-takeover
stop_service

echo "== 3. rust service loses its database and rebuilds it from the stream"
rm -f "$WORK"/mycelic.db*
start_rust rebuild
grep -q "replaying the full log" "$WORK/rust.rebuild.log" && echo "  ok   rust logged the full replay of the stream"
"$PYTHON" "$CHECK" rust-rebuild
stop_service

if [ -n "$REPORT" ]; then
  "$PYTHON" - "$CHECKS" "$REPORT" "$PY_COMMIT" <<'PYEOF'
import json, sys
checks = json.load(open(sys.argv[1]))
json.dump({"python_commit": sys.argv[3], "checks": checks, "passed": sum(c["ok"] for c in checks), "total": len(checks)},
          open(sys.argv[2], "w"), indent=1)
PYEOF
fi
echo "COMPAT PASS: $("$PYTHON" -c "import json,sys; c=json.load(open(sys.argv[1])); print(f'{sum(x[\"ok\"] for x in c)}/{len(c)} checks')" "$CHECKS")"
