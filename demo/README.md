# Demos

All three are binaries of the `mycelic-harness` crate (`cargo build --release` puts them in `target/release/`)
and drive the real stack (nats-server and the Mycelic service). Local processes need `nats-server` on
`PATH` or in `MYCELIC_NATS_SERVER_BIN`; with Docker, `--driver compose` runs the compose deployment instead.
Every number printed or drawn is read from the live deployment.

## End-to-end

```bash
./target/release/mycelic-demo                     # nats-server + service as local processes
./target/release/mycelic-demo --driver compose    # against `docker compose` (builds the image)
./target/release/mycelic-demo --keep              # leave everything running and print how to connect
```

Northwind Robotics' logistics, procurement and sales agents each hold one routine note about the SD-9 servo
drive. Mycelic composes an enterprise-level supply-risk conclusion whose lineage names every contributing
observation, agent, team and layer. Then the service is killed, the broker is killed and the database deleted;
everything comes back from the JetStream event log.

## Strategic synthesis

```bash
./target/release/mycelic-strategic-demo                   # local processes
./target/release/mycelic-strategic-demo --driver compose  # against `docker compose`
```

Two regions independently reach a regional supply-risk conclusion; head office adds two facts of its own; the
enterprise rule `strategic_second_source` fires only once the risk is corroborated in two regions. The demo
then retracts evidence (the strategy is withdrawn), adds new evidence (it returns as a new version), shows
lineage redaction from one agent's point of view, and rebuilds the whole chain from the event log.

## Live demo console

`mycelic-live` drives the real Mycelic stack through nine
acts and draws what happens as a mycelial network in the browser: the agents of Orrery Robotics register,
share some notes and keep others on their own disk, each region reaches a rule conclusion, the enterprise
combines them, a retraction and a recount change the outcome, and finally the service is killed, its database
deleted and everything rebuilt from the event log. Orrery Robotics is a fictional company and all its data is
synthetic. Nothing on screen is hard-coded: every number comes from the run (the service's admin API and the
engine's own actions), and every check shown was run against it. After the last act the engine stops the stack
and the console says so.

```bash
./target/release/mycelic-live                     # start the stack, open http://127.0.0.1:8765/, advance each act from the console
./target/release/mycelic-live --auto --dwell 8    # the acts advance by themselves, 8 s apart
./target/release/mycelic-live --replay            # no stack needed: serve the recorded run in demo/live/trace.json
./target/release/mycelic-live --export page.html  # a standalone replay page with the recorded run embedded
```

If the stack cannot start, the program says why and prints the `--replay`
command; if a live run fails midway, the console stays up with the error and the same command. Presenter notes
open at `?presenter`; `--record PATH` records a new trace headless. The module documentation of
`crates/mycelic-harness/src/live/` describes the trace format and the HTTP endpoints. The console page
(`demo/live/console.html`) and the recorded trace are shared with the original Python version unchanged.
