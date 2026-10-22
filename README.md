# Mycelic

**Shared organizational memory for AI agents.**

Each agent keeps its own private memory on its own disk and shares only what it chooses. Mycelic rolls the
shared notes up through the organization (agent → team → department → subsidiary → region → enterprise),
consolidates them, and composes conclusions no single agent could reach. Every conclusion carries its lineage
back to the observations, agents and teams it rests on, redacted to what the reader may see.

- **Hierarchy-scoped visibility.** An agent sees its own team's notes, notes shared org-wide, and the
  conclusions of the units above it; anything else in a lineage graph is redacted, not dropped.
- **Durable by construction.** Every write lands in SQLite and an outbox in one transaction, is published to
  NATS JetStream with an HMAC signature, and is applied idempotently. Kill the service, kill the broker,
  delete the database: everything is rebuilt from the event log.
- **MCP and REST.** Claude Code or any MCP client can query and write as an agent; the SDK and the agent
  process (`mycelic-sdk`) carry none of the server's dependencies.
- **One binary.** The service is written in Rust: `mycelic serve` is the whole server (API, MCP, outbox publisher,
  JetStream consumer and aggregator).
- **Self-hosted.** Docker Compose for one machine, kustomize manifests for Kubernetes, Prometheus metrics.

Mycelic is the enterprise sibling of [NeuralGraph](https://github.com/anovruzov/NeuralGraph), the local-only
memory for a single assistant. It moved here from that repository with its full history.

## Quick start

Docker Compose (needs Docker 24+ with Compose v2, `openssl`):

```bash
git clone https://github.com/PNurmanM/MycelicLabs.git
cd MycelicLabs
cp deploy/mycelic/.env.example deploy/mycelic/.env   # then fill in the secrets: see DEPLOYMENT.md §1
docker compose -f deploy/mycelic/docker-compose.yml up -d --build
curl -s http://localhost:8080/health
```

Without Docker, the end-to-end demo runs the broker and the service as local processes
(needs a Rust toolchain, 1.88 or newer, and [`nats-server`](https://github.com/nats-io/nats-server/releases) on `PATH`):

```bash
cargo build --release
./target/release/mycelic-demo
```

Connect Claude Code with an agent key:

```bash
claude mcp add --transport http mycelic http://localhost:8080/mcp --header "Authorization: Bearer $MYCELIC_API_KEY"
```

## Documentation

| Document | What it covers |
|---|---|
| [DEPLOYMENT.md](DEPLOYMENT.md) | Compose, production topology, configuration, operations, Kubernetes, troubleshooting |
| [SECURITY.md](SECURITY.md) | What is enforced, how it is tested, and what is not |
| [docs/MYCELIC_ARCHITECTURE.md](docs/MYCELIC_ARCHITECTURE.md) | What actually runs, module by module, and the test behind each claim |
| [demo/README.md](demo/README.md) | The end-to-end, strategic-synthesis and live-console demos |
| [docs/MYCELIC_ENTERPRISE.md](docs/MYCELIC_ENTERPRISE.md) | The enterprise-hierarchy benchmark study; code in [crates/mycelic-research/](crates/mycelic-research/), raw runs in [research/mycelic/](research/mycelic/) |
| [docs/MYCELIC_LOSS_ACCOUNTING.md](docs/MYCELIC_LOSS_ACCOUNTING.md), [docs/mycelic_vnext/](docs/mycelic_vnext/) | Where hidden patterns are lost, and the next-version research |

## Repository layout

```
crates/mycelic/          the service (binary `mycelic`): API, store, aggregation, lineage, retrieval, MCP, transport;
                         tests/ holds the unit, API, MCP, JetStream and golden-fixture tests
crates/mycelic-sdk/      the agent side: HTTP client, agent-local memory, the `mycelic-agent` reference process
crates/mycelic-harness/  process and compose drivers, the production smoke test (`mycelic-smoke`) and the demos
crates/mycelic-research/ the enterprise-hierarchy benchmark (binary `mycelic-research`): simulator, every
                         architecture, experiments E1-E12, statistics, document and figure generators
deploy/mycelic/          Dockerfile, docker-compose.yml, nats.conf, Prometheus, rules.json, k8s/
demo/                    the live console page and its recorded trace, demo outputs
scripts/                 the Python-to-Rust compatibility check and the HTTP benchmark
research/mycelic/        the enterprise-hierarchy benchmark's data: artifacts, live-measurement keys, logs, rerun scripts
docs/                    architecture, benchmark and research reports
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).
