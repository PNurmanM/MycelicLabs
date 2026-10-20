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
- **MCP and REST.** Claude Code or any MCP client can query and write as an agent; the SDK and agent process
  use the standard library alone.
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
(needs Python 3.11+ and [`nats-server`](https://github.com/nats-io/nats-server/releases) on `PATH`):

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python demo/mycelic_demo.py
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
| [docs/MYCELIC_ENTERPRISE.md](docs/MYCELIC_ENTERPRISE.md) | The enterprise-hierarchy benchmark study; code and raw runs in [research/mycelic/](research/mycelic/) |
| [docs/MYCELIC_LOSS_ACCOUNTING.md](docs/MYCELIC_LOSS_ACCOUNTING.md), [docs/mycelic_vnext/](docs/mycelic_vnext/) | Where hidden patterns are lost, and the next-version research |

## Repository layout

```
mycelic/          the service: API, store, aggregation, lineage, retrieval, MCP, transport, SDK
tests/mycelic/    unit, API, MCP, JetStream and process tests
tests/smoke/      the production smoke test (process or compose driver)
deploy/mycelic/   Dockerfile, docker-compose.yml, nats.conf, Prometheus, rules.json, k8s/
demo/             runnable demos
research/mycelic/ the enterprise-hierarchy benchmark: simulator, experiments, artifacts, logs
docs/             architecture, benchmark and research reports
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).
