# Contributing to Mycelic

## Setup

```bash
git clone https://github.com/PNurmanM/MycelicLabs.git
cd MycelicLabs
python3 -m venv .venv && source .venv/bin/activate     # Python 3.11+
pip install -r requirements.txt
```

## Tests

```bash
python -m pytest tests/mycelic -q                        # unit, API, MCP; JetStream + process tests need nats-server
python tests/smoke/mycelic_smoke.py --driver process     # the production smoke test against local processes
python tests/smoke/mycelic_smoke.py --driver compose     # the deployment test (builds the image)
```

The integration tests run a real `nats-server` when `MYCELIC_NATS_SERVER_BIN` (or `nats-server` on `PATH`) is
available and skip themselves otherwise. Run the compose smoke test before changing anything under
`deploy/mycelic/`.

## Where code goes

- **Agent side** (`mycelic/__init__.py`, `version.py`, `sdk/`, `harness.py`, the smoke test and the demos) runs on
  the standard library alone; `tests/mycelic/test_packaging.py` enforces it. Keep server imports out of them.
- **`mycelic/coordination/`, `mcp_protocol.py`, `textutil.py`** were copied from
  [NeuralGraph](https://github.com/anovruzov/NeuralGraph) at `e7ef73ea`. They are Mycelic's code now: change them
  here as needed; nothing syncs them back.

## Research

`research/mycelic/README.md` has the commands that reproduce every number in
[`docs/MYCELIC_ENTERPRISE.md`](docs/MYCELIC_ENTERPRISE.md). Raw per-run rows belong in
`research/mycelic/artifacts/`; feature dumps are gitignored and regenerate.

## Style

Match the file you are editing. The codebase favours explicit names, docstrings that say *why* rather than
*what*, and comments only where the reason is not obvious from the code.
