"""Generate the golden fixtures the Rust port is tested against, from the Python implementation.

    python crates/mycelic/tests/golden/generate_from_python.py        # from the repository root, Python mycelic importable

Every deterministic piece of the Python service is exercised here and its output written as JSON next to this
script.  The Rust tests (``crates/mycelic/tests/golden_*.rs``) read the same inputs, run the Rust code and assert
equality.  Clock and id generator are pinned (``FIXED_NOW``, a counter for ``new_id``) so whole service runs,
event ids and timestamps included, are reproducible; API keys stay random and are left out of the comparison.

This script ran against the Python implementation at ace3efb (the last commit with Python under ``mycelic/``);
it is kept in the history next to the fixtures it wrote, and removed together with the Python code.
"""
from __future__ import annotations

import asyncio
import json
import math
import os
import random
import sqlite3
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import mycelic.aggregation as aggregation_mod  # noqa: E402
import mycelic.models as models_mod  # noqa: E402
import mycelic.service as service_mod  # noqa: E402
import mycelic.store as store_mod  # noqa: E402
from mycelic import auth, hierarchy, textutil  # noqa: E402
from mycelic.auth import Principal, generate_api_key, hash_key, memory_visible, parse_agent_id  # noqa: E402
from mycelic.config import ConfigError, Settings  # noqa: E402
from mycelic.coordination.contracts import ClaimEnvelope, PolicyStatus  # noqa: E402
from mycelic.coordination.core import LineageAnalyzer, RuleBasedSynthesizer, to_jsonable  # noqa: E402
from mycelic.hierarchy import HierarchyError  # noqa: E402
from mycelic.metrics import Metrics  # noqa: E402
from mycelic.models import Agent, Memory, content_hash, derived_memory_id  # noqa: E402
from mycelic.retrieval import BM25, Retriever  # noqa: E402
from mycelic.service import Forbidden, MycelicService, NotFound, ValidationError  # noqa: E402
from mycelic.store import MycelicStore  # noqa: E402
from mycelic.transport import InProcessTransport  # noqa: E402

FIXED_NOW = "2026-10-01T12:00:00+00:00"
ADMIN_TOKEN = "test-admin-token-0123456789abcdef0123456789"
GENERATED_BY = "crates/mycelic/tests/golden/generate_from_python.py against the Python implementation at ace3efb"


def write(name: str, data: Any) -> None:
    data = {"_generated_by": GENERATED_BY, **data}
    (OUT / name).write_text(json.dumps(data, ensure_ascii=False, indent=1, sort_keys=False) + "\n", encoding="utf-8")
    print(f"wrote {name}")


# ---------------------------------------------------------------------------------------------------- determinism
class CounterIds:
    def __init__(self) -> None:
        self.n = 0

    def __call__(self, prefix: str) -> str:
        self.n += 1
        return f"{prefix}_{self.n:020d}"


def pin(ids: CounterIds) -> None:
    fixed = lambda: FIXED_NOW  # noqa: E731
    for mod in (models_mod, store_mod, service_mod, aggregation_mod):
        mod.now_iso = fixed
    models_mod.new_id = ids
    service_mod.new_id = ids


# ---------------------------------------------------------------------------------------------------- textutil
TOKEN_TEXTS = [
    "", "   ", "The port of Rotterdam terminal 3 strike announced for weeks 41-43.",
    "Zürich naïve café façade Ærø straße İstanbul ﬁnance Ⅻ x² ＡＢＣ résumé",
    "rock'n'roll o'neill's children's it's its don't can't let's 'quoted' --dash-- -lead trail-",
    "1990s 3d e-mail re-entry a1 x ab abc 42 7 000 0 12-34 'tis",
    "Running lives moved cities hoping stopped planned fixing hopping tries dies agreed played buses boxes buzzes wishes",
    "Kessler Antriebe confirms roughly two weeks of SD-9 inventory left; no second source qualified.",
    "Helios Automation committed to 40 RX-4 arms for November; every RX-4 uses an SD-9 drive.",
    "yeah thanks sounds good great awesome okay well still the and for are but not",
    "ÀÉÎÕÜ àéîõü ÇÑ çñ Œuvre œuvre Ðð Þþ ß ı Ĳ ĳ ŉ ǅ",
    "tab\tnew\nline\r\nmixed\u00a0nbsp\u2003emspace\u200bzw",
    "UPPER lower MiXeD caseS CASES Cases",
    "agreed fed bled sled need speed hoped hated rated fizzed fished",
    "agrees sees glasses misses classes passes kisses",
    "caresses ponies ties caress cats running sing ring ding king",
    "ssss's x's ab's abc's '' ''' -'- '-a-' a'-'b",
    "co-operate co--operate 'n' n' 'n -- '' a-b-c-d",
    "ÅNGSTRÖM Kelvin K Ω ohm µ micro ﬀ ﬃ ﬄ ﬅ ﬆ",
    "日本語 中文 한국어 العربية עברית ελληνικά русский",
    "Σίσυφος ΣΑΣ σας ΟΔΟΣ",
    "emoji 🚀 rocket 👍🏽 thumbs ❤️ heart",
    "supply:sd-9/transport supply:sd-9/supplier strategy:supply-risk",
    "x'''y ab'''cd abc'-'def 'abc' \"abc\" «abc» ‘abc’ “abc”",
]
STEM_WORDS = ["lives", "moved", "running", "cities", "hoping", "hopped", "planned", "fixing", "tries", "dies", "agreed",
              "played", "buses", "boxes", "buzzes", "wishes", "caresses", "ponies", "ties", "caress", "cats", "sing",
              "ring", "dinged", "kinged", "rated", "hated", "fizzed", "fished", "bled", "sled", "need", "speed", "seed",
              "feed", "fed", "red", "bed", "shed", "sped", "ies", "dies", "lies", "xies", "abed", "aed", "ing", "ning",
              "sting", "string", "bring", "ss", "s", "es", "ses", "xes", "zes", "hes", "ches", "shes", "gases", "pass",
              "class", "glass", "a", "ab", "abc", "abcd", "abcde", "conflated", "troubled", "sized", "hoping", "filing",
              "falling", "hissing", "tanned", "failing", "filed", "hopping", "controlled", "rolled", "bowled", "boxed",
              "taxed", "flying", "played", "stayed", "fixed", "mixed", "waxed", "plying"]


def gen_textutil() -> None:
    rnd = random.Random(7)
    vocab = ("port strike Rotterdam terminal carrier ETA supplier stock servo drive order customer November shipment "
             "delay forecast region EMEA APAC warehouse inventory quality defect recall audit invoice contract price "
             "the and for running moved cities it's o'neill's re-entry 1990s 42 Zürich naïve 'quoted' --x-- a b").split()
    texts = list(TOKEN_TEXTS)
    for _ in range(60):
        texts.append(" ".join(rnd.choice(vocab) for _ in range(rnd.randint(1, 18))) + rnd.choice(["", ".", "!", "?", ";"]))
    cases = [{"text": t, "tokens": textutil.tokenize(t), "tokens_nostem": textutil.tokenize(t, do_stem=False),
              "tokens_nonum": textutil.tokenize(t, keep_numbers=False), "fold": textutil.fold(t),
              "normalize_ws": textutil.normalize_ws(t)} for t in texts]
    write("textutil.json", {"cases": cases, "stem": [[w, textutil.stem(w)] for w in STEM_WORDS]})


# ---------------------------------------------------------------------------------------------------- bm25 + retrieval
WORDS = ("port strike rotterdam terminal carrier eta supplier stock servo drive order customer november shipment "
         "delay forecast region emea apac warehouse inventory quality defect recall audit invoice contract price").split()


def gen_bm25() -> None:
    rnd = random.Random(11)
    cases = []
    for _ in range(40):
        docs = [[rnd.choice(WORDS[: rnd.randint(3, len(WORDS))]) for _ in range(rnd.randint(0, 20))] or ["_empty_"]
                for _ in range(rnd.randint(1, 30))]
        query = [rnd.choice(WORDS) for _ in range(rnd.randint(0, 6))]
        k1, b = rnd.choice([(1.5, 0.75), (1.2, 0.5), (2.0, 1.0)])
        cases.append({"docs": docs, "query": query, "k1": k1, "b": b, "scores": BM25(docs, k1=k1, b=b).get_scores(query)})
    write("bm25.json", {"cases": cases})


LAYERS = hierarchy.LAYERS


def synthetic_memories(n: int, seed: int) -> list[Memory]:
    """Memories at every layer of two organizations, with org-visible notes, promotions and repeated topics."""
    rnd = random.Random(seed)
    teams = [f"acme/emea/gmbh/ops/t{i}" for i in range(6)] + [f"acme/emea/gmbh/sales/s{i}" for i in range(3)] + \
            [f"acme/apac/kk/ops/k{i}" for i in range(3)]
    out = []
    for i in range(n):
        layer = rnd.choice(["agent"] * 6 + ["team", "department", "subsidiary", "region", "enterprise"])
        team = rnd.choice(teams)
        if layer == "agent":
            scope = f"{team}/a{rnd.randint(0, 4)}"
        else:
            scope = hierarchy.unit_at_layer(team + "/x", layer)
        meta: dict[str, Any] = {}
        if layer != "agent" and out and rnd.random() < 0.2:
            meta["promoted_from"] = rnd.choice(out).memory_id
        text = " ".join(rnd.choice(WORDS) for _ in range(rnd.randint(0, 14)))
        out.append(Memory(
            memory_id=f"m{i:05d}", org_id="acme", layer=layer, scope=scope, text=text,
            topic=rnd.choice([None, "supply:sd-9", "supply:sd-3", "quality"]), slot=rnd.choice([None, "transport_disruption", "demand"]),
            entity=rnd.choice([None, "sd-9", "sd-3"]), kind=rnd.choice(["observation", "risk", "fact"]),
            confidence=rnd.choice([0.5, 0.6, 0.65, 0.7, 0.8, 0.85, 0.9, 0.95]), support=1, independent_teams=1,
            producer_id=f"a{i}", operator="agent_observation" if layer == "agent" else "topic_consolidation", rule_id=None,
            event_id=None, visibility="org" if rnd.random() < 0.25 else "team",
            created_at=f"2026-10-0{rnd.randint(1, 9)}T1{rnd.randint(0, 9)}:00:00+00:00", metadata=meta))
    return out


class RowsStore:
    revision = 1

    def __init__(self, rows: list[Memory]) -> None:
        self.rows = rows

    def visible_rows(self, *a: Any, **k: Any) -> list[Memory]:
        return self.rows


PRINCIPALS = [
    {"name": "admin", "agent_path": None, "team_path": None},
    {"name": "ops-t0-a0", "agent_path": "acme/emea/gmbh/ops/t0/a0", "team_path": "acme/emea/gmbh/ops/t0"},
    {"name": "ops-t3-a2", "agent_path": "acme/emea/gmbh/ops/t3/a2", "team_path": "acme/emea/gmbh/ops/t3"},
    {"name": "sales-s1-a1", "agent_path": "acme/emea/gmbh/sales/s1/a1", "team_path": "acme/emea/gmbh/sales/s1"},
    {"name": "apac-k2-a4", "agent_path": "acme/apac/kk/ops/k2/a4", "team_path": "acme/apac/kk/ops/k2"},
]


def gen_retrieval() -> None:
    rows = synthetic_memories(400, 3)
    retriever = Retriever(RowsStore(rows))  # type: ignore[arg-type]
    queries = [
        {"query": "port strike servo"}, {"query": "supplier stock november", "k": 25}, {"query": "carrier eta delay emea"},
        {"query": "quality recall audit", "min_layer": "team"}, {"query": "warehouse inventory", "scope": "acme/emea"},
        {"query": "the and for"}, {"query": ""}, {"query": "port port port strike", "k": 100},
        {"query": "invoice contract", "topic": "quality"}, {"query": "price", "entity": "sd-9"},
        {"query": "region apac forecast", "scope": "acme/apac/kk", "min_layer": "subsidiary"},
        {"query": "shipment", "scope": "acme/emea/gmbh/ops/t0", "k": 50}, {"query": "zzz nothing matches"},
        {"query": "drive", "min_layer": "enterprise", "k": 3}, {"query": "", "min_layer": "region", "k": 7},
        {"query": "Ports STRIKING delays", "k": 12},
    ]
    cases = []
    for p in PRINCIPALS:
        vis = (lambda m: True) if p["agent_path"] is None else \
            (lambda m, p=p: memory_visible(m, agent_path=p["agent_path"], team_path=p["team_path"]))
        for q in queries:
            hits = retriever.search("acme", q["query"], visible=vis, scope=q.get("scope"), min_layer=q.get("min_layer", "agent"),
                                    k=q.get("k", 10), topic=q.get("topic"), entity=q.get("entity"))
            cases.append({"principal": p["name"], "params": q,
                          "hits": [{"memory_id": h.memory.memory_id, **{k: v for k, v in h.to_dict().items() if k != "memory"}}
                                   for h in hits]})
    write("retrieval.json", {"memories": [m.to_dict() for m in rows], "principals": PRINCIPALS, "cases": cases})


# ---------------------------------------------------------------------------------------------------- visibility
def gen_visibility() -> None:
    rows = synthetic_memories(150, 5)
    rows += [Memory(memory_id="other-org", org_id="globex", layer="enterprise", scope="globex", text="x", topic=None, slot=None,
                    entity=None, kind="fact", confidence=0.5, support=1, independent_teams=1, producer_id="g1",
                    operator="topic_consolidation", rule_id=None, event_id=None, visibility="org", created_at=FIXED_NOW)]
    principals = []
    for p in PRINCIPALS:
        if p["agent_path"] is None:
            pr = Principal.admin()
        else:
            ag = Agent(agent_id=p["agent_path"].rsplit("/", 1)[1], org_id="acme", display_name="x", path=p["agent_path"],
                       scopes=["memory:read"], key_prefix="mk_")
            pr = Principal.for_agent(ag)
        principals.append((p, pr))
    matrix = {p["name"]: [mid for mid in (m.memory_id for m in rows if pr.can_read(m))] for p, pr in principals}
    scopes = ["acme", "acme/emea", "acme/emea/gmbh", "acme/emea/gmbh/ops", "acme/emea/gmbh/ops/t0", "acme/emea/gmbh/ops/t0/a0",
              "acme/emea/gmbh/ops/t0/a9", "acme/emea/gmbh/ops/t1", "acme/apac", "acme/emea/gmbh/sales/s1/a1", "globex",
              "acme/emea/gmbh/ops/t0x", "acme/emea/gmbh/opsx"]
    query_scope = {p["name"]: {s: pr.can_query_scope(s) for s in scopes} for p, pr in principals}
    owns = {p["name"]: [m.memory_id for m in rows if pr.owns(m)] for p, pr in principals}
    # the SQL twin: the same memories in a real store, read through visible_rows
    tmp = tempfile.mkdtemp()
    store = MycelicStore(Path(tmp) / "v.db")
    for m in rows:
        cols = m.to_dict()
        store._conn.execute(
            "INSERT INTO memories(memory_id, org_id, layer, scope, text, topic, slot, entity, kind, confidence, support, "
            "independent_teams, producer_id, operator, rule_id, agg_key, event_id, visibility, status, superseded_by, created_at, "
            "applied_at, source_event_ids, local_ref, metadata) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (m.memory_id, m.org_id, m.layer, m.scope, m.text, m.topic, m.slot, m.entity, m.kind, m.confidence, m.support,
             m.independent_teams, m.producer_id, m.operator, m.rule_id, None, m.event_id, m.visibility, m.status, None,
             m.created_at, None, "[]", None, json.dumps(cols["metadata"])))
    sql = []
    for p, pr in principals:
        for scope in (None, "acme/emea", "acme/emea/gmbh/ops/t0", "acme/apac/kk"):
            for min_layer in (0, 2):
                got = store.visible_rows("acme", agent_path=pr.path, team_path=pr.team_path, scope=scope, min_layer_index=min_layer)
                sql.append({"principal": p["name"], "scope": scope, "min_layer_index": min_layer, "ids": [m.memory_id for m in got]})
    write("visibility.json", {"memories": [m.to_dict() for m in rows], "principals": PRINCIPALS, "can_read": matrix,
                              "can_query_scope": query_scope, "owns": owns, "visible_rows": sql})


# ---------------------------------------------------------------------------------------------------- ids, keys, hierarchy
def gen_ids() -> None:
    rnd = random.Random(13)
    alphabet = "abcdefghijklmnopqrstuvwxyz0123456789_-:./|é漢 "
    rs = lambda n: "".join(rnd.choice(alphabet) for _ in range(n))  # noqa: E731
    derived = []
    for _ in range(80):
        args = {"operator": rnd.choice(["topic_consolidation", "slot_composition"]), "scope": rs(rnd.randint(1, 30)),
                "key": rs(rnd.randint(0, 20)), "parent_ids": [rs(rnd.randint(1, 26)) for _ in range(rnd.randint(0, 6))]}
        derived.append({**args, "id": derived_memory_id(**args)})
    hashes = []
    for _ in range(60):
        parts = [rs(rnd.randint(0, 25)) for _ in range(rnd.randint(1, 4))]
        hashes.append({"parts": parts, "hash": content_hash(*parts)})
    keys = []
    for agent in ("log-1", "a", "agent-7", "x" * 64):
        key, key_hash, prefix = generate_api_key(agent)
        keys.append({"agent_id": agent, "key": key, "hash": key_hash, "prefix": prefix, "parsed": parse_agent_id(key),
                     "prefix_len": len(prefix), "key_len": len(key)})
    parse = {k: parse_agent_id(k) for k in ("nope", "mk_", "mk_.x", "mk_a.b", "mk_abc", "mk_abc.", "mk_a.b.c", "Mk_a.b", "mk_é.x", "")}
    hk = {k: hash_key(k) for k in ("mk_a.b", "é", "", "mk_log-1.secret")}
    idem = []
    for agent, key in (("log-1", "note-1"), ("proc-1", "call-1"), ("a", "é漢"), ("sales-2", "x" * 200)):
        idem.append({"agent": agent, "key": key, "memory_id": f"mem_{content_hash(agent, key)[:22]}",
                     "event_id": f"evt_{content_hash(agent, 'event', key)[:22]}",
                     "event_memory_id": f"mem_{content_hash(agent, 'event-memory', key)[:22]}"})
    derived_event = {mid: f"evt_d{content_hash(mid)[:22]}" for mid in ("mem_dabc", "mem_d" + "0" * 22, "x")}
    write("ids.json", {"derived_memory_id": derived, "content_hash": hashes, "api_keys": keys, "parse_agent_id": parse,
                       "hash_key": hk, "idempotency": idem, "derived_event_id": derived_event,
                       "textutil_content_hash": [{"parts": c["parts"], "hash": textutil.content_hash(*c["parts"])} for c in hashes[:10]]})


def catch(fn: Any) -> Any:
    try:
        return {"ok": fn()}
    except HierarchyError as exc:
        return {"error": str(exc)}


def gen_hierarchy() -> None:
    paths = ["acme", "acme/eu", "acme/eu/de", "acme/eu/de/ops", "acme/eu/de/ops/log", "acme/eu/de/ops/log/a1",
             "acme/eu/de/ops/log/a1/x", "", "Acme", "acme//eu", "acme/eu/", "/acme", "a" * 64, "a" * 65, "a_b-c/d",
             "-a", "_a", "acme/EU", "acme/e u", "acme/é", "0/1/2/3/4/5", "acme/eu/de/ops/log/a1/x/y"]
    out = []
    for p in paths:
        row: dict[str, Any] = {"path": p}
        row["split"] = catch(lambda: hierarchy.split_path(p))
        row["layer_of_path"] = catch(lambda: hierarchy.layer_of_path(p))
        row["parent_path"] = catch(lambda: hierarchy.parent_path(p))
        row["ancestors"] = catch(lambda: hierarchy.ancestors(p))
        row["ancestors_strict"] = catch(lambda: hierarchy.ancestors(p, include_self=False))
        row["org_of"] = catch(lambda: hierarchy.org_of(p))
        row["unit_at_layer"] = {layer: catch(lambda layer=layer: hierarchy.unit_at_layer(p, layer)) for layer in LAYERS}
        row["agent_path"] = catch(lambda: hierarchy.AgentPath.parse(p).to_dict())
        out.append(row)
    pairs = [("acme/eu/de/ops/log/a1", "acme/eu"), ("acme/eu/de", "acme/eu"), ("acme/eu", "acme/eu"), ("acme/us/x", "acme/eu"),
             ("acme/eux/x", "acme/eu"), ("acme/eu/de/ops/log/a1", "acme")]
    child = [{"descendant": d, "unit": u, "result": catch(lambda d=d, u=u: hierarchy.child_unit_of(d, u)),
              "is_ancestor_or_self": hierarchy.is_ancestor_or_self(u, d)} for d, u in pairs]
    parts = [{"enterprise": "acme", "agent_id": "a1"}, {"enterprise": "acme", "region": "eu", "team": "log", "agent_id": "a1"},
             {"enterprise": "Acme Inc", "agent_id": "a1"}, {"enterprise": "acme", "agent_id": "A1"},
             {"enterprise": "acme", "region": "", "agent_id": "a1"}, {"enterprise": "acme", "agent_id": ""},
             {"enterprise": "", "agent_id": "a"}, {"enterprise": "acme", "department": "x/y", "agent_id": "a"}]
    from_parts = [{"parts": pp, "result": catch(lambda pp=pp: hierarchy.AgentPath.from_parts(**pp).to_dict())} for pp in parts]
    layer_index = {name: catch(lambda name=name: hierarchy.layer_index(name)) for name in (*LAYERS, "Team", "org", "")}
    write("hierarchy.json", {"paths": out, "child_unit_of": child, "from_parts": from_parts, "layer_index": layer_index})


def gen_iso() -> None:
    inputs = ["2026-10-01", "2026-10-01T10:00", "2026-10-01T10:00:00", "2026-10-01T10:00:00Z", "2026-10-01T10:00:00+02:00",
              "2026-10-01T10:00:00.123456+02:00", "2026-10-01T10:00:00.1234567", "2026-10-01 10:00:00", "2026-10-01T10",
              "20261001", "20261001T100000", "2026-W40-1", "2026W401", "2026-10-01T10:00:00-0530", "2026-10-01T10:00:00+05",
              "2026-10-01T10:00:00+05:30:15", "2026-10-01T10:00:00.5Z", "2026-02-30", "2026-13-01", "2026-10-01T25:00:00",
              "not a date", "2026-10-01T10:00:00+24:00", "2026-1-01", "10:00", "2026-10-01T10:00:60", "2026-10-01T23:59:59.999999",
              "2026-10-01T10:00:00,5", "2026-10-01X10:00:00", "2026-10-01é10:00:00", "2026-10-01T10:00:00 ", " 2026-10-01",
              "2026-10-01T1000", "2026-10-01T100000", "2026-10-01T10:00:00.", "2026-10-01T10:00:00+00:00:00.5",
              "0001-01-01T00:00:00Z", "9999-12-31T23:59:59Z", "2024-02-29", "2023-02-29", "2026-10-01T10:00:00z",
              "2026-10-01T10:00:00+0200", "2026-10-01T10:00:00-00:00", "2026-10-01T10:00Z", "2026-10-01T10Z",
              "2026-W53-7", "2020-W53-7", "2026-W00-1", "2026-W01-8", "2026-10-01T24:00:00"]
    cases = []
    for s in inputs:
        try:
            cases.append({"input": s, "output": service_mod._iso({"k": s}, "k")})
        except ValidationError as exc:
            cases.append({"input": s, "error": str(exc)})
    write("iso.json", {"cases": cases})


def gen_config() -> None:
    rules = str(ROOT / "deploy" / "mycelic" / "rules.json")
    envs = [
        {}, {"MYCELIC_HOST": "127.0.0.1"}, {"MYCELIC_HOST": "127.0.0.1", "MYCELIC_PORT": "abc"},
        {"MYCELIC_HOST": "127.0.0.1", "MYCELIC_PORT": "0"}, {"MYCELIC_ADMIN_TOKEN": "short"},
        {"MYCELIC_ADMIN_TOKEN": "x" * 32}, {"MYCELIC_ADMIN_TOKEN": "change-me-" + "x" * 32},
        {"MYCELIC_HOST": "localhost", "MYCELIC_NATS_URL": "http://nats:4222"},
        {"MYCELIC_HOST": "::1", "MYCELIC_NATS_URL": "nats://user:pw@nats:4222"},
        {"MYCELIC_HOST": "127.0.0.1", "MYCELIC_NATS_URL": ""},
        {"MYCELIC_HOST": "127.0.0.1", "MYCELIC_EVENT_SIGNING_KEY": "short"},
        {"MYCELIC_HOST": "127.0.0.1", "MYCELIC_EVENT_SIGNING_KEY": "ExAmPle" + "k" * 30},
        {"MYCELIC_HOST": "127.0.0.1", "MYCELIC_NATS_PASSWORD": "placeholder"},
        {"MYCELIC_HOST": "127.0.0.1", "MYCELIC_METRICS_TOKEN": "REPLACE_ME"},
        {"MYCELIC_HOST": "127.0.0.1", "MYCELIC_TLS_CERT_FILE": rules},
        {"MYCELIC_HOST": "127.0.0.1", "MYCELIC_TLS_CERT_FILE": rules, "MYCELIC_TLS_KEY_FILE": "/nonexistent/key.pem"},
        {"MYCELIC_HOST": "127.0.0.1", "MYCELIC_RULES_FILE": rules},
        {"MYCELIC_HOST": "127.0.0.1", "MYCELIC_RULES_FILE": "/nonexistent/rules.json"},
        {"MYCELIC_HOST": "127.0.0.1", "MYCELIC_PUBLISH_INTERVAL_SECONDS": "0.001"},
        {"MYCELIC_HOST": "127.0.0.1", "MYCELIC_PUBLISH_INTERVAL_SECONDS": "x"},
        {"MYCELIC_HOST": "127.0.0.1", "MYCELIC_RATE_LIMIT_RPS": "-1"},
        {"MYCELIC_HOST": "127.0.0.1", "MYCELIC_RATE_LIMIT_RPS": "2.5", "MYCELIC_RATE_LIMIT_BURST": "7"},
        {"MYCELIC_HOST": "127.0.0.1", "MYCELIC_MAX_BODY_BYTES": "100"},
        {"MYCELIC_HOST": "127.0.0.1", "MYCELIC_TRUST_PROXY_HEADERS": "YES", "MYCELIC_READY_REQUIRES_NATS": "on",
         "MYCELIC_ALLOWED_HOSTS": " a.example.com, ,b.example.com ", "MYCELIC_CORS_ORIGINS": "https://x"},
        {"MYCELIC_HOST": "127.0.0.1", "MYCELIC_TRUST_PROXY_HEADERS": "nope"},
        {"MYCELIC_HOST": "127.0.0.1", "MYCELIC_LOG_LEVEL": "debug", "MYCELIC_NATS_MAX_BYTES": "-5"},
        {"MYCELIC_HOST": "127.0.0.1", "MYCELIC_NATS_MAX_AGE_SECONDS": "-1"},
        {"MYCELIC_HOST": "127.0.0.1", "MYCELIC_PORT": " 8081"}, {"MYCELIC_HOST": "127.0.0.1", "MYCELIC_PORT": "1_000"},
        {"MYCELIC_HOST": "127.0.0.1", "MYCELIC_PUBLISH_INTERVAL_SECONDS": "1e-1"},
        {"MYCELIC_HOST": "127.0.0.1", "MYCELIC_PUBLISH_INTERVAL_SECONDS": " inf"},
        {"MYCELIC_HOST": "127.0.0.1", "MYCELIC_NATS_URL": "tls://nats:4222", "MYCELIC_NATS_TOKEN": "t"},
        {"MYCELIC_HOST": "0.0.0.0", "MYCELIC_ADMIN_TOKEN": "a" * 40, "MYCELIC_DB_PATH": "/tmp/x.db", "MYCELIC_INSTANCE_ID": "i"},
    ]
    cases = []
    saved = dict(os.environ)
    for env in envs:
        os.environ.clear()
        os.environ.update({k: v for k, v in saved.items() if not k.startswith("MYCELIC_")})
        os.environ.update(env)
        try:
            s = Settings.from_env()
            cases.append({"env": {k: v.replace(rules, "{rules}") for k, v in env.items()},
                          "settings": json.loads(json.dumps(s.redacted()).replace(json.dumps(rules)[1:-1], "{rules}"))})
        except ConfigError as exc:
            cases.append({"env": {k: v.replace(rules, "{rules}") for k, v in env.items()}, "error": str(exc).replace(rules, "{rules}")})
    os.environ.clear()
    os.environ.update(saved)
    write("config.json", {"cases": cases})


# ---------------------------------------------------------------------------------------------------- coordination
def gen_coordination() -> None:
    rnd = random.Random(17)
    cases = []
    for _ in range(120):
        slots = [f"s{i}" for i in range(rnd.randint(1, 4))]
        pool_roots = [f"r{i}" for i in range(rnd.randint(1, 6))]
        pool_domains = [f"d{i}" for i in range(rnd.randint(1, 5))]
        claims = []
        for j in range(rnd.randint(0, 12)):
            claims.append(ClaimEnvelope(
                claim_id=f"c{rnd.randint(0, 99):02d}{j}", query_id="q", producer_node_id=rnd.choice(["p1", "p2", "p3", "mycelic"]),
                content={"slot": rnd.choice(slots + ["other"]), "value": rnd.choice(["v", "w", "[REDACTED]"])},
                confidence=rnd.choice([0.0, 0.1, 0.33, 0.5, 0.55, 0.7, 0.85, 0.9, 0.99, 1.0, 0.123456789, 0.6666666]),
                evidence_refs=(), source_ids=(), parent_memory_ids=(),
                lineage_root_ids=tuple(rnd.sample(pool_roots, rnd.randint(0, len(pool_roots)))),
                failure_domains=tuple(rnd.sample(pool_domains, rnd.randint(0, len(pool_domains)))),
                policy_status=PolicyStatus.ALLOWED, created_at=FIXED_NOW))
        synthesis = RuleBasedSynthesizer(tuple(slots)).synthesize(tuple(claims))
        metrics = LineageAnalyzer().score(tuple(claims), tuple(slots), synthesis)
        coalitions = LineageAnalyzer().reconstruction_coalitions(tuple(claims), tuple(slots))
        cases.append({"required_slots": slots,
                      "claims": [{"claim_id": c.claim_id, "producer_node_id": c.producer_node_id, "content": c.content,
                                  "confidence": c.confidence, "lineage_root_ids": list(c.lineage_root_ids),
                                  "failure_domains": list(c.failure_domains)} for c in claims],
                      "synthesis": to_jsonable(synthesis), "fragility": to_jsonable(metrics), "coalitions": to_jsonable(coalitions)})
    write("coordination.json", {"cases": cases})


# ---------------------------------------------------------------------------------------------------- python json + rounding
def gen_pyjson() -> None:
    rnd = random.Random(19)
    floats = [0.0, -0.0, 1.0, 0.1, 0.5, 0.97, 0.9999, 1e16, 1e15, 9999999999999998.0, 1e-4, 1e-5, 0.00012345, 1.5e-7, 123456789.0,
              1.2345678901234568e+17, 1e22, 1e-300, 5e-324, 1.7976931348623157e308, 2.0 ** 53, 3.141592653589793, -2.5, 100.0, 0.3,
              0.1 + 0.2, 1 / 3, 2 / 3, 12345.6789, 1e21, 1e-7, 123e-20]
    floats += [rnd.uniform(-1e6, 1e6) for _ in range(30)] + [10 ** rnd.uniform(-12, 22) for _ in range(40)]
    values = [
        {"b": 1, "a": [1, 2.5, None, True, False, "x"], "c": {"z": {}, "y": []}},
        {"text": "quote \" backslash \\ slash / newline \n tab \t cr \r bs \b ff \f nul \x00 us \x1f del \x7f é 漢 🚀 \u2028"},
        [], {}, "", [[], [{}], {"a": [{}]}], {"é": 1, "a": 2, "B": 3, "_": 4, "aa": 5, "a b": 6},
        {"n": 12345678901234567890, "neg": -5, "f": 1e16, "small": 1e-5},
    ]
    out = {"floats": [{"value": f, "repr": repr(f), "json": json.dumps(f)} for f in floats],
           "dumps": [{"value": v, "default": json.dumps(v, ensure_ascii=False), "sorted": json.dumps(v, ensure_ascii=False, sort_keys=True),
                      "indent1": json.dumps(v, ensure_ascii=False, indent=1), "compact": json.dumps(v, ensure_ascii=False, separators=(",", ":")),
                      "ascii": json.dumps(v)} for v in values]}
    rounds = []
    for x in [0.125, 0.375, 2.675, 0.5, 1.5, 2.5, 0.00005, 0.00015, 0.123456789, 0.99995, 0.999949999, 1 - 1e-17] + \
             [rnd.random() for _ in range(200)] + [1 - (1 - 0.9) * (1 - 0.7), 1 - 0.1 * 0.3 * 0.15]:
        rounds.append({"x": x, "r4": round(x, 4), "r6": round(x, 6), "f3": f"{x:.3f}", "f2": f"{x:.2f}"})
    out["round"] = rounds
    write("pyjson.json", out)


# ---------------------------------------------------------------------------------------------------- rule validation
def gen_rules() -> None:
    base = {"rule_id": "r1", "target_layer": "team", "required_slots": ["a", "b"], "conclusion": "C {entity} {slot:a}"}
    bodies = [base, {**base, "rule_id": "bad id!"}, {**base, "rule_id": ""}, {**base, "target_layer": "agent"},
              {**base, "target_layer": "nope"}, {**base, "required_slots": []}, {**base, "required_slots": "a"},
              {**base, "required_slots": ["a", "a", "b"]}, {**base, "required_slots": list("abcdefg")},
              {**base, "required_slots": list("abcdef") + ["a"]}, {**base, "required_slots": ["bad slot"]},
              {**base, "conclusion": ""}, {**base, "kind": "nope"}, {**base, "kind": "decision"},
              {**base, "min_agents": 0}, {**base, "min_agents": True}, {**base, "min_agents": 2.0}, {**base, "min_teams": "2"},
              {**base, "sources": []}, {**base, "sources": ["x"]}, {**base, "sources": ["slot_composition", "slot_composition"]},
              {**base, "emits_slot": "a"}, {**base, "emits_slot": "c", "emits_topic": "t"}, {**base, "emits_slot": "bad slot"},
              {**base, "min_units": {"a": {"region": 2}}}, {**base, "min_units": {"c": {"region": 2}}},
              {**base, "min_units": {"a": {"agent": 2}}}, {**base, "min_units": {"a": {"region": 0}}},
              {**base, "min_units": {"a": {}}}, {**base, "min_units": {"a": {"region": True}}}, {**base, "min_units": []},
              {**base, "min_units": None}, {**base, "corroborate": 1}, {**base, "enabled": 0}, {**base, "org_id": "acme"},
              {**base, "topic_prefix": "  supply: "}, {**base, "metadata": {"x": "y" * 3000}}, {**base, "metadata": []},
              {**base, "conclusion": 5}, "not an object", {**base, "rule_id": " r2 "}, {**base, "required_slots": ["a", 1]}]
    svc = MycelicService.__new__(MycelicService)
    cases = []
    for b in bodies:
        try:
            cases.append({"body": b, "rule": svc._rule_from_body(b).to_dict()})
        except ValidationError as exc:
            cases.append({"body": b, "error": str(exc)})
    write("rules_validation.json", {"cases": cases})


# ---------------------------------------------------------------------------------------------------- service scenarios
class Det:
    """A pinned service: in-process transport, fixed clock, counter ids."""

    def __init__(self, tmp: str, overrides: dict[str, Any] | None = None, *, transport: InProcessTransport | None = None,
                 store: MycelicStore | None = None, ids: CounterIds | None = None) -> None:
        from tests.mycelic.helpers import settings
        self.ids = ids or CounterIds()
        pin(self.ids)
        self.settings = settings(tmp, **(overrides or {}))
        self.service = MycelicService(self.settings, store=store, transport=transport or InProcessTransport(), metrics=Metrics())
        self.service.aggregator.clock = lambda: FIXED_NOW
        self.keys: dict[str, str] = {}

    def principal(self, who: str) -> Principal:
        token = ADMIN_TOKEN if who == "admin" else self.keys[who]
        return self.service.authenticate(f"Bearer {token}")


def snapshot(store: MycelicStore) -> dict[str, Any]:
    c = store._conn
    events = []
    for r in c.execute("SELECT * FROM events ORDER BY rid").fetchall():
        e = store_mod.row_event(r).to_dict()
        if isinstance(e["payload"], dict):
            e["payload"] = {k: v for k, v in e["payload"].items() if k not in ("key_hash", "key_prefix")}
        events.append(e)
    return {
        "memories": [store_mod.row_memory(r).to_dict() for r in c.execute("SELECT * FROM memories ORDER BY rid").fetchall()],
        "agg_keys": [r[0] for r in c.execute("SELECT agg_key FROM memories ORDER BY rid").fetchall()],
        "edges": [store_mod.row_edge(r).to_dict() for r in c.execute("SELECT * FROM lineage_edges ORDER BY child_id, parent_id").fetchall()],
        "events": events,
        "audit": [{**dict(r), "detail": json.loads(r["detail"])} for r in c.execute("SELECT * FROM audit_log ORDER BY id").fetchall()],
        "rules": [{**store_mod.row_rule(r).to_dict(), "updated_at": r["updated_at"]} for r in c.execute("SELECT * FROM rules ORDER BY rule_id").fetchall()],
        "agents": [{k: v for k, v in store_mod.row_agent(r).to_dict().items() if k not in ("key_prefix", "last_seen_at")}
                   for r in c.execute("SELECT * FROM agents ORDER BY path").fetchall()],
        "orgs": [dict(r) for r in c.execute("SELECT * FROM orgs ORDER BY org_id").fetchall()],
        "meta": {r["key"]: r["value"] for r in c.execute("SELECT * FROM meta ORDER BY key").fetchall()},
    }


def err(exc: Exception) -> dict[str, Any]:
    kind = type(exc).__name__
    msg = exc.args[0] if isinstance(exc, NotFound) and exc.args else str(exc)
    return {"error": {"type": kind, "message": msg}}


def resolve(store: MycelicStore, ref: Any, aliases: dict[str, str]) -> str:
    if isinstance(ref, str):
        return ref
    if "alias" in ref:
        return aliases[ref["alias"]]
    sel = ref["select"]
    for r in store._conn.execute("SELECT * FROM memories ORDER BY rid").fetchall():
        m = store_mod.row_memory(r).to_dict()
        if all(m.get(k) == v for k, v in sel.items()):
            return m["memory_id"]
    raise KeyError(f"no memory matches {sel}")


def strip_latency(d: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in d.items() if k != "latency_ms"}


async def run_scenario(steps: list[dict[str, Any]], overrides: dict[str, Any] | None = None) -> list[Any]:
    tmp = tempfile.mkdtemp()
    det = Det(tmp, overrides)
    s = det.service
    await s.start()
    outputs: list[Any] = []
    aliases: dict[str, str] = {}
    try:
        for step in steps:
            op = step["op"]
            out: Any = None
            try:
                if op == "rule":
                    out = (await s.upsert_rule(step["body"])).to_dict()
                elif op == "delete_rule":
                    out = await s.delete_rule(step["rule_id"])
                elif op == "register":
                    agent, key = await s.register_agent(step["body"])
                    det.keys[agent.agent_id] = key
                    out = {k: v for k, v in agent.to_dict().items() if k != "key_prefix"}
                elif op == "revoke":
                    out = await s.revoke_agent(step["agent_id"])
                elif op == "observe":
                    m, created = await s.ingest_memory(det.principal(step["agent"]), step["body"])
                    if step.get("as"):
                        aliases[step["as"]] = m.memory_id
                    out = {"memory_id": m.memory_id, "event_id": m.event_id, "created": created}
                elif op == "events":
                    out = await s.ingest_events(det.principal(step["agent"]), step["items"])
                elif op == "retract":
                    ev = await s.retract(det.principal(step["agent"]), resolve(s.store, step["memory"], aliases),
                                         step.get("reason", "retracted by producer"))
                    out = ev.to_dict()
                elif op == "settle":
                    assert await s.wait_idle(60), "did not settle"
                    out = None
                elif op == "query":
                    res = s.query(det.principal(step["viewer"]), step["body"])
                    res["results"] = [{**h, "memory": s.public_view(h["memory"], det.principal(step["viewer"]))} for h in res["results"]]
                    out = strip_latency(res)
                elif op == "lineage":
                    out = s.lineage(det.principal(step["viewer"]), resolve(s.store, step["memory"], aliases))
                elif op == "get_memory":
                    p = det.principal(step["viewer"])
                    out = s.public_view(s.get_memory(p, resolve(s.store, step["memory"], aliases)), p)
                elif op == "list_memories":
                    p = det.principal(step["viewer"])
                    rows = s.list_memories(p, scope=step.get("scope"), layers=step.get("layers"), limit=step.get("limit", 50),
                                           status=step.get("status", "active"))
                    out = [s.public_view(m, p) for m in rows]
                elif op == "snapshot":
                    out = snapshot(s.store)
                elif op == "stats":
                    st = s.store.stats()
                    out = {k: v for k, v in st.items() if k not in ("revision",)}
                elif op == "rebuild":
                    transport = s.transport
                    await s.stop()
                    old_store = s.store
                    det2 = Det(tmp, overrides, transport=transport, store=MycelicStore(":memory:"), ids=det.ids)
                    det2.keys = det.keys
                    det, s = det2, det2.service
                    await s.start()
                    assert await s.wait_idle(60), "rebuild did not settle"
                    await old_store.close()
                    out = snapshot(s.store)
                else:
                    raise ValueError(op)
            except (ValidationError, Forbidden, NotFound, HierarchyError) as exc:
                out = err(exc)
            outputs.append(out)
    finally:
        await s.close()
    return outputs


DEMO_RULE = {
    "rule_id": "component_supply_risk", "target_layer": "enterprise",
    "required_slots": ["transport_disruption", "supplier_buffer_low", "demand_commitment"],
    "conclusion": ("Supply risk for {entity}: inbound transport is disrupted ({slot:transport_disruption}); "
                   "the supplier buffer is thin ({slot:supplier_buffer_low}); demand is committed ({slot:demand_commitment})."),
    "topic_prefix": "supply:", "min_agents": 3, "min_teams": 2, "kind": "risk",
}
RULES = json.loads((ROOT / "deploy" / "mycelic" / "rules.json").read_text())["rules"]
REGIONAL = next(r for r in RULES if r["rule_id"] == "regional_supply_risk")
STRATEGIC = next(r for r in RULES if r["rule_id"] == "strategic_second_source")
T = "supply:sd-9/transport"


def reg(agent_id: str, team: str, department: str = "ops", subsidiary: str = "nw-gmbh", region: str = "emea",
        enterprise: str = "northwind", **extra: Any) -> dict[str, Any]:
    return {"op": "register", "body": {"enterprise": enterprise, "region": region, "subsidiary": subsidiary,
                                       "department": department, "team": team, "agent_id": agent_id, **extra}}


def obs(agent: str, key: str, text: str, settle: bool = True, **fields: Any) -> list[dict[str, Any]]:
    steps = [{"op": "observe", "agent": agent, "as": f"{agent}:{key}", "body": {"text": text, "idempotency_key": key, **fields}}]
    return steps + ([{"op": "settle"}] if settle else [])


def datapath_base() -> list[dict[str, Any]]:
    return [{"op": "rule", "body": DEMO_RULE}, reg("log-1", "logistics"), reg("log-2", "logistics"), reg("proc-1", "procurement"),
            reg("sales-1", "field-sales", "commercial"), reg("sales-2", "field-sales", "commercial")]


def datapath_seed() -> list[dict[str, Any]]:
    return (obs("log-1", "n1", "Port of Rotterdam terminal 3 strike announced for weeks 41-43; SD-9 servo drive shipments route through it.",
                topic=T, slot="transport_disruption", entity="sd-9", confidence=0.9)
            + obs("log-2", "n1", "Carrier ETA for the SD-9 servo drive container slipped by 12 days.", topic=T,
                  slot="transport_disruption", entity="sd-9", confidence=0.7)
            + obs("proc-1", "n1", "Kessler Antriebe confirms roughly two weeks of SD-9 inventory left.", topic="supply:sd-9/supplier",
                  slot="supplier_buffer_low", entity="sd-9", confidence=0.85)
            + obs("sales-1", "n1", "Helios Automation committed to 40 RX-4 arms for November; every RX-4 uses an SD-9 drive.",
                  topic="supply:sd-9/demand", slot="demand_commitment", entity="sd-9", confidence=0.8))


ENT = {"select": {"layer": "enterprise", "status": "active"}}
TEAM = {"select": {"layer": "team", "status": "active"}}


def scenario_datapath() -> list[dict[str, Any]]:
    return datapath_base() + datapath_seed() + [
        {"op": "snapshot"}, {"op": "stats"},
        {"op": "query", "viewer": "sales-2", "body": {"query": "delivery risk for RX-4 and the SD-9 servo drive", "scope": "northwind"}},
        {"op": "query", "viewer": "sales-2", "body": {"query": "x", "scope": "northwind/emea/nw-gmbh/ops"}},
        {"op": "query", "viewer": "admin", "body": {"query": "delivery risk", "scope": "northwind/emea/nw-gmbh/ops"}},
        {"op": "query", "viewer": "admin", "body": {"query": "delivery risk"}},
        {"op": "query", "viewer": "admin", "body": {"query": "strike", "org_id": "northwind", "min_layer": "team", "k": 3}},
        {"op": "query", "viewer": "log-1", "body": {"query": "", "scope": "northwind"}},
        {"op": "query", "viewer": "log-1", "body": {"query": "strike", "min_layer": "Team"}},
        {"op": "query", "viewer": "log-1", "body": {"query": "strike", "k": 0}},
        {"op": "query", "viewer": "log-1", "body": {"query": "strike", "k": True}},
        {"op": "query", "viewer": "log-1", "body": {"query": "strike", "scope": "Northwind"}},
        {"op": "query", "viewer": "log-1", "body": {"text": "strike carrier", "include_lineage": False, "topic": T}},
        {"op": "query", "viewer": "log-1", "body": {"query": "strike", "entity": "sd-9", "scope": "northwind/emea/nw-gmbh/ops/logistics"}},
        {"op": "lineage", "viewer": "sales-2", "memory": ENT}, {"op": "lineage", "viewer": "admin", "memory": ENT},
        {"op": "lineage", "viewer": "log-1", "memory": TEAM}, {"op": "lineage", "viewer": "sales-2", "memory": TEAM},
        {"op": "lineage", "viewer": "sales-2", "memory": "mem_doesnotexist"},
        {"op": "lineage", "viewer": "log-1", "memory": {"alias": "log-1:n1"}},
        {"op": "get_memory", "viewer": "log-2", "memory": TEAM}, {"op": "get_memory", "viewer": "sales-2", "memory": TEAM},
        {"op": "get_memory", "viewer": "sales-2", "memory": {"alias": "log-1:n1"}},
        {"op": "get_memory", "viewer": "sales-2", "memory": ENT}, {"op": "get_memory", "viewer": "log-1", "memory": {"alias": "log-1:n1"}},
        {"op": "list_memories", "viewer": "sales-2", "scope": "northwind", "limit": 500},
        {"op": "list_memories", "viewer": "sales-2", "scope": "northwind", "layers": ["enterprise"]},
        {"op": "list_memories", "viewer": "admin", "scope": "northwind", "limit": 500},
        {"op": "list_memories", "viewer": "admin"}, {"op": "list_memories", "viewer": "log-1", "scope": "northwind/emea/nw-gmbh/sales"},
        {"op": "list_memories", "viewer": "log-1", "status": "superseded"},
        {"op": "observe", "agent": "log-1", "body": {"text": "same note", "idempotency_key": "n1"}},
        {"op": "observe", "agent": "log-1", "body": {"text": "spoof", "agent_id": "log-2"}},
        {"op": "observe", "agent": "admin", "body": {"text": "admins do not observe"}},
        {"op": "retract", "agent": "admin", "memory": TEAM},
        {"op": "retract", "agent": "sales-2", "memory": {"alias": "log-1:n1"}},
        {"op": "retract", "agent": "sales-2", "memory": ENT},
    ]


def scenario_supersede() -> list[dict[str, Any]]:
    return datapath_base() + datapath_seed() + [reg("log-3", "logistics")] + \
        obs("log-3", "n1", "Rotterdam customs broker warns of a two-week backlog for SD-9 containers.", topic=T,
            slot="transport_disruption", entity="sd-9", confidence=0.6) + [
        {"op": "snapshot"}, {"op": "lineage", "viewer": "log-1", "memory": TEAM},
        reg("log-4", "logistics")] + \
        obs("log-4", "n1", "Rail alternative for SD-9 is fully booked.", topic=T, slot="transport_disruption", entity="sd-9", confidence=0.5) + [
        {"op": "retract", "agent": "log-4", "memory": {"alias": "log-4:n1"}, "reason": "mistaken"}, {"op": "settle"},
        {"op": "snapshot"}, {"op": "lineage", "viewer": "admin", "memory": TEAM},
        {"op": "retract", "agent": "log-3", "memory": {"alias": "log-3:n1"}, "reason": "false alarm"}, {"op": "settle"},
        {"op": "snapshot"}, {"op": "lineage", "viewer": "log-1", "memory": TEAM},
        {"op": "retract", "agent": "log-3", "memory": {"alias": "log-3:n1"}, "reason": "again"}, {"op": "settle"},
    ]


def scenario_retraction() -> list[dict[str, Any]]:
    return datapath_base() + datapath_seed() + [
        {"op": "retract", "agent": "sales-1", "memory": {"alias": "sales-1:n1"}, "reason": "customer cancelled the order"},
        {"op": "settle"}, {"op": "snapshot"},
        {"op": "query", "viewer": "sales-2", "body": {"query": "supply risk sd-9", "scope": "northwind", "min_layer": "enterprise"}},
        {"op": "retract", "agent": "log-1", "memory": {"alias": "log-1:n1"}, "reason": "withdrawn"},
        {"op": "retract", "agent": "log-2", "memory": {"alias": "log-2:n1"}, "reason": "withdrawn"}, {"op": "settle"},
        *obs("proc-1", "pallets", "Unrelated note about pallets.", topic="ops:pallets"), {"op": "snapshot"},
        {"op": "lineage", "viewer": "admin", "memory": {"select": {"layer": "enterprise"}}},
        {"op": "observe", "agent": "log-1", "body": {"text": "x", "source_event_ids": ["evt_doesnotexist"]}},
        reg("acme-1", "acme", "acme", "acme", "acme", "acme"),
        {"op": "events", "agent": "acme-1", "items": [{"type": "x"}]},
        {"op": "observe", "agent": "log-1", "body": {"text": "x", "source_event_ids": ["evt_00000000000000000099"]}},
    ]


def scenario_events() -> list[dict[str, Any]]:
    item = {"type": "supplier.call", "payload": {"supplier": "Kessler"}, "idempotency_key": "call-1",
            "memory": {"text": "Kessler has two weeks of SD-9 stock.", "topic": "supply:sd-9/supplier", "slot": "supplier_buffer_low", "entity": "sd-9"}}
    return datapath_base() + [
        {"op": "events", "agent": "proc-1", "items": [item, {"type": "heartbeat"}]},
        {"op": "events", "agent": "proc-1", "items": [{"type": "supplier.call", "idempotency_key": "call-1"}]},
        {"op": "events", "agent": "proc-1", "items": [item]}, {"op": "events", "agent": "proc-1", "items": [item]},
        {"op": "events", "agent": "proc-1", "items": []}, {"op": "events", "agent": "proc-1", "items": "nope"},
        {"op": "events", "agent": "proc-1", "items": [{"type": "bad type!"}]},
        {"op": "events", "agent": "proc-1", "items": [{"type": "t", "memory": "x"}]},
        {"op": "events", "agent": "proc-1", "items": [{"type": "t", "payload": {"x": "y" * 17000}}]},
        {"op": "events", "agent": "proc-1", "items": [{"type": "t", "occurred_at": "yesterday"}]},
        {"op": "events", "agent": "proc-1", "items": [{"type": "t", "occurred_at": "2026-09-01T08:00:00+02:00", "idempotency_key": "k2",
                                                         "memory": {"text": "with own key", "idempotency_key": "own"}}]},
        {"op": "events", "agent": "admin", "items": [{"type": "t"}]},
        {"op": "settle"}, {"op": "snapshot"},
        {"op": "lineage", "viewer": "proc-1", "memory": {"select": {"layer": "agent", "producer_id": "proc-1"}}},
        {"op": "observe", "agent": "proc-1", "body": {"text": "x", "source_event_ids": ["evt_00000000000000000007"]}},
        {"op": "observe", "agent": "proc-1", "body": {"text": "y" * 4001}},
        {"op": "observe", "agent": "proc-1", "body": {"text": "   "}},
        {"op": "observe", "agent": "proc-1", "body": {"text": 5}},
        {"op": "observe", "agent": "proc-1", "body": {"text": "x", "kind": "nope"}},
        {"op": "observe", "agent": "proc-1", "body": {"text": "x", "visibility": "public"}},
        {"op": "observe", "agent": "proc-1", "body": {"text": "x", "confidence": 1.5}},
        {"op": "observe", "agent": "proc-1", "body": {"text": "x", "confidence": "0.5"}},
        {"op": "observe", "agent": "proc-1", "body": {"text": "x", "confidence": True}},
        {"op": "observe", "agent": "proc-1", "body": {"text": "x", "confidence": 1, "idempotency_key": "int-conf"}},
        {"op": "observe", "agent": "proc-1", "body": {"text": "x", "slot": "bad slot!"}},
        {"op": "observe", "agent": "proc-1", "body": {"text": "x", "topic": "t" * 201}},
        {"op": "observe", "agent": "proc-1", "body": {"text": "x", "metadata": {"blob": "y" * 5000}}},
        {"op": "observe", "agent": "proc-1", "body": {"text": "x", "metadata": "nope"}},
        {"op": "observe", "agent": "proc-1", "body": {"text": "x", "source_event_ids": "nope"}},
        {"op": "observe", "agent": "proc-1", "body": {"text": "x", "source_event_ids": ["bad id!"]}},
        {"op": "observe", "agent": "proc-1", "body": {"text": "x", "observed_at": "2026-09-01T08:30:15.5+02:00", "idempotency_key": "ts",
                                                       "topic": " padded ", "entity": "  ", "local_ref": "note-42",
                                                       "metadata": {"promoted_from": "x", "agg_key": "k", "note": "kept", "n": 1.5}}},
        {"op": "observe", "agent": "proc-1", "body": {"text": "x", "observed_at": "garbage"}},
        {"op": "observe", "agent": "proc-1", "body": "not a dict"},
        {"op": "settle"}, {"op": "snapshot"},
    ]


def region_obs(region: str, *, skip_supplier: bool = False, settle: bool = True) -> list[dict[str, Any]]:
    port = "Rotterdam" if region == "emea" else ("Busan" if region == "apac" else "Long Beach")
    steps = obs(f"{region}-logistics-1", "t1", f"Port of {port} strike announced; SD-9 shipments route through it.",
                topic=T, slot="transport_disruption", entity="sd-9", confidence=0.9)
    steps += obs(f"{region}-logistics-2", "t2", f"Carrier ETA for SD-9 containers via {port} slipped by 12 days.",
                 topic=T, slot="transport_disruption", entity="sd-9", confidence=0.7)
    if not skip_supplier:
        steps += obs(f"{region}-procurement-1", "s1", "Kessler Antriebe confirms two weeks of SD-9 inventory left.",
                     topic="supply:sd-9/supplier", slot="supplier_buffer_low", entity="sd-9", confidence=0.85)
    steps += obs(f"{region}-field-sales-1", "d1", "A customer committed to 40 RX-4 arms for November; every RX-4 uses an SD-9.",
                 topic="supply:sd-9/demand", slot="demand_commitment", entity="sd-9", confidence=0.8)
    return steps


def hq_obs() -> list[dict[str, Any]]:
    return obs("hq-analytics-1", "growth", "RX-4 order intake is up 40% year over year across all regions.", topic="strategy:demand",
               slot="demand_growth", entity="sd-9", confidence=0.8) + \
        obs("hq-sourcing-1", "conc", "Kessler Antriebe is the only qualified SD-9 source for every subsidiary.",
            topic="strategy:supply-base", slot="supplier_concentration", entity="sd-9", confidence=0.95)


def strategic_base() -> list[dict[str, Any]]:
    steps = [{"op": "rule", "body": REGIONAL}, {"op": "rule", "body": STRATEGIC}]
    for region, sub in (("emea", "nw-gmbh"), ("apac", "nw-kk")):
        for team, dept in (("logistics", "ops"), ("procurement", "ops"), ("field-sales", "commercial")):
            for i in (1, 2):
                steps.append(reg(f"{region}-{team}-{i}", team, dept, sub, region))
    steps.append(reg("hq-sourcing-1", "sourcing", "strategy", "northwind-hq", "hq"))
    steps.append(reg("hq-analytics-1", "analytics", "strategy", "northwind-hq", "hq"))
    return steps


STRAT = {"select": {"layer": "enterprise", "rule_id": "strategic_second_source", "status": "active"}}


def scenario_strategic() -> list[dict[str, Any]]:
    return strategic_base() + region_obs("emea") + hq_obs() + [{"op": "snapshot"}] + region_obs("apac") + [
        {"op": "snapshot"}, {"op": "lineage", "viewer": "admin", "memory": STRAT},
        {"op": "lineage", "viewer": "emea-field-sales-2", "memory": STRAT},
        {"op": "lineage", "viewer": "hq-sourcing-1", "memory": STRAT},
        {"op": "query", "viewer": "emea-field-sales-2", "body": {"query": "second source sourcing decision sd-9", "scope": "northwind", "min_layer": "region"}},
        {"op": "query", "viewer": "hq-sourcing-1", "body": {"query": "Should we qualify a second source for the SD-9 servo drive?", "scope": "northwind", "min_layer": "enterprise", "k": 5}},
        {"op": "query", "viewer": "apac-logistics-1", "body": {"query": "supply risk sd-9", "scope": "northwind", "k": 30}},
        {"op": "list_memories", "viewer": "emea-field-sales-2", "scope": "northwind", "limit": 500},
        {"op": "rebuild"},
        {"op": "lineage", "viewer": "hq-sourcing-1", "memory": STRAT},
    ]


def scenario_follows_evidence() -> list[dict[str, Any]]:
    steps = strategic_base() + region_obs("emea") + region_obs("apac") + hq_obs() + [
        {"op": "retract", "agent": "apac-procurement-1", "memory": {"alias": "apac-procurement-1:s1"}, "reason": "counted wrong"},
        {"op": "settle"}, {"op": "snapshot"}]
    steps += obs("apac-procurement-2", "recount", "Kessler Antriebe confirms two weeks of SD-9 inventory left.",
                 topic="supply:sd-9/supplier", slot="supplier_buffer_low", entity="sd-9", confidence=0.85)
    steps += [{"op": "snapshot"}]
    for team, dept, i in (("logistics", "ops", 1), ("logistics", "ops", 2), ("procurement", "ops", 1), ("field-sales", "commercial", 1)):
        steps.append(reg(f"amer-{team}-{i}", team, dept, "nw-inc", "amer"))
    steps += region_obs("amer") + [{"op": "snapshot"},
                                   {"op": "retract", "agent": "amer-procurement-1", "memory": {"alias": "amer-procurement-1:s1"}, "reason": "duplicate entry"},
                                   {"op": "settle"}, {"op": "snapshot"}, {"op": "lineage", "viewer": "admin", "memory": STRAT}]
    return steps


def scenario_corroborate_toggle() -> list[dict[str, Any]]:
    return strategic_base() + [{"op": "rule", "body": {**STRATEGIC, "corroborate": False}}] + region_obs("emea") + region_obs("apac") + \
        hq_obs() + [{"op": "snapshot"}, {"op": "rule", "body": STRATEGIC}] + \
        obs("hq-analytics-1", "growth2", "Second demand datapoint: backlog doubled.", topic="strategy:demand",
            slot="demand_growth", entity="sd-9", confidence=0.7) + [{"op": "snapshot"}]


def scenario_withdrawal() -> list[dict[str, Any]]:
    return strategic_base() + region_obs("emea") + region_obs("apac") + hq_obs() + \
        obs("emea-logistics-1", "sup", "Kessler told us there are two weeks of SD-9 inventory left.", topic="supply:sd-9/supplier",
            slot="supplier_buffer_low", entity="sd-9", confidence=0.95) + [
        {"op": "snapshot"},
        {"op": "query", "viewer": "hq-sourcing-1", "body": {"query": "second source sd-9", "scope": "northwind", "min_layer": "enterprise"}}] + \
        obs("emea-procurement-2", "recount", "Recount: two weeks of SD-9 inventory, no second source.", topic="supply:sd-9/supplier",
            slot="supplier_buffer_low", entity="sd-9", confidence=0.97) + [{"op": "snapshot"}]


def scenario_stale_dependents() -> list[dict[str, Any]]:
    steps = strategic_base() + [{"op": "rule", "body": {"rule_id": "r", "target_layer": "department",
                                                         "required_slots": ["transport_disruption", "supplier_buffer_low"],
                                                         "sources": ["topic_consolidation"], "min_agents": 2, "min_teams": 2,
                                                         "conclusion": "R for {entity}"}}]
    for a, slot, topic in (("emea-logistics-1", "transport_disruption", T), ("emea-logistics-2", "transport_disruption", T),
                           ("emea-procurement-1", "supplier_buffer_low", "supply:sd-9/supplier"),
                           ("emea-procurement-2", "supplier_buffer_low", "supply:sd-9/supplier")):
        steps += obs(a, "n", f"note by {a}", topic=topic, slot=slot, entity="sd-9", confidence=0.8)
    steps += [{"op": "snapshot"}]
    steps += obs("emea-logistics-1", "rx2", "Rail slot for RX-2 frames also affected by the strike.", topic=T,
                 slot="transport_disruption", entity="rx-2", confidence=0.6)
    return steps + [{"op": "snapshot"}]


def scenario_unanimous() -> list[dict[str, Any]]:
    return strategic_base() + [{"op": "rule", "body": {"rule_id": "r2", "target_layer": "region",
                                                        "required_slots": ["transport_disruption", "supplier_buffer_low"],
                                                        "sources": ["agent_observation", "topic_consolidation"], "min_agents": 3,
                                                        "conclusion": "R2 {entity}"}}] + \
        obs("emea-logistics-1", "a", "strike", topic=T, slot="transport_disruption", entity="sd-9", confidence=0.9) + \
        obs("emea-procurement-1", "b", "low stock", topic="supply:sd-9/supplier", slot="supplier_buffer_low", entity="sd-9", confidence=0.8) + \
        obs("emea-logistics-2", "c", "Port congestion note without a slot.", topic=T) + [{"op": "snapshot"}] + [
        {"op": "rule", "body": {"rule_id": "a", "target_layer": "team", "required_slots": ["x"], "emits_slot": "y", "conclusion": "A",
                                "sources": ["agent_observation", "slot_composition"]}},
        {"op": "rule", "body": {"rule_id": "b", "target_layer": "team", "required_slots": ["y"], "emits_slot": "x", "conclusion": "B",
                                "sources": ["slot_composition", "agent_observation"]}},
        {"op": "rule", "body": {"rule_id": "c", "target_layer": "enterprise", "required_slots": ["y"], "emits_slot": "z", "conclusion": "C",
                                "sources": ["slot_composition"]}},
        {"op": "rule", "body": {"rule_id": "too-many", "target_layer": "team", "required_slots": list("abcdefg"), "conclusion": "x"}},
        {"op": "rule", "body": {"rule_id": "d", "target_layer": "region", "required_slots": ["z"], "emits_slot": "y", "conclusion": "D",
                                "sources": ["slot_composition"]}},
        {"op": "rule", "body": {"rule_id": "e", "target_layer": "region", "required_slots": ["z"], "emits_slot": "y", "conclusion": "E",
                                "sources": ["slot_composition"], "enabled": False}},
        {"op": "delete_rule", "rule_id": "c"}, {"op": "delete_rule", "rule_id": "nope"},
        {"op": "rule", "body": {"rule_id": "d", "target_layer": "region", "required_slots": ["z"], "emits_slot": "y", "conclusion": "D",
                                "sources": ["slot_composition"]}},
        {"op": "settle"}, {"op": "snapshot"},
    ]


def scenario_fragility() -> list[dict[str, Any]]:
    steps = strategic_base()
    for i in range(12):
        for j in (1, 2):
            c = 0.5 + (2 * i + j) / 100
            steps += obs(f"emea-logistics-{j}", f"t{i}", f"transport note {i}-{j}", topic=T, slot="transport_disruption", entity="sd-9", confidence=c)
            steps += obs(f"emea-procurement-{j}", f"s{i}", f"supplier note {i}-{j}", topic="supply:sd-9/supplier",
                         slot="supplier_buffer_low", entity="sd-9", confidence=c)
            steps += obs(f"emea-field-sales-{j}", f"d{i}", f"demand note {i}-{j}", topic="supply:sd-9/demand",
                         slot="demand_commitment", entity="sd-9", confidence=c)
    return steps + [{"op": "snapshot"}, {"op": "lineage", "viewer": "admin", "memory": {"select": {"layer": "region", "rule_id": "regional_supply_risk", "status": "active"}}}]


def scenario_promotion() -> list[dict[str, Any]]:
    """A small organization: one team, so every unit above it has a single registered child and promotes."""
    steps = [reg("a1", "support", "acme", "acme", "acme", "acme"), reg("a2", "support", "acme", "acme", "acme", "acme")]
    steps += obs("a1", "k1", "Customer Zürich reports the RX-4 gripper firmware crashes on boot.", topic="support:rx-4", confidence=0.6)
    steps += obs("a2", "k1", "Second report: RX-4 firmware 2.1 crash loop at the Basel site.", topic="support:rx-4", confidence=0.7)
    steps += [{"op": "snapshot"}, {"op": "lineage", "viewer": "a1", "memory": {"select": {"layer": "enterprise", "status": "active"}}},
              {"op": "query", "viewer": "a1", "body": {"query": "firmware crash"}},
              reg("b1", "billing", "acme", "acme", "acme", "acme")]
    steps += obs("b1", "k1", "Billing sees refunds for RX-4 firmware crashes.", topic="support:rx-4", confidence=0.5)
    steps += [{"op": "snapshot"}, reg("c1", "billing", "finance", "acme", "acme", "acme")]
    steps += obs("c1", "k1", "Finance books RX-4 firmware crash refunds.", topic="support:rx-4", confidence=0.9, visibility="org")
    steps += [{"op": "snapshot"}, {"op": "query", "viewer": "a1", "body": {"query": "firmware crash refunds", "k": 20}},
              {"op": "revoke", "agent_id": "c1"}, {"op": "revoke", "agent_id": "nobody"}] + \
        obs("a1", "k2", "Third report from Bern.", topic="support:rx-4", confidence=0.4) + [{"op": "snapshot"}, {"op": "rebuild"}]
    return steps


async def gen_scenarios() -> None:
    scenarios = {"datapath": scenario_datapath(), "supersede_reactivate": scenario_supersede(), "retraction": scenario_retraction(),
                 "events": scenario_events(), "strategic": scenario_strategic(), "follows_evidence": scenario_follows_evidence(),
                 "corroborate_toggle": scenario_corroborate_toggle(), "withdrawal_cascade": scenario_withdrawal(),
                 "stale_dependents": scenario_stale_dependents(), "unanimous_and_cycles": scenario_unanimous(),
                 "fragility_bounded": scenario_fragility(), "promotion": scenario_promotion()}
    for name, steps in scenarios.items():
        outputs = await run_scenario(steps)
        write(f"scenario_{name}.json", {"fixed_now": FIXED_NOW, "admin_token": ADMIN_TOKEN, "steps": steps, "outputs": outputs})


# ---------------------------------------------------------------------------------------------------- HTTP API + MCP
IGNORE_KEYS = {"api_key", "key_prefix", "latency_ms"}
IGNORE_PATHS = {("checks", "db", "path"), ("settings", "db_path")}


def normalize(v: Any, path: tuple = ()) -> Any:
    if isinstance(v, dict):
        out = {}
        for k, x in v.items():
            if k in IGNORE_KEYS or (*path, k) in IGNORE_PATHS:
                out[k] = "<ignored>"
            elif k == "text" and isinstance(x, str) and path and path[-1] == "content[]" and x.startswith("{"):
                try:
                    out[k] = {"json": normalize(json.loads(x), ())}
                except json.JSONDecodeError:
                    out[k] = x
            else:
                out[k] = normalize(x, (*path, k))
        return out
    if isinstance(v, list):
        return [normalize(x, (*path[:-1], f"{path[-1]}[]") if path else ("[]",)) for x in v]
    return v


def mcp(method: str, params: Any = None, id_: Any = 1, **extra: Any) -> dict[str, Any]:
    msg: dict[str, Any] = {"jsonrpc": "2.0", "id": id_, "method": method}
    if params is not None:
        msg["params"] = params
    msg.update(extra)
    return msg


def api_requests() -> dict[str, dict[str, Any]]:
    A = {"Authorization": "Bearer {admin}"}
    K = lambda who: {"Authorization": "Bearer {key:%s}" % who}  # noqa: E731
    J = {"Accept": "application/json"}
    reg_body = lambda agent, **u: {"enterprise": "northwind", "region": "emea", "subsidiary": "nw-gmbh", "department": "ops",  # noqa: E731
                                   "team": "logistics", "agent_id": agent, **u}
    main = [
        {"method": "GET", "path": "/"}, {"method": "GET", "path": "/health"}, {"method": "GET", "path": "/ready"},
        {"method": "GET", "path": "/health", "headers": A}, {"method": "GET", "path": "/health", "headers": {"Authorization": "Bearer mk_nobody.wrong"}},
        {"method": "GET", "path": "/admin/status", "headers": A},
        {"method": "POST", "path": "/memory", "json": {"text": "x"}},
        {"method": "POST", "path": "/memory", "data": "text=1", "headers": {"Authorization": "Bearer mk_a.b", "Content-Type": "text/plain"}},
        {"method": "POST", "path": "/memory", "data": "text=1", "headers": {"Authorization": "Bearer mk_a.b"}},
        {"method": "GET", "path": "/whoami", "headers": {"Authorization": "Bearer mk_nobody.secret"}},
        {"method": "GET", "path": "/whoami", "headers": {"Authorization": "Bearer é"}},
        {"method": "GET", "path": "/whoami", "headers": {"Authorization": "Basic abc"}},
        {"method": "GET", "path": "/whoami", "headers": {"Authorization": "Bearer "}},
        {"method": "GET", "path": "/whoami", "headers": {"Authorization": "bearer {admin}"}},
        {"method": "GET", "path": "/whoami", "headers": A},
        {"method": "GET", "path": "/nope", "headers": A}, {"method": "GET", "path": "/nope"},
        {"method": "PUT", "path": "/memory", "headers": A},
        {"method": "OPTIONS", "path": "/memory"},
        {"method": "POST", "path": "/memory", "data_repeat": ["x", 1024 * 1024 + 10], "headers": {**A, "Content-Type": "application/json"}},
        {"method": "POST", "path": "/memory", "data": "{bad json", "headers": {**A, "Content-Type": "application/json"}},
        {"method": "POST", "path": "/memory", "json": [1, 2], "headers": A},
        {"method": "POST", "path": "/admin/rules", "json": DEMO_RULE, "headers": A},
        {"method": "POST", "path": "/admin/rules", "json": {"rule_id": "bad"}, "headers": A},
        {"method": "GET", "path": "/admin/rules", "headers": A},
        {"method": "POST", "path": "/admin/agents", "json": reg_body("log-1"), "headers": A, "capture_key": "log-1"},
        {"method": "POST", "path": "/admin/agents", "json": reg_body("log-2"), "headers": A, "capture_key": "log-2"},
        {"method": "POST", "path": "/admin/agents", "json": reg_body("proc-1", team="procurement"), "headers": A, "capture_key": "proc-1"},
        {"method": "POST", "path": "/admin/agents", "json": reg_body("sales-1", department="commercial", team="field-sales"), "headers": A, "capture_key": "sales-1"},
        {"method": "POST", "path": "/admin/agents", "json": reg_body("sales-2", department="commercial", team="field-sales"), "headers": A, "capture_key": "sales-2"},
        {"method": "POST", "path": "/admin/agents", "json": {"enterprise": "northwind", "agent_id": "evil"}, "headers": K("log-1")},
        {"method": "POST", "path": "/admin/agents", "json": {"enterprise": "northwind", "agent_id": "x", "scopes": ["admin"]}, "headers": A},
        {"method": "POST", "path": "/admin/agents", "json": {"enterprise": "northwind", "agent_id": "log-1"}, "headers": A},
        {"method": "POST", "path": "/admin/agents", "json": {"enterprise": "northwind", "agent_id": "mycelic"}, "headers": A},
        {"method": "POST", "path": "/admin/agents", "json": {"enterprise": "North Wind", "agent_id": "x"}, "headers": A},
        {"method": "POST", "path": "/admin/agents", "json": reg_body("events-only", scopes=["events:write"]), "headers": A, "capture_key": "events-only"},
        {"method": "POST", "path": "/admin/agents", "json": reg_body("read-only", scopes=["memory:read"]), "headers": A, "capture_key": "read-only"},
        {"method": "POST", "path": "/admin/agents", "json": {"enterprise": "northwind", "agent_id": "no-scopes", "scopes": []}, "headers": A, "capture_key": "no-scopes"},
        {"method": "POST", "path": "/admin/agents", "json": {"enterprise": "acme", "agent_id": "acme-1", "display_name": "Acme One", "metadata": {"x": 1}}, "headers": A, "capture_key": "acme-1"},
        {"method": "GET", "path": "/admin/agents", "headers": A}, {"method": "GET", "path": "/admin/agents?org=acme", "headers": A},
        {"method": "GET", "path": "/whoami", "headers": K("log-1")},
        {"method": "POST", "path": "/memory", "json": {"text": "admin note"}, "headers": A},
        {"method": "POST", "path": "/memory", "json": {"text": "x"}, "headers": K("no-scopes")},
        {"method": "POST", "path": "/memory", "json": {"text": "Port of Rotterdam terminal 3 strike announced for weeks 41-43.", "topic": T,
                                                       "slot": "transport_disruption", "entity": "sd-9", "confidence": 0.9, "idempotency_key": "a"},
         "headers": K("log-1"), "settle": True},
        {"method": "POST", "path": "/memory", "json": {"text": "Port of Rotterdam terminal 3 strike announced for weeks 41-43.", "idempotency_key": "a"},
         "headers": K("log-1")},
        {"method": "POST", "path": "/memory", "json": {"text": "Carrier ETA for the SD-9 container slipped by 12 days.", "topic": T,
                                                       "slot": "transport_disruption", "entity": "sd-9", "confidence": 0.7, "idempotency_key": "b"},
         "headers": K("log-2"), "settle": True},
        {"method": "POST", "path": "/memory", "json": {"text": "Kessler Antriebe has two weeks of SD-9 inventory left.", "topic": "supply:sd-9/supplier",
                                                       "slot": "supplier_buffer_low", "entity": "sd-9", "idempotency_key": "c"},
         "headers": K("proc-1"), "settle": True},
        {"method": "POST", "path": "/memory", "json": {"text": "Helios Automation committed to 40 RX-4 arms for November.", "topic": "supply:sd-9/demand",
                                                       "slot": "demand_commitment", "entity": "sd-9", "idempotency_key": "d", "local_ref": "my-note"},
         "headers": K("sales-1"), "settle": True},
        {"method": "GET", "path": "/memory/{mem:log-1:a}", "headers": K("log-1")},
        {"method": "GET", "path": "/memory/{mem:log-1:a}", "headers": K("sales-2")},
        {"method": "GET", "path": "/memory/{mem:log-1:a}", "headers": A},
        {"method": "GET", "path": "/memory/{mem:log-1:a}", "headers": K("events-only")},
        {"method": "GET", "path": "/memory/{mem:sales-1:d}", "headers": K("sales-1")},
        {"method": "GET", "path": "/memory/{mem:sales-1:d}", "headers": K("sales-2")},
        {"method": "GET", "path": "/memories?scope=northwind", "headers": K("events-only")},
        {"method": "POST", "path": "/query", "json": {"query": "note"}, "headers": K("events-only")},
        {"method": "POST", "path": "/query", "json": {"query": "delivery risk RX-4 SD-9", "scope": "northwind"}, "headers": K("sales-2")},
        {"method": "POST", "path": "/query", "json": {"query": "delivery risk RX-4 SD-9", "scope": "northwind"}, "headers": K("read-only")},
        {"method": "POST", "path": "/query", "json": {"query": "delivery risk", "scope": "northwind/emea/nw-gmbh/ops"}, "headers": K("sales-2")},
        {"method": "POST", "path": "/query", "json": {"query": "delivery risk", "scope": "northwind/emea/nw-gmbh/ops"}, "headers": A},
        {"method": "POST", "path": "/query", "json": {"query": "delivery risk"}, "headers": A},
        {"method": "POST", "path": "/query", "json": {"query": "x", "scope": "Bad Scope"}, "headers": A},
        {"method": "POST", "path": "/query", "json": {"query": "x", "min_layer": "nope"}, "headers": K("log-1")},
        {"method": "POST", "path": "/query", "json": {"query": "x", "k": 101}, "headers": K("log-1")},
        {"method": "GET", "path": "/lineage/{select:enterprise}", "headers": K("sales-2")},
        {"method": "GET", "path": "/lineage/{select:enterprise}", "headers": A},
        {"method": "GET", "path": "/lineage/{mem:log-1:a}", "headers": K("read-only")},
        {"method": "GET", "path": "/lineage/mem_doesnotexist", "headers": K("sales-2")},
        {"method": "GET", "path": "/memories?scope=northwind&layer=enterprise", "headers": K("sales-2")},
        {"method": "GET", "path": "/memories?scope=northwind&limit=500", "headers": K("sales-2")},
        {"method": "GET", "path": "/memories?scope=northwind&limit=500", "headers": A},
        {"method": "GET", "path": "/memories?limit=abc", "headers": A}, {"method": "GET", "path": "/memories", "headers": A},
        {"method": "GET", "path": "/memories?scope=bad%20scope", "headers": A},
        {"method": "GET", "path": "/memories?limit=0&layer=team,agent", "headers": K("log-1")},
        {"method": "GET", "path": "/memories?status=superseded", "headers": K("log-1")},
        {"method": "POST", "path": "/memory/{mem:log-1:a}/retract", "json": {"reason": "strike called off"}, "headers": K("log-2")},
        {"method": "POST", "path": "/memory/{mem:log-1:a}/retract", "json": {"reason": "strike called off"}, "headers": K("log-1"), "settle": True},
        {"method": "GET", "path": "/memory/{mem:log-1:a}", "headers": K("log-1")},
        {"method": "DELETE", "path": "/memory/{mem:log-2:b}", "headers": K("log-2"), "settle": True},
        {"method": "POST", "path": "/memory/{select:team}/retract", "headers": A},
        {"method": "POST", "path": "/memory/mem_nope/retract", "headers": A},
        {"method": "POST", "path": "/events", "json": {"events": [{"type": "call", "memory": {"text": "Kessler stock is low", "slot": "supplier_buffer_low"}}]},
         "headers": K("proc-1"), "settle": True},
        {"method": "POST", "path": "/events", "json": {"events": []}, "headers": K("proc-1")},
        {"method": "POST", "path": "/events", "json": {"nothing": 1}, "headers": K("proc-1")},
        {"method": "POST", "path": "/events", "json": {"events": [{"type": "x"}] * 101}, "headers": K("proc-1")},
        {"method": "POST", "path": "/events", "json": {"events": [{"type": "x"}]}, "headers": K("read-only")},
        {"method": "POST", "path": "/memory", "json": {"text": "x", "metadata": {"blob": "y" * 5000}}, "headers": K("proc-1")},
        {"method": "POST", "path": "/memory", "json": {"text": "x", "slot": "bad slot!"}, "headers": K("proc-1")},
        {"method": "POST", "path": "/memory", "json": {"text": "x", "agent_id": "log-1"}, "headers": K("proc-1")},
        {"method": "GET", "path": "/admin/audit", "headers": A}, {"method": "GET", "path": "/admin/audit?limit=3", "headers": A},
        {"method": "GET", "path": "/admin/audit?limit=x", "headers": A},
        {"method": "GET", "path": "/admin/events?org=northwind&kind=memory.observed", "headers": A},
        {"method": "GET", "path": "/admin/events?org=northwind&agent=log-1&limit=2", "headers": A},
        {"method": "GET", "path": "/admin/events?org=northwind&status=failed", "headers": A},
        {"method": "GET", "path": "/admin/events", "headers": A},
        {"method": "GET", "path": "/admin/audit", "headers": K("log-1")},
        {"method": "POST", "path": "/admin/agents/log-2/rotate", "headers": A, "capture_key": "log-2"},
        {"method": "POST", "path": "/admin/agents/nobody/rotate", "headers": A},
        {"method": "DELETE", "path": "/admin/agents/sales-2", "headers": A},
        {"method": "DELETE", "path": "/admin/agents/nobody", "headers": A},
        {"method": "GET", "path": "/whoami", "headers": K("sales-2")},
        {"method": "GET", "path": "/whoami", "headers": K("log-2")},
        {"method": "DELETE", "path": "/admin/rules/component_supply_risk", "headers": A},
        {"method": "DELETE", "path": "/admin/rules/component_supply_risk", "headers": A},
        {"method": "POST", "path": "/admin/replay", "headers": A},
        {"method": "POST", "path": "/admin/replay", "json": {}, "headers": A, "settle": True},
        # MCP
        {"method": "POST", "path": "/mcp", "json": mcp("tools/list"), "headers": J},
        {"method": "POST", "path": "/mcp", "json": mcp("initialize", {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "t", "version": "1"}}), "headers": {**K("log-1"), **J}},
        {"method": "POST", "path": "/mcp", "json": mcp("initialize", {"protocolVersion": "1999-01-01"}), "headers": {**K("log-1"), **J}},
        {"method": "POST", "path": "/mcp", "json": mcp("initialize", {"protocolVersion": "2024-11-05"}), "headers": {**K("log-1"), **J}},
        {"method": "POST", "path": "/mcp", "json": mcp("tools/list"), "headers": {**K("log-1"), **J}},
        {"method": "POST", "path": "/mcp", "json": mcp("ping", id_="abc"), "headers": {**K("log-1"), **J, "Mcp-Session-Id": "s1"}},
        {"method": "POST", "path": "/mcp", "json": {"jsonrpc": "2.0", "method": "notifications/initialized"}, "headers": {**K("log-1"), **J}},
        {"method": "POST", "path": "/mcp", "json": {"jsonrpc": "2.0", "method": "unknown/notification"}, "headers": {**K("log-1"), **J}},
        {"method": "POST", "path": "/mcp", "json": mcp("unknown/method"), "headers": {**K("log-1"), **J}},
        {"method": "POST", "path": "/mcp", "json": mcp("resources/list"), "headers": {**K("log-1"), **J}},
        {"method": "POST", "path": "/mcp", "json": mcp("resources/read", {"uri": "x://y"}), "headers": {**K("log-1"), **J}},
        {"method": "POST", "path": "/mcp", "json": mcp("resources/templates/list"), "headers": {**K("log-1"), **J}},
        {"method": "POST", "path": "/mcp", "json": mcp("prompts/list"), "headers": {**K("log-1"), **J}},
        {"method": "POST", "path": "/mcp", "json": mcp("prompts/get", {"name": "x"}), "headers": {**K("log-1"), **J}},
        {"method": "POST", "path": "/mcp", "json": mcp("logging/setLevel", {"level": "info"}), "headers": {**K("log-1"), **J}},
        {"method": "POST", "path": "/mcp", "json": mcp("ping", params=[1]), "headers": {**K("log-1"), **J}},
        {"method": "POST", "path": "/mcp", "json": mcp("ping", id_=1.5), "headers": {**K("log-1"), **J}},
        {"method": "POST", "path": "/mcp", "json": mcp("ping", id_=True), "headers": {**K("log-1"), **J}},
        {"method": "POST", "path": "/mcp", "json": mcp("ping", id_=None), "headers": {**K("log-1"), **J}},
        {"method": "POST", "path": "/mcp", "json": {"jsonrpc": "1.0", "id": 3, "method": "ping"}, "headers": {**K("log-1"), **J}},
        {"method": "POST", "path": "/mcp", "json": {"id": 3}, "headers": {**K("log-1"), **J}},
        {"method": "POST", "path": "/mcp", "json": "just a string", "headers": {**K("log-1"), **J}},
        {"method": "POST", "path": "/mcp", "json": [], "headers": {**K("log-1"), **J}},
        {"method": "POST", "path": "/mcp", "json": [mcp("ping", id_=1), {"jsonrpc": "2.0", "method": "notifications/x"}, mcp("tools/list", id_=2)], "headers": {**K("log-1"), **J}},
        {"method": "POST", "path": "/mcp", "json": [{"jsonrpc": "2.0", "method": "notifications/x"}], "headers": {**K("log-1"), **J}},
        {"method": "POST", "path": "/mcp", "data": "{nope", "headers": {**K("log-1"), **J, "Content-Type": "application/json"}},
        {"method": "POST", "path": "/mcp", "json": mcp("ping"), "headers": {**K("log-1"), **J, "MCP-Protocol-Version": "1999-01-01"}},
        {"method": "POST", "path": "/mcp", "json": mcp("ping"), "headers": {**K("log-1"), **J, "MCP-Protocol-Version": "2025-03-26"}},
        {"method": "POST", "path": "/mcp", "json": mcp("ping"), "headers": {**K("log-1"), "Accept": "text/event-stream"}},
        {"method": "POST", "path": "/mcp", "json": mcp("ping"), "headers": {**K("log-1"), "Accept": "text/event-stream, application/json"}},
        {"method": "GET", "path": "/mcp", "headers": {**K("log-1"), **J}},
        {"method": "DELETE", "path": "/mcp", "headers": {**K("log-1"), "Mcp-Session-Id": "s1"}},
        {"method": "PATCH", "path": "/mcp", "headers": K("log-1")},
        {"method": "POST", "path": "/mcp", "json": mcp("tools/call", {"name": "mycelic_remember", "arguments": {"text": "Terminal 3 strike announced.", "topic": T, "idempotency_key": "mcp-1"}}),
         "headers": {**K("log-1"), **J}, "settle": True},
        {"method": "POST", "path": "/mcp", "json": mcp("tools/call", {"name": "mycelic_remember", "arguments": {"text": "x", "confidence": 2}}), "headers": {**K("log-1"), **J}},
        {"method": "POST", "path": "/mcp", "json": mcp("tools/call", {"name": "mycelic_remember", "arguments": {"topic": "no text"}}), "headers": {**K("log-1"), **J}},
        {"method": "POST", "path": "/mcp", "json": mcp("tools/call", {"name": "mycelic_remember", "arguments": {"text": "x", "bogus": 1}}), "headers": {**K("log-1"), **J}},
        {"method": "POST", "path": "/mcp", "json": mcp("tools/call", {"name": "mycelic_query", "arguments": {"query": "terminal strike"}}), "headers": {**K("log-1"), **J}},
        {"method": "POST", "path": "/mcp", "json": mcp("tools/call", {"name": "mycelic_query", "arguments": {"query": "terminal strike", "k": 2.7, "min_layer": "agent", "scope": "northwind", "topic": T, "entity": None}}), "headers": {**K("log-1"), **J}},
        {"method": "POST", "path": "/mcp", "json": mcp("tools/call", {"name": "mycelic_query", "arguments": {"query": "x", "k": "3"}}), "headers": {**K("log-1"), **J}},
        {"method": "POST", "path": "/mcp", "json": mcp("tools/call", {"name": "mycelic_query", "arguments": {"query": "x", "min_layer": "nope"}}), "headers": {**K("log-1"), **J}},
        {"method": "POST", "path": "/mcp", "json": mcp("tools/call", {"name": "mycelic_query", "arguments": {"query": "x", "scope": "northwind/emea/nw-gmbh/commercial"}}), "headers": {**K("log-1"), **J}},
        {"method": "POST", "path": "/mcp", "json": mcp("tools/call", {"name": "mycelic_query", "arguments": {}}), "headers": {**K("log-1"), **J}},
        {"method": "POST", "path": "/mcp", "json": mcp("tools/call", {"name": "mycelic_get_memory", "arguments": {"memory_id": "{mem:log-1:mcp-1}"}}), "headers": {**K("sales-1"), **J}},
        {"method": "POST", "path": "/mcp", "json": mcp("tools/call", {"name": "mycelic_get_memory", "arguments": {"memory_id": "{mem:log-1:mcp-1}"}}), "headers": {**K("log-1"), **J}},
        {"method": "POST", "path": "/mcp", "json": mcp("tools/call", {"name": "mycelic_get_memory", "arguments": {"memory_id": "{mem:log-1:mcp-1}"}}), "headers": {**K("events-only"), **J}},
        {"method": "POST", "path": "/mcp", "json": mcp("tools/call", {"name": "mycelic_lineage", "arguments": {"memory_id": "{mem:log-1:mcp-1}"}}), "headers": {**K("log-1"), **J}},
        {"method": "POST", "path": "/mcp", "json": mcp("tools/call", {"name": "mycelic_lineage", "arguments": {"memory_id": "nope"}}), "headers": {**K("log-1"), **J}},
        {"method": "POST", "path": "/mcp", "json": mcp("tools/call", {"name": "mycelic_status", "arguments": {}}), "headers": {**K("log-1"), **J}},
        {"method": "POST", "path": "/mcp", "json": mcp("tools/call", {"name": "mycelic_status"}), "headers": {**K("acme-1"), **J}},
        {"method": "POST", "path": "/mcp", "json": mcp("tools/call", {"name": "mycelic_status", "arguments": {}}), "headers": {**A, **J}},
        {"method": "POST", "path": "/mcp", "json": mcp("tools/call", {"name": "mycelic_remember", "arguments": {"text": "admin"}}), "headers": {**A, **J}},
        {"method": "POST", "path": "/mcp", "json": mcp("tools/call", {"name": "nope", "arguments": {}}), "headers": {**K("log-1"), **J}},
        {"method": "POST", "path": "/mcp", "json": mcp("tools/call", {"name": "mycelic_status", "arguments": [1]}), "headers": {**K("log-1"), **J}},
        {"method": "POST", "path": "/mcp", "json": mcp("tools/call", {"arguments": {}}), "headers": {**K("log-1"), **J}},
        {"method": "POST", "path": "/mcp", "json": [mcp("ping", id_=i) for i in range(101)], "headers": {**K("log-1"), **J}},
        {"method": "POST", "path": "/mcp", "json": [mcp("ping", id_=i) for i in range(5)], "headers": {**K("log-1"), **J}},
        {"method": "GET", "path": "/admin/status", "headers": A},
        {"method": "GET", "path": "/health", "headers": A},
    ]
    return {
        "main": {"overrides": {}, "requests": main},
        "rate_limit": {"overrides": {"rate_limit_rps": 0.001, "rate_limit_burst": 3, "trust_proxy_headers": True}, "requests": [
            {"method": "GET", "path": "/whoami", "headers": A}, {"method": "GET", "path": "/whoami", "headers": A},
            {"method": "GET", "path": "/whoami", "headers": A}, {"method": "GET", "path": "/whoami", "headers": A},
            *[{"method": "GET", "path": "/whoami", "headers": {"Authorization": "Bearer mk_nobody.x", "X-Forwarded-For": f"10.0.0.{i}, 203.0.113.9"}} for i in range(5)],
            {"method": "GET", "path": "/health", "headers": {"X-Forwarded-For": "198.51.100.200"}},
            {"method": "POST", "path": "/admin/agents", "json": reg_body("log-1"), "headers": {**A, "X-Forwarded-For": "198.51.100.1"}},
            {"method": "GET", "path": "/whoami", "headers": {**A, "X-Forwarded-For": "198.51.100.2"}},
        ]},
        "rate_limit_mcp": {"overrides": {"rate_limit_rps": 0.001, "rate_limit_burst": 3, "trust_proxy_headers": True}, "requests": [
            {"method": "POST", "path": "/admin/agents", "json": reg_body("log-2"), "headers": {**A, "X-Forwarded-For": "198.51.100.9"}, "capture_key": "log-2"},
            {"method": "POST", "path": "/mcp", "json": [mcp("ping", id_=1), mcp("ping", id_=2)], "headers": {**K("log-2"), **J, "X-Forwarded-For": "198.51.100.1"}},
            {"method": "GET", "path": "/whoami", "headers": {**K("log-2"), "X-Forwarded-For": "198.51.100.2"}},
            {"method": "GET", "path": "/whoami", "headers": {**K("log-2"), "X-Forwarded-For": "198.51.100.3"}},
            {"method": "POST", "path": "/mcp", "json": [mcp("ping", id_=1)] * 5, "headers": {**A, **J, "X-Forwarded-For": "198.51.100.4"}},
        ]},
        "proxy_hops": {"overrides": {"rate_limit_rps": 0.001, "rate_limit_burst": 1, "trust_proxy_headers": True, "trusted_proxy_hops": 2}, "requests": [
            {"method": "GET", "path": "/whoami", "headers": {"Authorization": "Bearer mk_x.y", "X-Forwarded-For": "1.1.1.1, 2.2.2.2, 3.3.3.3"}},
            {"method": "GET", "path": "/whoami", "headers": {"Authorization": "Bearer mk_x.y", "X-Forwarded-For": "9.9.9.9, 2.2.2.2, 4.4.4.4"}},
            {"method": "GET", "path": "/whoami", "headers": {"Authorization": "Bearer mk_x.y", "X-Forwarded-For": "5.5.5.5"}},
            {"method": "GET", "path": "/whoami", "headers": {"Authorization": "Bearer mk_x.y", "X-Forwarded-For": "6.6.6.6"}},
        ]},
        "hosts": {"overrides": {"allowed_hosts": ["mycelic.example.com"], "cors_origins": ["https://app.example.com"]}, "requests": [
            {"method": "GET", "path": "/health", "headers": {"Host": "127.0.0.1:8080"}},
            {"method": "GET", "path": "/ready", "headers": {"Host": "127.0.0.1:8080"}},
            {"method": "GET", "path": "/whoami", "headers": {"Host": "127.0.0.1:8080", **A}},
            {"method": "GET", "path": "/whoami", "headers": {"Host": "mycelic.example.com", **A}},
            {"method": "GET", "path": "/whoami", "headers": {"Host": "MYCELIC.example.com:443", **A}},
            {"method": "OPTIONS", "path": "/memory", "headers": {"Host": "mycelic.example.com"}},
            {"method": "POST", "path": "/mcp", "json": mcp("ping"), "headers": {"Host": "mycelic.example.com", **A, **J, "Origin": "https://evil.example.com"}},
            {"method": "POST", "path": "/mcp", "json": mcp("ping"), "headers": {"Host": "mycelic.example.com", **A, **J, "Origin": "https://app.example.com"}},
        ]},
        "metrics_token": {"overrides": {"metrics_token": "metrics-token-0123456789abcdef0123456789"}, "requests": [
            {"method": "GET", "path": "/metrics", "status_only": True},
            {"method": "GET", "path": "/metrics", "headers": A, "status_only": True},
            {"method": "GET", "path": "/metrics", "headers": {"Authorization": "Bearer metrics-token-0123456789abcdef0123456789"}, "status_only": True},
        ]},
        "metrics_open": {"overrides": {}, "requests": [
            {"method": "GET", "path": "/metrics", "status_only": True},
            {"method": "GET", "path": "/metrics", "headers": {"Authorization": "Bearer mk_x.y"}, "status_only": True},
            {"method": "GET", "path": "/metrics", "headers": A, "status_only": True},
        ]},
    }


HEADER_NAMES = ("content-type", "www-authenticate", "cache-control", "allow", "mcp-protocol-version", "mcp-session-id")


async def gen_api() -> None:
    from aiohttp.test_utils import TestClient, TestServer

    from mycelic.api import create_app

    configs = api_requests()
    results = {}
    for name, cfg in configs.items():
        tmp = tempfile.mkdtemp()
        det = Det(tmp, cfg["overrides"])
        s = det.service
        await s.start()
        client = TestClient(TestServer(create_app(s)))
        await client.start_server()
        keys: dict[str, str] = {}
        outs = []
        try:
            for req in cfg["requests"]:
                path, headers, body = substitute(req, keys, s.store)
                kwargs: dict[str, Any] = {"headers": headers}
                if "json" in req:
                    kwargs["json"] = body
                if "data" in req:
                    kwargs["data"] = req["data"].encode("utf-8")
                if "data_repeat" in req:
                    kwargs["data"] = (req["data_repeat"][0] * req["data_repeat"][1]).encode("utf-8")
                r = await client.request(req["method"], path, **kwargs)
                raw = await r.read()
                out: dict[str, Any] = {"status": r.status}
                if not req.get("status_only"):
                    out["headers"] = {h: ("<present>" if h == "mcp-session-id" else r.headers[h]) for h in HEADER_NAMES if h in r.headers}
                    try:
                        out["json"] = normalize(json.loads(raw)) if raw else None
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        out["text"] = raw.decode("utf-8", "replace")
                if req.get("capture_key") and r.status in (200, 201):
                    keys[req["capture_key"]] = json.loads(raw)["api_key"]
                if req.get("settle"):
                    assert await s.wait_idle(30)
                outs.append(out)
        finally:
            await client.close()
            await s.close()
        results[name] = {"overrides": cfg["overrides"], "requests": cfg["requests"], "responses": outs}
    write("api.json", {"fixed_now": FIXED_NOW, "admin_token": ADMIN_TOKEN, "ignore_keys": sorted(IGNORE_KEYS),
                       "ignore_paths": [list(p) for p in sorted(IGNORE_PATHS)], "header_names": list(HEADER_NAMES), "configs": results})


def substitute(req: dict[str, Any], keys: dict[str, str], store: MycelicStore) -> tuple[str, dict[str, str], Any]:
    def sub(s: str) -> str:
        s = s.replace("{admin}", ADMIN_TOKEN)
        for who, key in keys.items():
            s = s.replace("{key:%s}" % who, key)
        while "{mem:" in s:
            i = s.index("{mem:")
            j = s.index("}", i)
            agent, idem = s[i + 5:j].rsplit(":", 1)
            s = s[:i] + f"mem_{content_hash(agent, idem)[:22]}" + s[j + 1:]
        while "{select:" in s:
            i = s.index("{select:")
            j = s.index("}", i)
            layer = s[i + 8:j]
            row = store._conn.execute("SELECT memory_id FROM memories WHERE layer=? AND status='active' ORDER BY rid LIMIT 1", (layer,)).fetchone()
            s = s[:i] + (row[0] if row else "none") + s[j + 1:]
        return s

    def walk(v: Any) -> Any:
        if isinstance(v, str):
            return sub(v)
        if isinstance(v, list):
            return [walk(x) for x in v]
        if isinstance(v, dict):
            return {k: walk(x) for k, x in v.items()}
        return v

    return sub(req["path"]), {k: sub(v) for k, v in (req.get("headers") or {}).items()}, walk(req.get("json"))


async def amain() -> None:
    gen_textutil()
    gen_bm25()
    gen_retrieval()
    gen_visibility()
    gen_ids()
    gen_hierarchy()
    gen_iso()
    gen_config()
    gen_coordination()
    gen_pyjson()
    gen_rules()
    await gen_scenarios()
    await gen_api()


if __name__ == "__main__":
    asyncio.run(amain())
