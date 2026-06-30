"""Auth + session-management endpoint tests (FastAPI TestClient + injected in-memory Store)."""
from fastapi.testclient import TestClient

from apcir.interactive.capacity import IndexRegistry, CapacityManager
from apcir.interactive.pipeline import InteractivePipeline, PipelineConfig
from apcir.interactive.search_server import create_app
from apcir.interactive.store import Store

import os

CONFIG_YAML = os.path.join(os.path.dirname(__file__), "..", "capacity_config.yaml")


def _app():
    store = Store(":memory:")
    store.create_user("admin", "pw", is_admin=True)
    reg = IndexRegistry.from_yaml(CONFIG_YAML)
    cap = CapacityManager(reg, free_ram_fn=lambda: 200.0, free_vram_fn=lambda: [24.0])
    pipe = InteractivePipeline(PipelineConfig(), registry=reg, capacity=cap)
    return TestClient(create_app(PipelineConfig(), eager_load=False, pipeline=pipe, store=store))


def _login(client, u="admin", p="pw"):
    tok = client.post("/auth/login", json={"username": u, "password": p}).json()["token"]
    return {"Authorization": f"Bearer {tok}"}


def test_login_and_me():
    client = _app()
    r = client.post("/auth/login", json={"username": "admin", "password": "pw"})
    assert r.status_code == 200
    tok = r.json()["token"]
    me = client.get("/auth/me", headers={"Authorization": f"Bearer {tok}"})
    assert me.status_code == 200 and me.json()["username"] == "admin"


def test_login_bad_password_401():
    client = _app()
    assert client.post("/auth/login", json={"username": "admin", "password": "x"}).status_code == 401


def test_me_requires_token():
    client = _app()
    assert client.get("/auth/me").status_code == 401
    assert client.get("/auth/me", headers={"Authorization": "Bearer garbage"}).status_code == 401


def test_logout_revokes_token():
    client = _app()
    H = _login(client)
    assert client.post("/auth/logout", headers=H).status_code == 200
    assert client.get("/auth/me", headers=H).status_code == 401


def test_sessions_crud_authed():
    client = _app()
    H = _login(client)
    sid = client.post("/sessions", json={"title": "c1"}, headers=H).json()["id"]
    assert [s["title"] for s in client.get("/sessions", headers=H).json()] == ["c1"]
    assert client.patch(f"/sessions/{sid}", json={"title": "c2"}, headers=H).status_code == 200
    assert client.get("/sessions", headers=H).json()[0]["title"] == "c2"
    assert client.delete(f"/sessions/{sid}", headers=H).status_code == 200
    assert client.get("/sessions", headers=H).json() == []


def test_sessions_require_auth():
    client = _app()
    assert client.get("/sessions").status_code == 401
    assert client.post("/sessions", json={"title": "x"}).status_code == 401


def test_ptkb_crud_authed():
    client = _app()
    H = _login(client)
    pid = client.post("/ptkb", json={"statement": "likes hiking"}, headers=H).json()["id"]
    assert [p["statement"] for p in client.get("/ptkb", headers=H).json()] == ["likes hiking"]
    assert client.get("/ptkb", headers=H).json()[0]["source"] == "manual"
    assert client.put(f"/ptkb/{pid}", json={"statement": "likes mountain hiking"},
                      headers=H).status_code == 200
    assert client.get("/ptkb", headers=H).json()[0]["statement"] == "likes mountain hiking"
    assert client.delete(f"/ptkb/{pid}", headers=H).status_code == 200
    assert client.get("/ptkb", headers=H).json() == []
    assert client.delete(f"/ptkb/{pid}", headers=H).status_code == 404   # already gone


def test_ptkb_requires_auth():
    client = _app()
    assert client.get("/ptkb").status_code == 401
    assert client.post("/ptkb", json={"statement": "x"}).status_code == 401


class _FakeResult:
    response = "an answer"
    citations = {"d1": 1.0}
    hits = [("d1", 1.0), ("d2", 0.5)]            # full fused ranking (TurnResult.hits is tuples)
    per_retriever = [{"retriever": "bm25", "hits": [["d1", 1.0]]}]
    shared_docs = {}
    reformulations = {"bm25": ["rewritten q"]}
    citation_spans = [{"n": 1, "docid": "d1", "start": 0, "end": 3}]


class _FakePipe:
    def extract_ptkb(self, current, utt, resp):
        return ["I am a vegetarian."]


def test_persist_turn_saves_and_extracts():
    from apcir.interactive.search_server import _persist_turn, SearchRequest
    store = Store(":memory:")
    uid = store.create_user("a", "p")
    sid = store.create_session(uid)
    req = SearchRequest(utterance="vegan?", session_id=sid, extract_ptkb=True, turn_index=0)
    _persist_turn(store, _FakePipe(), uid, req, _FakeResult())
    turns = store.list_turns(sid)
    assert len(turns) == 1 and turns[0]["utterance"] == "vegan?"
    assert turns[0]["response"] == "an answer"
    assert turns[0]["payload"]["citations"] == {"d1": 1.0}
    assert turns[0]["payload"]["per_retriever"][0]["retriever"] == "bm25"
    # the full fused passage list must be persisted (JSON-friendly [[docid, score]]) so a reloaded
    # session re-renders identically, not a degraded citations-only fallback.
    assert turns[0]["payload"]["hits"] == [["d1", 1.0], ["d2", 0.5]]
    assert [p["statement"] for p in store.list_ptkb(uid)] == ["I am a vegetarian."]
    assert store.list_ptkb(uid)[0]["source"] == "extracted"


def test_persist_turn_returns_new_extracted_facts():
    """_persist_turn returns (saved, new_facts) so /search can surface a per-turn 'learned …' line."""
    from apcir.interactive.search_server import _persist_turn, SearchRequest
    store = Store(":memory:")
    uid = store.create_user("a", "p")
    sid = store.create_session(uid)
    req = SearchRequest(utterance="vegan?", session_id=sid, extract_ptkb=True)
    saved, facts = _persist_turn(store, _FakePipe(), uid, req, _FakeResult())
    assert saved is True
    assert facts == ["I am a vegetarian."]                # only the NEW facts learned this turn
    # a second identical turn learns nothing new (deduped) -> empty list
    saved2, facts2 = _persist_turn(store, _FakePipe(), uid, req, _FakeResult())
    assert saved2 is True and facts2 == []


def test_persist_turn_auto_indexes_turns():
    from apcir.interactive.search_server import _persist_turn, SearchRequest
    store = Store(":memory:")
    uid = store.create_user("a", "p")
    sid = store.create_session(uid)
    assert _persist_turn(store, _FakePipe(), uid, SearchRequest(utterance="q1", session_id=sid),
                         _FakeResult())[0] is True
    _persist_turn(store, _FakePipe(), uid, SearchRequest(utterance="q2", session_id=sid), _FakeResult())
    turns = store.list_turns(sid)
    assert [t["idx"] for t in turns] == [0, 1]                # server-allocated, no client collision
    assert [t["utterance"] for t in turns] == ["q1", "q2"]


def test_persist_turn_dedups_extracted_ptkb():
    from apcir.interactive.search_server import _persist_turn, SearchRequest

    class DupPipe:
        def extract_ptkb(self, current, utt, resp):
            return ["I am vegetarian.", "I am vegetarian.", "I like hiking."]

    store = Store(":memory:")
    uid = store.create_user("a", "p")
    sid = store.create_session(uid)
    req = SearchRequest(utterance="q", session_id=sid, extract_ptkb=True)
    _persist_turn(store, DupPipe(), uid, req, _FakeResult())
    assert sorted(p["statement"] for p in store.list_ptkb(uid)) == ["I am vegetarian.", "I like hiking."]


def test_persist_turn_isolates_extract_failure():
    from apcir.interactive.search_server import _persist_turn, SearchRequest

    class BadPipe:
        def extract_ptkb(self, *a):
            raise RuntimeError("llm down")

    store = Store(":memory:")
    uid = store.create_user("a", "p")
    sid = store.create_session(uid)
    req = SearchRequest(utterance="q", session_id=sid, extract_ptkb=True)
    saved, facts = _persist_turn(store, BadPipe(), uid, req, _FakeResult())   # must NOT raise
    assert saved is True and facts == []                                      # extract failed -> none
    assert len(store.list_turns(sid)) == 1                                    # turn still saved
    assert store.list_ptkb(uid) == []                                        # extract failed -> none


def test_persist_turn_skips_when_unauthed_or_no_session():
    from apcir.interactive.search_server import _persist_turn, SearchRequest
    store = Store(":memory:")
    uid = store.create_user("a", "p")
    sid = store.create_session(uid)
    _persist_turn(store, _FakePipe(), uid, SearchRequest(utterance="x"), _FakeResult())   # no session
    _persist_turn(store, _FakePipe(), None, SearchRequest(utterance="x", session_id=sid), _FakeResult())
    assert store.list_turns(sid) == []
    # a session NOT owned by the user is ignored
    other = store.create_user("b", "p")
    _persist_turn(store, _FakePipe(), other, SearchRequest(utterance="x", session_id=sid), _FakeResult())
    assert store.list_turns(sid) == []
