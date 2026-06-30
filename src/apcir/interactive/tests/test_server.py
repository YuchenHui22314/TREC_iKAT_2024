"""Endpoint tests (FastAPI TestClient) for the RALI Searcher app: /models, /activate,
/activate/status, /search guards. eager_load=False + injected pipeline (fake capacity); /activate
loads the real tiny qrecc_ance_mini in a background thread."""
import os
import threading
import time

from fastapi.testclient import TestClient

from apcir.interactive.capacity import IndexRegistry, CapacityManager
from apcir.interactive.pipeline import InteractivePipeline, PipelineConfig
from apcir.interactive.search_server import create_app

CONFIG_YAML = os.path.join(os.path.dirname(__file__), "..", "capacity_config.yaml")


def _client(free_ram=200.0):
    reg = IndexRegistry.from_yaml(CONFIG_YAML)
    cap = CapacityManager(reg, free_ram_fn=lambda: free_ram,
                          free_vram_fn=lambda: [24.0, 24.0, 24.0, 24.0])
    pipe = InteractivePipeline(PipelineConfig(), registry=reg, capacity=cap)
    app = create_app(PipelineConfig(), eager_load=False, pipeline=pipe)
    return TestClient(app), pipe


def test_doc_endpoint_409_without_docfetch_then_returns_text():
    """GET /doc/{docid} -> 409 until a doc-fetch (sparse) index is resident, then the passage text."""
    import json as _json
    client, pipe = _client()
    assert pipe._docfetch is None
    assert client.get("/doc", params={"docid": "anything"}).status_code == 409

    class _FakeDoc:
        def __init__(self, did):
            self._did = did

        def raw(self):
            return _json.dumps({"contents": f"text of {self._did}"})

    class _FakeDocfetch:
        def doc(self, did):
            return _FakeDoc(did)

    pipe._docfetch = _FakeDocfetch()
    r = client.get("/doc", params={"docid": "somedoc"})
    assert r.status_code == 200
    assert r.json() == {"docid": "somedoc", "text": "text of somedoc"}


def _wait_status(client, tid, timeout=12.0):
    st = {"state": "running"}
    for _ in range(int(timeout / 0.2)):
        st = client.get(f"/activate/status/{tid}").json()
        if st["state"] != "running":
            break
        time.sleep(0.2)
    return st


def test_models_endpoint():
    client, _ = _client()
    r = client.get("/models")
    assert r.status_code == 200
    body = r.json()
    assert any(u["name"] == "qrecc_ance_mini" for u in body["units"])
    assert body["resident"] == []
    assert body["free_ram_gb"] == 200.0


def test_activate_loads_mini_then_status_done():
    client, _ = _client()
    r = client.post("/activate", json={"units": ["qrecc_ance_mini"]})
    assert r.status_code == 200
    body = r.json()
    assert body["plan"]["to_load"] == ["qrecc_ance_mini"]
    st = _wait_status(client, body["task_id"])
    assert st["state"] == "done", st
    assert "qrecc_ance_mini" in st["resident"]
    assert client.get("/models").json()["resident"] == ["qrecc_ance_mini"]


def test_activate_refuses_when_wont_fit():
    client, _ = _client(free_ram=237.0)
    r = client.post("/activate", json={"units": ["clueweb_qwen"]})
    assert r.status_code == 409
    assert "clueweb_qwen" in r.json()["detail"]


def test_activate_status_unknown_404():
    client, _ = _client()
    assert client.get("/activate/status/does-not-exist").status_code == 404


def test_activate_unknown_unit_400():
    client, _ = _client()
    r = client.post("/activate", json={"units": ["bogus_unit"]})
    assert r.status_code == 400
    assert "bogus_unit" in r.json()["detail"]


def test_activate_rejects_concurrent():
    client, pipe = _client()
    started, release = threading.Event(), threading.Event()
    orig = pipe.set_active

    def slow(units, progress_cb=None):
        started.set()
        release.wait(5)
        return orig(units, progress_cb)

    pipe.set_active = slow
    r1 = client.post("/activate", json={"units": ["qrecc_ance_mini"]})
    assert r1.status_code == 200
    assert started.wait(3)                              # first is blocking inside set_active
    r2 = client.post("/activate", json={"units": ["qrecc_ance_mini"]})
    assert r2.status_code in (409, 423)                # rejected while one in-flight
    release.set()
    _wait_status(client, r1.json()["task_id"])


def test_search_without_active_units_409():
    client, _ = _client()
    r = client.post("/search", json={"utterance": "hello"})   # default legacy retrievers, none loaded
    assert r.status_code == 409


def test_search_empty_retrievers_422():
    client, _ = _client()
    r = client.post("/search", json={"utterance": "hi", "retrievers": []})
    assert r.status_code == 422                         # pydantic rejects empty retrievers
