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
