"""SQLite data-layer tests (in-memory DB, no GPU/heavy deps)."""
from apcir.interactive.store import Store


def _store():
    return Store(":memory:")


def test_create_and_verify_user():
    s = _store()
    uid = s.create_user("alice", "pw123", is_admin=True)
    assert uid > 0
    u = s.verify_user("alice", "pw123")
    assert u and u["username"] == "alice" and u["is_admin"] == 1
    assert s.verify_user("alice", "wrong") is None          # wrong password
    assert s.verify_user("nobody", "pw123") is None          # unknown user


def test_password_is_hashed_not_plaintext():
    s = _store()
    s.create_user("alice", "secret")
    row = s._conn.execute("SELECT password_hash FROM users WHERE username='alice'").fetchone()
    assert "secret" not in row["password_hash"]               # not stored in the clear


def test_create_duplicate_user_raises_clear():
    s = _store()
    s.create_user("a", "p")
    try:
        s.create_user("a", "p2")
        assert False, "expected a clear duplicate-username error"
    except ValueError as e:
        assert "a" in str(e) and "exist" in str(e).lower()


def test_token_expires_after_ttl():
    clock = [1000.0]
    s = Store(":memory:", token_ttl_seconds=100, now_fn=lambda: clock[0])
    uid = s.create_user("a", "p")
    tok = s.create_token(uid)
    assert s.user_for_token(tok) == uid          # within TTL
    clock[0] = 1000.0 + 101                       # advance past TTL
    assert s.user_for_token(tok) is None          # expired (and purged)


def test_token_roundtrip_and_revoke():
    s = _store()
    uid = s.create_user("bob", "pw")
    tok = s.create_token(uid)
    assert s.user_for_token(tok) == uid
    s.delete_token(tok)
    assert s.user_for_token(tok) is None
    assert s.user_for_token("garbage") is None


def test_session_crud_is_user_scoped():
    s = _store()
    a = s.create_user("a", "p")
    b = s.create_user("b", "p")
    sid = s.create_session(a, "chat 1")
    assert sid > 0
    assert [x["title"] for x in s.list_sessions(a)] == ["chat 1"]
    assert s.rename_session(sid, a, "renamed")
    assert s.list_sessions(a)[0]["title"] == "renamed"
    # b cannot see, rename or delete a's session
    assert s.list_sessions(b) == []
    assert not s.rename_session(sid, b, "hax")
    assert not s.delete_session(sid, b)
    assert s.delete_session(sid, a)
    assert s.list_sessions(a) == []


def test_turns_persist_payload_and_cascade_delete():
    s = _store()
    a = s.create_user("a", "p")
    sid = s.create_session(a)
    s.add_turn(sid, 0, "hello", "hi there", {"citations": {"d1": 0.9}, "qr": "rewritten q"})
    s.add_turn(sid, 1, "more", "ok", {})
    turns = s.list_turns(sid)
    assert [t["utterance"] for t in turns] == ["hello", "more"]
    assert turns[0]["payload"]["citations"]["d1"] == 0.9
    assert turns[0]["payload"]["qr"] == "rewritten q"
    s.delete_session(sid, a)                                  # cascade removes turns
    assert s.list_turns(sid) == []


def test_ptkb_crud_is_user_scoped():
    s = _store()
    a = s.create_user("a", "p")
    b = s.create_user("b", "p")
    pid = s.add_ptkb(a, "likes hiking", source="extracted")
    assert [p["statement"] for p in s.list_ptkb(a)] == ["likes hiking"]
    assert s.list_ptkb(a)[0]["source"] == "extracted"
    assert not s.update_ptkb(pid, b, "hax")                  # b can't touch a's ptkb
    assert s.update_ptkb(pid, a, "likes mountain hiking")
    assert s.list_ptkb(a)[0]["statement"] == "likes mountain hiking"
    assert s.delete_ptkb(pid, a)
    assert s.list_ptkb(a) == []
