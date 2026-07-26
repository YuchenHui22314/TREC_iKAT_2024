"""Thin HTTP client for the iKAT'26 user-simulation API (Sim.API).

Schema is AUTHORITATIVE from the live OpenAPI spec (`/simulation/openapi.json`,
fetched 2026-06-10). Endpoints:

    GET  /auth/verify                      -> {team_id, ...}
    GET  /budget/check                     -> remaining debug/run budget
    POST /{mode}/start    {StartRequest}   -> UserUtteranceMessage   (first turn)
    POST /{mode}/continue {AssistantResponseMessage} -> UserUtteranceMessage (next turn)
    GET  /{mode}/session                   -> crash-recovery: current session state
    GET  /run/status?run_id=               -> run progress
    GET  /run/dump?run_id=                 -> the run the API assembled from our payloads
    GET  /run/dump-all                     -> all assembled runs

`{mode}` is "debug" (playground, 100 conversations, NOT scored) or "run"
(official, 6 runs).

Submission body (`AssistantResponseMessage`) has EXACTLY four keys:
    run_id   : str   (required)
    response : str   (required, <=512 spaCy tokens)
    citations: {str: number} | None   (optional; "docid:passage" -> score, top-10)
    meta     : object | None           (optional, free-form)
There is NO first-class ptkb / ptkb_provenance / persona field — a PTKB list, if we
emit one, goes under `meta`.

SECURITY: the bearer token is read ONLY from the env var IKAT_SIM_TOKEN. It is never
hardcoded, printed, logged, or written to disk. `__repr__` does not expose it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

import requests


DEFAULT_BASE_URL = "https://scai-ikat26.webis.de/simulation"
TOKEN_ENV_VAR = "IKAT_SIM_TOKEN"


# --------------------------------------------------------------------------- #
# Wire dataclasses
# --------------------------------------------------------------------------- #
@dataclass
class HistoryItem:
    """One past turn in the conversation. `role` is "user" | "assistant"
    ("assistant" content = a response WE sent on a previous turn)."""
    role: str
    content: str


@dataclass
class UserUtteranceMessage:
    """API -> us. Returned by /start and every /continue."""
    timestamp: Optional[str] = None
    run_id: Optional[str] = None
    topic_id: Optional[str] = None
    user_id: Optional[str] = None
    utterance: str = ""
    history: List[HistoryItem] = field(default_factory=list)
    last_response_of_session: bool = False
    last_response_of_run: bool = False
    user_meta: Dict[str, Any] = field(default_factory=dict)
    # keep the untouched server payload for inspection / forward-compat
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_json(cls, d: Dict[str, Any]) -> "UserUtteranceMessage":
        hist = []
        for h in d.get("history", []) or []:
            if isinstance(h, dict):
                hist.append(HistoryItem(role=h.get("role", ""), content=h.get("content", "")))
        return cls(
            timestamp=d.get("timestamp"),
            run_id=d.get("run_id"),
            topic_id=d.get("topic_id"),
            user_id=d.get("user_id"),
            utterance=d.get("utterance", "") or "",
            history=hist,
            last_response_of_session=bool(d.get("last_response_of_session", False)),
            last_response_of_run=bool(d.get("last_response_of_run", False)),
            user_meta=d.get("user_meta", {}) or {},
            raw=d,
        )


@dataclass
class AssistantResponseMessage:
    """us -> API. The /continue body. Exactly these four keys."""
    run_id: str
    response: str
    citations: Optional[Dict[str, float]] = None
    meta: Optional[Dict[str, Any]] = None

    def to_json(self) -> Dict[str, Any]:
        body: Dict[str, Any] = {"run_id": self.run_id, "response": self.response}
        if self.citations is not None:
            body["citations"] = self.citations
        if self.meta is not None:
            body["meta"] = self.meta
        return body


# --------------------------------------------------------------------------- #
# Client
# --------------------------------------------------------------------------- #
class SimAPIError(RuntimeError):
    pass


class SimClient:
    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        mode: str = "debug",
        timeout: float = 120.0,
        token: Optional[str] = None,
        verify: Optional[bool] = None,
    ):
        if mode not in ("debug", "run"):
            raise ValueError(f"mode must be 'debug' or 'run', got {mode!r}")
        self.base_url = base_url.rstrip("/")
        self.mode = mode
        self.timeout = timeout
        tok = token if token is not None else os.environ.get(TOKEN_ENV_VAR)
        if not tok:
            raise SimAPIError(
                f"No Sim.API token. Export it as the env var {TOKEN_ENV_VAR} "
                "(it must never be hardcoded/committed)."
            )
        self._session = requests.Session()
        # The token lives only in this header; never logged or repr'd.
        self._session.headers.update({"Authorization": f"Bearer {tok}"})
        # TLS verification ON by default. The webis host can present a hostname-mismatched cert
        # (CN=web.webis.de served for scai-ikat26.webis.de — a server-side vhost glitch, the LE
        # cert is otherwise valid); set verify=False or env IKAT_SIM_INSECURE=1 to keep submitting.
        if verify is None:
            verify = os.environ.get("IKAT_SIM_INSECURE", "").lower() not in ("1", "true", "yes")
        self._session.verify = verify
        if not verify:
            try:
                import urllib3
                urllib3.disable_warnings()
            except Exception:
                pass

    def __repr__(self) -> str:  # never leak the token
        return f"SimClient(base_url={self.base_url!r}, mode={self.mode!r})"

    # --- low-level ---------------------------------------------------------- #
    def _url(self, path: str) -> str:
        return f"{self.base_url}/{path.lstrip('/')}"

    def _request(self, method: str, path: str, **kw) -> Any:
        kw.setdefault("timeout", self.timeout)
        try:
            resp = self._session.request(method, self._url(path), **kw)
        except requests.RequestException as e:
            raise SimAPIError(f"{method} {path} failed: {e}") from e
        if resp.status_code >= 400:
            # body may carry a useful error; the token is never in the body
            raise SimAPIError(f"{method} {path} -> HTTP {resp.status_code}: {resp.text[:1000]}")
        if not resp.content:
            return None
        try:
            return resp.json()
        except ValueError:
            return resp.text

    # --- read-only helpers (NO conversation budget) ------------------------- #
    def verify(self) -> Dict[str, Any]:
        """GET /auth/verify -> team identity. Confirms the token works."""
        return self._request("GET", "/auth/verify")

    def budget_check(self) -> Dict[str, Any]:
        """GET /budget/check -> remaining debug/run budget."""
        return self._request("GET", "/budget/check")

    def session(self, run_id: str) -> Dict[str, Any]:
        """GET /{mode}/session?run_id= -> current session state (crash recovery).
        The API requires the run_id query param."""
        return self._request("GET", f"/{self.mode}/session", params={"run_id": run_id})

    def resume(self, run_id: str) -> "UserUtteranceMessage":
        """Resume an ALREADY-STARTED run: GET /{mode}/session parsed as the next
        UserUtteranceMessage awaiting our response. Use when /{mode}/start 412s (the run
        name already exists) — drive /continue from this turn. Read-only, no budget cost."""
        return UserUtteranceMessage.from_json(self.session(run_id))

    def run_status(self, run_id: str) -> Dict[str, Any]:
        return self._request("GET", "/run/status", params={"run_id": run_id})

    def run_dump(self, run_id: str) -> Dict[str, Any]:
        """GET /run/dump -> the run the API ASSEMBLED from our per-turn payloads."""
        return self._request("GET", "/run/dump", params={"run_id": run_id})

    def run_dump_all(self) -> Dict[str, Any]:
        return self._request("GET", "/run/dump-all")

    # --- conversation lifecycle (COSTS budget) ------------------------------ #
    def start(
        self, run_id: str, description: str = "", extra: Optional[Dict[str, Any]] = None
    ) -> UserUtteranceMessage:
        """POST /{mode}/start -> first user utterance.

        Body is `RunMetaMessage` = {run_id (req), description (req), extra (free-form)}.
        NOTE: there is NO `track_persona` field in the real schema (it was a research
        artifact). Anything extra goes in `extra` (free-form, may be ignored by the API).
        """
        body: Dict[str, Any] = {"run_id": run_id, "description": description}
        if extra:
            body["extra"] = extra
        d = self._request("POST", f"/{self.mode}/start", json=body)
        return UserUtteranceMessage.from_json(d)

    def continue_(self, msg: AssistantResponseMessage) -> UserUtteranceMessage:
        """POST /{mode}/continue -> next user utterance (advances the turn)."""
        d = self._request("POST", f"/{self.mode}/continue", json=msg.to_json())
        return UserUtteranceMessage.from_json(d)
