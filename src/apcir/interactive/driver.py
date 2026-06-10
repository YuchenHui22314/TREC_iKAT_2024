"""Sim.API driver: drives the start/continue loop, calls OUR search server per turn.

Flow per conversation (session):
    start(run_id, description) -> UserUtteranceMessage
    loop:
        POST our /search {utterance, history, ptkb, ids}  -> {response, citations, ...}
        continue_(AssistantResponseMessage{run_id, response, citations, meta})
        -> next UserUtteranceMessage
        stop the session at last_response_of_session; stop the run at last_response_of_run.

The official run is ASSEMBLED BY THE SIM.API from our per-turn payloads — we do NOT
upload a run file and do NOT call the old generate_and_save_ikat_submission. Locally we
save per-turn TREC rankings + a per-session JSON (for our own inspection) under
results/ClueWeb_ikat/<topics>/{ranking,interactive}/ with topics in
{ikat_26_sim_debug, ikat_26_sim_run} so they never collide with offline eval.

PTKB: the search server selects relevant statements (returned as ptkb_provenance); the
driver keeps a PTKBStore only for DIAGNOSTICS — whether user_meta.ptkb is present and
whether it changes across turns (this decides if the PTKB seam is worth lighting up).
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional

import requests

from .sim_client import SimClient, SimAPIError, AssistantResponseMessage, UserUtteranceMessage
from .ptkb_store import PTKBStore


def _now_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def _history_contents(msg: UserUtteranceMessage) -> List[str]:
    """API history -> interleaved [u1, r1, u2, r2, ...] content list (our server's
    fullconv_ctx). Each history item is a {role, content} string map."""
    return [h.content for h in msg.history]


def _ptkb_from_meta(user_meta: Dict[str, Any]) -> List[str]:
    return PTKBStore._extract_ptkb_list(user_meta)


class Driver:
    def __init__(self, args):
        self.args = args
        self.client = SimClient(base_url=args.base_url, mode=args.mode)
        self.server_url = args.server_url.rstrip("/")
        self.topics_tag = f"ikat_26_sim_{args.mode}"   # ikat_26_sim_debug | ikat_26_sim_run
        self.out_root = os.path.join(args.results_dir, "ClueWeb_ikat", self.topics_tag)
        os.makedirs(os.path.join(self.out_root, "ranking"), exist_ok=True)
        os.makedirs(os.path.join(self.out_root, "interactive"), exist_ok=True)
        self.run_id = args.run_id or f"rali_{args.mode}_{_now_tag()}"

    # --- our search server -------------------------------------------------- #
    def _search(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        r = requests.post(f"{self.server_url}/search", json=payload, timeout=600)
        r.raise_for_status()
        return r.json()

    def _continue_resilient(self, body: AssistantResponseMessage) -> Optional[UserUtteranceMessage]:
        """POST /continue with one retry. The iKAT user-simulator can return HTTP 500
        on its own LLM error (e.g. when stuck in a degenerate loop on weak responses).
        Returns None if it keeps failing so the caller can finalize gracefully instead
        of crashing."""
        for attempt in (1, 2):
            try:
                return self.client.continue_(body)
            except SimAPIError as e:
                print(f"[driver] /continue attempt {attempt} failed: {e}")
                if attempt == 2:
                    return None
                time.sleep(2)
        return None

    # --- persistence -------------------------------------------------------- #
    def _save_ranking(self, qid: str, hits: List[List[Any]]):
        path = os.path.join(self.out_root, "ranking", f"{self.run_id}_{qid}.txt")
        with open(path, "w") as f:
            for rank, (docid, score) in enumerate(hits, 1):
                f.write(f"{qid} Q0 {docid} {rank} {score} {self.run_id}\n")

    def _save_session(self, session_id: str, records: List[Dict[str, Any]]):
        path = os.path.join(self.out_root, "interactive", f"{self.run_id}_{session_id}.json")
        with open(path, "w") as f:
            json.dump({"run_id": self.run_id, "session_id": session_id, "turns": records},
                      f, indent=2, ensure_ascii=False)
        return path

    # --- main loop ---------------------------------------------------------- #
    def run(self) -> int:
        team = self.client.verify()
        budget_before = self.client.budget_check()
        print(f"[driver] team={team} run_id={self.run_id} mode={self.args.mode}")
        print(f"[driver] budget before: {json.dumps(budget_before, ensure_ascii=False)}")

        sessions_done = 0
        msg = self.client.start(self.run_id, self.args.description,
                                extra={"track_persona": bool(self.args.track_persona)})
        session_records: List[Dict[str, Any]] = []
        ptkb = PTKBStore(conversation_id=f"{msg.topic_id}-{msg.user_id}")
        turn_index = 0
        cur_session_id = f"{msg.topic_id}-{msg.user_id}"

        while True:
            ptkb.update(msg.user_meta)
            qid = f"{msg.topic_id}-{msg.user_id}-{turn_index}"
            payload = {
                "utterance": msg.utterance,
                "history": _history_contents(msg),
                "ptkb": _ptkb_from_meta(msg.user_meta),
                "topic_id": msg.topic_id or "0",
                "user_id": msg.user_id or "0",
                "turn_index": turn_index,
            }
            t0 = time.time()
            sr = self._search(payload)
            dt = time.time() - t0

            # persist locally
            self._save_ranking(qid, sr.get("hits", []))
            session_records.append({
                "turn_index": turn_index, "qid": qid,
                "utterance": msg.utterance,
                "response": sr.get("response", ""),
                "citations": sr.get("citations", {}),
                "ptkb_provenance": sr.get("ptkb_provenance", []),
                "latency_s": round(dt, 2),
                "last_response_of_session": msg.last_response_of_session,
            })
            print(f"[driver] turn {qid}: {len(sr.get('citations', {}))} citations, "
                  f"{dt:.1f}s, eos={msg.last_response_of_session} eor={msg.last_response_of_run} "
                  f"resp[:80]={sr.get('response','')[:80]!r}")

            # build the submission body; PTKB rides in free-form meta (best-effort)
            meta = None
            prov = sr.get("ptkb_provenance") or []
            if prov:
                meta = {"ptkb_provenance": prov}
            body = AssistantResponseMessage(
                run_id=self.run_id, response=sr.get("response", ""),
                citations=sr.get("citations") or None, meta=meta)

            last_session = msg.last_response_of_session
            last_run = msg.last_response_of_run

            # send our response; the API advances and returns the NEXT turn.
            # On a persistent /continue failure (simulator 500), finalize gracefully.
            next_msg = self._continue_resilient(body)
            if next_msg is None:
                p = self._save_session(cur_session_id, session_records)
                print(f"[driver] /continue failed after retry; saved partial session "
                      f"{cur_session_id} ({len(session_records)} turns) -> {p} | "
                      f"ptkb diag: {ptkb.diagnostics()}")
                sessions_done += 1
                break

            # safety: cap turns/session so a degenerate simulator loop can't run forever
            if not last_session and (turn_index + 1) >= self.args.max_turns_per_session:
                p = self._save_session(cur_session_id, session_records)
                print(f"[driver] hit max_turns_per_session={self.args.max_turns_per_session}; "
                      f"saved session {cur_session_id} -> {p} | ptkb diag: {ptkb.diagnostics()}")
                sessions_done += 1
                break

            if last_session:
                p = self._save_session(cur_session_id, session_records)
                print(f"[driver] session {cur_session_id} done ({len(session_records)} turns) "
                      f"-> {p} | ptkb diag: {ptkb.diagnostics()}")
                sessions_done += 1
                if last_run:
                    print("[driver] last_response_of_run -> run complete.")
                    break
                if sessions_done >= self.args.max_conversations:
                    print(f"[driver] reached max_conversations={self.args.max_conversations}; stopping.")
                    break
                # next session bookkeeping
                session_records = []
                turn_index = 0
                msg = next_msg
                cur_session_id = f"{msg.topic_id}-{msg.user_id}"
                ptkb = PTKBStore(conversation_id=cur_session_id)
                continue

            if last_run:
                print("[driver] last_response_of_run -> run complete.")
                break

            msg = next_msg
            turn_index += 1

        budget_after = self.client.budget_check()
        print(f"[driver] budget after: {json.dumps(budget_after, ensure_ascii=False)}")
        print(f"[driver] sessions_done={sessions_done}")
        return 0


def run_driver(args) -> int:
    return Driver(args).run()
