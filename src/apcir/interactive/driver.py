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


# --- dynamic PTKB extraction (gpt-5-mini) ----------------------------------- #
# When a NEW user (persona) has no prior session, we must grow their PTKB from the
# conversation itself. Per turn we ask the LLM for any NEW durable, personally-relevant
# facts the user revealed that are not already in the current PTKB. Few-shot facts are
# taken from the iKAT-25 organiser-oracle carried-over facts (ptkb-update.json), e.g.
# "I have acid reflux." / "I have a deadline approaching."
PTKB_EXTRACT_PROMPT = """You maintain a user's Personal Text Knowledge Base (PTKB) for a personalized conversational assistant. A PTKB is a set of concise, first-person, present-tense statements capturing DURABLE, personally-relevant facts about the user that help personalize future answers --- across: preferences / likes & dislikes; goals & plans; occupation & background / life facts; habits & routines; health & physical conditions; skills & abilities; values & concerns; possessions; dietary / lifestyle constraints; and salient ongoing situations (e.g. an approaching deadline).

You are given the user's CURRENT PTKB and the LATEST turn of an ongoing conversation. Output any NEW personal facts the user revealed THIS turn that are (a) about the USER (not the world / the assistant's content), (b) durable or decision-relevant (not one-off chit-chat), and (c) NOT already entailed by the current PTKB.

Rules:
- One atomic fact per line, as "I ..." (first person, present tense, concise).
- Capture DURABLE traits / preferences / conditions --- NOT the user's immediate request or task
  this turn (e.g. "help me find a gift" / "I'm looking for ways to do X" is a TASK, not a fact).
- Do NOT extract the fine-grained CONTENT or WORDING of something being co-designed or built in the
  conversation (the exact phrases of a script/prompt, the individual items in a plan or list, tiny
  settings being tweaked together turn by turn). Those are artifacts of THIS task, not the user.
  Extract a fact ONLY if it would still describe the user OUTSIDE this conversation; prefer ONE
  general durable fact over many hyper-specific ones, and stop once the durable preference is captured.
- Use ONLY what the USER stated or clearly implied --- NEVER the assistant's suggestions,
  inferences, or diagnoses.
- The PTKB below ALREADY includes facts extracted earlier in this conversation. Do NOT restate,
  paraphrase, refine, or re-emit anything already entailed by it.
- If the user corrects/updates a prior fact, output the corrected fact.
- If there is NO new durable personal fact this turn, output exactly: NONE

### Example A (a real durable fact)
Current PTKB: 1. I want to increase my protein intake. 2. I eat dinner late at night.
User: "I'm planning meals, but I have to be careful --- spicy and acidic stuff really sets off my acid reflux."
Assistant: "Sure, here are reflux-friendly high-protein options..."
New PTKB facts:
I have acid reflux.

### Example B (a task / request is NOT a fact)
Current PTKB: 1. I enjoy hiking.
User: "Can you help me find ways to cheer up a friend who's feeling down?"
Assistant: "Here are ideas to support a friend..."
New PTKB facts:
NONE

### Example C (extract the user's fact, NOT the assistant's inference)
Current PTKB: (empty)
User: "My finger really hurts at the base after a hard climb yesterday."
Assistant: "That could be an A2 pulley injury; rest and ..."
New PTKB facts:
I have finger pain at the base after climbing.

### Example D (co-designed wording / micro-details are NOT durable facts)
Current PTKB: 1. I find short phone voice-prompts helpful for snack reminders.
User: "Great --- let's have it say 'Two breaths', then 'Notice the taste', and start with 'Phone away.'"
Assistant: "Done --- here is your 8-second prompt: ..."
New PTKB facts:
NONE

### Now do it.
Current PTKB:
{ptkb}

Latest turn:
User: {utterance}
Assistant: {response}

New PTKB facts (one per line, or NONE):"""


def _parse_extracted_facts(ans: str) -> List[str]:
    """Parse the extractor LLM output into a clean list of first-person facts ([] on NONE)."""
    facts = []
    for line in (ans or "").splitlines():
        s = line.strip().lstrip("-*0123456789. ").strip()
        if not s or s.upper() == "NONE":
            continue
        if not (s.lower().startswith("i ") or s.lower().startswith("i'")):
            continue  # keep only first-person "I ..." statements
        if len(s) > 200:
            continue
        facts.append(s)
    return facts


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
        # dynamic PTKB extraction (gpt-5-mini), accumulated per PERSONA (= topic_id minus the
        # trailing conversation number; user_id is just "planning-user-"+topic_id, redundant).
        self._persona_facts: Dict[str, List[str]] = {}
        self._llm = None
        if getattr(args, "extract_ptkb", False):
            from .llm_client import SharedLLMClient
            self._llm = SharedLLMClient(backend="openai", model="gpt-5-mini")
            print("[driver] dynamic PTKB extraction ON (gpt-5-mini)")

    def _extract_ptkb(self, current_ptkb: List[str], utterance: str, response: str) -> List[str]:
        """Ask gpt-5-mini for NEW durable persona facts revealed this turn (not in current_ptkb)."""
        prompt = PTKB_EXTRACT_PROMPT.format(
            ptkb=("\n".join(f"{i}. {s}" for i, s in enumerate(current_ptkb, 1)) or "(empty)"),
            utterance=utterance, response=(response or "")[:1200])
        try:
            ans, _ = self._llm.generate(prompt)
        except Exception as e:
            print(f"[extract] LLM error: {e}", flush=True)
            return []
        return _parse_extracted_facts(ans)

    # --- our search server -------------------------------------------------- #
    def _search(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        r = requests.post(f"{self.server_url}/search", json=payload, timeout=600)
        r.raise_for_status()
        return r.json()

    def _continue_resilient(self, body: AssistantResponseMessage) -> Optional[UserUtteranceMessage]:
        """POST /continue with one retry. The iKAT user-simulator can return HTTP 500
        on its own LLM error -- either a TRANSIENT API hiccup (rate limit / timeout) or a
        persistent degenerate-loop state. Retry several times with exponential backoff to ride out
        transient 500s; return None only if it keeps failing so the caller can finalize gracefully
        instead of crashing."""
        max_attempts = 5
        for attempt in range(1, max_attempts + 1):
            try:
                return self.client.continue_(body)
            except SimAPIError as e:
                print(f"[driver] /continue attempt {attempt}/{max_attempts} failed: {e}")
                if attempt == max_attempts:
                    return None
                time.sleep(2 ** attempt)  # 2,4,8,16s backoff
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
        try:
            msg = self.client.start(self.run_id, self.args.description,
                                    extra={"track_persona": bool(self.args.track_persona)})
        except SimAPIError as e:
            if "HTTP 412" not in str(e):
                raise
            # run name already registered (a prior partial launch): RESUME the open sessions
            # via /session instead of re-/start (which 412s). No /start => no extra run slot.
            print(f"[driver] /{self.args.mode}/start -> 412 (run exists); RESUMING via "
                  f"/{self.args.mode}/session", flush=True)
            msg = self.client.resume(self.run_id)
            if msg.last_response_of_run:
                print(f"[driver] run {self.run_id} already COMPLETE; nothing to resume.")
                return 0
        session_records: List[Dict[str, Any]] = []
        ptkb = PTKBStore(conversation_id=f"{msg.topic_id}-{msg.user_id}")
        turn_index = 0
        cur_session_id = f"{msg.topic_id}-{msg.user_id}"

        while True:
            ptkb.update(msg.user_meta)
            # [probe] raw cross-session identity: does the same user_id recur across sessions, and
            # is user_meta.ptkb a static base each session? grep '\[probe\]' the driver log; the
            # ptkb_hash is stable within this run process so a repeat user_id can be checked for a
            # changed PTKB. (Instrumentation for the dynamic-PTKB feasibility probe.)
            _pk = _ptkb_from_meta(msg.user_meta)
            print(f"[probe] turn={turn_index} run_id={msg.run_id!r} topic_id={msg.topic_id!r} "
                  f"user_id={msg.user_id!r} ptkb_n={len(_pk)} "
                  f"ptkb_hash={hash(tuple(sorted(_pk))) & 0xffffffff:08x} "
                  f"eos={msg.last_response_of_session} eor={msg.last_response_of_run}", flush=True)
            qid = f"{msg.topic_id}-{msg.user_id}-{turn_index}"
            persona = (msg.topic_id or "").rsplit("-", 1)[0]
            # [override B] for a returning persona, REPLACE the simulator's static base PTKB with
            # base + our accumulated extracted facts (the simulator never carries facts across a
            # persona's sessions, so the carry-over is ours to inject for personalized retrieval).
            payload_ptkb = list(_pk)
            if getattr(self.args, "override_ptkb", False) and self._persona_facts.get(persona):
                extra = [f for f in self._persona_facts[persona] if f not in payload_ptkb]
                payload_ptkb = _pk + extra
                print(f"[override] turn={turn_index} persona={persona!r} "
                      f"base={len(_pk)} +accumulated={len(extra)} -> {len(payload_ptkb)}", flush=True)
            payload = {
                "utterance": msg.utterance,
                "history": _history_contents(msg),
                "ptkb": payload_ptkb,
                "topic_id": msg.topic_id or "0",
                "user_id": msg.user_id or "0",
                "turn_index": turn_index,
            }
            t0 = time.time()
            sr = self._search(payload)
            dt = time.time() - t0

            # [extract] dynamic PTKB: pull NEW persona facts from this turn (logged for analysis;
            # accumulated per persona for the cross-session override — points A + C of the design).
            new_facts = None
            if self._llm is not None:
                # feed base + already-accumulated facts so the model dedupes against what we've
                # already extracted (it was re-extracting variants of the same fact every turn).
                current = _pk + [f for f in self._persona_facts.get(persona, []) if f not in _pk]
                new_facts = self._extract_ptkb(current, msg.utterance, sr.get("response", ""))
                if new_facts:
                    store = self._persona_facts.setdefault(persona, [])
                    for f in new_facts:
                        if f not in store:
                            store.append(f)
                print(f"[extract] turn={turn_index} persona={persona!r} new={new_facts} "
                      f"persona_total={len(self._persona_facts.get(persona, []))}", flush=True)

            # persist locally
            self._save_ranking(qid, sr.get("hits", []))
            session_records.append({
                "turn_index": turn_index, "qid": qid,
                "utterance": msg.utterance,
                "response": sr.get("response", ""),
                "citations": sr.get("citations", {}),
                "ptkb_provenance": sr.get("ptkb_provenance", []),
                # dynamic-PTKB (only meaningful with --extract_ptkb): what gpt-5-mini pulled THIS
                # turn, the running per-persona accumulation, and base-vs-sent PTKB sizes (the
                # latter shows the --override_ptkb carry-over: sent_n > base_n when facts are injected).
                "ptkb_extracted_this_turn": new_facts or [],
                "ptkb_accumulated": list(self._persona_facts.get(persona, [])),
                "ptkb_base_n": len(_pk),
                "ptkb_sent_n": len(payload_ptkb),
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
