"""OnlineRewriter — run the offline QR promptors at REQUEST time.

Mirrors the exact call patterns in `rewrite/rewrite.py` (build_turn_prompt -> LLM ->
parse_returned_text -> add_reformulation), but driven live per turn and returning the
rewritten query STRING(S) for retrieval. Reuses the promptor classes verbatim; the LLM
is the shared `SharedLLMClient` (OpenAI or local vLLM).

Supported qr_names (all rewrite-producing; pure level-judging EXCLUDED):
  rar, rar_personalized_cot[0|1|N], rar_non_personalized_cot[0|1|N],
  MQ4CS_persq, GtR (two-stage), ptkb_sum (three-stage).

`rewrite()` returns a LIST of query strings (1 for single-rewrite QRs; phi for GtR ->
the pipeline retrieves each and fuses them, see pipeline._fuse). Memoized per Turn.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List

# Models decorate the rewrite differently (Qwen: "**Re-written query:** … **Explanation:** …";
# others: bare). Normalize: strip a leading label and any trailing Explanation/Reason block +
# markdown bold, so a single-string rewrite is clean regardless of model.
_LABEL_RE = re.compile(
    r"^\s*\**\s*(re-?written query|rewritten query|rewrite|search query|query)\s*\**\s*:\s*\**\s*",
    re.IGNORECASE)
_TRAIL_RE = re.compile(
    r"\n\s*\**\s*(explanation|reasoning|reason|rationale|note)s?\b\s*\**\s*:.*",
    re.IGNORECASE | re.DOTALL)


def clean_rewrite(text: str) -> str:
    text = (text or "").strip()
    text = _LABEL_RE.sub("", text)
    text = _TRAIL_RE.sub("", text)
    return text.strip().strip("*").strip()

from apcir.functional.topics import Turn
from apcir.functional.promptor import (
    RewriteAndResponsePromptor,
    RARPersonalizedCoTPromptor,
    RARNonPersonalizedCoTPromptor,
    MQ4CSRWPrompter,
    GtR_RS,
    GtR_RW,
    SummarizePTKBPromptor,
    PersonalizeViaPTKBSummaryPrompter,
)
from .llm_client import SharedLLMClient


_DEMO_DIR = "/data/rech/huiyuche/TREC_iKAT_2024/data/topics/ikat23"
_IKAT23_DEMO = f"{_DEMO_DIR}/original_demonstration.json"                       # rar: NO ptkb
_IKAT23_PERS_DEMO = f"{_DEMO_DIR}/demonstration_using_ikat23.json"             # personalized_cot: HAS ptkb
_IKAT23_NONPERS_DEMO = f"{_DEMO_DIR}/non_personalized_demonstration_using_ikat23.json"  # non_personalized_cot


@dataclass
class RewriterConfig:
    demo_file: str = _IKAT23_DEMO                          # rar (RewriteAndResponsePromptor)
    personalized_demo_file: str = _IKAT23_PERS_DEMO        # RARPersonalizedCoTPromptor (needs per-demo 'ptkb')
    non_personalized_demo_file: str = _IKAT23_NONPERS_DEMO # RARNonPersonalizedCoTPromptor
    gtr_phi: int = 2


def context_turns_from_history(history: List[str]) -> List[Turn]:
    """history = [u1, r1, u2, r2, ...] (interleaved) -> [Turn(current_utterance=u_i,
    current_response=r_i)]. The promptors read only current_utterance/current_response."""
    turns: List[Turn] = []
    for i in range(0, len(history) - 1, 2):
        t = Turn()
        t.current_utterance = history[i]
        t.current_response = history[i + 1]
        turns.append(t)
    # a dangling final user utterance with no response is the CURRENT turn (not context)
    return turns


class OnlineRewriter:
    def __init__(self, llm: SharedLLMClient, config: RewriterConfig = None):
        self.llm = llm
        self.config = config or RewriterConfig()
        self._promptor_cache: Dict[str, object] = {}

    # ---- promptor construction (lazy, cached; mirrors rewrite.py:240-347) -------- #
    def _promptor(self, qr_name: str):
        if qr_name in self._promptor_cache:
            return self._promptor_cache[qr_name]
        c = self.config
        if qr_name == "rar":
            p = RewriteAndResponsePromptor(demo_file=c.demo_file, enable_cot=False)
        elif "rar_personalized_cot" in qr_name:
            p = RARPersonalizedCoTPromptor(
                demo_file=c.personalized_demo_file, enable_cot=(qr_name[-1] != "N"),
                zero_shot_cot=(qr_name[-1] == "0"), one_shot_cot=(qr_name[-1] == "1"),
                cot_format="cot_seperate")
        elif "rar_non_personalized_cot" in qr_name:
            p = RARNonPersonalizedCoTPromptor(
                demo_file=c.non_personalized_demo_file, enable_cot=(qr_name[-1] != "N"),
                zero_shot_cot=(qr_name[-1] == "0"), one_shot_cot=(qr_name[-1] == "1"),
                cot_format="cot_seperate")
        elif "MQ4CS_persq" in qr_name:
            p = MQ4CSRWPrompter()
        else:
            raise ValueError(f"unsupported qr_name {qr_name!r}")
        self._promptor_cache[qr_name] = p
        return p

    # ---- main entry -------------------------------------------------------------- #
    def rewrite(self, turn: Turn, qr_name: str, context: List[Turn],
                ptkb_dict: Dict[int, str]) -> List[str]:
        """Return the rewritten query string(s) for this turn. Memoized on the Turn."""
        cache = getattr(turn, "_qr_cache", None)
        if cache is None:
            cache = {}
            turn._qr_cache = cache
        if qr_name in cache:
            return cache[qr_name]

        try:
            if "ptkb_sum" in qr_name:
                queries = self._ptkb_sum(turn, qr_name, context, ptkb_dict)
            elif "GtR" in qr_name:
                queries = self._gtr(turn, context, ptkb_dict)
            elif "MQ4CS_persq" in qr_name:
                queries = self._mq4cs_persq(turn, qr_name, context, ptkb_dict)
            elif "rar_personalized_cot" in qr_name or "rar_non_personalized_cot" in qr_name:
                queries = self._rar_cot(turn, qr_name, context, ptkb_dict)
            elif qr_name == "rar":
                queries = self._rar(turn, context)
            else:
                raise ValueError(f"unsupported qr_name {qr_name!r}")
        except Exception as e:  # noqa: BLE001 - graceful degrade to the raw utterance
            print(f"[rewriter] {qr_name} failed ({e}); falling back to raw utterance")
            queries = [turn.current_utterance]

        queries = [clean_rewrite(q) for q in queries if q and q.strip()]
        queries = [q for q in queries if q] or [turn.current_utterance]
        cache[qr_name] = queries
        return queries

    # ---- LLM call that ALSO stores the model's reasoning (parallel to the manual _cot) -- #
    def _gen(self, prompt: str, turn: Turn, think_key: str) -> str:
        """Call the LLM, store its <think> reasoning as the `think_key` reformulation
        (Qwen3 etc. think -> better quality; we keep thinking ON and record it), and
        return the clean post-</think> answer for the promptor parser."""
        answer, think = self.llm.generate(prompt)
        if think:
            turn.add_reformulation(think_key, think, [])
        return answer

    # ---- per-qr handlers (mirror rewrite.py exactly) ----------------------------- #
    def _rar(self, turn: Turn, context: List[Turn]) -> List[str]:
        p = self._promptor("rar")
        prompt = p.build_turn_prompt(context, turn)              # NO ptkb for rar
        parsed = p.parse_returned_text(self._gen(prompt, turn, "rar_think"))
        if not parsed:
            return [turn.current_utterance]
        turn.add_reformulation("rar_rw", parsed[0], [])
        turn.add_reformulation("rar_rs", parsed[1], [])
        return [parsed[0]]

    def _rar_cot(self, turn: Turn, qr_name: str, context: List[Turn],
                 ptkb_dict: Dict[int, str]) -> List[str]:
        p = self._promptor(qr_name)
        prompt = p.build_turn_prompt(context, ptkb_dict, turn)
        parsed = p.parse_returned_text(self._gen(prompt, turn, qr_name + "_think"))
        if not parsed:
            return [turn.current_utterance]
        rewrite, response = parsed[0], parsed[1]
        if "N" not in qr_name and len(parsed) > 2:
            turn.add_reformulation(qr_name + "_cot", parsed[2], [])
        turn.add_reformulation(qr_name + "_rw", rewrite, [])
        turn.add_reformulation(qr_name + "_rs", response, [])
        return [rewrite]

    def _mq4cs_persq(self, turn: Turn, qr_name: str, context: List[Turn],
                     ptkb_dict: Dict[int, str]) -> List[str]:
        p = self._promptor(qr_name)
        prompt = p.build_turn_prompt(context, ptkb_dict, turn)
        query = p.parse_returned_text(self._gen(prompt, turn, qr_name + "_think"))
        if not query:
            return [turn.current_utterance]
        turn.add_reformulation(qr_name + "_rw", query, [])
        return [query]

    def _gtr(self, turn: Turn, context: List[Turn], ptkb_dict: Dict[int, str]) -> List[str]:
        # stage 1: GtR_RS -> a free-text answer
        rs = GtR_RS()
        answer = self._gen(rs.build_turn_prompt(context, ptkb_dict, turn), turn, "GtR_rs_think")
        turn.add_reformulation("GtR_rs", answer, [])
        # stage 2: GtR_RW(phi) -> phi queries from the answer
        phi = self.config.gtr_phi
        rw = GtR_RW(phi=phi)
        queries = rw.parse_returned_text(
            self._gen(rw.build_turn_prompt(context, ptkb_dict, turn, answer), turn, "GtR_mq_think"))
        if not queries:
            return [turn.current_utterance]
        queries = [q for q in queries if q and q.strip() and q != "GG"]
        for i, q in enumerate(queries):
            turn.add_reformulation(f"GtR_mq_{i+1}", q, [])
        return queries or [turn.current_utterance]

    def _ptkb_sum(self, turn: Turn, qr_name: str, context: List[Turn],
                  ptkb_dict: Dict[int, str]) -> List[str]:
        # needs a decontextualized rar_rw first
        rar_q = self._rar(turn, context)[0]
        # summarize the PTKB
        summ_p = SummarizePTKBPromptor()
        summary = summ_p.parse_returned_text(
            self._gen(summ_p.build_turn_prompt(ptkb_dict), turn, "ptkb_summarize_think"))
        if not summary:
            return [rar_q]
        turn.add_reformulation("ptkb_summarize", summary, [])
        # personalize rar_rw via the summary
        pers_p = PersonalizeViaPTKBSummaryPrompter(enable_cot=("cot" in qr_name))
        parsed = pers_p.parse_returned_text(
            self._gen(pers_p.build_turn_prompt(summary=summary, user_query=rar_q),
                      turn, qr_name + "_think"))
        if not parsed:
            return [rar_q]
        turn.add_reformulation(qr_name + "_rw", parsed[0], [])
        turn.add_reformulation(qr_name + "_rs", parsed[1], [])
        return [parsed[0]]
