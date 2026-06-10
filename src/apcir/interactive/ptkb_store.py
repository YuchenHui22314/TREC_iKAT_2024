"""PTKBStore — OPTIONAL, minimal per-conversation PTKB seam.

The live Sim.API has NO structured ptkb/ptkb_provenance field (submission =
{run_id, response, citations, meta}); a PTKB list, if we emit one, rides in the
free-form `meta`. The scored-PTKB language on the guidelines page is likely stale
2024/25 boilerplate. So this is best-effort, not core.

We still keep a thin seam so we can light it up cheaply IF the debug run shows base
PTKB arriving in `user_meta` AND PTKB turns out to be scored:
  - hold the base PTKB statements (from user_meta);
  - log whether user_meta.ptkb is present and whether it changes across turns
    (this is what tells us PTKB is in play);
  - select relevant base statements for `meta.ptkb_provenance`.
The extractor (pull NEW persona facts from the dialogue) is intentionally a STUB —
do not build it until the debug run confirms it is worth it.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional


_WORD = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set:
    return set(_WORD.findall((text or "").lower()))


class PTKBStore:
    def __init__(self, conversation_id: str):
        self.conversation_id = conversation_id
        self.base: List[str] = []
        self.extracted: List[str] = []        # v1: always empty (extractor stubbed)
        # diagnostics gathered across the conversation (printed at finalize)
        self.ever_present: bool = False
        self.changed_across_turns: bool = False
        self._seen_signatures: set = set()
        self._n_updates: int = 0

    @staticmethod
    def _extract_ptkb_list(user_meta: Optional[Dict[str, Any]]) -> List[str]:
        """Pull a list[str] of PTKB statements out of a free-form user_meta, tolerating
        a few plausible shapes (list of str, list of {statement/text/content}, dict)."""
        if not user_meta:
            return []
        raw = user_meta.get("ptkb")
        if raw is None:
            return []
        out: List[str] = []
        if isinstance(raw, dict):
            raw = list(raw.values())
        if isinstance(raw, list):
            for item in raw:
                if isinstance(item, str):
                    out.append(item)
                elif isinstance(item, dict):
                    for key in ("statement", "text", "content", "ptkb"):
                        if key in item and isinstance(item[key], str):
                            out.append(item[key])
                            break
        elif isinstance(raw, str):
            out.append(raw)
        return out

    def update(self, user_meta: Optional[Dict[str, Any]]) -> None:
        """Refresh base PTKB from this turn's user_meta and record diagnostics."""
        self._n_updates += 1
        stmts = self._extract_ptkb_list(user_meta)
        if stmts:
            self.ever_present = True
            sig = tuple(stmts)
            if self._seen_signatures and sig not in self._seen_signatures:
                self.changed_across_turns = True
            self._seen_signatures.add(sig)
            self.base = stmts
        # extractor seam (STUB): would append newly-extracted persona facts to
        # self.extracted here once confirmed worth building.

    def relevant_for(self, turn) -> List[str]:
        """v1: return base statements whose words overlap the current utterance
        (simple, dependency-free). Empty if no base PTKB."""
        pool = self.base + self.extracted
        if not pool:
            return []
        q = _tokens(getattr(turn, "current_utterance", "") or "")
        if not q:
            return []
        scored = [(len(q & _tokens(s)), s) for s in pool]
        relevant = [s for n, s in scored if n > 0]
        return relevant

    def diagnostics(self) -> Dict[str, Any]:
        return {
            "conversation_id": self.conversation_id,
            "updates": self._n_updates,
            "ptkb_ever_present": self.ever_present,
            "ptkb_changed_across_turns": self.changed_across_turns,
            "base_size": len(self.base),
            "distinct_ptkb_signatures": len(self._seen_signatures),
        }
