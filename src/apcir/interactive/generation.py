"""RAG response generation for the interactive server, using the shared LLM.

Reuses `PersonalizedResponseGenPromptor` (functional/promptor.py:508) — the SAME prompt
the offline pipeline uses (dialog context + user profile + reference passages -> a fluent,
grounded, personalized answer, capped ~220 words in-prompt). This REPLACES the v1
extractive passage-dump that made the user-simulator loop. Falls back to extractive on
any LLM/parse failure.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from apcir.functional.topics import Turn, load_document_by_id
from apcir.functional.promptor import InteractiveResponseGenPromptor

# iKAT-scoring-tuned prompt (nugget coverage + groundedness + selective PTKB); offline
# PersonalizedResponseGenPromptor is left untouched. See promptor.InteractiveResponseGenPromptor.
_GEN_PROMPTOR = InteractiveResponseGenPromptor()

# Skip array[1]-style indexing (ASCII identifier char before '[') but NOT CJK text: unicode \w
# counts CJK as word chars, which silently dropped every marker in Chinese answers ("菜肴[2]。").
_CITE_RE = re.compile(r"(?<![A-Za-z0-9_])\[(\d+)\]")


def parse_citations(response: str, docids: List[str]) -> List[Dict[str, Any]]:
    """Extract inline [n] citation markers from `response`. Returns [{n, docid, start, end}] for
    each marker whose 1-based n indexes into `docids` (out-of-range markers are dropped). start/end
    are char offsets of the [n] token in `response` — the frontend underlines the cited span and
    jumps to that passage."""
    out = []
    for m in _CITE_RE.finditer(response):
        n = int(m.group(1))
        if 1 <= n <= len(docids):
            out.append({"n": n, "docid": docids[n - 1], "start": m.start(), "end": m.end()})
    return out


def rag_response(
    llm,
    docfetch,
    ranked: List[Any],
    context_turns: List[Turn],
    ptkb_dict: Dict[int, str],
    last_question: str,
    top_k: int,
    truncate_fn,
    max_tokens: int,
    cite: bool = False,
) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    """Fetch top-k passage texts, build the personalized RAG prompt, generate, truncate. Returns
    (response, citations): citations is [] unless cite=True, in which case the prompt asks the LLM to
    mark passage uses with [n] and the parsed {n,docid,start,end} spans are returned. Returns
    (None, []) on failure so the caller can fall back to extractive."""
    texts, docids, seen = [], [], set()
    for d in ranked[:top_k]:
        if d.docid in seen:
            continue
        seen.add(d.docid)
        try:
            txt = load_document_by_id(d.docid, docfetch)["contents"].strip()
        except Exception:
            txt = ""
        if txt:
            texts.append(txt)
            docids.append(d.docid)
    if not texts:
        return None, []
    prompt = _GEN_PROMPTOR.build_turn_prompt(
        context=context_turns, ptkb_dict=ptkb_dict or {},
        passages_list=texts, last_question=last_question)
    if cite:                                   # opt-in: leaves the iKAT-tuned base prompt untouched
        prompt += ("\n\nIMPORTANT — overriding any earlier instruction about NOT citing inline: in "
                   "this answer, cite each claim with the number of the Search Result you used, in "
                   "square brackets — e.g. [1] or [2][3] — where the number is that Search Result's "
                   "1-based position in the list above.")
    try:
        raw = llm.generate_text(prompt)[0]
    except Exception as e:  # noqa: BLE001
        print(f"[generation] LLM failed ({e}); falling back to extractive")
        return None, []
    resp = _GEN_PROMPTOR.parse_returned_text(raw)
    if not resp:
        # the model sometimes omits the "Response: " prefix — accept the raw text
        resp = raw.strip()
    if not resp:
        return None, []
    resp = truncate_fn(resp, max_tokens)
    return resp, (parse_citations(resp, docids) if cite else [])
