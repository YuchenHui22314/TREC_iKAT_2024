"""RAG response generation for the interactive server, using the shared LLM.

Reuses `PersonalizedResponseGenPromptor` (functional/promptor.py:508) — the SAME prompt
the offline pipeline uses (dialog context + user profile + reference passages -> a fluent,
grounded, personalized answer, capped ~220 words in-prompt). This REPLACES the v1
extractive passage-dump that made the user-simulator loop. Falls back to extractive on
any LLM/parse failure.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from apcir.functional.topics import Turn, load_document_by_id
from apcir.functional.promptor import PersonalizedResponseGenPromptor

_GEN_PROMPTOR = PersonalizedResponseGenPromptor()


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
) -> Optional[str]:
    """Fetch top-k passage texts, build the personalized RAG prompt, generate, truncate.
    Returns None on failure so the caller can fall back to extractive."""
    texts, seen = [], set()
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
    if not texts:
        return None
    prompt = _GEN_PROMPTOR.build_turn_prompt(
        context=context_turns, ptkb_dict=ptkb_dict or {},
        passages_list=texts, last_question=last_question)
    try:
        raw = llm.generate_text(prompt)[0]
    except Exception as e:  # noqa: BLE001
        print(f"[generation] LLM failed ({e}); falling back to extractive")
        return None
    resp = _GEN_PROMPTOR.parse_returned_text(raw)
    if not resp:
        # the model sometimes omits the "Response: " prefix — accept the raw text
        resp = raw.strip()
    if not resp:
        return None
    return truncate_fn(resp, max_tokens)
