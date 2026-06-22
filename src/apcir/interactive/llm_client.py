"""SharedLLMClient — one OpenAI-SDK client for BOTH query reformulation and RAG
generation, against EITHER the OpenAI API or a local vLLM OpenAI-compatible server.

The whole point of the unification: `openai.OpenAI` talks to any OpenAI-compatible
endpoint, so the backend is just a `base_url`+model swap and we need only ONE code path.
We do NOT use the old `functional/llm.py` `OpenAILM`/`LM` wrappers here.

backend="openai"     -> base_url=None (api.openai.com), api_key=os.environ['openai_key']
backend="local_vllm" -> base_url="http://127.0.0.1:8100/v1", api_key="EMPTY"
"""

from __future__ import annotations

import os
import re
import time
from typing import List, Optional

from openai import OpenAI

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def split_think(text: str) -> tuple:
    """Split a reasoning-model completion into (answer, think).

    answer = the text AFTER </think> (clean, for the promptors' strict parsers);
    think  = the <think>…</think> reasoning content (stored as a `{name}_think`
    reformulation, parallel to the manual `_cot`). Keeping thinking ON improves quality;
    we just separate it so the parser sees clean output and the reasoning is kept on record.
    """
    think_parts = _THINK_RE.findall(text)
    answer = _THINK_RE.sub("", text)
    if "<think>" in answer:                    # unclosed (truncated) thinking block
        idx = answer.index("<think>")
        think_parts.append(answer[idx + len("<think>"):])
        answer = answer[:idx]
    think = "\n".join(p.strip() for p in think_parts).strip()
    return answer.strip(), think


class SharedLLMClient:
    def __init__(
        self,
        backend: str = "openai",
        model: str = "gpt-4o-2024-08-06",
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.0,
        top_p: float = 1.0,
        timeout: float = 120.0,
        max_retries: int = 4,
        enable_thinking: bool = True,
        reasoning_effort: Optional[str] = None,
    ):
        if backend not in ("openai", "local_vllm"):
            raise ValueError(f"backend must be 'openai' or 'local_vllm', got {backend!r}")
        self.backend = backend
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.max_retries = max_retries
        # gpt-5 / o-series reasoning effort: minimal|low|medium|high. None -> API default
        # (medium). Falls back to env LLM_REASONING_EFFORT so the server's gen call AND the
        # driver's extraction call can be sped up by one env var, no flag threading.
        self.reasoning_effort = reasoning_effort or os.environ.get("LLM_REASONING_EFFORT")
        # Qwen3 etc. are reasoning models that emit <think>…</think>, which breaks the
        # promptors' strict parsers. Disable thinking on the local server (vLLM honours the
        # chat-template kwarg) and strip any residual block. QR/RAG don't need it.
        self.enable_thinking = enable_thinking

        if backend == "local_vllm":
            base_url = base_url or "http://127.0.0.1:8100/v1"
            api_key = api_key or "EMPTY"
        else:  # openai
            api_key = api_key or os.environ.get("openai_key") or os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError(
                    "openai backend needs an API key in env 'openai_key' (or 'OPENAI_API_KEY').")
        self.base_url = base_url
        # the SDK has its own retry; we add an outer loop for connection blips (vLLM warm-up)
        self._client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout, max_retries=2)

    def __repr__(self) -> str:
        return f"SharedLLMClient(backend={self.backend!r}, model={self.model!r}, base_url={self.base_url!r})"

    def _build_kwargs(self, prompt: str) -> dict:
        """Per-backend/model API quirks:
        - OpenAI chat API now wants `max_completion_tokens` (NOT `max_tokens`); GPT-5 / o-series
          REASONING models additionally reject non-default temperature/top_p -> omit them.
        - local vLLM uses the classic `max_tokens` + temperature/top_p, and honours the
          enable_thinking chat-template kwarg.
        """
        kw = dict(model=self.model, messages=[{"role": "user", "content": prompt}])
        if self.backend == "openai":
            kw["max_completion_tokens"] = self.max_tokens
            is_reasoning = self.model.startswith(("gpt-5", "o1", "o3", "o4"))
            if not is_reasoning:
                kw["temperature"] = self.temperature
                kw["top_p"] = self.top_p
            elif self.reasoning_effort:        # minimal|low|medium|high (speed vs quality)
                kw["reasoning_effort"] = self.reasoning_effort
        else:  # local_vllm
            kw["max_tokens"] = self.max_tokens
            kw["temperature"] = self.temperature
            kw["top_p"] = self.top_p
            if not self.enable_thinking:
                kw["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
        return kw

    def _raw(self, prompt: str) -> str:
        """Raw completion string (may contain a <think>…</think> block). One user message;
        the server applies the model's chat template."""
        kwargs = self._build_kwargs(prompt)
        last_err = None
        for attempt in range(self.max_retries):
            try:
                resp = self._client.chat.completions.create(**kwargs)
                return resp.choices[0].message.content or ""
            except Exception as e:  # noqa: BLE001 - want to retry any transient API/conn error
                last_err = e
                if attempt < self.max_retries - 1:
                    time.sleep(min(2 ** attempt, 10))
        raise RuntimeError(f"LLM generate failed after {self.max_retries} tries: {last_err}")

    def generate(self, prompt: str) -> tuple:
        """Return (answer, think): the clean post-</think> answer + the reasoning block."""
        return split_think(self._raw(prompt))

    def generate_text(self, prompt: str) -> List[str]:
        """Single prompt -> [clean answer] (think stripped). For callers that don't need
        the reasoning (e.g. RAG generation)."""
        return [self.generate(prompt)[0]]

    def health(self) -> bool:
        """True if the endpoint answers a models list (cheap, no generation)."""
        try:
            self._client.models.list()
            return True
        except Exception:
            return False
