# llm_client.py
from __future__ import annotations

import os
import json
import importlib.util
from typing import Any, Dict, List, Tuple

try:
    import requests  # pip install requests
except Exception:
    requests = None  # Optional; we fall back to SDK where possible


UsageDict = Dict[str, Any]  # prompt_tokens, completion_tokens, total_tokens, etc.


class LLMClient:
    """
    Provider order:
      1) FORCE_PROVIDER (lightpanda|openai|openllm)
      2) LIGHTPANDA_API_KEY -> lightpanda
      3) OPENAI_API_KEY     -> openai
      4) OPENLLM_BASE_URL   -> openllm
      else -> stub (offline)
    """

    def __init__(self) -> None:
        self.provider: str = self._select_provider()
        self.session = requests.Session() if requests else None

    def _select_provider(self) -> str:
        forced = (os.getenv("FORCE_PROVIDER") or "").strip().lower()
        if forced in ("lightpanda", "openai", "openllm"):
            return forced
        if os.getenv("LIGHTPANDA_API_KEY"):
            return "lightpanda"
        if os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_TOKEN"):
            return "openai"
        if os.getenv("OPENLLM_BASE_URL"):
            return "openllm"
        return "stub"

    # -------------------- public API --------------------

    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.2) -> Tuple[str, UsageDict]:
        """
        Returns (text, usage_dict). usage_dict follows OpenAI-style keys when available:
        { 'prompt_tokens': int, 'completion_tokens': int, 'total_tokens': int, ... }
        """
        if self.provider == "lightpanda":
            return self._chat_lightpanda(messages, temperature=temperature)
        if self.provider == "openai":
            return self._chat_openai(messages, temperature=temperature)
        if self.provider == "openllm":
            return self._chat_openllm(messages, temperature=temperature)
        return self._chat_stub(messages)

    # ------------------ provider impls ------------------

    def _chat_lightpanda(self, messages: List[Dict[str, str]], temperature: float) -> Tuple[str, UsageDict]:
        base = os.getenv("LIGHTPANDA_BASE_URL", "https://api.lightpanda.ai")
        url = f"{base.rstrip('/')}/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {os.getenv('LIGHTPANDA_API_KEY')}",
            "Content-Type": "application/json",
        }
        payload: Dict[str, Any] = {
            "model": os.getenv("LIGHTPANDA_MODEL", "gpt-4o-mini"),
            "messages": messages,
            "temperature": temperature,
        }
        return self._post_and_parse(url, headers, payload)

    def _chat_openai(self, messages: List[Dict[str, str]], temperature: float) -> Tuple[str, UsageDict]:
        # Try SDK first
        try:
            from openai import OpenAI  # pip install openai
            api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_TOKEN")
            client = OpenAI(api_key=api_key)
            model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            resp = client.chat.completions.create(model=model, messages=messages, temperature=temperature)
            text = resp.choices[0].message.content
            usage_obj = getattr(resp, "usage", None)
            if hasattr(usage_obj, "to_dict"):
                usage: UsageDict = usage_obj.to_dict()  # type: ignore[attr-defined]
            elif usage_obj is None:
                usage = {}
            else:
                usage = dict(usage_obj)  # type: ignore[arg-type]
            return text, usage
        except Exception:
            # HTTP fallback
            url = "https://api.openai.com/v1/chat/completions"
            api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_TOKEN")
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            payload: Dict[str, Any] = {
                "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                "messages": messages,
                "temperature": temperature,
            }
            return self._post_and_parse(url, headers, payload)

    def _chat_openllm(self, messages: List[Dict[str, str]], temperature: float) -> Tuple[str, UsageDict]:
        base = os.getenv("OPENLLM_BASE_URL", "http://localhost:8000")
        url = f"{base.rstrip('/')}/v1/chat/completions"
        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if os.getenv("OPENLLM_API_KEY"):
            headers["Authorization"] = f"Bearer {os.getenv('OPENLLM_API_KEY')}"
        payload: Dict[str, Any] = {
            "model": os.getenv("OPENLLM_MODEL", "qwen2.5-7b-instruct"),
            "messages": messages,
            "temperature": temperature,
        }
        return self._post_and_parse(url, headers, payload)

    def _chat_stub(self, messages: List[Dict[str, str]]) -> Tuple[str, UsageDict]:
        last_user = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
        return f"[STUB LLM REPLY] {last_user[:200]}", {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

    # ------------------ HTTP helper ------------------

    def _post_and_parse(self, url: str, headers: Dict[str, str], payload: Dict[str, Any]) -> Tuple[str, UsageDict]:
        if not self.session:
            raise RuntimeError("Install 'requests' to use HTTP providers. Try: pip install requests")
        r = self.session.post(url, headers=headers, json=payload, timeout=60)
        r.raise_for_status()
        data: Dict[str, Any] = r.json()

        # OpenAI-compatible response
        text: str
        if "choices" in data and data["choices"]:
            try:
                text = data["choices"][0]["message"]["content"]
            except Exception:
                # Some servers use 'text' directly
                text = data.get("text", "")
        else:
            text = data.get("text", "")

        usage: UsageDict = {}
        if isinstance(data.get("usage"), dict):
            usage = data["usage"]  # type: ignore[assignment]
        else:
            # Some servers don't return usage; leave zeros so your counters still work
            usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        return text, usage