import json
import os

import httpx

from agent12.llm import LLMResponse, ToolCall


class LLMError(RuntimeError):
    pass


class OpenAICompatLLM:
    """Works with any OpenAI-compatible /chat/completions API (OpenAI, Groq, ...)."""

    def __init__(self, api_key: str | None = None, base_url: str | None = None,
                 model: str | None = None, timeout_s: float = 30.0,
                 client: httpx.Client | None = None):
        self.api_key = api_key or os.environ.get("LLM_API_KEY")
        if not self.api_key:
            raise ValueError("LLM_API_KEY is not set")
        self.base_url = (base_url or os.environ.get("LLM_BASE_URL",
                                                   "https://api.openai.com/v1")).rstrip("/")
        self.model = model or os.environ.get("LLM_MODEL", "gpt-4o-mini")
        self.client = client or httpx.Client(timeout=timeout_s)

    def chat(self, messages: list[dict], tools: list[dict]) -> LLMResponse:
        r = self.client.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": self.model, "messages": messages, "tools": tools},
        )
        if r.status_code != 200:
            raise LLMError(f"HTTP {r.status_code}: {r.text[:200]}")

        data = r.json()
        msg = data["choices"][0]["message"]

        calls: list[ToolCall] = []
        for tc in msg.get("tool_calls") or []:
            raw = tc["function"].get("arguments") or "{}"
            try:
                args = json.loads(raw)
                if not isinstance(args, dict):
                    raise ValueError("arguments must be an object")
            except ValueError:
                # strict validation will reject this and the error goes back to the model
                args = {"_invalid_json": raw[:200]}
            calls.append(ToolCall(tc["id"], tc["function"]["name"], args))

        usage = data.get("usage") or {}
        return LLMResponse(
            text=msg.get("content") or "",
            tool_calls=calls,
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
        )
