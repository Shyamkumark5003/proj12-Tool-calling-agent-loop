import copy
import time
from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict


@dataclass
class LLMResponse:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0


class LLM(Protocol):
    def chat(self, messages: list[dict], tools: list[dict]) -> LLMResponse: ...


class ScriptedLLM:
    """Fake LLM for tests: returns pre-written responses one by one."""

    def __init__(self, script: list[LLMResponse], delay_s: float = 0.0):
        self.script = list(script)
        self.delay_s = delay_s
        self.seen: list[list[dict]] = []  # messages received on each call

    def chat(self, messages: list[dict], tools: list[dict]) -> LLMResponse:
        time.sleep(self.delay_s)
        self.seen.append(copy.deepcopy(messages))
        if not self.script:
            raise RuntimeError("ScriptedLLM: script ran out")
        return self.script.pop(0)
