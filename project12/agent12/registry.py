import logging
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict, ValidationError

log = logging.getLogger("agent12.tools")
_pool = ThreadPoolExecutor(max_workers=4)


class StrictArgs(BaseModel):
    """Base for all tool arguments. Extra/unknown fields = error."""
    model_config = ConfigDict(extra="forbid")


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    args_model: type[StrictArgs]
    func: Callable[[Any], Any]
    timeout_s: float = 5.0
    side_effect: bool = False
    preview: Callable[[Any], Any] | None = None  # dry-run; side_effect tool-ku kattaayam

    def __post_init__(self):
        if self.side_effect and self.preview is None:
            raise ValueError(f"tool '{self.name}' has a side effect but no dry-run preview")


def _short_error(e: ValidationError) -> str:
    parts = []
    for err in e.errors():
        where = ".".join(str(x) for x in err["loc"]) or "arguments"
        parts.append(f"{where}: {err['msg']}")
    return "; ".join(parts)


def run_tool(tool: Tool, raw_args: dict) -> dict:
    """Never raises. Always returns {'ok': True, 'data': ...} or {'ok': False, 'error': ...}."""
    start = time.perf_counter()

    # 1. validate
    try:
        args = tool.args_model.model_validate(raw_args)
    except ValidationError as e:
        return _finish(tool, raw_args, start,
                       {"ok": False, "error": "invalid arguments: " + _short_error(e)})

    # 2. run with timeout
    future = _pool.submit(tool.func, args)
    try:
        out = {"ok": True, "data": future.result(timeout=tool.timeout_s)}
    except FutureTimeout:
        out = {"ok": False, "error": f"timeout after {tool.timeout_s}s"}
    except Exception as e:  # 3. tool error -> goes back to the model, no crash
        out = {"ok": False, "error": f"{type(e).__name__}: {e}"}

    return _finish(tool, raw_args, start, out)


def preview_tool(tool: Tool, raw_args: dict) -> dict:
    """Validate + dry run. Must NOT change anything."""
    try:
        args = tool.args_model.model_validate(raw_args)
    except ValidationError as e:
        return {"ok": False, "error": "invalid arguments: " + _short_error(e)}
    try:
        return {"ok": True, "data": tool.preview(args)}
    except Exception as e:
        return {"ok": False, "error": f"preview failed: {type(e).__name__}: {e}"}


def _finish(tool: Tool, raw_args: dict, start: float, out: dict) -> dict:
    ms = int((time.perf_counter() - start) * 1000)
    out["duration_ms"] = ms
    # 4. log every call
    log.info("tool_call name=%s args=%s ok=%s ms=%d error=%s",
             tool.name, raw_args, out["ok"], ms, out.get("error"))
    return out
