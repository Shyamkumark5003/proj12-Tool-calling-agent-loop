import json
import threading
from pathlib import Path

from pydantic import Field

from agent12.registry import StrictArgs, Tool

MAX_KEYS = 20


class ThreadStore:
    """Conversation state. thread_id -> messages. Each thread belongs to ONE user."""

    def __init__(self):
        self._threads: dict[str, dict] = {}

    def get(self, thread_id: str, user_id: str) -> list[dict]:
        t = self._threads.get(thread_id)
        if t is None:
            t = {"user_id": user_id, "messages": []}
            self._threads[thread_id] = t
        elif t["user_id"] != user_id:
            raise PermissionError(f"thread '{thread_id}' belongs to another user")
        return t["messages"]


class MemoryStore:
    """Small long-term memory, scoped per user, saved in a JSON file."""

    def __init__(self, path):
        self.path = Path(path)
        self._lock = threading.Lock()

    def _load(self) -> dict:
        if not self.path.exists():
            return {}
        return json.loads(self.path.read_text())

    def all(self, user_id: str) -> dict:
        return dict(self._load().get(user_id, {}))

    def save(self, user_id: str, key: str, value: str) -> None:
        with self._lock:
            data = self._load()
            mine = data.setdefault(user_id, {})
            if key not in mine and len(mine) >= MAX_KEYS:
                raise ValueError(f"memory full ({MAX_KEYS} keys max)")
            mine[key] = value
            self.path.write_text(json.dumps(data, indent=2))


class SaveMemoryArgs(StrictArgs):
    key: str = Field(min_length=1, max_length=50)
    value: str = Field(min_length=1, max_length=200)
    # user_id illa. LLM-ku user-a choose panna anumathi illa.


def make_memory_tool(store: MemoryStore, user_id: str) -> Tool:
    """user_id inga bind aagudhu (agent kuduthadhu), LLM kudukradhu illa."""

    def save_memory(args: SaveMemoryArgs) -> dict:
        store.save(user_id, args.key, args.value)
        return {"saved": args.key}

    return Tool(
        "save_memory",
        "Save a short fact about the current user (for example preferred units) "
        "so it can be remembered in future conversations. Never store secrets.",
        SaveMemoryArgs,
        save_memory,
    )
