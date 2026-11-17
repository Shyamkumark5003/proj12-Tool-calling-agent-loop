import logging
import uuid

from agent12.agent import Agent, cli_approver
from agent12.openai_llm import OpenAICompatLLM
from agent12.state import MemoryStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s | %(message)s")

agent = Agent(OpenAICompatLLM(), approver=cli_approver, memory=MemoryStore("data/memory.json"))
user_id = input("user id: ").strip() or "shyam"
thread_id = uuid.uuid4().hex
print(f"thread {thread_id[:8]}  (type 'exit' to quit)")

while True:
    q = input("\nyou> ").strip()
    if q.lower() in {"exit", "quit"}:
        break
    r = agent.run(q, user_id=user_id, thread_id=thread_id)
    print("\nassistant>", r.answer or f"[stopped: {r.stop_reason}]")
    print(f"(steps={r.steps} tokens={r.tokens} cost=${r.cost_usd})")
