import json
import operator
from typing import Annotated, TypedDict

from agent12.agent import tool_specs
from agent12.openai_llm import OpenAICompatLLM
from agent12.tools import TOOLS

llm = OpenAICompatLLM()
specs = tool_specs(TOOLS)


class State(TypedDict):
    # operator.add = node return panna new messages, list la sernthukkum
    messages: Annotated[list, operator.add]


def llm_node(state: State) -> dict:
    resp = llm.chat(state["messages"], specs)
    msg = {"role": "assistant", "content": resp.text or None}
    if resp.tool_calls:
        msg["tool_calls"] = [
            {"id": c.id, "type": "function",
             "function": {"name": c.name, "arguments": json.dumps(c.args)}}
            for c in resp.tool_calls
        ]
    return {"messages": [msg]}


# ---------- Step 3: tools node + edge + compile ----------
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from agent12.agent import BASE_PROMPT
from agent12.registry import preview_tool, run_tool
from langgraph.types import interrupt


def tools_node(state: State) -> dict:
    last = state["messages"][-1]
    out = []
    for call in last["tool_calls"]:
        name = call["function"]["name"]
        args = json.loads(call["function"]["arguments"])
        tool = TOOLS.get(name)
        if tool is None:
            result = {"ok": False, "error": f"unknown tool '{name}'. available: {sorted(TOOLS)}"}
        elif tool.side_effect:
            preview = preview_tool(tool, args)
            if not preview["ok"]:
                result = preview  # bad args -> model ku error, human ah disturb pannala
            else:
                approved = interrupt({"tool": name, "preview": preview["data"]})
                result = (run_tool(tool, args) if approved else
                          {"ok": False, "error": "rejected by the human. NOTHING was changed. Tell the user it was not done."})
        else:
            result = run_tool(tool, args)
        out.append({"role": "tool", "tool_call_id": call["id"], "content": json.dumps(result)})
    return {"messages": out}


def route(state: State) -> str:
    # conditional edge: tool call irundha tools ku, illa na END
    return "tools" if state["messages"][-1].get("tool_calls") else END


builder = StateGraph(State)
builder.add_node("llm", llm_node)
builder.add_node("tools", tools_node)
builder.add_edge(START, "llm")
builder.add_conditional_edges("llm", route, {"tools": "tools", END: END})
builder.add_edge("tools", "llm")

app = builder.compile(checkpointer=InMemorySaver())
