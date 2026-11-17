import json
import uuid

from langgraph.errors import GraphRecursionError
from langgraph.types import Command

from agent12.agent import BASE_PROMPT
from graph12 import app

config = {"configurable": {"thread_id": "t1"}, "recursion_limit": 10}

first = True
while True:
    q = input("you> ")
    if q == "exit":
        break
    msgs = [{"role": "user", "content": q}]
    if first:
        msgs.insert(0, {"role": "system", "content": BASE_PROMPT})
        first = False
    try:
        result = app.invoke({"messages": msgs}, config)
    except GraphRecursionError:
        print("assistant> [stopped: recursion_limit]")
        # limit la nindra thread state half-done ah irukkum, so pudhu thread la start pannu
        config["configurable"]["thread_id"] = uuid.uuid4().hex
        first = True
        continue

    # graph pause aana, human kitta kelu, apuram resume pannu
    while result.get("__interrupt__"):
        payload = result["__interrupt__"][0].value
        print("\n=== APPROVAL NEEDED ===")
        print(json.dumps(payload, indent=2))
        ok = input("Approve? [y/N]: ").strip().lower() == "y"
        result = app.invoke(Command(resume=ok), config)

    print("assistant>", result["messages"][-1]["content"])
