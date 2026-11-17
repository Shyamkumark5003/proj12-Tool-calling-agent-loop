import json
from pathlib import Path
from typing import Literal

from pydantic import Field

from agent12.registry import StrictArgs, Tool

TICKETS_FILE = Path(__file__).resolve().parent.parent / "data" / "tickets.json"


# ---------- Tool 1: ticket lookup (read-only) ----------
class LookupTicketArgs(StrictArgs):
    ticket_id: int = Field(gt=0)


def lookup_ticket(args: LookupTicketArgs) -> dict:
    tickets = json.loads(TICKETS_FILE.read_text())
    ticket = tickets.get(str(args.ticket_id))
    if ticket is None:
        raise ValueError(f"ticket {args.ticket_id} not found")
    return ticket


# ---------- Tool 2: RAG search ----------
class SearchDocsArgs(StrictArgs):
    query: str = Field(min_length=3, max_length=200)
    top_k: int = Field(default=3, ge=1, le=5)


def search_docs(args: SearchDocsArgs) -> dict:
    from agent12.rag import search
    return {"results": search(args.query, args.top_k)}


# ---------- Tool 3: unit converter ----------
CONVERSIONS = {
    ("km", "miles"): lambda v: v * 0.621371,
    ("miles", "km"): lambda v: v * 1.609344,
    ("kg", "lb"): lambda v: v * 2.204623,
    ("lb", "kg"): lambda v: v * 0.453592,
    ("c", "f"): lambda v: v * 9 / 5 + 32,
    ("f", "c"): lambda v: (v - 32) * 5 / 9,
}


class ConvertUnitsArgs(StrictArgs):
    value: float
    from_unit: str
    to_unit: str


def convert_units(args: ConvertUnitsArgs) -> dict:
    key = (args.from_unit.lower(), args.to_unit.lower())
    fn = CONVERSIONS.get(key)
    if fn is None:
        supported = sorted({f"{a}->{b}" for a, b in CONVERSIONS})
        raise ValueError(f"unsupported conversion {key[0]}->{key[1]}. supported: {supported}")
    return {"result": round(fn(args.value), 4), "unit": key[1]}


# ---------- Tool 4: create ticket (SIDE EFFECT) ----------
class CreateTicketArgs(StrictArgs):
    title: str = Field(min_length=3, max_length=100)
    priority: Literal["low", "medium", "high"] = "medium"


def _next_id(tickets: dict) -> int:
    return max((int(k) for k in tickets), default=0) + 1


def create_ticket_preview(args: CreateTicketArgs) -> dict:
    tickets = json.loads(TICKETS_FILE.read_text())
    return {
        "action": "create_ticket",
        "will_create": {"id": _next_id(tickets), "title": args.title,
                        "priority": args.priority, "status": "open"},
        "note": "DRY RUN - nothing has been written yet",
    }


def create_ticket(args: CreateTicketArgs) -> dict:
    tickets = json.loads(TICKETS_FILE.read_text())
    new_id = _next_id(tickets)
    ticket = {"id": new_id, "title": args.title, "status": "open", "priority": args.priority}
    tickets[str(new_id)] = ticket
    TICKETS_FILE.write_text(json.dumps(tickets, indent=2))
    return ticket


TOOLS = {
    t.name: t
    for t in [
        Tool("lookup_ticket", "Get a support ticket by numeric id (read-only).",
             LookupTicketArgs, lookup_ticket),
        Tool("search_docs", "Search the knowledge base and return relevant passages.",
             SearchDocsArgs, search_docs, timeout_s=10.0),
        Tool("convert_units", "Convert a number between units (km/miles, kg/lb, c/f).",
             ConvertUnitsArgs, convert_units),
        Tool("create_ticket", "Create a new support ticket. Needs human approval.",
             CreateTicketArgs, create_ticket,
             side_effect=True, preview=create_ticket_preview),
    ]
}
