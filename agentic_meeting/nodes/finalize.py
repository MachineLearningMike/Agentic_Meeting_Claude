from ..state import message
from .common import agenda_overview, ask, language_rule, recent_messages


SYSTEM_PROMPT = """
You are the secretary of a structured human-AI meeting.

Produce a concise, accurate meeting report.

Use these sections:

1. Meeting objective
2. Background and established facts
3. Agenda items: for each, the discussion and its outcome
4. Proposals discussed
5. Decisions actually agreed upon
6. Action items
7. Unresolved issues
8. Recommended next steps

Important rules:
- Never invent decisions or agreements.
- Distinguish suggestions from accepted decisions.
- If no decision was reached, explicitly say so.
- Do not invent owners or deadlines for action items.
- Preserve important disagreements.
- Say which agenda items were deferred or not reached.
"""


def finalize_meeting(state: dict, ctx) -> dict:
    # Close the item under discussion when the meeting ends early.
    agenda = [dict(item) for item in state.get("agenda", [])]
    for item in agenda:
        if item.get("status") == "active":
            item["status"] = "concluded"
            item["round_count"] = state.get("round_count", 0)
            if state.get("agenda_summary"):
                item["summary"] = state["agenda_summary"]

    context = {
        "topic": state.get("topic", ""),
        "agenda": agenda_overview({"agenda": agenda}),
        "context_analysis": state.get("context_analysis", ""),
        "messages": recent_messages(state, limit=None),
    }

    summary = ask(ctx, "finalize", SYSTEM_PROMPT + language_rule(state), context)

    return {
        "meeting_summary": summary,
        "agenda": agenda,
        "messages": [message("assistant", f"[Meeting Report]\n\n{summary}")],
    }
