"""Write every prompt the model receives during one example meeting.

Runs a meeting with the scripted model, which records each call exactly as a
real model would receive it, and writes them in order as Markdown:

    python apps/cli/show_prompts.py --out prompts.md
    python apps/cli/show_prompts.py --language 日本語 --out prompts_ja.md

The example has two agenda items, two AI participants and one human, so all
node types appear. Use --solo for a meeting without participants.
"""

import argparse
import json
import sys

from agentic_meeting import Meeting
from agentic_meeting.testing import demo_model


PARTICIPANTS = [
    {"id": "researcher", "name": "Researcher", "role": "researcher",
     "description": "Develops alternative trading strategies."},
    {"id": "risk", "name": "Risk analyst", "role": "risk analyst",
     "instructions": "Always quantify the downside."},
    {"id": "owner", "name": "Haruto", "role": "project owner", "is_human": True},
]

MEETING = {
    "project_id": "trading-ai",
    "topic": "Reduce excessive trading frequency",
    "background": "The trading system generates too many trades; fees reduce profit.",
    "agenda": [
        {"id": "approach", "description": "Fixed versus adaptive frequency"},
        {"id": "evaluation", "description": "How to evaluate the policy"},
    ],
    "max_rounds_per_agenda": 2,
}


def main() -> None:
    sys.stdout.reconfigure(errors="replace")

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", help="Markdown file to write (default: print)")
    parser.add_argument("--language", default="", help="Meeting language")
    parser.add_argument("--solo", action="store_true", help="No participants")
    args = parser.parse_args()

    model = demo_model()
    meeting = Meeting(model)

    meeting_input = {**MEETING, "language": args.language}
    if not args.solo:
        meeting_input["participants"] = PARTICIPANTS

    result = meeting.start(meeting_input)
    while not result.finished:
        result = meeting.resume(
            result.thread_id, {"feedback": "Adaptive, if drawdown stays under 5%."}
        )

    lines = [
        "# Prompts of one example meeting",
        "",
        f"{len(model.calls)} model calls, in order. Each has a system prompt "
        "(instructions), a user message (the meeting data, as JSON) and, for "
        "the moderator and extraction, the JSON schema of the reply.",
        "",
    ]

    for number, call in enumerate(model.calls, start=1):
        who = f" — {call['participant']}" if call["participant"] else ""
        lines += [
            f"## {number}. {call['node']}{who}",
            "",
            "### System prompt",
            "",
            "```text",
            call["system"].strip(),
            "```",
            "",
            "### User message",
            "",
            "```json",
            call["user"].strip(),
            "```",
            "",
        ]

        if call["schema"]:
            lines += [
                "### Reply schema (structured output)",
                "",
                "```json",
                json.dumps(call["schema"], indent=2),
                "```",
                "",
            ]

    text = "\n".join(lines)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as file:
            file.write(text)
        print(f"Wrote {len(model.calls)} prompts to {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
