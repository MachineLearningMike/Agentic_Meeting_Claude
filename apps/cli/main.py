"""A meeting in the console.

    python apps/cli/main.py                      # Claude Opus 5.5
    python apps/cli/main.py --model claude-sonnet-5-5 --effort low
    python apps/cli/main.py --model fake         # offline demo, no API key
"""

import argparse
import os
import sys
from pathlib import Path

from agentic_meeting import ClaudeModel, Meeting, MeetingStore
from agentic_meeting.model import DEFAULT_MODEL
from agentic_meeting.testing import demo_model


# Shared with the other apps: <repo>/data/meetings.db, or $MEETING_DB.
DATA = Path(__file__).resolve().parents[2] / "data"


def make_model(name: str, effort: str):
    return demo_model() if name == "fake" else ClaudeModel(name, effort=effort)


def main():
    # Redirected output on Windows may use a code page without e.g. Japanese;
    # replace what it cannot show instead of crashing.
    sys.stdout.reconfigure(errors="replace")

    parser = argparse.ArgumentParser(description="Run a meeting.")
    parser.add_argument(
        "--model",
        default=os.environ.get("MEETING_MODEL", DEFAULT_MODEL),
        help=(
            "Claude model id, or 'fake' for an offline demo. "
            "Defaults to $MEETING_MODEL or " + DEFAULT_MODEL + "."
        ),
    )
    parser.add_argument(
        "--effort",
        default="medium",
        choices=["low", "medium", "high", "xhigh", "max"],
        help="How deeply Claude thinks (default: medium).",
    )
    parser.add_argument(
        "--project",
        default="",
        help="Project ID. Meetings of the same project share their memory.",
    )
    parser.add_argument(
        "--language",
        default="",
        help="Meeting language, e.g. Japanese. Default: the language you write in.",
    )
    args = parser.parse_args()

    meeting = Meeting(
        make_model(args.model, args.effort),
        store=MeetingStore(os.environ.get("MEETING_DB", str(DATA / "meetings.db"))),
    )

    result = meeting.start({
        "project_id": args.project,
        "language": args.language,
        "topic": "Adaptive trading frequency",
        "background": "Our trading system generates too many trades.",
        "problems": [
            {"id": "problem-001", "description": "Trading frequency is too high."}
        ],
        "max_rounds_per_agenda": 3,
    })

    while not result.finished:
        print("\nModerator asks:")
        print(result.question["question"])
        feedback = input("\nYour feedback: ")
        result = meeting.resume(result.thread_id, {"feedback": feedback})

    print("\nMeeting completed.")
    print(result.output.get("meeting_summary", ""))


if __name__ == "__main__":
    main()
