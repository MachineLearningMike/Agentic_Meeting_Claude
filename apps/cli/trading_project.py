"""The trading-frequency project as three meetings that build on each other.

Run one meeting at a time:

    python apps/cli/trading_project.py 1
    python apps/cli/trading_project.py 2
    python apps/cli/trading_project.py 3

The default model is Claude Opus 5.5; pass --model to choose another Claude model.

Use --model fake to try it offline with a scripted model.

Project memory is kept in data/meetings.db and checkpoints in
data/checkpoints.db, so a meeting paused for your answer can be resumed later
by running the same command again.
"""

import argparse
import os
import sys
from pathlib import Path

from agentic_meeting import ClaudeModel, Meeting, MeetingStore, SqliteCheckpoints
from agentic_meeting.model import DEFAULT_MODEL
from agentic_meeting.testing import demo_model


PROJECT_ID = "trading-ai"
# <repo>/data, shared with the other apps.
DATA = Path(__file__).resolve().parents[2] / "data"

PARTICIPANTS = [
    {"id": "researcher", "name": "Researcher", "role": "researcher",
     "description": "Develops alternative trading strategies."},
    {"id": "quant", "name": "Quant analyst", "role": "quant analyst",
     "description": "Evaluates mathematical and statistical assumptions."},
    {"id": "risk", "name": "Risk analyst", "role": "risk analyst",
     "description": "Identifies risks, drawdowns and failure modes."},
    {"id": "engineer", "name": "Engineer", "role": "engineer",
     "description": "Assesses implementation feasibility and cost."},
    {"id": "owner", "name": "Project owner", "role": "project owner",
     "is_human": True},
]

MEETINGS = {
    1: {
        "topic": "Reduce excessive trading frequency",
        "background": "The trading system generates too many trades; fees "
                      "and slippage reduce profitability.",
        "agenda": [
            {"description": "Fixed versus adaptive trading frequency"},
            {"description": "What is still unknown"},
        ],
    },
    2: {
        "topic": "Design the adaptive trading-frequency policy",
        "agenda": [
            {"description": "Which market-state variables control frequency?"},
            {"description": "How should the policy be evaluated?"},
        ],
    },
    3: {
        "topic": "Review the policy evaluation and decide",
        "background": "Paste backtest results here before running.",
        "agenda": [
            {"description": "Review evaluation results"},
            {"description": "Decide whether to adopt the policy"},
        ],
    },
}


def print_project_state(store: MeetingStore) -> None:
    state = store.get_project_state(PROJECT_ID)

    print("\n===== Current project state =====")
    for field, items in state.items():
        print(f"\n{field} ({len(items)})")
        for item in items:
            text = item.get("description") or item.get("content", "")
            status = item.get("status") or item.get("position", "")
            print(f"  - [{status}] {text}")


def main() -> None:
    # Redirected output on Windows may use a code page without e.g. Japanese;
    # replace what it cannot show instead of crashing.
    sys.stdout.reconfigure(errors="replace")

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("meeting", type=int, choices=sorted(MEETINGS))
    parser.add_argument(
        "--model", default=os.environ.get("MEETING_MODEL", DEFAULT_MODEL)
    )
    args = parser.parse_args()

    DATA.mkdir(exist_ok=True)
    store = MeetingStore(str(DATA / "meetings.db"))

    meeting = Meeting(
        demo_model() if args.model == "fake" else ClaudeModel(args.model),
        store=store,
        checkpoints=SqliteCheckpoints(str(DATA / "checkpoints.db")),
    )
    thread_id = f"{PROJECT_ID}-meeting-{args.meeting}"

    # Resume a paused meeting, or start it.
    state = meeting.get_state(thread_id)
    question = meeting.pending_question(thread_id)

    if question:
        print("Resuming the paused meeting.")
    elif state and state.get("meeting_saved"):
        print(f"Meeting {args.meeting} is already finished.")
        print_project_state(store)
        return
    else:
        result = meeting.start({
            "project_id": PROJECT_ID,
            "participants": PARTICIPANTS,
            "max_rounds_per_agenda": 2,
            **MEETINGS[args.meeting],
        }, thread_id=thread_id)
        question = result.question

    while question:
        asked = question.get("participant") or {}
        print(f"\nModerator asks {asked.get('name', 'you')}: {question['question']}")
        answer = input("Your answer (empty to skip): ")
        result = meeting.resume(thread_id, {"feedback": answer})
        question = result.question

    output = result.output

    print("\n===== Meeting report =====\n")
    print(output.get("meeting_summary", ""))

    for warning in output.get("extraction_warnings", []):
        print(f"Warning: {warning}")
    if output.get("extraction_error"):
        print(f"Extraction failed: {output['extraction_error']}")

    print_project_state(store)


if __name__ == "__main__":
    main()
