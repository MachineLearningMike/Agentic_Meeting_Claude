"""Test helpers for the meeting engine."""

from __future__ import annotations

from dataclasses import dataclass

from agentic_meeting import Meeting, MemoryCheckpoints
from agentic_meeting.testing import (  # noqa: F401  (re-exported for tests)
    ASK,
    CONCLUDE,
    CONTINUE,
    DEFAULT_REPLIES,
    EMPTY_ARTIFACTS,
    ScriptedModel,
    moderator_reply,
)


@dataclass
class Resume:
    """An answer to a waiting meeting, passed to invoke() instead of an input."""

    answer: object


@dataclass
class Question:
    value: dict


class Harness:
    """Runs meetings for tests.

    Each call builds a Meeting from ``config`` (model, store, participant
    models), sharing one checkpoint store, as a stateless server would.
    """

    def __init__(self, checkpoints=None):
        self.checkpoints = checkpoints if checkpoints is not None else MemoryCheckpoints()

    def meeting(self, config: dict) -> Meeting:
        options = config.get("configurable", {})
        return Meeting(
            options.get("meeting_model"),
            store=options.get("meeting_store"),
            participant_models=options.get("participant_models"),
            checkpoints=self.checkpoints,
        )

    def run(self, meeting_input, config: dict):
        thread_id = config["configurable"]["thread_id"]
        meeting = self.meeting(config)

        if isinstance(meeting_input, Resume):
            return meeting.resume(thread_id, meeting_input.answer)
        return meeting.start(meeting_input, thread_id=thread_id)


def new_harness() -> Harness:
    return Harness()


def invoke(harness: Harness, meeting_input, config: dict) -> dict:
    """Run until the meeting finishes or asks; return its full state.

    A waiting meeting's question is under "__question__".
    """

    result = harness.run(meeting_input, config)
    state = dict(result.state)

    if result.question is not None:
        state["__question__"] = [Question(result.question)]

    return state
