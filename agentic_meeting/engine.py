"""The meeting engine: runs the steps of a meeting in order.

    initialize -> load_project_memory -> analyze_context -> deliberate -> moderate
        moderate decides:
            continue           -> deliberate
            ask a human        -> pause; resume() -> analyze_context
            conclude / defer / revisit an agenda item
                               -> advance_agenda -> analyze_context, or finalize
            conclude meeting   -> finalize
    finalize -> extract_artifacts -> save_meeting -> done

The state is saved to the checkpoint store after every step, so a meeting
paused for a human answer can be resumed later, also by another process.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any, Callable

from .checkpoints import CheckpointStore, MemoryCheckpoints
from .model import Model
from .nodes.agenda import advance_agenda
from .nodes.analysis import analyze_context
from .nodes.common import Context
from .nodes.deliberation import deliberate
from .nodes.extract_artifacts import extract_meeting_artifacts
from .nodes.finalize import finalize_meeting
from .nodes.human_feedback import feedback_request, receive_feedback
from .nodes.initialization import initialize_meeting
from .nodes.moderator import moderate
from .nodes.project_memory import load_project_memory, save_meeting
from .state import MeetingInput, MeetingOutput, merge


STEPS: dict[str, Callable[[dict, Context], dict]] = {
    "initialize": initialize_meeting,
    "load_project_memory": load_project_memory,
    "analyze_context": analyze_context,
    "deliberate": deliberate,
    "moderate": moderate,
    "advance_agenda": advance_agenda,
    "finalize": finalize_meeting,
    "extract_artifacts": extract_meeting_artifacts,
    "save_meeting": save_meeting,
}

AWAIT_FEEDBACK = "await_feedback"
DONE = "done"


def next_step(step: str, state: dict) -> str:
    """The step that follows ``step``, given the state after it ran."""

    if step == "moderate":
        action = state.get("moderator_action")
        if action == "continue_discussion":
            return "deliberate"
        if action == "request_user_feedback":
            return AWAIT_FEEDBACK
        if action in ("conclude_agenda", "defer_agenda", "revisit_agenda"):
            return "advance_agenda"
        if action == "conclude_meeting":
            return "finalize"
        raise ValueError(f"Unknown moderator action: {action}")

    if step == "advance_agenda":
        return "finalize" if state.get("agenda_complete") else "analyze_context"

    return {
        "initialize": "load_project_memory",
        "load_project_memory": "analyze_context",
        "analyze_context": "deliberate",
        "deliberate": "moderate",
        "finalize": "extract_artifacts",
        "extract_artifacts": "save_meeting",
        "save_meeting": DONE,
    }[step]


@dataclass
class MeetingResult:
    """Where a meeting stands after start() or resume()."""

    thread_id: str
    state: dict
    question: dict | None = None  # set while the meeting waits for a human

    @property
    def finished(self) -> bool:
        return self.question is None

    @property
    def output(self) -> MeetingOutput:
        return {
            key: self.state[key]
            for key in MeetingOutput.__annotations__
            if key in self.state
        }


class Meeting:
    """Runs meetings.

    model:              the default Model, e.g. ClaudeModel().
    store:              a MeetingStorage for project memory (optional).
    checkpoints:        where running meetings are saved; MemoryCheckpoints by
                        default, SqliteCheckpoints to survive restarts.
    participant_models: {key: Model} for participants with "model": key.
    on_event:           called with {"type": "step_start" | "step_end" |
                        "step_error", "step": name, ...} around every step,
                        e.g. to show progress in a UI.
    """

    def __init__(
        self,
        model: Model | None,
        *,
        store: Any = None,
        checkpoints: CheckpointStore | None = None,
        participant_models: dict[str, Model] | None = None,
        on_event: Callable[[dict], None] | None = None,
    ):
        self.context = Context(
            model=model,
            participant_models=participant_models or {},
            store=store,
        )
        self.checkpoints = checkpoints if checkpoints is not None else MemoryCheckpoints()
        self.on_event = on_event or (lambda event: None)

    def start(self, meeting_input: MeetingInput, thread_id: str | None = None) -> MeetingResult:
        """Start a meeting and run it until it finishes or asks a human."""

        if self.context.model is None:
            raise ValueError("No meeting model supplied.")

        # Only MeetingInput keys are taken from the caller.
        state = {
            key: value
            for key, value in meeting_input.items()
            if key in MeetingInput.__annotations__
        }

        return self._run(thread_id or str(uuid.uuid4()), state, "initialize")

    def resume(self, thread_id: str, answer) -> MeetingResult:
        """Give the waiting meeting its answer and run it on.

        answer is a string, or a dict with "feedback" and optionally
        "decline_future_questions" and "participant_id".
        """

        saved = self.checkpoints.load(thread_id)

        if saved is None or saved[1] != AWAIT_FEEDBACK:
            raise ValueError(f"Meeting {thread_id!r} is not waiting for an answer.")

        state = merge(saved[0], receive_feedback(saved[0], answer))
        return self._run(thread_id, state, "analyze_context")

    def get_state(self, thread_id: str) -> dict | None:
        saved = self.checkpoints.load(thread_id)
        return saved[0] if saved else None

    def pending_question(self, thread_id: str) -> dict | None:
        """The question a paused meeting is waiting on, or None."""

        saved = self.checkpoints.load(thread_id)
        if saved is None or saved[1] != AWAIT_FEEDBACK:
            return None
        return feedback_request(saved[0])

    def _run(self, thread_id: str, state: dict, step: str) -> MeetingResult:
        self.checkpoints.save(thread_id, state, step)

        while step != DONE:
            if step == AWAIT_FEEDBACK:
                self.checkpoints.save(thread_id, state, step)
                return MeetingResult(thread_id, state, feedback_request(state))

            self.on_event({"type": "step_start", "step": step})

            try:
                update = STEPS[step](state, self.context)
            except Exception as error:
                # The last checkpoint already points at this step.
                self.on_event({"type": "step_error", "step": step, "error": str(error)})
                raise

            state = merge(state, update)
            self.on_event({"type": "step_end", "step": step, "update": update})

            step = next_step(step, state)
            self.checkpoints.save(thread_id, state, step)

        return MeetingResult(thread_id, state)
