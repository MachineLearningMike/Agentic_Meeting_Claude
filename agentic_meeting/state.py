"""Meeting state: plain, JSON-serializable dictionaries.

The whole state can be saved to a checkpoint store and loaded again, which is
how a meeting pauses for a human answer and resumes later.
"""

from __future__ import annotations

from typing import Any, Literal, TypedDict

from .schemas import (
    ActionItem,
    AgendaItem,
    Argument,
    Decision,
    MeetingObjective,
    MeetingProblem,
    MeetingRecord,
    Participant,
    ProjectContext,
    Proposal,
    UnresolvedIssue,
)


class Message(TypedDict, total=False):
    """One entry of the meeting transcript."""

    role: Literal["system", "user", "assistant"]
    content: str
    name: str  # participant id, when a participant spoke


def message(role: str, content: str, name: str | None = None) -> Message:
    entry: Message = {"role": role, "content": content}
    if name:
        entry["name"] = name
    return entry


class MeetingState(TypedDict, total=False):
    # Identity and context
    meeting_id: str
    project_id: str
    topic: str
    background: str
    language: str
    prior_state: dict[str, Any]
    project_context: ProjectContext

    # Meeting inputs
    objectives: list[MeetingObjective]
    problems: list[MeetingProblem]
    participants: list[Participant]

    # Transcript
    messages: list[Message]

    # Structured artifacts
    proposals: list[Proposal]
    arguments: list[Argument]
    decisions: list[Decision]
    action_items: list[ActionItem]
    unresolved_issues: list[UnresolvedIssue]

    # Aggregate structured record
    meeting_record: MeetingRecord
    extraction_error: str
    extraction_warnings: list[str]
    meeting_saved: bool

    # Agenda
    agenda: list[AgendaItem]
    agenda_complete: bool

    # Deliberation control
    current_agenda_id: str
    current_agenda: str
    round_count: int
    total_rounds: int
    max_rounds_per_agenda: int
    max_total_rounds: int
    moderator_action: str
    moderator_question: str
    moderator_reasoning: str
    moderator_target_agenda_id: str
    moderator_ask_participant_id: str
    agenda_summary: str
    user_feedback: str
    user_declined_feedback: bool

    # Analysis
    context_analysis: str
    latest_deliberation: str

    # Final output
    meeting_summary: str


class MeetingInput(TypedDict, total=False):
    """What a caller provides to start a meeting.

    meeting_id is optional; a new one is generated when it is missing. Reusing
    an id saves the meeting over the earlier record with that id.
    """

    meeting_id: str
    project_id: str
    topic: str
    background: str
    language: str
    prior_state: dict[str, Any]
    project_context: ProjectContext
    objectives: list[MeetingObjective]
    problems: list[MeetingProblem]
    participants: list[Participant]
    agenda: list[AgendaItem]
    max_rounds_per_agenda: int
    max_total_rounds: int


class MeetingOutput(TypedDict, total=False):
    """What a finished meeting returns to its caller.

    The transcript and deliberation control stay inside the meeting; read them
    from MeetingResult.state when needed.
    """

    meeting_id: str
    project_id: str
    meeting_record: MeetingRecord
    meeting_summary: str
    agenda: list[AgendaItem]
    extraction_error: str
    extraction_warnings: list[str]
    meeting_saved: bool


# Updates to these keys are appended rather than replaced.
APPEND_KEYS = frozenset({
    "messages",
    "proposals",
    "arguments",
    "decisions",
    "action_items",
    "unresolved_issues",
})


def merge(state: dict, update: dict) -> dict:
    """Return a new state with a node's update applied."""

    merged = dict(state)

    for key, value in update.items():
        if key in APPEND_KEYS:
            merged[key] = list(merged.get(key, [])) + list(value)
        else:
            merged[key] = value

    return merged
