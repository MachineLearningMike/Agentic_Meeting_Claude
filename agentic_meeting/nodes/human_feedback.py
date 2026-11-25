"""Asking a human and taking in the answer.

The meeting pauses after feedback_request (the engine saves a checkpoint and
returns the question); receive_feedback applies the answer when it resumes.
"""

from ..state import message


def find_human(state: dict, participant_id: str | None) -> dict | None:
    return next(
        (
            participant
            for participant in state.get("participants", [])
            if participant["id"] == participant_id and participant.get("is_human")
        ),
        None,
    )


def feedback_request(state: dict) -> dict:
    """The question shown to the human while the meeting is paused."""

    asked = find_human(state, state.get("moderator_ask_participant_id"))

    return {
        "type": "human_feedback",
        "question": state.get(
            "moderator_question", "What is your assessment of the discussion?"
        ),
        "participant": (
            {"id": asked["id"], "name": asked.get("name", asked["id"])}
            if asked
            else None
        ),
        "agenda": state.get("current_agenda", ""),
        "latest_deliberation": state.get("latest_deliberation", ""),
    }


def receive_feedback(state: dict, response) -> dict:
    """Apply an answer: a string, or a dict with "feedback" and optionally
    "decline_future_questions" and "participant_id" (who answered)."""

    asked = find_human(state, state.get("moderator_ask_participant_id"))

    if isinstance(response, dict):
        feedback = str(response.get("feedback", "")).strip()
        decline_future = bool(response.get("decline_future_questions", False))
        # Whoever answered a general question may identify themselves.
        speaker = find_human(state, response.get("participant_id")) or asked
    else:
        feedback = str(response).strip()
        decline_future = False
        speaker = asked

    return {
        "user_feedback": feedback,
        "user_declined_feedback": decline_future,
        "messages": [
            message(
                "user",
                f"[{speaker.get('name', speaker['id'])}] {feedback}" if speaker else feedback,
                name=speaker["id"] if speaker else None,
            )
        ],
    }
