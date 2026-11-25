import json

import pytest

from agentic_meeting import Meeting
from agentic_meeting.testing import ASK, CONCLUDE, CONTINUE, ScriptedModel


INITIAL_STATE = {
    "project_id": "trading-ai",
    "topic": "Adaptive trading frequency",
    "background": "The trading system generates too many trades.",
    "problems": [
        {"id": "problem-001", "description": "Trading frequency is too high."}
    ],
    "objectives": [],
    "max_rounds_per_agenda": 3,
}


def start(model, **overrides):
    """Start a meeting and return (meeting, result)."""

    meeting = Meeting(model)
    return meeting, meeting.start({**INITIAL_STATE, **overrides})


def test_normal_meeting_completes():
    model = ScriptedModel()

    _, result = start(model)

    assert result.finished
    assert result.finished
    assert result.state["meeting_summary"] == "Report."
    assert result.state["round_count"] == 1
    assert result.state["current_agenda"] == "Trading frequency is too high."

    record = result.state["meeting_record"]
    assert record["meeting_id"] == result.state["meeting_id"]
    assert record["project_id"] == "trading-ai"
    assert record["topic"] == "Adaptive trading frequency"

    assert [call["node"] for call in model.calls] == [
        "analysis", "deliberation", "moderator", "finalize", "extraction",
    ]


def test_feedback_pause_and_resume():
    model = ScriptedModel(replies={"moderator": [ASK, CONCLUDE]})

    meeting, result = start(model)

    assert result.question["question"] == "Which approach?"

    result = meeting.resume(result.thread_id, {"feedback": "I prefer adaptive frequency."})

    assert result.finished
    assert result.state["user_feedback"] == "I prefer adaptive frequency."
    assert result.state["round_count"] == 2
    assert any(
        message["role"] == "user"
        and message["content"] == "I prefer adaptive frequency."
        for message in result.state["messages"]
    )

    # Feedback reaches the analysis and deliberation that follow it.
    second_analysis = json.loads(model.calls_for("analysis")[1]["user"])
    assert second_analysis["user_feedback"] == "I prefer adaptive frequency."


def test_resume_accepts_plain_string():
    model = ScriptedModel(replies={"moderator": [ASK, CONCLUDE]})

    meeting, result = start(model)
    result = meeting.resume(result.thread_id, "Plain answer")

    assert result.state["user_feedback"] == "Plain answer"
    assert result.finished


def test_declining_feedback_stops_questions():
    model = ScriptedModel(replies={"moderator": [ASK] * 10})

    meeting, result = start(model, max_rounds_per_agenda=5)
    questions = 0

    while result.question:
        questions += 1
        result = meeting.resume(result.thread_id, {"feedback": "", "decline_future_questions": True})

    assert questions == 1
    assert result.finished
    assert result.state["round_count"] == 5


def test_round_cap_bounds_feedback_requests():
    model = ScriptedModel(replies={"moderator": [ASK] * 10})

    meeting, result = start(model, max_rounds_per_agenda=2)
    questions = 0

    while result.question:
        questions += 1
        result = meeting.resume(result.thread_id, {"feedback": "ok"})

    assert questions == 1
    assert result.state["round_count"] == 2
    assert result.finished


def test_round_cap_bounds_continued_discussion():
    model = ScriptedModel(replies={"moderator": [CONTINUE] * 10})

    _, result = start(model, max_rounds_per_agenda=3)

    assert result.state["round_count"] == 3
    assert result.state["moderator_action"] == "conclude_agenda"
    assert result.state["agenda"][0]["status"] == "concluded"


def test_invalid_moderator_output_continues_until_cap():
    model = ScriptedModel(replies={"moderator": ["not json"] * 10})

    _, result = start(model, max_rounds_per_agenda=2)

    assert result.state["round_count"] == 2
    assert result.finished
    assert result.state["moderator_reasoning"] == "Invalid moderator response."


def test_extracted_artifacts_are_recorded():
    extraction = json.dumps({
        "proposals": [
            {"description": "Develop an adaptive policy.", "status": "proposed"}
        ],
        "arguments": [
            {"content": "Reduces unnecessary trades.", "position": "support"}
        ],
        "decisions": [],
        "action_items": [],
        "unresolved_issues": [
            {"description": "Evaluate transaction costs.", "status": "open"}
        ],
    })
    model = ScriptedModel(replies={"extraction": [extraction]})

    _, result = start(model)
    record = result.state["meeting_record"]

    assert not result.state["extraction_error"]
    assert [p["description"] for p in record["proposals"]] == [
        "Develop an adaptive policy."
    ]
    assert record["arguments"][0]["position"] == "support"
    assert record["unresolved_issues"][0]["status"] == "open"
    assert all(item["id"] for item in record["proposals"] + record["arguments"])
    assert result.state["proposals"] == record["proposals"]


def test_invalid_extraction_is_retried_once_with_the_error():
    valid = json.dumps({"proposals": [{"description": "Adaptive policy"}]})
    model = ScriptedModel(replies={"extraction": ["not json", valid]})

    _, result = start(model)
    attempts = model.calls_for("extraction")

    assert len(attempts) == 2
    assert "previous_attempt_error" not in json.loads(attempts[0]["user"])
    assert "No JSON object found" in json.loads(attempts[1]["user"])["previous_attempt_error"]
    assert not result.state["extraction_error"]
    assert result.state["meeting_record"]["proposals"][0]["description"] == "Adaptive policy"


def test_invalid_extraction_keeps_meeting():
    model = ScriptedModel(replies={"extraction": ["not json", "still not json"]})

    _, result = start(model)
    record = result.state["meeting_record"]

    assert result.finished
    assert result.state["meeting_summary"] == "Report."
    assert result.state["extraction_error"] == "No JSON object found in model response."
    assert record["extraction_error"] == result.state["extraction_error"]
    assert record["proposals"] == []


def test_missing_model_raises():
    with pytest.raises(ValueError, match="No meeting model supplied"):
        Meeting(None).start(INITIAL_STATE)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
