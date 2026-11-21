import json

import pytest

from fakes import ASK, CONCLUDE, CONTINUE, Resume, ScriptedModel, invoke, new_harness


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
    """Start a meeting on a fresh harness and return (harness, config, result)."""

    harness = new_harness()
    config = {"configurable": {"thread_id": "test", "meeting_model": model}}
    result = invoke(harness, {**INITIAL_STATE, **overrides}, config)
    return harness, config, result


def test_normal_meeting_completes():
    model = ScriptedModel()

    _, _, result = start(model)

    assert "__question__" not in result
    assert result["current_phase"] == "completed"
    assert result["meeting_summary"] == "Report."
    assert result["round_count"] == 1
    assert result["current_agenda"] == "Trading frequency is too high."

    record = result["meeting_record"]
    assert record["meeting_id"] == result["meeting_id"]
    assert record["project_id"] == "trading-ai"
    assert record["topic"] == "Adaptive trading frequency"

    assert [call["node"] for call in model.calls] == [
        "analysis", "deliberation", "moderator", "finalize", "extraction",
    ]


def test_feedback_interrupt_and_resume():
    model = ScriptedModel(replies={"moderator": [ASK, CONCLUDE]})

    harness, config, result = start(model)

    assert result["__question__"][0].value["question"] == "Which approach?"

    result = invoke(harness, 
        Resume({"feedback": "I prefer adaptive frequency."}), config
    )

    assert "__question__" not in result
    assert result["user_feedback"] == "I prefer adaptive frequency."
    assert result["round_count"] == 2
    assert any(
        message["role"] == "human"
        and message["content"] == "I prefer adaptive frequency."
        for message in result["messages"]
    )

    # Feedback reaches the analysis and deliberation that follow it.
    second_analysis = json.loads(model.calls_for("analysis")[1]["human"])
    assert second_analysis["user_feedback"] == "I prefer adaptive frequency."


def test_resume_accepts_plain_string():
    model = ScriptedModel(replies={"moderator": [ASK, CONCLUDE]})

    harness, config, _ = start(model)
    result = invoke(harness, Resume("Plain answer"), config)

    assert result["user_feedback"] == "Plain answer"
    assert result["current_phase"] == "completed"


def test_declining_feedback_stops_questions():
    model = ScriptedModel(replies={"moderator": [ASK] * 10})

    harness, config, result = start(model, max_rounds_per_agenda=5)
    interrupts = 0

    while result.get("__question__"):
        interrupts += 1
        result = invoke(harness, 
            Resume({"feedback": "", "decline_future_questions": True}),
            config,
        )

    assert interrupts == 1
    assert result["current_phase"] == "completed"
    assert result["round_count"] == 5


def test_round_cap_bounds_feedback_requests():
    model = ScriptedModel(replies={"moderator": [ASK] * 10})

    harness, config, result = start(model, max_rounds_per_agenda=2)
    interrupts = 0

    while result.get("__question__"):
        interrupts += 1
        result = invoke(harness, Resume({"feedback": "ok"}), config)

    assert interrupts == 1
    assert result["round_count"] == 2
    assert result["current_phase"] == "completed"


def test_round_cap_bounds_continued_discussion():
    model = ScriptedModel(replies={"moderator": [CONTINUE] * 10})

    _, _, result = start(model, max_rounds_per_agenda=3)

    assert result["round_count"] == 3
    assert result["moderator_action"] == "conclude_agenda"
    assert result["agenda"][0]["status"] == "concluded"


def test_invalid_moderator_output_continues_until_cap():
    model = ScriptedModel(replies={"moderator": ["not json"] * 10})

    _, _, result = start(model, max_rounds_per_agenda=2)

    assert result["round_count"] == 2
    assert result["current_phase"] == "completed"
    assert result["moderator_reasoning"] == "Invalid moderator response."


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

    _, _, result = start(model)
    record = result["meeting_record"]

    assert not result["extraction_error"]
    assert [p["description"] for p in record["proposals"]] == [
        "Develop an adaptive policy."
    ]
    assert record["arguments"][0]["position"] == "support"
    assert record["unresolved_issues"][0]["status"] == "open"
    assert all(item["id"] for item in record["proposals"] + record["arguments"])
    assert result["proposals"] == record["proposals"]


def test_invalid_extraction_is_retried_once_with_the_error():
    valid = json.dumps({"proposals": [{"description": "Adaptive policy"}]})
    model = ScriptedModel(replies={"extraction": ["not json", valid]})

    _, _, result = start(model)
    attempts = model.calls_for("extraction")

    assert len(attempts) == 2
    assert "previous_attempt_error" not in json.loads(attempts[0]["human"])
    assert "No JSON object found" in json.loads(attempts[1]["human"])["previous_attempt_error"]
    assert not result["extraction_error"]
    assert result["meeting_record"]["proposals"][0]["description"] == "Adaptive policy"


def test_invalid_extraction_keeps_meeting():
    model = ScriptedModel(replies={"extraction": ["not json", "still not json"]})

    _, _, result = start(model)
    record = result["meeting_record"]

    assert result["current_phase"] == "completed"
    assert result["meeting_summary"] == "Report."
    assert result["extraction_error"] == "No JSON object found in model response."
    assert record["extraction_error"] == result["extraction_error"]
    assert record["proposals"] == []


def test_missing_model_raises():
    harness = new_harness()

    with pytest.raises(ValueError, match="No meeting model supplied"):
        invoke(harness, INITIAL_STATE, {"configurable": {"thread_id": "test"}})


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
