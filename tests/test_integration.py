"""End-to-end behavior: several meetings, restarts and concurrent projects."""

import json

import pytest

from agentic_meeting.testing import ASK, CONCLUDE, ScriptedModel, moderator_reply
from agentic_meeting import Meeting, MeetingStore, MemoryCheckpoints, SqliteCheckpoints


PARTICIPANTS = [
    {"id": "researcher", "name": "Researcher", "role": "develops strategies"},
    {"id": "risk", "name": "Risk analyst", "role": "identifies risks"},
    {"id": "owner", "name": "Haruto", "role": "project owner", "is_human": True},
]


@pytest.fixture
def store(tmp_path):
    return MeetingStore(str(tmp_path / "meetings.db"))


def next_item(summary):
    return moderator_reply("conclude_agenda", agenda_summary=summary)


def run(store, model, meeting_input, answers=()):
    """Run a meeting to the end, answering each feedback question in turn."""

    meeting = Meeting(model, store=store)
    result = meeting.start(meeting_input)

    for answer in answers:
        assert result.question, "expected a feedback question"
        result = meeting.resume(result.thread_id, {"feedback": answer})

    assert result.finished
    return result


def test_trading_project_across_three_meetings(store):
    # Meeting 1: two agenda items, a question to the owner, open issues.
    ask_owner = moderator_reply(
        "request_user_feedback", question="Fixed or adaptive?", ask_participant_id="owner"
    )
    first = run(store, ScriptedModel(replies={
        "moderator": [ask_owner, next_item("Adaptive preferred."), next_item("Costs open.")],
        "extraction": [json.dumps({
            "proposals": [{"ref": "p1", "description": "Adaptive trading frequency",
                           "created_by": "researcher", "agenda_id": "approach"}],
            "arguments": [{"content": "Drawdown may rise", "position": "oppose",
                           "target_id": "p1", "author_id": "risk", "agenda_id": "approach"}],
            "unresolved_issues": [
                {"description": "Which market variables drive frequency?", "agenda_id": "approach"},
                {"description": "Transaction costs are unevaluated", "agenda_id": "costs"},
            ],
        })],
    }), {
        "project_id": "trading-ai",
        "topic": "Reduce excessive trading frequency",
        "participants": PARTICIPANTS,
        "agenda": [
            {"id": "approach", "description": "Fixed versus adaptive frequency"},
            {"id": "costs", "description": "Cost impact"},
        ],
    }, answers=["Adaptive, if drawdown stays under 5%."])

    record = first.state["meeting_record"]
    proposal_id = record["proposals"][0]["id"]
    variables_id, costs_id = [i["id"] for i in record["unresolved_issues"]]

    assert first.state["extraction_warnings"] == []
    assert record["arguments"][0]["target_id"] == proposal_id
    assert [i["status"] for i in record["agenda_items"]] == ["concluded", "concluded"]
    assert any(m.get("name") == "owner" for m in first.state["messages"] if m["role"] == "user")

    # Meeting 2 starts from Meeting 1 and refines the policy.
    model = ScriptedModel(replies={
        "moderator": [CONCLUDE],
        "extraction": [json.dumps({
            "proposals": [{"description": "adaptive trading frequency"}],
            "decisions": [{"ref": "d1", "description": "Use volatility as the driver",
                           "status": "approved", "approved_by": "owner"}],
            "action_items": [{"description": "Backtest the volatility policy",
                              "owner_id": "researcher"}],
            "updates": [
                {"id": proposal_id, "status": "under_discussion"},
                {"id": variables_id, "status": "resolved", "note": "Volatility chosen"},
            ],
        })],
    })
    second = run(store, model, {
        "project_id": "trading-ai",
        "topic": "Design the adaptive policy",
        "participants": PARTICIPANTS,
    })

    analysis = json.loads(model.calls_for("analysis")[0]["user"])["project_context"]
    assert [p["id"] for p in analysis["current_state"]["proposals"]] == [proposal_id]
    assert analysis["history"][0]["meeting_id"] == first.state["meeting_id"]
    assert second.state["meeting_record"]["proposals"] == []  # repeat was dropped
    decision_id = second.state["meeting_record"]["decisions"][0]["id"]

    # Meeting 3 reviews the backtest, replaces the decision and accepts.
    third = run(store, ScriptedModel(replies={
        "moderator": [CONCLUDE],
        "extraction": [json.dumps({
            "decisions": [{"ref": "d2", "description": "Use volatility and spread",
                           "status": "approved", "approved_by": "owner"}],
            "updates": [
                {"id": decision_id, "status": "superseded", "superseded_by": "d2"},
                {"id": proposal_id, "status": "accepted"},
                {"id": costs_id, "status": "resolved"},
            ],
        })],
    }), {
        "project_id": "trading-ai",
        "topic": "Review backtest results",
        "participants": PARTICIPANTS,
    })

    assert third.state["extraction_warnings"] == []

    state = store.get_project_state("trading-ai")
    decisions = {d["id"]: d for d in state["decisions"]}
    new_decision_id = third.state["meeting_record"]["decisions"][0]["id"]

    assert [(p["id"], p["status"]) for p in state["proposals"]] == [(proposal_id, "accepted")]
    assert decisions[decision_id]["status"] == "superseded"
    assert decisions[decision_id]["superseded_by"] == new_decision_id
    assert decisions[new_decision_id]["approved_by"] == "owner"
    assert {i["status"] for i in state["unresolved_issues"]} == {"resolved"}
    assert state["action_items"][0]["owner_id"] == "researcher"
    assert len(store.list_meetings("trading-ai")) == 3

    # History keeps what each meeting saw at the time.
    assert store.get(second.state["meeting_id"])["decisions"][0]["status"] == "approved"


def test_paused_meeting_resumes_after_restart(tmp_path, store):
    checkpoints = str(tmp_path / "checkpoints.db")
    meeting_input = {"project_id": "trading-ai", "topic": "Adaptive trading frequency"}

    # First process: the meeting pauses for feedback.
    first_model = ScriptedModel(replies={"moderator": [ASK]})
    paused = Meeting(
        first_model, store=store, checkpoints=SqliteCheckpoints(checkpoints)
    ).start(meeting_input, thread_id="meeting-1")

    assert paused.question["question"] == "Which approach?"
    assert store.list_meetings() == []

    # Second process: new checkpoint connection and model pick the meeting up.
    second_model = ScriptedModel(replies={"moderator": [CONCLUDE]})
    result = Meeting(
        second_model, store=store, checkpoints=SqliteCheckpoints(checkpoints)
    ).resume("meeting-1", {"feedback": "Adaptive."})

    assert result.state["meeting_saved"] is True
    assert result.state["meeting_id"] == paused.state["meeting_id"]
    assert [c["node"] for c in second_model.calls] == [
        "analysis", "deliberation", "moderator", "finalize", "extraction",
    ]
    feedback = json.loads(second_model.calls_for("analysis")[0]["user"])["user_feedback"]
    assert feedback == "Adaptive."
    assert store.get(result.state["meeting_id"]) is not None


def test_interleaved_projects_stay_isolated(store):
    checkpoints = MemoryCheckpoints()

    def meeting(model):
        return Meeting(model, store=store, checkpoints=checkpoints)

    def extraction(text):
        return [json.dumps({"proposals": [{"description": text}]})]

    model_a = ScriptedModel(replies={"moderator": [ASK, CONCLUDE], "extraction": extraction("A idea")})
    model_b = ScriptedModel(replies={"extraction": extraction("B idea")})

    # A pauses; B runs to the end in between; then A resumes.
    paused = meeting(model_a).start({"project_id": "alpha", "topic": "A"}, thread_id="a")
    done_b = meeting(model_b).start({"project_id": "beta", "topic": "B"}, thread_id="b")
    done_a = meeting(model_a).resume("a", {"feedback": "go"})

    assert paused.question
    assert done_a.state["topic"] == "A" and done_b.state["topic"] == "B"
    assert done_a.state["meeting_id"] != done_b.state["meeting_id"]

    def proposals(project_id):
        return [p["description"] for p in store.get_project_state(project_id)["proposals"]]

    assert proposals("alpha") == ["A idea"]
    assert proposals("beta") == ["B idea"]
    assert [r["topic"] for r in store.list_meetings("alpha")] == ["A"]


def test_empty_feedback_lets_the_meeting_continue():
    meeting = Meeting(ScriptedModel(replies={"moderator": [ASK, CONCLUDE]}))

    paused = meeting.start({"topic": "Trading"})
    result = meeting.resume(paused.thread_id, {"feedback": ""})

    assert result.state["user_feedback"] == ""
    assert result.finished
    assert result.state["meeting_summary"] == "Report."


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
