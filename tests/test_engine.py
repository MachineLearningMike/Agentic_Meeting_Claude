"""The engine's own contract: results, checkpoints, events and errors."""

import pytest

from agentic_meeting.testing import ASK, CONCLUDE, ScriptedModel
from agentic_meeting import (
    Meeting,
    MeetingOutput,
    MeetingStore,
    MemoryCheckpoints,
    SqliteCheckpoints,
)
from agentic_meeting.engine import next_step


def test_finished_meeting_returns_its_output():
    result = Meeting(ScriptedModel()).start({"topic": "Trading"})

    assert result.finished
    assert result.question is None
    assert set(result.output) <= set(MeetingOutput.__annotations__)
    assert {"meeting_id", "meeting_record", "meeting_summary"} <= set(result.output)
    assert "messages" not in result.output
    assert result.state["messages"]


def test_thread_id_is_generated_or_kept():
    meeting = Meeting(ScriptedModel())

    assert meeting.start({"topic": "T"}).thread_id
    assert meeting.start({"topic": "T"}, thread_id="mine").thread_id == "mine"


def test_input_outside_the_contract_is_ignored():
    result = Meeting(ScriptedModel()).start(
        {"topic": "Trading", "round_count": 99, "moderator_action": "x"}
    )

    assert result.state["round_count"] == 1


def test_paused_meeting_reports_its_question():
    meeting = Meeting(ScriptedModel(replies={"moderator": [ASK, CONCLUDE]}))

    result = meeting.start({"topic": "Trading"}, thread_id="t")

    assert not result.finished
    assert result.question["question"] == "Which approach?"
    assert meeting.pending_question("t") == result.question
    assert meeting.get_state("t")["moderator_action"] == "request_user_feedback"

    done = meeting.resume("t", "Adaptive.")

    assert done.finished
    assert meeting.pending_question("t") is None


@pytest.mark.parametrize("thread_id", ["unknown", "finished"])
def test_resume_needs_a_waiting_meeting(thread_id):
    meeting = Meeting(ScriptedModel())
    meeting.start({"topic": "T"}, thread_id="finished")

    with pytest.raises(ValueError, match="is not waiting for an answer"):
        meeting.resume(thread_id, "x")


def test_missing_model_is_an_error():
    with pytest.raises(ValueError, match="No meeting model supplied"):
        Meeting(None).start({"topic": "T"})


def test_events_surround_every_step():
    events = []
    Meeting(ScriptedModel(), on_event=events.append).start({"topic": "T"})

    starts = [e["step"] for e in events if e["type"] == "step_start"]
    ends = [e["step"] for e in events if e["type"] == "step_end"]

    assert starts == ends == [
        "initialize", "load_project_memory", "analyze_context", "deliberate",
        "moderate", "finalize", "extract_artifacts", "save_meeting",
    ]
    moderate_end = next(e for e in events if e["type"] == "step_end" and e["step"] == "moderate")
    assert moderate_end["update"]["moderator_action"] == "conclude_meeting"


class Broken(ScriptedModel):
    def complete(self, request):
        if request.node == "deliberation":
            raise RuntimeError("network down")
        return super().complete(request)


def test_failing_step_raises_and_leaves_a_checkpoint_at_that_step():
    events = []
    checkpoints = MemoryCheckpoints()

    with pytest.raises(RuntimeError, match="network down"):
        Meeting(Broken(), checkpoints=checkpoints, on_event=events.append).start(
            {"topic": "T"}, thread_id="t"
        )

    assert events[-1] == {"type": "step_error", "step": "deliberate", "error": "network down"}
    state, step = checkpoints.load("t")
    assert step == "deliberate"
    assert state["context_analysis"] == "Analysis."


def test_sqlite_checkpoints_survive_a_new_engine(tmp_path):
    path = str(tmp_path / "checkpoints.db")
    first = Meeting(
        ScriptedModel(replies={"moderator": [ASK]}),
        checkpoints=SqliteCheckpoints(path),
    )
    first.start({"topic": "取引頻度"}, thread_id="t")

    second = Meeting(ScriptedModel(), checkpoints=SqliteCheckpoints(path))

    assert second.pending_question("t")["question"] == "Which approach?"
    result = second.resume("t", {"feedback": "適応型"})
    assert result.finished
    assert result.state["topic"] == "取引頻度"
    assert result.state["user_feedback"] == "適応型"


def test_checkpoints_are_copies():
    checkpoints = MemoryCheckpoints()
    state = {"items": [1]}
    checkpoints.save("t", state, "x")
    state["items"].append(2)

    assert checkpoints.load("t") == ({"items": [1]}, "x")
    assert checkpoints.load("missing") is None


def test_a_parent_process_can_run_meetings_as_a_step(tmp_path):
    store = MeetingStore(str(tmp_path / "meetings.db"))

    def project_step(question: str) -> str:
        """A parent workflow step that holds a meeting and uses its record."""
        result = Meeting(ScriptedModel(), store=store).start(
            {"project_id": "trading-ai", "topic": question}
        )
        return result.output["meeting_record"]["topic"]

    assert project_step("Trade less?") == "Trade less?"
    assert len(store.list_meetings("trading-ai")) == 1


def test_routing_rejects_unknown_moderator_actions():
    with pytest.raises(ValueError, match="Unknown moderator action"):
        next_step("moderate", {"moderator_action": "dance"})


def test_structured_nodes_send_a_schema_and_text_nodes_do_not():
    model = ScriptedModel()
    Meeting(model).start({"topic": "T"})

    with_schema = {c["node"] for c in model.calls if c["schema"] is not None}
    assert with_schema == {"moderator", "extraction"}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
