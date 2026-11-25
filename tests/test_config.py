import json
import sys
import threading

import pytest

from agentic_meeting.testing import ScriptedModel, moderator_reply
from agentic_meeting import (
    ConfigError,
    ConsoleHub,
    Meeting,
    SeatError,
    load_meeting_config,
    parse_meeting_config,
)


TOML = """
topic = "Trade less?"
background = "Too many trades."
max_rounds_per_agenda = 2
agenda = ["Fixed or adaptive?", { id = "a2", description = "Open questions" }]

[[participants]]
name = "Risk analyst"
role = "risk analyst"
statement = "Identifies risks."
instructions = "Quantify the downside."
model = "claude-sonnet-5-5"

[[participants]]
id = "owner"
name = "Haruto"
statement = "Owns the budget."
human = true
language = "日本語"
"""


def config(**changes):
    return {"topic": "Trade less?", **changes}


# --- configuration -----------------------------------------------------------


@pytest.mark.skipif(sys.version_info < (3, 11), reason="tomllib needs Python 3.11")
def test_toml_file_becomes_a_meeting_input(tmp_path):
    path = tmp_path / "meeting.toml"
    path.write_text(TOML, encoding="utf-8")

    assert load_meeting_config(path) == {
        "topic": "Trade less?",
        "background": "Too many trades.",
        "max_rounds_per_agenda": 2,
        "agenda": [
            {"description": "Fixed or adaptive?"},
            {"description": "Open questions", "id": "a2"},
        ],
        "participants": [
            {
                "id": "risk-analyst",
                "name": "Risk analyst",
                "role": "risk analyst",
                "description": "Identifies risks.",
                "instructions": "Quantify the downside.",
                "model": "claude-sonnet-5-5",
            },
            {
                "id": "owner",
                "name": "Haruto",
                "description": "Owns the budget.",
                "language": "日本語",
                "is_human": True,
            },
        ],
    }


def test_json_file_is_read_too(tmp_path):
    path = tmp_path / "meeting.json"
    path.write_text(json.dumps(config(agenda=["A"])), encoding="utf-8")

    assert load_meeting_config(path) == {"topic": "Trade less?", "agenda": [{"description": "A"}]}


def test_ids_are_made_from_names_without_clashing_with_given_ids():
    participants = parse_meeting_config(config(participants=[
        {"name": "Risk Analyst"},
        {"name": "Risk analyst"},
        {"id": "risk-analyst-2", "name": "Third"},
    ]))["participants"]

    assert [p["id"] for p in participants] == ["risk-analyst", "risk-analyst-3", "risk-analyst-2"]


@pytest.mark.parametrize("data, message", [
    ({}, "'topic' is required"),
    (config(agend=["A"]), "Unknown setting in the meeting: agend"),
    (config(agenda=[]), "at least one item"),
    (config(agenda=[" "]), "agenda item 1 has no description"),
    (config(max_rounds_per_agenda=0), "whole number"),
    (config(participants=[{"role": "x"}]), "participant 1 needs a 'name'"),
    (config(participants=[{"name": "A", "rol": "x"}]), "Unknown setting in participant 1: rol"),
    (config(participants=[{"name": "A", "human": "yes"}]), "must be true or false"),
    (config(participants=[{"name": "A", "human": True, "model": "m"}]), "cannot have a 'model'"),
    (config(participants=[{"id": "a"}, {"id": "a"}]), "used twice"),
    (config(participants=[{"name": "A", "statement": "x", "description": "y"}]), "given twice"),
])
def test_mistakes_are_reported(data, message):
    with pytest.raises(ConfigError, match=message):
        parse_meeting_config(data)


def test_file_errors_name_the_file(tmp_path):
    path = tmp_path / "meeting.json"
    path.write_text("{}", encoding="utf-8")

    with pytest.raises(ConfigError, match="meeting.json: 'topic' is required"):
        load_meeting_config(path)

    with pytest.raises(ConfigError, match="Cannot read"):
        load_meeting_config(tmp_path / "missing.json")


def test_role_statements_reach_the_prompts():
    model = ScriptedModel()
    Meeting(model).start(parse_meeting_config(config(participants=[
        {"name": "Risk", "statement": "Identifies risks."},
        {"name": "Haruto", "statement": "Owns the budget.", "human": True},
    ])))

    assert "Identifies risks." in model.calls_for("deliberation")[0]["system"]
    # The moderator knows the human's role statement, to decide whom to ask.
    seen = json.loads(model.calls_for("moderator")[0]["user"])["participants"]
    assert seen[1] == {
        "id": "haruto", "name": "Haruto", "role": "", "is_human": True,
        "description": "Owns the budget.",
    }


# --- consoles ----------------------------------------------------------------


PEOPLE = [
    {"id": "risk", "name": "Risk analyst"},
    {"id": "owner", "name": "Haruto", "is_human": True},
    {"id": "advisor", "name": "Mika", "is_human": True},
]


class Screen:
    """A console that keeps what it was shown."""

    def __init__(self):
        self.lines = []

    def __call__(self, text):
        self.lines.append(text)

    def saw(self, text):
        return any(text in line for line in self.lines)


def question(participant_id=None, text="Which approach?"):
    participant = {"id": participant_id, "name": participant_id} if participant_id else None
    return {"question": text, "participant": participant}


def asking(hub, payload):
    """Ask in the background; returns a function that waits for the reply."""

    replies = []
    thread = threading.Thread(target=lambda: replies.append(hub.ask(payload)))
    thread.start()

    # ask() announces the question before it waits.
    while hub._question is None and thread.is_alive():
        pass

    def reply():
        thread.join(timeout=5)
        return replies[0]

    return reply


def test_only_human_participants_have_seats():
    hub = ConsoleHub(PEOPLE)

    assert hub.register("Haruto", Screen())["id"] == "owner"  # by name
    assert [p["id"] for p in hub.missing()] == ["advisor"]

    with pytest.raises(SeatError, match="No human participant 'risk'"):
        hub.register("risk", Screen())
    with pytest.raises(SeatError, match="already has a console"):
        hub.register("owner", Screen())
    with pytest.raises(SeatError, match=r"Seats: owner \(Haruto, taken\), advisor \(Mika\)"):
        hub.register("", Screen())


def test_meeting_waits_until_every_human_has_a_console():
    hub = ConsoleHub(PEOPLE)
    hub.register("owner", Screen())

    waiting = threading.Thread(target=hub.wait_for_consoles)
    waiting.start()
    waiting.join(timeout=0.2)
    assert waiting.is_alive()

    hub.register("advisor", Screen())
    waiting.join(timeout=5)
    assert not waiting.is_alive() and hub.started


def test_start_anyway_stops_the_waiting():
    hub = ConsoleHub(PEOPLE)
    waiting = threading.Thread(target=hub.wait_for_consoles)
    waiting.start()

    hub.start_anyway()
    waiting.join(timeout=5)
    assert not waiting.is_alive()


def test_question_goes_to_the_asked_participants_console():
    owner, advisor, log = Screen(), Screen(), Screen()
    hub = ConsoleHub(PEOPLE, log=log)
    hub.register("owner", owner)
    hub.register("advisor", advisor)

    reply = asking(hub, question("advisor"))

    assert advisor.saw("Moderator asks you, Mika: Which approach?")
    assert owner.saw("Waiting for Mika to answer.") and log.saw("Waiting for Mika")

    assert hub.submit(owner, "Me first") is False
    assert owner.saw("The moderator asked Mika; please wait.")

    assert hub.submit(advisor, " Adaptive. ") is True
    assert reply() == {
        "feedback": "Adaptive.", "decline_future_questions": False, "participant_id": "advisor",
    }
    assert owner.saw("Mika: Adaptive.") and not advisor.saw("Mika: Adaptive.")


def test_general_question_is_answered_by_whoever_is_first():
    owner, advisor = Screen(), Screen()
    hub = ConsoleHub(PEOPLE)
    hub.register("owner", owner)
    hub.register("advisor", advisor)

    reply = asking(hub, question())

    assert hub.submit(owner, "Fixed.") is True
    assert hub.submit(advisor, "Adaptive.") is False
    assert reply()["participant_id"] == "owner"
    assert reply()["feedback"] == "Fixed."


def test_anyone_answers_for_a_participant_without_a_console():
    owner = Screen()
    hub = ConsoleHub(PEOPLE)
    hub.register("owner", owner)

    reply = asking(hub, question("advisor"))

    assert owner.saw("Mika (not here; anyone can answer)")
    assert hub.submit(owner, "Adaptive.") is True
    assert reply()["participant_id"] == "owner"


def test_question_is_shown_to_a_console_that_joins_late_and_freed_when_it_leaves():
    owner, advisor = Screen(), Screen()
    hub = ConsoleHub(PEOPLE)
    hub.register("owner", owner)

    reply = asking(hub, question("advisor"))
    hub.register("advisor", advisor)
    assert advisor.saw("Moderator asks you, Mika")

    hub.unregister(advisor)
    assert owner.saw("Mika left.") and owner.saw("Anyone can answer")
    assert hub.submit(owner, "Go.") is True
    assert reply()["feedback"] == "Go."


def test_one_console_can_hold_every_seat_and_sees_each_message_once():
    screen = Screen()
    hub = ConsoleHub(PEOPLE, log=screen)
    hub.register("owner", screen)
    hub.register("advisor", screen)

    hub.say("Round 1")
    assert screen.lines.count("Round 1") == 1

    reply = asking(hub, question("advisor"))
    assert hub.submit(screen, "Adaptive.") is True
    assert reply()["participant_id"] == "advisor"  # answered in the asked seat


def test_decline_and_empty_answers():
    owner = Screen()
    hub = ConsoleHub(PEOPLE)
    hub.register("owner", owner)

    reply = asking(hub, question("owner"))
    hub.submit(owner, "/decline")
    assert reply() == {
        "feedback": "", "decline_future_questions": True, "participant_id": "owner",
    }

    reply = asking(hub, question("owner"))
    hub.submit(owner, "")
    assert reply()["feedback"] == ""


def test_host_answers_when_the_meeting_has_no_human_participants():
    log = Screen()
    hub = ConsoleHub([PEOPLE[0]], log=log)

    assert hub.missing() == []
    reply = asking(hub, question())
    assert hub.submit(log, "Adaptive.") is True
    assert reply() == {"feedback": "Adaptive.", "decline_future_questions": False}


def test_console_without_a_seat_cannot_answer():
    stranger = Screen()
    hub = ConsoleHub(PEOPLE)

    assert hub.submit(stranger, "Hello") is False
    assert stranger.saw("no seat")


def test_hub_carries_a_whole_meeting():
    model = ScriptedModel(replies={"moderator": [
        moderator_reply("request_user_feedback", question="Budget?", ask_participant_id="owner"),
    ]})
    meeting_input = parse_meeting_config(config(participants=[
        {"name": "Risk"},
        {"id": "owner", "name": "Haruto", "human": True},
    ]))
    owner = Screen()
    hub = ConsoleHub(meeting_input["participants"])
    hub.register("owner", owner)

    meeting = Meeting(model)
    result = meeting.start(meeting_input)

    reply = asking(hub, result.question)
    assert owner.saw("Moderator asks you, Haruto: Budget?")
    hub.submit(owner, "Ten thousand.")
    result = meeting.resume(result.thread_id, reply())

    assert result.finished
    assert {"role": "user", "content": "[Haruto] Ten thousand.", "name": "owner"} in (
        result.state["messages"]
    )
