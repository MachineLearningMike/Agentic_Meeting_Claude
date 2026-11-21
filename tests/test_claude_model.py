"""ClaudeModel's requests to the Messages API, checked with a fake client."""

import json
from types import SimpleNamespace

import pytest

from agentic_meeting import ClaudeModel, Meeting, ModelRefusal, ModelRequest
from agentic_meeting.model import DEFAULT_MODEL, FALLBACK_BETA
from agentic_meeting.nodes.extract_artifacts import EXTRACTION_SCHEMA, validate_artifacts
from agentic_meeting.nodes.moderator import DECISION_SCHEMA, VALID_ACTIONS
from agentic_meeting.testing import demo_model


class FakeClient:
    """Stands in for anthropic.Anthropic: records requests, returns replies."""

    def __init__(self, replies=None):
        self.requests = []
        self.replies = list(replies or [])
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **params):
        self.requests.append(params)
        if self.replies:
            return self.replies.pop(0)
        return reply("ok")


def reply(*texts, stop_reason="end_turn", category=None):
    return SimpleNamespace(
        stop_reason=stop_reason,
        stop_details=SimpleNamespace(category=category) if category else None,
        content=[SimpleNamespace(type="thinking", thinking="")]
        + [SimpleNamespace(type="text", text=t) for t in texts],
    )


REQUEST = ModelRequest(node="analysis", system="Analyze.", user='{"topic": "T"}')


def test_default_request():
    client = FakeClient()

    ClaudeModel(client=client).complete(REQUEST)
    params = client.requests[0]

    assert params["model"] == DEFAULT_MODEL == "claude-opus-5-5"
    assert params["max_tokens"] == 16000
    assert params["system"] == [{
        "type": "text", "text": "Analyze.", "cache_control": {"type": "ephemeral"},
    }]
    assert params["messages"] == [{"role": "user", "content": '{"topic": "T"}'}]
    assert params["output_config"] == {"effort": "medium"}
    assert params["betas"] == [FALLBACK_BETA]
    assert params["fallbacks"] == "default"
    assert "thinking" not in params  # adaptive thinking is the default


def test_effort_model_and_fallbacks_are_configurable():
    client = FakeClient()
    model = ClaudeModel(
        "claude-sonnet-5-5",
        effort="high",
        node_effort={"analysis": "low"},
        fallbacks=False,
        client=client,
    )

    model.complete(REQUEST)
    model.complete(ModelRequest(node="moderator", system="s", user="{}"))

    assert client.requests[0]["model"] == "claude-sonnet-5-5"
    assert client.requests[0]["output_config"]["effort"] == "low"
    assert client.requests[1]["output_config"]["effort"] == "high"
    assert "betas" not in client.requests[0] and "fallbacks" not in client.requests[0]


def test_schema_becomes_a_structured_output_format():
    client = FakeClient()

    ClaudeModel(client=client).complete(
        ModelRequest(node="moderator", system="s", user="{}", schema=DECISION_SCHEMA)
    )

    assert client.requests[0]["output_config"]["format"] == {
        "type": "json_schema", "schema": DECISION_SCHEMA,
    }


def test_reply_text_joins_text_blocks_and_skips_thinking():
    client = FakeClient([reply("Hello ", "world")])

    assert ClaudeModel(client=client).complete(REQUEST) == "Hello world"


def test_refusal_raises():
    client = FakeClient([reply(stop_reason="refusal", category="cyber")])

    with pytest.raises(ModelRefusal, match=r"declined the analysis request \(cyber\)"):
        ClaudeModel(client=client).complete(REQUEST)


def test_cut_off_json_is_an_error_but_cut_off_text_is_returned():
    client = FakeClient([
        reply('{"action": "conti', stop_reason="max_tokens"),
        reply("A long analysis", stop_reason="max_tokens"),
    ])
    model = ClaudeModel(client=client)

    with pytest.raises(ValueError, match="cut off at max_tokens"):
        model.complete(ModelRequest(node="moderator", system="s", user="{}", schema={}))
    assert model.complete(REQUEST) == "A long analysis"


def test_a_whole_meeting_runs_through_claude_model():
    decision = json.dumps({
        "action": "conclude_meeting", "reason": "Done.", "question_for_user": "",
        "agenda_summary": "Settled.", "target_agenda_id": "", "ask_participant_id": "",
    })
    artifacts = json.dumps({
        "proposals": [{"description": "Adaptive frequency", "status": "proposed"}],
        "arguments": [], "decisions": [], "action_items": [],
        "unresolved_issues": [], "updates": [],
    })
    client = FakeClient([
        reply("Analysis."), reply("Deliberation."), reply(decision),
        reply("Report."), reply(artifacts),
    ])

    result = Meeting(ClaudeModel(client=client)).start({"topic": "Trading"})

    assert result.output["meeting_summary"] == "Report."
    assert result.output["meeting_record"]["proposals"][0]["description"] == "Adaptive frequency"
    formats = [r["output_config"].get("format") is not None for r in client.requests]
    assert formats == [False, False, True, False, True]


def test_a_refusal_during_extraction_keeps_the_meeting():
    decision = json.dumps({
        "action": "conclude_meeting", "reason": "", "question_for_user": "",
        "agenda_summary": "", "target_agenda_id": "", "ask_participant_id": "",
    })
    client = FakeClient([
        reply("A."), reply("D."), reply(decision), reply("Report."),
        reply(stop_reason="refusal"), reply(stop_reason="refusal"),
    ])

    result = Meeting(ClaudeModel(client=client)).start({"topic": "T"})

    assert result.output["meeting_summary"] == "Report."
    assert "declined the extraction request" in result.output["extraction_error"]


# --- the schemas themselves -------------------------------------------------


def walk_objects(schema):
    if isinstance(schema, dict):
        if schema.get("type") == "object":
            yield schema
        for value in schema.values():
            yield from walk_objects(value)
    elif isinstance(schema, list):
        for value in schema:
            yield from walk_objects(value)


@pytest.mark.parametrize("schema", [DECISION_SCHEMA, EXTRACTION_SCHEMA])
def test_schemas_meet_structured_output_rules(schema):
    objects = list(walk_objects(schema))

    assert objects
    for obj in objects:
        assert obj["additionalProperties"] is False
        assert set(obj["required"]) <= set(obj["properties"])


def test_decision_schema_allows_exactly_the_engine_actions():
    assert set(DECISION_SCHEMA["properties"]["action"]["enum"]) == VALID_ACTIONS


def test_extraction_schema_statuses_match_validation():
    # Every status the schema allows is accepted by validate_artifacts.
    props = EXTRACTION_SCHEMA["properties"]
    for field, status_key in [
        ("proposals", "status"), ("decisions", "status"),
        ("action_items", "status"), ("unresolved_issues", "status"),
        ("arguments", "position"),
    ]:
        item = props[field]["items"]["properties"]
        text_key = "content" if field == "arguments" else "description"
        for status in item[status_key]["enum"]:
            validate_artifacts({field: [{text_key: "x", status_key: status}]})


def test_demo_model_extraction_fits_the_schema_shape():
    data = json.loads(demo_model().defaults["extraction"])

    assert set(data) <= set(EXTRACTION_SCHEMA["properties"])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
