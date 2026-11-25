import json
from dataclasses import dataclass, field
from typing import Any

from ..model import Model, ModelRequest


@dataclass
class Context:
    """What nodes need besides the state: models and the optional store."""

    model: Model | None
    participant_models: dict[str, Model] = field(default_factory=dict)
    store: Any = None

    def model_for(self, participant: dict | None = None) -> Model:
        """The model for a call; a participant with "model" uses its own."""

        model_key = (participant or {}).get("model")

        if model_key:
            model = self.participant_models.get(model_key)

            if model is None:
                raise ValueError(
                    f"Participant {participant.get('id')!r} uses model "
                    f"{model_key!r}, which is not in participant_models."
                )

            return model

        if self.model is None:
            raise ValueError("No meeting model supplied.")

        return self.model


def ask(
    ctx: Context,
    node: str,
    system: str,
    data: dict,
    participant: dict | None = None,
    schema: dict | None = None,
) -> str:
    """Send one node's instructions and data to its model; return the reply."""

    shared, user = split_for_caching(data)

    return ctx.model_for(participant).complete(ModelRequest(
        node=node,
        system=system,
        shared=shared,
        user=user,
        participant=participant["id"] if participant else None,
        schema=schema,
    ))


# Data that stays the same for the whole meeting. It is sent first, so that
# Claude can cache it together with the node's instructions.
SHARED_KEYS = ("topic", "participants", "project_context")


def split_for_caching(data: dict) -> tuple[str, str]:
    """Split a node's data into the part shared by every round and the rest.

    The two texts joined are one JSON object, with the shared keys first.
    """

    def dump(value: dict) -> str:
        return json.dumps(value, ensure_ascii=False, default=str, indent=2)

    shared = {key: data[key] for key in SHARED_KEYS if key in data}
    rest = {key: value for key, value in data.items() if key not in shared}

    if not shared or not rest:
        return "", dump(data)

    # '{\n  ...shared' + ',\n' and '  ...rest\n}'
    return dump(shared)[:-2] + ",\n", dump(rest)[2:]


def recent_messages(state: dict, limit: int | None = 12) -> list[str]:
    messages = state.get("messages", [])
    if limit is not None:
        messages = messages[-limit:]
    return [entry.get("content", "") for entry in messages]


def participants_overview(state: dict) -> list[dict]:
    """The participants in the compact form used in prompts."""
    return [
        {
            "id": participant["id"],
            "name": participant.get("name", participant["id"]),
            "role": participant.get("role", ""),
            "is_human": participant.get("is_human", False),
            # The role statement, so that others know what to ask whom.
            **(
                {"description": participant["description"]}
                if participant.get("description")
                else {}
            ),
            **(
                {"language": participant["language"]}
                if participant.get("language")
                else {}
            ),
        }
        for participant in state.get("participants", [])
    ]


def language_rule(state: dict, structured: bool = False) -> str:
    """Prompt text telling a node which language to write in.

    With state["language"] the meeting uses that language; otherwise it uses
    the language the meeting is held in. For JSON output, only free text is
    written in that language.
    """

    language = (state.get("language") or "").strip()

    if language:
        rule = f"Write all your output in {language}."
    else:
        rule = (
            "Write in the language the meeting is held in: the language of "
            "the topic and of the human participants' messages. If that is "
            "unclear, use English."
        )

    if structured:
        rule += (
            " This applies to free-text values only; keep JSON keys, ids, "
            "actions and status values exactly as specified."
        )

    return f"\nLANGUAGE: {rule}\n"


def agenda_overview(state: dict) -> list[dict]:
    """The agenda in the compact form used in prompts."""
    return [
        {
            "id": item["id"],
            "description": item.get("description", ""),
            "status": item.get("status", ""),
            "summary": item.get("summary", ""),
        }
        for item in state.get("agenda", [])
    ]


def parse_json_object(text: str) -> dict:
    """Parse a JSON object, including one wrapped in a Markdown code block.

    Structured outputs make Claude's replies valid JSON already; this also
    accepts replies from models used without a schema.
    """

    text = text.strip()

    if text.startswith("```"):
        lines = text.splitlines()

        if lines[-1].strip() == "```":
            lines = lines[1:-1]
        else:
            lines = lines[1:]

        text = "\n".join(lines).strip()

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end < start:
        raise ValueError("No JSON object found in model response.")

    result = json.loads(text[start:end + 1])

    if not isinstance(result, dict):
        raise ValueError("Expected a JSON object.")

    return result
