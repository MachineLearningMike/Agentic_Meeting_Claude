"""Meeting configuration files.

A meeting is described in a TOML or JSON file: what it is about, its agenda,
and who takes part.

    topic = "Reduce excessive trading frequency"
    agenda = ["Fixed versus adaptive frequency", "What is still unknown"]

    [[participants]]
    name = "Risk analyst"
    role = "risk analyst"
    statement = "Identifies risks, drawdowns and failure modes."

    [[participants]]
    id = "owner"
    name = "Project owner"
    human = true

load_meeting_config() returns the MeetingInput for Meeting.start(). Mistakes
in the file (unknown keys, missing names, duplicate ids) raise ConfigError
with the place they were found.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .state import MeetingInput


class ConfigError(ValueError):
    """The meeting configuration is not valid."""


TEXT_KEYS = ("meeting_id", "project_id", "topic", "background", "language")
NUMBER_KEYS = ("max_rounds_per_agenda", "max_total_rounds")
# Passed to the engine as they are.
LIST_KEYS = ("problems", "objectives")
MEETING_KEYS = {*TEXT_KEYS, *NUMBER_KEYS, *LIST_KEYS, "agenda", "participants"}

# Config key -> Participant key. "statement" is the role statement.
PARTICIPANT_KEYS = {
    "id": "id",
    "name": "name",
    "role": "role",
    "statement": "description",
    "description": "description",
    "instructions": "instructions",
    "model": "model",
    "language": "language",
    "human": "is_human",
    "is_human": "is_human",
}


def load_meeting_config(path: str | Path) -> MeetingInput:
    """Read a meeting configuration from a .toml or .json file."""

    path = Path(path)

    try:
        text = path.read_text(encoding="utf-8-sig")
    except OSError as error:
        raise ConfigError(f"Cannot read {path}: {error.strerror or error}") from error

    try:
        if path.suffix.lower() == ".json":
            data = json.loads(text)
        else:
            data = _toml().loads(text)
    except ValueError as error:
        raise ConfigError(f"{path}: {error}") from error

    try:
        return parse_meeting_config(data)
    except ConfigError as error:
        raise ConfigError(f"{path}: {error}") from error


def parse_meeting_config(data: Any) -> MeetingInput:
    """Check a configuration and turn it into a MeetingInput."""

    if not isinstance(data, dict):
        raise ConfigError("The configuration must be a table of settings.")

    _no_unknown_keys(data, MEETING_KEYS, "the meeting")

    meeting: dict[str, Any] = {}

    for key in TEXT_KEYS:
        if key in data:
            meeting[key] = _text(data[key], key)

    if not meeting.get("topic"):
        raise ConfigError("'topic' is required.")

    for key in NUMBER_KEYS:
        if key in data:
            value = data[key]
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ConfigError(f"'{key}' must be a whole number of 1 or more.")
            meeting[key] = value

    for key in LIST_KEYS:
        if key in data:
            meeting[key] = _tables(data[key], key)

    if "agenda" in data:
        meeting["agenda"] = _agenda(data["agenda"])

    if "participants" in data:
        meeting["participants"] = _participants(data["participants"])

    return meeting  # type: ignore[return-value]


def _agenda(value: Any) -> list[dict]:
    if not isinstance(value, list) or not value:
        raise ConfigError("'agenda' must be a list with at least one item.")

    items = []
    ids: set[str] = set()

    for number, entry in enumerate(value, start=1):
        where = f"agenda item {number}"

        if isinstance(entry, str):
            entry = {"description": entry}
        elif not isinstance(entry, dict):
            raise ConfigError(f"{where} must be text, or a table with 'description'.")

        _no_unknown_keys(entry, {"id", "description"}, where)
        description = _text(entry.get("description", ""), f"{where}: 'description'")

        if not description:
            raise ConfigError(f"{where} has no description.")

        item = {"description": description}

        if "id" in entry:
            item_id = _text(entry["id"], f"{where}: 'id'")
            if not item_id or item_id in ids:
                raise ConfigError(f"{where}: id {item_id!r} is empty or used twice.")
            ids.add(item_id)
            item["id"] = item_id

        items.append(item)

    return items


def _participants(value: Any) -> list[dict]:
    entries = _tables(value, "participants")
    participants = []

    for number, entry in enumerate(entries, start=1):
        where = f"participant {number}"
        _no_unknown_keys(entry, set(PARTICIPANT_KEYS), where)

        participant: dict[str, Any] = {}

        for key, target in PARTICIPANT_KEYS.items():
            if key not in entry:
                continue
            if target in participant:
                raise ConfigError(f"{where}: '{key}' is given twice under two names.")
            if target == "is_human":
                if not isinstance(entry[key], bool):
                    raise ConfigError(f"{where}: '{key}' must be true or false.")
                participant[target] = entry[key]
            else:
                participant[target] = _text(entry[key], f"{where}: '{key}'")

        if not participant.get("name") and not participant.get("id"):
            raise ConfigError(f"{where} needs a 'name' (or an 'id').")

        if not participant.get("is_human"):
            participant.pop("is_human", None)
        elif participant.get("model"):
            raise ConfigError(f"{where} is human, so it cannot have a 'model'.")

        participants.append({k: v for k, v in participant.items() if v != ""})

    # Given ids are kept; the others are made from the names.
    taken = set()
    for number, participant in enumerate(participants, start=1):
        if "id" in participant:
            if participant["id"] in taken:
                raise ConfigError(
                    f"participant {number}: id {participant['id']!r} is used twice."
                )
            taken.add(participant["id"])

    for participant in participants:
        if "id" not in participant:
            participant["id"] = _slug(participant["name"], taken)
        participant.setdefault("name", participant["id"])

    return participants


def _slug(name: str, taken: set[str]) -> str:
    base = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "participant"
    candidate, suffix = base, 2

    while candidate in taken:
        candidate, suffix = f"{base}-{suffix}", suffix + 1

    taken.add(candidate)
    return candidate


def _text(value: Any, where: str) -> str:
    if not isinstance(value, str):
        raise ConfigError(f"{where} must be text.")
    return value.strip()


def _tables(value: Any, key: str) -> list[dict]:
    if not isinstance(value, list) or not all(isinstance(v, dict) for v in value):
        raise ConfigError(f"'{key}' must be a list of tables.")
    return [dict(entry) for entry in value]


def _no_unknown_keys(table: dict, known: set[str], where: str) -> None:
    unknown = sorted(str(key) for key in table if key not in known)

    if unknown:
        raise ConfigError(
            f"Unknown setting in {where}: {', '.join(unknown)}. "
            f"Known settings: {', '.join(sorted(known))}."
        )


def _toml():
    try:
        import tomllib  # Python 3.11+
        return tomllib
    except ImportError:
        pass

    try:
        import tomli
        return tomli
    except ImportError:
        raise ConfigError(
            "Reading .toml needs Python 3.11+, or `pip install tomli`. "
            "A .json file works on every version."
        ) from None
