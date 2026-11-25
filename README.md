# Agentic Meeting (Claude)

Structured human-AI meetings on Claude, with persistent project memory. AI
participants deliberate in their roles, a moderator steers the agenda and asks
people for input, and every meeting's proposals, arguments, decisions, action
items and open issues are kept with their status, so the next meeting
continues where the last one stopped.

This is the Claude-native version of the LangGraph meeting engine in
`Project_Engine`: no LangGraph or LangChain, only the official
[Anthropic SDK](https://pypi.org/project/anthropic/).

## Setup

```bash
pip install -e .            # the package and the anthropic SDK
pip install chainlit pytest # web UI and tests (optional)
```

Claude needs credentials: set `ANTHROPIC_API_KEY`, or run `ant auth login`.
Everything also runs offline with `--model fake`, a scripted demo model.

## Run

```bash
python apps/cli/main.py                        # console meeting, Claude Opus 5.5
python apps/cli/main.py --model fake           # offline demo
python apps/cli/meet.py host examples/meeting.toml   # your own agenda and participants
python apps/cli/trading_project.py 1           # three meetings that build on each other
python apps/cli/show_prompts.py --out p.md     # every prompt of an example meeting
cd apps/web && chainlit run app.py             # browser UI, shared meetings
```

All apps share `data/meetings.db` (override with `MEETING_DB`).

## Configure a meeting

A meeting's agenda and participants are set in a TOML (or JSON) file. Copy
[examples/meeting.toml](examples/meeting.toml) and change it:

```toml
topic = "Reduce excessive trading frequency"
background = "Fees and slippage reduce profitability."

agenda = [
    "Fixed versus adaptive trading frequency",
    "What is still unknown",
]

[[participants]]                 # agentic: speaks in every round
name = "Risk analyst"
role = "risk analyst"
statement = "Identifies risks, drawdowns and failure modes."

[[participants]]                 # human: answers from their console
id = "owner"
name = "Project owner"
statement = "Owns the budget and makes the final decisions."
human = true
```

| Participant key | Meaning |
| --- | --- |
| `name` | Shown in the meeting; required |
| `id` | Used to join and in records; made from the name if missing |
| `role` | Short role title |
| `statement` | The role statement. An agentic participant argues from it; for a human it tells the moderator what to ask them |
| `human` | `true` for a person; otherwise the participant is agentic |
| `instructions` | Extra rules for an agentic participant |
| `model` | A different Claude model id for an agentic participant |
| `language` | Questions to this human are written in it |

Meeting keys: `topic` (required), `background`, `agenda`, `participants`,
`language`, `project_id`, `max_rounds_per_agenda`, `max_total_rounds`,
`problems`, `objectives`, `meeting_id`. A misspelled key is reported, not
ignored.

Run it with `apps/cli/meet.py`. Human participants take part through
consoles that register with the meeting:

```bash
# One console: you answer for every human participant.
python apps/cli/meet.py host examples/meeting.toml

# One console per human. The host takes a seat; the meeting starts once the
# others have registered theirs (or when the host presses Enter).
python apps/cli/meet.py host examples/meeting.toml --as owner
python apps/cli/meet.py join advisor          # in the advisor's terminal

# The host takes no seat and only shows the meeting.
python apps/cli/meet.py host examples/meeting.toml --serve
```

Every console shows the meeting as it runs. A question the moderator
addresses to one person can only be answered from that person's console;
a general question, or one for somebody who is not connected, by anyone.
An empty answer skips the question and `/decline` stops further questions.

Consoles connect to `127.0.0.1:8765`. For other machines, host with
`--listen 0.0.0.0:8765` and join with `--address HOST:8765`. There is no
password, so only do this on a network you trust. `--model fake` works here
too.

In code, the same pieces are `load_meeting_config()` and `ConsoleHub`:

```python
from agentic_meeting import ClaudeModel, ConsoleHub, Meeting, load_meeting_config

meeting_input = load_meeting_config("examples/meeting.toml")
hub = ConsoleHub(meeting_input["participants"])
hub.register("owner", send=print)       # a console is a function that shows text
hub.wait_for_consoles()

meeting = Meeting(ClaudeModel())
result = meeting.start(meeting_input)
while not result.finished:
    result = meeting.resume(result.thread_id, hub.ask(result.question))

# whenever a console's person types a line: hub.submit(send, line)
```

## Use the engine

```python
from agentic_meeting import ClaudeModel, Meeting, MeetingStore, SqliteCheckpoints

meeting = Meeting(
    ClaudeModel(),                                 # claude-opus-5-5
    store=MeetingStore("data/meetings.db"),        # optional: project memory
    checkpoints=SqliteCheckpoints("data/checkpoints.db"),  # survive restarts
)

result = meeting.start({"project_id": "trading-ai", "topic": "Trade less?"})
while not result.finished:
    print(result.question["question"])
    result = meeting.resume(result.thread_id, {"feedback": input("> ")})

print(result.output["meeting_summary"])
```

- `start()` and `resume()` run until the meeting finishes or asks a human.
- `result.output` is the `MeetingOutput` (record, summary, agenda, warnings);
  `result.state` holds everything, including the transcript.
- `on_event=` receives a `step_start` / `step_end` / `step_error` event around
  every step, which the web UI uses to show progress.
- A parent workflow simply calls `start()` and uses the returned record.

### Input (`MeetingInput`)

| Key | Meaning |
| --- | --- |
| `topic`, `background` | What the meeting is about |
| `language` | Language of all output, e.g. `日本語`; if missing, the language the meeting is held in |
| `project_id` | Meetings with the same id share project memory |
| `agenda` | Items discussed one at a time; if missing, built from `problems` and `objectives`, else the topic |
| `participants` | AI participants speak in every round from their role; human ones answer questions, in their own `language` if set |
| `max_rounds_per_agenda` | Round limit per agenda item (default 3) |
| `max_total_rounds` | Limit for the whole meeting (default: per-item limit × (items + 1)) |
| `project_context` | Supply project memory yourself instead of loading it from the store |

## How it uses Claude

`ClaudeModel` (in `agentic_meeting/model.py`) sends every request through the
Messages API:

- **Model:** `claude-opus-5-5` by default; any Claude model id can be passed.
- **Thinking:** adaptive (the model's default); depth set with `effort`
  (`medium` by default, overridable per step with `node_effort`).
- **Structured outputs:** the moderator's decision and the extracted records
  are constrained to JSON schemas, so they always parse. Their meaning (known
  ids, lifecycle rules, duplicates) is still validated in code.
- **Refusal fallbacks:** server-side fallbacks are on (`fallbacks="default"`);
  a request Claude declines is retried on a fallback model in the same call.
  If it is still declined, `ModelRefusal` is raised (during extraction the
  meeting is kept and the failure reported instead).
- **Prompt caching:** each step's instructions, and the data that stays the
  same for the whole meeting (topic, participants, project memory), are sent
  first and marked for caching, since they repeat every round. Claude keeps
  them for five minutes after each use; with `ClaudeModel(cache_ttl="1h")`
  for an hour, which suits meetings where people take longer to answer (cache
  writes then cost more). `cache_ttl=None` turns caching off.
- **Per-participant models:** a participant with `"model": "key"` uses
  `participant_models={"key": ClaudeModel("claude-sonnet-5-5")}`.

## Persistence

- **Project memory** (`MeetingStore`): each meeting's record as history, plus
  each project's current state. A new meeting sees both, clearly labelled.
- **Checkpoints** (`MemoryCheckpoints`, `SqliteCheckpoints`): the running
  meeting, saved after every step, so a meeting paused for an answer resumes
  later, even in another process.

## Tests

```bash
pytest
```

191 tests, no API key needed: a scripted model stands in for Claude, and a
fake Anthropic client checks the exact requests `ClaudeModel` sends.

## Layout

```text
agentic_meeting/
├── engine.py          Meeting: runs the steps, pauses and resumes
├── model.py           ClaudeModel: requests to the Messages API
├── checkpoints.py     Memory and SQLite checkpoints
├── config.py          Meeting configuration files
├── consoles.py        ConsoleHub: the human participants' consoles
├── state.py           MeetingState, MeetingInput, MeetingOutput
├── schemas.py         Artifact types and lifecycle rules
├── storage.py         SQLite: meeting records + project state
├── history.py         Loads project memory for a meeting
├── testing.py         Scripted models for tests and demos
└── nodes/             The steps: analysis, deliberation, moderator, ...
apps/cli/              Console meeting, configured meeting (meet.py), trading example, prompt dump
examples/              A meeting configuration to copy
apps/web/              Chainlit browser UI
tests/
```
