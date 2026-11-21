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
python apps/cli/trading_project.py 1           # three meetings that build on each other
python apps/cli/show_prompts.py --out p.md     # every prompt of an example meeting
cd apps/web && chainlit run app.py             # browser UI, shared meetings
```

All apps share `data/meetings.db` (override with `MEETING_DB`).

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
- **Prompt caching:** each step's instructions are marked for caching, since
  they repeat every round.
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

159 tests, no API key needed: a scripted model stands in for Claude, and a
fake Anthropic client checks the exact requests `ClaudeModel` sends.

## Layout

```text
agentic_meeting/
├── engine.py          Meeting: runs the steps, pauses and resumes
├── model.py           ClaudeModel: requests to the Messages API
├── checkpoints.py     Memory and SQLite checkpoints
├── state.py           MeetingState, MeetingInput, MeetingOutput
├── schemas.py         Artifact types and lifecycle rules
├── storage.py         SQLite: meeting records + project state
├── history.py         Loads project memory for a meeting
├── testing.py         Scripted models for tests and demos
└── nodes/             The steps: analysis, deliberation, moderator, ...
apps/cli/              Console meeting, trading example, prompt dump
apps/web/              Chainlit browser UI
tests/
```
