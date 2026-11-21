"""Load project memory at the start of a meeting and save the record at the end.

Both steps use the meeting's store and do nothing when there is none.
"""

from ..history import load_project_context


def load_project_memory(state: dict, ctx) -> dict:
    project_id = state.get("project_id")

    # A caller may supply project_context directly; keep it.
    if ctx.store is None or not project_id or state.get("project_context"):
        return {}

    return {"project_context": load_project_context(ctx.store, project_id)}


def save_meeting(state: dict, ctx) -> dict:
    record = state.get("meeting_record")

    if ctx.store is None or not record:
        return {}

    ctx.store.save(record)
    return {"meeting_saved": True}
