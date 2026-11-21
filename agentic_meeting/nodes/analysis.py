from ..history import PROJECT_CONTEXT_RULES, prompt_context
from .common import (
    agenda_overview,
    ask,
    language_rule,
    participants_overview,
    recent_messages,
)


SYSTEM_PROMPT = """
You are an analytical assistant participating in a structured meeting.

Analyze the provided meeting context.

Identify:
1. Established facts.
2. Assumptions that have not been verified.
3. Constraints.
4. Important missing information.
5. The central question that the meeting should resolve.

Do not invent facts.
Do not make decisions on behalf of participants.
Be precise and concise.
"""


def analyze_context(state: dict, ctx) -> dict:
    context = {
        "topic": state.get("topic", ""),
        "background": state.get("background", ""),
        "prior_state": state.get("prior_state", ""),
        "objectives": state.get("objectives", []),
        "problems": state.get("problems", []),
        "current_agenda": state.get("current_agenda", ""),
        "agenda": agenda_overview(state),
        "participants": participants_overview(state),
        "user_feedback": state.get("user_feedback", ""),
        "recent_messages": recent_messages(state),
    }

    system_prompt = SYSTEM_PROMPT

    project_context = prompt_context(state.get("project_context"))

    if project_context:
        context["project_context"] = project_context
        system_prompt += PROJECT_CONTEXT_RULES

    system_prompt += language_rule(state)

    return {
        "context_analysis": ask(ctx, "analysis", system_prompt, context),
        "current_phase": "analysis",
    }
