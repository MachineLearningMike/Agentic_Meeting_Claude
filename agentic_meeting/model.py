"""Calling Claude.

Every node asks the model through one small interface, ``Model.complete``.
``ClaudeModel`` implements it with the Anthropic SDK; tests and offline demos
use ``agentic_meeting.testing.ScriptedModel`` instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import anthropic


DEFAULT_MODEL = "claude-opus-5-5"

# Server-side refusal fallbacks: if Claude declines a request, the API re-runs
# it on a fallback model chosen by refusal category, inside the same call.
FALLBACK_BETA = "server-side-fallback-2026-07-01"


@dataclass
class ModelRequest:
    """One model call made by a meeting node."""

    node: str                       # analysis, deliberation, moderator, finalize, extraction
    system: str                     # instructions for this node
    user: str                       # the meeting data, as JSON
    participant: str | None = None  # participant id for a participant's turn
    schema: dict | None = None      # JSON schema the reply must follow


class Model(Protocol):
    def complete(self, request: ModelRequest) -> str:
        """Return the model's reply text (JSON text when a schema is given)."""


class ModelRefusal(RuntimeError):
    """Claude declined the request, and no fallback model answered it."""


class ClaudeModel:
    """Claude through the official Anthropic SDK.

    model:       Claude model id, by default claude-opus-5-5.
    effort:      thinking depth for every node: low, medium, high, xhigh, max.
    node_effort: per-node overrides, e.g. {"analysis": "low"}.
    fallbacks:   use server-side refusal fallbacks (recommended).
    client:      an anthropic.Anthropic client; by default one is created from
                 the environment (ANTHROPIC_API_KEY or `ant auth login`).
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        effort: str = "medium",
        node_effort: dict[str, str] | None = None,
        max_tokens: int = 16000,
        fallbacks: bool = True,
        client: Any = None,
    ):
        self.model = model
        self.effort = effort
        self.node_effort = node_effort or {}
        self.max_tokens = max_tokens
        self.fallbacks = fallbacks
        self.client = client if client is not None else anthropic.Anthropic()

    def request_params(self, request: ModelRequest) -> dict:
        """The Messages API parameters for one request."""

        output_config: dict = {
            "effort": self.node_effort.get(request.node, self.effort),
        }

        if request.schema is not None:
            output_config["format"] = {
                "type": "json_schema",
                "schema": request.schema,
            }

        params: dict = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            # Each node's instructions repeat every round, so they are cached.
            "system": [{
                "type": "text",
                "text": request.system,
                "cache_control": {"type": "ephemeral"},
            }],
            "messages": [{"role": "user", "content": request.user}],
            "output_config": output_config,
        }

        if self.fallbacks:
            params["betas"] = [FALLBACK_BETA]
            params["fallbacks"] = "default"

        return params

    def complete(self, request: ModelRequest) -> str:
        response = self.client.beta.messages.create(**self.request_params(request))

        if response.stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            category = getattr(details, "category", None) if details else None
            raise ModelRefusal(
                f"Claude declined the {request.node} request"
                + (f" ({category})." if category else ".")
            )

        text = "".join(
            block.text for block in response.content if block.type == "text"
        )

        if request.schema is not None and response.stop_reason == "max_tokens":
            raise ValueError(
                f"The {request.node} reply was cut off at max_tokens; "
                "its JSON is incomplete."
            )

        return text
