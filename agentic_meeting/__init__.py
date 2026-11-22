"""Structured human-AI meetings on Claude, with persistent project memory."""

from .checkpoints import CheckpointStore, MemoryCheckpoints, SqliteCheckpoints
from .config import ConfigError, load_meeting_config, parse_meeting_config
from .consoles import ConsoleHub, SeatError
from .engine import Meeting, MeetingResult
from .history import load_project_context
from .model import ClaudeModel, Model, ModelRefusal, ModelRequest
from .state import MeetingInput, MeetingOutput, MeetingState
from .storage import MeetingStorage, MeetingStore

__all__ = [
    "CheckpointStore",
    "ClaudeModel",
    "ConfigError",
    "ConsoleHub",
    "Meeting",
    "MeetingInput",
    "MeetingOutput",
    "MeetingResult",
    "MeetingState",
    "MeetingStorage",
    "MeetingStore",
    "MemoryCheckpoints",
    "Model",
    "ModelRefusal",
    "ModelRequest",
    "SeatError",
    "SqliteCheckpoints",
    "load_meeting_config",
    "load_project_context",
    "parse_meeting_config",
]
