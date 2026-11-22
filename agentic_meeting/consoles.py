"""The consoles of a meeting's human participants.

Every human participant takes part through a console: a terminal, a browser
tab, anything that can show text and send back what the person typed. A
console registers with the meeting's ConsoleHub for a participant's seat. The
hub then shows the meeting to every console, puts the moderator's questions
to the right one and hands the answer back to the meeting:

    hub = ConsoleHub(meeting_input["participants"])
    hub.register("owner", send=print)          # a console is its send function
    hub.wait_for_consoles()

    result = meeting.start(meeting_input)
    while not result.finished:
        result = meeting.resume(result.thread_id, hub.ask(result.question))

    # elsewhere, whenever a console's person types a line:
    hub.submit(send, line)

The hub is thread-safe and knows nothing about how consoles are connected;
apps/cli/meet.py connects terminals to it over TCP.
"""

from __future__ import annotations

import threading
from typing import Any, Callable, Mapping, Sequence

Send = Callable[[str], None]

DECLINE = "/decline"


class SeatError(ValueError):
    """A console asked for a seat it cannot have."""


class ConsoleHub:
    """Registers consoles for human participants and routes questions to them.

    participants: the meeting's participants; only the human ones have seats.
    log:          a send function that is shown everything, e.g. the terminal
                  of whoever hosts the meeting. It may also hold seats.
    """

    def __init__(self, participants: Sequence[Mapping[str, Any]] | None = None, log: Send | None = None):
        self.humans = {
            participant["id"]: participant
            for participant in participants or []
            if participant.get("is_human")
        }
        self.log = log
        self.consoles: dict[str, Send] = {}  # participant id -> console
        self.started = False
        self._question: dict | None = None
        self._reply: dict | None = None
        self._changed = threading.Condition(threading.RLock())

    # --- seats ---------------------------------------------------------------

    def find_seat(self, name: str) -> Mapping[str, Any]:
        """The human participant with this id or name."""

        wanted = (name or "").strip().lower()

        for participant in self.humans.values():
            if wanted and wanted in (
                participant["id"].lower(), participant.get("name", "").lower()
            ):
                return participant

        raise SeatError(
            (f"No human participant {name.strip()!r}. " if wanted else "")
            + self.describe_seats()
        )

    def describe_seats(self) -> str:
        if not self.humans:
            return "This meeting has no human participants."

        seats = ", ".join(
            f"{pid} ({self._name(pid)}"
            + (", taken)" if pid in self.consoles else ")")
            for pid in self.humans
        )
        return f"Seats: {seats}."

    def register(self, name: str, send: Send) -> Mapping[str, Any]:
        """Give a console the seat of a human participant (by id or name)."""

        with self._changed:
            participant = self.find_seat(name)
            pid = participant["id"]

            if pid in self.consoles:
                raise SeatError(
                    f"{self._name(pid)} already has a console. " + self.describe_seats()
                )

            self.consoles[pid] = send
            self.say(f"* {self._name(pid)} joined.", exclude=send)

            if self._question is not None:
                self._announce(only=send)

            self._changed.notify_all()
            return participant

    def unregister(self, send: Send) -> None:
        """Free every seat a console holds, e.g. after it disconnected."""

        with self._changed:
            left = [pid for pid, console in self.consoles.items() if console is send]

            for pid in left:
                del self.consoles[pid]
                self.say(f"* {self._name(pid)} left.")

            if self._question is not None and self._asked() in left:
                self.say("  Anyone can answer the moderator's question now.")

            self._changed.notify_all()

    def missing(self) -> list[Mapping[str, Any]]:
        """Human participants without a console."""

        with self._changed:
            return [p for pid, p in self.humans.items() if pid not in self.consoles]

    def wait_for_consoles(self) -> None:
        """Block until every human has a console, or start_anyway() is called."""

        with self._changed:
            # The timeout keeps Ctrl+C working on Windows.
            while self.missing() and not self.started:
                self._changed.wait(0.5)
            self.started = True

    def start_anyway(self) -> None:
        with self._changed:
            self.started = True
            self._changed.notify_all()

    # --- talking -------------------------------------------------------------

    def say(self, text: str, only: Send | None = None, exclude: Send | None = None) -> None:
        """Show text on every console (once each), on one, or on all but one."""

        with self._changed:
            for send in [only] if only else self._everyone():
                if send is not exclude:
                    _deliver(send, text)

    def ask(self, question: dict) -> dict:
        """Put the moderator's question to the consoles and wait for the answer.

        question is MeetingResult.question; the return value goes to
        Meeting.resume(). A question for a participant without a console can
        be answered by anyone.
        """

        with self._changed:
            self._question, self._reply = question, None
            self._announce()

            while self._reply is None:
                self._changed.wait(0.5)

            reply, self._question, self._reply = self._reply, None, None
            return reply

    def submit(self, send: Send, text: str) -> bool:
        """Take a line typed at a console; True if it answered the question."""

        text = text.strip()

        with self._changed:
            seats = [pid for pid, console in self.consoles.items() if console is send]

            if not seats and self.humans:
                _deliver(send, "  This console has no seat in the meeting.")
                return False

            if self._question is None or self._reply is not None:
                _deliver(send, "  The moderator has not asked anything; please wait.")
                return False

            asked = self._asked()

            if asked in self.consoles and asked not in seats:
                _deliver(send, f"  The moderator asked {self._name(asked)}; please wait.")
                return False

            speaker = asked if asked in seats else (seats[0] if seats else None)
            decline = text.lower() == DECLINE
            name = self._name(speaker) if speaker else "The host"

            if decline:
                self.say(f"* {name} let the meeting continue without more questions.")
            else:
                self.say(f"{name}: {text or '(no answer)'}", exclude=send)

            self._reply = {
                "feedback": "" if decline else text,
                "decline_future_questions": decline,
            }
            if speaker:
                self._reply["participant_id"] = speaker

            self._changed.notify_all()
            return True

    # --- internals -----------------------------------------------------------

    def _name(self, participant_id: str) -> str:
        return self.humans[participant_id].get("name", participant_id)

    def _everyone(self) -> list[Send]:
        """Every console and the log, each once (a console may hold several seats)."""

        sends: list[Send] = []
        for send in [self.log, *self.consoles.values()]:
            if send is not None and not any(send is seen for seen in sends):
                sends.append(send)
        return sends

    def _asked(self) -> str | None:
        """The id of the human the open question is addressed to, if any."""

        participant = (self._question or {}).get("participant") or {}
        return participant.get("id") if participant.get("id") in self.humans else None

    def _announce(self, only: Send | None = None) -> None:
        text = (self._question or {}).get("question", "")
        asked = self._asked()
        target = self.consoles.get(asked) if asked else None
        how = f"(Type your answer; empty to skip, {DECLINE} to stop being asked.)"

        for send in [only] if only else self._everyone():
            if target is None:
                whom = f" {self._name(asked)} (not here; anyone can answer)" if asked else ""
                _deliver(send, f"\nModerator asks{whom}: {text}\n{how}")
            elif send is target:
                _deliver(send, f"\nModerator asks you, {self._name(asked)}: {text}\n{how}")
            else:
                _deliver(
                    send,
                    f"\nModerator asks {self._name(asked)}: {text}\n"
                    f"(Waiting for {self._name(asked)} to answer.)",
                )


def _deliver(send: Send, text: str) -> None:
    try:
        send(text)
    except Exception:
        # A console that cannot be reached is removed by whoever connects it.
        pass
