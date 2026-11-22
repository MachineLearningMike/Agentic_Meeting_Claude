"""A configured meeting, with a console for every human participant.

The meeting (topic, agenda, participants and their role statements) is read
from a TOML or JSON file; see examples/meeting.toml.

    python apps/cli/meet.py host examples/meeting.toml
        One console: you answer for every human participant.

    python apps/cli/meet.py host examples/meeting.toml --as owner
        You are "owner". The other human participants register their own
        consoles, and the meeting starts when everyone is here:

    python apps/cli/meet.py join advisor
        Run by each other human, in their own terminal.

    python apps/cli/meet.py host examples/meeting.toml --serve
        This console only shows the meeting; every human joins.

Consoles connect to 127.0.0.1:8765 by default. To let other machines join,
host with --listen 0.0.0.0:8765 and join with --address HOST:8765. There is
no password: anyone who can reach the port can take a free seat.

Add --model fake to try it offline with a scripted model.
"""

import argparse
import json
import os
import socket
import socketserver
import sys
import threading
from pathlib import Path

DEFAULT_ADDRESS = "127.0.0.1:8765"

STEP_NAMES = {
    "initialize": "Agenda",
    "load_project_memory": "Project memory",
    "advance_agenda": "Next agenda item",
    "analyze_context": "Context analysis",
    "deliberate": "Deliberation",
    "moderate": "Moderator",
    "finalize": "Meeting report",
    "extract_artifacts": "Meeting records",
}

# Steps that call the model and so take a while.
SLOW_STEPS = {"analyze_context", "deliberate", "moderate", "finalize", "extract_artifacts"}


def parse_address(text: str) -> tuple[str, int]:
    host, _, port = text.rpartition(":")
    try:
        return host or "127.0.0.1", int(port)
    except ValueError:
        raise SystemExit(f"Not an address (HOST:PORT): {text}")


# --- hosting -----------------------------------------------------------------


def format_step(step: str, update: dict) -> str:
    if step == "initialize":
        return "\n".join(
            f"{index}. {item['description']}"
            for index, item in enumerate(update.get("agenda", []), start=1)
        )

    if step == "load_project_memory":
        context = update.get("project_context")
        if not context:
            return ""
        return f"{len(context.get('history', []))} earlier meeting(s) of this project are used."

    if step == "advance_agenda":
        if update.get("agenda_complete"):
            return "All agenda items handled."
        return f"Now discussing: {update.get('current_agenda', '')}"

    if step == "analyze_context":
        return update.get("context_analysis", "")

    if step == "deliberate":
        return f"Round {update.get('round_count', '')}\n\n{update.get('latest_deliberation', '')}"

    if step == "moderate":
        # A question is shown by the hub, to whoever it is for.
        return f"{update.get('moderator_action', '')}: {update.get('moderator_reasoning', '')}"

    if step == "extract_artifacts":
        if update.get("extraction_error"):
            return f"Extraction failed: {update['extraction_error']}"
        lines = []
        for field in ("proposals", "arguments", "decisions", "action_items", "unresolved_issues"):
            for item in update.get(field, []):
                text = item.get("description") or item.get("content", "")
                label = item.get("status") or item.get("position", "")
                lines.append(f"- {field.replace('_', ' ')}: {text} ({label})")
        lines += [f"Warning: {warning}" for warning in update.get("extraction_warnings", [])]
        return "\n".join(lines)

    return ""


def serve_consoles(hub, address: tuple[str, int]) -> socketserver.ThreadingTCPServer:
    """Accept consoles over TCP: one JSON object per line, in both directions.

    A console first sends {"join": seat}, then {"say": text} for every line
    its person types. The server sends {"text": ...} to show.
    """

    from agentic_meeting.consoles import SeatError

    class Console(socketserver.StreamRequestHandler):
        def handle(self) -> None:
            writing = threading.Lock()

            def send(text: str) -> None:
                with writing:
                    self.wfile.write((json.dumps({"text": text}) + "\n").encode())
                    self.wfile.flush()

            try:
                seat = json.loads(self.rfile.readline() or b"{}").get("join", "")
                participant = hub.register(str(seat), send)
            except SeatError as error:
                send(str(error))
                return
            except (ValueError, AttributeError, OSError):
                return

            try:
                send(f"You joined as {participant.get('name', participant['id'])}.")

                if not hub.started:
                    send("Waiting for the meeting to start.")

                for line in self.rfile:
                    hub.submit(send, str(json.loads(line).get("say", "")))
            except (ValueError, AttributeError, OSError):
                pass
            finally:
                hub.unregister(send)

    class Server(socketserver.ThreadingTCPServer):
        daemon_threads = True

    try:
        server = Server(address, Console)
    except OSError as error:
        raise SystemExit(f"Cannot listen on {address[0]}:{address[1]}: {error}")

    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def host(args) -> None:
    from agentic_meeting import ClaudeModel, Meeting, MeetingStore
    from agentic_meeting.config import ConfigError, load_meeting_config
    from agentic_meeting.consoles import ConsoleHub, SeatError
    from agentic_meeting.testing import demo_model

    try:
        meeting_input = load_meeting_config(args.config)
    except ConfigError as error:
        raise SystemExit(f"Configuration error: {error}")

    participants = meeting_input.get("participants", [])

    def show(text: str) -> None:
        print(text, flush=True)

    hub = ConsoleHub(participants, log=show)

    # Which seats this console holds: all (default), one (--as) or none (--serve).
    try:
        if args.seat:
            hub.register(args.seat, show)
        elif not args.serve:
            for human in list(hub.humans):
                hub.register(human, show)
    except SeatError as error:
        raise SystemExit(str(error))

    def make_model(name: str):
        return demo_model() if args.model == "fake" else ClaudeModel(name, effort=args.effort)

    try:
        model = make_model(args.model)
        # A participant's "model" in the configuration is a Claude model id.
        participant_models = {
            name: make_model(name)
            for name in {p["model"] for p in participants if p.get("model")}
        }
    except Exception as error:
        raise SystemExit(
            f"Could not set up Claude: {error}\n"
            "Set ANTHROPIC_API_KEY, or use --model fake for an offline demo."
        )

    data = Path(__file__).resolve().parents[2] / "data"
    data.mkdir(exist_ok=True)

    def on_event(event: dict) -> None:
        step = event["step"]
        if step not in STEP_NAMES:
            return
        if event["type"] == "step_start" and step in SLOW_STEPS:
            hub.say(f"\n... {STEP_NAMES[step]} ...")
        elif event["type"] == "step_end":
            text = format_step(step, event["update"])
            if text:
                hub.say(f"\n--- {STEP_NAMES[step]} ---\n{text}")

    meeting = Meeting(
        model,
        store=MeetingStore(os.environ.get("MEETING_DB", str(data / "meetings.db"))),
        participant_models=participant_models,
        on_event=on_event,
    )

    server = None

    if hub.missing():
        server = serve_consoles(hub, parse_address(args.listen))
        where = "" if args.listen == DEFAULT_ADDRESS else (
            f" --address HOST:{server.server_address[1]}"
        )
        show(f"Meeting: {meeting_input['topic']}")
        show("Waiting for these participants to register their consoles:")
        for human in hub.missing():
            show(f"  {human.get('name', human['id'])}:  "
                 f"python apps/cli/meet.py join {human['id']}{where}")
        show("Press Enter to start without waiting.\n")

    def read_keyboard() -> None:
        for line in sys.stdin:
            if hub.started:
                hub.submit(show, line)
            else:
                hub.start_anyway()

    threading.Thread(target=read_keyboard, daemon=True).start()

    try:
        hub.wait_for_consoles()
        hub.say(f"\n=== Meeting started: {meeting_input['topic']} ===")

        result = meeting.start(meeting_input)

        while result.question is not None:
            result = meeting.resume(result.thread_id, hub.ask(result.question))

        output = result.output
        hub.say(f"\n=== Meeting report ===\n\n{output.get('meeting_summary', '')}")
        hub.say(
            f"\nSaved as meeting {output.get('meeting_id', '')}."
            if output.get("meeting_saved")
            else "\nNo meeting record was saved."
        )
    except KeyboardInterrupt:
        hub.say("\nThe host stopped the meeting.")
    except Exception as error:
        hub.say(f"\nThe meeting stopped with an error: {error}")
        raise SystemExit(1)
    finally:
        if server is not None:
            server.shutdown()
            server.server_close()


# --- joining -----------------------------------------------------------------


def join(args) -> None:
    address = parse_address(args.address)

    try:
        connection = socket.create_connection(address, timeout=10)
    except OSError as error:
        raise SystemExit(
            f"Cannot reach a meeting at {address[0]}:{address[1]} ({error}). "
            "Is the host running?"
        )

    connection.settimeout(None)

    def send(message: dict) -> None:
        connection.sendall((json.dumps(message) + "\n").encode())

    def read_keyboard() -> None:
        try:
            for line in sys.stdin:
                send({"say": line.rstrip("\r\n")})
        except OSError:
            pass

    try:
        send({"join": args.seat})
        threading.Thread(target=read_keyboard, daemon=True).start()

        # Everything the meeting sends is shown until it closes the connection.
        for line in connection.makefile("r", encoding="utf-8"):
            print(json.loads(line).get("text", ""), flush=True)
    except (OSError, ValueError):
        pass
    except KeyboardInterrupt:
        return

    print("\nDisconnected from the meeting.")


def main() -> None:
    # Redirected output on Windows may use a code page without e.g. Japanese;
    # replace what it cannot show instead of crashing.
    sys.stdout.reconfigure(errors="replace")

    parser = argparse.ArgumentParser(
        description="Run a configured meeting, or join one as a human participant."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    hosting = commands.add_parser("host", help="Run the meeting described in a file.")
    hosting.add_argument("config", help="Meeting configuration (.toml or .json).")
    seats = hosting.add_mutually_exclusive_group()
    seats.add_argument(
        "--as", dest="seat", default="", metavar="PARTICIPANT",
        help="Your seat (id or name). Other humans join from their own consoles.",
    )
    seats.add_argument(
        "--serve", action="store_true",
        help="Take no seat: every human participant joins from their own console.",
    )
    hosting.add_argument(
        "--listen", default=DEFAULT_ADDRESS, metavar="HOST:PORT",
        help=f"Where consoles connect (default {DEFAULT_ADDRESS}).",
    )
    hosting.add_argument(
        "--model", default=os.environ.get("MEETING_MODEL", ""),
        help="Claude model id, or 'fake' for an offline demo. Default: $MEETING_MODEL "
             "or the engine's default.",
    )
    hosting.add_argument(
        "--effort", default="medium", choices=["low", "medium", "high", "xhigh", "max"],
        help="How deeply Claude thinks (default: medium).",
    )
    hosting.set_defaults(run=host)

    joining = commands.add_parser("join", help="Register this console for a human participant.")
    joining.add_argument(
        "seat", nargs="?", default="",
        help="Your participant id or name. Leave out to see the seats.",
    )
    joining.add_argument(
        "--address", default=DEFAULT_ADDRESS, metavar="HOST:PORT",
        help=f"Where the meeting is hosted (default {DEFAULT_ADDRESS}).",
    )
    joining.set_defaults(run=join)

    args = parser.parse_args()

    if args.command == "host" and not args.model:
        from agentic_meeting.model import DEFAULT_MODEL
        args.model = DEFAULT_MODEL

    args.run(args)


if __name__ == "__main__":
    main()
