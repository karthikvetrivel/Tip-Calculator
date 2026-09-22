"""Store and publish the text of a session: its trace.

A trace is what the user said, what the model replied, what each tool was
given and what it returned, and the summary that replaced a compacted
context. The ledger keeps each event as one ``trace_events`` row, redacted
and capped before it is written. The pre-push snapshot publishes one trace
object per session, and the trusted workflow and the hosted service accept
it only through ``validate_trace``, so every side agrees on one shape.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import PurePosixPath
import re
import sqlite3
from typing import Any

from .activity import _SECRET_PREFIXES, _TOKEN_RE
from .store import git_common_dir


TRACE_VERSION = 1
# The hosted object type that carries one trace beside a PR artifact.
SCHEMA = "harness-attribution/pr-traces@1"
KINDS = (
    "user_prompt", "assistant", "tool_call", "tool_result", "compaction",
    "subagent_result",
)
# Caps on one event. A prompt, a reply, and a compaction summary keep their
# first 64 KiB. A tool input keeps 32 KiB of its JSON. A tool output keeps its
# first 12 KiB and its last 4 KiB, because the end of a command's output is
# where its result and its error live.
MAX_TEXT_BYTES = 64 * 1024
MAX_INPUT_BYTES = 32 * 1024
MAX_OUTPUT_BYTES = 16 * 1024
_OUTPUT_HEAD_BYTES = 12 * 1024
_OUTPUT_TAIL_BYTES = 4 * 1024
# Caps on one session in the ledger and on one published trace.
MAX_SESSION_BYTES = 4 * 1024 * 1024
MAX_SESSION_ROWS = 5_000
MAX_TRACE_BYTES = 4 * 1024 * 1024
MAX_SNAPSHOT_TRACE_BYTES = 32 * 1024 * 1024
TRUNCATION_MARK = "\n[... truncated ...]\n"
_REDACTED = "[redacted]"
_PEM_BLOCK = re.compile(r"-----BEGIN[^-]*-----.*?-----END[^-]*-----", re.DOTALL)
_OID = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_TEXT_KINDS = frozenset({"user_prompt", "assistant", "compaction", "subagent_result"})
_MAX_ID_CHARS = 2048
_MAX_LINE = 1_000_000


def traces_enabled(repo: Any) -> bool:
    """Return whether this repository keeps the text of its sessions.

    ``joyride install`` writes the choice into its manifest. A manifest
    written before the key existed keeps traces, and no manifest keeps none.
    """

    state_path = git_common_dir(repo) / "attribution" / "install.json"
    try:
        state = json.loads(state_path.read_bytes()[: 64 * 1024].decode("utf-8"))
    except (OSError, ValueError, UnicodeDecodeError):
        return False
    return isinstance(state, dict) and state.get("traces_enabled") is not False


def redact(text: str) -> str:
    """Replace every token that starts like a secret, and every PEM block."""

    text = _PEM_BLOCK.sub(_REDACTED, text)
    return _TOKEN_RE.sub(
        lambda match: _REDACTED if match.group().startswith(_SECRET_PREFIXES) else match.group(),
        text,
    )


def _encode(text: str) -> bytes:
    return text.encode("utf-8", errors="replace")


def _head(text: str, limit: int) -> str:
    return _encode(text)[:limit].decode("utf-8", errors="ignore")


def cap_text(text: str, limit: int = MAX_TEXT_BYTES) -> tuple[str, bool]:
    """Return the text within ``limit`` bytes and whether it was cut."""

    if len(_encode(text)) <= limit:
        return text, False
    return _head(text, limit) + TRUNCATION_MARK, True


def cap_output(text: str) -> tuple[str, bool]:
    """Return a tool output as its head and its tail when it is too long."""

    encoded = _encode(text)
    if len(encoded) <= MAX_OUTPUT_BYTES:
        return text, False
    tail = encoded[-_OUTPUT_TAIL_BYTES:].decode("utf-8", errors="ignore")
    return _head(text, _OUTPUT_HEAD_BYTES) + TRUNCATION_MARK + tail, True


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(value)


def _capped_input(value: Any) -> tuple[Any, bool]:
    """Return a tool input as a JSON object, or a marked cut of its text."""

    if not isinstance(value, Mapping):
        value = {} if value is None else {"value": value}
    encoded = _stringify(value)
    if len(_encode(encoded)) <= MAX_INPUT_BYTES:
        return json.loads(encoded), False
    return {"_truncated": True, "text": _head(encoded, MAX_INPUT_BYTES)}, True


def _capped_fields(kind: str, fields: Mapping[str, Any]) -> tuple[dict[str, Any], bool]:
    payload: dict[str, Any] = {}
    truncated = False
    for name, value in fields.items():
        if name == "text":
            value, cut = cap_text(_stringify(value))
        elif name == "output":
            value, cut = cap_output(_stringify(value))
        elif name == "input":
            value, cut = _capped_input(value)
        else:
            cut = False
        payload[name] = value
        truncated = truncated or cut
    return payload, truncated


def _redacted(value: Any) -> Any:
    """Redact every string of a payload before it is encoded."""

    if isinstance(value, str):
        return redact(value)
    if isinstance(value, Mapping):
        return {redact(str(key)): _redacted(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_redacted(item) for item in value]
    return value


def record_trace_event(
    connection: sqlite3.Connection,
    session_id: str,
    kind: str,
    fields: Mapping[str, Any],
    *,
    occurred_at: str,
    tool_use_id: str | None = None,
    agent_id: str | None = None,
    turn_id: str | None = None,
) -> bool:
    """Insert one redacted, capped event, and stop at the session caps.

    ``fields`` holds the kind's own values: ``text`` for a prompt, a reply, a
    compaction summary, or a subagent result; ``tool_name``, ``tool_class``,
    ``summary``, and ``input`` for a tool call; ``succeeded`` and ``output``
    for a tool result. Past the session caps the row is dropped and the
    session's ``trace_truncated`` flag is set.
    """

    if kind not in KINDS:
        raise ValueError(f"unknown trace event kind: {kind!r}")
    payload, truncated = _capped_fields(kind, fields)
    encoded = json.dumps(_redacted(payload), sort_keys=True, ensure_ascii=False)
    size = len(_encode(encoded))
    stored = connection.execute(
        "SELECT COUNT(*), COALESCE(SUM(size_bytes), 0) FROM trace_events WHERE session_id = ?",
        (session_id,),
    ).fetchone()
    if int(stored[0]) >= MAX_SESSION_ROWS or int(stored[1]) + size > MAX_SESSION_BYTES:
        connection.execute(
            "UPDATE sessions SET trace_truncated = 1 WHERE id = ?", (session_id,)
        )
        return False
    connection.execute(
        """
        INSERT INTO trace_events(
            session_id, kind, tool_use_id, agent_id, turn_id, occurred_at,
            payload, size_bytes, truncated
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            session_id, kind, tool_use_id, agent_id, turn_id, occurred_at,
            encoded, size, int(truncated),
        ),
    )
    return True


def canonical_bytes(value: Any) -> bytes:
    """Return the one encoding that sizes and digests a trace."""

    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def trace_digest(trace: Mapping[str, Any]) -> str:
    body = {key: value for key, value in trace.items() if key != "digest"}
    return "sha256:" + hashlib.sha256(canonical_bytes(body)).hexdigest()


def _note_edits(session_id: str, notes: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return the lines each tool call of one session wrote, from its notes."""

    grouped: dict[tuple[str, str, str], list[dict[str, int]]] = {}
    for note in notes:
        commit = note.get("commit")
        if not isinstance(commit, str):
            continue
        for entry in note.get("files") or []:
            path = entry.get("path")
            for item in entry.get("ranges") or []:
                tool_use_id = item.get("tool_use_id")
                if item.get("session_id") != session_id or not isinstance(tool_use_id, str):
                    continue
                grouped.setdefault((tool_use_id, commit, str(path)), []).append(
                    {"start": int(item["start"]), "end": int(item["end"])}
                )
    return [
        {"tool_use_id": tool_use_id, "commit": commit, "path": path, "ranges": ranges}
        for (tool_use_id, commit, path), ranges in sorted(grouped.items())
    ]


def build_trace(
    connection: sqlite3.Connection, session_id: str, notes: list[Mapping[str, Any]]
) -> dict[str, Any] | None:
    """Return the published trace of one session, or None without events."""

    rows = connection.execute(
        """
        SELECT id, kind, tool_use_id, agent_id, occurred_at, payload, truncated
        FROM trace_events WHERE session_id = ? ORDER BY occurred_at, id
        """,
        (session_id,),
    ).fetchall()
    if not rows:
        return None
    session = connection.execute(
        "SELECT harness, model, trace_truncated FROM sessions WHERE id = ?",
        (session_id,),
    ).fetchone()
    events: list[dict[str, Any]] = []
    turn = 0
    for row in rows:
        if row["kind"] == "user_prompt":
            turn += 1
        event: dict[str, Any] = {
            "seq": len(events) + 1, "at": row["occurred_at"], "kind": row["kind"],
            "turn": turn, **json.loads(row["payload"]),
        }
        if row["tool_use_id"] is not None:
            event["tool_use_id"] = row["tool_use_id"]
        if row["agent_id"] is not None:
            event["agent_id"] = row["agent_id"]
        if row["truncated"]:
            event["truncated"] = True
        events.append(event)
    trace: dict[str, Any] = {
        "version": TRACE_VERSION, "session_id": session_id,
        "harness": session["harness"], "model": session["model"],
        "events": events, "edits": _note_edits(session_id, notes),
        "truncated": bool(session["trace_truncated"]),
    }
    # A trace over the blob cap loses its oldest turns whole, so the newest
    # work, which is what a pull request shows, keeps its full context.
    while len(canonical_bytes(trace)) > MAX_TRACE_BYTES and trace["events"]:
        oldest = trace["events"][0]["turn"]
        trace["events"] = [event for event in trace["events"] if event["turn"] != oldest]
        trace["truncated"] = True
    for index, event in enumerate(trace["events"], 1):
        event["seq"] = index
    trace["digest"] = trace_digest(trace)
    return trace


def _fail(message: str) -> ValueError:
    return ValueError(f"Invalid trace: {message}.")


def _text_value(value: Any, name: str, limit: int) -> str:
    if not isinstance(value, str) or "\0" in value or len(_encode(value)) > limit:
        raise _fail(f"{name} is not text within its limit")
    return value


def _validate_event(event: Any, previous_seq: int) -> int:
    if not isinstance(event, Mapping):
        raise _fail("an event is not an object")
    kind = event.get("kind")
    if kind not in KINDS:
        raise _fail("an event kind is unknown")
    seq, turn = event.get("seq"), event.get("turn")
    if (
        isinstance(seq, bool) or not isinstance(seq, int) or seq != previous_seq + 1
        or isinstance(turn, bool) or not isinstance(turn, int) or turn < 0
    ):
        raise _fail("event order is broken")
    _text_value(event.get("at"), "an event time", 128)
    allowed = {"seq", "at", "kind", "turn", "agent_id", "truncated"}
    if "agent_id" in event:
        _text_value(event["agent_id"], "an agent id", _MAX_ID_CHARS)
    if "truncated" in event and not isinstance(event["truncated"], bool):
        raise _fail("a truncation flag is not a boolean")
    slack = len(_encode(TRUNCATION_MARK))
    if kind in _TEXT_KINDS:
        _text_value(event.get("text"), "event text", MAX_TEXT_BYTES + slack)
        allowed.add("text")
        if kind == "compaction":
            allowed.add("trigger")
            if event.get("trigger") is not None:
                _text_value(event["trigger"], "a compaction trigger", 256)
        if kind == "subagent_result":
            allowed.add("agent_type")
            if event.get("agent_type") is not None:
                _text_value(event["agent_type"], "an agent type", 256)
    else:
        _text_value(event.get("tool_use_id"), "a tool use id", _MAX_ID_CHARS)
        allowed.add("tool_use_id")
        if kind == "tool_call":
            allowed.update({"tool_name", "tool_class", "summary", "input"})
            _text_value(event.get("tool_name"), "a tool name", 512)
            _text_value(event.get("tool_class"), "a tool class", 64)
            if event.get("summary") is not None:
                _text_value(event["summary"], "a tool summary", 256)
            if not isinstance(event.get("input"), Mapping) or len(
                canonical_bytes(event["input"])
            ) > MAX_INPUT_BYTES + 4096:
                raise _fail("a tool input is not an object within its limit")
        else:
            allowed.update({"succeeded", "output"})
            if event.get("succeeded") is not None and not isinstance(event["succeeded"], bool):
                raise _fail("a tool outcome is not a boolean")
            _text_value(event.get("output"), "a tool output", MAX_OUTPUT_BYTES + slack)
    if set(event) - allowed:
        raise _fail("an event carries an unknown key")
    return seq


def _validate_edit(edit: Any) -> None:
    if not isinstance(edit, Mapping) or set(edit) != {"tool_use_id", "commit", "path", "ranges"}:
        raise _fail("an edit has the wrong keys")
    _text_value(edit["tool_use_id"], "an edit tool use id", _MAX_ID_CHARS)
    if not isinstance(edit["commit"], str) or _OID.fullmatch(edit["commit"]) is None:
        raise _fail("an edit commit is not an object id")
    path = _text_value(edit["path"], "an edit path", 4096)
    parts = PurePosixPath(path)
    if parts.is_absolute() or ".." in parts.parts or not path:
        raise _fail("an edit path is not repository relative")
    if not isinstance(edit["ranges"], list) or not edit["ranges"]:
        raise _fail("an edit has no ranges")
    for item in edit["ranges"]:
        if not isinstance(item, Mapping) or set(item) != {"start", "end"}:
            raise _fail("a range has the wrong keys")
        start, end = item["start"], item["end"]
        if (
            isinstance(start, bool) or isinstance(end, bool)
            or not isinstance(start, int) or not isinstance(end, int)
            or not 1 <= start <= end <= _MAX_LINE
        ):
            raise _fail("a range is out of order")


def validate_trace(value: Any) -> dict[str, Any]:
    """Return a trace object that every side of the pipeline accepts."""

    if not isinstance(value, Mapping) or set(value) != {
        "version", "session_id", "harness", "model", "events", "edits", "truncated", "digest",
    }:
        raise _fail("wrong top-level keys")
    if value["version"] != TRACE_VERSION or isinstance(value["version"], bool):
        raise _fail("unsupported version")
    _text_value(value["session_id"], "the session id", _MAX_ID_CHARS)
    _text_value(value["harness"], "the harness", 512)
    _text_value(value["model"], "the model", 512)
    if not isinstance(value["truncated"], bool):
        raise _fail("the truncation flag is not a boolean")
    events, edits = value["events"], value["edits"]
    if not isinstance(events, list) or not events or len(events) > MAX_SESSION_ROWS:
        raise _fail("events are missing or too many")
    previous = 0
    for event in events:
        previous = _validate_event(event, previous)
    if not isinstance(edits, list):
        raise _fail("edits are not a list")
    for edit in edits:
        _validate_edit(edit)
    if not isinstance(value["digest"], str) or _DIGEST.fullmatch(value["digest"]) is None:
        raise _fail("the digest is malformed")
    if len(canonical_bytes(value)) > MAX_TRACE_BYTES:
        raise _fail("the trace exceeds 4 MiB")
    if trace_digest(value) != value["digest"]:
        raise _fail("the digest does not match the trace")
    return dict(value)
