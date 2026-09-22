"""Say what one session loaded before it wrote a line of committed source.

``why`` answers the second question of this project's workflow tracking for a
single line: which agent owns it, and what had entered that agent's context
window before the edit that produced it. Ownership is the committed proof that
``code`` already reads. The ordered context beside it is local hook evidence,
which only the clone that recorded it holds; a clone with notes alone reports
the counts those notes carry and says that the order is unavailable.

The prompt of the turn that wrote the line is printed when this clone holds the
session's trace and the note names the tool call. Otherwise nothing here prints
text: a compaction summary is not listed, and every item that is listed is
named by its kind, its locator, and its size, the way the ledger holds it.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import sqlite3
from typing import Any
from urllib.parse import quote

from .code import _Reader, _validate_path
from .costing import allocate_session_usage
from .report import (
    _load_local_context_loads,
    _load_local_tool_calls,
    _parse_time,
    _session_activity,
)
from .workflow import iter_agents


# One owner lists at most this many loaded items, and names at most this many
# of the lines it owns. Both lists start at the oldest entry, as every other
# capped list in this project does, and report the total beside themselves.
MAX_CONTEXT_ITEMS = 20
MAX_OWNER_LINES = 100

# What entered an agent, as the two ledger tables record it. A read, a search,
# a web fetch, a skill, and an MCP call are tool calls; an instruction file and
# a subagent result are context loads. An edit, a write, and a shell command
# changed the work rather than informing it, and a prompt and a compaction
# summary are text this project never stored.
_LOADED_TOOL_CLASSES = frozenset({"read", "search", "web", "skill", "mcp"})
_LOADED_CONTEXT_KINDS = frozenset({"instruction_file", "subagent_result"})
# The classes of the call that writes a line of a file.
_EDIT_TOOL_CLASSES = frozenset({"edit", "write"})

# A line number in a ``FILE:LINE`` argument. A longer run of digits is part of
# the path: no file this inspector reads has that many lines.
_LINE_SUFFIX = re.compile(r"[0-9]{1,7}")
_UNKNOWN_TIME = datetime.min.replace(tzinfo=timezone.utc)


def split_location(value: str) -> tuple[str, int | None]:
    """Return the path and the optional line of a ``FILE[:LINE]`` argument."""

    path, separator, suffix = value.rpartition(":")
    if separator and path and _LINE_SUFFIX.fullmatch(suffix):
        return path, int(suffix)
    return value, None


def _order(row: Mapping[str, Any]) -> tuple[datetime, float]:
    """Return one row's position in the stream its session recorded.

    Both tables stamp ``occurred_at``; only a tool call counts a ``sequence``
    within its session. Two rows of one moment therefore keep the load ahead of
    the calls, and the calls in the order the harness made them.
    """

    return (
        _parse_time(row.get("occurred_at")) or _UNKNOWN_TIME,
        row.get("sequence") or 0,
    )


def _item(row: Mapping[str, Any], kind: str) -> dict[str, Any]:
    """Return one loaded item: its kind, where it came from, and how large.

    No text and no hash. The report already lists a tool call without its
    locator hash and a context load without its content hash, and this list
    names the same rows.
    """

    return {
        "kind": kind,
        "locator": row.get("locator"),
        "tool_name": row.get("tool_name"),
        "size_bytes": row.get("size_bytes"),
        "memory_type": row.get("memory_type"),
        "related_session_id": row.get("related_session_id"),
        "occurred_at": row.get("occurred_at"),
        "before_compaction": bool(row.get("before_compaction")),
    }


def _loaded(activity: Mapping[str, Any]) -> list[tuple[tuple[datetime, float], dict[str, Any]]]:
    """Return everything the session loaded, oldest first, in one order."""

    ordered: list[tuple[tuple[datetime, float], dict[str, Any]]] = []
    for row in activity["context_loads"]:
        if row.get("kind") in _LOADED_CONTEXT_KINDS:
            ordered.append((_order(row), _item(row, str(row["kind"]))))
    for row in activity["tool_calls"]:
        if row.get("tool_class") in _LOADED_TOOL_CLASSES:
            ordered.append((_order(row), _item(row, str(row["tool_class"]))))
    ordered.sort(key=lambda entry: entry[0])
    return ordered


def _cut(
    activity: Mapping[str, Any],
    path: str,
    commit_time: datetime,
    tool_use_id: str | None = None,
) -> tuple[tuple[datetime, float], str]:
    """Return where the loaded list stops, and how that point was found.

    The edit that wrote the line is the tool call its note range names, when
    the range names one. Otherwise it is the session's own edit or write of
    that path before the commit that carries it. One such call is the edit.
    Several are a session that wrote the file more than once, so the list
    stops at the last of them and says so. None leaves the commit itself as
    the boundary.
    """

    if tool_use_id is not None:
        for row in activity["tool_calls"]:
            if row.get("tool_use_id") == tool_use_id:
                return _order(row), "edit"
    edits = [
        row
        for row in activity["tool_calls"]
        if row.get("tool_class") in _EDIT_TOOL_CLASSES
        and row.get("locator") == path
        and _order(row)[0] <= commit_time
    ]
    if len(edits) == 1:
        return _order(edits[0]), "edit"
    if edits:
        return max(_order(row) for row in edits), "last_edit"
    return (commit_time, math.inf), "commit"


def _local_prompt(
    common_dir: Path, session_id: str, tool_use_id: str | None
) -> dict[str, Any] | None:
    """Return the prompt of the turn whose tool call wrote the line.

    Only the clone that recorded the session holds its trace, and only a range
    that names its tool call can be tied to one turn of it.
    """

    database = common_dir / "attribution" / "ledger.sqlite3"
    if tool_use_id is None or not database.is_file():
        return None
    try:
        connection = sqlite3.connect(f"file:{quote(str(database))}?mode=ro", uri=True, timeout=1)
        connection.row_factory = sqlite3.Row
        try:
            if connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'trace_events'"
            ).fetchone() is None:
                return None
            call = connection.execute(
                "SELECT id FROM trace_events WHERE session_id = ? AND tool_use_id = ?"
                " AND kind = 'tool_call'",
                (session_id, tool_use_id),
            ).fetchone()
            if call is None:
                return None
            prompt = connection.execute(
                "SELECT payload, occurred_at FROM trace_events WHERE session_id = ?"
                " AND kind = 'user_prompt' AND id < ? ORDER BY id DESC LIMIT 1",
                (session_id, int(call["id"])),
            ).fetchone()
        finally:
            connection.close()
    except (sqlite3.Error, OSError):
        return None
    if prompt is None:
        return None
    return {"text": json.loads(prompt["payload"])["text"], "at": prompt["occurred_at"]}


def _note_agent(reader: _Reader, commit: str, session_id: str) -> dict[str, Any] | None:
    """Return what the note of a commit records about one of its agents."""

    note = reader.note(commit)
    for agent in iter_agents(note.get("workflow") if note else None):
        if agent.get("session_id") == session_id:
            return {key: value for key, value in agent.items() if key != "children"}
    return None


def _owner(
    reader: _Reader,
    tool_calls: Mapping[str, dict[str, Any]],
    context_loads: Mapping[str, dict[str, Any]],
    session_id: str,
    lines: list[dict[str, Any]],
    usage: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Return one owning session, its commit, and what it had loaded."""

    session = reader.sessions[session_id]
    if usage is not None:
        # The cost a session's harness reported travels in its note. The cost
        # local telemetry allocated to it does not, and only this clone holds
        # it, so it rides beside the note session the way a report carries it.
        session = {**session, "telemetry": usage}

    def written(line: Mapping[str, Any]) -> tuple[datetime, int]:
        meta = reader.info(line["origin"]["commit"])
        return (_parse_time(meta["committed_at"]) or _UNKNOWN_TIME, line["number"])

    # A file covers many lines of one session, written over many commits. The
    # most recent of them is the commit this owner is reported at.
    origin = max(lines, key=written)["origin"]
    meta = reader.info(origin["commit"])
    commit_time = _parse_time(meta["committed_at"]) or _UNKNOWN_TIME
    span = reader.owner_span(origin) or {}
    tool_use_id = span.get("tool_use_id")
    owner = {
        "session": session,
        "prompt": _local_prompt(reader.common_dir, session_id, tool_use_id),
        "commit": {
            "sha": meta["sha"],
            "short_sha": meta["short_sha"],
            "subject": meta["subject"],
            "committed_at": meta["committed_at"],
            "path": origin["path"],
        },
        "lines": [line["number"] for line in lines[:MAX_OWNER_LINES]],
        "line_count": len(lines),
        "lines_truncated": len(lines) > MAX_OWNER_LINES,
        "context": [],
        "context_total": None,
        "context_truncated": False,
        "context_unavailable": None,
        "context_cut": None,
        "profile": None,
    }
    if session_id not in tool_calls and session_id not in context_loads:
        # Only the clone that ran the session holds its ordered activity, and a
        # clone that kept a ledger of its own recorded none of the work another
        # clone did. A note publishes what each agent did as counts, never as an
        # order, so an unknown order stays unknown rather than becoming an empty
        # list.
        owner["profile"] = _note_agent(reader, origin["commit"], session_id)
        owner["context_unavailable"] = (
            "No recorded activity for this session on this clone; its note"
            " carries the counts below."
            if owner["profile"]
            else "No recorded activity for this session on this clone."
        )
        return owner

    activity = _session_activity(tool_calls, context_loads, session)
    if activity["tool_calls_truncated"] or activity["context_loads_truncated"]:
        reader.warnings.add(
            f"Session {session_id} recorded more activity than a report lists."
        )
    limit, cut = _cut(activity, origin["path"], commit_time, tool_use_id)
    items = [item for position, item in _loaded(activity) if position < limit]
    owner.update(
        {
            "context": items[:MAX_CONTEXT_ITEMS],
            "context_total": len(items),
            "context_truncated": len(items) > MAX_CONTEXT_ITEMS,
            "context_cut": cut,
        }
    )
    return owner


def build_why(
    repo: str | Path,
    path: str,
    *,
    line: int | None = None,
    target_ref: str = "main",
) -> dict[str, Any]:
    """Return each owning session of a file or line, and what it loaded first.

    With ``line``, one line of the target revision is answered for. Without it,
    every attributed line of the file is grouped under the session that owns
    it, and each owner is reported at its own most recent commit of that file.
    """

    path = _validate_path(path)
    if line is not None and (
        isinstance(line, bool) or not isinstance(line, int) or line < 1
    ):
        raise ValueError("Choose a line number of 1 or more.")
    reader = _Reader(repo, target_ref)
    lines = reader.snapshot(reader.target, path)
    if not reader.snapshot_exists.get((reader.target, path), False):
        raise ValueError(f"File {path!r} does not exist in the target revision.")
    if line is not None:
        if line > len(lines):
            raise ValueError(
                f"File {path!r} has {len(lines):,} line(s) in the target revision."
            )
        lines = [lines[line - 1]]

    tool_calls = _load_local_tool_calls(reader.common_dir, reader.warnings)
    context_loads = _load_local_context_loads(reader.common_dir, reader.warnings)
    owned: dict[str, list[dict[str, Any]]] = {}
    for item in lines:
        if item["session_id"] is not None:
            owned.setdefault(item["session_id"], []).append(item)
    try:
        usage = allocate_session_usage(reader.common_dir, reader.sessions.values())
    except (OSError, ValueError, sqlite3.Error) as exc:
        usage = {}
        reader.warnings.add(f"Could not read local cost telemetry: {exc}.")
    owners = [
        _owner(
            reader, tool_calls, context_loads, session_id, items, usage.get(session_id)
        )
        for session_id, items in owned.items()
    ]
    return {
        # ``_Reader`` resolves the target or raises, so the header that renders
        # this payload is told the target was found rather than reading a key
        # that a repository record built for another command never carries.
        "repository": {**reader.repository, "target_exists": True},
        "file": path,
        "line": line,
        "target": target_ref,
        "owners": owners,
        "warnings": reader.warnings.items,
    }
