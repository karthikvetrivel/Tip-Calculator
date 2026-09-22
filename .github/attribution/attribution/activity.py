"""Workflow facts shared by both native hook receivers.

The installed receiver (``automation.handle_hook``) and the template receiver
(``hook_capture.record_hook_event``) read different payload shapes but record
the same workflow evidence. The pure functions here name that evidence, and the
ledger helpers below write it.

Nothing in this module stores prompt text, transcript paths, tool responses, or
MCP arguments; ``traces`` keeps the text of a session beside the rows written
here. A locator is a short, human-readable label; the hash beside it covers the
full value so a redacted locator still joins across events.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import os
from pathlib import Path
import re
import sqlite3
from typing import Any

from .hook_events import response_success
from .hooks import normalized_event


TOOL_CLASSES = (
    "read",
    "search",
    "edit",
    "write",
    "shell",
    "web",
    "skill",
    "mcp",
    "agent",
    "ask_user",
    "other",
)
CONTEXT_LOAD_KINDS = (
    "instruction_file",
    "compaction_summary",
    "subagent_result",
    "user_prompt",
    "injected_context",
)
LAUNCH_MODES = frozenset({"foreground", "background"})
SESSION_SOURCES = frozenset(
    {"startup", "resume", "fork", "clear", "compact", "subagent"}
)
MODEL_SWITCH_SOURCES = frozenset(
    {"user", "auto", "resume", "agent_response", "unknown"}
)

MAX_LOCATOR_CHARS = 200
# Each session stores at most this many tool calls. The count of calls keeps
# growing after the cap, so a report can still say how much it did not store.
MAX_TOOL_CALL_ROWS = 5_000
# Each session stores at most this many context loads. Both caps raise the one
# ``activity_truncated`` flag, which says that a session's activity is
# incomplete without claiming which half of it was cut.
MAX_CONTEXT_LOAD_ROWS = 5_000
# A tool that changes the working tree needs the snapshot pair that proves what
# it changed. Every other class is recorded from its completion alone.
SNAPSHOT_TOOL_CLASSES = frozenset({"edit", "write", "shell"})
_MAX_FACET_CHARS = 256
_MAX_MODELS_USED = 32

_TOOL_CLASS_BY_NAME = {
    "Read": "read",
    "NotebookRead": "read",
    "Grep": "search",
    "Glob": "search",
    "Edit": "edit",
    "NotebookEdit": "edit",
    "apply_patch": "edit",
    "Write": "write",
    "Bash": "shell",
    "PowerShell": "shell",
    # Codex runs commands through a tool named ``shell``, and accepts ``Bash``
    # as a hook matcher alias for it. Both names are the same tool, so both
    # take the snapshot pair that proves what a command changed.
    "shell": "shell",
    "WebFetch": "web",
    "WebSearch": "web",
    "Skill": "skill",
    "Agent": "agent",
    "Task": "agent",
    "AskUserQuestion": "ask_user",
}

# The installer subscribes its pre-tool hook to exactly these names, so a tool
# never reaches the receiver for a snapshot it does not need. A tool that
# changes files under a name the class table does not hold is named here rather
# than reclassified. ``MultiEdit`` is the one such Claude Code tool this
# project knows.
SNAPSHOT_TOOL_NAMES = frozenset(
    {
        name
        for name, tool_class in _TOOL_CLASS_BY_NAME.items()
        if tool_class in SNAPSHOT_TOOL_CLASSES
    }
    | {"MultiEdit"}
)

# A harness says that a tool call failed by sending its own failure event.
# These are the harnesses whose hooks this project subscribes to one: ``_EVENTS``
# in ``install.py`` and ``_NESTED_EVENTS`` and ``_COPILOT_EVENTS`` in
# ``hook_templates.py``. Anywhere else a completion event fires for a failed
# call too, so completion alone proves nothing about the outcome.
FAILURE_EVENT_HARNESSES = frozenset({"claude-code", "github-copilot", "qwen-code"})
# ``response_success`` reads a plain-string response only for a patch or a
# shell command, so only those classes name the shape it knows.
_RESPONSE_KINDS = {"edit": "patch", "write": "patch", "shell": "shell"}

# The events that load something into an agent, and the kind of load each one
# reports. The name is normalized before the lookup, so a harness that spells
# the event ``PostCompact`` and one that spells it ``post_compact`` agree, an
# aliased spelling such as ``user_input`` reaches the same row as
# ``UserPromptSubmit``, and a receiver holding the normalized name already
# looks up the same kind.
_CONTEXT_LOAD_EVENTS = {
    "user_prompt": "user_prompt",
    "post_compact": "compaction_summary",
    "instructions_loaded": "instruction_file",
    "subagent_stop": "subagent_result",
}
# The payload field whose size and hash stand in for text the ledger never
# stores.
_CONTEXT_LOAD_TEXT_FIELDS = {
    "user_prompt": "prompt",
    "compaction_summary": "compact_summary",
    "subagent_result": "last_assistant_message",
}

# A launched subagent either ran to completion inside the parent's tool call or
# was handed off to run beside it.
_LAUNCH_MODE_BY_STATUS = {
    "completed": "foreground",
    "async_launched": "background",
}

_SECRET_PREFIXES = (
    "sk-",
    "ghp_",
    "gho_",
    "github_pat_",
    "AKIA",
    "xoxb-",
    "xoxp-",
    "-----BEGIN",
    "AIza",
)
# Tokens keep the characters that every known secret prefix uses, so a path
# segment or command word that begins with one is found without splitting it.
_TOKEN_RE = re.compile(r"[A-Za-z0-9_-]+")
_OPAQUE_RUN_RE = re.compile(r"[A-Za-z0-9]{32,}")


def _text(value: Any, *, max_chars: int = _MAX_FACET_CHARS) -> str | None:
    """Return one bounded, NUL-free string, or None for anything else."""

    if not isinstance(value, str):
        return None
    result = value.strip()
    if not result or "\0" in result or len(result) > max_chars:
        return None
    return result


def _count(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if 0 <= value <= 2**63 - 1 else None


def content_digest(text: Any) -> tuple[str, int] | None:
    """Return the hex SHA-256 and byte size of text that is never stored."""

    if not isinstance(text, str):
        return None
    encoded = text.encode("utf-8", errors="surrogatepass")
    return hashlib.sha256(encoded).hexdigest(), len(encoded)


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="surrogatepass")).hexdigest()


def session_facets(harness_id: str, payload: Any) -> dict[str, Any]:
    """Return the workflow facets one payload carries, and nothing else.

    The payload is a hook payload or the ``Agent`` tool response inside one.
    Only the documented keys are read; every other key is ignored, so an
    unrecognized field can never reach the ledger.
    """

    del harness_id  # Every supported harness names these facets identically.
    if not isinstance(payload, Mapping):
        return {}
    facets: dict[str, Any] = {}

    agent_type = _text(payload.get("agent_type"))
    if agent_type is not None:
        facets["agent_type"] = agent_type
    permission_mode = _text(payload.get("permission_mode"))
    if permission_mode is not None:
        facets["permission_mode"] = permission_mode

    effort = payload.get("effort")
    effort_level = (
        _text(effort.get("level")) if isinstance(effort, Mapping) else _text(effort)
    )
    if effort_level is not None:
        facets["effort_level"] = effort_level

    session_source = _text(payload.get("source"))
    if session_source in SESSION_SOURCES:
        facets["session_source"] = session_source

    duration = _count(payload.get("duration_ms"))
    if duration is None:
        duration = _count(payload.get("totalDurationMs"))
    if duration is not None:
        facets["duration_ms"] = duration

    launch_mode = _LAUNCH_MODE_BY_STATUS.get(_text(payload.get("status")) or "")
    if launch_mode is not None:
        facets["launch_mode"] = launch_mode

    agent_id = _text(payload.get("agentId"))
    if agent_id is not None:
        facets["agent_id"] = agent_id
    resolved_model = _text(payload.get("resolvedModel"))
    if resolved_model is not None:
        facets["resolved_model"] = resolved_model

    models_used = payload.get("modelsUsed")
    if isinstance(models_used, (list, tuple)):
        named = [
            model
            for model in (_text(item) for item in models_used[:_MAX_MODELS_USED])
            if model is not None
        ]
        if named:
            facets["models_used"] = named

    tool_call_count = _count(payload.get("totalToolUseCount"))
    if tool_call_count is not None:
        facets["tool_call_count"] = tool_call_count
    return facets


def classify_tool(harness_id: str, tool_name: Any) -> str:
    """Return the ledger class of one tool call.

    Names outside the table are ``other``, so a local function tool named
    ``WriteReport`` is never counted as a file write.
    """

    del harness_id  # Tool names are harness-wide; only their payloads differ.
    name = _text(tool_name, max_chars=512)
    if name is None:
        return "other"
    if name.startswith("mcp__"):
        return "mcp"
    return _TOOL_CLASS_BY_NAME.get(name, "other")


def takes_snapshot(harness_id: str, tool_name: Any) -> bool:
    """Return whether one tool call needs the worktree snapshot pair.

    Only a tool that can change the working tree needs a baseline. Every other
    tool is recorded from its completion alone, which keeps a read, a search,
    or a subagent launch to one small database write.
    """

    del harness_id  # Tool names are harness-wide; only their payloads differ.
    return _text(tool_name, max_chars=512) in SNAPSHOT_TOOL_NAMES


def _has_secret_prefix(value: str) -> bool:
    return any(
        token.startswith(_SECRET_PREFIXES) for token in _TOKEN_RE.findall(value)
    )


def _has_opaque_token(value: str) -> bool:
    for run in _OPAQUE_RUN_RE.findall(value):
        if (
            any(character.isupper() for character in run)
            and any(character.islower() for character in run)
            and any(character.isdigit() for character in run)
        ):
            return True
    return False


def _sanitized(display: str, full: str) -> tuple[str | None, str | None]:
    """Apply the shared locator rules to one composed value.

    The rules read the display value, because that is what a report prints. The
    hash always covers the full value, so a dropped locator still identifies
    the same file, command, or URL across sessions.
    """

    if "\0" in display or "\0" in full:
        return None, None
    digest = _hash(full)
    if len(display) > MAX_LOCATOR_CHARS:
        return None, digest
    if _has_secret_prefix(display) or _has_opaque_token(display):
        return None, digest
    return display, digest


def _path_parts(value: Any, repo_root: Any) -> tuple[str | None, str | None]:
    """Return the display path and the value its hash covers.

    The filesystem is never read. A hook reports a path that may already be
    gone, and resolving symlinks would turn one file into several locators.
    """

    text = _text(value, max_chars=32 * 1024)
    if text is None:
        return None, None
    root = os.path.normpath(str(repo_root)) if repo_root is not None else None
    if os.path.isabs(text):
        absolute = os.path.normpath(text)
    elif root is not None:
        absolute = os.path.normpath(os.path.join(root, text))
    else:
        return text.replace(os.sep, "/"), text.replace(os.sep, "/")
    if root is not None:
        relative = os.path.relpath(absolute, root)
        if relative != os.pardir and not relative.startswith(os.pardir + os.sep):
            posix = relative.replace(os.sep, "/")
            return posix, posix
    return "outside_repo", absolute


def path_locator(
    value: Any, repo_root: str | Path | None
) -> tuple[str | None, str | None]:
    """Return the locator and the hash of one reported file path.

    A context load names a file the way a tool call does, so both read these
    rules: repo-relative inside the repository, ``outside_repo`` outside it,
    with the hash covering the full path either way.
    """

    display, full = _path_parts(value, repo_root)
    if display is None or full is None:
        return None, None
    return _sanitized(display, full)


def extract_locator(
    harness_id: str,
    tool_name: Any,
    tool_class: str,
    tool_input: Any,
    repo_root: str | Path | None,
) -> tuple[str | None, str | None]:
    """Return one tool call's locator and the hash of its full value.

    Only the fields the class table names are read. No other ``tool_input``
    value, and no ``tool_response`` value, reaches the result.
    """

    if not isinstance(tool_input, Mapping):
        tool_input = {}
    name = _text(tool_name, max_chars=512)

    if tool_class in {"read", "write"} or (
        tool_class == "edit" and name != "apply_patch"
    ):
        return path_locator(tool_input.get("file_path"), repo_root)

    if tool_class == "edit":
        # Codex applies one patch across several files, so no single path
        # describes the call.
        return None, None

    if tool_class == "search":
        pattern = _text(tool_input.get("pattern"), max_chars=32 * 1024)
        if pattern is None:
            return None, None
        display, full = _path_parts(tool_input.get("path"), repo_root)
        if display is None or full is None:
            return _sanitized(pattern, pattern)
        return _sanitized(f"{pattern} in {display}", f"{pattern} in {full}")

    if tool_class == "shell":
        command = _text(tool_input.get("command"), max_chars=128 * 1024)
        if command is None:
            return None, None
        words = command.split()
        if not words:
            return None, None
        return _sanitized(f"{words[0]} +{len(words) - 1}", command)

    if tool_class == "web":
        url = _text(tool_input.get("url"), max_chars=32 * 1024)
        if url is not None:
            return _sanitized(url.split("#", 1)[0].split("?", 1)[0], url)
        query = _text(tool_input.get("query"), max_chars=32 * 1024)
        if query is not None:
            return _sanitized(query, query)
        return None, None

    if tool_class == "skill":
        skill = _text(tool_input.get("skill")) or _text(tool_input.get("name"))
        if skill is None:
            return None, None
        return _sanitized(skill, skill)

    if tool_class == "mcp":
        parts = (name or "").split("__")
        if len(parts) < 3 or not parts[1] or not parts[2]:
            return None, None
        server, tool = parts[1], "__".join(parts[2:])
        return _sanitized(f"{server}/{tool}", f"{server}/{tool}")

    if tool_class == "agent":
        subagent_type = _text(tool_input.get("subagent_type"))
        if subagent_type is None:
            return None, None
        return _sanitized(subagent_type, subagent_type)

    return None, None


def agent_summary(tool_input: Any) -> str | None:
    """Return the short purpose a parent gave one subagent, or None.

    Claude Code carries it as the ``description`` of an ``Agent`` call: a few
    words such as ``Find flaky tests`` that the orchestrator wrote to say what
    the subagent is for. It is bounded exactly as a locator is, because it is
    text a model composed: over the cap, on more than one line, or holding a
    token that looks like a secret, nothing is stored.
    """

    if not isinstance(tool_input, Mapping):
        return None
    description = _text(tool_input.get("description"), max_chars=MAX_LOCATOR_CHARS)
    if description is None or "\n" in description or "\r" in description:
        return None
    summary, _digest = _sanitized(description, description)
    return summary


def tool_succeeded(
    harness_id: str,
    event_name: Any,
    *,
    tool_class: str | None = None,
    tool_response: Any = None,
) -> bool | None:
    """Return the outcome one tool event reports, or None for unknown.

    A failure event is an outcome by itself. A completion event is not: a
    harness with no failure event sends the same event for a call that failed,
    so completion means success only where a failure would have arrived on its
    own event. An explicit outcome in the response outranks both, and only the
    boolean it yields is returned; no part of the response is stored. A
    pre-tool event has no outcome yet.
    """

    name = _text(event_name, max_chars=512)
    if name is None:
        return None
    folded = name.casefold().replace("_", "")
    if "pretool" in folded:
        return None
    if folded.endswith("failure") or folded.endswith("failed"):
        return False
    if "posttool" not in folded and "aftertool" not in folded:
        return None
    reported = response_success(
        tool_response, _RESPONSE_KINDS.get(tool_class or "", "other")
    )
    if reported is not None:
        return reported
    return True if harness_id in FAILURE_EVENT_HARNESSES else None


def context_load_kind(event_name: Any) -> str | None:
    """Return the kind of context load one event reports, or None for others."""

    name = _text(event_name, max_chars=512)
    if name is None:
        return None
    normalized = normalized_event(name)
    if normalized is None:
        return None
    return _CONTEXT_LOAD_EVENTS.get(normalized)


def context_load(
    harness_id: str,
    payload: Any,
    repo_root: str | Path | None = None,
    *,
    event: str | None = None,
) -> dict[str, Any] | None:
    """Return the content-free record of what one event loaded, or None.

    The event decides which field is read, because a payload commonly repeats
    the prompt of the turn it belongs to. Text is reduced to its hash and its
    size here; the trace event the installed receiver writes beside this row
    is where the text itself is kept. An instruction file reports a path
    rather than text, so its hash covers that path.
    """

    del harness_id  # Every harness that reports a load names these fields alike.
    kind = context_load_kind(event)
    if kind is None or not isinstance(payload, Mapping):
        return None
    entry: dict[str, Any] = {
        "kind": kind,
        "locator": None,
        "content_hash": None,
        "size_bytes": None,
        "memory_type": _text(payload.get("memory_type")),
        "load_reason": _text(payload.get("load_reason")),
    }
    if kind == "instruction_file":
        locator, locator_hash = path_locator(payload.get("file_path"), repo_root)
        if locator is None and locator_hash is None:
            # Nothing in the payload names the file that was loaded.
            return None
        entry["locator"] = locator
        entry["content_hash"] = locator_hash
        return entry

    if kind == "compaction_summary":
        entry["load_reason"] = entry["load_reason"] or _text(payload.get("trigger"))
    digest = content_digest(payload.get(_CONTEXT_LOAD_TEXT_FIELDS[kind]))
    if digest is not None:
        entry["content_hash"], entry["size_bytes"] = digest
    return entry


def hook_activity(
    harness_id: str,
    payload: Any,
    repo_root: str | Path | None = None,
    *,
    event: str | None = None,
) -> dict[str, Any]:
    """Return the content-free workflow facts one hook payload carries.

    The template receiver is given a ``HookEvent``, never the payload, so this
    is the only path by which a facet, a tool class, or a locator reaches it.
    The result holds no prompt, transcript path, tool response body, or MCP
    argument, and its caller may pass it straight to the ledger.

    ``event`` is the harness event name when the caller was told it on the
    command line. It distinguishes a failed tool call from a completed one,
    which the normalized event no longer separates, and it names which field
    of the payload one context load reads.
    """

    if not isinstance(payload, Mapping):
        return {}
    result: dict[str, Any] = {"facets": session_facets(harness_id, payload)}
    event_name = event or payload.get("hook_event_name")

    loaded = context_load(harness_id, payload, repo_root, event=event_name)
    if loaded is not None:
        result["context_load"] = loaded

    tool_name = _text(payload.get("tool_name"), max_chars=512)
    tool_use_id = _text(payload.get("tool_use_id"), max_chars=512)
    if tool_name is not None and tool_use_id is not None:
        tool_class = classify_tool(harness_id, tool_name)
        locator, locator_hash = extract_locator(
            harness_id, tool_name, tool_class, payload.get("tool_input"), repo_root
        )
        result["tool_call"] = {
            "tool_use_id": tool_use_id,
            "tool_name": tool_name,
            "tool_class": tool_class,
            "locator": locator,
            "locator_hash": locator_hash,
            "succeeded": tool_succeeded(
                harness_id,
                event_name,
                tool_class=tool_class,
                tool_response=payload.get("tool_response"),
            ),
            "duration_ms": _count(payload.get("duration_ms")),
        }
        if tool_class == "agent":
            # Only a subagent launch names a purpose, and only the call that
            # made it carries one, so the key appears nowhere else.
            summary = agent_summary(payload.get("tool_input"))
            if summary is not None:
                result["tool_call"]["summary"] = summary

    response = payload.get("tool_response")
    if isinstance(response, Mapping):
        result["response_facets"] = session_facets(harness_id, response)

    # Only a model-switch event names a model it moved to.
    to_model = _text(payload.get("to_model"))
    if to_model is not None:
        result["model_switch"] = {
            "from_model": _text(payload.get("from_model")),
            "to_model": to_model,
            "source": _text(payload.get("source")) or "unknown",
        }
    return result


_TEXT_FACET_COLUMNS = (
    "agent_type",
    "launch_mode",
    "session_source",
    "permission_mode",
    "effort_level",
)


def apply_session_facets(
    connection: sqlite3.Connection,
    session_id: str,
    facets: Mapping[str, Any],
    *,
    fields: tuple[str, ...],
) -> None:
    """Write the named facets onto one session row.

    ``permission_mode`` and ``effort_level`` hold the most recent value the
    harness reported. The remaining facets describe how the session started, so
    a later event never overwrites a value that is already known.
    """

    assignments: list[str] = []
    values: list[Any] = []
    for field in fields:
        if field not in facets:
            continue
        value = facets[field]
        if field in {"launch_mode", "session_source"} and value not in (
            LAUNCH_MODES if field == "launch_mode" else SESSION_SOURCES
        ):
            continue
        if field in {"permission_mode", "effort_level"}:
            assignments.append(f"{field} = ?")
        elif field in _TEXT_FACET_COLUMNS or field == "duration_ms":
            assignments.append(f"{field} = COALESCE({field}, ?)")
        else:
            continue
        values.append(value)
    if not assignments:
        return
    connection.execute(
        f"UPDATE sessions SET {', '.join(assignments)} WHERE id = ?",
        (*values, session_id),
    )


def record_model_switch(
    connection: sqlite3.Connection,
    session_id: str,
    *,
    from_model: str | None,
    to_model: str | None,
    source: str,
    occurred_at: str,
) -> bool:
    """Insert one model switch and keep the session's switch count in step."""

    target = _text(to_model)
    if target is None:
        return False
    if source not in MODEL_SWITCH_SOURCES:
        source = "unknown"
    previous = _text(from_model)
    if previous == "unknown":
        previous = None
    if previous == target:
        return False
    if source == "agent_response":
        # A parent's Agent response summarizes a finished subagent, so it names
        # a set of models rather than an ordered history. Replaying the same
        # response must not append the same switch twice.
        duplicate = connection.execute(
            """
            SELECT 1 FROM model_switches
            WHERE session_id = ? AND to_model = ? AND source = 'agent_response'
            """,
            (session_id, target),
        ).fetchone()
        if duplicate is not None:
            return False
    connection.execute(
        """
        INSERT INTO model_switches(
            session_id, from_model, to_model, source, occurred_at
        ) VALUES (?, ?, ?, ?, ?)
        """,
        (session_id, previous, target, source, occurred_at),
    )
    connection.execute(
        "UPDATE sessions SET model_switch_count = model_switch_count + 1 WHERE id = ?",
        (session_id,),
    )
    return True


def session_model(connection: sqlite3.Connection, session_id: str) -> str | None:
    row = connection.execute(
        "SELECT model FROM sessions WHERE id = ?", (session_id,)
    ).fetchone()
    return None if row is None else str(row["model"])


def next_tool_sequence(connection: sqlite3.Connection, session_id: str) -> int:
    row = connection.execute(
        "SELECT COALESCE(MAX(sequence), 0) AS last FROM tool_calls WHERE session_id = ?",
        (session_id,),
    ).fetchone()
    return int(row["last"]) + 1


_SESSION_COUNTERS = frozenset(
    {"turn_count", "prompt_count", "interrupt_count", "compaction_count"}
)


def increment_session_counter(
    connection: sqlite3.Connection, session_id: str, field: str
) -> None:
    """Add one to a session's count of turns, prompts, interrupts, or compactions."""

    if field not in _SESSION_COUNTERS:
        raise ValueError(f"unknown session counter: {field!r}")
    connection.execute(
        f"UPDATE sessions SET {field} = {field} + 1 WHERE id = ?", (session_id,)
    )


def _complete_tool_call(
    connection: sqlite3.Connection,
    row_id: int,
    succeeded: bool | None,
    duration_ms: int | None,
) -> None:
    """Record the outcome a later event reports for a call already stored."""

    if succeeded is None and duration_ms is None:
        return
    connection.execute(
        """
        UPDATE tool_calls
        SET succeeded = COALESCE(?, succeeded),
            duration_ms = COALESCE(?, duration_ms)
        WHERE id = ?
        """,
        (None if succeeded is None else int(succeeded), duration_ms, row_id),
    )


def record_tool_call(
    connection: sqlite3.Connection,
    session_id: str,
    *,
    tool_use_id: str,
    tool_name: str,
    tool_class: str,
    occurred_at: str,
    turn_id: str | None = None,
    locator: str | None = None,
    locator_hash: str | None = None,
    child_session_id: str | None = None,
    succeeded: bool | None = None,
    duration_ms: int | None = None,
    source: str = "harness_reported",
) -> bool:
    """Insert one tool call, count it once, and stop inserting at the row cap.

    A repeat of a call already recorded completes the outcome of the row that
    exists instead of inserting a second one, so the pre-tool and post-tool
    events of one call count once. Past the cap the session's count of calls
    still grows and ``activity_truncated`` marks the session.
    """

    if tool_class not in TOOL_CLASSES:
        tool_class = "other"
    existing = connection.execute(
        "SELECT id FROM tool_calls WHERE session_id = ? AND tool_use_id = ?",
        (session_id, tool_use_id),
    ).fetchone()
    if existing is not None:
        _complete_tool_call(connection, int(existing["id"]), succeeded, duration_ms)
        return False

    # The session row exists: every caller resolves or creates it before it
    # records a call, and the insert below has a foreign key to it.
    session = connection.execute(
        "SELECT compaction_count FROM sessions WHERE id = ?", (session_id,)
    ).fetchone()
    connection.execute(
        "UPDATE sessions SET tool_call_count = tool_call_count + 1 WHERE id = ?",
        (session_id,),
    )
    sequence = next_tool_sequence(connection, session_id)
    if sequence > MAX_TOOL_CALL_ROWS:
        connection.execute(
            "UPDATE sessions SET activity_truncated = 1 WHERE id = ?", (session_id,)
        )
        return False
    connection.execute(
        """
        INSERT INTO tool_calls(
            session_id, tool_use_id, turn_id, sequence, tool_name, tool_class,
            locator, locator_hash, child_session_id, succeeded, duration_ms,
            compaction_epoch, occurred_at, source
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            session_id,
            tool_use_id,
            turn_id,
            sequence,
            tool_name,
            tool_class,
            locator,
            locator_hash,
            child_session_id,
            None if succeeded is None else int(succeeded),
            duration_ms,
            int(session["compaction_count"]),
            occurred_at,
            source,
        ),
    )
    return True


def record_context_load(
    connection: sqlite3.Connection,
    session_id: str,
    *,
    kind: str,
    occurred_at: str,
    locator: str | None = None,
    content_hash: str | None = None,
    size_bytes: int | None = None,
    memory_type: str | None = None,
    load_reason: str | None = None,
    turn_id: str | None = None,
    related_session_id: str | None = None,
    source: str = "harness_reported",
) -> bool:
    """Insert one context load, and stop inserting at the row cap.

    The row carries the session's compaction count at the moment it was
    written, so a later compaction leaves every load it summarized marked as
    loaded before it. Past the cap ``activity_truncated`` marks the session and
    nothing is inserted.
    """

    if kind not in CONTEXT_LOAD_KINDS:
        raise ValueError(f"unknown context load kind: {kind!r}")
    # The session row exists: every caller resolves or creates it before it
    # records a load, and the insert below has a foreign key to it.
    stored = connection.execute(
        "SELECT COUNT(*) AS count FROM context_loads WHERE session_id = ?",
        (session_id,),
    ).fetchone()
    if int(stored["count"]) >= MAX_CONTEXT_LOAD_ROWS:
        connection.execute(
            "UPDATE sessions SET activity_truncated = 1 WHERE id = ?", (session_id,)
        )
        return False
    session = connection.execute(
        "SELECT compaction_count FROM sessions WHERE id = ?", (session_id,)
    ).fetchone()
    connection.execute(
        """
        INSERT INTO context_loads(
            session_id, kind, locator, content_hash, size_bytes, memory_type,
            load_reason, turn_id, related_session_id, compaction_epoch,
            occurred_at, source
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            session_id,
            kind,
            locator,
            content_hash,
            size_bytes,
            memory_type,
            load_reason,
            turn_id,
            related_session_id,
            int(session["compaction_count"]),
            occurred_at,
            source,
        ),
    )
    return True


def record_session_summary(
    connection: sqlite3.Connection, session_id: str, summary: str | None
) -> bool:
    """Name what one session was asked to do, without renaming it.

    A summary already on the row was written by the session itself or by an
    earlier event of the same launch. Either is closer to the work than a
    repeat of the launch, so neither is overwritten.
    """

    if summary is None:
        return False
    connection.execute(
        "UPDATE sessions SET summary = COALESCE(summary, ?) WHERE id = ?",
        (summary, session_id),
    )
    return True


def record_agent_response(
    connection: sqlite3.Connection,
    *,
    parent_session_id: str,
    tool_use_id: str,
    child_session_id: str | None,
    facets: Mapping[str, Any],
    occurred_at: str,
    summary: str | None = None,
) -> None:
    """Record what the parent's ``Agent`` tool response says about its child.

    The response is the only place a parent reports which model actually served
    a subagent and whether the subagent ran inside the call or beside it.
    ``summary`` is the purpose the launching call named, which the child keeps
    as the one sentence saying what it was started for.
    """

    duration_ms = facets.get("duration_ms")
    connection.execute(
        """
        UPDATE tool_calls
        SET child_session_id = COALESCE(child_session_id, ?),
            duration_ms = COALESCE(?, duration_ms)
        WHERE session_id = ? AND tool_use_id = ?
        """,
        (child_session_id, duration_ms, parent_session_id, tool_use_id),
    )
    if child_session_id is None:
        return

    record_session_summary(connection, child_session_id, summary)
    apply_session_facets(
        connection,
        child_session_id,
        facets,
        fields=("launch_mode", "duration_ms"),
    )

    current = session_model(connection, child_session_id)
    resolved = _text(facets.get("resolved_model"))
    if resolved is not None and resolved != current:
        if record_model_switch(
            connection,
            child_session_id,
            from_model=current,
            to_model=resolved,
            source="agent_response",
            occurred_at=occurred_at,
        ):
            connection.execute(
                """
                UPDATE sessions
                SET model = ?, model_source = 'agent_response'
                WHERE id = ?
                """,
                (resolved, child_session_id),
            )
            current = resolved

    for model in facets.get("models_used") or ():
        if model == current:
            continue
        record_model_switch(
            connection,
            child_session_id,
            from_model=current,
            to_model=model,
            source="agent_response",
            occurred_at=occurred_at,
        )
        # The response reports no timing, but every row must still leave the
        # model the row before it named rather than claim a shared predecessor.
        current = model


__all__ = [
    "CONTEXT_LOAD_KINDS",
    "FAILURE_EVENT_HARNESSES",
    "LAUNCH_MODES",
    "MAX_CONTEXT_LOAD_ROWS",
    "MAX_LOCATOR_CHARS",
    "MAX_TOOL_CALL_ROWS",
    "MODEL_SWITCH_SOURCES",
    "SESSION_SOURCES",
    "SNAPSHOT_TOOL_CLASSES",
    "SNAPSHOT_TOOL_NAMES",
    "TOOL_CLASSES",
    "agent_summary",
    "apply_session_facets",
    "classify_tool",
    "content_digest",
    "context_load",
    "context_load_kind",
    "extract_locator",
    "hook_activity",
    "increment_session_counter",
    "next_tool_sequence",
    "path_locator",
    "record_agent_response",
    "record_context_load",
    "record_model_switch",
    "record_session_summary",
    "record_tool_call",
    "session_facets",
    "session_model",
    "takes_snapshot",
    "tool_succeeded",
]
