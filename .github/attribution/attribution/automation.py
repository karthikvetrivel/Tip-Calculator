"""Durable native-hook capture and deferred Git-note publication."""

from __future__ import annotations

from collections.abc import Mapping
import json
import os
from pathlib import Path
import sqlite3
import subprocess
from typing import Any
import uuid

from . import activity, traces
from .capture import _base_commit, _content_hash, _snapshot, _utc_now, _worktree_lock
from .hosted_runtime import WORKFLOW_PATH
from .notes import _record_commit_locked
from .runtime import system_subprocess_environment
from .store import RepoPath, git_common_dir, git_dir, open_db, repository_root


SUPPORTED_HARNESSES = frozenset({"codex", "claude-code"})
KNOWN_EVENTS = frozenset(
    {
        "SessionStart",
        "UserPromptSubmit",
        "PreToolUse",
        "PostToolUse",
        "PostToolUseFailure",
        "PostModelSwitch",
        "InstructionsLoaded",
        "PostCompact",
        "SubagentStart",
        "SubagentStop",
        "Interrupt",
        "Stop",
        "SessionEnd",
    }
)
ACTIVE_CAPTURE_STATES = ("pending", "contaminated", "limited", "imported")
INSTALL_STATE_VERSION = 1
MAX_NATIVE_SNAPSHOT_FILES = 10_000
MAX_NATIVE_SNAPSHOT_BYTES = 32 * 1024 * 1024
# A session counts the events it saw itself, so each one names the counter it
# advances. SessionEnd ends no turn, and a subagent stop ends its child's turn
# rather than this session's.
_COUNTED_EVENTS = {
    "UserPromptSubmit": "prompt_count",
    "Interrupt": "interrupt_count",
    "Stop": "turn_count",
}


def _result(
    status: str,
    *,
    session_id: str | None = None,
    changed_files: list[str] | None = None,
    recorded_commits: list[str] | None = None,
    warnings: list[str] | None = None,
) -> dict[str, object]:
    """Return one hook outcome.

    ``pending`` opened a snapshot that its completion will close, ``captured``
    closed one, and ``recorded`` wrote a tool call, a count, a facet, or a
    context load that needed no snapshot at all. ``ignored`` changed no
    attribution, and ``warning`` names what was lost.
    """

    return {
        "status": status,
        "session_id": session_id,
        "changed_files": changed_files or [],
        "recorded_commits": recorded_commits or [],
        "warnings": warnings or [],
    }


def _warning(message: object) -> str:
    text = str(message).strip()
    return text[:1000] if text else "Native attribution hook failed safely."


def _required_text(payload: Mapping[str, Any], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Hook payload requires a nonempty {field}")
    if len(value) > 4096:
        raise ValueError(f"Hook payload {field} is too long")
    return value


def _optional_text(payload: Mapping[str, Any], field: str) -> str | None:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip() or len(value) > 4096:
        return None
    return value


def _effective_native_session(payload: Mapping[str, Any], event: str) -> str:
    native_session = _required_text(payload, "session_id")
    agent_id = _optional_text(payload, "agent_id") or _optional_text(payload, "subagent_id")
    if agent_id is not None:
        return f"{native_session}::{agent_id}"

    # A SessionStart names the session it starts, and Claude reports the main
    # session's own agent type on it. A real subagent announces itself with
    # SubagentStart, which carries an agent ID and took the branch above.
    marked_subagent = event != "SessionStart" and (
        bool(payload.get("is_subagent"))
        or _optional_text(payload, "agent_type") is not None
    )
    if marked_subagent:
        # Some native hook versions identify a subagent only by type while
        # retaining the parent's session ID. Tool ID keeps this conservative
        # identity stable across the matching Pre/Post pair without inheritance.
        discriminator = (
            _optional_text(payload, "tool_use_id")
            or _optional_text(payload, "turn_id")
            or event
        )
        return f"{native_session}::unidentified-subagent::{discriminator}"
    return native_session


def _direct_model(
    harness: str,
    event: str,
    payload: Mapping[str, Any],
) -> tuple[str, str] | None:
    field = "to_model" if event == "PostModelSwitch" else "model"
    model = _optional_text(payload, field)
    if model is None:
        return None
    if harness == "claude-code" and event in {"SessionStart", "PostModelSwitch"}:
        # Claude documents these as selected session settings, not proof of the
        # provider model that served every individual request.
        return model, "session_setting"
    return model, "hook"


def _branch_feature(repo: Path) -> tuple[str, str]:
    result = subprocess.run(
        ["git", "-C", str(repo), "symbolic-ref", "--quiet", "--short", "HEAD"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=system_subprocess_environment(),
        check=False,
    )
    if result.returncode == 0:
        branch = result.stdout.decode("utf-8", errors="replace").strip()
        if branch:
            return branch, "branch"
    return "Detached HEAD", "fallback"


def _latest_head_action_is_direct_commit(repo: Path) -> bool:
    """Return whether HEAD's reflog proves a direct `git commit` action.

    Cherry-pick, revert, rebase, merge, pull, reset, and `git am` import or
    restore history. Their file transitions must not be credited to the model
    merely because they occurred inside a shell-tool interval. Missing reflog
    evidence is treated conservatively as an import.
    """

    result = subprocess.run(
        ["git", "-C", str(repo), "reflog", "show", "-1", "--format=%gs", "HEAD"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=system_subprocess_environment(),
        check=False,
    )
    if result.returncode != 0:
        return False
    action = result.stdout.decode("utf-8", errors="replace").strip()
    return action.startswith(("commit: ", "commit (initial): ", "commit (amend): "))


def _head_changed_without_direct_commit(repo: Path, base_commit: str | None) -> bool:
    current = _base_commit(repo)
    return current != base_commit and not _latest_head_action_is_direct_commit(repo)


def _session_context(
    connection: sqlite3.Connection,
    repo: Path,
    worktree_id: str,
    harness: str,
    native_session_id: str,
    event: str,
    payload: Mapping[str, Any],
) -> tuple[str, str, str, str]:
    existing = connection.execute(
        """
        SELECT model, model_source, feature, feature_source
        FROM hook_sessions
        WHERE worktree_id = ? AND harness = ? AND native_session_id = ?
        """,
        (worktree_id, harness, native_session_id),
    ).fetchone()
    direct_model = _direct_model(harness, event, payload)
    now = _utc_now()

    if existing is None:
        feature, feature_source = _branch_feature(repo)
        if direct_model is None:
            model, model_source = "unknown", "unknown"
        else:
            model, model_source = direct_model
        connection.execute(
            """
            INSERT INTO hook_sessions(
                worktree_id, harness, native_session_id, model, model_source,
                feature, feature_source, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                worktree_id,
                harness,
                native_session_id,
                model,
                model_source,
                feature,
                feature_source,
                now,
            ),
        )
        return model, model_source, feature, feature_source

    if direct_model is not None:
        model, model_source = direct_model
    elif harness == "claude-code" and existing["model_source"] == "session_setting":
        model = existing["model"]
        model_source = existing["model_source"]
    else:
        # Codex documents model on every event. A missing field is missing
        # evidence, not permission to reuse a potentially stale prior value.
        model, model_source = "unknown", "unknown"
    feature = existing["feature"]
    feature_source = existing["feature_source"]
    if event in {"SessionStart", "PreToolUse", "PostToolUse", "PostToolUseFailure"}:
        feature, feature_source = _branch_feature(repo)
    connection.execute(
        """
        UPDATE hook_sessions
        SET model = ?, model_source = ?, feature = ?, feature_source = ?, updated_at = ?
        WHERE worktree_id = ? AND harness = ? AND native_session_id = ?
        """,
        (
            model,
            model_source,
            feature,
            feature_source,
            now,
            worktree_id,
            harness,
            native_session_id,
        ),
    )
    return model, model_source, feature, feature_source


_FACET_COLUMNS = ("agent_type", "session_source", "permission_mode", "effort_level")


def _native_parent(native_session_id: str) -> str | None:
    """Return the parent native session of one subagent key, or None."""

    parent, separator, _ = native_session_id.partition("::")
    return parent if separator and parent else None


def _tool_class(harness: str, payload: Mapping[str, Any]) -> str:
    return activity.classify_tool(harness, _optional_text(payload, "tool_name"))


def _is_agent_call(harness: str, payload: Mapping[str, Any]) -> bool:
    """Return whether one tool event describes a subagent launch."""

    return _tool_class(harness, payload) == "agent"


def _takes_snapshot(harness: str, payload: Mapping[str, Any]) -> bool:
    """Return whether one tool event needs the worktree snapshot pair."""

    return activity.takes_snapshot(harness, _optional_text(payload, "tool_name"))


def _retain_facets(
    connection: sqlite3.Connection,
    worktree_id: str,
    harness: str,
    native_session_id: str,
    facets: Mapping[str, Any],
) -> None:
    """Keep session facets on the hook session until a ledger session exists.

    SessionStart arrives before any tool event, so the ledger session it
    describes is usually not created yet. Retaining the facets here also
    survives the branch change that commonly follows SessionStart.
    """

    assignments: list[str] = []
    values: list[Any] = []
    for column in _FACET_COLUMNS:
        if column not in facets:
            continue
        if column in {"permission_mode", "effort_level"}:
            assignments.append(f"{column} = ?")
        else:
            assignments.append(f"{column} = COALESCE({column}, ?)")
        values.append(facets[column])
    if not assignments:
        return
    connection.execute(
        f"UPDATE hook_sessions SET {', '.join(assignments)} "
        "WHERE worktree_id = ? AND harness = ? AND native_session_id = ?",
        (*values, worktree_id, harness, native_session_id),
    )


def _apply_retained_facets(
    connection: sqlite3.Connection,
    session_id: str,
    worktree_id: str,
    harness: str,
    native_session_id: str,
) -> None:
    retained = connection.execute(
        f"""
        SELECT {", ".join(_FACET_COLUMNS)}
        FROM hook_sessions
        WHERE worktree_id = ? AND harness = ? AND native_session_id = ?
        """,
        (worktree_id, harness, native_session_id),
    ).fetchone()
    if retained is None:
        return
    activity.apply_session_facets(
        connection,
        session_id,
        {
            column: retained[column]
            for column in _FACET_COLUMNS
            if retained[column] is not None
        },
        fields=_FACET_COLUMNS,
    )


def _ledger_session(
    connection: sqlite3.Connection,
    *,
    worktree_id: str,
    harness: str,
    native_session_id: str,
    model: str,
    model_source: str,
    feature: str,
    feature_source: str,
    base_commit: str | None,
    started_at: str,
) -> str:
    existing = connection.execute(
        """
        SELECT id
        FROM sessions
        WHERE worktree_id = ? AND harness = ? AND native_session_id = ?
          AND model = ? AND feature = ?
        ORDER BY rowid
        LIMIT 1
        """,
        (worktree_id, harness, native_session_id, model, feature),
    ).fetchone()
    if existing is None and model != "unknown":
        placeholder = connection.execute(
            "SELECT id, model FROM sessions WHERE worktree_id = ? AND harness = ? "
            "AND native_session_id = ? AND feature = ? ORDER BY rowid DESC LIMIT 1",
            (worktree_id, harness, native_session_id, feature),
        ).fetchone()
        if placeholder is not None and placeholder["model"] == "unknown":
            connection.execute(
                "UPDATE sessions SET model = ?, model_source = ? WHERE id = ?",
                (model, model_source, placeholder["id"]),
            )
            existing = placeholder
    if existing is None and model == "unknown":
        # Missing model metadata is not a model switch. A completed Agent
        # response may already have named this native agent's model.
        existing = connection.execute(
            "SELECT id FROM sessions WHERE worktree_id = ? AND harness = ? "
            "AND native_session_id = ? AND feature = ? ORDER BY rowid DESC LIMIT 1",
            (worktree_id, harness, native_session_id, feature),
        ).fetchone()
    if existing is not None:
        connection.execute(
            """
            UPDATE sessions
            SET harness_id = COALESCE(harness_id, ?),
                model_source = COALESCE(model_source, ?),
                harness_source = COALESCE(harness_source, 'native_hook'),
                integration_mode = COALESCE(integration_mode, 'native_hook')
            WHERE id = ?
            """,
            (
                harness,
                "unknown" if model == "unknown" else "native_hook",
                existing["id"],
            ),
        )
        _apply_retained_facets(
            connection, existing["id"], worktree_id, harness, native_session_id
        )
        return existing["id"]

    session_id = str(uuid.uuid4())
    connection.execute(
        """
        INSERT INTO sessions(
            id, worktree_id, native_session_id, feature, feature_source,
            model, harness, label_source, cost_usd, cost_source, started_at,
            base_commit, harness_id, model_source, harness_source,
            integration_mode
        ) VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, ?, ?, ?, ?,
            'native_hook', 'native_hook'
        )
        """,
        (
            session_id,
            worktree_id,
            native_session_id,
            feature,
            feature_source,
            model,
            harness,
            model_source,
            started_at,
            base_commit,
            harness,
            "unknown" if model == "unknown" else "native_hook",
        ),
    )
    _apply_retained_facets(
        connection, session_id, worktree_id, harness, native_session_id
    )
    parent_native = _native_parent(native_session_id)
    if parent_native is not None:
        parent = connection.execute(
            """
            SELECT id
            FROM sessions
            WHERE worktree_id = ? AND harness = ? AND native_session_id = ?
            ORDER BY rowid DESC
            LIMIT 1
            """,
            (worktree_id, harness, parent_native),
        ).fetchone()
        if parent is not None:
            connection.execute(
                "UPDATE sessions SET parent_session_id = ? WHERE id = ?",
                (parent["id"], session_id),
            )
    return session_id


def _existing_ledger_session(
    connection: sqlite3.Connection,
    worktree_id: str,
    harness: str,
    native_session_id: str,
) -> str | None:
    row = connection.execute(
        """
        SELECT id
        FROM sessions
        WHERE worktree_id = ? AND harness = ? AND native_session_id = ?
        ORDER BY rowid DESC
        LIMIT 1
        """,
        (worktree_id, harness, native_session_id),
    ).fetchone()
    return None if row is None else str(row["id"])


def _linked_ledger_session(
    connection: sqlite3.Connection,
    repo: Path,
    worktree_id: str,
    harness: str,
    native_session_id: str,
    base_commit: str | None,
    started_at: str,
) -> str:
    """Return the ledger session one agent link needs, creating it if absent.

    The named session's own model and feature state is read but never updated.
    A subagent event describes the child, so treating it as parent evidence
    could relabel work that the parent already recorded.
    """

    existing = _existing_ledger_session(
        connection, worktree_id, harness, native_session_id
    )
    if existing is not None:
        return existing
    state = connection.execute(
        """
        SELECT model, model_source, feature, feature_source
        FROM hook_sessions
        WHERE worktree_id = ? AND harness = ? AND native_session_id = ?
        """,
        (worktree_id, harness, native_session_id),
    ).fetchone()
    if state is None:
        feature, feature_source = _branch_feature(repo)
        model, model_source = "unknown", "unknown"
    else:
        model, model_source = str(state["model"]), str(state["model_source"])
        feature, feature_source = str(state["feature"]), str(state["feature_source"])
    return _ledger_session(
        connection,
        worktree_id=worktree_id,
        harness=harness,
        native_session_id=native_session_id,
        model=model,
        model_source=model_source,
        feature=feature,
        feature_source=feature_source,
        base_commit=base_commit,
        started_at=started_at,
    )


def _capture_identity(
    worktree_id: str,
    harness: str,
    native_session_id: str,
    tool_use_id: str,
) -> str:
    material = "\0".join((worktree_id, harness, native_session_id, tool_use_id))
    return str(uuid.uuid5(uuid.NAMESPACE_URL, material))


def _tool_session(
    connection: sqlite3.Connection,
    repo: Path,
    worktree_id: str,
    harness: str,
    native_session_id: str,
) -> str:
    """Return the ledger session one event belongs to, creating it if absent.

    The existing session is read first, so an event on the fast path costs one
    small query instead of the Git calls that opening a session needs.
    """

    existing = _existing_ledger_session(
        connection, worktree_id, harness, native_session_id
    )
    if existing is not None:
        return existing
    return _linked_ledger_session(
        connection,
        repo,
        worktree_id,
        harness,
        native_session_id,
        _base_commit(repo),
        _utc_now(),
    )


def _record_tool_activity(
    connection: sqlite3.Connection,
    repo: Path,
    payload: Mapping[str, Any],
    harness: str,
    session_id: str,
    tool_use_id: str,
    event: str,
    occurred_at: str,
) -> str | None:
    """Record one completed tool call on the session that made it.

    Only the fields the class table names are read, and the response is read
    only for the outcome it states, so no tool response body and no other tool
    input value reaches this row. The trace event the caller writes beside it
    is where the input and the response are kept. Returns the locator.
    """

    tool_name = _required_text(payload, "tool_name")
    tool_class = _tool_class(harness, payload)
    locator, locator_hash = activity.extract_locator(
        harness, tool_name, tool_class, payload.get("tool_input"), repo
    )
    facets = activity.session_facets(harness, payload)
    activity.apply_session_facets(
        connection,
        session_id,
        facets,
        fields=("permission_mode", "effort_level"),
    )
    activity.record_tool_call(
        connection,
        session_id,
        tool_use_id=tool_use_id,
        tool_name=tool_name,
        tool_class=tool_class,
        occurred_at=occurred_at,
        turn_id=_optional_text(payload, "turn_id"),
        locator=locator,
        locator_hash=locator_hash,
        succeeded=activity.tool_succeeded(
            harness,
            event,
            tool_class=tool_class,
            tool_response=payload.get("tool_response"),
        ),
        duration_ms=facets.get("duration_ms"),
    )
    return locator


def _record_context_load(
    connection: sqlite3.Connection,
    repo: Path,
    payload: Mapping[str, Any],
    harness: str,
    session_id: str,
    event: str,
    occurred_at: str,
    *,
    related_session_id: str | None = None,
) -> bool:
    """Record what one event loaded into a session, without its text.

    Only the fields the event mapping names are read: an instruction path, the
    two labels that describe it, and the size and hash of its text. The text
    itself is kept by the trace event the caller writes beside this row.
    """

    loaded = activity.context_load(harness, payload, repo, event=event)
    if loaded is None:
        return False
    return activity.record_context_load(
        connection,
        session_id,
        kind=str(loaded["kind"]),
        occurred_at=occurred_at,
        locator=loaded.get("locator"),
        content_hash=loaded.get("content_hash"),
        size_bytes=loaded.get("size_bytes"),
        memory_type=loaded.get("memory_type"),
        load_reason=loaded.get("load_reason"),
        turn_id=(
            _optional_text(payload, "prompt_id") or _optional_text(payload, "turn_id")
        ),
        related_session_id=related_session_id,
    )


def _record_trace(
    connection: sqlite3.Connection,
    repo: Path,
    session_id: str,
    kind: str,
    fields: Mapping[str, Any],
    *,
    occurred_at: str,
    tool_use_id: str | None = None,
    agent_id: str | None = None,
    turn_id: str | None = None,
) -> None:
    """Keep the text of one event unless the repository opted out."""

    if not traces.traces_enabled(repo):
        return
    traces.record_trace_event(
        connection, session_id, kind, fields, occurred_at=occurred_at,
        tool_use_id=tool_use_id, agent_id=agent_id, turn_id=turn_id,
    )


def _skipped_warning(skipped: Mapping[str, str]) -> list[str]:
    counts: dict[str, int] = {}
    for reason in skipped.values():
        if reason.startswith("ignored_preexisting"):
            continue
        counts[reason] = counts.get(reason, 0) + 1
    if not counts:
        return []
    detail = ", ".join(f"{count} {reason}" for reason, count in sorted(counts.items()))
    return [f"Skipped unsupported snapshot files ({detail}); no attribution was inferred for them."]


def _active_count(connection: sqlite3.Connection, worktree_id: str) -> int:
    placeholders = ",".join("?" for _ in ACTIVE_CAPTURE_STATES)
    row = connection.execute(
        f"SELECT COUNT(*) AS count FROM hook_captures WHERE worktree_id = ? AND status IN ({placeholders})",
        (worktree_id, *ACTIVE_CAPTURE_STATES),
    ).fetchone()
    count = int(row["count"])
    has_hook_units = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'hook_units'"
    ).fetchone()
    if has_hook_units is not None:
        unit_row = connection.execute(
            "SELECT COUNT(*) AS count FROM hook_units WHERE worktree_id = ? AND active = 1",
            (worktree_id,),
        ).fetchone()
        count += int(unit_row["count"])
    return count


def _handle_fast_pre(
    repo: Path,
    payload: Mapping[str, Any],
    harness: str,
    native_session_id: str,
) -> dict[str, object]:
    """Handle a pre-tool event for a tool that needs no worktree snapshot.

    Only an ``Agent`` call needs a row before it finishes: the subagent tree
    hangs on the launching call, and the child reports its own events while the
    parent call is still open. Recording it here keeps that call off the
    capture queue, so the subagent's own captures no longer overlap it.
    """

    if not _is_agent_call(harness, payload):
        # Every other class is recorded from its completion alone.
        return _result("ignored")
    tool_use_id = _required_text(payload, "tool_use_id")
    worktree_id = str(git_dir(repo))
    with _worktree_lock(repo, blocking=True):
        connection = open_db(repo)
        try:
            session_id = _tool_session(
                connection, repo, worktree_id, harness, native_session_id
            )
            _record_tool_activity(
                connection,
                repo,
                payload,
                harness,
                session_id,
                tool_use_id,
                "PreToolUse",
                _utc_now(),
            )
            connection.commit()
        finally:
            connection.close()
    # The launch is on the ledger, and it opened no snapshot to close.
    return _result("recorded", session_id=session_id)


def _handle_pre(
    repo: Path,
    payload: Mapping[str, Any],
    harness: str,
    native_session_id: str,
) -> dict[str, object]:
    tool_use_id = _required_text(payload, "tool_use_id")
    worktree_id = str(git_dir(repo))
    capture_id = _capture_identity(worktree_id, harness, native_session_id, tool_use_id)

    with _worktree_lock(repo, blocking=True):
        connection = open_db(repo)
        try:
            existing = connection.execute(
                """
                SELECT ledger_session_id
                FROM hook_captures
                WHERE worktree_id = ? AND harness = ?
                  AND native_session_id = ? AND tool_use_id = ?
                """,
                (worktree_id, harness, native_session_id, tool_use_id),
            ).fetchone()
            if existing is not None:
                return _result("ignored", session_id=existing["ledger_session_id"])

            before, skipped = _snapshot(
                repo,
                max_files=MAX_NATIVE_SNAPSHOT_FILES,
                max_total_bytes=MAX_NATIVE_SNAPSHOT_BYTES,
                include_ignored=True,
            )
            started_at = _utc_now()
            base_commit = _base_commit(repo)
            model, model_source, feature, feature_source = _session_context(
                connection,
                repo,
                worktree_id,
                harness,
                native_session_id,
                "PreToolUse",
                payload,
            )
            ledger_session_id = _ledger_session(
                connection,
                worktree_id=worktree_id,
                harness=harness,
                native_session_id=native_session_id,
                model=model,
                model_source=model_source,
                feature=feature,
                feature_source=feature_source,
                base_commit=base_commit,
                started_at=started_at,
            )
            limited = any(reason.startswith("snapshot_") for reason in skipped.values())
            overlap = _active_count(connection, worktree_id) > 0
            if overlap:
                placeholders = ",".join("?" for _ in ACTIVE_CAPTURE_STATES)
                connection.execute(
                    f"UPDATE hook_captures SET status = 'contaminated' WHERE worktree_id = ? AND status IN ({placeholders})",
                    (worktree_id, *ACTIVE_CAPTURE_STATES),
                )
                active_units = connection.execute(
                    """
                    SELECT unit_key, attribution_session_id
                    FROM hook_units
                    WHERE worktree_id = ? AND active = 1
                    """,
                    (worktree_id,),
                ).fetchall()
                if active_units:
                    # The public lifecycle receiver and the installed tool hook
                    # observed overlapping intervals. Close the public units
                    # without evidence so neither path can claim the shared edit.
                    from .hook_capture import _close_without_evidence

                    _close_without_evidence(
                        connection,
                        active_units,
                        started_at,
                        "automatic_overlap",
                    )
            connection.execute(
                """
                INSERT INTO hook_captures(
                    id, worktree_id, harness, native_session_id, tool_use_id,
                    turn_id, ledger_session_id, base_commit, started_at, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    capture_id,
                    worktree_id,
                    harness,
                    native_session_id,
                    tool_use_id,
                    _optional_text(payload, "turn_id"),
                    ledger_session_id,
                    base_commit,
                    started_at,
                    "contaminated" if overlap else ("limited" if limited else "pending"),
                ),
            )
            for path in ([] if limited else sorted(set(before) | set(skipped))):
                content = before.get(path)
                connection.execute(
                    """
                    INSERT INTO hook_capture_files(
                        capture_id, path, before_content, before_hash, skip_reason
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        capture_id,
                        path,
                        content,
                        _content_hash(content),
                        skipped.get(path),
                    ),
                )
            connection.commit()
        finally:
            connection.close()

    warnings = _skipped_warning(skipped)
    if overlap:
        warnings.append(
            "Overlapping native tool captures were detected; their file deltas will remain unknown."
        )
    elif limited:
        warnings.append(
            "The bounded native snapshot limit was reached; this tool's file deltas will remain unknown."
        )
    if model == "unknown":
        warnings.append("The native hook did not provide a model; it was recorded as unknown.")
    return _result("pending", session_id=ledger_session_id, warnings=warnings)


def _before_rows(
    connection: sqlite3.Connection,
    capture_id: str,
) -> dict[str, sqlite3.Row]:
    return {
        row["path"]: row
        for row in connection.execute(
            "SELECT * FROM hook_capture_files WHERE capture_id = ? ORDER BY path",
            (capture_id,),
        )
    }


def _drain_pending(repo: Path, worktree_id: str) -> tuple[list[str], list[str]]:
    recorded: list[str] = []
    warnings: list[str] = []
    while True:
        with _worktree_lock(repo, blocking=True):
            connection = open_db(repo)
            try:
                if _active_count(connection, worktree_id):
                    return recorded, warnings
                queued = connection.execute(
                    """
                    SELECT rowid, commit_sha
                    FROM pending_commits
                    WHERE worktree_id = ?
                    ORDER BY queued_at, rowid
                    LIMIT 1
                    """,
                    (worktree_id,),
                ).fetchone()
            finally:
                connection.close()
            if queued is None:
                return recorded, warnings

            commit_sha = queued["commit_sha"]
            try:
                # Keep the worktree serialized from the settled-capture check
                # through immutable note publication. A new Pre hook cannot
                # otherwise appear between those two decisions.
                _record_commit_locked(repo, commit_sha)
            except (OSError, ValueError, sqlite3.Error, subprocess.SubprocessError) as exc:
                warnings.append(
                    f"Deferred commit {commit_sha[:12]} was not recorded: {_warning(exc)}"
                )
                return recorded, warnings

            connection = open_db(repo)
            try:
                connection.execute(
                    "DELETE FROM pending_commits WHERE worktree_id = ? AND commit_sha = ?",
                    (worktree_id, commit_sha),
                )
                connection.commit()
            finally:
                connection.close()
        recorded.append(commit_sha)


def _handle_post(
    repo: Path,
    payload: Mapping[str, Any],
    harness: str,
    native_session_id: str,
    event: str,
) -> dict[str, object]:
    tool_use_id = _required_text(payload, "tool_use_id")
    worktree_id = str(git_dir(repo))
    changed_files: list[str] = []
    warnings: list[str] = []
    duplicate = False
    ledger_session_id: str | None = None

    with _worktree_lock(repo, blocking=True):
        connection = open_db(repo)
        try:
            capture = connection.execute(
                """
                SELECT *
                FROM hook_captures
                WHERE worktree_id = ? AND harness = ?
                  AND native_session_id = ? AND tool_use_id = ?
                """,
                (worktree_id, harness, native_session_id, tool_use_id),
            ).fetchone()
            if capture is None:
                # A tool on the fast path takes no snapshot, so its completion
                # arrives without a pre-tool capture. The call itself is still
                # evidence, and recording it changes no file attribution.
                activity_session_id = _tool_session(
                    connection, repo, worktree_id, harness, native_session_id
                )
            else:
                # The call belongs to the session that was running when it
                # began. A model change later in this handler moves the file
                # delta to a new session; it does not move the call that
                # produced it.
                activity_session_id = str(capture["ledger_session_id"])
            # The row is written before the Agent branch below, which updates
            # that row with the child session and the duration the response
            # reports. An Agent call takes no snapshot, so its completion is
            # commonly the first event that names it.
            occurred_at = _utc_now()
            locator = _record_tool_activity(
                connection,
                repo,
                payload,
                harness,
                activity_session_id,
                tool_use_id,
                event,
                occurred_at,
            )
            tool_class = _tool_class(harness, payload)
            _record_trace(
                connection, repo, activity_session_id, "tool_call",
                {
                    "tool_name": _required_text(payload, "tool_name"),
                    "tool_class": tool_class, "summary": locator,
                    "input": payload.get("tool_input"),
                },
                occurred_at=occurred_at, tool_use_id=tool_use_id,
                agent_id=_optional_text(payload, "agent_id"),
                turn_id=_optional_text(payload, "turn_id"),
            )
            _record_trace(
                connection, repo, activity_session_id, "tool_result",
                {
                    "succeeded": activity.tool_succeeded(
                        harness, event, tool_class=tool_class,
                        tool_response=payload.get("tool_response"),
                    ),
                    "output": payload.get("tool_response"),
                },
                occurred_at=occurred_at, tool_use_id=tool_use_id,
                agent_id=_optional_text(payload, "agent_id"),
                turn_id=_optional_text(payload, "turn_id"),
            )
            if _is_agent_call(harness, payload):
                _record_finished_agent_call(
                    connection,
                    repo,
                    payload,
                    harness,
                    worktree_id,
                    native_session_id,
                    tool_use_id,
                    capture,
                )
            connection.commit()
            if capture is None:
                if not _takes_snapshot(harness, payload):
                    # The call is on the ledger, and this tool needed no
                    # snapshot pair, so no file evidence is missing.
                    return _result("recorded", session_id=activity_session_id)
                return _result(
                    "warning",
                    session_id=activity_session_id,
                    warnings=["Post-tool hook had no matching durable pre-tool snapshot; no attribution was inferred."],
                )
            ledger_session_id = capture["ledger_session_id"]
            capture_status = capture["status"]
            if (
                capture_status == "pending"
                and _head_changed_without_direct_commit(repo, capture["base_commit"])
            ):
                capture_status = "imported"
                connection.execute(
                    "UPDATE hook_captures SET status = 'imported' WHERE id = ?",
                    (capture["id"],),
                )
            if capture_status not in ACTIVE_CAPTURE_STATES:
                duplicate = True
            else:
                if capture_status in {"contaminated", "limited", "imported"}:
                    after: dict[str, bytes | None] = {}
                    after_skips: dict[str, str] = {}
                else:
                    after, after_skips = _snapshot(
                        repo,
                        max_files=MAX_NATIVE_SNAPSHOT_FILES,
                        max_total_bytes=MAX_NATIVE_SNAPSHOT_BYTES,
                        include_ignored=True,
                    )
                before_rows = _before_rows(connection, capture["id"])
                if harness == "claude-code" and _direct_model(harness, event, payload) is None:
                    # An explicit model on Claude's matching Pre event is valid
                    # evidence for that tool interval even though Post commonly
                    # omits the field. This does not inherit a parent model: the
                    # selected segment already uses the effective subagent key.
                    captured_session = connection.execute(
                        """
                        SELECT model, label_source, feature, feature_source
                        FROM sessions WHERE id = ?
                        """,
                        (ledger_session_id,),
                    ).fetchone()
                    if captured_session is None:
                        raise ValueError("Durable native capture references a missing session")
                    model = captured_session["model"]
                    model_source = captured_session["label_source"]
                    feature, feature_source = _branch_feature(repo)
                    connection.execute(
                        """
                        UPDATE hook_sessions
                        SET feature = ?, feature_source = ?, updated_at = ?
                        WHERE worktree_id = ? AND harness = ? AND native_session_id = ?
                        """,
                        (
                            feature,
                            feature_source,
                            _utc_now(),
                            worktree_id,
                            harness,
                            native_session_id,
                        ),
                    )
                else:
                    model, model_source, feature, feature_source = _session_context(
                        connection,
                        repo,
                        worktree_id,
                        harness,
                        native_session_id,
                        event,
                        payload,
                    )
                selected_session = _ledger_session(
                    connection,
                    worktree_id=worktree_id,
                    harness=harness,
                    native_session_id=native_session_id,
                    model=model,
                    model_source=model_source,
                    feature=feature,
                    feature_source=feature_source,
                    base_commit=capture["base_commit"],
                    started_at=capture["started_at"],
                )
                if selected_session != ledger_session_id:
                    connection.execute(
                        "UPDATE sessions SET ended_at = ? WHERE id = ?",
                        (_utc_now(), ledger_session_id),
                    )
                    ledger_session_id = selected_session
                    connection.execute(
                        "UPDATE hook_captures SET ledger_session_id = ? WHERE id = ?",
                        (ledger_session_id, capture["id"]),
                    )

                after_limited = any(
                    reason.startswith("snapshot_") for reason in after_skips.values()
                )
                if capture_status == "contaminated":
                    warnings.append(
                        "This capture overlapped another native tool; its file deltas were left unknown."
                    )
                elif capture_status == "imported":
                    warnings.append(
                        "This tool imported committed history; its file deltas were left unknown."
                    )
                elif capture_status == "limited" or after_limited:
                    warnings.append(
                        "The bounded native snapshot limit was reached; this tool's file deltas were left unknown."
                    )
                else:
                    before_skipped = {
                        path: row["skip_reason"]
                        for path, row in before_rows.items()
                        if row["skip_reason"] is not None
                    }
                    ignored_directories = [
                        path
                        for path, reason in before_skipped.items()
                        if reason == "ignored_preexisting_directory"
                    ]
                    ignored_directories.extend(
                        path
                        for path, reason in after_skips.items()
                        if reason == "ignored_preexisting_directory"
                    )
                    for path in sorted(set(before_rows) | set(after) | set(after_skips)):
                        if (
                            path in before_skipped
                            or path in after_skips
                            or any(path.startswith(directory) for directory in ignored_directories)
                        ):
                            continue
                        before_row = before_rows.get(path)
                        before_content = (
                            bytes(before_row["before_content"])
                            if before_row is not None and before_row["before_content"] is not None
                            else None
                        )
                        after_content = after.get(path)
                        if before_content == after_content:
                            continue
                        connection.execute(
                            """
                            INSERT INTO edits(
                                session_id, path, before_content, after_content,
                                before_hash, after_hash, base_commit, tool_use_id
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                ledger_session_id,
                                path,
                                before_content,
                                after_content,
                                _content_hash(before_content),
                                _content_hash(after_content),
                                capture["base_commit"],
                                tool_use_id,
                            ),
                        )
                        changed_files.append(path)
                    warnings.extend(_skipped_warning(after_skips))

                completed_at = _utc_now()
                connection.execute(
                    "UPDATE sessions SET ended_at = ? WHERE id = ?",
                    (completed_at, ledger_session_id),
                )
                connection.execute(
                    "UPDATE hook_captures SET status = ? WHERE id = ?",
                    (
                        "completed_contaminated"
                        if capture_status == "contaminated"
                        else (
                            "completed_imported"
                            if capture_status == "imported"
                            else (
                                "completed_limited"
                                if capture_status == "limited" or after_limited
                                else "completed"
                            )
                        ),
                        capture["id"],
                    ),
                )
                connection.execute(
                    "DELETE FROM hook_capture_files WHERE capture_id = ?",
                    (capture["id"],),
                )
                connection.commit()
        finally:
            connection.close()

    recorded_commits, drain_warnings = _drain_pending(repo, worktree_id)
    warnings.extend(drain_warnings)
    if duplicate:
        status = "captured" if recorded_commits else "ignored"
    else:
        status = "captured"
    if warnings and status == "ignored":
        status = "warning"
    return _result(
        status,
        session_id=ledger_session_id,
        changed_files=changed_files,
        recorded_commits=recorded_commits,
        warnings=warnings,
    )


def _record_finished_agent_call(
    connection: sqlite3.Connection,
    repo: Path,
    payload: Mapping[str, Any],
    harness: str,
    worktree_id: str,
    native_session_id: str,
    tool_use_id: str,
    capture: sqlite3.Row | None,
) -> None:
    """Record a finished ``Agent`` call whether or not its capture survived.

    The response is metadata about the subagent that ran, so it needs no
    snapshot. A hook installed mid-call, a reached snapshot limit, or a capture
    that already closed must not drop the tree the response describes.
    """

    if capture is not None:
        base_commit = capture["base_commit"]
        started_at = capture["started_at"]
        parent_session_id = str(capture["ledger_session_id"])
    elif activity.session_facets(harness, payload.get("tool_response")):
        base_commit = _base_commit(repo)
        started_at = _utc_now()
        parent_session_id = _linked_ledger_session(
            connection,
            repo,
            worktree_id,
            harness,
            native_session_id,
            base_commit,
            started_at,
        )
    else:
        # Neither a capture nor a response. Opening a session for that would
        # leave a row behind that describes nothing.
        return
    activity.apply_session_facets(
        connection,
        parent_session_id,
        activity.session_facets(harness, payload),
        fields=("permission_mode", "effort_level"),
    )
    _record_agent_response(
        connection,
        repo,
        payload,
        harness,
        worktree_id,
        native_session_id,
        parent_session_id,
        tool_use_id,
        base_commit,
        started_at,
    )


def _record_agent_response(
    connection: sqlite3.Connection,
    repo: Path,
    payload: Mapping[str, Any],
    harness: str,
    worktree_id: str,
    native_session_id: str,
    parent_session_id: str,
    tool_use_id: str,
    base_commit: str | None,
    started_at: str,
) -> None:
    """Record what a finished Agent call reports about the subagent it ran."""

    facets = activity.session_facets(harness, payload.get("tool_response"))
    agent_id = facets.get("agent_id")
    child_session_id = None
    if agent_id is not None:
        child_session_id = _linked_ledger_session(
            connection,
            repo,
            worktree_id,
            harness,
            f"{native_session_id}::{agent_id}",
            base_commit,
            started_at,
        )
        activity.apply_session_facets(
            connection,
            child_session_id,
            {"session_source": "subagent"},
            fields=("session_source",),
        )
    activity.record_agent_response(
        connection,
        parent_session_id=parent_session_id,
        tool_use_id=tool_use_id,
        child_session_id=child_session_id,
        facets=facets,
        occurred_at=_utc_now(),
        # The launching call named the purpose; the response names the child
        # that was given it, so the two meet here.
        summary=activity.agent_summary(payload.get("tool_input")),
    )


def _record_model_switch(
    connection: sqlite3.Connection,
    worktree_id: str,
    harness: str,
    native_session_id: str,
    payload: Mapping[str, Any],
) -> None:
    """Record a model change against the session that was running.

    The row does not relabel that session. This receiver already segments a
    model change into a new ledger session, so rewriting the model here would
    credit the new model with edits the previous one produced.
    """

    to_model = _optional_text(payload, "to_model") or _optional_text(payload, "model")
    if to_model is None:
        return
    active = connection.execute(
        """
        SELECT id, model
        FROM sessions
        WHERE worktree_id = ? AND harness = ? AND native_session_id = ?
        ORDER BY rowid DESC
        LIMIT 1
        """,
        (worktree_id, harness, native_session_id),
    ).fetchone()
    if active is None:
        return
    activity.record_model_switch(
        connection,
        str(active["id"]),
        from_model=_optional_text(payload, "from_model") or str(active["model"]),
        to_model=to_model,
        source=_optional_text(payload, "source") or "unknown",
        occurred_at=_utc_now(),
    )


def _handle_metadata(
    repo: Path,
    payload: Mapping[str, Any],
    harness: str,
    native_session_id: str,
    event: str,
) -> dict[str, object]:
    worktree_id = str(git_dir(repo))
    with _worktree_lock(repo, blocking=True):
        connection = open_db(repo)
        try:
            _session_context(
                connection,
                repo,
                worktree_id,
                harness,
                native_session_id,
                event,
                payload,
            )
            if event == "SessionStart":
                _retain_facets(
                    connection,
                    worktree_id,
                    harness,
                    native_session_id,
                    activity.session_facets(harness, payload),
                )
                existing = _existing_ledger_session(
                    connection, worktree_id, harness, native_session_id
                )
                if existing is not None:
                    _apply_retained_facets(
                        connection, existing, worktree_id, harness, native_session_id
                    )
            elif event == "PostModelSwitch":
                _record_model_switch(
                    connection, worktree_id, harness, native_session_id, payload
                )
            connection.commit()
        finally:
            connection.close()
    return _result("ignored")


def _handle_subagent(
    repo: Path,
    payload: Mapping[str, Any],
    harness: str,
    native_session_id: str,
    event: str,
) -> dict[str, object]:
    """Give one subagent its own ledger session beneath its parent."""

    worktree_id = str(git_dir(repo))
    facets = activity.session_facets(harness, payload)
    # A subagent event names the subagent, so its own ``source`` field never
    # describes how this session began.
    facets["session_source"] = "subagent"
    with _worktree_lock(repo, blocking=True):
        connection = open_db(repo)
        try:
            started_at = _utc_now()
            base_commit = _base_commit(repo)
            parent_native = _native_parent(native_session_id)
            parent_session_id = None
            if parent_native is not None:
                # Resolve the parent first so the child can point at it.
                parent_session_id = _linked_ledger_session(
                    connection,
                    repo,
                    worktree_id,
                    harness,
                    parent_native,
                    base_commit,
                    started_at,
                )
            if event == "SubagentStart":
                model, model_source, feature, feature_source = _session_context(
                    connection,
                    repo,
                    worktree_id,
                    harness,
                    native_session_id,
                    event,
                    payload,
                )
                _retain_facets(
                    connection, worktree_id, harness, native_session_id, facets
                )
                session_id = _ledger_session(
                    connection,
                    worktree_id=worktree_id,
                    harness=harness,
                    native_session_id=native_session_id,
                    model=model,
                    model_source=model_source,
                    feature=feature,
                    feature_source=feature_source,
                    base_commit=base_commit,
                    started_at=started_at,
                )
            else:
                # A stop carries no new model evidence. Selecting the session by
                # model here would open a second one whenever the parent's Agent
                # response has already resolved the model this child ran on.
                session_id = _linked_ledger_session(
                    connection,
                    repo,
                    worktree_id,
                    harness,
                    native_session_id,
                    base_commit,
                    started_at,
                )
                connection.execute(
                    "UPDATE sessions SET ended_at = COALESCE(ended_at, ?) WHERE id = ?",
                    (_utc_now(), session_id),
                )
            activity.apply_session_facets(
                connection, session_id, facets, fields=_FACET_COLUMNS
            )
            if event == "SubagentStop" and parent_session_id is not None:
                # The child produced the result; the parent is the agent that
                # loaded it. Without a known parent no session loaded anything.
                _record_context_load(
                    connection,
                    repo,
                    payload,
                    harness,
                    parent_session_id,
                    event,
                    _utc_now(),
                    related_session_id=session_id,
                )
            if event == "SubagentStop" and isinstance(
                payload.get("last_assistant_message"), str
            ):
                occurred_at = _utc_now()
                agent_id = _optional_text(payload, "agent_id")
                _record_trace(
                    connection, repo, session_id, "assistant",
                    {"text": payload["last_assistant_message"]},
                    occurred_at=occurred_at, agent_id=agent_id,
                )
                if parent_session_id is not None:
                    _record_trace(
                        connection, repo, parent_session_id, "subagent_result",
                        {
                            "text": payload["last_assistant_message"],
                            "agent_type": _optional_text(payload, "agent_type"),
                        },
                        occurred_at=occurred_at, agent_id=agent_id,
                    )
            connection.commit()
        finally:
            connection.close()
    return _result("recorded", session_id=session_id)


def _handle_turn(
    repo: Path,
    payload: Mapping[str, Any],
    harness: str,
    native_session_id: str,
    event: str,
) -> dict[str, object]:
    """Count one prompt or interrupt on the session that received it.

    A prompt is also a context load: it is the one thing every turn loads. Its
    text is reduced to a size and a hash, so the counts and the loads describe
    the shape of the work without storing what was said.
    """

    worktree_id = str(git_dir(repo))
    with _worktree_lock(repo, blocking=True):
        connection = open_db(repo)
        try:
            session_id = _tool_session(
                connection, repo, worktree_id, harness, native_session_id
            )
            activity.increment_session_counter(
                connection, session_id, _COUNTED_EVENTS[event]
            )
            occurred_at = _utc_now()
            _record_context_load(
                connection, repo, payload, harness, session_id, event, occurred_at
            )
            if event == "UserPromptSubmit" and isinstance(payload.get("prompt"), str):
                _record_trace(
                    connection, repo, session_id, "user_prompt",
                    {"text": payload["prompt"]}, occurred_at=occurred_at,
                    turn_id=(
                        _optional_text(payload, "prompt_id")
                        or _optional_text(payload, "turn_id")
                    ),
                )
            connection.commit()
        finally:
            connection.close()
    return _result("recorded", session_id=session_id)


def _handle_context(
    repo: Path,
    payload: Mapping[str, Any],
    harness: str,
    native_session_id: str,
    event: str,
) -> dict[str, object]:
    """Record the instruction file or compaction summary one event loaded.

    A compaction closes the epoch that its own summary describes, so the row is
    written before the count advances. The summary then carries the epoch of
    the context it replaced rather than the epoch it opened.
    """

    worktree_id = str(git_dir(repo))
    with _worktree_lock(repo, blocking=True):
        connection = open_db(repo)
        try:
            session_id = _tool_session(
                connection, repo, worktree_id, harness, native_session_id
            )
            occurred_at = _utc_now()
            recorded = _record_context_load(
                connection, repo, payload, harness, session_id, event, occurred_at
            )
            if event == "PostCompact":
                if isinstance(payload.get("compact_summary"), str):
                    _record_trace(
                        connection, repo, session_id, "compaction",
                        {
                            "text": payload["compact_summary"],
                            "trigger": _optional_text(payload, "trigger"),
                        },
                        occurred_at=occurred_at,
                    )
                activity.increment_session_counter(
                    connection, session_id, "compaction_count"
                )
                recorded = True
            connection.commit()
        finally:
            connection.close()
    return _result("recorded" if recorded else "ignored", session_id=session_id)


def _handle_stop(
    repo: Path,
    payload: Mapping[str, Any],
    harness: str,
    native_session_id: str,
    event: str,
) -> dict[str, object]:
    worktree_id = str(git_dir(repo))
    warnings: list[str] = []
    abandoned = 0
    with _worktree_lock(repo, blocking=True):
        connection = open_db(repo)
        try:
            if event in _COUNTED_EVENTS:
                # A session end closes the session without ending a turn.
                session_id = _tool_session(
                    connection, repo, worktree_id, harness, native_session_id
                )
                activity.increment_session_counter(
                    connection, session_id, _COUNTED_EVENTS[event]
                )
                if isinstance(payload.get("last_assistant_message"), str):
                    _record_trace(
                        connection, repo, session_id, "assistant",
                        {"text": payload["last_assistant_message"]},
                        occurred_at=_utc_now(),
                    )
            active_rows = connection.execute(
                """
                SELECT id, native_session_id, ledger_session_id
                FROM hook_captures
                WHERE worktree_id = ? AND harness = ?
                  AND status IN ('pending', 'contaminated', 'limited', 'imported')
                """,
                (worktree_id, harness),
            ).fetchall()
            explicit_agent = "::" in native_session_id
            rows = [
                row
                for row in active_rows
                if row["native_session_id"] == native_session_id
                or (
                    not explicit_agent
                    and str(row["native_session_id"]).startswith(native_session_id + "::")
                )
            ]
            completed_at = _utc_now()
            for row in rows:
                connection.execute(
                    "UPDATE hook_captures SET status = 'abandoned' WHERE id = ?",
                    (row["id"],),
                )
                connection.execute(
                    "UPDATE sessions SET ended_at = ? WHERE id = ?",
                    (completed_at, row["ledger_session_id"]),
                )
                connection.execute(
                    "DELETE FROM hook_capture_files WHERE capture_id = ?",
                    (row["id"],),
                )
            abandoned = len(rows)
            connection.commit()
        finally:
            connection.close()

    if abandoned:
        warnings.append(
            f"Abandoned {abandoned} unmatched native capture(s); their changes remain unknown."
        )
    recorded_commits, drain_warnings = _drain_pending(repo, worktree_id)
    warnings.extend(drain_warnings)
    return _result(
        "captured" if recorded_commits else "ignored",
        recorded_commits=recorded_commits,
        warnings=warnings,
    )


def abandon_worktree(repo: RepoPath) -> dict[str, object]:
    """Settle all active capture state before one worktree is disabled."""

    root = repository_root(repo)
    worktree_id = str(git_dir(root))
    database_path = git_common_dir(root) / "attribution" / "ledger.sqlite3"
    if not database_path.is_file():
        return {
            "abandoned_captures": 0,
            "recorded_commits": [],
            "dropped_pending_commits": 0,
            "warnings": [],
        }

    with _worktree_lock(root, blocking=True):
        connection = open_db(root)
        try:
            active_rows = connection.execute(
                """
                SELECT id, ledger_session_id FROM hook_captures
                WHERE worktree_id = ?
                  AND status IN ('pending', 'contaminated', 'limited', 'imported')
                """,
                (worktree_id,),
            ).fetchall()
            completed_at = _utc_now()
            for row in active_rows:
                connection.execute(
                    "UPDATE hook_captures SET status = 'abandoned' WHERE id = ?",
                    (row["id"],),
                )
                connection.execute(
                    "UPDATE sessions SET ended_at = ? WHERE id = ?",
                    (completed_at, row["ledger_session_id"]),
                )
                connection.execute(
                    "DELETE FROM hook_capture_files WHERE capture_id = ?",
                    (row["id"],),
                )
            connection.commit()
        finally:
            connection.close()

    recorded, warnings = _drain_pending(root, worktree_id)
    dropped = 0
    with _worktree_lock(root, blocking=True):
        connection = open_db(root)
        try:
            dropped = int(
                connection.execute(
                    "SELECT COUNT(*) AS count FROM pending_commits WHERE worktree_id = ?",
                    (worktree_id,),
                ).fetchone()["count"]
            )
            if dropped:
                connection.execute(
                    "DELETE FROM pending_commits WHERE worktree_id = ?",
                    (worktree_id,),
                )
                connection.commit()
        finally:
            connection.close()
    if dropped:
        warnings.append(
            f"Discarded {dropped} pending commit record(s) that could not be published safely."
        )
    return {
        "abandoned_captures": len(active_rows),
        "recorded_commits": recorded,
        "dropped_pending_commits": dropped,
        "warnings": warnings,
    }


def _payload_repo_matches(repo: Path, payload: Mapping[str, Any]) -> bool:
    cwd = _optional_text(payload, "cwd")
    if cwd is None:
        return True
    try:
        return repository_root(cwd) == repo
    except ValueError:
        return False


def _register_telemetry_session(
    repo: Path, payload: Mapping[str, Any], harness: str, event: str
) -> None:
    """Make only this installed repository's native session eligible for OTel."""

    if event not in {"SessionStart", "PreToolUse"}:
        return
    state_path = git_common_dir(repo) / "attribution" / "install.json"
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return
    if not isinstance(state, dict) or state.get("telemetry_enabled") is not True:
        return
    from .telemetry import ensure_collector, register_session

    native_session_id = _required_text(payload, "session_id")
    if event == "SessionStart":
        ensure_collector()
    register_session(
        "claude" if harness == "claude-code" else "codex",
        native_session_id,
        str(git_common_dir(repo)),
        repository_path=str(repo),
    )


def handle_hook(repo: RepoPath, payload: Mapping[str, Any], harness: str) -> dict[str, object]:
    """Handle one Codex or Claude Code JSON hook event without policy output."""

    try:
        if (
            os.environ.get("ATTRIBUTION_WRAPPED_CAPTURE") == "1"
            or os.environ.get("ATTRIBUTION_WRAPPER_SESSION_ID")
        ):
            return _result("ignored")
        if harness not in SUPPORTED_HARNESSES:
            raise ValueError("harness must be codex or claude-code")
        if not isinstance(payload, Mapping):
            raise ValueError("Hook payload must be a JSON object")
        root = repository_root(repo)
        enabled, gate_warning = _install_enabled(root)
        if not enabled and not _enrolled(root):
            # Machine-level hooks fire in every repository. One that never
            # enrolled writes nothing and takes no worktree snapshot.
            return _result("ignored")
        if not _payload_repo_matches(root, payload):
            raise ValueError("Hook payload cwd belongs to a different Git worktree")
        if not enabled:
            enabled, gate_warning = _heal_worktree(root)
        if not enabled:
            return _result("ignored", warnings=[gate_warning] if gate_warning else [])
        event = _required_text(payload, "hook_event_name")
        if event not in KNOWN_EVENTS:
            return _result("ignored")
        try:
            _register_telemetry_session(root, payload, harness, event)
        except (OSError, ValueError, sqlite3.Error, subprocess.SubprocessError):
            # Cost telemetry is optional enrichment. It must never veto code
            # capture, a tool call, or a Git operation.
            pass
        native_session_id = _effective_native_session(payload, event)
        if event == "PreToolUse":
            if not _takes_snapshot(harness, payload):
                # A tool that changes no file needs no baseline. Its row is
                # inserted when it completes, which costs no snapshot.
                return _handle_fast_pre(root, payload, harness, native_session_id)
            return _handle_pre(root, payload, harness, native_session_id)
        if event in {"PostToolUse", "PostToolUseFailure"}:
            return _handle_post(root, payload, harness, native_session_id, event)
        if event in {"SessionStart", "PostModelSwitch"}:
            return _handle_metadata(root, payload, harness, native_session_id, event)
        if event in {"SubagentStart", "SubagentStop"}:
            return _handle_subagent(root, payload, harness, native_session_id, event)
        if event in {"UserPromptSubmit", "Interrupt"}:
            return _handle_turn(root, payload, harness, native_session_id, event)
        if event in {"InstructionsLoaded", "PostCompact"}:
            return _handle_context(root, payload, harness, native_session_id, event)
        if event in {"Stop", "SessionEnd"}:
            return _handle_stop(root, payload, harness, native_session_id, event)
        return _result("ignored")
    except Exception as exc:
        return _result("warning", warnings=[_warning(exc)])


def _enrolled(repo: Path) -> bool:
    """Return whether this checkout carries the hosted attribution workflow."""

    return (repo / WORKFLOW_PATH).is_file()


def _heal_worktree(repo: Path) -> tuple[bool, str | None]:
    """Install this enrolled clone's Git publisher from a machine-level install."""

    try:
        from .install import heal_worktree

        if not heal_worktree(repo):
            return False, "Native attribution automation is not enabled for this worktree."
    except Exception as exc:
        # Provisioning is best effort. A hook never blocks the coding tool.
        return False, _warning(exc)
    return _install_enabled(repo)


def _install_enabled(repo: Path) -> tuple[bool, str | None]:
    state_path = git_common_dir(repo) / "attribution" / "install.json"
    try:
        raw = state_path.read_bytes()
    except FileNotFoundError:
        return False, "Native attribution automation is not enabled for this worktree."
    except OSError as exc:
        return False, f"Could not read native automation state: {_warning(exc)}"
    if len(raw) > 64 * 1024:
        return False, "Native automation state is oversized and was ignored."
    try:
        state = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return False, "Native automation state is malformed and was ignored."
    if not isinstance(state, dict):
        return False, "Native automation state has an unsupported format and was ignored."
    enabled_worktrees = state.get("enabled_worktrees")
    if state.get("version") != INSTALL_STATE_VERSION or not isinstance(enabled_worktrees, list):
        return False, "Native automation state has an unsupported format and was ignored."
    if str(git_dir(repo)) not in enabled_worktrees:
        return False, "Native attribution automation is not enabled for this worktree."
    return True, None


def handle_git_hook(repo: RepoPath, event: str = "post-commit") -> dict[str, object]:
    """Queue a committed HEAD and publish it once native captures are settled."""

    try:
        root = repository_root(repo)
        enabled, gate_warning = _install_enabled(root)
        if not enabled:
            return _result("ignored", warnings=[gate_warning] if gate_warning else [])
        if event not in {"post-commit", "post-merge"}:
            return _result("ignored")
        commit_sha = _base_commit(root)
        if commit_sha is None:
            raise ValueError("post-commit hook could not resolve HEAD")
        worktree_id = str(git_dir(root))
        with _worktree_lock(root, blocking=True):
            connection = open_db(root)
            try:
                connection.execute(
                    """
                    INSERT OR IGNORE INTO pending_commits(worktree_id, commit_sha, queued_at)
                    VALUES (?, ?, ?)
                    """,
                    (worktree_id, commit_sha, _utc_now()),
                )
                history_import = event == "post-merge" or (
                    event == "post-commit"
                    and not _latest_head_action_is_direct_commit(root)
                )
                if history_import:
                    connection.execute(
                        """
                        UPDATE hook_captures SET status = 'imported'
                        WHERE worktree_id = ? AND status = 'pending'
                        """,
                        (worktree_id,),
                    )
                connection.commit()
                if _active_count(connection, worktree_id):
                    return _result("pending")
            finally:
                connection.close()
        recorded, warnings = _drain_pending(root, worktree_id)
        status = "captured" if recorded else ("warning" if warnings else "ignored")
        return _result(status, recorded_commits=recorded, warnings=warnings)
    except Exception as exc:
        return _result("warning", warnings=[_warning(exc)])


def automation_status(repo: RepoPath) -> dict[str, object]:
    """Return durable native-capture queue state for the current worktree."""

    warnings: list[str] = []
    try:
        root = repository_root(repo)
        enabled, gate_warning = _install_enabled(root)
        if not enabled and gate_warning:
            warnings.append(gate_warning)
        worktree_id = str(git_dir(root))
        database_path = git_common_dir(root) / "attribution" / "ledger.sqlite3"
        if not database_path.is_file():
            return {"pending_captures": 0, "pending_commits": 0, "warnings": warnings}
        connection = sqlite3.connect(
            f"{database_path.resolve().as_uri()}?mode=ro",
            uri=True,
            timeout=5,
        )
        connection.row_factory = sqlite3.Row
        try:
            tables = {
                row["name"]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                )
            }
            if not {"hook_captures", "pending_commits"}.issubset(tables):
                return {"pending_captures": 0, "pending_commits": 0, "warnings": warnings}
            pending_captures = _active_count(connection, worktree_id)
            pending_commits = int(
                connection.execute(
                    "SELECT COUNT(*) AS count FROM pending_commits WHERE worktree_id = ?",
                    (worktree_id,),
                ).fetchone()["count"]
            )
            contaminated = int(
                connection.execute(
                    """
                    SELECT COUNT(*) AS count FROM hook_captures
                    WHERE worktree_id = ? AND status IN ('contaminated', 'completed_contaminated')
                    """,
                    (worktree_id,),
                ).fetchone()["count"]
            )
            limited = int(
                connection.execute(
                    """
                    SELECT COUNT(*) AS count FROM hook_captures
                    WHERE worktree_id = ? AND status IN ('limited', 'completed_limited')
                    """,
                    (worktree_id,),
                ).fetchone()["count"]
            )
            imported = int(
                connection.execute(
                    """
                    SELECT COUNT(*) AS count FROM hook_captures
                    WHERE worktree_id = ? AND status IN ('imported', 'completed_imported')
                    """,
                    (worktree_id,),
                ).fetchone()["count"]
            )
            abandoned = int(
                connection.execute(
                    "SELECT COUNT(*) AS count FROM hook_captures WHERE worktree_id = ? AND status = 'abandoned'",
                    (worktree_id,),
                ).fetchone()["count"]
            )
        finally:
            connection.close()
        if contaminated:
            warnings.append(
                f"{contaminated} overlapping native capture(s) were kept unattributed."
            )
        if limited:
            warnings.append(
                f"{limited} native capture(s) exceeded snapshot limits and were kept unattributed."
            )
        if imported:
            warnings.append(
                f"{imported} native capture(s) imported committed history and were kept unattributed."
            )
        if abandoned:
            warnings.append(f"{abandoned} unmatched native capture(s) were abandoned safely.")
        return {
            "pending_captures": pending_captures,
            "pending_commits": pending_commits,
            "warnings": warnings,
        }
    except (OSError, ValueError, sqlite3.Error, subprocess.SubprocessError) as exc:
        return {"pending_captures": 0, "pending_commits": 0, "warnings": [_warning(exc)]}
