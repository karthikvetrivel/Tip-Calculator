"""Record repository transitions from native agent hook events."""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import os
from pathlib import Path
import sqlite3
import subprocess
from typing import Iterable, Iterator, Mapping
import uuid

from . import activity
from .capture import (
    SnapshotLimitError,
    _base_commit,
    _content_hash,
    _snapshot,
    _utc_now,
    _validated_label,
    _worktree_lock,
)
from .hooks import HookEvent, event_fingerprint
from .harnesses import harness_display_name
from .runtime import system_subprocess_environment
from .store import RepoPath, _run_git, git_dir, open_db, repository_root
from .tasks import resolve_task


_TERMINAL_EVENTS = {
    "error",
    "interrupt",
    "session_end",
    "stop",
    "subagent_stop",
    "turn_end",
}
_SUCCESS_EVENTS = {"session_end", "stop", "subagent_stop", "turn_end"}
# A session counts the events it saw itself. A session end closes the session
# without ending a turn, and a subagent stop ends its child's turn, not this
# session's.
_COUNTED_EVENTS = {
    "stop": "turn_count",
    "turn_end": "turn_count",
    "user_prompt": "prompt_count",
    "interrupt": "interrupt_count",
}
_FAILED_SESSION_REASONS = {
    "abort",
    "aborted",
    "error",
    "failed",
    "failure",
    "timeout",
    "timed_out",
}
HOOK_LOCK_WAIT_SECONDS = 2.0
MAX_HOOK_SNAPSHOT_BYTES = 64 * 1024 * 1024
MAX_HOOK_SNAPSHOT_FILES = 100_000
_LEGACY_ACTIVE_CAPTURE_STATES = ("pending", "contaminated", "limited", "imported")


@contextmanager
def _native_capture_lock(root: Path, worktree_id: str) -> Iterator[None]:
    """Serialize a native event, then drain deferred commits after unlocking."""

    acquired = False
    try:
        with _worktree_lock(root, wait_seconds=HOOK_LOCK_WAIT_SECONDS):
            acquired = True
            yield
    finally:
        if acquired:
            # Import late to avoid capture.py <-> automation.py import recursion.
            from .automation import _drain_pending

            _drain_pending(root, worktree_id)


def _same_repository(root: Path, event_cwd: str | None) -> None:
    if event_cwd is None:
        return
    try:
        event_root = repository_root(Path(event_cwd))
    except (OSError, ValueError) as exc:
        raise ValueError("The hook working directory is not in a Git repository") from exc
    if event_root != root:
        raise ValueError("The hook working directory belongs to a different Git repository")


def _load_snapshot(
    connection: sqlite3.Connection, unit_key: str
) -> tuple[dict[str, bytes | None], dict[str, str]]:
    files: dict[str, bytes | None] = {}
    skipped: dict[str, str] = {}
    rows = connection.execute(
        """
        SELECT files.path, files.content, files.content_hash,
               files.is_absent, files.skip_reason,
               blobs.content AS blob_content
        FROM hook_snapshot_files AS files
        LEFT JOIN hook_snapshot_blobs AS blobs
          ON blobs.content_hash = files.content_hash
        WHERE files.unit_key = ?
        ORDER BY files.path
        """,
        (unit_key,),
    ).fetchall()
    for row in rows:
        path = str(row["path"])
        if row["skip_reason"] is not None:
            skipped[path] = str(row["skip_reason"])
        elif row["is_absent"]:
            files[path] = None
        elif row["content"] is not None:
            # Read inline rows from the first development schema.
            files[path] = bytes(row["content"])
        elif row["content_hash"] is not None and row["blob_content"] is not None:
            files[path] = bytes(row["blob_content"])
        else:
            skipped[path] = "missing_snapshot_content"
    return files, skipped


def _snapshot_limit_reason(
    files: dict[str, bytes | None], skipped: dict[str, str]
) -> str | None:
    if len(files) + len(skipped) > MAX_HOOK_SNAPSHOT_FILES:
        return f"snapshot has more than {MAX_HOOK_SNAPSHOT_FILES:,} files"
    total_bytes = sum(len(content) for content in files.values() if content is not None)
    if total_bytes > MAX_HOOK_SNAPSHOT_BYTES:
        return f"snapshot exceeds {MAX_HOOK_SNAPSHOT_BYTES // (1024 * 1024)} MiB"
    return None


def _bounded_snapshot(
    root: Path,
) -> tuple[dict[str, bytes | None], dict[str, str], str | None]:
    """Take a hook snapshot without first materializing an oversized tree."""

    try:
        files, skipped = _snapshot(
            root,
            maximum_total_bytes=MAX_HOOK_SNAPSHOT_BYTES,
            maximum_files=MAX_HOOK_SNAPSHOT_FILES,
        )
    except SnapshotLimitError as exc:
        return {}, {}, str(exc)
    return files, skipped, _snapshot_limit_reason(files, skipped)


def _delete_snapshots(
    connection: sqlite3.Connection, unit_keys: Iterable[str]
) -> None:
    connection.executemany(
        "DELETE FROM hook_snapshot_files WHERE unit_key = ?",
        [(unit_key,) for unit_key in unit_keys],
    )
    connection.execute(
        """
        DELETE FROM hook_snapshot_blobs
        WHERE NOT EXISTS (
            SELECT 1
            FROM hook_snapshot_files
            WHERE hook_snapshot_files.content_hash = hook_snapshot_blobs.content_hash
        )
        """
    )


def _save_snapshot(
    connection: sqlite3.Connection,
    unit_keys: Iterable[str],
    files: dict[str, bytes | None],
    skipped: dict[str, str],
    baseline_commit: str | None,
) -> None:
    reason = _snapshot_limit_reason(files, skipped)
    if reason is not None:
        raise ValueError(reason)
    keys = tuple(unit_keys)
    _delete_snapshots(connection, keys)

    blobs: dict[str, bytes] = {}
    file_rows: list[tuple[str, str, str | None, int]] = []
    for path, content in sorted(files.items()):
        content_hash = _content_hash(content)
        if content_hash is not None and content is not None:
            blobs[content_hash] = content
        file_rows.append((path, content_hash, int(content is None)))
    connection.executemany(
        """
        INSERT OR IGNORE INTO hook_snapshot_blobs(content_hash, content)
        VALUES (?, ?)
        """,
        sorted(blobs.items()),
    )

    for unit_key in keys:
        connection.executemany(
            """
            INSERT INTO hook_snapshot_files(
                unit_key, path, content, content_hash, is_absent, skip_reason
            ) VALUES (?, ?, NULL, ?, ?, NULL)
            """,
            [
                (unit_key, path, content_hash, is_absent)
                for path, content_hash, is_absent in file_rows
            ],
        )
        connection.executemany(
            """
            INSERT INTO hook_snapshot_files(
                unit_key, path, content, is_absent, skip_reason
            ) VALUES (?, ?, NULL, 0, ?)
            """,
            [
                (unit_key, path, reason)
                for path, reason in sorted(skipped.items())
            ],
        )
        connection.execute(
            "UPDATE hook_units SET baseline_commit = ? WHERE unit_key = ?",
            (baseline_commit, unit_key),
        )
    _delete_snapshots(connection, ())


def _record_delta(
    connection: sqlite3.Connection,
    session_id: str,
    before: dict[str, bytes | None],
    before_skips: dict[str, str],
    after: dict[str, bytes | None],
    after_skips: dict[str, str],
    base_commit: str | None,
) -> list[str]:
    blocked = set(before_skips) | set(after_skips)
    changed: list[str] = []
    for path in sorted(set(before) | set(after)):
        if path in blocked:
            continue
        before_content = before.get(path)
        after_content = after.get(path)
        if before_content == after_content:
            continue
        changed.append(path)
        connection.execute(
            """
            INSERT INTO edits(
                session_id, path, before_content, after_content,
                before_hash, after_hash, base_commit
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                path,
                before_content,
                after_content,
                _content_hash(before_content),
                _content_hash(after_content),
                base_commit,
            ),
        )
    return changed


def _active_units(
    connection: sqlite3.Connection, worktree_id: str
) -> list[sqlite3.Row]:
    return connection.execute(
        """
        SELECT units.rowid AS ledger_order, units.*,
               sessions.provider AS provider,
               sessions.task_id AS task_id,
               sessions.base_commit AS session_base_commit
        FROM hook_units AS units
        JOIN sessions ON sessions.id = units.attribution_session_id
        WHERE units.worktree_id = ? AND units.active = 1
        ORDER BY ledger_order
        """,
        (worktree_id,),
    ).fetchall()


def _close_without_evidence(
    connection: sqlite3.Connection,
    units: Iterable[sqlite3.Row],
    now: str,
    last_event: str,
) -> int:
    materialized = list(units)
    for unit in materialized:
        _close_session(
            connection,
            str(unit["attribution_session_id"]),
            now,
            exit_code=None,
            outcome="abandoned",
        )
        connection.execute(
            """
            UPDATE hook_units
            SET active = 0, last_event = ?, updated_at = ?
            WHERE unit_key = ?
            """,
            (last_event, now, unit["unit_key"]),
        )
    _delete_snapshots(
        connection,
        [str(unit["unit_key"]) for unit in materialized],
    )
    return len(materialized)


def _matching_unit(
    active_units: list[sqlite3.Row], event: HookEvent, model: str
) -> sqlite3.Row | None:
    identity_matches = [
        row
        for row in active_units
        if row["harness_id"] == event.harness_id
        and row["native_session_id"] == event.native_session_id
        and row["native_turn_id"] == event.native_turn_id
        and row["native_agent_id"] == event.native_agent_id
    ]
    provider_matches = [
        row
        for row in identity_matches
        if event.provider is None
        or row["provider"] is None
        or row["provider"] == event.provider
    ]
    exact_matches = [row for row in provider_matches if row["model"] == model]
    if exact_matches:
        return exact_matches[-1]
    if len(provider_matches) == 1 and (
        model == "unknown" or provider_matches[0]["model"] == "unknown"
    ):
        return provider_matches[0]
    return None


def _known_metadata_conflicts(
    active_units: list[sqlite3.Row], event: HookEvent, model: str
) -> tuple[list[sqlite3.Row], bool, bool]:
    conflicts: list[sqlite3.Row] = []
    model_changed = False
    provider_changed = False
    for row in active_units:
        if (
            row["harness_id"] != event.harness_id
            or row["native_session_id"] != event.native_session_id
            or row["native_turn_id"] != event.native_turn_id
            or row["native_agent_id"] != event.native_agent_id
        ):
            continue
        row_model = str(row["model"])
        row_provider = row["provider"]
        model_conflict = (
            model != "unknown"
            and row_model != "unknown"
            and model != row_model
        )
        provider_conflict = (
            event.provider is not None
            and row_provider is not None
            and event.provider != row_provider
        )
        if model_conflict or provider_conflict:
            conflicts.append(row)
            model_changed = model_changed or model_conflict
            provider_changed = provider_changed or provider_conflict
    return conflicts, model_changed, provider_changed


def _has_ambiguous_overlap(
    active_units: list[sqlite3.Row],
    unit_key: str,
    event: HookEvent,
) -> bool:
    """Return whether another open unit prevents safe deepest-span ownership."""

    for row in active_units:
        if row["unit_key"] == unit_key:
            continue
        if row["native_session_id"] != event.native_session_id:
            return True

        other_agent = row["native_agent_id"]
        current_agent = event.native_agent_id
        if current_agent is None and other_agent is not None:
            # A parent normally waits while a child runs. A parent boundary
            # while the child remains open signals a concurrent or incomplete
            # span, so neither unit can safely own the transition.
            return True
        if (
            current_agent is not None
            and other_agent is not None
            and current_agent != other_agent
        ):
            return True
        if (
            current_agent == other_agent
            and event.native_turn_id is not None
            and row["native_turn_id"] is not None
            and event.native_turn_id != row["native_turn_id"]
        ):
            return True
    return False


def _event_is_terminal(event: HookEvent) -> bool:
    if event.event == "error" and event.recoverable is True:
        return False
    return event.event in _TERMINAL_EVENTS


def _event_exit_code(event: HookEvent) -> int:
    if event.event == "interrupt":
        return 130
    reason = (event.terminal_reason or "").strip().lower().replace("-", "_")
    if event.event == "session_end" and reason in _FAILED_SESSION_REASONS:
        return 1
    return 0 if event.event in _SUCCESS_EVENTS else 1


def _event_outcome(event: HookEvent) -> str:
    exit_code = _event_exit_code(event)
    if exit_code == 130:
        return "interrupted"
    return "completed" if exit_code == 0 else "failed"


def _touch_session_task(
    connection: sqlite3.Connection, session_id: str, now: str
) -> None:
    connection.execute(
        """
        UPDATE tasks
        SET updated_at = ?
        WHERE id = (SELECT task_id FROM sessions WHERE id = ?)
        """,
        (now, session_id),
    )


def _close_session(
    connection: sqlite3.Connection,
    session_id: str,
    now: str,
    *,
    exit_code: int | None,
    outcome: str,
) -> None:
    connection.execute(
        """
        UPDATE sessions
        SET ended_at = ?, exit_code = ?, outcome = ?
        WHERE id = ?
        """,
        (now, exit_code, outcome, session_id),
    )
    _touch_session_task(connection, session_id, now)


def _event_database_fingerprint(
    worktree_id: str, event: HookEvent
) -> str | None:
    # Some hook APIs, including Claude Code, do not provide a turn, event, or
    # timestamp identifier. Treating their event name as a replay key would
    # collapse every later turn in the same native session. Repeated calls are
    # still safe because open units match in place and orphan terminal events
    # are ignored.
    if (
        event.native_turn_id is None
        and event.native_event_id is None
        and event.occurred_at is None
    ):
        return None
    digest = hashlib.sha256()
    digest.update(worktree_id.encode("utf-8"))
    digest.update(b"\0")
    digest.update(event_fingerprint(event).encode("ascii"))
    return digest.hexdigest()


def _transition_spans_at_most_one_commit(
    root: Path,
    baseline_commit: str | None,
    current_commit: str | None,
) -> bool:
    """Return whether one snapshot interval can map to one commit parent."""

    if current_commit == baseline_commit:
        return True
    if current_commit is None:
        return False
    reflog = subprocess.run(
        ["git", "-C", str(root), "reflog", "show", "-1", "--format=%gs", "HEAD"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=system_subprocess_environment(),
        check=False,
    )
    if reflog.returncode != 0:
        return False
    action = reflog.stdout.decode("utf-8", errors="replace").strip()
    if not action.startswith(("commit: ", "commit (initial): ", "commit (amend): ")):
        return False
    if baseline_commit is None:
        try:
            commit_line = _run_git(
                root, "rev-list", "--parents", "-n", "1", current_commit
            ).stdout.split()
        except ValueError:
            return False
        return len(commit_line) == 1
    try:
        _run_git(root, "merge-base", "--is-ancestor", baseline_commit, current_commit)
        count = int(
            _run_git(
                root, "rev-list", "--count", f"{baseline_commit}..{current_commit}"
            ).stdout.strip()
        )
    except (ValueError, TypeError):
        return False
    return count == 1


def _create_unit(
    connection: sqlite3.Connection,
    root: Path,
    worktree_id: str,
    task: Mapping[str, object],
    event: HookEvent,
    model: str,
    harness_version: str | None,
    model_source: str,
    now: str,
    baseline_commit: str | None,
) -> tuple[str, str]:
    session_id = str(uuid.uuid4())
    unit_key = str(uuid.uuid4())
    connection.execute(
        """
        INSERT INTO sessions(
            id, worktree_id, task_id, feature, feature_source, model,
            harness, label_source, membership_source, role,
            cost_usd, cost_source, started_at, base_commit,
            harness_id, harness_version, native_session_id,
            native_turn_id, native_agent_id, native_parent_session_id, provider,
            model_source, harness_source, integration_mode, invocation_cwd
        ) VALUES (
            ?, ?, ?, ?, 'reported', ?, ?, ?, ?, 'implementation',
            NULL, NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
            'native_hook', 'native_hook', ?
        )
        """,
        (
            session_id,
            worktree_id,
            task["id"],
            task["name"],
            model,
            harness_display_name(event.harness_id),
            "mixed" if model_source == "reported" else "native_hook",
            task["membership_source"],
            now,
            baseline_commit,
            event.harness_id,
            harness_version,
            event.native_session_id,
            event.native_turn_id,
            event.native_agent_id,
            event.parent_session_id,
            event.provider,
            model_source,
            event.cwd or str(root),
        ),
    )
    connection.execute(
        """
        INSERT INTO hook_units(
            unit_key, worktree_id, harness_id, native_session_id,
            native_turn_id, native_agent_id, model, baseline_commit,
            attribution_session_id, last_event, active, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?)
        """,
        (
            unit_key,
            worktree_id,
            event.harness_id,
            event.native_session_id,
            event.native_turn_id,
            event.native_agent_id,
            model,
            baseline_commit,
            session_id,
            event.event,
            now,
            now,
        ),
    )
    if event.native_agent_id is not None:
        # A subagent reports its parent's native session, so the main agent of
        # that session is its parent in the ledger tree.
        parent = connection.execute(
            """
            SELECT id
            FROM sessions
            WHERE worktree_id = ? AND harness_id = ? AND native_session_id = ?
              AND native_agent_id IS NULL AND id <> ?
            ORDER BY rowid DESC
            LIMIT 1
            """,
            (worktree_id, event.harness_id, event.native_session_id, session_id),
        ).fetchone()
        if parent is not None:
            connection.execute(
                "UPDATE sessions SET parent_session_id = ? WHERE id = ?",
                (parent["id"], session_id),
            )
    return unit_key, session_id


# A session's duration is not among these: an event that reports a duration is
# reporting its own tool call, not the wall-clock life of the session.
_FACET_FIELDS = (
    "agent_type",
    "launch_mode",
    "session_source",
    "permission_mode",
    "effort_level",
)


def _child_session(
    connection: sqlite3.Connection,
    worktree_id: str,
    event: HookEvent,
    agent_id: str | None,
) -> str | None:
    if agent_id is None:
        return None
    row = connection.execute(
        """
        SELECT id
        FROM sessions
        WHERE worktree_id = ? AND harness_id = ? AND native_session_id = ?
          AND native_agent_id = ?
        ORDER BY rowid DESC
        LIMIT 1
        """,
        (worktree_id, event.harness_id, event.native_session_id, agent_id),
    ).fetchone()
    return None if row is None else str(row["id"])


def _record_workflow_switch(
    connection: sqlite3.Connection,
    session_id: str,
    workflow: Mapping[str, object] | None,
    previous_model: str | None,
    now: str,
) -> None:
    """Record a reported model change against the session that was running.

    The session keeps its own model. A model change opens a new unit here, so
    relabelling the closing session would credit the new model with the work
    the previous one produced.
    """

    switch = (workflow or {}).get("model_switch")
    if not isinstance(switch, Mapping):
        return
    activity.record_model_switch(
        connection,
        session_id,
        from_model=switch.get("from_model") or previous_model,
        to_model=switch.get("to_model"),
        source=str(switch.get("source") or "unknown"),
        occurred_at=now,
    )


def _apply_context_load(
    connection: sqlite3.Connection,
    session_id: str,
    event: HookEvent,
    workflow: Mapping[str, object],
    now: str,
) -> None:
    """Record what one event loaded into the session that loaded it.

    A subagent result is produced by the child and loaded by the parent that
    launched it, so the row names the child and belongs to the parent. A
    compaction summary is written before the count advances, so it carries the
    epoch of the context it replaced.
    """

    # ``activity.context_load`` read the payload by the harness spelling of the
    # event; the normalized name reaches the same kind, so the row this
    # receiver places on a session and the load it holds cannot disagree.
    kind = activity.context_load_kind(event.event)
    load = workflow.get("context_load")
    if kind is None or not isinstance(load, Mapping) or load.get("kind") != kind:
        return
    target, related = session_id, None
    if kind == "subagent_result":
        parent = connection.execute(
            "SELECT parent_session_id FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        if parent is None or parent["parent_session_id"] is None:
            # No parent session is known, so no session loaded this result.
            return
        target, related = str(parent["parent_session_id"]), session_id
    activity.record_context_load(
        connection,
        target,
        kind=kind,
        occurred_at=now,
        locator=load.get("locator"),
        content_hash=load.get("content_hash"),
        size_bytes=load.get("size_bytes"),
        memory_type=load.get("memory_type"),
        load_reason=load.get("load_reason"),
        turn_id=event.native_turn_id,
        related_session_id=related,
    )
    if kind == "compaction_summary":
        activity.increment_session_counter(
            connection, session_id, "compaction_count"
        )


def _apply_workflow(
    connection: sqlite3.Connection,
    worktree_id: str,
    session_id: str,
    event: HookEvent,
    workflow: Mapping[str, object] | None,
    now: str,
) -> None:
    """Record the workflow facts one event carries onto its ledger session."""

    if not workflow:
        return
    facets = dict(workflow.get("facets") or {})
    if event.event in {"subagent_start", "subagent_stop"}:
        facets["session_source"] = "subagent"
    elif event.event != "session_start":
        # Only a start says how its session began. Other events carry their own
        # unrelated ``source``; a model switch reports ``resume`` for a resumed
        # model, which must never be read as a resumed session.
        facets.pop("session_source", None)
    activity.apply_session_facets(
        connection, session_id, facets, fields=_FACET_FIELDS
    )
    counter = _COUNTED_EVENTS.get(event.event)
    if counter is not None:
        activity.increment_session_counter(connection, session_id, counter)
    _apply_context_load(connection, session_id, event, workflow, now)

    tool_call = workflow.get("tool_call")
    if not isinstance(tool_call, Mapping) or event.event not in {
        "pre_tool",
        "post_tool",
    }:
        return
    activity.record_tool_call(
        connection,
        session_id,
        tool_use_id=str(tool_call["tool_use_id"]),
        tool_name=str(tool_call["tool_name"]),
        tool_class=str(tool_call["tool_class"]),
        occurred_at=now,
        turn_id=event.native_turn_id,
        locator=tool_call.get("locator"),
        locator_hash=tool_call.get("locator_hash"),
        succeeded=tool_call.get("succeeded"),
        duration_ms=tool_call.get("duration_ms"),
    )
    response_facets = workflow.get("response_facets")
    if (
        tool_call.get("tool_class") != "agent"
        or event.event != "post_tool"
        or not isinstance(response_facets, Mapping)
    ):
        return
    activity.record_agent_response(
        connection,
        parent_session_id=session_id,
        tool_use_id=str(tool_call["tool_use_id"]),
        child_session_id=_child_session(
            connection, worktree_id, event, response_facets.get("agent_id")
        ),
        facets=response_facets,
        occurred_at=now,
        summary=tool_call.get("summary"),
    )


def record_hook_event(
    repo: RepoPath,
    feature: str,
    event: HookEvent,
    *,
    harness_version: str | None = None,
    model_source: str = "native_hook",
    workflow: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Capture one native hook boundary without storing the raw hook payload.

    ``workflow`` carries the content-free facts that ``activity.hook_activity``
    read from the payload. Without it this receiver behaves exactly as before.
    """

    wrapper_session = os.environ.get("ATTRIBUTION_WRAPPER_SESSION_ID")
    if wrapper_session:
        return {
            "ignored": True,
            "reason": "wrapped_command",
            "wrapper_session_id": wrapper_session,
        }

    feature = _validated_label(feature, "feature").strip()
    if harness_version is not None:
        harness_version = _validated_label(harness_version, "harness_version").strip()
        if len(harness_version) > 256:
            raise ValueError("harness_version is too long")
    model_source = _validated_label(model_source, "model_source")
    if model_source not in {"native_hook", "reported"}:
        raise ValueError("model_source must be native_hook or reported")
    model = (event.model or "unknown").strip()
    effective_model_source = model_source if event.model is not None else "unknown"
    root = repository_root(repo)
    _same_repository(root, event.cwd)
    worktree_id = str(git_dir(root))
    now = _utc_now()
    database_fingerprint = _event_database_fingerprint(worktree_id, event)
    terminal = _event_is_terminal(event)

    with _native_capture_lock(root, worktree_id):
        connection = open_db(root)
        try:
            duplicate = None
            if database_fingerprint is not None:
                duplicate = connection.execute(
                    "SELECT unit_key FROM hook_events WHERE fingerprint = ?",
                    (database_fingerprint,),
                ).fetchone()
            if duplicate is not None:
                return {
                    "duplicate": True,
                    "event": event.event,
                    "unit_key": duplicate["unit_key"],
                    "changed_files": [],
                }

            active = _active_units(connection, worktree_id)
            placeholders = ",".join("?" for _ in _LEGACY_ACTIVE_CAPTURE_STATES)
            legacy_active = int(
                connection.execute(
                    f"""
                    SELECT COUNT(*) AS count
                    FROM hook_captures
                    WHERE worktree_id = ? AND status IN ({placeholders})
                    """,
                    (worktree_id, *_LEGACY_ACTIVE_CAPTURE_STATES),
                ).fetchone()["count"]
            )
            if legacy_active:
                connection.execute(
                    f"""
                    UPDATE hook_captures
                    SET status = 'contaminated'
                    WHERE worktree_id = ? AND status IN ({placeholders})
                    """,
                    (worktree_id, *_LEGACY_ACTIVE_CAPTURE_STATES),
                )
                closed_units = _close_without_evidence(
                    connection,
                    active,
                    now,
                    "wrapper_overlap",
                )
                connection.commit()
                return {
                    "ignored": True,
                    "reason": "wrapper_overlap",
                    "event": event.event,
                    "changed_files": [],
                    "closed_units": closed_units,
                }
            unit = _matching_unit(active, event, model)
            conflicts, model_changed, provider_changed = _known_metadata_conflicts(
                active, event, model
            )
            if conflicts:
                # The transition since the old baseline can span both metadata
                # values. Do not assign it to either side of the boundary.
                current, current_skips, limit_reason = _bounded_snapshot(root)
                if limit_reason is not None:
                    closed_units = _close_without_evidence(
                        connection,
                        active,
                        now,
                        "snapshot_limit",
                    )
                    connection.commit()
                    return {
                        "ignored": True,
                        "reason": "snapshot_limit",
                        "detail": limit_reason,
                        "event": event.event,
                        "changed_files": [],
                        "closed_units": closed_units,
                    }

                current_commit = _base_commit(root)
                task = resolve_task(root, feature=feature) if not terminal else None
                _save_snapshot(
                    connection,
                    [str(row["unit_key"]) for row in active],
                    current,
                    current_skips,
                    current_commit,
                )
                same_identity = [
                    row
                    for row in active
                    if row["harness_id"] == event.harness_id
                    and row["native_session_id"] == event.native_session_id
                    and row["native_turn_id"] == event.native_turn_id
                    and row["native_agent_id"] == event.native_agent_id
                ]
                if model_changed and provider_changed:
                    boundary = "model_provider_change"
                elif model_changed:
                    boundary = "model_change"
                else:
                    boundary = "provider_change"
                boundary_exit_code = _event_exit_code(event) if terminal else 0
                for row in same_identity:
                    _record_workflow_switch(
                        connection,
                        str(row["attribution_session_id"]),
                        workflow,
                        str(row["model"]),
                        now,
                    )
                    _close_session(
                        connection,
                        str(row["attribution_session_id"]),
                        now,
                        exit_code=boundary_exit_code,
                        outcome=_event_outcome(event) if terminal else "completed",
                    )
                    connection.execute(
                        """
                        UPDATE hook_units
                        SET active = 0, last_event = ?, updated_at = ?
                        WHERE unit_key = ?
                        """,
                        (boundary, now, row["unit_key"]),
                    )
                _delete_snapshots(
                    connection,
                    [str(row["unit_key"]) for row in same_identity],
                )

                if terminal:
                    boundary_unit = same_identity[-1]
                    unit_key = str(boundary_unit["unit_key"])
                    session_id = str(boundary_unit["attribution_session_id"])
                else:
                    unit_key, session_id = _create_unit(
                        connection,
                        root,
                        worktree_id,
                        task,
                        event,
                        model,
                        harness_version,
                        effective_model_source,
                        now,
                        current_commit,
                    )
                    _save_snapshot(
                        connection,
                        [unit_key],
                        current,
                        current_skips,
                        current_commit,
                    )
                # The transition is unattributed, but the event itself still
                # happened. Its tool call and its counts belong to the session
                # that was running when it began, which is the session this
                # boundary closed.
                _apply_workflow(
                    connection,
                    worktree_id,
                    (
                        str(same_identity[-1]["attribution_session_id"])
                        if same_identity
                        else session_id
                    ),
                    event,
                    workflow,
                    now,
                )
                if database_fingerprint is not None:
                    connection.execute(
                        """
                        INSERT INTO hook_events(
                            fingerprint, unit_key, event_name, received_at
                        ) VALUES (?, ?, ?, ?)
                        """,
                        (database_fingerprint, unit_key, event.event, now),
                    )
                _touch_session_task(connection, session_id, now)
                connection.commit()
                return {
                    "session_id": session_id,
                    "unit_key": unit_key,
                    "event": event.event,
                    "native_session_id": event.native_session_id,
                    "native_turn_id": event.native_turn_id,
                    "baseline_created": not terminal,
                    "changed_files": [],
                    "ambiguous": True,
                    "reason": boundary,
                    "model_changed": model_changed,
                    "provider_changed": provider_changed,
                    "closed": terminal,
                    "closed_units": len(same_identity),
                    "skipped_files": [
                        {"path": path, "reason": reason}
                        for path, reason in sorted(current_skips.items())
                    ],
                }
            if unit is None and terminal:
                related = [
                    row
                    for row in active
                    if row["harness_id"] == event.harness_id
                    and row["native_session_id"] == event.native_session_id
                ]
                if len(related) == 1:
                    unit = related[0]
                elif not related:
                    return {
                        "ignored": True,
                        "reason": "orphan_terminal",
                        "event": event.event,
                        "changed_files": [],
                    }
                else:
                    current, current_skips, limit_reason = _bounded_snapshot(root)
                    if limit_reason is not None:
                        closed_units = _close_without_evidence(
                            connection,
                            active,
                            now,
                            "snapshot_limit",
                        )
                        connection.commit()
                        return {
                            "ignored": True,
                            "reason": "snapshot_limit",
                            "detail": limit_reason,
                            "event": event.event,
                            "changed_files": [],
                            "closed_units": closed_units,
                        }
                    current_commit = _base_commit(root)
                    _save_snapshot(
                        connection,
                        [str(row["unit_key"]) for row in active],
                        current,
                        current_skips,
                        current_commit,
                    )
                    exit_code = _event_exit_code(event)
                    for row in related:
                        _close_session(
                            connection,
                            str(row["attribution_session_id"]),
                            now,
                            exit_code=exit_code,
                            outcome=_event_outcome(event),
                        )
                        connection.execute(
                            """
                            UPDATE hook_units
                            SET active = 0, last_event = ?, updated_at = ?
                            WHERE unit_key = ?
                            """,
                            (event.event, now, row["unit_key"]),
                        )
                    _delete_snapshots(
                        connection,
                        [str(row["unit_key"]) for row in related],
                    )
                    unit_key = str(related[-1]["unit_key"])
                    if database_fingerprint is not None:
                        connection.execute(
                            """
                            INSERT INTO hook_events(
                                fingerprint, unit_key, event_name, received_at
                            ) VALUES (?, ?, ?, ?)
                            """,
                            (database_fingerprint, unit_key, event.event, now),
                        )
                    connection.commit()
                    return {
                        "event": event.event,
                        "native_session_id": event.native_session_id,
                        "changed_files": [],
                        "ambiguous": True,
                        "closed": True,
                        "closed_units": len(related),
                    }
            current, current_skips, limit_reason = _bounded_snapshot(root)
            if limit_reason is not None:
                closed_units = _close_without_evidence(
                    connection,
                    active,
                    now,
                    "snapshot_limit",
                )
                connection.commit()
                return {
                    "ignored": True,
                    "reason": "snapshot_limit",
                    "detail": limit_reason,
                    "event": event.event,
                    "changed_files": [],
                    "closed_units": closed_units,
                }
            current_commit = _base_commit(root)
            baseline_created = unit is None
            previous_model = None if unit is None else str(unit["model"])
            changed_files: list[str] = []
            ambiguous = False

            if unit is None:
                # Resolve only after a complete initial snapshot exists. This
                # avoids empty tasks for ignored or oversized hook events.
                task = resolve_task(root, feature=feature)
                unit_key, session_id = _create_unit(
                    connection,
                    root,
                    worktree_id,
                    task,
                    event,
                    model,
                    harness_version,
                    effective_model_source,
                    now,
                    current_commit,
                )
                _save_snapshot(
                    connection,
                    [unit_key],
                    current,
                    current_skips,
                    current_commit,
                )
                active = _active_units(connection, worktree_id)
            else:
                unit_key = str(unit["unit_key"])
                session_id = str(unit["attribution_session_id"])
                if unit["model"] == "unknown" and model != "unknown":
                    connection.execute(
                        "UPDATE hook_units SET model = ? WHERE unit_key = ?",
                        (model, unit_key),
                    )
                    connection.execute(
                        """
                        UPDATE sessions
                        SET model = ?, model_source = ?,
                            label_source = CASE
                                WHEN ? = 'reported' THEN 'mixed'
                                ELSE label_source
                            END
                        WHERE id = ? AND model_source = 'unknown'
                        """,
                        (
                            model,
                            effective_model_source,
                            effective_model_source,
                            session_id,
                        ),
                    )
                if event.provider is not None:
                    connection.execute(
                        "UPDATE sessions SET provider = COALESCE(provider, ?) WHERE id = ?",
                        (event.provider, session_id),
                    )
                if harness_version is not None:
                    connection.execute(
                        """
                        UPDATE sessions
                        SET harness_version = COALESCE(harness_version, ?)
                        WHERE id = ?
                        """,
                        (harness_version, session_id),
                    )
                before, before_skips = _load_snapshot(connection, unit_key)
                if _has_ambiguous_overlap(active, unit_key, event):
                    ambiguous = True
                elif not _transition_spans_at_most_one_commit(
                    root,
                    unit["baseline_commit"],
                    current_commit,
                ):
                    ambiguous = True
                else:
                    changed_files = _record_delta(
                        connection,
                        session_id,
                        before,
                        before_skips,
                        current,
                        current_skips,
                        unit["baseline_commit"],
                    )
                # Advance every active baseline. This prevents a parent unit from
                # claiming the same transition after a child or turn records it.
                _save_snapshot(
                    connection,
                    [str(row["unit_key"]) for row in active],
                    current,
                    current_skips,
                    current_commit,
                )

            _apply_workflow(connection, worktree_id, session_id, event, workflow, now)
            if event.event == "post_model_switch":
                _record_workflow_switch(
                    connection, session_id, workflow, previous_model, now
                )

            if terminal:
                exit_code = _event_exit_code(event)
                _close_session(
                    connection,
                    session_id,
                    now,
                    exit_code=exit_code,
                    outcome=_event_outcome(event),
                )
                connection.execute(
                    """
                    UPDATE hook_units
                    SET active = 0, last_event = ?, updated_at = ?
                    WHERE unit_key = ?
                    """,
                    (event.event, now, unit_key),
                )
                _delete_snapshots(connection, [unit_key])
            else:
                connection.execute(
                    """
                    UPDATE hook_units
                    SET last_event = ?, updated_at = ?
                    WHERE unit_key = ?
                    """,
                    (event.event, now, unit_key),
                )
                _touch_session_task(connection, session_id, now)

            if database_fingerprint is not None:
                connection.execute(
                    """
                    INSERT INTO hook_events(
                        fingerprint, unit_key, event_name, received_at
                    ) VALUES (?, ?, ?, ?)
                    """,
                    (database_fingerprint, unit_key, event.event, now),
                )
            connection.commit()
            return {
                "session_id": session_id,
                "unit_key": unit_key,
                "event": event.event,
                "native_session_id": event.native_session_id,
                "native_turn_id": event.native_turn_id,
                "baseline_created": baseline_created,
                "changed_files": changed_files,
                "ambiguous": ambiguous,
                "closed": terminal,
                "skipped_files": [
                    {"path": path, "reason": reason}
                    for path, reason in sorted(current_skips.items())
                ],
            }
        finally:
            connection.close()


def recover_hook_sessions(
    repo: RepoPath, *, capture_changes: bool = False
) -> dict[str, object]:
    """Close abandoned hook units, discarding later changes unless requested."""

    root = repository_root(repo)
    worktree_id = str(git_dir(root))
    now = _utc_now()
    with _native_capture_lock(root, worktree_id):
        connection = open_db(root)
        try:
            active = _active_units(connection, worktree_id)
            if not active:
                return {
                    "recovered": 0,
                    "changed_files": [],
                    "ambiguous": False,
                    "capture_changes": capture_changes,
                }
            changed_files: list[str] = []
            limit_reason = None
            ambiguous = len(active) != 1
            if capture_changes and not ambiguous:
                current, current_skips, limit_reason = _bounded_snapshot(root)
                ambiguous = limit_reason is not None
                if not ambiguous:
                    unit = active[0]
                    current_commit = _base_commit(root)
                    if not _transition_spans_at_most_one_commit(
                        root,
                        unit["baseline_commit"],
                        current_commit,
                    ):
                        ambiguous = True
                    else:
                        before, before_skips = _load_snapshot(
                            connection, str(unit["unit_key"])
                        )
                        changed_files = _record_delta(
                            connection,
                            str(unit["attribution_session_id"]),
                            before,
                            before_skips,
                            current,
                            current_skips,
                            unit["baseline_commit"],
                        )
            for unit in active:
                session_id = str(unit["attribution_session_id"])
                _close_session(
                    connection,
                    session_id,
                    now,
                    exit_code=None,
                    outcome="abandoned",
                )
                connection.execute(
                    """
                    UPDATE hook_units
                    SET active = 0, last_event = 'recovered', updated_at = ?
                    WHERE unit_key = ?
                    """,
                    (now, unit["unit_key"]),
                )
            _delete_snapshots(
                connection,
                [str(unit["unit_key"]) for unit in active],
            )
            connection.commit()
            result: dict[str, object] = {
                "recovered": len(active),
                "changed_files": changed_files,
                "ambiguous": ambiguous,
                "capture_changes": capture_changes,
            }
            if limit_reason is not None:
                result["reason"] = "snapshot_limit"
                result["detail"] = limit_reason
            return result
        finally:
            connection.close()
