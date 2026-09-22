"""Private, fail-open Codex/Claude hooks for exact structured edit evidence."""

from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
import time
import uuid

from .capture import _base_commit, _content_hash, _utc_now
from .edit_evidence import expected_edit, read_targets
from .hook_events import classify_tool, normalize_event
from .store import git_dir, open_db, repository_root
from .tasks import resolve_task


_MAX_EVENT_BYTES = 8 * 1024 * 1024
_MAX_PENDING = 64
_SCHEMA = """
CREATE TABLE IF NOT EXISTS structured_hook_sessions (
    id TEXT PRIMARY KEY,
    worktree_id TEXT NOT NULL,
    model TEXT NULL
);
CREATE TABLE IF NOT EXISTS structured_hook_calls (
    id TEXT PRIMARY KEY,
    source_session_id TEXT NOT NULL,
    worktree_id TEXT NOT NULL,
    model TEXT NOT NULL,
    base_commit TEXT NULL,
    started_at TEXT NOT NULL,
    created REAL NOT NULL,
    kind TEXT NOT NULL,
    pending INTEGER NOT NULL,
    blocked INTEGER NOT NULL,
    before_json TEXT NULL,
    expected_json TEXT NULL
);
CREATE INDEX IF NOT EXISTS structured_hook_calls_pending_idx
    ON structured_hook_calls(worktree_id, pending);
"""


def _identity(*values: str) -> str:
    return hashlib.sha256("\0".join(values).encode()).hexdigest()


def _pack(values: dict[str, bytes | None]) -> str:
    return json.dumps({path: base64.b64encode(content).decode() if content is not None else None
                       for path, content in values.items()}, sort_keys=True)


def _unpack(value: str | None) -> dict[str, bytes | None]:
    return {path: base64.b64decode(content, validate=True) if content is not None else None
            for path, content in json.loads(value or "{}").items()}


def _finish(connection: sqlite3.Connection, call_id: str) -> None:
    # Keep a small receipt for duplicate-event protection; release snapshots.
    connection.execute(
        "UPDATE structured_hook_calls "
        "SET pending = 0, before_json = NULL, expected_json = NULL WHERE id = ?",
        (call_id,),
    )


def _background(event: dict) -> bool:
    supplied = event.get("tool_input") or {}
    response = event.get("tool_response")
    return bool(supplied.get("run_in_background") or (isinstance(response, dict) and any(
        response.get(key) for key in ("backgroundTaskId", "background_task_id", "task_id", "shell_id")
    )))


def handle_event(provider: str, payload: dict) -> dict:
    """Consume one event without executing tool input or inferring manual edits."""
    if os.environ.get("ATTRIBUTION_WRAPPED_CAPTURE") == "1":
        return {"status": "ignored"}
    event = normalize_event(provider, payload)
    if event is None:
        return {"status": "ignored"}
    try:
        root = repository_root(event["cwd"])
    except (OSError, ValueError):
        return {"status": "ignored"}
    try:
        task = resolve_task(root)
    except (OSError, ValueError):
        return {"status": "ignored"}
    worktree = str(git_dir(root))
    provider = event["provider"]
    group = _identity(provider, worktree, event["external_session_id"])
    source = group + ":" + _identity(event["agent_id"]) if event.get("agent_id") else group
    phase = event["event"]
    call_id = _identity(source, event["tool_use_id"]) if event.get("tool_use_id") else None
    kind = classify_tool(provider, event.get("tool_name"))
    if phase in {"pre", "post", "failure"} and (call_id is None or kind == "other"):
        return {"status": "ignored"}
    connection = open_db(root)
    connection.row_factory = sqlite3.Row
    try:
        connection.executescript(_SCHEMA)
        connection.execute("BEGIN IMMEDIATE")
        row = connection.execute(
            "SELECT model FROM structured_hook_sessions WHERE id = ?", (source,)
        ).fetchone()
        model = event.get("model") or (row["model"] if row else None)
        if phase in {"start", "model"}:
            model = event.get("model")
            connection.execute(
                "INSERT INTO structured_hook_sessions(id, worktree_id, model) VALUES (?, ?, ?) "
                "ON CONFLICT(id) DO UPDATE SET model = excluded.model",
                (source, worktree, model),
            )
            connection.commit()
            return {"status": "started"}
        if phase == "end":
            connection.execute(
                "UPDATE structured_hook_calls "
                "SET pending = 0, before_json = NULL, expected_json = NULL "
                "WHERE (source_session_id = ? OR source_session_id LIKE ?) AND pending = 1",
                (source, source + ":%"),
            )
            connection.commit()
            return {"status": "ended"}
        call = connection.execute(
            "SELECT * FROM structured_hook_calls WHERE id = ?", (call_id,)
        ).fetchone()
        if phase == "pre":
            if call is not None:
                connection.commit()
                return {"status": "ignored"}
            pending = connection.execute(
                "SELECT * FROM structured_hook_calls "
                "WHERE worktree_id = ? AND pending = 1",
                (worktree,),
            ).fetchall()
            if len(pending) >= _MAX_PENDING:
                connection.execute(
                    "UPDATE structured_hook_calls "
                    "SET blocked = 1 WHERE worktree_id = ? AND pending = 1",
                    (worktree,),
                )
                # The tool still runs after our hook returns. Keep a small
                # barrier receipt even when we cannot afford another snapshot.
                connection.execute(
                    "INSERT INTO structured_hook_calls "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, 1, NULL, NULL)",
                    (call_id, source, worktree, model or "Unknown model", _base_commit(root), _utc_now(),
                     time.time(), kind),
                )
                connection.commit()
                return {"status": "unknown"}
            evidence = expected_edit(provider, event["tool_name"], event["tool_input"], Path(event["cwd"]), root)
            before, expected = evidence if evidence is not None else ({}, {})
            blocked = evidence is None
            for previous in pending:
                previous_paths = set(_unpack(previous["expected_json"]))
                if not expected or not previous_paths or previous_paths.intersection(expected):
                    blocked = True
                    connection.execute(
                        "UPDATE structured_hook_calls SET blocked = 1 WHERE id = ?",
                        (previous["id"],),
                    )
            connection.execute(
                "INSERT INTO structured_hook_calls "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?)",
                (call_id, source, worktree, model or "Unknown model", _base_commit(root), _utc_now(),
                 time.time(), kind, int(blocked), _pack(before), _pack(expected)),
            )
            connection.commit()
            return {"status": "started"}
        if call is None or not call["pending"]:
            connection.commit()
            return {"status": "ignored"}
        if call["kind"] == "shell" and _background(event):
            # A returned background shell is still capable of writing files.
            connection.commit()
            return {"status": "unknown"}
        if phase == "failure" or event.get("succeeded") is not True or call["blocked"]:
            _finish(connection, call_id)
            connection.commit()
            return {"status": "unknown"}
        before, expected = _unpack(call["before_json"]), _unpack(call["expected_json"])
        observed = read_targets(root, expected)
        if not expected or observed != expected or _base_commit(root) != call["base_commit"]:
            _finish(connection, call_id)
            connection.commit()
            return {"status": "unknown"}
        changed = [path for path in expected if before[path] != expected[path]]
        if not changed:
            _finish(connection, call_id)
            connection.commit()
            return {"status": "ignored"}
        # Segments keep commit-local reconciliation intact. source_session_id
        # lets the public report count one native session across those segments.
        session_id = str(uuid.uuid5(uuid.NAMESPACE_URL, "\0".join(
            (source, call["model"], call["base_commit"] or "root")
        )))
        now = _utc_now()
        harness_id = "codex" if provider == "codex" else "claude-code"
        model_source = "unknown" if call["model"] == "Unknown model" else "native_hook"
        connection.execute(
            """
            INSERT INTO sessions(
                id, worktree_id, task_id, feature, model, harness, actor_kind,
                source_session_id, label_source, membership_source, role,
                cost_usd, cost_source, started_at, ended_at, base_commit,
                exit_code, outcome, harness_id, model_source,
                harness_source, integration_mode
            ) VALUES (
                ?, ?, ?, ?, ?, ?, 'ai', ?, 'tool_hook', ?, 'implementation',
                NULL, NULL, ?, ?, ?, 0, 'completed', ?, ?,
                'native_hook', 'native_hook'
            )
            ON CONFLICT(id) DO UPDATE SET ended_at = excluded.ended_at
            """,
            (
                session_id,
                worktree,
                task["id"],
                task["name"],
                call["model"],
                "Codex" if provider == "codex" else "Claude Code",
                source,
                task["membership_source"],
                call["started_at"],
                now,
                call["base_commit"],
                harness_id,
                model_source,
            ),
        )
        for path in changed:
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
                    before[path],
                    expected[path],
                    _content_hash(before[path]),
                    _content_hash(expected[path]),
                    call["base_commit"],
                ),
            )
        _finish(connection, call_id)
        connection.commit()
        return {"status": "captured", "session_id": session_id}
    finally:
        connection.close()


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    if len(arguments) != 1:
        return 0
    try:
        raw = sys.stdin.buffer.read(_MAX_EVENT_BYTES + 1)
        if len(raw) <= _MAX_EVENT_BYTES:
            handle_event(arguments[0], json.loads(raw))
    except Exception:
        # Hook errors must not block coding or echo prompts, paths or tool input.
        print("Joyride: this tool edit could not be captured.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
