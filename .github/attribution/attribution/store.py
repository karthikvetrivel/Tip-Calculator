"""SQLite storage for the local attribution ledger."""

from __future__ import annotations

import fcntl
import os
from pathlib import Path
import sqlite3
import subprocess
from typing import Union

from .runtime import system_subprocess_environment


RepoPath = Union[str, Path]
SUPPORTED_SCHEMA_VERSION = 6


_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    worktree_id TEXT NOT NULL,
    native_session_id TEXT NULL,
    native_turn_id TEXT NULL,
    native_agent_id TEXT NULL,
    native_parent_session_id TEXT NULL,
    task_id TEXT NULL,
    feature TEXT NOT NULL,
    feature_source TEXT NULL,
    model TEXT NOT NULL,
    harness TEXT NOT NULL,
    actor_kind TEXT NOT NULL DEFAULT 'ai' CHECK(actor_kind IN ('ai', 'manual')),
    source_session_id TEXT NULL,
    harness_id TEXT NULL,
    harness_version TEXT NULL,
    provider TEXT NULL,
    model_source TEXT NULL,
    harness_source TEXT NULL,
    integration_mode TEXT NULL,
    invocation_cwd TEXT NULL,
    label_source TEXT NOT NULL DEFAULT 'reported',
    membership_source TEXT NOT NULL DEFAULT 'legacy',
    role TEXT NOT NULL DEFAULT 'implementation',
    summary TEXT NULL,
    parent_session_id TEXT NULL,
    token_count INTEGER NULL,
    token_source TEXT NULL,
    cost_usd REAL NULL,
    cost_source TEXT NULL,
    usage_includes_children INTEGER NOT NULL DEFAULT 0,
    agent_type TEXT NULL,
    launch_mode TEXT NULL,
    session_source TEXT NULL,
    permission_mode TEXT NULL,
    effort_level TEXT NULL,
    turn_count INTEGER NOT NULL DEFAULT 0,
    prompt_count INTEGER NOT NULL DEFAULT 0,
    interrupt_count INTEGER NOT NULL DEFAULT 0,
    compaction_count INTEGER NOT NULL DEFAULT 0,
    model_switch_count INTEGER NOT NULL DEFAULT 0,
    tool_call_count INTEGER NOT NULL DEFAULT 0,
    duration_ms INTEGER NULL,
    activity_truncated INTEGER NOT NULL DEFAULT 0,
    trace_truncated INTEGER NOT NULL DEFAULT 0,
    started_at TEXT NOT NULL,
    ended_at TEXT NULL,
    base_commit TEXT NULL,
    exit_code INTEGER NULL,
    outcome TEXT NULL,
    FOREIGN KEY (task_id) REFERENCES tasks(id),
    FOREIGN KEY (parent_session_id) REFERENCES sessions(id)
);

CREATE TABLE IF NOT EXISTS tasks (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    kind TEXT NOT NULL DEFAULT 'feature',
    pr_ref TEXT NULL,
    pr_url TEXT NULL,
    branch TEXT NULL,
    state TEXT NOT NULL DEFAULT 'active',
    inference_source TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    anchor_commit TEXT NULL,
    merged_into TEXT NULL,
    FOREIGN KEY (merged_into) REFERENCES tasks(id)
);

CREATE TABLE IF NOT EXISTS task_contexts (
    worktree_id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL,
    branch TEXT NULL,
    head_commit TEXT NULL,
    pinned INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (task_id) REFERENCES tasks(id)
);

CREATE TABLE IF NOT EXISTS edits (
    id INTEGER PRIMARY KEY,
    session_id TEXT NOT NULL,
    path TEXT NOT NULL,
    before_content BLOB NULL,
    after_content BLOB NULL,
    before_hash TEXT NULL,
    after_hash TEXT NULL,
    base_commit TEXT NULL,
    tool_use_id TEXT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);

CREATE TABLE IF NOT EXISTS recorded_commits (
    commit_sha TEXT PRIMARY KEY,
    recorded_at TEXT NOT NULL,
    note_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS hook_sessions (
    worktree_id TEXT NOT NULL,
    harness TEXT NOT NULL,
    native_session_id TEXT NOT NULL,
    model TEXT NOT NULL,
    model_source TEXT NOT NULL,
    feature TEXT NOT NULL,
    feature_source TEXT NOT NULL,
    agent_type TEXT NULL,
    session_source TEXT NULL,
    permission_mode TEXT NULL,
    effort_level TEXT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (worktree_id, harness, native_session_id)
);

CREATE TABLE IF NOT EXISTS hook_captures (
    id TEXT PRIMARY KEY,
    worktree_id TEXT NOT NULL,
    harness TEXT NOT NULL,
    native_session_id TEXT NOT NULL,
    tool_use_id TEXT NOT NULL,
    turn_id TEXT NULL,
    ledger_session_id TEXT NOT NULL,
    base_commit TEXT NULL,
    started_at TEXT NOT NULL,
    status TEXT NOT NULL,
    UNIQUE (worktree_id, harness, native_session_id, tool_use_id),
    FOREIGN KEY (ledger_session_id) REFERENCES sessions(id)
);

CREATE TABLE IF NOT EXISTS hook_capture_files (
    capture_id TEXT NOT NULL,
    path TEXT NOT NULL,
    before_content BLOB NULL,
    before_hash TEXT NULL,
    skip_reason TEXT NULL,
    PRIMARY KEY (capture_id, path),
    FOREIGN KEY (capture_id) REFERENCES hook_captures(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS pending_commits (
    worktree_id TEXT NOT NULL,
    commit_sha TEXT NOT NULL,
    queued_at TEXT NOT NULL,
    PRIMARY KEY (worktree_id, commit_sha)
);

CREATE TABLE IF NOT EXISTS hook_units (
    unit_key TEXT PRIMARY KEY,
    worktree_id TEXT NOT NULL,
    harness_id TEXT NOT NULL,
    native_session_id TEXT NOT NULL,
    native_turn_id TEXT NULL,
    native_agent_id TEXT NULL,
    model TEXT NOT NULL,
    baseline_commit TEXT NULL,
    attribution_session_id TEXT NOT NULL REFERENCES sessions(id),
    last_event TEXT NOT NULL,
    active INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS hook_snapshot_files (
    unit_key TEXT NOT NULL REFERENCES hook_units(unit_key) ON DELETE CASCADE,
    path TEXT NOT NULL,
    content BLOB NULL,
    content_hash TEXT NULL,
    is_absent INTEGER NOT NULL DEFAULT 0,
    skip_reason TEXT NULL,
    PRIMARY KEY (unit_key, path)
);

CREATE TABLE IF NOT EXISTS hook_snapshot_blobs (
    content_hash TEXT PRIMARY KEY,
    content BLOB NOT NULL
);

CREATE TABLE IF NOT EXISTS hook_events (
    fingerprint TEXT PRIMARY KEY,
    unit_key TEXT NOT NULL REFERENCES hook_units(unit_key) ON DELETE CASCADE,
    event_name TEXT NOT NULL,
    received_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS model_switches (
    id INTEGER PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id),
    from_model TEXT NULL,
    to_model TEXT NOT NULL,
    source TEXT NOT NULL CHECK(
        source IN ('user', 'auto', 'resume', 'agent_response', 'unknown')
    ),
    occurred_at TEXT NOT NULL
);

-- child_session_id names a sessions(id) row that a later event may still
-- create, so it carries no foreign key of its own.
CREATE TABLE IF NOT EXISTS tool_calls (
    id INTEGER PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id),
    tool_use_id TEXT NOT NULL,
    turn_id TEXT NULL,
    sequence INTEGER NOT NULL,
    tool_name TEXT NOT NULL,
    tool_class TEXT NOT NULL CHECK(
        tool_class IN (
            'read', 'search', 'edit', 'write', 'shell', 'web', 'skill',
            'mcp', 'agent', 'ask_user', 'other'
        )
    ),
    locator TEXT NULL,
    locator_hash TEXT NULL,
    child_session_id TEXT NULL,
    succeeded INTEGER NULL,
    duration_ms INTEGER NULL,
    compaction_epoch INTEGER NOT NULL DEFAULT 0,
    occurred_at TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT 'harness_reported',
    UNIQUE (session_id, tool_use_id)
);

CREATE TABLE IF NOT EXISTS context_loads (
    id INTEGER PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id),
    kind TEXT NOT NULL CHECK(
        kind IN (
            'instruction_file', 'compaction_summary', 'subagent_result',
            'user_prompt', 'injected_context'
        )
    ),
    locator TEXT NULL,
    content_hash TEXT NULL,
    size_bytes INTEGER NULL,
    memory_type TEXT NULL,
    load_reason TEXT NULL,
    turn_id TEXT NULL,
    related_session_id TEXT NULL,
    compaction_epoch INTEGER NOT NULL DEFAULT 0,
    occurred_at TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT 'harness_reported'
);

-- The text of a session: what the user said, what the model replied, and
-- what each tool was given and returned. Every row is redacted and capped
-- before it is written, and ``sessions.trace_truncated`` marks a session
-- whose later rows the caps dropped.
CREATE TABLE IF NOT EXISTS trace_events (
    id INTEGER PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id),
    kind TEXT NOT NULL CHECK(
        kind IN (
            'user_prompt', 'assistant', 'tool_call', 'tool_result',
            'compaction', 'subagent_result'
        )
    ),
    tool_use_id TEXT NULL,
    agent_id TEXT NULL,
    turn_id TEXT NULL,
    occurred_at TEXT NOT NULL,
    payload TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    truncated INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS edits_session_id_idx ON edits(session_id);
CREATE INDEX IF NOT EXISTS edits_path_idx ON edits(path);
CREATE INDEX IF NOT EXISTS sessions_base_commit_idx ON sessions(base_commit);
CREATE INDEX IF NOT EXISTS hook_captures_pending_idx ON hook_captures(worktree_id, status);
CREATE INDEX IF NOT EXISTS pending_commits_queue_idx ON pending_commits(worktree_id, queued_at);
CREATE INDEX IF NOT EXISTS tasks_branch_idx ON tasks(branch);
CREATE INDEX IF NOT EXISTS hook_units_native_session_idx
    ON hook_units(worktree_id, harness_id, native_session_id);
CREATE INDEX IF NOT EXISTS hook_units_attribution_session_idx
    ON hook_units(attribution_session_id);
CREATE INDEX IF NOT EXISTS hook_units_active_idx
    ON hook_units(worktree_id, harness_id, active);
CREATE INDEX IF NOT EXISTS hook_events_unit_key_idx ON hook_events(unit_key);
CREATE INDEX IF NOT EXISTS model_switches_session_id_idx
    ON model_switches(session_id);
CREATE INDEX IF NOT EXISTS tool_calls_session_id_idx ON tool_calls(session_id);
CREATE INDEX IF NOT EXISTS tool_calls_child_session_id_idx
    ON tool_calls(child_session_id);
CREATE INDEX IF NOT EXISTS context_loads_session_id_idx
    ON context_loads(session_id);
CREATE INDEX IF NOT EXISTS trace_events_session_id_idx
    ON trace_events(session_id, id);
"""


_SESSION_COLUMN_MIGRATIONS = {
    # ``worktree_id`` predates schema versioning and must remain part of the
    # migration path for development ledgers created before it was added.
    "worktree_id": "TEXT",
    "native_session_id": "TEXT",
    "native_turn_id": "TEXT",
    "native_agent_id": "TEXT",
    "native_parent_session_id": "TEXT",
    "task_id": "TEXT NULL",
    "feature_source": "TEXT NULL",
    "actor_kind": "TEXT NOT NULL DEFAULT 'ai' CHECK(actor_kind IN ('ai', 'manual'))",
    "source_session_id": "TEXT NULL",
    "membership_source": "TEXT NOT NULL DEFAULT 'legacy'",
    "role": "TEXT NOT NULL DEFAULT 'implementation'",
    "summary": "TEXT NULL",
    "parent_session_id": "TEXT",
    "token_count": "INTEGER NULL",
    "token_source": "TEXT NULL",
    "usage_includes_children": "INTEGER NOT NULL DEFAULT 0",
    "outcome": "TEXT NULL",
    "harness_id": "TEXT",
    "harness_version": "TEXT",
    "provider": "TEXT",
    "model_source": "TEXT",
    "harness_source": "TEXT",
    "integration_mode": "TEXT",
    "invocation_cwd": "TEXT",
    # Schema 5 workflow facets. Every count carries a default so a populated
    # table can gain the column, and every unknown value stays NULL.
    "agent_type": "TEXT NULL",
    "launch_mode": "TEXT NULL",
    "session_source": "TEXT NULL",
    "permission_mode": "TEXT NULL",
    "effort_level": "TEXT NULL",
    "turn_count": "INTEGER NOT NULL DEFAULT 0",
    "prompt_count": "INTEGER NOT NULL DEFAULT 0",
    "interrupt_count": "INTEGER NOT NULL DEFAULT 0",
    "compaction_count": "INTEGER NOT NULL DEFAULT 0",
    "model_switch_count": "INTEGER NOT NULL DEFAULT 0",
    "tool_call_count": "INTEGER NOT NULL DEFAULT 0",
    "duration_ms": "INTEGER NULL",
    "activity_truncated": "INTEGER NOT NULL DEFAULT 0",
    # Schema 6 trace storage.
    "trace_truncated": "INTEGER NOT NULL DEFAULT 0",
}

_HOOK_SNAPSHOT_COLUMN_MIGRATIONS = {
    "content_hash": "TEXT",
}

_HOOK_UNIT_COLUMN_MIGRATIONS = {
    "baseline_commit": "TEXT",
}

# A session facet can arrive before the first tool event creates the ledger
# session it belongs to, so the installed receiver retains it here.
_HOOK_SESSION_COLUMN_MIGRATIONS = {
    "agent_type": "TEXT NULL",
    "session_source": "TEXT NULL",
    "permission_mode": "TEXT NULL",
    "effort_level": "TEXT NULL",
}


def _run_git(repo: RepoPath, *args: str) -> subprocess.CompletedProcess[bytes]:
    """Run a read-only Git query and retain stderr for a useful ValueError."""

    try:
        result = subprocess.run(
            ["git", "-C", str(repo), *args],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=system_subprocess_environment(),
            check=False,
        )
    except OSError as exc:
        raise ValueError(f"Could not run Git: {exc}") from exc
    if result.returncode:
        message = result.stderr.decode("utf-8", errors="replace").strip()
        raise ValueError(message or f"Git command failed with exit code {result.returncode}")
    return result


_git_path_cache: dict[tuple[str, str], Path] = {}


def _git_path(repo: RepoPath, flag: str) -> Path:
    """Resolve a ``git rev-parse`` path once per repository path and flag.

    Callers ask for the same directories many times per hook event, and each
    query costs a Git subprocess. A cached path is reused while it exists.
    """

    key = (os.path.abspath(repo), flag)
    cached = _git_path_cache.get(key)
    if cached is not None and cached.exists():
        return cached
    output = _run_git(repo, "rev-parse", "--path-format=absolute", flag).stdout
    path = Path(output.decode("utf-8", errors="surrogateescape").strip()).resolve()
    _git_path_cache[key] = path
    return path


def repository_root(repo: RepoPath) -> Path:
    """Return the absolute root of the supplied Git worktree."""

    return _git_path(repo, "--show-toplevel")


def git_common_dir(repo: RepoPath) -> Path:
    """Return the absolute common Git directory shared by linked worktrees."""

    return _git_path(repo, "--git-common-dir")


def git_dir(repo: RepoPath) -> Path:
    """Return the absolute per-worktree Git directory."""

    return _git_path(repo, "--git-dir")


def open_db(repo: RepoPath) -> sqlite3.Connection:
    """Open and initialize this repository's worktree-shared ledger."""

    database_dir = git_common_dir(repo) / "attribution"
    database_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    schema_lock_path = database_dir / "schema.lock"
    with schema_lock_path.open("a+b") as schema_lock:
        fcntl.flock(schema_lock.fileno(), fcntl.LOCK_EX)
        connection = sqlite3.connect(database_dir / "ledger.sqlite3", timeout=30)
        connection.row_factory = sqlite3.Row
        try:
            schema_version = int(
                connection.execute("PRAGMA user_version").fetchone()[0]
            )
            if schema_version > SUPPORTED_SCHEMA_VERSION:
                raise ValueError(
                    f"Joyride ledger schema {schema_version} is newer than "
                    f"supported schema {SUPPORTED_SCHEMA_VERSION}"
                )

            preexisting_session_columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(sessions)")
            }
            vendor_parent_migration = (
                "parent_session_id" in preexisting_session_columns
                and "task_id" not in preexisting_session_columns
                and "harness_id" in preexisting_session_columns
            )

            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("PRAGMA foreign_keys = ON")
            connection.executescript(_SCHEMA)

            # SQLite cannot add a NOT NULL column to a populated table without
            # a default. Migrated columns therefore remain nullable, while
            # fresh ledgers retain the stronger constraint declared in _SCHEMA.
            columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(sessions)")
            }
            for name, declaration in _SESSION_COLUMN_MIGRATIONS.items():
                if name not in columns:
                    connection.execute(
                        f"ALTER TABLE sessions ADD COLUMN {name} {declaration}"
                    )

            # The harness-only schema briefly stored a vendor parent identifier
            # in parent_session_id. Task economics reserves that field for a
            # local sessions(id) relationship, so move the legacy values aside.
            if vendor_parent_migration:
                connection.execute(
                    """
                    UPDATE sessions
                    SET native_parent_session_id = parent_session_id,
                        parent_session_id = NULL
                    WHERE parent_session_id IS NOT NULL
                    """
                )

            edit_columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(edits)")
            }
            if "base_commit" not in edit_columns:
                connection.execute("ALTER TABLE edits ADD COLUMN base_commit TEXT")
            if "tool_use_id" not in edit_columns:
                connection.execute("ALTER TABLE edits ADD COLUMN tool_use_id TEXT")

            capture_columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(hook_captures)")
            }
            if "turn_id" not in capture_columns:
                connection.execute("ALTER TABLE hook_captures ADD COLUMN turn_id TEXT")

            snapshot_columns = {
                row["name"]
                for row in connection.execute(
                    "PRAGMA table_info(hook_snapshot_files)"
                )
            }
            for name, declaration in _HOOK_SNAPSHOT_COLUMN_MIGRATIONS.items():
                if name not in snapshot_columns:
                    connection.execute(
                        "ALTER TABLE hook_snapshot_files "
                        f"ADD COLUMN {name} {declaration}"
                    )

            hook_session_columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(hook_sessions)")
            }
            for name, declaration in _HOOK_SESSION_COLUMN_MIGRATIONS.items():
                if name not in hook_session_columns:
                    connection.execute(
                        f"ALTER TABLE hook_sessions ADD COLUMN {name} {declaration}"
                    )

            unit_columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(hook_units)")
            }
            for name, declaration in _HOOK_UNIT_COLUMN_MIGRATIONS.items():
                if name not in unit_columns:
                    connection.execute(
                        f"ALTER TABLE hook_units ADD COLUMN {name} {declaration}"
                    )
            connection.execute(
                """
                UPDATE hook_units
                SET baseline_commit = (
                    SELECT sessions.base_commit
                    FROM sessions
                    WHERE sessions.id = hook_units.attribution_session_id
                )
                WHERE baseline_commit IS NULL
                """
            )

            connection.execute(
                "CREATE INDEX IF NOT EXISTS edits_base_commit_idx ON edits(base_commit)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS hook_captures_turn_idx "
                "ON hook_captures(harness, native_session_id, turn_id)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS sessions_task_id_idx ON sessions(task_id)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS sessions_parent_session_id_idx "
                "ON sessions(parent_session_id)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS sessions_native_parent_session_id_idx "
                "ON sessions(native_parent_session_id)"
            )

            # Scope pre-worktree rows only during their migration. If several
            # worktrees exist, keep them unknown forever. Assigning them on a
            # later open after another worktree is removed could invent scope.
            if schema_version < SUPPORTED_SCHEMA_VERSION:
                try:
                    worktrees = [
                        line
                        for line in _run_git(
                            repo, "worktree", "list", "--porcelain"
                        ).stdout.splitlines()
                        if line.startswith(b"worktree ")
                    ]
                except ValueError:
                    worktrees = []
                if len(worktrees) == 1:
                    connection.execute(
                        """
                        UPDATE sessions
                        SET worktree_id = ?
                        WHERE worktree_id IS NULL
                        """,
                        (str(git_dir(repo)),),
                    )

            connection.execute(
                f"PRAGMA user_version = {SUPPORTED_SCHEMA_VERSION}"
            )
            connection.commit()
            return connection
        except BaseException:
            connection.close()
            raise
        finally:
            fcntl.flock(schema_lock.fileno(), fcntl.LOCK_UN)
