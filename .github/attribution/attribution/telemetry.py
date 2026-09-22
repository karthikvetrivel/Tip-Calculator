"""Privacy-preserving local OTLP/HTTP cost telemetry ingestion.

Only a small, explicit metadata allowlist is decoded and persisted.  In
particular, log bodies (other than an exact event-name match), prompts,
responses, tool output, and the original OTLP payload are never stored.

Codex ChatGPT credit rates
--------------------------

The rates below are effective 2026-09-14 and are expressed as ChatGPT credits
per one million uncached-input / cached-input / output tokens.  They are not
API prices and must never be reported as US dollars::

    gpt-6-astra       250 / 25    / 1250
    gpt-5.6-sol       100 / 10    / 500
    gpt-5.6-terra      50 /  5    / 300
    gpt-5.6-luna        5 /  0.5  /  30
    gpt-5.5            125 / 12.5 / 750
    gpt-5.4           62.5 /  6.25/ 375
    gpt-5.4-mini     18.75 /  1.875/ 113

There are no filesystem writes at import time.  The default state directory
can be overridden with ``ATTRIBUTION_TELEMETRY_DIR`` or passed to every public
entry point, which also makes tests independent of a user's real state.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import errno
import fcntl
import hashlib
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import ipaddress
import json
import math
import os
from pathlib import Path
import re
import secrets
import signal
import socket
import sqlite3
import stat
import subprocess
import threading
import time
from typing import Any, Iterator, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from .runtime import current_runtime


STATE_DIR_ENV = "ATTRIBUTION_TELEMETRY_DIR"
DEFAULT_OTLP_PORT = 4318
DEFAULT_MAX_REQUEST_BYTES = 1_048_576
SQLITE_INTEGER_MAX = 9_223_372_036_854_775_807
_COLLECTOR_IDENTITY = "harness-attribution-telemetry-v1"
_COLLECTOR_LOCK_FILENAME = "collector-startup.lock"
_MAX_IDENTITY_RESPONSE_BYTES = 4096
CODEX_CHATGPT_CREDIT_RATES_EFFECTIVE_DATE = "2026-09-14"
OPENAI_API_STANDARD_RATES_EFFECTIVE_DATE = "2026-09-14"
CODEX_CHATGPT_CREDIT_RATES: dict[str, dict[str, Decimal]] = {
    "gpt-6-astra": {
        "input": Decimal("250"),
        "cached_input": Decimal("25"),
        "output": Decimal("1250"),
    },
    "gpt-5.6-sol": {
        "input": Decimal("100"),
        "cached_input": Decimal("10"),
        "output": Decimal("500"),
    },
    "gpt-5.6-terra": {
        "input": Decimal("50"),
        "cached_input": Decimal("5"),
        "output": Decimal("300"),
    },
    "gpt-5.6-luna": {
        "input": Decimal("5"),
        "cached_input": Decimal("0.5"),
        "output": Decimal("30"),
    },
    "gpt-5.5": {
        "input": Decimal("125"),
        "cached_input": Decimal("12.5"),
        "output": Decimal("750"),
    },
    "gpt-5.4": {
        "input": Decimal("62.5"),
        "cached_input": Decimal("6.25"),
        "output": Decimal("375"),
    },
    "gpt-5.4-mini": {
        "input": Decimal("18.75"),
        "cached_input": Decimal("1.875"),
        "output": Decimal("113"),
    },
}
OPENAI_API_STANDARD_RATES: dict[str, dict[str, Decimal]] = {
    "gpt-6-astra": {"input": Decimal("10"), "cached_input": Decimal("1"), "output": Decimal("50")},
    "gpt-5.6-sol": {"input": Decimal("4"), "cached_input": Decimal("0.4"), "output": Decimal("20")},
    "gpt-5.6-terra": {"input": Decimal("2"), "cached_input": Decimal("0.2"), "output": Decimal("12")},
    "gpt-5.6-luna": {"input": Decimal("0.2"), "cached_input": Decimal("0.02"), "output": Decimal("1.2")},
    "gpt-5.5": {"input": Decimal("5"), "cached_input": Decimal("0.5"), "output": Decimal("30")},
    "gpt-5.4": {"input": Decimal("2.5"), "cached_input": Decimal("0.25"), "output": Decimal("15")},
    "gpt-5.4-mini": {"input": Decimal("0.75"), "cached_input": Decimal("0.075"), "output": Decimal("4.5")},
}

# Retaining handles for children started by this process avoids premature
# Popen finalization warnings and lets stop_collector reap them cleanly.  The
# mapping has no external side effects and remains empty until startup.
_COLLECTOR_CHILDREN: dict[str, subprocess.Popen[Any]] = {}


_SCHEMA = """
CREATE TABLE IF NOT EXISTS registered_sessions (
    provider TEXT NOT NULL,
    native_session_id TEXT NOT NULL,
    repository_id TEXT NOT NULL,
    repository_path TEXT NULL,
    active INTEGER NOT NULL DEFAULT 1,
    registered_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (provider, native_session_id, repository_id)
);

CREATE TABLE IF NOT EXISTS telemetry_events (
    id INTEGER PRIMARY KEY,
    event_key TEXT NOT NULL UNIQUE,
    provider TEXT NOT NULL,
    event_name TEXT NOT NULL,
    event_kind TEXT NULL,
    native_session_id TEXT NOT NULL,
    prompt_id TEXT NULL,
    turn_id TEXT NULL,
    request_id TEXT NULL,
    client_request_id TEXT NULL,
    model TEXT NULL,
    auth_mode TEXT NULL,
    service_tier TEXT NULL,
    input_tokens INTEGER NULL,
    cached_input_tokens INTEGER NULL,
    cache_creation_input_tokens INTEGER NULL,
    output_tokens INTEGER NULL,
    total_tokens INTEGER NULL,
    cost_amount TEXT NULL,
    cost_unit TEXT NULL,
    cost_source TEXT NULL,
    query_source TEXT NULL,
    agent_id TEXT NULL,
    parent_agent_id TEXT NULL,
    agent_name TEXT NULL,
    effort TEXT NULL,
    observed_at_unix_nano INTEGER NULL,
    received_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS telemetry_event_repositories (
    event_key TEXT NOT NULL,
    repository_id TEXT NOT NULL,
    repository_path TEXT NULL,
    PRIMARY KEY (event_key, repository_id),
    FOREIGN KEY (event_key) REFERENCES telemetry_events(event_key) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS claude_prompt_tool_links (
    link_key TEXT NOT NULL UNIQUE,
    native_session_id TEXT NOT NULL,
    prompt_id TEXT NOT NULL,
    tool_use_id TEXT NOT NULL,
    observed_at_unix_nano INTEGER NULL,
    received_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS claude_prompt_tool_link_repositories (
    link_key TEXT NOT NULL,
    repository_id TEXT NOT NULL,
    repository_path TEXT NULL,
    PRIMARY KEY (link_key, repository_id),
    FOREIGN KEY (link_key) REFERENCES claude_prompt_tool_links(link_key) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS telemetry_events_session_idx
    ON telemetry_events(provider, native_session_id);
CREATE INDEX IF NOT EXISTS telemetry_event_repositories_repository_idx
    ON telemetry_event_repositories(repository_id);
CREATE INDEX IF NOT EXISTS registered_sessions_repository_idx
    ON registered_sessions(repository_id);
CREATE INDEX IF NOT EXISTS claude_prompt_tool_links_prompt_idx
    ON claude_prompt_tool_links(native_session_id, prompt_id);
CREATE INDEX IF NOT EXISTS claude_prompt_tool_link_repositories_repository_idx
    ON claude_prompt_tool_link_repositories(repository_id);
"""


class TelemetryError(ValueError):
    """Raised when telemetry input or configuration is invalid."""


@dataclass(frozen=True)
class _NormalizedEvent:
    event_key: str
    provider: str
    event_name: str
    event_kind: str | None
    native_session_id: str
    prompt_id: str | None = None
    turn_id: str | None = None
    request_id: str | None = None
    client_request_id: str | None = None
    model: str | None = None
    auth_mode: str | None = None
    service_tier: str | None = None
    input_tokens: int | None = None
    cached_input_tokens: int | None = None
    cache_creation_input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    cost_amount: str | None = None
    cost_unit: str | None = None
    cost_source: str | None = None
    query_source: str | None = None
    agent_id: str | None = None
    parent_agent_id: str | None = None
    agent_name: str | None = None
    effort: str | None = None
    observed_at_unix_nano: int | None = None
    tool_use_id: str | None = None


def default_state_dir() -> Path:
    """Return the configured global telemetry directory without creating it."""

    override = os.environ.get(STATE_DIR_ENV)
    if override:
        return Path(override).expanduser().resolve()
    state_home = os.environ.get("XDG_STATE_HOME")
    base = Path(state_home).expanduser() if state_home else Path.home() / ".local" / "state"
    return (base / "harness-attribution" / "telemetry").resolve()


def _state_dir_path(state_dir: str | os.PathLike[str] | None) -> Path:
    return Path(state_dir).expanduser().resolve() if state_dir is not None else default_state_dir()


def secure_state_dir(state_dir: str | os.PathLike[str] | None = None) -> Path:
    """Create and return a private state directory.

    The leaf may not be a symbolic link.  POSIX permissions are narrowed to
    user-only access even when the directory existed before this call.
    """

    path = _state_dir_path(state_dir)
    if path.is_symlink():
        raise TelemetryError(f"Telemetry state directory may not be a symlink: {path}")
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    if not path.is_dir():
        raise TelemetryError(f"Telemetry state path is not a directory: {path}")
    if os.name == "posix":
        os.chmod(path, 0o700)
    return path


def _normalize_provider(provider: str) -> str:
    value = str(provider).strip().casefold()
    aliases = {"anthropic": "claude", "claude-code": "claude", "openai": "codex"}
    value = aliases.get(value, value)
    if value not in {"claude", "codex"}:
        raise TelemetryError("provider must be 'claude' or 'codex'")
    return value


def _bounded_text(value: Any, maximum: int = 512) -> str | None:
    if value is None or isinstance(value, (dict, list, tuple, set)):
        return None
    text = str(value).strip()
    if not text or len(text) > maximum or "\x00" in text:
        return None
    return text


class TelemetryStore:
    """SQLite-backed store containing normalized, registered-session metadata."""

    def __init__(self, state_dir: str | os.PathLike[str] | None = None) -> None:
        self.state_dir = secure_state_dir(state_dir)
        self.database_path = self.state_dir / "telemetry.sqlite3"
        with self._connection() as connection:
            existing_tables = {
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                )
            }
            connection.executescript(_SCHEMA)
            registration_columns = {
                row[1]
                for row in connection.execute("PRAGMA table_info(registered_sessions)")
            }
            if "active" not in registration_columns:
                connection.execute(
                    "ALTER TABLE registered_sessions "
                    "ADD COLUMN active INTEGER NOT NULL DEFAULT 1"
                )
            columns = {
                row[1]
                for row in connection.execute("PRAGMA table_info(telemetry_events)")
            }
            if "service_tier" not in columns:
                connection.execute(
                    "ALTER TABLE telemetry_events ADD COLUMN service_tier TEXT"
                )
            for workflow_column in (
                "query_source",
                "agent_id",
                "parent_agent_id",
                "agent_name",
                "effort",
            ):
                if workflow_column not in columns:
                    connection.execute(
                        f"ALTER TABLE telemetry_events ADD COLUMN {workflow_column} TEXT"
                    )
            if "telemetry_event_repositories" not in existing_tables:
                connection.execute(
                    """
                    INSERT OR IGNORE INTO telemetry_event_repositories (
                        event_key, repository_id, repository_path
                    )
                    SELECT e.event_key, r.repository_id, r.repository_path
                    FROM telemetry_events e
                    JOIN registered_sessions r
                      ON r.provider = e.provider
                     AND r.native_session_id = e.native_session_id
                    """
                )
            if "claude_prompt_tool_link_repositories" not in existing_tables:
                connection.execute(
                    """
                    INSERT OR IGNORE INTO claude_prompt_tool_link_repositories (
                        link_key, repository_id, repository_path
                    )
                    SELECT l.link_key, r.repository_id, r.repository_path
                    FROM claude_prompt_tool_links l
                    JOIN registered_sessions r
                      ON r.provider = 'claude'
                     AND r.native_session_id = l.native_session_id
                    """
                )
        if os.name == "posix":
            os.chmod(self.database_path, 0o600)

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        try:
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def register_session(
        self,
        provider: str,
        native_session_id: str,
        repository_id: str | os.PathLike[str],
        repository_path: str | os.PathLike[str] | None = None,
    ) -> dict[str, str | None]:
        provider_name = _normalize_provider(provider)
        session = _bounded_text(native_session_id)
        repository = _bounded_text(repository_id, 2048)
        if session is None:
            raise TelemetryError("native_session_id must be a non-empty short string")
        if repository is None:
            raise TelemetryError("repository_id must be a non-empty short string")
        path = _bounded_text(repository_path, 4096)
        if path is None and isinstance(repository_id, os.PathLike):
            path = str(Path(repository_id).expanduser().resolve())
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO registered_sessions (
                    provider, native_session_id, repository_id, repository_path, active
                ) VALUES (?, ?, ?, ?, 1)
                ON CONFLICT(provider, native_session_id, repository_id) DO UPDATE SET
                    repository_path = excluded.repository_path,
                    active = 1,
                    registered_at = CURRENT_TIMESTAMP
                """,
                (provider_name, session, repository, path),
            )
        return {
            "provider": provider_name,
            "native_session_id": session,
            "repository_id": repository,
            "repository_path": path,
        }

    def unregister_repository(self, repository_id: str | os.PathLike[str]) -> int:
        """Deactivate ingestion while retaining the historical repository link."""

        repository = _bounded_text(repository_id, 4096)
        if repository is None:
            raise TelemetryError("repository_id must be a non-empty short string")
        with self._connection() as connection:
            cursor = connection.execute(
                """
                UPDATE registered_sessions
                SET active = 0
                WHERE (repository_id = ? OR repository_path = ?) AND active = 1
                """,
                (repository, repository),
            )
            return cursor.rowcount

    def registered_native_sessions(
        self, repository_id: str | os.PathLike[str], provider: str | None = None
    ) -> list[dict[str, str]]:
        repository = _bounded_text(repository_id, 4096)
        if repository is None:
            raise TelemetryError("repository_id must be a non-empty short string")
        clauses = ["(repository_id = ? OR repository_path = ?)", "active = 1"]
        parameters: list[Any] = [repository, repository]
        if provider is not None:
            clauses.append("provider = ?")
            parameters.append(_normalize_provider(provider))
        with self._connection() as connection:
            rows = connection.execute(
                f"""
                SELECT provider, native_session_id, repository_id, repository_path
                FROM registered_sessions
                WHERE {' AND '.join(clauses)}
                ORDER BY provider, native_session_id
                """,
                parameters,
            ).fetchall()
        return [dict(row) for row in rows]

    def registered_repository_count(self) -> int:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT COUNT(DISTINCT repository_id) FROM registered_sessions "
                "WHERE active = 1"
            ).fetchone()
        return int(row[0])

    def _active_repositories(
        self, connection: sqlite3.Connection, provider: str, native_session_id: str
    ) -> list[sqlite3.Row]:
        return connection.execute(
            """
            SELECT repository_id, repository_path
            FROM registered_sessions
            WHERE provider = ? AND native_session_id = ? AND active = 1
            ORDER BY repository_id
            """,
            (provider, native_session_id),
        ).fetchall()

    def ingest_otlp(self, payload: Mapping[str, Any] | str | bytes) -> dict[str, int]:
        """Normalize and persist registered OTLP JSON log records idempotently."""

        if isinstance(payload, bytes):
            try:
                payload = json.loads(payload.decode("utf-8"))
            except (UnicodeDecodeError, ValueError, RecursionError) as exc:
                raise TelemetryError("body is not valid UTF-8 OTLP JSON") from exc
        elif isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except (ValueError, RecursionError) as exc:
                raise TelemetryError("body is not valid OTLP JSON") from exc
        if not isinstance(payload, Mapping):
            raise TelemetryError("OTLP JSON payload must be an object")

        stats = {
            "accepted": 0,
            "duplicates": 0,
            "ignored": 0,
            "unregistered": 0,
            "ambiguous": 0,
            "links_added": 0,
        }
        parsed_records = list(_parse_otlp_records(payload))
        if not parsed_records and not _has_log_container(payload):
            raise TelemetryError("payload does not contain OTLP log records")

        with self._connection() as connection:
            for parsed in parsed_records:
                if parsed is None:
                    stats["ignored"] += 1
                    continue
                repositories = self._active_repositories(
                    connection, parsed.provider, parsed.native_session_id
                )
                if not repositories:
                    stats["ignored"] += 1
                    stats["unregistered"] += 1
                    continue
                if len(repositories) != 1:
                    # OTLP records have no repository identity. Attaching one
                    # record to several active repositories would leak and
                    # double-count it, so ambiguous native IDs fail closed.
                    stats["ignored"] += 1
                    stats["ambiguous"] += 1
                    continue
                cursor = connection.execute(
                    """
                    INSERT OR IGNORE INTO telemetry_events (
                        event_key, provider, event_name, event_kind,
                        native_session_id, prompt_id, turn_id, request_id,
                        client_request_id, model, auth_mode, service_tier,
                        input_tokens,
                        cached_input_tokens, cache_creation_input_tokens,
                        output_tokens, total_tokens, cost_amount, cost_unit,
                        cost_source, query_source, agent_id, parent_agent_id,
                        agent_name, effort, observed_at_unix_nano
                    ) VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                        ?, ?, ?, ?, ?, ?, ?
                    )
                    """,
                    (
                        parsed.event_key,
                        parsed.provider,
                        parsed.event_name,
                        parsed.event_kind,
                        parsed.native_session_id,
                        parsed.prompt_id,
                        parsed.turn_id,
                        parsed.request_id,
                        parsed.client_request_id,
                        parsed.model,
                        parsed.auth_mode,
                        parsed.service_tier,
                        parsed.input_tokens,
                        parsed.cached_input_tokens,
                        parsed.cache_creation_input_tokens,
                        parsed.output_tokens,
                        parsed.total_tokens,
                        parsed.cost_amount,
                        parsed.cost_unit,
                        parsed.cost_source,
                        parsed.query_source,
                        parsed.agent_id,
                        parsed.parent_agent_id,
                        parsed.agent_name,
                        parsed.effort,
                        parsed.observed_at_unix_nano,
                    ),
                )
                if cursor.rowcount:
                    stats["accepted"] += 1
                    connection.executemany(
                        """
                        INSERT OR IGNORE INTO telemetry_event_repositories (
                            event_key, repository_id, repository_path
                        ) VALUES (?, ?, ?)
                        """,
                        (
                            (
                                parsed.event_key,
                                repository["repository_id"],
                                repository["repository_path"],
                            )
                            for repository in repositories
                        ),
                    )
                else:
                    stats["duplicates"] += 1

                if parsed.tool_use_id and parsed.prompt_id:
                    link_key = _digest_key(
                        "claude-tool-link",
                        parsed.native_session_id,
                        parsed.prompt_id,
                        parsed.tool_use_id,
                    )
                    link_cursor = connection.execute(
                        """
                        INSERT OR IGNORE INTO claude_prompt_tool_links (
                            link_key, native_session_id, prompt_id, tool_use_id,
                            observed_at_unix_nano
                        ) VALUES (?, ?, ?, ?, ?)
                        """,
                        (
                            link_key,
                            parsed.native_session_id,
                            parsed.prompt_id,
                            parsed.tool_use_id,
                            parsed.observed_at_unix_nano,
                        ),
                    )
                    stats["links_added"] += max(link_cursor.rowcount, 0)
                    if link_cursor.rowcount:
                        connection.executemany(
                            """
                            INSERT OR IGNORE INTO claude_prompt_tool_link_repositories (
                                link_key, repository_id, repository_path
                            ) VALUES (?, ?, ?)
                            """,
                            (
                                (
                                    link_key,
                                    repository["repository_id"],
                                    repository["repository_path"],
                                )
                                for repository in repositories
                            ),
                        )
        return stats

    def query_events(
        self,
        provider: str | None = None,
        native_session_id: str | None = None,
        repository_id: str | os.PathLike[str] | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return normalized events, optionally scoped to a registered repository."""

        clauses: list[str] = []
        parameters: list[Any] = []
        if provider is not None:
            clauses.append("e.provider = ?")
            parameters.append(_normalize_provider(provider))
        if native_session_id is not None:
            session = _bounded_text(native_session_id)
            if session is None:
                raise TelemetryError("native_session_id must be a non-empty short string")
            clauses.append("e.native_session_id = ?")
            parameters.append(session)
        if repository_id is not None:
            repository = _bounded_text(repository_id, 4096)
            if repository is None:
                raise TelemetryError("repository_id must be a non-empty short string")
            clauses.append(
                """EXISTS (
                    SELECT 1 FROM telemetry_event_repositories er
                    WHERE er.event_key = e.event_key
                      AND (er.repository_id = ? OR er.repository_path = ?)
                )"""
            )
            parameters.extend((repository, repository))
        if limit is not None:
            if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
                raise TelemetryError("limit must be a positive integer")

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        limit_sql = " LIMIT ?" if limit is not None else ""
        if limit is not None:
            parameters.append(limit)
        with self._connection() as connection:
            rows = connection.execute(
                f"""
                SELECT e.* FROM telemetry_events e
                {where}
                ORDER BY COALESCE(e.observed_at_unix_nano, 0), e.id
                {limit_sql}
                """,
                parameters,
            ).fetchall()
        return [_event_row(row) for row in rows]

    def query_claude_prompt_tool_links(
        self,
        native_session_id: str | None = None,
        prompt_id: str | None = None,
        repository_id: str | os.PathLike[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Return Claude prompt-to-tool-use links for registered sessions."""

        clauses: list[str] = []
        parameters: list[Any] = []
        if native_session_id is not None:
            session = _bounded_text(native_session_id)
            if session is None:
                raise TelemetryError("native_session_id must be a non-empty short string")
            clauses.append("l.native_session_id = ?")
            parameters.append(session)
        if prompt_id is not None:
            prompt = _bounded_text(prompt_id)
            if prompt is None:
                raise TelemetryError("prompt_id must be a non-empty short string")
            clauses.append("l.prompt_id = ?")
            parameters.append(prompt)
        if repository_id is not None:
            repository = _bounded_text(repository_id, 4096)
            if repository is None:
                raise TelemetryError("repository_id must be a non-empty short string")
            clauses.append(
                """EXISTS (
                    SELECT 1 FROM claude_prompt_tool_link_repositories lr
                    WHERE lr.link_key = l.link_key
                      AND (lr.repository_id = ? OR lr.repository_path = ?)
                )"""
            )
            parameters.extend((repository, repository))
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._connection() as connection:
            rows = connection.execute(
                f"""
                SELECT native_session_id, prompt_id, tool_use_id,
                       observed_at_unix_nano, received_at
                FROM claude_prompt_tool_links l
                {where}
                ORDER BY COALESCE(observed_at_unix_nano, 0), received_at,
                         native_session_id, prompt_id, tool_use_id
                """,
                parameters,
            ).fetchall()
        return [dict(row) for row in rows]


def register_session(
    provider: str,
    native_session_id: str,
    repository_id: str | os.PathLike[str],
    state_dir: str | os.PathLike[str] | None = None,
    *,
    repository_path: str | os.PathLike[str] | None = None,
) -> dict[str, str | None]:
    """Register a native session as eligible for telemetry persistence."""

    return TelemetryStore(state_dir).register_session(
        provider, native_session_id, repository_id, repository_path
    )


def unregister_repository(
    repository_id: str | os.PathLike[str],
    state_dir: str | os.PathLike[str] | None = None,
) -> int:
    """Deactivate repository ingestion without orphaning historical events."""

    return TelemetryStore(state_dir).unregister_repository(repository_id)


def registered_repository_count(
    state_dir: str | os.PathLike[str] | None = None,
) -> int:
    """Return the number of repositories with active registered sessions."""

    directory = _state_dir_path(state_dir)
    if not (directory / "telemetry.sqlite3").is_file():
        return 0
    return TelemetryStore(directory).registered_repository_count()


def query_events(
    provider: str | None = None,
    native_session_id: str | None = None,
    repository_id: str | os.PathLike[str] | None = None,
    state_dir: str | os.PathLike[str] | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Open the global store and return matching normalized telemetry events."""

    directory = _state_dir_path(state_dir)
    if not (directory / "telemetry.sqlite3").is_file():
        return []
    return TelemetryStore(directory).query_events(
        provider, native_session_id, repository_id, limit
    )


def query_claude_prompt_tool_links(
    native_session_id: str | None = None,
    prompt_id: str | None = None,
    repository_id: str | os.PathLike[str] | None = None,
    state_dir: str | os.PathLike[str] | None = None,
) -> list[dict[str, Any]]:
    """Open the global store and return matching Claude prompt/tool links."""

    directory = _state_dir_path(state_dir)
    if not (directory / "telemetry.sqlite3").is_file():
        return []
    return TelemetryStore(directory).query_claude_prompt_tool_links(
        native_session_id, prompt_id, repository_id
    )


def compute_codex_chatgpt_credits(
    model: str,
    input_tokens: int | None,
    cached_input_tokens: int | None,
    output_tokens: int | None,
    *,
    cache_write_input_tokens: int | None = None,
    input_includes_cached: bool = True,
    service_tier: str | None = None,
) -> Decimal | None:
    """Compute ChatGPT credits using the rate table effective 2026-09-14.

    Official usage reports total input including cached input, so that is the
    default.  Set ``input_includes_cached=False`` only for a source that reports
    uncached input separately.
    """

    model_name = str(model)
    rates = CODEX_CHATGPT_CREDIT_RATES.get(model_name)
    if rates is None:
        return None
    if input_tokens is None or output_tokens is None:
        return None
    multiplier = _chatgpt_service_tier_multiplier(model_name, service_tier)
    if multiplier is None:
        return None
    input_count = max(input_tokens or 0, 0)
    if input_count > 272_000:
        # Long-context and service-tier multipliers are model-specific. OTel
        # does not currently guarantee enough pricing metadata to infer them.
        return None
    cached_count = max(cached_input_tokens or 0, 0)
    cache_write_count = max(cache_write_input_tokens or 0, 0)
    output_count = max(output_tokens or 0, 0)
    if input_includes_cached:
        cached_count = min(cached_count, input_count)
        cache_write_count = min(cache_write_count, input_count - cached_count)
        # Codex credit pricing does not charge for prompt-cache writes.
        uncached_count = input_count - cached_count - cache_write_count
    else:
        uncached_count = input_count
    credits = (
        Decimal(uncached_count) * rates["input"]
        + Decimal(cached_count) * rates["cached_input"]
        + Decimal(output_count) * rates["output"]
    ) / Decimal(1_000_000)
    return credits * multiplier


def _chatgpt_service_tier_multiplier(
    model: str, service_tier: str | None
) -> Decimal | None:
    tier = (service_tier or "default").strip().casefold()
    if tier in {"", "default", "standard", "auto"}:
        return Decimal("1")
    if tier not in {"fast", "priority"}:
        return None
    if model in {"gpt-5.4", "gpt-5.4-mini"}:
        return Decimal("2")
    if model in {
        "gpt-6-astra",
        "gpt-5.6-sol",
        "gpt-5.6-terra",
        "gpt-5.6-luna",
        "gpt-5.5",
    }:
        return Decimal("2.5")
    return None


def _api_service_tier_multiplier(
    model: str, service_tier: str | None
) -> Decimal | None:
    tier = (service_tier or "default").strip().casefold()
    if tier in {"", "default", "standard", "auto"}:
        return Decimal("1")
    if tier not in {"fast", "priority"}:
        return None
    if model == "gpt-5.5":
        return Decimal("2.5")
    if model in {
        "gpt-6-astra",
        "gpt-5.6-sol",
        "gpt-5.6-terra",
        "gpt-5.6-luna",
        "gpt-5.4",
        "gpt-5.4-mini",
    }:
        return Decimal("2")
    return None


def compute_openai_api_usd(
    model: str,
    input_tokens: int | None,
    cached_input_tokens: int | None,
    output_tokens: int | None,
    *,
    cache_write_input_tokens: int | None = None,
    service_tier: str | None = None,
) -> Decimal | None:
    """Estimate standard-tier API cost; fail closed for long-context pricing."""

    rates = OPENAI_API_STANDARD_RATES.get(str(model))
    if rates is None:
        return None
    if input_tokens is None or output_tokens is None:
        return None
    model_name = str(model)
    multiplier = _api_service_tier_multiplier(model_name, service_tier)
    if multiplier is None:
        return None
    input_count = max(input_tokens or 0, 0)
    if input_count > 272_000:
        # Long-context and service-tier multipliers are model-specific. OTel
        # does not currently guarantee enough pricing metadata to infer them.
        return None
    cached_count = min(max(cached_input_tokens or 0, 0), input_count)
    cache_write_count = min(
        max(cache_write_input_tokens or 0, 0), input_count - cached_count
    )
    uncached_count = input_count - cached_count - cache_write_count
    output_count = max(output_tokens or 0, 0)
    cache_write_multiplier = (
        Decimal("1.25")
        if model_name
        in {"gpt-6-astra", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna"}
        else Decimal("1")
    )
    return multiplier * (
        Decimal(uncached_count) * rates["input"]
        + Decimal(cached_count) * rates["cached_input"]
        + Decimal(cache_write_count) * rates["input"] * cache_write_multiplier
        + Decimal(output_count) * rates["output"]
    ) / Decimal(1_000_000)


def _event_row(row: sqlite3.Row) -> dict[str, Any]:
    result = dict(row)
    result.pop("event_key", None)
    amount_text = result.get("cost_amount")
    amount = float(Decimal(amount_text)) if amount_text is not None else None
    result["cost_amount"] = amount
    result["cost_usd"] = amount if result.get("cost_unit") == "USD" else None
    result["credits"] = amount if result.get("cost_unit") == "credits" else None
    return result


def _canonical_key(key: Any) -> str:
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", str(key).strip())
    return re.sub(r"[^a-z0-9]+", "_", text.casefold()).strip("_")


_ALIASES: dict[str, tuple[str, ...]] = {
    "event_name": ("event_name", "otel_event_name", "name"),
    "event_kind": (
        "event_kind",
        "event_type",
        "kind",
        "type",
        "sse_event_type",
        "response_type",
    ),
    "session": (
        "session_id",
        "conversation_id",
        "thread_id",
        "codex_conversation_id",
    ),
    "prompt": ("prompt_id",),
    "turn": ("turn_id", "codex_turn_id"),
    "request": ("request_id", "response_id", "api_request_id"),
    "client_request": ("client_request_id", "client_requestid"),
    "tool_use": ("tool_use_id", "tool_id", "tool_useid"),
    "model": (
        "model",
        "model_name",
        "model_slug",
        "gen_ai_request_model",
        "gen_ai_response_model",
        "response_model",
    ),
    "auth_mode": ("auth_mode", "authentication_mode", "codex_auth_mode"),
    "service_tier": ("service_tier", "request_service_tier", "speed"),
    "cost_usd": ("cost_usd", "cost_us_dollars", "cost"),
    "cost_usd_micros": ("cost_usd_micros", "cost_micros", "usd_micros"),
    "input_tokens": (
        "input_tokens",
        "input_token_count",
        "tokens_input",
        "usage_input_tokens",
        "token_usage_input_tokens",
        "response_usage_input_tokens",
        "gen_ai_usage_input_tokens",
    ),
    "cached_input_tokens": (
        "cached_input_tokens",
        "cached_input_token_count",
        "cached_tokens",
        "cached_token_count",
        "input_cached_tokens",
        "cache_read_tokens",
        "cache_read_token_count",
        "cache_read_input_tokens",
        "usage_cached_input_tokens",
        "usage_input_tokens_details_cached_tokens",
        "token_usage_cached_input_tokens",
        "response_usage_input_tokens_details_cached_tokens",
        "gen_ai_usage_cached_input_tokens",
        "gen_ai_usage_cache_read_input_tokens",
    ),
    "cache_creation_input_tokens": (
        "cache_creation_tokens",
        "cache_creation_input_tokens",
        "cache_write_tokens",
        "cache_write_token_count",
        "cache_write_input_tokens",
        "usage_cache_creation_input_tokens",
        "usage_cache_write_input_tokens",
        "gen_ai_usage_cache_write_input_tokens",
    ),
    "output_tokens": (
        "output_tokens",
        "output_token_count",
        "tokens_output",
        "usage_output_tokens",
        "token_usage_output_tokens",
        "response_usage_output_tokens",
        "gen_ai_usage_output_tokens",
    ),
    "total_tokens": (
        "total_tokens",
        "total_token_count",
        "tokens_total",
        "usage_total_tokens",
        "token_usage_total_tokens",
        "response_usage_total_tokens",
        "codex_usage_total_tokens",
    ),
    "record_id": ("event_id", "log_record_id", "record_id"),
    # Workflow labels. Claude Code documents query_source and effort on
    # claude_code.api_request, and agent_id and parent_agent_id on request
    # spans. Whether its events carry agent_id is unverified, and Codex
    # per-agent usage is unknown, so every column below stays nullable.
    "query_source": ("query_source", "query_source_safe"),
    "agent_id": ("agent_id", "agent.id"),
    "parent_agent_id": ("parent_agent_id",),
    "effort": ("effort",),
    "agent_name": ("agent.name",),
}
_ALLOWED_ATTRIBUTE_KEYS = {
    _canonical_key(alias) for aliases in _ALIASES.values() for alias in aliases
}


def _decode_scalar(value: Any) -> str | int | float | bool | None:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if not isinstance(value, Mapping):
        return None
    for name in (
        "stringValue",
        "string_value",
        "intValue",
        "int_value",
        "doubleValue",
        "double_value",
        "boolValue",
        "bool_value",
    ):
        if name in value:
            inner = value[name]
            return inner if isinstance(inner, (str, int, float, bool)) else None
    return None


def _allowed_attributes(attributes: Any) -> dict[str, Any]:
    allowed: dict[str, Any] = {}
    if isinstance(attributes, Mapping):
        entries = attributes.items()
    elif isinstance(attributes, Sequence) and not isinstance(attributes, (str, bytes)):
        entries = (
            (entry.get("key"), entry.get("value"))
            for entry in attributes
            if isinstance(entry, Mapping) and "key" in entry
        )
    else:
        return allowed
    for key, encoded_value in entries:
        canonical = _canonical_key(key)
        if canonical not in _ALLOWED_ATTRIBUTE_KEYS:
            continue
        value = _decode_scalar(encoded_value)
        if value is not None:
            allowed[canonical] = value
    return allowed


def _get(attributes: Mapping[str, Any], field: str) -> Any:
    for alias in _ALIASES[field]:
        key = _canonical_key(alias)
        if key in attributes:
            return attributes[key]
    return None


def _container(mapping: Mapping[str, Any], camel: str, snake: str) -> Any:
    return mapping.get(camel, mapping.get(snake))


def _raw_log_records(payload: Mapping[str, Any]) -> Iterator[tuple[Mapping[str, Any], dict[str, Any]]]:
    resource_logs = _container(payload, "resourceLogs", "resource_logs")
    if isinstance(resource_logs, Sequence) and not isinstance(resource_logs, (str, bytes)):
        for resource_log in resource_logs:
            if not isinstance(resource_log, Mapping):
                continue
            resource = resource_log.get("resource")
            resource_attributes = _allowed_attributes(
                resource.get("attributes") if isinstance(resource, Mapping) else None
            )
            scope_logs = _container(resource_log, "scopeLogs", "scope_logs")
            if not isinstance(scope_logs, Sequence) or isinstance(scope_logs, (str, bytes)):
                continue
            for scope_log in scope_logs:
                if not isinstance(scope_log, Mapping):
                    continue
                scope = scope_log.get("scope")
                scope_attributes = _allowed_attributes(
                    scope.get("attributes") if isinstance(scope, Mapping) else None
                )
                records = _container(scope_log, "logRecords", "log_records")
                if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
                    continue
                for record in records:
                    if isinstance(record, Mapping):
                        inherited = {**resource_attributes, **scope_attributes}
                        yield record, inherited
        return

    records = _container(payload, "logRecords", "log_records")
    if records is None:
        records = payload.get("logs")
    if isinstance(records, Sequence) and not isinstance(records, (str, bytes)):
        for record in records:
            if isinstance(record, Mapping):
                yield record, {}


def _has_log_container(payload: Mapping[str, Any]) -> bool:
    return any(
        key in payload for key in ("resourceLogs", "resource_logs", "logRecords", "log_records", "logs")
    )


def _record_event_name(record: Mapping[str, Any], attributes: Mapping[str, Any]) -> str | None:
    direct = record.get("eventName", record.get("event_name"))
    candidates = (_get(attributes, "event_name"), _decode_scalar(direct), _decode_scalar(record.get("body")))
    known = {
        "claude_code.api_request",
        "claude_code.tool_result",
        "codex.sse_event",
        "codex.turn.token_usage",
        "response.completed",
    }
    for candidate in candidates:
        value = _bounded_text(candidate, 128)
        if value and value.casefold() in known:
            return value.casefold()
    return None


def _observed_time(record: Mapping[str, Any]) -> int | None:
    value = record.get(
        "timeUnixNano",
        record.get(
            "time_unix_nano",
            record.get("observedTimeUnixNano", record.get("observed_time_unix_nano")),
        ),
    )
    return _nonnegative_int(value)


def _nonnegative_int(value: Any) -> int | None:
    value = _decode_scalar(value)
    if isinstance(value, bool) or value is None:
        return None
    try:
        decimal = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None
    if not decimal.is_finite() or decimal < 0 or decimal != decimal.to_integral_value():
        return None
    integer = int(decimal)
    return integer if integer <= SQLITE_INTEGER_MAX else None


def _sqlite_safe_sum(*values: int | None) -> int | None:
    if any(value is None for value in values):
        return None
    total = sum(value for value in values if value is not None)
    return total if total <= SQLITE_INTEGER_MAX else None


def _nonnegative_decimal(value: Any) -> Decimal | None:
    value = _decode_scalar(value)
    if isinstance(value, bool) or value is None:
        return None
    try:
        decimal = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None
    return (
        decimal
        if decimal.is_finite() and Decimal("0") <= decimal <= Decimal("1000000000")
        else None
    )


def _decimal_text(value: Decimal | None) -> str | None:
    if value is None:
        return None
    text = format(value.normalize(), "f")
    return "0" if text == "-0" else text


def _digest_key(*parts: Any) -> str:
    encoded = json.dumps(parts, ensure_ascii=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _parse_otlp_records(payload: Mapping[str, Any]) -> Iterator[_NormalizedEvent | None]:
    for record, inherited in _raw_log_records(payload):
        attributes = {**inherited, **_allowed_attributes(record.get("attributes"))}
        event_name = _record_event_name(record, attributes)
        if event_name is None:
            yield None
            continue
        yield _parse_record(event_name, record, attributes)


def _parse_record(
    event_name: str, record: Mapping[str, Any], attributes: Mapping[str, Any]
) -> _NormalizedEvent | None:
    if event_name.startswith("claude_code."):
        return _parse_claude_record(event_name, record, attributes)
    return _parse_codex_record(event_name, record, attributes)


def _workflow_labels(attributes: Mapping[str, Any]) -> dict[str, str | None]:
    """Return the workflow labels a request carries, each one optional.

    ``query_source`` names what asked for the request. It is a label only: a
    request without an ``agent_id`` cannot be assigned to one agent's session.
    """

    return {
        "query_source": _bounded_text(_get(attributes, "query_source"), 128),
        "agent_id": _bounded_text(_get(attributes, "agent_id")),
        "parent_agent_id": _bounded_text(_get(attributes, "parent_agent_id")),
        "agent_name": _bounded_text(_get(attributes, "agent_name"), 128),
        "effort": _bounded_text(_get(attributes, "effort"), 64),
    }


def _parse_claude_record(
    event_name: str, record: Mapping[str, Any], attributes: Mapping[str, Any]
) -> _NormalizedEvent | None:
    session = _bounded_text(_get(attributes, "session"))
    if session is None:
        return None
    prompt = _bounded_text(_get(attributes, "prompt"))
    request = _bounded_text(_get(attributes, "request"))
    client_request = _bounded_text(_get(attributes, "client_request"))
    tool_use = _bounded_text(_get(attributes, "tool_use"))
    model = _bounded_text(_get(attributes, "model"), 256)
    observed = _observed_time(record)
    input_tokens = _nonnegative_int(_get(attributes, "input_tokens"))
    cached_tokens = _nonnegative_int(_get(attributes, "cached_input_tokens"))
    cache_creation = _nonnegative_int(_get(attributes, "cache_creation_input_tokens"))
    output_tokens = _nonnegative_int(_get(attributes, "output_tokens"))
    total_tokens = _nonnegative_int(_get(attributes, "total_tokens"))
    if total_tokens is None:
        # Claude Code reports its cached and cache creation counts beside its
        # input count rather than inside it, so the parts it reported add up,
        # exactly as ``costing._counted_tokens`` sums them. A request that
        # reported no count at all stays unknown.
        parts = [
            count
            for count in (input_tokens, cached_tokens, cache_creation, output_tokens)
            if count is not None
        ]
        if parts:
            total_tokens = _sqlite_safe_sum(*parts)

    amount: Decimal | None = None
    if event_name == "claude_code.api_request":
        micros = _nonnegative_int(_get(attributes, "cost_usd_micros"))
        amount = (
            Decimal(micros) / Decimal(1_000_000)
            if micros is not None
            else _nonnegative_decimal(_get(attributes, "cost_usd"))
        )
        if amount is not None and amount > Decimal("1000000000"):
            amount = None
    elif prompt is None or tool_use is None:
        return None

    identity = (
        ("request", request, client_request)
        if request or client_request
        else (
            "tool",
            prompt,
            tool_use,
        )
        if tool_use
        else (
            "metadata",
            prompt,
            model,
            input_tokens,
            cached_tokens,
            cache_creation,
            output_tokens,
            _decimal_text(amount),
            observed,
        )
    )
    event_key = _digest_key("claude", event_name, session, identity)
    return _NormalizedEvent(
        event_key=event_key,
        provider="claude",
        event_name=event_name,
        event_kind=None,
        native_session_id=session,
        prompt_id=prompt,
        request_id=request,
        client_request_id=client_request,
        model=model,
        input_tokens=input_tokens,
        cached_input_tokens=cached_tokens,
        cache_creation_input_tokens=cache_creation,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        cost_amount=_decimal_text(amount),
        cost_unit="USD" if amount is not None else None,
        cost_source="vendor_estimate" if amount is not None else None,
        **_workflow_labels(attributes),
        observed_at_unix_nano=observed,
        tool_use_id=tool_use,
    )


def _parse_codex_record(
    event_name: str, record: Mapping[str, Any], attributes: Mapping[str, Any]
) -> _NormalizedEvent | None:
    event_kind = _bounded_text(_get(attributes, "event_kind"), 128)
    body = _bounded_text(_decode_scalar(record.get("body")), 128)
    if event_name == "response.completed":
        event_name = "codex.sse_event"
        event_kind = "response.completed"
    elif event_name == "codex.sse_event":
        if event_kind is None and body and body.casefold() == "response.completed":
            event_kind = "response.completed"
        if event_kind is None or event_kind.casefold() != "response.completed":
            return None
        event_kind = "response.completed"
    elif event_name != "codex.turn.token_usage":
        return None

    session = _bounded_text(_get(attributes, "session"))
    if session is None:
        return None
    turn = _bounded_text(_get(attributes, "turn"))
    request = _bounded_text(_get(attributes, "request"))
    model = _bounded_text(_get(attributes, "model"), 256)
    auth_mode = _bounded_text(_get(attributes, "auth_mode"), 64)
    service_tier = _bounded_text(_get(attributes, "service_tier"), 64)
    input_tokens = _nonnegative_int(_get(attributes, "input_tokens"))
    cached_tokens = _nonnegative_int(_get(attributes, "cached_input_tokens"))
    cache_write_tokens = _nonnegative_int(
        _get(attributes, "cache_creation_input_tokens")
    )
    output_tokens = _nonnegative_int(_get(attributes, "output_tokens"))
    total_tokens = _nonnegative_int(_get(attributes, "total_tokens"))
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = _sqlite_safe_sum(input_tokens, output_tokens)
    observed = _observed_time(record)

    credits: Decimal | None = None
    api_cost: Decimal | None = None
    normalized_auth = auth_mode.casefold() if auth_mode else ""
    if normalized_auth in {"chatgpt", "swic"} and model is not None:
        credits = compute_codex_chatgpt_credits(
            model,
            input_tokens,
            cached_tokens,
            output_tokens,
            cache_write_input_tokens=cache_write_tokens,
            service_tier=service_tier,
        )
    elif normalized_auth in {"api", "api_key", "apikey"} and model is not None:
        api_cost = compute_openai_api_usd(
            model,
            input_tokens,
            cached_tokens,
            output_tokens,
            cache_write_input_tokens=cache_write_tokens,
            service_tier=service_tier,
        )

    identity = (
        ("request", request)
        if request
        else ("turn", turn)
        if turn
        else (
            "metadata",
            model,
            auth_mode,
            input_tokens,
            cached_tokens,
            cache_write_tokens,
            output_tokens,
            observed,
        )
    )
    event_key = _digest_key("codex", event_name, event_kind, session, identity)
    return _NormalizedEvent(
        event_key=event_key,
        provider="codex",
        event_name=event_name,
        event_kind=event_kind,
        native_session_id=session,
        turn_id=turn,
        request_id=request,
        model=model,
        auth_mode=auth_mode,
        service_tier=service_tier,
        input_tokens=input_tokens,
        cached_input_tokens=cached_tokens,
        cache_creation_input_tokens=cache_write_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        cost_amount=_decimal_text(credits if credits is not None else api_cost),
        cost_unit=(
            "credits" if credits is not None else "USD" if api_cost is not None else None
        ),
        cost_source=(
            f"chatgpt_credit_rate_{CODEX_CHATGPT_CREDIT_RATES_EFFECTIVE_DATE}"
            if credits is not None
            else f"openai_api_standard_rate_{OPENAI_API_STANDARD_RATES_EFFECTIVE_DATE}"
            if api_cost is not None
            else None
        ),
        **_workflow_labels(attributes),
        observed_at_unix_nano=observed,
    )


def _validate_loopback(host: str) -> str:
    host = str(host).strip()
    if host.casefold() == "localhost":
        return "127.0.0.1"
    try:
        address = ipaddress.ip_address(host)
    except ValueError as exc:
        raise TelemetryError("collector host must be a loopback IP address") from exc
    if not address.is_loopback:
        raise TelemetryError("collector may only bind to a loopback address")
    return str(address)


def make_handler(
    store: TelemetryStore,
    bearer_token: str,
    max_request_bytes: int = DEFAULT_MAX_REQUEST_BYTES,
    *,
    instance_id: str | None = None,
) -> type[BaseHTTPRequestHandler]:
    """Build an authenticated handler for ``/v1/logs`` and ``/health``."""

    token = _bounded_text(bearer_token, 4096)
    if token is None:
        raise TelemetryError("a non-empty bearer token is required")
    identity = _bounded_text(instance_id or secrets.token_hex(16), 128)
    if identity is None:
        raise TelemetryError("a non-empty collector instance ID is required")
    if max_request_bytes <= 0:
        raise TelemetryError("max_request_bytes must be positive")
    # The two spellings of one bearer token: the decoded header an HTTP client
    # sends, and the undecoded ``OTEL_EXPORTER_OTLP_HEADERS`` value that Claude
    # Code forwards. ``telemetry_settings`` writes the second form.
    accepted_authorizations = (
        f"Bearer {token}".encode(),
        f"Bearer%20{quote(token, safe='')}".encode(),
    )

    class TelemetryHandler(BaseHTTPRequestHandler):
        server_version = "JoyrideTelemetry/1"
        sys_version = ""

        def log_message(self, _format: str, *_args: Any) -> None:
            # Request paths and payload metadata should not leak to stderr.
            return

        def _authorized(self) -> bool:
            supplied = self.headers.get("Authorization", "").encode()
            # Claude Code forwards the value of ``OTEL_EXPORTER_OTLP_HEADERS``
            # exactly as it was written, so the ``Bearer%20<token>`` this
            # project writes there arrives with its space still
            # percent-encoded. Both spellings name the one token, and nothing
            # else is accepted. Every comparison runs, so a match in the first
            # form costs the same time as a match in the second.
            matched = False
            for expected in accepted_authorizations:
                matched |= hmac.compare_digest(supplied, expected)
            return matched

        def _json(self, status_code: int, payload: Mapping[str, Any]) -> None:
            body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _require_auth(self) -> bool:
            if self._authorized():
                return True
            self.send_response(401)
            self.send_header("WWW-Authenticate", "Bearer")
            self.send_header("Content-Length", "0")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            return False

        def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            if self.path == "/identity":
                self._json(
                    200,
                    {"service": _COLLECTOR_IDENTITY, "instance_id": identity},
                )
                return
            if not self._require_auth():
                return
            if self.path != "/health":
                self._json(404, {"error": "not_found"})
                return
            self._json(200, {"status": "ok"})

        def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            if not self._require_auth():
                return
            if self.path == "/shutdown":
                self._json(200, {"status": "stopping"})
                threading.Thread(target=self.server.shutdown, daemon=True).start()
                return
            if self.path != "/v1/logs":
                self._json(404, {"error": "not_found"})
                return
            content_type = self.headers.get("Content-Type", "").partition(";")[0].strip().casefold()
            if content_type not in {"application/json", "application/x-json"}:
                self._json(415, {"error": "content_type_must_be_json"})
                return
            if self.headers.get("Content-Encoding", "identity").casefold() not in {"", "identity"}:
                self._json(415, {"error": "content_encoding_not_supported"})
                return
            raw_length = self.headers.get("Content-Length")
            if raw_length is None:
                self._json(411, {"error": "content_length_required"})
                return
            try:
                length = int(raw_length)
            except ValueError:
                self._json(400, {"error": "invalid_content_length"})
                return
            if length < 0:
                self._json(400, {"error": "invalid_content_length"})
                return
            if length > max_request_bytes:
                self._json(413, {"error": "request_too_large"})
                return
            body = self.rfile.read(length)
            if len(body) != length:
                self._json(400, {"error": "incomplete_request_body"})
                return
            try:
                result = store.ingest_otlp(body)
            except TelemetryError:
                self._json(400, {"error": "invalid_otlp_json"})
                return
            self._json(200, result)

    return TelemetryHandler


class _LoopbackThreadingHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def make_server(
    store: TelemetryStore,
    bearer_token: str,
    host: str = "127.0.0.1",
    port: int = DEFAULT_OTLP_PORT,
    max_request_bytes: int = DEFAULT_MAX_REQUEST_BYTES,
    *,
    instance_id: str | None = None,
) -> ThreadingHTTPServer:
    """Create, but do not start, a loopback-only OTLP HTTP server."""

    loopback = _validate_loopback(host)
    if isinstance(port, bool) or not isinstance(port, int) or not 0 <= port <= 65535:
        raise TelemetryError("port must be an integer from 0 through 65535")
    selected_instance_id = _bounded_text(instance_id or secrets.token_hex(16), 128)
    if selected_instance_id is None:
        raise TelemetryError("a non-empty collector instance ID is required")
    handler = make_handler(
        store,
        bearer_token,
        max_request_bytes,
        instance_id=selected_instance_id,
    )
    address: tuple[Any, ...] = (loopback, port)
    if ":" in loopback:
        class IPv6LoopbackServer(_LoopbackThreadingHTTPServer):
            address_family = socket.AF_INET6

        server: ThreadingHTTPServer = IPv6LoopbackServer(address, handler)
    else:
        server = _LoopbackThreadingHTTPServer(address, handler)
    server.attribution_instance_id = selected_instance_id  # type: ignore[attr-defined]
    return server


def serve(
    state_dir: str | os.PathLike[str] | None = None,
    bearer_token: str | None = None,
    host: str = "127.0.0.1",
    port: int = DEFAULT_OTLP_PORT,
    max_request_bytes: int = DEFAULT_MAX_REQUEST_BYTES,
) -> None:
    """Run the authenticated local collector until interrupted."""

    directory = secure_state_dir(state_dir)
    token = bearer_token or _load_or_create_token(directory)
    store = TelemetryStore(directory)
    server = make_server(store, token, host, port, max_request_bytes)
    try:
        server.serve_forever(poll_interval=0.25)
    finally:
        server.server_close()


def _token_path(state_dir: Path) -> Path:
    return state_dir / "collector.token"


def _load_or_create_token(state_dir: Path) -> str:
    path = _token_path(state_dir)
    if path.is_symlink():
        raise TelemetryError("collector token file may not be a symlink")
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        token = path.read_text(encoding="ascii").strip()
    else:
        token = secrets.token_urlsafe(32)
        with os.fdopen(descriptor, "w", encoding="ascii") as file:
            file.write(token + "\n")
            file.flush()
            os.fsync(file.fileno())
    if _bounded_text(token, 4096) is None:
        raise TelemetryError("collector token file is invalid")
    if os.name == "posix":
        mode = stat.S_IMODE(path.stat().st_mode)
        if mode != 0o600:
            os.chmod(path, 0o600)
    return token


def _runtime_path(state_dir: Path) -> Path:
    return state_dir / "collector-runtime.json"


def _endpoint_path(state_dir: Path) -> Path:
    return state_dir / "collector-endpoint.json"


def _read_endpoint(state_dir: Path) -> dict[str, Any] | None:
    path = _endpoint_path(state_dir)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None
    if not isinstance(value, dict):
        return None
    host, port = value.get("host"), value.get("port")
    if not isinstance(host, str) or not isinstance(port, int):
        return None
    try:
        _validate_loopback(host)
    except TelemetryError:
        return None
    return value if 0 < port <= 65535 else None


def _read_runtime(state_dir: Path) -> dict[str, Any] | None:
    path = _runtime_path(state_dir)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None
    if not isinstance(value, dict):
        return None
    host = value.get("host")
    port = value.get("port")
    pid = value.get("pid")
    instance_id = _bounded_text(value.get("instance_id"), 128)
    if (
        not isinstance(host, str)
        or not isinstance(port, int)
        or not isinstance(pid, int)
        or instance_id is None
    ):
        return None
    try:
        _validate_loopback(host)
    except TelemetryError:
        return None
    if not 0 < port <= 65535 or pid <= 0:
        return None
    value["instance_id"] = instance_id
    return value


def _write_runtime(state_dir: Path, payload: Mapping[str, Any]) -> None:
    path = _runtime_path(state_dir)
    temporary = state_dir / f".collector-runtime-{os.getpid()}.tmp"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as file:
            json.dump(payload, file, separators=(",", ":"))
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _write_endpoint(state_dir: Path, host: str, port: int) -> None:
    path = _endpoint_path(state_dir)
    temporary = state_dir / f".collector-endpoint-{os.getpid()}.tmp"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as file:
            json.dump({"host": host, "port": port}, file, separators=(",", ":"))
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _host_for_url(host: str) -> str:
    return f"[{host}]" if ":" in host else host


def telemetry_settings(
    state_dir: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Return collector endpoint and authentication settings.

    Calling this function intentionally creates the private state directory and
    token.  Importing the module does not.  If no collector is running yet, the
    endpoint uses the standard OTLP/HTTP port; ``ensure_collector`` returns the
    authoritative endpoint after startup.
    """

    directory = secure_state_dir(state_dir)
    token = _load_or_create_token(directory)
    runtime = _read_runtime(directory)
    endpoint = _read_endpoint(directory)
    selected = runtime or endpoint
    host = selected["host"] if selected else "127.0.0.1"
    port = selected["port"] if selected else DEFAULT_OTLP_PORT
    instance_id = selected.get("instance_id") if selected else None
    base = f"http://{_host_for_url(host)}:{port}"
    authorization = f"Bearer {token}"
    return {
        "state_dir": str(directory),
        "database": str(directory / "telemetry.sqlite3"),
        "endpoint": f"{base}/v1/logs",
        "identity_endpoint": f"{base}/identity",
        "health_endpoint": f"{base}/health",
        "instance_id": instance_id,
        "token": token,
        "authorization_header": authorization,
        "headers": {"Authorization": authorization},
        "header": f"Authorization=Bearer%20{quote(token, safe='')}",
    }


def _request_identity(settings: Mapping[str, Any], timeout: float = 0.5) -> bool:
    expected = _bounded_text(settings.get("instance_id"), 128)
    endpoint = settings.get("identity_endpoint")
    if expected is None or not isinstance(endpoint, str):
        return False
    request = Request(endpoint, method="GET")
    try:
        with urlopen(request, timeout=timeout) as response:
            if response.status != 200 or response.geturl() != endpoint:
                return False
            body = response.read(_MAX_IDENTITY_RESPONSE_BYTES + 1)
    except (HTTPError, URLError, TimeoutError, OSError):
        return False
    if len(body) > _MAX_IDENTITY_RESPONSE_BYTES:
        return False
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, ValueError, RecursionError):
        return False
    actual = payload.get("instance_id") if isinstance(payload, Mapping) else None
    return (
        isinstance(actual, str)
        and payload.get("service") == _COLLECTOR_IDENTITY
        and hmac.compare_digest(actual.encode(), expected.encode())
    )


def _request_health(settings: Mapping[str, Any], timeout: float = 0.5) -> bool:
    if not _request_identity(settings, timeout=timeout):
        return False
    request = Request(
        str(settings["health_endpoint"]),
        headers={"Authorization": str(settings["authorization_header"])},
        method="GET",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            return response.status == 200
    except (HTTPError, URLError, TimeoutError, OSError):
        return False


def telemetry_status(
    state_dir: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Return collector status without creating state when none exists."""

    directory = _state_dir_path(state_dir)
    runtime = _read_runtime(directory) if directory.is_dir() else None
    token_path = _token_path(directory)
    if runtime is None or not token_path.is_file() or token_path.is_symlink():
        return {"running": False, "state_dir": str(directory), "pid": None, "endpoint": None}
    try:
        token = token_path.read_text(encoding="ascii").strip()
    except OSError:
        return {"running": False, "state_dir": str(directory), "pid": None, "endpoint": None}
    base = f"http://{_host_for_url(runtime['host'])}:{runtime['port']}"
    settings = {
        "identity_endpoint": f"{base}/identity",
        "health_endpoint": f"{base}/health",
        "instance_id": runtime["instance_id"],
        "authorization_header": f"Bearer {token}",
    }
    running = _request_health(settings)
    return {
        "running": running,
        "state_dir": str(directory),
        "pid": runtime["pid"] if running else None,
        "endpoint": f"{base}/v1/logs" if running else None,
    }


def _loopback_port_available(port: int) -> bool:
    if isinstance(port, bool) or not isinstance(port, int) or not 0 <= port <= 65535:
        raise TelemetryError("port must be an integer from 0 through 65535")
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.bind(("127.0.0.1", port))
        return True
    except OSError:
        return False
    finally:
        probe.close()


@contextmanager
def _collector_startup_lock(state_dir: Path) -> Iterator[None]:
    path = state_dir / _COLLECTOR_LOCK_FILENAME
    flags = os.O_RDWR | os.O_CREAT
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise TelemetryError("collector startup lock may not be a symlink") from exc
        raise
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise TelemetryError("collector startup lock must be a regular file")
        current = path.lstat()
        if (
            stat.S_ISLNK(current.st_mode)
            or opened.st_dev != current.st_dev
            or opened.st_ino != current.st_ino
        ):
            raise TelemetryError("collector startup lock changed while it was opened")
        if os.name == "posix":
            os.fchmod(descriptor, 0o600)
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _reap_collector_child(directory: Path) -> None:
    process = _COLLECTOR_CHILDREN.pop(str(directory), None)
    if process is None:
        return
    try:
        process.wait(timeout=1)
    except subprocess.TimeoutExpired:
        # The authenticated server has stopped accepting requests.  Leave the
        # detached process to finish its short shutdown path without killing a
        # possibly unrelated PID.
        _COLLECTOR_CHILDREN[str(directory)] = process


def ensure_collector(
    state_dir: str | os.PathLike[str] | None = None,
    *,
    port: int = DEFAULT_OTLP_PORT,
    max_request_bytes: int = DEFAULT_MAX_REQUEST_BYTES,
    startup_timeout: float = 5.0,
) -> dict[str, Any]:
    """Ensure a detached local collector is running and return its settings."""

    if startup_timeout <= 0 or startup_timeout > 60:
        raise TelemetryError("startup_timeout must be greater than zero and at most 60 seconds")
    directory = secure_state_dir(state_dir)
    with _collector_startup_lock(directory):
        return _ensure_collector_locked(
            directory,
            port=port,
            max_request_bytes=max_request_bytes,
            startup_timeout=startup_timeout,
        )


def _ensure_collector_locked(
    directory: Path,
    *,
    port: int,
    max_request_bytes: int,
    startup_timeout: float,
) -> dict[str, Any]:
    existing = telemetry_status(directory)
    if existing["running"]:
        return {**telemetry_settings(directory), **existing}

    configured_endpoint = _read_endpoint(directory)
    if configured_endpoint is not None:
        port = configured_endpoint["port"]

    runtime_path = _runtime_path(directory)
    try:
        runtime_path.unlink()
    except FileNotFoundError:
        pass
    token = _load_or_create_token(directory)
    if configured_endpoint is None and port and not _loopback_port_available(port):
        # Another local service may legitimately own the standard OTLP port.
        # An ephemeral loopback port keeps installation automatic; the actual
        # endpoint is returned below and persisted in the runtime record.
        port = 0
    invocation = current_runtime().module_child(
        "attribution.telemetry",
        (
            "_serve",
            "--state-dir",
            str(directory),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--max-request-bytes",
            str(max_request_bytes),
        ),
        standalone_action="_collector",
    )
    command = list(invocation.argv)
    process = subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=invocation.environment(),
        start_new_session=True,
        close_fds=True,
    )
    _COLLECTOR_CHILDREN[str(directory)] = process
    deadline = time.monotonic() + startup_timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            _COLLECTOR_CHILDREN.pop(str(directory), None)
            raise TelemetryError("telemetry collector exited during startup")
        runtime = _read_runtime(directory)
        if runtime is not None:
            settings = telemetry_settings(directory)
            if _request_health(settings):
                _write_endpoint(directory, runtime["host"], runtime["port"])
                return {
                    **settings,
                    "running": True,
                    "pid": runtime["pid"],
                }
        time.sleep(0.05)
    process.terminate()
    try:
        process.wait(timeout=1)
    except subprocess.TimeoutExpired:
        pass
    _COLLECTOR_CHILDREN.pop(str(directory), None)
    raise TelemetryError("telemetry collector did not become healthy before the timeout")


def stop_collector(
    state_dir: str | os.PathLike[str] | None = None,
    *,
    timeout: float = 5.0,
) -> bool:
    """Ask the authenticated local collector to stop; return whether one ran."""

    directory = _state_dir_path(state_dir)
    status = telemetry_status(directory)
    if not status["running"]:
        return False
    settings = telemetry_settings(directory)
    endpoint = str(settings["endpoint"])
    shutdown_endpoint = endpoint[: -len("/v1/logs")] + "/shutdown"
    request = Request(
        shutdown_endpoint,
        data=b"",
        headers={"Authorization": str(settings["authorization_header"])},
        method="POST",
    )
    try:
        with urlopen(request, timeout=min(timeout, 2.0)) as response:
            if response.status != 200:
                return False
    except (HTTPError, URLError, TimeoutError, OSError):
        return False
    deadline = time.monotonic() + max(timeout, 0)
    while time.monotonic() < deadline:
        if not telemetry_status(directory)["running"]:
            _reap_collector_child(directory)
            return True
        time.sleep(0.05)
    stopped = not telemetry_status(directory)["running"]
    if stopped:
        _reap_collector_child(directory)
    return stopped


def _collector_process(
    state_dir: str,
    host: str,
    port: int,
    max_request_bytes: int,
) -> int:
    directory = secure_state_dir(state_dir)
    token = _load_or_create_token(directory)
    store = TelemetryStore(directory)
    instance_id = secrets.token_hex(16)
    server = make_server(
        store,
        token,
        host,
        port,
        max_request_bytes,
        instance_id=instance_id,
    )
    bound_host = str(server.server_address[0])
    bound_port = int(server.server_address[1])
    runtime = {
        "pid": os.getpid(),
        "host": bound_host,
        "port": bound_port,
        "instance_id": instance_id,
        "started_at_unix": int(time.time()),
    }
    _write_runtime(directory, runtime)

    def request_shutdown(_signum: int, _frame: Any) -> None:
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, request_shutdown)
    signal.signal(signal.SIGINT, request_shutdown)
    try:
        server.serve_forever(poll_interval=0.25)
    finally:
        server.server_close()
        current = _read_runtime(directory)
        if current and current.get("instance_id") == instance_id:
            try:
                _runtime_path(directory).unlink()
            except FileNotFoundError:
                pass
    return 0


def _main(arguments: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("command", choices=["_serve"])
    parser.add_argument("--state-dir", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_OTLP_PORT)
    parser.add_argument("--max-request-bytes", type=int, default=DEFAULT_MAX_REQUEST_BYTES)
    options = parser.parse_args(arguments)
    return _collector_process(
        options.state_dir, options.host, options.port, options.max_request_bytes
    )


if __name__ == "__main__":
    raise SystemExit(_main())
