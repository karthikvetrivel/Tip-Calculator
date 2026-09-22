"""Read-only post-run metadata discovery for Hermes Agent.

Hermes keeps session metadata in ``state.db``.  This adapter opens that file
with SQLite's read-only URI mode, introspects the installed schema, and reads
only allowlisted columns from ``sessions`` and ``session_model_usage``.  It
never reads the ``messages`` table.

``session_model_usage`` holds one row per model a session billed, which is the
only record of the models it used, and only where that table was read in full.
Nothing in either table counts a prompt, a turn, or a tool call, so every other
workflow field stays ``None``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
import sqlite3
import stat
from typing import Mapping

from .base import AdapterSnapshot, NativeMetadata


HARNESS_ID = "hermes"
MAX_SESSIONS = 4096
MAX_USAGE_ROWS = 4096
MAX_TEXT_FIELD = 512

_PATH_COLUMNS = ("cwd", "git_repo_root")
_MARKER_COLUMNS = (
    "started_at",
    "ended_at",
    "last_activity_at",
    "message_count",
    "tool_call_count",
    "input_tokens",
    "output_tokens",
    "cache_read_tokens",
    "cache_write_tokens",
    "reasoning_tokens",
    "estimated_cost_usd",
    "actual_cost_usd",
    "model",
)
_METADATA_COLUMNS = (
    "model",
    "billing_provider",
    "estimated_cost_usd",
    "actual_cost_usd",
    "cost_status",
    "cost_source",
)
_USAGE_COLUMNS = (
    "model",
    "billing_provider",
    "estimated_cost_usd",
    "actual_cost_usd",
    "cost_status",
    "cost_source",
    "api_call_count",
)


@dataclass(frozen=True, slots=True)
class _HermesSession:
    marker: tuple[object, ...]
    cwd: Path | None
    git_repo_root: Path | None


@dataclass(frozen=True, slots=True)
class _HermesState:
    repo: Path
    database: Path
    sessions: tuple[tuple[str, tuple[object, ...]], ...]


def _short_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip()
    if not value or len(value) > MAX_TEXT_FIELD:
        return None
    if any(ord(character) < 32 for character in value):
        return None
    return value


def _resolved_path(value: object) -> Path | None:
    if not isinstance(value, str) or not value or len(value) > 8192:
        return None
    try:
        return Path(value).expanduser().resolve(strict=False)
    except (OSError, RuntimeError, ValueError):
        return None


def _environment_path(value: str, env: Mapping[str, str]) -> Path:
    home = env.get("HOME", "").strip()
    if home and (value == "~" or value.startswith("~/")):
        value = os.fspath(Path(home) / ("" if value == "~" else value[2:]))
    return Path(value).resolve(strict=False)


def _hermes_home(env: Mapping[str, str]) -> Path | None:
    explicit = env.get("HERMES_HOME", "").strip()
    if explicit:
        try:
            return _environment_path(explicit, env)
        except (OSError, RuntimeError, ValueError):
            return None
    home = env.get("HOME", "").strip()
    if not home:
        return None
    try:
        return (Path(home) / ".hermes").resolve(strict=False)
    except (OSError, RuntimeError, ValueError):
        return None


def _open_read_only(path: Path) -> sqlite3.Connection:
    # Do not add ``immutable=1``: it can hide committed records that are still
    # in an active WAL.  ``mode=ro`` and query_only prevent adapter writes.
    connection = sqlite3.connect(
        f"{path.resolve(strict=False).as_uri()}?mode=ro",
        uri=True,
        timeout=1,
    )
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only = ON")
    connection.execute("PRAGMA busy_timeout = 1000")
    return connection


def _table_columns(connection: sqlite3.Connection, table: str) -> set[str]:
    if table not in {"sessions", "session_model_usage"}:
        return set()
    present = connection.execute(
        "SELECT 1 FROM sqlite_schema WHERE type = 'table' AND name = ?",
        (table,),
    ).fetchone()
    if present is None:
        return set()
    return {str(row[1]) for row in connection.execute(f"PRAGMA table_info({table})")}


def _regular_file(path: Path) -> bool:
    try:
        return stat.S_ISREG(path.stat().st_mode)
    except OSError:
        return False


def _scan_sessions(path: Path) -> dict[str, _HermesSession] | None:
    if not path.exists():
        return {}
    if not _regular_file(path):
        return None
    try:
        connection = _open_read_only(path)
    except (OSError, sqlite3.Error, ValueError):
        return None
    try:
        columns = _table_columns(connection, "sessions")
        if "id" not in columns or not columns.intersection(_PATH_COLUMNS):
            return None
        selected = ["id"]
        selected.extend(column for column in _PATH_COLUMNS if column in columns)
        marker_columns = [column for column in _MARKER_COLUMNS if column in columns]
        if not marker_columns:
            return None
        selected.extend(marker_columns)
        quoted = ", ".join(f'"{column}"' for column in selected)
        rows = connection.execute(f"SELECT {quoted} FROM sessions LIMIT ?", (MAX_SESSIONS + 1,)).fetchall()
        if len(rows) > MAX_SESSIONS:
            return None

        sessions: dict[str, _HermesSession] = {}
        for row in rows:
            session_id = _short_text(row["id"])
            if session_id is None or session_id in sessions:
                return None
            sessions[session_id] = _HermesSession(
                marker=tuple(row[column] for column in marker_columns),
                cwd=_resolved_path(row["cwd"]) if "cwd" in columns else None,
                git_repo_root=(
                    _resolved_path(row["git_repo_root"]) if "git_repo_root" in columns else None
                ),
            )
        return sessions
    except sqlite3.Error:
        return None
    finally:
        connection.close()


def snapshot_hermes(
    repo: str | os.PathLike[str],
    env: Mapping[str, str],
) -> AdapterSnapshot | None:
    """Record Hermes session ids and safe update markers before a run."""

    home = _hermes_home(env)
    if home is None:
        return None
    database = home / "state.db"
    sessions = _scan_sessions(database)
    if sessions is None:
        return None
    return AdapterSnapshot(
        HARNESS_ID,
        _HermesState(
            repo=Path(repo).resolve(strict=False),
            database=database,
            sessions=tuple(sorted((session_id, item.marker) for session_id, item in sessions.items())),
        ),
    )


def _cost(value: object) -> tuple[float | None, bool]:
    if value is None:
        return None, False
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None, True
    result = float(value)
    if not math.isfinite(result) or result < 0:
        return None, True
    return result, False


def _one_or_none(values: set[str], label: str, warnings: list[str]) -> str | None:
    if len(values) == 1:
        return next(iter(values))
    if len(values) > 1:
        warnings.append(f"Hermes session used multiple {label}; {label[:-1]} was not collapsed")
    return None


def _row_cost(
    row: Mapping[str, object],
    warnings: list[str],
) -> tuple[float | None, str | None]:
    selected = _billing_value(row, warnings, "session")
    if selected is None:
        return None, None
    kind, value = selected
    return value, f"hermes-session-{kind}"


def _billing_value(
    row: Mapping[str, object],
    warnings: list[str],
    label: str,
) -> tuple[str, float] | None:
    actual, invalid_actual = _cost(row.get("actual_cost_usd"))
    estimated, invalid_estimated = _cost(row.get("estimated_cost_usd"))
    if invalid_actual or invalid_estimated:
        warnings.append(f"Hermes reported an invalid {label} cost; native cost was ignored")
        return None

    status_value = _short_text(row.get("cost_status"))
    status = status_value.casefold() if status_value is not None else None
    if status == "actual":
        if actual is None or (actual == 0 and estimated is not None and estimated > 0):
            warnings.append(f"Hermes {label} actual cost was incomplete; native cost was ignored")
            return None
        return "actual", actual
    if status == "estimated":
        if estimated is None or (actual is not None and actual > 0):
            warnings.append(f"Hermes {label} mixed actual and estimated cost; native cost was ignored")
            return None
        return "estimated", estimated
    if status == "included":
        if (actual or 0) > 0 or (estimated or 0) > 0:
            warnings.append(f"Hermes {label} included cost conflicted with USD values")
            return None
        return "included", 0.0
    if status not in {None, "unknown"}:
        warnings.append(f"Hermes {label} used an unknown cost status; native cost was ignored")
        return None

    if actual is not None and actual > 0:
        return "actual", actual
    if estimated is not None:
        return "estimated", estimated
    return None


def _usage_cost(
    rows: list[dict[str, object]],
    warnings: list[str],
) -> tuple[float | None, str | None]:
    if not rows:
        return None, None
    selected: list[tuple[str, float]] = []
    for row in rows:
        value = _billing_value(row, warnings, "per-model")
        if value is None:
            return None, None
        selected.append(value)

    kinds = {kind for kind, _ in selected}
    if len(kinds) != 1:
        warnings.append("Hermes per-model rows mixed billing kinds; native cost was ignored")
        return None, None
    kind = next(iter(kinds))
    try:
        total = math.fsum(value for _, value in selected)
    except OverflowError:
        total = math.inf
    if not math.isfinite(total):
        warnings.append("Hermes per-model costs overflowed; native cost was ignored")
        return None, None
    return total, f"hermes-model-usage-{kind}"


def _selected_metadata(
    path: Path,
    session_id: str,
    *,
    allow_cost: bool,
) -> NativeMetadata:
    warnings: list[str] = []
    try:
        connection = _open_read_only(path)
    except (OSError, sqlite3.Error, ValueError):
        return NativeMetadata(
            harness_id=HARNESS_ID,
            native_session_id=session_id,
            warnings=("Hermes state database became unavailable; native details were ignored",),
        )
    try:
        session_columns = _table_columns(connection, "sessions")
        selected = [column for column in _METADATA_COLUMNS if column in session_columns]
        if "id" not in session_columns:
            raise sqlite3.DatabaseError("sessions.id is unavailable")
        quoted = ", ".join(f'"{column}"' for column in selected)
        projection = quoted or "id"
        row = connection.execute(
            f"SELECT {projection} FROM sessions WHERE id = ? LIMIT 2",
            (session_id,),
        ).fetchall()
        if len(row) != 1:
            return NativeMetadata(
                harness_id=HARNESS_ID,
                native_session_id=session_id,
                warnings=("Hermes selected session changed during metadata collection",),
            )
        session = {column: row[0][column] for column in selected}

        usage_rows: list[dict[str, object]] = []
        usage_truncated = False
        usage_ordered = True
        usage_columns = _table_columns(connection, "session_model_usage")
        usage_readable = "session_id" in usage_columns
        if usage_readable:
            usage_selected = [column for column in _USAGE_COLUMNS if column in usage_columns]
            usage_projection = ", ".join(f'"{column}"' for column in usage_selected) or "session_id"
            filtered = (
                f"SELECT {usage_projection} FROM session_model_usage WHERE session_id = ?"
            )
            try:
                # ``rowid`` is the order SQLite stored the rows in, which is the
                # order this session first billed each model.  No allowlisted
                # column carries a time, so it is the only order to read.
                raw_usage = connection.execute(
                    f"{filtered} ORDER BY rowid LIMIT ?",
                    (session_id, MAX_USAGE_ROWS + 1),
                ).fetchall()
            except sqlite3.Error:
                # A table with no rowid still holds the costs.  The order
                # between its rows is then unknown rather than guessed.
                usage_ordered = False
                raw_usage = connection.execute(
                    f"{filtered} LIMIT ?",
                    (session_id, MAX_USAGE_ROWS + 1),
                ).fetchall()
            if len(raw_usage) > MAX_USAGE_ROWS:
                usage_truncated = True
                warnings.append("Hermes per-model usage exceeded the safe row limit; usage details were ignored")
            else:
                usage_rows = [
                    {column: usage_row[column] for column in usage_selected}
                    for usage_row in raw_usage
                ]
    except sqlite3.Error:
        return NativeMetadata(
            harness_id=HARNESS_ID,
            native_session_id=session_id,
            warnings=("Hermes metadata schema was unavailable; native details were ignored",),
        )
    finally:
        connection.close()

    models = {
        value
        for value in [_short_text(session.get("model")), *(_short_text(row.get("model")) for row in usage_rows)]
        if value is not None
    }
    providers = {
        value
        for value in [
            _short_text(session.get("billing_provider")),
            *(_short_text(row.get("billing_provider")) for row in usage_rows),
        ]
        if value is not None
    }
    model = _one_or_none(models, "models", warnings)
    provider = _one_or_none(providers, "providers", warnings)

    # The distinct models of ``session_model_usage``, in the order the table
    # stored them, are the models this session used, so each one after the first
    # is a model change.  A resumed session's rows cover the runs before this
    # one as its costs do, and a truncated or unordered read proves nothing
    # about either, so each leaves the history unknown.  A session whose
    # per-model table was read and held no row ran the one model that
    # ``sessions.model`` names; a table that was never read names nothing.
    ordered_models: tuple[str, ...] | None = None
    if allow_cost and usage_readable and usage_ordered and not usage_truncated:
        listed = [_short_text(row.get("model")) for row in usage_rows]
        if not usage_rows:
            listed = [_short_text(session.get("model"))]
        used: list[str] = []
        for value in listed:
            if value is not None and value not in used:
                used.append(value)
        ordered_models = tuple(used) or None

    cost_usd: float | None = None
    cost_source: str | None = None
    if not allow_cost:
        warnings.append("Hermes resumed an existing session; cumulative native cost was ignored")
    elif usage_truncated:
        warnings.append("Hermes could not prove a complete usage total; native cost was ignored")
    elif usage_rows:
        cost_usd, cost_source = _usage_cost(usage_rows, warnings)
    else:
        cost_usd, cost_source = _row_cost(session, warnings)

    return NativeMetadata(
        harness_id=HARNESS_ID,
        native_session_id=session_id,
        model=model,
        provider=provider,
        cost_usd=cost_usd,
        cost_source=cost_source,
        warnings=tuple(warnings),
        model_switch_count=(
            None if ordered_models is None else len(ordered_models) - 1
        ),
        models=ordered_models,
    )


def finalize_hermes(
    adapter_snapshot: AdapterSnapshot,
    repo: str | os.PathLike[str],
    env: Mapping[str, str],
) -> NativeMetadata | None:
    """Return metadata when exactly one matching Hermes session changed."""

    del env  # Use the exact database selected before the wrapped child ran.
    if adapter_snapshot.harness_id != HARNESS_ID or not isinstance(adapter_snapshot.state, _HermesState):
        return None
    state = adapter_snapshot.state
    resolved_repo = Path(repo).resolve(strict=False)
    if resolved_repo != state.repo:
        return None

    sessions = _scan_sessions(state.database)
    if sessions is None:
        return None
    before = dict(state.sessions)
    changed = [
        session_id
        for session_id, item in sessions.items()
        if (item.cwd == state.repo or item.git_repo_root == state.repo)
        and (session_id not in before or before[session_id] != item.marker)
    ]
    if len(changed) != 1:
        return None
    session_id = changed[0]
    return _selected_metadata(
        state.database,
        session_id,
        allow_cost=session_id not in before,
    )


snapshot = snapshot_hermes
finalize = finalize_hermes
