"""Build a read-only attribution report from local evidence and Git notes."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import ast
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import statistics
import subprocess
from typing import Any, Iterable, Mapping
from urllib.parse import quote

from .activity import CONTEXT_LOAD_KINDS, TOOL_CLASSES
from .harnesses import harness_display_name
from .notes import MAX_NOTE_BYTES
from .workflow import (
    build_profile,
    iter_agents,
    merge_activity,
    parse_profile,
    profile_activity,
)
from .runtime import system_subprocess_environment


_NOTES_REF = "refs/notes/attribution"
_TASK_NOTES_REF = "refs/notes/attribution-tasks"
_MAX_NOTE_BYTES = MAX_NOTE_BYTES
_MAX_NOTE_LINES = 1_000_000
_MAX_TARGET_FILES = 2_000
_MAX_TARGET_FILE_BYTES = 1024 * 1024
_MAX_TARGET_TOTAL_BYTES = 32 * 1024 * 1024
_GIT_TIMEOUT_SECONDS = 30
_OID_RE = re.compile(r"^[0-9a-fA-F]{40,64}$")
_BLAME_HEADER_RE = re.compile(
    r"^([0-9a-f]{40,64}) ([0-9]+) ([0-9]+)(?: ([0-9]+))?$"
)


@dataclass(frozen=True, order=True)
class _Origin:
    commit: str
    path: str
    line: int


class _Warnings:
    def __init__(self) -> None:
        self._items: list[str] = []
        self._seen: set[str] = set()

    def add(self, message: str) -> None:
        if message not in self._seen:
            self._seen.add(message)
            self._items.append(message)

    @property
    def items(self) -> list[str]:
        return self._items


def _run_git(
    repo: Path,
    *args: str,
    check: bool = True,
    text: bool = True,
) -> subprocess.CompletedProcess[Any]:
    try:
        result = subprocess.run(
            ["git", "-c", "core.quotePath=false", "-C", str(repo), *args],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=text,
            errors="replace" if text else None,
            env=system_subprocess_environment(),
            timeout=_GIT_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise ValueError(f"Git command timed out: git {args[0] if args else ''}") from exc
    if check and result.returncode != 0:
        stderr = result.stderr.strip() if text else result.stderr.decode(errors="replace").strip()
        raise ValueError(stderr or f"Git command failed: git {' '.join(args)}")
    return result


def _repository_paths(repo: str | Path) -> tuple[Path, Path, Path]:
    requested = Path(repo).expanduser().resolve()
    top = _run_git(requested, "rev-parse", "--show-toplevel").stdout.strip()
    root = Path(top).resolve()
    common_raw = _run_git(root, "rev-parse", "--git-common-dir").stdout.strip()
    common = Path(common_raw)
    if not common.is_absolute():
        common = (root / common).resolve()
    else:
        common = common.resolve()
    git_dir_raw = _run_git(root, "rev-parse", "--git-dir").stdout.strip()
    git_dir = Path(git_dir_raw)
    if not git_dir.is_absolute():
        git_dir = (root / git_dir).resolve()
    else:
        git_dir = git_dir.resolve()
    return root, common, git_dir


def _live_worktrees(
    repo: Path,
    common_dir: Path,
    current_git_dir: Path,
    warnings: _Warnings,
) -> list[dict[str, Any]]:
    """Describe live worktrees without trusting stale administrative entries."""

    listed = _run_git(
        repo,
        "worktree",
        "list",
        "--porcelain",
        "-z",
        check=False,
        text=False,
    )
    if listed.returncode != 0:
        warnings.add("Could not inspect linked worktrees.")
        return [
            {
                "id": str(current_git_dir),
                "path": str(repo),
                "branch": None,
                "current": True,
            }
        ]

    result: dict[Path, dict[str, Any]] = {}
    for record in listed.stdout.split(b"\x00\x00"):
        fields = [field for field in record.split(b"\x00") if field]
        paths = [
            field[len(b"worktree ") :]
            for field in fields
            if field.startswith(b"worktree ")
        ]
        if len(paths) != 1 or any(
            field == b"prunable" or field.startswith(b"prunable ") or field == b"bare"
            for field in fields
        ):
            continue
        checkout = Path(os.fsdecode(paths[0]))
        try:
            if not checkout.is_dir():
                continue
            git_dir_result = _run_git(
                checkout,
                "rev-parse",
                "--path-format=absolute",
                "--git-dir",
                check=False,
                text=False,
            )
            common_result = _run_git(
                checkout,
                "rev-parse",
                "--path-format=absolute",
                "--git-common-dir",
                check=False,
                text=False,
            )
            if git_dir_result.returncode != 0 or common_result.returncode != 0:
                continue
            candidate_git_dir = Path(
                os.fsdecode(git_dir_result.stdout.rstrip(b"\n"))
            ).resolve()
            candidate_common = Path(
                os.fsdecode(common_result.stdout.rstrip(b"\n"))
            ).resolve()
        except OSError:
            continue
        if candidate_common != common_dir:
            continue
        branch_fields = [
            field[len(b"branch ") :]
            for field in fields
            if field.startswith(b"branch ")
        ]
        branch: str | None = None
        if len(branch_fields) == 1:
            branch = os.fsdecode(branch_fields[0])
            if branch.startswith("refs/heads/"):
                branch = branch[len("refs/heads/") :]
        result[candidate_git_dir] = {
            "id": str(candidate_git_dir),
            "path": str(checkout.resolve()),
            "branch": branch,
            "current": candidate_git_dir == current_git_dir,
        }

    if current_git_dir not in result:
        result[current_git_dir] = {
            "id": str(current_git_dir),
            "path": str(repo),
            "branch": None,
            "current": True,
        }
    return sorted(
        result.values(),
        key=lambda item: (not item["current"], item["path"]),
    )


def _resolve_target(repo: Path, target_ref: str) -> str | None:
    if not target_ref or "\x00" in target_ref:
        return None
    result = _run_git(
        repo,
        "rev-parse",
        "--verify",
        "--quiet",
        "--end-of-options",
        f"{target_ref}^{{commit}}",
        check=False,
    )
    if result.returncode != 0:
        return None
    target = result.stdout.strip().lower()
    return target if _OID_RE.fullmatch(target) else None


def _parse_time(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    candidate = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _format_time(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


_OPTIONAL_SESSION_TEXT_FIELDS = (
    "workflow_identity",
    "harness_id",
    "harness_version",
    "native_session_id",
    "native_turn_id",
    "native_agent_id",
    "native_parent_session_id",
    "provider",
    "model_source",
    "harness_source",
    "integration_mode",
    "agent_type",
    "launch_mode",
    "session_source",
    "permission_mode",
    "effort_level",
)

# Counts that only ever grow as a session is observed for longer.
_SESSION_COUNT_FIELDS = (
    "turn_count",
    "prompt_count",
    "interrupt_count",
    "compaction_count",
    "model_switch_count",
    "tool_call_count",
)
_OPTIONAL_SESSION_INT_FIELDS = _SESSION_COUNT_FIELDS + ("duration_ms",)

# The harness reports its own most recent value for these, so a later record
# replaces an earlier one instead of conflicting with it.
_RECENT_SESSION_TEXT_FIELDS = ("permission_mode", "effort_level")
_IDENTITY_SESSION_TEXT_FIELDS = tuple(
    field
    for field in _OPTIONAL_SESSION_TEXT_FIELDS
    if field not in _RECENT_SESSION_TEXT_FIELDS
)

# One session lists at most this many tool calls, and at most this many
# context loads, in the report JSON.
MAX_JSON_ACTIVITY_ROWS = 200

_SESSION_SOURCE_VALUES = {
    "model_source": {
        "reported",
        "command",
        "unknown",
        "harness",
        "native_hook",
        "agent_response",
        # A label a report derived from the models its telemetry named. It is
        # never written to the ledger or to a note; only a report carries it.
        "telemetry",
    },
    "harness_source": {"reported", "command", "native_hook"},
    "integration_mode": {"wrapper", "detected_wrapper", "native_hook"},
    "launch_mode": {"foreground", "background"},
    "session_source": {
        "startup",
        "resume",
        "fork",
        "clear",
        "compact",
        "subagent",
    },
}


def _dominant_model(models: list[Any], counted: Any) -> str | None:
    """Return the model that produced most of a session's output, or None.

    A session that used several models is named by the model that wrote the
    most output tokens, because that is the model a reader would name for the
    work. A tie, or a spool that counted output for fewer than two of them,
    names nobody.
    """

    if not isinstance(counted, Mapping):
        return None
    totals: list[tuple[float, str]] = []
    for name in models:
        if not isinstance(name, str):
            continue
        value = counted.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        number = float(value)
        if math.isfinite(number) and number >= 0:
            totals.append((number, name))
    if len(totals) < 2:
        return None
    totals.sort(key=lambda item: -item[0])
    if totals[0][0] <= 0 or totals[0][0] == totals[1][0]:
        return None
    return totals[0][1]


def telemetry_model(session: Mapping[str, Any], usage: Any) -> str | None:
    """Return the model a session's allocated usage names, or None.

    A harness that reports no model when a session starts leaves the ledger
    saying ``unknown``, while every request event of that session names the
    model that served it. One named model is the answer. Several are a session
    that used more than one, and the model that wrote the most output tokens
    then names it. Where the usage counts no output per model, or two models
    wrote the same amount, the label stays unknown and the usage models list
    them as they always did.

    The answer is a label a report computes. Nothing writes it to the ledger or
    to a note, both of which keep what the harness actually reported.
    """

    model = session.get("model")
    if (
        isinstance(model, str)
        and model.strip()
        and model != "unknown"
        and session.get("model_source") != "unknown"
    ):
        return None
    models = usage.get("models") if isinstance(usage, Mapping) else None
    if not isinstance(models, list) or not models:
        return None
    if len(models) == 1:
        named = models[0]
    else:
        named = _dominant_model(models, usage.get("model_output_tokens"))
    if not isinstance(named, str) or not named.strip() or "\x00" in named:
        return None
    return named


def _normalise_session(
    raw: dict[str, Any],
    *,
    source: str,
    warnings: _Warnings,
) -> dict[str, Any] | None:
    session_id = raw.get("id")
    feature = raw.get("feature")
    model = raw.get("model")
    harness = raw.get("harness")
    if not all(
        isinstance(value, str) and value.strip()
        for value in (session_id, feature, model, harness)
    ):
        warnings.add(f"Ignored a malformed session from {source}.")
        return None

    actor_kind = raw.get("actor_kind", "ai")
    if not isinstance(actor_kind, str) or actor_kind not in {"ai", "manual"}:
        warnings.add(f"Ignored session {session_id!r} with an invalid actor kind.")
        return None
    source_session_id = raw.get("source_session_id")
    if source_session_id is not None and (
        not isinstance(source_session_id, str) or not source_session_id or len(source_session_id) > 256
    ):
        warnings.add(f"Ignored session {session_id!r} with an invalid source session ID.")
        return None

    started = _parse_time(raw.get("started_at"))
    if started is None:
        warnings.add(f"Ignored session {session_id!r} from {source}: invalid start time.")
        return None
    ended_raw = raw.get("ended_at")
    ended = _parse_time(ended_raw) if ended_raw is not None else None
    if ended_raw is not None and ended is None:
        warnings.add(f"Session {session_id!r} from {source} has an invalid end time.")

    cost_raw = raw.get("cost_usd")
    cost: float | None
    if cost_raw is None:
        cost = None
    elif (
        isinstance(cost_raw, (int, float))
        and not isinstance(cost_raw, bool)
        and math.isfinite(float(cost_raw))
        and float(cost_raw) >= 0
    ):
        cost = float(cost_raw)
    else:
        cost = None
        warnings.add(f"Session {session_id!r} from {source} has an invalid reported cost.")

    exit_raw = raw.get("exit_code")
    exit_code = exit_raw if isinstance(exit_raw, int) and not isinstance(exit_raw, bool) else None
    label_source = raw.get("label_source")
    if not isinstance(label_source, str) or not label_source:
        label_source = "reported"
    cost_source = raw.get("cost_source")
    if cost is None:
        cost_source = None
    elif not isinstance(cost_source, str) or not cost_source:
        cost_source = "reported"

    task_id = raw.get("task_id")
    if not isinstance(task_id, str) or not task_id:
        task_id = None
    membership_source = raw.get("membership_source")
    if not isinstance(membership_source, str) or not membership_source:
        membership_source = "legacy"
    role = raw.get("role")
    if role not in {"planning", "implementation", "testing", "review", "other"}:
        role = "implementation"
    summary = raw.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        summary = None
    parent_session_id = raw.get("parent_session_id")
    if not isinstance(parent_session_id, str) or not parent_session_id:
        parent_session_id = None
    token_raw = raw.get("token_count")
    if isinstance(token_raw, int) and not isinstance(token_raw, bool) and token_raw >= 0:
        token_count: int | None = token_raw
    else:
        token_count = None
        if token_raw is not None:
            warnings.add(f"Session {session_id!r} from {source} has an invalid token count.")
    token_source = raw.get("token_source")
    if token_count is None:
        token_source = None
    elif not isinstance(token_source, str) or not token_source:
        token_source = "reported"
    includes_children = raw.get("usage_includes_children", False)
    includes_children = includes_children is True or includes_children == 1
    outcome = raw.get("outcome")
    if outcome not in {"completed", "failed", "interrupted", "abandoned"}:
        if exit_code == 0:
            outcome = "completed"
        elif exit_code == 130:
            outcome = "interrupted"
        elif exit_code is not None:
            outcome = "failed"
        else:
            outcome = None

    result = {
        "id": session_id,
        "worktree_id": (
            raw.get("worktree_id")
            if isinstance(raw.get("worktree_id"), str) and raw.get("worktree_id")
            else None
        ),
        "feature_source": raw.get("feature_source") if isinstance(raw.get("feature_source"), str) else None,
        "task_id": task_id,
        "feature": feature,
        "model": model,
        "harness": harness,
        "actor_kind": actor_kind,
        "source_session_id": source_session_id,
        "label_source": label_source,
        "membership_source": membership_source,
        "role": role,
        "summary": summary,
        "parent_session_id": parent_session_id,
        "token_count": token_count,
        "token_source": token_source,
        "cost_usd": cost,
        "cost_source": cost_source,
        "usage_includes_children": includes_children,
        "started_at": _format_time(started),
        "ended_at": _format_time(ended) if ended else None,
        "exit_code": exit_code,
        "outcome": outcome,
    }
    for field in _OPTIONAL_SESSION_INT_FIELDS:
        value = raw.get(field)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            result[field] = value
        else:
            result[field] = None
            if value is not None:
                warnings.add(
                    f"Session {session_id!r} from {source} has an invalid {field}."
                )
    truncated = raw.get("activity_truncated")
    result["activity_truncated"] = truncated is True or truncated == 1

    for field in _OPTIONAL_SESSION_TEXT_FIELDS:
        value = raw.get(field)
        if value is None:
            result[field] = None
        elif (
            isinstance(value, str)
            and value.strip()
            and "\x00" not in value
            and (
                field not in _SESSION_SOURCE_VALUES
                or value in _SESSION_SOURCE_VALUES[field]
            )
        ):
            result[field] = value
        else:
            result[field] = None
            warnings.add(
                f"Session {session_id!r} from {source} has an invalid {field}."
            )
    harness_id = result.get("harness_id")
    if isinstance(harness_id, str):
        try:
            result["harness"] = harness_display_name(harness_id)
        except ValueError:
            # Keep the recorded display label for a forward-compatible ID that
            # this registry version does not know yet.
            pass
    return result


def _load_local_sessions(
    common_dir: Path, warnings: _Warnings
) -> list[dict[str, Any]]:
    database = common_dir / "attribution" / "ledger.sqlite3"
    if not database.is_file():
        return []
    if database.stat().st_size > 256 * 1024 * 1024:
        warnings.add("The local attribution ledger is too large to read safely.")
        return []
    uri = f"file:{quote(str(database))}?mode=ro"
    has_unscoped_sessions = False
    try:
        connection = sqlite3.connect(uri, uri=True, timeout=1)
        connection.row_factory = sqlite3.Row
        try:
            columns = {
                row[1] for row in connection.execute("PRAGMA table_info(sessions)").fetchall()
            }
            tables = {
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                ).fetchall()
            }
            optional_columns = []
            optional = {
                "worktree_id": "NULL",
                "native_session_id": "NULL",
                "native_turn_id": "NULL",
                "native_agent_id": "NULL",
                "native_parent_session_id": "NULL",
                "harness_id": "NULL",
                "harness_version": "NULL",
                "provider": "NULL",
                "model_source": "NULL",
                "harness_source": "NULL",
                "integration_mode": "NULL",
                "feature_source": "NULL",
                "source_session_id": "NULL",
                "actor_kind": "'ai'",
                "task_id": "NULL",
                "membership_source": "'legacy'",
                "role": "'implementation'",
                "summary": "NULL",
                "parent_session_id": "NULL",
                "token_count": "NULL",
                "token_source": "NULL",
                "usage_includes_children": "0",
                "outcome": "NULL",
                "agent_type": "NULL",
                "launch_mode": "NULL",
                "session_source": "NULL",
                "permission_mode": "NULL",
                "effort_level": "NULL",
                "turn_count": "0",
                "prompt_count": "0",
                "interrupt_count": "0",
                "compaction_count": "0",
                "model_switch_count": "0",
                "tool_call_count": "0",
                "duration_ms": "NULL",
                "activity_truncated": "0",
            }
            for field, default in optional.items():
                optional_columns.append(
                    field if field in columns else f"{default} AS {field}"
                )
            optional_fields = ", ".join(optional_columns)
            select = f"""
                SELECT id, feature, model, harness, label_source, cost_usd,
                       cost_source, started_at, ended_at, exit_code, {optional_fields}
                FROM sessions AS s
            """
            evidence = "s.cost_usd IS NOT NULL"
            if "task_id" in columns:
                evidence += " OR s.task_id IS NOT NULL"
            if "edits" in tables:
                evidence += (
                    " OR EXISTS (SELECT 1 FROM edits "
                    "WHERE edits.session_id = s.id)"
                )
            # Workflow evidence stands on its own. A subagent that only read
            # files still belongs in the agent tree of the task it served.
            if "agent_type" in columns:
                evidence += " OR s.agent_type IS NOT NULL"
            if "tool_calls" in tables:
                evidence += (
                    " OR EXISTS (SELECT 1 FROM tool_calls "
                    "WHERE tool_calls.session_id = s.id)"
                )
            # An agent that loaded its instructions and then answered from them
            # leaves a context load and nothing else.
            if "context_loads" in tables:
                evidence += (
                    " OR EXISTS (SELECT 1 FROM context_loads "
                    "WHERE context_loads.session_id = s.id)"
                )
            # A session that only answered prompts still worked on the task.
            # Its counts are the only evidence a turn without a tool call
            # leaves, so the report would otherwise drop the agent entirely.
            for counter in ("prompt_count", "turn_count", "interrupt_count"):
                if counter in columns:
                    evidence += f" OR s.{counter} > 0"
            if "worktree_id" in columns:
                evidence += " OR s.worktree_id IS NULL"
            else:
                # A schema without this column predates linked-worktree identity.
                # Keep its sessions as metadata, but never treat them as scoped
                # file evidence.
                evidence = "1"
            rows = connection.execute(
                select + f" WHERE {evidence} ORDER BY started_at, id"
            ).fetchall()
            has_unscoped_sessions = bool(rows) and (
                "worktree_id" not in columns
                or any(row["worktree_id"] is None for row in rows)
            )
        finally:
            connection.close()
    except (OSError, sqlite3.Error) as exc:
        warnings.add(f"Could not read the local attribution ledger: {exc}.")
        return []

    if has_unscoped_sessions:
        warnings.add(
            "Legacy sessions without a worktree identity appear as metadata, "
            "but their edits remain excluded from attribution evidence."
        )

    sessions: list[dict[str, Any]] = []
    for row in rows:
        normalised = _normalise_session(
            dict(row), source="the local ledger", warnings=warnings
        )
        if normalised is not None:
            sessions.append(normalised)
    return sessions


def _load_local_model_switches(
    common_dir: Path, warnings: _Warnings
) -> dict[str, list[dict[str, Any]]]:
    """Return each session's recorded model changes, oldest first."""

    database = common_dir / "attribution" / "ledger.sqlite3"
    if not database.is_file():
        return {}
    uri = f"file:{quote(str(database))}?mode=ro"
    switches: dict[str, list[dict[str, Any]]] = {}
    try:
        connection = sqlite3.connect(uri, uri=True, timeout=1)
        connection.row_factory = sqlite3.Row
        try:
            present = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'model_switches'"
            ).fetchone()
            if present is None:
                return {}
            rows = connection.execute(
                """
                SELECT session_id, from_model, to_model, source, occurred_at
                FROM model_switches
                ORDER BY session_id, id
                """
            ).fetchall()
        finally:
            connection.close()
    except (OSError, sqlite3.Error) as exc:
        warnings.add(f"Could not read local model switches: {exc}.")
        return {}
    for row in rows:
        switches.setdefault(str(row["session_id"]), []).append(
            {
                "from_model": row["from_model"],
                "to_model": row["to_model"],
                "source": row["source"],
                "occurred_at": row["occurred_at"],
            }
        )
    return switches


def _load_local_tool_calls(
    common_dir: Path, warnings: _Warnings
) -> dict[str, dict[str, Any]]:
    """Return each session's recorded tool calls, oldest first.

    Only the local ledger holds this activity. A note carries the counts of a
    session, never the locators of the files it read.
    """

    database = common_dir / "attribution" / "ledger.sqlite3"
    if not database.is_file():
        return {}
    uri = f"file:{quote(str(database))}?mode=ro"
    by_session: dict[str, dict[str, Any]] = {}
    try:
        connection = sqlite3.connect(uri, uri=True, timeout=1)
        connection.row_factory = sqlite3.Row
        try:
            present = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'tool_calls'"
            ).fetchone()
            if present is None:
                return {}
            # The counts cover every stored row; the listed rows stop at the
            # JSON limit.
            counts = connection.execute(
                """
                SELECT session_id, tool_class, COUNT(*) AS count
                FROM tool_calls
                GROUP BY session_id, tool_class
                """
            ).fetchall()
            # A sequence counts from 1 within its session, so the limit is a
            # bound on the read as well as on the output.
            rows = connection.execute(
                """
                SELECT session_id, tool_use_id, sequence, tool_name, tool_class,
                       locator, succeeded, duration_ms, compaction_epoch,
                       occurred_at
                FROM tool_calls
                WHERE sequence <= ?
                ORDER BY session_id, sequence, id
                """,
                (MAX_JSON_ACTIVITY_ROWS,),
            ).fetchall()
        finally:
            connection.close()
    except (OSError, sqlite3.Error) as exc:
        warnings.add(f"Could not read local tool calls: {exc}.")
        return {}

    def entry_for(session_id: str) -> dict[str, Any]:
        return by_session.setdefault(
            session_id,
            {
                "tool_calls": [],
                "tool_calls_truncated": False,
                "tool_call_counts": dict.fromkeys(TOOL_CLASSES, 0),
            },
        )

    for row in counts:
        entry = entry_for(str(row["session_id"]))
        tool_class = str(row["tool_class"])
        if tool_class in entry["tool_call_counts"]:
            entry["tool_call_counts"][tool_class] = int(row["count"])
    for entry in by_session.values():
        entry["tool_calls_truncated"] = (
            sum(entry["tool_call_counts"].values()) > MAX_JSON_ACTIVITY_ROWS
        )
    for row in rows:
        # The two statements do not share one snapshot, so a hook that inserted
        # the first row of a new session between them names a session the
        # counts do not hold. The report degrades; it does not fail.
        entry = entry_for(str(row["session_id"]))
        if len(entry["tool_calls"]) >= MAX_JSON_ACTIVITY_ROWS:
            continue
        entry["tool_calls"].append(
            {
                "tool_use_id": row["tool_use_id"],
                "sequence": row["sequence"],
                "tool_name": row["tool_name"],
                "tool_class": row["tool_class"],
                "locator": row["locator"],
                "succeeded": (
                    None if row["succeeded"] is None else bool(row["succeeded"])
                ),
                "duration_ms": row["duration_ms"],
                "compaction_epoch": row["compaction_epoch"],
                "occurred_at": row["occurred_at"],
            }
        )
    return by_session


def _load_local_context_loads(
    common_dir: Path, warnings: _Warnings
) -> dict[str, dict[str, Any]]:
    """Return each session's recorded context loads, oldest first.

    Only the local ledger holds them. A note carries what a session loaded as
    counts and repo-relative paths, never as a hash of anything it read.
    """

    database = common_dir / "attribution" / "ledger.sqlite3"
    if not database.is_file():
        return {}
    uri = f"file:{quote(str(database))}?mode=ro"
    by_session: dict[str, dict[str, Any]] = {}
    try:
        connection = sqlite3.connect(uri, uri=True, timeout=1)
        connection.row_factory = sqlite3.Row
        try:
            present = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'context_loads'"
            ).fetchone()
            if present is None:
                return {}
            # The counts cover every stored row; the listed rows stop at the
            # JSON limit. A load has no sequence of its own, so the limit is
            # applied per session in the read rather than to the whole table.
            counts = connection.execute(
                """
                SELECT session_id, kind, COUNT(*) AS count
                FROM context_loads
                GROUP BY session_id, kind
                """
            ).fetchall()
            rows = connection.execute(
                """
                SELECT session_id, kind, locator, size_bytes,
                       memory_type, load_reason, turn_id, related_session_id,
                       compaction_epoch, occurred_at
                FROM (
                    SELECT *, ROW_NUMBER() OVER (
                        PARTITION BY session_id ORDER BY id
                    ) AS position
                    FROM context_loads
                )
                WHERE position <= ?
                ORDER BY session_id, id
                """,
                (MAX_JSON_ACTIVITY_ROWS,),
            ).fetchall()
        finally:
            connection.close()
    except (OSError, sqlite3.Error) as exc:
        warnings.add(f"Could not read local context loads: {exc}.")
        return {}

    def entry_for(session_id: str) -> dict[str, Any]:
        return by_session.setdefault(
            session_id,
            {
                "context_loads": [],
                "context_loads_truncated": False,
                "context_load_counts": dict.fromkeys(CONTEXT_LOAD_KINDS, 0),
            },
        )

    for row in counts:
        entry = entry_for(str(row["session_id"]))
        kind = str(row["kind"])
        if kind in entry["context_load_counts"]:
            entry["context_load_counts"][kind] = int(row["count"])
    for entry in by_session.values():
        entry["context_loads_truncated"] = (
            sum(entry["context_load_counts"].values()) > MAX_JSON_ACTIVITY_ROWS
        )
    for row in rows:
        # The two statements do not share one snapshot, so a hook that inserted
        # the first row of a new session between them names a session the
        # counts do not hold. The report degrades; it does not fail.
        entry = entry_for(str(row["session_id"]))
        # The hash of a load stays in the ledger, where a later join reads it,
        # and out of the JSON, as a tool call's locator hash already does. Both
        # digest content this project refuses to store, and a report says what
        # a session loaded, not what the text of it was.
        entry["context_loads"].append(
            {
                "kind": row["kind"],
                "locator": row["locator"],
                "size_bytes": row["size_bytes"],
                "memory_type": row["memory_type"],
                "load_reason": row["load_reason"],
                "turn_id": row["turn_id"],
                "related_session_id": row["related_session_id"],
                "compaction_epoch": row["compaction_epoch"],
                "occurred_at": row["occurred_at"],
            }
        )
    return by_session


def _load_local_instruction_files(
    common_dir: Path, warnings: _Warnings
) -> dict[str, list[dict[str, Any]]]:
    """Return the distinct instruction files each session loaded.

    The workflow profile reads this list rather than the loads above, which
    stop at the JSON row limit that a session's prompts and subagent results
    share. A file that a busy session loaded after that limit still belongs in
    the profile its own note carries, so it belongs in this report as well.
    """

    database = common_dir / "attribution" / "ledger.sqlite3"
    if not database.is_file():
        return {}
    uri = f"file:{quote(str(database))}?mode=ro"
    by_session: dict[str, list[dict[str, Any]]] = {}
    try:
        connection = sqlite3.connect(uri, uri=True, timeout=1)
        connection.row_factory = sqlite3.Row
        try:
            present = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'context_loads'"
            ).fetchone()
            if present is None:
                return {}
            rows = connection.execute(
                """
                SELECT DISTINCT session_id, locator, memory_type
                FROM context_loads
                WHERE kind = 'instruction_file' AND locator IS NOT NULL
                ORDER BY session_id, locator, memory_type
                """
            ).fetchall()
        finally:
            connection.close()
    except (OSError, sqlite3.Error) as exc:
        warnings.add(f"Could not read local instruction files: {exc}.")
        return {}
    for row in rows:
        by_session.setdefault(str(row["session_id"]), []).append(
            {
                "kind": "instruction_file",
                "locator": row["locator"],
                "memory_type": row["memory_type"],
            }
        )
    return by_session


def _normalise_task(
    raw: dict[str, Any], *, source: str, warnings: _Warnings
) -> dict[str, Any] | None:
    task_id = raw.get("id")
    name = raw.get("name")
    if not isinstance(task_id, str) or not task_id or not isinstance(name, str) or not name:
        warnings.add(f"Ignored a malformed task from {source}.")
        return None
    kind = raw.get("kind")
    if kind not in {"feature", "pull_request", "unresolved"}:
        kind = "feature"
    state = raw.get("state")
    if state not in {"active", "shipped", "abandoned", "merged"}:
        state = "active"
    created = _parse_time(raw.get("created_at"))
    updated = _parse_time(raw.get("updated_at"))
    if created is None or updated is None:
        warnings.add(f"Ignored task {task_id!r} from {source}: invalid time.")
        return None
    result = {
        "id": task_id,
        "name": name,
        "kind": kind,
        "pr_ref": raw.get("pr_ref") if isinstance(raw.get("pr_ref"), str) else None,
        "pr_url": raw.get("pr_url") if isinstance(raw.get("pr_url"), str) else None,
        "branch": raw.get("branch") if isinstance(raw.get("branch"), str) else None,
        "state": state,
        "inference_source": (
            raw.get("inference_source")
            if isinstance(raw.get("inference_source"), str)
            else "unknown"
        ),
        "created_at": _format_time(created),
        "updated_at": _format_time(updated),
        "merged_into": (
            raw.get("merged_into") if isinstance(raw.get("merged_into"), str) else None
        ),
    }
    return result


def _load_local_tasks(common_dir: Path, warnings: _Warnings) -> list[dict[str, Any]]:
    database = common_dir / "attribution" / "ledger.sqlite3"
    if not database.is_file() or database.stat().st_size > 256 * 1024 * 1024:
        return []
    uri = f"file:{quote(str(database))}?mode=ro"
    try:
        connection = sqlite3.connect(uri, uri=True, timeout=1)
        connection.row_factory = sqlite3.Row
        try:
            exists = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'tasks'"
            ).fetchone()
            rows = connection.execute(
                "SELECT * FROM tasks ORDER BY updated_at, id"
            ).fetchall() if exists else []
        finally:
            connection.close()
    except (OSError, sqlite3.Error) as exc:
        warnings.add(f"Could not read local tasks: {exc}.")
        return []
    result: list[dict[str, Any]] = []
    for row in rows:
        task = _normalise_task(dict(row), source="the local ledger", warnings=warnings)
        if task is not None:
            result.append(task)
    return result


def _validate_range(raw: Any, *, where: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError(f"{where} contains a non-object range")
    start = raw.get("start")
    end = raw.get("end")
    session_id = raw.get("session_id")
    if (
        not isinstance(start, int)
        or isinstance(start, bool)
        or not isinstance(end, int)
        or isinstance(end, bool)
        or start < 1
        or end < start
        or end - start + 1 > _MAX_NOTE_LINES
        or not isinstance(session_id, str)
        or not session_id
    ):
        raise ValueError(f"{where} contains an invalid range")
    validated = {"start": start, "end": end, "session_id": session_id}
    tool_use_id = raw.get("tool_use_id")
    if isinstance(tool_use_id, str) and tool_use_id and len(tool_use_id) <= 256:
        # The tool call that wrote the range, kept for the local trace lookup.
        validated["tool_use_id"] = tool_use_id
    return validated


def _parse_note(
    raw_text: str,
    annotated_commit: str,
    *,
    warnings: _Warnings | None = None,
) -> dict[str, Any]:
    try:
        raw = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError("invalid JSON") from exc
    if not isinstance(raw, dict) or raw.get("version") != 1:
        raise ValueError("unsupported or missing version")
    commit = raw.get("commit")
    if not isinstance(commit, str) or commit.lower() != annotated_commit:
        raise ValueError("commit field does not match the noted commit")
    if _parse_time(raw.get("recorded_at")) is None:
        raise ValueError("invalid recorded_at time")

    raw_sessions = raw.get("sessions")
    raw_files = raw.get("files")
    raw_revisions = raw.get("revisions", [])
    if not isinstance(raw_sessions, list) or not isinstance(raw_files, list):
        raise ValueError("sessions and files must be arrays")
    if not isinstance(raw_revisions, list):
        raise ValueError("revisions must be an array")

    files: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    total_range_lines = 0
    for index, file_raw in enumerate(raw_files):
        if not isinstance(file_raw, dict):
            raise ValueError(f"file {index} is not an object")
        path = file_raw.get("path")
        added = file_raw.get("added_lines")
        ranges_raw = file_raw.get("ranges")
        if (
            not isinstance(path, str)
            or not path
            or "\x00" in path
            or path in seen_paths
            or not isinstance(added, int)
            or isinstance(added, bool)
            or added < 0
            or not isinstance(ranges_raw, list)
        ):
            raise ValueError(f"file {index} is invalid")
        seen_paths.add(path)
        ranges = [
            _validate_range(item, where=f"file {path!r}") for item in ranges_raw
        ]
        covered_count = 0
        previous_end = 0
        for item in sorted(ranges, key=lambda item: item["start"]):
            if item["start"] <= previous_end:
                raise ValueError(f"file {path!r} contains overlapping ranges")
            covered_count += item["end"] - item["start"] + 1
            previous_end = item["end"]
        if covered_count > added:
            raise ValueError(f"file {path!r} attributes more lines than it added")
        total_range_lines += sum(item["end"] - item["start"] + 1 for item in ranges)
        if total_range_lines > _MAX_NOTE_LINES:
            raise ValueError("note contains too many attributed lines")
        files.append({"path": path, "added_lines": added, "ranges": ranges})

    revisions: list[dict[str, Any]] = []
    for index, revision_raw in enumerate(raw_revisions):
        if not isinstance(revision_raw, dict):
            raise ValueError(f"revision {index} is not an object")
        from_commit = revision_raw.get("from_commit")
        from_path = revision_raw.get("from_path")
        start = revision_raw.get("from_start")
        end = revision_raw.get("from_end")
        from_session = revision_raw.get("from_session_id")
        to_session = revision_raw.get("to_session_id")
        if (
            not isinstance(from_commit, str)
            or not _OID_RE.fullmatch(from_commit)
            or not isinstance(from_path, str)
            or not from_path
            or "\x00" in from_path
            or not isinstance(start, int)
            or isinstance(start, bool)
            or not isinstance(end, int)
            or isinstance(end, bool)
            or start < 1
            or end < start
            or end - start + 1 > _MAX_NOTE_LINES
            or (from_session is not None and not isinstance(from_session, str))
            or (to_session is not None and not isinstance(to_session, str))
        ):
            raise ValueError(f"revision {index} is invalid")
        revisions.append(
            {
                "from_commit": from_commit.lower(),
                "from_path": from_path,
                "from_start": start,
                "from_end": end,
                "from_session_id": from_session,
                "to_session_id": to_session,
            }
        )

    participants = raw.get("contributing_session_ids")
    if participants is not None and (
        not isinstance(participants, list)
        or any(not isinstance(item, str) or not item for item in participants)
    ):
        raise ValueError("contributing_session_ids must be an array of session IDs")
    session_revisions = raw.get("session_revisions", [])
    if not isinstance(session_revisions, list):
        raise ValueError("session_revisions must be an array")
    for revision in session_revisions:
        if (
            not isinstance(revision, dict)
            or not isinstance(revision.get("path"), str)
            or not revision["path"]
            or "\x00" in revision["path"]
            or any(not isinstance(revision.get(field), str) or not revision[field]
                   for field in ("from_session_id", "to_session_id"))
            or type(revision.get("removed_lines")) is not int
            or not 0 < revision["removed_lines"] <= _MAX_NOTE_LINES
            or not isinstance(revision.get("kind"), str)
            or revision["kind"] not in {"replace", "delete"}
        ):
            raise ValueError("invalid session revision")

    workflow = None
    if raw.get("workflow") is not None:
        try:
            workflow = parse_profile(raw["workflow"])
        except ValueError as exc:
            # The profile explains the work around the proof; it is not the
            # proof. A note whose profile is unreadable still reports its lines.
            if warnings is not None:
                warnings.add(
                    f"Ignored the workflow profile in note "
                    f"{annotated_commit[:7]}: {exc}."
                )

    return {
        "commit": annotated_commit,
        "recorded_at": raw["recorded_at"],
        "sessions": raw_sessions,
        "workflow": workflow,
        "files": files,
        "revisions": revisions,
        "contributing_session_ids": participants,
        "session_revisions": session_revisions,
    }


def _load_notes(repo: Path, warnings: _Warnings) -> list[dict[str, Any]]:
    listed = _run_git(
        repo, "notes", f"--ref={_NOTES_REF}", "list", check=False
    )
    if listed.returncode != 0:
        stderr = listed.stderr.lower()
        if "cannot read notes" not in stderr and "bad ref" not in stderr:
            warnings.add("Could not enumerate attribution notes; recorded commits are unavailable.")
        return []

    notes: list[dict[str, Any]] = []
    for line in listed.stdout.splitlines():
        pieces = line.split()
        if len(pieces) != 2 or not all(_OID_RE.fullmatch(piece) for piece in pieces):
            warnings.add("Ignored a malformed entry in the attribution notes ref.")
            continue
        note_oid, annotated = (piece.lower() for piece in pieces)
        commit_check = _run_git(
            repo, "cat-file", "-e", f"{annotated}^{{commit}}", check=False
        )
        if commit_check.returncode != 0:
            warnings.add(f"Ignored attribution note for missing commit {annotated[:7]}.")
            continue
        size_result = _run_git(repo, "cat-file", "-s", note_oid, check=False)
        try:
            note_size = int(size_result.stdout.strip())
        except (TypeError, ValueError):
            note_size = -1
        if size_result.returncode != 0 or note_size < 0:
            warnings.add(f"Ignored unreadable attribution note for {annotated[:7]}.")
            continue
        if note_size > _MAX_NOTE_BYTES:
            warnings.add(f"Ignored oversized attribution note for {annotated[:7]}.")
            continue
        shown = _run_git(repo, "cat-file", "blob", note_oid, check=False)
        if shown.returncode != 0:
            warnings.add(f"Ignored unreadable attribution note for {annotated[:7]}.")
            continue
        try:
            notes.append(_parse_note(shown.stdout, annotated, warnings=warnings))
        except ValueError as exc:
            warnings.add(f"Ignored attribution note for {annotated[:7]}: {exc}.")
    return notes


def _load_task_notes(repo: Path, warnings: _Warnings) -> list[dict[str, Any]]:
    listed = _run_git(
        repo, "notes", f"--ref={_TASK_NOTES_REF}", "list", check=False
    )
    if listed.returncode != 0:
        stderr = listed.stderr.lower()
        if "cannot read notes" not in stderr and "bad ref" not in stderr:
            warnings.add("Could not enumerate task notes; shared task economics are unavailable.")
        return []

    entries: list[dict[str, Any]] = []
    for line in listed.stdout.splitlines():
        pieces = line.split()
        if len(pieces) != 2 or not all(_OID_RE.fullmatch(piece) for piece in pieces):
            warnings.add("Ignored a malformed entry in the task notes ref.")
            continue
        note_oid, annotated = (piece.lower() for piece in pieces)
        commit_check = _run_git(
            repo, "cat-file", "-e", f"{annotated}^{{commit}}", check=False
        )
        if commit_check.returncode != 0:
            warnings.add(f"Ignored task note for missing commit {annotated[:7]}.")
            continue
        size_result = _run_git(repo, "cat-file", "-s", note_oid, check=False)
        try:
            note_size = int(size_result.stdout.strip())
        except (TypeError, ValueError):
            note_size = -1
        if size_result.returncode != 0 or note_size < 0 or note_size > _MAX_NOTE_BYTES:
            warnings.add(f"Ignored unreadable task note for {annotated[:7]}.")
            continue
        shown = _run_git(repo, "cat-file", "blob", note_oid, check=False)
        if shown.returncode != 0:
            warnings.add(f"Ignored unreadable task note for {annotated[:7]}.")
            continue
        try:
            payload = json.loads(shown.stdout)
        except json.JSONDecodeError:
            warnings.add(f"Ignored invalid task note for {annotated[:7]}.")
            continue
        if (
            not isinstance(payload, dict)
            or payload.get("version") != 1
            or payload.get("commit") != annotated
            or not isinstance(payload.get("tasks"), list)
            or _parse_time(payload.get("recorded_at")) is None
        ):
            warnings.add(f"Ignored unsupported task note for {annotated[:7]}.")
            continue
        for raw_task in payload["tasks"]:
            if not isinstance(raw_task, dict):
                warnings.add(f"Ignored a malformed task in note {annotated[:7]}.")
                continue
            task = _normalise_task(
                raw_task, source=f"task note {annotated[:7]}", warnings=warnings
            )
            if task is None:
                continue
            raw_sessions = raw_task.get("sessions")
            if not isinstance(raw_sessions, list):
                warnings.add(f"Task {task['id']!r} has malformed sessions.")
                continue
            entries.append(
                {
                    "task": task,
                    "sessions": raw_sessions,
                    "recorded_at": payload["recorded_at"],
                    "commit": annotated,
                }
            )
    return entries


def _model_unknown(record: Mapping[str, Any]) -> bool:
    """Say whether a session record names no model yet."""

    return record.get("model") == "unknown" or record.get("model_source") == "unknown"


def _merge_session(
    sessions: dict[str, dict[str, Any]],
    candidate: dict[str, Any],
    warnings: _Warnings,
    *,
    immutable_existing: bool = False,
    local: bool = False,
) -> None:
    existing = sessions.get(candidate["id"])
    if existing is None:
        sessions[candidate["id"]] = candidate
        return
    # A record that names no model has no model identity to conflict with. A
    # note written while a subagent still ran carries ``unknown`` until the
    # ledger learns the model, so the record that names one labels the
    # session, whichever of the two loaded first, and neither is a conflict.
    if _model_unknown(existing) and not _model_unknown(candidate):
        existing["model"] = candidate["model"]
        existing["model_source"] = candidate.get("model_source")
    elif _model_unknown(candidate) and not _model_unknown(existing):
        candidate = {
            **candidate,
            "model": existing["model"],
            "model_source": existing.get("model_source"),
        }
    identity_fields = (
        "feature",
        "model",
        "harness",
        "actor_kind",
        "source_session_id",
        "label_source",
    )
    if any(existing[field] != candidate[field] for field in identity_fields):
        if immutable_existing:
            warnings.add(
                f"Session {candidate['id']!r} has later reported metadata that "
                "conflicts with its immutable note. The note value was used."
            )
        else:
            warnings.add(
                f"Session {candidate['id']!r} has conflicting reported metadata; "
                "the first record was used."
            )
    if immutable_existing:
        if candidate["started_at"] != existing["started_at"]:
            warnings.add(
                f"Session {candidate['id']!r} has later started_at metadata that "
                "conflicts with its immutable note. The note value was used."
            )
        if (
            candidate.get("ended_at") is not None
            and candidate.get("ended_at") != existing.get("ended_at")
        ):
            warnings.add(
                f"Session {candidate['id']!r} has later ended_at metadata that "
                "conflicts with its immutable note. The note value was used."
            )
    else:
        existing["started_at"] = min(existing["started_at"], candidate["started_at"])
        if candidate.get("ended_at") is not None:
            existing["ended_at"] = max(
                value
                for value in (existing.get("ended_at"), candidate["ended_at"])
                if value is not None
            )

    for field in ("cost_usd", "cost_source"):
        if immutable_existing:
            if candidate.get(field) is not None and candidate.get(field) != existing.get(field):
                warnings.add(
                    f"Session {candidate['id']!r} has later {field} metadata that "
                    "conflicts with its immutable note. The note value was used."
                )
            continue
        if existing.get(field) is None and candidate.get(field) is not None:
            existing[field] = candidate[field]
        elif (
            existing.get(field) is not None
            and candidate.get(field) is not None
            and existing[field] != candidate[field]
        ):
            warnings.add(
                f"Session {candidate['id']!r} has conflicting {field}; "
                "the first value was used."
            )

    for field in _IDENTITY_SESSION_TEXT_FIELDS:
        if immutable_existing:
            if candidate.get(field) is not None and candidate.get(field) != existing.get(field):
                warnings.add(
                    f"Session {candidate['id']!r} has later {field} metadata that "
                    "conflicts with its immutable note. The note value was used."
                )
            continue
        if existing.get(field) is None and candidate.get(field) is not None:
            existing[field] = candidate[field]
        elif (
            existing.get(field) is not None
            and candidate.get(field) is not None
            and existing[field] != candidate[field]
        ):
            warnings.add(
                f"Session {candidate['id']!r} has conflicting {field}; "
                "the first value was used."
            )
    # These hold the most recent value the harness reported, so the live ledger
    # wins over the value a note recorded earlier and neither is a conflict.
    for field in _RECENT_SESSION_TEXT_FIELDS:
        if candidate.get(field) is None:
            continue
        if local or existing.get(field) is None:
            existing[field] = candidate[field]

    for field in (
        "worktree_id",
        "feature_source",
        "task_id",
        "membership_source",
        "role",
        "summary",
        "parent_session_id",
        "token_count",
        "token_source",
        "exit_code",
        "outcome",
    ):
        if existing.get(field) is None and candidate.get(field) is not None:
            existing[field] = candidate[field]
        elif (
            existing.get(field) is not None
            and candidate.get(field) is not None
            and existing[field] != candidate[field]
        ):
            warnings.add(
                f"Session {candidate['id']!r} has conflicting {field}; "
                "the first value was used."
            )
    # Counts and durations are observations of one session, not identity. A
    # later read can only have seen more of the same session, so the larger
    # value wins instead of raising a conflict against an immutable note.
    for field in _OPTIONAL_SESSION_INT_FIELDS:
        observed = [
            value
            for value in (existing.get(field), candidate.get(field))
            if value is not None
        ]
        existing[field] = max(observed) if observed else None
    if candidate.get("activity_truncated"):
        existing["activity_truncated"] = True
    if candidate.get("usage_includes_children"):
        existing["usage_includes_children"] = True


def _merge_task(
    tasks: dict[str, dict[str, Any]], candidate: dict[str, Any], warnings: _Warnings
) -> None:
    existing = tasks.get(candidate["id"])
    if existing is None:
        tasks[candidate["id"]] = candidate
        return
    existing_time = _parse_time(existing.get("updated_at"))
    candidate_time = _parse_time(candidate.get("updated_at"))
    if candidate_time is not None and (existing_time is None or candidate_time > existing_time):
        tasks[candidate["id"]] = candidate
        return
    if existing.get("name") != candidate.get("name"):
        warnings.add(
            f"Task {candidate['id']!r} has conflicting metadata; the latest record was used."
        )


def _target_history(repo: Path, target: str, warnings: _Warnings) -> list[str]:
    result = _run_git(repo, "rev-list", target)
    return [line.lower() for line in result.stdout.splitlines() if _OID_RE.fullmatch(line)]


def _commit_info(
    repo: Path,
    commit: str,
    cache: dict[str, dict[str, Any] | None],
    warnings: _Warnings,
) -> dict[str, Any] | None:
    commit = commit.lower()
    if commit in cache:
        return cache[commit]
    result = _run_git(
        repo, "show", "-s", "--format=%H%x00%s%x00%cI", commit, check=False
    )
    if result.returncode != 0:
        warnings.add(f"Could not read commit metadata for {commit[:7]}.")
        cache[commit] = None
        return None
    pieces = result.stdout.rstrip("\n").split("\x00", 2)
    committed = _parse_time(pieces[2]) if len(pieces) == 3 else None
    if len(pieces) != 3 or committed is None:
        warnings.add(f"Commit {commit[:7]} has malformed metadata.")
        cache[commit] = None
        return None
    value = {
        "sha": pieces[0].lower(),
        "short_sha": pieces[0][:7].lower(),
        "subject": pieces[1],
        "committed_at": _format_time(committed),
        "_datetime": committed,
    }
    cache[commit] = value
    return value


def _decode_git_path(value: str) -> str:
    if value.startswith('"') and value.endswith('"'):
        try:
            decoded = ast.literal_eval(value)
            if isinstance(decoded, str):
                return decoded
        except (SyntaxError, ValueError):
            pass
    return value


def _parse_blame(output: str) -> set[_Origin]:
    result: set[_Origin] = set()
    lines = output.splitlines()
    index = 0
    while index < len(lines):
        header = _BLAME_HEADER_RE.fullmatch(lines[index])
        if header is None:
            raise ValueError("unexpected blame output")
        commit = header.group(1)
        original_line = int(header.group(2))
        index += 1
        filename: str | None = None
        while index < len(lines) and not lines[index].startswith("\t"):
            if lines[index].startswith("filename "):
                filename = _decode_git_path(lines[index][9:])
            index += 1
        if index >= len(lines) or filename is None:
            raise ValueError("incomplete blame output")
        result.add(_Origin(commit, filename, original_line))
        index += 1
    return result


def _target_blame(
    repo: Path,
    target: str,
    warnings: _Warnings,
) -> tuple[set[_Origin], bool]:
    tree = _run_git(repo, "ls-tree", "-r", "-l", "-z", target, text=False)
    entries: list[tuple[str, str, int, str]] = []
    for raw_entry in tree.stdout.split(b"\x00"):
        if not raw_entry:
            continue
        try:
            metadata, raw_path = raw_entry.split(b"\t", 1)
            mode, object_type, object_id, raw_size = metadata.decode("ascii").split()
            size = int(raw_size)
            path = raw_path.decode("utf-8", errors="surrogateescape")
        except (ValueError, UnicodeError):
            warnings.add("Skipped an unreadable entry in the target tree.")
            return set(), False
        if object_type == "blob" and mode in {"100644", "100755"}:
            entries.append((path, object_id, size, mode))

    complete = True
    if len(entries) > _MAX_TARGET_FILES:
        warnings.add(
            f"Target has more than {_MAX_TARGET_FILES:,} text candidates; "
            "retention metrics are unavailable."
        )
        entries = entries[:_MAX_TARGET_FILES]
        complete = False

    origins: set[_Origin] = set()
    inspected_bytes = 0
    for path, object_id, size, _mode in entries:
        display_path = path.encode("utf-8", errors="replace").decode("utf-8")
        if size > _MAX_TARGET_FILE_BYTES:
            warnings.add(
                f"Skipped large target file {display_path!r}; retention metrics are unavailable."
            )
            complete = False
            continue
        if inspected_bytes + size > _MAX_TARGET_TOTAL_BYTES:
            warnings.add(
                "Target text exceeds the reporting inspection limit; "
                "retention metrics are unavailable."
            )
            complete = False
            break
        blob = _run_git(repo, "cat-file", "blob", object_id, check=False, text=False)
        if blob.returncode != 0:
            warnings.add(
                f"Could not inspect target file {display_path!r}; retention metrics are unavailable."
            )
            complete = False
            continue
        inspected_bytes += size
        if b"\x00" in blob.stdout:
            warnings.add(f"Skipped binary target file {display_path!r}.")
            continue
        blamed = _run_git(
            repo,
            "blame",
            "--root",
            "--line-porcelain",
            "-M",
            "-C",
            "-C",
            target,
            "--",
            path,
            check=False,
        )
        if blamed.returncode != 0:
            warnings.add(
                f"Could not inspect target file {display_path!r}; retention metrics are unavailable."
            )
            complete = False
            continue
        try:
            origins.update(_parse_blame(blamed.stdout))
        except ValueError:
            warnings.add(
                f"Could not parse history for target file {display_path!r}; "
                "retention metrics are unavailable."
            )
            complete = False
    return origins, complete


def _coalesce_ranges(line_owners: dict[int, str]) -> list[dict[str, Any]]:
    ranges: list[dict[str, Any]] = []
    for line, session_id in sorted(line_owners.items()):
        if (
            ranges
            and ranges[-1]["session_id"] == session_id
            and ranges[-1]["end"] + 1 == line
        ):
            ranges[-1]["end"] = line
        else:
            ranges.append({"start": line, "end": line, "session_id": session_id})
    return ranges


def _added_lines_for_commit(repo: Path, commit: str, warnings: _Warnings) -> int:
    parent_result = _run_git(
        repo, "rev-list", "--parents", "-n", "1", commit, check=False
    )
    fields = parent_result.stdout.split()
    if parent_result.returncode != 0 or not fields:
        warnings.add(f"Could not inspect the parent of commit {commit[:7]}.")
        return 0
    if len(fields) == 1:
        result = _run_git(
            repo,
            "diff-tree",
            "--root",
            "--no-commit-id",
            "-r",
            "--numstat",
            "--find-renames",
            commit,
            "--",
            check=False,
        )
    else:
        # Joyride notes compare merge commits with their first parent. Use
        # the same baseline so a conflict resolution cannot disappear here.
        result = _run_git(
            repo,
            "diff",
            "--numstat",
            "--find-renames",
            fields[1],
            commit,
            "--",
            check=False,
        )
    if result.returncode != 0:
        warnings.add(f"Could not count added lines for commit {commit[:7]}.")
        return 0
    added = 0
    for line in result.stdout.splitlines():
        fields = line.split("\t", 2)
        if len(fields) < 3:
            continue
        if fields[0] == "-":
            warnings.add(f"Binary changes in commit {commit[:7]} have no line count.")
            continue
        try:
            added += int(fields[0])
        except ValueError:
            warnings.add(f"Could not parse an added-line count in commit {commit[:7]}.")
    return added


def _merge_additions(repo: Path, commit: str, parents: int) -> set[_Origin]:
    """Find lines new to every merge parent, not additions inherited from a branch."""
    patch = _run_git(
        repo, "show", "--format=", "--diff-merges=combined", "--unified=0",
        "--no-ext-diff", "--no-textconv", "--no-color", "--no-renames", "--src-prefix=a/",
        "--dst-prefix=b/", commit, "--",
    ).stdout
    origins: set[_Origin] = set()
    path: str | None = None
    result_line: int | None = None
    remaining = 0
    for line in patch.split("\n"):
        if line.startswith("diff --combined "):
            path, result_line, remaining = None, None, 0
        elif result_line is None and line.startswith("+++ "):
            raw_path = _decode_git_path(line[4:])
            path = raw_path[2:] if raw_path.startswith("b/") else None
        elif line.startswith("@" * (parents + 1) + " "):
            match = re.search(r" \+(\d+)(?:,(\d+))? @", line)
            if match is None:
                raise ValueError(f"Could not read merge additions at {commit[:7]}.")
            result_line = int(match.group(1))
            remaining = int(match.group(2)) if match.group(2) is not None else 1
        elif result_line is not None and remaining > 0:
            prefix = line[:parents]
            if len(prefix) != parents or any(char not in " +-" for char in prefix):
                continue  # For example, Git's no-newline marker.
            if "-" not in prefix:
                if prefix == "+" * parents and path is not None:
                    origins.add(_Origin(commit, path, result_line))
                result_line += 1
                remaining -= 1
    return origins


def _median(values: Iterable[float]) -> float | None:
    materialised = list(values)
    if not materialised:
        return None
    return round(float(statistics.median(materialised)), 1)


def _group_id(session: dict[str, Any], tasks: dict[str, dict[str, Any]]) -> str:
    task_id = session.get("task_id")
    visited: set[str] = set()
    while isinstance(task_id, str) and task_id and task_id not in visited:
        visited.add(task_id)
        task = tasks.get(task_id)
        if task is None:
            return str(session["feature"])
        if not task.get("merged_into"):
            return task_id
        task_id = task["merged_into"]
    return str(session["feature"])


def _task_economics(sessions: list[dict[str, Any]]) -> dict[str, Any]:
    """Sum reported and provider usage once, respecting inclusive parents."""

    by_id = {session["id"]: session for session in sessions}

    def cost_values(session: dict[str, Any]) -> dict[str, float | None]:
        reported = session.get("cost_usd")
        telemetry = session.get("telemetry")
        usage = telemetry if isinstance(telemetry, dict) else {}
        if reported is not None:
            return {
                "reported": float(reported),
                "estimated": None,
                "credits": None,
                "equivalent": None,
            }
        return {
            "reported": None,
            "estimated": usage.get("estimated_cost_usd"),
            "credits": usage.get("codex_credits"),
            "equivalent": usage.get("codex_api_equivalent_usd"),
        }

    def has_cost(session: dict[str, Any]) -> bool:
        return any(value is not None for value in cost_values(session).values())

    def covering_ancestor(
        session: dict[str, Any], predicate: Any
    ) -> str | None:
        parent_id = session.get("parent_session_id")
        visited: set[str] = set()
        while isinstance(parent_id, str) and parent_id and parent_id not in visited:
            visited.add(parent_id)
            parent = by_id.get(parent_id)
            if parent is None:
                return None
            if parent.get("usage_includes_children") and predicate(parent):
                return parent_id
            parent_id = parent.get("parent_session_id")
        return None

    reported_costs: list[float] = []
    estimated_costs: list[float] = []
    credit_costs: list[float] = []
    equivalent_costs: list[float] = []
    counted_tokens: list[int] = []
    missing_cost = 0
    missing_tokens = 0
    cost_covered_sessions = 0
    ai_session_count = 0
    for session in sessions:
        values = cost_values(session)
        # A manual session has no usage by definition, so it can be missing
        # none. Only an AI session without usage leaves a total incomplete.
        manual = session.get("actor_kind") == "manual"
        if not manual:
            ai_session_count += 1
        cost_ancestor = covering_ancestor(session, has_cost)
        token_ancestor = covering_ancestor(
            session, lambda item: item.get("token_count") is not None
        )
        session["cost_covered_by_parent"] = cost_ancestor is not None
        session["tokens_covered_by_parent"] = token_ancestor is not None
        session["cost_in_total"] = cost_ancestor is None and has_cost(session)
        session["tokens_in_total"] = (
            token_ancestor is None and session.get("token_count") is not None
        )
        if session["cost_in_total"]:
            destinations = (
                ("reported", reported_costs),
                ("estimated", estimated_costs),
                ("credits", credit_costs),
                ("equivalent", equivalent_costs),
            )
            for field, destination in destinations:
                value = values[field]
                if value is not None:
                    destination.append(float(value))
            telemetry = session.get("telemetry")
            complete = (
                session.get("cost_usd") is not None
                or not isinstance(telemetry, dict)
                or telemetry.get("cost_complete") is not False
            )
            if complete:
                cost_covered_sessions += 1
            else:
                missing_cost += 1
        elif cost_ancestor is None and not manual:
            missing_cost += 1
        if session["tokens_in_total"]:
            counted_tokens.append(int(session["token_count"]))
        elif token_ancestor is None and not manual:
            missing_tokens += 1

    return {
        "reported_cost_usd": (
            round(math.fsum(reported_costs), 10) if reported_costs else None
        ),
        "estimated_cost_usd": (
            round(math.fsum(estimated_costs), 10) if estimated_costs else None
        ),
        "codex_credits": (
            round(math.fsum(credit_costs), 10) if credit_costs else None
        ),
        "codex_api_equivalent_usd": (
            round(math.fsum(equivalent_costs), 10) if equivalent_costs else None
        ),
        "cost_covered_sessions": cost_covered_sessions,
        "cost_complete": ai_session_count > 0 and missing_cost == 0,
        "missing_cost_session_count": missing_cost,
        "total_tokens": sum(counted_tokens) if counted_tokens else None,
        "tokens_complete": ai_session_count > 0 and missing_tokens == 0,
        "missing_token_session_count": missing_tokens,
    }


def _read_demo_marker(common_dir: Path, warnings: _Warnings) -> bool:
    marker = common_dir / "attribution" / "demo.json"
    if not marker.is_file():
        return False
    try:
        if marker.stat().st_size > 4096:
            raise ValueError("marker is too large")
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        warnings.add(f"Ignored invalid example-data marker: {exc}.")
        return False
    return isinstance(payload, dict) and payload.get("example_data") is True


def _before_compaction(
    rows: list[dict[str, Any]], compactions: int
) -> list[dict[str, Any]]:
    """Mark every row that a later compaction has since summarized away.

    A row is before a compaction when it was written in an earlier epoch than
    the one the session ended in. What it names entered a context window that
    no longer exists.
    """

    return [
        {**row, "before_compaction": int(row["compaction_epoch"] or 0) < compactions}
        for row in rows
    ]


def _session_activity(
    tool_calls: dict[str, dict[str, Any]],
    context_loads: dict[str, dict[str, Any]],
    session: dict[str, Any],
) -> dict[str, Any]:
    """Return one session's recorded activity and its counts of every kind."""

    session_id = str(session["id"])
    compactions = session.get("compaction_count") or 0
    recorded = tool_calls.get(session_id) or {
        "tool_calls": [],
        "tool_calls_truncated": False,
        "tool_call_counts": dict.fromkeys(TOOL_CLASSES, 0),
    }
    loaded = context_loads.get(session_id) or {
        "context_loads": [],
        "context_loads_truncated": False,
        "context_load_counts": dict.fromkeys(CONTEXT_LOAD_KINDS, 0),
    }
    return {
        "tool_calls": _before_compaction(recorded["tool_calls"], compactions),
        "tool_calls_truncated": recorded["tool_calls_truncated"],
        "tool_call_counts": recorded["tool_call_counts"],
        "context_loads": _before_compaction(loaded["context_loads"], compactions),
        "context_loads_truncated": loaded["context_loads_truncated"],
        "context_load_counts": loaded["context_load_counts"],
    }


def _note_workflow(
    notes: list[dict[str, Any]]
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, list[dict[str, Any]]]]]:
    """Return what the workflow object of each note records about its sessions.

    A clone without the local ledger has only these objects. The first result
    holds each session's counts and tool calls; the second holds the
    instruction files of every note that names a session, keyed by the commit
    the note annotates, because a profile counts the agents that loaded a file
    without naming them.
    """

    activity: dict[str, dict[str, Any]] = {}
    files: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for note in notes:
        profile = note.get("workflow")
        if profile is None:
            continue
        recorded = profile_activity(profile)
        for agent in iter_agents(profile):
            session_id = agent["session_id"]
            entry = recorded.get(session_id)
            if entry is not None:
                activity[session_id] = merge_activity(activity.get(session_id), entry)
            if profile["instruction_files"]:
                files.setdefault(session_id, {})[note["commit"]] = profile[
                    "instruction_files"
                ]
    return activity, files


def _workflow_profile(
    sessions: list[dict[str, Any]],
    tool_calls: dict[str, dict[str, Any]],
    context_loads: dict[str, dict[str, Any]],
    instruction_files: dict[str, list[dict[str, Any]]],
    recorded: dict[str, dict[str, Any]],
    recorded_files: dict[str, dict[str, list[dict[str, Any]]]],
) -> dict[str, Any] | None:
    """Return one task's workflow profile from the ledger and from its notes."""

    activity: dict[str, dict[str, Any]] = {}
    files: list[dict[str, Any]] = []
    from_notes: set[str] = set()
    edited: set[str] = set()
    for session in sessions:
        session_id = str(session["id"])
        if (session.get("attributed_lines") or 0) > 0:
            edited.add(session_id)
        entry: dict[str, Any] = {}
        local = tool_calls.get(session_id)
        if local is not None:
            entry["tool_calls"] = local["tool_call_counts"]
        loaded = context_loads.get(session_id)
        if loaded is not None:
            entry["instruction_files"] = instruction_files.get(session_id, [])
        else:
            # This clone never ran the session, so every note that named it
            # contributes the files it saw, once per note.
            for commit, listed in (recorded_files.get(session_id) or {}).items():
                if commit not in from_notes:
                    from_notes.add(commit)
                    files.extend(listed)
        # The local ledger is the live record of a session. A note describes
        # the same session as it stood when its commit was recorded, so it
        # fills in only what this ledger does not hold.
        for field, value in (recorded.get(session_id) or {}).items():
            entry.setdefault(field, value)
        if entry:
            activity[session_id] = entry
    return build_profile(
        sessions, activity=activity, instruction_files=files, edited=edited
    )


def build_dashboard(repo: str | Path, target_ref: str = "main") -> dict[str, Any]:
    """Return one JSON-compatible, evidence-backed report payload.

    This function is intentionally read-only. It does not create a ledger, write a
    note, refresh a cache, or modify the repository.
    """

    warnings = _Warnings()
    root, common_dir, worktree_git_dir = _repository_paths(repo)
    target_ref = str(target_ref)
    target = _resolve_target(root, target_ref)
    if target is None:
        warnings.add(f"Target ref {target_ref!r} does not exist.")

    generated = _utc_now()
    notes = _load_notes(root, warnings)
    worktrees = _live_worktrees(root, common_dir, worktree_git_dir, warnings)
    worktrees_by_id = {item["id"]: item for item in worktrees}
    loaded_task_notes = _load_task_notes(root, warnings)
    latest_task_notes: dict[str, dict[str, Any]] = {}
    for entry in loaded_task_notes:
        task_id = entry["task"]["id"]
        previous = latest_task_notes.get(task_id)
        entry_time = _parse_time(entry["recorded_at"])
        previous_time = _parse_time(previous["recorded_at"]) if previous else None
        if previous is None or (
            entry_time is not None
            and (previous_time is None or entry_time > previous_time)
        ):
            latest_task_notes[task_id] = entry
    task_notes = list(latest_task_notes.values())
    tasks: dict[str, dict[str, Any]] = {}
    for task in _load_local_tasks(common_dir, warnings):
        _merge_task(tasks, task, warnings)
    sessions: dict[str, dict[str, Any]] = {}
    for note in notes:
        for raw_session in note["sessions"]:
            if not isinstance(raw_session, dict):
                warnings.add(f"Ignored a malformed session in note {note['commit'][:7]}.")
                continue
            session = _normalise_session(
                raw_session,
                source=f"note {note['commit'][:7]}",
                warnings=warnings,
            )
            if session is not None:
                _merge_session(sessions, session, warnings)
    model_switches = _load_local_model_switches(common_dir, warnings)
    tool_calls = _load_local_tool_calls(common_dir, warnings)
    context_loads = _load_local_context_loads(common_dir, warnings)
    instruction_files = _load_local_instruction_files(common_dir, warnings)
    note_activity, note_instruction_files = _note_workflow(notes)
    immutable_session_ids = set(sessions)
    for session in _load_local_sessions(common_dir, warnings):
        _merge_session(
            sessions,
            session,
            warnings,
            immutable_existing=session["id"] in immutable_session_ids,
            local=True,
        )
    for entry in task_notes:
        _merge_task(tasks, entry["task"], warnings)
        for raw_session in entry["sessions"]:
            if not isinstance(raw_session, dict):
                warnings.add(
                    f"Ignored a malformed session in task note {entry['commit'][:7]}."
                )
                continue
            session = _normalise_session(
                raw_session,
                source=f"task note {entry['commit'][:7]}",
                warnings=warnings,
            )
            if session is not None:
                _merge_session(
                    sessions,
                    session,
                    warnings,
                    immutable_existing=session["id"] in immutable_session_ids,
                )

    # A provider PR reference is a stronger task identity than a worktree or
    # branch. Join independently inferred tasks when they later resolve to the
    # same PR, without rewriting the read-only report inputs.
    tasks_by_pr: dict[str, list[dict[str, Any]]] = {}
    for task in tasks.values():
        pr_identity = task.get("pr_url") or task.get("pr_ref")
        if pr_identity and not task.get("merged_into"):
            tasks_by_pr.setdefault(str(pr_identity), []).append(task)
    for candidates in tasks_by_pr.values():
        if len(candidates) < 2:
            continue
        candidates.sort(
            key=lambda item: (
                item.get("inference_source") != "explicit",
                item.get("created_at") or "",
                item["id"],
            )
        )
        canonical = candidates[0]
        for duplicate in candidates[1:]:
            duplicate["merged_into"] = canonical["id"]

    try:
        from .costing import allocate_session_usage

        telemetry_usage = allocate_session_usage(common_dir, sessions.values())
    except (OSError, ValueError, sqlite3.Error) as exc:
        telemetry_usage = {}
        warnings.add(f"Could not read local cost telemetry: {exc}.")

    target_history = _target_history(root, target, warnings) if target else []
    target_set = set(target_history)
    commit_cache: dict[str, dict[str, Any] | None] = {}
    contribution_commits: dict[str, list[dict[str, Any]]] = {}
    note_totals: dict[str, tuple[int, int]] = {}
    session_origins: dict[str, set[_Origin]] = {session_id: set() for session_id in sessions}
    origin_sessions: dict[_Origin, str] = {}

    for note in notes:
        commit = note["commit"]
        per_path_owners: dict[str, dict[int, str | None]] = {}
        total_added = 0
        for file_entry in note["files"]:
            path = file_entry["path"]
            total_added += file_entry["added_lines"]
            owners = per_path_owners.setdefault(path, {})
            for item in file_entry["ranges"]:
                session_id = item["session_id"]
                if session_id not in sessions:
                    warnings.add(
                        f"Note {commit[:7]} references unknown session {session_id!r}; "
                        "those lines remain unattributed."
                    )
                    continue
                for line in range(item["start"], item["end"] + 1):
                    if line not in owners:
                        owners[line] = session_id
                    elif owners[line] != session_id:
                        owners[line] = None
                        warnings.add(
                            f"Note {commit[:7]} assigns one line to multiple sessions; "
                            "that line remains unattributed."
                        )

        credited = sum(
            1 for owners in per_path_owners.values() for owner in owners.values() if owner
        )
        if credited > total_added:
            warnings.add(
                f"Note {commit[:7]} attributes more lines than its recorded additions."
            )
        note_totals[commit] = (total_added, min(credited, total_added))

        by_feature: dict[str, dict[str, dict[int, str]]] = {}
        for path, owners in per_path_owners.items():
            for line, session_id in owners.items():
                if session_id is None:
                    continue
                session = sessions[session_id]
                feature = _group_id(session, tasks)
                by_feature.setdefault(feature, {}).setdefault(path, {})[line] = session_id
                origin = _Origin(commit, path, line)
                session_origins.setdefault(session_id, set()).add(origin)
                previous = origin_sessions.get(origin)
                if previous is None:
                    origin_sessions[origin] = session_id
                elif previous != session_id:
                    warnings.add(
                        f"Conflicting ownership was recorded for {path}:{line} at {commit[:7]}."
                    )

        commit_meta = _commit_info(root, commit, commit_cache, warnings)
        if commit_meta is None:
            continue
        unknown_added = max(total_added - min(credited, total_added), 0)
        for feature, paths in by_feature.items():
            files = [
                {"path": path, "ranges": _coalesce_ranges(owners)}
                for path, owners in sorted(paths.items())
            ]
            attributed = sum(
                item["end"] - item["start"] + 1
                for file_entry in files
                for item in file_entry["ranges"]
            )
            contribution_commits.setdefault(feature, []).append(
                {
                    "sha": commit_meta["sha"],
                    "short_sha": commit_meta["short_sha"],
                    "subject": commit_meta["subject"],
                    "committed_at": commit_meta["committed_at"],
                    "landed": commit in target_set if target is not None else None,
                    "attributed_lines": attributed,
                    "retained_lines": 0,
                    "unknown_added_lines": unknown_added,
                    "files": files,
                }
            )

    landed_origins = {
        origin
        for origins in session_origins.values()
        for origin in origins
        if origin.commit in target_set
    }
    if target and landed_origins:
        current_origins, blame_complete = _target_blame(root, target, warnings)
    else:
        current_origins, blame_complete = set(), True
    retained_origins = landed_origins & current_origins

    valid_notes = {note["commit"] for note in notes}
    merge_parents: dict[str, int] = {}
    if target:
        for line in _run_git(root, "rev-list", "--min-parents=2", "--parents", target).stdout.splitlines():
            fields = line.split()
            merge_parents[fields[0]] = len(fields) - 1
    target_unknown: dict[str, int] = {}
    for commit in target_history:
        if commit in merge_parents:
            # Branch additions were counted at their origin commits. Only new
            # merge-resolution lines belong to this commit's unknown total.
            additions = _merge_additions(root, commit, merge_parents[commit])
            target_unknown[commit] = len(additions - origin_sessions.keys())
            continue
        actual_added = _added_lines_for_commit(root, commit, warnings)
        if commit in valid_notes:
            recorded_added, credited = note_totals.get(commit, (0, 0))
            if actual_added > recorded_added:
                warnings.add(
                    f"Commit {commit[:7]} contains added lines outside its attribution note; "
                    "they remain unattributed."
                )
            known_added = max(recorded_added, actual_added)
            target_unknown[commit] = max(known_added - credited, 0)
        else:
            target_unknown[commit] = actual_added

    # Revisions are evidence that an original line was removed or replaced. Only
    # revisions reachable from the selected target can describe its current state.
    removal_evidence: dict[_Origin, datetime] = {}
    feature_revisions: dict[str, dict[tuple[Any, ...], dict[str, Any]]] = {}
    for note in notes:
        commit = note["commit"]
        if commit not in target_set:
            continue
        commit_meta = _commit_info(root, commit, commit_cache, warnings)
        if commit_meta is None:
            continue
        revision_time = commit_meta["_datetime"]
        for revision in note["revisions"]:
            source_meta = _commit_info(
                root, revision["from_commit"], commit_cache, warnings
            )
            lifetime = None
            if source_meta is not None:
                lifetime = round(
                    max(
                        0.0,
                        (revision_time - source_meta["_datetime"]).total_seconds() / 3600,
                    ),
                    1,
                )
            involved_features: set[str] = set()
            for session_id in (
                revision["from_session_id"],
                revision["to_session_id"],
            ):
                if session_id in sessions:
                    involved_features.add(_group_id(sessions[session_id], tasks))
            for line in range(revision["from_start"], revision["from_end"] + 1):
                origin = _Origin(revision["from_commit"], revision["from_path"], line)
                owner = origin_sessions.get(origin)
                if owner in sessions:
                    involved_features.add(_group_id(sessions[owner], tasks))
                previous = removal_evidence.get(origin)
                if previous is None or revision_time < previous:
                    removal_evidence[origin] = revision_time

            item = {
                "commit": commit,
                "committed_at": commit_meta["committed_at"],
                "from_commit": revision["from_commit"],
                "from_path": revision["from_path"],
                "from_start": revision["from_start"],
                "from_end": revision["from_end"],
                "from_session_id": revision["from_session_id"],
                "to_session_id": revision["to_session_id"],
                "observed_lifetime_hours": lifetime,
            }
            key = tuple(item.values())
            for feature in involved_features:
                feature_revisions.setdefault(feature, {})[key] = item

    features_to_sessions: dict[str, list[dict[str, Any]]] = {}
    for session in sessions.values():
        features_to_sessions.setdefault(_group_id(session, tasks), []).append(session)

    all_features = set(features_to_sessions) | set(contribution_commits)
    features: list[dict[str, Any]] = []
    for feature_name in all_features:
        task_meta = tasks.get(feature_name)
        display_name = task_meta["name"] if task_meta is not None else feature_name
        feature_sessions = features_to_sessions.get(feature_name, [])
        session_ids = {session["id"] for session in feature_sessions}
        feature_origins = {
            origin for session_id in session_ids for origin in session_origins.get(session_id, set())
        }
        feature_landed = {origin for origin in feature_origins if origin.commit in target_set}
        feature_retained = feature_landed & retained_origins
        retention_known = blame_complete or not feature_landed

        session_payloads: list[dict[str, Any]] = []
        for session in feature_sessions:
            origins = session_origins.get(session["id"], set())
            landed = {origin for origin in origins if origin.commit in target_set}
            retained = landed & retained_origins
            known = blame_complete or not landed
            worktree_id = session.get("worktree_id")
            worktree = worktrees_by_id.get(worktree_id)
            usage = telemetry_usage.get(session["id"])
            named = telemetry_model(session, usage)
            model = named if named is not None else session["model"]
            model_source = (
                "telemetry" if named is not None else session.get("model_source")
            )
            token_count = session["token_count"]
            token_source = session["token_source"]
            if token_count is None and isinstance(usage, Mapping):
                # The ledger holds a count only where the wrapper, an adapter,
                # or ``--tokens`` supplied one. A hook session supplies none,
                # while the spool of the same run counted every token, so the
                # count fills here exactly as the pull request report fills it. An
                # unknown count is never reported as a zero.
                counted = usage.get("total_tokens")
                if (
                    isinstance(counted, (int, float))
                    and not isinstance(counted, bool)
                    and math.isfinite(counted)
                    and counted > 0
                ):
                    token_count = int(round(counted))
                    token_source = "telemetry"
            session_payloads.append(
                {
                    "id": session["id"],
                    "worktree": (
                        {
                            "id": worktree_id,
                            "path": worktree.get("path") if worktree else None,
                            "branch": worktree.get("branch") if worktree else None,
                            "current": worktree.get("current", False) if worktree else False,
                        }
                        if worktree_id
                        else None
                    ),
                    "native_session_id": session.get("native_session_id"),
                    "feature_source": session.get("feature_source"),
                    "task_id": session.get("task_id"),
                    "model": model,
                    "harness": session["harness"],
                    "actor_kind": session["actor_kind"],
                    "source_session_id": session["source_session_id"],
                    "label_source": session["label_source"],
                    "membership_source": session["membership_source"],
                    "role": session["role"],
                    "summary": session["summary"],
                    "parent_session_id": session["parent_session_id"],
                    "token_count": token_count,
                    "token_source": token_source,
                    "cost_usd": session["cost_usd"],
                    "cost_source": session["cost_source"],
                    "telemetry": usage,
                    "usage_includes_children": session["usage_includes_children"],
                    "started_at": session["started_at"],
                    "ended_at": session["ended_at"],
                    "exit_code": session["exit_code"],
                    "outcome": session["outcome"],
                    **{
                        field: session.get(field)
                        for field in _OPTIONAL_SESSION_TEXT_FIELDS
                    },
                    # A model this report read out of telemetry says where the
                    # label came from. The ledger keeps what the harness said.
                    "model_source": model_source,
                    **{
                        field: session.get(field)
                        for field in _OPTIONAL_SESSION_INT_FIELDS
                    },
                    "activity_truncated": bool(session.get("activity_truncated")),
                    "model_switches": model_switches.get(session["id"], []),
                    **_session_activity(tool_calls, context_loads, session),
                    "attributed_lines": len(origins),
                    "landed_lines": len(landed) if target is not None else None,
                    "retained_lines": (
                        len(retained) if target is not None and known else None
                    ),
                    "removed_lines": (
                        len(landed - retained)
                        if target is not None and known
                        else None
                    ),
                }
            )
        session_payloads.sort(key=lambda item: (item["started_at"], item["id"]), reverse=True)

        commits = contribution_commits.get(feature_name, [])
        for commit_item in commits:
            if commit_item["landed"]:
                commit_item["unknown_added_lines"] = target_unknown.get(
                    commit_item["sha"], commit_item["unknown_added_lines"]
                )
            commit_origins = {
                origin
                for origin in feature_landed
                if origin.commit == commit_item["sha"]
            }
            commit_item["retained_lines"] = (
                len(commit_origins & retained_origins)
                if target is not None and (blame_complete or not commit_origins)
                else None
            )
        commits.sort(key=lambda item: (item["committed_at"], item["sha"]), reverse=True)
        revisions = list(feature_revisions.get(feature_name, {}).values())
        revisions.sort(
            key=lambda item: (item["committed_at"], item["commit"], item["from_start"]),
            reverse=True,
        )
        landed_commit_count = (
            sum(1 for item in commits if item["landed"] is True)
            if target is not None
            else None
        )
        if target is None:
            status = "unknown"
        elif revisions and not commits:
            # A deletion-only edit can be an enduring, landed contribution even
            # though it adds no line that can appear in the contribution table.
            status = "landed"
        elif not commits:
            status = "captured"
        elif landed_commit_count == 0:
            status = "unlanded"
        elif landed_commit_count == len(commits):
            status = "landed"
        else:
            status = "mixed"

        economics = _task_economics(session_payloads)
        activity_times = [
            item["ended_at"] or item["started_at"] for item in session_payloads
        ]
        if not activity_times and commits:
            activity_times = [item["committed_at"] for item in commits]
        last_activity = max(activity_times) if activity_times else generated.isoformat()

        retained_age_values: list[float] = []
        removed_lifetime_values: list[float] = []
        if retention_known:
            for origin in feature_retained:
                source = _commit_info(root, origin.commit, commit_cache, warnings)
                if source is not None:
                    retained_age_values.append(
                        max(0.0, (generated - source["_datetime"]).total_seconds() / 3600)
                    )
            for origin in feature_landed - feature_retained:
                removed_at = removal_evidence.get(origin)
                source = _commit_info(root, origin.commit, commit_cache, warnings)
                if removed_at is not None and source is not None:
                    removed_lifetime_values.append(
                        max(0.0, (removed_at - source["_datetime"]).total_seconds() / 3600)
                    )

        landed_count = len(feature_landed)
        retained_count = len(feature_retained) if retention_known else None
        removed_count = len(feature_landed - feature_retained) if retention_known else None
        retention_pct = None
        if landed_count and retained_count is not None:
            retention_pct = round(100.0 * retained_count / landed_count, 1)

        features.append(
            {
                "id": feature_name,
                "name": display_name,
                "kind": task_meta["kind"] if task_meta is not None else "feature",
                "pr_ref": task_meta["pr_ref"] if task_meta is not None else None,
                "pr_url": task_meta["pr_url"] if task_meta is not None else None,
                "branch": task_meta["branch"] if task_meta is not None else None,
                "state": task_meta["state"] if task_meta is not None else "active",
                "inference_source": (
                    task_meta["inference_source"] if task_meta is not None else "legacy"
                ),
                "status": status,
                "last_activity_at": last_activity,
                "session_count": len(session_payloads),
                "commit_count": len(commits),
                "landed_commit_count": landed_commit_count,
                **economics,
                "attributed_lines": len(feature_origins),
                "landed_lines": landed_count if target is not None else None,
                "retained_lines": retained_count if target is not None else None,
                "removed_lines": removed_count if target is not None else None,
                "retention_pct": retention_pct if target is not None else None,
                "median_retained_age_hours": (
                    _median(retained_age_values)
                    if target is not None and retention_known
                    else None
                ),
                "median_removed_lifetime_hours": (
                    _median(removed_lifetime_values)
                    if target is not None and retention_known
                    else None
                ),
                "workflow": _workflow_profile(
                    session_payloads,
                    tool_calls,
                    context_loads,
                    instruction_files,
                    note_activity,
                    note_instruction_files,
                ),
                "sessions": session_payloads,
                "commits": commits,
                "revisions": revisions,
            }
        )

    features.sort(key=lambda item: (item["last_activity_at"], item["name"]), reverse=True)

    unattributed_commits = (
        sum(1 for value in target_unknown.values() if value)
        if target is not None
        else None
    )
    unattributed_lines = sum(target_unknown.values()) if target is not None else None

    return {
        "repository": {
            "name": root.name,
            "path": str(root),
            "current_worktree_id": str(worktree_git_dir),
            "worktrees": worktrees,
            "target_ref": target_ref,
            "target_commit": target,
            "target_exists": target is not None,
            "example_data": _read_demo_marker(common_dir, warnings),
        },
        "generated_at": _format_time(generated),
        "tasks": features,
        "features": features,
        "unattributed": {
            "commit_count": unattributed_commits,
            "added_lines": unattributed_lines,
        },
        "warnings": warnings.items,
    }
