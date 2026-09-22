"""Capture an exact working-tree transition around one coding command."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import stat
import subprocess
import time
from typing import Iterator, Mapping, Sequence
import uuid

from .activity import apply_session_facets, record_model_switch
from .runtime import system_subprocess_environment
from .store import RepoPath, _run_git, git_dir, open_db, repository_root
from .tasks import ROLES, resolve_task, validated_token_count


MAX_TEXT_BYTES = 1024 * 1024
_ACTIVE_CAPTURE_STATES = ("pending", "contaminated", "limited", "imported")
_MANUAL_CAPTURE_HARNESS = "manual"
METADATA_LOCK_WAIT_SECONDS = 5.0

_LABEL_SOURCES = frozenset({"reported", "command", "mixed", "detected", "native_hook"})
_MODEL_SOURCES = frozenset({"reported", "command", "harness", "native_hook", "unknown"})
_HARNESS_SOURCES = frozenset({"reported", "command", "native_hook"})
_INTEGRATION_MODES = frozenset({"wrapper", "detected_wrapper", "native_hook"})
_OPENCODE_EXECUTABLE_ENV = "ATTRIBUTION_OPENCODE_EXECUTABLE"
_WINDOWS_EXECUTABLE_SUFFIXES = (".exe", ".cmd", ".bat", ".com")


@dataclass(frozen=True, slots=True)
class _AdapterContext:
    """The environment and directory seen by a directly invoked command."""

    environment: dict[str, str]
    cwd: Path
    executable: str | None
    executable_index: int | None
    uses_env: bool
    complete: bool


class SnapshotLimitError(ValueError):
    """Raised before a bounded repository snapshot exceeds its memory budget."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _content_hash(content: bytes | None) -> str | None:
    if content is None:
        return None
    return hashlib.sha256(content).hexdigest()


def _base_commit(repo: Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "--verify", "HEAD^{commit}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=system_subprocess_environment(),
        check=False,
    )
    if result.returncode:
        return None
    return result.stdout.decode("ascii").strip()


def _listed_paths(repo: Path) -> list[str]:
    output = _run_git(
        repo,
        "ls-files",
        "-z",
        "--cached",
        "--others",
        "--exclude-standard",
    ).stdout
    paths: list[str] = []
    for raw_path in output.split(b"\0"):
        if not raw_path:
            continue
        try:
            paths.append(raw_path.decode("utf-8"))
        except UnicodeDecodeError as exc:
            raise ValueError("Git paths must be valid UTF-8 for attribution") from exc
    return sorted(set(paths))


def _listed_ignored_paths(repo: Path) -> list[str]:
    """Return ignored pre-existing files, collapsing ignored directories."""

    output = _run_git(
        repo,
        "ls-files",
        "-z",
        "--others",
        "--ignored",
        "--exclude-standard",
        "--directory",
    ).stdout
    paths: list[str] = []
    for raw_path in output.split(b"\0"):
        if not raw_path:
            continue
        try:
            paths.append(raw_path.decode("utf-8"))
        except UnicodeDecodeError as exc:
            raise ValueError("Git paths must be valid UTF-8 for attribution") from exc
    return sorted(set(paths))


def _snapshot(
    repo: Path,
    *,
    max_files: int | None = None,
    max_total_bytes: int | None = None,
    include_ignored: bool = False,
    maximum_total_bytes: int | None = None,
    maximum_files: int | None = None,
) -> tuple[dict[str, bytes | None], dict[str, str]]:
    """Read eligible files without changing either the worktree or index.

    The ``max_*`` limits mark individual paths as skipped for the established
    automatic-capture flow. The ``maximum_*`` limits fail closed for native
    harness hooks, which must not retain a partial baseline.
    """

    files: dict[str, bytes | None] = {}
    skipped: dict[str, str] = {}
    paths = _listed_paths(repo)
    if maximum_files is not None and len(paths) > maximum_files:
        raise SnapshotLimitError(f"snapshot has more than {maximum_files:,} files")
    total_bytes = 0
    scanned_bytes = 0
    for file_number, relative_path in enumerate(paths, start=1):
        if max_files is not None and file_number > max_files:
            skipped[relative_path] = "snapshot_file_limit"
            continue
        path = repo / relative_path
        try:
            metadata = path.lstat()
        except FileNotFoundError:
            # A missing tracked path is an exact absence, distinct from an empty file.
            files[relative_path] = None
            continue
        except OSError:
            skipped[relative_path] = "unreadable"
            continue

        if not stat.S_ISREG(metadata.st_mode):
            skipped[relative_path] = "not_regular"
            continue
        if metadata.st_size > MAX_TEXT_BYTES:
            skipped[relative_path] = "too_large"
            continue
        if (
            maximum_total_bytes is not None
            and scanned_bytes + metadata.st_size > maximum_total_bytes
        ):
            raise SnapshotLimitError(
                f"snapshot exceeds {maximum_total_bytes // (1024 * 1024)} MiB"
            )
        if max_total_bytes is not None and total_bytes + metadata.st_size > max_total_bytes:
            skipped[relative_path] = "snapshot_total_limit"
            continue
        try:
            with path.open("rb") as handle:
                content = handle.read(MAX_TEXT_BYTES + 1)
        except OSError:
            skipped[relative_path] = "unreadable"
            continue
        if (
            maximum_total_bytes is not None
            and scanned_bytes + len(content) > maximum_total_bytes
        ):
            raise SnapshotLimitError(
                f"snapshot exceeds {maximum_total_bytes // (1024 * 1024)} MiB"
            )
        scanned_bytes += len(content)
        if len(content) > MAX_TEXT_BYTES:
            skipped[relative_path] = "too_large"
            continue
        if max_total_bytes is not None and total_bytes + len(content) > max_total_bytes:
            skipped[relative_path] = "snapshot_total_limit"
            continue
        if b"\0" in content:
            skipped[relative_path] = "binary"
            continue
        files[relative_path] = content
        total_bytes += len(content)
    if include_ignored:
        for relative_path in _listed_ignored_paths(repo):
            if relative_path in files or relative_path in skipped:
                continue
            if max_files is not None and len(files) + len(skipped) >= max_files:
                # Git paths cannot be empty, so this is an unambiguous internal
                # marker. Native capture discards the whole limited interval.
                skipped[""] = "snapshot_file_limit"
                break
            skipped[relative_path] = (
                "ignored_preexisting_directory"
                if relative_path.endswith("/")
                else "ignored_preexisting"
            )
    return files, skipped


@contextmanager
def _worktree_lock(
    repo: Path,
    *,
    blocking: bool = False,
    wait_seconds: float = 0.0,
) -> Iterator[None]:
    """Lock one worktree, either indefinitely or for a bounded interval."""

    lock_dir = git_dir(repo) / "attribution"
    lock_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    lock_path = lock_dir / "capture.lock"
    with lock_path.open("a+b") as lock_file:
        if blocking:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        else:
            deadline = time.monotonic() + max(0.0, float(wait_seconds))
            while True:
                try:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError as exc:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise ValueError(
                            "Another attribution capture is already running in this worktree"
                        ) from exc
                    time.sleep(min(0.05, remaining))
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


@contextmanager
def _manual_run_lock(repo: Path) -> Iterator[None]:
    """Serialize manual wrappers without blocking the Git hooks they invoke."""

    lock_dir = git_dir(repo) / "attribution"
    lock_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    lock_path = lock_dir / "manual-run.lock"
    with lock_path.open("a+b") as lock_file:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError(
                "Another attribution capture is already running in this worktree"
            ) from exc
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _settle_failed_manual_capture(
    repo: Path,
    capture_id: str,
    session_id: str,
    task_id: str,
    exit_code: int | None,
) -> None:
    """Best-effort terminal state for failures after the durable pre-snapshot."""

    with _worktree_lock(repo, blocking=True):
        connection = open_db(repo)
        try:
            ended_at = _utc_now()
            outcome = "interrupted" if exit_code == 130 else "failed"
            connection.execute(
                "UPDATE sessions SET ended_at = ?, exit_code = ?, outcome = ? WHERE id = ?",
                (ended_at, exit_code, outcome, session_id),
            )
            connection.execute(
                "UPDATE tasks SET updated_at = ? WHERE id = ?", (ended_at, task_id)
            )
            placeholders = ",".join("?" for _ in _ACTIVE_CAPTURE_STATES)
            connection.execute(
                f"""
                UPDATE hook_captures SET status = 'abandoned'
                WHERE id = ? AND status IN ({placeholders})
                """,
                (capture_id, *_ACTIVE_CAPTURE_STATES),
            )
            connection.execute(
                "DELETE FROM hook_capture_files WHERE capture_id = ?", (capture_id,)
            )
            connection.commit()
        finally:
            connection.close()


def _validated_label(value: str | None, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value.strip()


def _validated_choice(value: str, name: str, choices: frozenset[str]) -> str:
    result = _validated_label(value, name)
    if result not in choices:
        allowed = ", ".join(sorted(choices))
        raise ValueError(f"{name} must be one of: {allowed}")
    return result


def _validated_cost(cost_usd: float | None) -> float | None:
    if cost_usd is None:
        return None
    if isinstance(cost_usd, bool):
        raise ValueError("cost_usd must be a finite, nonnegative number or None")
    try:
        value = float(cost_usd)
    except (TypeError, ValueError) as exc:
        raise ValueError("cost_usd must be a finite, nonnegative number or None") from exc
    if not math.isfinite(value) or value < 0:
        raise ValueError("cost_usd must be a finite, nonnegative number or None")
    # A negative zero passes the check above and would print with its sign.
    return 0.0 if value == 0 else value


def _validated_count(value: int | None, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer or None")
    return value


def _validated_command(command: Sequence[str]) -> list[str]:
    if isinstance(command, (str, bytes)) or not isinstance(command, Sequence) or not command:
        raise ValueError("command must be a nonempty sequence of arguments")
    result: list[str] = []
    for argument in command:
        if not isinstance(argument, (str, os.PathLike)):
            raise ValueError("every command argument must be a string or path")
        result.append(os.fspath(argument))
    return result


def _executable_basename(value: str) -> str:
    name = value.replace("\\", "/").rsplit("/", 1)[-1]
    folded = name.casefold()
    for suffix in _WINDOWS_EXECUTABLE_SUFFIXES:
        if folded.endswith(suffix):
            return name[: -len(suffix)].casefold()
    return folded


def _environment_assignment(token: str) -> tuple[str, str] | None:
    """Parse one portable ``env`` assignment without invoking a shell."""

    name, separator, value = token.partition("=")
    if not separator or not name:
        return None
    if not (name[0].isalpha() or name[0] == "_"):
        return None
    if any(not (character.isalnum() or character == "_") for character in name):
        return None
    return name, value


def _adapter_context(
    argv: Sequence[str],
    invocation_cwd: Path,
    child_environment: Mapping[str, str],
) -> _AdapterContext:
    """Derive adapter inputs from a direct command or a leading ``env`` call.

    This parser is intentionally narrower than a shell. Unknown ``env`` options
    disable metadata discovery, while the original command still runs unchanged.
    """

    environment = dict(child_environment)
    environment.pop(_OPENCODE_EXECUTABLE_ENV, None)
    cwd = invocation_cwd
    if not argv:
        return _AdapterContext(environment, cwd, None, None, False, False)

    index = 0
    uses_env = _executable_basename(argv[0]) == "env"
    if uses_env:
        index = 1
        requested_cwd: str | None = None
        parsing_options = True
        while index < len(argv):
            token = argv[index]
            if parsing_options and token == "--":
                parsing_options = False
                index += 1
                continue
            if parsing_options and token in {"-i", "--ignore-environment", "-"}:
                environment.clear()
                index += 1
                continue
            if parsing_options and token in {"-u", "--unset"}:
                if index + 1 >= len(argv):
                    return _AdapterContext(environment, cwd, None, None, True, False)
                environment.pop(argv[index + 1], None)
                index += 2
                continue
            if parsing_options and token.startswith("--unset="):
                environment.pop(token.partition("=")[2], None)
                index += 1
                continue
            if parsing_options and token.startswith("-u") and token != "-u":
                environment.pop(token[2:], None)
                index += 1
                continue
            if parsing_options and token in {"-C", "--chdir"}:
                if index + 1 >= len(argv):
                    return _AdapterContext(environment, cwd, None, None, True, False)
                requested_cwd = argv[index + 1]
                index += 2
                continue
            if parsing_options and token.startswith("--chdir="):
                requested_cwd = token.partition("=")[2]
                index += 1
                continue
            if parsing_options and token.startswith("-C") and token != "-C":
                requested_cwd = token[2:]
                index += 1
                continue
            assignment = _environment_assignment(token)
            if assignment is not None:
                parsing_options = False
                name, value = assignment
                environment[name] = value
                index += 1
                continue
            if parsing_options and token.startswith("-"):
                return _AdapterContext(environment, cwd, None, None, True, False)
            break

        if requested_cwd is not None:
            if not requested_cwd:
                return _AdapterContext(environment, cwd, None, None, True, False)
            requested_path = Path(requested_cwd)
            if not requested_path.is_absolute():
                requested_path = invocation_cwd / requested_path
            try:
                cwd = requested_path.resolve(strict=False)
            except (OSError, RuntimeError, ValueError):
                return _AdapterContext(
                    environment, invocation_cwd, None, None, True, False
                )

    if index >= len(argv):
        return _AdapterContext(environment, cwd, None, None, uses_env, False)
    return _AdapterContext(environment, cwd, argv[index], index, uses_env, True)


def _command_with_wrapper_marker(
    argv: Sequence[str],
    context: _AdapterContext,
    session_id: str,
    *,
    extra_markers: Mapping[str, str] | None = None,
) -> list[str]:
    """Keep capture markers after a leading ``env -i`` clears its input."""

    result = list(argv)
    if context.uses_env and context.complete and context.executable_index is not None:
        markers = {"ATTRIBUTION_WRAPPER_SESSION_ID": session_id}
        markers.update(extra_markers or {})
        assignments = [f"{name}={value}" for name, value in markers.items()]
        result[context.executable_index:context.executable_index] = assignments
    return result


def _adapter_warning(harness: str, message: str, exc: Exception | None = None) -> str:
    if exc is None:
        return f"{harness} {message}"
    return f"{harness} {message} ({type(exc).__name__})"


def _safe_adapter_warnings(value: object) -> list[str]:
    """Keep adapter warnings bounded, printable, and JSON-safe."""

    result: list[str] = []
    try:
        for index, warning in enumerate(value):  # type: ignore[arg-type]
            if index == 32:
                break
            if not isinstance(warning, str):
                continue
            warning = warning.strip()
            if (
                warning
                and len(warning) <= 1024
                and not any(ord(character) < 32 for character in warning)
            ):
                result.append(warning)
    except Exception:
        return result
    return result


def _session_is_published(connection: sqlite3.Connection, session_id: str) -> bool:
    """Return whether an immutable cached note already contains the session."""

    for row in connection.execute("SELECT note_json FROM recorded_commits"):
        try:
            note = json.loads(row["note_json"])
        except (TypeError, ValueError, RecursionError) as exc:
            raise ValueError(
                "Could not verify immutable attribution records before metadata enrichment"
            ) from exc
        if not isinstance(note, dict) or not isinstance(note.get("sessions"), list):
            raise ValueError(
                "Could not verify immutable attribution records before metadata enrichment"
            )
        for session in note["sessions"]:
            if isinstance(session, dict) and session.get("id") == session_id:
                return True
    return False


def _enrich_session_metadata_locked(
    connection: sqlite3.Connection,
    session_id: str,
    *,
    harness_id: str | None = None,
    harness_version: str | None = None,
    native_session_id: str | None = None,
    model: str | None = None,
    provider: str | None = None,
    cost_usd: float | None = None,
    cost_source: str | None = None,
    agent_type: str | None = None,
    session_source: str | None = None,
    permission_mode: str | None = None,
    effort_level: str | None = None,
    turn_count: int | None = None,
    prompt_count: int | None = None,
    interrupt_count: int | None = None,
    compaction_count: int | None = None,
    model_switch_count: int | None = None,
    tool_call_count: int | None = None,
    duration_ms: int | None = None,
    models: Sequence[str] | None = None,
    refuse_published: bool,
    expected_worktree_id: str,
) -> dict[str, object]:
    """Apply allowlisted metadata while the caller holds the worktree lock."""

    text_values = {
        "harness_id": harness_id,
        "harness_version": harness_version,
        "native_session_id": native_session_id,
        "model": model,
        "provider": provider,
        "cost_source": cost_source,
    }
    for field, value in tuple(text_values.items()):
        if value is not None:
            text_values[field] = _validated_label(value, field)
    cost = _validated_cost(cost_usd)
    counts = {
        field: _validated_count(value, field)
        for field, value in (
            ("turn_count", turn_count),
            ("prompt_count", prompt_count),
            ("interrupt_count", interrupt_count),
            ("compaction_count", compaction_count),
            ("model_switch_count", model_switch_count),
            ("tool_call_count", tool_call_count),
        )
        if value is not None
    }
    facets: dict[str, object] = {
        field: _validated_label(value, field)
        for field, value in (
            ("agent_type", agent_type),
            ("session_source", session_source),
            ("permission_mode", permission_mode),
            ("effort_level", effort_level),
        )
        if value is not None
    }
    if duration_ms is not None:
        facets["duration_ms"] = _validated_count(duration_ms, "duration_ms")
    if isinstance(models, (str, bytes)):
        raise ValueError("models must be a sequence of model names or None")

    row = connection.execute(
        "SELECT * FROM sessions WHERE id = ?", (session_id,)
    ).fetchone()
    if row is None:
        raise ValueError(f"Unknown attribution session {session_id!r}")
    if row["worktree_id"] != expected_worktree_id:
        raise ValueError(
            "Joyride session does not belong to the supplied Git worktree"
        )
    if refuse_published and _session_is_published(connection, session_id):
        raise ValueError(
            "Joyride metadata is immutable after the session appears in a recorded commit"
        )

    updates: dict[str, object] = {}
    for field in (
        "harness_id",
        "harness_version",
        "native_session_id",
        "provider",
    ):
        if row[field] is None and text_values[field] is not None:
            updates[field] = text_values[field]
    model_was_reported = row["model_source"] == "reported" or (
        row["model_source"] is None and row["label_source"] == "reported"
    )
    if text_values["model"] is not None and not model_was_reported:
        updates["model"] = text_values["model"]
        updates["model_source"] = "harness"
    if cost is not None and row["cost_usd"] is None:
        updates["cost_usd"] = cost
        updates["cost_source"] = text_values["cost_source"] or "harness"

    effective_model_source = str(
        updates.get("model_source", row["model_source"] or "unknown")
    )
    harness_source = str(row["harness_source"] or "unknown")
    sources = {effective_model_source, harness_source}
    if "reported" in sources and len(sources) > 1:
        updates["label_source"] = "mixed"
    elif sources <= {"command", "harness", "native_hook"}:
        updates["label_source"] = "detected"

    if updates:
        assignments = ", ".join(f"{field} = ?" for field in updates)
        connection.execute(
            f"UPDATE sessions SET {assignments} WHERE id = ?",
            (*updates.values(), session_id),
        )

    # A wrapped harness describes its own workflow the way a hook payload does,
    # so the same ledger writers apply it. A facet that is already known keeps
    # the value it has; the permission mode and the effort level hold the most
    # recent one, exactly as they do for a hook event.
    if facets:
        apply_session_facets(connection, session_id, facets, fields=tuple(facets))

    # A native record lists the models a session used in order, which makes
    # each entry after the first one model change. The record says nothing
    # about who or what changed it, so the source stays unknown.
    previous: str | None = None
    for entry in models or ():
        if previous is not None:
            record_model_switch(
                connection,
                session_id,
                from_model=previous,
                to_model=entry,
                source="unknown",
                occurred_at=_utc_now(),
            )
        previous = entry

    # A count from a native record fills a counter that no event of this
    # session advanced. A session that counted its own events keeps its count,
    # so a wrapper that also ran hooks never counts one turn twice.
    for field, value in counts.items():
        connection.execute(
            f"UPDATE sessions SET {field} = ? WHERE id = ? AND {field} = 0",
            (value, session_id),
        )

    result = connection.execute(
        """
        SELECT id, model, harness, harness_id, harness_version,
               native_session_id, provider, model_source, harness_source,
               label_source, cost_usd, cost_source, integration_mode
        FROM sessions WHERE id = ?
        """,
        (session_id,),
    ).fetchone()
    return dict(result)


def run_session(
    repo: RepoPath,
    feature: str | None,
    model: str | None,
    harness: str | None,
    cost_usd: float | None,
    command: Sequence[str],
    *,
    actor_kind: str = "ai",
    task_query: str | None = None,
    role: str = "implementation",
    summary: str | None = None,
    parent_session_id: str | None = None,
    token_count: int | None = None,
    usage_includes_children: bool = False,
    harness_id: str | None = None,
    harness_version: str | None = None,
    provider: str | None = None,
    model_source: str | None = None,
    harness_source: str | None = None,
    label_source: str = "reported",
    integration_mode: str = "wrapper",
    metadata_adapter_id: str | None = None,
) -> dict[str, object]:
    """Run ``command`` and record its exact repository transition.

    Actor kind, labels and optional cost are explicitly reported by the caller.
    A manual capture observes an editor command's transition, not its keystrokes;
    concurrent or automated edits are indistinguishable inside the same capture.
    The command's arguments are intentionally not persisted or published.
    """

    if actor_kind not in ("ai", "manual"):
        raise ValueError("actor_kind must be 'ai' or 'manual'")
    if actor_kind == "manual":
        model = "Manual" if model is None else model
        harness = "editor" if harness is None else harness
    model = _validated_label(model, "model")
    harness = _validated_label(harness, "harness")
    label_source = _validated_choice(label_source, "label_source", _LABEL_SOURCES)
    integration_mode = _validated_choice(
        integration_mode, "integration_mode", _INTEGRATION_MODES
    )
    if harness_id is not None:
        harness_id = _validated_label(harness_id, "harness_id")
    if harness_version is not None:
        harness_version = _validated_label(harness_version, "harness_version")
    if provider is not None:
        provider = _validated_label(provider, "provider")
    if model_source is not None:
        model_source = _validated_choice(model_source, "model_source", _MODEL_SOURCES)
    if harness_source is not None:
        harness_source = _validated_choice(
            harness_source, "harness_source", _HARNESS_SOURCES
        )
    if metadata_adapter_id is not None:
        metadata_adapter_id = _validated_label(
            metadata_adapter_id, "metadata_adapter_id"
        )
        if harness_id is not None and metadata_adapter_id != harness_id:
            raise ValueError("metadata_adapter_id must match harness_id")
    cost = _validated_cost(cost_usd)
    tokens = validated_token_count(token_count)
    if role not in ROLES:
        raise ValueError(f"role must be one of {', '.join(ROLES)}")
    if summary is not None:
        summary = _validated_label(summary, "summary")
    inherited_task_id = (
        os.environ.get("ATTRIBUTION_TASK_ID")
        if task_query is None and feature is None
        else None
    )
    inherited_parent_id = (
        os.environ.get("ATTRIBUTION_SESSION_ID")
        if parent_session_id is None and task_query is None and feature is None
        else None
    )
    task_query = task_query or inherited_task_id
    parent_session_id = parent_session_id or inherited_parent_id
    if parent_session_id is not None:
        parent_session_id = _validated_label(parent_session_id, "parent_session_id")
    argv = _validated_command(command)
    root = repository_root(repo)
    invocation_cwd = Path(repo).resolve()
    try:
        valid_invocation_cwd = (
            invocation_cwd.is_dir() and repository_root(invocation_cwd) == root
        )
    except (OSError, ValueError):
        valid_invocation_cwd = False
    if not valid_invocation_cwd:
        invocation_cwd = root
    worktree_id = str(git_dir(root))
    parent_task_id: str | None = None
    if parent_session_id is not None:
        with open_db(root) as connection:
            parent = connection.execute(
                "SELECT task_id FROM sessions WHERE id = ?", (parent_session_id,)
            ).fetchone()
            if parent is None:
                raise ValueError(f"No parent session matches {parent_session_id!r}")
            parent_task_id = parent["task_id"]
    task = resolve_task(
        root,
        task_query=task_query or parent_task_id,
        feature=feature,
    )
    if inherited_task_id is not None:
        task["membership_source"] = "parent_context"
    if parent_task_id is not None and task["id"] != parent_task_id:
        raise ValueError("Parent and child sessions must belong to the same task")
    session_id = str(uuid.uuid4())
    capture_id = str(uuid.uuid4())
    capture_markers = {
        "ATTRIBUTION_TASK_ID": str(task["id"]),
        "ATTRIBUTION_SESSION_ID": session_id,
        "ATTRIBUTION_WRAPPED_CAPTURE": "1",
        "ATTRIBUTION_WRAPPER_SESSION_ID": session_id,
    }
    child_environment = {**os.environ, **capture_markers}
    command_context = _adapter_context(argv, invocation_cwd, child_environment)
    child_argv = _command_with_wrapper_marker(
        argv,
        command_context,
        session_id,
        extra_markers=capture_markers,
    )
    if command_context.complete:
        command_context.environment.update(capture_markers)

    adapter_snapshot = None
    adapter_environment: dict[str, str] | None = None
    adapter_repo: Path | None = None
    adapter_warnings: list[str] = []
    metadata_result: dict[str, object] | None = None
    effective_integration_mode = integration_mode

    # A dedicated lock prevents two manual wrappers from running together. The
    # shorter capture lock is deliberately released while the child runs: a Git
    # commit inside the child must be able to queue itself from post-commit.
    with _manual_run_lock(root):
        with _worktree_lock(root, blocking=True):
            connection = open_db(root)
            try:
                # A process crash releases manual-run.lock but can leave its
                # durable row behind. Retire that stale row before starting a
                # new wrapper so it cannot permanently block deferred commits.
                stale = connection.execute(
                    """
                    SELECT id, ledger_session_id
                    FROM hook_captures
                    WHERE worktree_id = ? AND harness = ?
                      AND status IN ('pending', 'contaminated', 'limited', 'imported')
                    """,
                    (worktree_id, _MANUAL_CAPTURE_HARNESS),
                ).fetchall()
                abandoned_at = _utc_now()
                for row in stale:
                    connection.execute(
                        "UPDATE hook_captures SET status = 'abandoned' WHERE id = ?",
                        (row["id"],),
                    )
                    connection.execute(
                        "UPDATE sessions SET ended_at = ?, outcome = 'abandoned' WHERE id = ?",
                        (abandoned_at, row["ledger_session_id"]),
                    )
                    connection.execute(
                        "DELETE FROM hook_capture_files WHERE capture_id = ?",
                        (row["id"],),
                    )
                placeholders = ",".join("?" for _ in _ACTIVE_CAPTURE_STATES)
                active = int(
                    connection.execute(
                        f"""
                        SELECT COUNT(*) AS count FROM hook_captures
                        WHERE worktree_id = ? AND status IN ({placeholders})
                        """,
                        (worktree_id, *_ACTIVE_CAPTURE_STATES),
                    ).fetchone()["count"]
                )
                native_units = connection.execute(
                    """
                    SELECT units.*, sessions.task_id AS task_id
                    FROM hook_units AS units
                    JOIN sessions ON sessions.id = units.attribution_session_id
                    WHERE units.worktree_id = ? AND units.active = 1
                    """,
                    (worktree_id,),
                ).fetchall()
                if native_units:
                    # A wrapper interval cannot safely overlap an existing
                    # native hook interval. Retire the native baselines and
                    # make the wrapper interval contaminated as well.
                    from .hook_capture import _close_without_evidence

                    active += _close_without_evidence(
                        connection,
                        native_units,
                        abandoned_at,
                        "wrapper_overlap",
                    )
                if metadata_adapter_id is not None:
                    try:
                        from .adapters import snapshot_adapter, supported_adapters

                        if metadata_adapter_id in supported_adapters():
                            if command_context.complete:
                                try:
                                    context_is_in_repo = (
                                        repository_root(command_context.cwd) == root
                                    )
                                except (OSError, ValueError):
                                    context_is_in_repo = False
                                if context_is_in_repo:
                                    adapter_environment = command_context.environment
                                    adapter_repo = command_context.cwd
                                    if (
                                        metadata_adapter_id == "opencode"
                                        and command_context.executable is not None
                                        and _executable_basename(
                                            command_context.executable
                                        ) in {"opencode", "opencode-ai"}
                                    ):
                                        adapter_environment[
                                            _OPENCODE_EXECUTABLE_ENV
                                        ] = command_context.executable
                                    try:
                                        adapter_snapshot = snapshot_adapter(
                                            metadata_adapter_id,
                                            adapter_repo,
                                            adapter_environment,
                                        )
                                    except Exception as exc:
                                        adapter_warnings.append(
                                            _adapter_warning(
                                                harness,
                                                "metadata discovery could not start",
                                                exc,
                                            )
                                        )
                            if adapter_snapshot is None and not adapter_warnings:
                                adapter_warnings.append(
                                    _adapter_warning(
                                        harness,
                                        "native session metadata was unavailable",
                                    )
                                )
                    except Exception as exc:
                        # Metadata is optional. Its failure must not prevent the
                        # requested command or its exact edit capture.
                        adapter_warnings.append(
                            _adapter_warning(
                                harness,
                                "metadata discovery could not start",
                                exc,
                            )
                        )

                if adapter_snapshot is not None and integration_mode == "wrapper":
                    effective_integration_mode = "detected_wrapper"
                if active:
                    # Neither side of an overlapping interval can claim an
                    # exclusive file transition.
                    before: dict[str, bytes | None] = {}
                    before_skips: dict[str, str] = {}
                    connection.execute(
                        f"""
                        UPDATE hook_captures SET status = 'contaminated'
                        WHERE worktree_id = ? AND status IN ({placeholders})
                        """,
                        (worktree_id, *_ACTIVE_CAPTURE_STATES),
                    )
                else:
                    before, before_skips = _snapshot(root, include_ignored=True)

                started_at = _utc_now()
                base_commit = _base_commit(root)
                connection.execute(
                    """
                    INSERT INTO sessions(
                        id, worktree_id, task_id, feature, feature_source, model,
                        harness, actor_kind, harness_id, harness_version, provider,
                        model_source, harness_source, integration_mode,
                        invocation_cwd, label_source, membership_source, role,
                        summary, parent_session_id, token_count, token_source,
                        cost_usd, cost_source, usage_includes_children,
                        started_at, base_commit
                    ) VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                        ?, ?, ?, ?, ?, ?, ?, ?, ?
                    )
                    """,
                    (
                        session_id,
                        worktree_id,
                        task["id"],
                        task["name"],
                        "reported" if feature is not None else task["membership_source"],
                        model,
                        harness,
                        actor_kind,
                        harness_id,
                        harness_version,
                        provider,
                        model_source,
                        harness_source,
                        effective_integration_mode,
                        str(invocation_cwd),
                        label_source,
                        task["membership_source"],
                        role,
                        summary,
                        parent_session_id,
                        tokens,
                        "reported" if tokens is not None else None,
                        cost,
                        "reported" if cost is not None else None,
                        int(bool(usage_includes_children)),
                        started_at,
                        base_commit,
                    ),
                )
                connection.execute(
                    """
                    INSERT INTO hook_captures(
                        id, worktree_id, harness, native_session_id, tool_use_id,
                        ledger_session_id, base_commit, started_at, status
                    ) VALUES (?, ?, ?, ?, 'run', ?, ?, ?, ?)
                    """,
                    (
                        capture_id,
                        worktree_id,
                        _MANUAL_CAPTURE_HARNESS,
                        session_id,
                        session_id,
                        base_commit,
                        started_at,
                        "contaminated" if active else "pending",
                    ),
                )
                connection.commit()
            finally:
                connection.close()

        command_error: BaseException | None = None
        exit_code: int | None
        try:
            # An explicitly wrapped session already captures this command;
            # native tool hooks must not record the same transition twice.
            exit_code = subprocess.run(
                child_argv,
                cwd=invocation_cwd,
                env=system_subprocess_environment(child_environment),
                check=False,
            ).returncode
        except KeyboardInterrupt as exc:
            # The child may have changed files before interruption. Preserve
            # that evidence, then re-raise so the CLI retains normal Ctrl+C
            # behavior.
            command_error = exc
            exit_code = 130
        except OSError as exc:
            command_error = exc
            exit_code = None
        except BaseException as exc:
            # Unexpected child-launch failures still need to release durable
            # capture state before their original exception is propagated.
            command_error = exc
            exit_code = None

        changed_files: list[str] = []
        after_skips: dict[str, str] = {}
        try:
            with _worktree_lock(root, blocking=True):
                connection = open_db(root)
                try:
                    capture = connection.execute(
                        "SELECT status FROM hook_captures WHERE id = ?",
                        (capture_id,),
                    ).fetchone()
                    if capture is None:
                        raise ValueError("Manual attribution capture state disappeared")
                    capture_status = str(capture["status"])
                    if capture_status == "pending":
                        after, after_skips = _snapshot(root, include_ignored=True)
                        blocked_paths = set(before_skips) | set(after_skips)
                        ignored_directories = {
                            path
                            for skipped in (before_skips, after_skips)
                            for path, reason in skipped.items()
                            if reason == "ignored_preexisting_directory"
                        }
                        for path in sorted(set(before) | set(after)):
                            if path in blocked_paths or any(
                                path.startswith(directory)
                                for directory in ignored_directories
                            ):
                                continue
                            before_content = before.get(path)
                            after_content = after.get(path)
                            if before_content == after_content:
                                continue
                            changed_files.append(path)
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

                    ended_at = _utc_now()
                    if exit_code == 0:
                        outcome = "completed"
                    elif exit_code == 130:
                        outcome = "interrupted"
                    else:
                        outcome = "failed"
                    connection.execute(
                        "UPDATE sessions SET ended_at = ?, exit_code = ?, outcome = ? WHERE id = ?",
                        (ended_at, exit_code, outcome, session_id),
                    )
                    connection.execute(
                        "UPDATE tasks SET updated_at = ? WHERE id = ?",
                        (ended_at, task["id"]),
                    )
                    if (
                        adapter_snapshot is not None
                        and adapter_environment is not None
                        and adapter_repo is not None
                    ):
                        try:
                            from .adapters import finalize_adapter

                            metadata = finalize_adapter(
                                adapter_snapshot,
                                adapter_repo,
                                adapter_environment,
                            )
                            if metadata is None:
                                adapter_warnings.append(
                                    _adapter_warning(
                                        harness,
                                        "native session metadata was ambiguous or unavailable",
                                    )
                                )
                            elif metadata.harness_id != metadata_adapter_id:
                                adapter_warnings.append(
                                    _adapter_warning(
                                        harness,
                                        "native session metadata named a different harness",
                                    )
                                )
                            else:
                                adapter_warnings.extend(
                                    _safe_adapter_warnings(metadata.warnings)
                                )
                                connection.execute("SAVEPOINT adapter_metadata")
                                try:
                                    metadata_result = _enrich_session_metadata_locked(
                                        connection,
                                        session_id,
                                        harness_id=metadata.harness_id,
                                        harness_version=metadata.harness_version,
                                        native_session_id=metadata.native_session_id,
                                        model=metadata.model,
                                        provider=metadata.provider,
                                        cost_usd=metadata.cost_usd,
                                        cost_source=metadata.cost_source,
                                        agent_type=metadata.agent_type,
                                        session_source=metadata.session_source,
                                        permission_mode=metadata.permission_mode,
                                        effort_level=metadata.effort_level,
                                        turn_count=metadata.turn_count,
                                        prompt_count=metadata.prompt_count,
                                        interrupt_count=metadata.interrupt_count,
                                        compaction_count=metadata.compaction_count,
                                        model_switch_count=metadata.model_switch_count,
                                        tool_call_count=metadata.tool_call_count,
                                        duration_ms=metadata.duration_ms,
                                        models=metadata.models,
                                        refuse_published=False,
                                        expected_worktree_id=worktree_id,
                                    )
                                    connection.execute("RELEASE adapter_metadata")
                                except Exception as exc:
                                    metadata_result = None
                                    connection.execute("ROLLBACK TO adapter_metadata")
                                    connection.execute("RELEASE adapter_metadata")
                                    adapter_warnings.append(
                                        _adapter_warning(
                                            harness,
                                            "native session metadata could not be saved",
                                            exc,
                                        )
                                    )
                        except Exception as exc:
                            adapter_warnings.append(
                                _adapter_warning(
                                    harness,
                                    "metadata discovery failed",
                                    exc,
                                )
                            )
                    terminal_status = {
                        "pending": "completed",
                        "contaminated": "completed_contaminated",
                        "limited": "completed_limited",
                        "imported": "completed_imported",
                    }.get(capture_status)
                    if terminal_status is not None:
                        connection.execute(
                            "UPDATE hook_captures SET status = ? WHERE id = ?",
                            (terminal_status, capture_id),
                        )
                    connection.execute(
                        "DELETE FROM hook_capture_files WHERE capture_id = ?",
                        (capture_id,),
                    )
                    connection.commit()
                finally:
                    connection.close()
        except BaseException as completion_error:
            try:
                _settle_failed_manual_capture(
                    root, capture_id, session_id, str(task["id"]), exit_code
                )
            except BaseException:
                # Preserve the original command/completion failure. A later
                # status or uninstall call can still abandon a durable row.
                pass
            try:
                from .automation import _drain_pending

                _drain_pending(root, worktree_id)
            except BaseException:
                pass
            if command_error is not None:
                raise command_error
            raise completion_error

        # Import late to avoid capture.py <-> automation.py import recursion.
        try:
            from .automation import _drain_pending

            _drain_pending(root, worktree_id)
        except BaseException:
            if command_error is not None:
                raise command_error
            raise

        from .task_notes import sync_task_if_anchored

        task_sync_error: str | None = None
        try:
            sync_task_if_anchored(root, str(task["id"]))
        except (OSError, ValueError) as exc:
            task_sync_error = str(exc)

        if command_error is not None:
            raise command_error

        skip_reasons: dict[str, set[str]] = {}
        for skipped in (before_skips, after_skips):
            for path, reason in skipped.items():
                if reason.startswith("ignored_preexisting"):
                    continue
                skip_reasons.setdefault(path, set()).add(reason)
        skipped_files = [
            {"path": path, "reason": ",".join(sorted(reasons))}
            for path, reasons in sorted(skip_reasons.items())
        ]
        result: dict[str, object] = {
            "session_id": session_id,
            "actor_kind": actor_kind,
            "task": {
                "id": task["id"],
                "name": task["name"],
                "membership_source": task["membership_source"],
            },
            "exit_code": exit_code,
            "changed_files": changed_files,
            "skipped_files": skipped_files,
            "task_sync_error": task_sync_error,
        }
        if metadata_result is not None:
            result["metadata"] = metadata_result
        if adapter_warnings:
            result["warnings"] = adapter_warnings
        return result


def enrich_session_metadata(
    repo: RepoPath,
    session_id: str,
    *,
    harness_id: str | None = None,
    harness_version: str | None = None,
    native_session_id: str | None = None,
    model: str | None = None,
    provider: str | None = None,
    cost_usd: float | None = None,
    cost_source: str | None = None,
) -> dict[str, object]:
    """Add metadata unless an immutable commit record already contains it."""

    session_id = _validated_label(session_id, "session_id")
    root = repository_root(repo)
    with _worktree_lock(root, wait_seconds=METADATA_LOCK_WAIT_SECONDS):
        connection = open_db(root)
        try:
            result = _enrich_session_metadata_locked(
                connection,
                session_id,
                harness_id=harness_id,
                harness_version=harness_version,
                native_session_id=native_session_id,
                model=model,
                provider=provider,
                cost_usd=cost_usd,
                cost_source=cost_source,
                refuse_published=True,
                expected_worktree_id=str(git_dir(root)),
            )
            connection.commit()
            return result
        finally:
            connection.close()
