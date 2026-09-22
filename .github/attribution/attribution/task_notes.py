"""Publish task-wide session economics without changing source commits."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import subprocess
from typing import Any, Iterator

from .runtime import system_subprocess_environment
from .store import RepoPath, git_common_dir, open_db, repository_root
from .tasks import resolve_task, select_task, update_task_anchor


TASK_NOTES_REF = "refs/notes/attribution-tasks"
_MAX_TASK_NOTE_BYTES = 2 * 1024 * 1024
_TASK_FIELDS = (
    "id",
    "name",
    "kind",
    "pr_ref",
    "pr_url",
    "branch",
    "state",
    "inference_source",
    "created_at",
    "updated_at",
    "merged_into",
)
_SESSION_FIELDS = (
    "id",
    "task_id",
    "feature",
    "feature_source",
    "model",
    "harness",
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
    "actor_kind",
    "source_session_id",
    "label_source",
    "membership_source",
    "role",
    "summary",
    "parent_session_id",
    "token_count",
    "token_source",
    "cost_usd",
    "cost_source",
    "usage_includes_children",
    "started_at",
    "ended_at",
    "exit_code",
    "outcome",
    # Workflow facets, kept in step with ``notes._SESSION_OPTIONAL_FIELDS`` so
    # a task note and a commit note describe the same session.
    "agent_type",
    "launch_mode",
    "session_source",
    "permission_mode",
    "effort_level",
    "turn_count",
    "prompt_count",
    "interrupt_count",
    "compaction_count",
    "model_switch_count",
    "tool_call_count",
    "duration_ms",
    "activity_truncated",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _resolve_commit(repo: Path, value: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "--verify", "--end-of-options", f"{value}^{{commit}}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=system_subprocess_environment(),
        text=True,
        check=False,
    )
    if result.returncode:
        message = result.stderr.strip()
        raise ValueError(message or f"Could not resolve commit {value!r}")
    return result.stdout.strip()


def _show_note(repo: Path, commit: str) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(repo), "notes", f"--ref={TASK_NOTES_REF}", "show", commit],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=system_subprocess_environment(),
        check=False,
    )
    if result.returncode:
        return None
    try:
        return result.stdout.decode("utf-8").rstrip("\n")
    except UnicodeDecodeError as exc:
        raise ValueError(f"Existing task note for {commit} is not UTF-8") from exc


def _existing_payload(repo: Path, commit: str) -> dict[str, Any]:
    raw = _show_note(repo, commit)
    if raw is None:
        return {
            "version": 1,
            "commit": commit,
            "recorded_at": _utc_now(),
            "tasks": [],
        }
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Existing task note for {commit} is not valid JSON") from exc
    if (
        not isinstance(payload, dict)
        or payload.get("version") != 1
        or payload.get("commit") != commit
        or not isinstance(payload.get("tasks"), list)
    ):
        raise ValueError(f"Existing task note for {commit} has an unsupported format")
    return payload


def _records(repo: Path, task_ids: set[str]) -> list[dict[str, Any]]:
    if not task_ids:
        return []
    placeholders = ",".join("?" for _ in task_ids)
    with open_db(repo) as connection:
        task_rows = connection.execute(
            f"SELECT * FROM tasks WHERE id IN ({placeholders}) ORDER BY created_at, id",
            tuple(sorted(task_ids)),
        ).fetchall()
        result: list[dict[str, Any]] = []
        for task in task_rows:
            session_rows = connection.execute(
                "SELECT * FROM sessions WHERE task_id = ? ORDER BY started_at, rowid",
                (task["id"],),
            ).fetchall()
            result.append(
                {
                    **{field: task[field] for field in _TASK_FIELDS},
                    "sessions": [
                        {field: session[field] for field in _SESSION_FIELDS}
                        for session in session_rows
                    ],
                }
            )
        return result


@contextmanager
def _task_note_lock(repo: Path) -> Iterator[None]:
    lock_dir = git_common_dir(repo) / "attribution"
    lock_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    with (lock_dir / "task-notes.lock").open("a+b") as lock_file:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("Another task-note update is already running") from exc
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _write_records(
    root: Path, commit_sha: str, records: list[dict[str, Any]]
) -> dict[str, Any]:
    replacements = {record["id"]: record for record in records}
    with _task_note_lock(root):
        payload = _existing_payload(root, commit_sha)
        retained = [
            item
            for item in payload["tasks"]
            if isinstance(item, dict) and item.get("id") not in replacements
        ]
        next_tasks = retained + records
        if payload["tasks"] == next_tasks:
            return {
                "commit": commit_sha,
                "task_ids": sorted(replacements),
                "note_written": False,
            }
        payload["recorded_at"] = _utc_now()
        payload["tasks"] = next_tasks
        note_json = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
        if len(note_json.encode("utf-8")) > _MAX_TASK_NOTE_BYTES:
            raise ValueError("Task note is too large to write safely")
        result = subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "notes",
                f"--ref={TASK_NOTES_REF}",
                "add",
                "-f",
                "-F",
                "-",
                commit_sha,
            ],
            input=note_json.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=system_subprocess_environment(),
            check=False,
        )
        if result.returncode:
            message = result.stderr.decode("utf-8", errors="replace").strip()
            raise ValueError(message or f"Could not write task note for {commit_sha}")
        return {
            "commit": commit_sha,
            "task_ids": sorted(replacements),
            "note_written": True,
        }


def sync_task_ids(repo: RepoPath, task_ids: set[str], commit: str) -> dict[str, Any]:
    """Write current snapshots for task IDs to one commit's task note."""

    root = repository_root(repo)
    commit_sha = _resolve_commit(root, commit)
    update_task_anchor(root, task_ids, commit_sha)
    records = _records(root, task_ids)
    if not records:
        return {"commit": commit_sha, "task_ids": [], "note_written": False}
    return _write_records(root, commit_sha, records)


def sync_task(
    repo: RepoPath,
    task_query: str | None = None,
    *,
    commit: str = "HEAD",
) -> dict[str, Any]:
    root = repository_root(repo)
    if task_query is None:
        task = resolve_task(root)
    else:
        with open_db(root) as connection:
            task = dict(select_task(connection, task_query))
    return sync_task_ids(root, {str(task["id"])}, commit)


def sync_task_if_anchored(repo: RepoPath, task_id: str) -> dict[str, Any] | None:
    """Refresh a task snapshot after a metadata-only or later review session."""

    root = repository_root(repo)
    with open_db(root) as connection:
        task = connection.execute(
            "SELECT anchor_commit FROM tasks WHERE id = ?", (task_id,)
        ).fetchone()
    if task is None or not task["anchor_commit"]:
        return None
    return sync_task_ids(root, {task_id}, str(task["anchor_commit"]))


__all__ = ["TASK_NOTES_REF", "sync_task", "sync_task_ids", "sync_task_if_anchored"]
