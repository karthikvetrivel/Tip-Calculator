"""Infer, manage, and correct task membership for captured sessions."""

from __future__ import annotations

from datetime import datetime, timezone
import math
import os
from pathlib import Path
import re
import sqlite3
import subprocess
from typing import Any
from urllib.parse import urlsplit
import uuid

from .runtime import system_subprocess_environment
from .store import RepoPath, git_dir, open_db, repository_root


ROLES = ("planning", "implementation", "testing", "review", "other")
OUTCOMES = ("completed", "failed", "interrupted", "abandoned")
_PRIMARY_BRANCHES = {"main", "master", "trunk", "develop", "development"}
_BRANCH_PREFIXES = ("feature/", "feat/", "fix/", "bugfix/", "chore/", "codex/")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _nonempty(value: str | None, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value.strip()


def _nullable_text(value: str | None, name: str) -> str | None:
    if value is None:
        return None
    return _nonempty(value, name)


def validated_token_count(value: int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("token_count must be a nonnegative integer or None")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("token_count must be a nonnegative integer or None") from exc
    if result != value or result < 0:
        raise ValueError("token_count must be a nonnegative integer or None")
    return result


def validated_cost(value: float | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("cost_usd must be a finite, nonnegative number or None")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("cost_usd must be a finite, nonnegative number or None") from exc
    if not math.isfinite(result) or result < 0:
        raise ValueError("cost_usd must be a finite, nonnegative number or None")
    return result


def current_branch(repo: RepoPath) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(repo), "symbolic-ref", "--quiet", "--short", "HEAD"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=system_subprocess_environment(),
        check=False,
    )
    if result.returncode:
        return None
    branch = result.stdout.decode("utf-8", errors="replace").strip()
    return branch or None


def current_head(repo: RepoPath) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "--verify", "--quiet", "HEAD^{commit}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=system_subprocess_environment(),
        check=False,
    )
    if result.returncode:
        return None
    head = result.stdout.decode("ascii", errors="replace").strip()
    return head or None


def _branch_task_name(branch: str) -> str:
    value = branch
    for prefix in _BRANCH_PREFIXES:
        if value.startswith(prefix):
            value = value[len(prefix) :]
            break
    value = re.sub(r"[-_.]+", " ", value).strip()
    return value[:1].upper() + value[1:] if value else branch


def _remote_web_url(repo: Path, kind: str, number: str) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(repo), "remote", "get-url", "origin"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=system_subprocess_environment(),
        text=True,
        check=False,
    )
    if result.returncode:
        return None
    remote = result.stdout.strip()
    match = re.fullmatch(r"git@([A-Za-z0-9.-]+):(.+)", remote)
    if match:
        host, path = match.groups()
    else:
        try:
            parsed = urlsplit(remote)
            host = parsed.hostname
            path = parsed.path.lstrip("/")
            port = parsed.port
        except ValueError:
            return None
        if parsed.scheme not in {"http", "https"} or not host:
            return None
        if port is not None:
            host = f"{host}:{port}"
    if path.endswith(".git"):
        path = path[:-4]
    if not path or any(ord(character) < 32 for character in path):
        return None
    suffix = f"pull/{number}" if kind == "pull" else f"-/merge_requests/{number}"
    return f"https://{host}/{path.rstrip('/')}/{suffix}"


def infer_pull_request(repo: RepoPath, commit: str | None) -> tuple[str, str | None] | None:
    """Infer a PR only from local provider refs that point at the task head."""

    if commit is None:
        return None
    result = subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "for-each-ref",
            "--format=%(objectname)%00%(refname)",
            "refs/pull",
            "refs/merge-requests",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=system_subprocess_environment(),
        check=False,
    )
    if result.returncode:
        return None
    for raw_line in result.stdout.splitlines():
        try:
            object_id, raw_ref = raw_line.split(b"\0", 1)
            ref = raw_ref.decode("utf-8")
        except (ValueError, UnicodeError):
            continue
        if object_id.decode("ascii", errors="ignore") != commit:
            continue
        pull = re.fullmatch(r"refs/pull/(\d+)/(?:head|merge)", ref)
        merge_request = re.fullmatch(r"refs/merge-requests/(\d+)/(?:head|merge)", ref)
        if pull:
            number = pull.group(1)
            return f"#{number}", _remote_web_url(Path(repo), "pull", number)
        if merge_request:
            number = merge_request.group(1)
            return f"!{number}", _remote_web_url(Path(repo), "merge_request", number)
    return None


def _task_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {key: row[key] for key in row.keys()}


def _active_task(connection: sqlite3.Connection, task_id: str) -> sqlite3.Row | None:
    return connection.execute(
        "SELECT * FROM tasks WHERE id = ? AND merged_into IS NULL", (task_id,)
    ).fetchone()


def select_task(connection: sqlite3.Connection, query: str) -> sqlite3.Row:
    query = _nonempty(query, "task")
    exact = connection.execute(
        "SELECT * FROM tasks WHERE merged_into IS NULL AND (id = ? OR name = ?)",
        (query, query),
    ).fetchall()
    if len(exact) == 1:
        return exact[0]
    folded = query.casefold()
    rows = connection.execute(
        "SELECT * FROM tasks WHERE merged_into IS NULL ORDER BY updated_at DESC, id"
    ).fetchall()
    matches = [
        row
        for row in rows
        if str(row["id"]).casefold().startswith(folded)
        or str(row["name"]).casefold() == folded
    ]
    if not matches:
        raise ValueError(f"No task matches {query!r}")
    if len(matches) > 1:
        raise ValueError(f"Task {query!r} is ambiguous; use its full ID")
    return matches[0]


def _bind(
    connection: sqlite3.Connection,
    *,
    worktree_id: str,
    task_id: str,
    branch: str | None,
    head: str | None,
    pinned: bool,
) -> None:
    connection.execute(
        """
        INSERT INTO task_contexts(worktree_id, task_id, branch, head_commit, pinned, updated_at)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(worktree_id) DO UPDATE SET
            task_id = excluded.task_id,
            branch = excluded.branch,
            head_commit = excluded.head_commit,
            pinned = excluded.pinned,
            updated_at = excluded.updated_at
        """,
        (worktree_id, task_id, branch, head, int(pinned), _utc_now()),
    )


def _create_task(
    connection: sqlite3.Connection,
    *,
    name: str,
    branch: str | None,
    inference_source: str,
    kind: str = "feature",
    pr_ref: str | None = None,
    pr_url: str | None = None,
) -> sqlite3.Row:
    task_id = str(uuid.uuid4())
    now = _utc_now()
    connection.execute(
        """
        INSERT INTO tasks(
            id, name, kind, pr_ref, pr_url, branch, state,
            inference_source, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, 'active', ?, ?, ?)
        """,
        (task_id, name, kind, pr_ref, pr_url, branch, inference_source, now, now),
    )
    row = _active_task(connection, task_id)
    if row is None:
        raise ValueError("Could not create task")
    return row


def resolve_task(
    repo: RepoPath,
    *,
    task_query: str | None = None,
    feature: str | None = None,
) -> dict[str, Any]:
    """Return an explicit or conservatively inferred task and bind the worktree."""

    root = repository_root(repo)
    branch = current_branch(root)
    head = current_head(root)
    worktree_id = str(git_dir(root))
    with open_db(root) as connection:
        if task_query is not None:
            task = select_task(connection, task_query)
            membership_source = "explicit"
            pinned = True
        elif feature is not None:
            name = _nonempty(feature, "feature")
            candidates = connection.execute(
                "SELECT * FROM tasks WHERE merged_into IS NULL AND name = ? ORDER BY updated_at DESC",
                (name,),
            ).fetchall()
            task = candidates[0] if candidates else _create_task(
                connection,
                name=name,
                branch=branch,
                inference_source="feature_label",
            )
            membership_source = "feature_label"
            pinned = True
        else:
            context = connection.execute(
                """
                SELECT t.*, c.branch AS context_branch, c.head_commit AS context_head,
                       c.pinned AS context_pinned
                FROM task_contexts AS c
                JOIN tasks AS t ON t.id = c.task_id
                WHERE c.worktree_id = ? AND t.merged_into IS NULL
                """,
                (worktree_id,),
            ).fetchone()
            context_matches = context is not None and context["context_branch"] == branch
            if context_matches and (
                bool(context["context_pinned"])
                or branch not in _PRIMARY_BRANCHES
                or context["context_head"] == head
            ):
                task = context
                membership_source = "active_context"
                pinned = bool(context["context_pinned"])
            else:
                candidates = []
                if branch and branch not in _PRIMARY_BRANCHES:
                    candidates = connection.execute(
                        """
                        SELECT * FROM tasks
                        WHERE branch = ? AND state = 'active' AND merged_into IS NULL
                        ORDER BY updated_at DESC
                        """,
                        (branch,),
                    ).fetchall()
                if len(candidates) == 1:
                    task = candidates[0]
                    membership_source = "branch"
                elif branch and branch not in _PRIMARY_BRANCHES:
                    task = _create_task(
                        connection,
                        name=_branch_task_name(branch),
                        branch=branch,
                        inference_source="branch",
                    )
                    membership_source = "branch"
                else:
                    suffix = head[:7] if head else "unborn"
                    task = _create_task(
                        connection,
                        name=f"Unresolved work at {suffix}",
                        branch=branch,
                        inference_source="unresolved",
                        kind="unresolved",
                    )
                    membership_source = "unresolved"
                pinned = False

        inferred_pr = infer_pull_request(root, head)
        if inferred_pr is not None and not task["pr_ref"]:
            pr_ref, pr_url = inferred_pr
            connection.execute(
                """
                UPDATE tasks
                SET kind = 'pull_request', pr_ref = ?, pr_url = ?,
                    inference_source = 'pull_request_ref', updated_at = ?
                WHERE id = ?
                """,
                (pr_ref, pr_url, _utc_now(), task["id"]),
            )
            task = _active_task(connection, task["id"])
            if task is None:
                raise ValueError("Could not update inferred task")

        _bind(
            connection,
            worktree_id=worktree_id,
            task_id=task["id"],
            branch=branch,
            head=head,
            pinned=pinned,
        )
        connection.commit()
        result = _task_dict(task)
        result["membership_source"] = membership_source
        return result


def start_task(
    repo: RepoPath,
    name: str,
    *,
    pr_ref: str | None = None,
    pr_url: str | None = None,
) -> dict[str, Any]:
    root = repository_root(repo)
    branch = current_branch(root)
    kind = "pull_request" if pr_ref or pr_url else "feature"
    with open_db(root) as connection:
        task = _create_task(
            connection,
            name=_nonempty(name, "name"),
            branch=branch,
            inference_source="explicit",
            kind=kind,
            pr_ref=_nullable_text(pr_ref, "pr_ref"),
            pr_url=_nullable_text(pr_url, "pr_url"),
        )
        _bind(
            connection,
            worktree_id=str(git_dir(root)),
            task_id=task["id"],
            branch=branch,
            head=current_head(root),
            pinned=True,
        )
        connection.commit()
        return _task_dict(task)


def use_task(repo: RepoPath, query: str) -> dict[str, Any]:
    root = repository_root(repo)
    with open_db(root) as connection:
        task = select_task(connection, query)
        _bind(
            connection,
            worktree_id=str(git_dir(root)),
            task_id=task["id"],
            branch=current_branch(root),
            head=current_head(root),
            pinned=True,
        )
        connection.commit()
        return _task_dict(task)


def list_tasks(repo: RepoPath) -> list[dict[str, Any]]:
    root = repository_root(repo)
    with open_db(root) as connection:
        rows = connection.execute(
            """
            SELECT t.*, COUNT(s.id) AS session_count
            FROM tasks AS t
            LEFT JOIN sessions AS s ON s.task_id = t.id
            WHERE t.merged_into IS NULL
            GROUP BY t.id
            ORDER BY t.updated_at DESC, t.name, t.id
            """
        ).fetchall()
        return [_task_dict(row) for row in rows]


def link_pull_request(
    repo: RepoPath,
    query: str,
    pr_ref: str,
    *,
    pr_url: str | None = None,
) -> dict[str, Any]:
    root = repository_root(repo)
    with open_db(root) as connection:
        task = select_task(connection, query)
        connection.execute(
            """
            UPDATE tasks
            SET kind = 'pull_request', pr_ref = ?, pr_url = ?, updated_at = ?
            WHERE id = ?
            """,
            (
                _nonempty(pr_ref, "pr_ref"),
                _nullable_text(pr_url, "pr_url"),
                _utc_now(),
                task["id"],
            ),
        )
        connection.commit()
        updated = _active_task(connection, task["id"])
        if updated is None:
            raise ValueError("Could not update task")
        return _task_dict(updated)


def merge_tasks(repo: RepoPath, source_query: str, destination_query: str) -> dict[str, Any]:
    root = repository_root(repo)
    with open_db(root) as connection:
        source = select_task(connection, source_query)
        destination = select_task(connection, destination_query)
        if source["id"] == destination["id"]:
            raise ValueError("Source and destination tasks must be different")
        now = _utc_now()
        anchors = {
            str(value)
            for value in (source["anchor_commit"], destination["anchor_commit"])
            if value
        }
        destination_anchor = destination["anchor_commit"] or source["anchor_commit"]
        connection.execute(
            "UPDATE sessions SET task_id = ?, feature = ?, membership_source = 'manual_merge' WHERE task_id = ?",
            (destination["id"], destination["name"], source["id"]),
        )
        connection.execute(
            "UPDATE task_contexts SET task_id = ?, updated_at = ? WHERE task_id = ?",
            (destination["id"], now, source["id"]),
        )
        connection.execute(
            "UPDATE tasks SET state = 'merged', merged_into = ?, updated_at = ? WHERE id = ?",
            (destination["id"], now, source["id"]),
        )
        connection.execute(
            "UPDATE tasks SET updated_at = ?, anchor_commit = ? WHERE id = ?",
            (now, destination_anchor, destination["id"]),
        )
        connection.commit()
        updated = _active_task(connection, destination["id"])
        if updated is None:
            raise ValueError("Could not merge tasks")
        return {
            "source_task_id": source["id"],
            "task": _task_dict(updated),
            "sync_commits": sorted(anchors),
        }


def move_session(repo: RepoPath, session_id: str, task_query: str) -> dict[str, Any]:
    root = repository_root(repo)
    session_id = _nonempty(session_id, "session_id")
    with open_db(root) as connection:
        task = select_task(connection, task_query)
        existing = connection.execute(
            "SELECT task_id FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        if existing is None:
            raise ValueError(f"No session matches {session_id!r}")
        source_task_id = existing["task_id"]
        task_ids = {
            str(value) for value in (source_task_id, task["id"]) if value is not None
        }
        placeholders = ",".join("?" for _ in task_ids)
        anchors = {
            str(row["anchor_commit"])
            for row in connection.execute(
                f"SELECT anchor_commit FROM tasks WHERE id IN ({placeholders})",
                tuple(sorted(task_ids)),
            ).fetchall()
            if row["anchor_commit"]
        } if task_ids else set()
        cursor = connection.execute(
            """
            UPDATE sessions
            SET task_id = ?, feature = ?, membership_source = 'manual_move'
            WHERE id = ?
            """,
            (task["id"], task["name"], session_id),
        )
        now = _utc_now()
        if task_ids:
            connection.execute(
                f"UPDATE tasks SET updated_at = ? WHERE id IN ({placeholders})",
                (now, *sorted(task_ids)),
            )
        connection.commit()
        return {
            "session_id": session_id,
            "source_task_id": source_task_id,
            "task": _task_dict(task),
            "sync_commits": sorted(anchors),
        }


def add_session(
    repo: RepoPath,
    *,
    model: str,
    harness: str,
    task_query: str | None = None,
    role: str = "other",
    summary: str | None = None,
    parent_session_id: str | None = None,
    token_count: int | None = None,
    cost_usd: float | None = None,
    usage_includes_children: bool = False,
    outcome: str = "completed",
) -> dict[str, Any]:
    """Add a metadata-only session, such as a review or imported subagent."""

    root = repository_root(repo)
    model = _nonempty(model, "model")
    harness = _nonempty(harness, "harness")
    if role not in ROLES:
        raise ValueError(f"role must be one of {', '.join(ROLES)}")
    if outcome not in OUTCOMES:
        raise ValueError(f"outcome must be one of {', '.join(OUTCOMES)}")
    summary = _nullable_text(summary, "summary")
    token_count = validated_token_count(token_count)
    cost = validated_cost(cost_usd)
    inherited_task_id = os.environ.get("ATTRIBUTION_TASK_ID") if task_query is None else None
    inherited_parent_id = (
        os.environ.get("ATTRIBUTION_SESSION_ID") if parent_session_id is None else None
    )
    task_query = task_query or inherited_task_id
    parent_session_id = _nullable_text(
        parent_session_id or inherited_parent_id, "parent_session_id"
    )

    parent_task: str | None = None
    if parent_session_id is not None:
        with open_db(root) as connection:
            parent = connection.execute(
                "SELECT task_id FROM sessions WHERE id = ?", (parent_session_id,)
            ).fetchone()
            if parent is None:
                raise ValueError(f"No parent session matches {parent_session_id!r}")
            parent_task = parent["task_id"]
    task = resolve_task(root, task_query=task_query or parent_task)
    if inherited_task_id is not None:
        task["membership_source"] = "parent_context"
    if parent_task is not None and task["id"] != parent_task:
        raise ValueError("Parent and child sessions must belong to the same task")

    session_id = str(uuid.uuid4())
    now = _utc_now()
    with open_db(root) as connection:
        connection.execute(
            """
            INSERT INTO sessions(
                id, worktree_id, task_id, feature, model, harness, label_source,
                membership_source, role, summary, parent_session_id, token_count,
                token_source, cost_usd, cost_source, usage_includes_children,
                started_at, ended_at, base_commit, exit_code, outcome
            ) VALUES (?, ?, ?, ?, ?, ?, 'reported', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?)
            """,
            (
                session_id,
                str(git_dir(root)),
                task["id"],
                task["name"],
                model,
                harness,
                task["membership_source"],
                role,
                summary,
                parent_session_id,
                token_count,
                "reported" if token_count is not None else None,
                cost,
                "reported" if cost is not None else None,
                int(bool(usage_includes_children)),
                now,
                now,
                current_head(root),
                outcome,
            ),
        )
        connection.execute(
            "UPDATE tasks SET updated_at = ? WHERE id = ?", (now, task["id"])
        )
        connection.commit()
    return {"session_id": session_id, "task": task, "outcome": outcome}


def task_for_session_ids(
    connection: sqlite3.Connection, session_ids: set[str]
) -> set[str]:
    if not session_ids:
        return set()
    placeholders = ",".join("?" for _ in session_ids)
    rows = connection.execute(
        f"SELECT DISTINCT task_id FROM sessions WHERE id IN ({placeholders}) AND task_id IS NOT NULL",
        tuple(sorted(session_ids)),
    ).fetchall()
    return {str(row["task_id"]) for row in rows}


def update_task_anchor(repo: RepoPath, task_ids: set[str], commit: str) -> None:
    if not task_ids:
        return
    root = repository_root(repo)
    placeholders = ",".join("?" for _ in task_ids)
    now = _utc_now()
    inferred_pr = infer_pull_request(root, commit)
    with open_db(root) as connection:
        connection.execute(
            f"UPDATE tasks SET anchor_commit = ? WHERE id IN ({placeholders})",
            (commit, *sorted(task_ids)),
        )
        connection.execute(
            f"""
            UPDATE task_contexts
            SET head_commit = ?, updated_at = ?
            WHERE task_id IN ({placeholders})
            """,
            (commit, now, *sorted(task_ids)),
        )
        if inferred_pr is not None:
            pr_ref, pr_url = inferred_pr
            connection.execute(
                f"""
                UPDATE tasks
                SET kind = 'pull_request', pr_ref = COALESCE(pr_ref, ?),
                    pr_url = COALESCE(pr_url, ?), inference_source = 'pull_request_ref',
                    updated_at = ?
                WHERE id IN ({placeholders})
                """,
                (pr_ref, pr_url, now, *sorted(task_ids)),
            )
        connection.commit()


__all__ = [
    "OUTCOMES",
    "ROLES",
    "add_session",
    "current_branch",
    "current_head",
    "infer_pull_request",
    "link_pull_request",
    "list_tasks",
    "merge_tasks",
    "move_session",
    "resolve_task",
    "select_task",
    "start_task",
    "task_for_session_ids",
    "update_task_anchor",
    "use_task",
    "validated_cost",
    "validated_token_count",
]
