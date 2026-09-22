"""Publish immutable, evidence-backed attribution as Git notes."""

from __future__ import annotations

import ast
import fcntl
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
import json
from pathlib import Path
import re
import sqlite3
import subprocess
from typing import Iterable

from .capture import MAX_TEXT_BYTES, _worktree_lock
from .runtime import system_subprocess_environment
from .store import RepoPath, _run_git, git_common_dir, git_dir, open_db, repository_root
from .workflow import MAX_WORKFLOW_DEPTH, build_profile


NOTES_REF = "refs/notes/attribution"
EMPTY_TREE = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"
MAX_NOTE_BYTES = 2 * 1024 * 1024
# The snapshot schema in ``sharing.py`` republishes at most this many agents in
# one workflow object. A note that named more would cost a push its whole
# snapshot, and this module cannot import that one, so the number is repeated
# here and pinned by a test.
MAX_WORKFLOW_SESSIONS = 500
RECORD_LOCK_WAIT_SECONDS = 5.0


@dataclass(frozen=True)
class _TreeEntry:
    mode: str
    object_id: str


@dataclass(frozen=True)
class _ChangedFile:
    status: str
    old_path: str | None
    new_path: str | None


@dataclass(frozen=True)
class _Edit:
    edit_id: int
    session_id: str
    before: bytes | None
    after: bytes | None
    tool_use_id: str | None = None


@dataclass
class _Reconciled:
    owners: list[str | None]
    tool_use_ids: list[str | None]
    removal_owners: dict[int, str]
    session_revisions: list[dict[str, object]]


@dataclass(frozen=True)
class _RevisionPoint:
    from_commit: str
    from_path: str
    from_line: int
    from_session_id: str | None
    to_session_id: str | None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _decode_path(raw_path: bytes) -> str:
    try:
        return raw_path.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("Git paths must be valid UTF-8 for attribution") from exc


def _resolve_commit(repo: Path, commit: str) -> str:
    if not isinstance(commit, str) or not commit:
        raise ValueError("commit must be a nonempty Git revision")
    output = _run_git(
        repo,
        "rev-parse",
        "--verify",
        "--end-of-options",
        f"{commit}^{{commit}}",
    ).stdout
    return output.decode("ascii").strip()


def _parents(repo: Path, commit: str) -> list[str]:
    fields = _run_git(repo, "rev-list", "--parents", "-n", "1", commit).stdout.decode("ascii").split()
    return fields[1:]


def _first_parent(repo: Path, commit: str) -> str | None:
    parents = _parents(repo, commit)
    return parents[0] if parents else None


def _tree(repo: Path, commit: str | None) -> dict[str, _TreeEntry]:
    if commit is None:
        return {}
    output = _run_git(repo, "ls-tree", "-r", "-z", "--full-tree", commit).stdout
    result: dict[str, _TreeEntry] = {}
    for record in output.split(b"\0"):
        if not record:
            continue
        try:
            header, raw_path = record.split(b"\t", 1)
            mode, object_type, object_id = header.decode("ascii").split()
        except (ValueError, UnicodeDecodeError) as exc:
            raise ValueError("Could not parse the committed Git tree") from exc
        if object_type == "blob":
            result[_decode_path(raw_path)] = _TreeEntry(mode=mode, object_id=object_id)
    return result


def _changed_files(repo: Path, parent: str | None, commit: str) -> list[_ChangedFile]:
    old = parent or EMPTY_TREE
    output = _run_git(
        repo,
        "diff",
        "--no-ext-diff",
        "--no-textconv",
        "--name-status",
        "-z",
        "--find-renames",
        old,
        commit,
        "--",
    ).stdout
    tokens = output.split(b"\0")
    changes: list[_ChangedFile] = []
    index = 0
    while index < len(tokens) and tokens[index]:
        status_token = tokens[index]
        index += 1
        embedded_path: bytes | None = None
        if b"\t" in status_token:
            status_token, embedded_path = status_token.split(b"\t", 1)
        status = status_token.decode("ascii")
        kind = status[:1]
        if kind in {"R", "C"}:
            if embedded_path is not None:
                old_raw = embedded_path
            else:
                old_raw = tokens[index]
                index += 1
            new_raw = tokens[index]
            index += 1
            changes.append(_ChangedFile(status=kind, old_path=_decode_path(old_raw), new_path=_decode_path(new_raw)))
        else:
            if embedded_path is not None:
                raw_path = embedded_path
            else:
                raw_path = tokens[index]
                index += 1
            path = _decode_path(raw_path)
            changes.append(
                _ChangedFile(
                    status=kind,
                    old_path=None if kind == "A" else path,
                    new_path=None if kind == "D" else path,
                )
            )
    return changes


def _blob_content(repo: Path, entry: _TreeEntry | None) -> tuple[bytes | None, str | None]:
    if entry is None:
        return None, None
    if not entry.mode.startswith("100"):
        return None, "not_regular"
    size = int(_run_git(repo, "cat-file", "-s", entry.object_id).stdout.decode("ascii"))
    if size > MAX_TEXT_BYTES:
        return None, "too_large"
    content = _run_git(repo, "cat-file", "blob", entry.object_id).stdout
    if b"\0" in content:
        return None, "binary"
    return content, None


def _lines(content: bytes | None) -> list[bytes]:
    """Split exactly as Git does: LF is the only line separator."""

    if not content:
        return []
    pieces = content.split(b"\n")
    result = [piece + b"\n" for piece in pieces[:-1]]
    if pieces[-1]:
        result.append(pieces[-1])
    return result


def _eligible_edits(
    connection: sqlite3.Connection,
    path: str,
    parent: str | None,
    worktree_id: str,
) -> list[_Edit]:
    if parent is None:
        base_clause = "COALESCE(e.base_commit, s.base_commit) IS NULL"
        parameters: tuple[object, ...] = (path, worktree_id)
    else:
        base_clause = "COALESCE(e.base_commit, s.base_commit) = ?"
        parameters = (path, worktree_id, parent)
    rows = connection.execute(
        f"""
        SELECT e.id, e.session_id, e.before_content, e.after_content, e.tool_use_id
        FROM edits AS e
        JOIN sessions AS s ON s.id = e.session_id
        WHERE e.path = ?
          AND s.worktree_id = ?
          AND {base_clause}
          AND s.ended_at IS NOT NULL
        ORDER BY e.id
        """,
        parameters,
    ).fetchall()
    return [
        _Edit(
            edit_id=row["id"],
            session_id=row["session_id"],
            before=bytes(row["before_content"]) if row["before_content"] is not None else None,
            after=bytes(row["after_content"]) if row["after_content"] is not None else None,
            tool_use_id=row["tool_use_id"],
        )
        for row in rows
    ]


def _exact_chain(edits: list[_Edit], target: bytes | None) -> list[_Edit]:
    """Find the latest backwards-connected chain ending at exact target bytes."""

    final = next((edit for edit in reversed(edits) if edit.after == target), None)
    if final is None:
        return []
    reverse_chain = [final]
    cursor = final
    used = {final.edit_id}
    while True:
        predecessor = next(
            (
                edit
                for edit in reversed(edits)
                if edit.edit_id < cursor.edit_id
                and edit.edit_id not in used
                and edit.after == cursor.before
            ),
            None,
        )
        if predecessor is None:
            break
        reverse_chain.append(predecessor)
        used.add(predecessor.edit_id)
        cursor = predecessor
    return list(reversed(reverse_chain))


def _initial_origins(parent: bytes | None, observed: bytes | None) -> list[int | None]:
    parent_lines = _lines(parent)
    observed_lines = _lines(observed)
    origins: list[int | None] = [None] * len(observed_lines)
    matcher = SequenceMatcher(None, parent_lines, observed_lines, autojunk=False)
    for operation, old_start, old_end, new_start, new_end in matcher.get_opcodes():
        if operation == "equal":
            for offset in range(new_end - new_start):
                origins[new_start + offset] = old_start + offset
    return origins


def _reconcile(parent: bytes | None, target: bytes | None, chain: list[_Edit]) -> _Reconciled:
    target_lines = _lines(target)
    if not chain:
        return _Reconciled(
            owners=[None] * len(target_lines), tool_use_ids=[None] * len(target_lines),
            removal_owners={}, session_revisions=[],
        )

    state_content = chain[0].before
    state_lines = _lines(state_content)
    owners: list[str | None] = [None] * len(state_lines)
    tool_use_ids: list[str | None] = [None] * len(state_lines)
    origins = _initial_origins(parent, state_content)
    removal_owners: dict[int, str] = {}
    session_revisions: list[dict[str, object]] = []

    for edit in chain:
        if edit.before != state_content:
            raise ValueError("Internal attribution chain is not exact")
        after_lines = _lines(edit.after)
        next_owners: list[str | None] = [None] * len(after_lines)
        next_tool_use_ids: list[str | None] = [None] * len(after_lines)
        next_origins: list[int | None] = [None] * len(after_lines)
        matcher = SequenceMatcher(None, state_lines, after_lines, autojunk=False)
        for operation, old_start, old_end, new_start, new_end in matcher.get_opcodes():
            if operation == "equal":
                next_owners[new_start:new_end] = owners[old_start:old_end]
                next_tool_use_ids[new_start:new_end] = tool_use_ids[old_start:old_end]
                next_origins[new_start:new_end] = origins[old_start:old_end]
                continue
            if operation in {"delete", "replace"}:
                # These are observed removal events, not a claim that each old
                # line has a one-to-one descendant in the replacement. Only
                # owners established inside this exact chain are eligible.
                removed_by_owner: dict[str, int] = {}
                for owner in owners[old_start:old_end]:
                    if owner is not None and owner != edit.session_id:
                        removed_by_owner[owner] = removed_by_owner.get(owner, 0) + 1
                session_revisions.extend(
                    {
                        "from_session_id": owner,
                        "to_session_id": edit.session_id,
                        "removed_lines": count,
                        "kind": operation,
                    }
                    for owner, count in sorted(removed_by_owner.items())
                )
                for origin in origins[old_start:old_end]:
                    if origin is not None:
                        removal_owners[origin] = edit.session_id
            if operation in {"insert", "replace"}:
                next_owners[new_start:new_end] = [edit.session_id] * (new_end - new_start)
                next_tool_use_ids[new_start:new_end] = [edit.tool_use_id] * (new_end - new_start)
        state_content = edit.after
        state_lines = after_lines
        owners = next_owners
        tool_use_ids = next_tool_use_ids
        origins = next_origins

    if state_content != target:
        raise ValueError("Internal attribution chain does not end at the committed blob")
    return _Reconciled(
        owners=owners, tool_use_ids=tool_use_ids, removal_owners=removal_owners,
        session_revisions=session_revisions,
    )


def _changed_line_opcodes(old: bytes | None, new: bytes | None) -> list[tuple[str, int, int, int, int]]:
    return SequenceMatcher(None, _lines(old), _lines(new), autojunk=False).get_opcodes()


def _owner_ranges(
    changed_indices: Iterable[int],
    owners: list[str | None],
    tool_use_ids: list[str | None],
) -> list[dict[str, object]]:
    """Group changed lines by the session, and the tool call, that wrote them.

    A range names its tool call only when the edit that produced it recorded
    one, so a note stays readable by every consumer that predates the key.
    """

    ranges: list[dict[str, object]] = []
    for index in changed_indices:
        owner = owners[index]
        if owner is None:
            continue
        tool_use_id = tool_use_ids[index]
        line_number = index + 1
        last = ranges[-1] if ranges else None
        if (
            last is not None and last["session_id"] == owner
            and last.get("tool_use_id") == tool_use_id and last["end"] == line_number - 1
        ):
            last["end"] = line_number
        else:
            item: dict[str, object] = {"start": line_number, "end": line_number, "session_id": owner}
            if tool_use_id is not None:
                item["tool_use_id"] = tool_use_id
            ranges.append(item)
    return ranges


_BLAME_HEADER = re.compile(rb"^([0-9a-f]{40,64}) (\d+) (\d+)(?: (\d+))?$")


def _decode_blame_path(raw_path: bytes) -> str:
    value = _decode_path(raw_path)
    if value.startswith('"') and value.endswith('"'):
        try:
            decoded = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return value
        if isinstance(decoded, str):
            return decoded
    return value


def _blame(repo: Path, commit: str, path: str) -> dict[int, tuple[str, str, int]]:
    output = _run_git(
        repo,
        "-c",
        "core.quotePath=false",
        "--literal-pathspecs",
        "blame",
        "--no-textconv",
        "--line-porcelain",
        "--root",
        commit,
        "--",
        path,
    ).stdout
    result: dict[int, tuple[str, str, int]] = {}
    # Git's porcelain records are LF-delimited. Bare CR and vertical-tab bytes
    # are valid file content and must not be mistaken for record boundaries.
    records = output.split(b"\n")
    index = 0
    while index < len(records):
        match = _BLAME_HEADER.match(records[index])
        if match is None:
            index += 1
            continue
        original_commit = match.group(1).decode("ascii")
        original_line = int(match.group(2))
        final_line = int(match.group(3))
        original_path = path
        index += 1
        while index < len(records):
            record = records[index]
            index += 1
            if record.startswith(b"filename "):
                original_path = _decode_blame_path(record[len(b"filename ") :])
            if record.startswith(b"\t"):
                break
        result[final_line - 1] = (original_commit, original_path, original_line)
    return result


class _HistoricalNotes:
    def __init__(self, repo: Path):
        self.repo = repo
        self.notes: dict[str, dict[str, object] | None] = {}
        self.session_metadata: dict[str, dict[str, object]] = {}

    def _load(self, commit: str) -> dict[str, object] | None:
        if commit in self.notes:
            return self.notes[commit]
        raw_note = _show_note(self.repo, commit)
        if raw_note is None:
            self.notes[commit] = None
            return None
        try:
            note = json.loads(raw_note)
        except (TypeError, json.JSONDecodeError):
            self.notes[commit] = None
            return None
        if not isinstance(note, dict) or note.get("version") != 1 or note.get("commit") != commit:
            self.notes[commit] = None
            return None
        self.notes[commit] = note
        sessions = note.get("sessions", [])
        if isinstance(sessions, list):
            for session in sessions:
                if isinstance(session, dict) and isinstance(session.get("id"), str):
                    self.session_metadata.setdefault(session["id"], session)
        return note

    def owner(self, commit: str, path: str, line: int) -> str | None:
        note = self._load(commit)
        if note is None:
            return None
        files = note.get("files", [])
        if not isinstance(files, list):
            return None
        for file_record in files:
            if not isinstance(file_record, dict) or file_record.get("path") != path:
                continue
            ranges = file_record.get("ranges", [])
            if not isinstance(ranges, list):
                return None
            for line_range in ranges:
                if not isinstance(line_range, dict):
                    continue
                start = line_range.get("start")
                end = line_range.get("end")
                owner = line_range.get("session_id")
                if isinstance(start, int) and isinstance(end, int) and start <= line <= end and isinstance(owner, str):
                    return owner
        return None


def _coalesce_revisions(points: list[_RevisionPoint]) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for point in points:
        if (
            result
            and result[-1]["from_commit"] == point.from_commit
            and result[-1]["from_path"] == point.from_path
            and result[-1]["from_session_id"] == point.from_session_id
            and result[-1]["to_session_id"] == point.to_session_id
            and result[-1]["from_end"] == point.from_line - 1
        ):
            result[-1]["from_end"] = point.from_line
        else:
            result.append(
                {
                    "from_commit": point.from_commit,
                    "from_path": point.from_path,
                    "from_start": point.from_line,
                    "from_end": point.from_line,
                    "from_session_id": point.from_session_id,
                    "to_session_id": point.to_session_id,
                }
            )
    return result


_SESSION_FIELDS = (
    "id",
    "feature",
    "model",
    "harness",
    "actor_kind",
    "source_session_id",
    "label_source",
    "cost_usd",
    "cost_source",
    "started_at",
    "ended_at",
)
_SESSION_OPTIONAL_FIELDS: dict[str, object] = {
    "native_session_id": None,
    "native_turn_id": None,
    "native_agent_id": None,
    "native_parent_session_id": None,
    "feature_source": None,
    "task_id": None,
    "membership_source": "legacy",
    "role": "implementation",
    "summary": None,
    "parent_session_id": None,
    "token_count": None,
    "token_source": None,
    "usage_includes_children": 0,
    "exit_code": None,
    "outcome": None,
    "harness_id": None,
    "harness_version": None,
    "provider": None,
    "model_source": None,
    "harness_source": None,
    "integration_mode": None,
    # Workflow facets. A note written before them simply lacks the keys, and
    # the defaults below keep an older note parsable without rewriting it.
    "agent_type": None,
    "launch_mode": None,
    "session_source": None,
    "permission_mode": None,
    "effort_level": None,
    "turn_count": 0,
    "prompt_count": 0,
    "interrupt_count": 0,
    "compaction_count": 0,
    "model_switch_count": 0,
    "tool_call_count": 0,
    "duration_ms": None,
    "activity_truncated": 0,
}


def _session_records(
    connection: sqlite3.Connection,
    session_ids: set[str],
    historical: _HistoricalNotes,
) -> list[dict[str, object]]:
    if not session_ids:
        return []
    placeholders = ",".join("?" for _ in session_ids)
    local_rows = connection.execute(
        f"SELECT rowid AS ledger_order, * FROM sessions WHERE id IN ({placeholders}) ORDER BY started_at, ledger_order",
        tuple(sorted(session_ids)),
    ).fetchall()
    records: list[dict[str, object]] = []
    seen: set[str] = set()
    for row in local_rows:
        records.append(
            {
                **{field: row[field] for field in _SESSION_FIELDS},
                **{field: row[field] for field in _SESSION_OPTIONAL_FIELDS},
            }
        )
        seen.add(row["id"])
    for session_id in sorted(session_ids - seen):
        source = historical.session_metadata.get(session_id)
        if source is None:
            continue
        # Notes written before actor-kind capture contain only AI sessions.
        source = {"actor_kind": "ai", "source_session_id": None, **source}
        if any(field not in source for field in _SESSION_FIELDS):
            continue
        records.append(
            {
                **{field: source[field] for field in _SESSION_FIELDS},
                **{
                    field: source.get(field, default)
                    for field, default in _SESSION_OPTIONAL_FIELDS.items()
                },
            }
        )
    return records


def _contributing_depths(
    connection: sqlite3.Connection, session_ids: set[str]
) -> dict[str, int]:
    """Return how deep each contributing session already sits in its own tree.

    A profile nests an agent under its parent whenever the note names both, so
    an orchestrator and the subagent beside it that each own a line are already
    two levels apart before the walk below adds one more.
    """

    if not session_ids:
        return {}
    placeholders = ",".join("?" for _ in session_ids)
    parents = {
        str(row["id"]): row["parent_session_id"]
        for row in connection.execute(
            f"SELECT id, parent_session_id FROM sessions WHERE id IN ({placeholders})",
            tuple(sorted(session_ids)),
        )
    }
    depths: dict[str, int] = {}
    for session_id in session_ids:
        depth = 1
        walked = {session_id}
        parent = parents.get(session_id)
        while (
            isinstance(parent, str)
            and parent in session_ids
            and parent not in walked
        ):
            walked.add(parent)
            depth += 1
            parent = parents.get(parent)
        depths[session_id] = depth
    return depths


def _descendant_sessions(
    connection: sqlite3.Connection, session_ids: set[str]
) -> tuple[set[str], int]:
    """Return the sessions the contributing ones launched, and the count cut.

    A subagent that only read the repository owns no line, so it is not a
    contributing session and never will be. It still worked the task: the
    orchestrator that read its report wrote the lines it found. A note that
    named two agents where three worked would describe the wrong work, so the
    walk below follows ``parent_session_id`` down from every contributing
    session and the note names what it finds.

    The walk takes one generation at a time, oldest first, and stops at two
    limits: the number of agents a snapshot republishes, and the depth one
    republishes them at. A session past either point is simply absent from the
    note, exactly as an agent this project never recorded is. Cutting the depth
    here is what keeps ``build_snapshot`` from refusing the whole note, and
    with it the pull request footer of that push. The returned count is of the
    agents the depth limit cut. The profile has no field for an omitted count,
    so ``record_commit`` reports it as a warning instead.
    """

    found: set[str] = set()
    depths = _contributing_depths(connection, session_ids)
    frontier = set(session_ids)
    seen = set(session_ids)
    omitted = 0
    while frontier:
        remaining = MAX_WORKFLOW_SESSIONS - len(session_ids | found)
        if remaining <= 0:
            break
        placeholders = ",".join("?" for _ in frontier)
        rows = connection.execute(
            f"""
            SELECT id, parent_session_id FROM sessions
            WHERE parent_session_id IN ({placeholders})
            ORDER BY started_at, id
            """,
            tuple(sorted(frontier)),
        ).fetchall()
        # The order is the ledger's own, so the set a cap cuts is the same set
        # on every clone that recorded the same work. ``seen`` closes a cycle
        # that a repaired parent link could otherwise open.
        children: list[tuple[str, int]] = []
        for row in rows:
            child = str(row["id"])
            if child in seen:
                continue
            depth = depths.get(str(row["parent_session_id"]), 1) + 1
            if depth > MAX_WORKFLOW_DEPTH:
                omitted += 1
                continue
            children.append((child, depth))
        if not children:
            break
        children = children[:remaining]
        for child, depth in children:
            depths[child] = depth
        seen.update(child for child, _ in children)
        found.update(child for child, _ in children)
        frontier = {child for child, _ in children}
    return found, omitted


def _workflow_profile(
    connection: sqlite3.Connection,
    records: list[dict[str, object]],
    session_ids: set[str],
    participants: set[str],
) -> dict[str, object] | None:
    """Return the workflow profile of the agents that worked on a commit.

    ``session_ids`` names the sessions that contributed a line of the commit.
    ``participants`` adds the sessions those launched, which shaped the work
    without owning a line of it.

    The profile carries counts, agent types, models, roles, launch modes, and
    the repo-relative paths of the instruction files each agent loaded. No
    locator of a tool call, no hash, and no text reaches it, because this
    object is published in a Git note.
    """

    sessions = [record for record in records if record["id"] in participants]
    if not sessions:
        return None
    placeholders = ",".join("?" for _ in participants)
    ordered = tuple(sorted(participants))
    activity: dict[str, dict[str, object]] = {}
    for row in connection.execute(
        f"""
        SELECT session_id, tool_class, COUNT(*) AS calls
        FROM tool_calls WHERE session_id IN ({placeholders})
        GROUP BY session_id, tool_class
        """,
        ordered,
    ):
        entry = activity.setdefault(row["session_id"], {})
        entry.setdefault("tool_calls", {})[row["tool_class"]] = row["calls"]
    for row in connection.execute(
        f"""
        SELECT DISTINCT session_id, locator, memory_type
        FROM context_loads
        WHERE kind = 'instruction_file' AND locator IS NOT NULL
          AND session_id IN ({placeholders})
        ORDER BY session_id, locator, memory_type
        """,
        ordered,
    ):
        entry = activity.setdefault(row["session_id"], {})
        entry.setdefault("instruction_files", []).append(
            {
                "kind": "instruction_file",
                "locator": row["locator"],
                "memory_type": row["memory_type"],
            }
        )
    # Only the contributing sessions are known to have worked by their lines.
    # A session this walk added answers the same question the profile already
    # asks of every other agent: it counts when it recorded a turn, a prompt,
    # or a tool call, and not when its harness only loaded its instructions.
    return build_profile(sessions, activity=activity, edited=session_ids)


def _note_task_ids(note: dict[str, object]) -> set[str]:
    result: set[str] = set()
    sessions = note.get("sessions", [])
    if not isinstance(sessions, list):
        return result
    for session in sessions:
        if isinstance(session, dict) and isinstance(session.get("task_id"), str):
            result.add(session["task_id"])
    return result


def _show_note(repo: Path, commit: str) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(repo), "notes", f"--ref={NOTES_REF}", "show", commit],
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
        raise ValueError(f"Existing attribution note for {commit} is not UTF-8") from exc


def _validated_note(raw_note: str, commit: str) -> dict[str, object]:
    try:
        note = json.loads(raw_note)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Existing note for {commit} is not a valid attribution note") from exc
    if not isinstance(note, dict) or note.get("version") != 1 or note.get("commit") != commit:
        raise ValueError(f"Existing note for {commit} conflicts with attribution note version 1")
    return note


def _summary(note: dict[str, object], *, note_written: bool) -> dict[str, object]:
    attributed_lines = 0
    added_lines = 0
    files = note.get("files", [])
    if not isinstance(files, list):
        raise ValueError("Joyride note has invalid files data")
    for file_record in files:
        if not isinstance(file_record, dict) or not isinstance(file_record.get("added_lines"), int):
            raise ValueError("Joyride note has invalid file data")
        added_lines += file_record["added_lines"]
        ranges = file_record.get("ranges", [])
        if not isinstance(ranges, list):
            raise ValueError("Joyride note has invalid range data")
        for line_range in ranges:
            if not isinstance(line_range, dict):
                raise ValueError("Joyride note has invalid range data")
            start, end = line_range.get("start"), line_range.get("end")
            if not isinstance(start, int) or not isinstance(end, int) or end < start:
                raise ValueError("Joyride note has invalid line bounds")
            attributed_lines += end - start + 1
    if attributed_lines > added_lines:
        raise ValueError("Joyride note attributes more lines than the commit added")
    return {
        "commit": note["commit"],
        "attributed_lines": attributed_lines,
        "unknown_lines": added_lines - attributed_lines,
        "note_written": note_written,
        # A note this run did not write recorded nothing to report.
        "warnings": [],
    }


def _write_note(repo: Path, commit: str, note_json: str) -> bool:
    encoded_note = note_json.encode("utf-8")
    if len(encoded_note) > MAX_NOTE_BYTES:
        raise ValueError(
            f"Joyride note exceeds the {MAX_NOTE_BYTES}-byte limit. "
            "Joyride did not write it."
        )

    # Linked worktrees share this ref. Serialize ref updates across all of them,
    # independently of their per-worktree capture locks.
    lock_path = git_common_dir(repo) / "attribution" / "notes.lock"
    lock_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with lock_path.open("a+b") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            return _write_note_locked(repo, commit, note_json)
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _write_note_locked(repo: Path, commit: str, note_json: str) -> bool:
    encoded_note = note_json.encode("utf-8")
    result = subprocess.run(
        ["git", "-C", str(repo), "notes", f"--ref={NOTES_REF}", "add", "-F", "-", commit],
        input=encoded_note,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=system_subprocess_environment(),
        check=False,
    )
    if result.returncode == 0:
        return True
    raced_note = _show_note(repo, commit)
    if raced_note == note_json:
        return False
    message = result.stderr.decode("utf-8", errors="replace").strip()
    raise ValueError(message or f"Could not write attribution note for {commit}")


def _cache_record(
    connection: sqlite3.Connection,
    commit: str,
    recorded_at: str,
    note_json: str,
) -> None:
    existing = connection.execute(
        "SELECT note_json FROM recorded_commits WHERE commit_sha = ?", (commit,)
    ).fetchone()
    if existing is not None:
        if existing["note_json"] != note_json:
            raise ValueError(f"Ledger record for {commit} conflicts with its immutable Git note")
        return
    connection.execute(
        "INSERT INTO recorded_commits(commit_sha, recorded_at, note_json) VALUES (?, ?, ?)",
        (commit, recorded_at, note_json),
    )
    connection.commit()


def record_commit(repo: RepoPath, commit: str = "HEAD") -> dict[str, object]:
    """Record conservative line attribution for an existing Git commit."""

    root = repository_root(repo)
    commit_sha = _resolve_commit(root, commit)
    with _worktree_lock(root, wait_seconds=RECORD_LOCK_WAIT_SECONDS):
        return _record_commit_locked(root, commit_sha)


def _record_commit_locked(root: Path, commit_sha: str) -> dict[str, object]:
    """Record a resolved commit while the caller holds the worktree lock."""

    connection = open_db(root)
    try:
        worktree_id = str(git_dir(root))
        active_hook = connection.execute(
            """
            SELECT 1 FROM hook_units
            WHERE worktree_id = ? AND active = 1
            UNION ALL
            SELECT 1 FROM hook_captures
            WHERE worktree_id = ?
              AND status IN ('pending', 'contaminated', 'limited', 'imported')
            LIMIT 1
            """,
            (worktree_id, worktree_id),
        ).fetchone()
        if active_hook is not None:
            raise ValueError(
                "Joyride did not record the commit because a native hook "
                "capture is active. Finish the capture or run `joyride recover`."
            )

        existing_note_json = _show_note(root, commit_sha)
        cached = connection.execute(
            "SELECT recorded_at, note_json FROM recorded_commits WHERE commit_sha = ?",
            (commit_sha,),
        ).fetchone()

        if existing_note_json is not None:
            existing_note = _validated_note(existing_note_json, commit_sha)
            if cached is not None and cached["note_json"] != existing_note_json:
                raise ValueError(f"Ledger record for {commit_sha} conflicts with its immutable Git note")
            _cache_record(
                connection,
                commit_sha,
                str(existing_note.get("recorded_at", _utc_now())),
                existing_note_json,
            )
            task_ids = _note_task_ids(existing_note)
            if task_ids:
                from .task_notes import sync_task_ids

                sync_task_ids(root, task_ids, commit_sha)
            return _summary(existing_note, note_written=False)

        if cached is not None:
            cached_note = _validated_note(cached["note_json"], commit_sha)
            note_written = _write_note(root, commit_sha, cached["note_json"])
            task_ids = _note_task_ids(cached_note)
            if task_ids:
                from .task_notes import sync_task_ids

                sync_task_ids(root, task_ids, commit_sha)
            return _summary(cached_note, note_written=note_written)

        parents = _parents(root, commit_sha)
        if len(parents) > 1:
            # The first-parent diff of a merge contains every line inherited
            # from the merged branch. Reusing edit evidence here would credit
            # those lines a second time. Conflict-resolution changes therefore
            # remain unknown; the report counts only lines new to every parent.
            recorded_at = _utc_now()
            note: dict[str, object] = {
                "version": 1,
                "commit": commit_sha,
                "recorded_at": recorded_at,
                "sessions": [],
                "files": [],
                "revisions": [],
            }
            note_json = json.dumps(note, separators=(",", ":"), ensure_ascii=False)
            note_written = _write_note(root, commit_sha, note_json)
            _cache_record(connection, commit_sha, recorded_at, note_json)
            return {
                "commit": commit_sha,
                "attributed_lines": 0,
                "unknown_lines": 0,
                "note_written": note_written,
                "warnings": [],
            }

        parent = parents[0] if parents else None
        parent_tree = _tree(root, parent)
        commit_tree = _tree(root, commit_sha)
        historical = _HistoricalNotes(root)
        files: list[dict[str, object]] = []
        revision_points: list[_RevisionPoint] = []
        session_revisions: list[dict[str, object]] = []
        attributed_lines = 0
        unknown_lines = 0
        matched_session_ids: set[str] = set()
        for change in _changed_files(root, parent, commit_sha):
            old_entry = parent_tree.get(change.old_path) if change.old_path is not None else None
            new_entry = commit_tree.get(change.new_path) if change.new_path is not None else None
            old_content, old_skip = _blob_content(root, old_entry)
            new_content, new_skip = _blob_content(root, new_entry)
            if old_skip is not None or new_skip is not None:
                continue

            ledger_path = change.new_path or change.old_path
            if ledger_path is None:
                continue
            edits = _eligible_edits(connection, ledger_path, parent, worktree_id)
            chain = _exact_chain(edits, new_content)
            matched_session_ids.update(edit.session_id for edit in chain)
            reconciled = _reconcile(old_content, new_content, chain)
            session_revisions.extend(
                {"path": ledger_path, **revision} for revision in reconciled.session_revisions
            )
            opcodes = _changed_line_opcodes(old_content, new_content)
            changed_new_indices: list[int] = []
            for operation, _old_start, _old_end, new_start, new_end in opcodes:
                if operation in {"insert", "replace"}:
                    changed_new_indices.extend(range(new_start, new_end))

            ranges = _owner_ranges(
                changed_new_indices, reconciled.owners, reconciled.tool_use_ids
            )
            file_attributed = sum(int(item["end"]) - int(item["start"]) + 1 for item in ranges)
            file_added = len(changed_new_indices)
            attributed_lines += file_attributed
            unknown_lines += file_added - file_attributed
            if file_added and change.new_path is not None:
                files.append({"path": change.new_path, "added_lines": file_added, "ranges": ranges})

            if parent is None or change.old_path is None:
                continue
            blame = _blame(root, parent, change.old_path)
            for operation, old_start, old_end, new_start, new_end in opcodes:
                if operation not in {"delete", "replace"}:
                    continue
                final_owners = reconciled.owners[new_start:new_end]
                unambiguous_final_owner: str | None = None
                if final_owners and final_owners[0] is not None and all(
                    owner == final_owners[0] for owner in final_owners
                ):
                    unambiguous_final_owner = final_owners[0]
                for old_index in range(old_start, old_end):
                    origin = blame.get(old_index)
                    if origin is None:
                        raise ValueError(f"Could not trace original line {old_index + 1} of {change.old_path}")
                    original_commit, original_path, original_line = origin
                    from_owner = historical.owner(original_commit, original_path, original_line)
                    if operation == "replace" and unambiguous_final_owner is not None:
                        to_owner = unambiguous_final_owner
                    else:
                        to_owner = reconciled.removal_owners.get(old_index)
                    revision_points.append(
                        _RevisionPoint(
                            from_commit=original_commit,
                            from_path=original_path,
                            from_line=original_line,
                            from_session_id=from_owner,
                            to_session_id=to_owner,
                        )
                    )

        revisions = _coalesce_revisions(revision_points)
        contributing_sessions: set[str] = set(matched_session_ids)
        for file_record in files:
            for line_range in file_record["ranges"]:
                contributing_sessions.add(str(line_range["session_id"]))
        for revision in revisions:
            if revision["to_session_id"] is not None:
                contributing_sessions.add(str(revision["to_session_id"]))
        needed_sessions: set[str] = set(contributing_sessions)
        for revision in revisions:
            if revision["from_session_id"] is not None:
                needed_sessions.add(str(revision["from_session_id"]))
            if revision["to_session_id"] is not None:
                needed_sessions.add(str(revision["to_session_id"]))

        # A subagent that owns no line of this commit is not a contributing
        # session, and the lines of the commit still belong to the sessions
        # that wrote them. The note records it beside them, because it is part
        # of how the work was done.
        descendants, deeper = _descendant_sessions(
            connection, contributing_sessions
        )

        recorded_at = _utc_now()
        session_records = _session_records(
            connection, needed_sessions | descendants, historical
        )
        profile = _workflow_profile(
            connection,
            session_records,
            contributing_sessions,
            contributing_sessions | descendants,
        )
        note: dict[str, object] = {
            "version": 1,
            "commit": commit_sha,
            "recorded_at": recorded_at,
            "sessions": session_records,
            # A note written before the profile simply lacks the key, and a
            # commit with no contributing session has no workflow to describe.
            **({"workflow": profile} if profile is not None else {}),
            "contributing_session_ids": sorted(contributing_sessions),
            "files": files,
            "revisions": revisions,
            "session_revisions": session_revisions,
        }
        # A snapshot republishes a workflow object only to a bounded depth, so
        # the walk above stops there. The reader is told what that cost rather
        # than losing the whole note to a refused snapshot.
        warnings = (
            [
                "The agent tree of this commit nests deeper than "
                f"{MAX_WORKFLOW_DEPTH} levels; {deeper} subagent(s) at that "
                "boundary, and anything they launched, are not named in the note."
            ]
            if deeper
            else []
        )
        note_json = json.dumps(note, separators=(",", ":"), ensure_ascii=False)
        note_written = _write_note(root, commit_sha, note_json)
        _cache_record(connection, commit_sha, recorded_at, note_json)
        from .task_notes import sync_task_ids
        from .tasks import task_for_session_ids

        task_ids = task_for_session_ids(connection, needed_sessions)
        if task_ids:
            sync_task_ids(root, task_ids, commit_sha)
        return {
            "commit": commit_sha,
            "attributed_lines": attributed_lines,
            "unknown_lines": unknown_lines,
            "note_written": note_written,
            "warnings": warnings,
        }
    finally:
        connection.close()
