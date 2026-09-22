"""Read committed source, line ownership, and conservative revision evidence.

All paths are literal Git tree paths. No worktree files, external diff drivers,
text converters, or writable ledger connections are used by this module.
"""

from __future__ import annotations

from difflib import SequenceMatcher
import json
import os
from pathlib import Path
import re
import selectors
import subprocess
import time
from typing import Any, Mapping

from .report import (
    _BLAME_HEADER_RE,
    _MAX_NOTE_BYTES,
    _Warnings,
    _decode_git_path,
    _merge_session,
    _normalise_session,
    _parse_note,
    _read_demo_marker,
    telemetry_model,
)
from .runtime import system_subprocess_environment


MAX_FILE_BYTES = 512 * 1024
MAX_FILE_LINES = 10_000
MAX_FILES = 2_000
MAX_HISTORY_COMMITS = 100
MAX_HISTORY_EVENTS = 500
MAX_HISTORY_LINES = 40_000
MAX_TOTAL_BYTES = 16 * 1024 * 1024
MAX_SECONDS = 25
_OID = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")


def _validate_path(path: str) -> str:
    if (
        not isinstance(path, str)
        or not path
        or len(path) > 4096
        or "\x00" in path
        or path.startswith("/")
        or any(part in {"", ".", ".."} for part in path.split("/"))
    ):
        raise ValueError("Choose a relative, literal repository file path without traversal.")
    try:
        path.encode("utf-8")
    except UnicodeError as exc:
        raise ValueError("File paths must be valid UTF-8.") from exc
    return path


class _Limit(ValueError):
    pass


class _Reader:
    def __init__(
        self, repo: str | Path, target_ref: str,
        *, notes: list[dict[str, Any]] | None = None,
        usage: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> None:
        self.root = Path(repo).expanduser().resolve()
        self.deadline = time.monotonic() + MAX_SECONDS
        self.warnings = _Warnings()
        self.truncated = False
        self.total_bytes = 0
        self.sessions: dict[str, dict[str, Any]] = {}
        self.usage = usage or {}
        self.notes: dict[str, dict[str, Any] | None] = {}
        self.supplied_notes: dict[str, dict[str, Any]] | None = None
        self.snapshots: dict[tuple[str, str], list[dict[str, Any]]] = {}
        self.snapshot_exists: dict[tuple[str, str], bool] = {}
        self.metadata: dict[str, dict[str, Any]] = {}
        bare = self.git("rev-parse", "--is-bare-repository").decode().strip() == "true"
        if not bare:
            self.root = Path(self.git("rev-parse", "--show-toplevel").decode().strip()).resolve()
        self.target = self.resolve(target_ref)
        common = Path(self.git("rev-parse", "--git-common-dir").decode().strip())
        if not common.is_absolute():
            common = self.root / common
        # The shared Git directory holds the ledger that ``why`` reads beside
        # these commits. Nothing in this module writes to it.
        self.common_dir = common
        if notes is not None:
            if not isinstance(notes, list):
                raise ValueError("Shared attribution notes must be an array.")
            supplied: dict[str, dict[str, Any]] = {}
            for raw in notes:
                if not isinstance(raw, dict) or not isinstance(raw.get("commit"), str):
                    raise ValueError("Invalid shared attribution note.")
                commit = raw["commit"].lower()
                if commit in supplied:
                    raise ValueError("Duplicate shared attribution note.")
                encoded = json.dumps(raw, ensure_ascii=False, allow_nan=False)
                if len(encoded.encode("utf-8")) > _MAX_NOTE_BYTES:
                    raise ValueError("Shared attribution note exceeds the size limit.")
                supplied[commit] = _parse_note(encoded, commit, warnings=self.warnings)
            self.supplied_notes = supplied
        name = self.root.name[:-4] if bare and self.root.name.endswith(".git") else self.root.name
        self.repository = {
            "name": name,
            "path": str(self.root),
            "target_ref": target_ref,
            "target_commit": self.target,
            "example_data": _read_demo_marker(common, self.warnings),
        }

    def git(self, *args: str, check: bool = True) -> bytes:
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise _Limit("Code inspection reached its time limit.")
        try:
            process = subprocess.Popen(
                ["git", "--no-replace-objects", "--literal-pathspecs", "-c", "core.quotePath=false",
                 "-C", str(self.root), *args],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                env=system_subprocess_environment(),
            )
        except OSError as exc:
            raise ValueError(f"Could not run Git: {exc}") from exc
        stdout, stderr = bytearray(), bytearray()
        command_deadline = min(self.deadline, time.monotonic() + 10)
        try:
            with selectors.DefaultSelector() as selector:
                selector.register(process.stdout, selectors.EVENT_READ, stdout)
                selector.register(process.stderr, selectors.EVENT_READ, stderr)
                while selector.get_map():
                    remaining = command_deadline - time.monotonic()
                    if remaining <= 0:
                        raise _Limit("Code inspection reached its time limit.")
                    for key, _ in selector.select(remaining):
                        chunk = os.read(key.fd, 64 * 1024)
                        if not chunk:
                            selector.unregister(key.fileobj)
                        else:
                            key.data.extend(chunk)
                            if len(stdout) + len(stderr) > MAX_TOTAL_BYTES:
                                raise _Limit("Git output exceeds the code inspection size limit.")
            process.wait(timeout=max(0.01, command_deadline - time.monotonic()))
        except subprocess.TimeoutExpired as exc:
            raise _Limit("Code inspection reached its time limit.") from exc
        finally:
            if process.poll() is None:
                process.kill()
            process.wait()
            process.stdout.close()
            process.stderr.close()
        if process.returncode:
            if not check:
                return b""
            raise ValueError(stderr.decode("utf-8", "replace").strip() or "Git query failed.")
        return bytes(stdout)

    def resolve(self, ref: str) -> str:
        if not isinstance(ref, str) or not ref or len(ref) > 1024 or "\x00" in ref:
            raise ValueError("Choose a valid Git commit or reference.")
        resolved = self.git("rev-parse", "--verify", "--quiet", "--end-of-options", f"{ref}^{{commit}}", check=False).decode().strip()
        if not _OID.fullmatch(resolved):
            raise ValueError(f"Git reference {ref!r} does not identify a commit.")
        return resolved

    def info(self, commit: str) -> dict[str, Any]:
        if commit not in self.metadata:
            fields = self.git("show", "--no-patch", "--format=%H%x00%P%x00%s%x00%cI", commit).decode("utf-8", "replace").rstrip("\n").split("\x00", 3)
            parents = fields[1].split()
            self.metadata[commit] = {
                "sha": fields[0], "short_sha": fields[0][:7],
                "parent_sha": parents[0] if parents else None,
                "subject": fields[2], "committed_at": fields[3],
            }
        return self.metadata[commit]

    def note(self, commit: str) -> dict[str, Any] | None:
        if commit in self.notes:
            return self.notes[commit]
        self.notes[commit] = None
        if self.supplied_notes is not None:
            note = self.supplied_notes.get(commit)
            if note is None:
                return None
        else:
            oid = self.git("notes", "--ref=refs/notes/attribution", "list", commit, check=False).decode().strip()
            if not _OID.fullmatch(oid):
                return None
            size = int(self.git("cat-file", "-s", oid))
            if size > _MAX_NOTE_BYTES:
                self.warnings.add(f"Ignored oversized attribution note for {commit[:7]}.")
                return None
            try:
                note = _parse_note(
                    self.git("cat-file", "blob", oid).decode("utf-8", "replace"),
                    commit,
                    warnings=self.warnings,
                )
            except ValueError as exc:
                self.warnings.add(f"Ignored attribution note for {commit[:7]}: {exc}.")
                return None
        if note is None:
            return None
        for raw in note["sessions"]:
            if not isinstance(raw, dict):
                self.warnings.add(f"Ignored malformed session metadata in {commit[:7]}.")
                continue
            session = _normalise_session(raw, source=f"note {commit[:7]}", warnings=self.warnings)
            if session:
                # Resolve the same session label as the summary before computing
                # revision arrows. Do not rewrite the immutable input note.
                named = telemetry_model(session, self.usage.get(session["id"]))
                if named is not None:
                    session["model"] = named
                _merge_session(self.sessions, session, self.warnings)
        self.notes[commit] = note
        return note

    def owner_span(self, origin: dict[str, Any]) -> dict[str, Any] | None:
        """Return the note range that owns one line: its session and tool call."""
        note = self.note(origin["commit"])
        if note:
            for file in note["files"]:
                if file["path"] != origin["path"]:
                    continue
                for span in file["ranges"]:
                    if span["start"] <= origin["line"] <= span["end"]:
                        if span["session_id"] in self.sessions:
                            return span
                        self.warnings.add(f"A range in {origin['commit'][:7]} references missing session metadata.")
        return None

    def owner(self, origin: dict[str, Any]) -> str | None:
        span = self.owner_span(origin)
        return span["session_id"] if span else None

    def tree(self, commit: str, path: str | None = None) -> list[dict[str, Any]]:
        args = ["ls-tree", "-r", "-l", "-z", "--full-tree", commit]
        if path is not None:
            args += ["--", path]
        result = []
        for row in self.git(*args).split(b"\0"):
            if not row:
                continue
            header, name = row.split(b"\t", 1)
            mode, kind, oid, size = header.decode("ascii").split()
            try:
                decoded = name.decode("utf-8")
            except UnicodeError:
                self.warnings.add("Skipped a file whose path is not UTF-8.")
                continue
            if path is not None and decoded != path:
                continue
            result.append({"path": decoded, "mode": mode, "kind": kind, "oid": oid, "size": int(size) if size != "-" else 0})
        return result

    def snapshot(self, commit: str | None, path: str | None) -> list[dict[str, Any]]:
        if commit is None or path is None:
            return []
        key = (commit, path)
        if key in self.snapshots:
            return [dict(line, history_ids=[]) for line in self.snapshots[key]]
        entries = self.tree(commit, path)
        self.snapshot_exists[key] = bool(entries)
        if not entries:
            self.snapshots[key] = []
            return []
        entry = entries[0]
        if entry["mode"] not in {"100644", "100755"} or entry["kind"] != "blob":
            raise ValueError("Code inspection supports regular committed files only (no symlinks or submodules).")
        if entry["size"] > MAX_FILE_BYTES:
            raise ValueError(f"File exceeds the {MAX_FILE_BYTES // 1024} KiB code inspection limit.")
        self.total_bytes += entry["size"]
        if self.total_bytes > MAX_TOTAL_BYTES:
            raise _Limit("File history exceeds the code inspection size limit.")
        content = self.git("cat-file", "blob", entry["oid"])
        if b"\0" in content:
            raise ValueError("Binary files cannot be shown in the code inspector.")
        try:
            decoded = content.decode("utf-8")
        except UnicodeError as exc:
            raise ValueError("Code inspection requires a UTF-8 text file.") from exc
        texts = decoded.split("\n")
        if not texts[-1]:
            texts.pop()
        if len(texts) > MAX_FILE_LINES:
            raise ValueError(f"File exceeds the {MAX_FILE_LINES:,} line code inspection limit.")
        if not texts:
            self.snapshots[key] = []
            return []
        output = self.git("blame", "--no-textconv", "--root", "--line-porcelain", commit, "--", path).decode("utf-8", "replace")
        rows = output.split("\n")
        index = 0
        lines: dict[int, dict[str, Any]] = {}
        while index < len(rows) and rows[index]:
            header = _BLAME_HEADER_RE.fullmatch(rows[index])
            if not header:
                raise ValueError("Could not parse Git line history.")
            number = int(header.group(3))
            origin = {"commit": header.group(1), "path": path, "line": int(header.group(2))}
            index += 1
            while index < len(rows) and not rows[index].startswith("\t"):
                if rows[index].startswith("filename "):
                    origin["path"] = _decode_git_path(rows[index][9:])
                index += 1
            if index >= len(rows) or number < 1 or number > len(texts):
                raise ValueError("Git line history does not match the committed file.")
            lines[number] = {"number": number, "content": texts[number - 1],
                             "has_newline": number < len(texts) or decoded.endswith("\n"),
                             "session_id": self.owner(origin), "origin": origin, "history_ids": []}
            index += 1
        if len(lines) != len(texts):
            raise ValueError("Git line history is incomplete.")
        self.snapshots[key] = [lines[number] for number in range(1, len(texts) + 1)]
        return [dict(line, history_ids=[]) for line in self.snapshots[key]]

    def commits(self, revision: str, path: str) -> list[str]:
        output = self.git("log", "--follow", "--topo-order", "--format=%H", f"--max-count={MAX_HISTORY_COMMITS + 1}", revision, "--", path)
        commits = output.decode().splitlines()
        if len(commits) > MAX_HISTORY_COMMITS:
            commits = commits[:MAX_HISTORY_COMMITS]
            self.truncated = True
            self.warnings.add(f"History is limited to the most recent {MAX_HISTORY_COMMITS} file commits.")
        return commits

    def change(self, commit: str, path: str) -> tuple[str | None, str | None]:
        parent = self.info(commit)["parent_sha"]
        args = ["diff-tree", "--no-commit-id", "--root", "-r", "-M", "--name-status", "-z", "--no-ext-diff", "--no-textconv"]
        args += [parent, commit] if parent else [commit]
        fields = self.git(*args).split(b"\0")
        index = 0
        while index < len(fields) and fields[index]:
            status = fields[index].decode("ascii")
            old = fields[index + 1].decode("utf-8", "replace")
            index += 2
            new = old
            if status.startswith(("R", "C")):
                new = fields[index].decode("utf-8", "replace")
                index += 1
            if new == path:
                return (None if status == "A" else old, None if status == "D" else new)
        return path, path

    def mark_partial(self, exc: Exception) -> None:
        self.truncated = True
        self.warnings.add(f"Some earlier history is unavailable: {exc}")


def list_code_files(repo: str | Path, target_ref: str = "main") -> dict[str, Any]:
    """List regular files in a resolved commit, without opening the ledger."""
    reader = _Reader(repo, target_ref)
    files = []
    for entry in reader.tree(reader.target):
        if entry["mode"] not in {"100644", "100755"}:
            continue
        if entry["size"] > MAX_FILE_BYTES:
            reader.warnings.add("Files larger than 512 KiB are omitted from the code inspector.")
            reader.truncated = True
            continue
        if len(files) == MAX_FILES:
            reader.warnings.add(f"The file list is limited to {MAX_FILES:,} files.")
            reader.truncated = True
            break
        files.append({"path": entry["path"], "size": entry["size"]})
    return {"repository": reader.repository, "files": files, "warnings": reader.warnings.items, "truncated": reader.truncated}


def _immutable_commit(reader: _Reader, value: str, label: str) -> str:
    """Resolve an object only when the caller supplied its complete object ID."""
    if not isinstance(value, str) or not _OID.fullmatch(value):
        raise ValueError(f"The PR {label} must be a complete lowercase Git object ID.")
    resolved = reader.resolve(value)
    if resolved != value:
        raise ValueError(f"The PR {label} does not identify an immutable commit.")
    return resolved


def _pr_scope(
    repo: str | Path,
    base_sha: str,
    head_sha: str,
    *,
    owner: str,
    repository: str,
    number: int,
    target_ref: str,
    notes: list[dict[str, Any]] | None = None,
) -> tuple[_Reader, dict[str, Any]]:
    """Return the literal file set for a frozen PR comparison.

    The supplied base is intentionally not trusted as the comparison point: a
    PR can have been opened before its base branch moved.  Git's merge-base is
    the only base used to choose files or construct the visible diff.
    """
    reader = _Reader(repo, target_ref, notes=notes)
    supplied_base = _immutable_commit(reader, base_sha, "base SHA")
    head = _immutable_commit(reader, head_sha, "head SHA")
    merge_base = reader.git("merge-base", supplied_base, head, check=False).decode().strip()
    if not _OID.fullmatch(merge_base):
        raise ValueError("The PR base and head do not have a usable merge base.")

    raw = reader.git(
        "diff", "--name-status", "-z", "-M", "--no-ext-diff", "--no-textconv",
        merge_base, head,
    ).split(b"\0")
    head_entries = {entry["path"]: entry for entry in reader.tree(head)}
    base_entries = {entry["path"]: entry for entry in reader.tree(merge_base)}
    files: list[dict[str, Any]] = []
    index = 0
    while index < len(raw) and raw[index]:
        try:
            status = raw[index].decode("ascii")
        except UnicodeError as exc:
            raise ValueError("Git returned an invalid PR file status.") from exc
        index += 1
        if index >= len(raw):
            raise ValueError("Git returned an incomplete PR file list.")
        try:
            first = raw[index].decode("utf-8")
        except UnicodeError as exc:
            raise ValueError("PR file paths must be valid UTF-8.") from exc
        index += 1
        old_path, new_path = first, first
        if status.startswith(("R", "C")):
            if index >= len(raw):
                raise ValueError("Git returned an incomplete renamed PR file.")
            try:
                new_path = raw[index].decode("utf-8")
            except UnicodeError as exc:
                raise ValueError("PR file paths must be valid UTF-8.") from exc
            index += 1
        if status.startswith("A"):
            old_path = None
        elif status.startswith("D"):
            new_path = None
        elif not status[:1] in {"M", "R", "C", "T"}:
            raise ValueError("Git returned an unsupported PR file status.")
        display_path = new_path or old_path
        if display_path is None:
            continue
        _validate_path(display_path)
        entry = head_entries.get(new_path or "")
        files.append({
            "path": display_path, "status": status, "base_path": old_path,
            "head_path": new_path, "size": entry["size"] if entry else 0,
            "base_blob_sha": base_entries.get(old_path or "", {}).get("oid"),
            "head_blob_sha": entry.get("oid") if entry else None,
        })
    scope = {
        "owner": owner, "repository": repository, "number": number,
        "requested_base_sha": supplied_base, "merge_base_sha": merge_base,
        "head_sha": head, "configured_target_sha": reader.target,
        # A local repository cannot authoritatively know GitHub's newest PR
        # head.  Keep the immutable comparison frozen, but never call a normal
        # open PR stale merely because its branch differs from local main.
        "freshness": "unknown", "stale": None,
        "files": files,
    }
    return reader, scope


def list_pr_code_files(
    repo: str | Path,
    base_sha: str,
    head_sha: str,
    *,
    owner: str,
    repository: str,
    number: int,
    target_ref: str = "main",
) -> dict[str, Any]:
    """List exactly the files changed from PR merge-base to pinned head."""
    reader, scope = _pr_scope(
        repo, base_sha, head_sha, owner=owner, repository=repository,
        number=number, target_ref=target_ref,
    )
    repo_info = dict(reader.repository)
    repo_info.update({"target_ref": f"PR #{number} pinned head", "target_commit": scope["head_sha"]})
    return {
        "repository": repo_info, "scope": {key: value for key, value in scope.items() if key != "files"},
        "files": scope["files"], "warnings": reader.warnings.items, "truncated": False,
    }


def _event(reader: _Reader, commit: str, old_path: str | None, new_path: str | None,
           before: list[dict[str, Any]], after: list[dict[str, Any]], kind: str, index: int) -> dict[str, Any]:
    note = reader.note(commit)
    revisions = note["revisions"] if note else []
    from_ids = sorted({line["session_id"] for line in before if line["session_id"]})
    to_ids = sorted({line["session_id"] for line in after if line["session_id"]})
    matches = []
    for line in before:
        origin = line["origin"]
        links = [link for link in revisions
                 if link["from_commit"] == origin["commit"]
                 and link["from_path"] == origin["path"]
                 and link["from_start"] <= origin["line"] <= link["from_end"]
                 and link["from_session_id"] == line["session_id"]
                 and (not after or link["to_session_id"] in to_ids)]
        matches.append(links)
    if before and not after:
        to_ids = sorted({link["to_session_id"] for links in matches for link in links if link["to_session_id"] in reader.sessions})
    all_known = all(line["session_id"] for line in before + after)
    integrated = any(line["origin"]["commit"] != commit for line in after)
    exact_pairs = {(line["session_id"], link["to_session_id"])
                   for line, links in zip(before, matches) for link in links
                   if link["to_session_id"] in reader.sessions}
    if (before and all(matches) and all_known and to_ids
            and all(len({link["to_session_id"] for link in links}) == 1 for links in matches)
            and set(to_ids) <= {pair[1] for pair in exact_pairs}):
        evidence = "recorded"
    elif not before and after and all_known:
        evidence = "recorded"
    elif before and from_ids and to_ids:
        evidence = "inferred"
    else:
        evidence = "unattributed"
    if integrated:
        # A merge can bring in code already authored on another parent. Its
        # first-parent diff is not evidence that a model revised that parent.
        evidence = "unattributed"
    if evidence == "recorded":
        cross_model = any(left is not None and reader.sessions[left]["model"] != reader.sessions[right]["model"]
                          for left, right in exact_pairs)
    else:
        from_models = {reader.sessions[owner]["model"] for owner in from_ids}
        to_models = {reader.sessions[owner]["model"] for owner in to_ids}
        cross_model = len(from_models) == len(to_models) == 1 and from_models != to_models
    if integrated:
        cross_model = False
    meta = reader.info(commit)
    return {
        "id": f"{commit}:{index}", "commit": commit, "short_sha": meta["short_sha"],
        "subject": meta["subject"], "committed_at": meta["committed_at"], "parent_sha": meta["parent_sha"],
        "before_path": old_path, "after_path": new_path,
        "before_start": before[0]["number"] if before else None,
        "before_end": before[-1]["number"] if before else None,
        "after_start": after[0]["number"] if after else None,
        "after_end": after[-1]["number"] if after else None,
        "before": before, "after": after, "from_session_ids": from_ids, "to_session_ids": to_ids,
        "cross_model": cross_model, "evidence": evidence, "kind": kind, "integration": integrated,
    }


def build_code_file(
    repo: str | Path, path: str, target_ref: str = "main",
    revision: str | None = None, *, notes: list[dict[str, Any]] | None = None,
    usage: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Inspect source at a reachable revision and trace replacement hunk history.

    ``path`` is the target's path, including when requesting an earlier revision.
    Ownership comes from blame origins, never the displayed line coordinates.
    Replacement relationships without exact note links are explicitly inferred.
    ``diff`` compares the selected commit with its first parent. Segment starts
    are one-based insertion anchors even on empty sides; a unified patch header
    subtracts one from an empty side's anchor. Line objects keep their actual
    one-based numbers and include final-newline state.
    """
    path = _validate_path(path)
    reader = _Reader(repo, target_ref, notes=notes, usage=usage)
    selected = reader.resolve(revision) if revision is not None else reader.target
    selected_info = dict(reader.info(selected))
    if selected != reader.target:
        # merge-base output is bounded and works for ancestors on merged branches.
        ancestor = reader.git("merge-base", reader.target, selected, check=False).decode().strip()
        if ancestor != selected:
            raise ValueError("The selected revision is outside the target's history.")
    selected_path = path
    if selected != reader.target:
        reached_boundary = False
        for commit in reader.commits(reader.target, path):
            # Trace backwards even when the same path existed earlier: that
            # earlier path may belong to an unrelated file replaced by a rename.
            ancestor = reader.git("merge-base", commit, selected, check=False).decode().strip()
            if ancestor == commit:
                reached_boundary = True
                break
            if ancestor != selected:
                continue
            old_path, new_path = reader.change(commit, selected_path)
            if old_path is None:
                raise ValueError("This file did not exist at the selected revision.")
            if old_path and old_path != new_path:
                selected_path = old_path
        if reader.truncated and not reached_boundary:
            raise ValueError("Cannot resolve this file's path within the bounded revision history.")
    entries = reader.tree(selected, selected_path)
    commits = reader.commits(selected, selected_path)
    if not entries and not commits:
        raise ValueError(f"File {path!r} does not exist in the selected revision's history.")
    current = reader.snapshot(selected, selected_path)
    selected_diff: dict[str, Any] = {
        "available": False,
        "reason": "The selected revision's parent comparison is unavailable within the code inspection limits.",
        "before_path": None, "after_path": selected_path if entries else None,
        "before_line_count": None, "after_line_count": len(current), "segments": [],
    }
    steps: list[tuple[str, str | None, str | None]] = []
    cursor = selected_path
    # A commit that does not change the file (and some merge commits) is absent
    # from --follow output. Still compare that exact revision with its parent,
    # using the same bounded snapshots and opcodes as the history below.
    for commit in dict.fromkeys([selected, *commits]):
        try:
            old_path, new_path = reader.change(commit, cursor)
            steps.append((commit, old_path, new_path))
            if commit == selected:
                selected_diff["before_path"] = old_path
                selected_diff["after_path"] = new_path
                if selected not in commits:
                    # A merge may add the file relative to its first parent
                    # while --follow still traces its authors on another one.
                    continue
            if old_path is None:
                break
            cursor = old_path
        except _Limit as exc:
            if commit == selected:
                selected_diff["reason"] = str(exc)
            reader.mark_partial(exc)
            break
    # Load newest commits first so any limit preserves recent evidence. Then
    # connect those bounded snapshots in topological order without further Git
    # calls; origin keys allow merged branch histories to meet at their merge.
    prepared = []
    event_count = 0
    history_lines = 0
    for commit, old_path, new_path in steps:
        try:
            parent = reader.info(commit)["parent_sha"]
            before = reader.snapshot(parent, old_path)
            after = reader.snapshot(commit, new_path)
            reader.note(commit)
            operations = SequenceMatcher(None, [(line["content"], line["has_newline"]) for line in before],
                                         [(line["content"], line["has_newline"]) for line in after], autojunk=False).get_opcodes()
            if time.monotonic() > reader.deadline:
                raise _Limit("Code inspection reached its time limit.")
            added_events = sum(operation != "equal" for operation, *_ in operations) + bool(old_path and new_path and old_path != new_path)
            added_lines = sum(b - a + d - c for operation, a, b, c, d in operations if operation != "equal")
            if event_count + added_events > MAX_HISTORY_EVENTS or history_lines + added_lines > MAX_HISTORY_LINES:
                raise _Limit("History exceeds the event or changed-line inspection limit.")
            event_count += added_events
            history_lines += added_lines
            prepared.append((commit, old_path, new_path, before, after, operations))
        except ValueError as exc:
            if commit == selected:
                selected_diff["reason"] = str(exc)
            reader.mark_partial(exc)
            if isinstance(exc, _Limit):
                break
    history: list[dict[str, Any]] = []
    histories: dict[tuple[str, str, int], list[str]] = {}
    for commit, old_path, new_path, before, after, operations in reversed(prepared):
        segments: list[dict[str, Any]] = []
        for line in before:
            origin = line["origin"]
            line["history_ids"] = list(histories.get((origin["commit"], origin["path"], origin["line"]), []))
        staged: list[dict[str, Any]] = []
        if old_path and new_path and old_path != new_path:
            staged.append(_event(reader, commit, old_path, new_path, [], [], "renamed", 0))
        for operation, a, b, c, d in operations:
            if operation == "equal":
                for previous, following in zip(before[a:b], after[c:d]):
                    following["history_ids"] = list(previous["history_ids"])
                if commit == selected:
                    segments.append({"kind": "context", "before_start": a + 1, "after_start": c + 1,
                                     "before": before[a:b], "after": after[c:d], "event_id": None})
                continue
            old_lines, new_lines = before[a:b], after[c:d]
            event = _event(reader, commit, old_path, new_path, old_lines, new_lines,
                           {"replace": "modified", "insert": "added", "delete": "deleted"}[operation], len(staged))
            prior = list(dict.fromkeys(event_id for line in old_lines for event_id in line["history_ids"]))
            for line in new_lines:
                origin = line["origin"]
                if origin["commit"] != commit:
                    line["history_ids"] = list(histories.get((origin["commit"], origin["path"], origin["line"]), []))
                else:
                    line["history_ids"] = [*prior, event["id"]]
            staged.append(event)
            if commit == selected:
                segments.append({"kind": "change", "before_start": a + 1, "after_start": c + 1,
                                 "before": old_lines, "after": new_lines, "event_id": event["id"]})
        if commit == selected:
            parent = selected_info["parent_sha"]
            selected_diff = {
                "available": True, "reason": None,
                "before_path": old_path if reader.snapshot_exists.get((parent, old_path), False) else None,
                "after_path": new_path if reader.snapshot_exists.get((commit, new_path), False) else None,
                "before_line_count": len(before), "after_line_count": len(after), "segments": segments,
            }
        history.extend(staged)
        for line in after:
            origin = line["origin"]
            histories[(origin["commit"], origin["path"], origin["line"])] = line["history_ids"]
    for line in current:
        origin = line["origin"]
        line["history_ids"] = histories.get((origin["commit"], origin["path"], origin["line"]), [])
    return {
        "repository": reader.repository, "path": path,
        "revision": {**selected_info, "path": selected_path},
        "sessions": reader.sessions, "lines": current,
        "diff": selected_diff,
        "history": list(reversed(history)), "warnings": reader.warnings.items,
        "truncated": reader.truncated,
    }


def build_pr_code_file(
    repo: str | Path,
    path: str,
    base_sha: str,
    head_sha: str,
    *,
    owner: str,
    repository: str,
    number: int,
    target_ref: str = "main",
    revision: str | None = None,
    notes: list[dict[str, Any]] | None = None,
    usage: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Inspect one file from the immutable PR file set.

    Revision history remains the normal conservative per-file history, while
    ``diff`` is deliberately replaced with the whole PR comparison: merge-base
    to the pinned head, never the head commit's first parent.
    """
    path = _validate_path(path)
    _, scope = _pr_scope(
        repo, base_sha, head_sha, owner=owner, repository=repository,
        number=number, target_ref=target_ref, notes=notes,
    )
    if revision is not None and revision != scope["head_sha"]:
        raise ValueError("PR file history is pinned to the immutable head SHA.")
    record = next((item for item in scope["files"] if item["path"] == path), None)
    if record is None:
        raise ValueError("This file is not part of the pinned PR diff.")
    current = build_code_file(repo, path, target_ref=scope["head_sha"], notes=notes, usage=usage)
    before_lines: list[dict[str, Any]] = []
    before_path = record["base_path"]
    before_error: str | None = None
    if before_path is not None:
        try:
            # The exact base path comes from Git's whole-PR comparison. Read
            # it directly: tracing backwards from head can misidentify a file
            # deleted and recreated at the same path, or hit the history cap.
            previous = build_code_file(
                repo, before_path, target_ref=scope["merge_base_sha"], notes=notes, usage=usage,
            )
            before_lines = previous["lines"]
            # Removed lines can belong to sessions absent from current source
            # or from the head's bounded history. Keep those labels and links.
            current["sessions"] = {**previous["sessions"], **current["sessions"]}
            known_events = {event["id"] for event in current["history"]}
            current["history"].extend(
                event for event in previous["history"] if event["id"] not in known_events
            )
            current["warnings"].extend(previous["warnings"])
            current["truncated"] = current["truncated"] or previous["truncated"]
        except ValueError as exc:
            # Missing baseline evidence must not become a fabricated all-added
            # diff. Current source and available history remain inspectable.
            before_error = f"The PR's earlier source is unavailable: {exc}"
            current["warnings"].append(before_error)
            current["truncated"] = True

    after_lines = current["lines"]
    operations = [] if before_error else SequenceMatcher(
        None,
        [(line["content"], line["has_newline"]) for line in before_lines],
        [(line["content"], line["has_newline"]) for line in after_lines],
        autojunk=False,
    ).get_opcodes()
    segments: list[dict[str, Any]] = []
    for operation, a, b, c, d in operations:
        segments.append({
            "kind": "context" if operation == "equal" else "change",
            "before_start": a + 1, "after_start": c + 1,
            "before": before_lines[a:b], "after": after_lines[c:d], "event_id": None,
        })
    current["diff"] = {
        "available": before_error is None, "reason": before_error,
        "before_path": before_path,
        "after_path": record["head_path"],
        "before_line_count": None if before_error else len(before_lines), "after_line_count": len(after_lines),
        "segments": segments,
        "comparison": "pr_merge_base_to_head",
        "merge_base_sha": scope["merge_base_sha"], "head_sha": scope["head_sha"],
    }
    current["scope"] = {key: value for key, value in scope.items() if key != "files"}
    current["repository"] = {
        **current["repository"], "target_ref": f"PR #{number} pinned head",
        "target_commit": scope["head_sha"],
    }
    return current
