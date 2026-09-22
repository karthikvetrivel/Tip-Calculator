"""Compute intended structured edits without executing tools or changing files.

Only exact, bounded text edits are accepted. This deliberately supports less
than the tools themselves: fuzzy patches and arbitrary shell commands cannot
provide dependable edit evidence. Callers must still compare the expected
result with the files after the tool runs.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import stat
import subprocess
from typing import Iterable

from .capture import MAX_TEXT_BYTES
from .hook_events import classify_tool
from .runtime import system_subprocess_environment


MAX_TARGET_FILES = 64
MAX_TOTAL_BYTES = 8 * 1024 * 1024


def _valid_parts(path: Path) -> bool:
    return bool(path.parts) and not any(part in {"..", ""} or part.casefold() == ".git" for part in path.parts)


def _relative_path(value: object, cwd: Path, repo: Path) -> str | None:
    if not isinstance(value, str) or not value or "\0" in value:
        return None
    path = Path(value)
    if not _valid_parts(path) or not cwd.is_absolute():
        return None
    path = path if path.is_absolute() else cwd / path
    if not _valid_parts(path):
        return None
    root = repo.resolve()
    # Canonicalize operating-system aliases above the repository (for example
    # macOS /var -> /private/var). Once we enter the repository, retain the raw
    # components so read_targets can open them without following any symlink.
    component = Path(path.anchor)
    entered = component == root
    relative_parts: list[str] = []
    for part in path.parts[1:]:
        if entered:
            relative_parts.append(part)
        else:
            component = (component / part).resolve()
            entered = component == root
    if not entered or not relative_parts:
        return None
    return Path(*relative_parts).as_posix()


def _read_one(root_fd: int, relative: str) -> bytes | None:
    """Open each component without following links, including parent links."""
    parent_fd = os.dup(root_fd)
    try:
        parts = Path(relative).parts
        for part in parts[:-1]:
            try:
                child_fd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent_fd)
            except FileNotFoundError:
                return None
            os.close(parent_fd)
            parent_fd = child_fd
        try:
            file_fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent_fd)
        except FileNotFoundError:
            return None
        with os.fdopen(file_fd, "rb") as stream:
            metadata = os.fstat(stream.fileno())
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > MAX_TEXT_BYTES:
                raise ValueError("Target is not a bounded regular file")
            content = stream.read(MAX_TEXT_BYTES + 1)
        _validate_content(content)
        return content
    finally:
        os.close(parent_fd)


def _validate_content(content: bytes) -> None:
    if len(content) > MAX_TEXT_BYTES or b"\0" in content:
        raise ValueError("Target is not bounded text")
    content.decode("utf-8")


def read_targets(repo: Path, paths: Iterable[str]) -> dict[str, bytes | None] | None:
    """Read eligible repository-relative paths; absent paths map to None.

    Any unsafe path, ignored untracked file, non-text file, read failure, or
    size limit invalidates the whole set. Tracked files remain eligible even
    when an ignore rule matches them. No working-tree data is written.
    """
    try:
        unique: list[str] = []
        for value in paths:
            if not isinstance(value, str) or not value or "\0" in value:
                return None
            path = Path(value)
            if path.is_absolute() or not _valid_parts(path) or path.as_posix() != value:
                return None
            if value not in unique:
                unique.append(value)
                if len(unique) > MAX_TARGET_FILES:
                    return None
        if not unique:
            return {}
        ignored = subprocess.run(
            ["git", "-C", str(repo), "check-ignore", "-z", "--stdin"],
            input=b"\0".join(value.encode("utf-8") for value in unique) + b"\0",
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env=system_subprocess_environment(), check=False,
        )
        if ignored.returncode != 1 or ignored.stdout:
            return None
        root_fd = os.open(repo, os.O_RDONLY | os.O_DIRECTORY)
        try:
            result: dict[str, bytes | None] = {}
            total = 0
            for relative in unique:
                content = _read_one(root_fd, relative)
                total += len(content) if content is not None else 0
                if total > MAX_TOTAL_BYTES:
                    return None
                result[relative] = content
            return result
        finally:
            os.close(root_fd)
    except (OSError, ValueError, TypeError, UnicodeError, RuntimeError):
        return None


@dataclass
class _Chunk:
    anchor: str | None
    old: list[str]
    new: list[str]
    eof: bool = False


@dataclass
class _Operation:
    kind: str
    path: str
    destination: str | None = None
    content: bytes | None = None
    chunks: list[_Chunk] | None = None


def _parse_patch(patch: object, cwd: Path, repo: Path) -> list[_Operation] | None:
    if not isinstance(patch, str) or "\0" in patch or len(patch.encode("utf-8")) > MAX_TOTAL_BYTES:
        return None
    # Only the canonical LF format is accepted. The real tool also accepts
    # lenient wrappers and alternate line-ending modes which we cannot infer.
    if "\r" in patch:
        return None
    lines = patch.removesuffix("\n").split("\n")
    if len(lines) < 3 or lines[0] != "*** Begin Patch" or lines[-1] != "*** End Patch":
        return None
    operations: list[_Operation] = []
    targets: set[str] = set()
    index = 1
    while index < len(lines) - 1:
        header = lines[index]
        kind = next((kind for kind in ("Add", "Delete", "Update") if header.startswith(f"*** {kind} File: ")), None)
        if kind is None:
            return None
        path = _relative_path(header[len(f"*** {kind} File: "):], cwd, repo)
        if path is None or path in targets:
            return None
        targets.add(path)
        index += 1
        operation = _Operation(kind.lower(), path)
        if kind == "Add":
            content: list[str] = []
            while index < len(lines) - 1 and lines[index].startswith("+"):
                content.append(lines[index][1:])
                index += 1
            if not content:
                return None
            operation.content = ("\n".join(content) + "\n").encode("utf-8")
        elif kind == "Update":
            if lines[index].startswith("*** Move to: "):
                destination = _relative_path(lines[index][len("*** Move to: "):], cwd, repo)
                if destination is None or destination in targets:
                    return None
                operation.destination = destination
                targets.add(destination)
                index += 1
            chunks: list[_Chunk] = []
            while index < len(lines) - 1 and not lines[index].startswith("*** "):
                anchor = None
                if lines[index] == "@@" or lines[index].startswith("@@ "):
                    anchor = lines[index][3:] if lines[index].startswith("@@ ") else None
                    index += 1
                elif chunks:
                    return None
                old: list[str] = []
                new: list[str] = []
                count = 0
                while index < len(lines) - 1 and lines[index] and lines[index][0] in " +-":
                    line = lines[index]
                    if line[0] in " -":
                        old.append(line[1:])
                    if line[0] in " +":
                        new.append(line[1:])
                    index += 1
                    count += 1
                if not count:
                    return None
                eof = lines[index] == "*** End of File"
                if eof:
                    index += 1
                chunks.append(_Chunk(anchor, old, new, eof))
                if eof:
                    break
            if not chunks:
                return None
            operation.chunks = chunks
        operations.append(operation)
        if len(targets) > MAX_TARGET_FILES:
            return None
    return operations or None


def _unique_match(lines: list[str], pattern: list[str], start: int, eof: bool = False) -> int | None:
    if eof:
        position = len(lines) - len(pattern)
        return position if position >= start and lines[position:] == pattern else None
    # A linear search avoids quadratic work on large repetitive generated files.
    prefix = [0] * len(pattern)
    matched = 0
    for index in range(1, len(pattern)):
        while matched and pattern[index] != pattern[matched]:
            matched = prefix[matched - 1]
        if pattern[index] == pattern[matched]:
            matched += 1
        prefix[index] = matched
    match = None
    matched = 0
    for index in range(start, len(lines)):
        while matched and lines[index] != pattern[matched]:
            matched = prefix[matched - 1]
        if lines[index] == pattern[matched]:
            matched += 1
        if matched == len(pattern):
            if match is not None:
                return None
            match = index - len(pattern) + 1
            matched = prefix[matched - 1]
    return match


def _updated_content(content: bytes, chunks: list[_Chunk]) -> bytes | None:
    # This reproduces the tool's default LF reconstruction, but requires exact
    # unique matches instead of its whitespace and punctuation fallbacks.
    text = content.decode("utf-8")
    if "\r" in text:
        return None
    lines = text.split("\n")
    if lines[-1] == "":
        lines.pop()
    replacements: list[tuple[int, int, list[str]]] = []
    start = 0
    for chunk in chunks:
        if chunk.anchor is not None:
            anchor = _unique_match(lines, [chunk.anchor], start)
            if anchor is None:
                return None
            start = anchor + 1
        if not chunk.old:
            position = len(lines) - 1 if lines and lines[-1] == "" else len(lines)
            if position < start:
                return None
        else:
            position = _unique_match(lines, chunk.old, start, chunk.eof)
            if position is None:
                return None
        if replacements and position < replacements[-1][0] + replacements[-1][1]:
            return None
        replacements.append((position, len(chunk.old), chunk.new))
        start = position + len(chunk.old)
    for position, count, replacement in reversed(replacements):
        lines[position:position + count] = replacement
    if not lines or lines[-1] != "":
        lines.append("")
    return "\n".join(lines).encode("utf-8")


def expected_edit(
    provider: str, tool_name: str, tool_input: dict, cwd: Path, repo: Path,
) -> tuple[dict[str, bytes | None], dict[str, bytes | None]] | None:
    """Return targeted before/expected bytes for a supported native edit.

    Invalid, ambiguous, unsupported, or unsafe input returns None. This never
    invokes the tool or writes source files, directories, or the Git index.
    """
    try:
        if not isinstance(tool_input, dict):
            return None
        kind = classify_tool(provider, tool_name)
        if kind not in {"write", "edit", "patch"}:
            return None
        if kind == "patch":
            operations = _parse_patch(tool_input.get("command"), cwd, repo)
            if operations is None:
                return None
            paths = [path for operation in operations for path in (operation.path, operation.destination) if path is not None]
            before = read_targets(repo, paths)
            if before is None:
                return None
            expected = dict(before)
            for operation in operations:
                original = before[operation.path]
                if operation.kind == "add":
                    expected[operation.path] = operation.content
                elif operation.kind == "delete":
                    if original is None:
                        return None
                    expected[operation.path] = None
                else:
                    if original is None:
                        return None
                    updated = _updated_content(original, operation.chunks or [])
                    if updated is None:
                        return None
                    expected[operation.path] = updated
                    if operation.destination is not None:
                        expected[operation.path] = None
                        expected[operation.destination] = updated
        else:
            path = _relative_path(tool_input.get("file_path"), cwd, repo)
            if path is None:
                return None
            before = read_targets(repo, [path])
            if before is None:
                return None
            if kind == "write":
                content = tool_input.get("content")
                if not isinstance(content, str):
                    return None
                expected = {path: content.encode("utf-8")}
            else:
                old = tool_input.get("old_string")
                new = tool_input.get("new_string")
                replace_all = tool_input.get("replace_all", False)
                if not isinstance(old, str) or not old or not isinstance(new, str) or not isinstance(replace_all, bool):
                    return None
                original = before[path]
                if original is None:
                    return None
                content = original.decode("utf-8")
                occurrences = content.count(old)
                if occurrences == 0 or (occurrences != 1 and not replace_all):
                    return None
                expected = {path: content.replace(old, new, -1 if replace_all else 1).encode("utf-8")}
        total = 0
        for values in (before, expected):
            for content in values.values():
                if content is not None:
                    _validate_content(content)
                    total += len(content)
                    if total > MAX_TOTAL_BYTES:
                        return None
        return before, expected
    except (OSError, ValueError, TypeError, UnicodeError, RuntimeError):
        return None
