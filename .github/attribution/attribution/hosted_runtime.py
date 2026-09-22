"""Build and verify the exact Python runtime vendored by hosted setup."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat


WORKFLOW_PATH = ".github/workflows/attribution-footer.yml"
RUNTIME_PREFIX = ".github/attribution/attribution/"
RUNTIME_MANIFEST_PATH = ".github/attribution/runtime-manifest.json"
MAX_SETUP_FILES = 256
MAX_SETUP_BYTES = 2 * 1024 * 1024
MAX_SETUP_FILE_BYTES = 2 * 1024 * 1024
_MAX_SCANNED_ENTRIES = 4096
_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9_.-]+$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CACHE_DIRECTORIES = frozenset(
    {"__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache"}
)
_REQUIRED_MODULES = frozenset(
    {
        "__init__.py",
        "adapters/__init__.py",
        "adapters/base.py",
        "github_footer.py",
        "hosted_artifact.py",
        "narrative_summary.py",
    "traces.py",
        "runtime.py",
        "tool_capture.py",
    }
)


class HostedRuntimeError(ValueError):
    """The local source or vendored runtime is unsafe or incomplete."""


def _safe_component(name: str) -> None:
    if (
        name in {"", ".", ".."}
        or _SAFE_COMPONENT.fullmatch(name) is None
        or any(ord(character) < 32 or ord(character) == 127 for character in name)
    ):
        raise HostedRuntimeError("The hosted runtime contains an unsafe path.")


def _read_regular_file(path: Path) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise HostedRuntimeError("The hosted runtime cannot be read safely.") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise HostedRuntimeError("The hosted runtime contains an unsafe file.")
        if before.st_size > MAX_SETUP_FILE_BYTES:
            raise HostedRuntimeError("A hosted runtime file exceeds its size limit.")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            content = stream.read(MAX_SETUP_FILE_BYTES + 1)
        after = os.fstat(descriptor)
        if (
            len(content) > MAX_SETUP_FILE_BYTES
            or len(content) != before.st_size
            or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        ):
            raise HostedRuntimeError(
                "A hosted runtime file changed while it was being read."
            )
        return content
    finally:
        os.close(descriptor)


def _runtime_tree(
    package_root: Path,
    *,
    python_only: bool,
    exclude_caches: bool,
) -> tuple[dict[str, bytes], set[str]]:
    """Read one bounded tree without following a symbolic link."""

    package = Path(package_root)
    try:
        details = package.lstat()
        if stat.S_ISLNK(details.st_mode) or not stat.S_ISDIR(details.st_mode):
            raise HostedRuntimeError("The hosted runtime root is not a safe directory.")
        root = package.resolve(strict=True)
    except FileNotFoundError as exc:
        raise HostedRuntimeError("The hosted runtime source is unavailable.") from exc
    except (OSError, RuntimeError) as exc:
        raise HostedRuntimeError("The hosted runtime root cannot be inspected safely.") from exc

    files: dict[str, bytes] = {}
    directories: set[str] = set()
    total_bytes = 0
    scanned_entries = 0

    def visit(directory: Path, relative: PurePosixPath) -> None:
        nonlocal scanned_entries, total_bytes
        try:
            directory_details = directory.lstat()
            if stat.S_ISLNK(directory_details.st_mode) or not stat.S_ISDIR(
                directory_details.st_mode
            ):
                raise HostedRuntimeError(
                    "The hosted runtime contains an unsafe directory."
                )
            directory.resolve(strict=True).relative_to(root)
            entries = sorted(directory.iterdir(), key=lambda candidate: candidate.name)
        except HostedRuntimeError:
            raise
        except (OSError, RuntimeError, ValueError) as exc:
            raise HostedRuntimeError(
                "The hosted runtime cannot be inspected safely."
            ) from exc

        for entry in entries:
            scanned_entries += 1
            if scanned_entries > _MAX_SCANNED_ENTRIES:
                raise HostedRuntimeError("The hosted runtime contains too many entries.")
            _safe_component(entry.name)
            try:
                entry_details = entry.lstat()
            except OSError as exc:
                raise HostedRuntimeError(
                    "The hosted runtime cannot be inspected safely."
                ) from exc
            if stat.S_ISLNK(entry_details.st_mode):
                raise HostedRuntimeError("The hosted runtime contains a symbolic link.")
            child = relative / entry.name
            if len(child.parts) > 32 or len(child.as_posix()) > 4000:
                raise HostedRuntimeError("The hosted runtime contains an unsafe path.")
            if stat.S_ISDIR(entry_details.st_mode):
                if exclude_caches and entry.name in _CACHE_DIRECTORIES:
                    continue
                directories.add(child.as_posix())
                visit(entry, child)
                continue
            if not stat.S_ISREG(entry_details.st_mode):
                raise HostedRuntimeError("The hosted runtime contains an unsafe file.")
            if python_only and entry.suffix != ".py":
                continue
            if len(files) >= MAX_SETUP_FILES - 2:
                raise HostedRuntimeError("The hosted runtime contains too many files.")
            key = child.as_posix()
            content = _read_regular_file(entry)
            total_bytes += len(content)
            if total_bytes > MAX_SETUP_BYTES:
                raise HostedRuntimeError("The hosted runtime exceeds its size limit.")
            files[key] = content

    visit(root, PurePosixPath())
    return files, directories


def source_runtime_files(package_root: Path) -> dict[str, bytes]:
    """Return every intended Python module below the source package."""

    files, _directories = _runtime_tree(
        package_root, python_only=True, exclude_caches=True
    )
    return files


def installed_runtime_files(package_root: Path) -> dict[str, bytes]:
    """Read every file in an installed runtime for exact-set comparison."""

    files, _directories = installed_runtime_tree(package_root)
    return files


def installed_runtime_tree(package_root: Path) -> tuple[dict[str, bytes], set[str]]:
    """Read every installed file and directory for exact-tree comparison."""

    return _runtime_tree(package_root, python_only=False, exclude_caches=False)


def _runtime_path(value: object) -> str:
    if not isinstance(value, str) or not value.startswith(RUNTIME_PREFIX):
        raise HostedRuntimeError("The hosted runtime manifest contains an unsafe path.")
    relative = value.removeprefix(RUNTIME_PREFIX)
    parsed = PurePosixPath(relative)
    if (
        not parsed.parts
        or parsed.is_absolute()
        or ".." in parsed.parts
        or len(parsed.parts) > 32
        or len(relative) > 4000
    ):
        raise HostedRuntimeError("The hosted runtime manifest contains an unsafe path.")
    for part in parsed.parts:
        _safe_component(part)
    return value


def runtime_manifest_files(content: bytes) -> dict[str, str]:
    """Validate and return the files owned by one generated runtime manifest."""

    if not isinstance(content, bytes) or len(content) > MAX_SETUP_FILE_BYTES:
        raise HostedRuntimeError("The hosted runtime manifest exceeds its size limit.")

    def object_without_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise HostedRuntimeError(
                    "The hosted runtime manifest contains duplicate fields."
                )
            result[key] = value
        return result

    try:
        value = json.loads(content, object_pairs_hook=object_without_duplicates)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise HostedRuntimeError("The hosted runtime manifest is invalid.") from exc
    if (
        not isinstance(value, dict)
        or set(value) != {"version", "files"}
        or type(value.get("version")) is not int
        or value.get("version") != 1
        or not isinstance(value.get("files"), dict)
    ):
        raise HostedRuntimeError("The hosted runtime manifest is invalid.")
    raw_files = value["files"]
    if not raw_files or len(raw_files) > MAX_SETUP_FILES - 2:
        raise HostedRuntimeError("The hosted runtime manifest is invalid.")
    files: dict[str, str] = {}
    for raw_path, digest in raw_files.items():
        path = _runtime_path(raw_path)
        if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
            raise HostedRuntimeError("The hosted runtime manifest is invalid.")
        files[path] = digest
    return files


def _runtime_manifest(runtime: dict[str, bytes]) -> bytes:
    files = {
        RUNTIME_PREFIX + relative: hashlib.sha256(content).hexdigest()
        for relative, content in sorted(runtime.items())
    }
    return (
        json.dumps(
            {"version": 1, "files": files},
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def setup_files(package_root: Path, workflow: bytes) -> dict[str, bytes]:
    """Return the single bounded setup payload used by every installer."""

    if not isinstance(workflow, bytes) or not workflow:
        raise HostedRuntimeError("The hosted setup workflow is unavailable.")
    if len(workflow) > MAX_SETUP_FILE_BYTES:
        raise HostedRuntimeError("The hosted setup workflow exceeds its size limit.")
    runtime = source_runtime_files(package_root)
    if not _REQUIRED_MODULES.issubset(runtime):
        raise HostedRuntimeError("The hosted setup runtime is incomplete.")
    files = {WORKFLOW_PATH: workflow}
    files.update(
        {RUNTIME_PREFIX + relative: content for relative, content in runtime.items()}
    )
    files[RUNTIME_MANIFEST_PATH] = _runtime_manifest(runtime)
    if len(files) > MAX_SETUP_FILES:
        raise HostedRuntimeError("The hosted setup contains too many files.")
    if sum(len(content) for content in files.values()) > MAX_SETUP_BYTES:
        raise HostedRuntimeError("The hosted setup runtime exceeds its size limit.")
    return files


__all__ = [
    "HostedRuntimeError",
    "MAX_SETUP_BYTES",
    "MAX_SETUP_FILE_BYTES",
    "MAX_SETUP_FILES",
    "RUNTIME_PREFIX",
    "RUNTIME_MANIFEST_PATH",
    "WORKFLOW_PATH",
    "installed_runtime_files",
    "installed_runtime_tree",
    "runtime_manifest_files",
    "setup_files",
    "source_runtime_files",
]
