"""Safe, repository-scoped installation for native attribution automation."""

from __future__ import annotations

from dataclasses import dataclass
import copy
import hashlib
import json
import os
from pathlib import Path
import shlex
import stat
import subprocess
import sys
import tempfile
from typing import Any

from .activity import SNAPSHOT_TOOL_NAMES
from .runtime import (
    RuntimeCommand,
    current_runtime,
    stored_runtime,
    system_subprocess_environment,
)
from .store import git_common_dir, git_dir, repository_root


# ``runtime_kind`` is an additive v1 field. Missing values are legacy Python
# installs and are migrated in place, so no destructive schema bump is needed.
_VERSION = 1
_MARKER = "# attribution-managed-v1"
# Manifests written before machine-level installs existed have no hook scope
# and own the native hook files inside their own checkout.
_HOOK_SCOPE_USER = "user"
_CURRENT_WORKTREE_ONLY_ENV = "_HARNESS_ATTRIBUTION_CURRENT_WORKTREE_ONLY"
_DISABLE_TELEMETRY_ENV = "HARNESS_ATTRIBUTION_DISABLE_TELEMETRY"
_METADATA_PUSH_GUARD = "ATTRIBUTION_METADATA_PUSH"
_CLAUDE_TELEMETRY_HEADER = "OTEL_EXPORTER_OTLP_HEADERS"
_CLAUDE_TELEMETRY_ENDPOINT = "OTEL_EXPORTER_OTLP_LOGS_ENDPOINT"
_CLAUDE_LEGACY_DEFAULT_ENDPOINT = "http://127.0.0.1:4318/v1/logs"
_CLAUDE_TRACKED_TELEMETRY_WARNING = (
    ".claude/settings.local.json is tracked by Git; Joyride did not write "
    "Claude telemetry credentials."
)
_NOTICE = (
    "Restart existing coding sessions so they load hooks. "
    "Review repo hooks in Codex when prompted."
)
_EPHEMERAL_RUNTIME_ERROR = (
    "This attribution runtime lives in a temporary tool cache, so the hooks "
    "would break when that cache is cleared. Install it persistently with "
    "uv tool install git+https://github.com/valite-ai/attribution-hosted "
    "(or pipx install with the same URL) and run the command again."
)
_MAX_CONFIG_BYTES = 2 * 1024 * 1024
_EXCLUDE_BLOCK = (
    b"# >>> attribution-managed-v1\n"
    b"/.codex/hooks.json\n"
    b"/.claude/settings.local.json\n"
    b"# <<< attribution-managed-v1"
)
# The subagent, prompt, interrupt, compaction, and instruction events open no
# capture and take no worktree snapshot, so subscribing to them costs one small
# database write. Claude Code has no interrupt event; Codex has one. Only
# Claude Code reports the instruction files it loaded. PreCompact changes no
# ledger row, so neither harness subscribes to it.
_EVENTS = {
    "codex": (
        "SessionStart",
        "UserPromptSubmit",
        "PreToolUse",
        "PostToolUse",
        "PostCompact",
        "SubagentStart",
        "SubagentStop",
        "Interrupt",
        "Stop",
    ),
    "claude-code": (
        "SessionStart",
        "UserPromptSubmit",
        "PreToolUse",
        "PostToolUse",
        "PostToolUseFailure",
        "PostModelSwitch",
        "InstructionsLoaded",
        "PostCompact",
        "SubagentStart",
        "SubagentStop",
        "Stop",
    ),
}
_TOOL_EVENTS = {"PreToolUse", "PostToolUse", "PostToolUseFailure"}
# Every completed tool call joins the activity ledger, so the post events match
# every tool. A pre-tool event exists to take the worktree snapshot that proves
# what a tool changed, so it matches only the tools that can change the tree.
# The receiver reads the same names, so a tool never reaches it for a snapshot
# it does not need, and never misses one it does.
_SNAPSHOT_TOOL_MATCHER = f"^({'|'.join(sorted(SNAPSHOT_TOOL_NAMES))})$"
_EVERY_TOOL_MATCHER = ".*"
_INTEGRATION_PATHS = {
    "codex": Path(".codex/hooks.json"),
    "claude-code": Path(".claude/settings.local.json"),
}
_STANDARD_GIT_HOOKS = {
    "applypatch-msg",
    "commit-msg",
    "fsmonitor-watchman",
    "p4-changelist",
    "p4-post-changelist",
    "p4-pre-submit",
    "p4-prepare-changelist",
    "post-applypatch",
    "post-checkout",
    "post-commit",
    "post-merge",
    "post-receive",
    "post-rewrite",
    "post-update",
    "pre-applypatch",
    "pre-auto-gc",
    "pre-commit",
    "pre-merge-commit",
    "pre-push",
    "pre-rebase",
    "pre-receive",
    "prepare-commit-msg",
    "proc-receive",
    "push-to-checkout",
    "reference-transaction",
    "sendemail-validate",
    "update",
}


@dataclass(frozen=True)
class _Repository:
    root: Path
    common_dir: Path
    git_dir: Path

    @property
    def state_dir(self) -> Path:
        return self.common_dir / "attribution"

    @property
    def manifest_path(self) -> Path:
        return self.state_dir / "install.json"

    @property
    def managed_hooks_path(self) -> Path:
        return self.state_dir / "hooks"

    @property
    def bootstrap_path(self) -> Path:
        return self.state_dir / "hook-bootstrap.py"


@dataclass(frozen=True)
class _LinkedWorktree:
    repository: _Repository
    branch: str | None


@dataclass(frozen=True)
class _Snapshot:
    exists: bool
    data: bytes | None = None
    mode: int | None = None
    atime_ns: int | None = None
    mtime_ns: int | None = None


class _Transaction:
    """Remember exact file preimages and restore them if an operation fails."""

    def __init__(self) -> None:
        self._snapshots: dict[Path, _Snapshot] = {}
        self._order: list[Path] = []
        self._created_directories: list[Path] = []

    def _capture(self, path: Path) -> None:
        if path in self._snapshots:
            return
        try:
            details = path.lstat()
        except FileNotFoundError:
            snapshot = _Snapshot(False)
        else:
            if stat.S_ISLNK(details.st_mode):
                raise ValueError(f"Refusing to replace symlink {path}.")
            if not stat.S_ISREG(details.st_mode):
                raise ValueError(f"Expected a regular file at {path}.")
            snapshot = _Snapshot(
                True,
                path.read_bytes(),
                stat.S_IMODE(details.st_mode),
                details.st_atime_ns,
                details.st_mtime_ns,
            )
        self._snapshots[path] = snapshot
        self._order.append(path)

    def _ensure_parent(self, parent: Path, mode: int = 0o700) -> None:
        missing: list[Path] = []
        cursor = parent
        while not cursor.exists():
            missing.append(cursor)
            if cursor.parent == cursor:
                break
            cursor = cursor.parent
        if cursor.is_symlink() or not cursor.is_dir():
            raise ValueError(f"Unsafe parent directory {cursor}.")
        for directory in reversed(missing):
            directory.mkdir(mode=mode)
            self._created_directories.append(directory)

    def write(
        self,
        path: Path,
        data: bytes,
        *,
        mode: int,
        times: tuple[int, int] | None = None,
    ) -> None:
        self._capture(path)
        self._ensure_parent(path.parent)
        _atomic_write(path, data, mode=mode, times=times)

    def delete(self, path: Path) -> None:
        self._capture(path)
        try:
            path.unlink()
        except FileNotFoundError:
            pass

    def rollback(self) -> None:
        failures: list[str] = []
        for path in reversed(self._order):
            snapshot = self._snapshots[path]
            try:
                if snapshot.exists:
                    assert snapshot.data is not None and snapshot.mode is not None
                    self._ensure_parent(path.parent)
                    _atomic_write(
                        path,
                        snapshot.data,
                        mode=snapshot.mode,
                        times=(snapshot.atime_ns or 0, snapshot.mtime_ns or 0),
                    )
                else:
                    try:
                        path.unlink()
                    except FileNotFoundError:
                        pass
            except OSError as exc:
                failures.append(f"{path}: {exc}")
        for directory in reversed(self._created_directories):
            try:
                directory.rmdir()
            except OSError:
                pass
        if failures:
            raise RuntimeError("Rollback could not restore " + "; ".join(failures))


def _atomic_write(
    path: Path,
    data: bytes,
    *,
    mode: int,
    times: tuple[int, int] | None = None,
) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.attribution-", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
        os.chmod(temporary, mode)
        if times is not None:
            os.utime(temporary, ns=times)
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _git(
    repo: str | Path,
    *args: str,
    check: bool = True,
) -> subprocess.CompletedProcess[bytes]:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), *args],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=system_subprocess_environment(),
            check=False,
            timeout=20,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ValueError(f"Could not run Git: {exc}") from exc
    if check and result.returncode:
        message = result.stderr.decode("utf-8", errors="replace").strip()
        raise ValueError(message or f"Git command failed with exit code {result.returncode}.")
    return result


def _repository(repo: str | Path) -> _Repository:
    return _Repository(
        root=repository_root(repo),
        common_dir=git_common_dir(repo),
        git_dir=git_dir(repo),
    )


def _discover_worktrees(
    repository: _Repository,
) -> tuple[list[_LinkedWorktree], list[str]]:
    """Return live linked worktrees that belong to the selected repository."""

    listed = _git(repository.root, "worktree", "list", "--porcelain", "-z")
    discovered: dict[Path, _LinkedWorktree] = {}
    skipped = 0
    for record in listed.stdout.split(b"\x00\x00"):
        if not record:
            continue
        fields = [field for field in record.split(b"\x00") if field]
        path_fields = [field[len(b"worktree ") :] for field in fields if field.startswith(b"worktree ")]
        if len(path_fields) != 1 or any(
            field == b"prunable" or field.startswith(b"prunable ") or field == b"bare"
            for field in fields
        ):
            skipped += 1
            continue
        path = Path(os.fsdecode(path_fields[0]))
        try:
            if not path.is_dir():
                skipped += 1
                continue
            candidate = _repository(path)
        except (OSError, ValueError):
            skipped += 1
            continue
        if candidate.common_dir != repository.common_dir:
            skipped += 1
            continue
        branch_fields = [
            field[len(b"branch ") :]
            for field in fields
            if field.startswith(b"branch ")
        ]
        branch: str | None = None
        if len(branch_fields) == 1:
            branch = os.fsdecode(branch_fields[0])
            prefix = "refs/heads/"
            if branch.startswith(prefix):
                branch = branch[len(prefix) :]
        discovered[candidate.git_dir] = _LinkedWorktree(candidate, branch)

    # A valid selected checkout must never disappear merely because an older Git
    # version produced an unfamiliar optional porcelain field.
    if repository.git_dir not in discovered:
        branch_result = _git(
            repository.root,
            "symbolic-ref",
            "--quiet",
            "--short",
            "HEAD",
            check=False,
        )
        branch = (
            branch_result.stdout.decode("utf-8", errors="surrogateescape").rstrip("\n")
            if branch_result.returncode == 0
            else None
        )
        discovered[repository.git_dir] = _LinkedWorktree(repository, branch)

    worktrees = sorted(
        discovered.values(),
        key=lambda item: (
            item.repository.git_dir != repository.git_dir,
            str(item.repository.root),
        ),
    )
    warnings = []
    if skipped:
        noun = "worktree" if skipped == 1 else "worktrees"
        warnings.append(f"Skipped {skipped} missing, prunable, or invalid {noun}.")
    return worktrees, warnings


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json_bytes(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def _file_details(path: Path) -> tuple[bytes, int, int, int]:
    details = path.lstat()
    if stat.S_ISLNK(details.st_mode):
        raise ValueError(f"Refusing to follow symlink {path}.")
    if not stat.S_ISREG(details.st_mode):
        raise ValueError(f"Expected a regular file at {path}.")
    if details.st_size > _MAX_CONFIG_BYTES:
        raise ValueError(f"Configuration file is too large: {path}.")
    return (
        path.read_bytes(),
        stat.S_IMODE(details.st_mode),
        details.st_atime_ns,
        details.st_mtime_ns,
    )


def _assert_native_target(path: Path, root: Path) -> None:
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Native hook path escapes the repository: {path}.") from exc
    cursor = root
    for part in relative.parts:
        cursor = cursor / part
        try:
            details = cursor.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(details.st_mode):
            raise ValueError(f"Refusing to follow symlink {cursor}.")
        if cursor != path and not stat.S_ISDIR(details.st_mode):
            raise ValueError(f"Expected a directory at {cursor}.")
        if cursor == path and not stat.S_ISREG(details.st_mode):
            raise ValueError(f"Expected a regular file at {cursor}.")


def _load_json_config(path: Path) -> tuple[dict[str, Any], bytes | None, int, int, int]:
    if not path.exists():
        return {}, None, 0o600, 0, 0
    raw, mode, atime_ns, mtime_ns = _file_details(path)
    try:
        decoded = raw.decode("utf-8")
        value = json.loads(decoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Refusing to overwrite malformed JSON in {path}: {exc}.") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Refusing to overwrite non-object JSON in {path}.")
    return value, raw, mode, atime_ns, mtime_ns


def _load_manifest(repository: _Repository) -> dict[str, Any] | None:
    path = repository.manifest_path
    if not path.exists():
        return None
    raw, _mode, _atime, _mtime = _file_details(path)
    try:
        manifest = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"The attribution install manifest is malformed: {exc}.") from exc
    if not isinstance(manifest, dict) or manifest.get("version") != _VERSION:
        raise ValueError("The attribution install manifest has an unsupported version.")
    enabled = manifest.get("enabled_worktrees")
    integrations = manifest.get("integrations")
    if (
        not isinstance(enabled, list)
        or any(not isinstance(item, str) for item in enabled)
        or len(set(enabled)) != len(enabled)
        or not isinstance(integrations, list)
        or any(not isinstance(item, dict) for item in integrations)
    ):
        raise ValueError("The attribution install manifest has invalid entries.")
    expected_hooks = str(repository.managed_hooks_path)
    if manifest.get("managed_hooks_path") != expected_hooks:
        raise ValueError("The attribution install manifest points outside its managed hook directory.")
    return manifest


def _local_hooks_path(repository: _Repository) -> tuple[bool, str | None]:
    result = _git(
        repository.root,
        "config",
        "--local",
        "--null",
        "--get-all",
        "core.hooksPath",
        check=False,
    )
    if result.returncode == 1:
        return False, None
    if result.returncode != 0:
        message = result.stderr.decode("utf-8", errors="replace").strip()
        raise ValueError(message or "Could not read the repository hook configuration.")
    values = [
        item.decode("utf-8", errors="surrogateescape")
        for item in result.stdout.split(b"\x00")
        if item != b""
    ]
    if len(values) != 1:
        raise ValueError("Multiple repo-local core.hooksPath values cannot be preserved safely.")
    return True, values[0]


def _effective_hooks_path(repository: _Repository) -> str:
    configured = _git(
        repository.root,
        "config",
        "--path",
        "--get",
        "core.hooksPath",
        check=False,
    )
    if configured.returncode == 0:
        return configured.stdout.decode("utf-8", errors="surrogateescape").rstrip("\n")
    if configured.returncode != 1:
        message = configured.stderr.decode("utf-8", errors="replace").strip()
        raise ValueError(message or "Could not resolve the existing Git hooks path.")
    default = _git(
        repository.root,
        "rev-parse",
        "--path-format=absolute",
        "--git-path",
        "hooks",
    )
    return default.stdout.decode("utf-8", errors="surrogateescape").rstrip("\n")


def _effective_hooks_match(
    repository: _Repository, managed_hooks_path: Path
) -> tuple[bool, str]:
    configured = _effective_hooks_path(repository)
    effective = _resolve_original_directory(repository, configured).resolve()
    return effective == managed_hooks_path.resolve(), configured


def _set_local_hooks_path(repository: _Repository, value: str) -> None:
    _git(
        repository.root,
        "config",
        "--local",
        "--replace-all",
        "core.hooksPath",
        value,
    )


def _restore_local_hooks_path(
    repository: _Repository, previous: tuple[bool, str | None]
) -> None:
    present, value = previous
    if present:
        assert value is not None
        _set_local_hooks_path(repository, value)
        return
    result = _git(
        repository.root,
        "config",
        "--local",
        "--unset-all",
        "core.hooksPath",
        check=False,
    )
    if result.returncode not in {0, 5}:
        message = result.stderr.decode("utf-8", errors="replace").strip()
        raise ValueError(message or "Could not restore core.hooksPath.")


def _ephemeral_runtime_path(path: Path) -> bool:
    """Report whether a runtime path lives in a tool cache that gets pruned.

    ``uvx`` builds its environment under uv's versioned cache buckets, and
    ``pipx run`` keeps its temporary venvs in a ``.cache`` directory. Hook
    commands outlive both, so an installation from one breaks silently.
    """

    parts = path.parts
    return any(
        part.startswith(("archive-v", "environments-v"))
        or (part == ".cache" and "pipx" in parts)
        for part in parts
    )


def _runtime() -> RuntimeCommand:
    if sys.version_info < (3, 11):
        raise ValueError("Automation installation requires Python 3.11 or newer.")
    runtime = current_runtime()
    for path in (runtime.executable, runtime.source_root):
        if path is not None and _ephemeral_runtime_path(path):
            raise ValueError(_EPHEMERAL_RUNTIME_ERROR)
    if not runtime.executable.is_file() or not os.access(
        runtime.executable, os.X_OK
    ):
        raise ValueError("The current Joyride runtime is not a stable executable.")
    if runtime.source_root is not None and not runtime.source_root.is_dir():
        raise ValueError("The current Joyride source root is not stable.")
    return runtime


def _manifest_runtime(manifest: dict[str, Any]) -> RuntimeCommand:
    return stored_runtime(
        executable=manifest.get("executable"),
        source_root=manifest.get("source_root"),
        kind=manifest.get("runtime_kind"),
    )


def _validate_install_scope(repository: _Repository) -> None:
    home = Path.home().resolve()
    filesystem_root = Path(repository.root.anchor).resolve()
    if repository.root in {home, filesystem_root}:
        raise ValueError(
            "Refusing to install repository hooks into a home or filesystem root. "
            "Select a narrower Git repository."
        )


def _managed_command(
    repository: _Repository,
    harness: str,
    runtime: RuntimeCommand,
) -> str:
    invocation = runtime.cli(
        ("_hook", "--harness", harness, "--repository-hook"),
        bootstrap=repository.bootstrap_path,
    )
    return f"{invocation.shell()} {_MARKER}"


def _bootstrap_bytes(source_root: Path) -> bytes:
    source = str(source_root)
    text = f'''#!/usr/bin/env python3
{_MARKER}
"""Private launcher generated by harness-attribution."""
import sys

SOURCE_ROOT = {source!r}
if SOURCE_ROOT in sys.path:
    sys.path.remove(SOURCE_ROOT)
sys.path.insert(0, SOURCE_ROOT)

from attribution.cli import main

raise SystemExit(main())
'''
    return text.encode("utf-8")


def _handler(command: str) -> dict[str, Any]:
    return {"type": "command", "command": command, "timeout": 10}


def _event_matcher(event: str) -> str | None:
    if event not in _TOOL_EVENTS:
        return None
    return _SNAPSHOT_TOOL_MATCHER if event == "PreToolUse" else _EVERY_TOOL_MATCHER


def _managed_group(event: str, command: str) -> dict[str, Any]:
    group: dict[str, Any] = {"hooks": [_handler(command)]}
    matcher = _event_matcher(event)
    if matcher is not None:
        # Keep the provider's conventional ordering without adding provider-only keys.
        group = {"matcher": matcher, "hooks": group["hooks"]}
    return group


def _iter_hook_commands(payload: dict[str, Any]):
    hooks = payload.get("hooks")
    if hooks is None:
        return
    if not isinstance(hooks, dict):
        raise ValueError("The top-level hooks value must be an object.")
    for event, groups in hooks.items():
        if not isinstance(groups, list):
            continue
        for group_index, group in enumerate(groups):
            if not isinstance(group, dict):
                continue
            handlers = group.get("hooks")
            if not isinstance(handlers, list):
                continue
            for handler_index, candidate in enumerate(handlers):
                if isinstance(candidate, dict) and isinstance(candidate.get("command"), str):
                    yield event, group_index, handler_index, group, candidate


def _validate_managed_commands(
    payload: dict[str, Any], command: str, events: tuple[str, ...]
) -> None:
    hooks = payload.get("hooks")
    if hooks is not None and not isinstance(hooks, dict):
        raise ValueError("The top-level hooks value must be an object.")
    if isinstance(hooks, dict):
        for event in events:
            groups = hooks.get(event)
            if groups is None:
                continue
            if not isinstance(groups, list):
                raise ValueError(f"Hook event {event} must contain an array.")
            for group in groups:
                if not isinstance(group, dict) or not isinstance(group.get("hooks"), list):
                    raise ValueError(f"Hook event {event} contains an invalid matcher group.")
    for event, _group_index, _handler_index, _group, candidate in _iter_hook_commands(payload):
        candidate_command = candidate["command"]
        if _MARKER in candidate_command and (
            candidate_command != command or event not in events
        ):
            raise ValueError(
                "A conflicting attribution-managed native hook already exists; no files were changed."
            )


def _merge_native_hooks(
    payload: dict[str, Any], command: str, events: tuple[str, ...]
) -> dict[str, Any]:
    _validate_managed_commands(payload, command, events)
    result = copy.deepcopy(payload)
    hooks = result.setdefault("hooks", {})
    assert isinstance(hooks, dict)
    for event in events:
        groups = hooks.setdefault(event, [])
        assert isinstance(groups, list)
        canonical = _managed_group(event, command)
        exact_locations: list[int] = []
        for index, group in enumerate(groups):
            if not isinstance(group, dict):
                continue
            handlers = group.get("hooks")
            if not isinstance(handlers, list):
                continue
            if any(
                isinstance(candidate, dict) and candidate.get("command") == command
                for candidate in handlers
            ):
                exact_locations.append(index)
        if len(exact_locations) == 1 and groups[exact_locations[0]] == canonical:
            continue

        cleaned: list[Any] = []
        for group in groups:
            if not isinstance(group, dict) or not isinstance(group.get("hooks"), list):
                cleaned.append(group)
                continue
            updated = copy.deepcopy(group)
            updated["hooks"] = [
                candidate
                for candidate in updated["hooks"]
                if not (
                    isinstance(candidate, dict) and candidate.get("command") == command
                )
            ]
            if updated["hooks"] or set(updated) - {"matcher", "hooks"}:
                cleaned.append(updated)
        cleaned.append(canonical)
        hooks[event] = cleaned
    return result


def _merge_claude_telemetry_env(
    payload: dict[str, Any],
    expected: dict[str, str],
    previously_owned: set[str],
    previous_expected: dict[str, Any],
) -> tuple[dict[str, Any], list[str], list[str]]:
    """Refresh proven-owned settings and preserve user changes as conflicts."""

    result = copy.deepcopy(payload)
    existing = result.get("env")
    if existing is not None and not isinstance(existing, dict):
        return result, [], ["env"]
    env = existing if isinstance(existing, dict) else {}
    if existing is None:
        result["env"] = env
    proven_owned: set[str] = set()
    missing: set[str] = set()
    conflicts: list[str] = []
    for key, value in expected.items():
        if key in previously_owned:
            prior_value = previous_expected.get(key)
            if key in env and isinstance(prior_value, str) and env[key] == prior_value:
                proven_owned.add(key)
            elif key in env and env[key] == value:
                # The value is already correct, but no longer matches the
                # recorded preimage. Do not claim a possible user edit.
                continue
            else:
                conflicts.append(key)
        elif key not in env:
            missing.add(key)
        elif env[key] != value:
            conflicts.append(key)
    for key in proven_owned:
        # The current value is still byte-for-byte what Joyride recorded,
        # so it is safe to migrate to the current endpoint.
        env[key] = expected[key]
    if not conflicts:
        for key in missing:
            env[key] = expected[key]
    owned = proven_owned | (missing if not conflicts else set())
    return result, sorted(owned), sorted(conflicts)


def _legacy_claude_endpoint_recovery(
    payload: dict[str, Any],
    raw: bytes | None,
    record: dict[str, Any],
    current_expected: dict[str, str],
) -> bool:
    """Recognize only the provenance state written by the stale-port bug."""

    if raw is None or record.get("installed_sha256") != _sha256(raw):
        return False
    env = payload.get("env")
    expected = record.get("telemetry_env_expected")
    owned = record.get("telemetry_env_keys")
    conflicts = record.get("telemetry_env_conflicts")
    if (
        not isinstance(env, dict)
        or not isinstance(expected, dict)
        or not isinstance(owned, list)
        or not isinstance(conflicts, list)
        or env.get(_CLAUDE_TELEMETRY_ENDPOINT)
        != _CLAUDE_LEGACY_DEFAULT_ENDPOINT
        or _CLAUDE_TELEMETRY_ENDPOINT in owned
        or _CLAUDE_TELEMETRY_ENDPOINT not in conflicts
        or _CLAUDE_TELEMETRY_HEADER not in owned
        or expected.get(_CLAUDE_TELEMETRY_ENDPOINT)
        != current_expected.get(_CLAUDE_TELEMETRY_ENDPOINT)
        or expected.get(_CLAUDE_TELEMETRY_ENDPOINT)
        == _CLAUDE_LEGACY_DEFAULT_ENDPOINT
    ):
        return False
    recorded_header = expected.get(_CLAUDE_TELEMETRY_HEADER)
    return isinstance(recorded_header, str) and env.get(
        _CLAUDE_TELEMETRY_HEADER
    ) == recorded_header


def _has_claude_telemetry_credentials(payload: dict[str, Any]) -> bool:
    """Return whether a Claude settings object contains exporter credentials."""

    env = payload.get("env")
    return (
        isinstance(env, dict)
        and isinstance(env.get(_CLAUDE_TELEMETRY_HEADER), str)
        and bool(env[_CLAUDE_TELEMETRY_HEADER].strip())
    )


def _remove_claude_telemetry_env(
    payload: dict[str, Any], record: dict[str, Any]
) -> dict[str, Any]:
    """Remove only unchanged environment keys created by Joyride."""

    result = copy.deepcopy(payload)
    env = result.get("env")
    expected = record.get("telemetry_env_expected")
    owned = record.get("telemetry_env_keys")
    if not isinstance(env, dict) or not isinstance(expected, dict) or not isinstance(owned, list):
        return result
    for key in owned:
        if isinstance(key, str) and env.get(key) == expected.get(key):
            env.pop(key, None)
    if not env and record.get("telemetry_env_object_created") is True:
        result.pop("env", None)
    return result


def _remove_native_hooks(
    payload: dict[str, Any], command: str, events: tuple[str, ...], *, created_file: bool
) -> tuple[dict[str, Any], int]:
    _validate_managed_commands(payload, command, events)
    result = copy.deepcopy(payload)
    hooks = result.get("hooks")
    if not isinstance(hooks, dict):
        return result, 0
    removed = 0
    for event in events:
        groups = hooks.get(event)
        if not isinstance(groups, list):
            continue
        cleaned: list[Any] = []
        for group in groups:
            if not isinstance(group, dict) or not isinstance(group.get("hooks"), list):
                cleaned.append(group)
                continue
            updated = copy.deepcopy(group)
            kept_handlers = []
            for candidate in updated["hooks"]:
                if isinstance(candidate, dict) and candidate.get("command") == command:
                    removed += 1
                else:
                    kept_handlers.append(candidate)
            updated["hooks"] = kept_handlers
            if updated["hooks"] or set(updated) - {"matcher", "hooks"}:
                cleaned.append(updated)
        if cleaned or not created_file:
            hooks[event] = cleaned
        else:
            hooks.pop(event, None)
    if created_file and not hooks:
        result.pop("hooks", None)
    return result, removed


def _config_health(
    path: Path, command: str, events: tuple[str, ...], harness: str,
    *, only_event: str | None = None,
) -> tuple[bool, str | None]:
    try:
        payload, raw, _mode, _atime, _mtime = _load_json_config(path)
        if raw is None:
            return False, "Native hook file is missing."
        _validate_managed_commands(payload, command, events)
    except (OSError, ValueError) as exc:
        return False, str(exc)
    if harness == "claude-code" and payload.get("disableAllHooks") is True:
        return (
            False,
            "Claude Code has disableAllHooks enabled, so attribution hooks cannot run.",
        )
    hooks = payload.get("hooks")
    if not isinstance(hooks, dict):
        return False, "Native hook configuration is missing its hooks object."
    for event in (only_event,) if only_event is not None else events:
        groups = hooks.get(event)
        if not isinstance(groups, list):
            return False, f"Managed {event} hook is missing."
        matches = 0
        for group in groups:
            if not isinstance(group, dict) or not isinstance(group.get("hooks"), list):
                continue
            expected_matcher = _event_matcher(event)
            matcher_ok = (
                group.get("matcher") == expected_matcher
                if expected_matcher is not None
                else "matcher" not in group
            )
            if not matcher_ok:
                continue
            matches += sum(
                1
                for candidate in group["hooks"]
                if isinstance(candidate, dict)
                and candidate.get("type") == "command"
                and candidate.get("command") == command
            )
        if matches != 1:
            return False, f"Managed {event} hook is missing or duplicated."
    return True, None


def _user_scope(manifest: dict[str, Any]) -> bool:
    """Return whether this repository's harness hooks live at user scope."""

    return manifest.get("hook_scope") == _HOOK_SCOPE_USER


def _user_hook_health() -> dict[str, dict[str, Any]]:
    from .user_install import user_hook_health

    return user_hook_health()


def _user_scope_harness(health: dict[str, Any], enabled: bool) -> dict[str, Any]:
    """Describe one harness for a worktree whose hooks live at user scope."""

    if not enabled:
        return {
            "installed": False,
            "state": "not-installed",
            "message": "Hooks are not installed for this worktree.",
        }
    installed = bool(health.get("installed"))
    return {
        "installed": installed,
        "state": "enabled" if installed else "needs-attention",
        "message": health.get("message"),
    }


def _worktree_health(
    linked: _LinkedWorktree,
    manifest: dict[str, Any],
    *,
    proxy_ok: bool,
    current: bool,
) -> dict[str, Any]:
    repository = linked.repository
    worktree_id = str(repository.git_dir)
    enabled = worktree_id in manifest["enabled_worktrees"]
    managed_path = repository.managed_hooks_path
    messages: list[str] = []

    try:
        local_present, local_value = _local_hooks_path(repository)
        effective_ok, effective_value = _effective_hooks_match(
            repository, managed_path
        )
    except ValueError as exc:
        local_present, local_value = False, None
        effective_ok, effective_value = False, None
        messages.append(str(exc))
    local_ok = local_present and local_value == str(managed_path)
    if enabled and not local_ok:
        messages.append("The repository-local Git hook dispatcher is not configured.")
    if enabled and not effective_ok:
        if effective_value is None:
            messages.append("The effective Git hooks path could not be resolved.")
        else:
            messages.append(
                "A worktree-specific core.hooksPath overrides the attribution dispatcher."
            )

    try:
        runtime = _manifest_runtime(manifest)
    except ValueError:
        runtime = None
    harnesses: dict[str, dict[str, Any]] = {}
    user_scope = _user_scope(manifest)
    user_health = _user_hook_health() if user_scope else {}
    for harness, relative in _INTEGRATION_PATHS.items():
        if user_scope:
            harnesses[harness] = _user_scope_harness(
                user_health.get(harness, {}), enabled
            )
            if enabled and harnesses[harness]["message"]:
                messages.append(harnesses[harness]["message"])
            continue
        healthy = False
        message: str | None = None
        try:
            record = _integration_record(manifest, worktree_id, harness)
        except ValueError as exc:
            record = None
            message = str(exc)
        if not enabled:
            message = "Hooks are not installed for this worktree."
            state = "not-installed"
        elif record is None:
            message = message or "The install manifest is missing this native integration."
            state = "needs-attention"
        elif (
            not isinstance(record.get("path"), str)
            or not isinstance(record.get("managed_command"), str)
            or runtime is None
        ):
            message = "The native integration manifest entry is invalid."
            state = "needs-attention"
        else:
            expected_command = _managed_command(
                repository,
                harness,
                runtime,
            )
            if record["managed_command"] != expected_command:
                message = "The native hook command needs migration; run joyride install."
            else:
                healthy, message = _config_health(
                    repository.root / relative,
                    expected_command,
                    _EVENTS[harness],
                    harness,
                )
            state = "enabled" if healthy else "needs-attention"
        harnesses[harness] = {
            "installed": bool(enabled and healthy),
            "state": state,
            "message": message,
        }
        if enabled and message:
            messages.append(message)

    git_hook_installed = bool(enabled and local_ok and effective_ok and proxy_ok)
    if enabled and not proxy_ok:
        messages.append("The managed Git hook dispatcher needs attention.")
    installed = bool(
        enabled
        and git_hook_installed
        and all(item["installed"] for item in harnesses.values())
    )
    return {
        "path": str(repository.root),
        "worktree_id": worktree_id,
        "branch": linked.branch,
        "current": current,
        "enabled": enabled,
        "installed": installed,
        "healthy": installed,
        "state": (
            "enabled" if installed else "needs-attention" if enabled else "not-installed"
        ),
        "git_hook_installed": git_hook_installed,
        "harnesses": harnesses,
        "message": "; ".join(dict.fromkeys(messages)) or None,
    }


def _resolve_original_directory(repository: _Repository, configured: str) -> Path:
    path = Path(configured).expanduser()
    return path if path.is_absolute() else repository.root / path


def _discovered_hook_names(repository: _Repository, original: str) -> set[str]:
    names = set(_STANDARD_GIT_HOOKS)
    directory = _resolve_original_directory(repository, original)
    if directory == repository.managed_hooks_path:
        raise ValueError("The existing Git hooks path conflicts with the managed hook directory.")
    try:
        entries = list(directory.iterdir())
    except FileNotFoundError:
        return names
    except OSError as exc:
        raise ValueError(f"Could not inspect existing Git hooks in {directory}: {exc}.") from exc
    for entry in entries:
        if "/" in entry.name or "\x00" in entry.name:
            continue
        try:
            details = entry.lstat()
        except OSError:
            continue
        if stat.S_ISREG(details.st_mode) or stat.S_ISLNK(details.st_mode):
            names.add(entry.name)
    return names


def _proxy_bytes(
    hook_name: str,
    original_hooks_path: str,
    runtime: RuntimeCommand,
    bootstrap: Path,
) -> bytes:
    original = shlex.quote(original_hooks_path)
    name = shlex.quote(hook_name)
    if hook_name == "pre-push":
        if runtime.standalone:
            sharing_command = runtime.module_child(
                "attribution.sharing",
                (),
                standalone_action="_share",
            ).shell()
        else:
            assert runtime.source_root is not None
            program = (
                "import sys; sys.dont_write_bytecode = True; "
                f"sys.path.append({str(runtime.source_root)!r}); "
                "from attribution.sharing import main; raise SystemExit(main())"
            )
            sharing_command = " ".join(
                shlex.quote(argument)
                for argument in (
                    str(runtime.executable),
                    "-I",
                    "-S",
                    "-B",
                    "-c",
                    program,
                )
            )
        temporary = shlex.quote(
            str(bootstrap.parent / "pre-push-input.XXXXXXXXXX")
        )
        text = f'''#!/bin/sh
{_MARKER}
_attribution_original_dir={original}
case "$_attribution_original_dir" in
  /*) ;;
  *) _attribution_original_dir="$PWD/$_attribution_original_dir" ;;
esac
_attribution_original="$_attribution_original_dir"/{name}
if [ "${{{_METADATA_PUSH_GUARD}:-}}" = "1" ]; then
  if [ -x "$_attribution_original" ]; then
    exec "$_attribution_original" "$@"
  fi
  exit 0
fi
umask 077
_attribution_input=$(mktemp {temporary} 2>/dev/null) || _attribution_input=
if [ -z "$_attribution_input" ]; then
  printf '%s\n' 'Joyride: metadata input could not be buffered; branch push will continue.' >&2
  if [ -x "$_attribution_original" ]; then
    exec "$_attribution_original" "$@"
  fi
  exit 0
fi
_attribution_cleanup() {{ rm -f -- "$_attribution_input"; }}
trap '_attribution_cleanup' 0
trap 'exit 129' 1
trap 'exit 130' 2
trap 'exit 143' 15
if ! cat >"$_attribution_input"; then
  if [ -x "$_attribution_original" ]; then
    printf '%s\n' 'Joyride: pre-push input could not be buffered; the original hook was not run and the push was stopped.' >&2
    exit 1
  fi
  printf '%s\n' 'Joyride: metadata input could not be buffered; branch push will continue.' >&2
  exit 0
fi
_attribution_status=0
if [ -x "$_attribution_original" ]; then
  "$_attribution_original" "$@" <"$_attribution_input" || _attribution_status=$?
fi
if [ "$_attribution_status" -ne 0 ]; then
  exit "$_attribution_status"
fi
{sharing_command} "$@" <"$_attribution_input" || printf '%s\n' 'Joyride: metadata hook failed; branch push will continue.' >&2
exit 0
'''
        return text.encode("utf-8")
    if hook_name not in {"post-checkout", "post-commit", "post-merge"}:
        text = f'''#!/bin/sh
{_MARKER}
_attribution_original_dir={original}
case "$_attribution_original_dir" in
  /*) ;;
  *) _attribution_original_dir="$PWD/$_attribution_original_dir" ;;
esac
_attribution_original="$_attribution_original_dir"/{name}
if [ -x "$_attribution_original" ]; then
  exec "$_attribution_original" "$@"
fi
exit 0
'''
        return text.encode("utf-8")

    runtime_command = runtime.cli((), bootstrap=bootstrap).shell()
    action = (
        f'{_CURRENT_WORKTREE_ONLY_ENV}=1 {runtime_command} '
        'install --repo "$_attribution_repo"'
        if hook_name == "post-checkout"
        else f'{runtime_command} --repo "$_attribution_repo" _git-hook {hook_name}'
    )
    text = f'''#!/bin/sh
{_MARKER}
_attribution_original_dir={original}
case "$_attribution_original_dir" in
  /*) ;;
  *) _attribution_original_dir="$PWD/$_attribution_original_dir" ;;
esac
_attribution_original="$_attribution_original_dir"/{name}
_attribution_status=0
if [ -x "$_attribution_original" ]; then
  "$_attribution_original" "$@" || _attribution_status=$?
fi
_attribution_repo=$(git rev-parse --show-toplevel 2>/dev/null) || _attribution_repo=
if [ -n "$_attribution_repo" ]; then
  {action} >/dev/null 2>&1 || :
fi
exit "$_attribution_status"
'''
    return text.encode("utf-8")


def _exclude_with_block(raw: bytes) -> bytes:
    occurrences = raw.count(_EXCLUDE_BLOCK)
    has_marker = b"attribution-managed-v1" in raw
    if occurrences == 1:
        return raw
    if occurrences or has_marker:
        raise ValueError("The Git info/exclude file contains a conflicting attribution block.")
    separator = b"" if not raw or raw.endswith(b"\n") else b"\n"
    return raw + separator + _EXCLUDE_BLOCK + b"\n"


def _exclude_without_block(raw: bytes) -> tuple[bytes, bool]:
    if raw.count(_EXCLUDE_BLOCK) != 1:
        return raw, False
    start = raw.index(_EXCLUDE_BLOCK)
    end = start + len(_EXCLUDE_BLOCK)
    if end < len(raw) and raw[end : end + 1] == b"\n":
        end += 1
    if start > 0 and raw[start - 1 : start] == b"\n" and end == len(raw):
        # This newline was the separator only when the preceding content did not
        # already end in one. Exact unchanged installs use their backup instead.
        pass
    return raw[:start] + raw[end:], True


def _backup_path(repository: _Repository, worktree_id: str, harness: str) -> Path:
    worktree_key = hashlib.sha256(worktree_id.encode("utf-8")).hexdigest()[:20]
    return repository.state_dir / "backups" / worktree_key / f"{harness}.json"


def _tracked(repository: _Repository, relative: Path) -> bool:
    result = _git(
        repository.root,
        "ls-files",
        "--error-unmatch",
        "--",
        relative.as_posix(),
        check=False,
    )
    return result.returncode == 0


def _integration_record(
    manifest: dict[str, Any], worktree_id: str, harness: str
) -> dict[str, Any] | None:
    matches = [
        item
        for item in manifest.get("integrations", [])
        if item.get("worktree_id") == worktree_id and item.get("harness") == harness
    ]
    if len(matches) > 1:
        raise ValueError(f"Duplicate {harness} entries exist in the install manifest.")
    return matches[0] if matches else None


def _safe_private_path(path: Path, repository: _Repository) -> None:
    normalised = Path(os.path.abspath(path))
    if normalised != path:
        raise ValueError(f"Managed path is not canonical: {path}.")
    try:
        normalised.relative_to(repository.state_dir)
    except ValueError as exc:
        raise ValueError(f"Managed path escapes the private attribution directory: {path}.") from exc
    cursor = repository.state_dir
    try:
        state_details = cursor.lstat()
    except FileNotFoundError:
        state_details = None
    if state_details is not None and (
        stat.S_ISLNK(state_details.st_mode) or not stat.S_ISDIR(state_details.st_mode)
    ):
        raise ValueError(f"Unsafe private attribution directory {cursor}.")
    relative = normalised.relative_to(repository.state_dir)
    for part in relative.parts:
        cursor = cursor / part
        try:
            details = cursor.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(details.st_mode):
            raise ValueError(f"Refusing to follow symlink {cursor}.")


def _prepare_exclude(
    repository: _Repository,
    manifest: dict[str, Any],
    *,
    first_active_install: bool,
) -> tuple[dict[str, Any], bytes, bytes | None, int]:
    info_directory = repository.common_dir / "info"
    try:
        info_details = info_directory.lstat()
    except FileNotFoundError:
        info_details = None
    if info_details is not None and (
        stat.S_ISLNK(info_details.st_mode) or not stat.S_ISDIR(info_details.st_mode)
    ):
        raise ValueError(f"Unsafe Git info directory {info_directory}.")
    path = repository.common_dir / "info" / "exclude"
    if path.exists():
        raw, mode, atime, mtime = _file_details(path)
    else:
        raw, mode, atime, mtime = b"", 0o600, 0, 0
    if first_active_install and b"attribution-managed-v1" in raw:
        raise ValueError(
            "Git info/exclude contains an attribution marker without an active manifest."
        )
    desired = _exclude_with_block(raw)
    if first_active_install:
        backup = repository.state_dir / "backups" / "shared" / "info-exclude"
        if backup.exists():
            raise ValueError(f"A stale attribution backup already exists at {backup}.")
        record = {
            "path": str(path),
            "created_file": not path.exists(),
            "backup_path": str(backup),
            "original_mode": mode,
            "original_atime_ns": atime,
            "original_mtime_ns": mtime,
            "preimage_sha256": _sha256(raw),
            "installed_sha256": _sha256(desired),
            "restore_preimage_when_unchanged": True,
        }
    else:
        record = manifest.get("exclude")
        if not isinstance(record, dict):
            raise ValueError("The active install manifest is missing its exclude record.")
        previous_hash = record.get("installed_sha256")
        if previous_hash != _sha256(raw):
            record["restore_preimage_when_unchanged"] = False
        record["installed_sha256"] = _sha256(desired)
    return record, desired, raw, mode


def _manifest_bytes(manifest: dict[str, Any]) -> bytes:
    manifest["enabled_worktrees"] = sorted(set(manifest["enabled_worktrees"]))
    manifest["integrations"] = sorted(
        manifest["integrations"], key=lambda item: (item["worktree_id"], item["harness"])
    )
    return _json_bytes(manifest)


def _empty_status(path: Path, *, warning: str | None = None) -> dict[str, Any]:
    message = "Hooks are not installed for this worktree."
    warnings = [warning] if warning else []
    return {
        "installed": False,
        "repository_path": str(path),
        "git_repository": False,
        "hook_scope": "repository",
        "worktree_id": None,
        "harnesses": {
            "codex": {"installed": False, "state": "not-installed", "message": message},
            "claude-code": {
                "installed": False,
                "state": "not-installed",
                "message": message,
            },
        },
        "git_hook_installed": False,
        "repository_installed": False,
        "repository_healthy": False,
        "discovered_worktree_count": 0,
        "enabled_worktree_count": 0,
        "healthy_worktree_count": 0,
        "needs_attention_worktree_count": 0,
        "unavailable_enabled_worktree_count": 0,
        "worktrees": [],
        "warnings": warnings,
        "notice": _NOTICE,
    }


def _claude_cost_status(
    repository: _Repository, manifest: dict[str, Any], worktree_id: str
) -> dict[str, Any]:
    if manifest.get("telemetry_enabled") is not True:
        return {"enabled": False, "state": "disabled"}
    try:
        record = _integration_record(manifest, worktree_id, "claude-code")
        if record is None:
            raise ValueError("Claude telemetry integration is missing.")
        blocked_reason = record.get("telemetry_env_blocked_reason")
        if isinstance(blocked_reason, str) and blocked_reason:
            raise ValueError(blocked_reason)
        expected = record.get("telemetry_env_expected")
        if not isinstance(expected, dict) or not expected:
            raise ValueError("Claude telemetry settings are missing.")
        payload, raw, _mode, _atime, _mtime = _load_json_config(
            repository.root / _INTEGRATION_PATHS["claude-code"]
        )
        env = payload.get("env") if raw is not None else None
        if not isinstance(env, dict):
            raise ValueError("Claude telemetry environment is missing.")
        mismatches = [key for key, value in expected.items() if env.get(key) != value]
        if mismatches:
            raise ValueError(
                "Claude telemetry conflicts with existing environment keys: "
                + ", ".join(sorted(mismatches))
            )
        return {"enabled": True, "state": "enabled"}
    except (OSError, ValueError) as exc:
        return {"enabled": False, "state": "needs-attention", "message": str(exc)}


def installation_status(repo: str | Path) -> dict[str, Any]:
    """Return read-only installation state for one Git worktree."""

    requested = Path(repo).expanduser().resolve()
    try:
        repository = _repository(requested)
    except ValueError:
        return _empty_status(requested, warning="The selected folder is not a Git repository.")

    base = _empty_status(repository.root)
    base["git_repository"] = True
    base["worktree_id"] = str(repository.git_dir)
    worktrees, discovery_warnings = _discover_worktrees(repository)
    base["discovered_worktree_count"] = len(worktrees)
    base["worktrees"] = [
        {
            "path": str(item.repository.root),
            "worktree_id": str(item.repository.git_dir),
            "branch": item.branch,
            "current": item.repository.git_dir == repository.git_dir,
            "enabled": False,
        }
        for item in worktrees
    ]
    warnings: list[str] = list(discovery_warnings)
    try:
        manifest = _load_manifest(repository)
    except (OSError, ValueError) as exc:
        message = str(exc)
        base["warnings"] = [message]
        for value in base["harnesses"].values():
            value.update(state="needs-attention", message=message)
        return base
    if manifest is None:
        # A marker without its manifest cannot be removed or adopted safely.
        for harness, relative in _INTEGRATION_PATHS.items():
            path = repository.root / relative
            try:
                payload, raw, _mode, _atime, _mtime = _load_json_config(path)
                conflict = raw is not None and any(
                    _MARKER in candidate[4]["command"]
                    for candidate in _iter_hook_commands(payload)
                )
            except (OSError, ValueError):
                conflict = False
            if conflict:
                message = "Managed hook markers exist without an install manifest."
                base["harnesses"][harness].update(
                    state="needs-attention", message=message
                )
                warnings.append(message)
        base["warnings"] = list(dict.fromkeys(warnings))
        return base

    worktree_id = str(repository.git_dir)
    base["telemetry"] = {
        "claude": _claude_cost_status(repository, manifest, worktree_id)
    }
    base["traces_enabled"] = manifest.get("traces_enabled") is not False
    enabled_ids = set(manifest["enabled_worktrees"])
    base["repository_installed"] = bool(enabled_ids)
    base["enabled_worktree_count"] = len(enabled_ids)
    for item in base["worktrees"]:
        item["enabled"] = item["worktree_id"] in enabled_ids
    selected_enabled = worktree_id in manifest["enabled_worktrees"]
    try:
        runtime = _manifest_runtime(manifest)
    except ValueError:
        runtime = None
    user_scope = _user_scope(manifest)
    base["hook_scope"] = _HOOK_SCOPE_USER if user_scope else "repository"
    user_health = _user_hook_health() if user_scope else {}
    for harness, relative in _INTEGRATION_PATHS.items():
        if user_scope:
            base["harnesses"][harness] = _user_scope_harness(
                user_health.get(harness, {}), selected_enabled
            )
            message = base["harnesses"][harness]["message"]
            if selected_enabled and message:
                warnings.append(message)
            continue
        try:
            record = _integration_record(manifest, worktree_id, harness)
        except ValueError as exc:
            message = str(exc)
            base["harnesses"][harness].update(state="needs-attention", message=message)
            warnings.append(message)
            continue
        if record is None:
            if selected_enabled:
                message = "The install manifest is missing this native integration."
                base["harnesses"][harness].update(
                    state="needs-attention", message=message
                )
                warnings.append(message)
            continue
        command = record.get("managed_command")
        expected_path = repository.root / relative
        if (
            not isinstance(command, str)
            or not isinstance(record.get("path"), str)
            or runtime is None
        ):
            message = "The native integration manifest entry is invalid."
            base["harnesses"][harness].update(state="needs-attention", message=message)
            warnings.append(message)
            continue
        expected_command = _managed_command(
            repository,
            harness,
            runtime,
        )
        if command != expected_command:
            message = "The native hook command needs migration; run joyride install."
            base["harnesses"][harness].update(state="needs-attention", message=message)
            warnings.append(message)
            continue
        healthy, message = _config_health(
            expected_path, expected_command, _EVENTS[harness], harness
        )
        state = "enabled" if healthy and selected_enabled else "needs-attention"
        if not selected_enabled and healthy:
            message = "Native hooks remain but this worktree is not enabled."
        base["harnesses"][harness] = {
            "installed": bool(healthy and selected_enabled),
            "state": state,
            "message": message,
        }
        if message:
            warnings.append(message)
        if _tracked(repository, relative):
            warnings.append(
                f"{relative.as_posix()} is tracked by Git; the local exclude rule cannot hide it."
            )

    if not enabled_ids:
        base["worktrees"] = [
            _worktree_health(
                item,
                manifest,
                proxy_ok=False,
                current=item.repository.git_dir == repository.git_dir,
            )
            for item in worktrees
        ]
        base["warnings"] = list(dict.fromkeys(warnings))
        return base

    try:
        local_present, local_value = _local_hooks_path(repository)
        effective_ok, _effective_value = _effective_hooks_match(
            repository, repository.managed_hooks_path
        )
    except ValueError as exc:
        local_present, local_value = False, None
        effective_ok = False
        warnings.append(str(exc))
    managed_path = str(repository.managed_hooks_path)
    proxy_hashes = manifest.get("proxy_hashes", {})
    proxy_ok = isinstance(proxy_hashes, dict) and bool(proxy_hashes)
    try:
        if not isinstance(proxy_hashes, dict):
            raise ValueError("The install manifest has invalid Git proxy metadata.")
        for name, expected_hash in proxy_hashes.items():
            if (
                not isinstance(name, str)
                or Path(name).name != name
                or name in {".", ".."}
                or not isinstance(expected_hash, str)
            ):
                raise ValueError("The install manifest contains an invalid Git proxy name.")
            proxy_path = repository.managed_hooks_path / name
            _safe_private_path(proxy_path, repository)
            proxy_raw, proxy_mode, _atime, _mtime = _file_details(proxy_path)
            if expected_hash != _sha256(proxy_raw) or not proxy_mode & 0o111:
                proxy_ok = False
        for required in ("post-commit", "post-merge"):
            if required not in proxy_hashes:
                proxy_ok = False
        runtime = _manifest_runtime(manifest)
        if runtime.standalone:
            try:
                repository.bootstrap_path.lstat()
                bootstrap_present = True
            except FileNotFoundError:
                bootstrap_present = False
            if (
                bootstrap_present
                or manifest.get("bootstrap_sha256") is not None
            ):
                proxy_ok = False
        else:
            bootstrap_raw, bootstrap_mode, _atime, _mtime = _file_details(
                repository.bootstrap_path
            )
            if (
                manifest.get("bootstrap_sha256") != _sha256(bootstrap_raw)
                or not bootstrap_mode & 0o111
            ):
                proxy_ok = False
        if (
            not runtime.executable.is_file()
            or not os.access(runtime.executable, os.X_OK)
            or runtime.source_root is not None
            and not runtime.source_root.is_dir()
        ):
            proxy_ok = False
    except (OSError, ValueError) as exc:
        proxy_ok = False
        warnings.append(str(exc))
    base["git_hook_installed"] = bool(
        selected_enabled
        and local_present
        and local_value == managed_path
        and effective_ok
        and proxy_ok
    )
    if selected_enabled and not effective_ok:
        warnings.append(
            "A worktree-specific core.hooksPath overrides the attribution dispatcher."
        )
    if selected_enabled and not base["git_hook_installed"]:
        warnings.append("The managed Git hook dispatcher needs attention.")
    base["installed"] = bool(
        selected_enabled
        and base["git_hook_installed"]
        and all(item["installed"] for item in base["harnesses"].values())
    )

    evaluated_worktrees = [
        _worktree_health(
            item,
            manifest,
            proxy_ok=proxy_ok,
            current=item.repository.git_dir == repository.git_dir,
        )
        for item in worktrees
    ]
    base["worktrees"] = evaluated_worktrees
    live_enabled_ids = {
        item["worktree_id"] for item in evaluated_worktrees if item["enabled"]
    }
    healthy_count = sum(
        1 for item in evaluated_worktrees if item["enabled"] and item["healthy"]
    )
    unavailable_count = len(enabled_ids - live_enabled_ids)
    needs_attention_count = len(enabled_ids) - healthy_count
    base["healthy_worktree_count"] = healthy_count
    base["needs_attention_worktree_count"] = needs_attention_count
    base["unavailable_enabled_worktree_count"] = unavailable_count
    base["repository_healthy"] = bool(enabled_ids) and needs_attention_count == 0
    current_entry = next(
        (item for item in evaluated_worktrees if item["current"]), None
    )
    if current_entry is not None:
        base["harnesses"] = current_entry["harnesses"]
        base["git_hook_installed"] = current_entry["git_hook_installed"]
        base["installed"] = current_entry["installed"]
    if needs_attention_count:
        noun = "worktree needs" if needs_attention_count == 1 else "worktrees need"
        warnings.append(
            f"{needs_attention_count} enabled {noun} attention; "
            f"{healthy_count} of {len(enabled_ids)} are healthy."
        )
    base["warnings"] = list(dict.fromkeys(warnings))
    return base


def _install_worktree(
    repo: str | Path, *, traces_enabled: bool | None = None, native_hooks: bool = True
) -> dict[str, Any]:
    """Idempotently install native and Git automation for one worktree.

    ``traces_enabled`` preserves the repository's recorded choice when None.
    ``native_hooks`` is false when machine-level hooks already run. That
    worktree receives its Git publisher and a manifest with the user scope.
    """

    repository = _repository(repo)
    _validate_install_scope(repository)
    runtime = _runtime()
    telemetry_enabled = (
        native_hooks and os.environ.get(_DISABLE_TELEMETRY_ENV) != "1"
    )
    worktree_id = str(repository.git_dir)
    manifest = _load_manifest(repository)
    active = bool(manifest and manifest["enabled_worktrees"])
    local_before = _local_hooks_path(repository)

    if active:
        assert manifest is not None
        try:
            installed_runtime = _manifest_runtime(manifest)
        except ValueError as exc:
            raise ValueError(
                "The active install manifest has invalid runtime metadata."
            ) from exc
        if installed_runtime != runtime:
            raise ValueError(
                "Another active worktree uses a different attribution runtime; "
                "uninstall it before changing runtimes."
            )
        if local_before != (True, str(repository.managed_hooks_path)):
            raise ValueError(
                "core.hooksPath changed after installation; refusing to overwrite it."
            )
        effective_ok, _effective_value = _effective_hooks_match(
            repository, repository.managed_hooks_path
        )
        if not effective_ok:
            raise ValueError(
                "A worktree-specific core.hooksPath overrides attribution's Git hook "
                "dispatcher; remove that override before installing."
            )
        original_hooks_path = manifest.get("previous_effective_hooks_path")
        if not isinstance(original_hooks_path, str):
            raise ValueError("The active install manifest is missing the original hooks path.")
    else:
        original_hooks_path = _effective_hooks_path(repository)
        if (
            _resolve_original_directory(repository, original_hooks_path).resolve()
            == repository.managed_hooks_path
        ):
            raise ValueError("The existing core.hooksPath conflicts with attribution's private path.")
        manifest = {
            "version": _VERSION,
            "enabled_worktrees": [],
            "runtime_kind": runtime.kind,
            "executable": str(runtime.executable),
            "source_root": (
                str(runtime.source_root) if runtime.source_root is not None else None
            ),
            "previous_local_hooks_path": {
                "present": local_before[0],
                "value": local_before[1],
            },
            "previous_effective_hooks_path": original_hooks_path,
            "managed_hooks_path": str(repository.managed_hooks_path),
            "integrations": [],
            "proxy_hashes": {},
        }

    assert manifest is not None
    _safe_private_path(repository.manifest_path, repository)
    _safe_private_path(repository.bootstrap_path, repository)
    _safe_private_path(repository.managed_hooks_path, repository)
    if repository.managed_hooks_path.exists() and not active:
        try:
            if any(repository.managed_hooks_path.iterdir()):
                raise ValueError(
                    "The private managed hooks directory already contains unowned files."
                )
        except NotADirectoryError as exc:
            raise ValueError("The managed hooks path is not a directory.") from exc

    bootstrap = (
        _bootstrap_bytes(runtime.source_root)
        if runtime.source_root is not None
        else None
    )
    try:
        repository.bootstrap_path.lstat()
        bootstrap_exists = True
    except FileNotFoundError:
        bootstrap_exists = False
    if bootstrap_exists:
        if not active:
            raise ValueError(
                "A private attribution bootstrap exists without an active installation."
            )
        if bootstrap is None:
            raise ValueError(
                "A standalone attribution installation contains an unexpected Python bootstrap."
            )
        current, _mode, _atime, _mtime = _file_details(repository.bootstrap_path)
        if current != bootstrap:
            raise ValueError("The private attribution bootstrap conflicts with this runtime.")

    hook_names = _discovered_hook_names(repository, original_hooks_path)
    existing_proxy_hashes = manifest.get("proxy_hashes", {})
    if not isinstance(existing_proxy_hashes, dict):
        raise ValueError("The install manifest has invalid Git proxy metadata.")
    # Generate persistent hooks only from the validated values that will be
    # stored in the private install manifest. This prevents a checkout or
    # ambient PATH/PYTHONPATH value from becoming the publisher runtime.
    persistent_runtime = _manifest_runtime(manifest)
    proxy_payloads: dict[str, bytes] = {}
    for name in sorted(hook_names | set(existing_proxy_hashes)):
        path = repository.managed_hooks_path / name
        _safe_private_path(path, repository)
        desired = _proxy_bytes(
            name,
            original_hooks_path,
            persistent_runtime,
            repository.bootstrap_path,
        )
        if path.exists():
            current, _mode, _atime, _mtime = _file_details(path)
            owned_hash = existing_proxy_hashes.get(name)
            if current != desired and (
                not active
                or not isinstance(owned_hash, str)
                or owned_hash != _sha256(current)
            ):
                raise ValueError(f"Managed Git hook proxy was modified: {path}.")
        proxy_payloads[name] = desired

    native_preflight: list[dict[str, Any]] = []
    native_targets = _INTEGRATION_PATHS.items() if native_hooks else ()
    for harness, relative in native_targets:
        path = repository.root / relative
        _assert_native_target(path, repository.root)
        payload, raw, mode, atime, mtime = _load_json_config(path)
        claude_settings_tracked = harness == "claude-code" and _tracked(
            repository, relative
        )
        command = _managed_command(repository, harness, runtime)
        record = _integration_record(manifest, worktree_id, harness)
        if record is None and any(
            _MARKER in candidate[4]["command"]
            for candidate in _iter_hook_commands(payload)
        ):
            raise ValueError(
                f"{path} contains attribution markers without an owned manifest entry."
            )
        payload_to_merge = payload
        if record is not None:
            previous_command = record.get("managed_command")
            if (
                not isinstance(record.get("path"), str)
                or not isinstance(previous_command, str)
                or _MARKER not in previous_command
            ):
                raise ValueError(
                    f"The {harness} install manifest entry is invalid."
                )
            if previous_command != command:
                payload_to_merge, _removed = _remove_native_hooks(
                    payload,
                    previous_command,
                    _EVENTS[harness],
                    created_file=False,
                )
        if claude_settings_tracked and record is not None:
            # A settings file can become tracked after an earlier install. Remove
            # credentials that are still byte-for-byte ours before rewriting it.
            payload_to_merge = _remove_claude_telemetry_env(
                payload_to_merge, record
            )
        if claude_settings_tracked and _has_claude_telemetry_credentials(
            payload_to_merge
        ):
            raise ValueError(
                "Refusing to rewrite tracked .claude/settings.local.json while it "
                "contains OTLP exporter credentials. Remove the credentials or "
                "untrack the file first."
            )
        desired_payload = _merge_native_hooks(
            payload_to_merge, command, _EVENTS[harness]
        )
        backup: Path | None = None
        if record is None:
            backup = _backup_path(repository, worktree_id, harness)
            _safe_private_path(backup, repository)
            if backup.exists():
                raise ValueError(f"A stale attribution backup already exists at {backup}.")
        elif raw is None and not record.get("created_file"):
            raise ValueError(f"Pre-existing native hook file was removed: {path}.")
        native_preflight.append(
            {
                "harness": harness,
                "path": path,
                "payload": desired_payload,
                "raw": raw,
                "mode": mode,
                "atime": atime,
                "mtime": mtime,
                "command": command,
                "record": record,
                "backup": backup,
                "claude_settings_tracked": claude_settings_tracked,
            }
        )

    exclude_record, exclude_desired, exclude_raw, exclude_mode = _prepare_exclude(
        repository, manifest, first_active_install=not active
    )

    if telemetry_enabled:
        from .telemetry import ensure_collector
        from .telemetry_setup import claude_telemetry_env

        # Start only after every repository, hook, native config, backup, and
        # exclude target has passed its read-only preflight.  Read settings
        # after startup because an occupied default port can select a new one.
        ensure_collector()
        claude_telemetry = claude_telemetry_env()
    else:
        claude_telemetry = {}

    prepared_integrations: list[
        tuple[dict[str, Any], Path, bytes, bytes | None, int, int]
    ] = []
    for item in native_preflight:
        harness = item["harness"]
        path = item["path"]
        desired_payload = item["payload"]
        raw = item["raw"]
        mode = item["mode"]
        atime = item["atime"]
        mtime = item["mtime"]
        command = item["command"]
        record = item["record"]
        backup = item["backup"]
        claude_settings_tracked = item["claude_settings_tracked"]
        telemetry_conflicts: list[str] = []
        telemetry_keys: list[str] = []
        telemetry_env_object_created = False
        if (
            harness == "claude-code"
            and claude_telemetry
            and not claude_settings_tracked
        ):
            prior_keys = (
                set(record.get("telemetry_env_keys", []))
                if isinstance(record, dict)
                and isinstance(record.get("telemetry_env_keys"), list)
                else set()
            )
            prior_expected = (
                record.get("telemetry_env_expected", {})
                if isinstance(record, dict)
                and isinstance(record.get("telemetry_env_expected"), dict)
                else {}
            )
            if (
                isinstance(record, dict)
                and _legacy_claude_endpoint_recovery(
                    desired_payload, raw, record, claude_telemetry
                )
            ):
                # A buggy reinstall could drop ownership after recording the
                # unchanged legacy endpoint as its own installed output. These
                # exact signals recover that one value and no arbitrary URL.
                prior_keys.add(_CLAUDE_TELEMETRY_ENDPOINT)
                prior_expected = dict(prior_expected)
                prior_expected[_CLAUDE_TELEMETRY_ENDPOINT] = (
                    _CLAUDE_LEGACY_DEFAULT_ENDPOINT
                )
            telemetry_env_object_created = (
                bool(record.get("telemetry_env_object_created"))
                if isinstance(record, dict)
                else "env" not in desired_payload
            )
            desired_payload, telemetry_keys, telemetry_conflicts = (
                _merge_claude_telemetry_env(
                    desired_payload,
                    claude_telemetry,
                    prior_keys,
                    prior_expected,
                )
            )
        desired_mode = (
            0o600
            if harness == "claude-code"
            and _has_claude_telemetry_credentials(desired_payload)
            else mode
        )
        desired = _json_bytes(desired_payload)
        if record is None:
            assert isinstance(backup, Path)
            record = {
                "worktree_id": worktree_id,
                "path": str(path),
                "harness": harness,
                "created_file": raw is None,
                "created_parent": not path.parent.exists(),
                "managed_command": command,
                "backup_path": str(backup),
                "original_mode": mode,
                "original_atime_ns": atime,
                "original_mtime_ns": mtime,
                "preimage_sha256": _sha256(raw or b""),
                "installed_sha256": _sha256(desired),
                "installed_mode": desired_mode,
                "restore_preimage_when_unchanged": True,
            }
            if harness == "claude-code" and claude_settings_tracked:
                record["telemetry_env_blocked_reason"] = (
                    _CLAUDE_TRACKED_TELEMETRY_WARNING
                )
            elif harness == "claude-code" and claude_telemetry:
                record.update(
                    telemetry_env_expected=claude_telemetry,
                    telemetry_env_keys=telemetry_keys,
                    telemetry_env_conflicts=telemetry_conflicts,
                    telemetry_env_object_created=telemetry_env_object_created,
                )
            manifest["integrations"].append(record)
        else:
            if raw is not None and (
                record.get("installed_sha256") != _sha256(raw)
                or record.get("installed_mode", mode) != mode
            ):
                record["restore_preimage_when_unchanged"] = False
            record["path"] = str(path)
            record["managed_command"] = command
            record["installed_sha256"] = _sha256(desired)
            record["installed_mode"] = desired_mode
            if harness == "claude-code" and claude_settings_tracked:
                for key in (
                    "telemetry_env_expected",
                    "telemetry_env_keys",
                    "telemetry_env_conflicts",
                    "telemetry_env_object_created",
                ):
                    record.pop(key, None)
                record["telemetry_env_blocked_reason"] = (
                    _CLAUDE_TRACKED_TELEMETRY_WARNING
                )
            elif harness == "claude-code" and claude_telemetry:
                record.pop("telemetry_env_blocked_reason", None)
                record.update(
                    telemetry_env_expected=claude_telemetry,
                    telemetry_env_keys=telemetry_keys,
                    telemetry_env_conflicts=telemetry_conflicts,
                    telemetry_env_object_created=telemetry_env_object_created,
                )
        prepared_integrations.append(
            (record, path, desired, raw, mode, desired_mode)
        )

    manifest["exclude"] = exclude_record
    if worktree_id not in manifest["enabled_worktrees"]:
        manifest["enabled_worktrees"].append(worktree_id)
    manifest["proxy_hashes"] = {
        name: _sha256(payload) for name, payload in sorted(proxy_payloads.items())
    }
    manifest["runtime_kind"] = runtime.kind
    manifest["executable"] = str(runtime.executable)
    manifest["source_root"] = (
        str(runtime.source_root) if runtime.source_root is not None else None
    )
    if bootstrap is None:
        manifest.pop("bootstrap_sha256", None)
    else:
        manifest["bootstrap_sha256"] = _sha256(bootstrap)
    manifest["telemetry_enabled"] = telemetry_enabled
    if traces_enabled is not None:
        manifest["traces_enabled"] = traces_enabled
    else:
        manifest.setdefault("traces_enabled", True)
    if native_hooks:
        manifest.pop("hook_scope", None)
    else:
        manifest["hook_scope"] = _HOOK_SCOPE_USER

    transaction = _Transaction()
    config_changed = local_before != (True, str(repository.managed_hooks_path))
    try:
        for (
            record,
            path,
            desired,
            raw,
            mode,
            desired_mode,
        ) in prepared_integrations:
            backup = Path(record["backup_path"])
            if raw is not None and not backup.exists():
                transaction.write(backup, raw, mode=0o600)
            if raw != desired or mode != desired_mode:
                transaction.write(path, desired, mode=desired_mode)
        exclude_backup = Path(exclude_record["backup_path"])
        if exclude_raw is not None and not exclude_backup.exists():
            transaction.write(exclude_backup, exclude_raw, mode=0o600)
        exclude_path = Path(exclude_record["path"])
        if exclude_raw != exclude_desired:
            transaction.write(exclude_path, exclude_desired, mode=exclude_mode)
        if bootstrap is not None and not repository.bootstrap_path.exists():
            transaction.write(repository.bootstrap_path, bootstrap, mode=0o700)
        for name, payload in proxy_payloads.items():
            path = repository.managed_hooks_path / name
            if path.exists():
                current, current_mode, _atime, _mtime = _file_details(path)
            else:
                current, current_mode = None, 0
            if current != payload or not current_mode & 0o111:
                transaction.write(path, payload, mode=0o755)
        if config_changed:
            _set_local_hooks_path(repository, str(repository.managed_hooks_path))
        effective_ok, _effective_value = _effective_hooks_match(
            repository, repository.managed_hooks_path
        )
        if not effective_ok:
            raise ValueError(
                "A worktree-specific core.hooksPath overrides attribution's Git hook "
                "dispatcher; remove that override before installing."
            )
        transaction.write(
            repository.manifest_path,
            _manifest_bytes(manifest),
            mode=0o600,
        )
    except BaseException:
        config_error: BaseException | None = None
        if config_changed:
            try:
                _restore_local_hooks_path(repository, local_before)
            except BaseException as exc:
                config_error = exc
        try:
            transaction.rollback()
        except BaseException as rollback_error:
            raise RuntimeError(
                f"Installation failed and rollback was incomplete: {rollback_error}"
            ) from rollback_error
        if config_error is not None:
            raise RuntimeError(
                f"Installation failed and core.hooksPath could not be restored: {config_error}"
            ) from config_error
        raise
    return installation_status(repository.root)


def heal_worktree(repo: str | Path) -> bool:
    """Provision one enrolled clone from the machine-level install.

    The harness hooks already run from user scope, so only the Git publisher,
    the exclude entries, and the manifest are missing. A clone that was
    deliberately uninstalled keeps its empty manifest and is not provisioned
    again. Returns whether this call enabled the worktree.
    """

    from .user_install import load_user_manifest

    if load_user_manifest() is None:
        return False
    repository = _repository(repo)
    manifest = _load_manifest(repository)
    if manifest is not None and (
        not _user_scope(manifest)
        or not manifest["enabled_worktrees"]
        or str(repository.git_dir) in manifest["enabled_worktrees"]
    ):
        return False
    _install_worktree(repository.root, native_hooks=False)
    return True


def install_repo(
    repo: str | Path, *, all_worktrees: bool = True, traces_enabled: bool = True
) -> dict[str, Any]:
    """Install automation throughout a repository's live worktrees.

    The per-worktree helper remains transactional. If provisioning a later
    checkout fails, checkouts newly enabled by this call are disabled again in
    reverse order.
    """

    selected = _repository(repo)
    _validate_install_scope(selected)
    telemetry_enabled = os.environ.get(_DISABLE_TELEMETRY_ENV) != "1"
    collector_was_running = False
    if telemetry_enabled:
        from .telemetry import telemetry_status

        collector_was_running = bool(telemetry_status().get("running"))
    if os.environ.get(_CURRENT_WORKTREE_ONLY_ENV) == "1":
        all_worktrees = False
    if all_worktrees:
        worktrees, discovery_warnings = _discover_worktrees(selected)
    else:
        worktrees = [_LinkedWorktree(selected, None)]
        discovery_warnings = []

    before = _load_manifest(selected)
    enabled_before = set(before["enabled_worktrees"]) if before is not None else set()
    newly_enabled: list[_Repository] = []
    try:
        for worktree in worktrees:
            candidate = worktree.repository
            _install_worktree(candidate.root, traces_enabled=traces_enabled)
            if str(candidate.git_dir) not in enabled_before:
                newly_enabled.append(candidate)
    except BaseException as install_error:
        rollback_failures: list[str] = []
        for candidate in reversed(newly_enabled):
            try:
                _uninstall_worktree(candidate.root)
            except BaseException as exc:
                rollback_failures.append(f"{candidate.root}: {exc}")
        if telemetry_enabled and not collector_was_running:
            configured_repositories: int | None = None
            registered_repositories: int | None = None
            try:
                from .telemetry_setup import telemetry_install_status

                setup_status = telemetry_install_status(str(selected.common_dir))
                codex_status = setup_status.get("codex")
                count = (
                    codex_status.get("repository_count")
                    if isinstance(codex_status, dict)
                    else None
                )
                if isinstance(count, int) and not isinstance(count, bool) and count >= 0:
                    configured_repositories = count
            except (OSError, ValueError):
                # An uncertain ownership check must preserve the shared collector.
                pass
            try:
                from .telemetry import registered_repository_count

                registered_repositories = registered_repository_count()
            except (OSError, ValueError):
                # Session registration state is also an ownership signal. If it
                # cannot be read, leave the collector running.
                pass
            if configured_repositories == 0 and registered_repositories == 0:
                try:
                    from .telemetry import stop_collector, telemetry_status

                    if telemetry_status().get("running"):
                        stop_collector()
                        if telemetry_status().get("running"):
                            rollback_failures.append(
                                "the telemetry collector could not be stopped"
                            )
                except BaseException as exc:
                    rollback_failures.append(f"telemetry collector: {exc}")
        if rollback_failures:
            raise RuntimeError(
                "Repository installation failed and rollback was incomplete: "
                + "; ".join(rollback_failures)
            ) from install_error
        raise

    status = installation_status(selected.root)
    if telemetry_enabled:
        try:
            from .telemetry_setup import configure_telemetry

            telemetry = configure_telemetry(str(selected.common_dir))
            existing_telemetry = status.get("telemetry")
            status["telemetry"] = {
                **(existing_telemetry if isinstance(existing_telemetry, dict) else {}),
                **telemetry,
            }
            claude = status["telemetry"].get("claude", {})
            if isinstance(claude, dict) and claude.get("enabled") is not True:
                message = claude.get("message")
                status["warnings"].append(
                    "Claude cost collection needs attention."
                    + (f" {message}" if isinstance(message, str) and message else "")
                )
            codex = telemetry.get("codex", {})
            if isinstance(codex, dict) and codex.get("enabled") is not True:
                message = codex.get("message")
                action = codex.get("action")
                detail = " ".join(
                    value for value in (message, action) if isinstance(value, str) and value
                )
                status["warnings"].append(
                    "Codex cost collection needs attention."
                    + (f" {detail}" if detail else "")
                )
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            status["telemetry"] = {"configured": False, "error": str(exc)}
            status["warnings"].append(
                f"Cost collection could not be configured: {exc}"
            )
    status["warnings"] = list(
        dict.fromkeys([*status["warnings"], *discovery_warnings])
    )
    return status


def _restore_recorded_file(
    transaction: _Transaction,
    repository: _Repository,
    record: dict[str, Any],
    path: Path,
    events: tuple[str, ...],
    warnings: list[str],
) -> None:
    _assert_native_target(path, repository.root)
    command = record.get("managed_command")
    if not isinstance(command, str):
        raise ValueError("Native integration record has no managed command.")
    if not path.exists():
        warnings.append(f"Native hook file was already removed: {path}.")
        return
    payload, raw, mode, _atime, _mtime = _load_json_config(path)
    assert raw is not None
    unchanged = (
        record.get("restore_preimage_when_unchanged", True)
        and record.get("installed_sha256") == _sha256(raw)
        and record.get("installed_mode", mode) == mode
    )
    if unchanged:
        if record.get("created_file"):
            transaction.delete(path)
            return
        backup = Path(str(record.get("backup_path", "")))
        _safe_private_path(backup, repository)
        backup_raw, _backup_mode, _backup_atime, _backup_mtime = _file_details(backup)
        if record.get("preimage_sha256") != _sha256(backup_raw):
            raise ValueError(f"Native hook backup failed validation: {backup}.")
        original_mode = record.get("original_mode")
        original_atime = record.get("original_atime_ns")
        original_mtime = record.get("original_mtime_ns")
        if not all(isinstance(value, int) for value in (original_mode, original_atime, original_mtime)):
            raise ValueError("Native hook backup metadata is invalid.")
        transaction.write(
            path,
            backup_raw,
            mode=original_mode,
            times=(original_atime, original_mtime),
        )
        return

    updated, removed = _remove_native_hooks(
        payload, command, events, created_file=bool(record.get("created_file"))
    )
    if record.get("harness") == "claude-code":
        updated = _remove_claude_telemetry_env(updated, record)
    if removed == 0:
        warnings.append(f"Managed hook entries were already absent from {path}.")
        if updated == payload:
            return
    if record.get("created_file") and not updated:
        transaction.delete(path)
    else:
        transaction.write(path, _json_bytes(updated), mode=mode)


def _restore_exclude(
    transaction: _Transaction,
    repository: _Repository,
    record: dict[str, Any],
    warnings: list[str],
) -> None:
    expected = repository.common_dir / "info" / "exclude"
    if record.get("path") != str(expected):
        raise ValueError("The exclude backup path is invalid.")
    if expected.exists():
        raw, mode, _atime, _mtime = _file_details(expected)
    else:
        raw, mode = b"", 0o600
    unchanged = (
        record.get("restore_preimage_when_unchanged", True)
        and record.get("installed_sha256") == _sha256(raw)
    )
    if unchanged:
        if record.get("created_file"):
            if expected.exists():
                transaction.delete(expected)
            return
        backup = Path(str(record.get("backup_path", "")))
        _safe_private_path(backup, repository)
        backup_raw, _backup_mode, _backup_atime, _backup_mtime = _file_details(backup)
        if record.get("preimage_sha256") != _sha256(backup_raw):
            raise ValueError("The Git exclude backup failed validation.")
        original_mode = record.get("original_mode")
        original_atime = record.get("original_atime_ns")
        original_mtime = record.get("original_mtime_ns")
        if not all(isinstance(value, int) for value in (original_mode, original_atime, original_mtime)):
            raise ValueError("The Git exclude backup metadata is invalid.")
        transaction.write(
            expected,
            backup_raw,
            mode=original_mode,
            times=(original_atime, original_mtime),
        )
        return
    updated, removed = _exclude_without_block(raw)
    if not removed:
        warnings.append("The attribution block was already absent from Git info/exclude.")
        return
    transaction.write(expected, updated, mode=mode)


def _uninstall_worktree(repo: str | Path) -> dict[str, Any]:
    """Remove only automation owned by attribution for one worktree."""

    repository = _repository(repo)
    manifest = _load_manifest(repository)
    if manifest is None:
        return installation_status(repository.root)
    worktree_id = str(repository.git_dir)
    if worktree_id not in manifest["enabled_worktrees"]:
        return installation_status(repository.root)

    records: list[dict[str, Any]] = []
    native_targets = () if _user_scope(manifest) else _INTEGRATION_PATHS.items()
    for harness, relative in native_targets:
        record = _integration_record(manifest, worktree_id, harness)
        if (
            record is None
            or not isinstance(record.get("path"), str)
            or not isinstance(record.get("managed_command"), str)
            or _MARKER not in record["managed_command"]
        ):
            raise ValueError(f"Cannot safely uninstall: {harness} manifest entry is missing.")
        # The worktree id is stable across `git worktree move`; always operate on
        # the native file inside the checkout Git currently resolves for it.
        record["path"] = str(repository.root / relative)
        records.append(record)

    remaining_worktrees = [
        item for item in manifest["enabled_worktrees"] if item != worktree_id
    ]
    last = not remaining_worktrees
    warnings: list[str] = []
    local_before = _local_hooks_path(repository)
    previous_record = manifest.get("previous_local_hooks_path")
    if not isinstance(previous_record, dict):
        raise ValueError("The install manifest is missing the previous Git hook setting.")
    previous_local = (
        previous_record.get("present") is True,
        previous_record.get("value"),
    )
    if previous_local[0] and not isinstance(previous_local[1], str):
        raise ValueError("The saved Git hook setting is invalid.")

    # Validate all mutable native files before changing any of them.
    for record in records:
        path = Path(record["path"])
        _assert_native_target(path, repository.root)
        if path.exists():
            payload, _raw, _mode, _atime, _mtime = _load_json_config(path)
            _validate_managed_commands(
                payload, record["managed_command"], _EVENTS[record["harness"]]
            )
    if last:
        exclude_record = manifest.get("exclude")
        if not isinstance(exclude_record, dict):
            raise ValueError("The install manifest is missing its Git exclude record.")
        exclude_path = repository.common_dir / "info" / "exclude"
        if exclude_path.exists():
            _file_details(exclude_path)

    # Settle capture state while this worktree is still enabled. Otherwise a
    # later reinstall would see the orphaned pending capture as overlap and
    # conservatively contaminate unrelated edits.
    from .automation import abandon_worktree

    abandoned = abandon_worktree(repository.root)
    abandoned_count = abandoned.get("abandoned_captures", 0)
    if isinstance(abandoned_count, int) and abandoned_count:
        noun = "capture" if abandoned_count == 1 else "captures"
        warnings.append(f"Abandoned {abandoned_count} active {noun} safely.")
    abandon_warnings = abandoned.get("warnings")
    if isinstance(abandon_warnings, list):
        warnings.extend(
            item for item in abandon_warnings if isinstance(item, str) and item
        )

    transaction = _Transaction()
    config_restored = False
    try:
        for record in records:
            _restore_recorded_file(
                transaction,
                repository,
                record,
                Path(record["path"]),
                _EVENTS[record["harness"]],
                warnings,
            )

        manifest["enabled_worktrees"] = remaining_worktrees
        manifest["integrations"] = [
            item
            for item in manifest["integrations"]
            if item.get("worktree_id") != worktree_id
        ]

        if last:
            _restore_exclude(
                transaction, repository, manifest["exclude"], warnings
            )
            managed_path = str(repository.managed_hooks_path)
            if local_before == (True, managed_path):
                _restore_local_hooks_path(repository, previous_local)
                config_restored = True
            else:
                warnings.append(
                    "core.hooksPath changed after installation; the newer value was preserved."
                )

            proxy_hashes = manifest.get("proxy_hashes", {})
            if isinstance(proxy_hashes, dict):
                for name, expected_hash in proxy_hashes.items():
                    path = repository.managed_hooks_path / name
                    _safe_private_path(path, repository)
                    if not path.exists():
                        continue
                    raw, _mode, _atime, _mtime = _file_details(path)
                    if expected_hash == _sha256(raw):
                        transaction.delete(path)
                    else:
                        warnings.append(f"Modified managed Git proxy was preserved: {path}.")
            if repository.bootstrap_path.exists():
                raw, _mode, _atime, _mtime = _file_details(repository.bootstrap_path)
                if manifest.get("bootstrap_sha256") == _sha256(raw):
                    transaction.delete(repository.bootstrap_path)
                else:
                    warnings.append("Modified private hook bootstrap was preserved.")
            manifest["proxy_hashes"] = {}
            manifest.pop("bootstrap_sha256", None)
            manifest.pop("exclude", None)

        for record in records:
            backup = Path(str(record.get("backup_path", "")))
            _safe_private_path(backup, repository)
            if backup.exists():
                transaction.delete(backup)
        if last:
            exclude_backup = repository.state_dir / "backups" / "shared" / "info-exclude"
            _safe_private_path(exclude_backup, repository)
            if exclude_backup.exists():
                transaction.delete(exclude_backup)

        transaction.write(
            repository.manifest_path,
            _manifest_bytes(manifest),
            mode=0o600,
        )
    except BaseException:
        if config_restored:
            try:
                _restore_local_hooks_path(repository, local_before)
            except BaseException as config_error:
                raise RuntimeError(
                    f"Uninstall failed and core.hooksPath rollback failed: {config_error}"
                ) from config_error
        try:
            transaction.rollback()
        except BaseException as rollback_error:
            raise RuntimeError(
                f"Uninstall failed and rollback was incomplete: {rollback_error}"
            ) from rollback_error
        raise

    status = installation_status(repository.root)
    status["warnings"] = list(dict.fromkeys([*status["warnings"], *warnings]))
    return status


def _unregister_cost_telemetry_if_unused(
    repository: _Repository, status: dict[str, Any]
) -> dict[str, Any]:
    if (
        os.environ.get(_DISABLE_TELEMETRY_ENV) == "1"
        or status.get("repository_installed") is True
    ):
        return status
    try:
        from .telemetry_setup import unregister_telemetry

        status["telemetry"] = unregister_telemetry(str(repository.common_dir))
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        status["telemetry"] = {"unregistered": False, "error": str(exc)}
        status.setdefault("warnings", []).append(
            f"Cost collection could not be removed safely: {exc}"
        )
    return status


def uninstall_repo(
    repo: str | Path, *, all_worktrees: bool = True
) -> dict[str, Any]:
    """Disable automation throughout a repository's live worktrees by default."""

    selected = _repository(repo)
    if not all_worktrees:
        return _unregister_cost_telemetry_if_unused(
            selected, _uninstall_worktree(selected.root)
        )

    manifest = _load_manifest(selected)
    if manifest is None:
        return installation_status(selected.root)
    enabled = set(manifest["enabled_worktrees"])
    worktrees, discovery_warnings = _discover_worktrees(selected)
    targets = [
        item.repository
        for item in worktrees
        if str(item.repository.git_dir) in enabled
    ]
    removed: list[_Repository] = []
    operation_warnings: list[str] = []
    try:
        for candidate in targets:
            candidate_status = _uninstall_worktree(candidate.root)
            candidate_warnings = candidate_status.get("warnings")
            if isinstance(candidate_warnings, list):
                operation_warnings.extend(
                    item
                    for item in candidate_warnings
                    if isinstance(item, str) and item
                )
            removed.append(candidate)
    except BaseException as uninstall_error:
        rollback_failures: list[str] = []
        for candidate in reversed(removed):
            try:
                _install_worktree(candidate.root)
            except BaseException as exc:
                rollback_failures.append(f"{candidate.root}: {exc}")
        if rollback_failures:
            raise RuntimeError(
                "Repository uninstall failed and rollback was incomplete: "
                + "; ".join(rollback_failures)
            ) from uninstall_error
        raise

    status = installation_status(selected.root)
    unavailable_count = len(enabled) - len(targets)
    warnings = [*status["warnings"], *operation_warnings, *discovery_warnings]
    if unavailable_count:
        noun = "worktree" if unavailable_count == 1 else "worktrees"
        warnings.append(
            f"Could not disable {unavailable_count} unavailable {noun}; "
            "shared Git hooks were preserved."
        )
    status["warnings"] = list(dict.fromkeys(warnings))
    return _unregister_cost_telemetry_if_unused(selected, status)


__all__ = ["heal_worktree", "install_repo", "uninstall_repo", "installation_status"]
