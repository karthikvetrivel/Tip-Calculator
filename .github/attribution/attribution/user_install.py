"""Machine-level installation of the native harness hooks at user scope.

Claude Code applies ``~/.claude/settings.json`` to every project it opens and
Codex reads ``~/.codex/hooks.json`` the same way, so one install per machine
covers every clone. The repository installer in :mod:`attribution.install`
still owns the Git publisher and the per-clone manifest; this module writes
only the two user-scope configuration files and its own small manifest.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .install import (
    _EVENTS,
    _MARKER,
    _assert_native_target,
    _atomic_write,
    _config_health,
    _file_details,
    _json_bytes,
    _load_json_config,
    _merge_native_hooks,
    _remove_native_hooks,
    _runtime,
)
from .runtime import RuntimeCommand


_VERSION = 1
_NOTICE = (
    "Restart existing coding sessions so they load the machine hooks. "
    "Review the hooks in Codex when prompted."
)
USER_PATHS = {
    "codex": Path(".codex/hooks.json"),
    "claude-code": Path(".claude/settings.json"),
}


def _home() -> Path:
    return Path(os.path.abspath(Path.home()))


def user_manifest_path() -> Path:
    """Return the machine-level manifest path without creating it."""

    return _home() / ".attribution" / "user-install.json"


def _config_path(harness: str) -> Path:
    return _home() / USER_PATHS[harness]


def _managed_command(harness: str, runtime: RuntimeCommand) -> str:
    return f"{runtime.cli(('_hook', '--harness', harness)).shell()} {_MARKER}"


def _write(path: Path, data: bytes, *, mode: int) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    _atomic_write(path, data, mode=mode)


def load_user_manifest() -> dict[str, Any] | None:
    """Return the machine-level manifest, or ``None`` when none exists."""

    path = user_manifest_path()
    if not path.exists():
        return None
    raw, _mode, _atime, _mtime = _file_details(path)
    try:
        manifest = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"The machine install manifest is malformed: {exc}.") from exc
    if not isinstance(manifest, dict) or manifest.get("version") != _VERSION:
        raise ValueError("The machine install manifest has an unsupported version.")
    integrations = manifest.get("integrations")
    if not isinstance(integrations, dict) or any(
        not isinstance(item, dict) for item in integrations.values()
    ):
        raise ValueError("The machine install manifest has invalid entries.")
    return manifest


def _record(manifest: dict[str, Any] | None, harness: str) -> dict[str, Any] | None:
    if manifest is None:
        return None
    record = manifest["integrations"].get(harness)
    return record if isinstance(record, dict) else None


def install_user_hooks() -> dict[str, Any]:
    """Write the harness hooks once for this machine and every repository."""

    runtime = _runtime()
    previous = load_user_manifest()
    home = _home()
    manifest: dict[str, Any] = {
        "version": _VERSION,
        "runtime_kind": runtime.kind,
        "executable": str(runtime.executable),
        "source_root": (
            str(runtime.source_root) if runtime.source_root is not None else None
        ),
        "integrations": {},
    }
    prepared: list[tuple[Path, bytes, int]] = []
    for harness in USER_PATHS:
        path = _config_path(harness)
        _assert_native_target(path, home)
        payload, raw, mode, _atime, _mtime = _load_json_config(path)
        command = _managed_command(harness, runtime)
        record = _record(previous, harness)
        installed = record.get("managed_command") if record is not None else None
        if isinstance(installed, str) and installed != command:
            # A moved checkout or a new runtime changes the command. Retire the
            # command this installer wrote before merging the current one.
            payload, _removed = _remove_native_hooks(
                payload, installed, _EVENTS[harness], created_file=False
            )
        desired = _json_bytes(_merge_native_hooks(payload, command, _EVENTS[harness]))
        manifest["integrations"][harness] = {
            "path": str(path),
            "managed_command": command,
            "created_file": (
                bool(record.get("created_file")) if record is not None else raw is None
            ),
        }
        if raw != desired:
            prepared.append((path, desired, mode))
    # Record ownership before the hooks exist so that an interrupted install
    # still leaves an uninstall that can remove every command it wrote.
    _write(user_manifest_path(), _json_bytes(manifest), mode=0o600)
    for path, desired, mode in prepared:
        _write(path, desired, mode=mode)
    return user_install_status()


def uninstall_user_hooks() -> dict[str, Any]:
    """Remove only the machine-level hook commands that this installer wrote."""

    manifest = load_user_manifest()
    if manifest is None:
        return user_install_status()
    warnings: list[str] = []
    for harness in USER_PATHS:
        record = _record(manifest, harness)
        command = record.get("managed_command") if record is not None else None
        if not isinstance(command, str) or _MARKER not in command:
            continue
        path = _config_path(harness)
        payload, raw, mode, _atime, _mtime = _load_json_config(path)
        if raw is None:
            continue
        created_file = bool(record.get("created_file"))
        # Pruning the containers that become empty undoes exactly what the
        # merge created, because the merge is what added each managed event.
        updated, removed = _remove_native_hooks(
            payload, command, _EVENTS[harness], created_file=True
        )
        if removed == 0:
            warnings.append(f"Managed hook entries were already absent from {path}.")
        if created_file and not updated:
            path.unlink()
        elif updated != payload:
            _write(path, _json_bytes(updated), mode=mode)
    user_manifest_path().unlink(missing_ok=True)
    status = user_install_status()
    status["warnings"] = warnings
    return status


def user_hook_covers(harness: str, event: str | None) -> bool:
    """Return whether one healthy machine hook owns this event."""

    if event not in _EVENTS.get(harness, ()):
        return False
    try:
        manifest = load_user_manifest()
        record = _record(manifest, harness)
        command = record.get("managed_command") if record is not None else None
        if not isinstance(command, str):
            return False
        # A machine hook whose runtime is gone records nothing, so it owns no event.
        executable, source_root = manifest.get("executable"), manifest.get("source_root")
        if not isinstance(executable, str) or not os.access(executable, os.X_OK):
            return False
        if source_root is not None and not (isinstance(source_root, str) and os.path.isdir(source_root)):
            return False
        healthy, _message = _config_health(
            _config_path(harness), command, _EVENTS[harness], harness, only_event=event
        )
        return healthy
    except (OSError, ValueError):
        return False


def user_hook_health() -> dict[str, dict[str, Any]]:
    """Report each harness's machine-level hook state without changing it."""

    try:
        manifest = load_user_manifest()
    except (OSError, ValueError) as exc:
        message = str(exc)
        return {
            harness: {
                "installed": False,
                "state": "needs-attention",
                "message": message,
            }
            for harness in USER_PATHS
        }
    health: dict[str, dict[str, Any]] = {}
    for harness in USER_PATHS:
        record = _record(manifest, harness)
        command = record.get("managed_command") if record is not None else None
        if not isinstance(command, str):
            health[harness] = {
                "installed": False,
                "state": "not-installed",
                "message": "Machine-level hooks are not installed.",
            }
            continue
        healthy, message = _config_health(
            _config_path(harness), command, _EVENTS[harness], harness
        )
        health[harness] = {
            "installed": healthy,
            "state": "enabled" if healthy else "needs-attention",
            "message": message,
        }
    return health


def user_install_status() -> dict[str, Any]:
    """Return read-only machine-level installation state."""

    harnesses = user_hook_health()
    return {
        "installed": all(item["installed"] for item in harnesses.values()),
        "scope": "user",
        "manifest_path": str(user_manifest_path()),
        "harnesses": harnesses,
        "warnings": [],
        "notice": _NOTICE,
    }


__all__ = [
    "install_user_hooks",
    "load_user_manifest",
    "uninstall_user_hooks",
    "user_hook_covers",
    "user_hook_health",
    "user_install_status",
    "user_manifest_path",
]
