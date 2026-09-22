"""Safe machine-level setup for local cost telemetry collection.

Codex only reads OpenTelemetry configuration from the user configuration
layer, so this module manages one narrowly-scoped block in ``config.toml``.
Ownership is recorded separately as an exact block plus a repository ref-set.
That lets several repositories share the machine configuration and lets the
last unregister remove only bytes that this module wrote.

Importing this module has no filesystem or process side effects.
"""

from __future__ import annotations

from contextlib import contextmanager
import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
import tomllib
from typing import Any, Iterator, Mapping
from urllib.parse import urlsplit

from .telemetry import (
    ensure_collector,
    registered_repository_count,
    stop_collector,
    telemetry_settings,
    telemetry_status,
    unregister_repository,
)


CODEX_CONFIG_PATH_ENV = "ATTRIBUTION_CODEX_CONFIG_PATH"
CODEX_HOME_ENV = "CODEX_HOME"
MANIFEST_FILENAME = "telemetry-setup.json"
LOCK_FILENAME = "telemetry-setup.lock"
MAX_CODEX_CONFIG_BYTES = 2 * 1024 * 1024
MAX_MANIFEST_BYTES = 512 * 1024

MANAGED_BLOCK_BEGIN = "# >>> harness-attribution-otel-v1"
MANAGED_BLOCK_END = "# <<< harness-attribution-otel-v1"

_MANIFEST_VERSION = 1
_CONFIG_MODE = 0o600
_STATE_MODE = 0o700
_MAX_REPOSITORY_ID_BYTES = 16 * 1024


class TelemetrySetupError(ValueError):
    """Raised when telemetry setup cannot safely inspect or update a file."""


def _repository_id(value: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise TelemetrySetupError("repository_id must be a non-empty string")
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise TelemetrySetupError("repository_id must be valid UTF-8") from exc
    if len(encoded) > _MAX_REPOSITORY_ID_BYTES:
        raise TelemetrySetupError("repository_id is too long")
    return value


def _absolute_path(value: str | os.PathLike[str]) -> Path:
    expanded = os.path.expanduser(os.fspath(value))
    if "\x00" in expanded:
        raise TelemetrySetupError("configuration path contains a NUL byte")
    # abspath normalizes a relative test override without following a final
    # symlink.  Resolving here would defeat the explicit symlink check below.
    return Path(os.path.abspath(expanded))


def _codex_config_path(
    config_path: str | os.PathLike[str] | None,
) -> Path:
    if config_path is not None:
        return _absolute_path(config_path)
    override = os.environ.get(CODEX_CONFIG_PATH_ENV)
    if override:
        return _absolute_path(override)
    codex_home = os.environ.get(CODEX_HOME_ENV)
    if codex_home:
        return _absolute_path(codex_home) / "config.toml"
    return _absolute_path(Path.home() / ".codex" / "config.toml")


def _state_path(state_dir: str | os.PathLike[str] | None) -> Path:
    if state_dir is not None:
        # Preserve the final component until our own lstat check.  The
        # collector normalizes paths internally, but setup must not silently
        # accept a caller-supplied symlink as the manifest directory.
        return _absolute_path(state_dir)
    # telemetry_status is deliberately read-only and also resolves the default
    # directory in the same way as the collector implementation.
    status = telemetry_status(state_dir)
    value = status.get("state_dir")
    if not isinstance(value, str) or not value:
        raise TelemetrySetupError("collector returned no telemetry state directory")
    return Path(value)


def _check_existing_directory(path: Path, *, description: str) -> bool:
    """Validate an existing directory without creating or chmodding it."""

    try:
        details = path.lstat()
    except FileNotFoundError:
        return False
    if stat.S_ISLNK(details.st_mode):
        raise TelemetrySetupError(f"refusing symlink {description} at {path}")
    if not stat.S_ISDIR(details.st_mode):
        raise TelemetrySetupError(f"expected a directory for {description} at {path}")
    return True


def _check_config_parent(path: Path) -> None:
    """Reject an existing non-directory or symlink parent without writing."""

    cursor = path.parent
    while True:
        try:
            details = cursor.lstat()
        except FileNotFoundError:
            if cursor.parent == cursor:
                raise TelemetrySetupError(
                    f"could not find a safe parent directory for {path}"
                )
            cursor = cursor.parent
            continue
        if stat.S_ISLNK(details.st_mode):
            raise TelemetrySetupError(f"refusing symlink directory {cursor}")
        if not stat.S_ISDIR(details.st_mode):
            raise TelemetrySetupError(f"expected a directory at {cursor}")
        return


def _ensure_private_directory(path: Path) -> None:
    try:
        details = path.lstat()
    except FileNotFoundError:
        missing: list[Path] = []
        cursor = path
        while True:
            try:
                parent_details = cursor.lstat()
            except FileNotFoundError:
                missing.append(cursor)
                if cursor.parent == cursor:
                    raise TelemetrySetupError(
                        f"could not find a safe parent directory for {path}"
                    )
                cursor = cursor.parent
                continue
            if stat.S_ISLNK(parent_details.st_mode):
                raise TelemetrySetupError(f"refusing symlink directory {cursor}")
            if not stat.S_ISDIR(parent_details.st_mode):
                raise TelemetrySetupError(f"expected a directory at {cursor}")
            break
        for directory in reversed(missing):
            try:
                directory.mkdir(mode=_STATE_MODE)
            except FileExistsError:
                pass
            created = directory.lstat()
            if stat.S_ISLNK(created.st_mode) or not stat.S_ISDIR(created.st_mode):
                raise TelemetrySetupError(f"unsafe directory appeared at {directory}")
    else:
        if stat.S_ISLNK(details.st_mode):
            raise TelemetrySetupError(f"refusing symlink directory {path}")
        if not stat.S_ISDIR(details.st_mode):
            raise TelemetrySetupError(f"expected a directory at {path}")
    if os.name == "posix":
        os.chmod(path, _STATE_MODE, follow_symlinks=False)


def _ensure_config_parent(path: Path) -> None:
    parent = path.parent
    try:
        details = parent.lstat()
    except FileNotFoundError:
        _ensure_private_directory(parent)
        return
    if stat.S_ISLNK(details.st_mode):
        raise TelemetrySetupError(f"refusing symlink directory {parent}")
    if not stat.S_ISDIR(details.st_mode):
        raise TelemetrySetupError(f"expected a directory at {parent}")


def _same_file(left: os.stat_result, right: os.stat_result) -> bool:
    return left.st_dev == right.st_dev and left.st_ino == right.st_ino


def _read_regular_file(
    path: Path,
    *,
    maximum: int,
    description: str,
) -> tuple[bytes, int] | None:
    """Read a bounded regular file without following a final symlink."""

    try:
        before = path.lstat()
    except FileNotFoundError:
        return None
    if stat.S_ISLNK(before.st_mode):
        raise TelemetrySetupError(f"refusing symlink {description} at {path}")
    if not stat.S_ISREG(before.st_mode):
        raise TelemetrySetupError(f"expected a regular {description} at {path}")
    if before.st_size > maximum:
        raise TelemetrySetupError(
            f"refusing oversized {description} at {path} (limit: {maximum} bytes)"
        )

    flags = os.O_RDONLY
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise TelemetrySetupError(f"refusing symlink {description} at {path}") from exc
        raise
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise TelemetrySetupError(f"expected a regular {description} at {path}")
        if not _same_file(before, opened):
            raise TelemetrySetupError(f"{description} changed while it was being opened: {path}")
        if opened.st_size > maximum:
            raise TelemetrySetupError(
                f"refusing oversized {description} at {path} (limit: {maximum} bytes)"
            )
        with os.fdopen(descriptor, "rb", closefd=True) as source:
            descriptor = -1
            data = source.read(maximum + 1)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if len(data) > maximum:
        raise TelemetrySetupError(
            f"refusing oversized {description} at {path} (limit: {maximum} bytes)"
        )
    return data, stat.S_IMODE(opened.st_mode)


def _assert_config_unchanged(
    path: Path,
    expected: tuple[bytes, int] | None,
) -> None:
    """Fail closed if another process replaced or edited the Codex config."""

    observed = _read_regular_file(
        path,
        maximum=MAX_CODEX_CONFIG_BYTES,
        description="Codex config",
    )
    expected_data = expected[0] if expected is not None else None
    observed_data = observed[0] if observed is not None else None
    if observed_data != expected_data:
        raise TelemetrySetupError(
            f"Codex config changed concurrently; refusing to overwrite it: {path}"
        )


def _fsync_directory(directory: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    directory_flag = getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(directory, flags | directory_flag)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write(path: Path, data: bytes, *, mode: int = _CONFIG_MODE) -> None:
    _ensure_config_parent(path)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.attribution-", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "wb", closefd=True) as output:
            descriptor = -1
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        if os.name == "posix":
            os.chmod(path, mode, follow_symlinks=False)
        _fsync_directory(path.parent)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _atomic_delete(path: Path) -> None:
    path.unlink()
    _fsync_directory(path.parent)


@contextmanager
def _setup_lock(state_dir: Path) -> Iterator[None]:
    _ensure_private_directory(state_dir)
    path = state_dir / LOCK_FILENAME
    flags = os.O_RDWR | os.O_CREAT
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, _CONFIG_MODE)
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise TelemetrySetupError(f"refusing symlink setup lock at {path}") from exc
        raise
    try:
        details = os.fstat(descriptor)
        if not stat.S_ISREG(details.st_mode):
            raise TelemetrySetupError(f"expected a regular setup lock at {path}")
        current = path.lstat()
        if stat.S_ISLNK(current.st_mode) or not _same_file(details, current):
            raise TelemetrySetupError(f"unsafe setup lock at {path}")
        os.fchmod(descriptor, _CONFIG_MODE)
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _parse_toml(data: bytes, path: Path) -> dict[str, Any]:
    try:
        text = data.decode("utf-8")
        parsed = tomllib.loads(text)
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise TelemetrySetupError(f"refusing malformed TOML config at {path}: {exc}") from exc
    if not isinstance(parsed, dict):  # pragma: no cover - guaranteed by tomllib
        raise TelemetrySetupError(f"expected a TOML document at {path}")
    return parsed


def _toml_string(value: str) -> str:
    # JSON basic strings are a strict-enough subset for the endpoint and bearer
    # values used here and correctly escape control characters and quotes.
    return json.dumps(value, ensure_ascii=True)


def _validate_collector_settings(settings: Mapping[str, Any]) -> tuple[str, str]:
    endpoint = settings.get("endpoint")
    authorization = settings.get("authorization_header")
    if not isinstance(endpoint, str) or not isinstance(authorization, str):
        raise TelemetrySetupError("collector returned incomplete telemetry settings")
    parsed = urlsplit(endpoint)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path != "/v1/logs"
        or parsed.query
        or parsed.fragment
    ):
        raise TelemetrySetupError("collector endpoint must be a local HTTP /v1/logs URL")
    if not authorization.startswith("Bearer ") or len(authorization) <= len("Bearer "):
        raise TelemetrySetupError("collector returned an invalid bearer authorization value")
    if "\r" in authorization or "\n" in authorization:
        raise TelemetrySetupError("collector returned an invalid bearer authorization value")
    return endpoint, authorization


def _managed_block(settings: Mapping[str, Any]) -> bytes:
    endpoint, authorization = _validate_collector_settings(settings)
    text = (
        f"{MANAGED_BLOCK_BEGIN}\n"
        "[otel]\n"
        'environment = "harness-attribution"\n'
        "log_user_prompt = false\n"
        'metrics_exporter = "none"\n'
        'trace_exporter = "none"\n'
        "exporter = { otlp-http = { "
        f"endpoint = {_toml_string(endpoint)}, protocol = \"json\", "
        f"headers = {{ Authorization = {_toml_string(authorization)} }} }} }}\n"
        f"{MANAGED_BLOCK_END}\n"
    )
    data = text.encode("utf-8")
    # Treat our own output as untrusted until tomllib accepts it too.
    parsed = _parse_toml(data, Path("<generated attribution OTel block>"))
    if "otel" not in parsed:
        raise TelemetrySetupError("generated Codex telemetry block has no otel table")
    return data


def _append_block(original: bytes, block: bytes) -> tuple[bytes, bytes]:
    if not original:
        insertion = block
    elif original.endswith(b"\n"):
        insertion = b"\n" + block
    else:
        insertion = b"\n\n" + block
    return original + insertion, insertion


def _manifest_path(state_dir: Path) -> Path:
    return state_dir / MANIFEST_FILENAME


def _load_manifest(state_dir: Path) -> dict[str, Any] | None:
    path = _manifest_path(state_dir)
    result = _read_regular_file(
        path, maximum=MAX_MANIFEST_BYTES, description="telemetry setup manifest"
    )
    if result is None:
        return None
    data, mode = result
    if os.name == "posix" and mode & 0o077:
        raise TelemetrySetupError(f"telemetry setup manifest is not private: {path}")
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TelemetrySetupError(f"malformed telemetry setup manifest at {path}") from exc
    if not isinstance(value, dict) or value.get("version") != _MANIFEST_VERSION:
        raise TelemetrySetupError(f"unsupported telemetry setup manifest at {path}")
    repositories = value.get("repositories")
    codex = value.get("codex")
    if (
        not isinstance(repositories, list)
        or not repositories
        or any(not isinstance(item, str) or not item for item in repositories)
        or len(set(repositories)) != len(repositories)
        or not isinstance(codex, dict)
        or not isinstance(codex.get("config_path"), str)
        or not isinstance(codex.get("block"), str)
        or not isinstance(codex.get("block_sha256"), str)
        or not isinstance(codex.get("created_file"), bool)
        or (
            codex.get("original_mode") is not None
            and not isinstance(codex.get("original_mode"), int)
        )
    ):
        raise TelemetrySetupError(f"invalid telemetry setup manifest at {path}")
    try:
        block = codex["block"].encode("utf-8")
    except UnicodeEncodeError as exc:
        raise TelemetrySetupError(f"invalid telemetry setup manifest at {path}") from exc
    digest = hashlib.sha256(block).hexdigest()
    if digest != codex["block_sha256"]:
        raise TelemetrySetupError(f"telemetry setup manifest checksum failed at {path}")
    if not (
        codex["block"].lstrip().startswith(MANAGED_BLOCK_BEGIN)
        and codex["block"].rstrip().endswith(MANAGED_BLOCK_END)
    ):
        raise TelemetrySetupError(f"invalid managed block in telemetry setup manifest at {path}")
    return value


def _manifest_bytes(manifest: Mapping[str, Any]) -> bytes:
    return (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _new_manifest(
    repository_id: str,
    config_path: Path,
    insertion: bytes,
    *,
    created_file: bool,
    original_mode: int | None,
) -> dict[str, Any]:
    block = insertion.decode("utf-8")
    return {
        "version": _MANIFEST_VERSION,
        "repositories": [repository_id],
        "codex": {
            "config_path": str(config_path),
            "block": block,
            "block_sha256": hashlib.sha256(insertion).hexdigest(),
            "created_file": created_file,
            "original_mode": original_mode,
        },
    }


def _write_manifest(state_dir: Path, manifest: Mapping[str, Any]) -> None:
    data = _manifest_bytes(manifest)
    if len(data) > MAX_MANIFEST_BYTES:
        raise TelemetrySetupError("telemetry setup manifest would exceed its size limit")
    _atomic_write(_manifest_path(state_dir), data, mode=_CONFIG_MODE)


def _collector_result(settings: Mapping[str, Any]) -> dict[str, Any]:
    running = bool(settings.get("running"))
    return {
        "enabled": running,
        "state": "enabled" if running else "stopped",
        "running": running,
        "pid": settings.get("pid") if isinstance(settings.get("pid"), int) else None,
        "endpoint": settings.get("endpoint")
        if isinstance(settings.get("endpoint"), str)
        else None,
    }


def _codex_result(
    config_path: Path,
    *,
    state: str,
    enabled: bool,
    managed: bool,
    repository_count: int,
    changed: bool = False,
    message: str | None = None,
    action: str | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "enabled": enabled,
        "state": state,
        "managed": managed,
        "config_path": str(config_path),
        "repository_count": repository_count,
        "changed": changed,
    }
    if message is not None:
        result["message"] = message
    if action is not None:
        result["action"] = action
    return result


def _conflict(
    config_path: Path,
    *,
    message: str,
    action: str,
    repository_count: int = 0,
    managed: bool = False,
) -> dict[str, Any]:
    return _codex_result(
        config_path,
        state="conflict",
        enabled=False,
        managed=managed,
        repository_count=repository_count,
        message=message,
        action=action,
    )


def claude_telemetry_env(
    *, state_dir: str | os.PathLike[str] | None = None
) -> dict[str, str]:
    """Return privacy-preserving Claude Code ``settings.json`` env values.

    The values export log events only.  They deliberately disable metrics,
    traces, prompt text, tool arguments/content, and raw API bodies.
    """

    settings = telemetry_settings(state_dir)
    endpoint, _ = _validate_collector_settings(settings)
    header = settings.get("header")
    if not isinstance(header, str) or not header.startswith("Authorization=Bearer%20"):
        raise TelemetrySetupError("collector returned an invalid OTLP header value")
    if "\r" in header or "\n" in header:
        raise TelemetrySetupError("collector returned an invalid OTLP header value")
    return {
        "CLAUDE_CODE_ENABLE_TELEMETRY": "1",
        "CLAUDE_CODE_ENHANCED_TELEMETRY_BETA": "0",
        "OTEL_LOGS_EXPORTER": "otlp",
        "OTEL_METRICS_EXPORTER": "none",
        "OTEL_TRACES_EXPORTER": "none",
        "OTEL_EXPORTER_OTLP_PROTOCOL": "http/json",
        "OTEL_EXPORTER_OTLP_LOGS_PROTOCOL": "http/json",
        "OTEL_EXPORTER_OTLP_LOGS_ENDPOINT": endpoint,
        "OTEL_EXPORTER_OTLP_HEADERS": header,
        "OTEL_LOG_USER_PROMPTS": "0",
        "OTEL_LOG_TOOL_DETAILS": "0",
        "OTEL_LOG_TOOL_CONTENT": "0",
        "OTEL_LOG_RAW_API_BODIES": "0",
    }


def configure_telemetry(
    repository_id: str,
    *,
    state_dir: str | os.PathLike[str] | None = None,
    config_path: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Start the collector and safely enable Codex telemetry for a repository."""

    repository = _repository_id(repository_id)
    config = _codex_config_path(config_path)
    state = _state_path(state_dir)
    restart_required = False

    with _setup_lock(state):
        manifest = _load_manifest(state)
        current_result = _read_regular_file(
            config,
            maximum=MAX_CODEX_CONFIG_BYTES,
            description="Codex config",
        )
        _check_config_parent(config)
        current = current_result[0] if current_result is not None else b""
        current_mode = current_result[1] if current_result is not None else None
        parsed = _parse_toml(current, config)

        if manifest is not None:
            recorded = manifest["codex"]
            repositories = list(manifest["repositories"])
            recorded_path = Path(recorded["config_path"])
            if recorded_path != config:
                collector = ensure_collector(state)
                codex = _conflict(
                    config,
                    message=(
                        f"Joyride already manages Codex telemetry at {recorded_path}; "
                        f"the requested path is {config}."
                    ),
                    action=(
                        "Use the recorded config path, or unregister every repository "
                        "before choosing a different path."
                    ),
                    repository_count=len(repositories),
                    managed=True,
                )
                return {
                    "configured": False,
                    "repository_id": repository,
                    "collector": _collector_result(collector),
                    "codex": codex,
                    "restart_required": False,
                }

            owned = recorded["block"].encode("utf-8")
            if current.count(owned) != 1:
                collector = ensure_collector(state)
                codex = _conflict(
                    config,
                    message=(
                        "The recorded attribution-managed OTel block is missing or was "
                        "modified; no Codex configuration was overwritten."
                    ),
                    action=(
                        "Restore the managed block exactly or remove the conflicting "
                        "top-level [otel] configuration manually, then retry."
                    ),
                    repository_count=len(repositories),
                    managed=True,
                )
                return {
                    "configured": False,
                    "repository_id": repository,
                    "collector": _collector_result(collector),
                    "codex": codex,
                    "restart_required": False,
                }

            collector = ensure_collector(state)
            expected_block = _managed_block(collector)
            config_changed = False
            if owned != expected_block:
                marker_offset = owned.find(MANAGED_BLOCK_BEGIN.encode("utf-8"))
                if marker_offset < 0:  # guarded by manifest validation
                    raise TelemetrySetupError("managed OTel block marker is missing")
                expected_insertion = owned[:marker_offset] + expected_block
                updated = current.replace(owned, expected_insertion, 1)
                _parse_toml(updated, config)
                _assert_config_unchanged(config, current_result)
                _atomic_write(config, updated, mode=_CONFIG_MODE)
                config_changed = True
                restart_required = True
                recorded["block"] = expected_insertion.decode("utf-8")
                recorded["block_sha256"] = hashlib.sha256(expected_insertion).hexdigest()

            repository_added = repository not in repositories
            if repository_added:
                repositories.append(repository)
                repositories.sort()
                manifest["repositories"] = repositories
            if config_changed or repository_added:
                try:
                    _write_manifest(state, manifest)
                except BaseException:
                    if config_changed:
                        _assert_config_unchanged(
                            config,
                            (updated, _CONFIG_MODE),
                        )
                        _atomic_write(config, current, mode=current_mode or _CONFIG_MODE)
                    raise
            codex = _codex_result(
                config,
                state="enabled",
                enabled=True,
                managed=True,
                repository_count=len(repositories),
                changed=config_changed or repository_added,
            )
            return {
                "configured": True,
                "repository_id": repository,
                "collector": _collector_result(collector),
                "codex": codex,
                "restart_required": restart_required,
            }

        if "otel" in parsed:
            collector = ensure_collector(state)
            codex = _conflict(
                config,
                message=(
                    f"Codex config {config} already defines top-level OTel settings; "
                    "attribution left them unchanged."
                ),
                action=(
                    "Remove or manually reconcile the existing [otel] configuration, "
                    "then run telemetry configuration again."
                ),
            )
            return {
                "configured": False,
                "repository_id": repository,
                "collector": _collector_result(collector),
                "codex": codex,
                "restart_required": False,
            }

        collector = ensure_collector(state)
        block = _managed_block(collector)
        updated, insertion = _append_block(current, block)
        _parse_toml(updated, config)
        new_manifest = _new_manifest(
            repository,
            config,
            insertion,
            created_file=current_result is None,
            original_mode=current_mode,
        )
        _assert_config_unchanged(config, current_result)
        _atomic_write(config, updated, mode=_CONFIG_MODE)
        try:
            _write_manifest(state, new_manifest)
        except BaseException:
            if current_result is None:
                _assert_config_unchanged(config, (updated, _CONFIG_MODE))
                try:
                    _atomic_delete(config)
                except FileNotFoundError:
                    pass
            else:
                _assert_config_unchanged(config, (updated, _CONFIG_MODE))
                _atomic_write(config, current, mode=current_mode or _CONFIG_MODE)
            raise

        restart_required = True
        codex = _codex_result(
            config,
            state="enabled",
            enabled=True,
            managed=True,
            repository_count=1,
            changed=True,
        )
        return {
            "configured": True,
            "repository_id": repository,
            "collector": _collector_result(collector),
            "codex": codex,
            "restart_required": restart_required,
        }


def _read_only_codex_status(
    repository: str,
    state: Path,
    config: Path,
) -> dict[str, Any]:
    manifest = _load_manifest(state) if state.is_dir() else None
    _check_config_parent(config)
    current_result = _read_regular_file(
        config,
        maximum=MAX_CODEX_CONFIG_BYTES,
        description="Codex config",
    )
    current = current_result[0] if current_result is not None else b""
    parsed = _parse_toml(current, config)

    if manifest is None:
        if "otel" in parsed:
            return _conflict(
                config,
                message=(
                    f"Codex config {config} contains OTel settings not owned by attribution."
                ),
                action=(
                    "Keep the existing integration, or remove it before asking attribution "
                    "to configure Codex telemetry."
                ),
            )
        return _codex_result(
            config,
            state="disabled",
            enabled=False,
            managed=False,
            repository_count=0,
        )

    recorded = manifest["codex"]
    repositories = manifest["repositories"]
    if Path(recorded["config_path"]) != config:
        return _conflict(
            config,
            message=(
                f"Joyride manages Codex telemetry at {recorded['config_path']}, not {config}."
            ),
            action="Use the recorded config path when checking or removing telemetry.",
            repository_count=len(repositories),
            managed=True,
        )
    owned = recorded["block"].encode("utf-8")
    if current.count(owned) != 1:
        return _conflict(
            config,
            message="The attribution-managed OTel block is missing or modified.",
            action="Resolve the top-level [otel] configuration manually before continuing.",
            repository_count=len(repositories),
            managed=True,
        )
    registered = repository in repositories
    return _codex_result(
        config,
        state="enabled" if registered else "not_registered",
        enabled=registered,
        managed=True,
        repository_count=len(repositories),
    )


def telemetry_install_status(
    repository_id: str,
    *,
    state_dir: str | os.PathLike[str] | None = None,
    config_path: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Report collector and machine configuration state without writing files."""

    repository = _repository_id(repository_id)
    config = _codex_config_path(config_path)
    collector_status = telemetry_status(state_dir)
    state_value = collector_status.get("state_dir")
    if not isinstance(state_value, str) or not state_value:
        raise TelemetrySetupError("collector returned no telemetry state directory")
    state = _state_path(state_dir)
    state_exists = _check_existing_directory(
        state, description="telemetry state directory"
    )
    codex = _read_only_codex_status(repository, state, config)
    return {
        "configured": bool(codex["enabled"]),
        "repository_id": repository,
        "collector": _collector_result(collector_status),
        "codex": codex,
        "restart_required": False,
    }


def unregister_telemetry(
    repository_id: str,
    *,
    state_dir: str | os.PathLike[str] | None = None,
    config_path: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Unregister a repository and remove owned Codex config on the last ref."""

    repository = _repository_id(repository_id)
    config = _codex_config_path(config_path)
    collector_before = telemetry_status(state_dir)
    state_value = collector_before.get("state_dir")
    if not isinstance(state_value, str) or not state_value:
        raise TelemetrySetupError("collector returned no telemetry state directory")
    state = _state_path(state_dir)
    changed = False
    conflict: dict[str, Any] | None = None
    repository_count = 0

    state_exists = _check_existing_directory(
        state, description="telemetry state directory"
    )
    if state_exists:
        with _setup_lock(state):
            manifest = _load_manifest(state)
            if manifest is not None:
                repositories = list(manifest["repositories"])
                repository_count = len(repositories)
                recorded = manifest["codex"]
                recorded_path = Path(recorded["config_path"])
                if recorded_path != config:
                    conflict = _conflict(
                        config,
                        message=(
                            f"Joyride manages Codex telemetry at {recorded_path}; "
                            f"the requested path is {config}."
                        ),
                        action="Retry with the recorded config path.",
                        repository_count=repository_count,
                        managed=True,
                    )
                elif repository in repositories and len(repositories) > 1:
                    repositories.remove(repository)
                    manifest["repositories"] = repositories
                    _write_manifest(state, manifest)
                    repository_count = len(repositories)
                    changed = True
                elif repository in repositories:
                    current_result = _read_regular_file(
                        config,
                        maximum=MAX_CODEX_CONFIG_BYTES,
                        description="Codex config",
                    )
                    current = current_result[0] if current_result is not None else b""
                    current_mode = current_result[1] if current_result is not None else None
                    _parse_toml(current, config)
                    owned = recorded["block"].encode("utf-8")
                    if current.count(owned) != 1:
                        conflict = _conflict(
                            config,
                            message=(
                                "The exact attribution-managed OTel block is missing or was "
                                "modified, so no Codex configuration was removed."
                            ),
                            action=(
                                "Remove or repair the top-level [otel] configuration manually, "
                                "then retry unregistering."
                            ),
                            repository_count=1,
                            managed=True,
                        )
                    else:
                        remaining = current.replace(owned, b"", 1)
                        _parse_toml(remaining, config)
                        created_file = bool(recorded["created_file"])
                        delete_config = created_file and remaining == b""
                        if delete_config:
                            _assert_config_unchanged(config, current_result)
                            _atomic_delete(config)
                            installed_result: tuple[bytes, int] | None = None
                        else:
                            target_mode = current_mode or _CONFIG_MODE
                            original_mode = recorded.get("original_mode")
                            if (
                                not created_file
                                and current_mode == _CONFIG_MODE
                                and isinstance(original_mode, int)
                            ):
                                target_mode = original_mode
                            _assert_config_unchanged(config, current_result)
                            _atomic_write(config, remaining, mode=target_mode)
                            installed_result = (remaining, target_mode)
                        try:
                            _atomic_delete(_manifest_path(state))
                        except BaseException:
                            _assert_config_unchanged(config, installed_result)
                            _atomic_write(config, current, mode=current_mode or _CONFIG_MODE)
                            raise
                        repository_count = 0
                        changed = True

    registrations_removed = (
        unregister_repository(repository, state) if state_exists else 0
    )
    if (
        state_exists
        and conflict is None
        and repository_count == 0
        and (changed or registered_repository_count(state) == 0)
    ):
        stop_collector(state)
    collector_after = telemetry_status(state) if state_exists else collector_before
    if conflict is not None:
        codex = conflict
    elif repository_count:
        codex = _codex_result(
            config,
            state="enabled",
            enabled=True,
            managed=True,
            repository_count=repository_count,
            changed=changed,
            message=(
                f"Codex telemetry remains enabled for {repository_count} other "
                f"repositor{'y' if repository_count == 1 else 'ies'}."
            ),
        )
    else:
        codex = _codex_result(
            config,
            state="disabled",
            enabled=False,
            managed=False,
            repository_count=0,
            changed=changed,
        )
    return {
        "unregistered": conflict is None,
        "repository_id": repository,
        "registrations_removed": registrations_removed,
        "collector": _collector_result(collector_after),
        "codex": codex,
        "restart_required": changed and repository_count == 0,
    }


__all__ = [
    "CODEX_CONFIG_PATH_ENV",
    "CODEX_HOME_ENV",
    "MANAGED_BLOCK_BEGIN",
    "MANAGED_BLOCK_END",
    "MAX_CODEX_CONFIG_BYTES",
    "TelemetrySetupError",
    "claude_telemetry_env",
    "configure_telemetry",
    "telemetry_install_status",
    "unregister_telemetry",
]
