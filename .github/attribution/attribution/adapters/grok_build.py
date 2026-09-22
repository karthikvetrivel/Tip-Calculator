"""Read-only post-run metadata discovery for local Grok Build sessions.

``summary.json`` names the session, its directory, and its current model, and
``usage.json`` holds the cost.  Neither file counts a prompt, a tool call, or a
compaction, and neither names an agent or an earlier model.  ``usage.json``
also carries a ``turns`` array beside its session totals, which the records
this project has read hold empty and whose entries nothing here documents, so
its length is not read as a turn count.  This adapter therefore reports no
workflow field at all: every one of them stays ``None`` rather than being
guessed from that array or from a message count.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import stat
from typing import Mapping

from .base import AdapterSnapshot, NativeMetadata, read_stable_regular_file


HARNESS_ID = "grok-build"
MAX_SCAN_FILES = 512
MAX_SCAN_TOTAL_BYTES = 64 * 1024 * 1024
MAX_SUMMARY_BYTES = 512 * 1024
MAX_USAGE_BYTES = 2 * 1024 * 1024
MAX_TEXT_FIELD = 512
USD_TICKS_PER_USD = 10_000_000_000


@dataclass(frozen=True, slots=True)
class _FileStamp:
    size: int
    mtime_ns: int
    inode: int


@dataclass(frozen=True, slots=True)
class _GrokState:
    repo: Path
    sessions_root: Path
    files: tuple[tuple[Path, _FileStamp], ...]
    scan_complete: bool


@dataclass(frozen=True, slots=True)
class _ParsedSummary:
    session_id: str
    cwd: Path
    git_root: Path | None
    model: str | None
    directory: Path


def _sessions_root(env: Mapping[str, str]) -> Path | None:
    grok_home = env.get("GROK_HOME", "").strip()
    if grok_home:
        return (_environment_path(grok_home, env) / "sessions").resolve(strict=False)
    home = env.get("HOME", "").strip()
    if not home:
        return None
    return (Path(home) / ".grok" / "sessions").resolve(strict=False)


def _environment_path(value: str, env: Mapping[str, str]) -> Path:
    """Expand a current-user tilde against the child environment, not ours."""

    home = env.get("HOME", "").strip()
    if home and (value == "~" or value.startswith("~/")):
        value = os.fspath(Path(home) / ("" if value == "~" else value[2:]))
    return Path(value).resolve(strict=False)


def _stamp(path: Path) -> _FileStamp | None:
    try:
        metadata = path.lstat()
    except OSError:
        return None
    if not stat.S_ISREG(metadata.st_mode):
        return None
    return _FileStamp(metadata.st_size, metadata.st_mtime_ns, metadata.st_ino)


def _scan(root: Path) -> tuple[tuple[tuple[Path, _FileStamp], ...], bool]:
    """Scan the documented encoded-cwd/session-id directory shape."""

    if not root.is_dir():
        return (), True
    try:
        resolved_root = root.resolve(strict=False)
    except (OSError, RuntimeError, ValueError):
        return (), False
    candidates: list[Path] = []
    seen: set[Path] = set()
    complete = True
    try:
        for pattern in ("*/summary.json", "*/*/summary.json"):
            for path in root.glob(pattern):
                resolved = path.resolve(strict=False)
                try:
                    resolved.relative_to(resolved_root)
                except ValueError:
                    complete = False
                    break
                if resolved in seen:
                    continue
                seen.add(resolved)
                if len(candidates) == MAX_SCAN_FILES:
                    complete = False
                    break
                candidates.append(resolved)
            if not complete:
                break
    except (OSError, RuntimeError, ValueError):
        return (), False

    files: list[tuple[Path, _FileStamp]] = []
    for path in sorted(candidates, key=lambda item: os.fspath(item)):
        stamp = _stamp(path)
        if stamp is not None:
            files.append((path, stamp))
    if sum(stamp.size for _, stamp in files) > MAX_SCAN_TOTAL_BYTES:
        complete = False
    return tuple(files), complete


def snapshot(repo: str | os.PathLike[str], env: Mapping[str, str]) -> AdapterSnapshot | None:
    root = _sessions_root(env)
    if root is None:
        return None
    files, complete = _scan(root)
    return AdapterSnapshot(
        HARNESS_ID,
        _GrokState(
            repo=Path(repo).resolve(strict=False),
            sessions_root=root,
            files=files,
            scan_complete=complete,
        ),
    )


def _load_object(path: Path, maximum_bytes: int) -> dict[str, object] | None:
    content = read_stable_regular_file(path, maximum_bytes)
    if not content:
        return None
    try:
        value = json.loads(content.decode("utf-8"))
    except (OSError, UnicodeError, ValueError, RecursionError):
        return None
    return value if isinstance(value, dict) else None


def _short_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip()
    if not value or len(value) > MAX_TEXT_FIELD or any(ord(char) < 32 for char in value):
        return None
    return value


def _path(value: object) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return Path(value).resolve(strict=False)
    except (OSError, RuntimeError, ValueError):
        return None


def _parse_summary(path: Path) -> _ParsedSummary | None:
    value = _load_object(path, MAX_SUMMARY_BYTES)
    if value is None:
        return None
    info = value.get("info")
    if not isinstance(info, dict):
        return None
    session_id = _short_text(info.get("id"))
    cwd = _path(info.get("cwd"))
    if session_id is None or cwd is None:
        return None

    raw_git_root = value.get("git_root_dir")
    if raw_git_root is None:
        raw_git_root = value.get("gitRootDir")
    git_root = _path(raw_git_root)
    model = _short_text(value.get("current_model_id"))
    if model is None:
        model = _short_text(value.get("currentModelId"))
    return _ParsedSummary(
        session_id=session_id,
        cwd=cwd,
        git_root=git_root,
        model=model,
        directory=path.parent,
    )


def _parse_usage(
    path: Path, expected_session_id: str
) -> tuple[float | None, str | None, tuple[str, ...]]:
    value = _load_object(path, MAX_USAGE_BYTES)
    if value is None:
        return None, None, ("Grok Build usage.json is missing, malformed, or too large",)

    usage_session_id = value.get("sessionId")
    if usage_session_id is None:
        usage_session_id = value.get("session_id")
    if usage_session_id is not None and usage_session_id != expected_session_id:
        return None, None, ("Grok Build usage.json belongs to a different session",)

    session = value.get("session")
    if not isinstance(session, dict):
        return None, None, ("Grok Build usage.json has no session totals",)
    ticks = session.get("costUsdTicks")
    if ticks is None:
        ticks = session.get("cost_usd_ticks")

    warnings: list[str] = []
    is_partial = session.get("costIsPartial", session.get("cost_is_partial", False)) is True
    is_incomplete = session.get("usageIsIncomplete", session.get("usage_is_incomplete", False)) is True
    if is_partial:
        warnings.append("Grok Build reported a partial cost")
    if is_incomplete:
        warnings.append("Grok Build reported incomplete usage")

    if isinstance(ticks, bool) or not isinstance(ticks, int) or ticks < 0:
        if ticks is not None:
            warnings.append("Grok Build reported invalid cost ticks; cost was ignored")
        return None, None, tuple(warnings)
    try:
        cost_usd = ticks / USD_TICKS_PER_USD
    except OverflowError:
        warnings.append("Grok Build reported invalid cost ticks; cost was ignored")
        return None, None, tuple(warnings)
    if not math.isfinite(cost_usd):
        warnings.append("Grok Build reported invalid cost ticks; cost was ignored")
        return None, None, tuple(warnings)
    if is_partial or is_incomplete:
        # The current report schema has no separate completeness bit.  Do not
        # let a known subtotal masquerade as a complete session cost.
        return None, None, tuple(warnings)
    return cost_usd, "grok-usage", tuple(warnings)


def finalize(
    adapter_snapshot: AdapterSnapshot,
    repo: str | os.PathLike[str],
    env: Mapping[str, str],
) -> NativeMetadata | None:
    del env  # Discovery must use the same root captured before the child ran.
    if adapter_snapshot.harness_id != HARNESS_ID or not isinstance(adapter_snapshot.state, _GrokState):
        return None
    state = adapter_snapshot.state
    resolved_repo = Path(repo).resolve(strict=False)
    if resolved_repo != state.repo or not state.scan_complete:
        return None

    after_files, complete = _scan(state.sessions_root)
    if not complete:
        return None
    before = dict(state.files)
    changed = [path for path, stamp in after_files if before.get(path) != stamp]

    matches: list[_ParsedSummary] = []
    for path in changed:
        parsed = _parse_summary(path)
        if parsed is None:
            continue
        if parsed.cwd == state.repo or parsed.git_root == state.repo:
            matches.append(parsed)
    if len(matches) != 1:
        return None

    match = matches[0]
    if match.directory / "summary.json" in before:
        cost_usd = cost_source = None
        warnings = (
            "Grok Build resumed an existing session; cumulative native cost was ignored",
        )
    else:
        cost_usd, cost_source, warnings = _parse_usage(
            match.directory / "usage.json", match.session_id
        )
    return NativeMetadata(
        harness_id=HARNESS_ID,
        native_session_id=match.session_id,
        model=match.model,
        provider=None,
        cost_usd=cost_usd,
        cost_source=cost_source,
        warnings=warnings,
    )
