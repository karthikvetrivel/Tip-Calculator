"""Read-only post-run metadata discovery for the Pi coding agent.

Pi stores append-only JSONL session trees.  This adapter records only file
metadata before a wrapped run, then parses the one changed session belonging to
the repository after the run.  Prompt, response, and tool content is never
retained or returned.

The message roles on the active branch also give the prompts and turns of the
wrapped run, and the assistant messages give the models it used in order.
``parentId`` links one message to the message before it on its branch.  It is
not a subagent tree, so this adapter reports no agent type and no session
source, and a Pi session never nests under another one here.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import stat
from typing import Mapping

from .base import AdapterSnapshot, NativeMetadata, read_stable_regular_file


HARNESS_ID = "pi"
MAX_SCAN_FILES = 512
MAX_SESSION_BYTES = 8 * 1024 * 1024
MAX_BASELINE_TOTAL_BYTES = 64 * 1024 * 1024
MAX_JSONL_ENTRIES = 100_000
MAX_TEXT_FIELD = 512


@dataclass(frozen=True, slots=True)
class _FileStamp:
    size: int
    mtime_ns: int
    inode: int


@dataclass(frozen=True, slots=True)
class _PiState:
    repo: Path
    session_root: Path
    files: tuple[tuple[Path, "_PiBaseline"], ...]
    scan_complete: bool


@dataclass(frozen=True, slots=True)
class _PiBaseline:
    stamp: _FileStamp
    size: int
    digest: str
    session_id: str | None
    cwd: Path | None
    leaf_id: str | None
    parse_valid: bool


@dataclass(frozen=True, slots=True)
class _AssistantMetadata:
    provider: str | None
    model: str | None
    cost_usd: float | None
    cost_warning: str | None


@dataclass(frozen=True, slots=True)
class _EntryMetadata:
    parent_id: str | None
    role: str | None
    assistant: _AssistantMetadata | None
    byte_offset: int


@dataclass(frozen=True, slots=True)
class _ParsedSession:
    cwd: Path
    session_id: str
    new_assistants: tuple[_AssistantMetadata, ...]
    branch_ambiguous: bool
    model: str | None
    provider: str | None
    cost_usd: float | None
    cost_warnings: tuple[str, ...]
    prompt_count: int | None
    turn_count: int | None
    models: tuple[str, ...] | None


@dataclass(frozen=True, slots=True)
class _ParsedTree:
    cwd: Path
    session_id: str
    leaf_id: str | None
    entries: dict[str, _EntryMetadata]


def _session_root(env: Mapping[str, str]) -> Path | None:
    explicit = env.get("PI_CODING_AGENT_SESSION_DIR", "").strip()
    if explicit:
        return _environment_path(explicit, env)

    agent_dir = env.get("PI_CODING_AGENT_DIR", "").strip()
    if agent_dir:
        return (_environment_path(agent_dir, env) / "sessions").resolve(strict=False)

    home = env.get("HOME", "").strip()
    if not home:
        return None
    return (Path(home) / ".pi" / "agent" / "sessions").resolve(strict=False)


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
    """Scan Pi's documented direct and per-cwd session locations."""

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
        for pattern in ("*.jsonl", "*/*.jsonl"):
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
    return tuple(files), complete


def snapshot(repo: str | os.PathLike[str], env: Mapping[str, str]) -> AdapterSnapshot | None:
    root = _session_root(env)
    if root is None:
        return None
    scanned, complete = _scan(root)
    if sum(stamp.size for _, stamp in scanned) > MAX_BASELINE_TOTAL_BYTES:
        complete = False

    files: list[tuple[Path, _PiBaseline]] = []
    if complete:
        for path, original_stamp in scanned:
            content = read_stable_regular_file(path, MAX_SESSION_BYTES)
            stamp = _stamp(path)
            if content is None or stamp is None or stamp != original_stamp:
                complete = False
                break
            parsed = _parse_tree(content)
            files.append(
                (
                    path,
                    _PiBaseline(
                        stamp=stamp,
                        size=len(content),
                        digest=hashlib.sha256(content).hexdigest(),
                        session_id=parsed.session_id if parsed is not None else None,
                        cwd=parsed.cwd if parsed is not None else None,
                        leaf_id=parsed.leaf_id if parsed is not None else None,
                        parse_valid=parsed is not None,
                    ),
                )
            )
    return AdapterSnapshot(
        HARNESS_ID,
        _PiState(
            repo=Path(repo).resolve(strict=False),
            session_root=root,
            files=tuple(files),
            scan_complete=complete,
        ),
    )


def _short_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip()
    if not value or len(value) > MAX_TEXT_FIELD or any(ord(char) < 32 for char in value):
        return None
    return value


def _cost(value: object) -> tuple[float | None, str | None]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        if value is None:
            return None, None
        return None, "Pi reported a nonnumeric assistant cost; cost was ignored"
    try:
        result = float(value)
    except (OverflowError, ValueError):
        return None, "Pi reported an invalid assistant cost; cost was ignored"
    if not math.isfinite(result) or result < 0:
        return None, "Pi reported an invalid assistant cost; cost was ignored"
    return result, None


def _message_role(entry: dict[str, object]) -> str | None:
    """Return the role of one message entry, and nothing else it holds."""

    if entry.get("type") != "message":
        return None
    message = entry.get("message")
    return None if not isinstance(message, dict) else _short_text(message.get("role"))


def _assistant_metadata(entry: dict[str, object]) -> _AssistantMetadata | None:
    if entry.get("type") != "message":
        return None
    message = entry.get("message")
    if not isinstance(message, dict) or message.get("role") != "assistant":
        return None

    usage = message.get("usage")
    cost_value: object = None
    if isinstance(usage, dict):
        cost_record = usage.get("cost")
        if isinstance(cost_record, dict):
            cost_value = cost_record.get("total")
    cost_usd, warning = _cost(cost_value)
    return _AssistantMetadata(
        provider=_short_text(message.get("provider")),
        model=_short_text(message.get("model")),
        cost_usd=cost_usd,
        cost_warning=warning,
    )


def _parse_tree(content: bytes) -> _ParsedTree | None:
    header: dict[str, object] | None = None
    entries: dict[str, _EntryMetadata] = {}
    leaf_id: str | None = None
    try:
        pieces = content.split(b"\n")
        byte_offset = 0
        for line_number, raw_line in enumerate(pieces, start=1):
            if line_number > MAX_JSONL_ENTRIES:
                return None
            line_offset = byte_offset
            byte_offset += len(raw_line) + (1 if line_number < len(pieces) else 0)
            if not raw_line.strip():
                continue
            try:
                item = json.loads(raw_line.decode("utf-8"))
            except (ValueError, UnicodeError, RecursionError):
                return None
            if not isinstance(item, dict):
                return None
            if header is None:
                if item.get("type") != "session":
                    return None
                header = item
                continue

            if item.get("type") == "session" or _short_text(item.get("type")) is None:
                return None

            entry_id = _short_text(item.get("id"))
            parent_value = item.get("parentId")
            parent_id = None if parent_value is None else _short_text(parent_value)
            if entry_id is None or (parent_value is not None and parent_id is None):
                return None
            if entry_id in entries:
                return None
            entries[entry_id] = _EntryMetadata(
                parent_id=parent_id,
                role=_message_role(item),
                assistant=_assistant_metadata(item),
                byte_offset=line_offset,
            )
            leaf_id = entry_id
    except (MemoryError, OverflowError):
        return None

    if header is None:
        return None
    session_id = _short_text(header.get("id"))
    cwd_value = header.get("cwd")
    if session_id is None or not isinstance(cwd_value, str) or not cwd_value:
        return None
    try:
        cwd = Path(cwd_value).resolve(strict=False)
    except (OSError, RuntimeError, ValueError):
        return None

    return _ParsedTree(
        cwd=cwd,
        session_id=session_id,
        leaf_id=leaf_id,
        entries=entries,
    )


def _single_value(
    assistants: list[_AssistantMetadata],
    field: str,
    warnings: list[str],
) -> str | None:
    values = [getattr(item, field) for item in assistants]
    if values and all(value is not None for value in values) and len(set(values)) == 1:
        return values[0]
    if values:
        warnings.append(
            f"Pi new assistant messages used missing or multiple {field} values"
        )
    return None


def _parse(path: Path, baseline: _PiBaseline | None) -> _ParsedSession | None:
    content = read_stable_regular_file(path, MAX_SESSION_BYTES)
    if content is None or not content:
        return None
    if baseline is not None:
        if len(content) < baseline.size:
            return None
        prefix = content[: baseline.size]
        if hashlib.sha256(prefix).hexdigest() != baseline.digest:
            return None

    tree = _parse_tree(content)
    if tree is None:
        return None
    if baseline is not None and baseline.parse_valid:
        if tree.session_id != baseline.session_id or tree.cwd != baseline.cwd:
            return None

    current_id = tree.leaf_id
    active_entries: list[_EntryMetadata] = []
    active_ids: set[str] = set()
    while current_id is not None:
        if current_id in active_ids:
            return None
        active_ids.add(current_id)
        entry = tree.entries.get(current_id)
        if entry is None:
            return None
        active_entries.append(entry)
        current_id = entry.parent_id

    branch_ambiguous = baseline is not None and (
        not baseline.parse_valid
        or (baseline.leaf_id is not None and baseline.leaf_id not in active_ids)
        or (
            len(content) > baseline.size
            and baseline.size > 0
            and content[baseline.size - 1 : baseline.size] != b"\n"
        )
    )
    threshold = baseline.size if baseline is not None else 0
    new_assistants = [
        entry.assistant
        for entry in active_entries
        if entry.byte_offset >= threshold and entry.assistant is not None
    ]

    cost_warnings: list[str] = []
    if branch_ambiguous:
        cost_warnings.append(
            "Pi active branch changed across the wrapped run; native model and cost were ignored"
        )
        return _ParsedSession(
            cwd=tree.cwd,
            session_id=tree.session_id,
            new_assistants=(),
            branch_ambiguous=True,
            model=None,
            provider=None,
            cost_usd=None,
            cost_warnings=tuple(cost_warnings),
            prompt_count=None,
            turn_count=None,
            models=None,
        )

    # The workflow the wrapped run added, in the order Pi appended it.  One
    # ``user`` message is one prompt, and a turn is the assistant work that
    # answered it, so a turn is counted where an assistant message first
    # follows a user one.  A resumed session whose prompt is older than the
    # baseline therefore adds its answer to no turn, because that turn began
    # before this run did.
    prompt_count = 0
    turn_count = 0
    ordered_models: list[str] = []
    previous_role: str | None = None
    for entry in reversed(active_entries):
        if entry.byte_offset < threshold:
            continue
        if entry.role == "user":
            prompt_count += 1
        elif entry.role == "assistant" and previous_role == "user":
            turn_count += 1
        previous_role = entry.role
        model = entry.assistant.model if entry.assistant is not None else None
        if model is not None and (not ordered_models or ordered_models[-1] != model):
            ordered_models.append(model)

    cost_records: list[_AssistantMetadata] = []
    costs: list[float] = []
    for item in new_assistants:
        if item.cost_usd is None:
            cost_warnings.append(
                item.cost_warning
                or "Pi new assistant usage has no numeric cost; wrapped-run cost was ignored"
            )
        else:
            cost_records.append(item)
            costs.append(item.cost_usd)
    cost_usd: float | None = None
    if new_assistants and len(costs) == len(new_assistants):
        try:
            total = math.fsum(costs)
        except OverflowError:
            total = math.inf
        if math.isfinite(total):
            cost_usd = total
        else:
            cost_warnings.append("Pi new assistant costs overflowed; wrapped-run cost was ignored")

    model = _single_value(new_assistants, "model", cost_warnings)
    provider = _single_value(new_assistants, "provider", cost_warnings)

    return _ParsedSession(
        cwd=tree.cwd,
        session_id=tree.session_id,
        new_assistants=tuple(cost_records),
        branch_ambiguous=False,
        model=model,
        provider=provider,
        cost_usd=cost_usd,
        cost_warnings=tuple(dict.fromkeys(cost_warnings)),
        prompt_count=prompt_count,
        turn_count=turn_count,
        models=tuple(ordered_models) or None,
    )


def finalize(
    adapter_snapshot: AdapterSnapshot,
    repo: str | os.PathLike[str],
    env: Mapping[str, str],
) -> NativeMetadata | None:
    del env  # Discovery must use the same root captured before the child ran.
    if adapter_snapshot.harness_id != HARNESS_ID or not isinstance(adapter_snapshot.state, _PiState):
        return None
    state = adapter_snapshot.state
    resolved_repo = Path(repo).resolve(strict=False)
    if resolved_repo != state.repo or not state.scan_complete:
        return None

    after_files, complete = _scan(state.session_root)
    if (
        not complete
        or sum(stamp.size for _, stamp in after_files)
        > MAX_BASELINE_TOTAL_BYTES
    ):
        return None
    before = dict(state.files)
    changed = [
        path
        for path, stamp in after_files
        if path not in before or before[path].stamp != stamp
    ]

    matches: list[_ParsedSession] = []
    for path in changed:
        parsed = _parse(path, before.get(path))
        if parsed is not None and parsed.cwd == state.repo:
            matches.append(parsed)
    if len(matches) != 1:
        return None

    match = matches[0]
    warnings = list(match.cost_warnings)
    if not match.new_assistants and not match.branch_ambiguous:
        warnings.append("Pi session added no cost-bearing assistant message on its active branch")

    return NativeMetadata(
        harness_id=HARNESS_ID,
        native_session_id=match.session_id,
        model=match.model,
        provider=match.provider,
        cost_usd=match.cost_usd,
        cost_source="pi-session-usage" if match.cost_usd is not None else None,
        warnings=tuple(warnings),
        turn_count=match.turn_count,
        prompt_count=match.prompt_count,
        models=match.models,
    )
