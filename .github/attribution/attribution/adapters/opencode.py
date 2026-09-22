"""Read-only post-run metadata discovery for OpenCode.

The adapter uses OpenCode's supported ``session list --format json`` and
``export`` commands.  A before/after list comparison identifies the session
changed by the wrapped command.  Exported conversation content is neither
returned nor written to the attribution ledger; only a small allowlist of
session metadata is retained.

The export also answers part of the workflow question: its message roles give
the prompt and turn counts, its assistant messages give the models in order,
and its message parts give the tool mix wherever they name a tool.  An export
covers the whole session rather than one run, so those counts are reported only
for a session the wrapped command started, exactly as its cost is.  Nothing
else about the work is recoverable from an export, so every other workflow
field stays ``None``.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import selectors
import subprocess
import time
from typing import Mapping, Sequence

from .base import AdapterSnapshot, NativeMetadata
from ..runtime import system_subprocess_environment


HARNESS_ID = "opencode"
COMMAND_TIMEOUT_SECONDS = 8
MAX_LIST_BYTES = 2 * 1024 * 1024
MAX_EXPORT_BYTES = 8 * 1024 * 1024
MAX_SESSIONS = 512
MAX_MESSAGES = 100_000
MAX_PARTS = 10_000
MAX_TEXT_FIELD = 512
ADAPTER_EXECUTABLE_ENV = "ATTRIBUTION_OPENCODE_EXECUTABLE"

_LAUNCHER_NAMES = {"bun", "bunx", "npm", "npx", "uv", "uvx"}
# OpenCode names its built-in tools in lower case, and an export part names the
# tool it ran rather than the class of work it did.  The classes are the ones
# in ``activity.TOOL_CLASSES``; a name this table does not hold counts as
# ``other``, exactly as an unrecognized name does in the shared class table.
_TOOL_CLASS_BY_NAME = {
    "read": "read",
    "grep": "search",
    "glob": "search",
    "list": "search",
    "edit": "edit",
    "patch": "edit",
    "write": "write",
    "bash": "shell",
    "webfetch": "web",
    "task": "agent",
}
_PROBE_ENVIRONMENT = {
    "OPENCODE_DISABLE_AUTOUPDATE": "true",
    "OPENCODE_DISABLE_DEFAULT_PLUGINS": "true",
    "OPENCODE_DISABLE_MODELS_FETCH": "true",
    "OPENCODE_DISABLE_PRUNE": "true",
    "OPENCODE_DISABLE_TERMINAL_TITLE": "true",
}


@dataclass(frozen=True, slots=True)
class _ListedSession:
    updated: int | float | str | None
    created_ms: float | None
    directory: Path | None


@dataclass(frozen=True, slots=True)
class _OpenCodeState:
    repo: Path
    sessions: tuple[tuple[str, int | float | str | None], ...]
    captured_at_ms: float
    executable: str


def _short_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip()
    if not value or len(value) > MAX_TEXT_FIELD:
        return None
    if any(ord(character) < 32 for character in value):
        return None
    return value


def _updated_marker(value: object) -> int | float | str | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return value if math.isfinite(float(value)) else None
    if isinstance(value, str):
        value = value.strip()
        if value and len(value) <= 128 and not any(ord(character) < 32 for character in value):
            return value
    return None


def _resolved_path(value: object) -> Path | None:
    if not isinstance(value, str) or not value or len(value) > 8192:
        return None
    try:
        return Path(value).expanduser().resolve(strict=False)
    except (OSError, RuntimeError, ValueError):
        return None


def _created_millis(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    if not math.isfinite(result) or result < 0:
        return None
    return result


def _direct_executable(
    env: Mapping[str, str],
    explicit: str | None,
) -> str | None:
    value = explicit if explicit is not None else env.get(ADAPTER_EXECUTABLE_ENV)
    if not isinstance(value, str):
        return None
    value = value.strip()
    if not value or len(value) > 8192 or any(ord(character) < 32 for character in value):
        return None
    name = value.replace("\\", "/").rsplit("/", 1)[-1].casefold()
    for suffix in (".exe", ".cmd", ".bat", ".com"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    if name in _LAUNCHER_NAMES:
        return None
    return value


def _terminate(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is None:
        try:
            process.kill()
        except OSError:
            pass
    try:
        process.wait(timeout=1)
    except (OSError, subprocess.TimeoutExpired):
        pass


def _run_json(
    executable: str,
    arguments: Sequence[str],
    *,
    repo: Path,
    env: Mapping[str, str],
    output_limit: int,
) -> object | None:
    """Run one noninteractive OpenCode query with bounded time and parsing."""

    probe_environment = dict(env)
    probe_environment.update(_PROBE_ENVIRONMENT)
    probe_environment["ATTRIBUTION_WRAPPER_SESSION_ID"] = "opencode-metadata-probe"
    process: subprocess.Popen[bytes] | None = None
    selector: selectors.BaseSelector | None = None
    stdout_buffer = bytearray()
    stderr_buffer = bytearray()
    try:
        process = subprocess.Popen(
            [executable, "--pure", *arguments],
            cwd=repo,
            env=system_subprocess_environment(probe_environment),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if process.stdout is None or process.stderr is None:
            _terminate(process)
            return None

        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ, (stdout_buffer, output_limit))
        selector.register(process.stderr, selectors.EVENT_READ, (stderr_buffer, MAX_LIST_BYTES))
        deadline = time.monotonic() + COMMAND_TIMEOUT_SECONDS
        while selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                _terminate(process)
                return None
            events = selector.select(min(remaining, 0.25))
            if not events:
                continue
            for key, _ in events:
                buffer, limit = key.data
                chunk = os.read(key.fd, min(64 * 1024, limit - len(buffer) + 1))
                if not chunk:
                    selector.unregister(key.fileobj)
                    continue
                buffer.extend(chunk)
                if len(buffer) > limit:
                    _terminate(process)
                    return None

        remaining = max(0.001, deadline - time.monotonic())
        return_code = process.wait(timeout=remaining)
        stdout = bytes(stdout_buffer)
    except (OSError, subprocess.TimeoutExpired, ValueError):
        return None
    finally:
        if process is not None:
            if process.poll() is None:
                _terminate(process)
            for stream in (process.stdout, process.stderr):
                if stream is not None:
                    try:
                        stream.close()
                    except OSError:
                        pass
        if selector is not None:
            selector.close()

    if return_code != 0:
        return None
    if not stdout.strip():
        return []
    try:
        return json.loads(stdout.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError, RecursionError):
        return None


def _list_sessions(
    executable: str,
    repo: Path,
    env: Mapping[str, str],
) -> dict[str, _ListedSession] | None:
    data = _run_json(
        executable,
        ("session", "list", "--format", "json", "--max-count", str(MAX_SESSIONS + 1)),
        repo=repo,
        env=env,
        output_limit=MAX_LIST_BYTES,
    )
    if not isinstance(data, list) or len(data) > MAX_SESSIONS:
        return None

    sessions: dict[str, _ListedSession] = {}
    for item in data:
        if not isinstance(item, dict):
            return None
        session_id = _short_text(item.get("id"))
        if session_id is None or session_id in sessions:
            return None
        sessions[session_id] = _ListedSession(
            updated=_updated_marker(item.get("updated")),
            created_ms=_created_millis(item.get("created")),
            directory=_resolved_path(item.get("directory")),
        )
    return sessions


def snapshot_opencode(
    repo: str | os.PathLike[str],
    env: Mapping[str, str],
    executable: str | None = None,
) -> AdapterSnapshot | None:
    """Record OpenCode session ids and update markers before a wrapped run."""

    direct_executable = _direct_executable(env, executable)
    if direct_executable is None:
        return None
    resolved_repo = Path(repo).resolve(strict=False)
    sessions = _list_sessions(direct_executable, resolved_repo, env)
    if sessions is None:
        return None
    return AdapterSnapshot(
        HARNESS_ID,
        _OpenCodeState(
            repo=resolved_repo,
            sessions=tuple(sorted((session_id, item.updated) for session_id, item in sessions.items())),
            captured_at_ms=time.time() * 1000,
            executable=direct_executable,
        ),
    )


def _cost(value: object) -> tuple[float | None, bool]:
    """Return a valid USD amount and whether a supplied value was invalid."""

    if value is None:
        return None, False
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None, True
    result = float(value)
    if not math.isfinite(result) or result < 0:
        return None, True
    return result, False


def _message_info(message: object) -> dict[str, object] | None:
    """Return the non-content header of one exported message."""

    if not isinstance(message, dict):
        return None
    info = message.get("info", message)
    return info if isinstance(info, dict) else None


def _tool_class_counts(message: object) -> dict[str, int] | None:
    """Return the classes of the tools one message's parts name.

    A part of type ``tool`` names the tool it ran in ``tool``.  A message whose
    parts name none returns an empty tally; a part list too large to walk
    returns ``None``, because what it holds is then unknown.
    """

    if not isinstance(message, dict):
        return {}
    parts = message.get("parts")
    if not isinstance(parts, list):
        return {}
    if len(parts) > MAX_PARTS:
        return None
    counts: dict[str, int] = {}
    for part in parts:
        if not isinstance(part, dict) or part.get("type") != "tool":
            continue
        name = _short_text(part.get("tool"))
        if name is None:
            continue
        tool_class = _TOOL_CLASS_BY_NAME.get(name.casefold(), "other")
        counts[tool_class] = counts.get(tool_class, 0) + 1
    return counts


def _assistant_record(message: object) -> dict[str, object] | None:
    info = _message_info(message)
    if info is None or info.get("role") != "assistant":
        return None

    # OpenCode has emitted both the direct legacy assistant shape and a newer
    # metadata.assistant shape.  Read only the common, non-content fields.
    metadata = info.get("metadata")
    assistant = metadata.get("assistant") if isinstance(metadata, dict) else None
    if isinstance(assistant, dict):
        return assistant
    return info


def _model_values(record: Mapping[str, object]) -> tuple[str | None, str | None]:
    model = _short_text(record.get("modelID"))
    if model is None:
        model = _short_text(record.get("modelId"))
    if model is None:
        model = _short_text(record.get("id"))
    provider = _short_text(record.get("providerID"))
    if provider is None:
        provider = _short_text(record.get("providerId"))
    if provider is None:
        provider = _short_text(record.get("provider"))
    return model, provider


def _metadata_from_export(
    data: object,
    expected_id: str,
    *,
    allow_cost: bool,
    ignored_cost_warning: str | None = None,
) -> NativeMetadata | None:
    warnings: list[str] = []
    if not isinstance(data, dict):
        return None
    info = data.get("info")
    if not isinstance(info, dict) or _short_text(info.get("id")) != expected_id:
        return None

    models: set[str] = set()
    providers: set[str] = set()
    info_model = info.get("model")
    if isinstance(info_model, dict):
        model, provider = _model_values(info_model)
        if model is not None:
            models.add(model)
        if provider is not None:
            providers.add(provider)

    messages = data.get("messages")
    message_costs: list[float] = []
    missing_message_cost = False
    invalid_message_cost = False
    # The workflow that ``messages`` describes.  One ``user`` message is one
    # prompt, and a turn is the assistant work that answered a prompt, so a
    # turn is counted where an assistant message first follows a user one.
    # ``ordered_models`` keeps the ``modelID`` of each assistant message in
    # export order and leaves out a repeat of the model already running, so a
    # second entry is a model change.  A message list this adapter cannot walk
    # leaves every one of these counts unknown.
    messages_readable = isinstance(messages, list) and len(messages) <= MAX_MESSAGES
    prompt_count = 0
    turn_count = 0
    ordered_models: list[str] = []
    tool_counts: dict[str, int] = {}
    tools_incomplete = False
    previous_role: str | None = None
    if messages_readable:
        for message in messages:
            header = _message_info(message)
            role = _short_text(header.get("role")) if header is not None else None
            if role == "user":
                prompt_count += 1
            elif role == "assistant" and previous_role == "user":
                turn_count += 1
            previous_role = role
            counted = _tool_class_counts(message)
            if counted is None:
                tools_incomplete = True
            else:
                for tool_class, count in counted.items():
                    tool_counts[tool_class] = tool_counts.get(tool_class, 0) + count

            assistant = _assistant_record(message)
            if assistant is None:
                continue
            model, provider = _model_values(assistant)
            if model is not None:
                models.add(model)
                if not ordered_models or ordered_models[-1] != model:
                    ordered_models.append(model)
            if provider is not None:
                providers.add(provider)
            if allow_cost:
                value, invalid = _cost(assistant.get("cost"))
                invalid_message_cost = invalid_message_cost or invalid
                if value is None:
                    missing_message_cost = True
                else:
                    message_costs.append(value)
    elif messages is not None:
        warnings.append("OpenCode message metadata was malformed; message details were ignored")

    # An export is cumulative exactly as a session cost is, so the messages of
    # a session this run did not start describe the runs before it as well.
    # Every count below therefore needs the same proof of a new session that
    # the cost needs, and a session without that proof reports none of them.
    workflow_readable = messages_readable and allow_cost
    if messages_readable and not allow_cost:
        warnings.append(
            "OpenCode could not prove that the session was new; cumulative workflow counts were ignored"
        )

    # A tool mix is reported only where a part actually named a tool.  An
    # export whose parts name none records no tool call this adapter can see,
    # which is an unknown mix rather than a mix of nothing.
    if tools_incomplete:
        warnings.append("OpenCode message parts exceeded the safe limit; the tool mix was ignored")
    tool_call_counts = (
        None
        if not workflow_readable or tools_incomplete or not tool_counts
        else {name: tool_counts[name] for name in sorted(tool_counts)}
    )

    model = next(iter(models)) if len(models) == 1 else None
    provider = next(iter(providers)) if len(providers) == 1 else None
    if len(models) > 1:
        warnings.append("OpenCode session used multiple models; model was not collapsed")
    if len(providers) > 1:
        warnings.append("OpenCode session used multiple providers; provider was not collapsed")

    cost_usd: float | None = None
    cost_source: str | None = None
    if allow_cost:
        cost_usd, invalid_session_cost = _cost(info.get("cost"))
        if cost_usd is not None:
            cost_source = "opencode-session-total"
        elif message_costs and not missing_message_cost and not invalid_message_cost:
            try:
                cost_usd = math.fsum(message_costs)
            except OverflowError:
                cost_usd = None
            if cost_usd is not None and math.isfinite(cost_usd):
                cost_source = "opencode-message-usage"
            else:
                cost_usd = None
        if invalid_session_cost or invalid_message_cost:
            warnings.append("OpenCode reported an invalid cost; the invalid value was ignored")
        elif cost_usd is None and message_costs and missing_message_cost:
            warnings.append("OpenCode message costs were incomplete; cost was not undercounted")
    elif ignored_cost_warning is not None:
        warnings.append(ignored_cost_warning)

    return NativeMetadata(
        harness_id=HARNESS_ID,
        native_session_id=expected_id,
        model=model,
        provider=provider,
        cost_usd=cost_usd,
        cost_source=cost_source,
        harness_version=_short_text(info.get("version")),
        warnings=tuple(warnings),
        turn_count=turn_count if workflow_readable else None,
        prompt_count=prompt_count if workflow_readable else None,
        tool_call_count=(
            None if tool_call_counts is None else sum(tool_call_counts.values())
        ),
        tool_call_counts=tool_call_counts,
        models=(tuple(ordered_models) or None) if workflow_readable else None,
    )


def finalize_opencode(
    adapter_snapshot: AdapterSnapshot,
    repo: str | os.PathLike[str],
    env: Mapping[str, str],
    executable: str | None = None,
) -> NativeMetadata | None:
    """Return metadata when exactly one repository session changed."""

    if adapter_snapshot.harness_id != HARNESS_ID or not isinstance(adapter_snapshot.state, _OpenCodeState):
        return None
    state = adapter_snapshot.state
    resolved_repo = Path(repo).resolve(strict=False)
    if resolved_repo != state.repo:
        return None

    direct_executable = state.executable
    if executable is not None:
        requested_executable = _direct_executable(env, executable)
        if requested_executable is None or requested_executable != direct_executable:
            return None

    sessions = _list_sessions(direct_executable, resolved_repo, env)
    if sessions is None:
        return None
    before = dict(state.sessions)
    changed = [
        session_id
        for session_id, item in sessions.items()
        if item.directory == state.repo and (session_id not in before or before[session_id] != item.updated)
    ]
    if len(changed) != 1:
        return None

    session_id = changed[0]
    listed_session = sessions[session_id]
    exported = _run_json(
        direct_executable,
        ("export", session_id, "--sanitize"),
        repo=resolved_repo,
        env=env,
        output_limit=MAX_EXPORT_BYTES,
    )
    if exported is None:
        return None

    existed_before = session_id in before
    created_after_snapshot = (
        listed_session.created_ms is not None
        and listed_session.created_ms >= state.captured_at_ms
    )
    allow_cost = not existed_before and created_after_snapshot
    ignored_cost_warning = None
    if existed_before:
        ignored_cost_warning = (
            "OpenCode resumed an existing session; cumulative native cost was ignored"
        )
    elif not created_after_snapshot:
        ignored_cost_warning = (
            "OpenCode could not prove that the selected session was new; cumulative native cost was ignored"
        )
    return _metadata_from_export(
        exported,
        session_id,
        allow_cost=allow_cost,
        ignored_cost_warning=ignored_cost_warning,
    )


# The common names make this module usable by a simple module-based registry;
# the explicit names above are the stable public entry points.
snapshot = snapshot_opencode
finalize = finalize_opencode
