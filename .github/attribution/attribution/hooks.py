"""Normalize native coding-agent hook payloads without retaining their content."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Any


MAX_HOOK_BYTES = 1024 * 1024

_MAX_IDENTIFIER_CHARS = 1024
_MAX_PATH_CHARS = 32 * 1024
_MAX_JSON_DEPTH = 64


@dataclass(frozen=True)
class HookEvent:
    """The small, content-free subset of a native hook used for attribution."""

    harness_id: str
    event: str
    native_session_id: str
    native_turn_id: str | None
    parent_session_id: str | None
    native_agent_id: str | None
    cwd: str | None
    model: str | None
    provider: str | None
    occurred_at: str | None
    native_event_id: str | None
    tool_name: str | None
    terminal_reason: str | None
    recoverable: bool | None


_EVENT_ALIASES = {
    "session_started": "session_start",
    "start_session": "session_start",
    "session_stop": "session_end",
    "session_stopped": "session_end",
    "session_complete": "session_end",
    "session_completed": "session_end",
    "end_session": "session_end",
    "task_start": "session_start",
    "task_started": "session_start",
    "task_complete": "session_end",
    "task_completed": "session_end",
    "task_end": "session_end",
    "user_prompt_submit": "user_prompt",
    "user_prompt_submitted": "user_prompt",
    "prompt_submit": "user_prompt",
    "prompt_submitted": "user_prompt",
    "before_submit_prompt": "user_prompt",
    "pre_user_prompt": "user_prompt",
    "user_input": "user_prompt",
    "pre_tool_use": "pre_tool",
    "before_tool": "pre_tool",
    "before_tool_use": "pre_tool",
    "before_tool_call": "pre_tool",
    "tool_call_start": "pre_tool",
    "tool_start": "pre_tool",
    "post_tool_use": "post_tool",
    "post_tool_use_failure": "post_tool",
    "after_tool": "post_tool",
    "after_tool_use": "post_tool",
    "after_tool_call": "post_tool",
    "tool_call_end": "post_tool",
    "tool_end": "post_tool",
    "tool_result": "post_tool",
    "before_agent": "turn_start",
    "agent_start": "turn_start",
    "agent_started": "turn_start",
    "agent_run_start": "turn_start",
    "turn_started": "turn_start",
    "before_model": "turn_start",
    "after_agent": "turn_end",
    "agent_end": "turn_end",
    "agent_ended": "turn_end",
    "agent_settled": "turn_end",
    "agent_run_end": "turn_end",
    "turn_complete": "turn_end",
    "turn_completed": "turn_end",
    "agent_turn_complete": "turn_end",
    "after_model": "turn_end",
    "agent_stop": "stop",
    "stopped": "stop",
    "cancel": "interrupt",
    "canceled": "interrupt",
    "cancelled": "interrupt",
    "task_cancel": "interrupt",
    "task_canceled": "interrupt",
    "task_cancelled": "interrupt",
    "abort": "interrupt",
    "aborted": "interrupt",
    "interrupted": "interrupt",
    "subagent_started": "subagent_start",
    "sub_agent_start": "subagent_start",
    "sub_agent_started": "subagent_start",
    "subagent_end": "subagent_stop",
    "subagent_ended": "subagent_stop",
    "sub_agent_stop": "subagent_stop",
    "sub_agent_end": "subagent_stop",
    "sub_agent_ended": "subagent_stop",
    "failure": "error",
    "failed": "error",
    "error_occurred": "error",
    "stop_failure": "error",
    "pre_compact": "compact",
    "compaction": "compact",
    "compact_start": "compact",
}


def _validate_json(value: object, *, depth: int = 0) -> None:
    """Reject values that could not have come from a bounded JSON document."""

    if depth > _MAX_JSON_DEPTH:
        raise ValueError("hook payload is nested too deeply")
    if value is None or isinstance(value, (bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("hook payload contains a non-finite number")
        return
    if isinstance(value, str):
        if "\0" in value:
            raise ValueError("hook payload contains a NUL character")
        return
    if isinstance(value, list):
        for item in value:
            _validate_json(item, depth=depth + 1)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("hook payload object keys must be strings")
            if "\0" in key:
                raise ValueError("hook payload contains a NUL character")
            _validate_json(item, depth=depth + 1)
        return
    raise ValueError("hook payload must contain only JSON values")


def _validate_payload(payload: object) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("hook payload must be a JSON object")
    _validate_json(payload)
    try:
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    except (RecursionError, TypeError, ValueError) as exc:
        raise ValueError("hook payload must be valid JSON") from exc
    if len(encoded) > MAX_HOOK_BYTES:
        raise ValueError(f"hook payload exceeds {MAX_HOOK_BYTES} bytes")
    return payload


def _text(
    value: object,
    field: str,
    *,
    required: bool = False,
    max_chars: int = _MAX_IDENTIFIER_CHARS,
) -> str | None:
    if value is None:
        if required:
            raise ValueError(f"{field} is required")
        return None
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    result = value.strip()
    if not result:
        if required:
            raise ValueError(f"{field} is required")
        return None
    if "\0" in result:
        raise ValueError(f"{field} must not contain NUL characters")
    if len(result) > max_chars:
        raise ValueError(f"{field} is too long")
    return result


def _first_text(
    payload: dict[str, Any],
    keys: tuple[str, ...],
    field: str,
    *,
    max_chars: int = _MAX_IDENTIFIER_CHARS,
) -> str | None:
    for key in keys:
        if key in payload and payload[key] is not None:
            return _text(payload[key], field, max_chars=max_chars)
    return None


def _first_bool(
    payload: dict[str, Any], keys: tuple[str, ...], field: str
) -> bool | None:
    for key in keys:
        if key not in payload or payload[key] is None:
            continue
        value = payload[key]
        if not isinstance(value, bool):
            raise ValueError(f"{field} must be a boolean")
        return value
    return None


def _snake_case_event(value: str) -> str:
    # Split both ``HTTPError`` and ``preToolUse`` before replacing punctuation.
    result = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", value)
    result = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", result)
    result = re.sub(r"[^A-Za-z0-9]+", "_", result).strip("_").lower()
    if not result:
        raise ValueError("event is required")
    return _EVENT_ALIASES.get(result, result)


def normalized_event(name: object) -> str | None:
    """Return the normalized name of one harness event spelling, or None.

    ``parse_hook_payload`` normalizes the event of every payload it validates.
    A caller that holds the name alone reads the same table through here, so a
    harness spelling means the same event wherever it is read.
    """

    if not isinstance(name, str):
        return None
    try:
        return _snake_case_event(name)
    except ValueError:
        return None


def _workspace(payload: dict[str, Any]) -> str | None:
    direct = _first_text(
        payload,
        ("cwd", "workspace_root", "workspaceRoot", "worktree"),
        "cwd",
        max_chars=_MAX_PATH_CHARS,
    )
    if direct is not None:
        return direct
    if "workspaceRoots" not in payload or payload["workspaceRoots"] is None:
        return None
    roots = payload["workspaceRoots"]
    if not isinstance(roots, list):
        raise ValueError("workspaceRoots must be an array")
    if not roots:
        return None
    return _text(roots[0], "cwd", max_chars=_MAX_PATH_CHARS)


def _timestamp(payload: dict[str, Any]) -> str | None:
    for key in ("occurred_at", "occurredAt", "timestamp", "created_at", "createdAt"):
        if key not in payload or payload[key] is None:
            continue
        value = payload[key]
        if isinstance(value, str):
            return _text(value, "occurred_at")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("occurred_at must be a string or finite number")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("occurred_at must be a string or finite number")
        return str(value)
    return None


def _model_metadata(payload: dict[str, Any]) -> tuple[str | None, str | None]:
    model = _first_text(
        payload,
        ("model_name", "modelName", "model_id", "modelId"),
        "model",
    )
    nested: dict[str, Any] | None = None
    if model is None and "model" in payload and payload["model"] is not None:
        raw_model = payload["model"]
        if isinstance(raw_model, str):
            model = _text(raw_model, "model")
        elif isinstance(raw_model, dict):
            nested = raw_model
            model = _first_text(nested, ("slug", "id", "model"), "model")
        else:
            raise ValueError("model must be a string or object")
    elif isinstance(payload.get("model"), dict):
        nested = payload["model"]

    provider = _first_text(
        payload,
        ("provider", "provider_name", "providerName", "provider_id", "providerId"),
        "provider",
    )
    if provider is None and nested is not None and nested.get("provider") is not None:
        raw_provider = nested["provider"]
        if isinstance(raw_provider, str):
            provider = _text(raw_provider, "provider")
        elif isinstance(raw_provider, dict):
            provider = _first_text(raw_provider, ("slug", "id", "name"), "provider")
        else:
            raise ValueError("provider must be a string or object")
    return model, provider


def _tool_name(payload: dict[str, Any]) -> str | None:
    direct = _first_text(payload, ("tool_name", "toolName"), "tool_name")
    if direct is not None:
        return direct
    for container_name in ("preToolUse", "postToolUse", "pre_tool_use", "post_tool_use"):
        if container_name not in payload or payload[container_name] is None:
            continue
        container = payload[container_name]
        if not isinstance(container, dict):
            raise ValueError(f"{container_name} must be an object")
        if "tool" not in container or container["tool"] is None:
            continue
        raw_tool = container["tool"]
        if isinstance(raw_tool, str):
            return _text(raw_tool, "tool_name")
        if isinstance(raw_tool, dict):
            return _first_text(raw_tool, ("name", "id"), "tool_name")
        raise ValueError("tool_name must be a string or object")
    return None


def parse_hook_payload(
    payload: object,
    *,
    harness_id: str,
    event_override: str | None = None,
    session_override: str | None = None,
    model_override: str | None = None,
) -> HookEvent:
    """Return normalized attribution metadata from a native hook payload.

    Prompt text, transcripts, tool arguments, tool results, and unknown fields are
    deliberately neither returned nor included in the event fingerprint.
    """

    data = _validate_payload(payload)
    normalized_harness = _text(harness_id, "harness_id", required=True)

    if event_override is not None:
        raw_event = _text(event_override, "event", required=True)
    else:
        raw_event = _first_text(
            data,
            ("hook_event_name", "event", "eventName", "agent_action_name", "type"),
            "event",
        )
        if raw_event is None:
            raise ValueError("event is required")
    event = _snake_case_event(raw_event)

    if session_override is not None:
        session_id = _text(session_override, "native_session_id", required=True)
    else:
        session_id = _first_text(
            data,
            (
                "session_id",
                "sessionId",
                "conversation_id",
                "conversationId",
                "trajectory_id",
                "thread_id",
                "threadId",
                "taskId",
                "task_id",
            ),
            "native_session_id",
        )
        if session_id is None:
            raise ValueError("native_session_id is required")

    model, provider = _model_metadata(data)
    if model_override is not None:
        model = _text(model_override, "model", required=True)

    terminal_reason = None
    if event in {
        "error",
        "interrupt",
        "session_end",
        "stop",
        "subagent_stop",
        "turn_end",
    }:
        terminal_reason = _first_text(
            data,
            (
                "terminal_reason",
                "terminalReason",
                "stop_reason",
                "stopReason",
                "exit_reason",
                "exitReason",
                "error_context",
                "errorContext",
                "reason",
            ),
            "terminal_reason",
        )

    return HookEvent(
        harness_id=normalized_harness,
        event=event,
        native_session_id=session_id,
        native_turn_id=_first_text(
            data,
            (
                "turn_id",
                "turnId",
                "prompt_id",
                "promptId",
                "generation_id",
                "generationId",
                "execution_id",
                "executionId",
                "run_id",
                "runId",
            ),
            "native_turn_id",
        ),
        parent_session_id=_first_text(
            data,
            (
                "parent_session_id",
                "parentSessionId",
                "parent_id",
                "parentId",
                "parent_thread_id",
                "parentThreadId",
                "parent_task_id",
                "parentTaskId",
                "parent_conversation_id",
                "parentConversationId",
            ),
            "parent_session_id",
        ),
        native_agent_id=_first_text(
            data,
            ("agent_id", "agentId", "subagent_id", "subagentId"),
            "native_agent_id",
        ),
        cwd=_workspace(data),
        model=model,
        provider=provider,
        occurred_at=_timestamp(data),
        native_event_id=_first_text(
            data,
            (
                "event_id",
                "eventId",
                "hook_id",
                "hookId",
                "tool_use_id",
                "toolUseId",
                "call_id",
                "callId",
            ),
            "native_event_id",
        ),
        tool_name=_tool_name(data),
        terminal_reason=terminal_reason,
        recoverable=_first_bool(
            data,
            ("recoverable", "is_recoverable", "isRecoverable"),
            "recoverable",
        ),
    )


def event_fingerprint(event: HookEvent) -> str:
    """Return a stable SHA-256 fingerprint of the allowlisted event fields."""

    if not isinstance(event, HookEvent):
        raise TypeError("event must be a HookEvent")
    fields = {
        "cwd": event.cwd,
        "event": event.event,
        "harness_id": event.harness_id,
        "model": event.model,
        "native_agent_id": event.native_agent_id,
        "native_event_id": event.native_event_id,
        "native_session_id": event.native_session_id,
        "native_turn_id": event.native_turn_id,
        "occurred_at": event.occurred_at,
        "parent_session_id": event.parent_session_id,
        "provider": event.provider,
        "recoverable": event.recoverable,
        "terminal_reason": event.terminal_reason,
        "tool_name": event.tool_name,
    }
    encoded = json.dumps(
        fields,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "HookEvent",
    "MAX_HOOK_BYTES",
    "event_fingerprint",
    "normalized_event",
    "parse_hook_payload",
]
