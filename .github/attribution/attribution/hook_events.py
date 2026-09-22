"""Normalize native coding-tool hooks without treating every file change as AI.

The returned input and response are transient evidence for the capture engine,
not data to retain in session metadata; the installed receiver keeps them in
the session's trace. Transcript paths are deliberately excluded.

Schemas: https://learn.chatgpt.com/docs/hooks and
https://code.claude.com/docs/en/hooks
"""

from __future__ import annotations

from copy import deepcopy
import re
from typing import Any


_PROVIDERS = {"codex": "codex", "claude": "claude", "claude-code": "claude", "claude_code": "claude"}
_EVENTS = {
    "SessionStart": "start",
    "PreToolUse": "pre",
    "PostToolUse": "post",
    "SessionEnd": "end",
}
_CLAUDE_EVENTS = {"PostToolUseFailure": "failure", "PostModelSwitch": "model"}


def _text(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip() or "\0" in value:
        return None
    return value.strip()


def classify_tool(provider: str, tool_name: str | None) -> str:
    """Classify only native tools whose input format we recognize.

    MCP and arbitrary function tools stay ``other``, even if their names end in
    ``Write``. Shell commands also remain distinct: success does not prove which
    of the working tree's concurrent changes the command produced.
    """
    provider = _PROVIDERS.get(provider, "")
    if provider == "codex":
        # Edit and Write are supported hook matcher aliases for apply_patch.
        if tool_name in {"apply_patch", "Edit", "Write"}:
            return "patch"
        if tool_name == "Bash":
            return "shell"
    elif provider == "claude":
        if tool_name == "Write":
            return "write"
        if tool_name == "Edit":
            return "edit"
        if tool_name in {"Bash", "PowerShell"}:
            return "shell"
    return "other"


def response_success(response: Any, kind: str) -> bool | None:
    """Read explicit outcomes; a post hook alone does not mean Codex succeeded."""
    if isinstance(response, dict):
        if any(response.get(key) is True for key in ("is_error", "isError")):
            return False
        if response.get("success") is False:
            return False
        if response.get("error") not in (None, False, ""):
            return False
        status = response.get("status")
        if status in ("failed", "error", "cancelled", "canceled"):
            return False
        exit_code = response.get("exit_code")
        if isinstance(exit_code, int) and not isinstance(exit_code, bool):
            return exit_code == 0
        if response.get("success") is True:
            return True
        if status in ("completed", "success", "succeeded"):
            return True
        # Some local function tools wrap the model-facing response in output.
        if isinstance(response.get("output"), str):
            return response_success(response["output"], kind)
        return None
    if isinstance(response, str):
        if kind == "patch":
            if response.startswith("Success. Updated the following files:"):
                return True
            if response.startswith(("apply_patch verification failed:", "Failed to apply patch")):
                return False
        if kind == "shell":
            match = re.search(r"(?:^|\n)Process exited with code (-?\d+)(?:\n|$)", response)
            if match:
                return int(match.group(1)) == 0
    return None


# The wrapper capture engine in this module has called this outcome reader by
# its private name since before the activity ledger shared it.
_response_success = response_success


def normalize_event(provider: str, payload: dict) -> dict | None:
    """Return a validated capture event, or None for unsupported/malformed input.

    Every result has event, external_session_id, model, cwd, tool_use_id,
    tool_name, tool_input and succeeded. It also carries canonical provider,
    tool_response, failure_class and agent_id. None success means unknown, not
    successful. Callers must verify a tool's intended edit against actual bytes.
    """
    provider = _PROVIDERS.get(provider, "")
    if not provider or not isinstance(payload, dict):
        return None
    name = payload.get("hook_event_name")
    if not isinstance(name, str):
        return None
    event = _EVENTS.get(name)
    if provider == "claude":
        event = _CLAUDE_EVENTS.get(name, event)
    if event is None:
        return None
    session_id = _text(payload.get("session_id"))
    cwd = _text(payload.get("cwd"))
    if session_id is None or cwd is None:
        return None

    tool_name = None
    tool_use_id = None
    tool_input: dict = {}
    response = None
    succeeded = None
    failure_class = None
    if event in {"pre", "post", "failure"}:
        tool_name = _text(payload.get("tool_name"))
        tool_use_id = _text(payload.get("tool_use_id"))
        if tool_name is None or tool_use_id is None:
            return None
        if isinstance(payload.get("tool_input"), dict):
            tool_input = deepcopy(payload["tool_input"])
        if event == "post":
            response = deepcopy(payload.get("tool_response"))
            succeeded = response_success(response, classify_tool(provider, tool_name))
            # Claude's event is documented to fire only for successful calls.
            # Explicit failure data takes precedence over that event name.
            if succeeded is None and provider == "claude":
                succeeded = True
            if succeeded is False:
                failure_class = "tool_error"
        elif event == "failure":
            succeeded = False
            failure_class = "interrupted" if payload.get("is_interrupt") is True else "tool_error"

    if provider == "codex" or event == "start":
        model = _text(payload.get("model"))
    elif event == "model":
        model = _text(payload.get("to_model"))
    else:
        # Claude's tool events do not promise an active model. The engine can
        # associate main-agent calls with SessionStart/PostModelSwitch evidence.
        model = None
    return {
        "provider": provider,
        "event": event,
        "external_session_id": session_id,
        "model": model,
        "cwd": cwd,
        "tool_use_id": tool_use_id,
        "tool_name": tool_name,
        "tool_input": tool_input,
        "tool_response": response,
        "succeeded": succeeded,
        "failure_class": failure_class,
        "agent_id": _text(payload.get("agent_id")),
    }
