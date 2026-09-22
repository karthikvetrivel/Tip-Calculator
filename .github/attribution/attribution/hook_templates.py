"""Generate repository-scoped native hook configurations.

The returned dictionaries are file contents, not installation plans.  Callers
choose whether and where to write them.  Every command hook forwards only the
vendor-provided JSON on stdin; no prompt or transcript value is interpolated
into the command line.
"""

from __future__ import annotations

import shlex

from .runtime import default_hook_executable


_SUPPORTED_HARNESSES = (
    "claude-code",
    "codex",
    "gemini",
    "github-copilot",
    "qwen-code",
)
_DEFAULT_COMMAND = object()


# These boundaries are deliberately conservative.  A turn-start event creates
# the baseline, post-tool events provide intermediate checkpoints, a normal
# stop closes the turn, and SessionEnd is retained as a recovery boundary.
# Codex includes its model on these turn events. Qwen and Claude expose a model
# on SessionStart, but adding that session-wide boundary would overlap every
# turn unit. Gemini exposes a model on BeforeModel, before generated tool edits.
# Those model-only events stay out until metadata can be stored without opening
# an attribution unit. Claude's PostModelSwitch is the exception: it fires only
# when the model actually changed, which is already a unit boundary here, so it
# opens no unit that a turn event would not have opened anyway.
# The post-tool events carry no matcher, so every completed tool call reaches
# the activity ledger. No template subscribes to a pre-tool event, so no tool
# costs a second worktree snapshot before it runs.
# PostCompact records the summary that replaced a context window, and Claude's
# InstructionsLoaded names each memory file that entered one. PreCompact
# changes no ledger row, so no template subscribes to it. Only Claude Code
# reports its instruction files.
_NESTED_EVENTS: dict[str, tuple[str, ...]] = {
    "claude-code": (
        "UserPromptSubmit",
        "PostModelSwitch",
        "InstructionsLoaded",
        "PostCompact",
        "SubagentStart",
        "PostToolUse",
        "PostToolUseFailure",
        "SubagentStop",
        "Stop",
        "StopFailure",
        "SessionEnd",
    ),
    "codex": (
        "UserPromptSubmit",
        "PostCompact",
        "SubagentStart",
        "PostToolUse",
        "SubagentStop",
        "Stop",
        "Interrupt",
        "SessionEnd",
    ),
    "gemini": (
        "BeforeAgent",
        "AfterTool",
        "AfterAgent",
        "SessionEnd",
    ),
    "qwen-code": (
        "UserPromptSubmit",
        "SubagentStart",
        "PostToolUse",
        "PostToolUseFailure",
        "SubagentStop",
        "Stop",
        "SessionEnd",
    ),
}


# PascalCase selects Copilot's VS Code-compatible payload. The generated command
# also supplies the event name so the receiver does not depend on that payload
# detail. Copilot's SubagentStart payload has no stable agent id, so starting a
# child attribution unit there would not pair safely with SubagentStop.
_COPILOT_EVENTS = (
    "UserPromptSubmit",
    "PostToolUse",
    "PostToolUseFailure",
    "Stop",
    "ErrorOccurred",
    "SessionEnd",
)


def _command(
    attribution_command: str,
    feature: str,
    harness_id: str,
    event: str,
) -> str:
    return shlex.join(
        (
            attribution_command,
            "hook",
            "--feature",
            feature,
            "--harness",
            harness_id,
            "--event",
            event,
            "--observer",
        )
    )


def _nested_template(
    events: tuple[str, ...],
    attribution_command: str,
    feature: str,
    harness_id: str,
) -> dict[str, object]:
    """Return the matcher-group schema shared by four supported harnesses."""

    def handler(event: str) -> dict[str, object]:
        result: dict[str, object] = {
            "type": "command",
            "command": _command(
                attribution_command,
                feature,
                harness_id,
                event,
            ),
        }
        if harness_id == "codex" and event in {"Interrupt", "SessionEnd"}:
            # Codex limits these events to three seconds. The default is one.
            result["timeout"] = 3
        elif harness_id == "claude-code" and event == "SessionEnd":
            # Claude defaults this event to 1.5 seconds.
            result["timeout"] = 10
        return result

    return {
        "hooks": {
            event: [
                {
                    "hooks": [handler(event)]
                }
            ]
            for event in events
        }
    }


def _copilot_template(
    attribution_command: str,
    feature: str,
) -> dict[str, object]:
    """Return a versioned repository hook file for GitHub Copilot."""

    return {
        "version": 1,
        "hooks": {
            event: [
                {
                    "type": "command",
                    "exec": attribution_command,
                    "args": [
                        "hook",
                        "--feature",
                        feature,
                        "--harness",
                        "github-copilot",
                        "--event",
                        event,
                        "--observer",
                    ],
                }
            ]
            for event in _COPILOT_EVENTS
        },
    }


def supported_template_harnesses() -> tuple[str, ...]:
    """Return canonical harness ids with repository hook templates."""

    return _SUPPORTED_HARNESSES


def hook_template(
    harness_id: str,
    *,
    feature: str,
    attribution_command: object = _DEFAULT_COMMAND,
) -> dict[str, object]:
    """Return a JSON-ready native hook configuration for ``harness_id``.

    Only canonical ids are accepted.  ``attribution_command`` is one executable
    name or path, not a shell fragment.  When omitted, source installs use the
    public ``joyride`` command and standalone builds pin their own executable
    path. It and every argument are shell-quoted as independent tokens before
    being placed in vendor command-hook fields.
    """

    if not isinstance(harness_id, str) or harness_id not in _SUPPORTED_HARNESSES:
        raise ValueError(f"unsupported hook template harness: {harness_id!r}")
    if not isinstance(feature, str) or not feature.strip():
        raise ValueError("feature must be nonempty")
    selected_command = (
        default_hook_executable()
        if attribution_command is _DEFAULT_COMMAND
        else attribution_command
    )
    if not isinstance(selected_command, str) or not selected_command.strip():
        raise ValueError("attribution_command must be nonempty")
    if "\0" in feature or "\0" in selected_command:
        raise ValueError("hook command values must not contain NUL characters")

    normalized_feature = feature.strip()
    if harness_id == "github-copilot":
        return _copilot_template(selected_command, normalized_feature)
    return _nested_template(
        _NESTED_EVENTS[harness_id],
        selected_command,
        normalized_feature,
        harness_id,
    )


__all__ = ["hook_template", "supported_template_harnesses"]
