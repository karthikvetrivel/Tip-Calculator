"""Private pre-push integration for immutable, metadata-only PR snapshots."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
from typing import Any, TextIO

from .notes import (
    NOTES_REF, _blob_content, _changed_files, _eligible_edits, _exact_chain,
    _tree, record_commit,
)
from .runtime import system_subprocess_environment
from .store import git_common_dir, git_dir, open_db, repository_root
from .task_notes import TASK_NOTES_REF
from .traces import MAX_SNAPSHOT_TRACE_BYTES, build_trace, canonical_bytes, traces_enabled
from .workflow import MAX_WORKFLOW_DEPTH, native_identity_hash


MAX_SNAPSHOT_BYTES = 8 * 1024 * 1024
_MAX_NOTE_BYTES = 2 * 1024 * 1024
_MAX_COMMITS = 100_000
_MAX_CAPTURE_COMMITS = 250
_OID = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_REF_PREFIX = "refs/notes/attribution-pr/"
# A trace is stored under its session id, so the id must be a plain file name.
_TRACE_NAME = re.compile(r"^[A-Za-z0-9._:-]{1,200}$")
_GUARD = "ATTRIBUTION_METADATA_PUSH"
_SESSION_FIELDS = (
    "id", "task_id", "feature", "model", "harness", "actor_kind", "source_session_id",
    "label_source", "membership_source", "role", "summary", "parent_session_id", "token_count",
    "token_source", "cost_usd", "cost_source", "usage_includes_children", "started_at",
    "ended_at", "exit_code", "outcome", "agent_type", "launch_mode", "session_source",
    "permission_mode", "effort_level", "turn_count", "prompt_count", "interrupt_count",
    "compaction_count", "model_switch_count", "tool_call_count", "duration_ms",
    "activity_truncated",
)
# Workflow facets describe how the work was done, so they are published with
# bounded values only: no locator, no hash, and no text.
_SESSION_FACET_TEXT_FIELDS = (
    "agent_type", "launch_mode", "session_source", "permission_mode", "effort_level",
)
_SESSION_FACET_COUNT_FIELDS = (
    "turn_count", "prompt_count", "interrupt_count", "compaction_count",
    "model_switch_count", "tool_call_count", "duration_ms",
)
_LAUNCH_MODES = ("foreground", "background")
_SESSION_SOURCES = ("startup", "resume", "fork", "clear", "compact", "subagent")
# The workflow object of a note says how the work was done. Its vocabulary is
# pinned here, as the two tuples above are, because a snapshot is a published
# schema: a ledger that later learns a tool class must not silently widen what
# a push shares.
_TOOL_CLASSES = (
    "read", "search", "edit", "write", "shell", "web", "skill", "mcp", "agent",
    "ask_user", "other",
)
_WORKFLOW_AGENT_TEXT_FIELDS = (
    "agent_type", "harness", "model", "role", "launch_mode", "session_source",
    "permission_mode", "effort_level",
)
_WORKFLOW_AGENT_COUNT_FIELDS = (
    "turn_count", "prompt_count", "interrupt_count", "compaction_count",
    "model_switch_count", "duration_ms",
)
_WORKFLOW_TOTAL_FIELDS = (
    "agents", "subagents", "prompts", "turns", "interrupts", "compactions",
    "model_switches",
)
_MAX_WORKFLOW_AGENTS = 500
# One number, read by the walk in ``notes.py`` that writes the object and by
# the schema here that republishes it, so a note this project writes is never
# refused for its depth.
_MAX_WORKFLOW_DEPTH = MAX_WORKFLOW_DEPTH
_MAX_WORKFLOW_INSTRUCTION_FILES = 1_000
_REVISION_FIELDS = (
    "from_commit", "from_path", "from_start", "from_end",
    "from_session_id", "to_session_id",
)
_SESSION_REVISION_FIELDS = (
    "path", "from_session_id", "to_session_id", "removed_lines", "kind",
)
# The telemetry this clone allocated to one session: what its provider priced
# and counted for the requests that session made. Like every object above it,
# the record is rebuilt from named keys only, so it carries no prompt, no path,
# no request ID, and no native session ID.
_USAGE_COUNT_FIELDS = (
    "input_tokens", "cached_input_tokens", "cache_creation_input_tokens",
    "output_tokens", "total_tokens", "request_count", "priced_request_count",
    "unpriced_request_count",
)
_USAGE_ALLOCATIONS = (
    "agent-routed", "tool-linked", "turn-linked", "session-allocated", "mixed",
)
_MAX_USAGE_MODELS = 8
_MAX_USAGE_MODEL_CHARS = 120
# One push publishes usage for no more agents than one workflow object may
# name, so the map cannot grow past a bound the snapshot already obeys.
_MAX_USAGE_SESSIONS = _MAX_WORKFLOW_AGENTS


def snapshot_ref(head: str) -> str:
    if not isinstance(head, str) or _OID.fullmatch(head) is None or not head.strip("0"):
        raise ValueError("Snapshot head must be a full commit object ID")
    return _REF_PREFIX + head


def _git(repo: Path, *args: str, input_bytes: bytes | None = None,
         environment: dict[str, str] | None = None, check: bool = True) -> subprocess.CompletedProcess[bytes]:
    result = subprocess.run(
        ["git", "-C", str(repo), *args], input=input_bytes,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        env=system_subprocess_environment(environment), timeout=30, check=False,
    )
    if check and result.returncode:
        # Git errors can include credential-bearing remote URLs. The hook only
        # reports a generic failure; callers can inspect Git independently.
        raise ValueError(f"Joyride Git operation failed: {args[0]}")
    return result


def _text(value: Any, name: str, *, limit: int = 2048) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > limit or "\0" in value:
        raise ValueError(f"Invalid attribution {name}")
    return value


def _optional_text(value: Any, name: str, *, limit: int = 2048) -> str | None:
    if value is None:
        return None
    return _text(value, name, limit=limit)


def _sanitize_session(source: Any) -> dict[str, Any]:
    if not isinstance(source, dict):
        raise ValueError("Invalid attribution session")
    candidate = {field: source.get(field) for field in _SESSION_FIELDS}
    identity = native_identity_hash(source)
    if identity:
        # Preserve the equality relation needed to count native agents while
        # keeping the original harness identifier out of shared metadata.
        candidate["workflow_identity"] = identity
    elif isinstance(source.get("workflow_identity"), str) and re.fullmatch(r"[0-9a-f]{64}", source["workflow_identity"]):
        candidate["workflow_identity"] = source["workflow_identity"]
    for field in ("id", "feature", "model", "harness", "started_at"):
        candidate[field] = _text(candidate[field], field)
    for field in ("task_id", "source_session_id", "parent_session_id"):
        candidate[field] = _optional_text(candidate[field], field.replace("_", " "), limit=256)
    candidate["actor_kind"] = source.get("actor_kind", "ai")
    if candidate["actor_kind"] not in ("ai", "manual"):
        raise ValueError("Invalid attribution actor")
    candidate["label_source"] = _text(source.get("label_source", "reported"), "label source")
    candidate["membership_source"] = _text(
        source.get("membership_source", "legacy"), "membership source"
    )
    candidate["role"] = source.get("role", "implementation")
    if candidate["role"] not in ("planning", "implementation", "testing", "review", "other"):
        raise ValueError("Invalid attribution role")
    # One line saying what the agent was asked to do. It is published because a
    # PR footer names the purpose of a subagent. Unlike every field above it,
    # nothing bounded this one when it was written: ``joyride run
    # --summary`` takes any text a user types. A summary outside the bound is
    # therefore dropped rather than rejected, because a cosmetic field must not
    # cost a push its whole snapshot and the PR footer with it.
    summary = candidate["summary"]
    if (
        not isinstance(summary, str)
        or not summary.strip()
        or len(summary) > 256
        or any(character in summary for character in "\0\r\n")
    ):
        summary = None
    candidate["summary"] = summary
    tokens = candidate["token_count"]
    if tokens is not None and (
        isinstance(tokens, bool) or not isinstance(tokens, int) or tokens < 0
    ):
        raise ValueError("Invalid attribution token count")
    candidate["token_source"] = None if tokens is None else _text(
        source.get("token_source") or "reported", "token source"
    )
    cost = candidate["cost_usd"]
    if cost is not None and (
        isinstance(cost, bool) or not isinstance(cost, (int, float))
        or not math.isfinite(cost) or cost < 0
    ):
        raise ValueError("Invalid attribution cost")
    candidate["cost_source"] = None if cost is None else _text(
        source.get("cost_source") or "reported", "cost source"
    )
    includes_children = candidate["usage_includes_children"]
    if includes_children is None:
        includes_children = False
    elif type(includes_children) is bool:
        pass
    elif type(includes_children) is int and includes_children in (0, 1):
        pass
    else:
        raise ValueError("Invalid child-usage marker")
    candidate["usage_includes_children"] = bool(includes_children)
    candidate["ended_at"] = _optional_text(candidate["ended_at"], "end time", limit=128)
    exit_code = candidate["exit_code"]
    if exit_code is not None and (isinstance(exit_code, bool) or not isinstance(exit_code, int)):
        raise ValueError("Invalid attribution exit code")
    outcome = candidate["outcome"]
    if outcome is not None and outcome not in ("completed", "failed", "interrupted", "abandoned"):
        raise ValueError("Invalid attribution outcome")
    for field in _SESSION_FACET_TEXT_FIELDS:
        candidate[field] = _optional_text(candidate[field], field.replace("_", " "), limit=256)
    if candidate["launch_mode"] is not None and candidate["launch_mode"] not in _LAUNCH_MODES:
        raise ValueError("Invalid attribution launch mode")
    if candidate["session_source"] is not None and candidate["session_source"] not in _SESSION_SOURCES:
        raise ValueError("Invalid attribution session source")
    for field in _SESSION_FACET_COUNT_FIELDS:
        count = candidate[field]
        if count is None:
            continue
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError(f"Invalid attribution {field.replace('_', ' ')}")
    truncated = candidate["activity_truncated"]
    if truncated is None:
        truncated = False
    elif type(truncated) is bool or (type(truncated) is int and truncated in (0, 1)):
        pass
    else:
        raise ValueError("Invalid attribution truncation marker")
    candidate["activity_truncated"] = bool(truncated)
    return candidate


def _workflow_count(value: Any, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"Invalid attribution workflow {name}")
    return value


def _workflow_tool_calls(source: Any) -> dict[str, int] | None:
    """Republish one tool mix, or None where no harness reported a mix."""
    if source is None:
        return None
    if not isinstance(source, dict):
        raise ValueError("Invalid attribution workflow tool calls")
    counts: dict[str, int] = {}
    for name, value in source.items():
        if name not in _TOOL_CLASSES:
            raise ValueError("Invalid attribution workflow tool class")
        count = _workflow_count(value, f"{name} tool calls")
        if count is None:
            raise ValueError(f"Invalid attribution workflow {name} tool calls")
        counts[name] = count
    return {name: counts[name] for name in _TOOL_CLASSES if name in counts}


def _sanitize_workflow_agent(source: Any, depth: int, budget: list[int]) -> dict[str, Any]:
    if not isinstance(source, dict):
        raise ValueError("Invalid attribution workflow agent")
    if depth > _MAX_WORKFLOW_DEPTH:
        raise ValueError("Joyride workflow agents nest too deeply")
    budget[0] -= 1
    if budget[0] < 0:
        raise ValueError("Too many attribution workflow agents")
    agent: dict[str, Any] = {
        "session_id": _text(source.get("session_id"), "workflow session ID", limit=256),
        "parent_session_id": _optional_text(
            source.get("parent_session_id"), "workflow parent session ID", limit=256
        ),
    }
    for field in _WORKFLOW_AGENT_TEXT_FIELDS:
        agent[field] = _optional_text(
            source.get(field), f"workflow {field.replace('_', ' ')}", limit=256
        )
    for field in _WORKFLOW_AGENT_COUNT_FIELDS:
        agent[field] = _workflow_count(source.get(field), field.replace("_", " "))
    agent["tool_calls"] = _workflow_tool_calls(source.get("tool_calls"))
    cost = source.get("cost_usd")
    if cost is not None and (
        isinstance(cost, bool) or not isinstance(cost, (int, float))
        or not math.isfinite(cost) or cost < 0
    ):
        raise ValueError("Invalid attribution workflow cost")
    agent["cost_usd"] = cost
    truncated = source.get("activity_truncated")
    if truncated is None:
        truncated = False
    elif type(truncated) is bool or (type(truncated) is int and truncated in (0, 1)):
        pass
    else:
        raise ValueError("Invalid attribution workflow truncation marker")
    agent["activity_truncated"] = bool(truncated)
    children = source.get("children")
    if children is None:
        children = []
    if not isinstance(children, list):
        raise ValueError("Invalid attribution workflow children")
    agent["children"] = [
        _sanitize_workflow_agent(child, depth + 1, budget) for child in children
    ]
    return agent


def _sanitize_workflow(raw: Any) -> dict[str, Any]:
    """Republish how the work was done: counts, facets, and instruction paths.

    The object carries no locator, no hash, and no text, exactly as the note it
    came from does. Every field is rebuilt from a known key, so a note written
    by some other tool publishes nothing this schema does not name.
    """
    if not isinstance(raw, dict) or raw.get("version") != 1:
        raise ValueError("Invalid attribution workflow")
    agents = raw.get("agents")
    if not isinstance(agents, list):
        raise ValueError("Invalid attribution workflow agents")
    budget = [_MAX_WORKFLOW_AGENTS]
    parsed_agents = [_sanitize_workflow_agent(item, 1, budget) for item in agents]

    raw_totals = raw.get("totals")
    if raw_totals is None:
        raw_totals = {}
    if not isinstance(raw_totals, dict):
        raise ValueError("Invalid attribution workflow totals")
    totals: dict[str, Any] = {
        name: _workflow_count(raw_totals.get(name), f"total {name}")
        for name in _WORKFLOW_TOTAL_FIELDS
    }
    totals["tool_calls"] = _workflow_tool_calls(raw_totals.get("tool_calls"))

    raw_files = raw.get("instruction_files")
    if raw_files is None:
        raw_files = []
    if not isinstance(raw_files, list) or len(raw_files) > _MAX_WORKFLOW_INSTRUCTION_FILES:
        raise ValueError("Invalid attribution workflow instruction files")
    instruction_files: list[dict[str, Any]] = []
    for source in raw_files:
        if not isinstance(source, dict):
            raise ValueError("Invalid attribution workflow instruction file")
        path = _text(source.get("path"), "workflow instruction path", limit=4096)
        parsed_path = PurePosixPath(path)
        if parsed_path.is_absolute() or ".." in parsed_path.parts:
            raise ValueError("Attribution paths must be unique repository-relative paths")
        agent_count = _workflow_count(source.get("agent_count"), "instruction file agents")
        if agent_count is None:
            raise ValueError("Invalid attribution workflow instruction file agents")
        instruction_files.append({
            "path": path,
            "memory_type": _optional_text(
                source.get("memory_type"), "workflow memory type", limit=256
            ),
            "agent_count": agent_count,
        })

    raw_models = raw.get("models")
    if raw_models is None:
        raw_models = []
    if not isinstance(raw_models, list) or len(raw_models) > _MAX_WORKFLOW_AGENTS:
        raise ValueError("Invalid attribution workflow models")
    models: list[dict[str, Any]] = []
    for source in raw_models:
        if not isinstance(source, dict):
            raise ValueError("Invalid attribution workflow model")
        agent_count = _workflow_count(source.get("agents"), "model agents")
        if agent_count is None:
            raise ValueError("Invalid attribution workflow model agents")
        models.append({
            "model": _text(source.get("model"), "workflow model name", limit=256),
            "agents": agent_count,
        })

    return {
        "version": 1,
        "agents": parsed_agents,
        "totals": totals,
        "instruction_files": instruction_files,
        "models": models,
    }


def sanitize_note(raw: Any, commit: str) -> dict[str, Any]:
    """Select public report fields; never publish ledger snapshots or commands."""
    if not isinstance(raw, dict) or raw.get("version") != 1 or raw.get("commit") != commit:
        raise ValueError("Invalid attribution note")
    raw_sessions, raw_files = raw.get("sessions"), raw.get("files")
    if not isinstance(raw_sessions, list) or not isinstance(raw_files, list):
        raise ValueError("Invalid attribution note collections")
    sessions: list[dict[str, Any]] = []
    session_ids: set[str] = set()
    for source in raw_sessions:
        candidate = _sanitize_session(source)
        if candidate["id"] in session_ids:
            raise ValueError("Duplicate attribution session")
        sessions.append(candidate)
        session_ids.add(candidate["id"])

    files: list[dict[str, Any]] = []
    current_owners: set[str] = set()
    paths: set[str] = set()
    for source in raw_files:
        if not isinstance(source, dict):
            raise ValueError("Invalid attribution file")
        path = _text(source.get("path"), "path", limit=4096)
        parsed_path = PurePosixPath(path)
        if parsed_path.is_absolute() or ".." in parsed_path.parts or path in paths:
            raise ValueError("Attribution paths must be unique repository-relative paths")
        added = source.get("added_lines")
        if isinstance(added, bool) or not isinstance(added, int) or not 0 <= added <= 1_000_000:
            raise ValueError("Invalid attribution line count")
        raw_ranges = source.get("ranges", [])
        if not isinstance(raw_ranges, list):
            raise ValueError("Invalid attribution ranges")
        ranges: list[dict[str, Any]] = []
        previous_end, attributed = 0, 0
        for entry in raw_ranges:
            if not isinstance(entry, dict):
                raise ValueError("Invalid attribution range")
            start, end, owner = entry.get("start"), entry.get("end"), entry.get("session_id")
            if (isinstance(start, bool) or isinstance(end, bool)
                    or not isinstance(start, int) or not isinstance(end, int)
                    or not previous_end < start <= end <= 1_000_000 or owner not in session_ids):
                raise ValueError("Invalid attribution line ownership")
            item: dict[str, Any] = {"start": start, "end": end, "session_id": owner}
            tool_use_id = _optional_text(entry.get("tool_use_id"), "tool use id", limit=256)
            if tool_use_id is not None:
                item["tool_use_id"] = tool_use_id
            ranges.append(item)
            previous_end = end
            attributed += end - start + 1
            current_owners.add(owner)
        if attributed > added:
            raise ValueError("Attribution exceeds added lines")
        files.append({"path": path, "added_lines": added, "ranges": ranges})
        paths.add(path)

    revisions: list[dict[str, Any]] = []
    for source in raw.get("revisions", []):
        if not isinstance(source, dict):
            raise ValueError("Invalid attribution revision")
        revision = {field: source.get(field) for field in _REVISION_FIELDS}
        from_commit = revision["from_commit"]
        from_path = revision["from_path"]
        start, end = revision["from_start"], revision["from_end"]
        from_session, to_session = revision["from_session_id"], revision["to_session_id"]
        if (
            not isinstance(from_commit, str) or _OID.fullmatch(from_commit) is None
            or not isinstance(from_path, str) or not from_path or len(from_path) > 4096
            or "\0" in from_path or PurePosixPath(from_path).is_absolute()
            or ".." in PurePosixPath(from_path).parts
            or isinstance(start, bool) or isinstance(end, bool)
            or not isinstance(start, int) or not isinstance(end, int)
            or not 1 <= start <= end <= 1_000_000
            or (from_session is not None and from_session not in session_ids)
            or (to_session is not None and to_session not in session_ids)
        ):
            raise ValueError("Invalid attribution revision")
        revisions.append(revision)

    session_revisions: list[dict[str, Any]] = []
    raw_session_revisions = raw.get("session_revisions", [])
    if not isinstance(raw_session_revisions, list):
        raise ValueError("Invalid attribution session revisions")
    for source in raw_session_revisions:
        if not isinstance(source, dict):
            raise ValueError("Invalid attribution session revision")
        revision = {field: source.get(field) for field in _SESSION_REVISION_FIELDS}
        path = revision["path"]
        if (
            not isinstance(path, str) or not path or len(path) > 4096 or "\0" in path
            or PurePosixPath(path).is_absolute() or ".." in PurePosixPath(path).parts
            or revision["from_session_id"] not in session_ids
            or revision["to_session_id"] not in session_ids
            or isinstance(revision["removed_lines"], bool)
            or not isinstance(revision["removed_lines"], int)
            or not 1 <= revision["removed_lines"] <= 1_000_000
            or revision["kind"] not in {"replace", "delete"}
        ):
            raise ValueError("Invalid attribution session revision")
        session_revisions.append(revision)

    participants = raw.get("contributing_session_ids")
    if participants is None:
        # Legacy notes mix current participants with historical sources. Keep
        # overwritten attempts but remove source-only historical metadata.
        historical_sources: set[str] = set()
        for revision in revisions:
            source, target = revision["from_session_id"], revision["to_session_id"]
            if source is not None:
                historical_sources.add(source)
            if target is not None:
                current_owners.add(target)
        participants = sorted(session_ids - (historical_sources - current_owners))
    if (not isinstance(participants, list)
            or any(not isinstance(item, str) or item not in session_ids for item in participants)):
        raise ValueError("Invalid attribution participants")
    workflow = raw.get("workflow")
    return {
        "version": 1, "commit": commit,
        "recorded_at": _text(raw.get("recorded_at"), "recording time", limit=128),
        "sessions": sessions, "files": files,
        "revisions": revisions, "session_revisions": session_revisions,
        "contributing_session_ids": sorted(set(participants)),
        # A note written before this object existed simply lacks the key, and a
        # snapshot of it says nothing about how the work was done.
        **({"workflow": _sanitize_workflow(workflow)} if workflow is not None else {}),
    }


def sanitize_task(raw: Any, commit: str) -> dict[str, Any]:
    """Select the task economics needed by a PR report, excluding commands.

    A session summary is published with the rest of the session record, because
    a PR footer names what each agent was asked to do.
    """
    if not isinstance(raw, dict):
        raise ValueError("Invalid attribution task")
    task_id = _text(raw.get("id"), "task ID", limit=256)
    kind = raw.get("kind", "feature")
    if kind not in ("feature", "pull_request", "unresolved"):
        raise ValueError("Invalid attribution task kind")
    state = raw.get("state", "active")
    if state not in ("active", "shipped", "abandoned", "merged"):
        raise ValueError("Invalid attribution task state")
    sessions_raw = raw.get("sessions")
    if not isinstance(sessions_raw, list):
        raise ValueError("Invalid attribution task sessions")
    sessions: list[dict[str, Any]] = []
    session_ids: set[str] = set()
    for source in sessions_raw:
        session = _sanitize_session(source)
        if session["id"] in session_ids or session["task_id"] != task_id:
            raise ValueError("Invalid attribution task membership")
        sessions.append(session)
        session_ids.add(session["id"])
    return {
        "id": task_id,
        "name": _text(raw.get("name"), "task name"),
        "kind": kind,
        "pr_ref": _optional_text(raw.get("pr_ref"), "pull request", limit=256),
        "state": state,
        "inference_source": _text(
            raw.get("inference_source", "unknown"), "task inference source"
        ),
        "created_at": _text(raw.get("created_at"), "task creation time", limit=128),
        "updated_at": _text(raw.get("updated_at"), "task update time", limit=128),
        "merged_into": _optional_text(raw.get("merged_into"), "merged task ID", limit=256),
        "anchor_commit": commit,
        "sessions": sessions,
    }


def _usage_model(value: Any) -> str:
    """Return one published model name from an allocated usage record."""
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value) > _MAX_USAGE_MODEL_CHARS
        or any(character < " " or character == "\x7f" for character in value)
    ):
        raise ValueError("Invalid allocated usage model")
    return value


def _usage_number(value: Any) -> float | None:
    """Return one allocated count as a number, or None where nothing was counted."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Invalid allocated usage count")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError("Invalid allocated usage count")
    return number


def _usage_count(value: Any) -> int | None:
    """Return one published count, or None where the spool counted nothing.

    An allocation divides one request between the sessions that shared it, so a
    count arrives as a fraction of an event. The snapshot publishes whole
    numbers, and the fraction it rounds away is below anything a report prints.
    """
    number = _usage_number(value)
    return None if number is None else int(round(number))


def _usage_record(raw: dict[str, Any]) -> dict[str, Any]:
    record: dict[str, Any] = {}
    cost = raw.get("estimated_cost_usd")
    if cost is not None:
        if (
            isinstance(cost, bool) or not isinstance(cost, (int, float))
            or not math.isfinite(cost) or cost < 0
        ):
            raise ValueError("Invalid allocated usage cost")
        record["estimated_cost_usd"] = float(cost)
    for credit_field in ("codex_credits", "codex_api_equivalent_usd"):
        # A Codex subscription session prices in credits, not dollars, so the
        # snapshot carries the credits and their standard-rate dollar
        # comparison beside a Claude session's estimated dollars.
        amount = raw.get(credit_field)
        if amount is not None:
            if (
                isinstance(amount, bool) or not isinstance(amount, (int, float))
                or not math.isfinite(amount) or amount < 0
            ):
                raise ValueError("Invalid allocated usage cost")
            record[credit_field] = float(amount)
    models = raw.get("models")
    if models is not None:
        if not isinstance(models, list) or len(models) > _MAX_USAGE_MODELS:
            raise ValueError("Invalid allocated usage models")
        named = [_usage_model(value) for value in models]
        if named:
            record["models"] = named
    outputs = raw.get("model_output_tokens")
    if outputs is not None:
        if not isinstance(outputs, dict) or len(outputs) > _MAX_USAGE_MODELS:
            raise ValueError("Invalid allocated usage model output tokens")
        counted: dict[str, float] = {}
        for name, value in outputs.items():
            # This count decides which model names a session, so it keeps the
            # three decimals the allocation keeps: rounded to a whole number,
            # two models the local report tells apart can tie on the page.
            number = _usage_number(value)
            if number is None:
                raise ValueError("Invalid allocated usage model output tokens")
            counted[_usage_model(name)] = round(number, 3)
        if counted:
            record["model_output_tokens"] = counted
    for field in _USAGE_COUNT_FIELDS:
        count = _usage_count(raw.get(field))
        if count is not None:
            record[field] = count
    complete = raw.get("cost_complete")
    if complete is not None:
        if type(complete) is not bool:
            raise ValueError("Invalid allocated usage completeness marker")
        record["cost_complete"] = complete
    allocation = raw.get("allocation")
    if allocation is not None:
        if allocation not in _USAGE_ALLOCATIONS:
            raise ValueError("Invalid allocated usage allocation")
        record["allocation"] = allocation
    return record


def sanitize_usage(raw: Any) -> dict[str, Any] | None:
    """Select the allocated telemetry a PR report reads, or None to drop it.

    A note is evidence a push must publish exactly or not at all. This record
    is optional beside it: a malformed or oversized one is dropped, because an
    estimated cost must never cost a push its whole snapshot.
    """
    if not isinstance(raw, dict):
        return None
    try:
        record = _usage_record(raw)
    except ValueError:
        return None
    return record or None


def _snapshot_usage(
    root: Path, notes: list[dict[str, Any]], tasks: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    """Return the telemetry this clone allocated to the sessions it publishes.

    A note records what the harness reported, and a headless session reports no
    model and no cost. The provider spool of that same run priced every request
    it made, so the snapshot carries the allocation beside the notes and a clone
    that reads no ledger still names the model and the estimate. A task names
    the sessions that worked it without writing a line, so its sessions are
    priced here too, or the footer would call their cost unknown.
    """
    named: list[str] = []
    seen: set[str] = set()
    for record in (*notes, *tasks):
        for session in record["sessions"]:
            if session["id"] not in seen:
                seen.add(session["id"])
                named.append(session["id"])
    if not named:
        return {}
    try:
        # Imported here so an ordinary push loads the ledger reader only when
        # there are sessions to price.
        from .costing import allocate_session_usage
        from .report import _Warnings, _load_local_sessions

        common = git_common_dir(root)
        allocated = allocate_session_usage(
            common, _load_local_sessions(common, _Warnings())
        )
    except Exception:
        # Telemetry is optional evidence, so a spool this push cannot read
        # leaves the snapshot without it rather than failing the push.
        print(
            "Joyride: local cost telemetry could not be read; "
            "the PR footer may not show a cost.",
            file=sys.stderr,
        )
        return {}
    usage: dict[str, dict[str, Any]] = {}
    for identifier in named:
        record = sanitize_usage(allocated.get(identifier))
        if record is None:
            continue
        usage[identifier] = record
        if len(usage) >= _MAX_USAGE_SESSIONS:
            break
    return usage


def _reachable_commits(repo: Path, head: str) -> list[str]:
    snapshot_ref(head)
    lines = _git(repo, "rev-list", "--topo-order", "--reverse",
                 f"--max-count={_MAX_COMMITS + 1}", head).stdout.decode("ascii").splitlines()
    if len(lines) > _MAX_COMMITS:
        raise ValueError("Joyride history exceeds the sharing limit")
    return lines


def _note_objects(repo: Path, ref: str = NOTES_REF) -> dict[str, str]:
    result = _git(repo, "notes", f"--ref={ref}", "list", check=False)
    if result.returncode:
        return {}
    notes: dict[str, str] = {}
    for row in result.stdout.decode("ascii").splitlines():
        blob, commit = row.split()
        if not _OID.fullmatch(blob) or not _OID.fullmatch(commit):
            raise ValueError("Invalid attribution note object")
        notes[commit] = blob
        if len(notes) > _MAX_COMMITS:
            raise ValueError("Too many attribution notes to share")
    return notes


def _record_captured_commits(repo: Path, commits: list[str], existing: dict[str, str]) -> None:
    database = git_common_dir(repo) / "attribution" / "ledger.sqlite3"
    if not database.is_file():
        return
    worktree = str(git_dir(repo))
    connection = open_db(repo)
    try:
        bases = {row[0] for row in connection.execute(
            "SELECT DISTINCT base_commit FROM sessions WHERE worktree_id = ? AND ended_at IS NOT NULL",
            (worktree,),
        )}
        if not bases:
            return
        parent_rows = _git(repo, "rev-list", "--parents", "--no-walk=unsorted", "--stdin",
                           input_bytes=("\n".join(commits) + "\n").encode()).stdout.decode("ascii").splitlines()
        parents_by_commit = {}
        for row in parent_rows:
            fields = row.split()
            parents_by_commit[fields[0]] = fields[1] if len(fields) > 1 else None
        checked = 0
        for commit in commits:
            if commit in existing:
                continue
            parent = parents_by_commit[commit]
            if parent not in bases:
                continue
            checked += 1
            if checked > _MAX_CAPTURE_COMMITS:
                raise ValueError("Too many captured commits for one push")
            target_tree = _tree(repo, commit)
            for change in _changed_files(repo, parent, commit):
                path = change.new_path or change.old_path
                if path is None:
                    continue
                entry = target_tree.get(change.new_path) if change.new_path else None
                target, skipped = _blob_content(repo, entry)
                if skipped is not None:
                    continue
                edits = _eligible_edits(connection, path, parent, worktree)
                if _exact_chain(edits, target):
                    record_commit(repo, commit)
                    break
    finally:
        connection.close()


def _read_note_payloads(
    repo: Path, selected: list[tuple[str, str]]
) -> tuple[list[tuple[str, Any]], int]:
    if not selected:
        return [], 0
    object_input = ("\n".join(blob for _commit, blob in selected) + "\n").encode()
    headers = _git(repo, "cat-file", "--batch-check", input_bytes=object_input).stdout.splitlines()
    if len(headers) != len(selected):
        raise ValueError("Could not inspect attribution notes")
    sizes: list[int] = []
    raw_total = 0
    for (_commit, blob), header in zip(selected, headers):
        fields = header.split()
        if len(fields) != 3 or fields[0].decode() != blob or fields[1] != b"blob":
            raise ValueError("Joyride note is not a blob")
        size = int(fields[2])
        if not 0 <= size <= _MAX_NOTE_BYTES:
            raise ValueError("Joyride note exceeds the sharing limit")
        sizes.append(size)
        raw_total += size
    contents = _git(repo, "cat-file", "--batch", input_bytes=object_input).stdout
    payloads: list[tuple[str, Any]] = []
    cursor = 0
    for (commit, blob), size in zip(selected, sizes):
        header_end = contents.find(b"\n", cursor)
        expected_header = f"{blob} blob {size}".encode()
        if header_end < 0 or contents[cursor:header_end] != expected_header:
            raise ValueError("Could not read attribution note")
        start, end = header_end + 1, header_end + 1 + size
        if contents[end:end + 1] != b"\n":
            raise ValueError("Truncated attribution note")
        payloads.append((commit, json.loads(contents[start:end])))
        cursor = end + 1
    return payloads, raw_total


def build_snapshot(repo: str | Path, head: str, *, record_missing: bool = True) -> bytes:
    root = repository_root(repo)
    commits = _reachable_commits(root, head)
    objects = _note_objects(root)
    if record_missing:
        _record_captured_commits(root, commits, objects)
        objects = _note_objects(root)
    selected = [(commit, objects[commit]) for commit in commits if commit in objects]
    task_objects = _note_objects(root, TASK_NOTES_REF)
    selected_tasks = [
        (commit, task_objects[commit]) for commit in commits if commit in task_objects
    ]
    note_payloads, note_raw_total = _read_note_payloads(root, selected)
    task_payloads, task_raw_total = _read_note_payloads(root, selected_tasks)
    if note_raw_total + task_raw_total > MAX_SNAPSHOT_BYTES * 4:
        raise ValueError("Joyride snapshot exceeds 8 MiB")
    notes: list[dict[str, Any]] = []
    note_bytes = 0
    for commit, raw in note_payloads:
        notes.append(sanitize_note(raw, commit))
        # Checking as we accumulate also bounds the total generated metadata.
        note_bytes += len(json.dumps(notes[-1], ensure_ascii=False).encode())
        if note_bytes > MAX_SNAPSHOT_BYTES:
            raise ValueError("Joyride snapshot exceeds 8 MiB")
    tasks: dict[str, dict[str, Any]] = {}
    for commit, raw in task_payloads:
        if (
            not isinstance(raw, dict)
            or raw.get("version") != 1
            or raw.get("commit") != commit
            or not isinstance(raw.get("tasks"), list)
        ):
            raise ValueError("Invalid attribution task note")
        _text(raw.get("recorded_at"), "task recording time", limit=128)
        for source in raw["tasks"]:
            task = sanitize_task(source, commit)
            tasks[task["id"]] = task
            note_bytes += len(json.dumps(task, ensure_ascii=False).encode())
            if note_bytes > MAX_SNAPSHOT_BYTES:
                raise ValueError("Joyride snapshot exceeds 8 MiB")
    payload: dict[str, Any] = {
        "version": 1,
        "head_commit": head,
        "notes": notes,
        "tasks": list(tasks.values()),
    }
    # A snapshot of a push with no telemetry keeps the exact bytes it had
    # before this object existed, so the key appears only where usage does.
    usage = _snapshot_usage(root, notes, list(tasks.values()))
    if usage:
        payload["usage"] = usage
    snapshot = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False).encode()
    if len(snapshot) > MAX_SNAPSHOT_BYTES:
        raise ValueError("Joyride snapshot exceeds 8 MiB")
    return snapshot


def build_traces(repo: str | Path, snapshot: bytes) -> dict[str, bytes]:
    """Return the trace of each session the snapshot names, by session id.

    Over the snapshot's trace budget, the sessions that own the fewest lines
    lose their traces first, so the sessions a review looks at keep theirs.
    """

    root = repository_root(repo)
    if not traces_enabled(root):
        return {}
    notes = json.loads(snapshot)["notes"]
    owned: dict[str, int] = {}
    for note in notes:
        for entry in note["files"]:
            for item in entry["ranges"]:
                owned[item["session_id"]] = (
                    owned.get(item["session_id"], 0) + item["end"] - item["start"] + 1
                )
        for session_id in note["contributing_session_ids"]:
            owned.setdefault(session_id, 0)
    if not owned or not (git_common_dir(root) / "attribution" / "ledger.sqlite3").is_file():
        return {}
    traces: dict[str, bytes] = {}
    connection = open_db(root)
    try:
        for session_id in sorted(owned):
            if _TRACE_NAME.fullmatch(session_id) is None:
                continue
            trace = build_trace(connection, session_id, notes)
            if trace is not None:
                traces[session_id] = canonical_bytes(trace)
    finally:
        connection.close()
    total = sum(len(body) for body in traces.values())
    for session_id in sorted(traces, key=lambda item: (owned[item], item)):
        if total <= MAX_SNAPSHOT_TRACE_BYTES:
            break
        total -= len(traces.pop(session_id))
        print(
            f"Joyride: the trace of session {session_id} was left out of the"
            " snapshot because the traces exceed 32 MiB.",
            file=sys.stderr,
        )
    return traces


def _snapshot_evidence(raw: bytes) -> tuple[bytes, bool] | None:
    """Return a snapshot without its telemetry, and whether it carried any.

    The notes and tasks of a snapshot are evidence one push publishes exactly
    once. The telemetry beside them is optional, so two snapshots of one head
    are the same publication when everything but that object matches.
    """
    try:
        payload = json.loads(raw)
    except ValueError:
        return None
    if not isinstance(payload, dict):
        return None
    usage = payload.pop("usage", None)
    evidence = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False).encode()
    return evidence, bool(usage)


def _published_snapshot(
    root: Path, remote: str, reference: str, environment: dict[str, str]
) -> tuple[str, bytes] | None:
    """Return the metadata commit and snapshot bytes the remote already holds."""
    listed = _git(root, "ls-remote", "--exit-code", "--refs", "--", remote, reference,
                  environment=environment, check=False)
    rows = listed.stdout.decode("ascii", "replace").splitlines()
    if listed.returncode or len(rows) != 1:
        return None
    commit = rows[0].split("\t", 1)[0]
    if _OID.fullmatch(commit) is None:
        return None
    fetched = _git(root, "fetch", "--no-tags", "--depth=1", "--", remote, reference,
                   environment=environment, check=False)
    published = _git(root, "cat-file", "blob", f"{commit}:attribution.json",
                     environment=environment, check=False)
    if fetched.returncode or published.returncode or len(published.stdout) > MAX_SNAPSHOT_BYTES:
        return None
    return commit, published.stdout


def share_snapshot(repo: str | Path, head: str, remote: str) -> str:
    root = repository_root(repo)
    reference = snapshot_ref(head)
    snapshot = build_snapshot(root, head)
    blob = _git(root, "hash-object", "-w", "--stdin", input_bytes=snapshot).stdout.decode().strip()
    entries = [f"100644 blob {blob}\tattribution.json"]
    traces = build_traces(root, snapshot)
    if traces:
        trace_entries = []
        for session_id, body in sorted(traces.items()):
            trace_blob = _git(root, "hash-object", "-w", "--stdin", input_bytes=body).stdout.decode().strip()
            trace_entries.append(f"100644 blob {trace_blob}\t{session_id}.json")
        trace_tree = _git(root, "mktree", input_bytes=("\n".join(trace_entries) + "\n").encode()).stdout.decode().strip()
        entries.append(f"040000 tree {trace_tree}\ttraces")
    tree = _git(root, "mktree", input_bytes=("\n".join(entries) + "\n").encode()).stdout.decode().strip()
    environment = os.environ.copy()
    environment.update({
        "GIT_AUTHOR_NAME": "Joyride metadata", "GIT_AUTHOR_EMAIL": "attribution@localhost",
        "GIT_COMMITTER_NAME": "Joyride metadata", "GIT_COMMITTER_EMAIL": "attribution@localhost",
        "GIT_AUTHOR_DATE": "2000-01-01T00:00:00Z", "GIT_COMMITTER_DATE": "2000-01-01T00:00:00Z",
        "GIT_TERMINAL_PROMPT": "0", _GUARD: "1",
    })
    metadata_commit = _git(root, "-c", "commit.gpgsign=false", "commit-tree", tree,
                           input_bytes=b"PR attribution metadata\n", environment=environment).stdout.decode().strip()
    # No local ref mutation. Parentless, deterministic metadata commits make
    # each exact-head ref repeat exactly once its evidence has been published.
    pushed = _git(root, "push", "--no-verify", "--porcelain", "--", remote,
                  f"{metadata_commit}:{reference}", environment=environment, check=False)
    if pushed.returncode:
        # The remote already holds a snapshot of this head, and it differs. A
        # provider prices the requests of a session after the push that carried
        # its commit, so a headless session that reported no model and no cost
        # would keep an unknown footer for good. Refresh the ref where the notes
        # and the tasks are byte-identical and only the telemetry beside them
        # arrived, and never replace published telemetry with none.
        published = _published_snapshot(root, remote, reference, environment)
        evidence, carries_usage = _snapshot_evidence(snapshot)
        earlier = None if published is None else _snapshot_evidence(published[1])
        if earlier is None or earlier[0] != evidence or not carries_usage:
            raise ValueError("Joyride Git operation failed: push")
        _git(root, "push", "--no-verify", "--porcelain",
             f"--force-with-lease={reference}:{published[0]}", "--", remote,
             f"{metadata_commit}:{reference}", environment=environment)
    return reference


def pre_push(repo: str | Path, remote: str, rows: str, *, stderr: TextIO = sys.stderr) -> int:
    if os.environ.get(_GUARD) == "1":
        return 0
    heads: set[str] = set()
    for row in rows.splitlines():
        fields = row.split()
        if len(fields) != 4:
            print("Joyride: skipped malformed push metadata; branch push will continue.", file=stderr)
            continue
        _local_ref, local_oid, remote_ref, _remote_oid = fields
        if remote_ref.startswith("refs/heads/") and local_oid.strip("0"):
            heads.add(local_oid)
    for head in sorted(heads):
        try:
            share_snapshot(repo, head, remote)
        except Exception:
            print("Joyride: metadata could not be shared; branch push will continue and the PR footer may be unavailable.", file=stderr)
    return 0


def _push_is_dry_run(arguments: str) -> bool | None:
    """Conservatively recognize Git's flattened process command line.

    ``ps`` does not preserve argument boundaries or shell quoting. Unrecognized
    global arguments therefore suppress sharing instead of guessing. Splitting
    on whitespace still recognizes an actual dry-run flag, including when
    another argument contains literal quotation marks.
    """
    fields = arguments.split()
    if not fields or Path(fields[0]).name != "git":
        return None
    index = 1
    values = {"-C", "-c", "--git-dir", "--work-tree", "--namespace", "--config-env"}
    switches = {
        "-p", "-P", "--paginate", "--no-pager", "--bare", "--literal-pathspecs",
        "--no-literal-pathspecs", "--no-replace-objects", "--no-optional-locks",
        "--no-lazy-fetch", "--no-advice",
    }
    while index < len(fields):
        argument = fields[index]
        if argument == "push":
            break
        if argument in values:
            index += 2
        elif argument in switches or argument.startswith(("-C", "-c", "--git-dir=", "--work-tree=", "--namespace=", "--config-env=")):
            index += 1
        else:
            return None
    if index >= len(fields):
        return None
    for argument in fields[index + 1:]:
        # Git accepts abbreviated long options and combined short options.
        # Do not trust an apparent "--" separator: it could be inside a
        # flattened option value. A false positive only skips optional sharing.
        if argument == "--":
            continue
        if argument.startswith("--") and "--dry-run".startswith(argument):
            return True
        if re.fullmatch(r"-[A-Za-z0-9]+", argument) and "n" in argument[1:]:
            return True
    return False


def _outer_push_is_dry_run() -> bool | None:
    """Inspect Unix ancestors without exposing credential-bearing arguments."""
    if sys.platform not in ("darwin", "linux"):
        return None
    process = os.getppid()
    visited: set[int] = set()
    for _ in range(24):
        if process <= 1 or process in visited:
            return None
        visited.add(process)
        try:
            identity = subprocess.run(
                ["ps", "-ww", "-p", str(process), "-o", "ppid=", "-o", "comm="],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                env=system_subprocess_environment(),
                text=True, timeout=1, check=False,
            )
            if identity.returncode:
                return None
            parent, executable = identity.stdout.strip().split(maxsplit=1)
            if Path(executable).name == "git":
                command = subprocess.run(
                    ["ps", "-ww", "-p", str(process), "-o", "args="],
                    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                    env=system_subprocess_environment(),
                    text=True, timeout=1, check=False,
                )
                return _push_is_dry_run(command.stdout) if command.returncode == 0 else None
            process = int(parent)
        except (OSError, ValueError, subprocess.TimeoutExpired):
            return None
    return None


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    if len(arguments) != 2 or os.environ.get(_GUARD) == "1":
        return 0
    dry_run = _outer_push_is_dry_run()
    if dry_run is not False:
        if dry_run is None:
            print("Joyride: metadata skipped because the outer push context could not be verified.", file=sys.stderr)
        return 0
    try:
        return pre_push(Path.cwd(), arguments[1], sys.stdin.read())
    except Exception:
        print("Joyride: metadata capture failed; branch push will continue.", file=sys.stderr)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
