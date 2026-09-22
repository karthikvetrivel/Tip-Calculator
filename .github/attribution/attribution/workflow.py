"""One workflow profile, shared by the report, a note, and a pull request.

A profile says how a task was worked: which agents ran, what each one loaded,
and what each one cost. It carries counts, agent types, models, roles, launch
modes, and repo-relative instruction paths. It never carries a tool locator, a
hash, a URL, or any text, because ``record_commit`` publishes this object in a
Git note, and a note publishes nothing the repository does not already hold.

``build_profile`` reads evidence this project recorded. ``parse_profile`` reads
the same object back out of a note that some other clone wrote, so it rebuilds
every field from a known key and raises on anything else.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import hashlib
import math
from typing import Any

from .activity import TOOL_CLASSES


def native_identity_hash(session: Mapping[str, Any]) -> str | None:
    """Join both native session and child identity without exporting either."""
    native = session.get("native_session_id")
    if not isinstance(native, str) or not native or len(native) > 2048:
        return None
    child = session.get("native_agent_id")
    if isinstance(child, str) and child and not native.endswith("::" + child):
        native += "::" + child
    return hashlib.sha256(native.encode()).hexdigest()


PROFILE_VERSION = 1
# How deep one profile may nest its agents. The snapshot schema in
# ``sharing.py`` refuses a workflow object that nests deeper, and a note the
# snapshot refuses costs that push its whole snapshot, so the writer in
# ``notes.py`` and the reader in ``sharing.py`` both read the number here.
MAX_WORKFLOW_DEPTH = 8
# A commit's contributing sessions are few, and a task's are bounded by the
# sessions one repository recorded. Both limits guard the reader against a note
# that some other tool wrote.
MAX_PROFILE_AGENTS = 5_000
MAX_PROFILE_INSTRUCTION_FILES = 1_000
_MAX_TEXT_CHARS = 256
_MAX_PATH_CHARS = 1_024
_MAX_COUNT = 1_000_000_000

# The text facets of one agent, in the order the profile prints them. ``model``
# and ``harness`` label the work; the rest say how the agent started and how it
# was allowed to act.
_AGENT_TEXT_FIELDS = (
    "agent_type",
    "harness",
    "model",
    "role",
    "launch_mode",
    "session_source",
    "permission_mode",
    "effort_level",
)
# Counts a session accumulates. Null means that no harness reported one.
_AGENT_COUNT_FIELDS = (
    "turn_count",
    "prompt_count",
    "interrupt_count",
    "compaction_count",
    "model_switch_count",
)
# The totals of a profile, in the order this project's specification writes
# them. One agent lists its turns first; the totals of a task lead with the
# prompts that asked for the work.
_TOTAL_NAMES = {
    "prompt_count": "prompts",
    "turn_count": "turns",
    "interrupt_count": "interrupts",
    "compaction_count": "compactions",
    "model_switch_count": "model_switches",
}


def _text(value: Any, *, limit: int = _MAX_TEXT_CHARS) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    if not cleaned or len(cleaned) > limit or "\x00" in cleaned:
        return None
    return cleaned


def _count(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if 0 <= value <= _MAX_COUNT else None


def _cost(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) and number >= 0 else None


def _flag(value: Any) -> bool:
    return value is True or value == 1


def tool_class_counts(value: Any) -> dict[str, int] | None:
    """Return one agent's tool calls by class, or None when none are known."""

    if not isinstance(value, Mapping):
        return None
    counts = dict.fromkeys(TOOL_CLASSES, 0)
    known = False
    for name, raw in value.items():
        # A class this version does not know belongs to a later one. Its count
        # is dropped rather than folded into ``other``, which would claim a
        # classification that this table did not make.
        if name not in counts:
            continue
        number = _count(raw)
        if number is not None:
            counts[name] = number
            known = True
    return counts if known else None


def instruction_loads(rows: Iterable[Any]) -> list[dict[str, Any]]:
    """Return the distinct instruction files that one session loaded."""

    result: list[dict[str, Any]] = []
    seen: set[tuple[str, str | None]] = set()
    for row in rows or ():
        if not isinstance(row, Mapping) or row.get("kind") != "instruction_file":
            continue
        # The locator of an instruction file is its repo-relative path, or the
        # ``outside_repo`` label. Both are publishable; the hash beside them in
        # the ledger is not.
        path = _text(row.get("locator"), limit=_MAX_PATH_CHARS)
        if path is None:
            continue
        memory_type = _text(row.get("memory_type"))
        if (path, memory_type) in seen:
            continue
        seen.add((path, memory_type))
        result.append({"path": path, "memory_type": memory_type})
    return result


def _agent(session: Mapping[str, Any], activity: Mapping[str, Any]) -> dict[str, Any] | None:
    session_id = _text(session.get("id"))
    if session_id is None:
        return None
    counts: dict[str, int | None] = {}
    recorded = activity.get("counts")
    recorded = recorded if isinstance(recorded, Mapping) else {}
    for field in _AGENT_COUNT_FIELDS:
        value = _count(session.get(field))
        counts[field] = value if value is not None else _count(recorded.get(field))
    return {
        "session_id": session_id,
        "parent_session_id": _text(session.get("parent_session_id")),
        **{field: _text(session.get(field)) for field in _AGENT_TEXT_FIELDS},
        **counts,
        "tool_calls": tool_class_counts(activity.get("tool_calls")),
        "duration_ms": _count(session.get("duration_ms")),
        "cost_usd": _cost(session.get("cost_usd")),
        "activity_truncated": _flag(session.get("activity_truncated")),
        "children": [],
    }


def _worked(
    agent: Mapping[str, Any], activity: Mapping[str, Any], edited: frozenset[str]
) -> bool:
    """Say whether this agent did more than open and read its instructions.

    A harness loads its instruction files into every agent it starts, including
    one whose user then typed nothing. Such a session is real evidence that an
    agent opened, and the report still lists it, but it worked no task: every
    total here, and the agent count of each instruction file with them, would
    otherwise describe one more agent than the task had. An agent whose loads
    this project never recorded is unknown rather than idle, so the question is
    asked only of a session that recorded one.
    """

    if agent["session_id"] in edited:
        return True
    if any(agent[field] for field in _AGENT_COUNT_FIELDS):
        return True
    calls = agent["tool_calls"]
    if calls is not None and any(calls.values()):
        return True
    return not activity.get("instruction_files")


def _nests_under(agent: dict[str, Any], by_id: Mapping[str, dict[str, Any]]) -> bool:
    """Say whether this agent's parent chain ends instead of closing a loop."""

    seen = {agent["session_id"]}
    parent_id = agent["parent_session_id"]
    while parent_id is not None:
        if parent_id in seen:
            return False
        seen.add(parent_id)
        parent = by_id.get(parent_id)
        if parent is None:
            return False
        parent_id = parent["parent_session_id"]
    return True


def _total(agents: list[dict[str, Any]], field: str) -> int | None:
    values = [agent[field] for agent in agents if agent[field] is not None]
    return sum(values) if values else None


def _instruction_files(
    agents: list[dict[str, Any]],
    activity: Mapping[str, Mapping[str, Any]],
    recorded: Iterable[Any],
) -> list[dict[str, Any]]:
    counted: dict[tuple[str, str | None], int] = {}
    for agent in agents:
        entry = activity.get(agent["session_id"]) or {}
        for item in instruction_loads(entry.get("instruction_files") or ()):
            key = (item["path"], item["memory_type"])
            counted[key] = counted.get(key, 0) + 1
    for item in recorded or ():
        if not isinstance(item, Mapping):
            continue
        path = _text(item.get("path"), limit=_MAX_PATH_CHARS)
        if path is None:
            continue
        key = (path, _text(item.get("memory_type")))
        # A note says how many agents loaded a file, not which. Its count
        # therefore raises the count this profile counted rather than adding to
        # it, and never claims more agents than the tree holds.
        agent_count = min(_count(item.get("agent_count")) or 0, len(agents))
        counted[key] = max(counted.get(key, 0), agent_count)
    files = [
        {"path": path, "memory_type": memory_type, "agent_count": count}
        for (path, memory_type), count in sorted(
            counted.items(), key=lambda item: (item[0][0], item[0][1] or "")
        )
        if count
    ]
    return files[:MAX_PROFILE_INSTRUCTION_FILES]


def build_profile(
    sessions: Iterable[Mapping[str, Any]],
    *,
    activity: Mapping[str, Mapping[str, Any]] | None = None,
    instruction_files: Iterable[Any] | None = None,
    edited: Iterable[str] | None = None,
) -> dict[str, Any] | None:
    """Return the workflow profile of one task, or None when no agent is known.

    ``activity`` holds what each session did, keyed by session ID: its
    ``tool_calls`` by class, its ``instruction_files`` as ``context_loads``
    rows, and the ``counts`` a note recorded for a session this clone never
    ran. ``instruction_files`` holds the profile-level entries of a note, which
    name a file and how many agents loaded it without naming which. ``edited``
    names the sessions that own a line of the repository, which worked the task
    whatever their harness reported.
    """

    activity = activity or {}
    edited = frozenset(edited or ())
    # Oldest first, so the agent that began the work leads its own subtree.
    records = sorted(
        (session for session in sessions if isinstance(session, Mapping)),
        key=lambda session: (
            _text(session.get("started_at"), limit=_MAX_TEXT_CHARS) or "",
            _text(session.get("id")) or "",
        ),
    )
    agents: list[dict[str, Any]] = []
    idle: set[str] = set()
    for session in records:
        entry = activity.get(_text(session.get("id")) or "") or {}
        agent = _agent(session, entry)
        if agent is None:
            continue
        agents.append(agent)
        if not _worked(agent, entry, edited):
            idle.add(agent["session_id"])
    if idle:
        # An agent that launched another shaped the work, whatever its own
        # session recorded, and dropping it would orphan the tree below it.
        idle -= {agent["parent_session_id"] for agent in agents}
        agents = [agent for agent in agents if agent["session_id"] not in idle]
    if not agents:
        return None

    by_id: dict[str, dict[str, Any]] = {}
    for agent in agents:
        by_id.setdefault(agent["session_id"], agent)
    roots: list[dict[str, Any]] = []
    for agent in agents:
        parent = by_id.get(agent["parent_session_id"] or "")
        # An agent whose parent is unknown, or whose parents close a loop,
        # stays at the top level with its parent still named.
        if parent is None or parent is agent or not _nests_under(agent, by_id):
            roots.append(agent)
        else:
            parent["children"].append(agent)

    # A task whose sessions recorded no tool call at all has an unknown mix,
    # not an empty one. Only a recorded session turns the total into a count.
    tool_calls: dict[str, int] | None = None
    for agent in agents:
        if agent["tool_calls"] is None:
            continue
        tool_calls = tool_calls or dict.fromkeys(TOOL_CLASSES, 0)
        for name, count in agent["tool_calls"].items():
            tool_calls[name] += count
    totals = {
        "agents": len(agents),
        "subagents": sum(1 for agent in agents if agent["parent_session_id"]),
        **{_TOTAL_NAMES[field]: _total(agents, field) for field in _TOTAL_NAMES},
        "tool_calls": tool_calls,
    }
    models: dict[str, int] = {}
    for agent in agents:
        if agent["model"] is not None:
            models[agent["model"]] = models.get(agent["model"], 0) + 1
    return {
        "version": PROFILE_VERSION,
        "agents": roots,
        "totals": totals,
        "instruction_files": _instruction_files(agents, activity, instruction_files or ()),
        "models": [
            {"model": model, "agents": count}
            for model, count in sorted(models.items(), key=lambda item: (-item[1], item[0]))
        ],
    }


def iter_agents(profile: Any) -> list[dict[str, Any]]:
    """Return every agent of a profile, each parent before its children."""

    result: list[dict[str, Any]] = []
    if not isinstance(profile, Mapping):
        return result
    pending = list(profile.get("agents") or ())
    pending.reverse()
    while pending:
        agent = pending.pop()
        if not isinstance(agent, Mapping):
            continue
        result.append(dict(agent))
        children = [child for child in (agent.get("children") or ())]
        pending.extend(reversed(children))
        if len(result) > MAX_PROFILE_AGENTS:
            break
    return result


def profile_activity(profile: Any) -> dict[str, dict[str, Any]]:
    """Return what a profile records for each session it names."""

    result: dict[str, dict[str, Any]] = {}
    for agent in iter_agents(profile):
        session_id = _text(agent.get("session_id"))
        if session_id is None:
            continue
        entry: dict[str, Any] = {}
        calls = tool_class_counts(agent.get("tool_calls"))
        if calls is not None:
            entry["tool_calls"] = calls
        counts = {
            field: agent[field]
            for field in _AGENT_COUNT_FIELDS
            if _count(agent.get(field)) is not None
        }
        if counts:
            entry["counts"] = counts
        if entry:
            result[session_id] = entry
    return result


def merge_activity(
    existing: Mapping[str, Any] | None, candidate: Mapping[str, Any]
) -> dict[str, Any]:
    """Keep the larger of two records of one session's activity.

    Two notes describe the same session at two moments of its life. Neither
    unrecorded a call, so the later note is the one that saw more.
    """

    if existing is None:
        return dict(candidate)
    merged: dict[str, Any] = dict(existing)
    calls = existing.get("tool_calls")
    other = candidate.get("tool_calls")
    if isinstance(calls, Mapping) and isinstance(other, Mapping):
        merged["tool_calls"] = {
            name: max(calls.get(name, 0), other.get(name, 0)) for name in TOOL_CLASSES
        }
    elif isinstance(other, Mapping):
        merged["tool_calls"] = dict(other)
    counts = existing.get("counts")
    counts = dict(counts) if isinstance(counts, Mapping) else {}
    for field, value in (candidate.get("counts") or {}).items():
        counts[field] = max(counts.get(field, 0), value)
    if counts:
        merged["counts"] = counts
    return merged


def _parse_agent(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ValueError("an agent is not an object")
    session_id = _text(raw.get("session_id"))
    if session_id is None:
        raise ValueError("an agent has no session ID")
    agent: dict[str, Any] = {
        "session_id": session_id,
        "parent_session_id": _text(raw.get("parent_session_id")),
    }
    for field in _AGENT_TEXT_FIELDS:
        agent[field] = _text(raw.get(field))
    for field in _AGENT_COUNT_FIELDS:
        value = raw.get(field)
        if value is not None and _count(value) is None:
            raise ValueError(f"agent {session_id!r} has an invalid {field}")
        agent[field] = _count(value)
    calls = raw.get("tool_calls")
    if calls is not None and not isinstance(calls, Mapping):
        raise ValueError(f"agent {session_id!r} has invalid tool calls")
    for name in (calls or {}):
        if name in TOOL_CLASSES and _count(calls[name]) is None:
            raise ValueError(f"agent {session_id!r} has an invalid {name} count")
    agent["tool_calls"] = tool_class_counts(calls)
    duration = raw.get("duration_ms")
    if duration is not None and _count(duration) is None:
        raise ValueError(f"agent {session_id!r} has an invalid duration")
    agent["duration_ms"] = _count(duration)
    cost = raw.get("cost_usd")
    if cost is not None and _cost(cost) is None:
        raise ValueError(f"agent {session_id!r} has an invalid cost")
    agent["cost_usd"] = _cost(cost)
    agent["activity_truncated"] = _flag(raw.get("activity_truncated"))
    agent["children"] = []
    return agent


def parse_profile(raw: Any) -> dict[str, Any]:
    """Return the workflow object of a note, rebuilt from its known fields.

    A note is read-only evidence that another clone wrote, so every value is
    rebuilt rather than copied: an unknown key never reaches a report, and a
    malformed one raises for the caller to record as a warning.
    """

    if not isinstance(raw, Mapping):
        raise ValueError("workflow is not an object")
    if raw.get("version") != PROFILE_VERSION:
        raise ValueError("unsupported workflow version")
    if not isinstance(raw.get("agents"), list):
        raise ValueError("workflow agents must be an array")

    roots: list[dict[str, Any]] = []
    total = 0
    # A note nests its agents, so the walk carries the parsed parent that each
    # raw child belongs under.
    pending: list[tuple[Any, dict[str, Any] | None]] = [
        (item, None) for item in reversed(raw["agents"])
    ]
    seen: set[str] = set()
    while pending:
        item, parent = pending.pop()
        agent = _parse_agent(item)
        if agent["session_id"] in seen:
            raise ValueError("workflow names one agent twice")
        seen.add(agent["session_id"])
        total += 1
        if total > MAX_PROFILE_AGENTS:
            raise ValueError("workflow contains too many agents")
        if parent is None:
            roots.append(agent)
        else:
            parent["children"].append(agent)
        children = item.get("children")
        if children is not None and not isinstance(children, list):
            raise ValueError(f"agent {agent['session_id']!r} has invalid children")
        pending.extend((child, agent) for child in reversed(children or ()))

    files = raw.get("instruction_files")
    if files is not None and not isinstance(files, list):
        raise ValueError("workflow instruction files must be an array")
    if files is not None and len(files) > MAX_PROFILE_INSTRUCTION_FILES:
        raise ValueError("workflow lists too many instruction files")
    instruction_files: list[dict[str, Any]] = []
    for item in files or ():
        if not isinstance(item, Mapping):
            raise ValueError("an instruction file is not an object")
        path = _text(item.get("path"), limit=_MAX_PATH_CHARS)
        agent_count = _count(item.get("agent_count"))
        if path is None or agent_count is None:
            raise ValueError("an instruction file is invalid")
        instruction_files.append(
            {
                "path": path,
                "memory_type": _text(item.get("memory_type")),
                "agent_count": agent_count,
            }
        )

    models = raw.get("models")
    if models is not None and not isinstance(models, list):
        raise ValueError("workflow models must be an array")
    parsed_models: list[dict[str, Any]] = []
    for item in models or ():
        if not isinstance(item, Mapping):
            raise ValueError("a model entry is not an object")
        model = _text(item.get("model"))
        agent_count = _count(item.get("agents"))
        if model is None or agent_count is None:
            raise ValueError("a model entry is invalid")
        parsed_models.append({"model": model, "agents": agent_count})

    raw_totals = raw.get("totals")
    if raw_totals is not None and not isinstance(raw_totals, Mapping):
        raise ValueError("workflow totals must be an object")
    raw_totals = raw_totals or {}
    totals: dict[str, Any] = {}
    for name in ("agents", "subagents", *_TOTAL_NAMES.values()):
        value = raw_totals.get(name)
        if value is not None and _count(value) is None:
            raise ValueError(f"workflow total {name} is invalid")
        totals[name] = _count(value)
    calls = raw_totals.get("tool_calls")
    if calls is not None and not isinstance(calls, Mapping):
        raise ValueError("workflow total tool calls must be an object")
    for name in (calls or {}):
        if name in TOOL_CLASSES and _count(calls[name]) is None:
            raise ValueError(f"workflow total {name} calls are invalid")
    totals["tool_calls"] = tool_class_counts(calls)
    if totals["agents"] is None:
        totals["agents"] = total
    if totals["subagents"] is None:
        totals["subagents"] = sum(
            1 for agent in iter_agents({"agents": roots}) if agent["parent_session_id"]
        )
    return {
        "version": PROFILE_VERSION,
        "agents": roots,
        "totals": totals,
        "instruction_files": instruction_files,
        "models": parsed_models,
    }
