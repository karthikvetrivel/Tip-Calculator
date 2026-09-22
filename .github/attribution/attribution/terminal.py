"""Small, dependency-free terminal renderers for attribution reports."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
import math
import os
import re
import shlex
import shutil
import sys
import unicodedata
from typing import Any

from .workflow import iter_agents


_DEFAULT_WIDTH = 88
_MAX_WIDTH = 120
_MAX_FEATURES = 30
_MAX_SESSIONS = 30
_MAX_COMMITS = 30
_MAX_REVISIONS = 30
_MAX_FILES_PER_COMMIT = 8
_MAX_RANGES_PER_FILE = 8
_MAX_INSTRUCTION_FILES = 10
_MAX_AGENT_DEPTH = 6
# What a report says one agent loaded, beside the instruction files it read.
_LOADED_TOOL_CLASSES = (
    ("read", "read", "reads"),
    ("search", "search", "searches"),
    ("web", "web fetch", "web fetches"),
    ("skill", "skill", "skills"),
    ("mcp", "MCP call", "MCP calls"),
)

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
_RESET = "\x1b[0m"
_BOLD_CYAN = "\x1b[1;36m"
_BOLD = "\x1b[1m"
_DIM = "\x1b[2m"
_CYAN = "\x1b[36m"
_GREEN = "\x1b[32m"
_YELLOW = "\x1b[33m"
_RED = "\x1b[31m"
_MAGENTA = "\x1b[35m"


def safe_text(value: Any) -> str:
    """Return terminal-safe text with every control/format character visible."""

    if value is None:
        return ""
    text = str(value)
    escaped: list[str] = []
    for character in text:
        codepoint = ord(character)
        if character == "\n":
            escaped.append(r"\n")
        elif character == "\r":
            escaped.append(r"\r")
        elif character == "\t":
            escaped.append(r"\t")
        elif (
            codepoint < 32
            or 127 <= codepoint <= 159
            or unicodedata.category(character).startswith("C")
            or unicodedata.category(character) in {"Zl", "Zp"}
        ):
            if codepoint <= 0xFF:
                escaped.append(f"\\x{codepoint:02x}")
            elif codepoint <= 0xFFFF:
                escaped.append(f"\\u{codepoint:04x}")
            else:
                escaped.append(f"\\U{codepoint:08x}")
        elif character.isspace():
            escaped.append(" ")
        else:
            escaped.append(character)
    return " ".join("".join(escaped).split())


def sanitize_terminal(value: Any) -> str:
    """Compatibility alias for :func:`safe_text`."""

    return safe_text(value)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _items(value: Any) -> list[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return list(value)
    return []


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _integer(value: Any) -> int | None:
    number = _number(value)
    if number is None or not number.is_integer():
        return None
    return int(number)


def _count(value: Any) -> str:
    number = _integer(value)
    return f"{number:,}" if number is not None else "unknown"


def _counted(value: Any, singular: str, plural: str | None = None) -> str:
    number = _integer(value)
    noun = singular if number == 1 else (plural or singular + "s")
    return f"{number:,} {noun}" if number is not None else f"unknown {plural or singular + 's'}"


def _money(value: Any, complete: Any = True) -> str:
    number = _number(value)
    if number is None:
        return "unknown"
    rendered = f"${number:,.2f}"
    return f"{rendered} (incomplete)" if complete is False else rendered


def _table_money(value: Any, complete: Any = True) -> str:
    number = _number(value)
    if number is None:
        return "unknown"
    rendered = f"${number:,.2f}"
    return f"{rendered}*" if complete is False else rendered


def _credits(value: Any) -> str:
    number = _number(value)
    if number is None:
        return "unknown"
    if number == 0:
        return "0 cr"
    if abs(number) < 0.01:
        return f"{number:.4f} cr"
    return f"{number:,.2f} cr"


def _table_credits(value: Any) -> str:
    number = _number(value)
    if number is None:
        return "unknown"
    return f"{number:,.0f} cr" if number.is_integer() else f"{number:,.2f} cr"


def _api_equivalent(value: Any) -> str:
    number = _number(value)
    if number is None:
        return "unknown"
    if 0 < number < 0.01:
        return "<$0.01"
    return f"${number:,.2f}"


def _feature_cost(feature: Mapping[str, Any], *, compact: bool) -> str:
    reported = _number(feature.get("reported_cost_usd"))
    estimated = _number(feature.get("estimated_cost_usd"))
    credits = _number(feature.get("codex_credits"))
    equivalent = _number(feature.get("codex_api_equivalent_usd"))
    components = sum(value is not None for value in (reported, estimated, credits))
    if components == 0:
        return "unknown"
    incomplete = feature.get("cost_complete") is False
    if credits is not None and (reported is not None or estimated is not None):
        result = "mixed units" if compact else (
            f"{_money((reported or 0) + (estimated or 0))} USD + {_credits(credits)}"
        )
    elif credits is not None:
        result = _table_credits(credits) if compact else _credits(credits)
        if equivalent is not None:
            result += (
                f" ≈{_api_equivalent(equivalent)}"
                if compact
                else f" ≈ {_api_equivalent(equivalent)}"
            )
    else:
        usd = (reported or 0) + (estimated or 0)
        if estimated is not None and reported is not None:
            suffix = " mix" if compact else " (reported + estimated)"
        elif estimated is not None:
            suffix = " est." if compact else " estimated"
        else:
            suffix = "" if compact else " reported"
        result = f"${usd:,.2f}{suffix}"
    return f"{result}*" if incomplete and compact else (
        f"{result} (partial)" if incomplete else result
    )


def _session_cost(session: Mapping[str, Any]) -> str:
    reported = _number(session.get("cost_usd"))
    telemetry = _mapping(session.get("telemetry"))
    estimated = _number(telemetry.get("estimated_cost_usd"))
    credits = _number(telemetry.get("codex_credits"))
    equivalent = _number(telemetry.get("codex_api_equivalent_usd"))
    partial = telemetry.get("cost_complete") is False
    if reported is not None:
        return f"${reported:,.2f} reported"
    if estimated is not None:
        result = f"${estimated:,.2f} estimated"
        return f"{result} (partial)" if partial else result
    if credits is not None:
        result = _credits(credits).replace(" cr", " Codex credits")
        if equivalent is not None:
            comparison = _api_equivalent(equivalent)
            result += f" ({comparison} at standard API rates)"
        return f"{result} (partial)" if partial else result
    return "unknown"


def _percent(value: Any) -> str:
    number = _number(value)
    if number is None:
        return "unknown"
    return f"{number:.1f}%" if not number.is_integer() else f"{int(number)}%"


def _duration(value: Any) -> str:
    hours = _number(value)
    if hours is None:
        return "unknown"
    if hours < 1:
        return f"{round(hours * 60):,} min"
    if hours < 48:
        return f"{hours:.1f} hr" if not hours.is_integer() else f"{int(hours)} hr"
    days = hours / 24
    return f"{days:.1f} days" if not days.is_integer() else f"{int(days)} days"


def _date(value: Any) -> str:
    raw = safe_text(value)
    if not raw:
        return "unknown"
    candidate = raw[:-1] + "+00:00" if raw.endswith("Z") else raw
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError:
        return raw
    suffix = "Z" if parsed.utcoffset() is not None and parsed.utcoffset().total_seconds() == 0 else ""
    return parsed.strftime("%Y-%m-%d %H:%M") + suffix


def _width(value: int | None) -> int:
    if value is None:
        value = shutil.get_terminal_size(fallback=(_DEFAULT_WIDTH, 24)).columns
    if isinstance(value, bool) or not isinstance(value, int):
        value = _DEFAULT_WIDTH
    return max(24, min(_MAX_WIDTH, value))


def _display_width(text: str) -> int:
    width = 0
    for character in _ANSI_RE.sub("", text):
        if unicodedata.combining(character):
            continue
        width += 2 if unicodedata.east_asian_width(character) in {"W", "F"} else 1
    return width


def _color_enabled(value: bool | None) -> bool:
    if value is not None:
        return value
    if "NO_COLOR" in os.environ:
        return False
    if os.environ.get("CLICOLOR") == "0" or os.environ.get("TERM") == "dumb":
        return False
    force = os.environ.get("CLICOLOR_FORCE")
    if force not in {None, "", "0"}:
        return True
    return bool(getattr(sys.stdout, "isatty", lambda: False)())


def _style(value: str, code: str, enabled: bool) -> str:
    return f"{code}{value}{_RESET}" if enabled and value else value


def _state_style(value: str) -> str:
    lowered = value.casefold()
    if any(word in lowered for word in ("needs attention", "target missing", "failed")):
        return _RED
    if any(word in lowered for word in ("not landed", "partly landed", "awaiting", "unknown")):
        return _YELLOW
    if any(word in lowered for word in ("enabled", "landed", "written")):
        return _GREEN
    return _DIM


def _paint_common(lines: list[str], enabled: bool) -> list[str]:
    """Add restrained color after layout so ANSI codes cannot affect wrapping."""

    if not enabled:
        return lines
    result: list[str] = []
    warning_section = False
    section_names = {"Sessions", "Git commits", "Revision evidence", "Warnings", "Automatic tracking", "Harness retention", "Tasks", "Workflow", "Owners", "More information"}
    field_labels = {
        "ID", "Repository", "Target", "Status", "Feature", "Session", "Changed", "Skipped",
        "Worktree", "Current worktree", "Worktrees", "Codex",
        "Enabled worktrees", "Claude Code", "Git post-commit hook", "Traces", "Pending", "Notice", "Enable",
        "Model / harness", "Served model", "Native session", "Label source", "Lines", "Cost",
        "Usage", "Cost match", "Telemetry collector", "Cost collection",
        "Claude cost", "Codex cost",
        "Models", "Sessions", "Reported cost",
        "Role", "Parent session", "Outcome", "Tokens", "Task type", "Pull request",
        "Branch", "Total cost", "Total tokens",
        "Command", "Evidence", "Unknown additions", "Original session", "Revised by",
        "Observed lifetime", "Retained / landed", "Recorded", "Last activity",
        "Workflow", "Agent", "Launch mode", "Commit", "File", "Prompt",
    }
    for line in lines:
        stripped = line.strip()
        if line.startswith(("Joyride · ", "Joyride demo · ")):
            result.append(_style(line, _BOLD_CYAN, True))
            warning_section = False
            continue
        if stripped in section_names:
            code = _BOLD if stripped != "Warnings" else _YELLOW + "\x1b[1m"
            result.append(_style(line, code, True))
            warning_section = stripped == "Warnings"
            continue
        if warning_section and stripped:
            result.append(_style(line, _YELLOW, True))
            continue
        field = re.match(r"^(\s*)([^:]{1,28}):(.*)$", line)
        if field is not None and field.group(2) in field_labels:
            indent, label, value = field.groups()
            if label in {
                "Status", "Codex", "Claude Code", "Git post-commit hook",
                "Telemetry collector", "Claude cost", "Codex cost", "Cost collection",
            }:
                value_code = _state_style(value)
            elif label == "Model / harness":
                value_code = ""
            elif label == "Retained / landed":
                value_code = _DIM if value.strip() in {"—", "unknown/unknown"} else ""
            elif label in {"Cost", "Reported cost"}:
                value_code = _DIM if "unknown" in value else ""
            else:
                value_code = ""
            painted_value = _style(value, value_code, bool(value_code))
            result.append(f"{indent}{_style(label + ':', _DIM, True)}{painted_value}")
            continue
        result.append(line)
    return result


def _head(text: str, limit: int) -> tuple[str, str]:
    if limit <= 0:
        return "", text
    used = 0
    split_at = 0
    for index, character in enumerate(text):
        char_width = 0 if unicodedata.combining(character) else (
            2 if unicodedata.east_asian_width(character) in {"W", "F"} else 1
        )
        if used + char_width > limit:
            break
        used += char_width
        split_at = index + 1
    return text[:split_at], text[split_at:]


def _clip(text: str, limit: int) -> str:
    text = safe_text(text)
    if _display_width(text) <= limit:
        return text
    if limit <= 3:
        return _head(text, limit)[0]
    return _head(text, limit - 3)[0] + "..."


def _pad(text: str, limit: int, *, right: bool = False) -> str:
    clipped = _clip(text, limit)
    padding = " " * max(0, limit - _display_width(clipped))
    return padding + clipped if right else clipped + padding


def _wrap(text: Any, limit: int) -> list[str]:
    clean = safe_text(text) or "unknown"
    clean = _clip(clean, max(80, limit * 4))
    if limit <= 1:
        return [_clip(clean, max(1, limit))]
    lines: list[str] = []
    current = ""
    for original_word in clean.split(" "):
        word = original_word
        while _display_width(word) > limit:
            if current:
                lines.append(current)
                current = ""
            chunk, remainder = _head(word, limit)
            if not chunk:
                chunk, remainder = "?", word[1:]
            lines.append(chunk)
            word = remainder
        if not word:
            continue
        candidate = word if not current else f"{current} {word}"
        if _display_width(candidate) <= limit:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines or ["unknown"]


def _append_wrapped(lines: list[str], value: Any, width: int, *, indent: str = "") -> None:
    available = max(1, width - _display_width(indent))
    lines.extend(indent + item for item in _wrap(value, available))


def _append_hanging(
    lines: list[str], value: Any, width: int, *, first: str, continuation: str
) -> None:
    available = max(1, width - max(_display_width(first), _display_width(continuation)))
    wrapped = _wrap(value, available)
    lines.append(first + wrapped[0])
    lines.extend(continuation + item for item in wrapped[1:])


def _append_field(lines: list[str], label: str, value: Any, width: int, *, indent: str = "") -> None:
    prefix = f"{indent}{label}: "
    available = width - _display_width(prefix)
    if available < 10:
        lines.append(_clip(prefix.rstrip(), width))
        _append_wrapped(lines, value, width, indent=indent + "  ")
        return
    wrapped = _wrap(value, available)
    lines.append(prefix + wrapped[0])
    continuation = " " * _display_width(prefix)
    lines.extend(continuation + item for item in wrapped[1:])


def _short(value: Any, length: int = 10) -> str:
    text = safe_text(value) or "unknown"
    return text if len(text) <= length else text[:length]


def _source(value: Any) -> str:
    return {
        "hook": "tool-reported",
        "session_setting": "session setting",
        "unknown": "unknown",
        "reported": "supplied",
    }.get(value, "unknown")


def _status(value: Any, *, target_exists: bool) -> str:
    if not target_exists:
        return "target missing"
    return {
        "landed": "landed",
        "mixed": "partly landed",
        "unlanded": "not landed",
        "captured": "awaiting commit",
    }.get(value, "unknown")


def _line_pair(feature: Mapping[str, Any], *, target_exists: bool) -> str:
    if not target_exists:
        return "unknown/unknown"
    landed = _integer(feature.get("landed_lines"))
    if landed == 0:
        return "—"
    pair = f"{_count(feature.get('retained_lines'))}/{_count(feature.get('landed_lines'))}"
    retention = _percent(feature.get("retention_pct"))
    return f"{pair}  {retention}" if retention != "unknown" else pair


def _label(value: Any) -> str:
    return value if isinstance(value, str) and value.strip() else "unknown"


def _session_model(session: Mapping[str, Any]) -> str:
    # The report already resolved the label: the hook model where the harness
    # reported one, else the dominant telemetry model. The auxiliary requests a
    # session also makes, such as a title request under another model, stay
    # in ``telemetry.models`` and never rename the session here.
    return _label(session.get("model"))


def _elapsed(value: Any) -> str | None:
    """Return a wall-clock duration from milliseconds, or None when unknown."""

    milliseconds = _number(value)
    if milliseconds is None:
        return None
    seconds = milliseconds / 1000
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, whole_seconds = divmod(round(seconds), 60)
    if minutes < 60:
        return f"{minutes}m {whole_seconds}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes}m"


def _agent_model(agent: Mapping[str, Any]) -> str:
    return safe_text(_label(agent.get("model")))


def _agent_type(agent: Mapping[str, Any]) -> str | None:
    value = safe_text(agent.get("agent_type"))
    return value or None


def _tally(values: list[Any]) -> list[tuple[Any, int]]:
    """Return each distinct value with its count, most frequent first."""

    counts: dict[Any, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))


def _agent_noun(agent: Mapping[str, Any]) -> str:
    """Return what one agent is: a subagent, an orchestrator, or an agent."""

    if agent.get("parent_session_id"):
        return "subagent"
    return "orchestrator" if _items(agent.get("children")) else "agent"


def _agent_phrases(agents: list[Mapping[str, Any]]) -> list[str]:
    """Return one phrase for each model, with the noun each agent earns.

    The noun is taken per agent, as the tree of ``show`` takes it, so one root
    that launched subagents never makes an orchestrator of a root beside it
    that launched none. Each phrase names the agent types of its own group: a
    second model's types belong to that model, not to the phrase printed last.
    """

    grouped: dict[tuple[str, str], list[str]] = {}
    for agent in agents:
        key = (_agent_model(agent), _agent_noun(agent))
        listed = grouped.setdefault(key, [])
        agent_type = _agent_type(agent)
        if agent_type:
            listed.append(agent_type)
    phrases = []
    for key, count in _tally([(_agent_model(agent), _agent_noun(agent)) for agent in agents]):
        model, noun = key
        phrase = f"{count:,} {model} {noun}" if count == 1 else f"{count:,} {model} {noun}s"
        types = _tally(grouped[key])
        if types:
            phrase += " (" + ", ".join(f"{number:,} {name}" for name, number in types) + ")"
        phrases.append(phrase)
    return phrases


def _workflow_line(feature: Mapping[str, Any]) -> str | None:
    """Return one line that says how a task was worked, or None when unknown.

    Every part reports evidence. A part the ledger never recorded is left out
    rather than printed as a zero.
    """

    profile = _mapping(feature.get("workflow"))
    agents = iter_agents(profile)
    if not agents:
        return None
    roots = [agent for agent in agents if not agent.get("parent_session_id")]
    subagents = [agent for agent in agents if agent.get("parent_session_id")]
    parts = _agent_phrases(roots) + _agent_phrases(subagents)
    totals = _mapping(profile.get("totals"))
    prompts = _integer(totals.get("prompts"))
    if prompts:
        parts.append(_counted(prompts, "prompt"))
    reads = _integer(_mapping(totals.get("tool_calls")).get("read"))
    if reads:
        parts.append(_counted(reads, "read"))
    modes = {value for agent in roots if (value := safe_text(agent.get("permission_mode")))}
    if len(modes) == 1:
        parts.append(f"{modes.pop()} mode")
    return " · ".join(parts) if parts else None


def _unique_sessions(features: list[Any]) -> list[Mapping[str, Any]]:
    sessions: list[Mapping[str, Any]] = []
    seen: set[str] = set()
    for raw_feature in features:
        for raw_session in _items(_mapping(raw_feature).get("sessions")):
            session = _mapping(raw_session)
            session_id = session.get("id")
            if isinstance(session_id, str) and session_id:
                if session_id in seen:
                    continue
                seen.add(session_id)
            sessions.append(session)
    return sessions


def _summarise_sessions(sessions: list[Mapping[str, Any]]) -> dict[str, Any]:
    def total(field: str) -> int | None:
        values = [_integer(session.get(field)) for session in sessions]
        if any(value is None or value < 0 for value in values):
            return None
        return sum(values)

    landed = total("landed_lines")
    retained = total("retained_lines")
    attributed = total("attributed_lines")
    reported_costs: list[float] = []
    estimated_costs: list[float] = []
    credit_costs: list[float] = []
    equivalent_costs: list[float] = []
    covered = 0
    cost_sessions = 0
    for session in sessions:
        if session.get("cost_in_total") is False:
            continue
        cost_sessions += 1
        telemetry = _mapping(session.get("telemetry"))
        reported = _number(session.get("cost_usd"))
        estimated = _number(telemetry.get("estimated_cost_usd"))
        credits = _number(telemetry.get("codex_credits"))
        equivalent = _number(telemetry.get("codex_api_equivalent_usd"))
        if reported is not None and reported >= 0:
            reported_costs.append(reported)
        else:
            if estimated is not None and estimated >= 0:
                estimated_costs.append(estimated)
            if credits is not None and credits >= 0:
                credit_costs.append(credits)
            if equivalent is not None and equivalent >= 0:
                equivalent_costs.append(equivalent)
        if reported is not None or (
            telemetry.get("cost_complete") is not False
            and (estimated is not None or credits is not None)
        ):
            covered += 1
    tokens = [
        _integer(session.get("token_count"))
        for session in sessions
        if session.get("tokens_in_total") is not False
    ]
    known_tokens = [value for value in tokens if value is not None and value >= 0]
    if attributed is None or landed is None:
        status = "unknown"
    elif attributed == 0:
        status = "captured"
    elif landed == 0:
        status = "unlanded"
    elif landed == attributed:
        status = "landed"
    else:
        status = "mixed"
    return {
        "models": sorted({_session_model(session) for session in sessions}),
        "session_count": len(sessions),
        "attributed_lines": attributed,
        "landed_lines": landed,
        "retained_lines": retained,
        "retention_pct": round(100 * retained / landed, 1)
        if landed and retained is not None else None,
        "reported_cost_usd": (
            round(math.fsum(reported_costs), 10) if reported_costs else None
        ),
        "estimated_cost_usd": (
            round(math.fsum(estimated_costs), 10) if estimated_costs else None
        ),
        "codex_credits": (
            round(math.fsum(credit_costs), 10) if credit_costs else None
        ),
        "codex_api_equivalent_usd": (
            round(math.fsum(equivalent_costs), 10) if equivalent_costs else None
        ),
        "cost_complete": bool(cost_sessions) and covered == cost_sessions,
        "total_tokens": sum(known_tokens) if known_tokens else None,
        "tokens_complete": bool(sessions) and len(known_tokens) == len(tokens),
        "status": status,
    }


def _harness_rows(features: list[Any]) -> list[dict[str, Any]]:
    by_harness: dict[str, list[Mapping[str, Any]]] = {}
    for session in _unique_sessions(features):
        by_harness.setdefault(_label(session.get("harness")), []).append(session)
    return [
        {"harness": harness, **_summarise_sessions(sessions)}
        for harness, sessions in sorted(by_harness.items())
    ]


def _repository_lines(data: Mapping[str, Any], width: int) -> tuple[list[str], bool]:
    repository = _mapping(data.get("repository"))
    name = safe_text(repository.get("name")) or "unknown repository"
    path = safe_text(repository.get("path"))
    target = safe_text(repository.get("target_ref")) or "unknown"
    target_exists = repository.get("target_exists") is True
    target_commit = _short(repository.get("target_commit")) if target_exists else "not found"
    title = (
        f"Joyride demo · {name}"
        if repository.get("example_data") is True
        else f"Joyride · {name}"
    )
    lines: list[str] = []
    _append_wrapped(lines, title, width)
    if path:
        _append_field(lines, "Repository", path, width)
    _append_field(lines, "Target", f"{target} @ {target_commit}", width)
    if not target_exists:
        _append_wrapped(
            lines,
            "Target ref was not found. Landed and retained results are unavailable.",
            width,
        )
    return lines, target_exists


def _feature_rows(
    features: list[Any], *, target_exists: bool
) -> tuple[list[tuple[str, str, str, str, str]], dict[int, str]]:
    """Return the task rows and the workflow line that follows each task."""

    rows: list[tuple[str, str, str, str, str]] = []
    notes: dict[int, str] = {}
    for raw_feature in features[:_MAX_FEATURES]:
        feature = _mapping(raw_feature)
        workflows: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
        for session in _unique_sessions([feature]):
            key = (_session_model(session), _label(session.get("harness")))
            workflows.setdefault(key, []).append(session)
        if not workflows:
            workflows[("unknown", "unknown")] = []
        for (model, harness), sessions in sorted(workflows.items()):
            summary = _summarise_sessions(sessions) if sessions else feature
            rows.append(
                (
                    safe_text(feature.get("name")) or "unnamed",
                    f"{safe_text(model)} / {safe_text(harness)}",
                    _line_pair(summary, target_exists=target_exists),
                    _feature_cost(summary, compact=True),
                    _status(summary.get("status"), target_exists=target_exists),
                )
            )
        workflow = _workflow_line(feature)
        if workflow is not None:
            notes[len(rows) - 1] = workflow
    return rows, notes


def _render_feature_table(
    rows: list[tuple[str, str, str, str, str]], width: int, *, color: bool,
    harnesses: bool = False, notes: dict[int, str] | None = None,
) -> list[str]:
    notes = notes or {}
    if width < 84:
        lines: list[str] = []
        for index, (feature, model, line_pair, cost, status) in enumerate(rows):
            if lines:
                lines.append("")
            _append_wrapped(lines, feature, width)
            _append_field(lines, "Models" if harnesses else "Model / harness", model, width, indent="  ")
            _append_field(lines, "Retained / landed", line_pair, width, indent="  ")
            _append_field(lines, "Cost", cost, width, indent="  ")
            _append_field(lines, "Sessions" if harnesses else "Status", status, width, indent="  ")
            if index in notes:
                _append_field(lines, "Workflow", notes[index], width, indent="  ")
        return lines

    gap = "  "
    line_width, cost_width, status_width = 15, 14, 14
    remaining = width - line_width - cost_width - status_width - 4 * len(gap)
    model_width = min(30, max(20, remaining // 2))
    feature_width = max(12, remaining - model_width)
    widths = (feature_width, model_width, line_width, cost_width, status_width)
    headers = ("HARNESS", "MODELS", "RETAINED/LANDED", "COST", "SESSIONS") if harnesses else (
        "TASK", "MODEL / HARNESS", "RETAINED/LANDED", "COST", "STATUS"
    )

    def format_row(row: tuple[str, str, str, str, str], *, header: bool = False) -> str:
        cells = []
        for index, (value, cell_width) in enumerate(zip(row, widths)):
            cell = _pad(value, cell_width, right=not header and index in {2, 3})
            if color:
                if header:
                    cell = _style(cell, _BOLD, True)
                elif index == 0:
                    cell = _style(cell, _CYAN, True)
                elif index == 2:
                    cell = _style(cell, _DIM, True) if value == "—" else cell
                elif index == 3:
                    cell = _style(cell, _DIM, True) if value == "unknown" else cell
                elif index == 4:
                    cell = _style(cell, _state_style(value), True)
            cells.append(cell)
        return gap.join(cells)

    divider = "-" * width
    def wrap_row(row: tuple[str, str, str, str, str], *, header: bool = False) -> list[str]:
        columns = [_wrap(value, cell_width) for value, cell_width in zip(row, widths)]
        return [
            format_row(tuple(column[index] if index < len(column) else "" for column in columns), header=header)
            for index in range(max(map(len, columns)))
        ]

    lines = [*wrap_row(headers, header=True), _style(divider, _DIM, color)]
    for index, row in enumerate(rows):
        lines.extend(wrap_row(row))
        if index in notes:
            workflow: list[str] = []
            _append_wrapped(workflow, notes[index], width, indent="  ")
            lines.extend(_style(line, _DIM, color) for line in workflow)
    return lines


def _render_harnesses(features: list[Any], width: int, *, target_exists: bool, color: bool) -> list[str]:
    summaries = _harness_rows(features)
    if not summaries:
        return []
    rows = [
        (
            safe_text(item["harness"]),
            ", ".join(safe_text(model) for model in item["models"]),
            _line_pair(item, target_exists=target_exists),
            _feature_cost(item, compact=True),
            _count(item["session_count"]),
        )
        for item in summaries
    ]
    return ["Harness retention", *_render_feature_table(rows, width, color=color, harnesses=True)]


def _append_command(lines: list[str], args: list[str], width: int) -> None:
    if any(
        unicodedata.category(character).startswith("C")
        or unicodedata.category(character) in {"Zl", "Zp"}
        for argument in args for character in argument
    ):
        _append_wrapped(lines, "For names with control characters, use the command help:", width)
        _append_wrapped(lines, "joyride show --help", width)
        return
    # Break only between shell arguments (or concatenate quoted fragments of a
    # long argument). Backslash continuations keep even narrow output runnable.
    current = ""
    for argument in args:
        remaining = argument
        fragments: list[str] = []
        while _display_width(shlex.quote(remaining)) > width - 2:
            limit = width - 4
            chunk, tail = _head(remaining, limit)
            while _display_width(shlex.quote(chunk)) > width - 2:
                limit -= 1
                chunk, tail = _head(remaining, limit)
            fragments.append(shlex.quote(chunk))
            remaining = tail
        fragments.append(shlex.quote(remaining))
        if len(fragments) > 1:
            if current:
                lines.append(current + " \\")
                current = ""
            lines.extend(fragment + "\\" for fragment in fragments[:-1])
            current = fragments[-1]
        else:
            token = fragments[0]
            if current and _display_width(current + " " + token) > width - 2:
                lines.append(current + " \\")
                current = ""
            current = current + " " + token if current else token
    if current:
        lines.append(current)


def _more_information(report: Mapping[str, Any], width: int) -> list[str]:
    repository = _mapping(report.get("repository"))
    features = _items(report.get("features"))
    base = ["joyride"]
    path = repository.get("path")
    if isinstance(path, str) and path:
        base.extend(["--repo", path])
    target = repository.get("target_ref")
    scope = ["--target", target] if isinstance(target, str) and target else []
    lines = ["", "More information"]
    if features:
        feature = _mapping(features[0])
        query = _label(feature.get("id") or feature.get("name"))
        _append_wrapped(lines, "Sessions, commits, and replacement history:", width)
        show = ["show"] + scope + ["--", query] if query.startswith("-") else ["show", query] + scope
        _append_command(lines, base + show, width)
        if len(features) > 1:
            _append_wrapped(lines, "Replace the task name to inspect another task.", width)
    _append_wrapped(lines, "Full report as JSON:", width)
    _append_command(lines, base + ["report", "--json"] + scope, width)
    return lines


def _warning_lines(*values: Any, width: int) -> list[str]:
    warnings: list[str] = []
    seen: set[str] = set()
    for value in values:
        for warning in _items(value):
            text = safe_text(warning)
            if text and text not in seen:
                seen.add(text)
                warnings.append(text)
    binary_commits = [
        item for item in warnings
        if re.fullmatch(r"Binary changes in commit [0-9a-fA-F]+ have no line count\.", item)
    ]
    binary_files = [item for item in warnings if item.startswith("Skipped binary target file ")]
    warnings = [item for item in warnings if item not in set(binary_commits + binary_files)]
    if binary_commits:
        verb = "contains" if len(binary_commits) == 1 else "contain"
        warnings.append(
            f"{_counted(len(binary_commits), 'commit')} {verb} binary changes without line counts."
        )
    if binary_files:
        verb = "was" if len(binary_files) == 1 else "were"
        warnings.append(
            f"{_counted(len(binary_files), 'binary target file')} {verb} excluded from line analysis."
        )
    if not warnings:
        return []
    lines = ["", "Warnings"]
    for warning in warnings[:10]:
        _append_hanging(lines, warning, width, first="- ", continuation="  ")
    if len(warnings) > 10:
        lines.append(f"- ... {len(warnings) - 10:,} more warnings")
    return lines


def render_report(
    data: Mapping[str, Any], *, width: int | None = None, color: bool | None = None
) -> str:
    """Render a compact repository attribution summary."""

    report = _mapping(data)
    output_width = _width(width)
    use_color = _color_enabled(color)
    lines, target_exists = _repository_lines(report, output_width)
    features = _items(report.get("features"))
    lines.append("")

    if not features:
        lines.append("No attributed tasks yet.")
        _append_wrapped(lines, "Check setup with: joyride status", output_width)
        _append_wrapped(
            lines,
            "After tracking is enabled, use Codex or Claude Code and commit normally.",
            output_width,
        )
        _append_wrapped(lines, "Available commands: joyride help", output_width)
    else:
        lines.extend(_render_harnesses(features, output_width, target_exists=target_exists, color=use_color))
        lines.extend(["", "Tasks"])
        rows, workflow_notes = _feature_rows(features, target_exists=target_exists)
        lines.extend(
            _render_feature_table(
                rows,
                output_width,
                color=use_color,
                notes=workflow_notes,
            )
        )
        if len(features) > _MAX_FEATURES:
            _append_wrapped(lines, f"... {len(features) - _MAX_FEATURES:,} more tasks not shown", output_width)
        visible_features = [_mapping(feature) for feature in features[:_MAX_FEATURES]]
        if any(feature.get("cost_complete") is False and _feature_cost(feature, compact=True) != "unknown" for feature in visible_features):
            _append_wrapped(lines, "* Cost coverage is partial.", output_width)
        if any(feature.get("estimated_cost_usd") is not None for feature in visible_features):
            _append_wrapped(
                lines,
                "est. = provider or API price-table estimate, not an authoritative bill.",
                output_width,
            )
        if any(feature.get("codex_credits") is not None for feature in visible_features):
            _append_wrapped(
                lines,
                "cr = estimated credits. ≈$ is standard API pricing, not subscription spend.",
                output_width,
            )
        if any(
            _feature_cost(_mapping(item), compact=True) != "unknown"
            and not item.get("cost_complete", False)
            for item in _harness_rows(features)
        ):
            _append_wrapped(lines, "* Harness cost is incomplete.", output_width)

    unattributed = _mapping(report.get("unattributed"))
    unattributed_lines = _integer(unattributed.get("added_lines"))
    if unattributed_lines is not None and unattributed_lines > 0:
        lines.append("")
        _append_wrapped(
            lines,
            f"Other target history: {_counted(unattributed_lines, 'unattributed line')} "
            f"across {_counted(unattributed.get('commit_count'), 'commit')}",
            output_width,
        )
    lines.extend(_warning_lines(report.get("warnings"), width=output_width))
    return "\n".join(_paint_common(lines, use_color)).rstrip("\n") + "\n"


def _select_feature(data: Mapping[str, Any], query: str) -> Mapping[str, Any]:
    if not isinstance(query, str) or not query:
        raise ValueError("Task query cannot be empty.")
    features = [_mapping(feature) for feature in _items(data.get("features"))]
    for feature in features:
        if feature.get("id") == query or feature.get("name") == query:
            return feature
    folded = query.casefold()
    matches = [feature for feature in features if isinstance(feature.get("name"), str) and feature["name"].casefold() == folded]
    if not matches:
        prefix = folded.removesuffix("...").removesuffix("…").rstrip()
        if prefix:
            matches = [
                feature
                for feature in features
                if any(
                    isinstance(feature.get(field), str)
                    and feature[field].casefold().startswith(prefix)
                    for field in ("name", "id")
                )
            ]
    if not matches:
        raise ValueError(f'No task matches "{_clip(safe_text(query), 80)}".')
    if len(matches) > 1:
        candidates = ", ".join(
            f"{_clip(safe_text(feature.get('name')) or 'unnamed', 60)} "
            f"[{_clip(safe_text(feature.get('id')) or 'no id', 40)}]"
            for feature in matches[:5]
        )
        suffix = ", ..." if len(matches) > 5 else ""
        raise ValueError(
            f'Ambiguous task "{_clip(safe_text(query), 80)}": {candidates}{suffix}. Use an exact name or ID.'
        )
    return matches[0]


def _session_index(feature: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for raw_session in _items(feature.get("sessions")):
        session = _mapping(raw_session)
        session_id = session.get("id")
        if isinstance(session_id, str):
            result[session_id] = session
    return result


def _session_reference(value: Any, sessions: Mapping[str, Mapping[str, Any]]) -> str:
    session_id = value if isinstance(value, str) else ""
    safe_id = safe_text(session_id) or "unknown"
    display_id = _short(safe_id, 12)
    session = sessions.get(session_id)
    if session is None:
        return display_id
    model = safe_text(session.get("model")) or "unknown"
    harness = safe_text(session.get("harness")) or "unknown"
    return f"{display_id} ({model} / {harness})"


def _render_sessions(feature: Mapping[str, Any], width: int, *, target_exists: bool) -> list[str]:
    sessions = _items(feature.get("sessions"))
    lines = ["", "Sessions"]
    if not sessions:
        lines.append("  none")
        return lines
    for raw_session in sessions[:_MAX_SESSIONS]:
        session = _mapping(raw_session)
        lines.append("")
        summary = safe_text(session.get("summary"))
        role = safe_text(session.get("role")) or "implementation"
        heading = summary or f"{role.capitalize()} session"
        _append_wrapped(lines, heading, width, indent="  ")
        _append_field(
            lines,
            "Session",
            safe_text(session.get("id")) or "unknown session",
            width,
            indent="    ",
        )
        _append_field(lines, "Role", role, width, indent="    ")
        _append_field(
            lines,
            "Model / harness",
            f"{safe_text(session.get('model')) or 'unknown'} / {safe_text(session.get('harness')) or 'unknown'}",
            width,
            indent="    ",
        )
        telemetry = _mapping(session.get("telemetry"))
        served_models = [safe_text(item) for item in _items(telemetry.get("models"))]
        served_models = [item for item in served_models if item]
        if served_models and served_models != [safe_text(session.get("model"))]:
            _append_field(
                lines,
                "Served model",
                ", ".join(served_models),
                width,
                indent="    ",
            )
        native_id = safe_text(session.get("native_session_id"))
        if native_id:
            _append_field(lines, "Native session", native_id, width, indent="    ")
        worktree = _mapping(session.get("worktree"))
        worktree_path = safe_text(worktree.get("path"))
        worktree_branch = safe_text(worktree.get("branch"))
        worktree_id = safe_text(worktree.get("id"))
        if worktree_path or worktree_branch or worktree_id:
            worktree_parts = [
                value for value in (worktree_branch, worktree_path or worktree_id) if value
            ]
            if worktree.get("current") is True:
                worktree_parts.append("current")
            _append_field(
                lines,
                "Worktree",
                " · ".join(worktree_parts),
                width,
                indent="    ",
            )
        _append_field(lines, "Label source", _source(session.get("label_source")), width, indent="    ")
        landed = _count(session.get("landed_lines")) if target_exists else "unknown"
        retained = _count(session.get("retained_lines")) if target_exists else "unknown"
        _append_field(
            lines,
            "Lines",
            (
                f"{_count(session.get('attributed_lines'))} committed, "
                f"{landed} landed, {retained} retained"
            ),
            width,
            indent="    ",
        )
        _append_field(lines, "Cost", _session_cost(session), width, indent="    ")
        _append_field(lines, "Tokens", _count(session.get("token_count")), width, indent="    ")
        parent = safe_text(session.get("parent_session_id"))
        if parent:
            _append_field(lines, "Parent session", parent, width, indent="    ")
        if telemetry:
            input_tokens = _number(telemetry.get("input_tokens"))
            cached_tokens = _number(telemetry.get("cached_input_tokens"))
            output_tokens = _number(telemetry.get("output_tokens"))
            if any(value is not None for value in (input_tokens, cached_tokens, output_tokens)):
                parts = []
                if input_tokens is not None:
                    parts.append(f"{round(input_tokens):,} input")
                if cached_tokens is not None:
                    parts.append(f"{round(cached_tokens):,} cached")
                if output_tokens is not None:
                    parts.append(f"{round(output_tokens):,} output")
                _append_field(lines, "Usage", " · ".join(parts), width, indent="    ")
            allocation = {
                "agent-routed": "the agent that made the request",
                "tool-linked": "exact prompt + tool",
                "turn-linked": "exact Codex turn",
                "session-allocated": "native session allocation",
                "mixed": "mixed exact and session allocation",
            }.get(telemetry.get("allocation"))
            if allocation:
                _append_field(lines, "Cost match", allocation, width, indent="    ")
        outcome = safe_text(session.get("outcome"))
        exit_code = _integer(session.get("exit_code"))
        if outcome:
            _append_field(lines, "Outcome", outcome, width, indent="    ")
        elif exit_code is not None:
            command_outcome = "completed" if exit_code == 0 else f"failed with exit {exit_code}"
            _append_field(lines, "Command", command_outcome, width, indent="    ")
    if len(sessions) > _MAX_SESSIONS:
        _append_wrapped(lines, f"... {len(sessions) - _MAX_SESSIONS:,} more sessions not shown", width, indent="  ")
    return lines


def _agent_label(agent: Mapping[str, Any]) -> str:
    kind = _agent_noun(agent)
    agent_type = _agent_type(agent)
    return f"{agent_type} {kind}" if agent_type else kind


def _agent_text(agent: Mapping[str, Any]) -> str:
    """Return one agent with its tool count, its duration, and its cost."""

    parts = [
        _agent_model(agent),
        safe_text(_label(agent.get("harness"))),
        _agent_label(agent),
    ]
    calls = _mapping(agent.get("tool_calls"))
    if calls:
        total = sum(_integer(value) or 0 for value in calls.values())
        parts.append(_counted(total, "tool call"))
    elapsed = _elapsed(agent.get("duration_ms"))
    if elapsed is not None:
        parts.append(elapsed)
    cost = _number(agent.get("cost_usd"))
    if cost is not None:
        parts.append(f"${cost:,.2f}")
    return " · ".join(parts)


def _agent_rows(profile: Mapping[str, Any]) -> list[tuple[int, Mapping[str, Any]]]:
    """Return every agent with its depth, each parent before its children."""

    rows: list[tuple[int, Mapping[str, Any]]] = []
    pending = [(agent, 0) for agent in reversed(_items(profile.get("agents")))]
    while pending and len(rows) <= _MAX_SESSIONS:
        raw_agent, depth = pending.pop()
        agent = _mapping(raw_agent)
        rows.append((depth, agent))
        pending.extend(
            (child, depth + 1) for child in reversed(_items(agent.get("children")))
        )
    return rows


def _render_workflow(feature: Mapping[str, Any], width: int) -> list[str]:
    """Render the agent tree, then what those agents loaded."""

    profile = _mapping(feature.get("workflow"))
    rows = _agent_rows(profile)
    if not rows:
        return []
    lines = ["", "Workflow"]
    summary = _workflow_line(feature)
    if summary is not None:
        _append_wrapped(lines, summary, width, indent="  ")
    lines.append("")
    for depth, agent in rows[:_MAX_SESSIONS]:
        # A deep tree keeps its shape without running off the right margin.
        indent = "  " * (1 + min(depth, _MAX_AGENT_DEPTH))
        _append_hanging(
            lines, _agent_text(agent), width, first=indent, continuation=indent + "  "
        )
    if len(rows) > _MAX_SESSIONS:
        _append_wrapped(lines, "... more agents not shown", width, indent="  ")

    files = [_mapping(item) for item in _items(profile.get("instruction_files"))]
    totals = _mapping(_mapping(profile.get("totals")).get("tool_calls"))
    loaded = {name: _integer(totals.get(name)) for name, _, _ in _LOADED_TOOL_CLASSES}
    counted = [
        _counted(loaded[name], singular, plural)
        for name, singular, plural in _LOADED_TOOL_CLASSES
        if loaded[name] is not None
    ]
    if not files and not any(loaded.values()):
        return lines
    lines.extend(["", "  Loaded"])
    for item in files[:_MAX_INSTRUCTION_FILES]:
        parts = [safe_text(item.get("path")) or "unknown file"]
        memory_type = safe_text(item.get("memory_type"))
        if memory_type:
            parts.append(memory_type)
        parts.append(_counted(item.get("agent_count"), "agent"))
        _append_wrapped(lines, " · ".join(parts), width, indent="    ")
    if len(files) > _MAX_INSTRUCTION_FILES:
        remaining = len(files) - _MAX_INSTRUCTION_FILES
        _append_wrapped(
            lines, f"... {remaining:,} more instruction files not shown", width, indent="    "
        )
    if counted:
        _append_wrapped(lines, " · ".join(counted), width, indent="    ")
    # A read proves that content entered the context window. It does not prove
    # that the model relied on it.
    _append_wrapped(lines, "Loaded means it entered the context window.", width, indent="  ")
    return lines


def _range_text(raw_range: Any, sessions: Mapping[str, Mapping[str, Any]]) -> str:
    line_range = _mapping(raw_range)
    start, end = _integer(line_range.get("start")), _integer(line_range.get("end"))
    if start is None or end is None:
        location = "lines unknown"
    elif start == end:
        location = f"L{start:,}"
    else:
        location = f"L{start:,}-{end:,}"
    return f"{location} <- {_session_reference(line_range.get('session_id'), sessions)}"


def _render_commits(feature: Mapping[str, Any], width: int, *, target_exists: bool) -> list[str]:
    commits = _items(feature.get("commits"))
    sessions = _session_index(feature)
    lines = ["", "Git commits"]
    if not commits:
        lines.append("  none")
        return lines
    for raw_commit in commits[:_MAX_COMMITS]:
        commit = _mapping(raw_commit)
        sha = safe_text(commit.get("short_sha")) or _short(commit.get("sha"))
        subject = safe_text(commit.get("subject")) or "untitled commit"
        lines.append("")
        _append_hanging(
            lines,
            subject,
            width,
            first=f"  {sha}  ",
            continuation=" " * (_display_width(sha) + 4),
        )
        commit_status = "target missing" if not target_exists else (
            "on target" if commit.get("landed") is True else "not on target"
        )
        _append_field(
            lines,
            "Evidence",
            (
                f"{commit_status}; {_count(commit.get('attributed_lines'))} attributed, "
                f"{_count(commit.get('retained_lines')) if target_exists else 'unknown'} retained; "
                f"{_date(commit.get('committed_at'))}"
            ),
            width,
            indent="    ",
        )
        unknown_added = _integer(commit.get("unknown_added_lines"))
        if unknown_added is not None and unknown_added > 0:
            _append_field(lines, "Unknown additions", f"{unknown_added:,} lines", width, indent="    ")
        files = _items(commit.get("files"))
        for raw_file in files[:_MAX_FILES_PER_COMMIT]:
            file_entry = _mapping(raw_file)
            path = safe_text(file_entry.get("path")) or "unknown path"
            ranges = _items(file_entry.get("ranges"))
            summary = ", ".join(_range_text(item, sessions) for item in ranges[:_MAX_RANGES_PER_FILE])
            if len(ranges) > _MAX_RANGES_PER_FILE:
                summary += f", +{len(ranges) - _MAX_RANGES_PER_FILE} ranges"
            _append_field(lines, path, summary or "no attributed ranges", width, indent="    ")
        if len(files) > _MAX_FILES_PER_COMMIT:
            _append_wrapped(lines, f"... {len(files) - _MAX_FILES_PER_COMMIT:,} more files not shown", width, indent="    ")
    if len(commits) > _MAX_COMMITS:
        _append_wrapped(lines, f"... {len(commits) - _MAX_COMMITS:,} more commits not shown", width, indent="  ")
    return lines


def _render_revisions(feature: Mapping[str, Any], width: int) -> list[str]:
    revisions = _items(feature.get("revisions"))
    sessions = _session_index(feature)
    lines = ["", "Revision evidence"]
    if not revisions:
        lines.append("  none")
        return lines
    for raw_revision in revisions[:_MAX_REVISIONS]:
        revision = _mapping(raw_revision)
        start, end = _integer(revision.get("from_start")), _integer(revision.get("from_end"))
        if start is None or end is None:
            line_range = "lines unknown"
        elif start == end:
            line_range = f"L{start:,}"
        else:
            line_range = f"L{start:,}-{end:,}"
        heading = (
            f"{_short(revision.get('from_commit'))} -> {_short(revision.get('commit'))}  "
            f"{safe_text(revision.get('from_path')) or 'unknown path'}:{line_range}"
        )
        lines.append("")
        _append_wrapped(lines, heading, width, indent="  ")
        _append_field(
            lines,
            "Original session",
            _session_reference(revision.get("from_session_id"), sessions),
            width,
            indent="    ",
        )
        _append_field(
            lines,
            "Revised by",
            _session_reference(revision.get("to_session_id"), sessions),
            width,
            indent="    ",
        )
        _append_field(
            lines,
            "Observed lifetime",
            _duration(revision.get("observed_lifetime_hours")),
            width,
            indent="    ",
        )
    if len(revisions) > _MAX_REVISIONS:
        _append_wrapped(lines, f"... {len(revisions) - _MAX_REVISIONS:,} more revisions not shown", width, indent="  ")
    return lines


def feature_detail(data: Mapping[str, Any], query: str) -> Mapping[str, Any]:
    """Return the report's detail dictionary for one task.

    ``show`` prints this task, and ``show --json`` prints the same dictionary
    without rendering it, so both select a task by the same rules.
    """

    return _select_feature(_mapping(data), query)


def render_feature(
    data: Mapping[str, Any], query: str, *, width: int | None = None, color: bool | None = None
) -> str:
    """Render one selected attribution group."""

    report = _mapping(data)
    output_width = _width(width)
    use_color = _color_enabled(color)
    feature = _select_feature(report, query)
    repository = _mapping(report.get("repository"))
    target_exists = repository.get("target_exists") is True
    name = safe_text(feature.get("name")) or "unnamed"
    feature_id = safe_text(feature.get("id")) or "unknown"
    lines: list[str] = []
    _append_wrapped(lines, name, output_width)
    if feature_id != name:
        _append_field(lines, "ID", feature_id, output_width)
    _append_field(lines, "Repository", safe_text(repository.get("name")) or "unknown", output_width)
    target = safe_text(repository.get("target_ref")) or "unknown"
    _append_field(lines, "Target", f"{target} ({'available' if target_exists else 'not found'})", output_width)
    _append_field(lines, "Status", _status(feature.get("status"), target_exists=target_exists), output_width)
    kind = safe_text(feature.get("kind"))
    if kind:
        _append_field(lines, "Task type", kind.replace("_", " "), output_width)
    pr_ref = safe_text(feature.get("pr_ref"))
    if pr_ref:
        _append_field(lines, "Pull request", pr_ref, output_width)
    branch = safe_text(feature.get("branch"))
    if branch:
        _append_field(lines, "Branch", branch, output_width)
    line_summary = _line_pair(feature, target_exists=target_exists)
    _append_field(lines, "Retained / landed", line_summary, output_width)
    _append_field(
        lines,
        "Recorded",
        f"{_counted(feature.get('session_count'), 'session')}, "
        f"{_counted(feature.get('commit_count'), 'commit')}",
        output_width,
    )
    _append_field(
        lines,
        "Total cost",
        _feature_cost(feature, compact=False),
        output_width,
    )
    token_text = _count(feature.get("total_tokens"))
    if feature.get("tokens_complete") is False and token_text != "unknown":
        token_text += " (incomplete)"
    _append_field(lines, "Total tokens", token_text, output_width)
    _append_field(lines, "Last activity", _date(feature.get("last_activity_at")), output_width)
    if not target_exists:
        _append_wrapped(lines, "Target ref is missing; landed and retention evidence is unavailable.", output_width)

    lines.append("")
    lines.extend(_render_harnesses([feature], output_width, target_exists=target_exists, color=use_color))
    lines.extend(_render_workflow(feature, output_width))
    lines.extend(_render_sessions(feature, output_width, target_exists=target_exists))
    lines.extend(_render_commits(feature, output_width, target_exists=target_exists))
    lines.extend(_render_revisions(feature, output_width))
    lines.append("")
    _append_wrapped(
        lines,
        "This task includes every linked session, including work without surviving code.",
        output_width,
    )
    lines.extend(_warning_lines(report.get("warnings"), width=output_width))
    painted = _paint_common(lines, use_color)
    if painted:
        painted[0] = _style(painted[0], _BOLD_CYAN, use_color)
    return "\n".join(painted).rstrip() + "\n"


# What one loaded item is called, by the tool class or the context load kind
# that recorded it.
_CONTEXT_KIND_NAMES = {
    "read": "read",
    "search": "search",
    "web": "web fetch",
    "skill": "skill",
    "mcp": "MCP call",
    "instruction_file": "instruction file",
    "subagent_result": "subagent result",
}
# How the loaded list of one owner was cut, and what that says about it.
_CONTEXT_CUT_TEXT = {
    "edit": "Loaded before the edit that wrote these lines",
    "last_edit": (
        "Loaded before this session's last edit of the file, because no single "
        "edit matched these lines"
    ),
    "commit": (
        "Loaded before this commit, because no recorded edit of this file "
        "matched these lines"
    ),
}


def _line_ranges(numbers: list[Any]) -> str:
    """Return one-based line numbers as compact ranges, for example L4-7, L9."""

    ordered = sorted({value for number in numbers if (value := _integer(number))})
    spans: list[list[int]] = []
    for number in ordered:
        if spans and number == spans[-1][1] + 1:
            spans[-1][1] = number
        else:
            spans.append([number, number])
    text = ", ".join(
        f"L{start:,}" if start == end else f"L{start:,}-{end:,}"
        for start, end in spans[:_MAX_RANGES_PER_FILE]
    )
    if len(spans) > _MAX_RANGES_PER_FILE:
        text += ", ..."
    return text or "unknown"


def _context_text(item: Mapping[str, Any]) -> str:
    """Return one loaded item, by what it was rather than by what it said."""

    kind = safe_text(item.get("kind"))
    parts = [_CONTEXT_KIND_NAMES.get(kind, kind or "unknown")]
    locator = safe_text(item.get("locator"))
    if locator:
        parts.insert(0, locator)
    memory_type = safe_text(item.get("memory_type"))
    if memory_type:
        parts.append(memory_type)
    related = safe_text(item.get("related_session_id"))
    if related:
        parts.append(f"from {_short(related, 12)}")
    size = _integer(item.get("size_bytes"))
    if size is not None:
        parts.append(_counted(size, "byte"))
    if item.get("before_compaction") is True:
        parts.append("before compaction")
    return " · ".join(parts)


def _profile_counts(profile: Mapping[str, Any]) -> str | None:
    """Return what the note of a commit counted for one agent, or None."""

    calls = _mapping(profile.get("tool_calls"))
    parts = [
        _counted(calls.get(name), singular, plural)
        for name, singular, plural in _LOADED_TOOL_CLASSES
        if _integer(calls.get(name)) is not None
    ]
    prompts = _integer(profile.get("prompt_count"))
    if prompts:
        parts.insert(0, _counted(prompts, "prompt"))
    return " · ".join(parts) or None


def _render_owner(owner: Mapping[str, Any], width: int) -> list[str]:
    """Render one owning session, its commit, and what it had loaded."""

    session = _mapping(owner.get("session"))
    commit = _mapping(owner.get("commit"))
    lines = [""]
    _append_field(
        lines,
        "Session",
        safe_text(session.get("id")) or "unknown session",
        width,
        indent="  ",
    )
    _append_field(
        lines,
        "Model / harness",
        f"{safe_text(session.get('model')) or 'unknown'} / "
        f"{safe_text(session.get('harness')) or 'unknown'}",
        width,
        indent="    ",
    )
    agent_type = safe_text(session.get("agent_type"))
    if agent_type:
        _append_field(lines, "Agent", agent_type, width, indent="    ")
    _append_field(
        lines,
        "Role",
        safe_text(session.get("role")) or "implementation",
        width,
        indent="    ",
    )
    launch_mode = safe_text(session.get("launch_mode"))
    if launch_mode:
        _append_field(lines, "Launch mode", launch_mode, width, indent="    ")
    parent = safe_text(session.get("parent_session_id"))
    if parent:
        _append_field(lines, "Parent session", parent, width, indent="    ")
    _append_field(lines, "Cost", _session_cost(session), width, indent="    ")
    _append_field(
        lines,
        "Commit",
        " · ".join(
            part
            for part in (
                safe_text(commit.get("short_sha")) or "unknown",
                safe_text(commit.get("subject")),
                _date(commit.get("committed_at")),
            )
            if part
        ),
        width,
        indent="    ",
    )
    numbers = _items(owner.get("lines"))
    count = _integer(owner.get("line_count")) or len(numbers)
    ranges = _line_ranges(numbers)
    owned = ranges if count == 1 else f"{ranges} · {_counted(count, 'line')}"
    if owner.get("lines_truncated") is True:
        # The ranges describe the listed lines; the count describes them all.
        owned += f" · the first {len(numbers):,} are listed"
    _append_field(lines, "Lines", owned, width, indent="    ")
    prompt = _mapping(owner.get("prompt"))
    if prompt:
        _append_field(
            lines, "Prompt", safe_text(prompt.get("text")).strip().splitlines()[0][:200],
            width, indent="    ",
        )

    lines.append("")
    unavailable = safe_text(owner.get("context_unavailable"))
    if unavailable:
        _append_wrapped(lines, unavailable, width, indent="    ")
        counts = _profile_counts(_mapping(owner.get("profile")))
        if counts:
            _append_wrapped(lines, counts, width, indent="      ")
        return lines
    items = _items(owner.get("context"))
    total = _integer(owner.get("context_total"))
    heading = _CONTEXT_CUT_TEXT.get(
        safe_text(owner.get("context_cut")), "Loaded before this commit"
    )
    if total is not None and total > len(items):
        heading += f" · {len(items):,} of {total:,}"
    else:
        heading += f" · {_counted(total if total is not None else len(items), 'item')}"
    _append_wrapped(lines, heading, width, indent="    ")
    for item in items:
        _append_wrapped(lines, _context_text(_mapping(item)), width, indent="      ")
    if not items:
        lines.append("      none")
    elif total is not None and total > len(items):
        _append_wrapped(
            lines, f"... {total - len(items):,} more items not shown", width, indent="      "
        )
    return lines


def render_why(
    data: Mapping[str, Any], *, width: int | None = None, color: bool | None = None
) -> str:
    """Render each owning session of a line or file, and what it had loaded."""

    payload = _mapping(data)
    output_width = _width(width)
    use_color = _color_enabled(color)
    file_path = safe_text(payload.get("file")) or "unknown file"
    line_number = _integer(payload.get("line"))
    # The same header as the report, so an example repository is marked as one
    # here too. The file this command answers for follows it.
    lines, _ = _repository_lines(payload, output_width)
    _append_field(
        lines,
        "File",
        file_path if line_number is None else f"{file_path}:{line_number:,}",
        output_width,
    )
    owners = _items(payload.get("owners"))
    lines.append("")
    lines.append("Owners")
    if not owners:
        lines.append("  none recorded for these lines")
    for owner in owners[:_MAX_SESSIONS]:
        lines.extend(_render_owner(_mapping(owner), output_width))
    if len(owners) > _MAX_SESSIONS:
        _append_wrapped(
            lines,
            f"... {len(owners) - _MAX_SESSIONS:,} more sessions not shown",
            output_width,
            indent="  ",
        )
    lines.append("")
    # A load proves that content entered the context window. It does not prove
    # that the model relied on it.
    _append_wrapped(
        lines,
        "Loaded means it entered the context window. It does not prove that the "
        "model relied on it.",
        output_width,
    )
    lines.extend(_warning_lines(payload.get("warnings"), width=output_width))
    return "\n".join(_paint_common(lines, use_color)).rstrip() + "\n"


def _install_state(value: Any) -> str:
    return {
        "enabled": "enabled",
        "not-installed": "not installed",
        "needs-attention": "needs attention",
    }.get(value, "unknown")


def render_status(
    installation: Mapping[str, Any],
    automation: Mapping[str, Any] | None,
    *,
    width: int | None = None,
    color: bool | None = None,
) -> str:
    """Render automatic-tracking installation and queue status."""

    status = _mapping(installation)
    queue = _mapping(automation) or _mapping(status.get("automation"))
    output_width = _width(width)
    use_color = _color_enabled(color)
    harnesses = _mapping(status.get("harnesses"))
    harness_values = [_mapping(value) for value in harnesses.values()]
    needs_attention = any(item.get("state") == "needs-attention" for item in harness_values)
    if status.get("installed") is True and status.get("git_hook_installed") is not True:
        needs_attention = True
    if needs_attention:
        overall = "needs attention"
    elif status.get("installed") is True:
        overall = "enabled"
    elif status.get("installed") is False:
        overall = "not installed"
    else:
        overall = "unknown"

    lines = ["Automatic tracking"]
    _append_field(lines, "Status", overall, output_width)
    repository_path = safe_text(status.get("repository_path")) or "unknown"
    if status.get("git_repository") is False:
        repository_path += " (not a Git repository)"
    _append_field(lines, "Repository", repository_path, output_width)
    user_scope = _mapping(status.get("user_scope"))
    if user_scope:
        _append_field(
            lines,
            "Machine hooks",
            "enabled" if user_scope.get("installed") is True else "not installed",
            output_width,
        )
    if status.get("git_repository") is False:
        lines.extend(_warning_lines(status.get("warnings"), width=output_width))
        return "\n".join(_paint_common(lines, use_color)).rstrip() + "\n"
    enabled_worktrees = _integer(status.get("enabled_worktree_count"))
    discovered_worktrees = _integer(status.get("discovered_worktree_count"))
    if enabled_worktrees is not None and discovered_worktrees is not None:
        if enabled_worktrees <= discovered_worktrees:
            coverage = f"{enabled_worktrees:,} of {discovered_worktrees:,} enabled"
        else:
            coverage = (
                f"{enabled_worktrees:,} enabled; "
                f"{discovered_worktrees:,} currently available"
            )
        _append_field(lines, "Worktrees", coverage, output_width)
    current_worktree = next(
        (
            _mapping(item)
            for item in _items(status.get("worktrees"))
            if _mapping(item).get("current") is True
        ),
        {},
    )
    current_branch = safe_text(current_worktree.get("branch"))
    current_path = safe_text(current_worktree.get("path"))
    if current_branch or current_path:
        _append_field(
            lines,
            "Current worktree",
            current_branch or current_path,
            output_width,
        )
    else:
        _append_field(
            lines,
            "Worktree",
            safe_text(status.get("worktree_id")) or "unknown",
            output_width,
        )

    for key, label in (("codex", "Codex"), ("claude-code", "Claude Code")):
        harness = _mapping(harnesses.get(key))
        harness_state = _install_state(harness.get("state"))
        message = safe_text(harness.get("message"))
        if message and harness.get("state") == "needs-attention":
            harness_state += f" - {message}"
        _append_field(lines, label, harness_state, output_width)

    if status.get("git_hook_installed") is True:
        git_hook = "enabled"
    elif status.get("git_hook_installed") is False:
        git_hook = "not installed"
    else:
        git_hook = "unknown"
    _append_field(lines, "Git post-commit hook", git_hook, output_width)
    if "traces_enabled" in status:
        _append_field(
            lines, "Traces", "on" if status.get("traces_enabled") is not False else "off",
            output_width,
        )
    telemetry = _mapping(status.get("telemetry"))
    if telemetry:
        collector = _mapping(telemetry.get("collector"))
        claude_cost = _mapping(telemetry.get("claude"))
        codex_cost = _mapping(telemetry.get("codex"))
        _append_field(
            lines,
            "Telemetry collector",
            "enabled" if collector.get("enabled") is True else "needs attention",
            output_width,
        )
        if claude_cost:
            claude_value = (
                "enabled" if claude_cost.get("enabled") is True else "needs attention"
            )
            message = safe_text(claude_cost.get("message"))
            if message and claude_cost.get("enabled") is not True:
                claude_value += f" - {message}"
            _append_field(
                lines,
                "Claude cost",
                claude_value,
                output_width,
            )
        if codex_cost:
            codex_value = "enabled" if codex_cost.get("enabled") is True else "needs attention"
            message = safe_text(codex_cost.get("message"))
            if message and codex_cost.get("enabled") is not True:
                codex_value += f" - {message}"
            _append_field(lines, "Codex cost", codex_value, output_width)
    _append_field(
        lines,
        "Pending",
        f"{_counted(queue.get('pending_captures'), 'capture')}, "
        f"{_counted(queue.get('pending_commits'), 'commit')}",
        output_width,
    )
    notice = safe_text(status.get("notice"))
    if notice and status.get("installed") is True:
        _append_field(lines, "Notice", notice, output_width)
    elif status.get("installed") is False and status.get("git_repository") is True:
        enable = (
            "joyride install --user"
            if user_scope and user_scope.get("installed") is not True
            else "joyride install ."
        )
        _append_field(lines, "Enable", enable, output_width)
    installation_warnings = _items(status.get("warnings"))
    queue_warnings = _items(queue.get("warnings"))
    if status.get("installed") is False:
        installation_warnings = [
            item for item in installation_warnings
            if "not enabled for this worktree" not in safe_text(item).casefold()
        ]
        queue_warnings = [
            item for item in queue_warnings
            if "not enabled for this worktree" not in safe_text(item).casefold()
        ]
    lines.extend(_warning_lines(installation_warnings, queue_warnings, width=output_width))
    return "\n".join(_paint_common(lines, use_color)).rstrip() + "\n"


def render_setup(
    status: Mapping[str, Any], *, installed: bool, width: int | None = None,
    color: bool | None = None,
) -> str:
    """Render a concise result for repository installation or removal."""

    data = _mapping(status)
    output_width = _width(width)
    use_color = _color_enabled(color)
    repository = safe_text(data.get("repository_path")) or "the selected repository"
    if installed and data.get("installed") is True:
        heading, heading_color = "Joyride enabled", _GREEN
    elif installed:
        heading, heading_color = "Joyride needs attention", _YELLOW
    else:
        heading, heading_color = "Joyride disabled", _DIM
    lines = [_style(heading, _BOLD + heading_color, use_color)]
    _append_field(lines, "Repository", repository, output_width)
    count = _integer(data.get("enabled_worktree_count"))
    if count is not None:
        _append_field(lines, "Enabled worktrees", f"{count:,}", output_width)
    telemetry = _mapping(data.get("telemetry"))
    if installed and telemetry:
        collector = _mapping(telemetry.get("collector"))
        claude_cost = _mapping(telemetry.get("claude"))
        codex_cost = _mapping(telemetry.get("codex"))
        enabled_count = sum(
            item.get("enabled") is True for item in (claude_cost, codex_cost) if item
        )
        expected_count = sum(bool(item) for item in (claude_cost, codex_cost))
        if collector.get("enabled") is not True:
            cost_state = "needs attention"
        elif expected_count and enabled_count == expected_count:
            cost_state = "enabled for Claude and Codex"
        elif enabled_count:
            cost_state = "partly enabled"
        else:
            cost_state = "needs attention"
        _append_field(lines, "Cost collection", cost_state, output_width)
    if installed and isinstance(data.get("notice"), str) and data["notice"]:
        _append_field(lines, "Notice", data["notice"], output_width)
    if not installed:
        _append_wrapped(lines, "Existing attribution data was preserved.", output_width)
    lines.extend(_warning_lines(data.get("warnings"), width=output_width))
    return "\n".join(_paint_common(lines, use_color)).rstrip() + "\n"


def render_user_setup(
    status: Mapping[str, Any], *, installed: bool, width: int | None = None,
    color: bool | None = None,
) -> str:
    """Render the result of a machine-level install or removal."""

    data = _mapping(status)
    output_width = _width(width)
    use_color = _color_enabled(color)
    if installed and data.get("installed") is True:
        heading, heading_color = "Machine hooks enabled", _GREEN
    elif installed:
        heading, heading_color = "Machine hooks need attention", _YELLOW
    else:
        heading, heading_color = "Machine hooks removed", _DIM
    lines = [_style(heading, _BOLD + heading_color, use_color)]
    _append_field(lines, "Scope", "every repository on this machine", output_width)
    harnesses = _mapping(data.get("harnesses"))
    for key, label in (("codex", "Codex"), ("claude-code", "Claude Code")):
        harness = _mapping(harnesses.get(key))
        harness_state = _install_state(harness.get("state"))
        message = safe_text(harness.get("message"))
        if message and harness.get("state") == "needs-attention":
            harness_state += f" - {message}"
        _append_field(lines, label, harness_state, output_width)
    if installed and isinstance(data.get("notice"), str) and data["notice"]:
        _append_field(lines, "Notice", data["notice"], output_width)
    if installed:
        _append_wrapped(
            lines,
            "Enrolled repositories finish their own setup at the next hook event.",
            output_width,
        )
    else:
        _append_wrapped(lines, "Existing attribution data was preserved.", output_width)
    lines.extend(_warning_lines(data.get("warnings"), width=output_width))
    return "\n".join(_paint_common(lines, use_color)).rstrip() + "\n"


def render_capture(
    result: Mapping[str, Any],
    *,
    feature: str,
    model: str,
    harness: str,
    cost_usd: float | None,
    width: int | None = None,
    color: bool | None = None,
) -> str:
    """Render the result of the manual command wrapper."""

    data = _mapping(result)
    output_width = _width(width)
    use_color = _color_enabled(color)
    exit_code = _integer(data.get("exit_code"))
    succeeded = exit_code == 0
    heading = "Capture complete" if succeeded else "Capture finished with errors"
    lines = [_style(heading, _BOLD + (_GREEN if succeeded else _RED), use_color)]
    _append_field(lines, "Feature", feature, output_width)
    _append_field(lines, "Model / harness", f"{model} / {harness}", output_width)
    _append_field(lines, "Session", safe_text(data.get("session_id")) or "unknown", output_width)
    changed = [safe_text(item) for item in _items(data.get("changed_files"))]
    if not changed:
        changed_summary = "none"
    elif len(changed) <= 3:
        changed_summary = ", ".join(changed)
    else:
        changed_summary = _counted(len(changed), "file")
    _append_field(lines, "Changed", changed_summary, output_width)
    skipped = _items(data.get("skipped_files"))
    if skipped:
        _append_field(lines, "Skipped", _counted(len(skipped), "unsupported file"), output_width)
    _append_field(lines, "Reported cost", _money(cost_usd), output_width)
    command_status = (
        "completed" if exit_code == 0 else (
            f"failed with exit {exit_code}" if exit_code is not None else "could not start"
        )
    )
    _append_field(lines, "Command", command_status, output_width)
    return "\n".join(_paint_common(lines, use_color)).rstrip() + "\n"


__all__ = [
    "feature_detail", "render_capture", "render_feature", "render_report",
    "render_setup", "render_status", "render_user_setup", "safe_text",
    "sanitize_terminal",
]
