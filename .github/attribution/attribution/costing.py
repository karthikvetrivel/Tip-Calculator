"""Allocate local provider telemetry to durable attribution sessions."""

from __future__ import annotations

from collections import Counter, defaultdict
import math
from pathlib import Path
import sqlite3
from typing import Any, Iterable, Mapping
from urllib.parse import quote


_COMPLETED_CAPTURE_STATES = {
    "completed",
    "completed_contaminated",
    "completed_imported",
    "completed_limited",
}
# What Claude Code names as the source of a request that its own loop made,
# rather than one an agent it started made.
_MAIN_QUERY_SOURCES = {"sdk", "repl_main_thread"}


def _base_native_session(value: Any) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    return value.split("::", 1)[0]


def _model_key(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip().casefold()


def _provider_harness(provider: Any) -> str | None:
    return {"claude": "claude-code", "codex": "codex"}.get(provider)


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) and number >= 0 else None


def _identity(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip()


def _session_agent_id(session: Mapping[str, Any]) -> str | None:
    """Return the harness agent ID of one session, or None for a main session.

    A subagent's session is keyed by its parent's native session and its own
    agent ID, so the key names the agent even where the column does not.
    """

    agent = _identity(session.get("native_agent_id"))
    if agent is not None:
        return agent
    native = session.get("native_session_id")
    if isinstance(native, str) and "::" in native:
        return _identity(native.split("::", 1)[1])
    return None


def _session_agent_type(session: Mapping[str, Any]) -> str | None:
    agent_type = _identity(session.get("agent_type"))
    return agent_type.casefold() if agent_type is not None else None


def _event_agent_type(event: Mapping[str, Any]) -> str | None:
    """Return the agent type a request names, folded for comparison.

    Claude Code names the agent in ``agent_name`` where it has one, and
    otherwise in the last segment of a ``query_source`` such as
    ``agent:builtin:Explore``.
    """

    name = _identity(event.get("agent_name"))
    if name is None:
        source = _identity(event.get("query_source"))
        if source is None or not source.startswith("agent:"):
            return None
        name = _identity(source.rsplit(":", 1)[-1])
    return name.casefold() if name is not None else None


def _agent_targets(
    event: Mapping[str, Any],
    eligible: set[str],
    by_id: Mapping[str, Mapping[str, Any]],
) -> set[str]:
    """Return the sessions of one native session that a request's agent names.

    A subagent runs its tools inside its parent's prompt, so the prompt-to-tool
    link cannot tell the two apart and spreads every request of that prompt
    across both. The request itself says which agent asked for it, and each
    agent has its own ledger session, so that identity is read first.
    """

    agent = _identity(event.get("agent_id"))
    if agent is not None:
        matched = {
            session_id
            for session_id in eligible
            if _session_agent_id(by_id[session_id]) == agent
        }
        # An ID both sides carry is exact. One that names no session here falls
        # through to the agent type, which more harnesses report.
        if matched:
            return matched
    agent_type = _event_agent_type(event)
    if agent_type is not None:
        return {
            session_id
            for session_id in eligible
            if _session_agent_type(by_id[session_id]) == agent_type
        }
    source = event.get("query_source")
    if (
        isinstance(source, str)
        and source.strip().split(":", 1)[0] in _MAIN_QUERY_SOURCES | {""}
    ):
        # The main loop's own requests. Claude Code appends the active output
        # style to the loop's name, as ``repl_main_thread:outputStyle:custom``,
        # so the first segment names the loop. Every other session of this
        # native session belongs to an agent that loop started. An exporter
        # that reports no source at all names no agent, and nothing is
        # inferred from that absence: the links below decide instead.
        return {
            session_id
            for session_id in eligible
            if _session_agent_type(by_id[session_id]) is None
        }
    return set()


def _capture_weights(captured: Mapping[str, int], targets: Iterable[str]) -> Counter[str]:
    """Weight sessions by the completed captures each one recorded.

    A session that recorded no work takes no share while a sibling recorded
    some. Where none of them recorded any, nothing says one worked more than
    another, and the request is split evenly.
    """

    weights = Counter({session_id: captured.get(session_id, 0) for session_id in targets})
    if any(weight > 0 for weight in weights.values()):
        return Counter({key: value for key, value in weights.items() if value > 0})
    return Counter({session_id: 1 for session_id in targets})


def _agent_route(
    targets: set[str],
    event: Mapping[str, Any],
    harness: str,
    native: str,
    prompt_links: Mapping[tuple[str, str], list[str]],
    tools: Mapping[tuple[str, str], Counter[str]],
    native_routes: Mapping[tuple[str, str], Counter[str]],
) -> Counter[str]:
    """Weight one request across the sessions its agent identity names."""

    if len(targets) == 1:
        return Counter({next(iter(targets)): 1})
    # Two agents of one type can run at once. Both measures below already say
    # how much work each session recorded, so they split the request here as
    # well, restricted to the agent's own sessions.
    linked: Counter[str] = Counter()
    prompt = event.get("prompt_id")
    if isinstance(prompt, str) and prompt:
        for tool in prompt_links.get((native, prompt), []):
            linked.update(tools.get((native, tool), {}))
    selected = Counter(
        {key: value for key, value in linked.items() if key in targets and value > 0}
    )
    if selected:
        return selected
    return _capture_weights(native_routes.get((harness, native), Counter()), targets)


def _counted_tokens(provider: Any, values: Mapping[str, Any]) -> float | None:
    """Return the tokens one request used where it reported no total itself.

    Claude Code reports four counts and no total, and its cached and cache
    creation counts are beside its input count rather than inside it, so the
    four add up. Codex reports an input count that already holds its cached
    part, which is why that part is left out of its sum rather than added
    twice. A request that reported no count at all stays unknown: a total is
    never a zero this function invented.
    """

    fields = ["input_tokens", "output_tokens", "cache_creation_input_tokens"]
    if provider != "codex":
        fields.append("cached_input_tokens")
    counted = [
        number
        for number in (_number(values.get(field)) for field in fields)
        if number is not None
    ]
    return math.fsum(counted) if counted else None


def _read_capture_routes(
    common_dir: Path,
) -> tuple[
    dict[tuple[str, str], Counter[str]],
    dict[tuple[str, str, str], Counter[str]],
    dict[tuple[str, str], Counter[str]],
]:
    """Return tool, turn, and native-session routes without mutating the ledger.

    A capture row is one tool call that took a snapshot. The fast path takes
    none, so a session whose tools were all reads opens no capture and would be
    reachable only by its native session. The activity ledger names those calls
    with the same tool use ID the exporter correlates on, so ``tool_calls`` is
    read beside ``hook_captures`` and a call that both tables name counts once.
    """

    database = common_dir / "attribution" / "ledger.sqlite3"
    if not database.is_file():
        return {}, {}, {}
    uri = f"file:{quote(str(database))}?mode=ro"
    connection = sqlite3.connect(uri, uri=True, timeout=1)
    connection.row_factory = sqlite3.Row
    try:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        rows: list[sqlite3.Row] = []
        if "hook_captures" in tables:
            columns = {
                row[1]
                for row in connection.execute("PRAGMA table_info(hook_captures)")
            }
            turn_select = "turn_id" if "turn_id" in columns else "NULL AS turn_id"
            rows = connection.execute(
                f"""
                SELECT harness, native_session_id, tool_use_id, ledger_session_id,
                       status, {turn_select}
                FROM hook_captures
                """
            ).fetchall()
        calls: list[sqlite3.Row] = []
        if {"tool_calls", "sessions"} <= tables:
            # A ``tool_calls`` row names its session, not its harness or its
            # native ID, so the session row carries both. An older ledger
            # without those columns simply contributes no route.
            session_columns = {
                row[1] for row in connection.execute("PRAGMA table_info(sessions)")
            }
            if {"harness", "harness_id", "native_session_id"} <= session_columns:
                calls = connection.execute(
                    """
                    SELECT COALESCE(sessions.harness_id, sessions.harness) AS harness,
                           sessions.native_session_id AS native_session_id,
                           tool_calls.tool_use_id AS tool_use_id,
                           tool_calls.session_id AS ledger_session_id,
                           tool_calls.turn_id AS turn_id
                    FROM tool_calls
                    JOIN sessions ON sessions.id = tool_calls.session_id
                    """
                ).fetchall()
    finally:
        connection.close()

    tools: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    turns: dict[tuple[str, str, str], Counter[str]] = defaultdict(Counter)
    sessions: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    captured: set[tuple[str, str, str]] = set()

    def route(harness: Any, raw_native: Any, ledger: Any, tool: Any, turn: Any) -> str | None:
        native = _base_native_session(raw_native)
        if not all(isinstance(value, str) and value for value in (harness, native, ledger)):
            return None
        sessions[(harness, native)][ledger] += 1
        if isinstance(tool, str) and tool:
            tools[(native, tool)][ledger] += 1
        if isinstance(turn, str) and turn:
            turns[(harness, native, turn)][ledger] += 1
        return native

    for row in rows:
        if row["status"] not in _COMPLETED_CAPTURE_STATES:
            continue
        native = route(
            row["harness"],
            row["native_session_id"],
            row["ledger_session_id"],
            row["tool_use_id"],
            row["turn_id"],
        )
        tool = row["tool_use_id"]
        if native is not None and isinstance(tool, str) and tool:
            captured.add((row["harness"], native, tool))
    for row in calls:
        tool = row["tool_use_id"]
        native = _base_native_session(row["native_session_id"])
        if (
            native is not None
            and isinstance(tool, str)
            and tool
            and (row["harness"], native, tool) in captured
        ):
            # The capture above already routed this call. One tool call is one
            # route, whichever table recorded it.
            continue
        route(
            row["harness"],
            row["native_session_id"],
            row["ledger_session_id"],
            tool,
            row["turn_id"],
        )
    return dict(tools), dict(turns), dict(sessions)


def _telemetry_rows(repository_id: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Read the global telemetry spool only when it already exists."""

    from . import telemetry

    state_dir = telemetry.default_state_dir()
    if not (state_dir / "telemetry.sqlite3").is_file():
        return [], []
    return (
        telemetry.query_events(repository_id=repository_id, state_dir=state_dir),
        telemetry.query_claude_prompt_tool_links(
            repository_id=repository_id, state_dir=state_dir
        ),
    )


def _weighted_targets(routes: Counter[str], valid: set[str]) -> dict[str, float]:
    # The total is over every session the route names, so the share of a
    # session this caller did not pass is left out rather than handed to its
    # siblings. A report that lists fewer sessions then counts less, never more.
    total = sum(value for value in routes.values() if value > 0)
    if total <= 0:
        return {}
    return {
        key: value / total
        for key, value in routes.items()
        if key in valid and value > 0
    }


def allocate_session_usage(
    common_dir: str | Path,
    sessions: Iterable[Mapping[str, Any]],
    *,
    events: list[dict[str, Any]] | None = None,
    prompt_tool_links: list[dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Allocate deduplicated usage events without counting one event twice.

    A Claude request first follows the agent that made it: an orchestrator and
    the subagents it starts share one native session but each keep their own
    ledger session, and only the request says whose it was. A request that
    names no agent, or one this clone recorded no session for, then follows the
    documented prompt-to-tool correlation. Codex turns first follow the native
    turn ID. If an exporter does not expose either link, an event is allocated
    within its native session by completed capture count. The returned
    ``allocation`` value makes these distinctions visible to callers.
    """

    common = Path(common_dir).resolve()
    session_rows = [dict(item) for item in sessions]
    by_id = {
        item["id"]: item
        for item in session_rows
        if isinstance(item.get("id"), str) and item["id"]
    }
    if not by_id:
        return {}
    if events is None or prompt_tool_links is None:
        loaded_events, loaded_links = _telemetry_rows(str(common))
        if events is None:
            events = loaded_events
        if prompt_tool_links is None:
            prompt_tool_links = loaded_links
    if not events:
        return {}

    tools, turns, native_routes = _read_capture_routes(common)
    valid_ids = set(by_id)
    candidates: dict[tuple[str, str], set[str]] = defaultdict(set)
    for session_id, session in by_id.items():
        harness = session.get("harness_id") or session.get("harness")
        native = _base_native_session(session.get("native_session_id"))
        if isinstance(harness, str) and native:
            candidates[(harness, native)].add(session_id)

    prompt_links: dict[tuple[str, str], list[str]] = defaultdict(list)
    for link in prompt_tool_links or []:
        native = _base_native_session(link.get("native_session_id"))
        prompt = link.get("prompt_id")
        tool = link.get("tool_use_id")
        if native and isinstance(prompt, str) and prompt and isinstance(tool, str) and tool:
            prompt_links[(native, prompt)].append(tool)

    totals: dict[str, dict[str, Any]] = {}
    for raw_event in events:
        provider = raw_event.get("provider")
        harness = _provider_harness(provider)
        native = _base_native_session(raw_event.get("native_session_id"))
        if harness is None or native is None:
            continue
        event_values = dict(raw_event)
        if provider == "codex" and _number(event_values.get("credits")) is not None:
            from .telemetry import compute_openai_api_usd

            equivalent = compute_openai_api_usd(
                str(event_values.get("model") or ""),
                event_values.get("input_tokens"),
                event_values.get("cached_input_tokens"),
                event_values.get("output_tokens"),
                cache_write_input_tokens=event_values.get(
                    "cache_creation_input_tokens"
                ),
                service_tier=event_values.get("service_tier"),
            )
            if equivalent is not None:
                event_values["codex_api_equivalent_usd"] = float(equivalent)
        if _number(event_values.get("total_tokens")) is None:
            counted = _counted_tokens(provider, event_values)
            if counted is not None:
                event_values["total_tokens"] = counted
        eligible = candidates.get((harness, native), set())
        if not eligible:
            continue

        route = Counter()
        allocation = "session-allocated"
        if provider == "claude":
            targets = _agent_targets(raw_event, eligible, by_id)
            if targets and targets != eligible:
                # The agent identity narrows this request to fewer sessions
                # than the native session holds. Where it names every one of
                # them, or none, it separates nothing and the links below
                # decide exactly as they did before.
                route.update(
                    _agent_route(
                        targets,
                        raw_event,
                        harness,
                        native,
                        prompt_links,
                        tools,
                        native_routes,
                    )
                )
                allocation = "agent-routed"
            if not route:
                prompt = raw_event.get("prompt_id")
                if isinstance(prompt, str) and prompt:
                    for tool in prompt_links.get((native, prompt), []):
                        route.update(tools.get((native, tool), {}))
                    if route:
                        allocation = "tool-linked"
        elif provider == "codex":
            turn = raw_event.get("turn_id")
            if isinstance(turn, str) and turn:
                route.update(turns.get((harness, native, turn), {}))
                if route:
                    allocation = "turn-linked"

        if not route:
            model = _model_key(raw_event.get("model"))
            model_eligible = {
                session_id
                for session_id in eligible
                if model is not None and _model_key(by_id[session_id].get("model")) == model
            }
            selected = model_eligible or eligible
            route.update(
                _capture_weights(native_routes.get((harness, native), Counter()), selected)
            )

        weights = _weighted_targets(route, valid_ids & eligible)
        if not weights:
            continue
        numeric_fields = (
            "cost_usd",
            "credits",
            "codex_api_equivalent_usd",
            "input_tokens",
            "cached_input_tokens",
            "cache_creation_input_tokens",
            "output_tokens",
            "total_tokens",
        )
        is_request = raw_event.get("event_name") in {
            "claude_code.api_request",
            "codex.sse_event",
            "codex.turn.token_usage",
        }
        request_has_cost = (
            _number(event_values.get("cost_usd")) is not None
            or _number(event_values.get("credits")) is not None
        )
        for session_id, weight in weights.items():
            usage = totals.setdefault(
                session_id,
                {
                    "estimated_cost_usd": 0.0,
                    "codex_credits": 0.0,
                    "codex_api_equivalent_usd": 0.0,
                    "input_tokens": 0.0,
                    "cached_input_tokens": 0.0,
                    "cache_creation_input_tokens": 0.0,
                    "output_tokens": 0.0,
                    "total_tokens": 0.0,
                    "request_count": 0.0,
                    "priced_request_count": 0.0,
                    "models": set(),
                    "model_output_tokens": {},
                    "sources": set(),
                    "allocations": set(),
                    "has_usd": False,
                    "has_credits": False,
                    "has_equivalent_usd": False,
                },
            )
            for field in numeric_fields:
                number = _number(event_values.get(field))
                if number is None:
                    continue
                destination = {
                    "cost_usd": "estimated_cost_usd",
                    "credits": "codex_credits",
                }.get(field, field)
                usage[destination] += number * weight
                if field == "cost_usd":
                    usage["has_usd"] = True
                elif field == "credits":
                    usage["has_credits"] = True
                elif field == "codex_api_equivalent_usd":
                    usage["has_equivalent_usd"] = True
            if is_request:
                usage["request_count"] += weight
                if request_has_cost:
                    usage["priced_request_count"] += weight
            model = raw_event.get("model")
            if isinstance(model, str) and model:
                usage["models"].add(model)
                # Which model wrote how much, so a session that used several of
                # them can be labelled by the one that did most of the work.
                output = _number(event_values.get("output_tokens"))
                if output is not None:
                    outputs = usage["model_output_tokens"]
                    outputs[model] = outputs.get(model, 0.0) + output * weight
            source = raw_event.get("cost_source")
            if isinstance(source, str) and source:
                usage["sources"].add(source)
            usage["allocations"].add(allocation)

    result: dict[str, dict[str, Any]] = {}
    for session_id, usage in totals.items():
        has_usd = bool(usage.pop("has_usd"))
        has_credits = bool(usage.pop("has_credits"))
        has_equivalent_usd = bool(usage.pop("has_equivalent_usd"))
        usage["estimated_cost_usd"] = (
            round(usage["estimated_cost_usd"], 10) if has_usd else None
        )
        usage["codex_credits"] = (
            round(usage["codex_credits"], 10) if has_credits else None
        )
        usage["codex_api_equivalent_usd"] = (
            round(usage["codex_api_equivalent_usd"], 10)
            if has_equivalent_usd
            else None
        )
        for field in (
            "input_tokens",
            "cached_input_tokens",
            "cache_creation_input_tokens",
            "output_tokens",
            "total_tokens",
        ):
            usage[field] = round(usage[field], 3)
        request_count = usage["request_count"]
        priced_request_count = usage["priced_request_count"]
        usage["request_count"] = round(request_count, 3)
        usage["priced_request_count"] = round(priced_request_count, 3)
        usage["unpriced_request_count"] = round(
            max(request_count - priced_request_count, 0.0), 3
        )
        usage["cost_complete"] = request_count > 0 and math.isclose(
            priced_request_count,
            request_count,
            rel_tol=1e-9,
            abs_tol=1e-9,
        )
        usage["models"] = sorted(usage["models"])
        usage["model_output_tokens"] = {
            model: round(counted, 3)
            for model, counted in sorted(usage["model_output_tokens"].items())
        }
        usage["sources"] = sorted(usage["sources"])
        allocations = usage.pop("allocations")
        usage["allocation"] = (
            next(iter(allocations)) if len(allocations) == 1 else "mixed"
        )
        result[session_id] = usage
    return result


__all__ = ["allocate_session_usage"]
