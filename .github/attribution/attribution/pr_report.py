"""Measure a PR's final diff and render its small, replaceable description footer."""

from __future__ import annotations

from collections import Counter
import html
import json
import math
from pathlib import Path
import re
import sqlite3
from typing import Any, Mapping

from .costing import allocate_session_usage
from .harnesses import harness_display_name
from .notes import _blame, _blob_content, _changed_files, _tree, record_commit
from .report import (
    _Warnings, _decode_git_path, _load_local_sessions, _load_notes,
    _load_task_notes, _merge_session, _normalise_session, _normalise_task,
    _note_workflow, _parse_note, _parse_time, _resolve_target, _run_git,
    _task_economics, _workflow_profile, telemetry_model,
)
from .store import git_common_dir
from .workflow import iter_agents, native_identity_hash

START = "<!-- attribution:start -->"
END = "<!-- attribution:end -->"
_HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
_MAX_FILES = 2000
_MAX_BYTES = 32 * 1024 * 1024
_MAX_FILE_LINES = 100_000
_MAX_LINES = 500_000
# The structured narrative feeds the paragraph of the hosted page and the
# bounded model input, not the footer. These caps keep both of them to a size a
# reader holds; each one is reported beside the list it cut, so nothing is lost
# in silence.
_MAX_NARRATIVE_AGENTS = 12
_MAX_NARRATIVE_GROUPS = 6
_MAX_NARRATIVE_FILES = 5


def _scope(repo: Path, base_ref: str, head_ref: str) -> tuple[str, str, str, list[str]]:
    # A bare repository is used by the trusted GitHub workflow.
    _run_git(repo, "rev-parse", "--git-dir")
    base = _resolve_target(repo, base_ref)
    head = _resolve_target(repo, head_ref)
    if base is None or head is None:
        raise ValueError("The PR base and head must resolve to commits")
    bases = _run_git(repo, "merge-base", "--all", base, head).stdout.splitlines()
    if len(bases) != 1:
        raise ValueError("The PR must have one unambiguous merge base")
    commits = _run_git(repo, "rev-list", "--reverse", f"{base}..{head}").stdout.splitlines()
    return base, head, bases[0], commits


def _positions(patch: str) -> dict[str, set[int]]:
    """Parse actual Git added-line positions, including quoted paths and no-EOL files."""
    positions: dict[str, set[int]] = {}
    path: str | None = None
    old_left = new_left = next_line = 0
    for line in patch.split("\n"):
        if old_left or new_left:
            if line.startswith("+"):
                if path is None or new_left <= 0:
                    raise ValueError("Unexpected added line in PR diff")
                positions.setdefault(path, set()).add(next_line)
                next_line += 1
                new_left -= 1
            elif line.startswith("-"):
                old_left -= 1
            elif line.startswith(" "):
                old_left -= 1
                new_left -= 1
                next_line += 1
            elif not line.startswith("\\"):
                raise ValueError("Incomplete PR diff hunk")
            if old_left < 0 or new_left < 0:
                raise ValueError("Invalid PR diff hunk")
            continue
        if line.startswith("+++ "):
            value = _decode_git_path(line[4:])
            path = value[2:] if value.startswith("b/") else None
        match = _HUNK.match(line)
        if match:
            old_left = int(match[2]) if match[2] is not None else 1
            next_line = int(match[3])
            new_left = int(match[4]) if match[4] is not None else 1
    if old_left or new_left:
        raise ValueError("Incomplete PR diff")
    return positions


def _added_positions(repo: Path, base: str, head: str, warnings: _Warnings):
    old_tree, new_tree = _tree(repo, base), _tree(repo, head)
    changes = _changed_files(repo, base, head)
    allowed: set[str] = set()
    pathspecs: set[str] = set()
    inspected = 0
    inspected_lines = 0
    complete = True
    for index, change in enumerate(changes):
        if index >= _MAX_FILES:
            complete = False
            warnings.add("PR text inspection reached its file limit.")
            break
        old_entry, new_entry = old_tree.get(change.old_path), new_tree.get(change.new_path)
        if old_entry is None and new_entry is None:
            # Gitlinks are omitted by _tree: their synthetic 'Subproject commit'
            # diff line is repository metadata, not a contributed text line.
            continue
        old, old_skip = _blob_content(repo, old_entry)
        new, new_skip = _blob_content(repo, new_entry)
        reasons = {reason for reason in (old_skip, new_skip) if reason}
        if reasons:
            # Binary files have no text-line denominator. Other skipped text
            # candidates make the measured scope incomplete.
            if reasons != {"binary"}:
                complete = False
                warnings.add("Some changed files could not be measured as text.")
            continue
        inspected += len(old or b"") + len(new or b"")
        line_counts = [content.count(b"\n") + int(not content.endswith(b"\n"))
                       for content in (old, new) if content]
        if any(count > _MAX_FILE_LINES for count in line_counts):
            complete = False
            warnings.add("Some changed files exceed the text-line inspection limit.")
            continue
        inspected_lines += sum(line_counts)
        if inspected_lines > _MAX_LINES:
            complete = False
            warnings.add("PR text inspection reached its line limit.")
            break
        if inspected > _MAX_BYTES:
            complete = False
            warnings.add("PR text inspection reached its byte limit.")
            break
        if change.new_path is not None and new_entry is not None:
            allowed.add(change.new_path)
        pathspecs.update(path for path in (change.old_path, change.new_path) if path is not None)
    if not pathspecs:
        return {}, complete
    patch = _run_git(
        repo, "--literal-pathspecs", "diff", "--no-ext-diff", "--no-textconv",
        "--no-color", "--unified=0", "--find-renames", "--src-prefix=a/",
        "--dst-prefix=b/", base, head, "--", *sorted(pathspecs), text=False,
    ).stdout.decode("utf-8", errors="replace")
    return {path: lines for path, lines in _positions(patch).items() if path in allowed}, complete


def _task_records(
    repo: Path,
    supplied: list[dict[str, Any]] | None,
    warnings: _Warnings,
) -> list[dict[str, Any]]:
    if supplied is None:
        return []
    if not isinstance(supplied, list):
        raise ValueError("Shared attribution tasks must be an array")
    records: list[dict[str, Any]] = []
    for raw in supplied:
        if not isinstance(raw, dict):
            raise ValueError("Invalid shared attribution task")
        encoded = json.dumps(raw)
        if len(encoded.encode()) > 2 * 1024 * 1024:
            raise ValueError("Shared attribution task exceeds the size limit")
        task = _normalise_task(raw, source="shared task metadata", warnings=warnings)
        anchor = raw.get("anchor_commit")
        sessions = raw.get("sessions")
        if task is None or not isinstance(anchor, str) or not isinstance(sessions, list):
            raise ValueError("Invalid shared attribution task")
        records.append({
            **task,
            "anchor_commit": anchor.lower(),
            "recorded_at": raw.get("updated_at"),
            "sessions": sessions,
        })
    return records


def _subagent_groups(
    agent: dict[str, Any],
    sessions: dict[str, dict[str, Any]],
    owned_lines: Counter[str],
) -> tuple[list[dict[str, Any]], int]:
    """Return what one agent's subagents did, one row for each kind of them.

    Ten Explore subagents are one fact about the work, not ten. Grouping them
    by agent type and model is what turns a tree into a sentence a reader can
    hold, and the reads and searches of a group are what those agents did.
    """
    groups: dict[tuple[str | None, str | None], dict[str, Any]] = {}
    for child in iter_agents({"agents": agent.get("children") or []}):
        key = (child.get("agent_type"), child.get("model"))
        group = groups.setdefault(key, {
            "agent_type": key[0], "model": key[1], "count": 0, "reads": None,
            "searches": None, "lines": 0, "did": None, "_purposes": [],
        })
        group["count"] += 1
        group["lines"] += owned_lines.get(child["session_id"], 0)
        calls = child.get("tool_calls")
        if isinstance(calls, dict):
            # A group whose agents never reached the activity ledger has an
            # unknown tool mix, not an empty one.
            for field, name in (("reads", "read"), ("searches", "search")):
                counted = calls.get(name)
                group[field] = (group[field] or 0) + (counted if isinstance(counted, int) else 0)
        purpose = (sessions.get(child["session_id"]) or {}).get("summary")
        if purpose:
            group["_purposes"].append(purpose)
    for group in groups.values():
        # Subagents of one group were commonly launched for one reason, and that
        # reason then names the whole group. A few distinct purposes are listed
        # instead; more than a few say nothing a reader can hold.
        distinct = list(dict.fromkeys(group.pop("_purposes")))
        group["did"] = distinct if 1 <= len(distinct) <= 3 else None
    ordered = sorted(groups.values(), key=lambda group: (
        -group["count"], -group["lines"], group["agent_type"] or "", group["model"] or "",
    ))
    return ordered[:_MAX_NARRATIVE_GROUPS], max(len(ordered) - _MAX_NARRATIVE_GROUPS, 0)


def _narrative(
    workflow: dict[str, Any] | None,
    sessions: dict[str, dict[str, Any]],
    owned_lines: Counter[str],
    owned_files: Counter[tuple[str, str]],
    task_names: dict[str, str],
) -> tuple[list[dict[str, Any]], int]:
    """Return who did what in this PR, and how many agents did not fit.

    The tree of the workflow profile says which agents ran and what each one
    did; the blame of this PR says which lines each of them still owns. One
    entry joins the two for each agent a reader would name.
    """
    if not isinstance(workflow, dict):
        return [], 0
    files_by_session: dict[str, list[tuple[str, int]]] = {}
    for (owner, path), lines in owned_files.items():
        files_by_session.setdefault(owner, []).append((path, lines))
    entries: list[dict[str, Any]] = []
    for agent in workflow.get("agents") or ():
        session_id = agent["session_id"]
        session = sessions.get(session_id) or {}
        groups, groups_omitted = _subagent_groups(agent, sessions, owned_lines)
        lines = owned_lines.get(session_id, 0)
        prompts = agent.get("prompt_count")
        # An agent that owns no line, launched nobody, and was never prompted
        # says nothing about how this PR was built.
        if not lines and not groups and not prompts:
            continue
        counted = sorted(
            files_by_session.get(session_id, ()), key=lambda item: (-item[1], item[0])
        )
        task_id = session.get("task_id")
        entries.append({
            "label": _agent_label(agent),
            "role": agent.get("role"),
            "prompt_count": prompts,
            "lines": lines,
            "files": [
                {"path": path, "lines": count}
                for path, count in counted[:_MAX_NARRATIVE_FILES]
            ],
            "files_omitted": max(len(counted) - _MAX_NARRATIVE_FILES, 0),
            "did": (task_names.get(task_id) if task_id else None) or session.get("feature"),
            "subagents": groups,
            "subagents_omitted": groups_omitted,
            "_started_at": session.get("started_at") or "",
            "_session_id": session_id,
        })
    entries.sort(key=lambda entry: (
        -entry["lines"], entry["_started_at"], entry["_session_id"],
    ))
    for entry in entries:
        del entry["_started_at"], entry["_session_id"]
    return entries[:_MAX_NARRATIVE_AGENTS], max(len(entries) - _MAX_NARRATIVE_AGENTS, 0)


def _display_harness(value: Any) -> Any:
    """Return the public name of a known harness, or the recorded label.

    A note carries the harness ID the session recorded, so a report built from
    notes alone would print ``claude-code`` where a local report prints
    ``Claude Code``. An ID this registry version does not know yet keeps the
    label it was recorded with.
    """
    try:
        return harness_display_name(value)
    except ValueError:
        return value


def _amount(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) and number >= 0 else None


def _allocated(session: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the telemetry a local run allocated to one session, or nothing."""
    value = session.get("telemetry")
    return value if isinstance(value, Mapping) else {}


def _usage_tokens(usage: Mapping[str, Any]) -> int | None:
    """Return the tokens a session's allocated telemetry counted, or None.

    An unknown count is never reported as a zero, so a record that counted no
    token leaves the session's count unknown rather than claiming it used none.
    """
    counted = _amount(usage.get("total_tokens"))
    if counted is None or counted <= 0:
        return None
    return int(round(counted))


def build_pr_report(
    repo: str | Path, base_ref: str = "main", head_ref: str = "HEAD", *,
    notes: list[dict[str, Any]] | None = None,
    tasks: list[dict[str, Any]] | None = None,
    usage: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Read only committed text and note evidence; never infer manual work from gaps.

    ``usage`` is the telemetry a caller allocated to each session it recorded.
    It names no line and changes no cost total; it only labels a session whose
    harness never reported a model. A caller with no local spool passes none,
    and the footer renders exactly as it did.
    """
    root = Path(repo).resolve()
    base, head, merge_base, commits = _scope(root, base_ref, head_ref)
    warnings = _Warnings()
    history = set(_run_git(root, "rev-list", head).stdout.splitlines())
    if notes is None:
        loaded = _load_notes(root, warnings)
    else:
        if not isinstance(notes, list):
            raise ValueError("Shared attribution notes must be an array")
        loaded = []
        for note in notes:
            if not isinstance(note, dict) or not isinstance(note.get("commit"), str):
                raise ValueError("Invalid shared attribution note")
            encoded = json.dumps(note)
            if len(encoded.encode()) > 2 * 1024 * 1024:
                raise ValueError("Shared attribution note exceeds the size limit")
            loaded.append(_parse_note(encoded, note["commit"], warnings=warnings))
    by_commit: dict[str, dict[str, Any]] = {}
    sessions: dict[str, dict[str, Any]] = {}
    for note in loaded:
        if note["commit"] not in history:
            continue
        if note["commit"] in by_commit:
            raise ValueError("Duplicate attribution note for a commit")
        by_commit[note["commit"]] = note
        for raw in note["sessions"]:
            if not isinstance(raw, dict):
                warnings.add("Ignored a malformed session.")
                continue
            session = _normalise_session(raw, source="Git notes", warnings=warnings)
            if session:
                _merge_session(sessions, session, warnings)

    ranges = {
        (sha, file["path"]): file["ranges"]
        for sha, note in by_commit.items() for file in note["files"]
    }
    participants: set[str] = set()
    membership_complete = True
    for sha in commits:
        note = by_commit.get(sha)
        if note is None:
            continue
        explicit = note.get("contributing_session_ids")
        if explicit is not None:
            participants.update(explicit)
        else:
            # Older notes mix real attempts with source-only historical context.
            # Use evidenced current participants; do not charge historical work.
            identified = {r["session_id"] for file in note["files"] for r in file["ranges"]}
            identified.update(r["to_session_id"] for r in note["revisions"] if r["to_session_id"])
            participants.update(identified)
            if {s.get("id") for s in note["sessions"] if isinstance(s, dict)} - identified:
                membership_complete = False
                warnings.add("Some legacy session costs cannot be scoped to this PR.")
    if participants - sessions.keys():
        membership_complete = False
        warnings.add("Some contributing sessions have missing metadata.")

    economic_participants = set(participants)
    task_names: dict[str, str] = {}
    if tasks is not None:
        latest_tasks: dict[str, dict[str, Any]] = {}
        for record in _task_records(root, tasks, warnings):
            if record["anchor_commit"] not in history:
                continue
            previous = latest_tasks.get(record["id"])
            previous_time = _parse_time(previous.get("updated_at")) if previous else None
            record_time = _parse_time(record.get("updated_at"))
            if previous is None or (
                record_time is not None
                and (previous_time is None or record_time >= previous_time)
            ):
                latest_tasks[record["id"]] = record
        # The name of a task is what a top-level agent was working on, which the
        # footer prints beside the agent that worked it.
        task_names = {record["id"]: record["name"] for record in latest_tasks.values()}

        participant_task_ids = {
            str(sessions[session_id]["task_id"])
            for session_id in participants & sessions.keys()
            if sessions[session_id].get("task_id")
        }
        pr_commits = set(commits)
        relevant_task_ids = participant_task_ids or set(
            record["id"] for record in latest_tasks.values()
            if record["anchor_commit"] in pr_commits
        )
        for task_id in sorted(relevant_task_ids):
            record = latest_tasks.get(task_id)
            if record is None:
                membership_complete = False
                warnings.add("Some task-wide session economics are unavailable.")
                continue
            for raw in record["sessions"]:
                if not isinstance(raw, dict):
                    membership_complete = False
                    warnings.add(f"Task {task_id!r} contains a malformed session.")
                    continue
                session = _normalise_session(
                    raw, source="task economics metadata", warnings=warnings
                )
                if session is None or session.get("task_id") != task_id:
                    membership_complete = False
                    warnings.add(f"Task {task_id!r} contains invalid session membership.")
                    continue
                _merge_session(sessions, session, warnings)
                economic_participants.add(session["id"])

    # A note or a task record says what the harness reported, which for a
    # headless hook session is an ``unknown`` model, no tokens, and no cost.
    # The local spool of the same run named all three, so a local run fills
    # what those records left unknown here, once every session of this PR is
    # known and before any source row, cost total, workflow profile, or
    # narrative bullet reads it. What a record did report is never replaced.
    for session_id, session in sessions.items():
        allocated = (usage or {}).get(session_id)
        named = telemetry_model(session, allocated)
        if named is not None:
            session["model"] = named
            session["model_source"] = "telemetry"
        if not isinstance(allocated, Mapping):
            continue
        if session.get("cost_usd") is None and (
            _amount(allocated.get("estimated_cost_usd")) is not None
            or _amount(allocated.get("codex_credits")) is not None
        ):
            # ``_task_economics`` reads this exactly as the local report does:
            # a session with no reported cost counts by its estimate or its
            # Codex credits, and an incomplete price marks the total partial.
            session["telemetry"] = allocated
        if session.get("token_count") is None:
            counted = _usage_tokens(allocated)
            if counted is not None:
                session["token_count"] = counted
                session["token_source"] = "telemetry"

    # The footer, the artifact, and the hosted page all read this one field,
    # so a known harness is named here, once, after every session of this PR is
    # known and before any row, profile, or narrative label reads it.
    for session in sessions.values():
        session["harness"] = _display_harness(session["harness"])

    positions, complete = _added_positions(root, merge_base, head, warnings)
    counts = {"ai": 0, "manual": 0, "unknown": 0}
    owned_lines: Counter[str] = Counter()
    # Which file each session still owns lines in, so the footer can name the
    # files an agent wrote rather than only how many lines it wrote.
    owned_files: Counter[tuple[str, str]] = Counter()
    for path, line_numbers in positions.items():
        try:
            blame = _blame(root, head, path)
        except (ValueError, OSError):
            counts["unknown"] += len(line_numbers)
            warnings.add("Some added lines could not be traced to capture evidence.")
            continue
        for line in line_numbers:
            origin = blame.get(line - 1)
            owner = None
            if origin:
                sha, original_path, original_line = origin
                owner = next((r["session_id"] for r in ranges.get((sha, original_path), [])
                              if r["start"] <= original_line <= r["end"]), None)
            session = sessions.get(owner)
            if session is None:
                counts["unknown"] += 1
            else:
                counts[session["actor_kind"]] += 1
                owned_lines[owner] += 1
                owned_files[(owner, path)] += 1

    ai_sessions = [
        sessions[session_id]
        for session_id in economic_participants & sessions.keys()
        if sessions[session_id]["actor_kind"] == "ai"
    ]
    economics = _task_economics(ai_sessions)
    grouped: dict[tuple[str, str, str], dict[str, Any]] = {}
    for session_id in sorted((economic_participants & sessions.keys()) | owned_lines.keys()):
        session = sessions[session_id]
        key = (("manual", "Manual", "") if session["actor_kind"] == "manual" else
               (session["actor_kind"], session["model"], session["harness"]))
        group = grouped.setdefault(key, {
            "actor_kind": key[0], "model": key[1], "harness": key[2], "lines": 0,
            "session_count": 0, "_sessions": set(), "_costs": [], "_tokens": [],
            "_estimates": [], "_credits": [], "_equivalents": [],
            "_partial_costs": 0, "_unknown_costs": 0,
            "_unknown_tokens": 0, "_covered_costs": 0, "_covered_tokens": 0,
        })
        group["lines"] += owned_lines[session_id]
        if session_id in economic_participants:
            group["_sessions"].add(session.get("source_session_id") or session_id)
            if session["actor_kind"] == "ai":
                if session.get("cost_in_total"):
                    # A cost this run filled from telemetry is an estimate the
                    # provider priced, so it is summed apart from the cost a
                    # session reported and printed as the estimate it is.
                    estimated = _amount(
                        _allocated(session).get("estimated_cost_usd")
                    )
                    if session["cost_usd"] is not None:
                        group["_costs"].append(session["cost_usd"])
                    elif estimated is not None:
                        group["_estimates"].append(estimated)
                        if _allocated(session).get("cost_complete") is False:
                            # The same rule the local report applies: an
                            # estimate that priced only some of the requests
                            # of a session is a partial cost, not a whole one.
                            group["_partial_costs"] += 1
                    elif _amount(_allocated(session).get("codex_credits")) is not None:
                        # A Codex subscription session is priced in credits, so
                        # it counts here in its own unit rather than reading as
                        # an unknown dollar cost that marks the total partial.
                        group["_credits"].append(
                            _amount(_allocated(session).get("codex_credits"))
                        )
                        equivalent = _amount(
                            _allocated(session).get("codex_api_equivalent_usd")
                        )
                        if equivalent is not None:
                            group["_equivalents"].append(equivalent)
                        if _allocated(session).get("cost_complete") is False:
                            group["_partial_costs"] += 1
                    else:
                        group["_unknown_costs"] += 1
                elif session.get("cost_covered_by_parent"):
                    group["_covered_costs"] += 1
                elif session["cost_usd"] is None and not session.get("cost_covered_by_parent"):
                    group["_unknown_costs"] += 1
                if session.get("tokens_in_total"):
                    group["_tokens"].append(session["token_count"])
                elif session.get("tokens_covered_by_parent"):
                    group["_covered_tokens"] += 1
                elif session["token_count"] is None and not session.get("tokens_covered_by_parent"):
                    group["_unknown_tokens"] += 1
    sources = []
    for key, group in sorted(grouped.items()):
        group["session_count"] = len(group.pop("_sessions"))
        costs = group.pop("_costs")
        estimates = group.pop("_estimates")
        credits = group.pop("_credits")
        equivalents = group.pop("_equivalents")
        partial_costs = group.pop("_partial_costs")
        tokens = group.pop("_tokens")
        unknown_costs = group.pop("_unknown_costs")
        unknown_tokens = group.pop("_unknown_tokens")
        covered_costs = group.pop("_covered_costs")
        covered_tokens = group.pop("_covered_tokens")
        group["reported_cost_usd"] = round(math.fsum(costs), 10) if costs else None
        # The estimate, the credits, and their dollar comparison are carried
        # only where this run filled them, so a report built from committed
        # evidence alone keeps the exact shape it had.
        if estimates:
            group["estimated_cost_usd"] = round(math.fsum(estimates), 10)
        if credits:
            group["codex_credits"] = round(math.fsum(credits), 10)
        if equivalents:
            group["codex_api_equivalent_usd"] = round(math.fsum(equivalents), 10)
        group["cost_complete"] = (
            not unknown_costs and not partial_costs and membership_complete
        )
        group["cost_in_parent"] = bool(
            covered_costs and not costs and not estimates
            and not credits and not unknown_costs
        )
        group["total_tokens"] = sum(tokens) if tokens else None
        group["tokens_complete"] = not unknown_tokens and membership_complete
        group["tokens_in_parent"] = bool(covered_tokens and not tokens and not unknown_tokens)
        sources.append(group)
    logical_sessions = {
        (sessions[s]["harness"], sessions[s].get("source_session_id") or s)
        for s in economic_participants & sessions.keys()
    }
    logical_ai_sessions = {(s["harness"], s.get("source_session_id") or s["id"]) for s in ai_sessions}
    # A PR report reads committed evidence only, so the workflow profile comes
    # from the notes of this branch and never from a local ledger.
    note_activity, note_instruction_files = _note_workflow(loaded)
    # A subagent that owns no line of this PR is not a contributing session, so
    # no cost total charges it and no source row counts it. It still worked the
    # task, and the workflow object of each note names it, so the profile
    # describes it beside the agents that own the lines.
    profile_participants = set(economic_participants)
    for sha in commits:
        note = by_commit.get(sha)
        if note is None:
            continue
        profile_participants.update(
            agent["session_id"] for agent in iter_agents(note.get("workflow"))
        )
    profile_sessions = [
        sessions[session_id]
        for session_id in sorted(profile_participants & sessions.keys())
        if sessions[session_id]["actor_kind"] == "ai"
    ]
    # One native conversation can have multiple ledger segments. Join their
    # descriptive evidence without merging costs or rewriting stored notes.
    identities: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for session in profile_sessions:
        identity = (native_identity_hash(session) or session.get("workflow_identity")
                    or session.get("source_session_id") or session["id"])
        identities.setdefault((session.get("harness_id") or session["harness"], identity), []).append(session)
    aliases = {}
    combined = []
    narrative_sessions = dict(sessions)
    narrative_lines = Counter(owned_lines)
    narrative_files = Counter(owned_files)
    for members in identities.values():
        representative = dict(members[0])
        identifier = representative["id"]
        for member in members:
            aliases[member["id"]] = identifier
            if member["id"] != identifier:
                # Notes of the same ledger row are already merged by maximum.
                # Distinct ledger segments contain separate events, so add
                # their activity instead of discarding the smaller segment.
                target_activity = dict(note_activity.get(identifier, {}))
                for category in ("counts", "tool_calls"):
                    incoming = note_activity.get(member["id"], {}).get(category)
                    if isinstance(incoming, dict):
                        counters = dict(target_activity.get(category) or {})
                        for name, count in incoming.items():
                            counters[name] = counters.get(name, 0) + count
                        target_activity[category] = counters
                note_activity[identifier] = target_activity
                note_instruction_files.setdefault(identifier, {}).update(
                    note_instruction_files.get(member["id"], {})
                )
        for field in ("prompt_count", "turn_count", "tool_call_count", "interrupt_count", "compaction_count"):
            values = [member.get(field) for member in members if isinstance(member.get(field), int)]
            if values:
                representative[field] = sum(values)
        for field in ("model", "agent_type", "summary", "feature", "role",
                      "parent_session_id", "session_source"):
            values = list(dict.fromkeys(
                member.get(field) for member in members
                if member.get(field) not in (None, "", "unknown")
            ))
            representative[field] = values[0] if len(values) == 1 else (
                " / ".join(values) if field == "model" and values else
                "unknown" if field == "model" else None
            )
        for member in members[1:]:
            old = member["id"]
            narrative_lines[identifier] += narrative_lines.pop(old, 0)
            for (owner, path), count in list(narrative_files.items()):
                if owner == old:
                    narrative_files[(identifier, path)] += count
                    del narrative_files[(owner, path)]
        narrative_sessions[identifier] = representative
        combined.append(representative)
    for session in combined:
        session["parent_session_id"] = aliases.get(session.get("parent_session_id"), session.get("parent_session_id"))
    # Legacy stop-only placeholders have no launch/model/purpose or work
    # evidence. Keep them inspectable, but do not present them as confirmed
    # participants. Unknown models alone are never grounds for exclusion.
    parents = {session.get("parent_session_id") for session in combined}
    unconfirmed_agents = []
    profile_sessions = []
    for session in combined:
        identifier = session["id"]
        recorded = note_activity.get(identifier) or {}
        has_activity = any(
            isinstance(value, (int, float)) and value > 0
            for category in ("counts", "tool_calls")
            for value in (recorded.get(category) or {}).values()
        ) or any((session.get(field) or 0) > 0 for field in
                 ("prompt_count", "turn_count", "tool_call_count"))
        unconfirmed = (
            session.get("parent_session_id") is not None
            and session.get("session_source") == "subagent"
            and session.get("ended_at") is not None
            and session.get("model") in (None, "", "unknown")
            and not session.get("agent_type") and not session.get("summary")
            and not session.get("launch_mode")
            and not has_activity and not narrative_lines.get(identifier)
            and identifier not in parents
        )
        if unconfirmed:
            unconfirmed_agents.append({
                "session_id": identifier,
                "reason": "No launch, model, purpose, or work evidence in the report",
            })
        else:
            profile_sessions.append(session)
    workflow = _workflow_profile(
        profile_sessions,
        {},
        {},
        {},
        note_activity,
        note_instruction_files,
    )
    narrative, narrative_omitted = _narrative(
        workflow, narrative_sessions, narrative_lines, narrative_files, task_names
    )
    report = {
        "workflow": workflow,
        "unconfirmed_agents": unconfirmed_agents,
        "narrative": narrative, "narrative_omitted": narrative_omitted,
        "base_commit": base, "merge_base": merge_base, "head_commit": head,
        "added_lines": sum(counts.values()), "counts": counts, "sources": sources,
        "session_count": len(logical_sessions),
        "ai_session_count": len(logical_ai_sessions),
        "reported_cost_usd": economics["reported_cost_usd"],
        "cost_complete": membership_complete and economics["cost_complete"],
        "total_tokens": economics["total_tokens"],
        "tokens_complete": membership_complete and economics["tokens_complete"],
        "complete": complete, "warnings": warnings.items,
    }
    # As with each source row, the key exists only where a local run filled a
    # cost the notes did not carry.
    if economics["estimated_cost_usd"] is not None:
        report["estimated_cost_usd"] = economics["estimated_cost_usd"]
    # Credits and their dollar comparison ride beside the dollar estimate, and
    # only where a Codex subscription session was priced.
    if economics["codex_credits"] is not None:
        report["codex_credits"] = economics["codex_credits"]
    if economics["codex_api_equivalent_usd"] is not None:
        report["codex_api_equivalent_usd"] = economics["codex_api_equivalent_usd"]
    return report


def _local_task_records(root: Path, warnings: _Warnings) -> list[dict[str, Any]]:
    """Return this repository's task notes in the shape shared metadata carries.

    The trusted workflow reads task metadata out of the pushed snapshot, which
    anchors each task to the commit its note annotates. A local run reads the
    same records from the task notes ref, so both paths name the task each
    top-level agent worked on.
    """
    return [
        {
            **entry["task"],
            "anchor_commit": entry["commit"],
            "sessions": entry["sessions"],
        }
        for entry in _load_task_notes(root, warnings)
    ]


def _local_usage(root: Path, warnings: _Warnings) -> dict[str, dict[str, Any]]:
    """Return the telemetry this clone allocated to each session it recorded.

    A note carries the model its session recorded, and a headless session whose
    harness reported none recorded ``unknown``. The local spool of that same
    run still names the model each request used, so a local run can label the
    agent that a clone reading notes alone cannot. The trusted workflow reads
    no ledger and passes nothing, exactly as it reads no task note ref.
    """
    try:
        common_dir = git_common_dir(root)
        return allocate_session_usage(
            common_dir, _load_local_sessions(common_dir, warnings)
        )
    except (OSError, ValueError, sqlite3.Error) as exc:
        warnings.add(f"Could not read local cost telemetry: {exc}.")
        return {}


def prepare_pr_report(repo: str | Path, base_ref: str = "main", head_ref: str = "HEAD") -> dict[str, Any]:
    """Record this worktree's matching evidence before building the committed report."""
    root = Path(repo).resolve()
    base, head, _, commits = _scope(root, base_ref, head_ref)
    for sha in commits:
        record_commit(root, sha)
    warnings = _Warnings()
    tasks = _local_task_records(root, warnings)
    # A repository with no task note keeps the report it always built: passing
    # an empty list would ask for task-wide economics that no note can answer.
    report = build_pr_report(
        root, base, head, tasks=tasks or None, usage=_local_usage(root, warnings)
    )
    for message in warnings.items:
        if message not in report["warnings"]:
            report["warnings"].append(message)
    return report


def _label(value: str) -> str:
    # Labels are supplied by users. Prevent markup, mentions, links and footer
    # marker injection without exporting raw paths or command arguments.
    text = " ".join(str(value).split())[:120]
    escaped = html.escape(text, quote=True)
    for character in "|`*_[]()!@\\:":
        escaped = escaped.replace(character, f"&#{ord(character)};")
    return escaped


def _agent_label(agent: dict[str, Any]) -> str:
    """Name one agent by the harness it ran in and the model that served it."""
    named = [_label(value) for value in (agent.get("harness"), agent.get("model")) if value]
    return " / ".join(named) or "Unknown agent"


def _plural(value: int, singular: str, plural: str) -> str:
    return f"{value:,} {singular if value == 1 else plural}"


def _money(value: float | None, complete: bool) -> str:
    if value is None:
        return "Unknown"
    if value == 0:
        # A negative zero from an older record would print as "$-0.00".
        value = 0.0
    return f"${value:.2f}" + (" (partial)" if not complete else "")


def _api_equivalent(value: float) -> str:
    """Show the standard-rate dollar comparison of a credit-priced session."""
    if 0 < value < 0.01:
        return "<$0.01"
    return f"${value:,.2f}"


def _credit_text(value: float) -> str:
    """Name an amount of Codex credits, never as a dollar figure."""
    if value == 0:
        return "0 Codex credits"
    if abs(value) < 0.01:
        return f"{value:.4f} Codex credits"
    return f"{value:,.2f} Codex credits"


def _credit_cost(
    reported: float | None,
    estimated: float | None,
    credits: float,
    equivalent: float | None,
    complete: bool,
) -> str:
    """Render a cost a Codex subscription priced in credits, never as dollars.

    A session that also holds a reported or estimated dollar cost shows both
    units side by side, so no reader reads one number as the sum of the other.
    """
    if reported is not None or estimated is not None:
        usd = (reported or 0.0) + (estimated or 0.0)
        text = f"${usd:.2f} USD + {_credit_text(credits)}"
    else:
        text = _credit_text(credits)
        if equivalent is not None:
            text += f" ≈ {_api_equivalent(equivalent)}"
    return text if complete else f"{text} (partial)"


def _estimated_money(
    reported: float | None, estimated: float, complete: bool, suffix: str = ""
) -> str:
    """Return a cost that holds an estimate, saying that it holds one.

    A session records the cost its harness reported. A local run also fills the
    cost of a session that reported none from this clone's provider telemetry,
    and Claude Code prices each request itself rather than billing it, so a
    number that carries one of those prices is never printed as a reported
    cost. A report with no estimate never reaches this function.
    """
    basis = "reported + estimated" if reported is not None else "estimated"
    text = f"${(reported or 0.0) + estimated:.2f} {basis}{suffix}"
    return text if complete else f"{text} (partial)"


def fallback_narrative(report: dict[str, Any]) -> str | None:
    """Describe the bounded narrative facts without making another claim.

    The trusted workflow can replace this deliberately simple prose with a
    small-model summary. Keeping a local version gives the hosted page a
    paragraph when that optional request is disabled or unavailable. It stays
    beside the report it reads, which this module builds.
    """
    narrative = report.get("narrative")
    if not isinstance(report.get("workflow"), dict) or not narrative:
        return None
    entries = [entry for entry in narrative if isinstance(entry, dict)]
    if not entries:
        return None
    totals = report["workflow"].get("totals")
    totals = totals if isinstance(totals, dict) else {}
    agents = totals.get("agents")
    if not isinstance(agents, int) or isinstance(agents, bool) or agents < 1:
        agents = len(entries)
    sentences = [
        f"The user used {_plural(agents, 'coding agent', 'coding agents')} "
        "to build this change."
    ]

    clauses = []
    for entry in entries[:3]:
        # Narrative labels were escaped when the public report was built.
        # Decode them before escaping the finished paragraph once at the edge.
        label = html.unescape(str(entry.get("label") or "An agent"))
        task = entry.get("did")
        role = entry.get("role")
        if isinstance(task, str) and task.strip():
            clause = f"{label} worked on {task.strip()}"
        else:
            verb = {
                "planning": "planned", "implementation": "implemented",
                "testing": "tested", "review": "reviewed",
            }.get(role, "worked on")
            clause = f"{label} {verb} the change"
        groups = entry.get("subagents")
        if isinstance(groups, list):
            children = sum(
                group.get("count", 0) for group in groups
                if isinstance(group, dict)
                and isinstance(group.get("count"), int)
                and not isinstance(group.get("count"), bool)
                and group["count"] > 0
            )
            if children:
                clause += " with support from " + _plural(
                    children, "subagent", "subagents"
                )
        clauses.append(clause)
    if clauses:
        joined = clauses[0] if len(clauses) == 1 else (
            ", ".join(clauses[:-1]) + ", and " + clauses[-1]
        )
        if len(entries) > len(clauses):
            more_agents = len(entries) - len(clauses)
            joined += "; " + _plural(
                more_agents, "more top-level agent", "more top-level agents"
            ) + " contributed"
        sentences.append(joined + ".")

    files: dict[str, int] = {}
    omitted = 0
    for entry in entries:
        listed = entry.get("files")
        if isinstance(listed, list):
            for item in listed:
                if isinstance(item, dict) and isinstance(item.get("path"), str):
                    lines = item.get("lines")
                    files[item["path"]] = max(
                        files.get(item["path"], 0),
                        lines if isinstance(lines, int) and not isinstance(lines, bool) else 0,
                    )
        count = entry.get("files_omitted")
        if isinstance(count, int) and not isinstance(count, bool) and count > 0:
            omitted += count
    if files:
        ordered = sorted(files, key=lambda path: (-files[path], path))
        shown = ordered[:3]
        if len(shown) == 1:
            named = shown[0]
        else:
            named = ", ".join(shown[:-1]) + " and " + shown[-1]
        more = len(ordered) - len(shown) + omitted
        suffix = f" and {_plural(more, 'more file', 'more files')}" if more else ""
        sentences.append(f"The final work spans {named}{suffix}.")
    return " ".join(sentences)


def _source_label(source: dict[str, Any]) -> str:
    """Name one coding agent by the harness it ran in and the model it used."""
    named = [_label(value) for value in (source.get("harness"), source.get("model")) if value]
    return " · ".join(named) or "Unknown agent"


def _sessions_cell(count: int) -> str:
    """Show how many sessions one row counted, or a dash for none.

    A harness that reported two models inside one session leaves a row with no
    session of its own, and a zero there reads as a measured nothing rather
    than as the absence of a measurement.
    """
    return str(count) if count > 0 else "—"


def _cost_cell(entry: dict[str, Any]) -> str:
    """Render the cost of one agent row, or of the report that holds them.

    A source row and the report itself carry the same cost fields, so one
    reader serves both and the Total row cannot drift from the rows above it.
    """
    if entry.get("cost_in_parent"):
        return "Included in parent"
    credits = _amount(entry.get("codex_credits"))
    if credits is not None:
        return _credit_cost(
            _amount(entry.get("reported_cost_usd")),
            _amount(entry.get("estimated_cost_usd")),
            credits,
            _amount(entry.get("codex_api_equivalent_usd")),
            entry["cost_complete"],
        )
    estimated = entry.get("estimated_cost_usd")
    if estimated is None:
        return _money(entry["reported_cost_usd"], entry["cost_complete"])
    return _estimated_money(
        entry["reported_cost_usd"], estimated, entry["cost_complete"]
    )


def _total_cost_phrase(report: dict[str, Any]) -> str:
    """Say what this pull request cost, and on which basis that number stands."""
    reported = report.get("reported_cost_usd")
    estimated = report.get("estimated_cost_usd")
    credits = _amount(report.get("codex_credits"))
    partial = "" if report["cost_complete"] else " (partial)"
    if reported is None and estimated is None and credits is None:
        return "Unknown total reported cost"
    if reported is not None or estimated is not None:
        if estimated is None:
            total, basis = reported, "reported"
        else:
            total = (reported or 0.0) + estimated
            basis = "reported + estimated" if reported is not None else "estimated"
        if total == 0:
            total = 0.0
        phrase = f"${total:.2f} total {basis} cost"
        # Credits ride beside the dollars in their own unit, never added in.
        if credits is not None:
            phrase += f" + {_credit_text(credits)}"
        return f"{phrase}{partial}"
    equivalent = _amount(report.get("codex_api_equivalent_usd"))
    phrase = f"{_credit_text(credits)} total"
    if equivalent is not None:
        phrase += f" ≈ {_api_equivalent(equivalent)}"
    return f"{phrase}{partial}"


def render_footer(report: dict[str, Any], *, details_url: str | None = None) -> str:
    """Render the managed footer: which coding agents ran, and what they cost.

    Everything else the report measures stays on the hosted page. The footer
    reads the sources and the cost fields only, so it is byte-stable for
    identical input and unchanged by evidence it does not print.
    """
    agents = [source for source in report["sources"] if source["actor_kind"] == "ai"]
    rows = [START, "---", "**Joyride:** " + (
        f"{_plural(len(agents), 'coding agent', 'coding agents')}"
        f" · {_total_cost_phrase(report)}"
        if agents else "No coding agent recorded"
    )]
    if agents:
        rows.extend([
            "", "| Coding agent | Sessions | Cost |", "| --- | ---: | ---: |",
        ])
        # The sources are sorted where they are built; keep that order.
        rows.extend(
            f"| {_source_label(source)} | {_sessions_cell(source['session_count'])}"
            f" | {_cost_cell(source)} |"
            for source in agents
        )
        # One harness session that reported two models fills two rows, so the
        # total adds up the rows a reader can see instead of the count of the
        # distinct sessions behind them.
        counted = sum(
            source["session_count"] for source in agents if source["session_count"] > 0
        )
        rows.extend([
            f"| Total | {_sessions_cell(counted)} | {_cost_cell(report)} |", "",
        ])
    if details_url:
        if not agents:
            rows.append("")
        rows.append(
            f"[See how agents built this and optimization notes →]({details_url})"
        )
    rows.append(END)
    return "\n".join(rows)


def update_pr_body(body: str | None, footer: str) -> str:
    """Replace our one marked block byte-for-byte, preserving all outside prose."""
    body = body or ""
    starts, ends = body.count(START), body.count(END)
    if not starts and not ends:
        if not footer:
            return body
        return body + ("\n\n" if body else "") + footer
    if starts != 1 or ends != 1 or body.index(START) >= body.index(END):
        raise ValueError("The PR description contains malformed attribution markers")
    start, end = body.index(START), body.index(END) + len(END)
    return body[:start] + footer + body[end:]
