"""Command-line interface for local coding attribution."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, TextIO
from urllib.parse import urlencode

from . import __version__


_HOOK_INPUT_LIMIT = 2 * 1024 * 1024
_PUBLIC_COMMANDS = (
    "{install,uninstall,report,show,serve,code,why,status,run,task,session,hook,"
    "recover,harnesses,hook-template,record,hosted,doctor,demo,help}"
)


class _ArgumentParser(argparse.ArgumentParser):
    """Keep private hook entry points out of user-facing parse errors."""

    def error(self, message: str) -> None:
        for hidden in (
            ", '_hook'",
            ", '_git-hook'",
            ", '_collector'",
            ", '_share'",
            ", _hook",
            ", _git-hook",
            ", _collector",
            ", _share",
        ):
            message = message.replace(hidden, "")
        super().error(message)


def _cost(value: str) -> float:
    try:
        cost = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Cost must be a number in USD.") from exc
    if not math.isfinite(cost) or cost < 0:
        raise argparse.ArgumentTypeError("Cost must be a finite, nonnegative number in USD.")
    return cost


def _repo_option(command: argparse.ArgumentParser) -> None:
    command.add_argument(
        "--repo",
        dest="command_repo",
        type=Path,
        help="Git repository (default: current directory)",
    )


def _line(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Line must be an integer.") from exc
    if number < 1:
        raise argparse.ArgumentTypeError("Line must be 1 or more.")
    return number


def _port(value: str) -> int:
    try:
        port = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Port must be an integer.") from exc
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("Port must be between 1 and 65535.")
    return port


def _tokens(value: str) -> int:
    try:
        tokens = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Tokens must be a nonnegative integer.") from exc
    if tokens < 0:
        raise argparse.ArgumentTypeError("Tokens must be a nonnegative integer.")
    return tokens


def parser() -> argparse.ArgumentParser:
    result = _ArgumentParser(
        prog="joyride",
        description="See which coding sessions shaped a task, what they cost, and what lasted.",
        epilog="With no command, show the report. Use show <task> to inspect session and commit evidence.",
    )
    result.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    result.add_argument(
        "--repo",
        type=Path,
        default=Path.cwd(),
        help="Git repository (default: current directory)",
    )
    commands = result.add_subparsers(dest="action", metavar=_PUBLIC_COMMANDS)

    commands.add_parser("help", help="Show all available commands")

    install = commands.add_parser("install", help="Enable automatic tracking in a repository")
    install.add_argument(
        "folder", nargs="?", type=Path, help="Repository folder (default: current directory)"
    )
    install.add_argument(
        "--no-traces",
        action="store_true",
        help="Do not keep or publish the text of sessions in this repository",
    )
    install.add_argument(
        "--user",
        action="store_true",
        help="Install the harness hooks once for this machine",
    )
    _repo_option(install)

    uninstall = commands.add_parser(
        "uninstall", help="Disable tracking while preserving attribution data"
    )
    uninstall.add_argument(
        "folder", nargs="?", type=Path, help="Repository folder (default: current directory)"
    )
    uninstall.add_argument(
        "--user",
        action="store_true",
        help="Remove the machine-level harness hooks",
    )
    _repo_option(uninstall)

    hook = commands.add_parser("hook", help="Capture one native coding-agent hook event")
    hook.add_argument(
        "--feature",
        default=None,
        help="Feature name (or set ATTRIBUTION_FEATURE)",
    )
    hook.add_argument("--harness", required=True, help="Supported harness id or name")
    hook.add_argument("--harness-version", help="Harness version reported by the hook setup")
    hook.add_argument("--model", help="Model override when the payload omits it")
    hook.add_argument("--event", help="Event override when the payload omits it")
    hook.add_argument("--session-id", help="Session override when the payload omits it")
    hook.add_argument("--json", action="store_true", help="Print the capture result as JSON")
    hook.add_argument("--observer", action="store_true", help=argparse.SUPPRESS)
    _repo_option(hook)

    recover = commands.add_parser(
        "recover", help="Close unfinished native-hook captures"
    )
    recover.add_argument(
        "--capture-changes",
        action="store_true",
        help="Assign later changes when exactly one unfinished session exists",
    )
    _repo_option(recover)
    commands.add_parser("harnesses", help="List recognized coding harnesses")
    template = commands.add_parser(
        "hook-template", help="Print a repository hook configuration"
    )
    template.add_argument("--feature", required=True, help="Feature name for captured sessions")
    template.add_argument("--harness", required=True, help="Harness id or name")
    template.add_argument(
        "--joyride-command",
        dest="attribution_command",
        default=None,
        help=(
            "Installed joyride executable name or path "
            "(default: current standalone binary or joyride)"
        ),
    )

    report = commands.add_parser("report", help="Show the attribution summary")
    report.add_argument("--target", default="main", help="Target Git ref (default: main)")
    report.add_argument("--json", action="store_true", help="Print the raw report JSON")
    _repo_option(report)

    show = commands.add_parser("show", help="Show details for one feature")
    show.add_argument(
        "group", metavar="feature", help="Feature name (case-insensitive when unique)"
    )
    show.add_argument("--target", default="main", help="Target Git ref (default: main)")
    show.add_argument(
        "--json", action="store_true", help="Print the raw task detail JSON"
    )
    _repo_option(show)

    serve = commands.add_parser("serve", help="Open the read-only local dashboard")
    serve.add_argument("--target", default="main", help="Target Git ref (default: main)")
    serve.add_argument("--port", type=_port, default=8765, help="Local port (default: 8765)")
    _repo_option(serve)

    code = commands.add_parser(
        "code", help="Open code with model attribution and revision history"
    )
    code.add_argument(
        "path", nargs="?", help="Repository-relative file path (default: choose in the browser)"
    )
    code.add_argument("--target", default="main", help="Target Git ref (default: main)")
    code.add_argument("--port", type=_port, default=8765, help="Local port (default: 8765)")
    code.add_argument(
        "--no-open", action="store_true", help="Print the local URL without opening a browser"
    )
    code.add_argument(
        "--json",
        action="store_true",
        help="Print file attribution, or the file list, without starting a server",
    )
    _repo_option(code)

    why = commands.add_parser(
        "why", help="Show what a session loaded before it wrote a line"
    )
    why.add_argument(
        "path",
        metavar="FILE[:LINE]",
        help="Repository-relative file path, with an optional line number",
    )
    why.add_argument(
        "--line", type=_line, help="Line number, when the path does not carry one"
    )
    why.add_argument("--target", default="main", help="Target Git ref (default: main)")
    why.add_argument("--json", action="store_true", help="Print the raw why JSON")
    _repo_option(why)

    status = commands.add_parser("status", help="Show installation and capture status")
    status.add_argument("--json", action="store_true", help="Print the raw status JSON")
    _repo_option(status)

    doctor = commands.add_parser(
        "doctor", help="Check hosted GitHub and local installation readiness"
    )
    doctor.add_argument("--json", action="store_true", help="Print the raw readiness JSON")
    doctor.add_argument(
        "--hosted-url",
        help="Hosted HTTPS origin to check for repository variables",
    )
    doctor.add_argument(
        "--no-network",
        action="store_true",
        help="Skip the GitHub CLI authentication and variable checks",
    )
    _repo_option(doctor)

    hosted = commands.add_parser(
        "hosted", help="Set up the hosted GitHub integration"
    )
    hosted_commands = hosted.add_subparsers(dest="hosted_action", required=True)
    hosted_setup = hosted_commands.add_parser(
        "setup", help="Create a reviewable hosted setup pull request"
    )
    hosted_setup.add_argument(
        "--hosted-url",
        help="Hosted HTTPS origin (also sets ATTRIBUTION_UI_URL and ingest variables)",
    )
    hosted_setup.add_argument(
        "--branch",
        default="attribution/setup",
        help="Setup branch name (default: attribution/setup)",
    )
    hosted_setup.add_argument(
        "--base",
        dest="base_branch",
        help="Target branch (default: detected GitHub default branch)",
    )
    hosted_setup.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan files without authenticating, pushing, or creating a PR",
    )
    hosted_setup.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not start GitHub's browser authentication flow",
    )
    hosted_setup.add_argument(
        "--no-open",
        action="store_true",
        help="Do not open the created pull request in a browser",
    )
    hosted_setup.add_argument("--json", action="store_true", help="Print the setup result as JSON")
    _repo_option(hosted_setup)

    run = commands.add_parser("run", help="Capture one coding session and infer its task")
    task_choice = run.add_mutually_exclusive_group()
    task_choice.add_argument("--feature", help="Legacy feature label; creates or selects that task")
    task_choice.add_argument("--task", help="Explicit task name or ID instead of inference")
    run.add_argument("--model", help="Model name or version (detected from the command when omitted)")
    run.add_argument("--harness", help="Coding tool or workflow (detected from the command when omitted)")
    run.add_argument("--manual", action="store_true", help="Declare this session as manual editing")
    run.add_argument(
        "--role",
        choices=("planning", "implementation", "testing", "review", "other"),
        default="implementation",
        help="How this session shaped the task",
    )
    run.add_argument("--summary", help="Short description of the session's work")
    run.add_argument("--parent-session", help="Parent session ID for subagent work")
    run.add_argument("--tokens", type=_tokens, default=None, help="Reported token count")
    run.add_argument(
        "--cost-usd",
        type=_cost,
        default=None,
        help="Reported session cost in USD (unknown if omitted)",
    )
    run.add_argument(
        "--usage-includes-children",
        action="store_true",
        help="Reported tokens and cost already include descendant sessions",
    )
    run.add_argument("--json", action="store_true", help="Print the raw capture JSON")
    _repo_option(run)
    run.add_argument(
        "command", nargs=argparse.REMAINDER, help="Command after --, for example: -- claude"
    )

    task = commands.add_parser("task", help="Inspect or correct inferred task groups")
    task_commands = task.add_subparsers(dest="task_action", required=True)
    task_start = task_commands.add_parser("start", help="Create and select an explicit task")
    task_start.add_argument("name")
    task_start.add_argument("--pr", dest="pr_ref", help="Pull-request number or reference")
    task_start.add_argument("--pr-url", help="Pull-request URL")
    task_commands.add_parser("list", help="List local inferred and explicit tasks")
    task_use = task_commands.add_parser("use", help="Select a task for this worktree")
    task_use.add_argument("task")
    task_link = task_commands.add_parser("link-pr", help="Link a task to a pull request")
    task_link.add_argument("task")
    task_link.add_argument("pr_ref")
    task_link.add_argument("--url", dest="pr_url")
    task_merge = task_commands.add_parser("merge", help="Merge an incorrect task into another")
    task_merge.add_argument("source")
    task_merge.add_argument("destination")
    task_sync = task_commands.add_parser("sync", help="Publish all task sessions in Git notes")
    task_sync.add_argument("task", nargs="?")
    task_sync.add_argument("--commit", default="HEAD")

    session = commands.add_parser("session", help="Add or correct session metadata")
    session_commands = session.add_subparsers(dest="session_action", required=True)
    session_add = session_commands.add_parser("add", help="Add a planning, review, or imported session")
    session_add.add_argument("--task")
    session_add.add_argument("--model", required=True)
    session_add.add_argument("--harness", required=True)
    session_add.add_argument(
        "--role",
        choices=("planning", "implementation", "testing", "review", "other"),
        default="other",
    )
    session_add.add_argument("--summary")
    session_add.add_argument("--parent-session")
    session_add.add_argument("--tokens", type=_tokens, default=None)
    session_add.add_argument("--cost-usd", type=_cost, default=None)
    session_add.add_argument("--usage-includes-children", action="store_true")
    session_add.add_argument(
        "--outcome",
        choices=("completed", "failed", "interrupted", "abandoned"),
        default="completed",
    )
    session_move = session_commands.add_parser("move", help="Move a session to the correct task")
    session_move.add_argument("session_id")
    session_move.add_argument("task")

    record = commands.add_parser("record", help="Manually attach attribution to a Git commit")
    record.add_argument(
        "commit", nargs="?", default="HEAD", help="Commit to record (default: HEAD)"
    )
    record.add_argument("--json", action="store_true", help="Print the raw result JSON")
    _repo_option(record)

    demo = commands.add_parser("demo", help="Print a report from an isolated example repository")
    demo.add_argument("--json", action="store_true", help="Print the raw report JSON")

    # Private commands are deliberately absent from the public help listing.
    # Both global and suffix --repo forms remain compatible with installed hooks.
    native = commands.add_parser("_hook", add_help=False)
    native.add_argument("--harness", required=True, choices=("codex", "claude-code"))
    native.add_argument("--repo", dest="hook_repo", type=Path)
    native.add_argument("--repository-hook", action="store_true")
    git_hook = commands.add_parser("_git-hook", add_help=False)
    git_hook.add_argument("event", choices=("post-commit", "post-merge", "post-rewrite"))
    git_hook.add_argument("--repo", dest="hook_repo", type=Path)
    collector = commands.add_parser("_collector", add_help=False)
    collector.add_argument("collector_command", choices=("_serve",))
    collector.add_argument("--state-dir", required=True)
    collector.add_argument("--host", choices=("127.0.0.1",), required=True)
    collector.add_argument("--port", type=int, required=True)
    collector.add_argument("--max-request-bytes", type=int, required=True)
    share = commands.add_parser("_share", add_help=False)
    share.add_argument("hook_arguments", nargs=argparse.REMAINDER)
    return result


def _selected_repo(args: argparse.Namespace) -> Path:
    return (
        getattr(args, "command_repo", None)
        or getattr(args, "folder", None)
        or getattr(args, "hook_repo", None)
        or args.repo
    )


def _emit(value: str, *, stream: TextIO | None = None) -> None:
    output = stream or sys.stdout
    output.write(value)
    if not value.endswith("\n"):
        output.write("\n")


def _emit_json(value: Any, *, stream: TextIO | None = None) -> None:
    _emit(json.dumps(value, indent=2, ensure_ascii=False), stream=stream)


def _setup_message(status: dict[str, Any], *, installed: bool) -> str:
    from .terminal import render_setup

    return render_setup(status, installed=installed)


def _record_message(result: dict[str, Any]) -> str:
    from .terminal import safe_text

    commit = safe_text(str(result.get("commit", "unknown"))[:12])
    attributed = safe_text(result.get("attributed_lines", 0))
    unknown = safe_text(result.get("unknown_lines", 0))
    state = "written" if result.get("note_written") else "already present"
    lines = [
        f"Recorded {commit}: {attributed} attributed line(s), "
        f"{unknown} unknown line(s); note {state}."
    ]
    warnings = result.get("warnings")
    lines.extend(
        f"Warning: {safe_text(message)}"
        for message in (warnings if isinstance(warnings, list) else [])
        if isinstance(message, str)
    )
    return "\n".join(lines)


def _hosted_result_message(result: dict[str, Any]) -> str:
    from .terminal import safe_text

    repository = result.get("repository")
    if isinstance(repository, dict):
        identity = safe_text(repository.get("full_name")) or "the detected repository"
    else:
        identity = "the detected repository"
    status = safe_text(result.get("status")) or "unknown"
    lines = [f"Hosted setup {status} for {identity}."]
    if result.get("pull_request_url"):
        lines.append(f"Pull request: {safe_text(result['pull_request_url'])}")
    if result.get("branch"):
        lines.append(f"Branch: {safe_text(result['branch'])}")
    files = result.get("files")
    if isinstance(files, list):
        lines.append(f"Files in review: {len(files)}")
    for warning in result.get("warnings", []):
        lines.append(f"Warning: {safe_text(warning)}")
    return "\n".join(lines)


def _doctor_message(result: dict[str, Any]) -> str:
    from .terminal import safe_text

    lines = ["Hosted installation readiness"]
    lines.append(f"Status: {'ready' if result.get('ready') is True else 'needs attention'}")
    repository = result.get("repository")
    if isinstance(repository, dict):
        lines.append(f"Repository: {safe_text(repository.get('full_name'))}")
    for check in result.get("checks", []):
        if not isinstance(check, dict):
            continue
        state = safe_text(check.get("state")) or "unknown"
        name = safe_text(check.get("name")) or "check"
        message = safe_text(check.get("message"))
        lines.append(f"{name}: {state}" + (f" - {message}" if message else ""))
    return "\n".join(lines)


def _read_hook_payload() -> dict[str, Any]:
    source = sys.stdin.buffer if hasattr(sys.stdin, "buffer") else sys.stdin
    raw = source.read(_HOOK_INPUT_LIMIT + 1)
    if len(raw) > _HOOK_INPUT_LIMIT:
        raise ValueError("Hook payload is too large.")
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("Hook payload must be a JSON object.")
    return payload


def main(argv: list[str] | None = None) -> int:
    command_parser = parser()
    args = command_parser.parse_args(argv)
    try:
        if args.action == "help":
            command_parser.print_help()
            print(
                "\nReading the report\n"
                "Tasks use branch evidence unless you supply a task.\n"
                "Retention = retained / landed lines; it does not measure task success.\n"
                "No landed lines: —. Unknown evidence stays unknown.\n"
                "Tasks collect every inferred or explicitly linked session.\n"
                "\nSessions, commits, and replacement history:\n"
                "  joyride show <task>\n"
                "Full report as JSON:\n"
                "  joyride report --json\n"
                "For another repository or target:\n"
                "  joyride --repo /path/to/repo report --json --target main"
            )
            return 0

        if args.action == "_hook":
            from .automation import handle_hook

            payload = _read_hook_payload()
            if args.repository_hook:
                from .user_install import user_hook_covers

                if user_hook_covers(args.harness, payload.get("hook_event_name")):
                    return 0
            payload_cwd = payload.get("cwd")
            selected_hook_repo = (
                args.hook_repo
                or (
                    Path(payload_cwd)
                    if isinstance(payload_cwd, str) and payload_cwd.strip()
                    else args.repo
                )
            )
            handle_hook(selected_hook_repo, payload, args.harness)
            return 0
        if args.action == "_git-hook":
            from .automation import handle_git_hook

            handle_git_hook(_selected_repo(args), args.event)
            return 0
        if args.action == "_share":
            from .sharing import main as share

            return share(args.hook_arguments)
        if args.action == "_collector":
            from .telemetry import _collector_process

            if not 0 <= args.port <= 65535:
                raise ValueError("collector port must be between 0 and 65535")
            if args.max_request_bytes <= 0:
                raise ValueError("collector max request bytes must be positive")
            return _collector_process(
                args.state_dir,
                args.host,
                args.port,
                args.max_request_bytes,
            )

        if args.action == "install":
            if args.user:
                from .terminal import render_user_setup
                from .user_install import install_user_hooks

                _emit(render_user_setup(install_user_hooks(), installed=True))
                return 0
            from .install import install_repo

            status = install_repo(_selected_repo(args), traces_enabled=not args.no_traces)
            _emit(_setup_message(status, installed=True))
            return 0

        if args.action == "uninstall":
            if args.user:
                from .terminal import render_user_setup
                from .user_install import uninstall_user_hooks

                _emit(render_user_setup(uninstall_user_hooks(), installed=False))
                return 0
            from .install import uninstall_repo

            status = uninstall_repo(_selected_repo(args))
            _emit(_setup_message(status, installed=False))
            return 0

        if args.action == "run":
            from .capture import run_session
            from .harnesses import (
                canonical_harness_id,
                detect_harness,
                detect_model,
                harness_display_name,
            )

            command = args.command[1:] if args.command[:1] == ["--"] else args.command
            if not command:
                command_parser.error("run requires a command after --")
            if args.feature is not None and not args.feature.strip():
                command_parser.error("--feature cannot be empty")
            if args.task is not None and not args.task.strip():
                command_parser.error("--task cannot be empty")

            if args.manual:
                for label in ("model", "harness"):
                    value = getattr(args, label)
                    if value is not None and not value.strip():
                        command_parser.error(f"--{label} cannot be empty")
                model = args.model.strip() if args.model is not None else None
                harness = args.harness.strip() if args.harness is not None else None
                harness_id = None
                model_source = "reported" if model is not None else None
                harness_source = "reported" if harness is not None else None
                label_source = "reported"
            else:
                detected_harness = detect_harness(command)
                if args.harness is not None:
                    if not args.harness.strip():
                        command_parser.error("--harness cannot be empty")
                    harness = args.harness.strip()
                    harness_source = "reported"
                    try:
                        harness_id = canonical_harness_id(harness)
                    except ValueError:
                        harness_id = None
                    else:
                        harness = harness_display_name(harness_id)
                elif detected_harness is not None:
                    harness = detected_harness.display_name
                    harness_id = detected_harness.id
                    harness_source = "command"
                else:
                    command_parser.error(
                        "could not detect the coding harness; use --harness"
                    )

                detected_model = detect_model(command)
                if args.model is not None:
                    if not args.model.strip():
                        command_parser.error("--model cannot be empty")
                    model = args.model.strip()
                    model_source = "reported"
                elif detected_model is not None:
                    model = detected_model
                    model_source = "command"
                else:
                    model = "unknown"
                    model_source = "unknown"

                sources = {harness_source, model_source}
                label_source = sources.pop() if len(sources) == 1 else "mixed"

            if args.manual and args.cost_usd is not None:
                command_parser.error(
                    "--cost-usd records AI cost and cannot be used with --manual"
                )
            result = run_session(
                _selected_repo(args),
                args.feature,
                model,
                harness,
                args.cost_usd,
                command,
                actor_kind="manual" if args.manual else "ai",
                task_query=args.task,
                role=args.role,
                summary=args.summary,
                parent_session_id=args.parent_session,
                token_count=args.tokens,
                usage_includes_children=args.usage_includes_children,
                harness_id=harness_id,
                model_source=model_source,
                harness_source=harness_source,
                label_source=label_source,
                integration_mode="wrapper",
                metadata_adapter_id=harness_id if not args.manual else None,
            )
            if args.json:
                _emit_json(result, stream=sys.stderr)
            else:
                from .terminal import render_capture

                task_data = result.get("task")
                task_name = (
                    task_data.get("name")
                    if isinstance(task_data, dict)
                    and isinstance(task_data.get("name"), str)
                    else args.feature or args.task or "inferred task"
                )
                _emit(
                    render_capture(
                        result,
                        feature=task_name,
                        model=model or "Manual",
                        harness=harness or "editor",
                        cost_usd=args.cost_usd,
                    ),
                    stream=sys.stderr,
                )
            code = result["exit_code"]
            if not isinstance(code, int):
                return 1
            return code if code >= 0 else 128 - code
        if args.action == "task":
            from .task_notes import sync_task, sync_task_ids, sync_task_if_anchored
            from .tasks import (
                link_pull_request,
                list_tasks,
                merge_tasks,
                start_task,
                use_task,
            )

            if args.task_action == "start":
                result = start_task(
                    _selected_repo(args), args.name, pr_ref=args.pr_ref, pr_url=args.pr_url
                )
            elif args.task_action == "list":
                result = list_tasks(_selected_repo(args))
            elif args.task_action == "use":
                result = use_task(_selected_repo(args), args.task)
            elif args.task_action == "link-pr":
                result = link_pull_request(
                    _selected_repo(args), args.task, args.pr_ref, pr_url=args.pr_url
                )
                sync_task_if_anchored(_selected_repo(args), str(result["id"]))
            elif args.task_action == "merge":
                result = merge_tasks(_selected_repo(args), args.source, args.destination)
                task_ids = {str(result["source_task_id"]), str(result["task"]["id"])}
                for commit in result["sync_commits"]:
                    sync_task_ids(_selected_repo(args), task_ids, str(commit))
            else:
                result = sync_task(
                    _selected_repo(args), args.task, commit=args.commit
                )
            _emit_json(result)
            return 0
        if args.action == "session":
            from .task_notes import sync_task_ids, sync_task_if_anchored
            from .tasks import add_session, move_session

            if args.session_action == "add":
                result = add_session(
                    _selected_repo(args),
                    model=args.model,
                    harness=args.harness,
                    task_query=args.task,
                    role=args.role,
                    summary=args.summary,
                    parent_session_id=args.parent_session,
                    token_count=args.tokens,
                    cost_usd=args.cost_usd,
                    usage_includes_children=args.usage_includes_children,
                    outcome=args.outcome,
                )
            else:
                result = move_session(
                    _selected_repo(args), args.session_id, args.task
                )
            if args.session_action == "move" and result["sync_commits"]:
                task_ids = {str(result["task"]["id"])}
                if result["source_task_id"]:
                    task_ids.add(str(result["source_task_id"]))
                for commit in result["sync_commits"]:
                    sync_task_ids(_selected_repo(args), task_ids, str(commit))
            else:
                sync_task_if_anchored(
                    _selected_repo(args), str(result["task"]["id"])
                )
            _emit_json(result)
            return 0
        if args.action == "hook":
            from .activity import hook_activity
            from .harnesses import canonical_harness_id
            from .hook_capture import record_hook_event
            from .hooks import MAX_HOOK_BYTES, parse_hook_payload
            from .store import repository_root

            feature = args.feature or os.environ.get("ATTRIBUTION_FEATURE")
            if feature is None or not feature.strip():
                command_parser.error(
                    "hook requires --feature or ATTRIBUTION_FEATURE"
                )
            harness_id = canonical_harness_id(args.harness)
            source = getattr(sys.stdin, "buffer", sys.stdin)
            raw = source.read(MAX_HOOK_BYTES + 1)
            if isinstance(raw, str):
                raw = raw.encode("utf-8")
            if len(raw) > MAX_HOOK_BYTES:
                raise ValueError(f"hook payload exceeds {MAX_HOOK_BYTES} bytes")
            if not raw.strip():
                raise ValueError("hook payload is empty")
            try:
                payload = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
                raise ValueError("hook payload must be one UTF-8 JSON object") from exc
            event = parse_hook_payload(
                payload,
                harness_id=harness_id,
                event_override=args.event,
                session_override=args.session_id,
                model_override=args.model,
            )
            repo = _selected_repo(args)
            # Locators are repository-relative, so they need the root rather
            # than whichever directory the hook was invoked from. Recording the
            # event reports an unusable repository; this step only prepares.
            try:
                repo_root = repository_root(repo)
            except (OSError, ValueError):
                repo_root = repo
            result = record_hook_event(
                repo,
                feature,
                event,
                harness_version=args.harness_version,
                model_source="reported" if args.model is not None else "native_hook",
                workflow=hook_activity(
                    harness_id, payload, repo_root, event=args.event
                ),
            )
            if args.json:
                _emit_json(result)
            return 0
        if args.action == "recover":
            from .hook_capture import recover_hook_sessions

            _emit_json(
                recover_hook_sessions(
                    _selected_repo(args),
                    capture_changes=args.capture_changes,
                )
            )
            return 0
        if args.action == "harnesses":
            from .harnesses import harness_catalog

            _emit_json(harness_catalog())
            return 0
        if args.action == "hook-template":
            from .harnesses import canonical_harness_id
            from .hook_templates import hook_template

            harness_id = canonical_harness_id(args.harness)
            template = (
                hook_template(harness_id, feature=args.feature)
                if args.attribution_command is None
                else hook_template(
                    harness_id,
                    feature=args.feature,
                    attribution_command=args.attribution_command,
                )
            )
            _emit_json(template)
            return 0
        if args.action == "record":
            from .notes import record_commit

            result = record_commit(_selected_repo(args), args.commit)
            if args.json:
                _emit_json(result)
            else:
                _emit(_record_message(result))
            return 0

        if args.action == "status":
            from .automation import automation_status
            from .install import installation_status

            from .user_install import user_install_status

            selected = _selected_repo(args)
            installation = installation_status(selected)
            installation["user_scope"] = user_install_status()
            if installation.get("repository_installed") is True:
                try:
                    from .store import git_common_dir
                    from .telemetry_setup import telemetry_install_status

                    machine_telemetry = telemetry_install_status(
                        str(git_common_dir(selected))
                    )
                    existing_telemetry = installation.get("telemetry")
                    installation["telemetry"] = {
                        **(
                            existing_telemetry
                            if isinstance(existing_telemetry, dict)
                            else {}
                        ),
                        **machine_telemetry,
                    }
                except (OSError, ValueError):
                    # Installation state remains useful even when optional cost
                    # telemetry cannot be inspected.
                    pass
            automation = (
                automation_status(selected)
                if installation.get("git_repository") is not False
                else {"pending_captures": 0, "pending_commits": 0, "warnings": []}
            )
            if args.json:
                _emit_json({"installation": installation, "automation": automation})
            else:
                from .terminal import render_status

                _emit(render_status(installation, automation))
            return 0

        if args.action == "doctor":
            from .hosted import doctor

            result = doctor(
                _selected_repo(args),
                check_github=not args.no_network,
                hosted_url=args.hosted_url,
            )
            if args.json:
                _emit_json(result)
            else:
                _emit(_doctor_message(result))
            return 0 if result.get("ready") is True else 1

        if args.action == "hosted":
            if args.hosted_action != "setup":
                command_parser.error("hosted requires setup")
            from .hosted import setup

            result = setup(
                _selected_repo(args),
                hosted_url=args.hosted_url,
                branch=args.branch,
                base_branch=args.base_branch,
                dry_run=args.dry_run,
                open_browser=not args.no_browser,
                open_pull_request=not args.no_open,
            ).as_dict()
            if args.json:
                _emit_json(result)
            else:
                _emit(_hosted_result_message(result))
            return 0

        if args.action == "demo":
            from .demo import create_demo
            from .report import build_dashboard
            demo_repo = create_demo()
            dashboard = build_dashboard(demo_repo, target_ref="main")
            if args.json:
                _emit_json(dashboard)
            else:
                from .terminal import render_report

                _emit(render_report(dashboard))
            return 0

        if args.action == "serve":
            from .server import serve

            serve(
                _selected_repo(args),
                port=args.port,
                target_ref=args.target,
            )
            return 0

        if args.action == "code":
            from .code import build_code_file, list_code_files

            selected = _selected_repo(args)
            payload = (
                build_code_file(selected, args.path, target_ref=args.target)
                if args.path is not None
                else list_code_files(selected, target_ref=args.target)
            )
            if args.json:
                _emit_json(payload)
            else:
                from .server import serve

                start_path = "/code"
                if args.path is not None:
                    start_path += "#" + urlencode({"path": args.path})
                serve(
                    selected,
                    port=args.port,
                    target_ref=args.target,
                    start_path=start_path,
                    open_browser=not args.no_open,
                )
            return 0

        if args.action == "why":
            from .why import build_why, split_location

            path, suffix = split_location(args.path)
            if suffix is not None and args.line is not None and suffix != args.line:
                command_parser.error(
                    "give the line number in the path or with --line, not both"
                )
            payload = build_why(
                _selected_repo(args),
                path,
                line=args.line if args.line is not None else suffix,
                target_ref=args.target,
            )
            if args.json:
                _emit_json(payload)
            else:
                from .terminal import render_why

                _emit(render_why(payload))
            return 0

        from .report import build_dashboard

        target = args.target if args.action in {"report", "show"} else "main"
        dashboard = build_dashboard(_selected_repo(args), target_ref=target)
        if args.action == "show":
            if args.json:
                from .terminal import feature_detail

                _emit_json(feature_detail(dashboard, args.group))
                return 0
            from .terminal import render_feature

            _emit(render_feature(dashboard, args.group))
        elif args.action == "report" and args.json:
            _emit_json(dashboard)
        else:
            from .terminal import render_report

            _emit(render_report(dashboard))
        return 0
    except BrokenPipeError:
        try:
            with open(os.devnull, "w") as sink:
                os.dup2(sink.fileno(), sys.stdout.fileno())
        except (AttributeError, OSError, ValueError):
            pass
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        # Hook observers must never block an edit or Git operation, and their
        # generated invocations intentionally produce no output.
        if args.action in {"_hook", "_git-hook", "_share"} or (
            args.action == "hook" and getattr(args, "observer", False)
        ):
            return 0
        message = getattr(exc, "stderr", None) or str(exc)
        if isinstance(message, bytes):
            message = message.decode("utf-8", errors="replace")
        from .terminal import safe_text

        try:
            _emit(f"attribution: {safe_text(message)}", stream=sys.stderr)
        except BrokenPipeError:
            return 0
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
