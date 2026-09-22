"""Canonical metadata and command-line detection for coding harnesses.

This module deliberately does not execute commands or inspect the environment.
Detection is limited to command tokens supplied by a caller, so importing it is
safe in reports, migrations, and command-line completion code.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Sequence


_CANONICAL_ID = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*\Z")
_ENV_ASSIGNMENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*=.*\Z", re.DOTALL)
_WINDOWS_EXECUTABLE_SUFFIXES = (".exe", ".cmd", ".bat", ".com")


def _normalize_name(value: str) -> str:
    """Normalize human spelling without applying harness-specific guesses."""

    return re.sub(r"[^a-z0-9]+", "-", value.strip().casefold()).strip("-")


@dataclass(frozen=True, slots=True)
class HarnessSpec:
    """Stable public metadata for a supported coding harness.

    ``hook_support`` means that the harness exposes a documented hook, event,
    extension, or SDK callback surface suitable for lifecycle attribution.  It
    does not mean that this package has installed those hooks.
    """

    id: str
    display_name: str
    command_names: tuple[str, ...]
    hook_support: bool
    package_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not _CANONICAL_ID.fullmatch(self.id):
            raise ValueError("harness id must be a nonempty canonical lowercase id")
        if not isinstance(self.display_name, str) or not self.display_name.strip():
            raise ValueError("harness display name must be nonempty")
        if not isinstance(self.command_names, tuple):
            raise ValueError("harness command names must be a tuple")
        if not isinstance(self.hook_support, bool):
            raise ValueError("harness hook_support must be a boolean")
        if not isinstance(self.package_names, tuple):
            raise ValueError("harness package names must be a tuple")

        normalized_commands: set[str] = set()
        for command_name in self.command_names:
            if not isinstance(command_name, str) or not command_name.strip():
                raise ValueError("harness command names must contain nonempty strings")
            normalized = _normalize_name(command_name)
            if not normalized:
                raise ValueError("harness command names must contain a letter or number")
            if normalized in normalized_commands:
                raise ValueError("harness command names must be unique")
            normalized_commands.add(normalized)

        normalized_packages: set[str] = set()
        for package_name in self.package_names:
            if (
                not isinstance(package_name, str)
                or not package_name.strip()
                or any(character.isspace() for character in package_name)
            ):
                raise ValueError("harness package names must contain package identifiers")
            normalized = package_name.casefold()
            if normalized in normalized_packages:
                raise ValueError("harness package names must be unique")
            normalized_packages.add(normalized)


# Keep this tuple in id order.  The ids are persisted values; display names are
# presentation values; command_names are conservative executable aliases; and
# package_names are packages accepted by a supported one-shot launcher.  A
# harness with no safe blocking command can still appear in the registry.  In
# particular, generic executable names such as ``agent`` are intentionally not
# included even when a vendor accepts them.
HARNESS_SPECS: tuple[HarnessSpec, ...] = (
    HarnessSpec("aider", "Aider", ("aider",), False, ("aider-chat",)),
    HarnessSpec("amp", "Amp", ("amp",), True, ("@ampcode/cli",)),
    HarnessSpec(
        "claude-code",
        "Claude Code",
        ("claude",),
        True,
        ("@anthropic-ai/claude-code",),
    ),
    HarnessSpec("cline", "Cline", ("cline",), True, ("cline", "@cline/cli")),
    HarnessSpec("codex", "Codex", ("codex",), True, ("@openai/codex",)),
    HarnessSpec("continue", "Continue", ("cn",), False, ("@continuedev/cli",)),
    HarnessSpec("cursor", "Cursor", ("cursor-agent",), True),
    HarnessSpec(
        "gemini",
        "Gemini CLI",
        ("gemini",),
        True,
        ("@google/gemini-cli",),
    ),
    HarnessSpec(
        "github-copilot",
        "GitHub Copilot",
        ("copilot",),
        True,
        ("@github/copilot",),
    ),
    HarnessSpec("goose", "Goose", ("goose",), True),
    HarnessSpec(
        "grok-build",
        "Grok Build",
        ("grok",),
        True,
        ("@xai-official/grok",),
    ),
    HarnessSpec("hermes", "Hermes Agent", ("hermes",), True),
    HarnessSpec("opencode", "OpenCode", ("opencode",), True, ("opencode-ai",)),
    HarnessSpec("openhands", "OpenHands", ("openhands",), True, ("openhands",)),
    HarnessSpec(
        "pi",
        "Pi",
        ("pi",),
        True,
        ("@earendil-works/pi-coding-agent",),
    ),
    HarnessSpec(
        "qwen-code",
        "Qwen Code",
        ("qwen",),
        True,
        ("@qwen-code/qwen-code",),
    ),
    HarnessSpec("roo-code", "Roo Code", (), True),
    HarnessSpec("windsurf", "Windsurf", (), True),
)


_EXTRA_ALIASES: dict[str, tuple[str, ...]] = {
    "aider": ("aider chat",),
    "amp": ("amp code", "ampcode"),
    "claude-code": ("anthropic claude", "anthropic claude code", "claude cli"),
    "cline": ("cline cli",),
    "codex": ("openai codex", "codex cli"),
    "continue": ("continue cli", "continuedev", "continuedev cli"),
    "cursor": ("cursor agent", "cursor cli"),
    "gemini": ("google gemini", "google gemini cli"),
    "github-copilot": ("copilot cli", "github copilot cli", "gh copilot"),
    "goose": ("block goose", "goose cli"),
    "grok-build": ("xai grok build", "grok cli"),
    "hermes": ("nous hermes", "nousresearch hermes"),
    "openhands": ("open hands", "openhands cli"),
    "opencode": ("open code", "opencode cli"),
    "pi": ("pi coding agent", "pi dev"),
    "qwen-code": ("qwen", "qwen code cli", "qwen cli"),
    "roo-code": ("roo", "roo code cli"),
    "windsurf": ("windsurf cascade", "windsurf cli"),
}


def _build_aliases() -> dict[str, HarnessSpec]:
    result: dict[str, HarnessSpec] = {}
    for spec in HARNESS_SPECS:
        aliases = (spec.id, spec.display_name, *spec.command_names, *_EXTRA_ALIASES.get(spec.id, ()))
        for alias in aliases:
            normalized = _normalize_name(alias)
            previous = result.get(normalized)
            if previous is not None and previous.id != spec.id:
                raise RuntimeError(f"ambiguous harness alias: {alias!r}")
            result[normalized] = spec
    return result


_ALIASES = _build_aliases()


def _build_command_aliases() -> dict[str, HarnessSpec]:
    result: dict[str, HarnessSpec] = {}
    for spec in HARNESS_SPECS:
        for command_name in spec.command_names:
            normalized = _normalize_name(command_name)
            previous = result.get(normalized)
            if previous is not None and previous.id != spec.id:
                raise RuntimeError(f"ambiguous harness command: {command_name!r}")
            result[normalized] = spec
    return result


def _build_package_aliases() -> dict[str, HarnessSpec]:
    result: dict[str, HarnessSpec] = {}
    for spec in HARNESS_SPECS:
        for package_name in spec.package_names:
            normalized = package_name.casefold()
            previous = result.get(normalized)
            if previous is not None and previous.id != spec.id:
                raise RuntimeError(f"ambiguous harness package: {package_name!r}")
            result[normalized] = spec
    return result


_COMMAND_ALIASES = _build_command_aliases()
_PACKAGE_ALIASES = _build_package_aliases()
_PYTHON_MODULES = {"aider": _ALIASES["aider"]}


def canonical_harness_id(value: str) -> str:
    """Return the stable id for an id, display name, or known alias.

    Matching is case-insensitive and treats punctuation, spaces, and underscores
    as equivalent separators.  Unknown and empty values fail closed.
    """

    if not isinstance(value, str):
        raise ValueError("unknown harness")
    spec = _ALIASES.get(_normalize_name(value))
    if spec is None:
        raise ValueError(f"unknown harness: {value!r}")
    return spec.id


def harness_display_name(value: str) -> str:
    """Return the public display name for a known harness id or alias."""

    if not isinstance(value, str):
        raise ValueError("unknown harness")
    spec = _ALIASES.get(_normalize_name(value))
    if spec is None:
        raise ValueError(f"unknown harness: {value!r}")
    return spec.display_name


def _basename(value: str) -> str:
    """Return a platform-independent executable basename."""

    name = value.replace("\\", "/").rsplit("/", 1)[-1]
    folded = name.casefold()
    for suffix in _WINDOWS_EXECUTABLE_SUFFIXES:
        if folded.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _without_package_version(value: str) -> str:
    """Remove an npm-style version suffix while preserving a package scope."""

    value = value.removeprefix("npm:")
    if value.startswith("@"):
        slash = value.find("/")
        version = value.rfind("@")
        if slash != -1 and version > slash:
            return value[:version]
        return value
    return value.split("@", 1)[0]


def _spec_for_executable(value: str) -> HarnessSpec | None:
    """Match one executable or launcher package token, never its arguments."""

    if not value or "=" in value and _ENV_ASSIGNMENT.fullmatch(value):
        return None

    return _COMMAND_ALIASES.get(_normalize_name(_basename(value)))


def _spec_for_package(value: str) -> HarnessSpec | None:
    """Match one verified package name used by a one-shot launcher."""

    if not value or _ENV_ASSIGNMENT.fullmatch(value):
        return None
    package = _without_package_version(value).casefold()
    return _PACKAGE_ALIASES.get(package)


def _skip_environment_prefix(tokens: Sequence[str]) -> int | None:
    """Return the executable index after a real ``env`` command.

    Direct ``NAME=value command`` syntax belongs to a shell and cannot be
    executed by the wrapper. Unknown ``env`` flags fail closed because some of
    them, such as ``-S``, change how the remaining tokens are interpreted.
    """

    index = 1 if tokens and tokens[0] == "--" else 0
    if index >= len(tokens) or _normalize_name(_basename(tokens[index])) != "env":
        return index

    index += 1
    env_options_without_values = {
        "-0",
        "--ignore-environment",
        "--null",
        "-i",
    }
    env_options_with_values = {"-u", "--unset", "-C", "--chdir"}
    while index < len(tokens):
        token = tokens[index]
        if token == "--":
            return index + 1
        if _ENV_ASSIGNMENT.fullmatch(token):
            index += 1
            continue
        if token in env_options_without_values:
            index += 1
            continue
        if token in env_options_with_values:
            if index + 1 >= len(tokens):
                return None
            index += 2
            continue
        if token.startswith("--unset=") or token.startswith("--chdir="):
            index += 1
            continue
        if token == "-S" or token == "--split-string" or token.startswith("--split-string="):
            return None
        if token.startswith("-"):
            return None
        return index
    return None


@dataclass(frozen=True, slots=True)
class _Invocation:
    spec: HarnessSpec
    argument_start: int


def _package_launcher_start(tokens: Sequence[str], index: int) -> int | None:
    """Return the first package-launcher argument, or ``None``."""

    launcher = _normalize_name(_basename(tokens[index]))
    next_index = index + 1
    if launcher in {"npx", "pnpx", "bunx", "uvx"}:
        return next_index
    if (
        launcher == "bun"
        and next_index < len(tokens)
        and tokens[next_index] == "x"
    ):
        return next_index + 1
    if (
        launcher == "uv"
        and next_index + 1 < len(tokens)
        and tuple(tokens[next_index : next_index + 2]) == ("tool", "run")
    ):
        return next_index + 2
    if (
        launcher in {"npm", "pnpm", "yarn", "pipx"}
        and next_index < len(tokens)
        and (
            (launcher == "npm" and tokens[next_index] in {"exec", "x"})
            or (launcher in {"pnpm", "yarn"} and tokens[next_index] == "dlx")
            or (launcher == "pipx" and tokens[next_index] == "run")
        )
    ):
        return next_index + 1
    return None


def _launcher_target(tokens: Sequence[str], index: int) -> _Invocation | None:
    """Resolve a verified package used by a known one-shot launcher."""

    index = _package_launcher_start(tokens, index) or -1
    if index < 0:
        return None

    options_without_values = {
        "--bun",
        "--frozen",
        "--ignore-existing",
        "--isolated",
        "--no-cache",
        "--no-install",
        "--no-python-downloads",
        "--offline",
        "--quiet",
        "--refresh",
        "--silent",
        "--verbose",
        "--yes",
        "-q",
        "-v",
        "-y",
    }
    options_with_values = {
        "--cache",
        "--call",
        "--from",
        "--index",
        "--index-url",
        "--package",
        "--python",
        "--registry",
        "--userconfig",
        "--with",
        "-c",
        "-p",
    }
    package_source: HarnessSpec | None = None
    while index < len(tokens):
        token = tokens[index]
        if token == "--":
            index += 1
            continue
        if token in options_without_values:
            index += 1
            continue
        if token in options_with_values:
            if index + 1 >= len(tokens):
                return None
            if token in {"--from", "--package", "-p"}:
                candidate = _spec_for_package(tokens[index + 1])
                if candidate is None:
                    return None
                if package_source is not None and package_source != candidate:
                    return None
                package_source = candidate
            index += 2
            continue
        matched_option = next(
            (
                option
                for option in options_with_values
                if option.startswith("--") and token.startswith(f"{option}=")
            ),
            None,
        )
        if matched_option is not None:
            value = token.partition("=")[2]
            if not value:
                return None
            if matched_option in {"--from", "--package"}:
                candidate = _spec_for_package(value)
                if candidate is None:
                    return None
                if package_source is not None and package_source != candidate:
                    return None
                package_source = candidate
            index += 1
            continue
        if token.startswith("-"):
            return None

        if package_source is None:
            package_source = _spec_for_package(token)
            if package_source is None:
                return None
            return _Invocation(package_source, index + 1)

        executable = _spec_for_executable(token)
        if executable != package_source:
            return None
        return _Invocation(package_source, index + 1)
    return None


def _python_module_target(tokens: Sequence[str], index: int) -> _Invocation | None:
    """Resolve a small allowlist of documented ``python -m`` entry points."""

    executable = _basename(tokens[index]).casefold()
    if not re.fullmatch(r"(?:python|pythonw|pypy)(?:\d+(?:\.\d+)?)?|py", executable):
        return None
    index += 1
    if executable == "py":
        while index < len(tokens) and re.fullmatch(r"-\d+(?:\.\d+)?(?:-\d+)?", tokens[index]):
            index += 1
    if index + 1 >= len(tokens) or tokens[index] != "-m":
        return None
    spec = _PYTHON_MODULES.get(tokens[index + 1].casefold())
    if spec is None:
        return None
    return _Invocation(spec, index + 2)


def _resolve_invocation(command: Sequence[str]) -> _Invocation | None:
    """Resolve the harness and the start of its own arguments."""

    if isinstance(command, (str, bytes)) or not isinstance(command, Sequence):
        return None
    if any(not isinstance(token, str) for token in command):
        return None
    index = _skip_environment_prefix(command)
    if index is None or index >= len(command):
        return None

    package_target = _launcher_target(command, index)
    if package_target is not None:
        return package_target
    if _package_launcher_start(command, index) is not None:
        return None

    python_target = _python_module_target(command, index)
    if python_target is not None:
        return python_target

    executable = _normalize_name(_basename(command[index]))
    if executable == "gh" and index + 1 < len(command) and command[index + 1] == "copilot":
        return _Invocation(_ALIASES["github-copilot"], index + 2)

    spec = _spec_for_executable(command[index])
    if spec is None:
        return None
    return _Invocation(spec, index + 1)


def detect_harness(command: Sequence[str]) -> HarnessSpec | None:
    """Detect a harness from a tokenized command without executing it.

    Only documented executable, module, and package positions are considered.
    The ``env`` command and common one-shot package launchers are understood.
    Arbitrary shell arguments are never scanned.
    """

    invocation = _resolve_invocation(command)
    return invocation.spec if invocation is not None else None


def detect_model(command: Sequence[str]) -> str | None:
    """Return the last model flag only for a recognized harness command."""

    invocation = _resolve_invocation(command)
    if invocation is None:
        return None

    result: str | None = None
    index = invocation.argument_start
    while index < len(command):
        token = command[index]
        if token == "--":
            break
        if token.startswith("--model="):
            value = token.partition("=")[2].strip()
            if value:
                result = value
        elif token in {"--model", "-m"}:
            if index + 1 >= len(command) or command[index + 1] == "--":
                break
            value = command[index + 1].strip()
            if value and not value.startswith("-"):
                result = value
            index += 1
        index += 1
    return result


def harness_catalog() -> list[dict[str, object]]:
    """Return a stable, JSON-serializable catalog sorted by canonical id."""

    from .adapters import supported_adapters
    from .hook_templates import supported_template_harnesses

    adapter_ids = set(supported_adapters())
    template_ids = set(supported_template_harnesses())

    return [
        {
            "id": spec.id,
            "display_name": spec.display_name,
            "command_names": list(spec.command_names),
            "package_names": list(spec.package_names),
            "hook_support": spec.hook_support,
            "hook_template_support": spec.id in template_ids,
            "metadata_adapter_support": spec.id in adapter_ids,
        }
        for spec in sorted(HARNESS_SPECS, key=lambda item: item.id)
    ]
