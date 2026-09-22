"""Stable command construction for Python and standalone Joyride runtimes.

Commands written into repository hooks outlive the process that installs them.
An ordinary Python installation therefore uses its interpreter plus a private
bootstrap, while a PyInstaller build must invoke the frozen executable itself.
Keeping that distinction here prevents callers from accidentally treating the
standalone executable as a Python interpreter or relying on PyInstaller's
temporary extraction directory.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shlex
import sys
from typing import Iterable, Mapping


PYTHON_RUNTIME = "python"
STANDALONE_RUNTIME = "standalone"
_RUNTIME_KINDS = {PYTHON_RUNTIME, STANDALONE_RUNTIME}


def is_frozen() -> bool:
    """Return whether the current process is a packaged standalone build."""

    return bool(getattr(sys, "frozen", False))


@dataclass(frozen=True)
class RuntimeInvocation:
    """One shell-free command and its optional trusted Python import root."""

    argv: tuple[str, ...]
    pythonpath: Path | None = None
    reset_frozen_environment: bool = False

    def shell(self) -> str:
        """Render the invocation for a vendor hook without interpolating data."""

        command = " ".join(shlex.quote(argument) for argument in self.argv)
        assignments: list[str] = []
        if self.pythonpath is not None:
            assignments.append(f"PYTHONPATH={shlex.quote(str(self.pythonpath))}")
        if self.reset_frozen_environment:
            assignments.append("PYINSTALLER_RESET_ENVIRONMENT=1")
        return " ".join((*assignments, command))

    def environment(self) -> dict[str, str] | None:
        """Return an import-safe child environment when Python needs one."""

        if self.pythonpath is None and not self.reset_frozen_environment:
            return None
        environment = dict(os.environ)
        if self.pythonpath is not None:
            environment["PYTHONPATH"] = str(self.pythonpath)
        if self.reset_frozen_environment:
            environment["PYINSTALLER_RESET_ENVIRONMENT"] = "1"
        return environment


@dataclass(frozen=True)
class RuntimeCommand:
    """A persistent Joyride runtime that can construct child commands."""

    kind: str
    executable: Path
    source_root: Path | None = None

    def __post_init__(self) -> None:
        if self.kind not in _RUNTIME_KINDS:
            raise ValueError("The Joyride runtime kind is invalid.")
        if (
            not self.executable.is_absolute()
            or self.executable != Path(os.path.abspath(self.executable))
        ):
            raise ValueError("The Joyride runtime executable must be canonical.")
        if self.kind == PYTHON_RUNTIME and self.source_root is None:
            raise ValueError("A Python Joyride runtime requires its source root.")
        if self.kind == STANDALONE_RUNTIME and self.source_root is not None:
            raise ValueError("A standalone Joyride runtime cannot use a source root.")
        if self.source_root is not None and (
            not self.source_root.is_absolute()
            or self.source_root != Path(os.path.abspath(self.source_root))
        ):
            raise ValueError("The Joyride source root must be canonical.")

    @property
    def standalone(self) -> bool:
        return self.kind == STANDALONE_RUNTIME

    def cli(
        self,
        arguments: Iterable[str],
        *,
        bootstrap: Path | None = None,
    ) -> RuntimeInvocation:
        """Build a persistent invocation of the Joyride CLI."""

        selected = tuple(str(argument) for argument in arguments)
        if self.standalone:
            return RuntimeInvocation((str(self.executable), *selected))
        if bootstrap is not None:
            return RuntimeInvocation(
                (str(self.executable), str(bootstrap), *selected),
                self.source_root,
            )
        return RuntimeInvocation(
            (str(self.executable), "-P", "-m", "attribution", *selected),
            self.source_root,
        )

    def module_child(
        self,
        module: str,
        arguments: Iterable[str],
        *,
        standalone_action: str,
    ) -> RuntimeInvocation:
        """Build a module child command with a frozen CLI equivalent."""

        selected = tuple(str(argument) for argument in arguments)
        if self.standalone:
            return RuntimeInvocation(
                (str(self.executable), standalone_action, *selected),
                reset_frozen_environment=True,
            )
        return RuntimeInvocation(
            (str(self.executable), "-P", "-m", module, *selected),
            self.source_root,
        )


def current_runtime(*, source_root: Path | None = None) -> RuntimeCommand:
    """Describe the runtime whose process is currently executing."""

    executable = Path(sys.executable).resolve()
    if is_frozen():
        return RuntimeCommand(STANDALONE_RUNTIME, executable)
    selected_root = (
        Path(source_root).resolve()
        if source_root is not None
        else Path(__file__).resolve().parent.parent
    )
    return RuntimeCommand(PYTHON_RUNTIME, executable, selected_root)


def stored_runtime(
    *,
    executable: object,
    source_root: object,
    kind: object = None,
) -> RuntimeCommand:
    """Reconstruct a validated runtime from install-manifest primitives.

    Manifests written before runtime kinds existed are ordinary Python
    installations and remain readable.
    """

    selected_kind = PYTHON_RUNTIME if kind is None else kind
    if selected_kind not in _RUNTIME_KINDS or not isinstance(executable, str):
        raise ValueError("The Joyride runtime metadata is invalid.")
    if selected_kind == STANDALONE_RUNTIME:
        if source_root is not None:
            raise ValueError("The standalone Joyride runtime metadata is invalid.")
        return RuntimeCommand(STANDALONE_RUNTIME, Path(executable))
    if not isinstance(source_root, str):
        raise ValueError("The Python Joyride runtime metadata is invalid.")
    return RuntimeCommand(PYTHON_RUNTIME, Path(executable), Path(source_root))


def default_hook_executable() -> str:
    """Return the safe default executable for generated native templates."""

    runtime = current_runtime()
    return str(runtime.executable) if runtime.standalone else "joyride"


def system_subprocess_environment(
    environment: Mapping[str, str] | None = None,
) -> dict[str, str] | None:
    """Restore system library paths before a frozen app launches OS tools.

    ``environment`` may be a deliberately restricted child environment, such
    as the noninteractive one used for Git publication. It is copied before
    sanitizing so callers never mutate their own mapping.
    """

    if environment is None:
        if not is_frozen():
            return None
        selected = dict(os.environ)
    else:
        selected = dict(environment)
        if not is_frozen():
            return selected
    for variable in ("LD_LIBRARY_PATH", "LIBPATH", "DYLD_LIBRARY_PATH"):
        original = selected.get(variable + "_ORIG")
        if original is None:
            selected.pop(variable, None)
        else:
            selected[variable] = original
    return selected


__all__ = [
    "PYTHON_RUNTIME",
    "STANDALONE_RUNTIME",
    "RuntimeCommand",
    "RuntimeInvocation",
    "current_runtime",
    "default_hook_executable",
    "is_frozen",
    "stored_runtime",
    "system_subprocess_environment",
]
