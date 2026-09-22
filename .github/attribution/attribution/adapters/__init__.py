"""Read-only native session metadata adapters.

The public entry points are deliberately best-effort: unsupported harnesses and
ambiguous native sessions return ``None`` and never interfere with repository
capture.
"""

from __future__ import annotations

from importlib import import_module
import os
from typing import Mapping, Protocol

from .base import AdapterSnapshot, NativeMetadata


class _AdapterModule(Protocol):
    def snapshot(
        self, repo: str | os.PathLike[str], env: Mapping[str, str]
    ) -> AdapterSnapshot | None: ...

    def finalize(
        self,
        adapter_snapshot: AdapterSnapshot,
        repo: str | os.PathLike[str],
        env: Mapping[str, str],
    ) -> NativeMetadata | None: ...


_ADAPTER_MODULES = {
    "grok-build": ".grok_build",
    "hermes": ".hermes",
    "opencode": ".opencode",
    "pi": ".pi",
}


def _adapter(harness_id: str) -> _AdapterModule | None:
    module_name = _ADAPTER_MODULES.get(harness_id)
    if module_name is None:
        return None
    try:
        return import_module(module_name, __package__)  # type: ignore[return-value]
    except ModuleNotFoundError as exc:
        # Other adapters can be omitted from small source distributions.  Do
        # not hide a missing dependency imported *by* an existing adapter.
        expected = f"{__package__}.{module_name.removeprefix('.')}"
        if exc.name != expected:
            raise
        return None


def snapshot_adapter(
    harness_id: str,
    repo: str | os.PathLike[str],
    env: Mapping[str, str],
) -> AdapterSnapshot | None:
    """Capture bounded file metadata before running a supported harness."""

    adapter = _adapter(harness_id)
    return None if adapter is None else adapter.snapshot(repo, env)


def finalize_adapter(
    adapter_snapshot: AdapterSnapshot,
    repo: str | os.PathLike[str],
    env: Mapping[str, str],
) -> NativeMetadata | None:
    """Return allowlisted metadata from the one native session changed since snapshot."""

    adapter = _adapter(adapter_snapshot.harness_id)
    return None if adapter is None else adapter.finalize(adapter_snapshot, repo, env)


def supported_adapters() -> tuple[str, ...]:
    """Return canonical harness ids with post-run metadata adapters."""

    return tuple(sorted(_ADAPTER_MODULES))


__all__ = [
    "AdapterSnapshot",
    "NativeMetadata",
    "finalize_adapter",
    "snapshot_adapter",
    "supported_adapters",
]
