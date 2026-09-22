"""Small, dependency-free value types shared by native metadata adapters."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import os
from pathlib import Path
import stat


@dataclass(frozen=True, slots=True)
class NativeMetadata:
    """Allowlisted metadata recovered from one native harness session.

    Adapters intentionally return no prompt, response, or tool-call content.
    Missing or untrustworthy native values stay ``None`` rather than being
    guessed from a filename or configuration default.
    """

    harness_id: str
    native_session_id: str | None = None
    model: str | None = None
    provider: str | None = None
    cost_usd: float | None = None
    cost_source: str | None = None
    harness_version: str | None = None
    warnings: tuple[str, ...] = ()

    # How the session was worked, for the adapters whose records say.  A
    # harness that records none of this leaves every field ``None``: a count
    # the record does not contain is unknown, never zero.  ``models`` is the
    # models the session used in the order its record lists them, so a second
    # entry is a model change; ``tool_call_counts`` is keyed by the tool
    # classes of ``activity.TOOL_CLASSES`` and describes the mix a report
    # shows, which no column holds.  Only ``tool_call_count`` is stored.
    agent_type: str | None = None
    session_source: str | None = None
    permission_mode: str | None = None
    effort_level: str | None = None
    turn_count: int | None = None
    prompt_count: int | None = None
    interrupt_count: int | None = None
    compaction_count: int | None = None
    model_switch_count: int | None = None
    tool_call_count: int | None = None
    duration_ms: int | None = None
    tool_call_counts: Mapping[str, int] | None = None
    models: tuple[str, ...] | None = None


@dataclass(frozen=True, slots=True)
class AdapterSnapshot:
    """Opaque before-state used to identify the native session changed by a run."""

    harness_id: str
    state: object


def read_stable_regular_file(
    path: Path,
    maximum_bytes: int,
) -> bytes | None:
    """Read one regular file without following a final symlink.

    The size cap applies to the open file descriptor.  Identity and metadata
    checks before and after the read reject replacement, truncation, and growth
    races instead of parsing a mixed snapshot.
    """

    if isinstance(maximum_bytes, bool) or not isinstance(maximum_bytes, int) or maximum_bytes < 0:
        raise ValueError("maximum_bytes must be a nonnegative integer")
    try:
        path_before = path.lstat()
        if not stat.S_ISREG(path_before.st_mode) or path_before.st_size > maximum_bytes:
            return None

        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        try:
            with os.fdopen(descriptor, "rb", closefd=True) as handle:
                descriptor = -1
                opened = os.fstat(handle.fileno())
                if not stat.S_ISREG(opened.st_mode) or opened.st_size > maximum_bytes:
                    return None
                if (opened.st_dev, opened.st_ino) != (path_before.st_dev, path_before.st_ino):
                    return None
                content = handle.read(maximum_bytes + 1)
                after_read = os.fstat(handle.fileno())
        finally:
            if descriptor >= 0:
                os.close(descriptor)
        path_after = path.lstat()
    except OSError:
        return None

    if len(content) > maximum_bytes or len(content) != opened.st_size:
        return None
    opened_signature = (
        opened.st_dev,
        opened.st_ino,
        opened.st_size,
        opened.st_mtime_ns,
        opened.st_ctime_ns,
    )
    after_read_signature = (
        after_read.st_dev,
        after_read.st_ino,
        after_read.st_size,
        after_read.st_mtime_ns,
        after_read.st_ctime_ns,
    )
    path_after_signature = (
        path_after.st_dev,
        path_after.st_ino,
        path_after.st_size,
        path_after.st_mtime_ns,
        path_after.st_ctime_ns,
    )
    if opened_signature != after_read_signature or after_read_signature != path_after_signature:
        return None
    return content
