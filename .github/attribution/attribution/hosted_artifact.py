"""Build the bounded, source-free PR artifact accepted by the hosted service.

The trusted base-branch workflow performs all Git history computation.  The
hosted service receives only validated attribution metadata and immutable Git
object locators; it hydrates source later, after checking viewer access.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import html
import json
import math
from pathlib import Path, PurePosixPath
import re
import subprocess
from typing import Any, Mapping
import unicodedata

from .code import build_pr_code_file, list_pr_code_files
from .pr_report import build_pr_report, fallback_narrative
from .runtime import system_subprocess_environment


SCHEMA = "harness-attribution/pr-artifact@1"
MAX_ARTIFACT_BYTES = 3 * 1024 * 1024
MAX_DETAILED_FILES = 100
MAX_NARRATIVE_CHARS = 1000
_OID = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_REPOSITORY = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_SESSION_FIELDS = (
    "id", "task_id", "feature", "model", "harness", "actor_kind",
    "source_session_id", "label_source", "membership_source", "role",
    "parent_session_id", "token_count", "token_source", "cost_usd",
    "cost_source", "usage_includes_children", "started_at", "ended_at",
    "exit_code", "outcome",
)
_FORBIDDEN_KEYS = {
    "content", "before_text", "after_text", "source_text", "prompt",
    "command", "transcript", "worktree_id", "feature_source", "summary_text",
}
# The paragraph is display prose, not evidence, and a model drafts a different
# one on every run. Keeping it out of the digest lets a re-run of the same
# revision stay a retry instead of a conflict the hosted service rejects.
_UNSIGNED_SUMMARY_KEYS = frozenset({"narrative_summary", "narrative_source"})


def canonical_json(value: Any) -> bytes:
    """Return the single canonical encoding used for limits and digests."""
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ValueError("Artifact contains a value that cannot be encoded safely.") from exc


def artifact_digest(value: dict[str, Any]) -> str:
    """Digest an artifact without its ``digest`` member or its narrative prose."""
    if not isinstance(value, dict):
        raise ValueError("Artifact must be an object.")
    unsigned = {key: item for key, item in value.items() if key != "digest"}
    summary = unsigned.get("summary")
    if isinstance(summary, dict) and _UNSIGNED_SUMMARY_KEYS & set(summary):
        # Copy rather than mutate: the caller keeps the artifact it passed, and
        # an artifact built before these keys existed keeps its stored digest.
        unsigned["summary"] = {
            key: item for key, item in summary.items()
            if key not in _UNSIGNED_SUMMARY_KEYS
        }
    return "sha256:" + hashlib.sha256(canonical_json(unsigned)).hexdigest()


def artifact_bytes(value: dict[str, Any]) -> bytes:
    validated = validate_artifact(value)
    return canonical_json(validated)


def _path(value: Any, *, optional: bool = False) -> str | None:
    if value is None and optional:
        return None
    if not isinstance(value, str) or not value or len(value) > 4096 or "\0" in value:
        raise ValueError("Artifact contains an invalid repository path.")
    parsed = PurePosixPath(value)
    if parsed.is_absolute() or ".." in parsed.parts or any(part in {"", "."} for part in parsed.parts):
        raise ValueError("Artifact paths must be repository-relative.")
    return value


def _oid(value: Any, name: str, *, optional: bool = False) -> str | None:
    if value is None and optional:
        return None
    if not isinstance(value, str) or _OID.fullmatch(value) is None or not value.strip("0"):
        raise ValueError(f"Artifact {name} must be a complete Git object ID.")
    return value


def _blob_oid(repo: Path, commit: str, path: str, cache: dict[tuple[str, str], str]) -> str:
    key = (commit, path)
    if key in cache:
        return cache[key]
    try:
        result = subprocess.run(
            ["git", "--no-replace-objects", "--literal-pathspecs", "-c", "core.quotePath=false",
             "-C", str(repo), "ls-tree", "-z", "--full-tree", commit, "--", path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env=system_subprocess_environment(), timeout=10, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ValueError("Could not resolve an immutable source locator.") from exc
    rows = [row for row in result.stdout.split(b"\0") if row]
    if result.returncode or len(rows) != 1:
        raise ValueError("Could not resolve an immutable source locator.")
    try:
        header, raw_path = rows[0].split(b"\t", 1)
        mode, kind, oid = header.decode("ascii").split()
        actual = raw_path.decode("utf-8")
    except (ValueError, UnicodeError) as exc:
        raise ValueError("Git returned an invalid immutable source locator.") from exc
    if actual != path or mode not in {"100644", "100755"} or kind != "blob" or _OID.fullmatch(oid) is None:
        raise ValueError("Source locator does not identify an ordinary file blob.")
    cache[key] = oid
    return oid


def _line(
    repo: Path, raw: dict[str, Any], blobs: dict[tuple[str, str], str],
    source_commit: str | None, source_path: str | None,
) -> dict[str, Any]:
    origin = raw.get("origin")
    if not isinstance(origin, dict):
        raise ValueError("Code projection contains a line without an origin.")
    commit = _oid(origin.get("commit"), "line origin")
    path = _path(origin.get("path"))
    number = origin.get("line")
    if isinstance(number, bool) or not isinstance(number, int) or number < 1:
        raise ValueError("Code projection contains an invalid origin line.")
    display_number = raw.get("number")
    if (
        source_commit is None or source_path is None
        or isinstance(display_number, bool) or not isinstance(display_number, int)
        or display_number < 1
    ):
        raise ValueError("Code projection contains an invalid source line.")
    return {
        "number": display_number,
        "has_newline": raw.get("has_newline"),
        "session_id": raw.get("session_id"),
        "origin": {"commit": commit, "path": path, "line": number},
        "source": {
            "commit": source_commit, "path": source_path, "line": display_number,
            "blob_sha": _blob_oid(repo, source_commit, source_path, blobs),
        },
        "history_ids": list(raw.get("history_ids", [])),
    }


def _segments(
    repo: Path, raw: Any, blobs: dict[tuple[str, str], str], *,
    before_source: tuple[str | None, str | None],
    after_source: tuple[str | None, str | None],
) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        raise ValueError("Code projection contains invalid diff segments.")
    return [
        {
            "kind": segment.get("kind"),
            "before_start": segment.get("before_start"),
            "after_start": segment.get("after_start"),
            "before": [_line(repo, line, blobs, *before_source) for line in segment.get("before", [])],
            "after": [_line(repo, line, blobs, *after_source) for line in segment.get("after", [])],
            "event_id": segment.get("event_id"),
        }
        for segment in raw
    ]


def _projection(
    repo: Path, raw: dict[str, Any], source_refs: dict[str, Any], *,
    repository_name: str,
) -> dict[str, Any]:
    blobs: dict[tuple[str, str], str] = {}
    revision = raw["revision"]
    diff = raw["diff"]
    history = []
    for event in raw["history"]:
        history.append({
            key: deepcopy(event.get(key))
            for key in (
                "id", "commit", "short_sha", "subject", "committed_at", "parent_sha",
                "before_path", "after_path", "before_start", "before_end", "after_start",
                "after_end", "from_session_ids", "to_session_ids", "cross_model",
                "evidence", "kind", "integration",
            )
        } | {
            "before": [
                _line(repo, line, blobs, event.get("parent_sha"), event.get("before_path"))
                for line in event.get("before", [])
            ],
            "after": [
                _line(repo, line, blobs, event.get("commit"), event.get("after_path"))
                for line in event.get("after", [])
            ],
        })
    reasons = list(dict.fromkeys(str(item)[:512] for item in raw.get("warnings", [])))
    if raw.get("truncated") and not reasons:
        reasons.append("Per-file history reached an inspection limit.")
    return {
        "repository": {
            key: deepcopy(raw["repository"].get(key))
            for key in ("target_ref", "target_commit", "example_data")
        } | {"name": repository_name},
        "path": raw["path"],
        "revision": {
            key: deepcopy(revision.get(key))
            for key in ("sha", "short_sha", "parent_sha", "subject", "committed_at", "path")
        },
        "sessions": {
            session_id: {key: deepcopy(session.get(key)) for key in _SESSION_FIELDS}
            for session_id, session in raw.get("sessions", {}).items()
        },
        "lines": [
            _line(repo, line, blobs, revision.get("sha"), revision.get("path"))
            for line in raw.get("lines", [])
        ],
        "diff": {
            key: deepcopy(diff.get(key))
            for key in (
                "available", "reason", "before_path", "after_path", "before_line_count",
                "after_line_count", "comparison", "merge_base_sha", "head_sha",
            )
        } | {"segments": _segments(
            repo, diff.get("segments", []), blobs,
            before_source=(diff.get("merge_base_sha"), diff.get("before_path")),
            after_source=(diff.get("head_sha"), diff.get("after_path")),
        )},
        "history": history,
        "warnings": reasons,
        "truncated": bool(raw.get("truncated")),
        "scope": deepcopy(raw.get("scope")),
        "source_refs": source_refs,
        "completeness": {
            "status": "partial" if raw.get("truncated") else "complete",
            "reasons": reasons,
        },
    }


def _stub(path: str, source_refs: dict[str, Any], reason: str, *, repository: str, head: str) -> dict[str, Any]:
    return {
        "repository": {"name": repository, "target_ref": head, "target_commit": head, "example_data": False},
        "path": path, "revision": None, "sessions": {}, "lines": [],
        "diff": {
            "available": False, "reason": reason, "before_path": source_refs.get("base", {}).get("path") if source_refs.get("base") else None,
            "after_path": source_refs.get("head", {}).get("path") if source_refs.get("head") else None,
            "before_line_count": None, "after_line_count": None, "comparison": "pr_merge_base_to_head",
            "merge_base_sha": None, "head_sha": head, "segments": [],
        },
        "history": [], "warnings": [reason], "truncated": True, "scope": None,
        "source_refs": source_refs,
        "completeness": {"status": "partial", "reasons": [reason]},
    }


def _source_refs(record: dict[str, Any], merge_base: str, head: str) -> dict[str, Any]:
    def ref(side: str, commit: str) -> dict[str, Any] | None:
        path = record.get(f"{side}_path")
        blob = record.get(f"{side}_blob_sha")
        if path is None or blob is None:
            return None
        return {"commit": commit, "path": path, "blob_sha": blob}
    return {"base": ref("base", merge_base), "head": ref("head", head)}


def _narrative_paragraph(value: Any) -> str | None:
    """Make one bounded, control-free paragraph out of untrusted prose."""
    if not isinstance(value, str):
        return None
    # A model may echo the escaped entities it was shown, and the fallback
    # prose unescapes its own labels. The control pass runs after this, so an
    # entity cannot smuggle a control character past it.
    visible = "".join(
        " " if unicodedata.category(character) in {"Cc", "Cf", "Cs"} else character
        for character in html.unescape(value)
    )
    # The page renders this as a text node, so it is stored as plain text and
    # never escaped for Markdown here.
    return " ".join(visible.split())[:MAX_NARRATIVE_CHARS].strip() or None


def _narrative_fields(
    report: dict[str, Any], supplied: str | None,
) -> tuple[str | None, str | None]:
    """Return the paragraph this artifact carries and where it came from."""
    text = _narrative_paragraph(supplied)
    if text is not None:
        return text, "model"
    text = _narrative_paragraph(fallback_narrative(report))
    if text is not None:
        return text, "local"
    return None, None


def build_artifact(
    repo: str | Path, *, repository_id: str | int, full_name: str, pr_number: int,
    base_sha: str, head_sha: str, metadata_sha: str,
    notes: list[dict[str, Any]], tasks: list[dict[str, Any]],
    narrative_summary: str | None = None,
    usage: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Precompute one immutable PR report and bounded code-history projection."""
    root = Path(repo).expanduser().resolve()
    repo_id = str(repository_id)
    if not repo_id.isascii() or not repo_id.isdecimal() or repo_id.startswith("0") or repo_id == "0":
        raise ValueError("Repository ID must be a positive canonical decimal integer.")
    if not isinstance(full_name, str) or _REPOSITORY.fullmatch(full_name) is None:
        raise ValueError("Repository full name is invalid.")
    if isinstance(pr_number, bool) or not isinstance(pr_number, int) or not 1 <= pr_number <= 999_999_999:
        raise ValueError("Pull request number is invalid.")
    _oid(base_sha, "base SHA")
    _oid(head_sha, "head SHA")
    _oid(metadata_sha, "metadata SHA")
    owner, repository = full_name.split("/", 1)
    summary = build_pr_report(
        root, base_ref=base_sha, head_ref=head_sha, notes=notes, tasks=tasks,
        usage=usage,
    )
    # The paragraph is read from the structured narrative before that narrative
    # is dropped, so the hosted page keeps the account the PR no longer carries.
    narrative, narrative_source = _narrative_fields(summary, narrative_summary)
    # The artifact schema has no workflow field yet. The profile, and the
    # per-agent narrative the paragraph is built from, stay in the local report
    # and the commit note until the hosted schema adds them.
    dropped = {"workflow", "narrative", "narrative_omitted", "unconfirmed_agents"}
    summary = {key: value for key, value in summary.items() if key not in dropped}
    summary["narrative_summary"] = narrative
    summary["narrative_source"] = narrative_source
    listing = list_pr_code_files(
        root, base_sha, head_sha, owner=owner, repository=repository,
        number=pr_number, target_ref=head_sha,
    )
    merge_base = listing["scope"]["merge_base_sha"]
    artifact: dict[str, Any] = {
        "schema": SCHEMA,
        "repository": {"id": repo_id, "full_name": full_name},
        "pull_request": {
            "number": pr_number, "base_sha": base_sha, "head_sha": head_sha,
            "merge_base_sha": merge_base,
        },
        "metadata_sha": metadata_sha,
        "summary": summary,
        "file_index": [],
        "files": {},
        "completeness": {"status": "complete", "partial_files": 0, "reasons": []},
    }
    partial_reasons: list[str] = []
    for index, record in enumerate(listing["files"]):
        path = record["path"]
        refs = _source_refs(record, merge_base, head_sha)
        reason: str | None = None
        if index >= MAX_DETAILED_FILES:
            reason = f"Detailed code history is limited to {MAX_DETAILED_FILES} changed files."
            projection = _stub(path, refs, reason, repository=repository, head=head_sha)
        else:
            try:
                raw = build_pr_code_file(
                    root, path, base_sha, head_sha, owner=owner, repository=repository,
                    number=pr_number, target_ref=head_sha, notes=notes, usage=usage,
                )
                projection = _projection(
                    root, raw, refs, repository_name=repository,
                )
                if projection["completeness"]["status"] == "partial":
                    partial_reasons.extend(projection["completeness"]["reasons"])
            except (OSError, ValueError) as exc:
                reason = "Detailed code history is unavailable: " + str(exc)[:384]
                projection = _stub(path, refs, reason, repository=repository, head=head_sha)
        candidate = deepcopy(artifact)
        candidate["files"][path] = projection
        if len(canonical_json(candidate)) > MAX_ARTIFACT_BYTES:
            reason = "Detailed code history was omitted to keep the artifact within its 3 MiB limit."
            projection = _stub(path, refs, reason, repository=repository, head=head_sha)
        if projection["completeness"]["status"] == "partial":
            partial_reasons.extend(projection["completeness"]["reasons"])
        artifact["files"][path] = projection
        artifact["file_index"].append({
            key: deepcopy(record.get(key))
            for key in (
                "path", "status", "base_path", "head_path", "size",
                "base_blob_sha", "head_blob_sha",
            )
        } | {"completeness": deepcopy(projection["completeness"])})
    partial_files = sum(
        item["completeness"]["status"] == "partial" for item in artifact["file_index"]
    )
    reasons = list(dict.fromkeys([*listing.get("warnings", []), *partial_reasons]))
    artifact["completeness"] = {
        "status": "partial" if partial_files or listing.get("truncated") else "complete",
        "partial_files": partial_files,
        "reasons": [str(item)[:512] for item in reasons[:100]],
    }
    artifact["digest"] = artifact_digest(artifact)
    if len(canonical_json(artifact)) > MAX_ARTIFACT_BYTES:
        raise ValueError("Source-free PR artifact exceeds the 3 MiB hosted request limit.")
    return validate_artifact(artifact)


def _walk_source_free(value: Any) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str) or key in _FORBIDDEN_KEYS:
                raise ValueError("Artifact contains a forbidden source-bearing field.")
            if key in {"path", "before_path", "after_path", "base_path", "head_path"} and item is not None:
                _path(item)
            _walk_source_free(item)
    elif isinstance(value, list):
        for item in value:
            _walk_source_free(item)
    elif isinstance(value, str):
        if "\0" in value or len(value) > 8192:
            raise ValueError("Artifact contains an invalid string value.")


def _keys(
    value: Any, expected: set[str], label: str, *,
    optional: frozenset[str] | set[str] = frozenset(),
) -> dict[str, Any]:
    """Require every expected key and allow only the optional ones beside them."""
    if (
        not isinstance(value, dict) or not expected <= set(value)
        or not set(value) <= expected | optional
    ):
        raise ValueError(f"Artifact {label} has missing or unexpected fields.")
    return value


def _validate_narrative(text: Any, origin: Any) -> None:
    """Accept one bounded paragraph with its source, or neither of them."""
    if text is None and origin is None:
        return
    if (
        not isinstance(text, str) or not text or text != text.strip()
        or len(text) > MAX_NARRATIVE_CHARS
        or any(
            unicodedata.category(character) in {"Cc", "Cf", "Cs"}
            for character in text
        )
        or origin not in {"model", "local"}
    ):
        raise ValueError("Artifact summary narrative is invalid.")


def _strings(value: Any, label: str, *, optional: bool = False) -> list[str]:
    if value is None and optional:
        return []
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise ValueError(f"Artifact {label} must be an array of strings.")
    return value


def _validate_completeness(value: Any, label: str) -> dict[str, Any]:
    result = _keys(value, {"status", "reasons"}, label)
    if result["status"] not in {"complete", "partial"}:
        raise ValueError(f"Artifact {label} status is invalid.")
    _strings(result["reasons"], f"{label} reasons")
    return result


def _nonnegative_int(value: Any, label: str, *, optional: bool = False) -> None:
    if value is None and optional:
        return
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"Artifact {label} must be a nonnegative integer.")


def _nonnegative_number(value: Any, label: str, *, optional: bool = False) -> None:
    if value is None and optional:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"Artifact {label} must be a nonnegative finite number.")


def _validate_line(value: Any, sessions: set[str]) -> None:
    line = _keys(
        value, {"number", "has_newline", "session_id", "origin", "source", "history_ids"},
        "line projection",
    )
    if (
        isinstance(line["number"], bool) or not isinstance(line["number"], int)
        or line["number"] < 1 or type(line["has_newline"]) is not bool
        or (line["session_id"] is not None and line["session_id"] not in sessions)
    ):
        raise ValueError("Artifact line projection is invalid.")
    origin = _keys(line["origin"], {"commit", "path", "line"}, "line origin")
    _oid(origin["commit"], "line origin commit")
    _path(origin["path"])
    if isinstance(origin["line"], bool) or not isinstance(origin["line"], int) or origin["line"] < 1:
        raise ValueError("Artifact line origin is invalid.")
    source = _keys(line["source"], {"commit", "path", "line", "blob_sha"}, "line source")
    _oid(source["commit"], "line source commit")
    _oid(source["blob_sha"], "line source blob")
    _path(source["path"])
    if source["line"] != line["number"]:
        raise ValueError("Artifact line source coordinate is invalid.")
    _strings(line["history_ids"], "line history IDs")


def _validate_ref(value: Any, label: str) -> None:
    if value is None:
        return
    ref = _keys(value, {"commit", "path", "blob_sha"}, label)
    _oid(ref["commit"], f"{label} commit")
    _oid(ref["blob_sha"], f"{label} blob")
    _path(ref["path"])


def _validate_projection(
    value: Any, expected_path: str, pr: dict[str, Any], *, repository_name: str,
) -> None:
    projection = _keys(value, {
        "repository", "path", "revision", "sessions", "lines", "diff", "history",
        "warnings", "truncated", "scope", "source_refs", "completeness",
    }, "file projection")
    if projection["path"] != expected_path or type(projection["truncated"]) is not bool:
        raise ValueError("Artifact file projection does not match its index.")
    projected_repository = _keys(
        projection["repository"], {"name", "target_ref", "target_commit", "example_data"},
        "projected repository",
    )
    if (
        projected_repository["name"] != repository_name
        or projected_repository["target_commit"] != pr["head_sha"]
        or not isinstance(projected_repository["target_ref"], str)
        or type(projected_repository["example_data"]) is not bool
    ):
        raise ValueError("Artifact projected repository identity is invalid.")
    if projection["revision"] is not None:
        revision = _keys(
            projection["revision"],
            {"sha", "short_sha", "parent_sha", "subject", "committed_at", "path"},
            "file revision",
        )
        _oid(revision["sha"], "file revision")
        _oid(revision["parent_sha"], "file revision parent", optional=True)
        _path(revision["path"])
    sessions = projection["sessions"]
    if not isinstance(sessions, dict) or len(sessions) > 10_000:
        raise ValueError("Artifact projected sessions are invalid.")
    for session_id, session in sessions.items():
        if not isinstance(session_id, str) or not session_id or len(session_id) > 2048:
            raise ValueError("Artifact projected session ID is invalid.")
        record = _keys(session, set(_SESSION_FIELDS), "projected session")
        if record["id"] != session_id:
            raise ValueError("Artifact projected session identity is invalid.")
        required_text = ("feature", "model", "harness", "label_source", "membership_source", "started_at")
        if any(not isinstance(record[key], str) or not record[key] for key in required_text):
            raise ValueError("Artifact projected session labels are invalid.")
        for key in ("task_id", "source_session_id", "parent_session_id", "token_source", "cost_source", "ended_at", "outcome"):
            if record[key] is not None and (not isinstance(record[key], str) or not record[key]):
                raise ValueError("Artifact projected session metadata is invalid.")
        if record["actor_kind"] not in {"ai", "manual"} or record["role"] not in {"planning", "implementation", "testing", "review", "other"}:
            raise ValueError("Artifact projected session classification is invalid.")
        _nonnegative_int(record["token_count"], "session token count", optional=True)
        _nonnegative_number(record["cost_usd"], "session cost", optional=True)
        if type(record["usage_includes_children"]) is not bool:
            raise ValueError("Artifact projected session usage marker is invalid.")
        if record["exit_code"] is not None and (isinstance(record["exit_code"], bool) or not isinstance(record["exit_code"], int)):
            raise ValueError("Artifact projected session exit code is invalid.")
    session_ids = set(sessions)
    if not isinstance(projection["lines"], list):
        raise ValueError("Artifact projected lines are invalid.")
    for line in projection["lines"]:
        _validate_line(line, session_ids)
    diff = _keys(projection["diff"], {
        "available", "reason", "before_path", "after_path", "before_line_count",
        "after_line_count", "comparison", "merge_base_sha", "head_sha", "segments",
    }, "projected diff")
    if type(diff["available"]) is not bool or not isinstance(diff["segments"], list):
        raise ValueError("Artifact projected diff is invalid.")
    _nonnegative_int(diff["before_line_count"], "diff before line count", optional=True)
    _nonnegative_int(diff["after_line_count"], "diff after line count", optional=True)
    _path(diff["before_path"], optional=True)
    _path(diff["after_path"], optional=True)
    _oid(diff["merge_base_sha"], "projected merge base", optional=True)
    _oid(diff["head_sha"], "projected head", optional=True)
    for segment in diff["segments"]:
        item = _keys(
            segment, {"kind", "before_start", "after_start", "before", "after", "event_id"},
            "diff segment",
        )
        if item["kind"] not in {"context", "change"}:
            raise ValueError("Artifact diff segment kind is invalid.")
        _nonnegative_int(item["before_start"], "diff before start")
        _nonnegative_int(item["after_start"], "diff after start")
        if not isinstance(item["before"], list) or not isinstance(item["after"], list):
            raise ValueError("Artifact diff segment lines are invalid.")
        for line in [*item["before"], *item["after"]]:
            _validate_line(line, session_ids)
    if not isinstance(projection["history"], list):
        raise ValueError("Artifact projected history is invalid.")
    event_ids: set[str] = set()
    for event in projection["history"]:
        item = _keys(event, {
            "id", "commit", "short_sha", "subject", "committed_at", "parent_sha",
            "before_path", "after_path", "before_start", "before_end", "after_start",
            "after_end", "from_session_ids", "to_session_ids", "cross_model", "evidence",
            "kind", "integration", "before", "after",
        }, "history event")
        if not isinstance(item["id"], str) or not item["id"] or item["id"] in event_ids:
            raise ValueError("Artifact history event identity is invalid.")
        event_ids.add(item["id"])
        _oid(item["commit"], "history commit")
        _oid(item["parent_sha"], "history parent", optional=True)
        _path(item["before_path"], optional=True)
        _path(item["after_path"], optional=True)
        for key in ("before_start", "before_end", "after_start", "after_end"):
            _nonnegative_int(item[key], f"history {key}", optional=True)
        from_ids = _strings(item["from_session_ids"], "history source sessions")
        to_ids = _strings(item["to_session_ids"], "history target sessions")
        if not set(from_ids + to_ids) <= session_ids:
            raise ValueError("Artifact history references an unknown session.")
        if type(item["cross_model"]) is not bool or type(item["integration"]) is not bool:
            raise ValueError("Artifact history flags are invalid.")
        if item["evidence"] not in {"recorded", "inferred", "unattributed"} or item["kind"] not in {"renamed", "modified", "added", "deleted"}:
            raise ValueError("Artifact history classification is invalid.")
        if not isinstance(item["before"], list) or not isinstance(item["after"], list):
            raise ValueError("Artifact history lines are invalid.")
        for line in [*item["before"], *item["after"]]:
            _validate_line(line, session_ids)
    _strings(projection["warnings"], "projection warnings")
    if projection["scope"] is not None:
        scope = _keys(projection["scope"], {
            "owner", "repository", "number", "requested_base_sha", "merge_base_sha",
            "head_sha", "configured_target_sha", "freshness", "stale",
        }, "projection scope")
        if scope["number"] != pr["number"] or scope["requested_base_sha"] != pr["base_sha"] or scope["merge_base_sha"] != pr["merge_base_sha"] or scope["head_sha"] != pr["head_sha"]:
            raise ValueError("Artifact projection scope does not match the PR identity.")
    refs = _keys(projection["source_refs"], {"base", "head"}, "source references")
    _validate_ref(refs["base"], "base source reference")
    _validate_ref(refs["head"], "head source reference")
    if refs["base"] is not None and refs["base"]["commit"] != pr["merge_base_sha"]:
        raise ValueError("Artifact base source reference is not merge-base pinned.")
    if refs["head"] is not None and refs["head"]["commit"] != pr["head_sha"]:
        raise ValueError("Artifact head source reference is not head pinned.")
    _validate_completeness(projection["completeness"], "projection completeness")


def validate_artifact(value: Any, *, require_digest: bool = True) -> dict[str, Any]:
    """Validate identity, source exclusion, canonical digest, and request size."""
    if not isinstance(value, dict):
        raise ValueError("Artifact must be an object.")
    required = {
        "schema", "repository", "pull_request", "metadata_sha", "summary",
        "file_index", "files", "completeness", "digest",
    }
    if set(value) != required:
        raise ValueError("Artifact has missing or unexpected top-level fields.")
    if value.get("schema") != SCHEMA:
        raise ValueError("Artifact schema is unsupported.")
    repository = value.get("repository")
    if not isinstance(repository, dict) or set(repository) != {"id", "full_name"}:
        raise ValueError("Artifact repository identity is invalid.")
    repo_id = repository.get("id")
    if not isinstance(repo_id, str) or not repo_id.isascii() or not repo_id.isdecimal() or repo_id.startswith("0") or repo_id == "0":
        raise ValueError("Artifact repository ID is invalid.")
    if not isinstance(repository.get("full_name"), str) or _REPOSITORY.fullmatch(repository["full_name"]) is None:
        raise ValueError("Artifact repository full name is invalid.")
    pr = value.get("pull_request")
    if not isinstance(pr, dict) or set(pr) != {"number", "base_sha", "head_sha", "merge_base_sha"}:
        raise ValueError("Artifact PR identity is invalid.")
    if isinstance(pr.get("number"), bool) or not isinstance(pr.get("number"), int) or not 1 <= pr["number"] <= 999_999_999:
        raise ValueError("Artifact PR number is invalid.")
    for key in ("base_sha", "head_sha", "merge_base_sha"):
        _oid(pr.get(key), key)
    _oid(value.get("metadata_sha"), "metadata SHA")
    summary = value.get("summary")
    summary = _keys(summary, {
        "base_commit", "merge_base", "head_commit", "added_lines", "counts", "sources",
        "session_count", "ai_session_count", "reported_cost_usd", "cost_complete",
        "total_tokens", "tokens_complete", "complete", "warnings",
    }, "summary", optional=set(_UNSIGNED_SUMMARY_KEYS) | {
        "estimated_cost_usd", "codex_credits", "codex_api_equivalent_usd",
    })
    _validate_narrative(
        summary.get("narrative_summary"), summary.get("narrative_source")
    )
    if summary.get("base_commit") != pr["base_sha"] or summary.get("head_commit") != pr["head_sha"] or summary.get("merge_base") != pr["merge_base_sha"]:
        raise ValueError("Artifact report does not match its immutable PR identity.")
    counts = _keys(summary["counts"], {"ai", "manual", "unknown"}, "summary counts")
    if any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in counts.values()):
        raise ValueError("Artifact summary counts are invalid.")
    if summary["added_lines"] != sum(counts.values()):
        raise ValueError("Artifact summary line count is inconsistent.")
    for key in ("session_count", "ai_session_count"):
        _nonnegative_int(summary[key], f"summary {key}")
    _nonnegative_number(summary["reported_cost_usd"], "summary cost", optional=True)
    # A report carries this key only where a cost was estimated rather than
    # reported, so an artifact built from notes alone keeps the shape it had.
    _nonnegative_number(
        summary.get("estimated_cost_usd"), "summary estimated cost", optional=True
    )
    # A Codex subscription session prices in credits, so a report can carry a
    # credit total and its standard-rate dollar comparison beside the estimate.
    _nonnegative_number(
        summary.get("codex_credits"), "summary credits", optional=True
    )
    _nonnegative_number(
        summary.get("codex_api_equivalent_usd"), "summary credit comparison",
        optional=True,
    )
    _nonnegative_int(summary["total_tokens"], "summary token count", optional=True)
    for key in ("cost_complete", "tokens_complete", "complete"):
        if type(summary[key]) is not bool:
            raise ValueError("Artifact summary completeness flag is invalid.")
    if not isinstance(summary["sources"], list):
        raise ValueError("Artifact summary sources are invalid.")
    source_fields = {
        "actor_kind", "model", "harness", "lines", "session_count", "reported_cost_usd",
        "cost_complete", "cost_in_parent", "total_tokens", "tokens_complete", "tokens_in_parent",
    }
    for source in summary["sources"]:
        record = _keys(
            source, source_fields, "summary source",
            optional={
                "estimated_cost_usd", "codex_credits", "codex_api_equivalent_usd",
            },
        )
        if record["actor_kind"] not in {"ai", "manual"}:
            raise ValueError("Artifact summary source actor is invalid.")
        if not isinstance(record["model"], str) or not record["model"] or not isinstance(record["harness"], str):
            raise ValueError("Artifact summary source label is invalid.")
        _nonnegative_int(record["lines"], "summary source lines")
        _nonnegative_int(record["session_count"], "summary source session count")
        _nonnegative_number(record["reported_cost_usd"], "summary source cost", optional=True)
        _nonnegative_number(
            record.get("estimated_cost_usd"), "summary source estimated cost",
            optional=True,
        )
        _nonnegative_number(
            record.get("codex_credits"), "summary source credits", optional=True
        )
        _nonnegative_number(
            record.get("codex_api_equivalent_usd"),
            "summary source credit comparison", optional=True,
        )
        _nonnegative_int(record["total_tokens"], "summary source tokens", optional=True)
        for key in ("cost_complete", "cost_in_parent", "tokens_complete", "tokens_in_parent"):
            if type(record[key]) is not bool:
                raise ValueError("Artifact summary source completeness flag is invalid.")
    _strings(summary["warnings"], "summary warnings")
    file_index, files = value.get("file_index"), value.get("files")
    if not isinstance(file_index, list) or not isinstance(files, dict) or len(file_index) != len(files):
        raise ValueError("Artifact file projection is invalid.")
    paths = []
    for item in file_index:
        item = _keys(item, {
            "path", "status", "base_path", "head_path", "size", "base_blob_sha",
            "head_blob_sha", "completeness",
        }, "file index entry")
        path = _path(item.get("path"))
        if path in paths or path not in files or not isinstance(files[path], dict):
            raise ValueError("Artifact file paths must be unique and indexed.")
        paths.append(path)
        for key in ("base_blob_sha", "head_blob_sha"):
            _oid(item.get(key), key, optional=True)
        _path(item.get("base_path"), optional=True)
        _path(item.get("head_path"), optional=True)
        _nonnegative_int(item.get("size"), "file size")
        if not isinstance(item.get("status"), str) or not re.fullmatch(r"(?:[AMDT]|[RC][0-9]{1,3})", item["status"]):
            raise ValueError("Artifact file status is invalid.")
        _validate_completeness(item.get("completeness"), "file completeness")
        projection = files[path]
        _validate_projection(
            projection, path, pr,
            repository_name=repository["full_name"].split("/", 1)[1],
        )
    completeness = value.get("completeness")
    completeness = _keys(
        completeness, {"status", "partial_files", "reasons"}, "completeness",
    )
    if completeness.get("status") not in {"complete", "partial"}:
        raise ValueError("Artifact completeness is invalid.")
    if isinstance(completeness["partial_files"], bool) or not isinstance(completeness["partial_files"], int) or not 0 <= completeness["partial_files"] <= len(files):
        raise ValueError("Artifact partial file count is invalid.")
    expected_partial = sum(
        item["completeness"]["status"] == "partial" for item in file_index
    )
    if completeness["partial_files"] != expected_partial or (
        completeness["status"] == "complete" and expected_partial != 0
    ):
        raise ValueError("Artifact completeness does not match its file projections.")
    _strings(completeness["reasons"], "artifact completeness reasons")
    _walk_source_free(value)
    digest = value.get("digest")
    if require_digest and (not isinstance(digest, str) or digest != artifact_digest(value)):
        raise ValueError("Artifact digest does not match its canonical content.")
    encoded = canonical_json(value)
    if len(encoded) > MAX_ARTIFACT_BYTES:
        raise ValueError("Artifact exceeds the 3 MiB hosted request limit.")
    return json.loads(encoded)
