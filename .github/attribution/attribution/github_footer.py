"""Update one PR-body footer using trusted code and untrusted Git data only.

Run this module from the base repository's trusted workflow checkout. PR source
is fetched into a temporary bare repository and is never checked out or run.
"""

from __future__ import annotations

import base64
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qsl, quote, urlencode, urlsplit, urlunsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from .runtime import system_subprocess_environment


MAX_SNAPSHOT_BYTES = 8 * 1024 * 1024
_MAX_HTTP_BYTES = 16 * 1024 * 1024
_OID = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
_REPOSITORY = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_UI_SEGMENT = re.compile(r"^[A-Za-z0-9_.-]{1,100}$")
_METADATA_PREFIX = "refs/notes/attribution-pr/"
_OIDC_HTTP_BYTES = 1024 * 1024
_NARRATIVE_AUTHORS = frozenset({"OWNER", "MEMBER", "COLLABORATOR"})
# The allocated telemetry a snapshot may carry for one of its sessions. The
# clone that pushed the branch wrote it, so every value is untrusted here and
# is rebuilt from these names within these bounds.
_USAGE_COUNT_FIELDS = (
    "input_tokens", "cached_input_tokens", "cache_creation_input_tokens",
    "output_tokens", "total_tokens", "request_count", "priced_request_count",
    "unpriced_request_count",
)
_USAGE_ALLOCATIONS = frozenset({
    "agent-routed", "tool-linked", "turn-linked", "session-allocated", "mixed",
})
_MAX_USAGE_SESSIONS = 500
_MAX_USAGE_MODELS = 8
_MAX_USAGE_MODEL_CHARS = 120


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Never forward the workflow token to a redirected host.
        return None


def _host_url(value: str, *, api: bool = False) -> str:
    parsed = urlsplit(value)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or (not api and parsed.path not in {"", "/"})
    ):
        raise ValueError("GitHub server and API URLs must be trusted HTTPS URLs.")
    return value.rstrip("/")


def _ingest_url(value: str) -> tuple[str, str]:
    """Validate the configured endpoint and return it with its OIDC audience."""
    parsed = urlsplit(value.strip())
    try:
        parsed.port
    except ValueError as exc:
        raise ValueError("ATTRIBUTION_INGEST_URL has an invalid port.") from exc
    loopback = parsed.scheme == "http" and parsed.hostname in {"127.0.0.1", "localhost"}
    if (
        (parsed.scheme != "https" and not loopback)
        or not parsed.hostname or parsed.username is not None or parsed.password is not None
        or parsed.query or parsed.fragment or parsed.path != "/v1/artifacts/pr"
    ):
        raise ValueError("ATTRIBUTION_INGEST_URL must be an HTTPS /v1/artifacts/pr endpoint.")
    audience = f"{parsed.scheme}://{parsed.netloc}"
    return value.strip(), audience


def _oidc_token(request_url: str, request_token: str, audience: str) -> str:
    if not request_url or not request_token:
        raise ValueError("GitHub Actions OIDC is unavailable for artifact publication.")
    parsed = urlsplit(request_url)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password or parsed.fragment:
        raise ValueError("GitHub Actions supplied an invalid OIDC request URL.")
    query = parse_qsl(parsed.query, keep_blank_values=True)
    query.append(("audience", audience))
    url = urlunsplit((parsed.scheme, parsed.netloc, parsed.path, urlencode(query), ""))
    request = Request(
        url, method="GET",
        headers={"Authorization": f"Bearer {request_token}", "User-Agent": "attribution-pr-footer"},
    )
    try:
        with build_opener(_NoRedirect()).open(request, timeout=30) as response:
            raw = response.read(_OIDC_HTTP_BYTES + 1)
    except HTTPError as exc:
        raise ValueError(f"GitHub Actions OIDC request failed (HTTP {exc.code}).") from exc
    except (URLError, TimeoutError) as exc:
        raise ValueError("GitHub Actions OIDC request could not be completed.") from exc
    if len(raw) > _OIDC_HTTP_BYTES:
        raise ValueError("GitHub Actions OIDC response exceeded the size limit.")
    try:
        payload = json.loads(raw)
    except (ValueError, UnicodeError) as exc:
        raise ValueError("GitHub Actions OIDC returned invalid JSON.") from exc
    token = payload.get("value") if isinstance(payload, dict) else None
    if not isinstance(token, str) or not token or len(token) > 16_384 or "\0" in token:
        raise ValueError("GitHub Actions OIDC returned an invalid token.")
    return token


def _publish_artifact(
    ingest_url: str, artifact: dict[str, Any], *, oidc_request_url: str,
    oidc_request_token: str, repository: str, repository_id: int,
    workflow_ref: str, delivery_id: str,
) -> dict[str, Any]:
    from .hosted_artifact import artifact_bytes

    endpoint, audience = _ingest_url(ingest_url)
    if (
        not isinstance(repository_id, int) or isinstance(repository_id, bool) or repository_id < 1
        or not workflow_ref or len(workflow_ref) > 2048
        or any(character in workflow_ref for character in "\r\n\0")
        or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,200}", delivery_id or "")
    ):
        raise ValueError("Artifact publication identity is invalid.")
    digest = artifact.get("digest")
    if not isinstance(digest, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
        raise ValueError("Artifact publication digest is invalid.")
    bound_delivery = f"{delivery_id}:{digest[7:39]}"
    if len(bound_delivery) > 200:
        raise ValueError("Artifact publication delivery identifier is too long.")
    token = _oidc_token(oidc_request_url, oidc_request_token, audience)
    request = Request(
        endpoint, data=artifact_bytes(artifact), method="POST",
        headers={
            "Accept": "application/json", "Authorization": f"Bearer {token}",
            "Content-Type": "application/json", "User-Agent": "attribution-pr-footer",
            "X-GitHub-Repository": repository,
            "X-GitHub-Repository-ID": str(repository_id),
            "X-GitHub-Workflow-Ref": workflow_ref,
            "X-Attribution-Delivery": bound_delivery,
        },
    )
    try:
        with build_opener(_NoRedirect()).open(request, timeout=60) as response:
            raw = response.read(_OIDC_HTTP_BYTES + 1)
    except HTTPError as exc:
        raise ValueError(f"Artifact publication failed (HTTP {exc.code}).") from exc
    except (URLError, TimeoutError) as exc:
        raise ValueError("Artifact publication could not be completed.") from exc
    if len(raw) > _OIDC_HTTP_BYTES:
        raise ValueError("Artifact publication response exceeded the size limit.")
    try:
        payload = json.loads(raw or b"{}")
    except (ValueError, UnicodeError) as exc:
        raise ValueError("Artifact publication returned invalid JSON.") from exc
    if not isinstance(payload, dict):
        raise ValueError("Artifact publication returned an invalid response.")
    if payload.get("status") not in {"created", "unchanged", "duplicate"} or payload.get("digest") != digest:
        raise ValueError("Artifact publication response did not confirm the immutable digest.")
    return payload


def _api(url: str, token: str, *, body: dict[str, str] | None = None) -> dict[str, Any]:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    request = Request(
        url,
        data=data,
        method="PATCH" if body is not None else "GET",
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "User-Agent": "attribution-pr-footer",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    try:
        with build_opener(_NoRedirect()).open(request, timeout=30) as response:
            raw = response.read(_MAX_HTTP_BYTES + 1)
    except HTTPError as exc:
        raise ValueError(f"GitHub API {request.method} failed (HTTP {exc.code}).") from exc
    except (URLError, TimeoutError) as exc:
        raise ValueError(f"GitHub API {request.method} could not be completed.") from exc
    if len(raw) > _MAX_HTTP_BYTES:
        raise ValueError("GitHub API response exceeded the inspection limit.")
    try:
        result = json.loads(raw)
    except (ValueError, UnicodeError) as exc:
        raise ValueError("GitHub returned invalid JSON.") from exc
    if not isinstance(result, dict):
        raise ValueError("GitHub returned an unexpected PR response.")
    return result


def _repository_name(value: Any) -> str:
    if (
        not isinstance(value, str)
        or not _REPOSITORY.fullmatch(value)
        or any(part in {".", ".."} for part in value.split("/"))
    ):
        raise ValueError("PR repository identity is missing or invalid.")
    return value


def _identity(pr: dict[str, Any]) -> tuple[Any, ...]:
    try:
        number = pr["number"]
        if not isinstance(number, int) or isinstance(number, bool) or number < 1:
            raise ValueError("Invalid PR number.")
        result: list[Any] = [number]
        for side in ("base", "head"):
            item = pr[side]
            repository = _repository_name(item["repo"]["full_name"])
            sha, ref = item["sha"], item["ref"]
            if not isinstance(sha, str) or not _OID.fullmatch(sha):
                raise ValueError("Invalid PR commit identifier.")
            if not isinstance(ref, str) or not ref or any(ord(c) < 32 for c in ref):
                raise ValueError("Invalid PR branch name.")
            result.extend((repository.lower(), sha, ref))
        return tuple(result)
    except (KeyError, TypeError) as exc:
        raise ValueError("PR head or base identity is unavailable.") from exc


def _details_url(base_url: str | None, pr: dict[str, Any]) -> str | None:
    """Build one immutable UI route from trusted configuration and PR identity."""
    if base_url is None or not base_url.strip():
        return None
    value = base_url.strip().rstrip("/")
    parsed = urlsplit(value)
    try:
        parsed.port
    except ValueError as exc:
        raise ValueError("ATTRIBUTION_UI_URL has an invalid port.") from exc
    loopback_http = parsed.scheme == "http" and parsed.hostname in {
        "127.0.0.1", "localhost"
    }
    if (
        (parsed.scheme != "https" and not loopback_http)
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/"}
        or any(
            character.isspace() or ord(character) < 32 or ord(character) == 127
            for character in value
        )
        or "\\" in value
        or any(character in "[]()" for character in value)
    ):
        raise ValueError(
            "ATTRIBUTION_UI_URL must be an HTTPS base URL, or an explicit loopback HTTP URL."
        )
    number, repository, base, head = _identity(pr)[0], _repository_name(
        pr["base"]["repo"]["full_name"]
    ), pr["base"]["sha"], pr["head"]["sha"]
    owner, name = repository.split("/", 1)
    if (
        number > 999_999_999
        or not _UI_SEGMENT.fullmatch(owner)
        or not _UI_SEGMENT.fullmatch(name)
        or owner in {".", ".."}
        or name in {".", ".."}
    ):
        raise ValueError(
            "The PR identity cannot be represented by the attribution UI route."
        )
    route = "/".join(
        quote(str(component), safe="")
        for component in (owner, name, number, base, head)
    )
    return f"{value}/pr/{route}"


def _git_environment(server_url: str, token: str) -> dict[str, str]:
    environment = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    authorization = base64.b64encode(f"x-access-token:{token}".encode()).decode("ascii")
    settings = [
        ("protocol.allow", "never"),
        ("protocol.https.allow", "always"),
        ("http.followRedirects", "false"),
        (f"http.{server_url}/.extraHeader", f"Authorization: Basic {authorization}"),
        ("core.hooksPath", os.devnull),
    ]
    environment.update(
        GIT_TERMINAL_PROMPT="0",
        GIT_CONFIG_NOSYSTEM="1",
        GIT_CONFIG_GLOBAL=os.devnull,
        GIT_NO_REPLACE_OBJECTS="1",
        GIT_CONFIG_COUNT=str(len(settings)),
    )
    for index, (key, value) in enumerate(settings):
        environment[f"GIT_CONFIG_KEY_{index}"] = key
        environment[f"GIT_CONFIG_VALUE_{index}"] = value
    return environment


@contextmanager
def _report_git_environment(environment: dict[str, str]):
    """Apply the same Git isolation to report helpers in this single-job process."""
    previous = {key: value for key, value in os.environ.items() if key.startswith("GIT_")}
    try:
        for key in previous:
            del os.environ[key]
        os.environ.update({key: value for key, value in environment.items() if key.startswith("GIT_")})
        yield
    finally:
        for key in list(os.environ):
            if key.startswith("GIT_"):
                del os.environ[key]
        os.environ.update(previous)


def _git(repo: Path, environment: dict[str, str], *arguments: str, check: bool = True):
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), *arguments],
            env=system_subprocess_environment(environment),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=120,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ValueError("Git could not inspect the PR objects.") from exc
    if check and result.returncode:
        # Remote errors can contain attacker-controlled text; do not echo it.
        raise ValueError(f"Git {arguments[0]} failed while inspecting the PR objects.")
    return result


def _usage_model(value: Any) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value) > _MAX_USAGE_MODEL_CHARS
        or any(character < " " or character == "\x7f" for character in value)
    ):
        raise ValueError("Joyride usage names an unpublishable model.")
    return value


def _usage_count(value: Any) -> int | None:
    if value is None:
        return None
    if type(value) is not int or value < 0:
        raise ValueError("Joyride usage carries an invalid count.")
    return value


def _usage_record(source: dict[str, Any]) -> dict[str, Any]:
    record: dict[str, Any] = {}
    cost = source.get("estimated_cost_usd")
    if cost is not None:
        if (
            isinstance(cost, bool) or not isinstance(cost, (int, float))
            or not math.isfinite(cost) or cost < 0
        ):
            raise ValueError("Joyride usage carries an invalid cost.")
        record["estimated_cost_usd"] = float(cost)
    for credit_field in ("codex_credits", "codex_api_equivalent_usd"):
        # A Codex subscription session prices in credits; the footer reads them
        # and their standard-rate dollar comparison from the same snapshot.
        amount = source.get(credit_field)
        if amount is not None:
            if (
                isinstance(amount, bool) or not isinstance(amount, (int, float))
                or not math.isfinite(amount) or amount < 0
            ):
                raise ValueError("Joyride usage carries an invalid cost.")
            record[credit_field] = float(amount)
    models = source.get("models")
    if models is not None:
        if not isinstance(models, list) or len(models) > _MAX_USAGE_MODELS:
            raise ValueError("Joyride usage carries invalid models.")
        record["models"] = [_usage_model(value) for value in models]
    outputs = source.get("model_output_tokens")
    if outputs is not None:
        if not isinstance(outputs, dict) or len(outputs) > _MAX_USAGE_MODELS:
            raise ValueError("Joyride usage carries invalid model tokens.")
        counted: dict[str, float] = {}
        for name, value in outputs.items():
            # A share of a request, published with its decimals so the footer
            # names the same dominant model the local report does.
            if (
                isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value < 0
            ):
                raise ValueError("Joyride usage carries invalid model tokens.")
            counted[_usage_model(name)] = float(value)
        record["model_output_tokens"] = counted
    for field in _USAGE_COUNT_FIELDS:
        count = _usage_count(source.get(field))
        if count is not None:
            record[field] = count
    complete = source.get("cost_complete")
    if complete is not None:
        if type(complete) is not bool:
            raise ValueError("Joyride usage carries an invalid cost marker.")
        record["cost_complete"] = complete
    allocation = source.get("allocation")
    if allocation is not None:
        if allocation not in _USAGE_ALLOCATIONS:
            raise ValueError("Joyride usage carries an invalid allocation.")
        record["allocation"] = allocation
    return record


def _snapshot_usage(raw: Any) -> dict[str, dict[str, Any]]:
    """Return the allocated telemetry of a snapshot, dropping what it may not carry.

    The notes of a snapshot are evidence a footer refuses to render when it is
    malformed. This object is optional beside them: an entry outside its
    published bounds is dropped, and a whole object that is not a map of
    records is ignored, so the footer the notes alone describe still renders.
    """
    if not isinstance(raw, dict):
        return {}
    usage: dict[str, dict[str, Any]] = {}
    for session_id, source in raw.items():
        if (
            not isinstance(session_id, str) or not session_id or len(session_id) > 256
            or not isinstance(source, dict)
        ):
            continue
        try:
            record = _usage_record(source)
        except ValueError:
            continue
        if not record:
            continue
        usage[session_id] = record
        if len(usage) >= _MAX_USAGE_SESSIONS:
            break
    return usage


def _read_traces(repo: Path, environment: dict[str, str], tree: str) -> list[dict[str, Any]]:
    """Return the valid traces of a snapshot's ``traces`` directory.

    A trace is optional evidence beside the metadata, so one that is oversized,
    malformed, or misnamed is skipped with a warning, and reading stops at the
    directory's byte budget, instead of failing the footer.
    """

    from .traces import MAX_SNAPSHOT_TRACE_BYTES, MAX_TRACE_BYTES, validate_trace

    traces: list[dict[str, Any]] = []
    total = 0
    for entry in _git(repo, environment, "ls-tree", "-z", tree).stdout.rstrip(b"\x00").split(b"\x00"):
        if not entry:
            continue
        try:
            fields, name = entry.split(b"\t", 1)
            mode, kind, blob = fields.decode("ascii").split()
        except (ValueError, UnicodeError) as exc:
            raise ValueError("Joyride traces have an invalid tree.") from exc
        if mode != "100644" or kind != "blob" or not _OID.fullmatch(blob) or not name.endswith(b".json"):
            raise ValueError("Joyride traces must be ordinary JSON files.")
        label = name[:-5].decode("utf-8", "replace")
        size = int(_git(repo, environment, "cat-file", "-s", blob).stdout)
        if size > MAX_TRACE_BYTES or total + size > MAX_SNAPSHOT_TRACE_BYTES:
            print(f"Joyride: skipped the oversized trace of session {label}.", file=sys.stderr)
            continue
        total += size
        try:
            trace = validate_trace(json.loads(_git(repo, environment, "cat-file", "blob", blob).stdout))
        except (ValueError, UnicodeError):
            print(f"Joyride: skipped an invalid trace for session {label}.", file=sys.stderr)
            continue
        if trace["session_id"] != label:
            print(f"Joyride: skipped a misnamed trace for session {label}.", file=sys.stderr)
            continue
        traces.append(trace)
    return traces


def _read_snapshot(
    repo: Path, environment: dict[str, str], commit: str, head: str
) -> tuple[
    list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, Any]], list[dict[str, Any]]
]:
    tree = _git(repo, environment, "ls-tree", "-z", commit).stdout
    entries = []
    for entry in tree.rstrip(b"\x00").split(b"\x00"):
        try:
            fields, name = entry.split(b"\t", 1)
            entries.append((*fields.decode("ascii").split(), name))
        except (ValueError, UnicodeError) as exc:
            raise ValueError("Joyride metadata has an invalid tree.") from exc
    # The snapshot is one JSON file, beside an optional ``traces`` directory
    # that holds one file per session and is read on its own.
    metadata = [entry for entry in entries if entry[3] == b"attribution.json"]
    others = [entry for entry in entries if entry[3] != b"attribution.json"]
    if len(metadata) != 1 or len(others) > 1 or any(
        entry[:2] != ("040000", "tree") or entry[3] != b"traces" for entry in others
    ):
        raise ValueError("Joyride metadata must contain only attribution.json and traces.")
    mode, kind, blob, name = metadata[0]
    if mode != "100644" or kind != "blob" or not _OID.fullmatch(blob):
        raise ValueError("Joyride metadata must contain one ordinary JSON file.")
    traces = _read_traces(repo, environment, others[0][2]) if others else []
    size = int(_git(repo, environment, "cat-file", "-s", blob).stdout)
    if size > MAX_SNAPSHOT_BYTES:
        raise ValueError("Joyride metadata exceeds the 8 MiB limit.")
    raw = _git(repo, environment, "cat-file", "blob", blob).stdout
    if len(raw) > MAX_SNAPSHOT_BYTES:
        raise ValueError("Joyride metadata exceeds the 8 MiB limit.")
    try:
        payload = json.loads(raw)
    except (ValueError, UnicodeError) as exc:
        raise ValueError("Joyride metadata is not valid JSON.") from exc
    if (
        not isinstance(payload, dict)
        or type(payload.get("version")) is not int
        or payload["version"] != 1
        or payload.get("head_commit") != head
        or not isinstance(payload.get("notes"), list)
        or any(not isinstance(note, dict) for note in payload["notes"])
        or not isinstance(payload.get("tasks", []), list)
        or any(not isinstance(task, dict) for task in payload.get("tasks", []))
    ):
        raise ValueError("Joyride metadata does not match this PR head.")
    return payload["notes"], payload.get("tasks", []), _snapshot_usage(payload.get("usage")), traces


def _report_and_artifact_for_pr(
    pr: dict[str, Any],
    server_url: str,
    token: str,
    *,
    repository_id: int | None = None,
    narrative_api_key: str = "",
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, list[dict[str, Any]]]:
    from .pr_report import build_pr_report

    head = pr["head"]["sha"]
    base = pr["base"]["sha"]
    head_url = f"{server_url}/{_repository_name(pr['head']['repo']['full_name'])}.git"
    base_url = f"{server_url}/{_repository_name(pr['base']['repo']['full_name'])}.git"
    metadata_ref = _METADATA_PREFIX + head
    environment = _git_environment(server_url, token)
    with tempfile.TemporaryDirectory(prefix="attribution-pr-") as directory:
        repo = Path(directory)
        _git(repo, environment, "init", "--bare", "--quiet")
        found = _git(repo, environment, "ls-remote", "--exit-code", "--refs", head_url, metadata_ref, check=False)
        if found.returncode == 2:
            return None, None, []
        if found.returncode:
            raise ValueError("Could not check the attribution metadata ref.")
        matches = found.stdout.decode("ascii").splitlines()
        if len(matches) != 1:
            raise ValueError("Joyride metadata ref is ambiguous.")
        metadata_commit, advertised_ref = matches[0].split("\t", 1)
        if not _OID.fullmatch(metadata_commit) or advertised_ref != metadata_ref:
            raise ValueError("Joyride metadata ref is invalid.")
        _git(repo, environment, "fetch", "--quiet", "--no-tags", "--no-recurse-submodules", "--depth=1", head_url, f"{metadata_ref}:refs/attribution/metadata")
        fetched = _git(repo, environment, "rev-parse", "refs/attribution/metadata^{commit}").stdout.decode().strip()
        if fetched != metadata_commit:
            raise ValueError("Joyride metadata changed while it was being fetched.")
        notes, tasks, usage, traces = _read_snapshot(repo, environment, fetched, head)
        _git(repo, environment, "fetch", "--quiet", "--no-tags", "--no-recurse-submodules", base_url, f"{base}:refs/attribution/base")
        _git(repo, environment, "fetch", "--quiet", "--no-tags", "--no-recurse-submodules", head_url, f"{head}:refs/attribution/head")
        with _report_git_environment(environment):
            report = build_pr_report(
                repo, base_ref=base, head_ref=head, notes=notes, tasks=tasks,
                usage=usage,
            )
        if repository_id is None:
            return report, None, []
        # The model request carries a credential over HTTPS, so it runs between
        # the two Git blocks and never while the isolation environment of the
        # untrusted PR objects is installed. Re-entering that block is safe:
        # it saves and restores every GIT_ variable of this process.
        from .narrative_summary import generate_narrative_summary
        narrative = generate_narrative_summary(report, narrative_api_key)
        from .hosted_artifact import build_artifact
        full_name = _repository_name(pr["base"]["repo"]["full_name"])
        with _report_git_environment(environment):
            artifact = build_artifact(
                repo, repository_id=repository_id, full_name=full_name,
                pr_number=pr["number"], base_sha=base, head_sha=head,
                metadata_sha=fetched, notes=notes, tasks=tasks,
                narrative_summary=narrative, usage=usage,
            )
        return report, artifact, traces


def _publish_traces(
    ingest_url: str, traces: list[dict[str, Any]], artifact: dict[str, Any], *,
    oidc_request_url: str, oidc_request_token: str, repository: str,
    repository_id: int, workflow_ref: str, delivery_id: str,
) -> dict[str, int]:
    """Upload each trace beside its accepted artifact, counting what failed.

    A trace is optional evidence, so a failed upload is counted and reported
    on stderr and never removes the footer or fails the run.
    """

    from .traces import SCHEMA as TRACES_SCHEMA, canonical_bytes

    result = {"uploaded": 0, "failed": 0}
    if not traces:
        return result
    endpoint, audience = _ingest_url(ingest_url)
    digest = artifact["digest"]
    pull_request = {
        key: artifact["pull_request"][key] for key in ("number", "base_sha", "head_sha")
    }
    try:
        token = _oidc_token(oidc_request_url, oidc_request_token, audience)
    except (OSError, ValueError):
        result["failed"] = len(traces)
        return result
    for trace in traces:
        session_id = trace["session_id"]
        try:
            bound_delivery = f"{delivery_id}:{digest[7:39]}:{session_id}"
            if not re.fullmatch(r"[A-Za-z0-9_.:-]{1,200}", bound_delivery):
                raise ValueError("Trace publication delivery identifier is invalid.")
            request = Request(
                endpoint + "/traces",
                data=canonical_bytes({
                    "schema": TRACES_SCHEMA, "repository": artifact["repository"],
                    "pull_request": pull_request, "artifact_digest": digest, "trace": trace,
                }),
                method="POST",
                headers={
                    "Accept": "application/json", "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json", "User-Agent": "attribution-pr-footer",
                    "X-GitHub-Repository": repository,
                    "X-GitHub-Repository-ID": str(repository_id),
                    "X-GitHub-Workflow-Ref": workflow_ref,
                    "X-Attribution-Delivery": bound_delivery,
                },
            )
            with build_opener(_NoRedirect()).open(request, timeout=60) as response:
                raw = response.read(_OIDC_HTTP_BYTES + 1)
            payload = json.loads(raw or b"{}")
            if (
                not isinstance(payload, dict)
                or payload.get("status") not in {"created", "unchanged"}
                or payload.get("digest") != trace["digest"]
            ):
                raise ValueError("Trace publication response did not confirm the digest.")
        except (OSError, ValueError):
            result["failed"] += 1
            print(
                f"Joyride: the trace of session {session_id} could not be published.",
                file=sys.stderr,
            )
            continue
        result["uploaded"] += 1
    return result


def _footer_for_pr(
    pr: dict[str, Any], server_url: str, token: str, *,
    attribution_ui_url: str | None = None,
) -> str | None:
    from .pr_report import render_footer

    report, _artifact, _traces = _report_and_artifact_for_pr(pr, server_url, token)
    if report is None:
        return None
    return render_footer(report, details_url=_details_url(attribution_ui_url, pr))


def _narrative_key(pr: dict[str, Any], api_key: str) -> str:
    """Allow paid prose generation only for trusted same-repository authors."""
    if not api_key or pr.get("author_association") not in _NARRATIVE_AUTHORS:
        return ""
    try:
        base = _repository_name(pr["base"]["repo"]["full_name"])
        head = _repository_name(pr["head"]["repo"]["full_name"])
    except (KeyError, TypeError, ValueError):
        return ""
    return api_key if base.lower() == head.lower() else ""


def run_from_event(
    event_path: str | Path,
    *,
    token: str,
    api_url: str = "https://api.github.com",
    server_url: str = "https://github.com",
    attribution_ui_url: str | None = None,
    attribution_ingest_url: str | None = None,
    oidc_request_url: str = "",
    oidc_request_token: str = "",
    workflow_ref: str = "",
    delivery_id: str = "",
    narrative_api_key: str = "",
) -> dict[str, Any]:
    """Append/update one managed footer, or remove it when metadata is absent."""
    from .pr_report import update_pr_body

    raw = Path(event_path).read_bytes()
    if len(raw) > _MAX_HTTP_BYTES:
        raise ValueError("GitHub event exceeded the inspection limit.")
    event = json.loads(raw)
    if not isinstance(event, dict):
        raise ValueError("Invalid GitHub event.")
    action = event.get("action")
    if action not in {"opened", "reopened", "synchronize", "edited"}:
        return {"status": "ignored"}
    if action == "edited" and "base" not in (event.get("changes") or {}):
        return {"status": "ignored"}
    if not token:
        raise ValueError("GITHUB_TOKEN is required to update the PR footer.")
    api_url, server_url = _host_url(api_url, api=True), _host_url(server_url)
    pr = event.get("pull_request")
    if not isinstance(pr, dict):
        raise ValueError("The event has no pull request.")
    expected = _identity(pr)
    repository = _repository_name((event.get("repository") or {}).get("full_name"))
    if repository.lower() != expected[1]:
        raise ValueError("The event repository does not match the PR base.")
    endpoint = f"{api_url}/repos/{repository}/pulls/{expected[0]}"
    current = _api(endpoint, token)
    if current.get("state") != "open" or _identity(current) != expected:
        return {"status": "superseded"}
    narrative_api_key = _narrative_key(current, narrative_api_key)

    failure: str | None = None
    traces_result = {"uploaded": 0, "failed": 0}
    try:
        if attribution_ingest_url:
            repository_id = (event.get("repository") or {}).get("id")
            report, artifact, traces = _report_and_artifact_for_pr(
                pr, server_url, token, repository_id=repository_id,
                narrative_api_key=narrative_api_key,
            )
            if report is None or artifact is None:
                footer = None
            else:
                _publish_artifact(
                    attribution_ingest_url, artifact,
                    oidc_request_url=oidc_request_url,
                    oidc_request_token=oidc_request_token,
                    repository=repository,
                    repository_id=repository_id,
                    workflow_ref=workflow_ref,
                    delivery_id=delivery_id,
                )
                traces_result = _publish_traces(
                    attribution_ingest_url, traces, artifact,
                    oidc_request_url=oidc_request_url,
                    oidc_request_token=oidc_request_token,
                    repository=repository,
                    repository_id=repository_id,
                    workflow_ref=workflow_ref,
                    delivery_id=delivery_id,
                )
                print(
                    f"Joyride traces: {traces_result['uploaded']} uploaded,"
                    f" {traces_result['failed']} failed.",
                    file=sys.stderr,
                )
                from .pr_report import render_footer
                footer = render_footer(
                    report, details_url=_details_url(attribution_ui_url, pr),
                )
        else:
            # A hosted link is published only after immutable ingestion succeeds.
            footer = _footer_for_pr(pr, server_url, token, attribution_ui_url=None)
    except (OSError, ValueError) as exc:
        footer, failure = None, str(exc)

    # Read current prose after the expensive work and refuse an outdated event.
    current = _api(endpoint, token)
    if current.get("state") != "open" or _identity(current) != expected:
        return {"status": "superseded"}
    old_body = current.get("body") or ""
    if not isinstance(old_body, str):
        raise ValueError("The current PR description is invalid.")
    body = update_pr_body(old_body, footer or "")
    changed = body != old_body
    if changed:
        updated = _api(endpoint, token, body={"body": body})
        if _identity(updated) != expected or updated.get("body") != body:
            raise ValueError("The PR changed during publication; the footer update could not be verified.")
    if failure:
        raise ValueError(f"Joyride is unavailable; any previous managed footer was removed. {failure}")
    return {
        "status": "metadata-missing" if footer is None else ("updated" if changed else "unchanged"),
        "number": expected[0],
        "head_commit": expected[5],
        "traces": traces_result,
    }


def main() -> int:
    try:
        if os.environ.get("GITHUB_EVENT_NAME") != "pull_request_target":
            raise ValueError("This module requires a trusted pull_request_target workflow.")
        # Do not let a model credential enter the environment inherited by the
        # Git subprocesses which inspect the untrusted PR objects.
        narrative_api_key = os.environ.pop("ATTRIBUTION_OPENAI_API_KEY", "")
        result = run_from_event(
            os.environ["GITHUB_EVENT_PATH"],
            token=os.environ.get("GITHUB_TOKEN", ""),
            api_url=os.environ.get("GITHUB_API_URL", "https://api.github.com"),
            server_url=os.environ.get("GITHUB_SERVER_URL", "https://github.com"),
            attribution_ui_url=os.environ.get("ATTRIBUTION_UI_URL"),
            attribution_ingest_url=os.environ.get("ATTRIBUTION_INGEST_URL"),
            oidc_request_url=os.environ.get("ACTIONS_ID_TOKEN_REQUEST_URL", ""),
            oidc_request_token=os.environ.get("ACTIONS_ID_TOKEN_REQUEST_TOKEN", ""),
            workflow_ref=os.environ.get("GITHUB_WORKFLOW_REF", ""),
            delivery_id=(
                f"{os.environ.get('GITHUB_RUN_ID', '')}:"
                f"{os.environ.get('GITHUB_RUN_ATTEMPT', '')}:"
                f"{os.environ.get('GITHUB_EVENT_NAME', '')}"
            ),
            narrative_api_key=narrative_api_key,
        )
        print(json.dumps(result))
        return 0
    except (KeyError, OSError, ValueError) as exc:
        print(f"Joyride footer: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
