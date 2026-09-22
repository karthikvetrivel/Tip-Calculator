"""Local onboarding helpers for the hosted GitHub integration.

The hosted integration is deliberately provisioned through a normal pull
request.  This module only uses the GitHub CLI for authentication and the
repository's own Git remote for the write path; it never edits the currently
checked-out worktree.  The planning functions are kept separate from the
subprocess orchestration so ``doctor`` and dry runs remain useful offline.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
import errno
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import webbrowser
from typing import Any, Mapping
from urllib.parse import quote, urlsplit

from .hosted_runtime import (
    HostedRuntimeError,
    MAX_SETUP_FILE_BYTES as _MAX_SETUP_FILE_BYTES,
    RUNTIME_MANIFEST_PATH as _RUNTIME_MANIFEST_PATH,
    RUNTIME_PREFIX as _RUNTIME_PREFIX,
    WORKFLOW_PATH as _WORKFLOW_PATH,
    installed_runtime_tree,
    runtime_manifest_files,
    setup_files as build_setup_files,
)
from .runtime import is_frozen, system_subprocess_environment


_GITHUB_HOST = "github.com"
_REMOTE_RE = re.compile(r"^(?P<owner>[A-Za-z0-9_.-]+)/(?P<name>[A-Za-z0-9_.-]+)$")
_BRANCH_RE = re.compile(r"^[A-Za-z0-9._/-]+$")
# The isolated workflow appends .github/attribution after the standard library,
# so copied modules remain in an importable package below that directory.
_DEFAULT_BRANCH = "main"
_DEFAULT_SETUP_BRANCH = "attribution/setup"
_MAX_COMMAND_OUTPUT = 1_000_000
_STANDALONE_SOURCE_ERROR = (
    "Hosted setup is unavailable in the standalone Joyride binary because "
    "the one-file bundle does not contain the exact reviewable Python source "
    "that must be vendored into a repository. Install the Python package or run "
    "Joyride from a source checkout, then retry."
)


_WORKFLOW = """name: Coding attribution footer

on:
  pull_request_target:
    types: [opened, reopened, synchronize, edited]

permissions:
  contents: read
  id-token: write
  pull-requests: write

jobs:
  footer:
    if: github.event.action != 'edited' || github.event.changes.base != null
    concurrency:
      group: attribution-footer-${{ github.event.pull_request.number }}
      cancel-in-progress: true
    runs-on: ubuntu-latest
    timeout-minutes: 10
    steps:
      # pull_request_target's SHA is the trusted default-branch workflow revision.
      # Do not use the PR head OR its chosen target branch as runnable code.
      - name: Check out trusted workflow code
        uses: actions/checkout@11bd71901bbe5b1630ceea73d27597364c9af683 # v4.2.2
        with:
          repository: ${{ github.repository }}
          ref: ${{ github.sha }}
          persist-credentials: false
      - name: Update the attribution footer
        env:
          GITHUB_TOKEN: ${{ github.token }}
          ATTRIBUTION_INGEST_URL: ${{ vars.ATTRIBUTION_INGEST_URL }}
          ATTRIBUTION_UI_URL: ${{ vars.ATTRIBUTION_UI_URL }}
          ATTRIBUTION_OPENAI_API_KEY: ${{ secrets.ATTRIBUTION_OPENAI_API_KEY }}
        run: python3 -I -S -c "import sys; sys.dont_write_bytecode = True; sys.path.append('.github/attribution'); from attribution.github_footer import main; raise SystemExit(main())"
"""


class HostedSetupError(ValueError):
    """An actionable hosted setup or readiness error."""


@dataclass(frozen=True)
class GitHubRepository:
    """The GitHub identity discovered from the selected local checkout."""

    root: Path
    owner: str
    name: str
    remote: str
    base_branch: str
    remote_url: str

    @property
    def full_name(self) -> str:
        return f"{self.owner}/{self.name}"

    @property
    def web_url(self) -> str:
        return f"https://{_GITHUB_HOST}/{self.full_name}"


@dataclass(frozen=True)
class HostedSetupPlan:
    """Files and metadata that a setup pull request will add."""

    repository: GitHubRepository
    branch: str
    base_branch: str
    files: Mapping[str, bytes]
    hosted_url: str | None
    title: str
    body: str


@dataclass(frozen=True)
class HostedSetupResult:
    """The externally visible result of setup or a dry run."""

    status: str
    repository: GitHubRepository
    branch: str
    base_branch: str
    files: tuple[str, ...]
    pull_request_url: str | None = None
    hosted_url: str | None = None
    warnings: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "repository": {
                "root": str(self.repository.root),
                "full_name": self.repository.full_name,
                "owner": self.repository.owner,
                "name": self.repository.name,
                "remote": self.repository.remote,
                "remote_url": self.repository.remote_url,
                "base_branch": self.repository.base_branch,
                "web_url": self.repository.web_url,
            },
            "branch": self.branch,
            "base_branch": self.base_branch,
            "files": list(self.files),
            "pull_request_url": self.pull_request_url,
            "hosted_url": self.hosted_url,
            "warnings": list(self.warnings),
        }


def _command(
    arguments: list[str],
    *,
    cwd: Path | None = None,
    capture: bool = True,
    timeout: int = 30,
) -> subprocess.CompletedProcess[str]:
    """Run one bounded command without invoking a shell."""

    try:
        result = subprocess.run(
            arguments,
            cwd=str(cwd) if cwd is not None else None,
            stdin=None,
            stdout=subprocess.PIPE if capture else None,
            stderr=subprocess.PIPE if capture else None,
            env=system_subprocess_environment(),
            text=True,
            check=False,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise HostedSetupError(f"Could not run {arguments[0]}: {exc}") from exc
    if capture:
        # A Git remote can contain arbitrary text. Keep diagnostics bounded and
        # do not echo it verbatim in the terminal renderer.
        result.stdout = (result.stdout or "")[:_MAX_COMMAND_OUTPUT]
        result.stderr = (result.stderr or "")[:_MAX_COMMAND_OUTPUT]
    return result


def _git(repo: Path, *arguments: str, check: bool = True) -> str:
    result = _command(["git", "-C", str(repo), *arguments], cwd=repo)
    if check and result.returncode:
        message = (result.stderr or "").strip()
        raise HostedSetupError(message or f"Git {arguments[0]} failed.")
    return (result.stdout or "").strip()


def _validate_branch(value: str, *, option: str) -> str:
    branch = value.strip()
    if (
        not branch
        or len(branch) > 200
        or branch.startswith(("/", ".", "-"))
        or branch.endswith(("/", "."))
        or ".." in branch
        or "//" in branch
        or not _BRANCH_RE.fullmatch(branch)
    ):
        raise HostedSetupError(f"{option} must be a valid Git branch name.")
    return branch


def _remote_identity(remote_url: str) -> tuple[str, str] | None:
    """Extract a GitHub owner/repository from HTTPS, SSH, or scp syntax."""

    value = remote_url.strip()
    if not value:
        return None
    if value.startswith("git@"):
        prefix, separator, path = value.partition(":")
        if separator and prefix[4:].lower() == _GITHUB_HOST:
            candidate = path
        else:
            return None
    else:
        parsed = urlsplit(value)
        if parsed.scheme not in {"https", "ssh", "git"} or parsed.hostname is None:
            return None
        if parsed.hostname.lower() != _GITHUB_HOST or parsed.username not in {None, "git"}:
            return None
        candidate = parsed.path.lstrip("/")
    if candidate.endswith(".git"):
        candidate = candidate[:-4]
    match = _REMOTE_RE.fullmatch(candidate.rstrip("/"))
    if match is None or any(part in {".", ".."} for part in candidate.split("/")):
        return None
    return match.group("owner"), match.group("name")


def discover_repository(repo: str | Path = ".") -> GitHubRepository:
    """Discover and validate the GitHub repository for a local checkout."""

    requested = Path(repo).expanduser().resolve()
    try:
        root = Path(_git(requested, "rev-parse", "--show-toplevel")).resolve()
    except HostedSetupError as exc:
        raise HostedSetupError(
            f"{requested} is not a Git repository; run this command inside a GitHub checkout."
        ) from exc
    remotes = _git(root, "remote").splitlines()
    ordered = ["origin", *[item for item in remotes if item != "origin"]]
    selected: tuple[str, str, str] | None = None
    for remote in ordered:
        if not remote:
            continue
        url = _git(root, "remote", "get-url", remote, check=False)
        identity = _remote_identity(url)
        if identity is not None:
            selected = (remote, url, "/".join(identity))
            break
    if selected is None:
        if not remotes:
            raise HostedSetupError("The repository has no Git remote pointing to GitHub.")
        raise HostedSetupError(
            "The repository remotes do not point to github.com; hosted setup supports GitHub repositories."
        )
    remote, remote_url, full_name = selected
    owner, name = full_name.split("/", 1)
    remote_head = _git(
        root,
        "symbolic-ref",
        "--quiet",
        "--short",
        f"refs/remotes/{remote}/HEAD",
        check=False,
    )
    base_branch = remote_head.split("/", 1)[1] if remote_head.startswith(remote + "/") else ""
    if not base_branch:
        base_branch = _git(root, "branch", "--show-current", check=False) or _DEFAULT_BRANCH
    return GitHubRepository(root, owner, name, remote, _validate_branch(base_branch, option="base branch"), remote_url)


def _normalise_hosted_url(value: str) -> str:
    candidate = value.strip().rstrip("/")
    parsed = urlsplit(candidate)
    try:
        port = parsed.port
    except ValueError as exc:
        raise HostedSetupError(
            "--hosted-url must be an HTTPS origin without a path, query, or fragment."
        ) from exc
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/"}
        or port is not None and not 1 <= port <= 65535
    ):
        raise HostedSetupError("--hosted-url must be an HTTPS origin without a path, query, or fragment.")
    return candidate


def _require_source_runtime() -> None:
    if is_frozen():
        raise HostedSetupError(_STANDALONE_SOURCE_ERROR)


def _runtime_files() -> dict[str, bytes]:
    _require_source_runtime()
    package = Path(__file__).resolve().parent
    try:
        return build_setup_files(package, _WORKFLOW.encode("utf-8"))
    except HostedRuntimeError as exc:
        raise HostedSetupError(str(exc)) from exc


def make_setup_plan(
    repository: GitHubRepository,
    *,
    hosted_url: str | None = None,
    branch: str = _DEFAULT_SETUP_BRANCH,
    base_branch: str | None = None,
) -> HostedSetupPlan:
    """Build the reviewable setup files without touching the repository."""

    normalised_url = _normalise_hosted_url(hosted_url) if hosted_url else None
    selected_base = _validate_branch(base_branch or repository.base_branch, option="--base")
    selected_branch = _validate_branch(branch, option="--branch")
    files = _runtime_files()
    title = "Add hosted attribution setup"
    variable_note = (
        f"The command will configure `ATTRIBUTION_UI_URL` and `ATTRIBUTION_INGEST_URL` for `{normalised_url}` after the PR is created."
        if normalised_url
        else "Set `ATTRIBUTION_UI_URL` (and `ATTRIBUTION_INGEST_URL` for hosted ingestion) as repository variables after review."
    )
    body = (
        "## Hosted Joyride setup\n\n"
        "This pull request adds the trusted GitHub Actions workflow and its pinned, "
        "reviewable attribution runtime. The workflow builds a source-free metadata "
        "artifact from the default branch with `pull_request_target`; it does not check "
        "out or execute pull-request source.\n\n"
        f"{variable_note}\n\n"
        "Review the generated workflow and runtime before merging. Each developer "
        "clone still needs `joyride install .` so its ordinary pushes publish the "
        "bounded local attribution metadata used by the workflow."
    )
    return HostedSetupPlan(repository, selected_branch, selected_base, files, normalised_url, title, body)


def _gh(repo: GitHubRepository, *arguments: str, capture: bool = True) -> subprocess.CompletedProcess[str]:
    executable = shutil.which("gh")
    if executable is None:
        raise HostedSetupError(
            "GitHub CLI (`gh`) is required for hosted setup. Install it from https://cli.github.com/."
        )
    return _command([executable, *arguments], cwd=repo.root, capture=capture, timeout=120)


def ensure_github_auth(repository: GitHubRepository, *, open_browser: bool = True) -> None:
    """Verify ``gh`` auth, using its browser flow only in an interactive shell."""

    status = _gh(repository, "auth", "status", "--hostname", _GITHUB_HOST)
    if status.returncode == 0:
        return
    interactive = bool(getattr(sys.stdin, "isatty", lambda: False)())
    if not open_browser or not interactive:
        raise HostedSetupError(
            "GitHub CLI is not authenticated. Run `gh auth login --web` and retry hosted setup."
        )
    login = _gh(
        repository,
        "auth",
        "login",
        "--hostname",
        _GITHUB_HOST,
        "--git-protocol",
        "https",
        "--web",
        capture=False,
    )
    if login.returncode:
        raise HostedSetupError("GitHub browser authentication did not complete.")
    status = _gh(repository, "auth", "status", "--hostname", _GITHUB_HOST)
    if status.returncode:
        raise HostedSetupError("GitHub CLI authentication could not be verified.")


def _variable_values(repository: GitHubRepository) -> dict[str, str] | None:
    result = _gh(
        repository,
        "variable",
        "list",
        "--repo",
        repository.full_name,
        "--json",
        "name,value",
    )
    if result.returncode:
        return None
    try:
        payload = json.loads(result.stdout or "[]")
    except (TypeError, ValueError):
        return None
    if not isinstance(payload, list):
        return None
    variables: dict[str, str] = {}
    for item in payload:
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("name"), str)
            or not isinstance(item.get("value"), str)
            or item["name"] in variables
        ):
            return None
        variables[item["name"]] = item["value"]
    return variables


def _github_default_branch(repository: GitHubRepository) -> str | None:
    result = _gh(repository, "repo", "view", repository.full_name, "--json", "defaultBranchRef")
    if result.returncode:
        return None
    try:
        payload = json.loads(result.stdout or "{}")
    except (TypeError, ValueError):
        return None
    ref = payload.get("defaultBranchRef") if isinstance(payload, dict) else None
    name = ref.get("name") if isinstance(ref, dict) else None
    if not isinstance(name, str):
        return None
    try:
        return _validate_branch(name, option="GitHub default branch")
    except HostedSetupError:
        return None


def _gh_json(
    repository: GitHubRepository, *arguments: str
) -> dict[str, Any] | None:
    result = _gh(repository, *arguments)
    if result.returncode:
        return None
    try:
        value = json.loads(result.stdout or "{}")
    except (TypeError, ValueError, RecursionError):
        return None
    return value if isinstance(value, dict) else None


def _git_blob_sha(content: bytes) -> str:
    return hashlib.sha1(
        b"blob " + str(len(content)).encode("ascii") + b"\0" + content
    ).hexdigest()


def _remote_workflow_check(repository: GitHubRepository) -> tuple[str, str]:
    """Verify generated files on GitHub's current default branch, not HEAD."""

    branch = _github_default_branch(repository)
    if branch is None:
        return "fail", "Could not determine GitHub's current default branch."
    owner = quote(repository.owner, safe="")
    name = quote(repository.name, safe="")
    prefix = f"repos/{owner}/{name}"
    ref = _gh_json(
        repository,
        "api",
        f"{prefix}/git/ref/heads/{quote(branch, safe='/')}",
    )
    target = ref.get("object") if isinstance(ref, dict) else None
    sha = target.get("sha") if isinstance(target, dict) else None
    if (
        not isinstance(ref, dict)
        or ref.get("ref") != f"refs/heads/{branch}"
        or not isinstance(target, dict)
        or target.get("type") != "commit"
        or not isinstance(sha, str)
        or re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", sha) is None
    ):
        return "fail", "GitHub returned invalid default-branch information."
    commit = _gh_json(repository, "api", f"{prefix}/git/commits/{sha}")
    tree_sha = (commit.get("tree") or {}).get("sha") if isinstance(commit, dict) else None
    if (
        not isinstance(commit, dict)
        or commit.get("sha") != sha
        or not isinstance(tree_sha, str)
        or re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", tree_sha) is None
    ):
        return "fail", "GitHub returned invalid default-branch commit information."
    tree = _gh_json(repository, "api", f"{prefix}/git/trees/{tree_sha}?recursive=1")
    entries = tree.get("tree") if isinstance(tree, dict) else None
    if (
        not isinstance(tree, dict)
        or tree.get("truncated") is not False
        or not isinstance(entries, list)
    ):
        return "fail", "GitHub did not return a complete default-branch tree."
    blobs: dict[str, tuple[str, str, str]] = {}
    for entry in entries:
        path = entry.get("path") if isinstance(entry, dict) else None
        mode = entry.get("mode") if isinstance(entry, dict) else None
        kind = entry.get("type") if isinstance(entry, dict) else None
        object_sha = entry.get("sha") if isinstance(entry, dict) else None
        if (
            not isinstance(path, str)
            or not path
            or path in blobs
            or not isinstance(mode, str)
            or re.fullmatch(r"[0-7]{6}", mode) is None
            or kind not in {"blob", "tree", "commit"}
            or not isinstance(object_sha, str)
            or re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", object_sha) is None
        ):
            return "fail", "GitHub returned an invalid default-branch tree."
        if kind != "tree":
            blobs[path] = (mode, kind, object_sha)
    try:
        expected = _runtime_files()
    except HostedSetupError as exc:
        return "fail", str(exc)
    expected_runtime = {
        path for path in expected if path.startswith(_RUNTIME_PREFIX)
    }
    actual_runtime = {
        path for path in blobs if path.startswith(_RUNTIME_PREFIX)
    }
    if actual_runtime != expected_runtime or any(
        blobs.get(path) != ("100644", "blob", _git_blob_sha(content))
        for path, content in expected.items()
    ):
        return (
            "fail",
            f"GitHub's {branch} branch does not contain this release's exact generated workflow and runtime.",
        )
    return (
        "pass",
        f"GitHub's {branch} branch contains the exact generated workflow and runtime.",
    )


def _remote_blob(
    repository: GitHubRepository, prefix: str, object_sha: str
) -> bytes | None:
    value = _gh_json(repository, "api", f"{prefix}/git/blobs/{object_sha}")
    content = value.get("content") if isinstance(value, dict) else None
    if (
        not isinstance(value, dict)
        or value.get("encoding") != "base64"
        or not isinstance(content, str)
    ):
        return None
    try:
        decoded = base64.b64decode(content, validate=False)
    except (ValueError, TypeError):
        return None
    if len(decoded) > _MAX_SETUP_FILE_BYTES:
        return None
    return decoded


def _remote_setup_branch_matches(
    repository: GitHubRepository, plan: HostedSetupPlan
) -> bool:
    """Verify one exact generated orphan branch without checking it out."""

    owner = quote(repository.owner, safe="")
    name = quote(repository.name, safe="")
    prefix = f"repos/{owner}/{name}"

    def reference(branch: str) -> str | None:
        value = _gh_json(
            repository,
            "api",
            f"{prefix}/git/ref/heads/{quote(branch, safe='/')}",
        )
        target = value.get("object") if isinstance(value, dict) else None
        sha = target.get("sha") if isinstance(target, dict) else None
        if (
            not isinstance(value, dict)
            or value.get("ref") != f"refs/heads/{branch}"
            or not isinstance(target, dict)
            or target.get("type") != "commit"
            or not isinstance(sha, str)
            or re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", sha) is None
        ):
            return None
        return sha

    base_sha = reference(plan.base_branch)
    head_sha = reference(plan.branch)
    if base_sha is None or head_sha is None or base_sha == head_sha:
        return False
    head_commit = _gh_json(repository, "api", f"{prefix}/git/commits/{head_sha}")
    parents = head_commit.get("parents") if isinstance(head_commit, dict) else None
    if (
        not isinstance(head_commit, dict)
        or head_commit.get("sha") != head_sha
        or not isinstance(parents, list)
        or len(parents) != 1
        or not isinstance(parents[0], dict)
        or parents[0].get("sha") != base_sha
    ):
        return False

    def tree(commit_sha: str) -> dict[str, tuple[str, str, str]] | None:
        commit = _gh_json(repository, "api", f"{prefix}/git/commits/{commit_sha}")
        tree_sha = (commit.get("tree") or {}).get("sha") if isinstance(commit, dict) else None
        if (
            not isinstance(commit, dict)
            or commit.get("sha") != commit_sha
            or not isinstance(tree_sha, str)
            or re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", tree_sha) is None
        ):
            return None
        payload = _gh_json(
            repository, "api", f"{prefix}/git/trees/{tree_sha}?recursive=1"
        )
        entries = payload.get("tree") if isinstance(payload, dict) else None
        if (
            not isinstance(payload, dict)
            or payload.get("truncated") is not False
            or not isinstance(entries, list)
        ):
            return None
        result: dict[str, tuple[str, str, str]] = {}
        for entry in entries:
            path = entry.get("path") if isinstance(entry, dict) else None
            mode = entry.get("mode") if isinstance(entry, dict) else None
            kind = entry.get("type") if isinstance(entry, dict) else None
            object_sha = entry.get("sha") if isinstance(entry, dict) else None
            if (
                not isinstance(path, str)
                or not path
                or path in result
                or not isinstance(mode, str)
                or re.fullmatch(r"[0-7]{6}", mode) is None
                or kind not in {"blob", "tree", "commit"}
                or not isinstance(object_sha, str)
                or re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", object_sha)
                is None
            ):
                return None
            if kind != "tree":
                result[path] = (mode, kind, object_sha)
        return result

    base_tree, head_tree = tree(base_sha), tree(head_sha)
    if base_tree is None or head_tree is None:
        return False
    expected = {
        path: ("100644", "blob", _git_blob_sha(content))
        for path, content in plan.files.items()
    }
    expected_runtime = {
        path for path in expected if path.startswith(_RUNTIME_PREFIX)
    }
    head_runtime = {
        path for path in head_tree if path.startswith(_RUNTIME_PREFIX)
    }
    if head_runtime != expected_runtime or any(
        head_tree.get(path) != entry for path, entry in expected.items()
    ):
        return False

    stale_runtime = {
        path for path in base_tree if path.startswith(_RUNTIME_PREFIX)
    } - expected_runtime
    if stale_runtime:
        manifest_entry = base_tree.get(_RUNTIME_MANIFEST_PATH)
        if manifest_entry is None or manifest_entry[:2] != ("100644", "blob"):
            return False
        manifest = _remote_blob(repository, prefix, manifest_entry[2])
        if manifest is None:
            return False
        try:
            owned = runtime_manifest_files(manifest)
        except HostedRuntimeError:
            return False
        for path in stale_runtime:
            entry = base_tree[path]
            content = (
                _remote_blob(repository, prefix, entry[2])
                if entry[:2] == ("100644", "blob")
                else None
            )
            if (
                content is None
                or owned.get(path) != hashlib.sha256(content).hexdigest()
                or path in head_tree
            ):
                return False

    actual_delta = {
        path
        for path in set(base_tree) | set(head_tree)
        if base_tree.get(path) != head_tree.get(path)
    }
    expected_delta = {
        path for path, entry in expected.items() if base_tree.get(path) != entry
    } | stale_runtime
    return actual_delta == expected_delta


def _existing_setup_pull_request(
    repository: GitHubRepository, plan: HostedSetupPlan
) -> str | None:
    prefix = f"repos/{quote(repository.owner, safe='')}/{quote(repository.name, safe='')}"

    def branch_oid(branch: str) -> str | None:
        value = _gh_json(
            repository,
            "api",
            f"{prefix}/git/ref/heads/{quote(branch, safe='/')}",
        )
        target = value.get("object") if isinstance(value, dict) else None
        oid = target.get("sha") if isinstance(target, dict) else None
        if (
            not isinstance(value, dict)
            or value.get("ref") != f"refs/heads/{branch}"
            or not isinstance(target, dict)
            or target.get("type") != "commit"
            or not isinstance(oid, str)
            or re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", oid) is None
        ):
            return None
        return oid

    head_oid = branch_oid(plan.branch)
    base_oid = branch_oid(plan.base_branch)
    if head_oid is None or base_oid is None:
        return None
    result = _gh(
        repository,
        "pr",
        "list",
        "--repo",
        repository.full_name,
        "--state",
        "open",
        "--head",
        plan.branch,
        "--base",
        plan.base_branch,
        "--limit",
        "10",
        "--json",
        "url,headRefName,baseRefName,headRefOid,baseRefOid,isCrossRepository,headRepository,headRepositoryOwner",
    )
    if result.returncode:
        return None
    try:
        values = json.loads(result.stdout or "[]")
    except (TypeError, ValueError, RecursionError):
        return None
    matches: list[str] = []
    if isinstance(values, list):
        expected_url = f"https://github.com/{repository.full_name}/pull/"
        for item in values:
            head_repository = item.get("headRepository") if isinstance(item, dict) else None
            head_owner = item.get("headRepositoryOwner") if isinstance(item, dict) else None
            owner_login = head_owner.get("login") if isinstance(head_owner, dict) else None
            name_with_owner = (
                head_repository.get("nameWithOwner")
                if isinstance(head_repository, dict)
                else None
            )
            if (
                name_with_owner is None
                and isinstance(head_repository, dict)
                and isinstance(head_repository.get("name"), str)
                and isinstance(owner_login, str)
            ):
                name_with_owner = f"{owner_login}/{head_repository['name']}"
            url = item.get("url") if isinstance(item, dict) else None
            if (
                isinstance(item, dict)
                and item.get("isCrossRepository") is False
                and isinstance(owner_login, str)
                and owner_login.lower() == repository.owner.lower()
                and isinstance(name_with_owner, str)
                and name_with_owner.lower() == repository.full_name.lower()
                and item.get("headRefName") == plan.branch
                and item.get("baseRefName") == plan.base_branch
                and item.get("headRefOid") == head_oid
                and item.get("baseRefOid") == base_oid
                and isinstance(url, str)
                and url.startswith(expected_url)
                and url.removeprefix(expected_url).isdigit()
            ):
                matches.append(url)
    return matches[0] if len(matches) == 1 else None


def _runtime_check(repository: GitHubRepository) -> tuple[str, str]:
    """Verify the complete vendored package without following symbolic links."""

    package = PurePosixPath(_RUNTIME_PREFIX)
    expected: dict[str, bytes] = {}
    expected_directories: set[str] = set()
    try:
        generated = _runtime_files()
    except HostedSetupError as exc:
        return "fail", str(exc)
    except OSError:
        return "fail", "Joyride's expected hosted runtime is unavailable."
    for relative, content in generated.items():
        path = PurePosixPath(relative)
        try:
            child = path.relative_to(package)
        except ValueError:
            continue
        key = child.as_posix()
        if not child.parts or key in expected:
            return "fail", "Joyride's expected hosted runtime layout is invalid."
        expected[key] = content
        parent = child.parent
        while parent.parts:
            expected_directories.add(parent.as_posix())
            parent = parent.parent
    if not expected:
        return "fail", "Joyride's expected hosted runtime is unavailable."

    try:
        root = repository.root.resolve(strict=True)
        runtime = root
        for part in package.parts:
            runtime = runtime / part
            details = runtime.lstat()
            if stat.S_ISLNK(details.st_mode) or not stat.S_ISDIR(details.st_mode):
                return "fail", "The vendored hosted runtime contains an unsafe directory."
        runtime.resolve(strict=True).relative_to(root)
        actual, actual_directories = installed_runtime_tree(runtime)
    except FileNotFoundError:
        return "fail", f"{_RUNTIME_PREFIX} is missing."
    except HostedRuntimeError as exc:
        return "fail", str(exc)
    except (OSError, RuntimeError, ValueError):
        return "fail", "The vendored hosted runtime cannot be inspected safely."

    if set(actual) != set(expected) or actual_directories != expected_directories:
        return "fail", "The vendored hosted runtime has missing or unexpected files."
    for relative, content in actual.items():
        if content != expected[relative]:
            return "fail", "The vendored hosted runtime does not match this Joyride release."
    return "pass", "The complete hosted runtime matches this Joyride release."


def _workflow_check(repository: GitHubRepository) -> tuple[str, str]:
    workflow = repository.root / _WORKFLOW_PATH
    try:
        details = workflow.lstat()
        if (
            stat.S_ISLNK(details.st_mode)
            or not stat.S_ISREG(details.st_mode)
            or details.st_size > _MAX_SETUP_FILE_BYTES
        ):
            return "fail", f"{_WORKFLOW_PATH} is not an ordinary file."
        content = workflow.read_bytes()
    except FileNotFoundError:
        return "fail", f"{_WORKFLOW_PATH} is missing."
    except OSError as exc:
        return "fail", f"Could not read {_WORKFLOW_PATH}: {exc}."
    if content != _WORKFLOW.encode("utf-8"):
        return "fail", f"{_WORKFLOW_PATH} does not match Joyride's generated security policy."
    manifest = repository.root / _RUNTIME_MANIFEST_PATH
    try:
        details = manifest.lstat()
        if (
            stat.S_ISLNK(details.st_mode)
            or not stat.S_ISREG(details.st_mode)
            or details.st_size > _MAX_SETUP_FILE_BYTES
        ):
            return "fail", f"{_RUNTIME_MANIFEST_PATH} is not an ordinary file."
        expected_manifest = _runtime_files()[_RUNTIME_MANIFEST_PATH]
        if manifest.read_bytes() != expected_manifest:
            return "fail", "The hosted runtime ownership manifest does not match this Joyride release."
    except FileNotFoundError:
        return "fail", f"{_RUNTIME_MANIFEST_PATH} is missing."
    except HostedSetupError as exc:
        return "fail", str(exc)
    except OSError as exc:
        return "fail", f"Could not read {_RUNTIME_MANIFEST_PATH}: {exc}."
    runtime_state, runtime_message = _runtime_check(repository)
    if runtime_state != "pass":
        return runtime_state, runtime_message
    return "pass", "Trusted workflow and complete hosted runtime match this Joyride release."


def doctor(
    repo: str | Path = ".",
    *,
    check_github: bool = True,
    hosted_url: str | None = None,
) -> dict[str, Any]:
    """Return a safe, structured readiness report without changing state."""

    checks: list[dict[str, str]] = []
    try:
        repository = discover_repository(repo)
    except HostedSetupError as exc:
        return {
            "ready": False,
            "repository": None,
            "checks": [{"name": "github_repository", "state": "fail", "message": str(exc)}],
            "warnings": [],
        }

    checks.append({
        "name": "github_repository",
        "state": "pass",
        "message": f"Detected {repository.full_name} from the {repository.remote} remote.",
    })
    workflow_state, workflow_message = _workflow_check(repository)
    checks.append({"name": "workflow", "state": workflow_state, "message": workflow_message})
    executable = shutil.which("gh")
    if not check_github:
        checks.append(
            {
                "name": "github_cli",
                "state": "pass" if executable else "unknown",
                "message": (
                    "GitHub CLI is installed."
                    if executable
                    else "GitHub CLI availability was not required in offline mode."
                ),
            }
        )
        checks.append({"name": "github_auth", "state": "unknown", "message": "GitHub auth check was skipped."})
        checks.append(
            {
                "name": "default_branch_workflow",
                "state": "unknown",
                "message": "GitHub default-branch workflow verification was skipped.",
            }
        )
    elif executable is None:
        checks.append({"name": "github_cli", "state": "fail", "message": "GitHub CLI (`gh`) is not installed."})
        checks.append({"name": "github_auth", "state": "fail", "message": "GitHub authentication could not be checked."})
        checks.append(
            {
                "name": "default_branch_workflow",
                "state": "unknown",
                "message": "GitHub default-branch workflow could not be verified.",
            }
        )
    else:
        checks.append({"name": "github_cli", "state": "pass", "message": "GitHub CLI is installed."})
        try:
            auth = _gh(repository, "auth", "status", "--hostname", _GITHUB_HOST)
        except HostedSetupError:
            auth = None
        if auth is None or auth.returncode:
            checks.append({"name": "github_auth", "state": "fail", "message": "GitHub CLI is not authenticated."})
            checks.append(
                {
                    "name": "default_branch_workflow",
                    "state": "unknown",
                    "message": "GitHub default-branch workflow could not be verified.",
                }
            )
        else:
            checks.append({"name": "github_auth", "state": "pass", "message": "GitHub CLI authentication is available."})
            remote_state, remote_message = _remote_workflow_check(repository)
            checks.append(
                {
                    "name": "default_branch_workflow",
                    "state": remote_state,
                    "message": remote_message,
                }
            )
            selected_url = hosted_url or os.environ.get("ATTRIBUTION_HOSTED_URL")
            if selected_url:
                try:
                    target = _normalise_hosted_url(selected_url)
                except HostedSetupError as exc:
                    checks.append({"name": "hosted_url", "state": "fail", "message": str(exc)})
                    target = None
            else:
                target = None
            if not selected_url or target is not None:
                variables = _variable_values(repository)
                required = {"ATTRIBUTION_UI_URL", "ATTRIBUTION_INGEST_URL"}
                if variables is None:
                    checks.append({
                        "name": "hosted_variables",
                        "state": "fail",
                        "message": "Could not read hosted repository variables.",
                    })
                else:
                    missing = required - set(variables)
                    if target is None and not missing:
                        try:
                            target = _normalise_hosted_url(variables["ATTRIBUTION_UI_URL"])
                        except HostedSetupError:
                            target = None
                    expected = (
                        {
                            "ATTRIBUTION_UI_URL": target,
                            "ATTRIBUTION_INGEST_URL": target + "/v1/artifacts/pr",
                        }
                        if target is not None
                        else {}
                    )
                    mismatched = {
                        name for name, value in expected.items()
                        if variables.get(name) != value
                    }
                    if missing:
                        message = "Missing repository variables: " + ", ".join(sorted(missing)) + "."
                    elif target is None:
                        message = "Hosted repository variables contain an invalid UI origin."
                    elif mismatched:
                        message = "Repository variables do not match the hosted origin: " + ", ".join(sorted(mismatched)) + "."
                    else:
                        message = "Hosted repository variable values match the hosted origin."
                    checks.append({
                        "name": "hosted_variables",
                        "state": "pass" if not missing and target is not None and not mismatched else "fail",
                        "message": message,
                    })
                    if target is not None:
                        checks.append({
                            "name": "hosted_url",
                            "state": "pass",
                            "message": f"Hosted origin is {target}.",
                        })
    checks.append(
        {
            "name": "github_app_installation",
            "state": "warn" if check_github else "unknown",
            "message": (
                "GitHub App installation and permission state must be verified in the hosted UI; "
                "the user GitHub CLI token cannot securely inspect this App installation."
            ),
        }
    )

    try:
        from .install import installation_status

        local = installation_status(repository.root)
    except (OSError, ValueError) as exc:
        local = {"installed": False, "warnings": [str(exc)]}
    if local.get("installed") is True:
        checks.append({"name": "local_installation", "state": "pass", "message": "Local attribution hooks are healthy."})
    else:
        checks.append({
            "name": "local_installation",
            "state": "warn",
            "message": "Local hooks are not healthy; run `joyride install .` in each developer clone.",
        })
    ready = all(item["state"] in {"pass", "warn"} for item in checks) and not any(
        item["state"] == "fail" for item in checks
    )
    return {
        "ready": ready,
        "repository": {
            "root": str(repository.root),
            "full_name": repository.full_name,
            "remote": repository.remote,
            "remote_url": repository.remote_url,
            "base_branch": repository.base_branch,
            "web_url": repository.web_url,
        },
        "checks": checks,
        "installation": local,
        "warnings": [item["message"] for item in checks if item["state"] == "warn"],
    }


def _setup_destination(worktree: Path, relative: str) -> Path:
    """Return a contained setup path after rejecting unsafe path components."""

    root = Path(worktree)
    try:
        root_details = root.lstat()
    except OSError as exc:
        raise HostedSetupError("The temporary setup worktree is unavailable.") from exc
    if stat.S_ISLNK(root_details.st_mode) or not stat.S_ISDIR(root_details.st_mode):
        raise HostedSetupError("The temporary setup worktree is unsafe.")
    if not isinstance(relative, str):
        raise HostedSetupError("Setup paths must be strings.")
    parsed = PurePosixPath(relative)
    if (
        not relative
        or not parsed.parts
        or parsed.is_absolute()
        or ".." in parsed.parts
        or "\\" in relative
        or "\0" in relative
    ):
        raise HostedSetupError(f"Refusing an unsafe setup path: {relative!r}.")
    resolved_root = root.resolve(strict=True)
    parent = root
    for part in parsed.parts[:-1]:
        parent = parent / part
        try:
            details = parent.lstat()
        except FileNotFoundError:
            try:
                parent.mkdir()
                details = parent.lstat()
            except OSError as exc:
                raise HostedSetupError(f"Could not create a safe setup directory for {relative}.") from exc
        except OSError as exc:
            raise HostedSetupError(f"Could not inspect the setup path: {relative}.") from exc
        if stat.S_ISLNK(details.st_mode) or not stat.S_ISDIR(details.st_mode):
            raise HostedSetupError(f"Refusing an unsafe setup parent for {relative}.")
    try:
        parent.resolve(strict=True).relative_to(resolved_root)
    except (OSError, ValueError) as exc:
        raise HostedSetupError(f"Setup path escapes the temporary worktree: {relative}.") from exc
    destination = parent / parsed.name
    try:
        details = destination.lstat()
    except FileNotFoundError:
        return destination
    except OSError as exc:
        raise HostedSetupError(f"Could not inspect the setup path: {relative}.") from exc
    if stat.S_ISLNK(details.st_mode) or not stat.S_ISREG(details.st_mode):
        raise HostedSetupError(f"Refusing to replace an unsafe setup path: {relative}.")
    return destination


def _stale_runtime_files(worktree: Path, files: Mapping[str, bytes]) -> tuple[str, ...]:
    """Find unchanged, previously owned runtime files that are no longer shipped."""

    if _RUNTIME_MANIFEST_PATH not in files:
        return ()
    manifest_path = _setup_destination(worktree, _RUNTIME_MANIFEST_PATH)
    owned: dict[str, str] = {}
    if manifest_path.exists():
        try:
            details = manifest_path.lstat()
            if details.st_size > _MAX_SETUP_FILE_BYTES:
                raise HostedSetupError("The hosted runtime ownership manifest is too large.")
            owned = runtime_manifest_files(manifest_path.read_bytes())
        except HostedRuntimeError as exc:
            raise HostedSetupError(str(exc)) from exc
        except OSError as exc:
            raise HostedSetupError(
                "The hosted runtime ownership manifest cannot be read safely."
            ) from exc

    runtime_root = worktree.joinpath(*PurePosixPath(_RUNTIME_PREFIX).parts)
    try:
        runtime_root.lstat()
    except FileNotFoundError:
        return ()
    except OSError as exc:
        raise HostedSetupError(
            "The existing hosted runtime cannot be inspected safely."
        ) from exc
    try:
        actual, _directories = installed_runtime_tree(runtime_root)
    except HostedRuntimeError as exc:
        raise HostedSetupError(str(exc)) from exc
    desired = {path for path in files if path.startswith(_RUNTIME_PREFIX)}
    stale: list[str] = []
    for relative, content in sorted(actual.items()):
        path = _RUNTIME_PREFIX + relative
        if path in desired:
            continue
        digest = owned.get(path)
        if digest is None or hashlib.sha256(content).hexdigest() != digest:
            raise HostedSetupError(
                f"The existing hosted runtime contains an unowned or modified file: {path}. "
                "Review or remove it before retrying setup."
            )
        stale.append(path)
    return tuple(stale)


def _prune_empty_runtime_directories(worktree: Path, stale: tuple[str, ...]) -> None:
    runtime_root = worktree.joinpath(*PurePosixPath(_RUNTIME_PREFIX).parts)
    for relative in sorted(stale, key=lambda value: value.count("/"), reverse=True):
        directory = worktree.joinpath(*PurePosixPath(relative).parts).parent
        while directory != runtime_root:
            try:
                details = directory.lstat()
                if stat.S_ISLNK(details.st_mode) or not stat.S_ISDIR(details.st_mode):
                    raise HostedSetupError(
                        "An obsolete hosted runtime directory became unsafe during setup."
                    )
                directory.resolve(strict=True).relative_to(runtime_root.resolve(strict=True))
                directory.rmdir()
            except FileNotFoundError:
                pass
            except OSError as exc:
                if exc.errno in {errno.ENOTEMPTY, errno.EEXIST}:
                    break
                raise HostedSetupError(
                    "Could not remove an obsolete hosted runtime directory."
                ) from exc
            except (RuntimeError, ValueError) as exc:
                raise HostedSetupError(
                    "An obsolete hosted runtime directory escaped the setup root."
                ) from exc
            directory = directory.parent


def _write_files(worktree: Path, files: Mapping[str, bytes]) -> tuple[str, ...]:
    destinations: list[tuple[str, Path, bytes]] = []
    for relative, content in files.items():
        if not isinstance(content, bytes):
            raise HostedSetupError(f"Setup content for {relative} must be bytes.")
        path = _setup_destination(worktree, relative)
        destinations.append((relative, path, content))
    stale = _stale_runtime_files(worktree, files)
    for relative, path, content in destinations:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".attribution-setup-", dir=path.parent
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(content)
            temporary.chmod(0o644)
            os.replace(temporary, path)
        except OSError as exc:
            raise HostedSetupError(f"Could not write the setup file: {relative}.") from exc
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
    for relative in stale:
        path = _setup_destination(worktree, relative)
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        except OSError as exc:
            raise HostedSetupError(
                f"Could not remove the obsolete hosted runtime file: {relative}."
            ) from exc
    _prune_empty_runtime_directories(worktree, stale)
    return stale


def _create_pull_request(plan: HostedSetupPlan) -> str:
    repository = plan.repository
    base_ref = f"{repository.remote}/{plan.base_branch}"
    remote_branch = f"refs/heads/{plan.branch}"
    exists = _command(
        [
            "git",
            "-C",
            str(repository.root),
            "ls-remote",
            "--exit-code",
            "--heads",
            "--",
            repository.remote,
            remote_branch,
        ],
        cwd=repository.root,
        timeout=120,
    )
    branch_exists = False
    if exists.returncode == 0:
        lines = [line for line in (exists.stdout or "").splitlines() if line]
        if (
            len(lines) != 1
            or not lines[0].endswith("\t" + remote_branch)
            or re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", lines[0].split("\t", 1)[0])
            is None
        ):
            raise HostedSetupError("Git returned invalid setup-branch information.")
        branch_exists = True
    if exists.returncode not in {2}:
        if not branch_exists:
            raise HostedSetupError(
                "Could not verify whether the remote setup branch already exists."
            )
    _git(repository.root, "fetch", "--quiet", "--", repository.remote, plan.base_branch)
    create_commit = True
    if branch_exists:
        advanced = _command(
            [
                "git",
                "-C",
                str(repository.root),
                "push",
                "--",
                repository.remote,
                f"{base_ref}:{remote_branch}",
            ],
            cwd=repository.root,
            timeout=120,
        )
        if advanced.returncode:
            if not _remote_setup_branch_matches(repository, plan):
                raise HostedSetupError(
                    f"The remote setup branch {plan.branch!r} contains changes Joyride "
                    "cannot recover. Review or remove it before retrying."
                )
            existing_url = _existing_setup_pull_request(repository, plan)
            if existing_url:
                if _remote_setup_branch_matches(repository, plan):
                    return existing_url
                raise HostedSetupError(
                    "The remote setup branch changed while its pull request was inspected."
                )
            create_commit = False
    if create_commit:
        with tempfile.TemporaryDirectory(prefix="attribution-hosted-") as directory:
            worktree = Path(directory) / "checkout"
            added = _command(
                ["git", "-C", str(repository.root), "worktree", "add", "--detach", str(worktree), base_ref],
                cwd=repository.root,
            )
            if added.returncode:
                message = (added.stderr or "").strip()
                raise HostedSetupError(message or "Could not create an isolated setup worktree.")
            try:
                removed_files = _write_files(worktree, plan.files)
                add = _command(
                    ["git", "-C", str(worktree), "add", "--", *plan.files, *removed_files],
                    cwd=worktree,
                )
                if add.returncode:
                    raise HostedSetupError("Could not stage the hosted setup files.")
                changed = _command(["git", "-C", str(worktree), "diff", "--cached", "--quiet"], cwd=worktree)
                if changed.returncode == 0:
                    raise HostedSetupError(
                        "The hosted setup files already match the base branch; no reviewable PR is needed."
                    )
                commit = _command(
                    ["git", "-C", str(worktree), "-c", "user.name=Joyride Setup", "-c", "user.email=attribution-setup@users.noreply.github.com", "commit", "--message", plan.title],
                    cwd=worktree,
                )
                if commit.returncode:
                    raise HostedSetupError((commit.stderr or "").strip() or "Could not commit the hosted setup branch.")
                push = _command(
                    [
                        "git",
                        "-C",
                        str(worktree),
                        "push",
                        "--",
                        repository.remote,
                        f"HEAD:{remote_branch}",
                    ],
                    cwd=worktree,
                    timeout=120,
                )
                if push.returncode:
                    raise HostedSetupError((push.stderr or "").strip() or "Could not push the hosted setup branch.")
            finally:
                removed = _command(["git", "-C", str(repository.root), "worktree", "remove", "--force", str(worktree)], cwd=repository.root)
                if removed.returncode:
                    # The temporary directory is still removed by its context
                    # manager. Keep the cleanup issue visible without masking the
                    # actual setup result.
                    pass
    pr = _gh(
        repository,
        "pr",
        "create",
        "--repo",
        repository.full_name,
        "--head",
        plan.branch,
        "--base",
        plan.base_branch,
        "--title",
        plan.title,
        "--body",
        plan.body,
    )
    if pr.returncode:
        if _remote_setup_branch_matches(repository, plan):
            existing_url = _existing_setup_pull_request(repository, plan)
            if existing_url and _remote_setup_branch_matches(repository, plan):
                return existing_url
        raise HostedSetupError((pr.stderr or "").strip() or "Could not create the setup pull request.")
    url = (pr.stdout or "").strip().splitlines()[-1:]
    if not url or not url[0].startswith("https://github.com/"):
        raise HostedSetupError("GitHub did not return a setup pull-request URL.")
    if not _remote_setup_branch_matches(repository, plan):
        raise HostedSetupError(
            "The remote setup branch changed while its pull request was created."
        )
    return url[0]


def _set_variables(repository: GitHubRepository, hosted_url: str) -> tuple[str, ...]:
    variables = {
        "ATTRIBUTION_UI_URL": hosted_url,
        "ATTRIBUTION_INGEST_URL": hosted_url + "/v1/artifacts/pr",
    }
    warnings: list[str] = []
    for name, value in variables.items():
        result = _gh(
            repository,
            "variable",
            "set",
            name,
            "--repo",
            repository.full_name,
            "--body",
            value,
        )
        if result.returncode:
            warnings.append(f"Could not configure {name}; set it with `gh variable set` before merging.")
    return tuple(warnings)


def setup(
    repo: str | Path = ".",
    *,
    hosted_url: str | None = None,
    branch: str = _DEFAULT_SETUP_BRANCH,
    base_branch: str | None = None,
    dry_run: bool = False,
    open_browser: bool = True,
    open_pull_request: bool = True,
) -> HostedSetupResult:
    """Create a hosted setup PR from an isolated worktree."""

    # A one-file build cannot reproduce reviewable source bytes. Reject before
    # repository discovery or GitHub authentication can perform external work.
    _require_source_runtime()
    repository = discover_repository(repo)
    selected_url = hosted_url or os.environ.get("ATTRIBUTION_HOSTED_URL")
    if dry_run:
        plan = make_setup_plan(
            repository,
            hosted_url=selected_url,
            branch=branch,
            base_branch=base_branch,
        )
        return HostedSetupResult(
            "dry-run",
            repository,
            plan.branch,
            plan.base_branch,
            tuple(sorted(plan.files)),
            hosted_url=plan.hosted_url,
        )
    ensure_github_auth(repository, open_browser=open_browser)
    selected_base = base_branch or _github_default_branch(repository) or repository.base_branch
    plan = make_setup_plan(
        repository,
        hosted_url=selected_url,
        branch=branch,
        base_branch=selected_base,
    )
    pull_request_url = _create_pull_request(plan)
    warnings: tuple[str, ...] = ()
    if plan.hosted_url:
        warnings = _set_variables(repository, plan.hosted_url)
    if open_pull_request:
        try:
            webbrowser.open(pull_request_url)
        except Exception:
            warnings = (*warnings, "The pull request was created, but the browser could not be opened.")
    return HostedSetupResult(
        "created",
        repository,
        plan.branch,
        plan.base_branch,
        tuple(sorted(plan.files)),
        pull_request_url,
        plan.hosted_url,
        warnings,
    )


__all__ = [
    "GitHubRepository",
    "HostedSetupError",
    "HostedSetupPlan",
    "HostedSetupResult",
    "discover_repository",
    "doctor",
    "ensure_github_auth",
    "make_setup_plan",
    "setup",
]
