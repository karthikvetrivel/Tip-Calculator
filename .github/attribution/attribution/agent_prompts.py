"""The setup prompts that a customer's own coding agent runs.

The README, `docs/github-actions.md`, and the hosted Setup page show the same
two prompts, so the text lives here once. `render` fills the values that vary
by deployment: the hosted Setup page pins the install source to the revision it
runs and names its own hosted URL, while the guides keep the defaults.
"""

from __future__ import annotations

INSTALL_SOURCE = "git+https://github.com/valite-ai/attribution-hosted"
HOSTED_URL = "<HOSTED_URL>"

DEVELOPER_PROMPT = """Set up Joyride local capture in this repository. Joyride records which coding agent sessions shaped a change and what they cost. Work from the repository root and follow these steps in order. Stop and report if any step fails.

1. Run `git status --short`. Do not commit anything during this task.
2. Install the Joyride CLI once for this machine. Prefer uv: run `uv tool install <INSTALL_SOURCE>`. If uv is missing but pipx exists, run `pipx install <INSTALL_SOURCE>`. If neither exists, stop and tell me to install uv from https://docs.astral.sh/uv/. Do not use `uvx`, because the hooks need an installation that persists.
3. Make sure that `joyride --version` prints a version. If the command is not on PATH, run `uv tool update-shell`, tell me to open a new shell, and stop.
4. Run `joyride install .` from the repository root. This writes hook entries for Claude Code and Codex into untracked local files, adds a Git pre-push hook that publishes bounded metadata, and starts a local collector on the loopback interface. The install itself sends nothing. Each later push publishes that metadata with a redacted, capped trace of each session, which holds prompts, replies, and tool activity. Add `--no-traces` to the install command to keep and publish no trace.
5. Run `joyride status` and show me the result.
6. List every file that the install created or changed. Make sure that `git status --short` shows no new tracked changes.
7. Finish with this exact instruction, because hooks load only when a session starts: "Restart this session now. If you are in Codex, run /hooks after the restart and trust the Joyride hook definitions." Then stop."""

ADMIN_PROMPT = """Create the Joyride hosted setup pull request for this repository. Joyride adds a trusted GitHub Actions workflow that posts an attribution footer on each pull request. Work from the repository root. Stop and report if any step fails.

1. Make sure that `joyride --version` prints a version. If it does not, run `uv tool install <INSTALL_SOURCE>` first.
2. Run `gh auth status`. If the GitHub CLI is missing or not signed in, tell me to run `gh auth login --web` myself, then stop. Never ask me for a token, and never put a token in a command or a file.
3. Run `joyride hosted setup --repo . --hosted-url <HOSTED_URL> --dry-run` and show me the planned files and variables.
4. Run `joyride hosted setup --repo . --hosted-url <HOSTED_URL> --no-open --json` and give me the pull request URL. The command works in a temporary worktree and does not change tracked files here.
5. Tell me to review and merge that pull request, and to run `joyride doctor --repo . --hosted-url <HOSTED_URL>` after the merge. Then stop."""


def render(
    prompt: str,
    *,
    hosted_url: str = HOSTED_URL,
    install_source: str = INSTALL_SOURCE,
) -> str:
    """Return one prompt with its deployment values filled in."""

    return prompt.replace("<INSTALL_SOURCE>", install_source).replace(
        "<HOSTED_URL>", hosted_url
    )
