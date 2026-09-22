"""An isolated example made with the same capture and notes code as real data."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from .capture import run_session
from .notes import record_commit
from .runtime import system_subprocess_environment
from .task_notes import sync_task_if_anchored
from .tasks import add_session


def _git(repo: Path, *args: str, date: datetime | None = None) -> str:
    env = os.environ.copy()
    if date is not None:
        stamp = date.isoformat()
        env.update(GIT_AUTHOR_DATE=stamp, GIT_COMMITTER_DATE=stamp)
    process = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=system_subprocess_environment(env),
    )
    return process.stdout.strip()


def _write_command(path: str, content: str) -> list[str]:
    return [
        sys.executable,
        "-c",
        f"from pathlib import Path; Path({path!r}).write_text({content!r}, encoding='utf-8')",
    ]


def _backdate(repo: Path, session_id: str, date: datetime) -> None:
    from .store import open_db

    db = open_db(repo)
    try:
        with db:
            db.execute(
                "UPDATE sessions SET started_at = ?, ended_at = ? WHERE id = ?",
                (date.isoformat(), (date + timedelta(minutes=3)).isoformat(), session_id),
            )
    finally:
        db.close()


def _capture(
    repo: Path,
    *,
    feature: str,
    model: str,
    harness: str,
    cost: float | None,
    content: str,
    path: str,
    date: datetime,
    role: str = "implementation",
    summary: str | None = None,
    tokens: int | None = None,
) -> dict:
    result = run_session(
        repo,
        feature,
        model,
        harness,
        cost,
        _write_command(path, content),
        role=role,
        summary=summary,
        token_count=tokens,
    )
    if result["exit_code"] != 0:
        raise ValueError("The example coding command failed.")
    # Only this new fixture is backdated. Production capture timestamps are unchanged.
    _backdate(repo, result["session_id"], date)
    return result


def create_demo() -> Path:
    """Return a new example repository; never read or change the caller's repository."""
    repo = Path(tempfile.mkdtemp(prefix="attribution-example-"))
    now = datetime.now(timezone.utc).replace(microsecond=0)
    start = now - timedelta(days=45)
    _git(repo, "init", "--initial-branch=main")
    _git(repo, "config", "user.name", "Example developer")
    _git(repo, "config", "user.email", "example@localhost")
    _git(repo, "config", "commit.gpgsign", "false")
    # This is a new, private fixture. Disable inherited hooks for deterministic setup.
    _git(repo, "config", "core.hooksPath", str(repo / ".git" / "no-hooks"))
    (repo / "README.md").write_text("# Example shop\n\nFictional data for the Joyride demo.\n", encoding="utf-8")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-m", "Create example shop", date=start)

    marker = repo / ".git" / "attribution" / "demo.json"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps({"example_data": True}), encoding="utf-8")

    _git(repo, "checkout", "-b", "feature/coupons")
    planning = run_session(
        repo,
        "Coupon stacking",
        "Model A",
        "Harness X",
        0.05,
        [sys.executable, "-c", "pass"],
        role="planning",
        summary="Planned coupon precedence and edge cases",
        token_count=50_000,
    )
    _backdate(repo, planning["session_id"], start + timedelta(minutes=15))
    original = (
        "def apply_coupon(total, code):\n"
        "    discounts = {\n"
        '        "WELCOME": 0.10,\n'
        '        "LOYAL": 0.15,\n'
        "    }\n"
        "    rate = discounts.get(code, 0)\n"
        "    discount = round(total * rate, 2)\n"
        "    discounted = total - discount\n"
        "    return max(discounted, 0)\n"
        "\n"
    )
    _capture(
        repo, feature="Coupon stacking", model="Model A", harness="Harness X",
        cost=0.30, content=original, path="coupons.py", date=start + timedelta(hours=1),
        summary="Implemented the first coupon design", tokens=300_000,
    )
    revised = original.replace('"WELCOME": 0.10', '"WELCOME": 0.20').replace(
        '"LOYAL": 0.15', '"LOYAL": 0.25'
    ).replace("discounts.get(code, 0)", "discounts.get(code.upper(), 0)")
    _capture(
        repo, feature="Coupon stacking", model="Model B", harness="Harness Y",
        cost=0.10, content=revised, path="coupons.py", date=start + timedelta(hours=2),
        summary="Reworked coupon normalization and rates", tokens=100_000,
    )
    _git(repo, "add", "coupons.py")
    _git(repo, "commit", "-m", "Add coupon stacking", date=start + timedelta(hours=3))
    record_commit(repo)
    review = add_session(
        repo,
        task_query=planning["task"]["id"],
        model="Model B",
        harness="Harness Y",
        role="review",
        summary="Reviewed the landed behavior and edge cases",
        token_count=50_000,
        cost_usd=0.05,
    )
    _backdate(repo, review["session_id"], start + timedelta(hours=4))
    sync_task_if_anchored(repo, planning["task"]["id"])
    _git(repo, "checkout", "main")
    _git(repo, "merge", "--no-ff", "feature/coupons", "-m", "Merge coupon stacking", date=start + timedelta(days=1))

    # Two originally A-attributed lines change outside a capture, one month later.
    # They stay unknown; their revision records still point back to A.
    maintained = revised.replace(
        "discount = round(total * rate, 2)", "discount = round(max(total, 0) * rate, 2)"
    ).replace("discounted = total - discount", "discounted = round(total - discount, 2)")
    (repo / "coupons.py").write_text(maintained, encoding="utf-8")
    _git(repo, "add", "coupons.py")
    _git(repo, "commit", "-m", "Handle coupon rounding and negative totals", date=start + timedelta(days=31))
    record_commit(repo)

    # A second feature illustrates real captured work that has not landed on main.
    _git(repo, "checkout", "-b", "feature/exports")
    export = (
        "import csv\n"
        "import io\n"
        "\n"
        "\n"
        "def export_orders(orders):\n"
        "    buffer = io.StringIO()\n"
        "    writer = csv.writer(buffer)\n"
        '    writer.writerow(["order_id", "total"])\n'
        "    for order in orders:\n"
        '        writer.writerow([order["id"], order["total"]])\n'
        "    return buffer.getvalue()\n"
    )
    _capture(
        repo, feature="CSV exports", model="Model A", harness="Harness X",
        cost=None, content=export, path="exports.py", date=now - timedelta(hours=2),
        summary="Implemented CSV exports", tokens=None,
    )
    _git(repo, "add", "exports.py")
    _git(repo, "commit", "-m", "Export orders to CSV", date=now - timedelta(hours=1))
    record_commit(repo)
    _git(repo, "checkout", "main")
    return repo
