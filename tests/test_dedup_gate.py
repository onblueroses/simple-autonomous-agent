"""C3 gate: invokes scripts/check_dedup.py and asserts all sync/async pairs are within budget."""

import pathlib
import subprocess
import sys


_SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "check_dedup.py"


def test_sync_async_dedup_within_budget():
    result = subprocess.run(
        [sys.executable, str(_SCRIPT)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        "check_dedup reported failures.\n"
        f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
    )
