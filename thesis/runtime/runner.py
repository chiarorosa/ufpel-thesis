"""Subprocess runner for canonical thesis entrypoints."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence
import shlex
import subprocess


def run_command(args: Sequence[str], cwd: Path) -> None:
    command = " ".join(shlex.quote(part) for part in args)
    print(f"[cmd] {command}")
    result = subprocess.run(args, cwd=str(cwd), check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}: {command}"
        )
