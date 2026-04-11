"""Fresh-start cleanup utilities for thesis runtime outputs."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import json
import shutil


PROTECTED_ROOTS = (
    "uvg",
    "runtime",
    "scripts",
    "pipeline",
    "documents",
)


def _now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _is_protected(path: Path, thesis_root: Path) -> bool:
    resolved = path.resolve()
    for rel in PROTECTED_ROOTS:
        if _is_relative_to(resolved, (thesis_root / rel).resolve()):
            return True
    return False


def _add_candidate(candidates: list[Path], seen: set[Path], path: Path) -> None:
    resolved = path.resolve()
    if resolved in seen:
        return
    seen.add(resolved)
    candidates.append(resolved)


def collect_cleanup_candidates(
    *,
    thesis_root: Path,
    run_name: str | None,
    include_caches: bool,
) -> list[Path]:
    candidates: list[Path] = []
    seen: set[Path] = set()

    runs_root = (thesis_root / "runs").resolve()
    if run_name:
        _add_candidate(candidates, seen, runs_root / run_name)
    else:
        if runs_root.exists():
            for child in sorted(runs_root.iterdir()):
                if child.is_dir():
                    _add_candidate(candidates, seen, child)

    if run_name is None:
        logs_root = (thesis_root / "logs").resolve()
        _add_candidate(candidates, seen, logs_root)

        cleanup_reports_root = (thesis_root / "cleanup_reports").resolve()
        _add_candidate(candidates, seen, cleanup_reports_root)

    if include_caches:
        for cache_dir in thesis_root.rglob("__pycache__"):
            if cache_dir.is_dir():
                _add_candidate(candidates, seen, cache_dir)
        for pyc_file in thesis_root.rglob("*.pyc"):
            if pyc_file.is_file():
                _add_candidate(candidates, seen, pyc_file)

    return sorted(candidates)


def cleanup_thesis_outputs(
    *,
    repo_root: Path,
    run_name: str | None = None,
    include_caches: bool = True,
    execute: bool = False,
    manifest_path: Path | None = None,
) -> dict:
    thesis_root = (repo_root / "thesis").resolve()
    if not thesis_root.exists():
        raise FileNotFoundError(f"Missing thesis root: {thesis_root}")

    candidates = collect_cleanup_candidates(
        thesis_root=thesis_root,
        run_name=run_name,
        include_caches=include_caches,
    )

    removed: list[str] = []
    planned: list[str] = []
    skipped: list[dict[str, str]] = []
    missing: list[str] = []

    for candidate in candidates:
        if _is_protected(candidate, thesis_root):
            skipped.append({"path": str(candidate), "reason": "protected-root"})
            continue
        if not _is_relative_to(candidate, thesis_root):
            skipped.append({"path": str(candidate), "reason": "outside-thesis-root"})
            continue
        if not candidate.exists():
            missing.append(str(candidate))
            continue

        planned.append(str(candidate))
        if not execute:
            continue

        if candidate.is_dir():
            shutil.rmtree(candidate)
        else:
            candidate.unlink()
        removed.append(str(candidate))

    report = {
        "timestamp": _now_iso(),
        "mode": "execute" if execute else "dry-run",
        "thesis_root": str(thesis_root),
        "protected_roots": [str((thesis_root / rel).resolve()) for rel in PROTECTED_ROOTS],
        "run_name": run_name,
        "include_caches": include_caches,
        "planned_paths": planned,
        "removed_paths": removed,
        "skipped_paths": skipped,
        "missing_paths": missing,
    }

    target = manifest_path
    if target is None:
        reports_root = thesis_root / "cleanup_reports"
        reports_root.mkdir(parents=True, exist_ok=True)
        suffix = "execute" if execute else "dry_run"
        stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        target = reports_root / f"cleanup_{suffix}_{stamp}.json"

    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    report["manifest_path"] = str(target)

    return report


def validate_cleanup_safety(
    *,
    repo_root: Path,
    include_caches: bool = True,
) -> dict[str, int]:
    """Validate cleanup planning never includes protected thesis paths for removal."""
    report = cleanup_thesis_outputs(
        repo_root=repo_root,
        run_name=None,
        include_caches=include_caches,
        execute=False,
        manifest_path=(repo_root / "thesis" / "cleanup_reports" / "cleanup_safety_validation.json"),
    )

    thesis_root = (repo_root / "thesis").resolve()
    protected_violations: list[str] = []
    for entry in report["planned_paths"]:
        path = Path(entry).resolve()
        if _is_protected(path, thesis_root):
            protected_violations.append(entry)

    if protected_violations:
        joined = "\n".join(protected_violations)
        raise RuntimeError(
            "Cleanup safety validation failed: protected paths planned for removal:\n"
            f"{joined}"
        )

    return {
        "planned_paths": len(report["planned_paths"]),
        "protected_skips": len(report["skipped_paths"]),
        "violations": 0,
    }
