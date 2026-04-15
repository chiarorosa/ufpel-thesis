#!/usr/bin/env python3
"""Download and sync thesis/uvg contract data from Google Drive."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thesis.runtime import DEFAULT_UVG_DRIVE_URL, bootstrap_uvg_from_drive
from thesis.scripts._common import repo_root_from_script


def _default_temp_root(repo_root: Path) -> str:
    return str((repo_root / "thesis" / "runs" / "_bootstrap" / "uvg_drive").relative_to(repo_root))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Download UVG contract data from a public Google Drive ZIP file and "
            "sync it into thesis/uvg."
        )
    )
    parser.add_argument(
        "--drive-url",
        default=DEFAULT_UVG_DRIVE_URL,
        help="Public Google Drive ZIP file URL.",
    )
    parser.add_argument(
        "--target-root",
        default="thesis/uvg",
        help="Target root for UVG contract data.",
    )
    parser.add_argument(
        "--temp-root",
        default=None,
        help="Temporary download staging root.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files in target root.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan sync using existing staged files only; no download/move.",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip post-sync contract validation.",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep temporary staging directory after success.",
    )
    parser.add_argument(
        "--fresh-download",
        action="store_true",
        help="Clear temporary download cache before fetching files.",
    )
    parser.add_argument(
        "--max-retries-per-mode",
        type=int,
        default=2,
        help="Retries per cookie mode for ZIP download.",
    )
    parser.add_argument(
        "--retry-backoff-seconds",
        type=float,
        default=1.5,
        help="Base backoff delay (seconds) between retries.",
    )
    parser.add_argument(
        "--archive-timeout-seconds",
        type=int,
        default=1800,
        help="Timeout in seconds for each ZIP archive download attempt.",
    )
    parser.add_argument(
        "--no-auto-bootstrap-legacy-contract",
        action="store_true",
        help="Do not generate labels/qps from partitions when missing.",
    )
    parser.add_argument(
        "--min-intra-raw-sequences",
        type=int,
        default=1,
        help="Minimum sequence count for intra_raw_blocks validation.",
    )
    parser.add_argument(
        "--python",
        default=None,
        help="Python executable used to invoke gdown (defaults to current interpreter).",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    repo_root = repo_root_from_script(Path(__file__))
    temp_root_rel = args.temp_root or _default_temp_root(repo_root)

    report = bootstrap_uvg_from_drive(
        repo_root=repo_root,
        drive_url=args.drive_url,
        target_root=(repo_root / args.target_root).resolve(),
        temp_root=(repo_root / temp_root_rel).resolve(),
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        skip_validation=args.skip_validation,
        keep_temp=args.keep_temp,
        min_intra_raw_sequences=args.min_intra_raw_sequences,
        python_executable=args.python,
        clear_existing_download_cache=args.fresh_download,
        max_retries_per_mode=max(1, args.max_retries_per_mode),
        retry_backoff_seconds=max(0.0, args.retry_backoff_seconds),
        archive_timeout_seconds=max(120, args.archive_timeout_seconds),
        auto_bootstrap_legacy_contract=not args.no_auto_bootstrap_legacy_contract,
    )

    print(f"Drive URL: {report['drive_url']}")
    print(f"Target root: {report['target_root']}")
    print(f"Temp root: {report['temp_root']}")
    print(f"Dry run: {report['dry_run']}")
    print(f"Downloaded: {report['downloaded']}")
    print(f"Download exit code: {report['download_exit_code']}")
    print(f"Download mode: {report['download_mode']}")
    print(f"Resolved download mode: {report['resolved_download_mode']}")
    print(f"Fresh download: {report['clear_existing_download_cache']}")
    print(f"Retries per mode: {report['max_retries_per_mode']}")
    print(f"Archive timeout seconds: {report['archive_timeout_seconds']}")
    print(f"Auto legacy bootstrap: {report['auto_bootstrap_legacy_contract']}")
    print(f"Retry backoff seconds: {report['retry_backoff_seconds']}")
    print(f"Moved files: {report['moved_files']}")
    print(f"Moved bytes: {report['moved_bytes']}")
    print(f"Skipped existing files: {report['skipped_existing_files']}")
    source_root = report.get("source_root")
    if source_root:
        print(f"Selected source root: {source_root}")

    skipped_preview = report.get("skipped_existing_preview") or []
    if skipped_preview:
        print("Skipped existing preview:")
        for path in skipped_preview:
            print(f"- {path}")

    download_errors = report.get("download_errors") or []
    if download_errors:
        print("Download warnings:")
        for item in download_errors:
            print(f"- {item}")

    fallback = report.get("best_effort_fallback")
    if fallback:
        print("Best-effort fallback:")
        print(f"- Listed files: {fallback['listed_files']}")
        print(f"- Selected contract files: {fallback['selected_contract_files']}")
        print(f"- Downloaded files: {fallback['downloaded_files']}")
        print(f"- Reused existing files: {fallback['reused_existing_files']}")
        archive_source = fallback.get("archive_source")
        if archive_source:
            print(f"- Archive source: {archive_source}")
        archive_path = fallback.get("archive_path")
        if archive_path:
            print(f"- Archive path: {archive_path}")
        failed = fallback.get("failed_files") or []
        print(f"- Failed files: {len(failed)}")

    validation = report.get("validation") or {}
    if validation:
        print("Validation: ok")
        flow = validation.get("flow_validation")
        if flow is not None:
            print(f"Flow validation ok: {flow['ok']}")
            print(f"Flow warnings: {len(flow['warnings'])}")

    validated_sequences = report.get("validated_sequences") or []
    if validated_sequences:
        print(f"Validated sequences: {len(validated_sequences)}")

    legacy_bootstrap = report.get("legacy_bootstrap") or {}
    if legacy_bootstrap.get("executed"):
        print("Legacy bootstrap:")
        print(f"- Sequences: {len(legacy_bootstrap.get('sequences') or [])}")
        print(f"- Stats count: {legacy_bootstrap.get('stats_count', 0)}")

    print(f"Temp cleaned: {report['temp_cleaned']}")


if __name__ == "__main__":
    main()
