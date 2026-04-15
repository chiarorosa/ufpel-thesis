"""Google Drive bootstrap helpers for thesis/uvg contract data."""

from __future__ import annotations

from pathlib import Path
import importlib.util
import os
import re
import shutil
import subprocess
import sys
import time
import zipfile

from .contracts import (
    discover_sequences,
    ensure_dir,
    validate_intra_raw_blocks_contract,
    validate_raw_input_contract,
)
from .flow_validation import validate_expected_raw_flow
from .legacy_contract import bootstrap_legacy_labels_qps_from_partition


DEFAULT_UVG_DRIVE_URL = (
    "https://drive.google.com/file/d1c3uY4yeOgpyc8O2ta2kLMXP2Z5FU5d-8"
)

_SAMPLE_FILE_RE = re.compile(r"^.+_sample_(8|16|32|64)\.txt$")
_LABEL_FILE_RE = re.compile(r"^.+_labels_(8|16|32|64)_intra\.txt$")
_QPS_FILE_RE = re.compile(r"^.+_qps_(8|16|32|64)_intra\.txt$")
_DRIVE_FILE_ID_PATTERNS = (
    re.compile(r"/file/d/([A-Za-z0-9_-]+)"),
    re.compile(r"/file/d([A-Za-z0-9_-]+)"),
    re.compile(r"[?&]id=([A-Za-z0-9_-]+)"),
)


def _is_uvg_contract_root(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False

    has_sequence_partition = any(
        child.is_dir() and (child / "partition_frame_0.txt").exists()
        for child in path.iterdir()
    )
    if has_sequence_partition:
        return True

    has_contract_dirs = any(
        (path / name).is_dir() for name in ("intra_raw_blocks", "labels", "qps")
    )
    return has_contract_dirs


def _discover_uvg_candidates(download_root: Path, max_depth: int = 4) -> list[Path]:
    candidates: list[Path] = []
    if not download_root.exists():
        return candidates

    for dirpath, dirnames, _ in os.walk(download_root):
        current = Path(dirpath)
        depth = len(current.relative_to(download_root).parts)
        if depth > max_depth:
            dirnames[:] = []
            continue
        if _is_uvg_contract_root(current):
            candidates.append(current)

    return sorted(set(candidates))


def _select_uvg_source_root(download_root: Path) -> Path:
    candidates = _discover_uvg_candidates(download_root)
    if not candidates:
        raise FileNotFoundError(
            "No valid UVG contract root found after download. "
            "Expected either a folder named 'uvg' or a root containing "
            "<sequence>/partition_frame_0.txt."
        )

    def sort_key(path: Path) -> tuple[int, int]:
        depth = len(path.relative_to(download_root).parts)
        preferred_name = 0 if path.name.lower() == "uvg" else 1
        return (preferred_name, depth)

    ordered = sorted(candidates, key=lambda path: (sort_key(path), str(path)))
    best = ordered[0]
    best_key = sort_key(best)
    ties = [path for path in ordered if sort_key(path) == best_key]
    if len(ties) > 1:
        joined = "\n".join(str(path) for path in ties)
        raise RuntimeError(
            "Ambiguous UVG contract roots found in downloaded content. "
            "Please inspect and select one manually:\n"
            f"{joined}"
        )
    return best


def _sync_tree(
    *,
    source_root: Path,
    target_root: Path,
    overwrite: bool,
    dry_run: bool,
    skip_preview_limit: int = 50,
) -> tuple[int, int, int, list[str]]:
    moved_files = 0
    moved_bytes = 0
    skipped_existing_files = 0
    skipped_preview: list[str] = []

    files = sorted(path for path in source_root.rglob("*") if path.is_file())
    for source_file in files:
        relative = source_file.relative_to(source_root)
        target_file = target_root / relative

        if target_file.exists() and not overwrite:
            skipped_existing_files += 1
            if len(skipped_preview) < skip_preview_limit:
                skipped_preview.append(str(target_file))
            continue

        file_size = source_file.stat().st_size
        moved_files += 1
        moved_bytes += file_size
        if dry_run:
            continue

        ensure_dir(target_file.parent)
        if target_file.exists():
            if target_file.is_dir():
                shutil.rmtree(target_file)
            else:
                target_file.unlink()
        shutil.move(str(source_file), str(target_file))

    return moved_files, moved_bytes, skipped_existing_files, skipped_preview


def _is_gdown_installed(python_executable: str) -> bool:
    if python_executable == sys.executable:
        return importlib.util.find_spec("gdown") is not None
    probe = subprocess.run(
        [python_executable, "-c", "import gdown"],
        check=False,
        capture_output=True,
        text=True,
    )
    return probe.returncode == 0


def _has_any_downloaded_file(download_root: Path) -> bool:
    if not download_root.exists():
        return False
    return any(path.is_file() for path in download_root.rglob("*"))


def _is_valid_zip_archive(archive_path: Path) -> bool:
    if not archive_path.exists() or not archive_path.is_file() or archive_path.stat().st_size <= 0:
        return False
    try:
        with zipfile.ZipFile(archive_path, "r") as bundle:
            return bundle.testzip() is None
    except zipfile.BadZipFile:
        return False


def _extract_drive_file_id(drive_url: str) -> str | None:
    for pattern in _DRIVE_FILE_ID_PATTERNS:
        match = pattern.search(drive_url)
        if match:
            return match.group(1)
    return None


def _normalize_drive_file_download_source(drive_url: str) -> str:
    file_id = _extract_drive_file_id(drive_url)
    if file_id:
        return f"https://drive.google.com/uc?id={file_id}"
    return drive_url


def _download_with_gdown_cli(
    *,
    source: str,
    output_path: Path,
    python_executable: str,
    use_cookies: bool,
    timeout_seconds: int,
) -> tuple[bool, str]:
    args = [
        python_executable,
        "-m",
        "gdown",
        source,
        "-O",
        str(output_path),
        "--continue",
        "--quiet",
    ]
    if not use_cookies:
        args.append("--no-cookies")

    try:
        result = subprocess.run(
            args,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        return False, f"timeout after {timeout_seconds}s"

    if result.returncode == 0 and output_path.exists() and output_path.stat().st_size > 0:
        return True, "ok"

    stderr = (result.stderr or "").strip()
    stdout = (result.stdout or "").strip()
    detail = stderr or stdout or f"exit code {result.returncode}"
    if len(detail) > 300:
        detail = detail[:300] + "..."
    return False, detail


def _is_contract_file(relative_path: str) -> bool:
    normalized = relative_path.replace("\\", "/")
    name = Path(normalized).name

    if name.startswith("partition_frame_") and name.endswith(".txt"):
        return True
    if normalized.startswith("intra_raw_blocks/") and _SAMPLE_FILE_RE.match(name):
        return True
    if normalized.startswith("labels/") and _LABEL_FILE_RE.match(name):
        return True
    if normalized.startswith("qps/") and _QPS_FILE_RE.match(name):
        return True
    return False


def _download_zip_archive_contract_files(
    *,
    drive_url: str,
    download_root: Path,
    python_executable: str,
    max_retries_per_mode: int = 2,
    archive_timeout_seconds: int = 1800,
    retry_backoff_seconds: float = 1.5,
) -> dict[str, object]:
    archive_path = download_root / "uvg_contract_bundle.zip"
    extract_root = download_root / "zip_extract"
    archive_source = _normalize_drive_file_download_source(drive_url)

    if archive_path.exists() and archive_path.is_file() and archive_path.stat().st_size == 0:
        archive_path.unlink()
    if archive_path.exists() and archive_path.is_file() and not _is_valid_zip_archive(archive_path):
        archive_path.unlink()

    reused_existing_archive = _is_valid_zip_archive(archive_path)
    no_cookies_attempts = 0
    no_cookies_successes = 0
    failed_files: list[dict[str, str]] = []

    if not reused_existing_archive:
        success = False
        last_error = "unknown error"
        for use_cookies in (True, False):
            for attempt in range(1, max_retries_per_mode + 1):
                if not use_cookies:
                    no_cookies_attempts += 1
                ok, detail = _download_with_gdown_cli(
                    source=archive_source,
                    output_path=archive_path,
                    python_executable=python_executable,
                    use_cookies=use_cookies,
                    timeout_seconds=archive_timeout_seconds,
                )
                if ok:
                    if _is_valid_zip_archive(archive_path):
                        if not use_cookies:
                            no_cookies_successes += 1
                        success = True
                        break
                    last_error = "downloaded content is not a valid ZIP archive"
                    if archive_path.exists() and archive_path.is_file():
                        archive_path.unlink()
                    continue
                last_error = detail
                if archive_path.exists() and archive_path.is_file() and archive_path.stat().st_size == 0:
                    archive_path.unlink()
                if attempt < max_retries_per_mode and retry_backoff_seconds > 0:
                    sleep_seconds = retry_backoff_seconds * attempt
                    time.sleep(sleep_seconds)
            if success:
                break

        if not success and not _is_valid_zip_archive(archive_path):
            failed_files.append(
                {
                    "path": str(archive_path),
                    "id": archive_source,
                    "error": last_error,
                }
            )
            return {
                "listed_files": 1,
                "selected_contract_files": 0,
                "downloaded_files": 0,
                "reused_existing_files": 0,
                "no_cookies_attempts": no_cookies_attempts,
                "no_cookies_successes": no_cookies_successes,
                "failed_files": failed_files,
                "archive_path": str(archive_path),
                "archive_source": archive_source,
                "extracted_root": str(extract_root),
            }

    if extract_root.exists():
        shutil.rmtree(extract_root)
    ensure_dir(extract_root)

    with zipfile.ZipFile(archive_path, "r") as bundle:
        bundle.extractall(extract_root)

    selected_contract_files = 0
    for file_path in extract_root.rglob("*"):
        if not file_path.is_file():
            continue
        relative = str(file_path.relative_to(extract_root)).replace("\\", "/")
        if _is_contract_file(relative):
            selected_contract_files += 1

    return {
        "listed_files": 1,
        "selected_contract_files": selected_contract_files,
        "downloaded_files": 0 if reused_existing_archive else 1,
        "reused_existing_files": 1 if reused_existing_archive else 0,
        "no_cookies_attempts": no_cookies_attempts,
        "no_cookies_successes": no_cookies_successes,
        "failed_files": failed_files,
        "archive_path": str(archive_path),
        "archive_source": archive_source,
        "extracted_root": str(extract_root),
    }


def bootstrap_uvg_from_drive(
    *,
    repo_root: Path,
    drive_url: str = DEFAULT_UVG_DRIVE_URL,
    target_root: Path,
    temp_root: Path,
    overwrite: bool = False,
    dry_run: bool = False,
    skip_validation: bool = False,
    keep_temp: bool = False,
    min_intra_raw_sequences: int = 1,
    python_executable: str | None = None,
    clear_existing_download_cache: bool = False,
    max_retries_per_mode: int = 2,
    retry_backoff_seconds: float = 1.5,
    archive_timeout_seconds: int = 1800,
    auto_bootstrap_legacy_contract: bool = True,
) -> dict[str, object]:
    """Download UVG contract data from Google Drive and sync to thesis/uvg."""
    repo_root = repo_root.resolve()
    target_root = target_root.resolve()
    temp_root = temp_root.resolve()
    download_root = temp_root / "google_drive_download"
    python_exec = python_executable or sys.executable

    downloaded = False
    download_exit_code: int | None = None
    download_retry_no_cookies = False
    download_errors: list[str] = []
    best_effort_report: dict[str, object] | None = None
    source_root: Path | None = None
    resolved_mode = "zip"

    if not dry_run:
        if not _is_gdown_installed(python_exec):
            raise RuntimeError(
                "Missing dependency 'gdown'. Install with: "
                "pip install -r requirements-lock.txt"
            )
        if download_root.exists():
            if clear_existing_download_cache:
                shutil.rmtree(download_root)
                ensure_dir(download_root)
            else:
                print(
                    "[info] Reusing staged downloads for resume support. "
                    "Use --fresh-download to clear cache."
                )
        else:
            ensure_dir(download_root)
        print("[info] Download mode: zip archive (single-file download + extract)")
        best_effort_report = _download_zip_archive_contract_files(
            drive_url=drive_url,
            download_root=download_root,
            python_executable=python_exec,
            max_retries_per_mode=max_retries_per_mode,
            archive_timeout_seconds=archive_timeout_seconds,
            retry_backoff_seconds=retry_backoff_seconds,
        )

        if best_effort_report["selected_contract_files"] == 0:
            raise RuntimeError(
                "Zip download succeeded but no contract files were found after extract. "
                "Expected partition/intra_raw_blocks/labels/qps inside the archive."
            )
        failed_count = len(best_effort_report["failed_files"])
        download_exit_code = 0 if failed_count == 0 else 1
        download_retry_no_cookies = best_effort_report["no_cookies_attempts"] > 0
        if failed_count > 0:
            message = (
                "Contract download finished with errors. "
                "Some contract files could not be fetched from Google Drive."
            )
            download_errors.append(message)
            failed_preview = [item["path"] for item in best_effort_report["failed_files"][:5]]
            if failed_preview:
                download_errors.append("Failed files preview: " + ", ".join(failed_preview))
        if not _has_any_downloaded_file(download_root):
            failed_details = ""
            if best_effort_report and best_effort_report.get("failed_files"):
                failed_preview = best_effort_report["failed_files"][:3]
                joined = "; ".join(
                    f"{item['path']}: {item['error']}" for item in failed_preview
                )
                failed_details = f" First errors: {joined}"
            raise RuntimeError(
                "No contract file was downloaded from Google Drive. "
                "Check folder permissions/network access and retry. "
                "Tip: rerun without --fresh-download to reuse partial cache."
                f"{failed_details}"
            )

        downloaded = download_exit_code == 0
        extracted_root = Path(str(best_effort_report.get("extracted_root") or download_root))
        source_root = _select_uvg_source_root(extracted_root)
    elif download_root.exists():
        source_root = _select_uvg_source_root(download_root)

    ensure_dir(target_root)
    moved_files = 0
    moved_bytes = 0
    skipped_existing_files = 0
    skipped_preview: list[str] = []
    if source_root is not None:
        moved_files, moved_bytes, skipped_existing_files, skipped_preview = _sync_tree(
            source_root=source_root,
            target_root=target_root,
            overwrite=overwrite,
            dry_run=dry_run,
        )

    legacy_bootstrap_report: dict[str, object] = {
        "enabled": auto_bootstrap_legacy_contract,
        "executed": False,
        "sequences": [],
        "stats_count": 0,
    }
    if auto_bootstrap_legacy_contract and not dry_run and target_root.exists():
        sequences = discover_sequences(target_root)
        missing_legacy_contract = False
        labels_root = target_root / "labels"
        qps_root = target_root / "qps"
        for sequence in sequences:
            for block_size in (8, 16, 32, 64):
                labels_path = labels_root / f"{sequence}_labels_{block_size}_intra.txt"
                qps_path = qps_root / f"{sequence}_qps_{block_size}_intra.txt"
                if not labels_path.exists() or not qps_path.exists():
                    missing_legacy_contract = True
                    break
            if missing_legacy_contract:
                break

        if sequences and missing_legacy_contract:
            stats = bootstrap_legacy_labels_qps_from_partition(
                uvg_root=target_root,
                contract_root=target_root,
                sequences=sequences,
            )
            legacy_bootstrap_report = {
                "enabled": True,
                "executed": True,
                "sequences": sequences,
                "stats_count": len(stats),
            }

    validation: dict[str, object] = {}
    validated_sequences: list[str] = []
    has_local_sequences = target_root.exists() and len(discover_sequences(target_root)) > 0

    if not skip_validation:
        if dry_run and source_root is None and not has_local_sequences:
            validation["status"] = "skipped (dry-run without local downloaded content)"
        else:
            raw_validated = validate_raw_input_contract(uvg_root=target_root)
            validated_sequences = discover_sequences(target_root)
            validation["raw_input_contract_files"] = [str(path) for path in raw_validated]

            intra_root = target_root / "intra_raw_blocks"
            labels_root = target_root / "labels"
            qps_root = target_root / "qps"

            if intra_root.exists():
                validation["intra_raw_blocks_sequences"] = validate_intra_raw_blocks_contract(
                    intra_raw_blocks_root=intra_root,
                    min_sequences=min_intra_raw_sequences,
                )

            if intra_root.exists() and labels_root.exists() and qps_root.exists():
                flow = validate_expected_raw_flow(
                    raw_root=target_root,
                    legacy_base_path=target_root,
                )
                validation["flow_validation"] = {
                    "ok": flow["ok"],
                    "active_sequences": flow["active_sequences"],
                    "warnings": flow["warnings"],
                    "errors": flow["errors"],
                }
                if not flow["ok"]:
                    joined = "\n".join(flow["errors"])
                    raise RuntimeError(
                        "Flow validation failed after Drive bootstrap:\n"
                        f"{joined}"
                    )

    temp_cleaned = False
    if not keep_temp and not dry_run and temp_root.exists():
        shutil.rmtree(temp_root)
        temp_cleaned = True

    report: dict[str, object] = {
        "drive_url": drive_url,
        "target_root": str(target_root),
        "temp_root": str(temp_root),
        "download_root": str(download_root),
        "source_root": str(source_root) if source_root else None,
        "python_executable": python_exec,
        "downloaded": downloaded,
        "download_exit_code": download_exit_code,
        "download_retry_no_cookies": download_retry_no_cookies,
        "download_errors": download_errors,
        "download_mode": "zip",
        "resolved_download_mode": resolved_mode,
        "best_effort_fallback": best_effort_report,
        "dry_run": dry_run,
        "overwrite": overwrite,
        "moved_files": moved_files,
        "moved_bytes": moved_bytes,
        "skipped_existing_files": skipped_existing_files,
        "skipped_existing_preview": skipped_preview,
        "validated_sequences": validated_sequences,
        "skip_validation": skip_validation,
        "validation": validation,
        "keep_temp": keep_temp,
        "temp_cleaned": temp_cleaned,
        "clear_existing_download_cache": clear_existing_download_cache,
        "max_retries_per_mode": max_retries_per_mode,
        "retry_backoff_seconds": retry_backoff_seconds,
        "archive_timeout_seconds": archive_timeout_seconds,
        "auto_bootstrap_legacy_contract": auto_bootstrap_legacy_contract,
        "legacy_bootstrap": legacy_bootstrap_report,
    }
    return report
