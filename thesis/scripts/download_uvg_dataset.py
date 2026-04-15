#!/usr/bin/env python3
"""
download_uvg_dataset.py
========================
Download the UVG (Ultra Video Group) dataset from a Google Drive ZIP link
and extract it into the project's thesis/uvg/ directory.

Usage
-----
    python -m thesis.scripts.download_uvg_dataset \\
        --url "https://drive.google.com/file/d/<FILE_ID>/view?usp=sharing" \\
        --dest thesis/uvg

Arguments
---------
    --url       Public Google Drive sharing URL for the dataset ZIP.
                Defaults to the value stored in UVG_GDRIVE_URL env var.
    --dest      Destination directory where the ZIP will be extracted.
                Defaults to thesis/uvg (relative to repository root).
    --force     Re-download even if the destination already looks populated.
    --no-extract
                Download the ZIP but do not extract it.
    --zip-name  Override the name used for the local ZIP file (default: uvg_dataset.zip).

Dependencies
------------
    pip install requests tqdm
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Optional

try:
    import requests
except ImportError:  # pragma: no cover
    sys.exit(
        "[ERROR] 'requests' is not installed.\n"
        "Run: pip install requests tqdm"
    )

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Google's large-file virus-scan confirmation endpoint.
_GDRIVE_CONFIRM_URL = "https://drive.usercontent.google.com/download"

# Chunk size used while streaming (8 MiB).
_CHUNK_SIZE = 8 * 1024 * 1024


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_file_id(url: str) -> str:
    """
    Extract the Google Drive file ID from a sharing URL.

    Supported URL formats
    ~~~~~~~~~~~~~~~~~~~~~
    * https://drive.google.com/file/d/<ID>/view?usp=sharing
    * https://drive.google.com/open?id=<ID>
    * https://drive.google.com/uc?id=<ID>
    """
    # Pattern: /d/<ID>/
    m = re.search(r"/d/([a-zA-Z0-9_-]+)", url)
    if m:
        return m.group(1)
    # Pattern: id=<ID>
    m = re.search(r"[?&]id=([a-zA-Z0-9_-]+)", url)
    if m:
        return m.group(1)
    raise ValueError(
        f"Could not extract Google Drive file ID from URL: {url!r}\n"
        "Expected a URL like https://drive.google.com/file/d/<ID>/view"
    )


def _build_download_url(file_id: str) -> str:
    """Build the direct download URL that bypasses the virus-scan interstitial."""
    return f"{_GDRIVE_CONFIRM_URL}?id={file_id}&export=download&confirm=t&uuid=uvg"


def _download_with_progress(
    url: str,
    dest_path: Path,
    session: requests.Session,
) -> None:
    """
    Stream *url* into *dest_path* with an optional tqdm progress bar.

    The file is written to a sibling `.tmp` file first and then renamed
    atomically so interrupted downloads leave no corrupt artefact behind.
    """
    tmp_path = dest_path.with_suffix(".tmp")

    with session.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0)) or None

        print(f"[INFO] Downloading to: {dest_path}")
        if total:
            print(f"[INFO] File size: {total / 1024 / 1024:.1f} MiB")

        progress = None
        if tqdm is not None:
            progress = tqdm(
                total=total,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=dest_path.name,
                dynamic_ncols=True,
            )

        try:
            with tmp_path.open("wb") as fh:
                for chunk in r.iter_content(chunk_size=_CHUNK_SIZE):
                    if chunk:
                        fh.write(chunk)
                        if progress is not None:
                            progress.update(len(chunk))
        finally:
            if progress is not None:
                progress.close()

    tmp_path.rename(dest_path)
    print(f"[OK]   Download complete: {dest_path}")


def _validate_zip(path: Path) -> None:
    """Raise *zipfile.BadZipFile* if *path* is not a valid ZIP archive."""
    if not zipfile.is_zipfile(path):
        path.unlink(missing_ok=True)
        raise zipfile.BadZipFile(
            f"Downloaded file is not a valid ZIP archive: {path}\n"
            "Please verify the Google Drive URL and sharing permissions."
        )
    print("[OK]   ZIP integrity check passed.")


def _extract_zip(zip_path: Path, dest_dir: Path) -> None:
    """Extract *zip_path* into *dest_dir*, printing a summary."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Extracting {zip_path.name} to {dest_dir} ...")

    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.namelist()
        if tqdm is not None:
            for member in tqdm(members, desc="Extracting", unit="file", dynamic_ncols=True):
                zf.extract(member, dest_dir)
        else:
            zf.extractall(dest_dir)

    print(f"[OK]   Extracted {len(members)} file(s).")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def download_uvg_dataset(
    url: str,
    dest: Path,
    *,
    zip_name: str = "uvg_dataset.zip",
    force: bool = False,
    extract: bool = True,
) -> Path:
    """
    Download the UVG dataset ZIP from Google Drive and (optionally) extract it.

    Parameters
    ----------
    url:
        Public Google Drive sharing URL.
    dest:
        Directory where the ZIP will be saved **and** where its contents
        will be extracted (if *extract* is True).
    zip_name:
        Filename for the downloaded ZIP archive inside *dest*.
    force:
        If True, re-download even if *dest* already contains files.
    extract:
        If True (default), extract the ZIP after a successful download.

    Returns
    -------
    Path
        Path to the downloaded ZIP file.
    """
    dest = Path(dest).expanduser().resolve()
    zip_path = dest / zip_name

    # --- Idempotency guard ---------------------------------------------------
    if not force and zip_path.exists():
        print(f"[SKIP] ZIP already exists: {zip_path}")
        print("       Use --force to re-download.")
        if extract:
            _extract_zip(zip_path, dest)
        return zip_path

    dest.mkdir(parents=True, exist_ok=True)

    # --- Build download URL --------------------------------------------------
    file_id = _extract_file_id(url)
    download_url = _build_download_url(file_id)
    print(f"[INFO] Google Drive file ID: {file_id}")
    print(f"[INFO] Download URL: {download_url}")

    # --- Download ------------------------------------------------------------
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (UVG-Dataset-Downloader/1.0)"})

    _download_with_progress(download_url, zip_path, session)

    # --- Validate ------------------------------------------------------------
    _validate_zip(zip_path)

    # --- Extract -------------------------------------------------------------
    if extract:
        _extract_zip(zip_path, dest)

    return zip_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="download_uvg_dataset",
        description="Download the UVG dataset ZIP from Google Drive.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--url",
        default=os.environ.get("UVG_GDRIVE_URL", ""),
        help=(
            "Public Google Drive sharing URL for the dataset ZIP. "
            "Defaults to the UVG_GDRIVE_URL environment variable."
        ),
    )
    parser.add_argument(
        "--dest",
        default=str(Path(__file__).resolve().parents[2] / "uvg"),
        help="Destination directory (default: thesis/uvg).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the destination already contains the ZIP.",
    )
    parser.add_argument(
        "--no-extract",
        dest="extract",
        action="store_false",
        default=True,
        help="Download the ZIP but do not extract it.",
    )
    parser.add_argument(
        "--zip-name",
        default="uvg_dataset.zip",
        help="Filename for the local ZIP file (default: uvg_dataset.zip).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)

    if not args.url:
        print(
            "[ERROR] No URL provided.\n"
            "        Pass --url or set the UVG_GDRIVE_URL environment variable.",
            file=sys.stderr,
        )
        return 1

    try:
        zip_path = download_uvg_dataset(
            url=args.url,
            dest=Path(args.dest),
            zip_name=args.zip_name,
            force=args.force,
            extract=args.extract,
        )
        print(f"[DONE] Dataset available at: {zip_path.parent}")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
