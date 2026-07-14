#!/usr/bin/env python3
"""
Download and extract the AeroNet dataset.

This script intentionally contains placeholder URLs. Replace DATASET_URL and
SHA256 with the official archival link and checksum before public release.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

DATASET_URL = "REPLACE_WITH_OFFICIAL_DATASET_URL"
SHA256 = "REPLACE_WITH_SHA256_CHECKSUM"


def sha256sum(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response, destination.open("wb") as out:
        total = response.headers.get("Content-Length")
        total_bytes = int(total) if total else None
        downloaded = 0
        while True:
            block = response.read(1024 * 1024)
            if not block:
                break
            out.write(block)
            downloaded += len(block)
            if total_bytes:
                percent = 100.0 * downloaded / total_bytes
                print(f"\rDownloading: {percent:6.2f}%", end="", flush=True)
        print()


def extract_archive(archive: Path, output_dir: Path) -> None:
    with zipfile.ZipFile(archive, "r") as zf:
        zf.extractall(output_dir)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download and extract the official AeroNet release."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data"),
        help="Destination directory (default: data)",
    )
    parser.add_argument(
        "--keep-archive",
        action="store_true",
        help="Keep the downloaded archive after extraction",
    )
    args = parser.parse_args()

    if DATASET_URL.startswith("REPLACE_"):
        print(
            "ERROR: DATASET_URL has not been configured. "
            "Please set the official archival URL in this script.",
            file=sys.stderr,
        )
        return 2

    archive = args.output / "AeroNet.zip"
    print(f"Downloading AeroNet to {archive}")
    download(DATASET_URL, archive)

    if not SHA256.startswith("REPLACE_"):
        actual = sha256sum(archive)
        if actual.lower() != SHA256.lower():
            archive.unlink(missing_ok=True)
            print(
                f"ERROR: checksum mismatch.\nExpected: {SHA256}\nActual:   {actual}",
                file=sys.stderr,
            )
            return 3
        print("Checksum verified.")

    print(f"Extracting to {args.output}")
    extract_archive(archive, args.output)

    if not args.keep_archive:
        archive.unlink(missing_ok=True)

    print("AeroNet is ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
