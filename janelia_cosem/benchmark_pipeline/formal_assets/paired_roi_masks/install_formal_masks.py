#!/usr/bin/env python3
"""Install the three archived masks under the 15 filenames expected by runners."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import tifffile


HERE = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def logical_uint8_sha256(path: Path) -> str:
    """Hash mask content independently of TIFF compression or metadata."""
    binary = np.squeeze(np.asarray(tifffile.imread(path))) > 0
    return hashlib.sha256(binary.astype(np.uint8).tobytes(order="C")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    manifest = json.loads((HERE / "fixed_paired_roi_masks.json").read_text(encoding="utf-8"))
    args.data_root.mkdir(parents=True, exist_ok=True)
    installed = 0
    for row in manifest["rows"]:
        source = HERE / row["mask_asset"]
        mask_record = manifest["masks"][str(row["roi_num"])]
        expected_archive = mask_record["compressed_tiff_sha256"]
        expected_logical = mask_record["logical_uint8_sha256"]
        if sha256(source) != expected_archive:
            raise RuntimeError(f"archived mask hash mismatch: {source}")
        if logical_uint8_sha256(source) != expected_logical:
            raise RuntimeError(f"archived mask content mismatch: {source}")
        target = args.data_root / row["installed_sparse_filename"]
        if target.exists() and not args.overwrite:
            if logical_uint8_sha256(target) != expected_logical:
                raise FileExistsError(f"different file already exists: {target}")
            continue
        shutil.copy2(source, target)
        installed += 1
    print(f"installed {installed} masks in {args.data_root}")


if __name__ == "__main__":
    main()
