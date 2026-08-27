#!/usr/bin/env python3
"""Build compact, pixel-equivalent TIFF assets for the formal paired ROI masks."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import tifffile


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        required=True,
        help="Directory containing label_hela2_mito_independent_100_{1,5,10}.tif",
    )
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "masks")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, object] = {}
    expected_shape: tuple[int, ...] | None = None
    for roi in (1, 5, 10):
        source = args.source_dir / f"label_hela2_mito_independent_100_{roi}.tif"
        array = np.asarray(tifffile.imread(source), dtype=np.uint8)
        if expected_shape is None:
            expected_shape = array.shape
        if array.shape != expected_shape:
            raise ValueError(f"Shape mismatch for {source}: {array.shape} != {expected_shape}")
        target = args.output_dir / f"paired_roi_{roi}.tif"
        tifffile.imwrite(
            target,
            array,
            compression="zlib",
            compressionargs={"level": 9},
            metadata={"axes": "ZYX"},
        )
        metadata[str(roi)] = {
            "shape": list(array.shape),
            "foreground_voxels": int(np.count_nonzero(array)),
            "logical_uint8_sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
            "compressed_tiff_sha256": file_sha256(target),
            "compressed_bytes": target.stat().st_size,
        }

    assert expected_shape is not None
    validity = np.ones(expected_shape, dtype=np.uint8)
    validity_path = args.output_dir / "cellmap_validity_all_voxels.tif"
    tifffile.imwrite(
        validity_path,
        validity,
        compression="zlib",
        compressionargs={"level": 9},
        metadata={"axes": "ZYX"},
    )
    metadata["cellmap_validity"] = {
        "semantics": (
            "all voxels valid; reproduces the native CellMap baseline treatment "
            "of unlabeled voxels as background"
        ),
        "shape": list(validity.shape),
        "valid_voxels": int(validity.size),
        "compressed_tiff_sha256": file_sha256(validity_path),
        "compressed_bytes": validity_path.stat().st_size,
    }
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
