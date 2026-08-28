#!/usr/bin/env python3
"""Download, convert, and validate the public inputs for the formal benchmark."""

from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path


PUBLIC_N5_ROOT = "s3://janelia-cosem-datasets/jrc_hela-2/jrc_hela-2.n5"
DATASET_DOI = "https://doi.org/10.25378/janelia.13114211"
EXPECTED_SHAPE = (200, 1500, 796)
PUBLIC_INPUTS = {
    "hela2_em_s3.tif": {
        "array": "em/fibsem-uint16/s3",
        "dtype": "uint16",
        "logical_sha256": "1551fc1532e34aacba6cf7f3cf1b68bb473db1a4cdac74e668ce36e09192716a",
    },
    "hela2_mito_s3.tif": {
        "array": "labels/mito_seg/s3",
        "dtype": "uint8",
        "logical_sha256": "0f3e252e49c7a063227d0ab24d2cc5ab936f6189cedfe1a7b2d50490bf310d44",
    },
}
NEGATIVE_NAME = "negative_hela2_em_s3.tif"
BUNDLED_NEGATIVE = Path(__file__).resolve().parent / "formal_assets" / "negative_mask" / NEGATIVE_NAME
EXPECTED_NEGATIVE_LOGICAL_SHA256 = "822c3b00e2e7d6f0c30e3733361057b7b00646b2c596a89ba9bd7bbf47339446"
EXPECTED_NEGATIVE_FOREGROUND_VOXELS = 35876640



def logical_sha256(array) -> str:
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def read_tiff(path: Path):
    import numpy as np
    import tifffile

    return np.squeeze(np.asarray(tifffile.imread(path)))


def validate_public_tiff(path: Path, spec: dict[str, str]) -> None:
    array = read_tiff(path)
    if tuple(array.shape) != EXPECTED_SHAPE:
        raise ValueError(f"{path}: shape {array.shape} != {EXPECTED_SHAPE}")
    if str(array.dtype) != spec["dtype"]:
        raise ValueError(f"{path}: dtype {array.dtype} != {spec['dtype']}")
    actual = logical_sha256(array)
    if actual != spec["logical_sha256"]:
        raise ValueError(
            f"{path}: logical SHA-256 {actual} != {spec['logical_sha256']}"
        )
    print(f"validated {path.name}: shape={array.shape}, dtype={array.dtype}, logical_sha256={actual}")


def validate_negative(path: Path) -> str:
    import numpy as np

    array = read_tiff(path)
    if tuple(array.shape) != EXPECTED_SHAPE:
        raise ValueError(f"{path}: shape {array.shape} != {EXPECTED_SHAPE}")
    binary = (array > 0).astype(np.uint8)
    digest = logical_sha256(binary)
    print(
        f"validated {path.name}: shape={array.shape}, "
        f"foreground_voxels={int(binary.sum())}, logical_binary_sha256={digest}"
    )
    return digest


def open_public_n5():
    try:
        import zarr
    except ImportError as exc:
        raise RuntimeError(
            "Public download requires zarr<3, dask[array], and s3fs. "
            "Install benchmark_pipeline/requirements.txt."
        ) from exc
    if not hasattr(zarr, "N5FSStore"):
        raise RuntimeError("zarr.N5FSStore is unavailable; install zarr>=2.16,<3")
    return zarr.open(zarr.N5FSStore(PUBLIC_N5_ROOT, anon=True), mode="r")


def download_public_tiff(root, path: Path, spec: dict[str, str]) -> None:
    import dask.array as da
    import numpy as np
    import tifffile

    source = root[spec["array"]]
    array = da.from_array(source, chunks=source.chunks).compute()
    array = np.asarray(array).transpose(1, 2, 0).astype(spec["dtype"], copy=False)
    if tuple(array.shape) != EXPECTED_SHAPE:
        raise ValueError(
            f"public array {spec['array']} produced {array.shape}, expected {EXPECTED_SHAPE}"
        )
    actual = logical_sha256(array)
    if actual != spec["logical_sha256"]:
        raise ValueError(
            f"public array {spec['array']} logical SHA-256 {actual} != "
            f"{spec['logical_sha256']}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(path, array, compression="zlib")
    print(f"wrote {path} from {PUBLIC_N5_ROOT}/{spec['array']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--negative-from",
        type=Path,
        help=(
            "Optional StackC.tif generated from the public raw stack with "
            "../preprocessing.ijm; it is copied to the formal filename."
        ),
    )
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--require-negative",
        action="store_true",
        help="Fail unless negative_hela2_em_s3.tif is present and valid.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.data_root.mkdir(parents=True, exist_ok=True)
    remote = None
    for filename, spec in PUBLIC_INPUTS.items():
        target = args.data_root / filename
        if args.validate_only:
            if not target.is_file():
                raise FileNotFoundError(target)
        elif args.overwrite or not target.is_file():
            if remote is None:
                print(f"public dataset: {DATASET_DOI}")
                remote = open_public_n5()
            download_public_tiff(remote, target, spec)
        validate_public_tiff(target, spec)

    negative = args.data_root / NEGATIVE_NAME
    if args.negative_from is not None:
        source = args.negative_from.resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        validate_negative(source)
        if source != negative.resolve():
            if negative.exists() and not args.overwrite:
                raise FileExistsError(f"{negative} exists; pass --overwrite to replace it")
            shutil.copy2(source, negative)
            print(f"copied {source} -> {negative}")
    if not negative.is_file() and args.negative_from is None and not args.validate_only:
        if not BUNDLED_NEGATIVE.is_file():
            raise FileNotFoundError(BUNDLED_NEGATIVE)
        shutil.copy2(BUNDLED_NEGATIVE, negative)
        print(f"copied bundled formal negative mask {BUNDLED_NEGATIVE} -> {negative}")
    if negative.is_file():
        digest = validate_negative(negative)
        if digest != EXPECTED_NEGATIVE_LOGICAL_SHA256:
            raise ValueError(
                f"{negative}: logical binary SHA-256 {digest} != "
                f"{EXPECTED_NEGATIVE_LOGICAL_SHA256}"
            )
        foreground = int((read_tiff(negative) > 0).sum())
        if foreground != EXPECTED_NEGATIVE_FOREGROUND_VOXELS:
            raise ValueError(
                f"{negative}: foreground voxels {foreground} != "
                f"{EXPECTED_NEGATIVE_FOREGROUND_VOXELS}"
            )
    elif args.require_negative or args.validate_only:
        raise FileNotFoundError(
            f"{negative} is required; run this command without --validate-only "
            "to install the bundled formal negative mask."
        )


if __name__ == "__main__":
    main()
