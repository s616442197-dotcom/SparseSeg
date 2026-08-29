"""Train/predict the pinned official CellMap 2D/3D U-Net architectures."""
from __future__ import annotations
import argparse, json, random, time
from importlib import metadata
from pathlib import Path
from common import add_standard_arguments, check_inputs, write_timing

CELLMAP_DISTRIBUTION = "cellmap-segmentation-challenge"
CELLMAP_REPOSITORY = "https://github.com/janelia-cellmap/cellmap-segmentation-challenge"
CELLMAP_COMMIT = "0300239cd0b4867d4bab008aa9e95161b2442d93"
CELLMAP_INSTALL_SPEC = f"git+{CELLMAP_REPOSITORY}.git@{CELLMAP_COMMIT}"


def require_pinned_cellmap_models():
    """Load CellMap U-Nets only when the installed distribution matches the pinned commit."""
    install_command = f'python -m pip install \"{CELLMAP_INSTALL_SPEC}\"'
    try:
        distribution = metadata.distribution(CELLMAP_DISTRIBUTION)
    except metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            "The official CellMap package is required; the portable fallback has been removed. "
            f"Install the pinned source with: {install_command}"
        ) from exc

    direct_url_text = distribution.read_text("direct_url.json")
    if not direct_url_text:
        raise RuntimeError(
            "Cannot verify the installed CellMap source revision because direct_url.json is "
            f"missing. Reinstall from the pinned Git commit with: {install_command}"
        )
    try:
        direct_url = json.loads(direct_url_text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "The installed CellMap direct_url.json is invalid; reinstall the pinned package "
            f"with: {install_command}"
        ) from exc

    vcs_info = direct_url.get("vcs_info") or {}
    installed_commit = str(vcs_info.get("commit_id") or "").lower()
    installed_url = str(direct_url.get("url") or "").lower().removesuffix(".git")
    expected_url = CELLMAP_REPOSITORY.lower().removesuffix(".git")
    if vcs_info.get("vcs") != "git" or installed_commit != CELLMAP_COMMIT or installed_url != expected_url:
        raise RuntimeError(
            "The installed CellMap package does not match the required official source. "
            f"Expected {CELLMAP_REPOSITORY}@{CELLMAP_COMMIT}, got "
            f"url={direct_url.get('url')!r}, commit={installed_commit or None!r}. "
            f"Reinstall with: {install_command}"
        )

    try:
        from cellmap_segmentation_challenge.models import UNet_2D, UNet_3D
    except ImportError as exc:
        raise RuntimeError(
            "The pinned CellMap distribution is installed but its UNet_2D/UNet_3D models "
            "could not be imported. Recreate the documented COSEM U-Net environment."
        ) from exc
    return UNet_2D, UNet_3D

def crop2(array, z, y, x, size):
    return array[z, y:y+size, x:x+size]

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_standard_arguments(parser)
    parser.add_argument("--dimension", choices=("2d", "3d"), required=True)
    parser.add_argument("--steps-per-epoch", type=int, default=8)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--inference-overlap", type=float, default=0.5)
    args = parser.parse_args()
    if not 0.0 <= args.inference_overlap < 1.0:
        raise ValueError("--inference-overlap must be in [0, 1)")
    raw, sparse, _ = check_inputs(args)
    started = time.perf_counter()
    import numpy as np, torch, tifffile
    UNet_2D, UNet_3D = require_pinned_cellmap_models()
    architecture_source = f"{CELLMAP_DISTRIBUTION}@{CELLMAP_COMMIT}"
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    image = raw.astype(np.float32)
    image = (image - np.percentile(image, 1)) / max(1e-6, np.percentile(image, 99.8)-np.percentile(image, 1))
    image = np.clip(image, 0, 1)
    label = sparse > 0
    model = (UNet_2D if args.dimension == "2d" else UNet_3D)(1, 1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    positives = np.argwhere(label > 0)
    if len(positives) == 0:
        raise ValueError("COSEM-architecture training needs at least one positive voxel")
    xy = min(80, image.shape[1], image.shape[2])
    dz = min(16, image.shape[0])
    batch_size = args.batch_size or (8 if args.dimension == "2d" else 2)
    for _epoch in range(args.epochs):
        model.train()
        for _ in range(args.steps_per_epoch):
            x_patches, y_patches = [], []
            for _batch in range(batch_size):
                z, cy, cx = positives[random.randrange(len(positives))]
                y = int(np.clip(cy - xy//2, 0, image.shape[1]-xy))
                x = int(np.clip(cx - xy//2, 0, image.shape[2]-xy))
                if args.dimension == "2d":
                    x_patches.append(crop2(image, z, y, x, xy))
                    y_patches.append(crop2(label, z, y, x, xy))
                else:
                    z0 = int(np.clip(z-dz//2, 0, image.shape[0]-dz))
                    x_patches.append(image[z0:z0+dz,y:y+xy,x:x+xy])
                    y_patches.append(label[z0:z0+dz,y:y+xy,x:x+xy])
            xb = torch.from_numpy(np.asarray(x_patches, dtype=np.float32)[:,None]).to(device)
            yb = torch.from_numpy(np.asarray(y_patches, dtype=np.float32)[:,None]).to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            if isinstance(logits, (tuple, list)): logits = logits[0]
            if isinstance(logits, dict): logits = next(iter(logits.values()))
            pos = float(yb.sum().item()); neg = float(yb.numel()-pos)
            weight = torch.tensor(min(100.0, neg/max(pos,1.0)), device=device)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, yb, pos_weight=weight)
            loss.backward(); optimizer.step()
    model.eval()
    probabilities = np.zeros(image.shape, dtype=np.float32)
    counts = np.zeros(image.shape, dtype=np.uint16)
    with torch.no_grad():
        if args.dimension == "2d":
            infer_batch = max(1, min(batch_size, 4))
            for start in range(0, image.shape[0], infer_batch):
                stop = min(image.shape[0], start + infer_batch)
                tensor = torch.from_numpy(image[start:stop,None]).to(device)
                output = model(tensor)
                if isinstance(output, (tuple,list)): output=output[0]
                if isinstance(output, dict): output=next(iter(output.values()))
                probabilities[start:stop] = torch.sigmoid(output)[:,0].cpu().numpy()
                counts[start:stop] = 1
        else:
            strides = (
                max(1, int(round(dz * (1.0-args.inference_overlap)))),
                max(1, int(round(xy * (1.0-args.inference_overlap)))),
                max(1, int(round(xy * (1.0-args.inference_overlap)))),
            )
            def starts(length, patch, stride):
                values = list(range(0, max(1, length-patch+1), stride))
                last = max(0, length-patch)
                if not values or values[-1] != last: values.append(last)
                return values
            for z0 in starts(image.shape[0], dz, strides[0]):
                for y0 in starts(image.shape[1], xy, strides[1]):
                    for x0 in starts(image.shape[2], xy, strides[2]):
                        patch = image[z0:z0+dz,y0:y0+xy,x0:x0+xy]
                        tensor = torch.from_numpy(patch[None,None]).to(device)
                        output = model(tensor)
                        if isinstance(output, (tuple,list)): output=output[0]
                        if isinstance(output, dict): output=next(iter(output.values()))
                        prob = torch.sigmoid(output)[0,0].cpu().numpy()
                        probabilities[z0:z0+dz,y0:y0+xy,x0:x0+xy] += prob
                        counts[z0:z0+dz,y0:y0+xy,x0:x0+xy] += 1
    probabilities /= np.maximum(counts, 1)
    tifffile.imwrite(args.output, (probabilities >= 0.5).astype(np.uint8), compression="zlib")
    torch.save(model.state_dict(), args.work_dir / "model.pt")
    write_timing(args.output, model=f"cosem_{args.dimension}_unet", started=started,
                 epochs=args.epochs,
                 extra={"sampler": "direct positive-centered patches",
                        "architecture_source": architecture_source,
                        "cellmap_repository": CELLMAP_REPOSITORY,
                        "cellmap_commit": CELLMAP_COMMIT,
                        "seed": args.seed,
                        "batch_size": batch_size,
                        "steps_per_epoch": args.steps_per_epoch,
                        "inference_overlap": args.inference_overlap,
                        "loss": "weighted BCE; pos_weight=min(100,N_negative/max(N_positive,1))",
                        "label_treatment": "unselected voxels treated as background",
                        "ignore_mask": "none",
                        "formal_metrics_source": "fresh fixed-mask replay"})

if __name__ == "__main__":
    main()
