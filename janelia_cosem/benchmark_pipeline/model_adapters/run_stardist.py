"""Train and predict with the official StarDist3D API on the packaged example."""
from __future__ import annotations
import argparse, time
from pathlib import Path
from common import add_standard_arguments, check_inputs, normalize_prediction, write_timing

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_standard_arguments(parser)
    parser.add_argument("--steps-per-epoch", type=int, default=8)
    args = parser.parse_args()
    raw, sparse, _ = check_inputs(args)
    started = time.perf_counter()
    import numpy as np
    import tensorflow as tf
    from scipy import ndimage
    from csbdeep.utils import normalize
    from stardist import Rays_GoldenSpiral
    from stardist.models import Config3D, StarDist3D

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    image = normalize(raw.astype(np.float32), 1, 99.8, axis=(0, 1, 2))
    instances, _ = ndimage.label(sparse > 0)
    if int(instances.max()) == 0:
        raise ValueError("StarDist requires at least one sparse foreground instance")
    patch = (min(16, raw.shape[0]), min(128, raw.shape[1]), min(128, raw.shape[2]))
    use_gpu = args.device != "cpu" and bool(tf.config.list_physical_devices("GPU"))
    config = Config3D(
        rays=Rays_GoldenSpiral(32), grid=(1, 2, 2), anisotropy=(2.0, 1.0, 1.0),
        train_patch_size=patch, train_batch_size=1, train_epochs=args.epochs,
        train_steps_per_epoch=args.steps_per_epoch, train_learning_rate=3e-4,
        use_gpu=use_gpu,
    )
    model = StarDist3D(config, name="stardist_example", basedir=str(args.work_dir))
    model.train([image], [instances.astype(np.uint16)],
                validation_data=([image], [instances.astype(np.uint16)]), augmenter=None)
    prediction, _ = model.predict_instances(image)
    temp = args.work_dir / "prediction_instances.tif"
    import tifffile
    tifffile.imwrite(temp, prediction.astype(np.uint16), compression="zlib")
    normalize_prediction(temp, args.output, raw.shape)
    write_timing(args.output, model="stardist", started=started, epochs=args.epochs,
                 extra={"execution_device": "gpu" if use_gpu else "cpu",
                        "seed": args.seed})

if __name__ == "__main__":
    main()
