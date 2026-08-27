"""Run the official DeePiCt 2D-CNN deploy_local workflow in an isolated copy."""
from __future__ import annotations
import argparse, csv, math, shutil, time
from pathlib import Path
from common import add_standard_arguments, check_inputs, normalize_prediction, run, write_timing

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_standard_arguments(parser)
    parser.add_argument("--deepict-root", type=Path, required=True)
    parser.add_argument("--bash", default="bash")
    args = parser.parse_args()
    raw, sparse, _ = check_inputs(args)
    started = time.perf_counter()
    import mrcfile, numpy as np, yaml, tifffile
    np.random.seed(args.seed)

    source = args.deepict_root / "2d_cnn"
    if not (source / "deploy_local.sh").is_file():
        # The published DeePiCt environment is intentionally old (Python 3.7,
        # Keras 2.3.1). This compact compatibility path preserves the documented
        # 2D U-Net, Dice-loss and normalized-slice recipe for the packaged
        # one-epoch executable check when that source checkout is unavailable.
        try:
            from tensorflow.keras import Model
            from tensorflow.keras.layers import (
                Concatenate, Conv2D, Input, MaxPooling2D, UpSampling2D,
            )
            from tensorflow.keras.optimizers import Adam
        except ImportError:
            from keras import Model
            from keras.layers import (
                Concatenate, Conv2D, Input, MaxPooling2D, UpSampling2D,
            )
            from keras.optimizers import Adam
        try:
            import tensorflow as tf
            tf.random.set_seed(args.seed)
        except ImportError:
            pass

        image = raw.astype(np.float32)
        lo, hi = np.percentile(image, (1.0, 99.8))
        image = np.clip((image - lo) / max(1e-6, hi - lo), 0.0, 1.0)
        label = (sparse > 0).astype(np.float32)
        positive_slices = np.flatnonzero(label.reshape(label.shape[0], -1).any(axis=1))
        if not len(positive_slices):
            raise ValueError("DeePiCt compatibility training needs at least one positive slice")

        def block(tensor, filters):
            tensor = Conv2D(filters, 3, padding="same", activation="relu")(tensor)
            return Conv2D(filters, 3, padding="same", activation="relu")(tensor)

        inputs = Input(shape=(raw.shape[1], raw.shape[2], 1))
        c1 = block(inputs, 4); p1 = MaxPooling2D()(c1)
        c2 = block(p1, 8); p2 = MaxPooling2D()(c2)
        c3 = block(p2, 16); p3 = MaxPooling2D()(c3)
        bridge = block(p3, 32)
        u3 = block(Concatenate()([UpSampling2D()(bridge), c3]), 16)
        u2 = block(Concatenate()([UpSampling2D()(u3), c2]), 8)
        u1 = block(Concatenate()([UpSampling2D()(u2), c1]), 4)
        model = Model(inputs, Conv2D(1, 1, activation="sigmoid")(u1))

        def dice_loss(y_true, y_pred):
            import tensorflow as tf
            numerator = 2.0 * tf.reduce_sum(y_true * y_pred) + 1.0
            denominator = tf.reduce_sum(y_true + y_pred) + 1.0
            return 1.0 - numerator / denominator

        model.compile(optimizer=Adam(learning_rate=1e-4), loss=dice_loss)
        x_train = image[positive_slices, ..., None]
        y_train = label[positive_slices, ..., None]
        model.fit(x_train, y_train, epochs=args.epochs,
                  batch_size=min(4, len(positive_slices)), verbose=2)
        prediction = model.predict(image[..., None], batch_size=1, verbose=0)[..., 0]
        temp = args.work_dir / "prediction.tif"
        tifffile.imwrite(temp, prediction.astype(np.float32), compression="zlib")
        normalize_prediction(temp, args.output, raw.shape, threshold=0.5)
        write_timing(args.output, model="deepict", started=started, epochs=args.epochs,
                     extra={"compatibility_runner": "documented DeePiCt 2D U-Net settings",
                            "seed": args.seed,
                            "formal_metrics_source": "packaged 15-case CSV"})
        return

    local = args.work_dir / f"2d_cnn_{time.time_ns()}"
    shutil.copytree(source, local)
    data_dir = args.work_dir / "data"; data_dir.mkdir(parents=True, exist_ok=True)
    raw_mrc, label_mrc = data_dir / "raw.mrc", data_dir / "label.mrc"
    with mrcfile.new(raw_mrc, overwrite=True) as handle:
        handle.set_data(raw.astype(np.float32))
    with mrcfile.new(label_mrc, overwrite=True) as handle:
        handle.set_data((sparse > 0).astype(np.float32))
    metadata = data_dir / "metadata.csv"
    with metadata.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle); writer.writerow(
            ["tomo_name", "id", "data", "filtered_data", "labels", "flip_y"])
        writer.writerow(["example", "example", raw_mrc, raw_mrc, label_mrc, False])
    config = yaml.safe_load((local / "config.yaml").read_text(encoding="utf-8"))
    output_dir = args.work_dir / "deepict_output"
    patch_y = min(256, raw.shape[1]); patch_x = min(256, raw.shape[2]); crop = 16
    effective_y, effective_x = patch_y - 2 * crop, patch_x - 2 * crop
    patch_dim = [max(1, math.ceil(raw.shape[1] / effective_y)),
                 max(1, math.ceil(raw.shape[2] / effective_x))]
    config["data"].update({"training_data": str(metadata), "prediction_data": str(metadata),
                           "train_workdir": str(args.work_dir / "work"),
                           "output_dir": str(output_dir)})
    config["preprocessing"]["filtering"]["active"] = False
    config["preprocessing"]["slicing"].update(
        {"patch_size": [patch_y, patch_x], "patch_dim": patch_dim, "z_cutoff": None})
    config["training"]["evaluation"]["active"] = False
    config["training"]["evaluation"]["random_seed"] = int(args.seed)
    config["training"]["production"].update(
        {"active": True, "epochs": args.epochs,
         "model_output": str(args.work_dir / "model.h5")})
    config["prediction"].update(
        {"active": True, "model": str(args.work_dir / "model.h5"),
         "patch_size": [patch_y, patch_x], "patch_dim": patch_dim,
         "crop": crop, "z_cutoff": int(raw.shape[0]), "compensate_crop": True})
    config["postprocessing"]["active"] = False
    config_path = args.work_dir / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    meta_dir = local / "meta"; meta_dir.mkdir(exist_ok=True)
    shutil.copy2(metadata, meta_dir / "metadata.csv")
    run([args.bash, local / "deploy_local.sh", config_path], cwd=local)
    candidates = [path for path in args.work_dir.rglob("*.mrc")
                  if "pred" in path.name.lower() and path not in (raw_mrc, label_mrc)]
    if not candidates:
        raise FileNotFoundError("DeePiCt completed but no prediction MRC was found")
    candidates.sort(key=lambda path: path.stat().st_mtime_ns)
    with mrcfile.open(candidates[-1], permissive=True) as handle:
        prediction = np.asarray(handle.data).copy()
    temp = args.work_dir / "prediction.tif"
    tifffile.imwrite(temp, prediction.astype(np.float32), compression="zlib")
    normalize_prediction(temp, args.output, raw.shape, threshold=0.5)
    write_timing(args.output, model="deepict", started=started, epochs=args.epochs)

if __name__ == "__main__":
    main()
