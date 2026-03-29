"""
scripts/train.py
-----------------
Main training entry point. Reads a YAML config and trains the baseline model.

Usage:
    python scripts/train.py --config configs/isic_baseline.yaml
    python scripts/train.py --config configs/brats_baseline.yaml
"""

import os
import sys
import argparse
import yaml
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from data.isic_loader import ISICDataset
from data.brats_loader import BraTSDataset
from models.baseline import (
    build_efficientnetb0,
    build_unet_full,
)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def build_model(config: dict) -> tf.keras.Model:
    task = config.get("task", "classification")
    model_cfg = config.get("model", {})
    data_cfg = config["data"]

    if task == "classification":
        return build_efficientnetb0(
            num_classes=model_cfg.get("num_classes", 1),
            dropout=model_cfg.get("dropout", 0.3),
            input_shape=(data_cfg["image_size"], data_cfg["image_size"], 3),
        )
    else:
        n_channels = config["data"]["n_slices"] * len(config["data"]["modalities"])
        return build_unet_full(
            num_classes=model_cfg.get("num_classes", 4),
            n_channels=n_channels,
            input_size=data_cfg["patch_size"],
        )


def get_callbacks(config: dict) -> list:
    out_dir = config["output"]["dir"]
    os.makedirs(out_dir, exist_ok=True)
    task = config.get("task", "classification")
    monitor = "val_auc" if task == "classification" else "val_dice"

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(out_dir, "best_model.keras"),
            monitor=monitor, mode="max",
            save_best_only=True, verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config["training"].get("early_stopping_patience", 7),
            restore_best_weights=True, verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(
            os.path.join(out_dir, "training_log.csv")
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3,
            min_lr=1e-7, verbose=1,
        ),
    ]

    # Optional W&B logging
    if config["output"].get("log_wandb", False):
        try:
            import wandb
            from wandb.keras import WandbCallback
            wandb.init(
                project=config["output"].get("wandb_project", "medcompress"),
                config=config,
            )
            callbacks.append(WandbCallback())
        except ImportError:
            print("[train] wandb not installed, skipping W&B logging.")

    return callbacks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.get("seed", 42))

    print(f"\n{'='*60}")
    print(f"  MedCompress Training")
    print(f"  Task    : {config['task']}")
    print(f"  Dataset : {config['dataset']}")
    print(f"  Model   : {config['model']['name']}")
    print(f"{'='*60}\n")

    # ── Load dataset ────────────────────────────────────────────────── #
    if config["dataset"] == "isic":
        dataset = ISICDataset(config)
        class_weights = dataset.class_weights if config["training"].get("class_weight") == "auto" else None
    else:
        dataset = BraTSDataset(config)
        class_weights = None

    train_ds = dataset.get_train_dataset()
    val_ds = dataset.get_val_dataset()

    # ── Build model ─────────────────────────────────────────────────── #
    model = build_model(config)
    model.summary()

    # ── Train ────────────────────────────────────────────────────────── #
    callbacks = get_callbacks(config)

    fit_kwargs = dict(
        validation_data=val_ds,
        epochs=config["training"]["epochs"],
        callbacks=callbacks,
    )
    if class_weights:
        fit_kwargs["class_weight"] = class_weights

    history = model.fit(train_ds, **fit_kwargs)

    # ── Save final model ─────────────────────────────────────────────── #
    final_path = os.path.join(config["output"]["dir"], "final_model.keras")
    model.save(final_path)
    print(f"\n[train] Final model saved to {final_path}")

    return history


if __name__ == "__main__":
    main()
