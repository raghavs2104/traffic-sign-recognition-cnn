from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras

from class_names import CLASS_NAMES
from config import (
    BATCH_SIZE,
    CLASS_NAMES_PATH,
    DATASET_DIR,
    EPOCHS,
    MODEL_DIR,
    MODEL_PATH,
    OUTPUT_DIR,
    RANDOM_SEED,
)
from data_utils import (
    collect_image_paths,
    create_dataset,
    find_training_root,
    print_class_distribution,
    split_paths,
)
from model_utils import build_cnn_model


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a CNN on the GTSRB traffic-sign dataset."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DATASET_DIR,
        help="Folder containing the extracted GTSRB dataset.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
        help=f"Maximum training epochs. Default: {EPOCHS}",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"Batch size. Default: {BATCH_SIZE}",
    )
    return parser.parse_args()


def set_random_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def configure_gpu_memory_growth() -> None:
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as error:
            print(f"Could not enable GPU memory growth: {error}")

    if gpus:
        print(f"TensorFlow GPU devices: {gpus}")
    else:
        print("No TensorFlow GPU detected. Training will use the CPU.")


def save_history_csv(history: keras.callbacks.History, output_path: Path) -> None:
    keys = list(history.history.keys())
    rows = zip(*(history.history[key] for key in keys))

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["epoch", *keys])
        for epoch_index, row in enumerate(rows, start=1):
            writer.writerow([epoch_index, *row])


def save_training_plot(history: keras.callbacks.History, output_path: Path) -> None:
    history_values = history.history

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(history_values["accuracy"], label="Training accuracy")
    axes[0].plot(history_values["val_accuracy"], label="Validation accuracy")
    axes[0].set_title("Model Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(history_values["loss"], label="Training loss")
    axes[1].plot(history_values["val_loss"], label="Validation loss")
    axes[1].set_title("Model Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_arguments()

    if args.epochs <= 0:
        raise ValueError("--epochs must be greater than zero.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be greater than zero.")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    set_random_seeds(RANDOM_SEED)
    configure_gpu_memory_growth()

    training_root = find_training_root(args.dataset)
    print(f"\nUsing training data from:\n{training_root}")

    image_paths, labels = collect_image_paths(training_root)
    print(f"Total images found: {len(image_paths)}")

    (
        train_paths,
        validation_paths,
        train_labels,
        validation_labels,
    ) = split_paths(image_paths, labels)

    print(f"Training images:   {len(train_paths)}")
    print(f"Validation images: {len(validation_paths)}")

    print_class_distribution(train_labels, "Training class distribution")

    train_dataset = create_dataset(
        train_paths,
        train_labels,
        batch_size=args.batch_size,
        training=True,
    )
    validation_dataset = create_dataset(
        validation_paths,
        validation_labels,
        batch_size=args.batch_size,
        training=False,
    )

    class_ids = np.arange(43)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=class_ids,
        y=np.asarray(train_labels),
    )
    class_weight = {
        int(class_id): float(weight)
        for class_id, weight in zip(class_ids, weights)
    }

    model = build_cnn_model()
    model.summary()

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=MODEL_PATH,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=6,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.35,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
        keras.callbacks.CSVLogger(
            filename=OUTPUT_DIR / "training_log.csv",
            append=False,
        ),
    ]

    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=args.epochs,
        class_weight=class_weight,
        callbacks=callbacks,
    )

    validation_loss, validation_accuracy = model.evaluate(
        validation_dataset,
        verbose=1,
    )

    # Save the restored best model again, in case EarlyStopping restored weights
    # from a better epoch than the final callback state.
    model.save(MODEL_PATH)

    with CLASS_NAMES_PATH.open("w", encoding="utf-8") as file:
        json.dump(CLASS_NAMES, file, indent=2, ensure_ascii=False)

    save_history_csv(history, OUTPUT_DIR / "history.csv")
    save_training_plot(history, OUTPUT_DIR / "training_curves.png")

    print("\nTraining complete.")
    print(f"Validation loss:     {validation_loss:.4f}")
    print(f"Validation accuracy: {validation_accuracy * 100:.2f}%")
    print(f"Saved model:         {MODEL_PATH}")
    print(f"Training graph:      {OUTPUT_DIR / 'training_curves.png'}")


if __name__ == "__main__":
    main()
