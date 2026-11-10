from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

from class_names import CLASS_NAMES
from config import BATCH_SIZE, DATASET_DIR, OUTPUT_DIR
from data_utils import (
    collect_image_paths,
    create_dataset,
    find_training_root,
    split_paths,
)
from inference_utils import load_trained_model


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the trained CNN on the validation split."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DATASET_DIR,
        help="Folder containing the extracted GTSRB dataset.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"Batch size. Default: {BATCH_SIZE}",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    training_root = find_training_root(args.dataset)
    image_paths, labels = collect_image_paths(training_root)

    _, validation_paths, _, validation_labels = split_paths(image_paths, labels)

    validation_dataset = create_dataset(
        validation_paths,
        validation_labels,
        batch_size=args.batch_size,
        training=False,
    )

    model = load_trained_model()
    probabilities = model.predict(validation_dataset, verbose=1)
    predicted_labels = np.argmax(probabilities, axis=1)

    report = classification_report(
        validation_labels,
        predicted_labels,
        labels=list(range(43)),
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    )

    report_path = OUTPUT_DIR / "classification_report.txt"
    report_path.write_text(report, encoding="utf-8")

    matrix = confusion_matrix(
        validation_labels,
        predicted_labels,
        labels=list(range(43)),
    )

    figure, axis = plt.subplots(figsize=(20, 20))
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=[str(index) for index in range(43)],
    )
    display.plot(
        ax=axis,
        cmap="Blues",
        colorbar=False,
        values_format="d",
    )
    axis.set_title("GTSRB Validation Confusion Matrix")
    figure.tight_layout()

    confusion_matrix_path = OUTPUT_DIR / "confusion_matrix.png"
    figure.savefig(confusion_matrix_path, dpi=170, bbox_inches="tight")
    plt.close(figure)

    print("\nClassification report")
    print("=====================")
    print(report)
    print(f"Saved report: {report_path}")
    print(f"Saved confusion matrix: {confusion_matrix_path}")


if __name__ == "__main__":
    main()
