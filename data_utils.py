from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from config import (
    DATASET_DIR,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    RANDOM_SEED,
    SUPPORTED_IMAGE_EXTENSIONS,
    VALIDATION_SIZE,
)


def find_training_root(dataset_dir: Path = DATASET_DIR) -> Path:
    """
    Locate the numeric class folders in either the common Kaggle layout or
    the official GTSRB extracted layout.
    """
    candidates = [
        dataset_dir / "Train",
        dataset_dir / "train",
        dataset_dir / "GTSRB" / "Final_Training" / "Images",
        dataset_dir / "Final_Training" / "Images",
        dataset_dir,
    ]

    for candidate in candidates:
        if not candidate.exists() or not candidate.is_dir():
            continue

        numeric_directories = [
            child for child in candidate.iterdir()
            if child.is_dir() and child.name.isdigit()
        ]
        if numeric_directories:
            return candidate

    checked = "\n".join(f"  - {candidate}" for candidate in candidates)
    raise FileNotFoundError(
        "Could not find GTSRB training class folders.\n"
        "Expected numeric folders such as 0, 1, ..., 42 or "
        "00000, 00001, ..., 00042.\n"
        f"Checked:\n{checked}\n\n"
        "Extract the dataset inside the project's 'dataset' folder."
    )


def collect_image_paths(training_root: Path) -> tuple[list[str], list[int]]:
    image_paths: list[str] = []
    labels: list[int] = []

    class_directories = sorted(
        (
            path for path in training_root.iterdir()
            if path.is_dir() and path.name.isdigit()
        ),
        key=lambda path: int(path.name),
    )

    if len(class_directories) != 43:
        raise ValueError(
            f"Expected 43 class folders, but found {len(class_directories)} "
            f"inside: {training_root}"
        )

    found_labels = {int(path.name) for path in class_directories}
    expected_labels = set(range(43))
    if found_labels != expected_labels:
        missing = sorted(expected_labels - found_labels)
        extra = sorted(found_labels - expected_labels)
        raise ValueError(
            f"Dataset class labels are incorrect. Missing={missing}, extra={extra}"
        )

    for class_directory in class_directories:
        class_id = int(class_directory.name)

        for file_path in class_directory.rglob("*"):
            if (
                file_path.is_file()
                and file_path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
            ):
                image_paths.append(str(file_path))
                labels.append(class_id)

    if not image_paths:
        raise FileNotFoundError(
            f"No supported images were found inside: {training_root}"
        )

    counts = Counter(labels)
    missing_images = [class_id for class_id in range(43) if counts[class_id] == 0]
    if missing_images:
        raise ValueError(f"No images found for classes: {missing_images}")

    return image_paths, labels


def split_paths(
    image_paths: list[str],
    labels: list[int],
    validation_size: float = VALIDATION_SIZE,
    random_seed: int = RANDOM_SEED,
) -> tuple[list[str], list[str], list[int], list[int]]:
    return train_test_split(
        image_paths,
        labels,
        test_size=validation_size,
        random_state=random_seed,
        stratify=labels,
        shuffle=True,
    )


def _read_image_with_opencv(
    path_tensor,
    label_tensor,
) -> tuple[np.ndarray, np.int32]:

    # TensorFlow may provide the path as bytes, np.bytes_,
    # or a zero-dimensional NumPy array.
    if hasattr(path_tensor, "item"):
        path_value = path_tensor.item()
    else:
        path_value = path_tensor

    if isinstance(path_value, (bytes, bytearray, np.bytes_)):
        image_path = path_value.decode("utf-8")
    else:
        image_path = str(path_value)

    # The label may also be a NumPy scalar or normal integer.
    if hasattr(label_tensor, "item"):
        label_value = label_tensor.item()
    else:
        label_value = label_tensor

    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError(f"OpenCV could not read image: {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = cv2.resize(
        image,
        (IMAGE_WIDTH, IMAGE_HEIGHT),
        interpolation=cv2.INTER_AREA,
    )

    image = image.astype(np.float32) / 255.0

    return image, np.int32(label_value)


def _tensorflow_load_image(
    path: tf.Tensor,
    label: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    image, parsed_label = tf.numpy_function(
        _read_image_with_opencv,
        [path, label],
        [tf.float32, tf.int32],
    )

    image.set_shape((IMAGE_HEIGHT, IMAGE_WIDTH, 3))
    parsed_label.set_shape(())
    return image, parsed_label


def create_dataset(
    image_paths: Iterable[str],
    labels: Iterable[int],
    batch_size: int,
    training: bool,
    random_seed: int = RANDOM_SEED,
) -> tf.data.Dataset:
    paths_list = list(image_paths)
    labels_list = list(labels)

    dataset = tf.data.Dataset.from_tensor_slices((paths_list, labels_list))

    if training:
        dataset = dataset.shuffle(
            buffer_size=min(len(paths_list), 10_000),
            seed=random_seed,
            reshuffle_each_iteration=True,
        )

    dataset = dataset.map(
        _tensorflow_load_image,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=not training,
    )

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


def print_class_distribution(labels: Iterable[int], title: str) -> None:
    counts = Counter(labels)
    print(f"\n{title}")
    print("-" * len(title))
    for class_id in range(43):
        print(f"Class {class_id:02d}: {counts[class_id]:5d} images")
