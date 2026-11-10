from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from tensorflow import keras

from class_names import CLASS_NAMES
from config import CLASS_NAMES_PATH, IMAGE_HEIGHT, IMAGE_WIDTH, MODEL_PATH


def load_trained_model(model_path: Path = MODEL_PATH) -> keras.Model:
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            "Run train.py before running prediction."
        )
    return keras.models.load_model(model_path)


def load_class_names(path: Path = CLASS_NAMES_PATH) -> list[str]:
    if path.exists():
        with path.open("r", encoding="utf-8") as file:
            names = json.load(file)
        if len(names) != 43:
            raise ValueError(f"Expected 43 class names, found {len(names)}")
        return names

    return CLASS_NAMES


def preprocess_bgr_image(image_bgr: np.ndarray) -> np.ndarray:
    if image_bgr is None:
        raise ValueError("Received an empty image.")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_rgb = cv2.resize(
        image_rgb,
        (IMAGE_WIDTH, IMAGE_HEIGHT),
        interpolation=cv2.INTER_AREA,
    )
    image_rgb = image_rgb.astype(np.float32) / 255.0
    return np.expand_dims(image_rgb, axis=0)


def predict_top_k(
    model: keras.Model,
    image_bgr: np.ndarray,
    class_names: list[str],
    top_k: int = 3,
) -> list[tuple[int, str, float]]:
    batch = preprocess_bgr_image(image_bgr)
    probabilities = model.predict(batch, verbose=0)[0]

    top_k = max(1, min(top_k, len(class_names)))
    top_indices = np.argsort(probabilities)[-top_k:][::-1]

    return [
        (int(index), class_names[int(index)], float(probabilities[int(index)]))
        for index in top_indices
    ]
