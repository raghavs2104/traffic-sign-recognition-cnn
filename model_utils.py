from __future__ import annotations

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from config import IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CLASSES


def build_cnn_model() -> keras.Model:
    data_augmentation = keras.Sequential(
        [
            layers.RandomRotation(0.08),
            layers.RandomZoom(0.12),
            layers.RandomTranslation(height_factor=0.08, width_factor=0.08),
            layers.RandomContrast(0.15),
        ],
        name="data_augmentation",
    )

    inputs = keras.Input(
        shape=(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS),
        name="traffic_sign_image",
    )

    x = data_augmentation(inputs)

    x = layers.Conv2D(32, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(64, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(64, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(128, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(128, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Dropout(0.30)(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.40)(x)

    outputs = layers.Dense(
        NUM_CLASSES,
        activation="softmax",
        name="class_probabilities",
    )(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="traffic_sign_cnn")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    return model
