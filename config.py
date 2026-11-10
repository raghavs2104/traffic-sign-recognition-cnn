from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent

# Put the extracted GTSRB dataset inside this folder.
DATASET_DIR = PROJECT_DIR / "dataset"

MODEL_DIR = PROJECT_DIR / "models"
OUTPUT_DIR = PROJECT_DIR / "outputs"

MODEL_PATH = MODEL_DIR / "traffic_sign_cnn.keras"
CLASS_NAMES_PATH = MODEL_DIR / "class_names.json"

IMAGE_HEIGHT = 48
IMAGE_WIDTH = 48
IMAGE_CHANNELS = 3
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)

NUM_CLASSES = 43
BATCH_SIZE = 64
EPOCHS = 25
VALIDATION_SIZE = 0.20
RANDOM_SEED = 42

SUPPORTED_IMAGE_EXTENSIONS = {
    ".ppm", ".png", ".jpg", ".jpeg", ".bmp", ".webp"
}
