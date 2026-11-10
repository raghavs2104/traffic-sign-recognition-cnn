from __future__ import annotations

import argparse
from pathlib import Path
from tkinter import Tk, filedialog

import cv2

from inference_utils import (
    load_class_names,
    load_trained_model,
    predict_top_k,
)


def choose_image_with_dialog() -> Path | None:
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    selected = filedialog.askopenfilename(
        title="Select a traffic-sign image",
        filetypes=[
            ("Image files", "*.png *.jpg *.jpeg *.ppm *.bmp *.webp"),
            ("All files", "*.*"),
        ],
    )
    root.destroy()

    return Path(selected) if selected else None


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify one cropped traffic-sign image."
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Image path. If omitted, a file-selection dialog opens.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of predictions to show. Default: 3",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    image_path = args.image or choose_image_with_dialog()

    if image_path is None:
        print("No image selected.")
        return
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    model = load_trained_model()
    class_names = load_class_names()
    predictions = predict_top_k(
        model=model,
        image_bgr=image,
        class_names=class_names,
        top_k=args.top_k,
    )

    print(f"\nImage: {image_path}")
    print("Top predictions:")
    for rank, (class_id, class_name, confidence) in enumerate(
        predictions,
        start=1,
    ):
        print(
            f"{rank}. Class {class_id:02d} - {class_name}: "
            f"{confidence * 100:.2f}%"
        )

    best_id, best_name, best_confidence = predictions[0]
    display = image.copy()
    text = f"{best_id}: {best_name} ({best_confidence * 100:.1f}%)"

    # Wrap long label text across two lines.
    cv2.rectangle(display, (0, 0), (display.shape[1], 70), (0, 0, 0), -1)
    cv2.putText(
        display,
        text[:55],
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    if len(text) > 55:
        cv2.putText(
            display,
            text[55:110],
            (10, 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    cv2.imshow("Traffic Sign Prediction - press any key to close", display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
