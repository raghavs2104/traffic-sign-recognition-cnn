from __future__ import annotations

import argparse
import time

import cv2

from inference_utils import (
    load_class_names,
    load_trained_model,
    predict_top_k,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Classify a traffic sign placed inside a centre webcam box. "
            "This is recognition, not full object detection."
        )
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera index. Default: 0",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.70,
        help="Minimum confidence to show a class. Default: 0.70",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be between 0 and 1.")

    model = load_trained_model()
    class_names = load_class_names()

    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        raise RuntimeError(
            f"Could not open camera index {args.camera}. "
            "Try: python webcam.py --camera 1"
        )

    latest_text = "Place one sign inside the box"
    latest_confidence = 0.0
    last_prediction_time = 0.0
    prediction_interval_seconds = 0.15

    print("Press Q to close the webcam window.")

    try:
        while True:
            success, frame = camera.read()
            if not success:
                print("Could not read a frame from the webcam.")
                break

            frame_height, frame_width = frame.shape[:2]
            box_size = int(min(frame_width, frame_height) * 0.55)
            x1 = (frame_width - box_size) // 2
            y1 = (frame_height - box_size) // 2
            x2 = x1 + box_size
            y2 = y1 + box_size

            current_time = time.monotonic()
            if current_time - last_prediction_time >= prediction_interval_seconds:
                roi = frame[y1:y2, x1:x2]
                predictions = predict_top_k(
                    model=model,
                    image_bgr=roi,
                    class_names=class_names,
                    top_k=1,
                )
                class_id, class_name, confidence = predictions[0]

                latest_confidence = confidence
                if confidence >= args.threshold:
                    latest_text = f"{class_id}: {class_name}"
                else:
                    latest_text = "Uncertain - move the sign closer"

                last_prediction_time = current_time

            box_color = (
                (0, 200, 0)
                if latest_confidence >= args.threshold
                else (0, 180, 255)
            )
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 3)

            cv2.rectangle(
                frame,
                (0, 0),
                (frame_width, 85),
                (0, 0, 0),
                -1,
            )
            cv2.putText(
                frame,
                latest_text[:65],
                (15, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                box_color,
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"Confidence: {latest_confidence * 100:.1f}% | Press Q to quit",
                (15, 68),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("Traffic Sign Recognition - centre box", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
