from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
from PIL import Image, ImageTk

from inference_utils import (
    load_class_names,
    load_trained_model,
    predict_top_k,
)


class TrafficSignApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Traffic Sign Recognition using CNN")
        self.root.geometry("800x650")
        self.root.minsize(720, 600)

        try:
            self.model = load_trained_model()
            self.class_names = load_class_names()
        except Exception as error:
            messagebox.showerror("Model error", str(error))
            self.root.destroy()
            return

        self.preview_photo: ImageTk.PhotoImage | None = None
        self.selected_path: Path | None = None

        self._build_interface()

    def _build_interface(self) -> None:
        main_frame = ttk.Frame(self.root, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)

        title = ttk.Label(
            main_frame,
            text="Traffic Sign Recognition using CNN",
            font=("Segoe UI", 20, "bold"),
        )
        title.pack(pady=(0, 8))

        note = ttk.Label(
            main_frame,
            text=(
                "Select a cropped traffic-sign image. "
                "This model classifies signs; it does not locate them in a full road scene."
            ),
            font=("Segoe UI", 10),
            wraplength=720,
            justify=tk.CENTER,
        )
        note.pack(pady=(0, 15))

        self.image_label = ttk.Label(
            main_frame,
            text="No image selected",
            anchor=tk.CENTER,
        )
        self.image_label.pack(fill=tk.BOTH, expand=True, pady=10)

        button_frame = ttk.Frame(main_frame)
        button_frame.pack(pady=10)

        select_button = ttk.Button(
            button_frame,
            text="Select Image",
            command=self.select_image,
        )
        select_button.grid(row=0, column=0, padx=8)

        predict_button = ttk.Button(
            button_frame,
            text="Predict",
            command=self.predict_image,
        )
        predict_button.grid(row=0, column=1, padx=8)

        self.result_label = ttk.Label(
            main_frame,
            text="Prediction will appear here.",
            font=("Segoe UI", 13),
            wraplength=720,
            justify=tk.LEFT,
        )
        self.result_label.pack(fill=tk.X, pady=(10, 0))

    def select_image(self) -> None:
        selected = filedialog.askopenfilename(
            title="Select a traffic-sign image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.ppm *.bmp *.webp"),
                ("All files", "*.*"),
            ],
        )
        if not selected:
            return

        self.selected_path = Path(selected)

        try:
            image = Image.open(self.selected_path).convert("RGB")
            image.thumbnail((520, 380))
            self.preview_photo = ImageTk.PhotoImage(image)
            self.image_label.configure(image=self.preview_photo, text="")
            self.result_label.configure(text="Click Predict.")
        except Exception as error:
            messagebox.showerror("Image error", str(error))
            self.selected_path = None

    def predict_image(self) -> None:
        if self.selected_path is None:
            messagebox.showwarning("No image", "Select an image first.")
            return

        image_bgr = cv2.imread(str(self.selected_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            messagebox.showerror(
                "Image error",
                f"Could not read: {self.selected_path}",
            )
            return

        try:
            predictions = predict_top_k(
                model=self.model,
                image_bgr=image_bgr,
                class_names=self.class_names,
                top_k=3,
            )
        except Exception as error:
            messagebox.showerror("Prediction error", str(error))
            return

        lines = ["Top predictions:"]
        for rank, (class_id, class_name, confidence) in enumerate(
            predictions,
            start=1,
        ):
            lines.append(
                f"{rank}. Class {class_id:02d} — {class_name} "
                f"({confidence * 100:.2f}%)"
            )

        self.result_label.configure(text="\n".join(lines))


def main() -> None:
    root = tk.Tk()
    TrafficSignApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
