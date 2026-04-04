"""
MedCompress Desktop Application
--------------------------------
Tkinter-based GUI for running medical image inference on any
Mac, Windows, or Linux machine. No GPU required.

Usage:
    python deploy/app.py --model path/to/model.tflite

Or double-click the packaged binary (see deploy instructions).
"""

import argparse
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox

from PIL import Image, ImageTk

from deploy.inference import MedCompressInference


class MedCompressApp:
    """Desktop GUI for medical image analysis."""

    def __init__(self, model_path: str):
        self.engine = MedCompressInference(model_path)

        self.root = tk.Tk()
        self.root.title(f"MedCompress - {self.engine.model_name}")
        self.root.geometry("720x560")
        self.root.configure(bg="#f5f6fa")

        self._build_ui()

    def _build_ui(self) -> None:
        # Header
        header = tk.Frame(self.root, bg="#2c3e50", height=50)
        header.pack(fill="x")
        tk.Label(
            header, text="MedCompress", font=("Helvetica", 16, "bold"),
            fg="white", bg="#2c3e50",
        ).pack(side="left", padx=15, pady=10)
        tk.Label(
            header,
            text=f"Model: {self.engine.model_name} | Task: {self.engine.task}",
            font=("Helvetica", 10), fg="#bdc3c7", bg="#2c3e50",
        ).pack(side="right", padx=15)

        # Image display area
        self.canvas_frame = tk.Frame(self.root, bg="#ecf0f1")
        self.canvas_frame.pack(fill="both", expand=True, padx=15, pady=10)
        self.image_label = tk.Label(
            self.canvas_frame, text="Drop an image or click 'Open Image'",
            font=("Helvetica", 12), fg="#7f8c8d", bg="#ecf0f1",
        )
        self.image_label.pack(expand=True)

        # Controls
        controls = tk.Frame(self.root, bg="#f5f6fa")
        controls.pack(fill="x", padx=15, pady=(0, 5))

        tk.Button(
            controls, text="Open Image", command=self._open_image,
            font=("Helvetica", 11), bg="#3498db", fg="white",
            activebackground="#2980b9", padx=20, pady=5,
        ).pack(side="left")

        tk.Button(
            controls, text="Run Analysis", command=self._run_inference,
            font=("Helvetica", 11), bg="#27ae60", fg="white",
            activebackground="#1e8449", padx=20, pady=5,
        ).pack(side="left", padx=10)

        tk.Button(
            controls, text="Benchmark (50 runs)", command=self._run_benchmark,
            font=("Helvetica", 11), bg="#8e44ad", fg="white",
            activebackground="#6c3483", padx=15, pady=5,
        ).pack(side="left")

        # Results area
        results_frame = tk.Frame(self.root, bg="#f5f6fa")
        results_frame.pack(fill="x", padx=15, pady=(5, 15))
        self.result_text = tk.Text(
            results_frame, height=5, font=("Courier", 10),
            bg="white", relief="flat", wrap="word",
        )
        self.result_text.pack(fill="x")
        self.result_text.insert("1.0", "Ready. Load a model and open an image.")
        self.result_text.config(state="disabled")

        self.current_image_path = None

    def _open_image(self) -> None:
        path = filedialog.askopenfilename(
            title="Select Medical Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return

        self.current_image_path = path
        img = Image.open(path)
        # Scale to fit display
        display_size = (400, 350)
        img.thumbnail(display_size, Image.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        self.image_label.configure(image=photo, text="")
        self.image_label.image = photo

        self._set_result(f"Loaded: {Path(path).name}\nClick 'Run Analysis' to classify.")

    def _run_inference(self) -> None:
        if not self.current_image_path:
            messagebox.showwarning("No Image", "Please open an image first.")
            return

        self._set_result("Running inference...")
        self.root.update()

        result = self.engine.predict(self.current_image_path)
        self._set_result(result.summary())

    def _run_benchmark(self) -> None:
        if not self.current_image_path:
            messagebox.showwarning("No Image", "Please open an image first.")
            return

        self._set_result("Benchmarking (50 runs)...")
        self.root.update()

        stats = self.engine.benchmark(self.current_image_path, runs=50)
        self._set_result(
            f"Benchmark Results (50 runs):\n"
            f"  Median: {stats['median_ms']:.1f} ms\n"
            f"  95th percentile: {stats['p95_ms']:.1f} ms\n"
            f"  Min: {stats['min_ms']:.1f} ms  |  Max: {stats['max_ms']:.1f} ms"
        )

    def _set_result(self, text: str) -> None:
        self.result_text.config(state="normal")
        self.result_text.delete("1.0", "end")
        self.result_text.insert("1.0", text)
        self.result_text.config(state="disabled")

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MedCompress Desktop - Medical Image Analysis")
    parser.add_argument("--model", required=True,
                        help="Path to .tflite or .onnx model file")
    args = parser.parse_args()

    app = MedCompressApp(args.model)
    app.run()


if __name__ == "__main__":
    main()
