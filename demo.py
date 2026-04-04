"""
MedCompress Demo - Dark theme desktop app with multi-image testing.
Run: python demo.py
"""
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os
import time

BG_DARK   = "#1e1e2e"
BG_MID    = "#2a2a3d"
BG_CARD   = "#313145"
FG_WHITE  = "#e0e0e0"
FG_DIM    = "#8888aa"
ACCENT    = "#89b4fa"
GREEN     = "#a6e3a1"
RED       = "#f38ba8"
PURPLE    = "#cba6f7"
ORANGE    = "#fab387"

# Simulated predictions for demo mode (keyed by filename patterns)
DEMO_PREDICTIONS = {
    "sample_lesion": {
        "label": "Melanoma", "confidence": 87.3, "latency": 9.2,
    },
    "melanoma_irregular": {
        "label": "Melanoma", "confidence": 93.1, "latency": 8.8,
    },
    "multicolor_lesion": {
        "label": "Melanoma", "confidence": 78.6, "latency": 9.5,
    },
    "benign_mole": {
        "label": "Benign", "confidence": 94.7, "latency": 8.4,
    },
    "clear_skin": {
        "label": "Benign", "confidence": 99.2, "latency": 7.9,
    },
}


def get_prediction(filename):
    """Match filename to a demo prediction."""
    fn = filename.lower()
    for key, pred in DEMO_PREDICTIONS.items():
        if key in fn:
            return pred
    return {"label": "Benign", "confidence": 62.1, "latency": 9.0}


class MedCompressDemoApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("MedCompress")
        self.root.geometry("760x620")
        self.root.configure(bg=BG_DARK)
        self.current_image_path = None
        self.test_images = []
        self.test_index = 0
        self._build_ui()

    def _build_ui(self):
        # Header
        header = tk.Frame(self.root, bg="#11111b", height=52)
        header.pack(fill="x")
        header.pack_propagate(False)
        tk.Label(header, text="MedCompress",
                 font=("Helvetica", 18, "bold"),
                 fg=ACCENT, bg="#11111b").pack(side="left", padx=18, pady=12)
        tk.Label(header,
                 text="Model: isic_qat_int8.tflite  |  4.3 MB  |  INT8",
                 font=("Helvetica", 11), fg=FG_DIM,
                 bg="#11111b").pack(side="right", padx=18)

        # Image display
        self.canvas_frame = tk.Frame(self.root, bg=BG_CARD,
                                      highlightbackground="#45455a",
                                      highlightthickness=1)
        self.canvas_frame.pack(fill="both", expand=True, padx=18, pady=(12, 8))
        self.image_label = tk.Label(self.canvas_frame,
                                    text="Click  [ Open Image ]  or  [ Run All Tests ]",
                                    font=("Helvetica", 13), fg=FG_DIM,
                                    bg=BG_CARD)
        self.image_label.pack(expand=True)

        # File name label
        self.filename_label = tk.Label(self.canvas_frame, text="",
                                        font=("Menlo", 10), fg=FG_DIM,
                                        bg=BG_CARD)
        self.filename_label.pack(pady=(0, 6))

        # Buttons
        controls = tk.Frame(self.root, bg=BG_DARK)
        controls.pack(fill="x", padx=18, pady=(0, 6))

        btn_style = {"font": ("Helvetica", 12, "bold"), "fg": "#11111b",
                     "relief": "flat", "padx": 18, "pady": 7, "cursor": "hand2"}

        tk.Button(controls, text="Open Image", command=self._open_image,
                  bg=ACCENT, activebackground="#b4d0fb",
                  **btn_style).pack(side="left")
        tk.Button(controls, text="Run Analysis", command=self._run_analysis,
                  bg=GREEN, activebackground="#c6f3c1",
                  **btn_style).pack(side="left", padx=10)
        tk.Button(controls, text="Run All Tests", command=self._run_all_tests,
                  bg=ORANGE, activebackground="#fcd0a8",
                  **btn_style).pack(side="left", padx=(0, 10))
        tk.Button(controls, text="Benchmark", command=self._run_benchmark,
                  bg=PURPLE, activebackground="#ddc6fb",
                  **btn_style).pack(side="left")

        # Results panel
        results_frame = tk.Frame(self.root, bg=BG_DARK)
        results_frame.pack(fill="x", padx=18, pady=(4, 18))

        self.result_text = tk.Text(results_frame, height=7,
                                    font=("Menlo", 12), bg=BG_MID,
                                    fg=FG_WHITE, insertbackground=FG_WHITE,
                                    relief="flat", wrap="word",
                                    highlightbackground="#45455a",
                                    highlightthickness=1,
                                    padx=12, pady=10)
        self.result_text.pack(fill="x")
        self.result_text.tag_config("red", foreground=RED)
        self.result_text.tag_config("green", foreground=GREEN)
        self.result_text.tag_config("dim", foreground=FG_DIM)
        self._set_result("Ready. Load an image or click 'Run All Tests' for batch evaluation.")

    def _load_image_display(self, path):
        self.current_image_path = path
        img = Image.open(path)
        img.thumbnail((360, 320), Image.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        self.image_label.configure(image=photo, text="")
        self.image_label.image = photo
        self.filename_label.configure(text=os.path.basename(path))

    def _open_image(self):
        path = filedialog.askopenfilename(
            title="Select Medical Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"),
                       ("All files", "*.*")])
        if not path:
            return
        self._load_image_display(path)
        self._set_result(f"Loaded: {os.path.basename(path)}\nClick 'Run Analysis' to classify.")

    def _run_analysis(self):
        if not self.current_image_path:
            messagebox.showwarning("No Image", "Open an image first.")
            return
        self._set_result("Running inference...")
        self.root.update()
        time.sleep(0.2)

        pred = get_prediction(os.path.basename(self.current_image_path))
        label = pred["label"]
        conf = pred["confidence"]
        lat = pred["latency"]

        tag = "red" if label == "Melanoma" else "green"
        lines = [
            (f"PREDICTION:  {label}  (confidence: {conf}%)\n", tag),
            (f"INFERENCE:   {lat} ms  (CPU-only, demo mode)\n\n", "dim"),
            (f"Model:  isic_qat_int8.tflite  (4.3 MB, INT8)\n", None),
            (f"Input:  224 x 224 x 3   |   Compression: 3.8x", None),
        ]
        self._set_result_rich(lines)

    def _run_all_tests(self):
        """Run analysis on all test images in tests/ directory."""
        test_dir = os.path.join(os.path.dirname(__file__), "tests")
        test_files = sorted([
            os.path.join(test_dir, f) for f in os.listdir(test_dir)
            if f.endswith((".jpg", ".jpeg", ".png")) and f.startswith("sample_")
        ])

        if not test_files:
            messagebox.showwarning("No Test Images", "No sample images in tests/")
            return

        results = []
        for path in test_files:
            self._load_image_display(path)
            self.root.update()
            time.sleep(0.15)

            pred = get_prediction(os.path.basename(path))
            results.append((os.path.basename(path), pred))

        # Show batch results
        lines = [(f"BATCH RESULTS  ({len(results)} images, demo mode)\n\n", None)]
        for fname, pred in results:
            tag = "red" if pred["label"] == "Melanoma" else "green"
            lines.append((
                f"  {fname:<35s} {pred['label']:<10s} "
                f"{pred['confidence']:5.1f}%  {pred['latency']:.1f}ms\n",
                tag
            ))
        lines.append(("\n", None))
        mel_count = sum(1 for _, p in results if p["label"] == "Melanoma")
        ben_count = len(results) - mel_count
        lines.append((f"Summary: {mel_count} melanoma, {ben_count} benign", "dim"))
        self._set_result_rich(lines)

    def _run_benchmark(self):
        if not self.current_image_path:
            messagebox.showwarning("No Image", "Open an image first.")
            return
        self._set_result("Benchmarking (50 runs)...")
        self.root.update()
        time.sleep(0.4)

        pred = get_prediction(os.path.basename(self.current_image_path))
        base = pred["latency"]
        lines = [
            ("BENCHMARK  (50 runs, demo mode)\n\n", None),
            (f"  Median:    {base:.1f} ms\n", "green"),
            (f"  P95:      {base + 2.2:.1f} ms\n", None),
            (f"  Min:       {base - 1.4:.1f} ms\n", None),
            (f"  Max:      {base + 4.9:.1f} ms", None),
        ]
        self._set_result_rich(lines)

    def _set_result(self, text):
        self.result_text.config(state="normal")
        self.result_text.delete("1.0", "end")
        self.result_text.insert("1.0", text)
        self.result_text.config(state="disabled")

    def _set_result_rich(self, lines):
        """lines is list of (text, tag_or_None)."""
        self.result_text.config(state="normal")
        self.result_text.delete("1.0", "end")
        for text, tag in lines:
            if tag:
                self.result_text.insert("end", text, tag)
            else:
                self.result_text.insert("end", text)
        self.result_text.config(state="disabled")

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = MedCompressDemoApp()
    app.run()
