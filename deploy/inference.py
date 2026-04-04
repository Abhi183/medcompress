"""
MedCompress Inference Engine
----------------------------
Cross-platform inference for compressed medical imaging models.
Supports TFLite and ONNX models on CPU (macOS, Windows, Linux).

Usage:
    from deploy.inference import MedCompressInference

    engine = MedCompressInference("model.tflite")
    result = engine.predict("skin_lesion.jpg")
    print(result)
"""

import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class PredictionResult:
    """Inference result from a medical imaging model."""

    task: str                    # "classification" or "segmentation"
    label: str                   # human-readable prediction
    confidence: float            # 0.0 to 1.0
    inference_time_ms: float     # wall-clock time
    model_name: str
    input_shape: tuple[int, ...]
    raw_output: np.ndarray       # full model output

    def summary(self) -> str:
        if self.task == "classification":
            return (
                f"Prediction: {self.label} "
                f"(confidence: {self.confidence:.1%})\n"
                f"Inference: {self.inference_time_ms:.1f} ms"
            )
        return (
            f"Segmentation complete: {self.label}\n"
            f"Output shape: {self.raw_output.shape}\n"
            f"Inference: {self.inference_time_ms:.1f} ms"
        )


ISIC_LABELS = {0: "Benign", 1: "Melanoma"}
BRATS_LABELS = {0: "Background", 1: "Necrotic Core",
                2: "Peritumoral Edema", 3: "Enhancing Tumor"}


class MedCompressInference:
    """Unified inference engine for TFLite and ONNX medical models."""

    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        self.model_name = self.model_path.stem
        ext = self.model_path.suffix.lower()

        if ext == ".tflite":
            self._backend = "tflite"
            self._load_tflite()
        elif ext == ".onnx":
            self._backend = "onnx"
            self._load_onnx()
        else:
            raise ValueError(f"Unsupported format: {ext} (use .tflite or .onnx)")

        self.task = self._detect_task()

    def _load_tflite(self) -> None:
        try:
            import tflite_runtime.interpreter as tflite
            self._interpreter = tflite.Interpreter(
                model_path=str(self.model_path))
        except ImportError:
            import tensorflow as tf
            self._interpreter = tf.lite.Interpreter(
                model_path=str(self.model_path))

        self._interpreter.allocate_tensors()
        self._input_details = self._interpreter.get_input_details()[0]
        self._output_details = self._interpreter.get_output_details()[0]
        self.input_shape = tuple(self._input_details["shape"])

    def _load_onnx(self) -> None:
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = os.cpu_count() or 4
        self._session = ort.InferenceSession(
            str(self.model_path), opts, providers=["CPUExecutionProvider"])
        inp = self._session.get_inputs()[0]
        self.input_shape = tuple(inp.shape)
        self._input_name = inp.name

    def _detect_task(self) -> str:
        if self._backend == "tflite":
            out_shape = self._output_details["shape"]
        else:
            out_shape = self._session.get_outputs()[0].shape
        # Classification: output is (1, 1) or (1, num_classes)
        # Segmentation: output is (1, H, W, num_classes)
        if len(out_shape) == 4 and out_shape[1] > 1 and out_shape[2] > 1:
            return "segmentation"
        return "classification"

    def _preprocess_classification(self, image_path: str) -> np.ndarray:
        h, w = self.input_shape[1], self.input_shape[2]
        img = Image.open(image_path).convert("RGB").resize((w, h))
        arr = np.array(img, dtype=np.float32) / 127.5 - 1.0  # [-1, 1]
        return np.expand_dims(arr, axis=0)

    def _preprocess_segmentation(self, image_path: str) -> np.ndarray:
        h, w = self.input_shape[1], self.input_shape[2]
        channels = self.input_shape[3]
        img = Image.open(image_path).convert("RGB").resize((w, h))
        arr = np.array(img, dtype=np.float32) / 255.0
        # Pad or tile to match expected channels (e.g. 12 for BraTS 2.5D)
        if channels > 3:
            arr = np.tile(arr, (1, 1, (channels // 3) + 1))[:, :, :channels]
        return np.expand_dims(arr, axis=0)

    def predict(self, image_path: str) -> PredictionResult:
        """Run inference on a single image.

        Args:
            image_path: Path to the input image (JPEG or PNG).

        Returns:
            PredictionResult with label, confidence, and timing.
        """
        if self.task == "classification":
            input_data = self._preprocess_classification(image_path)
        else:
            input_data = self._preprocess_segmentation(image_path)

        # Handle INT8 quantization
        if (self._backend == "tflite"
                and self._input_details["dtype"] == np.uint8):
            scale = self._input_details["quantization"][0]
            zero_point = self._input_details["quantization"][1]
            input_data = (input_data / scale + zero_point).astype(np.uint8)

        start = time.perf_counter()
        if self._backend == "tflite":
            self._interpreter.set_tensor(
                self._input_details["index"], input_data)
            self._interpreter.invoke()
            output = self._interpreter.get_tensor(
                self._output_details["index"])
            # De-quantize INT8 output
            if self._output_details["dtype"] == np.uint8:
                scale = self._output_details["quantization"][0]
                zp = self._output_details["quantization"][1]
                output = (output.astype(np.float32) - zp) * scale
        else:
            output = self._session.run(
                None, {self._input_name: input_data})[0]
        elapsed_ms = (time.perf_counter() - start) * 1000

        if self.task == "classification":
            prob = float(1.0 / (1.0 + np.exp(-output[0][0])))  # sigmoid
            label_idx = 1 if prob > 0.5 else 0
            return PredictionResult(
                task="classification",
                label=ISIC_LABELS.get(label_idx, f"Class {label_idx}"),
                confidence=prob if label_idx == 1 else 1 - prob,
                inference_time_ms=elapsed_ms,
                model_name=self.model_name,
                input_shape=self.input_shape,
                raw_output=output,
            )
        else:
            seg_map = np.argmax(output[0], axis=-1)
            class_counts = {BRATS_LABELS[i]: int(np.sum(seg_map == i))
                           for i in range(output.shape[-1])
                           if i in BRATS_LABELS}
            summary = ", ".join(f"{k}: {v}px" for k, v in class_counts.items()
                               if v > 0)
            return PredictionResult(
                task="segmentation",
                label=summary,
                confidence=float(np.max(output)),
                inference_time_ms=elapsed_ms,
                model_name=self.model_name,
                input_shape=self.input_shape,
                raw_output=seg_map,
            )

    def benchmark(self, image_path: str, runs: int = 50) -> dict:
        """Run repeated inference and report latency statistics."""
        times = []
        for _ in range(runs):
            result = self.predict(image_path)
            times.append(result.inference_time_ms)
        return {
            "median_ms": float(np.median(times)),
            "p95_ms": float(np.percentile(times, 95)),
            "min_ms": float(np.min(times)),
            "max_ms": float(np.max(times)),
            "runs": runs,
        }
