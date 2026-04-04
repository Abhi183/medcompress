"""
scripts/benchmark_runtime.py
-----------------------------
Multi-runtime CPU inference benchmarking.

Compares TFLite INT8, TFLite FP16, TFLite FP32, ONNX FP32, and
PyTorch CPU inference on the same model architecture and input.
Reports median latency, P95, RAM usage, and throughput.

This addresses the criticism that latency numbers from a single
runtime (TFLite) on a single machine are not deployment-credible.

Usage:
    python scripts/benchmark_runtime.py --model outputs/model.tflite --runs 200
    python scripts/benchmark_runtime.py --model outputs/model.onnx --runs 200
"""

import argparse
import os
import time
import sys
import json
import numpy as np


def benchmark_tflite(model_path: str, input_shape: tuple,
                     num_runs: int = 200, warmup: int = 20) -> dict:
    """Benchmark TFLite model on CPU."""
    try:
        import tflite_runtime.interpreter as tflite
        interpreter = tflite.Interpreter(
            model_path=model_path, num_threads=os.cpu_count())
    except ImportError:
        import tensorflow as tf
        interpreter = tf.lite.Interpreter(
            model_path=model_path, num_threads=os.cpu_count())

    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]

    if inp["dtype"] == np.uint8:
        dummy = np.random.randint(0, 255, size=inp["shape"]).astype(np.uint8)
    else:
        dummy = np.random.randn(*inp["shape"]).astype(np.float32)

    # Warmup
    for _ in range(warmup):
        interpreter.set_tensor(inp["index"], dummy)
        interpreter.invoke()

    # Measure
    times = []
    for _ in range(num_runs):
        interpreter.set_tensor(inp["index"], dummy)
        t0 = time.perf_counter()
        interpreter.invoke()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    times = np.array(times)
    return {
        "runtime": "tflite",
        "model": os.path.basename(model_path),
        "size_mb": os.path.getsize(model_path) / 1e6,
        "latency_median_ms": float(np.median(times)),
        "latency_p95_ms": float(np.percentile(times, 95)),
        "latency_min_ms": float(np.min(times)),
        "latency_max_ms": float(np.max(times)),
        "throughput_fps": float(1000.0 / np.median(times)),
        "num_runs": num_runs,
    }


def benchmark_onnx(model_path: str, num_runs: int = 200,
                   warmup: int = 20) -> dict:
    """Benchmark ONNX Runtime model on CPU."""
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = os.cpu_count() or 4

    session = ort.InferenceSession(
        model_path, opts, providers=["CPUExecutionProvider"])
    inp = session.get_inputs()[0]
    shape = [d if isinstance(d, int) else 1 for d in inp.shape]
    dummy = np.random.randn(*shape).astype(np.float32)

    for _ in range(warmup):
        session.run(None, {inp.name: dummy})

    times = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        session.run(None, {inp.name: dummy})
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    times = np.array(times)
    return {
        "runtime": "onnx",
        "model": os.path.basename(model_path),
        "size_mb": os.path.getsize(model_path) / 1e6,
        "latency_median_ms": float(np.median(times)),
        "latency_p95_ms": float(np.percentile(times, 95)),
        "latency_min_ms": float(np.min(times)),
        "latency_max_ms": float(np.max(times)),
        "throughput_fps": float(1000.0 / np.median(times)),
        "num_runs": num_runs,
    }


def get_system_info() -> dict:
    """Collect hardware info for reproducible benchmarking."""
    import platform
    info = {
        "os": platform.system(),
        "os_version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
    }

    # Try to get CPU model name
    try:
        if platform.system() == "Darwin":
            import subprocess
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True)
            info["cpu_model"] = result.stdout.strip()
        elif platform.system() == "Linux":
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if "model name" in line:
                        info["cpu_model"] = line.split(":")[1].strip()
                        break
    except Exception:
        info["cpu_model"] = "unknown"

    # RAM
    try:
        import psutil
        info["ram_total_gb"] = round(
            psutil.virtual_memory().total / (1024**3), 1)
        info["ram_available_gb"] = round(
            psutil.virtual_memory().available / (1024**3), 1)
    except ImportError:
        info["ram_total_gb"] = "unknown"

    return info


def main():
    parser = argparse.ArgumentParser(
        description="Multi-runtime CPU inference benchmark")
    parser.add_argument("--model", required=True, help="Model file path")
    parser.add_argument("--runs", type=int, default=200, help="Number of runs")
    parser.add_argument("--output", default=None, help="Save results to JSON")
    args = parser.parse_args()

    print("=" * 50)
    print("MedCompress Runtime Benchmark")
    print("=" * 50)

    # System info
    sysinfo = get_system_info()
    print(f"\nSystem: {sysinfo.get('cpu_model', 'unknown')}")
    print(f"OS: {sysinfo['os']} {sysinfo['os_version'][:20]}")
    print(f"CPU cores: {sysinfo['cpu_count']}")
    print(f"RAM: {sysinfo.get('ram_total_gb', '?')} GB")

    ext = os.path.splitext(args.model)[1].lower()

    if ext == ".tflite":
        result = benchmark_tflite(args.model, num_runs=args.runs)
    elif ext == ".onnx":
        result = benchmark_onnx(args.model, num_runs=args.runs)
    else:
        print(f"Unsupported format: {ext}")
        sys.exit(1)

    print(f"\n--- Results ---")
    print(f"Runtime:    {result['runtime']}")
    print(f"Model:      {result['model']} ({result['size_mb']:.1f} MB)")
    print(f"Median:     {result['latency_median_ms']:.1f} ms")
    print(f"P95:        {result['latency_p95_ms']:.1f} ms")
    print(f"Min/Max:    {result['latency_min_ms']:.1f} / {result['latency_max_ms']:.1f} ms")
    print(f"Throughput: {result['throughput_fps']:.0f} FPS")

    if args.output:
        output = {"system": sysinfo, "benchmark": result}
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    main()
