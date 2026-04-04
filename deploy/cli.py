"""
MedCompress CLI
---------------
Command-line interface for medical image inference.

Usage:
    python deploy/cli.py --model model.tflite --image skin_lesion.jpg
    python deploy/cli.py --model model.tflite --image scan.png --benchmark
    python deploy/cli.py --model model.onnx --dir /path/to/images/
"""

import argparse
import json
import sys
from pathlib import Path

from deploy.inference import MedCompressInference


def run_single(engine: MedCompressInference, image_path: str,
               benchmark: bool = False) -> None:
    result = engine.predict(image_path)
    print(result.summary())

    if benchmark:
        print("\nRunning benchmark (50 iterations)...")
        stats = engine.benchmark(image_path, runs=50)
        print(f"  Median: {stats['median_ms']:.1f} ms")
        print(f"  P95:    {stats['p95_ms']:.1f} ms")
        print(f"  Min:    {stats['min_ms']:.1f} ms")
        print(f"  Max:    {stats['max_ms']:.1f} ms")


def run_batch(engine: MedCompressInference, image_dir: str,
              output_json: str | None = None) -> None:
    dir_path = Path(image_dir)
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    images = sorted(p for p in dir_path.iterdir()
                    if p.suffix.lower() in extensions)

    if not images:
        print(f"No images found in {image_dir}")
        return

    print(f"Processing {len(images)} images from {image_dir}\n")
    results = []

    for img_path in images:
        result = engine.predict(str(img_path))
        print(f"  {img_path.name:40s} -> {result.label:15s} "
              f"({result.confidence:.1%}, {result.inference_time_ms:.1f}ms)")
        results.append({
            "file": img_path.name,
            "label": result.label,
            "confidence": round(result.confidence, 4),
            "time_ms": round(result.inference_time_ms, 1),
        })

    if output_json:
        with open(output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_json}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MedCompress CLI - Medical Image Inference")
    parser.add_argument("--model", required=True,
                        help="Path to .tflite or .onnx model")
    parser.add_argument("--image", help="Path to a single image")
    parser.add_argument("--dir", help="Path to directory of images (batch mode)")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run latency benchmark on the image")
    parser.add_argument("--output", help="Save batch results to JSON file")
    args = parser.parse_args()

    if not args.image and not args.dir:
        parser.error("Provide --image for single inference or --dir for batch")

    engine = MedCompressInference(args.model)
    print(f"Model: {engine.model_name}")
    print(f"Task:  {engine.task}")
    print(f"Input: {engine.input_shape}\n")

    if args.image:
        run_single(engine, args.image, benchmark=args.benchmark)
    elif args.dir:
        run_batch(engine, args.dir, output_json=args.output)


if __name__ == "__main__":
    main()
