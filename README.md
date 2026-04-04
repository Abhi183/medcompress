# MedCompress

**Compressing medical imaging models for cross-platform endpoint deployment.**

MedCompress is a compression benchmark for medical imaging deep learning models. It evaluates quantization-aware training (QAT), knowledge distillation (KD), and sparse attention compression on melanoma classification (ISIC 2020) and brain tumor segmentation (BraTS 2021), with export to TFLite and ONNX for CPU inference on macOS, Windows, and Linux endpoints.

**Paper:** [paper/medcompress.pdf](paper/medcompress.pdf) (LaTeX source: [paper/medcompress.tex](paper/medcompress.tex))

**Author:** Abhishek Shekhar

---

## Key Results

| Model | Method | Size | Compression | Accuracy | CPU Latency |
|-------|--------|------|-------------|----------|-------------|
| EfficientNetB0 | QAT INT8 | 4.3 MB | 3.8x | 0.898 AUC | 14.1 ms |
| MobileNetV3-Small | KD + QAT INT8 | 2.7 MB | 6.0x | 0.884 AUC | 8.3 ms |
| U-Net Lite | KD + QAT INT8 | 4.1 MB | 30.3x | 0.804 Dice | 12.6 ms |

All compressed models run under 20 ms on CPU with no GPU required.

---

## Repository Structure

```
medcompress/
├── compression/
│   ├── qat.py                 # Quantization-aware training pipeline
│   ├── distillation.py        # Knowledge distillation with feature matching
│   └── sparse_attention.py    # MSA-inspired sparse attention compression
├── models/
│   └── baseline.py            # EfficientNetB0 + U-Net architectures
├── data/
│   ├── isic_loader.py         # ISIC 2020 dataset loader
│   └── brats_loader.py        # BraTS 2021 2.5D loader
├── configs/                   # YAML experiment configurations
├── scripts/
│   ├── train.py               # Baseline training
│   ├── compress.py            # Compression pipeline (QAT / KD)
│   └── evaluate.py            # Evaluation and benchmarking
├── paper/
│   ├── medcompress.tex        # LaTeX source
│   ├── medcompress.pdf        # Compiled paper (7 pages, 5 figures, 5 tables)
│   ├── references.bib         # BibTeX bibliography (20 references)
│   └── fig*.png               # Publication figures
├── results/
│   ├── compression_results.csv
│   ├── sparse_attention_ablation.csv
│   ├── distillation_ablation.csv
│   └── endpoint_profiling.csv
├── deploy/
│   ├── app.py                 # Desktop GUI (tkinter, cross-platform)
│   ├── cli.py                 # CLI for single/batch inference
│   └── inference.py           # Core inference engine (TFLite + ONNX)
├── figures/
│   ├── generate_figures.py    # Script to regenerate all paper figures
│   ├── fig1_compression_pareto.png
│   ├── fig2_sparse_attention_ablation.png
│   ├── fig3_distillation_ablation.png
│   ├── fig4_endpoint_latency.png
│   └── fig5_model_size.png
├── tests/
│   ├── test_pipeline.py       # Core ML pipeline tests
│   └── test_sparse_attention.py  # Sparse attention tests
└── notebooks/
    └── MedCompress_Demo.ipynb # Reproducible demo
```

---

## Quickstart

```bash
pip install -r requirements.txt

# Train baseline
python scripts/train.py --config configs/isic_baseline.yaml

# Compress with QAT
python scripts/compress.py --config configs/isic_qat.yaml

# Compress with knowledge distillation
python scripts/compress.py --config configs/isic_kd.yaml

# Evaluate
python scripts/evaluate.py --config configs/isic_qat.yaml --tflite outputs/isic_qat_int8.tflite
```

## Datasets

- **ISIC 2020:** [Kaggle](https://www.kaggle.com/c/siim-isic-melanoma-classification) (33,126 dermoscopy images, binary melanoma classification)
- **BraTS 2021:** [Synapse](https://www.synapse.org/brats2021) (multi-modal brain MRI, 4-class tumor segmentation)

## Sparse Attention Compression

The sparse attention module (`compression/sparse_attention.py`) adapts techniques from [Memory Sparse Attention (Chen et al., 2026)](https://github.com/EverMind-AI/MSA) for medical Vision Transformers:

- **KV cache pooling** reduces spatial token sequences by chunk-mean averaging
- **Top-k sparse routing** selects only the most relevant spatial regions per query
- **Decoupled router** uses separate Q/K projections trained with InfoNCE loss

Kernel=4, top-k=8 achieves 24.5x attention memory reduction with 0.7% AUC loss on ISIC classification.

## Deploy on Mac / Windows / Linux

MedCompress includes a ready-to-use desktop application and CLI for running compressed models on any endpoint. No GPU required.

**GUI (desktop app):**
```bash
pip install pillow numpy
# For TFLite models:
pip install tflite-runtime  # or tensorflow
# For ONNX models:
pip install onnxruntime

python deploy/app.py --model path/to/model.tflite
```

**CLI (single image):**
```bash
python deploy/cli.py --model model.tflite --image skin_lesion.jpg
# Output: Prediction: Melanoma (confidence: 87.3%), Inference: 9.2 ms
```

**CLI (batch processing):**
```bash
python deploy/cli.py --model model.tflite --dir /path/to/images/ --output results.json
```

**CLI (benchmark):**
```bash
python deploy/cli.py --model model.tflite --image scan.jpg --benchmark
# Output: Median: 8.3 ms, P95: 9.7 ms (50 runs)
```

The deployment engine supports both `.tflite` and `.onnx` models, auto-detects whether the model is for classification or segmentation, and handles INT8 quantized inputs/outputs transparently.

### Packaging as a Standalone App

To distribute as a standalone binary (no Python required on the target machine):

```bash
pip install pyinstaller
pyinstaller --onefile --windowed deploy/app.py --name MedCompress
# Produces dist/MedCompress.app (macOS) or dist/MedCompress.exe (Windows)
```

## Citation

If you use MedCompress in your research, please cite:

```bibtex
@misc{shekhar2026medcompress,
    title={MedCompress: Compressing Medical Imaging Models for Cross-Platform Endpoint Deployment},
    author={Abhishek Shekhar},
    year={2026},
    url={https://github.com/Abhi183/medcompress}
}
```

This work builds on Memory Sparse Attention:

```bibtex
@misc{chen2026msa,
    title={MSA: Memory Sparse Attention for Efficient End-to-End Memory Model Scaling to 100M Tokens},
    author={Yu Chen and Runkai Chen and Sheng Yi and Xinda Zhao and Xiaohong Li and Shun Fan and Jiangning Zhang and Yabiao Wang},
    year={2026},
    eprint={2603.23516},
    archivePrefix={arXiv}
}
```

## License

MIT
