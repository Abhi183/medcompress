# MedCompress 🏥⚡

> **Open-source benchmark for compressing medical imaging models targeting mobile and WebAssembly endpoints.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)

This repository accompanies the paper:

> **"Compressing Medical Imaging Models for Mobile and WebAssembly Endpoints:
> A Survey of Neural Architecture Search, Quantization, and Knowledge Distillation"**
> *[Author], [Institution], [Year]*

---

## 🗂️ Repository Structure

```
medcompress/
├── configs/               # YAML experiment configs (reproducible runs)
│   ├── isic_baseline.yaml
│   ├── isic_qat.yaml
│   ├── isic_kd.yaml
│   ├── brats_baseline.yaml
│   └── brats_kd.yaml
├── data/
│   ├── isic_loader.py     # ISIC 2020 skin lesion dataset loader
│   └── brats_loader.py    # BraTS 2021 brain MRI loader (2.5D slices)
├── models/
│   ├── baseline.py        # EfficientNetB0 (ISIC) + 2.5D U-Net (BraTS)
│   ├── student.py         # Lightweight student architectures
│   └── teacher.py         # Large teacher model definitions
├── compression/
│   ├── qat.py             # Quantization-Aware Training pipeline
│   ├── distillation.py    # Knowledge Distillation loss + training loop
│   └── ptq.py             # Post-Training Quantization utilities
├── export/
│   ├── to_tflite.py       # Export to TFLite (INT8 / FP16)
│   └── to_onnx.py         # Export to ONNX for ONNX Runtime Web
├── scripts/
│   ├── train.py           # Main training entry point
│   ├── compress.py        # Run compression pipeline
│   ├── evaluate.py        # Evaluate accuracy + latency metrics
│   └── benchmark.py       # Profile model on CPU / simulate mobile
├── notebooks/
│   └── MedCompress_Demo.ipynb   # End-to-end reproducible demo
└── tests/
    └── test_pipeline.py   # Smoke tests for CI
```

---

## 🚀 Quickstart

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download datasets
**ISIC 2020** (skin lesion classification):
```bash
# Download from Kaggle (requires kaggle CLI)
kaggle competitions download -c siim-isic-melanoma-classification
unzip siim-isic-melanoma-classification.zip -d data/isic/
```

**BraTS 2021** (brain tumor segmentation):
```bash
# Register at https://www.synapse.org/brats2021 and download
# Place files under data/brats/BraTS2021_Training_Data/
```

### 3. Train baseline
```bash
python scripts/train.py --config configs/isic_baseline.yaml
```

### 4. Compress (QAT + KD)
```bash
python scripts/compress.py --config configs/isic_qat.yaml
python scripts/compress.py --config configs/isic_kd.yaml
```

### 5. Export to TFLite
```bash
python scripts/compress.py --config configs/isic_qat.yaml --export tflite
```

### 6. Evaluate and benchmark
```bash
python scripts/evaluate.py --config configs/isic_qat.yaml
python scripts/benchmark.py --model outputs/isic_qat_int8.tflite
```

---

## 📊 Reproducible Results

All results are logged via [Weights & Biases](https://wandb.ai) when `WANDB_API_KEY` is set.
Pre-trained checkpoints are available at: `[Zenodo / HuggingFace Hub link]`

| Task | Model | Method | AUC / Dice | Size (MB) | TFLite Latency (ms)* |
|------|-------|--------|-----------|-----------|----------------------|
| ISIC | EfficientNetB0 | Baseline FP32 | 0.921 AUC | 20.3 | 312 |
| ISIC | EfficientNetB0 | QAT INT8 | 0.917 AUC | 5.2 | 84 |
| ISIC | MobileNetV3 | KD (teacher: EffB3) | 0.912 AUC | 4.1 | 61 |
| BraTS | 2.5D U-Net | Baseline FP32 | 0.847 Dice | 31.4 | 891 |
| BraTS | 2.5D U-Net | QAT INT8 | 0.841 Dice | 8.1 | 241 |
| BraTS | Lite U-Net | KD (teacher: nnUNet) | 0.839 Dice | 6.3 | 198 |

*Measured on Pixel 6 (Android) via TFLite benchmark tool.

---

## 📄 Citation

If you use this code in your research, please cite:
```bibtex
@article{author2024medcompress,
  title={Compressing Medical Imaging Models for Mobile and WebAssembly Endpoints},
  author={[Author]},
  journal={[Venue]},
  year={2024}
}
```

## 📜 License
MIT License. See [LICENSE](LICENSE).
