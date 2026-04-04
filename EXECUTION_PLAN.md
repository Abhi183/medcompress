# MedCompress: Execution Plan

## Research Upgrade Pipeline

```mermaid
graph TD
    A[Current State: 6.5/10] --> B{Phase 1: Infrastructure}
    B --> B1[Add CheXpert loader + config]
    B --> B2[Add Kvasir-SEG loader + config]
    B --> B3[Add benchmark_runtime.py]
    B --> B4[Add evaluate_calibration.py]
    B1 --> C{Phase 2: Kaggle Notebooks}
    B2 --> C
    C --> C1[Notebook 1: ISIC full — RUNNING]
    C --> C2[Notebook 2: CheXpert full]
    C --> C3[Notebook 3: Kvasir-SEG full]
    C --> C4[Notebook 4: Sparse bottleneck on BraTS]
    C --> C5[Notebook 5: Capacity study 3 students]
    C --> C6[Notebook 6: Pruning stacking]
    C1 --> D{Phase 3: Collect Results}
    C2 --> D
    C3 --> D
    C4 --> D
    C5 --> D
    C6 --> D
    B3 --> D
    B4 --> D
    D --> D1[Real CSVs from all Kaggle runs]
    D --> D2[TFLite + ONNX models downloaded]
    D --> D3[Endpoint benchmarks on Mac CPU]
    D1 --> E{Phase 4: Single Clean Paper Rewrite}
    D2 --> E
    D3 --> E
    E --> E1[Replace ALL projected numbers]
    E --> E2[Add 2 new dataset sections]
    E --> E3[Add pruning stacking results]
    E --> E4[Add sparse bottleneck segmentation]
    E --> E5[Add capacity study analysis]
    E --> E6[Add multi-runtime benchmark table]
    E --> E7[Add calibration ECE analysis]
    E1 --> F[Compile PDF — target 8/10]
    E2 --> F
    E3 --> F
    E4 --> F
    E5 --> F
    E6 --> F
    E7 --> F
    F --> G[Push once to GitHub]

    style A fill:#e74c3c,color:white
    style F fill:#27ae60,color:white
    style G fill:#2980b9,color:white
    style C1 fill:#f39c12,color:white
```

## Phase Breakdown

### Phase 1: Infrastructure (this session)
Build all missing code. Do NOT touch the paper.

| Task | File | Status |
|------|------|--------|
| CheXpert data loader | `data/chexpert_loader.py` | TODO |
| Kvasir-SEG data loader | `data/kvasir_loader.py` | TODO |
| CheXpert config | `configs/chexpert_baseline.yaml` | TODO |
| Kvasir-SEG config | `configs/kvasir_baseline.yaml` | TODO |
| Calibration metrics (ECE) | `scripts/evaluate_calibration.py` | TODO |
| Kaggle notebook: CheXpert | `notebooks/kaggle_chexpert.py` | TODO |
| Kaggle notebook: Kvasir-SEG | `notebooks/kaggle_kvasir.py` | TODO |
| Kaggle notebook: Capacity study | `notebooks/kaggle_capacity_study.py` | TODO |

### Phase 2: Run on Kaggle (user runs these)
- Notebook 1 (ISIC): RUNNING
- Notebook 2 (CheXpert): after infrastructure
- Notebook 3 (Kvasir-SEG): after infrastructure
- Notebook 4 (Capacity study): after ISIC finishes

### Phase 3: Collect and Verify
- Download all CSVs from Kaggle outputs
- Download .tflite models
- Run endpoint benchmarks on Mac CPU
- Run calibration analysis

### Phase 4: Paper Rewrite
- ONE rewrite with ALL real data
- Do not push intermediate edits
