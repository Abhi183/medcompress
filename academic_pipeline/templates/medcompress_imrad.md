# MedCompress: Efficient Compression of Medical Imaging Models via Quantization, Distillation, and Sparse Attention for Edge Deployment

**Author(s):** [Author Name(s)]
**Affiliation(s):** [Department, Institution]
**Corresponding Author:** [Name, Email]
**Date:** [Date]

---

## Abstract

[Background: 1-2 sentences on the growing need for deploying medical imaging AI at the edge — mobile clinics, resource-limited settings, browser-based diagnostics.]
[Problem: 1 sentence on the gap between large model accuracy and deployment constraints (memory, latency, power).]
[Method: 1-2 sentences describing the MedCompress pipeline — QAT, KD, and MSA-inspired sparse attention compression with TFLite/ONNX export — evaluated on ISIC 2020 and BraTS 2021.]
[Findings: 2-3 sentences with specific compression ratios, accuracy retention, and latency improvements across all three compression methods.]
[Implications: 1 sentence on what this enables for point-of-care medical imaging.]

**Keywords**: model compression, quantization-aware training, knowledge distillation, sparse attention, KV cache pooling, medical imaging, mobile deployment, TFLite, ONNX

---

## 1. Introduction

### 1.1 Clinical Need for Edge Deployment
[Why medical imaging AI must move beyond centralized GPU servers: connectivity gaps, latency-sensitive workflows, privacy constraints, cost of cloud inference in low-resource settings.]

### 1.2 The Compression Gap
[State-of-the-art medical imaging models (EfficientNet, U-Net) achieve strong accuracy but exceed mobile memory and latency budgets. Quantify the gap: model sizes, inference times on target hardware.]

### 1.3 Research Gap
[Existing compression literature focuses on natural images (ImageNet, COCO). Medical imaging compression is underexplored, especially for: (a) segmentation tasks with spatial precision requirements, (b) multi-modal inputs like BraTS 2.5D, (c) combined QAT + KD pipelines, and (d) attention-level compression for ViT-based medical models using sparse routing.]

### 1.4 Contributions
This paper makes the following contributions:
1. **MedCompress**, an open-source benchmark for compressing medical imaging models targeting mobile and WASM endpoints.
2. A systematic evaluation of QAT, KD, and **MSA-inspired sparse attention compression** on two clinical tasks: binary melanoma classification (ISIC 2020) and multi-class brain tumor segmentation (BraTS 2021).
3. Adaptation of Memory Sparse Attention (Chen et al., 2026) techniques, including KV cache chunk-mean pooling and top-k sparse routing, for Vision Transformer attention layers in medical imaging.
4. A 2.5D compression strategy for volumetric segmentation that enables mobile deployment without full 3D convolutions.
5. Reproducible export pipelines to TFLite (INT8/FP16) and ONNX with latency profiling.

### 1.5 Paper Organization
[Section 2 reviews related work. Section 3 describes the MedCompress methodology. Section 4 presents experimental results. Section 5 discusses findings and limitations. Section 6 concludes.]

---

## 2. Literature Review

### 2.1 Model Compression Techniques
[Survey QAT, PTQ, KD, pruning, neural architecture search. Cite foundational works: Hinton et al. (2015) for KD, Jacob et al. (2018) for quantization, Han et al. (2016) for pruning.]

### 2.2 Sparse Attention and Memory-Efficient Transformers
[Review efficient attention mechanisms: Linformer, Performer, FlashAttention, and Memory Sparse Attention (Chen et al., 2026). MSA introduces KV cache chunk-mean pooling and top-k document routing to achieve 100M-token context with less than 9% degradation. Discuss applicability to Vision Transformers in medical imaging where spatial attention patterns are structured and compressible.]

### 2.3 Medical Image Classification at the Edge
[Review mobile-targeted classification: skin lesion analysis, retinal imaging, chest X-ray. Cite relevant ISIC challenge work and mobile health deployments.]

### 2.4 Medical Image Segmentation Compression
[Review compressed segmentation: lightweight U-Net variants, MobileNet-based decoders, 2D vs 2.5D vs 3D trade-offs for brain MRI.]

### 2.5 Deployment Formats and Runtimes
[Review TFLite, ONNX Runtime, WebAssembly for medical AI. Compare inference engines, hardware support, quantization support.]

### 2.6 Summary and Gap Analysis
[Synthesize: no existing benchmark combines QAT + KD + sparse attention compression for both classification and segmentation in medical imaging with mobile export evaluation. While MSA (Chen et al., 2026) demonstrates sparse attention scaling for LLMs, its core techniques (KV pooling, top-k routing) have not been adapted for medical Vision Transformers.]

---

## 3. Methodology

### 3.1 Overview
[High-level description of the MedCompress pipeline: baseline training → compression (QAT or KD) → export (TFLite/ONNX) → evaluation.]

### 3.2 Datasets

#### 3.2.1 ISIC 2020
[33,126 dermoscopy images, binary melanoma classification. Preprocessing: resize 224×224, normalize to [-1, 1]. Class imbalance handling via inverse-frequency weighting. Train/val/test split with stratification.]

#### 3.2.2 BraTS 2021
[Multi-modal brain MRI (T1, T1ce, T2, FLAIR). 2.5D approach: stack N adjacent axial slices across 4 modalities → (128, 128, N×4) input. Label remapping {0,1,2,4} → {0,1,2,3}. Z-score normalization per volume. Background slice filtering.]

### 3.3 Baseline Models

#### 3.3.1 EfficientNetB0 (Classification)
[Transfer learning from ImageNet. Architecture: GlobalAveragePooling2D → Dense(256, ReLU) → Dense(1, sigmoid). Freeze all but last 20 layers.]

#### 3.3.2 U-Net Full and Lite (Segmentation)
[Full U-Net: 4 encoder stages (64→128→256→512→1024). Lite U-Net: 3 stages (32→64→128→256), ~8× fewer parameters. Both use Dice + CE loss.]

### 3.4 Compression Methods

#### 3.4.1 Quantization-Aware Training
[Fake-quantization nodes inserted via TensorFlow Model Optimization. Fine-tune for T epochs at reduced learning rate. Strip QAT wrappers. Export to TFLite INT8 with representative calibration dataset.]

#### 3.4.2 Knowledge Distillation
[Teacher: larger model (EfficientNetB3 or Full U-Net). Student: smaller model (MobileNetV3Small or Lite U-Net). Loss: α × KL(teacher_soft ‖ student_soft) + (1−α) × CE(y_true, student). Temperature T softens distributions. Optional feature-level MSE matching with 1×1 conv adapters.]

#### 3.4.3 MSA-Inspired Sparse Attention Compression
[Adaptation of Memory Sparse Attention (Chen et al., 2026) for Vision Transformer layers in medical imaging. Three components: (a) KV cache chunk-mean pooling reduces the key/value sequence by a configurable kernel factor (default 4x), compressing spatial attention memory; (b) Top-k sparse routing selects only the k most relevant spatial chunks per query via scaled dot-product scoring with head and query reduction; (c) Optional decoupled router with separate Q/K projections trained via InfoNCE auxiliary loss. For medical imaging, spatial tokens from ViT patch embeddings are treated analogously to document chunks in MSA, enabling structured compression of spatially coherent regions in dermoscopy and MRI data.]

### 3.5 Export and Deployment
[TFLite conversion (INT8, FP16, FP32). ONNX export via tf2onnx. Latency measurement protocol: median and p95 over N inference runs.]

### 3.6 Evaluation Metrics
[Classification: ROC-AUC. Segmentation: mean Dice coefficient (excluding background). Size: model file size (MB). Latency: ms per inference (median, p95). Compression ratio: baseline size / compressed size.]

---

## 4. Results

### 4.1 Baseline Performance
[Table: baseline model sizes, parameter counts, AUC/Dice scores, inference latency.]

### 4.2 QAT Results
[Table: INT8 vs FP16 vs FP32 — accuracy retention, size reduction, latency improvement for both ISIC and BraTS tasks.]

### 4.3 Knowledge Distillation Results
[Table: teacher vs student accuracy, distillation temperature ablation, alpha weighting ablation. Feature matching vs logit-only comparison.]

### 4.4 Sparse Attention Compression Results
[Table: attention memory reduction vs accuracy retention for different kernel_size (2, 4, 8) and top_k (4, 8, 16) settings. Comparison of coupled vs decoupled router. InfoNCE auxiliary loss ablation. Per-task analysis: classification (global features) vs segmentation (spatial precision).]

### 4.5 Combined Pipeline Results
[Table: best compression pipeline per task — QAT alone, KD alone, MSA alone, QAT+KD, QAT+MSA, KD+MSA, QAT+KD+MSA. Pareto analysis of accuracy vs size vs latency vs attention memory. Comparison with prior medical compression work.]

### 4.6 Deployment Profiling
[TFLite and ONNX runtime benchmarks on representative mobile hardware. Memory footprint during inference.]

---

## 5. Discussion

### 5.1 Key Findings
[Summarize: what compression ratios are achievable? What accuracy is retained? Which method works best for which task?]

### 5.2 Classification vs Segmentation Trade-offs
[Discuss why QAT works well for classification but KD is preferable for segmentation. Sparse attention compression sits between: it reduces attention compute without sacrificing spatial precision when top-k is set appropriately. Spatial precision requirements of segmentation vs global feature requirements of classification.]

### 5.3 The 2.5D Compromise
[Discuss trade-offs of 2.5D vs full 3D for BraTS. What volumetric context is lost? When is 2.5D sufficient for clinical use?]

### 5.4 Clinical Deployment Considerations
[Regulatory implications, model validation requirements, failure modes in clinical settings, need for uncertainty quantification.]

### 5.5 Sparse Attention: From LLMs to Medical Vision
[Discuss the transfer of MSA (Chen et al., 2026) techniques from language model context compression to medical imaging attention compression. Key insight: spatial tokens in ViT are analogous to document chunks in MSA. KV pooling exploits spatial coherence in dermoscopy and MRI. Limitations: MSA was designed for sequential text, not 2D spatial grids; future work could explore 2D-aware pooling kernels.]

### 5.6 Limitations
[Data: only two datasets/tasks. Hardware: no real mobile device profiling. Clinical: no clinical validation study. Compression: no pruning or NAS methods evaluated. MSA adaptation: preliminary; deeper integration with ViT architectures needed.]

### 5.7 Future Work
[Extend to more tasks (chest X-ray, retinal), add pruning and NAS, real mobile device benchmarks, clinical pilot study, uncertainty-aware compression. For sparse attention: explore learnable 2D pooling kernels, end-to-end MSA training with medical ViTs, and dynamic top-k selection based on image complexity.]

---

## 6. Conclusion

[2 paragraphs summarizing contributions and impact. End with a forward-looking statement about democratizing medical AI through compression.]

---

## AI Disclosure

This paper was prepared with the assistance of AI-powered academic writing tools. All content, experimental design, and results have been reviewed and verified by the author(s).

---

## References

[IEEE format. Numbered in order of appearance.]

---

## Appendices

### Appendix A: Hyperparameter Configurations
[Full YAML configs for reproducibility.]

### Appendix B: Additional Results
[Per-class segmentation Dice scores, confusion matrices, additional ablation studies.]
