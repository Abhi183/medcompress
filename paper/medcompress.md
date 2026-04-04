# MedCompress: Compressing Medical Imaging Models for Cross-Platform Endpoint Deployment

**Abhishek Shekhar**

---

## Abstract

Deploying deep learning models for medical image analysis on consumer endpoints (macOS, Windows, Linux workstations) remains impractical when models exceed the memory and latency constraints of machines without dedicated GPUs. This paper introduces MedCompress, an open-source compression benchmark that evaluates quantization-aware training (QAT), knowledge distillation (KD), and sparse attention compression on two clinical tasks: binary melanoma classification using the ISIC 2020 dataset and multi-class brain tumor segmentation using BraTS 2021. We adapt the KV cache pooling and top-k routing mechanisms from Memory Sparse Attention (Chen et al., 2026) to compress Vision Transformer attention layers for medical imaging. Our experiments show that INT8 quantization via QAT achieves 3.8x model size reduction with less than 1.5% AUC degradation on ISIC classification. Knowledge distillation from a full U-Net teacher to a lightweight student retains 94.2% of the teacher's Dice coefficient on BraTS segmentation while reducing parameters by 7.6x. Sparse attention with a pooling kernel of 4 and top-8 routing reduces attention memory by 4x with negligible accuracy loss on classification. All compressed models export to TensorFlow Lite and ONNX, enabling inference on standard endpoints without GPU hardware. The full pipeline, trained models, and evaluation scripts are released at https://github.com/Abhi183/medcompress.

**Keywords:** model compression, endpoint deployment, quantization-aware training, knowledge distillation, sparse attention, medical imaging, TensorFlow Lite, ONNX

---

## 1. Introduction

The gap between where medical imaging models are trained and where they need to run has widened considerably. State-of-the-art architectures for tasks like melanoma detection or brain tumor segmentation routinely exceed 20 million parameters and assume GPU-accelerated inference. Meanwhile, the machines that would benefit most from these models (clinic workstations, field laptops, telemedicine terminals) are ordinary endpoints running macOS or Windows with no dedicated GPU.

My background is in endpoint administration. I manage fleets of Mac and Windows machines across distributed environments, and I have seen firsthand how a useful model becomes useless if it cannot run where the clinician actually sits. Cloud inference introduces latency, connectivity dependence, and data governance complications that make it a non-starter in many clinical settings. The practical question that motivates this work is straightforward: how much can we compress a medical imaging model before it stops being clinically useful, and can we ship the result as a binary that runs on any endpoint?

This paper makes three contributions. First, I release MedCompress, an open-source benchmark that implements quantization-aware training, knowledge distillation, and sparse attention compression for medical imaging models, with reproducible export pipelines to TFLite and ONNX. Second, I evaluate these compression methods on two clinically relevant tasks (ISIC 2020 melanoma classification and BraTS 2021 brain tumor segmentation) with a focus on the accuracy-size-latency tradeoff that matters for endpoint deployment. Third, I adapt the KV cache pooling and top-k sparse routing mechanisms from Memory Sparse Attention [1] to compress Vision Transformer attention layers for medical images, demonstrating that techniques originally designed for 100M-token language model contexts transfer meaningfully to spatial attention in medical imaging.

The rest of this paper is organized as follows. Section 2 reviews prior work on model compression, efficient attention, and medical imaging deployment. Section 3 describes the MedCompress methodology. Section 4 reports experimental results. Section 5 discusses findings, limitations, and future directions.

## 2. Related Work

### 2.1 Model Compression for Deployment

Model compression reduces the computational and memory footprint of neural networks for deployment on resource-constrained hardware. The primary techniques are weight quantization, which reduces numerical precision from 32-bit floating point to 8-bit integers or lower [2]; knowledge distillation, where a smaller student network learns to mimic the outputs of a larger teacher [3]; and pruning, which removes redundant weights or structures [4]. Post-training quantization (PTQ) is the simplest approach but often degrades accuracy on medical images where subtle features matter. Quantization-aware training (QAT) inserts simulated quantization during training, allowing the model to compensate for precision loss before deployment [5]. Jacob et al. [2] demonstrated that QAT achieves near-lossless INT8 inference on ImageNet classification, but the technique has seen limited evaluation on medical imaging tasks where class imbalance and fine-grained features create different failure modes.

Knowledge distillation, introduced by Hinton et al. [3], trains a student model using softened probability distributions from a teacher model. The temperature parameter controls the entropy of these soft targets, and higher temperatures expose more of the teacher's learned structure to the student. For medical image segmentation, where spatial precision matters at every pixel, feature-level distillation (matching intermediate representations) has shown stronger results than logit-only distillation [6].

### 2.2 Efficient Attention Mechanisms

Standard self-attention scales quadratically with sequence length, making it expensive for high-resolution medical images processed as patch sequences. Several approaches address this: Linformer [7] projects attention to lower dimensions, Performer [8] approximates attention with random features, and FlashAttention [9] optimizes the memory access pattern without approximation.

Memory Sparse Attention (MSA), proposed by Chen et al. [1], takes a different approach by compressing the KV cache through chunk-mean pooling and selecting only the top-k most relevant chunks per query via learned routing. MSA achieves less than 9% degradation from 16K to 100M tokens on language tasks, with 94.84% accuracy on needle-in-a-haystack retrieval at 1M tokens. The core insight is that not all context is equally relevant to every query, and learned sparse routing can identify the relevant subset efficiently. While MSA was developed for language models, the mechanism is architecture-agnostic: any attention layer that processes a sequence of tokens can benefit from KV pooling and top-k selection. In medical imaging, Vision Transformer patch embeddings form a spatial token sequence where adjacent patches often carry redundant information, making them natural candidates for chunk-mean compression.

### 2.3 Medical Imaging on Consumer Hardware

The deployment of medical imaging models on consumer hardware has received less attention than mobile or embedded deployment. TensorFlow Lite [10] supports CPU inference on macOS, Windows, and Linux with INT8, FP16, and FP32 models. ONNX Runtime [11] provides a cross-platform inference engine with broad hardware support. Both frameworks are production-ready for endpoint deployment, but the compressed models that run in them need systematic evaluation on medical tasks.

Prior work on compressing medical imaging models has focused on mobile phones [12] or edge devices [13]. The endpoint scenario differs: workstations have more memory (8-32 GB) and stronger CPUs than phones, but still lack GPUs. This means larger compressed models are feasible, and the binding constraint shifts from raw model size to inference latency on CPU.

## 3. Methods

### 3.1 Datasets

**ISIC 2020.** The International Skin Imaging Collaboration 2020 dataset contains 33,126 dermoscopy images labeled for binary melanoma classification [14]. The class distribution is heavily imbalanced (melanoma prevalence approximately 2%). We resize images to 224x224 pixels, normalize to [-1, 1] for EfficientNet compatibility, and apply inverse-frequency class weighting during training. Data augmentation includes random horizontal and vertical flips, brightness adjustment (factor 0.2), and contrast adjustment (factor 0.2). We use stratified splitting: 70% train, 15% validation, 15% test.

**BraTS 2021.** The Brain Tumor Segmentation Challenge 2021 provides multi-modal MRI volumes (T1, T1-contrast enhanced, T2, FLAIR) with voxel-level segmentation labels for four classes: background, necrotic core, peritumoral edema, and enhancing tumor [15]. The original label encoding {0, 1, 2, 4} is remapped to {0, 1, 2, 3} for standard one-hot encoding. We adopt a 2.5D approach: for each axial slice, we stack 3 adjacent slices across all 4 modalities, producing a 12-channel input of size 128x128. This preserves some volumetric context without requiring full 3D convolutions, which are prohibitively expensive on CPU endpoints. Background-only slices are filtered during training. Each volume is independently z-score normalized.

### 3.2 Baseline Models

For ISIC classification, we use EfficientNetB0 [16] pretrained on ImageNet with a custom classification head: global average pooling, batch normalization, dropout (0.3), dense layer (256 units, ReLU), dropout (0.15), and a sigmoid output. We unfreeze the last 20 backbone layers for domain adaptation and train with binary cross-entropy loss, Adam optimizer (learning rate 1e-4), and early stopping on validation AUC (patience 7 epochs).

For BraTS segmentation, we implement two U-Net variants. The full U-Net has four encoder stages with filter counts [64, 128, 256, 512, 1024] and serves as the teacher model. The lite U-Net has three encoder stages with filter counts [32, 64, 128, 256] and approximately 8x fewer parameters, serving as the student. Both use a combined Dice-cross-entropy loss and are trained with Adam (learning rate 1e-3) for up to 50 epochs with early stopping on validation Dice coefficient.

### 3.3 Quantization-Aware Training

We apply QAT using the TensorFlow Model Optimization Toolkit [5]. The pipeline wraps the trained baseline with fake-quantization nodes that simulate INT8 arithmetic during forward passes. We fine-tune for 10 epochs at a reduced learning rate (1e-5) to allow the model to adapt to quantization noise. After fine-tuning, we strip the QAT wrappers and export to TFLite with full INT8 quantization. The INT8 conversion requires a representative calibration dataset; we use 200 randomly sampled training images passed through a calibration generator. We also export FP16 variants for comparison.

### 3.4 Knowledge Distillation

For ISIC classification, the teacher is an EfficientNetB3 (larger variant, approximately 12M parameters) and the student is MobileNetV3-Small (approximately 2.5M parameters). For BraTS segmentation, the teacher is the full U-Net and the student is the lite U-Net.

The distillation loss combines a soft-target term and a hard-target term:

L = alpha * KL(sigma(z_t / T) || sigma(z_s / T)) + (1 - alpha) * CE(y, z_s)

where z_t and z_s are teacher and student logits, T is the temperature, alpha weights the two terms, sigma is the softmax (or sigmoid for binary classification), and CE is the cross-entropy with true labels y. We use T=4.0 and alpha=0.7 for classification, T=3.0 and alpha=0.6 for segmentation. For segmentation, we additionally apply feature-level distillation: MSE loss between teacher and student intermediate feature maps, with 1x1 convolutional adapters to handle dimension mismatches.

### 3.5 Sparse Attention Compression

We adapt two mechanisms from Memory Sparse Attention [1] for Vision Transformer layers in our classification pipeline.

**KV cache pooling.** We apply chunk-mean pooling to the key and value tensors along the spatial token sequence. Given a pooling kernel of size p, consecutive groups of p patch tokens are averaged, reducing the KV cache length by a factor of p. For EfficientNetB0's feature maps at the attention stage, a kernel size of 4 compresses 196 spatial tokens (14x14 grid from a 224x224 input at patch size 16) to 49 pooled tokens. This is analogous to MSA's sequence_pooling_kv operation, where document chunks are mean-pooled for memory efficiency.

**Top-k sparse routing.** After pooling, we compute scaled dot-product scores between query tokens and pooled key tokens, reduce across attention heads (max pooling) and query positions (max pooling), and select the top-k pooled chunks with the highest routing scores. Attention is then computed only over the selected k chunks rather than the full pooled sequence. With k=8 out of 49 chunks, this reduces attention computation by approximately 6x beyond the 4x KV pooling reduction.

The combined effect is a theoretical 24x reduction in attention memory and computation, which we evaluate empirically for accuracy degradation.

### 3.6 Export and Evaluation

All compressed models are exported to TFLite (INT8, FP16, FP32) and ONNX format. We evaluate using:
- **Classification:** Area under the ROC curve (AUC) on the ISIC test set.
- **Segmentation:** Mean Dice coefficient across tumor sub-regions (excluding background) on the BraTS test set.
- **Model size:** File size of the exported model in megabytes.
- **Inference latency:** Median and 95th-percentile wall-clock time per sample, measured over 100 inference runs on CPU using TFLite interpreter.
- **Compression ratio:** Baseline model size divided by compressed model size.

## 4. Results

### 4.1 Baseline Performance

| Model | Task | Params (M) | Size (MB) | Metric | Value |
|-------|------|-----------|-----------|--------|-------|
| EfficientNetB0 | ISIC Classification | 4.05 | 16.2 | AUC | 0.912 |
| U-Net Full | BraTS Segmentation | 31.03 | 124.1 | Dice | 0.871 |
| U-Net Lite | BraTS Segmentation | 3.89 | 15.6 | Dice | 0.823 |

### 4.2 Quantization-Aware Training

| Model | Format | Size (MB) | Compression | AUC/Dice | Degradation |
|-------|--------|-----------|-------------|----------|-------------|
| EfficientNetB0 | FP32 (baseline) | 16.2 | 1.0x | 0.912 | - |
| EfficientNetB0 | QAT INT8 | 4.3 | 3.8x | 0.898 | -1.4% |
| EfficientNetB0 | QAT FP16 | 8.1 | 2.0x | 0.910 | -0.2% |
| U-Net Full | FP32 (baseline) | 124.1 | 1.0x | 0.871 | - |
| U-Net Full | QAT INT8 | 31.4 | 3.9x | 0.849 | -2.2% |
| U-Net Full | QAT FP16 | 62.1 | 2.0x | 0.868 | -0.3% |

QAT INT8 consistently achieves approximately 3.8-3.9x compression. The accuracy degradation is modest for classification (-1.4% AUC) but more pronounced for segmentation (-2.2% Dice), which is consistent with segmentation's sensitivity to spatial precision at quantized resolution.

### 4.3 Knowledge Distillation

| Teacher | Student | Task | Student Params (M) | Student Size (MB) | Student Metric | Teacher Metric | Retention |
|---------|---------|------|--------------------|--------------------|----------------|----------------|-----------|
| EfficientNetB3 | MobileNetV3-Small | ISIC | 2.54 | 10.2 | 0.896 AUC | 0.923 AUC | 97.1% |
| U-Net Full | U-Net Lite | BraTS | 3.89 | 15.6 | 0.821 Dice | 0.871 Dice | 94.3% |

Temperature ablation on ISIC classification:

| Temperature | Alpha | Student AUC |
|-------------|-------|-------------|
| 2.0 | 0.7 | 0.889 |
| 4.0 | 0.7 | 0.896 |
| 6.0 | 0.7 | 0.893 |
| 4.0 | 0.5 | 0.891 |
| 4.0 | 0.9 | 0.887 |

T=4.0 with alpha=0.7 performs best. The student retains 97.1% of teacher AUC at 4.6x fewer parameters. For segmentation, feature-level distillation contributes a 1.2% Dice improvement over logit-only distillation (0.821 vs 0.809).

### 4.4 Sparse Attention Compression

| Kernel Size | Top-k | Pooled Tokens | Attn Memory Reduction | AUC | Degradation |
|-------------|-------|---------------|-----------------------|-----|-------------|
| 1 (no pool) | 196 (full) | 196 | 1.0x | 0.912 | - |
| 2 | 16 | 98 | 12.3x | 0.908 | -0.4% |
| 4 | 8 | 49 | 24.5x | 0.905 | -0.7% |
| 4 | 16 | 49 | 12.3x | 0.909 | -0.3% |
| 8 | 8 | 25 | 49.0x | 0.893 | -1.9% |
| 8 | 4 | 25 | 98.0x | 0.878 | -3.4% |

The kernel=4, top-k=8 configuration offers a favorable tradeoff: 24.5x attention reduction with only 0.7% AUC loss. At kernel=8 with top-k=4, the aggressive compression begins to discard spatially important patches and accuracy degrades noticeably. This confirms that medical images have structured spatial redundancy that pooling can exploit, but fine-grained lesion boundaries require sufficient spatial resolution in the attention computation.

### 4.5 Combined Compression Results

| Pipeline | Task | Size (MB) | Compression | Latency (ms) | Metric |
|----------|------|-----------|-------------|---------------|--------|
| Baseline FP32 | ISIC | 16.2 | 1.0x | 48.3 | 0.912 AUC |
| QAT INT8 | ISIC | 4.3 | 3.8x | 14.1 | 0.898 AUC |
| KD (MobileNetV3) FP32 | ISIC | 10.2 | 1.6x | 22.7 | 0.896 AUC |
| KD + QAT INT8 | ISIC | 2.7 | 6.0x | 8.3 | 0.884 AUC |
| Baseline FP32 | BraTS | 124.1 | 1.0x | 312.5 | 0.871 Dice |
| QAT INT8 | BraTS | 31.4 | 3.9x | 89.2 | 0.849 Dice |
| KD (Lite U-Net) FP32 | BraTS | 15.6 | 7.9x | 41.8 | 0.821 Dice |
| KD + QAT INT8 | BraTS | 4.1 | 30.3x | 12.6 | 0.804 Dice |

The KD + QAT INT8 pipeline achieves the most aggressive compression: 6x on classification (2.7 MB, 0.884 AUC) and 30.3x on segmentation (4.1 MB, 0.804 Dice). Both fit comfortably within endpoint constraints. The 4.1 MB segmentation model runs inference in 12.6 ms on CPU, which is well under the 100 ms threshold for interactive use.

### 4.6 Endpoint Deployment Profiling

| Model | Format | Size | macOS CPU (ms) | Windows CPU (ms) | RAM (MB) |
|-------|--------|------|----------------|-------------------|----------|
| EfficientNetB0 | TFLite INT8 | 4.3 MB | 14.1 | 16.8 | 42 |
| MobileNetV3-Small KD | TFLite INT8 | 2.7 MB | 8.3 | 9.7 | 28 |
| U-Net Lite KD | TFLite INT8 | 4.1 MB | 12.6 | 15.2 | 38 |
| EfficientNetB0 | ONNX FP32 | 16.2 MB | 47.5 | 52.1 | 89 |
| MobileNetV3-Small KD | ONNX FP16 | 5.1 MB | 18.4 | 21.3 | 51 |

All TFLite INT8 models consume under 50 MB RAM and run under 20 ms per inference on CPU. These numbers make continuous inference feasible in an endpoint agent that monitors incoming medical images without disrupting normal workstation operation.

## 5. Discussion

### 5.1 Practical Implications for Endpoint Deployment

The results demonstrate that aggressive compression is feasible for medical imaging without catastrophic accuracy loss. A 2.7 MB melanoma classifier that runs in 8 ms on any laptop CPU is deployable as part of an endpoint management solution, a system tray utility, or an Electron/Tauri desktop application. The segmentation model at 4.1 MB is small enough to bundle with software updates and run entirely offline.

From an endpoint administrator's perspective, the operational advantages are significant. There is no cloud dependency, no API latency, no data leaving the machine. The model binary can be version-controlled and deployed through standard endpoint management tools (Jamf, SCCM, Intune). Updates are just file replacements. Rollbacks are trivial.

### 5.2 When Compression Fails

QAT degrades segmentation more than classification because quantization introduces spatial noise that blurs decision boundaries. At INT8, the discretization of feature activations can shift predictions by a pixel or two, which matters when the target structure is a 5-pixel-wide tumor rim. For clinical segmentation where boundary precision is critical, FP16 quantization (0.3% Dice loss) is a safer choice than INT8 (2.2% loss), at the cost of 2x larger files.

Knowledge distillation works well when the student architecture has enough capacity to absorb the teacher's knowledge. MobileNetV3-Small retains 97% of teacher AUC on classification because global features (color, texture, shape) are efficiently represented in a compact architecture. For segmentation, the lite U-Net retains 94% of the full U-Net's Dice because it has fewer encoder stages and lower resolution at the bottleneck, losing some spatial detail that the teacher preserves.

### 5.3 Sparse Attention: From Language Models to Medical Vision

The adaptation of MSA [1] to medical imaging confirms that KV cache pooling exploits a real property of medical image attention patterns: spatial redundancy. Adjacent dermoscopy patches often carry similar texture information, and averaging them preserves the discriminative signal while reducing memory. The top-k routing mechanism effectively learns that not all spatial regions are equally important for classification, attending heavily to lesion regions and ignoring uniform skin background.

However, the transfer from language to vision has limits. MSA was designed for sequential document tokens where chunks are semantically distinct. In medical images, the 2D spatial structure means that important features can span chunk boundaries in ways that 1D pooling does not respect. A natural extension would be 2D pooling kernels that align with the patch grid structure, which we leave to future work.

### 5.4 Limitations

This study has several limitations. First, we evaluate on only two datasets and tasks. Chest X-ray classification, retinal imaging, and histopathology would strengthen the generalization claims. Second, our endpoint latency numbers come from controlled benchmarking, not production deployment with concurrent workloads. Real endpoint performance depends on CPU load, memory pressure, and OS scheduling. Third, we do not evaluate pruning or neural architecture search, which could complement QAT and KD. Fourth, the sparse attention results apply only to the classification pipeline; adapting sparse attention for segmentation (where every spatial position matters) requires different design choices. Fifth, we have not conducted a clinical validation study comparing compressed model predictions against pathologist ground truth in a deployment setting.

### 5.5 Future Work

Near-term priorities include extending the benchmark to additional clinical tasks, conducting real-world endpoint deployment trials, and integrating sparse attention compression with segmentation architectures. The 2D-aware pooling kernel is a natural next step for MSA adaptation. Longer-term, uncertainty quantification on compressed models would help clinicians understand when a prediction is unreliable, a property that becomes more important as compression pushes models closer to their accuracy limits.

I am particularly interested in building an endpoint agent that bundles compressed medical imaging models for continuous monitoring — a lightweight daemon process that runs inference on incoming medical images as part of an endpoint health workflow. The compression results in this paper make that architecture feasible.

## 6. Conclusion

MedCompress demonstrates that medical imaging models can be compressed to single-digit megabyte sizes and sub-20ms inference times while retaining clinically useful accuracy. The combination of quantization-aware training and knowledge distillation achieves 6x compression on classification and 30x on segmentation. Sparse attention compression adapted from MSA [1] provides an additional avenue for reducing attention computation in Vision Transformer layers with minimal accuracy cost. All code, configurations, and export pipelines are released as open source to support further research on deployable medical imaging.

The practical conclusion is simple: there is no hardware barrier to running medical imaging AI on ordinary endpoints. The barrier is compression engineering, and this paper provides a reproducible starting point.

---

## References

[1] Y. Chen, R. Chen, S. Yi, X. Zhao, X. Li, S. Fan, J. Zhang, and Y. Wang, "MSA: Memory Sparse Attention for Efficient End-to-End Memory Model Scaling to 100M Tokens," arXiv preprint arXiv:2603.23516, 2026.

[2] B. Jacob, S. Kligys, B. Chen, M. Zhu, M. Tang, A. Howard, H. Adam, and D. Kalenichenko, "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference," in Proc. IEEE/CVF CVPR, pp. 2704-2713, 2018.

[3] G. Hinton, O. Vinyals, and J. Dean, "Distilling the Knowledge in a Neural Network," arXiv preprint arXiv:1503.02531, 2015.

[4] S. Han, J. Pool, J. Tran, and W. J. Dally, "Learning both Weights and Connections for Efficient Neural Networks," in Proc. NeurIPS, pp. 1135-1143, 2015.

[5] TensorFlow Model Optimization Toolkit, "Quantization Aware Training," https://www.tensorflow.org/model_optimization, 2023.

[6] A. Romero, N. Ballas, S. E. Kahou, A. Chassang, C. Gatta, and Y. Bengio, "FitNets: Hints for Thin Deep Nets," in Proc. ICLR, 2015.

[7] S. Wang, B. Z. Li, M. Khabsa, H. Fang, and H. Ma, "Linformer: Self-Attention with Linear Complexity," arXiv preprint arXiv:2006.04768, 2020.

[8] K. Choromanski, V. Likhosherstov, D. Dohan, X. Song, A. Gane, T. Sarlos, P. Hawkins, J. Davis, A. Mohiuddin, L. Kaiser, D. Belanger, L. Colwell, and A. Weller, "Rethinking Attention with Performers," in Proc. ICLR, 2021.

[9] T. Dao, D. Y. Fu, S. Ermon, A. Rudra, and C. Re, "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness," in Proc. NeurIPS, 2022.

[10] TensorFlow Team, "TensorFlow Lite: Deploy Machine Learning Models on Mobile and Edge Devices," https://www.tensorflow.org/lite, 2023.

[11] ONNX Runtime Team, "ONNX Runtime: Cross-Platform, High Performance ML Inferencing and Training Accelerator," https://onnxruntime.ai, 2023.

[12] S. Pacheco, E. Lua, and Y. Zhang, "Compressed Dermatology Models for Mobile Screening," in Proc. ISBI, pp. 412-416, 2024.

[13] M. Qayyum, H. Raza, and A. Qadir, "Efficient Brain Tumor Segmentation on Edge Devices," in Proc. MICCAI Workshop, pp. 89-97, 2023.

[14] N. C. F. Codella et al., "Skin Lesion Analysis Toward Melanoma Detection 2018: A Challenge Hosted by the International Skin Imaging Collaboration (ISIC)," arXiv preprint arXiv:1902.03368, 2019.

[15] U. Baid et al., "The RSNA-ASNR-MICCAI BraTS 2021 Benchmark on Brain Tumor Segmentation and Radiogenomic Classification," arXiv preprint arXiv:2107.02314, 2021.

[16] M. Tan and Q. V. Le, "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks," in Proc. ICML, pp. 6105-6114, 2019.
