"""
Generate publication-quality figures for MedCompress paper.
Run: python figures/generate_figures.py
Outputs PNG files to figures/
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
})

COLORS = {
    "baseline": "#2c3e50",
    "qat": "#e74c3c",
    "kd": "#2980b9",
    "kd_qat": "#27ae60",
    "sparse": "#8e44ad",
    "fp16": "#e67e22",
}


def fig1_compression_pareto():
    """Figure 1: Compression ratio vs accuracy (Pareto front)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))

    # ISIC Classification
    isic = {
        "Baseline FP32":     (1.0, 0.912, COLORS["baseline"], "o"),
        "QAT INT8":          (3.8, 0.898, COLORS["qat"], "s"),
        "QAT FP16":          (2.0, 0.910, COLORS["fp16"], "D"),
        "KD (MobileNetV3)":  (1.6, 0.896, COLORS["kd"], "^"),
        "KD + QAT INT8":     (6.0, 0.884, COLORS["kd_qat"], "P"),
    }
    for label, (cr, acc, c, m) in isic.items():
        ax1.scatter(cr, acc, color=c, marker=m, s=80, zorder=5)
        offset = (5, 5) if label != "QAT FP16" else (5, -12)
        ax1.annotate(label, (cr, acc), textcoords="offset points",
                     xytext=offset, fontsize=7.5)

    ax1.set_xlabel("Compression Ratio")
    ax1.set_ylabel("AUC")
    ax1.set_title("(a) ISIC 2020 Melanoma Classification")
    ax1.set_xlim(0, 7)
    ax1.set_ylim(0.875, 0.920)

    # BraTS Segmentation
    brats = {
        "Baseline FP32":     (1.0, 0.871, COLORS["baseline"], "o"),
        "QAT INT8":          (3.9, 0.849, COLORS["qat"], "s"),
        "QAT FP16":          (2.0, 0.868, COLORS["fp16"], "D"),
        "KD (U-Net Lite)":   (7.9, 0.821, COLORS["kd"], "^"),
        "KD + QAT INT8":     (30.3, 0.804, COLORS["kd_qat"], "P"),
    }
    for label, (cr, acc, c, m) in brats.items():
        ax2.scatter(cr, acc, color=c, marker=m, s=80, zorder=5)
        offset = (5, 5) if "KD + QAT" not in label else (5, -12)
        ax2.annotate(label, (cr, acc), textcoords="offset points",
                     xytext=offset, fontsize=7.5)

    ax2.set_xlabel("Compression Ratio")
    ax2.set_ylabel("Dice Coefficient")
    ax2.set_title("(b) BraTS 2021 Brain Tumor Segmentation")
    ax2.set_xlim(0, 34)
    ax2.set_ylim(0.790, 0.880)

    plt.tight_layout()
    plt.savefig("figures/fig1_compression_pareto.png")
    plt.close()
    print("Saved fig1_compression_pareto.png")


def fig2_sparse_attention_ablation():
    """Figure 2: Sparse attention kernel size and top-k vs AUC."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Left: kernel size sweep at fixed top-k=8
    ks = [1, 2, 4, 8]
    auc_k8 = [0.912, 0.910, 0.905, 0.893]
    mem_red = [1, 6.1, 24.5, 49.0]

    ax1b = ax1.twinx()
    bars = ax1.bar(range(len(ks)), auc_k8, color=COLORS["sparse"],
                   alpha=0.7, width=0.5)
    ax1b.plot(range(len(ks)), mem_red, "o-", color=COLORS["qat"],
              linewidth=2, markersize=6)

    ax1.set_xticks(range(len(ks)))
    ax1.set_xticklabels([str(k) for k in ks])
    ax1.set_xlabel("Pooling Kernel Size")
    ax1.set_ylabel("AUC", color=COLORS["sparse"])
    ax1b.set_ylabel("Attention Memory Reduction (x)", color=COLORS["qat"])
    ax1.set_ylim(0.885, 0.915)
    ax1.set_title("(a) Kernel Size Sweep (top-k=8)")

    # Right: top-k sweep at fixed kernel=4
    topk = [4, 8, 16, 49]
    auc_k4 = [0.899, 0.905, 0.909, 0.911]

    ax2.plot(topk, auc_k4, "o-", color=COLORS["sparse"],
             linewidth=2, markersize=7)
    ax2.fill_between(topk, [a - 0.003 for a in auc_k4],
                     [a + 0.003 for a in auc_k4],
                     alpha=0.15, color=COLORS["sparse"])
    ax2.axhline(y=0.912, color=COLORS["baseline"], linestyle="--",
                linewidth=1, label="Baseline (no compression)")
    ax2.set_xlabel("Top-k Selected Chunks")
    ax2.set_ylabel("AUC")
    ax2.set_title("(b) Top-k Sweep (kernel=4)")
    ax2.set_ylim(0.893, 0.916)
    ax2.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig("figures/fig2_sparse_attention_ablation.png")
    plt.close()
    print("Saved fig2_sparse_attention_ablation.png")


def fig3_distillation_ablation():
    """Figure 3: Knowledge distillation temperature and alpha sweep."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Temperature sweep
    temps = [2.0, 4.0, 6.0]
    aucs_t = [0.889, 0.896, 0.893]
    ax1.bar(range(len(temps)), aucs_t, color=COLORS["kd"],
            alpha=0.8, width=0.5)
    ax1.set_xticks(range(len(temps)))
    ax1.set_xticklabels([f"T={t}" for t in temps])
    ax1.set_ylabel("Student AUC")
    ax1.set_title(r"(a) Temperature Sweep ($\alpha$=0.7)")
    ax1.set_ylim(0.882, 0.900)
    for i, v in enumerate(aucs_t):
        ax1.text(i, v + 0.0005, f"{v:.3f}", ha="center", fontsize=9)

    # Alpha sweep
    alphas = [0.5, 0.7, 0.9]
    aucs_a = [0.891, 0.896, 0.887]
    ax2.bar(range(len(alphas)), aucs_a, color=COLORS["kd_qat"],
            alpha=0.8, width=0.5)
    ax2.set_xticks(range(len(alphas)))
    ax2.set_xticklabels([fr"$\alpha$={a}" for a in alphas])
    ax2.set_ylabel("Student AUC")
    ax2.set_title("(b) Alpha Sweep (T=4.0)")
    ax2.set_ylim(0.882, 0.900)
    for i, v in enumerate(aucs_a):
        ax2.text(i, v + 0.0005, f"{v:.3f}", ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig("figures/fig3_distillation_ablation.png")
    plt.close()
    print("Saved fig3_distillation_ablation.png")


def fig4_endpoint_latency():
    """Figure 4: Endpoint deployment latency comparison."""
    fig, ax = plt.subplots(figsize=(8, 4.5))

    models = [
        "EfficientNetB0\nTFLite INT8",
        "MobileNetV3\nKD+INT8",
        "U-Net Lite\nKD+INT8",
        "EfficientNetB0\nONNX FP32",
        "MobileNetV3\nONNX FP16",
    ]
    macos = [14.1, 8.3, 12.6, 47.5, 18.4]
    windows = [16.8, 9.7, 15.2, 52.1, 21.3]

    x = np.arange(len(models))
    w = 0.32
    ax.barh(x + w/2, macos, w, label="macOS CPU", color="#2980b9", alpha=0.85)
    ax.barh(x - w/2, windows, w, label="Windows CPU", color="#e74c3c", alpha=0.85)

    ax.set_yticks(x)
    ax.set_yticklabels(models)
    ax.set_xlabel("Inference Latency (ms)")
    ax.set_title("Cross-Platform Endpoint Inference Latency")
    ax.axvline(x=20, color="gray", linestyle="--", linewidth=1, alpha=0.6)
    ax.text(21, 4.3, "20 ms\ninteractive\nthreshold", fontsize=7.5,
            color="gray", va="top")
    ax.legend(loc="lower right")
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig("figures/fig4_endpoint_latency.png")
    plt.close()
    print("Saved fig4_endpoint_latency.png")


def fig5_model_size_comparison():
    """Figure 5: Model size waterfall showing compression stages."""
    fig, ax = plt.subplots(figsize=(7, 4.5))

    stages = ["Baseline\nFP32", "After\nKD", "After\nQAT INT8", "After\nKD+QAT"]
    isic_sizes = [16.2, 10.2, 4.3, 2.7]
    brats_sizes = [124.1, 15.6, 31.4, 4.1]

    x = np.arange(len(stages))
    w = 0.32
    ax.bar(x - w/2, isic_sizes, w, label="ISIC (Classification)",
           color=COLORS["kd"], alpha=0.85)
    ax.bar(x + w/2, brats_sizes, w, label="BraTS (Segmentation)",
           color=COLORS["qat"], alpha=0.85)

    for i in range(len(stages)):
        ax.text(i - w/2, isic_sizes[i] + 1.5, f"{isic_sizes[i]}",
                ha="center", fontsize=8)
        ax.text(i + w/2, brats_sizes[i] + 1.5, f"{brats_sizes[i]}",
                ha="center", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(stages)
    ax.set_ylabel("Model Size (MB)")
    ax.set_title("Model Size Across Compression Stages")
    ax.legend()
    ax.set_ylim(0, 140)

    plt.tight_layout()
    plt.savefig("figures/fig5_model_size.png")
    plt.close()
    print("Saved fig5_model_size.png")


if __name__ == "__main__":
    fig1_compression_pareto()
    fig2_sparse_attention_ablation()
    fig3_distillation_ablation()
    fig4_endpoint_latency()
    fig5_model_size_comparison()
    print("\nAll figures generated.")
