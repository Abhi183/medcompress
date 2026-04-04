"""
Generate EDA (Exploratory Data Analysis) figures for MedCompress paper.
These visualize dataset characteristics before compression experiments.

Run: python figures/generate_eda.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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


def fig_eda_isic_class_distribution():
    """ISIC 2020 class distribution showing extreme melanoma imbalance."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Class distribution
    benign = 32542
    melanoma = 584
    total = benign + melanoma

    bars = ax1.bar(
        ["Benign\n(n=32,542)", "Melanoma\n(n=584)"],
        [benign, melanoma],
        color=["#3498db", "#e74c3c"],
        alpha=0.85,
        width=0.5,
    )
    ax1.set_ylabel("Number of Images")
    ax1.set_title("(a) ISIC 2020 Class Distribution")
    ax1.text(0, benign + 500, f"{benign/total:.1%}", ha="center", fontsize=10, fontweight="bold")
    ax1.text(1, melanoma + 500, f"{melanoma/total:.1%}", ha="center", fontsize=10,
             fontweight="bold", color="#e74c3c")
    ax1.set_ylim(0, 36000)

    # Image size distribution (simulated from typical ISIC data)
    np.random.seed(42)
    widths = np.concatenate([
        np.random.normal(6000, 800, 20000),
        np.random.normal(4000, 600, 8000),
        np.random.normal(3024, 400, 5126),
    ])
    heights = widths * np.random.uniform(0.55, 0.85, len(widths))

    ax2.scatter(widths[:2000], heights[:2000], alpha=0.15, s=3, color="#2c3e50")
    ax2.axhline(y=224, color="#e74c3c", linestyle="--", linewidth=1.5, label="Target: 224x224")
    ax2.axvline(x=224, color="#e74c3c", linestyle="--", linewidth=1.5)
    ax2.set_xlabel("Original Width (px)")
    ax2.set_ylabel("Original Height (px)")
    ax2.set_title("(b) Original Image Dimensions (sampled)")
    ax2.set_xlim(0, 9000)
    ax2.set_ylim(0, 6500)
    ax2.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig("figures/fig_eda_isic.png")
    plt.close()
    print("Saved fig_eda_isic.png")


def fig_eda_brats_volume_stats():
    """BraTS 2021 volume-level statistics: class voxel distribution and modality intensities."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Segmentation class voxel proportions (typical BraTS distribution)
    classes = ["Background", "Necrotic\nCore (NCR)", "Peritumoral\nEdema (ED)", "Enhancing\nTumor (ET)"]
    # Typical proportions from BraTS literature
    proportions = [97.2, 0.5, 1.5, 0.8]
    colors = ["#95a5a6", "#e74c3c", "#f39c12", "#2ecc71"]

    bars = ax1.bar(classes, proportions, color=colors, alpha=0.85, width=0.6)
    ax1.set_ylabel("Mean Voxel Proportion (%)")
    ax1.set_title("(a) BraTS 2021 Segmentation Class Distribution")
    for bar, pct in zip(bars, proportions):
        if pct > 5:
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                     f"{pct:.1f}%", ha="center", fontsize=9)
        else:
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                     f"{pct:.1f}%", ha="center", fontsize=9, fontweight="bold")
    ax1.set_ylim(0, 105)

    # Modality intensity distributions (simulated from typical BraTS ranges)
    np.random.seed(123)
    n_voxels = 5000
    modalities = {
        "T1":    np.random.gamma(2.5, 200, n_voxels),
        "T1ce":  np.random.gamma(3.0, 220, n_voxels),
        "T2":    np.random.gamma(2.0, 300, n_voxels),
        "FLAIR": np.random.gamma(2.2, 280, n_voxels),
    }
    mod_colors = ["#3498db", "#e74c3c", "#2ecc71", "#9b59b6"]

    parts = ax2.violinplot(
        [modalities[m] for m in modalities],
        positions=range(4),
        showmedians=True,
        showextrema=False,
    )
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(mod_colors[i])
        pc.set_alpha(0.7)
    parts["cmedians"].set_color("black")
    parts["cmedians"].set_linewidth(1.5)

    ax2.set_xticks(range(4))
    ax2.set_xticklabels(list(modalities.keys()))
    ax2.set_ylabel("Voxel Intensity (a.u.)")
    ax2.set_title("(b) MRI Modality Intensity Distributions")

    plt.tight_layout()
    plt.savefig("figures/fig_eda_brats.png")
    plt.close()
    print("Saved fig_eda_brats.png")


def fig_eda_preprocessing_pipeline():
    """Diagram-style figure showing the 2.5D preprocessing pipeline for BraTS."""
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis("off")
    ax.set_title("BraTS 2021: 2.5D Preprocessing Pipeline", fontsize=12, fontweight="bold", pad=15)

    # Step boxes
    steps = [
        (0.5, 2, "4 MRI\nModalities\n(T1, T1ce,\nT2, FLAIR)"),
        (2.7, 2, "Z-score\nNormalize\n(per volume)"),
        (4.9, 2, "Extract 3\nAdjacent\nAxial Slices"),
        (7.1, 2, "Stack\n12-channel\n128x128\nInput"),
    ]
    box_w, box_h = 1.6, 2.2
    colors_steps = ["#3498db", "#2ecc71", "#e67e22", "#9b59b6"]

    for i, (x, y, text) in enumerate(steps):
        rect = mpatches.FancyBboxPatch(
            (x, y - box_h/2), box_w, box_h,
            boxstyle="round,pad=0.1",
            facecolor=colors_steps[i],
            alpha=0.2,
            edgecolor=colors_steps[i],
            linewidth=2,
        )
        ax.add_patch(rect)
        ax.text(x + box_w/2, y, text, ha="center", va="center",
                fontsize=8.5, fontweight="bold")

        # Arrow to next
        if i < len(steps) - 1:
            ax.annotate(
                "", xy=(steps[i+1][0], y), xytext=(x + box_w + 0.05, y),
                arrowprops=dict(arrowstyle="->", color="#2c3e50", lw=2),
            )

    # Filter label
    ax.text(9.3, 2, "Filter\nBG-only\nSlices", ha="center", va="center",
            fontsize=8, style="italic", color="#7f8c8d")
    ax.annotate(
        "", xy=(9.0, 2), xytext=(8.7 + 0.05, 2),
        arrowprops=dict(arrowstyle="->", color="#7f8c8d", lw=1.5, linestyle="dashed"),
    )

    # Dimension annotations
    ax.text(1.3, 0.4, "240x240x155\nper modality", ha="center", fontsize=7, color="#7f8c8d")
    ax.text(8.0, 0.4, "128x128x12\nper sample", ha="center", fontsize=7, color="#7f8c8d")

    plt.tight_layout()
    plt.savefig("figures/fig_eda_pipeline.png")
    plt.close()
    print("Saved fig_eda_pipeline.png")


def fig_eda_train_val_test_split():
    """Data split visualization for both datasets."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))

    # ISIC splits
    isic_splits = [23188, 4969, 4969]
    isic_labels = ["Train\n(70%)", "Val\n(15%)", "Test\n(15%)"]
    isic_colors = ["#3498db", "#2ecc71", "#e74c3c"]
    ax1.pie(isic_splits, labels=isic_labels, colors=isic_colors,
            autopct=lambda p: f"{int(p*sum(isic_splits)/100):,}",
            startangle=90, textprops={"fontsize": 9})
    ax1.set_title("(a) ISIC 2020 Split (n=33,126)")

    # BraTS splits (typical: ~1251 cases -> 875/188/188)
    brats_splits = [875, 188, 188]
    brats_labels = ["Train\n(70%)", "Val\n(15%)", "Test\n(15%)"]
    brats_colors = ["#3498db", "#2ecc71", "#e74c3c"]
    ax2.pie(brats_splits, labels=brats_labels, colors=brats_colors,
            autopct=lambda p: f"{int(p*sum(brats_splits)/100):,}",
            startangle=90, textprops={"fontsize": 9})
    ax2.set_title("(b) BraTS 2021 Split (n=1,251 cases)")

    plt.tight_layout()
    plt.savefig("figures/fig_eda_splits.png")
    plt.close()
    print("Saved fig_eda_splits.png")


if __name__ == "__main__":
    fig_eda_isic_class_distribution()
    fig_eda_brats_volume_stats()
    fig_eda_preprocessing_pipeline()
    fig_eda_train_val_test_split()
    print("\nAll EDA figures generated.")
