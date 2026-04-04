"""MedCompress domain glossary for consistent terminology.

Maps canonical terms to their definitions and acceptable synonyms.
Used by the stop-slop synonym cycling checker and by the pipeline
to enforce consistent terminology throughout the paper.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class GlossaryEntry:
    """A domain term with its canonical form and definition."""

    canonical: str
    definition: str
    abbreviation: str
    synonyms: tuple[str, ...]  # acceptable alternatives
    avoid: tuple[str, ...]  # terms to avoid in favor of canonical


GLOSSARY: dict[str, GlossaryEntry] = {
    "qat": GlossaryEntry(
        canonical="quantization-aware training",
        definition=(
            "A compression technique that inserts fake-quantization nodes "
            "during training, enabling the model to learn to compensate for "
            "quantization error before deployment."
        ),
        abbreviation="QAT",
        synonyms=("quantization-aware fine-tuning",),
        avoid=("quantization training", "quant training"),
    ),
    "kd": GlossaryEntry(
        canonical="knowledge distillation",
        definition=(
            "A compression technique where a smaller student model is trained "
            "to mimic the soft probability outputs of a larger teacher model, "
            "transferring learned representations."
        ),
        abbreviation="KD",
        synonyms=("model distillation",),
        avoid=("distilling", "knowledge transfer training"),
    ),
    "ptq": GlossaryEntry(
        canonical="post-training quantization",
        definition=(
            "Quantization applied after training without further fine-tuning. "
            "Requires a representative calibration dataset for INT8."
        ),
        abbreviation="PTQ",
        synonyms=("static quantization",),
        avoid=("offline quantization",),
    ),
    "tflite": GlossaryEntry(
        canonical="TensorFlow Lite",
        definition=(
            "A lightweight inference framework optimized for mobile and "
            "embedded devices. Supports INT8, FP16, and FP32 models."
        ),
        abbreviation="TFLite",
        synonyms=("TF Lite",),
        avoid=("tensorflow mobile", "tf-lite"),
    ),
    "onnx": GlossaryEntry(
        canonical="Open Neural Network Exchange",
        definition=(
            "An open format for representing machine learning models, "
            "enabling interoperability between frameworks."
        ),
        abbreviation="ONNX",
        synonyms=(),
        avoid=("open neural network format",),
    ),
    "dice": GlossaryEntry(
        canonical="Dice coefficient",
        definition=(
            "A spatial overlap metric for segmentation evaluation, computed as "
            "2|A∩B|/(|A|+|B|). Ranges from 0 (no overlap) to 1 (perfect)."
        ),
        abbreviation="DSC",
        synonyms=("Dice similarity coefficient", "Sørensen-Dice coefficient"),
        avoid=("dice score", "dice metric"),
    ),
    "auc": GlossaryEntry(
        canonical="area under the ROC curve",
        definition=(
            "A threshold-independent metric for binary classification "
            "performance. 0.5 = random, 1.0 = perfect discrimination."
        ),
        abbreviation="AUC",
        synonyms=("ROC-AUC", "AUROC"),
        avoid=("area under curve",),
    ),
    "isic": GlossaryEntry(
        canonical="International Skin Imaging Collaboration",
        definition=(
            "A consortium and challenge series for dermoscopic image analysis. "
            "ISIC 2020 provides 33,126 images for binary melanoma detection."
        ),
        abbreviation="ISIC",
        synonyms=(),
        avoid=("skin image dataset",),
    ),
    "brats": GlossaryEntry(
        canonical="Brain Tumor Segmentation Challenge",
        definition=(
            "A benchmark for multi-class brain tumor segmentation using "
            "multi-modal MRI (T1, T1ce, T2, FLAIR)."
        ),
        abbreviation="BraTS",
        synonyms=(),
        avoid=("brain tumor dataset",),
    ),
    "2.5d": GlossaryEntry(
        canonical="2.5D approach",
        definition=(
            "An intermediate strategy between 2D and 3D processing that "
            "stacks adjacent slices across modalities as multi-channel input, "
            "preserving some volumetric context without full 3D convolutions."
        ),
        abbreviation="2.5D",
        synonyms=("pseudo-3D", "multi-slice 2D"),
        avoid=("two and a half D", "2.5 dimensional"),
    ),
    "wasm": GlossaryEntry(
        canonical="WebAssembly",
        definition=(
            "A binary instruction format for stack-based virtual machines, "
            "enabling near-native performance in web browsers."
        ),
        abbreviation="WASM",
        synonyms=("Wasm",),
        avoid=("web assembly",),
    ),
    "msa": GlossaryEntry(
        canonical="Memory Sparse Attention",
        definition=(
            "An end-to-end trainable sparse attention framework that uses "
            "KV cache chunk-mean pooling and top-k document routing to "
            "extend context windows while reducing memory and compute. "
            "Adapted for Vision Transformer compression in medical imaging."
        ),
        abbreviation="MSA",
        synonyms=("sparse attention compression",),
        avoid=("memory attention", "sparse memory"),
    ),
    "kv_pooling": GlossaryEntry(
        canonical="KV cache pooling",
        definition=(
            "Chunk-mean averaging of key and value representations along "
            "the sequence dimension, reducing the KV cache size by the "
            "pooling kernel factor while preserving gradient flow."
        ),
        abbreviation="KV pooling",
        synonyms=("key-value pooling", "KV compression"),
        avoid=("cache compression", "attention compression"),
    ),
    "topk_routing": GlossaryEntry(
        canonical="top-k sparse routing",
        definition=(
            "A selection mechanism that computes routing scores between "
            "queries and pooled key chunks, then attends only to the k "
            "most relevant chunks, reducing attention complexity from "
            "O(n^2) to O(n*k)."
        ),
        abbreviation="top-k routing",
        synonyms=("sparse routing", "top-k selection"),
        avoid=("attention pruning", "token selection"),
    ),
    "infonce": GlossaryEntry(
        canonical="InfoNCE loss",
        definition=(
            "A contrastive learning objective that trains routing "
            "projections to discriminate relevant from irrelevant "
            "chunks, using normalized temperature-scaled cross-entropy."
        ),
        abbreviation="InfoNCE",
        synonyms=("noise-contrastive estimation",),
        avoid=("contrastive loss", "NCE"),
    ),
}


def get_term(key: str) -> GlossaryEntry:
    """Look up a glossary entry by key."""
    entry = GLOSSARY.get(key.lower())
    if entry is None:
        raise KeyError(f"Unknown glossary term: {key}")
    return entry


def get_all_canonical_terms() -> list[str]:
    """Return all canonical term names."""
    return [entry.canonical for entry in GLOSSARY.values()]


def get_abbreviation_map() -> dict[str, str]:
    """Return mapping of abbreviation -> canonical term."""
    return {
        entry.abbreviation: entry.canonical
        for entry in GLOSSARY.values()
    }
