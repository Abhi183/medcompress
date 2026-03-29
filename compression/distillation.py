"""
compression/distillation.py
-----------------------------
Knowledge Distillation (KD) training loop.
Supports response-based (soft-label) distillation with optional
feature-level distillation via intermediate layer matching.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras


# =========================================================================== #
#  Distillation Loss                                                           #
# =========================================================================== #

def kd_loss(y_true, student_logits, teacher_logits,
            temperature: float, alpha: float, task: str = "classification"):
    """
    Combined distillation + hard-label loss.

    Loss = alpha * KL(teacher_soft || student_soft) + (1-alpha) * CE(y_true, student)

    Args:
        y_true: Ground truth labels.
        student_logits: Raw (pre-softmax) output of student model.
        teacher_logits: Raw (pre-softmax) output of teacher model.
        temperature: Softens distributions; higher T → softer targets.
        alpha: Weight for distillation loss (0=hard labels only, 1=soft only).
        task: 'classification' or 'segmentation'.

    Returns:
        Scalar loss tensor.
    """
    T = temperature

    if task == "classification":
        # Soft labels (temperature-scaled probabilities)
        teacher_soft = tf.nn.sigmoid(teacher_logits / T)
        student_soft = tf.nn.sigmoid(student_logits / T)

        # KL divergence for binary case
        eps = 1e-7
        kl = teacher_soft * tf.math.log((teacher_soft + eps) / (student_soft + eps))
        kl += (1 - teacher_soft) * tf.math.log(
            (1 - teacher_soft + eps) / (1 - student_soft + eps)
        )
        distill_loss = tf.reduce_mean(kl) * (T ** 2)

        # Hard label loss
        hard_loss = keras.losses.binary_crossentropy(
            y_true, tf.nn.sigmoid(student_logits)
        )
        hard_loss = tf.reduce_mean(hard_loss)

    else:  # segmentation
        # Softmax-based KL divergence over num_classes
        teacher_soft = tf.nn.softmax(teacher_logits / T, axis=-1)
        student_soft = tf.nn.softmax(student_logits / T, axis=-1)

        kl = tf.keras.losses.KLDivergence()(teacher_soft, student_soft)
        distill_loss = kl * (T ** 2)

        # Hard label loss (Dice + CE for segmentation)
        hard_loss = keras.losses.categorical_crossentropy(
            y_true, tf.nn.softmax(student_logits, axis=-1)
        )
        hard_loss = tf.reduce_mean(hard_loss)

    return alpha * distill_loss + (1.0 - alpha) * hard_loss


# =========================================================================== #
#  Feature Distillation                                                        #
# =========================================================================== #

class FeatureDistillationLoss:
    """
    MSE loss between teacher and student intermediate feature maps.
    A 1x1 conv adapter is used when channel dimensions differ.
    """

    def __init__(self, teacher_shapes: list, student_shapes: list):
        self.adapters = []
        for t_shape, s_shape in zip(teacher_shapes, student_shapes):
            if t_shape[-1] != s_shape[-1]:
                # Learnable projection to match channel dimensions
                adapter = keras.layers.Conv2D(
                    t_shape[-1], kernel_size=1, use_bias=False
                )
                self.adapters.append(adapter)
            else:
                self.adapters.append(None)

    def __call__(self, teacher_feats: list, student_feats: list) -> tf.Tensor:
        total = tf.constant(0.0)
        for i, (tf_feat, sf_feat) in enumerate(zip(teacher_feats, student_feats)):
            if self.adapters[i] is not None:
                sf_feat = self.adapters[i](sf_feat)
            # Resize student features to teacher spatial resolution if needed
            if tf_feat.shape[1:3] != sf_feat.shape[1:3]:
                sf_feat = tf.image.resize(sf_feat, tf_feat.shape[1:3])
            mse = tf.reduce_mean(tf.square(tf_feat - sf_feat))
            total = total + mse
        return total / len(teacher_feats)


# =========================================================================== #
#  Distillation Trainer                                                        #
# =========================================================================== #

class DistillationTrainer:
    """
    Manages the KD training loop with optional feature distillation.

    Usage:
        trainer = DistillationTrainer(teacher, student, config)
        history = trainer.train(train_ds, val_ds)
    """

    def __init__(self, teacher: keras.Model, student: keras.Model, config: dict):
        self.teacher = teacher
        self.student = student
        self.config = config

        distill_cfg = config["distillation"]
        self.temperature = distill_cfg["temperature"]
        self.alpha = distill_cfg["alpha"]
        self.use_feature = distill_cfg.get("feature_distillation", False)
        self.task = config.get("task", "classification")

        train_cfg = config["training"]
        self.lr = train_cfg["learning_rate"]
        self.epochs = train_cfg["epochs"]

        self.optimizer = keras.optimizers.Adam(self.lr)

        # Build feature extraction models if needed
        if self.use_feature:
            feat_layers = distill_cfg.get("feature_layers", [])
            self._build_feature_models(feat_layers)
        else:
            self.teacher_feat_model = None
            self.student_feat_model = None
            self.feat_loss_fn = None

        # Metrics
        self.train_loss_tracker = keras.metrics.Mean(name="train_loss")
        self.val_loss_tracker = keras.metrics.Mean(name="val_loss")

        if self.task == "classification":
            self.train_auc = keras.metrics.AUC(name="train_auc")
            self.val_auc = keras.metrics.AUC(name="val_auc")
        else:
            from models.baseline import DiceCoefficient
            num_classes = student.output_shape[-1]
            self.train_dice = DiceCoefficient(num_classes, name="train_dice")
            self.val_dice = DiceCoefficient(num_classes, name="val_dice")

        self.teacher.trainable = False

        self.history = {"train_loss": [], "val_loss": []}

    def _build_feature_models(self, layer_names: list):
        """Create sub-models that output intermediate feature maps."""
        def _get_feature_model(model, names):
            outputs = []
            for name in names:
                if name in [l.name for l in model.layers]:
                    outputs.append(model.get_layer(name).output)
            if not outputs:
                print("[KD] Warning: No feature layers found. Disabling feature distillation.")
                return None
            return keras.Model(inputs=model.input, outputs=outputs)

        self.teacher_feat_model = _get_feature_model(self.teacher, layer_names)
        self.student_feat_model = _get_feature_model(self.student, layer_names)

        if self.teacher_feat_model and self.student_feat_model:
            t_shapes = [o.shape for o in self.teacher_feat_model.outputs]
            s_shapes = [o.shape for o in self.student_feat_model.outputs]
            self.feat_loss_fn = FeatureDistillationLoss(t_shapes, s_shapes)
        else:
            self.teacher_feat_model = None
            self.feat_loss_fn = None

    # ------------------------------------------------------------------ #
    #  Training step                                                       #
    # ------------------------------------------------------------------ #

    @tf.function
    def _train_step(self, images, labels):
        # Teacher forward pass (no gradient)
        teacher_out = self.teacher(images, training=False)

        with tf.GradientTape() as tape:
            student_out = self.student(images, training=True)

            # Response distillation loss
            loss = kd_loss(
                labels, student_out, teacher_out,
                self.temperature, self.alpha, self.task
            )

            # Optional feature distillation
            if self.use_feature and self.feat_loss_fn is not None:
                t_feats = self.teacher_feat_model(images, training=False)
                s_feats = self.student_feat_model(images, training=True)
                if not isinstance(t_feats, list):
                    t_feats, s_feats = [t_feats], [s_feats]
                feat_loss = self.feat_loss_fn(t_feats, s_feats)
                loss = loss + 0.1 * feat_loss  # feature loss weight

        grads = tape.gradient(loss, self.student.trainable_variables)
        self.optimizer.apply_gradients(
            zip(grads, self.student.trainable_variables)
        )
        return loss, student_out

    @tf.function
    def _val_step(self, images, labels):
        student_out = self.student(images, training=False)
        loss = kd_loss(
            labels, student_out,
            self.teacher(images, training=False),
            self.temperature, self.alpha, self.task
        )
        return loss, student_out

    # ------------------------------------------------------------------ #
    #  Full training loop                                                  #
    # ------------------------------------------------------------------ #

    def train(self, train_ds: tf.data.Dataset,
              val_ds: tf.data.Dataset,
              output_dir: str = "outputs/kd/") -> dict:
        os.makedirs(output_dir, exist_ok=True)
        best_val = float("inf")
        best_path = os.path.join(output_dir, "student_best.keras")

        for epoch in range(self.epochs):
            # ── Train ── #
            self.train_loss_tracker.reset_state()
            for images, labels in train_ds:
                loss, preds = self._train_step(images, labels)
                self.train_loss_tracker.update_state(loss)
                if self.task == "classification":
                    self.train_auc.update_state(labels, preds)
                else:
                    self.train_dice.update_state(labels, preds)

            # ── Validate ── #
            self.val_loss_tracker.reset_state()
            for images, labels in val_ds:
                loss, preds = self._val_step(images, labels)
                self.val_loss_tracker.update_state(loss)
                if self.task == "classification":
                    self.val_auc.update_state(labels, preds)
                else:
                    self.val_dice.update_state(labels, preds)

            train_loss = self.train_loss_tracker.result().numpy()
            val_loss = self.val_loss_tracker.result().numpy()

            if self.task == "classification":
                metric_str = (
                    f"train_auc={self.train_auc.result().numpy():.4f}  "
                    f"val_auc={self.val_auc.result().numpy():.4f}"
                )
                self.train_auc.reset_state()
                self.val_auc.reset_state()
            else:
                metric_str = (
                    f"train_dice={self.train_dice.result().numpy():.4f}  "
                    f"val_dice={self.val_dice.result().numpy():.4f}"
                )
                self.train_dice.reset_state()
                self.val_dice.reset_state()

            print(
                f"Epoch {epoch+1:03d}/{self.epochs}  "
                f"loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                f"{metric_str}"
            )

            # Save best
            if val_loss < best_val:
                best_val = val_loss
                self.student.save(best_path)
                print(f"  ✓ Saved best student → {best_path}")

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)

        print("[KD] Training complete.")
        return self.history
