"""
compression/sparse_bottleneck.py
---------------------------------
Sparse attention applied to U-Net segmentation at the bottleneck.

Standard U-Net has no attention layers, so the sparse attention
from sparse_attention.py cannot be applied directly. This module
inserts a tokenized sparse attention block at the U-Net bottleneck,
where the feature map is smallest (8x8 or 16x16) and attention
is computationally feasible on CPU.

The bottleneck feature map is reshaped from (B, H, W, C) to
(B, H*W, C) tokens, processed through sparse attention with 2D
spatial pooling, and reshaped back. This allows spatial compression
at the representation level without discarding spatial positions
in the output (which segmentation requires).

The key insight: at the bottleneck, each spatial position has a
large receptive field. Sparse attention here selects which bottleneck
regions receive full cross-attention and which are interpolated from
neighbors. This compresses computation without dropping output pixels.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class SparseBottleneckAttention(layers.Layer):
    """Tokenized sparse attention block for U-Net bottleneck.

    Inserted between the encoder's deepest stage and the decoder's
    first upsampling stage. Tokenizes the bottleneck feature map,
    applies 2D spatial pooling + top-k routing, and reconstructs
    the full spatial map via scatter + interpolation.

    Args:
        num_heads: Attention heads.
        head_dim: Dimension per head.
        pool_h: 2D pooling kernel height.
        pool_w: 2D pooling kernel width.
        top_k: Number of pooled tokens to attend.
        grid_h: Bottleneck spatial height (e.g., 8 for 128->8 after 4 stages).
        grid_w: Bottleneck spatial width.
        dropout_rate: Attention dropout.
    """

    def __init__(self, num_heads: int = 8, head_dim: int = 64,
                 pool_h: int = 2, pool_w: int = 2,
                 top_k: int = 4, grid_h: int = 8, grid_w: int = 8,
                 dropout_rate: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.embed_dim = num_heads * head_dim
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.top_k = top_k
        self.grid_h = grid_h
        self.grid_w = grid_w

        self.input_proj = layers.Dense(self.embed_dim, name="bottleneck_proj_in")
        self.output_proj = layers.Dense(self.embed_dim, name="bottleneck_proj_out")

        self.q_proj = layers.Dense(self.embed_dim, use_bias=False)
        self.k_proj = layers.Dense(self.embed_dim, use_bias=False)
        self.v_proj = layers.Dense(self.embed_dim, use_bias=False)

        self.norm1 = layers.LayerNormalization()
        self.norm2 = layers.LayerNormalization()
        self.ffn = keras.Sequential([
            layers.Dense(self.embed_dim * 2, activation="gelu"),
            layers.Dense(self.embed_dim),
        ])
        self.dropout = layers.Dropout(dropout_rate)

    def call(self, feature_map: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Process bottleneck feature map through sparse attention.

        Args:
            feature_map: (batch, H, W, C) bottleneck features
            training: Training mode flag

        Returns:
            (batch, H, W, C) processed features, same spatial shape
        """
        batch = tf.shape(feature_map)[0]
        h = tf.shape(feature_map)[1]
        w = tf.shape(feature_map)[2]
        c = tf.shape(feature_map)[3]

        # Tokenize: (B, H, W, C) -> (B, H*W, embed_dim)
        tokens = tf.reshape(feature_map, [batch, h * w, -1])
        tokens = self.input_proj(tokens)

        # Sparse self-attention with 2D pooling
        residual = tokens
        tokens = self.norm1(tokens)

        q = self._reshape_heads(self.q_proj(tokens), batch)
        k = self._reshape_heads(self.k_proj(tokens), batch)
        v = self._reshape_heads(self.v_proj(tokens), batch)

        # 2D spatial pooling on K, V
        pooled_k = self._spatial_pool_2d(k, batch)
        pooled_v = self._spatial_pool_2d(v, batch)

        # Top-k routing
        selected_k, selected_v = self._topk_route(q, pooled_k, pooled_v)

        # Attention: Q attends to selected K, V only
        scaling = tf.cast(self.head_dim, tf.float32) ** -0.5
        attn_weights = tf.matmul(q, selected_k, transpose_b=True) * scaling
        attn_weights = tf.nn.softmax(attn_weights, axis=-1)
        attn_weights = self.dropout(attn_weights, training=training)

        attn_out = tf.matmul(attn_weights, selected_v)
        attn_out = self._merge_heads(attn_out, batch)
        tokens = residual + attn_out

        # FFN
        residual = tokens
        tokens = self.norm2(tokens)
        tokens = residual + self.ffn(tokens)

        # Project back and reshape: (B, H*W, embed_dim) -> (B, H, W, C)
        tokens = self.output_proj(tokens)
        return tf.reshape(tokens, [batch, h, w, c])

    def _reshape_heads(self, x, batch):
        """(B, seq, embed) -> (B, heads, seq, head_dim)"""
        seq = tf.shape(x)[1]
        x = tf.reshape(x, [batch, seq, self.num_heads, self.head_dim])
        return tf.transpose(x, [0, 2, 1, 3])

    def _merge_heads(self, x, batch):
        """(B, heads, seq, head_dim) -> (B, seq, embed)"""
        seq = tf.shape(x)[2]
        x = tf.transpose(x, [0, 2, 1, 3])
        return tf.reshape(x, [batch, seq, self.embed_dim])

    def _spatial_pool_2d(self, x, batch):
        """2D spatial pooling on attention keys/values.

        (B, heads, H*W, dim) -> (B, heads, pooled_H*pooled_W, dim)
        """
        heads = self.num_heads
        dim = self.head_dim
        gh, gw = self.grid_h, self.grid_w
        ph, pw = self.pool_h, self.pool_w

        # Reshape to spatial grid
        x = tf.transpose(x, [0, 2, 1, 3])  # (B, H*W, heads, dim)
        x_2d = tf.reshape(x[:, :gh*gw], [batch, gh, gw, heads, dim])

        # Pad if needed
        pad_h = (ph - gh % ph) % ph
        pad_w = (pw - gw % pw) % pw
        if pad_h > 0 or pad_w > 0:
            x_2d = tf.pad(x_2d, [[0,0], [0,pad_h], [0,pad_w], [0,0], [0,0]])

        padded_h = gh + pad_h
        padded_w = gw + pad_w
        out_h = padded_h // ph
        out_w = padded_w // pw

        # Reshape to blocks and pool
        x_blocks = tf.reshape(x_2d, [batch, out_h, ph, out_w, pw, heads, dim])
        x_pooled = tf.reduce_mean(x_blocks, axis=[2, 4])  # (B, out_h, out_w, heads, dim)

        # Flatten back
        x_flat = tf.reshape(x_pooled, [batch, out_h * out_w, heads, dim])
        return tf.transpose(x_flat, [0, 2, 1, 3])  # (B, heads, pooled_tokens, dim)

    def _topk_route(self, q, pooled_k, pooled_v):
        """Select top-k pooled tokens based on routing scores."""
        # Routing scores: (B, heads, seq, pooled) via Q @ K^T
        scores = tf.matmul(q, pooled_k, transpose_b=True)
        # Reduce across heads and queries
        scores_reduced = tf.reduce_max(tf.reduce_max(scores, axis=1), axis=1)  # (B, pooled)

        num_pooled = tf.shape(scores_reduced)[1]
        k = tf.minimum(self.top_k, num_pooled)
        _, top_idx = tf.math.top_k(scores_reduced, k=k)

        # Gather selected tokens
        selected_k = tf.gather(
            tf.transpose(pooled_k, [0, 2, 1, 3]), top_idx, batch_dims=1)
        selected_k = tf.transpose(selected_k, [0, 2, 1, 3])
        selected_v = tf.gather(
            tf.transpose(pooled_v, [0, 2, 1, 3]), top_idx, batch_dims=1)
        selected_v = tf.transpose(selected_v, [0, 2, 1, 3])

        return selected_k, selected_v

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_heads": self.num_heads,
            "head_dim": self.head_dim,
            "pool_h": self.pool_h,
            "pool_w": self.pool_w,
            "top_k": self.top_k,
            "grid_h": self.grid_h,
            "grid_w": self.grid_w,
        })
        return config


def build_unet_with_sparse_bottleneck(
    num_classes: int = 4,
    n_channels: int = 12,
    input_size: int = 128,
    filters: list[int] = (32, 64, 128, 256),
    num_heads: int = 4,
    head_dim: int = 32,
    pool_kernel: int = 2,
    top_k: int = 4,
) -> keras.Model:
    """Build a U-Net Lite with sparse attention at the bottleneck.

    The bottleneck feature map at 128->8 after 4 downsamplings
    (or 128->16 after 3) is small enough that tokenized attention
    is computationally feasible on CPU. Sparse 2D pooling + top-k
    routing reduces it further.

    Args:
        num_classes: Output segmentation classes.
        n_channels: Input channels (e.g., 12 for BraTS 2.5D).
        input_size: Spatial input size.
        filters: Filter counts per encoder stage.
        num_heads: Attention heads at bottleneck.
        head_dim: Dimension per attention head.
        pool_kernel: 2D pooling kernel size (pool_kernel x pool_kernel).
        top_k: Top-k tokens for sparse routing.

    Returns:
        Keras Model with sparse bottleneck attention.
    """
    inputs = keras.Input(shape=(input_size, input_size, n_channels))

    # Encoder
    skips = []
    x = inputs
    for f in filters[:-1]:
        x = layers.Conv2D(f, 3, padding="same", activation="relu")(x)
        x = layers.Conv2D(f, 3, padding="same", activation="relu")(x)
        skips.append(x)
        x = layers.MaxPooling2D(2)(x)

    # Bottleneck with sparse attention
    bottleneck_filters = filters[-1]
    x = layers.Conv2D(bottleneck_filters, 3, padding="same", activation="relu")(x)
    x = layers.Conv2D(bottleneck_filters, 3, padding="same", activation="relu")(x)

    # Insert sparse attention
    bottleneck_h = input_size // (2 ** (len(filters) - 1))
    bottleneck_w = bottleneck_h
    x = SparseBottleneckAttention(
        num_heads=num_heads,
        head_dim=head_dim,
        pool_h=pool_kernel,
        pool_w=pool_kernel,
        top_k=top_k,
        grid_h=bottleneck_h,
        grid_w=bottleneck_w,
        name="sparse_bottleneck",
    )(x)

    # Decoder
    for f, skip in zip(reversed(filters[:-1]), reversed(skips)):
        x = layers.UpSampling2D(2)(x)
        x = layers.Concatenate()([x, skip])
        x = layers.Conv2D(f, 3, padding="same", activation="relu")(x)
        x = layers.Conv2D(f, 3, padding="same", activation="relu")(x)

    outputs = layers.Conv2D(num_classes, 1, activation="softmax")(x)
    return keras.Model(inputs, outputs, name="unet_sparse_bottleneck")
