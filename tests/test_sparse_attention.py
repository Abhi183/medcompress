"""Tests for MSA-inspired sparse attention compression module.

Covers KV cache pooling, top-k sparse routing, decoupled router,
InfoNCE loss, the composed SparseAttentionBlock, compression stats,
and citation constants.
"""

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow", reason="TensorFlow required for sparse attention tests")

from compression.sparse_attention import (
    MSA_CITATION_APA,
    MSA_CITATION_BIBTEX,
    MSA_CITATION_IEEE,
    DecoupledRouter,
    KVCachePooling,
    SparseAttentionBlock,
    TopKSparseRouter,
    get_compression_stats,
    info_nce_loss,
)


# -----------------------------------------------------------------------
# KV Cache Pooling Tests
# -----------------------------------------------------------------------

class TestKVCachePooling:
    """Tests for chunk-mean pooling of key/value tensors."""

    def test_basic_pooling_reduces_sequence(self):
        pooling = KVCachePooling(kernel_size=4)
        keys = tf.random.normal([2, 16, 8, 64])   # batch=2, seq=16, heads=8, dim=64
        values = tf.random.normal([2, 16, 8, 64])
        pk, pv = pooling(keys, values)
        assert pk.shape == (2, 4, 8, 64)  # 16 / 4 = 4 chunks
        assert pv.shape == (2, 4, 8, 64)

    def test_pooling_kernel_1_is_identity(self):
        pooling = KVCachePooling(kernel_size=1)
        keys = tf.random.normal([1, 8, 4, 32])
        values = tf.random.normal([1, 8, 4, 32])
        pk, pv = pooling(keys, values)
        assert pk.shape == keys.shape
        np.testing.assert_allclose(pk.numpy(), keys.numpy(), atol=1e-5)

    def test_pooling_with_non_divisible_length(self):
        """Sequence length not divisible by kernel_size should pad."""
        pooling = KVCachePooling(kernel_size=4)
        keys = tf.random.normal([1, 7, 2, 16])  # 7 not divisible by 4
        values = tf.random.normal([1, 7, 2, 16])
        pk, pv = pooling(keys, values)
        # Padded to 8, then 8/4 = 2 chunks
        assert pk.shape == (1, 2, 2, 16)

    def test_mean_pooling_correctness(self):
        """Verify chunk-mean is computed correctly."""
        pooling = KVCachePooling(kernel_size=2, pool_method="mean")
        # Known values: [[1, 2], [3, 4]] -> mean of chunks -> [1.5, 3.5]
        keys = tf.constant([[[[1.0], [2.0]], [[3.0], [4.0]]]])  # (1, 2, 2, 1)
        values = tf.constant([[[[10.0], [20.0]], [[30.0], [40.0]]]])
        pk, pv = pooling(keys, values)
        assert pk.shape == (1, 1, 2, 1)  # 2/2 = 1 chunk
        np.testing.assert_allclose(pk[0, 0, 0, 0].numpy(), 2.0, atol=1e-5)
        np.testing.assert_allclose(pk[0, 0, 1, 0].numpy(), 3.5, atol=1e-5)

    def test_max_pooling(self):
        pooling = KVCachePooling(kernel_size=2, pool_method="max")
        keys = tf.constant([[[[1.0], [3.0]], [[2.0], [4.0]]]])
        values = tf.constant([[[[1.0], [3.0]], [[2.0], [4.0]]]])
        pk, _ = pooling(keys, values)
        np.testing.assert_allclose(pk[0, 0, 0, 0].numpy(), 3.0, atol=1e-5)
        np.testing.assert_allclose(pk[0, 0, 1, 0].numpy(), 4.0, atol=1e-5)

    def test_invalid_kernel_size(self):
        with pytest.raises(ValueError, match="kernel_size must be >= 1"):
            KVCachePooling(kernel_size=0)

    def test_invalid_pool_method(self):
        pooling = KVCachePooling(kernel_size=2, pool_method="invalid")
        keys = tf.random.normal([1, 4, 2, 8])
        values = tf.random.normal([1, 4, 2, 8])
        with pytest.raises(ValueError, match="Unknown pool_method"):
            pooling(keys, values)

    def test_3d_input(self):
        """Works with (batch, seq, dim) tensors too."""
        pooling = KVCachePooling(kernel_size=4)
        keys = tf.random.normal([2, 16, 256])
        values = tf.random.normal([2, 16, 256])
        pk, pv = pooling(keys, values)
        assert pk.shape == (2, 4, 256)

    def test_serialization(self):
        pooling = KVCachePooling(kernel_size=8, pool_method="max")
        config = pooling.get_config()
        assert config["kernel_size"] == 8
        assert config["pool_method"] == "max"


# -----------------------------------------------------------------------
# Top-k Sparse Router Tests
# -----------------------------------------------------------------------

class TestTopKSparseRouter:
    """Tests for top-k spatial region selection."""

    def test_basic_routing(self):
        router = TopKSparseRouter(top_k=4)
        queries = tf.random.normal([2, 8, 16, 64])       # batch=2, heads=8, q=16, dim=64
        pooled_keys = tf.random.normal([2, 10, 8, 64])    # batch=2, chunks=10, heads=8
        indices, scores = router(queries, pooled_keys)
        assert indices.shape == (2, 4)
        assert scores.shape == (2, 4)

    def test_top_k_greater_than_chunks(self):
        """When top_k > num_chunks, should return all chunks."""
        router = TopKSparseRouter(top_k=20)
        queries = tf.random.normal([1, 4, 8, 32])
        pooled_keys = tf.random.normal([1, 5, 4, 32])
        indices, scores = router(queries, pooled_keys)
        assert indices.shape[1] == 5  # min(20, 5) = 5

    def test_head_reduce_mean(self):
        router = TopKSparseRouter(top_k=2, head_reduce="mean")
        queries = tf.random.normal([1, 4, 8, 32])
        pooled_keys = tf.random.normal([1, 6, 4, 32])
        indices, scores = router(queries, pooled_keys)
        assert indices.shape == (1, 2)

    def test_query_reduce_last(self):
        router = TopKSparseRouter(top_k=3, query_reduce="last")
        queries = tf.random.normal([1, 4, 8, 32])
        pooled_keys = tf.random.normal([1, 6, 4, 32])
        indices, scores = router(queries, pooled_keys)
        assert indices.shape == (1, 3)

    def test_query_reduce_mean(self):
        router = TopKSparseRouter(top_k=3, query_reduce="mean")
        queries = tf.random.normal([1, 4, 8, 32])
        pooled_keys = tf.random.normal([1, 6, 4, 32])
        indices, scores = router(queries, pooled_keys)
        assert indices.shape == (1, 3)

    def test_invalid_top_k(self):
        with pytest.raises(ValueError, match="top_k must be >= 1"):
            TopKSparseRouter(top_k=0)

    def test_scores_are_ordered(self):
        """Top-k scores should be in descending order."""
        router = TopKSparseRouter(top_k=3)
        queries = tf.random.normal([1, 4, 16, 64])
        pooled_keys = tf.random.normal([1, 10, 4, 64])
        _, scores = router(queries, pooled_keys)
        scores_np = scores.numpy()[0]
        assert all(scores_np[i] >= scores_np[i + 1]
                   for i in range(len(scores_np) - 1))

    def test_serialization(self):
        router = TopKSparseRouter(top_k=8, head_reduce="mean",
                                  query_reduce="last")
        config = router.get_config()
        assert config["top_k"] == 8
        assert config["head_reduce"] == "mean"


# -----------------------------------------------------------------------
# Decoupled Router Tests
# -----------------------------------------------------------------------

class TestDecoupledRouter:
    """Tests for separate routing projections."""

    def test_basic_routing_projection(self):
        router = DecoupledRouter(hidden_dim=256, routing_dim=128)
        q_hidden = tf.random.normal([2, 16, 256])
        k_hidden = tf.random.normal([2, 32, 256])
        rq, rk = router(q_hidden, k_hidden)
        assert rq.shape == (2, 16, 128)
        assert rk.shape == (2, 32, 128)

    def test_outputs_are_normalized(self):
        router = DecoupledRouter(hidden_dim=64, routing_dim=32)
        q = tf.random.normal([1, 8, 64])
        k = tf.random.normal([1, 8, 64])
        rq, rk = router(q, k)
        norms_q = tf.norm(rq, axis=-1).numpy()
        norms_k = tf.norm(rk, axis=-1).numpy()
        np.testing.assert_allclose(norms_q, 1.0, atol=1e-5)
        np.testing.assert_allclose(norms_k, 1.0, atol=1e-5)

    def test_serialization(self):
        router = DecoupledRouter(hidden_dim=512, routing_dim=256)
        config = router.get_config()
        assert config["hidden_dim"] == 512
        assert config["routing_dim"] == 256


# -----------------------------------------------------------------------
# InfoNCE Loss Tests
# -----------------------------------------------------------------------

class TestInfoNCELoss:
    """Tests for the contrastive auxiliary loss."""

    def test_loss_is_non_negative(self):
        rq = tf.random.normal([4, 64])
        rk = tf.random.normal([4, 10, 64])
        labels = tf.one_hot(tf.random.uniform([4], 0, 10, dtype=tf.int32), 10)
        loss = info_nce_loss(rq, rk, labels)
        assert loss.numpy() >= 0.0

    def test_perfect_alignment_low_loss(self):
        """When query perfectly matches the correct key, loss should be low."""
        rq = tf.constant([[1.0, 0.0, 0.0]])  # batch=1, dim=3
        rk = tf.constant([[[1.0, 0.0, 0.0],    # chunk 0 = match
                           [0.0, 1.0, 0.0],     # chunk 1 = orthogonal
                           [0.0, 0.0, 1.0]]])   # chunk 2 = orthogonal
        labels = tf.constant([[1.0, 0.0, 0.0]])  # chunk 0 is relevant
        loss = info_nce_loss(rq, rk, labels, temperature=0.07)
        assert loss.numpy() < 2.0  # should be relatively low

    def test_temperature_affects_loss(self):
        rq = tf.random.normal([2, 32])
        rk = tf.random.normal([2, 8, 32])
        labels = tf.one_hot([0, 3], 8)
        loss_low_t = info_nce_loss(rq, rk, labels, temperature=0.01)
        loss_high_t = info_nce_loss(rq, rk, labels, temperature=1.0)
        # Lower temperature makes loss more extreme
        assert loss_low_t.numpy() != loss_high_t.numpy()


# -----------------------------------------------------------------------
# SparseAttentionBlock Tests
# -----------------------------------------------------------------------

class TestSparseAttentionBlock:
    """Tests for the full composable sparse attention block."""

    def test_forward_pass_shape(self):
        block = SparseAttentionBlock(
            num_heads=4, head_dim=32, kernel_size=4, top_k=4,
        )
        x = tf.random.normal([2, 16, 128])  # batch=2, seq=16, dim=4*32=128
        out = block(x, training=False)
        assert out.shape == (2, 16, 128)

    def test_forward_pass_training(self):
        block = SparseAttentionBlock(
            num_heads=2, head_dim=16, kernel_size=2, top_k=2,
        )
        x = tf.random.normal([1, 8, 32])  # dim = 2*16 = 32
        out = block(x, training=True)
        assert out.shape == (1, 8, 32)

    def test_with_decoupled_router(self):
        block = SparseAttentionBlock(
            num_heads=4, head_dim=16, kernel_size=2, top_k=4,
            use_decoupled_router=True,
        )
        x = tf.random.normal([1, 8, 64])
        out = block(x, training=False)
        assert out.shape == (1, 8, 64)

    def test_gradient_flows(self):
        block = SparseAttentionBlock(
            num_heads=2, head_dim=16, kernel_size=2, top_k=2,
        )
        x = tf.random.normal([1, 8, 32])
        with tf.GradientTape() as tape:
            tape.watch(x)
            out = block(x, training=True)
            loss = tf.reduce_mean(out)
        grads = tape.gradient(loss, block.trainable_variables)
        assert any(g is not None for g in grads), "No gradients computed"

    def test_different_kernel_sizes(self):
        for ks in [1, 2, 4, 8]:
            block = SparseAttentionBlock(
                num_heads=2, head_dim=16, kernel_size=ks, top_k=2,
            )
            x = tf.random.normal([1, 16, 32])
            out = block(x)
            assert out.shape == (1, 16, 32), f"Failed for kernel_size={ks}"

    def test_serialization(self):
        block = SparseAttentionBlock(
            num_heads=8, head_dim=64, kernel_size=4, top_k=8,
            use_decoupled_router=True,
        )
        config = block.get_config()
        assert config["num_heads"] == 8
        assert config["kernel_size"] == 4
        assert config["top_k"] == 8
        assert config["use_decoupled_router"] is True


# -----------------------------------------------------------------------
# Compression Stats Tests
# -----------------------------------------------------------------------

class TestCompressionStats:
    """Tests for theoretical compression statistics."""

    def test_basic_stats(self):
        from tensorflow.keras import Sequential
        from tensorflow.keras.layers import Dense
        model = Sequential([Dense(10, input_shape=(5,))])
        stats = get_compression_stats(model, kernel_size=4, top_k=8,
                                      seq_len=196)
        assert stats["original_seq_len"] == 196
        assert stats["pooled_seq_len"] == 49  # 196 / 4
        assert stats["effective_top_k"] == 8
        assert stats["kv_memory_reduction"] == "4x"
        assert stats["model_params"] > 0

    def test_top_k_capped_at_num_chunks(self):
        from tensorflow.keras import Sequential
        from tensorflow.keras.layers import Dense
        model = Sequential([Dense(10, input_shape=(5,))])
        stats = get_compression_stats(model, kernel_size=16, top_k=100,
                                      seq_len=64)
        assert stats["effective_top_k"] == 4  # 64/16 = 4 chunks < 100


# -----------------------------------------------------------------------
# Citation Constants Tests
# -----------------------------------------------------------------------

class TestCitations:
    """Tests for MSA citation strings."""

    def test_bibtex_has_required_fields(self):
        assert "chen2026" in MSA_CITATION_BIBTEX.lower()
        assert "2603.23516" in MSA_CITATION_BIBTEX
        assert "arXiv" in MSA_CITATION_BIBTEX

    def test_ieee_format(self):
        assert "Y. Chen et al." in MSA_CITATION_IEEE
        assert "2026" in MSA_CITATION_IEEE

    def test_apa_format(self):
        assert "Chen, Y." in MSA_CITATION_APA
        assert "(2026)" in MSA_CITATION_APA
