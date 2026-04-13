"""
Tests for svd_via_eigh_full and CrossRIELayer graph-mode gradients.

These tests verify that tf.linalg.qr gradient NotImplementedError with
dynamic shapes is permanently resolved (the stop_gradient fix).
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import unittest
import numpy as np
import tensorflow as tf
from crossrie.custom_layers import svd_via_eigh_full, reconstruct_matrix_from_svd
from crossrie import CrossRIELayer


class TestSvdViaEighFull(unittest.TestCase):
    """Tests for the svd_via_eigh_full function."""

    def test_forward_correctness(self):
        """SVD reconstruction should approximate the original matrix."""
        C = tf.random.normal((4, 6, 8), dtype=tf.float32)
        s_k, U_full, V_full = svd_via_eigh_full(C)
        C_hat = reconstruct_matrix_from_svd(s_k, U_full, V_full)
        np.testing.assert_allclose(C.numpy(), C_hat.numpy(), atol=1e-4)

    def test_singular_values_nonnegative(self):
        """Singular values must be non-negative."""
        C = tf.random.normal((3, 10, 5), dtype=tf.float32)
        s_k, _, _ = svd_via_eigh_full(C)
        self.assertTrue(np.all(s_k.numpy() >= 0))

    def test_orthonormality_U(self):
        """U_full columns should be orthonormal."""
        C = tf.random.normal((2, 7, 9), dtype=tf.float32)
        _, U_full, _ = svd_via_eigh_full(C)
        UtU = tf.matmul(U_full, U_full, transpose_a=True)
        eye = tf.eye(tf.shape(U_full)[2], batch_shape=[2])
        np.testing.assert_allclose(UtU.numpy(), eye.numpy(), atol=1e-4)

    def test_orthonormality_V(self):
        """V_full columns should be orthonormal."""
        C = tf.random.normal((2, 5, 8), dtype=tf.float32)
        _, _, V_full = svd_via_eigh_full(C)
        VtV = tf.matmul(V_full, V_full, transpose_a=True)
        eye = tf.eye(tf.shape(V_full)[2], batch_shape=[2])
        np.testing.assert_allclose(VtV.numpy(), eye.numpy(), atol=1e-4)

    def test_square_matrix(self):
        """Should handle square matrices (n == m)."""
        C = tf.random.normal((3, 6, 6), dtype=tf.float32)
        s_k, U_full, V_full = svd_via_eigh_full(C)
        C_hat = reconstruct_matrix_from_svd(s_k, U_full, V_full)
        np.testing.assert_allclose(C.numpy(), C_hat.numpy(), atol=1e-4)

    def test_tall_matrix(self):
        """Should handle tall matrices (n > m)."""
        C = tf.random.normal((2, 12, 4), dtype=tf.float32)
        s_k, U_full, V_full = svd_via_eigh_full(C)
        self.assertEqual(s_k.shape[1], 4)  # r = min(n, m)
        C_hat = reconstruct_matrix_from_svd(s_k, U_full, V_full)
        np.testing.assert_allclose(C.numpy(), C_hat.numpy(), atol=1e-4)


class TestSvdGradientGraphMode(unittest.TestCase):
    """
    Tests that gradients through svd_via_eigh_full work in graph mode
    with dynamic shapes (the original NotImplementedError scenario).
    """

    def test_gradient_exists_graph_mode(self):
        """Gradient through s_k must be non-None in @tf.function."""

        @tf.function
        def compute_grad(C):
            with tf.GradientTape() as tape:
                tape.watch(C)
                s_k, _, _ = svd_via_eigh_full(C)
                loss = tf.reduce_sum(s_k)
            return tape.gradient(loss, C)

        C = tf.random.normal((2, 5, 8))
        grad = compute_grad(C)
        self.assertIsNotNone(grad)
        self.assertFalse(np.any(np.isnan(grad.numpy())))

    def test_gradient_varying_shapes_graph_mode(self):
        """Single @tf.function handles multiple dynamic shapes without error."""

        @tf.function(reduce_retracing=True)
        def compute_grad(C):
            with tf.GradientTape() as tape:
                tape.watch(C)
                s_k, _, _ = svd_via_eigh_full(C)
                loss = tf.reduce_sum(s_k ** 2)
            return tape.gradient(loss, C)

        for n, m in [(5, 8), (10, 3), (7, 7), (20, 15)]:
            C = tf.random.normal((2, n, m))
            grad = compute_grad(C)
            self.assertEqual(grad.shape, C.shape,
                             f"Gradient shape mismatch for n={n}, m={m}")
            self.assertFalse(np.any(np.isnan(grad.numpy())),
                             f"NaN gradient for n={n}, m={m}")

    def test_gradient_through_reconstruction(self):
        """Gradient flows through full SVD → reconstruct pipeline."""

        @tf.function
        def compute_grad(C):
            with tf.GradientTape() as tape:
                tape.watch(C)
                s_k, U_full, V_full = svd_via_eigh_full(C)
                C_hat = reconstruct_matrix_from_svd(s_k, U_full, V_full)
                loss = tf.reduce_sum(C_hat ** 2)
            return tape.gradient(loss, C)

        C = tf.random.normal((3, 6, 9))
        grad = compute_grad(C)
        self.assertIsNotNone(grad)
        self.assertFalse(np.any(np.isnan(grad.numpy())))


class TestCrossRIELayerGradient(unittest.TestCase):
    """
    End-to-end gradient tests for the full CrossRIELayer in graph mode
    with dynamic matrix dimensions.
    """

    def _build_model(self, **layer_kwargs):
        layer = CrossRIELayer(
            encoding_units=[8], lstm_units=[16],
            final_hidden_layer_sizes=[8], **layer_kwargs
        )
        in_Cxx = tf.keras.Input(shape=(None, None))
        in_Cyy = tf.keras.Input(shape=(None, None))
        in_Cxy = tf.keras.Input(shape=(None, None))
        in_t = tf.keras.Input(shape=())
        out = layer([in_Cxx, in_Cyy, in_Cxy, in_t])
        model = tf.keras.Model(inputs=[in_Cxx, in_Cyy, in_Cxy, in_t], outputs=out)
        model.compile(optimizer='adam', loss='mse')
        return model

    def _make_batch(self, B, N, M, T):
        Cxx = tf.random.normal((B, N, N))
        Cyy = tf.random.normal((B, M, M))
        Cxy = tf.random.normal((B, N, M))
        T_s = tf.constant([float(T)] * B)
        target = tf.random.normal((B, N, M))
        return [Cxx, Cyy, Cxy, T_s], target

    def test_train_step_additive(self):
        """Single train_on_batch in additive mode succeeds."""
        model = self._build_model(multiplicative=False, final_activation='linear')
        inputs, target = self._make_batch(4, 8, 10, 100)
        loss = model.train_on_batch(inputs, target)
        self.assertGreater(loss, 0)

    def test_train_step_multiplicative(self):
        """Single train_on_batch in multiplicative mode succeeds."""
        model = self._build_model(multiplicative=True, final_activation='softplus')
        inputs, target = self._make_batch(4, 8, 10, 100)
        loss = model.train_on_batch(inputs, target)
        self.assertGreater(loss, 0)

    def test_varying_shapes_sequential_training(self):
        """Model trains on batches with different N, M without errors."""
        model = self._build_model(multiplicative=False, final_activation='linear')
        shapes = [(6, 10), (12, 5), (8, 8), (4, 15)]
        for N, M in shapes:
            inputs, target = self._make_batch(4, N, M, 100)
            loss = model.train_on_batch(inputs, target)
            self.assertGreater(loss, 0, f"Zero loss for N={N}, M={M}")

    def test_loss_decreases(self):
        """Loss decreases over multiple training steps (basic sanity)."""
        model = self._build_model(multiplicative=False, final_activation='linear')
        inputs, target = self._make_batch(8, 6, 8, 100)
        losses = []
        for _ in range(10):
            loss = model.train_on_batch(inputs, target)
            losses.append(loss)
        self.assertLess(losses[-1], losses[0],
                        "Loss did not decrease over 10 steps on the same batch")


if __name__ == '__main__':
    unittest.main()
