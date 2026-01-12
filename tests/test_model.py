import unittest
import sys
import os
import tensorflow as tf
import numpy as np

from ccc.model import GeneralizedCCCModel

class TestGeneralizedCCCModel(unittest.TestCase):
    def setUp(self):
        # Default baseline parameters
        self.B = 2
        self.N = 5
        self.M = 10
        self.T = 100
        self.encoding_units = [8]
        self.lstm_units = [16]
        self.final_hidden_layer_sizes = [8]
        
    def test_model_initialization_additive(self):
        """Test instantiation and forward pass in Additive mode"""
        model = GeneralizedCCCModel(
            encoding_units=self.encoding_units,
            lstm_units=self.lstm_units,
            final_hidden_layer_sizes=self.final_hidden_layer_sizes,
            multiplicative=False,
            final_activation='linear',
            use_raw_returns=False
        )
        
        Cxx = tf.random.normal((self.B, self.N, self.N))
        Cyy = tf.random.normal((self.B, self.M, self.M))
        Cxy = tf.random.normal((self.B, self.N, self.M))
        n_samples = tf.constant([100.0] * self.B)
        
        output = model([Cxx, Cyy, Cxy, n_samples])
        
        # Output should be (B, N, M) - corresponding to Cxy shape
        self.assertEqual(output.shape, (self.B, self.N, self.M))

    def test_model_initialization_multiplicative(self):
        """Test instantiation and forward pass in Multiplicative mode"""
        model = GeneralizedCCCModel(
            encoding_units=self.encoding_units,
            lstm_units=self.lstm_units,
            final_hidden_layer_sizes=self.final_hidden_layer_sizes,
            multiplicative=True,
            final_activation='softplus',
            use_raw_returns=False
        )
        
        Cxx = tf.random.normal((self.B, self.N, self.N))
        Cyy = tf.random.normal((self.B, self.M, self.M))
        Cxy = tf.random.normal((self.B, self.N, self.M))
        n_samples = tf.constant([100.0] * self.B)
        
        output = model([Cxx, Cyy, Cxy, n_samples])
        self.assertEqual(output.shape, (self.B, self.N, self.M))

    def test_use_raw_returns(self):
        """Test the model when use_raw_returns=True (Inputting raw time series Rx, Ry)"""
        model = GeneralizedCCCModel(
            encoding_units=self.encoding_units,
            lstm_units=self.lstm_units,
            final_hidden_layer_sizes=self.final_hidden_layer_sizes,
            multiplicative=True,
            final_activation='softplus',
            use_raw_returns=True
        )
        
        # Inputs: Rx (B, N, T), Ry (B, M, T)
        Rx = tf.random.normal((self.B, self.N, self.T))
        Ry = tf.random.normal((self.B, self.M, self.T))
        
        # Models with use_raw_returns expect [Rx, Ry]
        output = model([Rx, Ry])
        
        self.assertEqual(output.shape, (self.B, self.N, self.M))

    def test_varying_architectures(self):
        """Test different network depths and widths"""
        configs = [
            {'encoding': [32, 16], 'lstm': [64], 'final': [32]},
            {'encoding': [], 'lstm': [16], 'final': []}, # Minimal
            {'encoding': [8], 'lstm': [8, 8], 'final': [4]} # Deep recurrent
        ]
        
        for conf in configs:
            with self.subTest(config=conf):
                model = GeneralizedCCCModel(
                    encoding_units=conf['encoding'],
                    lstm_units=conf['lstm'],
                    final_hidden_layer_sizes=conf['final'],
                    multiplicative=True,
                    final_activation='softplus'
                )
                
                Cxx = tf.random.normal((self.B, self.N, self.N))
                Cyy = tf.random.normal((self.B, self.M, self.M))
                Cxy = tf.random.normal((self.B, self.N, self.M))
                n_samples = tf.constant([100.0] * self.B)
                
                output = model([Cxx, Cyy, Cxy, n_samples])
                self.assertEqual(output.shape, (self.B, self.N, self.M))

    def test_valid_activations(self):
        """Test all supported activation functions for multiplicative mode"""
        activations = ['softplus', 'relu', 'sigmoid']
        for act in activations:
            with self.subTest(activation=act):
                model = GeneralizedCCCModel(
                    encoding_units=self.encoding_units,
                    lstm_units=self.lstm_units,
                    final_hidden_layer_sizes=self.final_hidden_layer_sizes,
                    multiplicative=True,
                    final_activation=act
                )
                # Just verify it builds and runs without error
                Cxx = tf.random.normal((self.B, self.N, self.N))
                Cyy = tf.random.normal((self.B, self.M, self.M))
                Cxy = tf.random.normal((self.B, self.N, self.M))
                n_samples = tf.constant([100.0] * self.B)
                model([Cxx, Cyy, Cxy, n_samples])

    def test_additive_activations(self):
        """Test supported activations for additive mode"""
        activations = ['linear', 'tanh']
        for act in activations:
             with self.subTest(activation=act):
                model = GeneralizedCCCModel(
                    encoding_units=self.encoding_units,
                    lstm_units=self.lstm_units,
                    final_hidden_layer_sizes=self.final_hidden_layer_sizes,
                    multiplicative=False,
                    final_activation=act
                )
                Cxx = tf.random.normal((self.B, self.N, self.N))
                Cyy = tf.random.normal((self.B, self.M, self.M))
                Cxy = tf.random.normal((self.B, self.N, self.M))
                n_samples = tf.constant([100.0] * self.B)
                model([Cxx, Cyy, Cxy, n_samples])

    def test_invalid_activation_config(self):
        """Test validation logic for activation functions"""
        with self.assertRaises(ValueError):
            GeneralizedCCCModel(
                encoding_units=self.encoding_units,
                lstm_units=self.lstm_units,
                final_hidden_layer_sizes=self.final_hidden_layer_sizes,
                multiplicative=True,
                final_activation='linear', # Invalid for multiplicative
                use_raw_returns=False
            )

    def test_large_dimensions(self):
        """Stress test with slightly larger matrices"""
        N_large = 50
        M_large = 60
        model = GeneralizedCCCModel(
            encoding_units=[16],
            lstm_units=[16],
            final_hidden_layer_sizes=[16],
            multiplicative=True,
            final_activation='softplus'
        )
        Cxx = tf.random.normal((self.B, N_large, N_large))
        Cyy = tf.random.normal((self.B, M_large, M_large))
        Cxy = tf.random.normal((self.B, N_large, M_large))
        n_samples = tf.constant([200.0] * self.B)
        
        output = model([Cxx, Cyy, Cxy, n_samples])
        self.assertEqual(output.shape, (self.B, N_large, M_large))

if __name__ == '__main__':
    unittest.main()
