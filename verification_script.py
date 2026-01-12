import tensorflow as tf
import numpy as np
import os
import sys

# Ensure src is in python path


from ccc import custom_layers
# from ccc import helper_functions # Implicitly tested via custom_layers

def verify_layers():
    print("Verifying SVDViaEighFullLayer...")
    # SVDViaEighFullLayer
    layer_svd = custom_layers.SVDViaEighFullLayer(eps=1e-6)
    B, n, m = 2, 3, 4
    C = tf.random.normal((B, n, m), dtype=tf.float32)
    s_k, U_full, V_full = layer_svd(C)
    print(f"SVDViaEighFullLayer output shapes: s_k={s_k.shape}, U_full={U_full.shape}, V_full={V_full.shape}")
    print(f"s_k sum: {tf.reduce_sum(s_k).numpy()}")
    
    print("\nVerifying DiagVtCvLayer...")
    # DiagVtCvLayer
    layer_diag = custom_layers.DiagVtCvLayer()
    r = min(n, m)
    # C is [B, n, n] in docstring but standard usage might vary? 
    # Looking at docstring: C: [B, n, n], V: [B, n, r]
    # Let's create dummy consistent inputs
    C_sq = tf.random.normal((B, n, n), dtype=tf.float32)
    V = tf.random.normal((B, n, r), dtype=tf.float32)
    d = layer_diag([C_sq, V])
    print(f"DiagVtCvLayer output shape: {d.shape}")
    print(f"d sum: {tf.reduce_sum(d).numpy()}")

    print("\nVerifying PadA2BSimpleLayer...")
    # PadA2BSimpleLayer
    layer_pad = custom_layers.PadA2BSimpleLayer()
    A = tf.random.normal((B, 3, 5), dtype=tf.float32)
    B_tensor = tf.random.normal((B, 4, 6), dtype=tf.float32) # n_b = max(4, 6) = 6
    padded = layer_pad([A, B_tensor])
    print(f"PadA2BSimpleLayer output shape: {padded.shape}")
    print(f"padded sum: {tf.reduce_sum(padded).numpy()}")
    
    print("\nVerifying SVDReconstructFromFullLayer...")
    # SVDReconstructFromFullLayer
    layer_recon = custom_layers.SVDReconstructFromFullLayer()
    # reuse outputs from svd
    # note: svd_via_eigh_full inputs C: [B, n, m]
    # returns s_k [B, r], U_full [B, n, n], V_full [B, m, m]
    # svd_reconstruct_from_full inputs: s_k, U_full, V_full
    recon = layer_recon([s_k, U_full, V_full])
    print(f"SVDReconstructFromFullLayer output shape: {recon.shape}")
    print(f"recon sum: {tf.reduce_sum(recon).numpy()}")

    # Save results to compare later if needed, but printing sums is a good quick check
    return {
        "s_k_sum": tf.reduce_sum(s_k).numpy(),
        "d_sum": tf.reduce_sum(d).numpy(),
        "padded_sum": tf.reduce_sum(padded).numpy(),
        "recon_sum": tf.reduce_sum(recon).numpy()
    }

if __name__ == "__main__":
    tf.random.set_seed(42)
    verify_layers()
