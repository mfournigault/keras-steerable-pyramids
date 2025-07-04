#!/usr/bin/env python3
"""
Test TensorFlow math utilities migration.

This test validates that the TensorFlow implementations of the math utilities
produce results consistent with the original PyTorch implementations.
"""

import numpy as np
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    import tensorflow as tf
    from utils.math_utils_tf import batch_fftshift2d, batch_ifftshift2d, tf_fft2d, tf_ifft2d, tf_fftshift2d, tf_ifftshift2d
    TF_AVAILABLE = True
except ImportError as e:
    print(f"TensorFlow not available: {e}")
    TF_AVAILABLE = False

def test_fft_shift_consistency():
    """Test that TensorFlow FFT shift operations are consistent with NumPy."""
    if not TF_AVAILABLE:
        print("Skipping TensorFlow tests - TensorFlow not available")
        return
    
    print("Testing FFT shift consistency...")
    
    # Create a test image
    np.random.seed(42)
    test_image = np.random.randn(32, 32).astype(np.float32)
    
    # NumPy reference
    fft_numpy = np.fft.fft2(test_image)
    fftshift_numpy = np.fft.fftshift(fft_numpy)
    ifftshift_numpy = np.fft.ifftshift(fftshift_numpy)
    
    # TensorFlow implementation
    test_image_tf = tf.constant(test_image)
    fft_tf = tf_fft2d(test_image_tf)
    fftshift_tf = tf_fftshift2d(fft_tf)
    ifftshift_tf = tf_ifftshift2d(fftshift_tf)
    
    # Convert back to numpy for comparison
    fftshift_tf_np = fftshift_tf.numpy()
    ifftshift_tf_np = ifftshift_tf.numpy()
    
    # Compare results
    tolerance = 1e-6
    
    fftshift_close = np.allclose(fftshift_numpy, fftshift_tf_np, atol=tolerance)
    ifftshift_close = np.allclose(ifftshift_numpy, ifftshift_tf_np, atol=tolerance)
    
    print(f"FFT shift close to NumPy: {fftshift_close}")
    print(f"IFFT shift close to NumPy: {ifftshift_close}")
    
    if fftshift_close and ifftshift_close:
        print("✓ FFT shift tests passed!")
    else:
        print("✗ FFT shift tests failed!")
        
    return fftshift_close and ifftshift_close

def test_batch_fft_operations():
    """Test batch FFT operations."""
    if not TF_AVAILABLE:
        print("Skipping TensorFlow batch tests - TensorFlow not available")
        return
        
    print("\nTesting batch FFT operations...")
    
    # Create batch of test images
    np.random.seed(42)
    batch_size = 4
    height, width = 64, 64
    test_batch = np.random.randn(batch_size, height, width).astype(np.float32)
    
    # Test batch operations
    test_batch_tf = tf.constant(test_batch)
    
    # Forward: real -> complex FFT -> shift
    fft_batch = tf_fft2d(test_batch_tf)
    shifted_batch = tf_fftshift2d(fft_batch)
    
    # Backward: unshift -> inverse FFT
    unshifted_batch = tf_ifftshift2d(shifted_batch)
    reconstructed_batch = tf_ifft2d(unshifted_batch)
    reconstructed_real = tf.math.real(reconstructed_batch).numpy()
    
    # Check reconstruction
    reconstruction_error = np.mean(np.abs(test_batch - reconstructed_real))
    tolerance = 1e-5
    
    print(f"Reconstruction error: {reconstruction_error}")
    print(f"Tolerance: {tolerance}")
    
    success = reconstruction_error < tolerance
    if success:
        print("✓ Batch FFT operations test passed!")
    else:
        print("✗ Batch FFT operations test failed!")
        
    return success

def test_complex_tensor_operations():
    """Test complex tensor operations with our batch functions."""
    if not TF_AVAILABLE:
        print("Skipping complex tensor tests - TensorFlow not available")
        return
        
    print("\nTesting complex tensor operations...")
    
    # Create complex test data (simulating the format from PyTorch)
    np.random.seed(42)
    batch_size = 2
    height, width = 32, 32
    
    # Create complex tensor
    real_part = np.random.randn(batch_size, height, width).astype(np.float32)
    imag_part = np.random.randn(batch_size, height, width).astype(np.float32)
    complex_tensor = tf.complex(real_part, imag_part)
    
    # Test our batch shift functions
    shifted = batch_fftshift2d(complex_tensor)
    unshifted = batch_ifftshift2d(shifted)
    
    # Check that we get back to original (should be close)
    original_real = tf.math.real(complex_tensor).numpy()
    original_imag = tf.math.imag(complex_tensor).numpy()
    
    final_real = tf.math.real(unshifted).numpy()
    final_imag = tf.math.imag(unshifted).numpy()
    
    real_error = np.mean(np.abs(original_real - final_real))
    imag_error = np.mean(np.abs(original_imag - final_imag))
    tolerance = 1e-6
    
    print(f"Real part error: {real_error}")
    print(f"Imaginary part error: {imag_error}")
    
    success = real_error < tolerance and imag_error < tolerance
    if success:
        print("✓ Complex tensor operations test passed!")
    else:
        print("✗ Complex tensor operations test failed!")
        
    return success

def run_all_tests():
    """Run all tests."""
    print("Running TensorFlow math utilities tests...")
    print("=" * 50)
    
    results = []
    
    if TF_AVAILABLE:
        results.append(test_fft_shift_consistency())
        results.append(test_batch_fft_operations())
        results.append(test_complex_tensor_operations())
    else:
        print("TensorFlow not available - cannot run tests")
        return False
    
    print("\n" + "=" * 50)
    if all(results):
        print("✓ All tests passed!")
        return True
    else:
        print("✗ Some tests failed!")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)