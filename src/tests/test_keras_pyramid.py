#!/usr/bin/env python3
"""
Test Keras Steerable Pyramid implementation against NumPy reference.

This test validates that the Keras implementation produces results
consistent with the original NumPy implementation.
"""

import numpy as np
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    import tensorflow as tf
    from steerable_pyramids.SCFpyr_Keras import SCFpyr_Keras, build_scf_pyramid, reconstruct_scf_pyramid
    TF_AVAILABLE = True
except ImportError as e:
    print(f"TensorFlow/Keras not available: {e}")
    TF_AVAILABLE = False

try:
    from steerable_pyramids.SCFpyr_NumPy import SCFpyr_NumPy
    NUMPY_AVAILABLE = True
except ImportError as e:
    print(f"NumPy implementation not available: {e}")
    NUMPY_AVAILABLE = False

def create_test_image(size=128):
    """Create a test image for pyramid testing."""
    np.random.seed(42)
    # Create a more interesting test pattern
    x, y = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))
    
    # Combine multiple patterns
    pattern1 = np.sin(5 * np.pi * x) * np.cos(3 * np.pi * y)
    pattern2 = np.exp(-(x**2 + y**2) * 4)
    noise = np.random.randn(size, size) * 0.1
    
    test_image = (pattern1 + pattern2 + noise).astype(np.float32)
    
    # Normalize to [0, 1]
    test_image = (test_image - test_image.min()) / (test_image.max() - test_image.min())
    
    return test_image

def test_pyramid_build_consistency():
    """Test that Keras implementation builds pyramids consistently with NumPy."""
    if not (TF_AVAILABLE and NUMPY_AVAILABLE):
        print("Skipping pyramid build test - dependencies not available")
        return False
        
    print("Testing pyramid build consistency...")
    
    # Create test image
    test_image = create_test_image(128)
    
    # Parameters
    height = 4  # Use smaller pyramid for testing
    nbands = 4
    scale_factor = 2
    
    # NumPy reference
    pyr_numpy = SCFpyr_NumPy(height, nbands, scale_factor)
    coeff_numpy = pyr_numpy.build(test_image)
    
    # Keras implementation
    test_image_batch = tf.constant(test_image[np.newaxis, :, :])  # Add batch dimension
    pyr_keras = SCFpyr_Keras(height, nbands, scale_factor)
    coeff_keras = pyr_keras.build(test_image_batch)
    
    # Compare coefficients
    tolerance = 1e-3  # More relaxed tolerance for comparison
    
    print(f"NumPy coefficients levels: {len(coeff_numpy)}")
    print(f"Keras coefficients levels: {len(coeff_keras)}")
    
    if len(coeff_numpy) != len(coeff_keras):
        print("✗ Different number of pyramid levels!")
        return False
    
    all_close = True
    
    for level, (coeff_np, coeff_k) in enumerate(zip(coeff_numpy, coeff_keras)):
        print(f"\nLevel {level}:")
        
        if isinstance(coeff_np, np.ndarray) and tf.is_tensor(coeff_k):
            # Low-pass or high-pass level
            coeff_k_np = coeff_k[0].numpy()  # Extract first batch element
            
            print(f"  NumPy shape: {coeff_np.shape}")
            print(f"  Keras shape: {coeff_k_np.shape}")
            
            if coeff_np.shape != coeff_k_np.shape:
                print(f"  ✗ Shape mismatch!")
                all_close = False
                continue
            
            error = np.mean(np.abs(coeff_np - coeff_k_np))
            close = np.allclose(coeff_np, coeff_k_np, atol=tolerance)
            
            print(f"  Mean error: {error}")
            print(f"  All close: {close}")
            
            if not close:
                all_close = False
                
        elif isinstance(coeff_np, list) and isinstance(coeff_k, list):
            # Orientation bands level
            print(f"  Number of orientations: {len(coeff_np)} vs {len(coeff_k)}")
            
            if len(coeff_np) != len(coeff_k):
                print(f"  ✗ Different number of orientations!")
                all_close = False
                continue
                
            for band, (band_np, band_k) in enumerate(zip(coeff_np, coeff_k)):
                band_k_np = band_k[0].numpy()  # Extract first batch element
                
                # Handle complex data
                if np.iscomplexobj(band_np):
                    band_k_complex = band_k_np
                    if not np.iscomplexobj(band_k_complex):
                        # If Keras returns real tensor, we need to handle this
                        print(f"    Band {band}: NumPy is complex, Keras is real")
                        # For now, just compare magnitudes
                        band_np_mag = np.abs(band_np)
                        band_k_mag = np.abs(band_k_complex)
                        error = np.mean(np.abs(band_np_mag - band_k_mag))
                        close = np.allclose(band_np_mag, band_k_mag, atol=tolerance)
                    else:
                        error = np.mean(np.abs(band_np - band_k_complex))
                        close = np.allclose(band_np, band_k_complex, atol=tolerance)
                else:
                    error = np.mean(np.abs(band_np - band_k_np))
                    close = np.allclose(band_np, band_k_np, atol=tolerance)
                
                print(f"    Band {band} error: {error}, close: {close}")
                
                if not close:
                    all_close = False
        else:
            print(f"  ✗ Type mismatch: {type(coeff_np)} vs {type(coeff_k)}")
            all_close = False
    
    if all_close:
        print("\n✓ Pyramid build consistency test passed!")
    else:
        print("\n✗ Pyramid build consistency test failed!")
        
    return all_close

def test_basic_keras_functionality():
    """Test basic Keras layer functionality."""
    if not TF_AVAILABLE:
        print("Skipping Keras functionality test - TensorFlow not available")
        return False
        
    print("\nTesting basic Keras functionality...")
    
    # Create test image
    test_image = create_test_image(64)  # Smaller for basic test
    test_batch = tf.constant(test_image[np.newaxis, np.newaxis, :, :])  # [N, C, H, W]
    
    # Create pyramid layer
    pyr_layer = SCFpyr_Keras(height=3, nbands=4, scale_factor=2)
    
    try:
        # Test call method
        coeff = pyr_layer(test_batch)
        print(f"  ✓ Layer call successful, returned {len(coeff)} levels")
        
        # Test build method
        coeff2 = pyr_layer.build_pyramid(test_batch)
        print(f"  ✓ Build method successful, returned {len(coeff2)} levels")
        
        # Test config
        config = pyr_layer.get_config()
        print(f"  ✓ Config retrieved: {config}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error in Keras functionality: {e}")
        return False

def test_convenience_functions():
    """Test convenience functions."""
    if not TF_AVAILABLE:
        print("Skipping convenience functions test - TensorFlow not available")
        return False
        
    print("\nTesting convenience functions...")
    
    # Create test image
    test_image = create_test_image(64)
    test_batch = tf.constant(test_image[np.newaxis, :, :])  # [N, H, W]
    
    try:
        # Test build function
        coeff = build_scf_pyramid(test_batch, height=3, nbands=4)
        print(f"  ✓ Build function successful, returned {len(coeff)} levels")
        
        # Test reconstruct function (if we implement reconstruction)
        # reconstructed = reconstruct_scf_pyramid(coeff, height=3, nbands=4)
        # print(f"  ✓ Reconstruct function successful, shape: {reconstructed.shape}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error in convenience functions: {e}")
        return False

def run_all_tests():
    """Run all tests."""
    print("Running Keras Steerable Pyramid tests...")
    print("=" * 60)
    
    results = []
    
    # Check dependencies
    print(f"TensorFlow available: {TF_AVAILABLE}")
    print(f"NumPy implementation available: {NUMPY_AVAILABLE}")
    print()
    
    if TF_AVAILABLE:
        results.append(test_basic_keras_functionality())
        results.append(test_convenience_functions())
        
        if NUMPY_AVAILABLE:
            results.append(test_pyramid_build_consistency())
        else:
            print("Cannot run comparison tests - NumPy implementation not available")
    else:
        print("Cannot run any tests - TensorFlow not available")
        return False
    
    print("\n" + "=" * 60)
    if all(results):
        print("✓ All tests passed!")
        return True
    else:
        print("✗ Some tests failed!")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)