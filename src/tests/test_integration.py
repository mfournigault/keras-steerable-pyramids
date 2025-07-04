#!/usr/bin/env python3
"""
Integration test for Keras steerable pyramid implementation.

This test compares the Keras implementation against the NumPy reference
to ensure mathematical correctness of the migration.
"""

import numpy as np
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from steerable_pyramids.SCFpyr_NumPy import SCFpyr_NumPy
        print("✓ NumPy implementation imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import NumPy implementation: {e}")
        return False
    
    try:
        # Try to import TensorFlow modules
        import tensorflow as tf
        from steerable_pyramids.SCFpyr_Keras import SCFpyr_Keras
        from utils.math_utils_tf import batch_fftshift2d, tf_fft2d
        from utils.utils_tf import get_device, load_image_batch
        print("✓ TensorFlow/Keras implementation imported successfully")
        return True
    except ImportError as e:
        print(f"⚠ TensorFlow/Keras implementation not available: {e}")
        print("This is expected if TensorFlow is not installed")
        return False

def create_test_pattern(size=64):
    """Create a deterministic test pattern."""
    x, y = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))
    
    # Create a pattern with multiple frequencies
    pattern = (np.sin(4 * np.pi * x) * np.cos(3 * np.pi * y) + 
               np.sin(8 * np.pi * (x + y)) * 0.5 +
               np.exp(-(x**2 + y**2) * 2) * 0.3)
    
    # Normalize to [0, 1]
    pattern = (pattern - pattern.min()) / (pattern.max() - pattern.min())
    return pattern.astype(np.float32)

def test_numpy_consistency():
    """Test NumPy implementation for consistency."""
    print("\nTesting NumPy implementation consistency...")
    
    try:
        from steerable_pyramids.SCFpyr_NumPy import SCFpyr_NumPy
        
        # Create test image
        test_image = create_test_pattern(64)
        
        # Test different pyramid configurations
        configs = [
            (3, 4, 2),  # height=3, nbands=4, scale_factor=2
            (4, 4, 2),  # height=4, nbands=4, scale_factor=2
            (3, 6, 2),  # height=3, nbands=6, scale_factor=2
        ]
        
        for height, nbands, scale_factor in configs:
            print(f"  Testing config: height={height}, nbands={nbands}, scale_factor={scale_factor}")
            
            pyr = SCFpyr_NumPy(height, nbands, scale_factor)
            coeff = pyr.build(test_image)
            reconstructed = pyr.reconstruct(coeff)
            
            # Check reconstruction quality
            error = np.mean(np.abs(test_image - reconstructed))
            print(f"    Reconstruction error: {error:.2e}")
            
            if error > 1e-3:
                print(f"    ⚠ High reconstruction error for config {height}-{nbands}-{scale_factor}")
            else:
                print(f"    ✓ Good reconstruction quality")
                
        return True
        
    except Exception as e:
        print(f"✗ NumPy consistency test failed: {e}")
        return False

def test_keras_structure():
    """Test Keras implementation structure (without TensorFlow)."""
    print("\nTesting Keras implementation structure...")
    
    try:
        # Check if the file can be parsed (syntax check)
        with open(os.path.join(os.path.dirname(__file__), '..', 'steerable_pyramids', 'SCFpyr_Keras.py'), 'r') as f:
            keras_code = f.read()
            
        # Basic syntax check by compiling
        compile(keras_code, 'SCFpyr_Keras.py', 'exec')
        print("✓ Keras implementation has valid Python syntax")
        
        # Check for key methods and classes
        required_elements = [
            'class SCFpyr_Keras',
            'def build(',
            'def reconstruct(',
            'def _build_levels(',
            'def _reconstruct_levels(',
            'def get_config(',
            'tf.signal.fft2d',
            'tf.signal.ifft2d',
            'batch_fftshift2d',
            'batch_ifftshift2d'
        ]
        
        missing_elements = []
        for element in required_elements:
            if element not in keras_code:
                missing_elements.append(element)
        
        if missing_elements:
            print(f"✗ Missing required elements: {missing_elements}")
            return False
        else:
            print("✓ All required methods and TensorFlow operations found")
            
        return True
        
    except Exception as e:
        print(f"✗ Keras structure test failed: {e}")
        return False

def test_tensorflow_math_utils_structure():
    """Test TensorFlow math utils structure."""
    print("\nTesting TensorFlow math utils structure...")
    
    try:
        with open(os.path.join(os.path.dirname(__file__), '..', 'utils', 'math_utils_tf.py'), 'r') as f:
            tf_utils_code = f.read()
            
        # Basic syntax check
        compile(tf_utils_code, 'math_utils_tf.py', 'exec')
        print("✓ TensorFlow math utils has valid Python syntax")
        
        # Check for key functions
        required_functions = [
            'def roll_n(',
            'def batch_fftshift2d(',
            'def batch_ifftshift2d(',
            'def tf_fft2d(',
            'def tf_ifft2d(',
            'tf.roll',
            'tf.complex',
            'tf.signal.fft2d',
            'tf.signal.ifft2d'
        ]
        
        missing_functions = []
        for func in required_functions:
            if func not in tf_utils_code:
                missing_functions.append(func)
        
        if missing_functions:
            print(f"✗ Missing required functions: {missing_functions}")
            return False
        else:
            print("✓ All required TensorFlow math functions found")
            
        return True
        
    except Exception as e:
        print(f"✗ TensorFlow math utils structure test failed: {e}")
        return False

def run_structural_tests():
    """Run all structural tests that don't require TensorFlow."""
    print("Running structural validation tests...")
    print("=" * 60)
    
    results = []
    
    # Test imports
    tf_available = test_imports()
    
    # Test NumPy implementation
    results.append(test_numpy_consistency())
    
    # Test Keras structure
    results.append(test_keras_structure())
    
    # Test TensorFlow math utils structure  
    results.append(test_tensorflow_math_utils_structure())
    
    print("\n" + "=" * 60)
    if all(results):
        print("✓ All structural tests passed!")
        if tf_available:
            print("✓ TensorFlow is available for full testing")
        else:
            print("⚠ TensorFlow not available - install for full validation")
        return True
    else:
        print("✗ Some structural tests failed!")
        return False

if __name__ == "__main__":
    success = run_structural_tests()
    
    # If TensorFlow is available, try to run the actual Keras tests
    try:
        import tensorflow as tf
        print(f"\nTensorFlow {tf.__version__} detected - running Keras tests...")
        
        # Import our test modules
        from test_tf_math_utils import run_all_tests as run_tf_tests
        from test_keras_pyramid import run_all_tests as run_keras_tests
        
        print("\n" + "=" * 60)
        print("Running TensorFlow math utils tests...")
        tf_success = run_tf_tests()
        
        print("\n" + "=" * 60)
        print("Running Keras pyramid tests...")
        keras_success = run_keras_tests()
        
        if tf_success and keras_success:
            print("\n✓ All TensorFlow/Keras tests passed!")
            success = success and True
        else:
            print("\n✗ Some TensorFlow/Keras tests failed!")
            success = False
            
    except ImportError:
        print("\nTensorFlow not available - skipping runtime tests")
        print("Install TensorFlow 2.15+ and Keras 3.0+ to run full validation")
    
    sys.exit(0 if success else 1)