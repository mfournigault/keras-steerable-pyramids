#!/usr/bin/env python3
"""
Example usage of the Keras Steerable Pyramid implementation.

This script demonstrates how to use the migrated Keras implementation
for building and reconstructing steerable pyramids.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def create_demo_image(size=128):
    """Create a demo image with interesting features."""
    x, y = np.meshgrid(np.linspace(-2, 2, size), np.linspace(-2, 2, size))
    
    # Create multiple patterns
    pattern1 = np.sin(4 * np.pi * x) * np.cos(3 * np.pi * y)
    pattern2 = np.exp(-(x**2 + y**2))
    pattern3 = np.sin(8 * np.pi * (x + y)) * 0.5
    
    # Combine patterns
    image = pattern1 * pattern2 + pattern3 * 0.3
    
    # Add some noise
    noise = np.random.randn(size, size) * 0.1
    image += noise
    
    # Normalize to [0, 1]
    image = (image - image.min()) / (image.max() - image.min())
    
    return image.astype(np.float32)

def demonstrate_numpy_implementation():
    """Demonstrate the NumPy reference implementation."""
    print("=" * 60)
    print("Demonstrating NumPy Steerable Pyramid Implementation")
    print("=" * 60)
    
    from steerable_pyramids.SCFpyr_NumPy import SCFpyr_NumPy
    
    # Define make_grid_coeff locally to avoid utils import issues
    def make_grid_coeff(coeff, normalize=True):
        """Create visualization grid of pyramid coefficients."""
        M, N = coeff[1][0].shape
        Norients = len(coeff[1])
        out = np.zeros((M * 2 - coeff[-1].shape[0], Norients * N))
        currentx, currenty = 0, 0

        for i in range(1, len(coeff[:-1])):
            for j in range(len(coeff[1])):
                if hasattr(coeff[i][j], 'real'):
                    tmp = coeff[i][j].real
                else:
                    tmp = np.real(coeff[i][j])
                m, n = tmp.shape
                if normalize:
                    tmp = 255 * tmp/tmp.max()
                tmp[m-1, :] = 255
                tmp[:, n-1] = 255
                out[currentx:currentx+m, currenty:currenty+n] = tmp
                currenty += n
            currentx += coeff[i][0].shape[0]
            currenty = 0

        m, n = coeff[-1].shape
        out[currentx: currentx+m, currenty: currenty+n] = 255 * coeff[-1]/coeff[-1].max()
        out[0, :] = 255
        out[:, 0] = 255
        return out.astype(np.uint8)
    
    # Create test image
    test_image = create_demo_image(128)
    print(f"Created test image: {test_image.shape}, range: [{test_image.min():.3f}, {test_image.max():.3f}]")
    
    # Build pyramid
    pyr = SCFpyr_NumPy(height=4, nbands=6, scale_factor=2)
    print(f"Building pyramid: height={pyr.height}, nbands={pyr.nbands}, scale_factor={pyr.scale_factor}")
    
    coeff = pyr.build(test_image)
    print(f"Pyramid built successfully with {len(coeff)} levels")
    
    # Print pyramid structure
    for i, level in enumerate(coeff):
        if isinstance(level, list):
            print(f"  Level {i}: {len(level)} orientation bands")
            for j, band in enumerate(level):
                print(f"    Band {j}: shape {band.shape}, dtype {band.dtype}")
        else:
            print(f"  Level {i}: shape {level.shape}, dtype {level.dtype}")
    
    # Reconstruct image
    reconstructed = pyr.reconstruct(coeff)
    reconstruction_error = np.mean(np.abs(test_image - reconstructed))
    print(f"Reconstruction error: {reconstruction_error:.2e}")
    
    # Create visualization
    coeff_grid = make_grid_coeff(coeff, normalize=True)
    
    # Plot results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(test_image, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(coeff_grid, cmap='gray')
    axes[1].set_title('Pyramid Coefficients')
    axes[1].axis('off')
    
    axes[2].imshow(reconstructed, cmap='gray')
    axes[2].set_title(f'Reconstructed (Error: {reconstruction_error:.2e})')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig('/tmp/numpy_steerable_pyramid_demo.png', dpi=150, bbox_inches='tight')
    print("Saved visualization to /tmp/numpy_steerable_pyramid_demo.png")
    
    return test_image, coeff, reconstructed

def demonstrate_keras_implementation():
    """Demonstrate the Keras implementation (if TensorFlow is available)."""
    print("\n" + "=" * 60)
    print("Demonstrating Keras Steerable Pyramid Implementation")
    print("=" * 60)
    
    try:
        import tensorflow as tf
        from steerable_pyramids.SCFpyr_Keras import SCFpyr_Keras, build_scf_pyramid
        from utils.utils_tf import extract_from_batch
        print(f"TensorFlow {tf.__version__} detected")
        
        # Create test image
        test_image = create_demo_image(128)
        test_batch = tf.constant(test_image[np.newaxis, :, :])  # Add batch dimension
        print(f"Created test batch: {test_batch.shape}")
        
        # Build pyramid using Keras layer
        pyr_layer = SCFpyr_Keras(height=4, nbands=6, scale_factor=2)
        print(f"Created Keras layer: height={pyr_layer.height}, nbands={pyr_layer.nbands}")
        
        coeff = pyr_layer.build_pyramid(test_batch)
        print(f"Pyramid built successfully with {len(coeff)} levels")
        
        # Print pyramid structure
        for i, level in enumerate(coeff):
            if isinstance(level, list):
                print(f"  Level {i}: {len(level)} orientation bands")
                for j, band in enumerate(level):
                    print(f"    Band {j}: shape {band.shape}, dtype {band.dtype}")
            else:
                print(f"  Level {i}: shape {level.shape}, dtype {level.dtype}")
        
        # Reconstruct image
        reconstructed_batch = pyr_layer.reconstruct(coeff)
        reconstructed = reconstructed_batch[0].numpy()  # Extract first batch element
        reconstruction_error = np.mean(np.abs(test_image - reconstructed))
        print(f"Reconstruction error: {reconstruction_error:.2e}")
        
        # Plot results
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(test_image, cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # For visualization, extract coefficients from batch
        coeff_single = extract_from_batch(coeff, 0)
        
        # Define make_grid_coeff locally 
        def make_grid_coeff(coeff, normalize=True):
            """Create visualization grid of pyramid coefficients."""
            M, N = coeff[1][0].shape
            Norients = len(coeff[1])
            out = np.zeros((M * 2 - coeff[-1].shape[0], Norients * N))
            currentx, currenty = 0, 0

            for i in range(1, len(coeff[:-1])):
                for j in range(len(coeff[1])):
                    if hasattr(coeff[i][j], 'real'):
                        tmp = coeff[i][j].real
                    else:
                        tmp = np.real(coeff[i][j])
                    m, n = tmp.shape
                    if normalize:
                        tmp = 255 * tmp/tmp.max()
                    tmp[m-1, :] = 255
                    tmp[:, n-1] = 255
                    out[currentx:currentx+m, currenty:currenty+n] = tmp
                    currenty += n
                currentx += coeff[i][0].shape[0]
                currenty = 0

            m, n = coeff[-1].shape
            out[currentx: currentx+m, currenty: currenty+n] = 255 * coeff[-1]/coeff[-1].max()
            out[0, :] = 255
            out[:, 0] = 255
            return out.astype(np.uint8)
        
        coeff_grid = make_grid_coeff(coeff_single, normalize=True)
        
        axes[1].imshow(coeff_grid, cmap='gray')
        axes[1].set_title('Pyramid Coefficients (Keras)')
        axes[1].axis('off')
        
        axes[2].imshow(reconstructed, cmap='gray')
        axes[2].set_title(f'Reconstructed (Error: {reconstruction_error:.2e})')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig('/tmp/keras_steerable_pyramid_demo.png', dpi=150, bbox_inches='tight')
        print("Saved visualization to /tmp/keras_steerable_pyramid_demo.png")
        
        return test_image, coeff, reconstructed
        
    except ImportError as e:
        print(f"TensorFlow not available: {e}")
        print("Please install TensorFlow 2.15+ and Keras 3.0+ to run Keras demo")
        return None, None, None

def compare_implementations():
    """Compare NumPy and Keras implementations if both are available."""
    print("\n" + "=" * 60)
    print("Comparing NumPy and Keras Implementations")
    print("=" * 60)
    
    try:
        import tensorflow as tf
        from steerable_pyramids.SCFpyr_NumPy import SCFpyr_NumPy
        from steerable_pyramids.SCFpyr_Keras import SCFpyr_Keras
        from utils.utils_tf import extract_from_batch
        
        # Use same test image
        test_image = create_demo_image(64)  # Smaller for comparison
        
        # NumPy implementation
        pyr_numpy = SCFpyr_NumPy(height=3, nbands=4, scale_factor=2)
        coeff_numpy = pyr_numpy.build(test_image)
        reconstructed_numpy = pyr_numpy.reconstruct(coeff_numpy)
        error_numpy = np.mean(np.abs(test_image - reconstructed_numpy))
        
        # Keras implementation
        test_batch = tf.constant(test_image[np.newaxis, :, :])
        pyr_keras = SCFpyr_Keras(height=3, nbands=4, scale_factor=2)
        coeff_keras = pyr_keras.build_pyramid(test_batch)
        reconstructed_keras_batch = pyr_keras.reconstruct(coeff_keras)
        reconstructed_keras = reconstructed_keras_batch[0].numpy()
        error_keras = np.mean(np.abs(test_image - reconstructed_keras))
        
        # Compare results
        print(f"NumPy reconstruction error: {error_numpy:.2e}")
        print(f"Keras reconstruction error: {error_keras:.2e}")
        
        # Compare coefficients
        coeff_keras_single = extract_from_batch(coeff_keras, 0)
        
        print(f"\\nCoefficient comparison:")
        for i, (coeff_np, coeff_k) in enumerate(zip(coeff_numpy, coeff_keras_single)):
            if isinstance(coeff_np, np.ndarray) and isinstance(coeff_k, np.ndarray):
                diff = np.mean(np.abs(coeff_np - coeff_k))
                print(f"  Level {i}: mean absolute difference = {diff:.2e}")
            elif isinstance(coeff_np, list) and isinstance(coeff_k, list):
                for j, (band_np, band_k) in enumerate(zip(coeff_np, coeff_k)):
                    if np.iscomplexobj(band_np) and np.iscomplexobj(band_k):
                        diff = np.mean(np.abs(band_np - band_k))
                        print(f"  Level {i}, Band {j}: mean absolute difference = {diff:.2e}")
        
        # Overall comparison
        implementation_diff = np.mean(np.abs(reconstructed_numpy - reconstructed_keras))
        print(f"\\nOverall reconstruction difference: {implementation_diff:.2e}")
        
        if implementation_diff < 1e-3:
            print("✓ Implementations are numerically consistent!")
        else:
            print("⚠ Implementations show significant differences")
            
    except ImportError:
        print("TensorFlow not available - cannot compare implementations")

def main():
    """Main demonstration function."""
    print("Steerable Pyramid Implementation Demo")
    print("This demo shows both NumPy and Keras implementations")
    
    # Always run NumPy demo
    numpy_results = demonstrate_numpy_implementation()
    
    # Try to run Keras demo
    keras_results = demonstrate_keras_implementation()
    
    # Compare if both are available
    if keras_results[0] is not None:
        compare_implementations()
    
    print("\\n" + "=" * 60)
    print("Demo completed!")
    print("Check /tmp/ for saved visualization images")

if __name__ == "__main__":
    main()