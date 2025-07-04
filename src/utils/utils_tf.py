# MIT License
#
# Copyright (c) 2018 Tom Runia
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Tom Runia
# Date Created: 2018-12-10
# Modified for TensorFlow/Keras by migration script

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

try:
    import skimage
    import skimage.io
except ImportError:
    print("Warning: scikit-image not available, image loading functions may not work")
    skimage = None

################################################################################

def get_device(device='gpu:0'):
    """Get TensorFlow device.
    
    Args:
        device: Device string (e.g., 'gpu:0', 'cpu:0')
        
    Returns:
        Device context or None for default placement
    """
    if isinstance(device, str):
        if 'gpu' in device or 'cuda' in device:
            # Check if GPU is available
            if tf.config.list_physical_devices('GPU'):
                return device.replace('cuda', 'gpu')  # Convert CUDA notation to TF notation
            else:
                print('No GPU devices found, falling back to CPU')
                return 'cpu:0'
        elif 'cpu' in device:
            return 'cpu:0'
    
    return None  # Use default device placement

def load_image_batch(image_file, batch_size, image_size=200):
    """Load and preprocess image batch.
    
    Args:
        image_file: Path to image file
        batch_size: Number of images in batch
        image_size: Target image size
        
    Returns:
        Preprocessed image batch as numpy array
    """
    if not os.path.isfile(image_file):
        raise FileNotFoundError('Image file not found on disk: {}'.format(image_file))
    
    if skimage is None:
        raise ImportError("scikit-image is required for image loading")
    
    # Load image
    im = skimage.io.imread(image_file)
    
    # Convert to grayscale if needed
    if len(im.shape) == 3:
        im = np.mean(im, axis=-1)
    
    # Resize using TensorFlow
    im_tf = tf.image.resize(
        tf.expand_dims(tf.cast(im, tf.float32), -1),
        [image_size, image_size]
    )
    im_tf = tf.squeeze(im_tf, -1)
    
    # Create batch with random crops
    im_batch = np.zeros((batch_size, image_size, image_size), np.float32)
    for i in range(batch_size):
        # For simplicity, just use the resized image
        # In practice, you might want to implement random cropping
        im_batch[i] = im_tf.numpy()
    
    # Add channel dimension and rescale
    return im_batch[:, None, :, :] / 255.0

def load_image_batch_tf(image_file, batch_size, image_size=200):
    """Load and preprocess image batch using TensorFlow ops.
    
    Args:
        image_file: Path to image file
        batch_size: Number of images in batch
        image_size: Target image size
        
    Returns:
        Preprocessed image batch as TensorFlow tensor
    """
    # Read image file
    image_raw = tf.io.read_file(image_file)
    image = tf.image.decode_image(image_raw, channels=1, dtype=tf.float32)
    
    # Resize image
    image = tf.image.resize(image, [image_size, image_size])
    image = tf.squeeze(image, -1)  # Remove channel dim
    
    # Create batch (repeat the same image)
    # In practice, you might want different augmentations
    image_batch = tf.stack([image] * batch_size)
    
    # Add channel dimension
    image_batch = tf.expand_dims(image_batch, 1)
    
    return image_batch

def show_image_batch(im_batch):
    """Visualize image batch.
    
    Args:
        im_batch: Image batch tensor (numpy or TensorFlow)
        
    Returns:
        Visualization array
    """
    if tf.is_tensor(im_batch):
        im_batch = im_batch.numpy()
    
    # Create grid visualization
    batch_size = im_batch.shape[0]
    if batch_size == 1:
        im_vis = im_batch[0].squeeze()
    else:
        # Simple grid layout
        grid_size = int(np.ceil(np.sqrt(batch_size)))
        im_vis = np.zeros((
            grid_size * im_batch.shape[-2],
            grid_size * im_batch.shape[-1]
        ))
        
        for i in range(batch_size):
            row = i // grid_size
            col = i % grid_size
            h_start = row * im_batch.shape[-2]
            h_end = h_start + im_batch.shape[-2]
            w_start = col * im_batch.shape[-1]
            w_end = w_start + im_batch.shape[-1]
            im_vis[h_start:h_end, w_start:w_end] = im_batch[i].squeeze()
    
    plt.imshow(im_vis, cmap='gray')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    return im_vis

def extract_from_batch(coeff_batch, example_idx=0):
    """Extract coefficients for a single example from the batch.
    
    Args:
        coeff_batch: List containing batch of coefficients
        example_idx: Index of example to extract
        
    Returns:
        List containing coefficients for single example
    """
    if not isinstance(coeff_batch, list):
        raise ValueError('Batch of coefficients must be a list')
    
    coeff = []  # coefficients for single example
    for coeff_level in coeff_batch:
        if tf.is_tensor(coeff_level):
            # Low- or High-Pass
            coeff_level_numpy = coeff_level[example_idx].numpy()
            coeff.append(coeff_level_numpy)
        elif isinstance(coeff_level, list):
            coeff_orientations_numpy = []
            for coeff_orientation in coeff_level:
                if tf.is_tensor(coeff_orientation):
                    coeff_orientation_numpy = coeff_orientation[example_idx].numpy()
                    # Handle complex tensors
                    if coeff_orientation_numpy.dtype == np.complex64 or coeff_orientation_numpy.dtype == np.complex128:
                        coeff_orientations_numpy.append(coeff_orientation_numpy)
                    else:
                        # Assume last dimension contains real/imaginary parts
                        if len(coeff_orientation_numpy.shape) > 2 and coeff_orientation_numpy.shape[-1] == 2:
                            coeff_orientation_complex = coeff_orientation_numpy[:, :, 0] + 1j * coeff_orientation_numpy[:, :, 1]
                            coeff_orientations_numpy.append(coeff_orientation_complex)
                        else:
                            coeff_orientations_numpy.append(coeff_orientation_numpy)
                else:
                    coeff_orientations_numpy.append(coeff_orientation)
            coeff.append(coeff_orientations_numpy)
        else:
            raise ValueError('coeff level must be of type (list, tf.Tensor)')
    return coeff

################################################################################

def make_grid_coeff(coeff, normalize=True):
    """Visualization function for building a large image containing all pyramid levels.
    
    Args:
        coeff: Complex pyramid stored as list containing all levels
        normalize: Whether to normalize each band
        
    Returns:
        Large image containing grid of all bands and orientations
    """
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

################################################################################
# TensorFlow-specific utilities

def ensure_complex64(x):
    """Ensure tensor is complex64.
    
    Args:
        x: Input tensor
        
    Returns:
        Complex64 tensor
    """
    if tf.is_tensor(x):
        if x.dtype in [tf.complex64, tf.complex128]:
            return tf.cast(x, tf.complex64)
        else:
            # Assume real tensor, convert to complex
            return tf.complex(tf.cast(x, tf.float32), tf.zeros_like(x, dtype=tf.float32))
    else:
        return tf.constant(x, dtype=tf.complex64)

def device_context(device_name):
    """Create device context for TensorFlow operations.
    
    Args:
        device_name: Device name string
        
    Returns:
        Context manager for device placement
    """
    if device_name is None:
        # Return a dummy context manager
        from contextlib import nullcontext
        return nullcontext()
    else:
        return tf.device(device_name)