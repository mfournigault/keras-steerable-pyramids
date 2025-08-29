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
# Date Created: 2018-12-04
# Modified for TensorFlow/Keras by migration script

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

################################################################################
################################################################################

def roll_n(X, axis, n):
    """Roll tensor X along specified axis by n positions.
    
    Args:
        X: Input tensor
        axis: Axis along which to roll
        n: Number of positions to roll
        
    Returns:
        Rolled tensor
    """
    return tf.roll(X, shift=n, axis=axis)

def batch_fftshift2d(x):
    """Apply 2D FFT shift to batched complex tensors.
    
    Args:
        x: Complex tensor of shape [..., H, W] (complex64 or complex128)
        
    Returns:
        FFT-shifted complex tensor
    """
    # Get the real and imaginary parts
    real = tf.math.real(x)
    imag = tf.math.imag(x)
    
    # Apply shift to spatial dimensions (assuming last 2 dims are spatial)
    ndims = len(x.shape)
    for dim in range(ndims - 2, ndims):  # Last 2 dimensions (H, W)
        n_shift = tf.shape(real)[dim] // 2
        # For odd-sized images
        if tf.shape(real)[dim] % 2 != 0:
            n_shift += 1
        real = tf.roll(real, shift=n_shift, axis=dim)
        imag = tf.roll(imag, shift=n_shift, axis=dim)
    
    # Combine back to complex tensor
    return tf.complex(real, imag)

def batch_ifftshift2d(x):
    """Apply 2D inverse FFT shift to batched complex tensors.
    
    Args:
        x: Complex tensor of shape [..., H, W] (complex64 or complex128)
        
    Returns:
        Inverse FFT-shifted complex tensor
    """
    # Get the real and imaginary parts
    real = tf.math.real(x)
    imag = tf.math.imag(x)
    
    # Apply inverse shift to spatial dimensions (reverse order)
    ndims = len(x.shape)
    for dim in range(ndims - 1, ndims - 3, -1):  # Last 2 dimensions in reverse
        n_shift = tf.shape(real)[dim] // 2
        real = tf.roll(real, shift=n_shift, axis=dim)
        imag = tf.roll(imag, shift=n_shift, axis=dim)
    
    # Combine back to complex tensor
    return tf.complex(real, imag)

################################################################################
################################################################################

def prepare_grid(m, n):
    """Prepare frequency grid for steerable pyramid.
    
    Args:
        m: Height of the grid
        n: Width of the grid
        
    Returns:
        log_rad: Logarithmic radial frequency grid
        angle: Angular frequency grid
    """
    x = np.linspace(-(m // 2)/(m / 2), (m // 2)/(m / 2) - (1 - m % 2)*2/m, num=m)
    y = np.linspace(-(n // 2)/(n / 2), (n // 2)/(n / 2) - (1 - n % 2)*2/n, num=n)
    xv, yv = np.meshgrid(y, x)
    angle = np.arctan2(yv, xv)
    rad = np.sqrt(xv**2 + yv**2)
    rad[m//2][n//2] = rad[m//2][n//2 - 1]
    log_rad = np.log2(rad)
    return log_rad, angle

def rcosFn(width, position):
    """Raised cosine function for steerable pyramid filters.
    
    Args:
        width: Width parameter
        position: Position parameter
        
    Returns:
        X: X values
        Y: Y values (raised cosine)
    """
    N = 256  # arbitrary
    X = np.pi * np.array(range(-(2*N+1), (N+2)))/2/N
    Y = np.cos(X)**2
    Y[0] = Y[1]
    Y[N+2] = Y[N+1]
    X = position + 2*width/np.pi*(X + np.pi/4)
    return X, Y

def pointOp(im, Y, X):
    """Point operation for interpolation.
    
    Args:
        im: Input image/array
        Y: Y values for interpolation
        X: X values for interpolation
        
    Returns:
        Interpolated result
    """
    out = np.interp(im.flatten(), X, Y)
    return np.reshape(out, im.shape)

def getlist(coeff):
    """Flatten pyramid coefficients into a list.
    
    Args:
        coeff: Pyramid coefficients
        
    Returns:
        Flattened list of coefficients
    """
    straight = [bands for scale in coeff[1:-1] for bands in scale]
    straight = [coeff[0]] + straight + [coeff[-1]]
    return straight

################################################################################
# TensorFlow FFT utilities

def tf_fft2d(x):
    """2D FFT for real inputs.
    
    Args:
        x: Real tensor of shape [..., H, W]
        
    Returns:
        Complex tensor after 2D FFT
    """
    return tf.signal.fft2d(tf.cast(x, tf.complex64))

def tf_ifft2d(x):
    """2D inverse FFT for complex inputs.
    
    Args:
        x: Complex tensor of shape [..., H, W]
        
    Returns:
        Complex tensor after 2D inverse FFT
    """
    return tf.signal.ifft2d(x)

def tf_fftshift2d(x):
    """2D FFT shift using TensorFlow operations.
    
    Args:
        x: Input tensor (real or complex)
        
    Returns:
        FFT-shifted tensor
    """
    # For 2D, shift both spatial dimensions
    ndims = len(x.shape)
    shifts = []
    for dim in range(ndims - 2, ndims):
        shifts.append(tf.shape(x)[dim] // 2)
    
    # Apply shifts to the last 2 dimensions
    axes = list(range(ndims - 2, ndims))
    return tf.roll(x, shift=shifts, axis=axes)

def tf_ifftshift2d(x):
    """2D inverse FFT shift using TensorFlow operations.
    
    Args:
        x: Input tensor (real or complex)
        
    Returns:
        Inverse FFT-shifted tensor
    """
    # For 2D, shift both spatial dimensions (negative shifts)
    ndims = len(x.shape)
    shifts = []
    for dim in range(ndims - 2, ndims):
        shifts.append(-(tf.shape(x)[dim] // 2))
    
    # Apply shifts to the last 2 dimensions
    axes = list(range(ndims - 2, ndims))
    return tf.roll(x, shift=shifts, axis=axes)

################################################################################
# NumPy reference implementation (fftshift and ifftshift) - kept for reference

# def fftshift(x, axes=None):
#     """
#     Shift the zero-frequency component to the center of the spectrum.
#     This function swaps half-spaces for all axes listed (defaults to all).
#     Note that ``y[0]`` is the Nyquist component only if ``len(x)`` is even.
#     Parameters
#     """
#     x = np.asarray(x)
#     if axes is None:
#         axes = tuple(range(x.ndim))
#         shift = [dim // 2 for dim in x.shape]
#     shift = [x.shape[ax] // 2 for ax in axes]
#     return np.roll(x, shift, axes)
#
# def ifftshift(x, axes=None):
#     """
#     The inverse of `fftshift`. Although identical for even-length `x`, the
#     functions differ by one sample for odd-length `x`.
#     """
#     x = np.asarray(x)
#     if axes is None:
#         axes = tuple(range(x.ndim))
#         shift = [-(dim // 2) for dim in x.shape]
#     shift = [-(x.shape[ax] // 2) for ax in axes]
#     return np.roll(x, shift, axes)