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
# Modified for Keras 3 + TensorFlow 2 by migration script

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import keras
from math import factorial

try:
    # Try importing the new math utils
    from utils.math_utils_tf import *
    from utils.math_utils import pointOp, prepare_grid, rcosFn
except ImportError:
    # Fallback to local imports
    import sys
    import os
    sys.path.append(os.path.dirname(__file__))
    from math_utils_tf import *
    from math_utils import pointOp, prepare_grid, rcosFn

################################################################################
################################################################################

class SCFpyr_Keras(keras.layers.Layer):
    """
    Complex Steerable Pyramid implementation using Keras 3 + TensorFlow 2.
    
    This is a modified version of buildSFpyr, that constructs a
    complex-valued steerable pyramid using Hilbert-transform pairs
    of filters. Note that the imaginary parts will *not* be steerable.

    Description of this transform appears in: Portilla & Simoncelli,
    International Journal of Computer Vision, 40(1):49-71, Oct 2000.
    Further information: http://www.cns.nyu.edu/~eero/STEERPYR/

    Modified code from the perceptual repository:
      https://github.com/andreydung/Steerable-filter

    This code looks very similar to the original Matlab code:
      https://github.com/LabForComputationalVision/matlabPyrTools/blob/master/buildSCFpyr.m

    Also looks very similar to the original Python code presented here:
      https://github.com/LabForComputationalVision/pyPyrTools/blob/master/pyPyrTools/SCFpyr.py
    """

    def __init__(self, height=5, nbands=4, scale_factor=2, **kwargs):
        """Initialize the Steerable Pyramid layer.
        
        Args:
            height: Number of pyramid levels (including low-pass and high-pass)
            nbands: Number of orientation bands
            scale_factor: Scale factor between pyramid levels
            **kwargs: Additional keyword arguments for the Layer
        """
        super(SCFpyr_Keras, self).__init__(**kwargs)
        
        self.height = height
        self.nbands = nbands
        self.scale_factor = scale_factor

        # Cache constants
        self.lutsize = 1024
        self.Xcosn = np.pi * np.array(range(-(2*self.lutsize+1), (self.lutsize+2)))/self.lutsize
        self.alpha = (self.Xcosn + np.pi) % (2*np.pi) - np.pi
        
        # Complex factors for construction and reconstruction
        self.complex_fact_construct = np.power(np.complex(0, -1), self.nbands-1)
        self.complex_fact_reconstruct = np.power(np.complex(0, 1), self.nbands-1)
        
        # Convert to TensorFlow constants
        self.complex_fact_construct_tf = tf.constant(self.complex_fact_construct, dtype=tf.complex64)
        self.complex_fact_reconstruct_tf = tf.constant(self.complex_fact_reconstruct, dtype=tf.complex64)

    def call(self, inputs, training=None):
        """Forward pass - build the steerable pyramid.
        
        Args:
            inputs: Input tensor of shape [N, C, H, W] or [N, H, W]
            training: Training mode flag
            
        Returns:
            List containing pyramid coefficients
        """
        return self.build(inputs)

    def build_pyramid(self, inputs):
        """Public interface for building the pyramid.
        
        Args:
            inputs: Input tensor
            
        Returns:
            Pyramid coefficients
        """
        return self.build(inputs)

    def build(self, im_batch):
        """Decomposes a batch of images into a complex steerable pyramid.
        
        Args:
            im_batch: Batch of images of shape [N, C, H, W] or [N, H, W]
        
        Returns:
            pyramid: list containing tf.Tensor objects storing the pyramid
        """
        # Handle input dimensions
        if len(im_batch.shape) == 3:
            # Add channel dimension if missing
            im_batch = tf.expand_dims(im_batch, 1)
        elif len(im_batch.shape) == 4:
            # Squeeze channel dimension if single channel
            if im_batch.shape[1] == 1:
                im_batch = tf.squeeze(im_batch, 1)
        
        # Ensure float32
        im_batch = tf.cast(im_batch, tf.float32)
        
        height, width = tf.shape(im_batch)[1], tf.shape(im_batch)[2]
        
        # Check whether image size is sufficient for number of levels
        min_size = tf.minimum(width, height)
        max_levels = tf.cast(tf.floor(tf.math.log(tf.cast(min_size, tf.float32)) / tf.math.log(2.0)) - 2, tf.int32)
        tf.debugging.assert_greater_equal(
            max_levels, self.height,
            message=f'Cannot build {self.height} levels, image too small.'
        )
        
        # Prepare a grid
        log_rad, angle = prepare_grid(height.numpy(), width.numpy())

        # Radial transition function (a raised cosine in log-frequency):
        Xrcos, Yrcos = rcosFn(1, -0.5)
        Yrcos = np.sqrt(Yrcos)

        YIrcos = np.sqrt(1 - Yrcos**2)

        lo0mask = pointOp(log_rad, YIrcos, Xrcos)
        hi0mask = pointOp(log_rad, Yrcos, Xrcos)

        # Convert to TensorFlow tensors with proper broadcasting dimensions
        lo0mask = tf.constant(lo0mask, dtype=tf.float32)
        hi0mask = tf.constant(hi0mask, dtype=tf.float32)
        
        # Add batch and channel dimensions for broadcasting
        lo0mask = tf.expand_dims(tf.expand_dims(lo0mask, 0), -1)
        hi0mask = tf.expand_dims(tf.expand_dims(hi0mask, 0), -1)

        # Fourier transform (2D) and shifting
        batch_dft = tf.signal.fft2d(tf.cast(im_batch, tf.complex64))
        batch_dft = batch_fftshift2d(batch_dft)

        # Low-pass
        lo0dft = batch_dft * tf.cast(lo0mask, tf.complex64)

        # Start recursively building the pyramids
        coeff = self._build_levels(lo0dft, log_rad, angle, Xrcos, Yrcos, self.height-1)

        # High-pass
        hi0dft = batch_dft * tf.cast(hi0mask, tf.complex64)
        hi0 = batch_ifftshift2d(hi0dft)
        hi0 = tf.signal.ifft2d(hi0)
        hi0_real = tf.math.real(hi0)
        coeff.insert(0, hi0_real)
        
        return coeff

    def _build_levels(self, lodft, log_rad, angle, Xrcos, Yrcos, height):
        """Recursively build pyramid levels.
        
        Args:
            lodft: Low-pass DFT coefficients
            log_rad: Logarithmic radial grid
            angle: Angular grid
            Xrcos: X values for raised cosine
            Yrcos: Y values for raised cosine
            height: Remaining levels to build
            
        Returns:
            List of pyramid coefficients
        """
        if height <= 1:
            # Low-pass
            lo0 = batch_ifftshift2d(lodft)
            lo0 = tf.signal.ifft2d(lo0)
            lo0_real = tf.math.real(lo0)
            coeff = [lo0_real]
        else:
            Xrcos = Xrcos - np.log2(self.scale_factor)

            ####################################################################
            ####################### Orientation bandpass ######################
            ####################################################################

            himask = pointOp(log_rad, Yrcos, Xrcos)
            himask = tf.constant(himask, dtype=tf.float32)
            himask = tf.expand_dims(tf.expand_dims(himask, 0), -1)

            order = self.nbands - 1
            const = np.power(2, 2*order) * np.square(factorial(order)) / (self.nbands * factorial(2*order))
            Ycosn = 2*np.sqrt(const) * np.power(np.cos(self.Xcosn), order) * (np.abs(self.alpha) < np.pi/2)

            # Loop through all orientation bands
            orientations = []
            for b in range(self.nbands):
                anglemask = pointOp(angle, Ycosn, self.Xcosn + np.pi*b/self.nbands)
                anglemask = tf.constant(anglemask, dtype=tf.float32)
                anglemask = tf.expand_dims(tf.expand_dims(anglemask, 0), -1)

                # Bandpass filtering                
                banddft = lodft * tf.cast(anglemask, tf.complex64) * tf.cast(himask, tf.complex64)

                # Multiply with complex number
                # (x+yi)(u+vi) = (xu-yv) + (xv+yu)i
                banddft = banddft * self.complex_fact_construct_tf

                band = batch_ifftshift2d(banddft)
                band = tf.signal.ifft2d(band)
                orientations.append(band)

            ####################################################################
            ######################## Subsample lowpass #########################
            ####################################################################

            # Get current dimensions
            dims = tf.shape(lodft)[1:3]  # Height and width
            dims_np = dims.numpy()

            # Both are tuples of size 2
            low_ind_start = (np.ceil((dims_np+0.5)/2) - np.ceil((np.ceil((dims_np-0.5)/2)+0.5)/2)).astype(int)
            low_ind_end = (low_ind_start + np.ceil((dims_np-0.5)/2)).astype(int)

            # Subsampling indices
            log_rad = log_rad[low_ind_start[0]:low_ind_end[0], low_ind_start[1]:low_ind_end[1]]
            angle = angle[low_ind_start[0]:low_ind_end[0], low_ind_start[1]:low_ind_end[1]]

            # Actual subsampling
            lodft = lodft[:, low_ind_start[0]:low_ind_end[0], low_ind_start[1]:low_ind_end[1]]

            # Filtering
            YIrcos = np.abs(np.sqrt(1 - Yrcos**2))
            lomask = pointOp(log_rad, YIrcos, Xrcos)
            lomask = tf.constant(lomask, dtype=tf.float32)
            lomask = tf.expand_dims(tf.expand_dims(lomask, 0), -1)

            # Convolution in spatial domain
            lodft = tf.cast(lomask, tf.complex64) * lodft

            ####################################################################
            ####################### Recursion next level ######################
            ####################################################################

            coeff = self._build_levels(lodft, log_rad, angle, Xrcos, Yrcos, height-1)
            coeff.insert(0, orientations)

        return coeff

    def reconstruct(self, coeff):
        """Reconstruct image from pyramid coefficients.
        
        Args:
            coeff: Pyramid coefficients from build()
            
        Returns:
            Reconstructed image tensor
        """
        if self.nbands != len(coeff[1]):
            raise Exception("Unmatched number of orientations")

        height, width = tf.shape(coeff[0])[1], tf.shape(coeff[0])[2]
        log_rad, angle = prepare_grid(height.numpy(), width.numpy())

        Xrcos, Yrcos = rcosFn(1, -0.5)
        Yrcos = np.sqrt(Yrcos)
        YIrcos = np.sqrt(np.abs(1 - Yrcos**2))

        lo0mask = pointOp(log_rad, YIrcos, Xrcos)
        hi0mask = pointOp(log_rad, Yrcos, Xrcos)

        # Convert to TensorFlow tensors with proper broadcasting dimensions
        lo0mask = tf.constant(lo0mask, dtype=tf.float32)
        hi0mask = tf.constant(hi0mask, dtype=tf.float32)
        lo0mask = tf.expand_dims(tf.expand_dims(lo0mask, 0), -1)
        hi0mask = tf.expand_dims(tf.expand_dims(hi0mask, 0), -1)

        # Start recursive reconstruction
        tempdft = self._reconstruct_levels(coeff[1:], log_rad, Xrcos, Yrcos, angle)

        hidft = tf.signal.fft2d(tf.cast(coeff[0], tf.complex64))
        hidft = batch_fftshift2d(hidft)

        outdft = tempdft * tf.cast(lo0mask, tf.complex64) + hidft * tf.cast(hi0mask, tf.complex64)

        reconstruction = batch_ifftshift2d(outdft)
        reconstruction = tf.signal.ifft2d(reconstruction)
        reconstruction = tf.math.real(reconstruction)  # Take real part

        return reconstruction

    def _reconstruct_levels(self, coeff, log_rad, Xrcos, Yrcos, angle):
        """Recursively reconstruct pyramid levels.
        
        Args:
            coeff: Remaining coefficients to reconstruct
            log_rad: Logarithmic radial grid
            Xrcos: X values for raised cosine
            Yrcos: Y values for raised cosine
            angle: Angular grid
            
        Returns:
            Reconstructed DFT coefficients
        """
        if len(coeff) == 1:
            dft = tf.signal.fft2d(tf.cast(coeff[0], tf.complex64))
            dft = batch_fftshift2d(dft)
            return dft

        Xrcos = Xrcos - np.log2(self.scale_factor)

        ####################################################################
        ####################### Orientation Residue #######################
        ####################################################################

        himask = pointOp(log_rad, Yrcos, Xrcos)
        himask = tf.constant(himask, dtype=tf.float32)
        himask = tf.expand_dims(tf.expand_dims(himask, 0), -1)

        lutsize = 1024
        Xcosn = np.pi * np.array(range(-(2*lutsize+1), (lutsize+2)))/lutsize
        order = self.nbands - 1
        const = np.power(2, 2*order) * np.square(factorial(order)) / (self.nbands * factorial(2*order))
        Ycosn = np.sqrt(const) * np.power(np.cos(Xcosn), order)

        orientdft = tf.zeros_like(coeff[0][0], dtype=tf.complex64)
        for b in range(self.nbands):
            anglemask = pointOp(angle, Ycosn, Xcosn + np.pi * b/self.nbands)
            anglemask = tf.constant(anglemask, dtype=tf.float32)
            anglemask = tf.expand_dims(tf.expand_dims(anglemask, 0), -1)

            banddft = tf.signal.fft2d(tf.cast(coeff[0][b], tf.complex64))
            banddft = batch_fftshift2d(banddft)

            banddft = banddft * tf.cast(anglemask, tf.complex64) * tf.cast(himask, tf.complex64)
            banddft = banddft * self.complex_fact_reconstruct_tf

            orientdft = orientdft + banddft

        ####################################################################
        ########## Lowpass component are upsampled and convoluted ##########
        ####################################################################
        
        dims = tf.shape(coeff[0][0])[1:3]
        dims_np = dims.numpy()
        
        lostart = (np.ceil((dims_np+0.5)/2) - np.ceil((np.ceil((dims_np-0.5)/2)+0.5)/2)).astype(np.int32)
        loend = lostart + np.ceil((dims_np-0.5)/2).astype(np.int32)

        nlog_rad = log_rad[lostart[0]:loend[0], lostart[1]:loend[1]]
        nangle = angle[lostart[0]:loend[0], lostart[1]:loend[1]]
        YIrcos = np.sqrt(np.abs(1 - Yrcos**2))
        lomask = pointOp(nlog_rad, YIrcos, Xrcos)

        # Filtering
        lomask = tf.constant(lomask, dtype=tf.float32)
        lomask = tf.expand_dims(tf.expand_dims(lomask, 0), -1)

        ################################################################################

        # Recursive call for image reconstruction        
        nresdft = self._reconstruct_levels(coeff[1:], nlog_rad, Xrcos, Yrcos, nangle)

        # Create result tensor and place upsampled content
        resdft = tf.zeros_like(coeff[0][0], dtype=tf.complex64)
        
        # Use tf.tensor_scatter_nd_update for placing the upsampled content
        batch_size = tf.shape(nresdft)[0]
        indices = []
        updates = []
        
        for b in range(batch_size):
            for h in range(lostart[0], loend[0]):
                for w in range(lostart[1], loend[1]):
                    indices.append([b, h, w])
                    h_idx = h - lostart[0]
                    w_idx = w - lostart[1]
                    updates.append(nresdft[b, h_idx, w_idx] * tf.cast(lomask[0, h_idx, w_idx], tf.complex64))
        
        if indices:  # Only update if there are indices
            indices = tf.constant(indices)
            updates = tf.stack(updates)
            resdft = tf.tensor_scatter_nd_update(resdft, indices, updates)

        return resdft + orientdft

    def get_config(self):
        """Get layer configuration for serialization."""
        config = super(SCFpyr_Keras, self).get_config()
        config.update({
            'height': self.height,
            'nbands': self.nbands,
            'scale_factor': self.scale_factor,
        })
        return config

################################################################################
# Convenience functions

def build_scf_pyramid(image, height=5, nbands=4, scale_factor=2):
    """Convenience function to build a steerable pyramid.
    
    Args:
        image: Input image tensor
        height: Number of pyramid levels
        nbands: Number of orientation bands
        scale_factor: Scale factor between levels
        
    Returns:
        Pyramid coefficients
    """
    pyramid = SCFpyr_Keras(height=height, nbands=nbands, scale_factor=scale_factor)
    return pyramid.build(image)

def reconstruct_scf_pyramid(coeff, height=5, nbands=4, scale_factor=2):
    """Convenience function to reconstruct from pyramid coefficients.
    
    Args:
        coeff: Pyramid coefficients
        height: Number of pyramid levels
        nbands: Number of orientation bands
        scale_factor: Scale factor between levels
        
    Returns:
        Reconstructed image
    """
    pyramid = SCFpyr_Keras(height=height, nbands=nbands, scale_factor=scale_factor)
    return pyramid.reconstruct(coeff)