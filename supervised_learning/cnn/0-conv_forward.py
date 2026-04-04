#!/usr/bin/env python3
"""
Convolutional Forward Propagation module
"""
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """
    Performs forward propagation over a convolutional layer of a neural network
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, c_new = W.shape
    sh, sw = stride

    if padding == "same":
        # Calculate padding to keep same dimensions
        ph = ((h_prev - 1) * sh + kh - h_prev) // 2
        pw = ((w_prev - 1) * sw + kw - w_prev) // 2
    else:
        ph, pw = 0, 0

    # Apply padding
    A_padded = np.pad(A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                      mode='constant', constant_values=0)

    # Output dimensions
    h_out = ((h_prev + 2 * ph - kh) // sh) + 1
    w_out = ((w_prev + 2 * pw - kw) // sw) + 1

    # Initialize output
    Z = np.zeros((m, h_out, w_out, c_new))

    # Convolution loop
    for i in range(h_out):
        for j in range(w_out):
            # Extract slice
            h_start, h_end = i * sh, i * sh + kh
            w_start, w_end = j * sw, j * sw + kw
            a_slice = A_padded[:, h_start:h_end, w_start:w_end, :]
            
            # Element-wise multiplication and summation over h, w, c_prev
            # Result shape will be (m, c_new)
            Z[:, i, j, :] = np.sum(a_slice[:, :, :, :, np.newaxis] * W[np.newaxis, :, :, :, :], 
                                   axis=(1, 2, 3))

    return activation(Z + b)
