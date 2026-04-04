#!/usr/bin/env python3
"""
Module for forward propagation in a convolutional layer
"""
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """
    Performs forward propagation over a convolutional layer
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, c_new = W.shape
    sh, sw = stride

    if padding == "same":
        ph = int(np.ceil(((h_prev - 1) * sh + kh - h_prev) / 2))
        pw = int(np.ceil(((w_prev - 1) * sw + kw - w_prev) / 2))
    else:
        ph, pw = 0, 0

    A_padded = np.pad(A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                      mode='constant', constant_values=0)

    h_out = int((h_prev + 2 * ph - kh) / sh) + 1
    w_out = int((w_prev + 2 * pw - kw) / sw) + 1

    output = np.zeros((m, h_out, w_out, c_new))

    for h in range(h_out):
        for w in range(w_out):
            h_start = h * sh
            h_end = h_start + kh
            w_start = w * sw
            w_end = w_start + kw

            # Extract the slice for all examples at once
            a_slice = A_padded[:, h_start:h_end, w_start:w_end, :]

            # Matrix multiplication equivalent for convolution
            # Reshape slice to (m, kh*kw*c_prev) and W to (kh*kw*c_prev, c_new)
            # This is faster and avoids deep nesting
            res = np.tensordot(a_slice, W, axes=((1, 2, 3), (0, 1, 2)))
            output[:, h, w, :] = res

    return activation(output + b)
