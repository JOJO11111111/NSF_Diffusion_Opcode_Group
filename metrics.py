"""
Evaluation metrics for all models
"""

from functools import partial

import keras.backend as K
import numpy as np


def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)


def partial_loss(interpolated_sample):
    partial_gp_loss = partial(
        gradient_penalty_loss, averaged_samples=interpolated_sample
    )
    partial_gp_loss.__name__ = 'gradient_penalty'
    return partial_gp_loss


def gradient_penalty_loss(y_true, y_pred, averaged_samples):
    """
    L2 norm of Eucledian norm calculation:
    1. Square the gradients
    2. L2 Norm = Sum over rows of squared gradients and take square root
    3. Gradient penalty = lambda*(l2_norm - 1)^2
    """

    # Get gradients wrt interpolated samples
    gradients = K.gradients(y_pred, averaged_samples)[0]

    # Square the gradients
    grad_squared = K.square(gradients)

    # Sum the squared gradients along each row and take square root
    grad_norm = K.sqrt(K.sum(grad_squared,
                             axis=np.arange(1, len(grad_squared.shape))))

    # Calculate penalty
    grad_penalty = K.square(1 - grad_norm)

    # Return the mean penalty accross the rows
    return K.mean(grad_penalty)