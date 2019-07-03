# utilities for neural networks

import gin
import numpy as np


@gin.configurable
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


@gin.configurable
def relu(x):
    return np.maximum(x, 0, x)
