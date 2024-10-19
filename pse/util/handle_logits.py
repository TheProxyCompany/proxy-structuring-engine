import numpy as np
import logging
from typing import List, Tuple
logger = logging.getLogger(__name__)

# Attempt to import optional dependencies
try:
    import mlx.core as mx

    _has_mlx = True
except ImportError:
    _has_mlx = False

try:
    import jax.numpy as jnp

    _has_jax = True
except ImportError:
    _has_jax = False

try:
    import torch

    _has_torch = True
except ImportError:
    _has_torch = False

def handle_logits(logits, top_k: int = 64) -> List[Tuple[int, float]]:
    """
    Returns the top_k logits and their corresponding token ids.

    This function dispatches to the appropriate implementation based on the type of `logits`.

    Args:
        logits: The logits array of shape (vocab_size,), which can be an array from MLX, NumPy, JAX, or PyTorch.
        top_k (int): The number of top tokens to return.

    Returns:
        A tuple of (top_k_indices, top_k_values), both arrays of length top_k.

    Raises:
        ValueError: If `logits` is not a 1-dimensional array or if `top_k` is not a positive integer.
        TypeError: If `logits` is not an instance of one of the supported array types.
    """
    if _has_mlx and isinstance(logits, mx.array):
        indices, values = handle_logits_mlx(logits, top_k)
    elif isinstance(logits, np.ndarray):
        indices, values = handle_logits_numpy(logits, top_k)
    elif _has_jax and isinstance(logits, jnp.ndarray):
        indices, values = handle_logits_jax(logits, top_k)
    elif _has_torch and isinstance(logits, torch.Tensor):
        indices, values = handle_logits_pytorch(logits, top_k)
    else:
        raise TypeError(f"Unsupported array type for logits: {type(logits)}")

    return [(int(idx), float(score)) for idx, score in zip(indices, values)]

def handle_logits_mlx(logits, top_k: int):
    """
    Implementation using MLX arrays optimized for large vocabularies.
    """
    if not _has_mlx:
        raise ImportError("MLX module is not installed. Please install it with 'pip install mlx'.")

    if not isinstance(logits, mx.array):
        raise TypeError("Expected logits to be an instance of mx.array.")

    if logits.ndim != 1:
        raise ValueError("Logits must be a 1-dimensional array.")

    vocab_size = logits.shape[0]

    if not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")

    top_k = min(top_k, vocab_size)

    if vocab_size == 0 or top_k == 0:
        return mx.array([]), mx.array([], dtype=logits.dtype)

    # Use argpartition for efficient top-k selection without full sort
    top_k_indices = mx.argpartition(-logits, top_k - 1)[:top_k]
    top_k_values = logits[top_k_indices]

    # Sort the top_k values for consistency
    sorted_order = mx.argsort(-top_k_values)
    top_k_indices = top_k_indices[sorted_order]
    top_k_values = top_k_values[sorted_order]

    return top_k_indices, top_k_values

def handle_logits_numpy(logits, top_k: int):
    """
    Implementation using NumPy arrays optimized for large vocabularies.
    """
    if not isinstance(logits, np.ndarray):
        raise TypeError("Expected logits to be a numpy.ndarray.")

    if logits.ndim != 1:
        raise ValueError("Logits must be a 1-dimensional array.")

    vocab_size = logits.size

    if not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")

    top_k = min(top_k, vocab_size)

    if vocab_size == 0 or top_k == 0:
        return np.array([], dtype=int), np.array([], dtype=logits.dtype)

    # Use argpartition for efficient top-k selection without full sort
    top_k_indices = np.argpartition(-logits, list(range(top_k)))[:top_k]
    top_k_values = logits[top_k_indices]

    # Sort the top_k values for consistency
    sorted_order = np.argsort(-top_k_values)
    top_k_indices = top_k_indices[sorted_order]
    top_k_values = top_k_values[sorted_order]

    return top_k_indices, top_k_values

def handle_logits_jax(logits, top_k: int):
    """
    Implementation using JAX arrays optimized for large vocabularies.
    """
    if not _has_jax:
        raise ImportError("JAX module is not installed. Please install it with 'pip install jax jaxlib'.")

    if not isinstance(logits, jnp.ndarray):
        raise TypeError("Expected logits to be a jax.numpy.ndarray.")

    if logits.ndim != 1:
        raise ValueError("Logits must be a 1-dimensional array.")

    vocab_size = logits.size

    if not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")

    top_k = min(top_k, vocab_size)

    if vocab_size == 0 or top_k == 0:
        return jnp.array([], dtype=int), jnp.array([], dtype=logits.dtype)

    # Use argpartition for efficient top-k selection without full sort
    top_k_indices = jnp.argpartition(-logits, top_k - 1)[:top_k]
    top_k_values = logits[top_k_indices]

    # Sort the top_k values for consistency
    sorted_order = jnp.argsort(-top_k_values)
    top_k_indices = top_k_indices[sorted_order]
    top_k_values = top_k_values[sorted_order]

    return top_k_indices, top_k_values

def handle_logits_pytorch(logits, top_k: int):
    """
    Implementation using PyTorch tensors optimized for large vocabularies.
    """
    if not _has_torch:
        raise ImportError("PyTorch module is not installed. Please install it with 'pip install torch'.")

    if not isinstance(logits, torch.Tensor):
        raise TypeError("Expected logits to be a torch.Tensor.")

    if logits.dim() != 1:
        raise ValueError("Logits must be a 1-dimensional tensor.")

    vocab_size = logits.size(0)

    if not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")

    top_k = min(top_k, vocab_size)

    if vocab_size == 0 or top_k == 0:
        return torch.tensor([], dtype=torch.long), torch.tensor([], dtype=logits.dtype)

    # Use torch.topk which is optimized and avoids sorting the entire array
    top_k_values, top_k_indices = torch.topk(
        logits, k=top_k, largest=True, sorted=True
    )

    return top_k_indices, top_k_values
