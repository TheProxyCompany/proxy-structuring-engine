"""Utility functions for extracting top logits from various array types.

This module provides a unified interface for extracting top-k logits from different
array types (MLX, NumPy, JAX, PyTorch). It handles the necessary type checking and
provides optimized implementations for each backend.
"""

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Optional dependencies
try:
    import mlx.core as mx

    _HAS_MLX = True
except ImportError:
    _HAS_MLX = False

try:
    import jax.numpy as jnp

    _HAS_JAX = True
except ImportError:
    _HAS_JAX = False

try:
    import torch

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

# Constants
DEFAULT_TOP_K = 10
MIN_TOP_K = 1


def get_top_k(logits: Any, top_k: int = DEFAULT_TOP_K) -> dict[int, float]:
    """
    Extract the top-k logits and their corresponding token ids.

    This function dispatches to the appropriate implementation based on the array type.
    It supports MLX, NumPy, JAX, and PyTorch arrays.

    Args:
        logits: Array of shape (vocab_size,) or (1, vocab_size) containing the logits
        top_k: Number of top tokens to return (default: 64)

    Returns:
        Dictionary mapping token ids to their logit values, sorted by descending logit value

    Raises:
        ValueError: If logits shape is invalid or top_k is not positive
        TypeError: If logits type is not supported
        ImportError: If required backend is not installed
    """
    if top_k < MIN_TOP_K:
        raise ValueError(f"top_k must be >= {MIN_TOP_K}, got {top_k}")
    # Dispatch based on array type
    if (
        _HAS_MLX
        and mx is not None
        and hasattr(mx, "array")
        and isinstance(logits, mx.array)
    ):
        indices, values = get_top_logits_mlx(logits, top_k)
    elif isinstance(logits, np.ndarray):
        indices, values = get_top_logits_numpy(logits, top_k)
    elif (
        _HAS_JAX
        and jnp is not None
        and hasattr(jnp, "ndarray")
        and isinstance(logits, jnp.ndarray)
    ):
        indices, values = get_top_logits_jax(logits, top_k)
    elif (
        _HAS_TORCH
        and torch is not None
        and hasattr(torch, "Tensor")
        and isinstance(logits, torch.Tensor)
    ):
        indices, values = get_top_logits_pytorch(logits, top_k)
    else:
        raise TypeError(f"Unsupported array type: {type(logits)}")

    return {int(i): float(v) for i, v in zip(indices, values, strict=True)}


def get_top_logits_mlx(logits: "mx.array", top_k: int) -> tuple[Any, Any]:
    """Extract top-k logits using MLX arrays.

    Optimized implementation for MLX arrays that handles both 1D and 2D inputs.

    Args:
        logits: MLX array of shape (vocab_size,) or (1, vocab_size)
        top_k: Number of top tokens to return

    Returns:
        Tuple of (indices, values) arrays containing the top-k results

    Raises:
        ImportError: If MLX is not installed
        TypeError: If input is not an MLX array
        ValueError: If input dimensions are invalid
    """
    if mx is None:
        raise ImportError(
            "MLX is required but not installed. Install with 'pip install mlx'"
        )

    ndim = logits.ndim
    if ndim == 2:
        logits = mx.squeeze(logits, axis=0)
    elif ndim != 1:
        raise ValueError(f"Expected 1D or 2D array, got {ndim}D")

    vocab_size = logits.shape[0]
    top_k = min(top_k, vocab_size)

    if vocab_size == 0 or top_k == 0:
        empty = mx.array([], dtype=logits.dtype)
        return empty, empty

    # Use argpartition for efficient selection
    top_k_indices = mx.argpartition(-logits, top_k - 1)[:top_k]
    top_k_values = logits[top_k_indices]

    # Sort for consistency
    sorted_order = mx.argsort(-top_k_values)
    return top_k_indices[sorted_order], top_k_values[sorted_order]


def get_top_logits_numpy(logits: "np.ndarray", top_k: int) -> tuple[Any, Any]:
    """
    Implementation using NumPy arrays optimized for large vocabularies.
    If 2d, squeeze the last axis (1, vocab_size) -> (vocab_size,).
    """
    if not isinstance(logits, np.ndarray):
        raise TypeError("Expected logits to be a numpy.ndarray.")

    ndim = logits.ndim
    if ndim == 2:
        logits = np.squeeze(logits, axis=0)
    elif ndim != 1:
        raise ValueError("Logits must be a 1D or 2D array.")

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


def get_top_logits_jax(logits: "jnp.ndarray", top_k: int) -> tuple[Any, Any]:
    """
    Implementation using JAX arrays optimized for large vocabularies.
    If 2d, squeeze the last axis (1, vocab_size) -> (vocab_size,).
    """
    if not _HAS_JAX:
        raise ImportError(
            "JAX module is not installed. Please install it with 'pip install jax jaxlib'."
        )

    if not isinstance(logits, jnp.ndarray):
        raise TypeError("Expected logits to be a jax.numpy.ndarray.")

    ndim = logits.ndim
    if ndim == 2:
        logits = jnp.squeeze(logits, axis=0)
    elif ndim != 1:
        raise ValueError("Logits must be a 1D or 2D array.")

    vocab_size = logits.size

    if not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer.")

    top_k = min(top_k, vocab_size)

    if vocab_size == 0 or top_k == 0:
        return jnp.array([], dtype=int), jnp.array([], dtype=logits.dtype)

    # Use argpartition for efficient top-k selection without full sort
    top_k_indices = jnp.argpartition(-logits, top_k - 1)[:top_k]
    top_k_values = logits[top_k_indices]

    # Sort the top_k values
    sorted_order = jnp.argsort(-top_k_values)
    top_k_indices = top_k_indices[sorted_order]
    top_k_values = top_k_values[sorted_order]

    return top_k_indices, top_k_values


def get_top_logits_pytorch(logits: "torch.Tensor", top_k: int) -> tuple[Any, Any]:
    """
    Implementation using PyTorch tensors optimized for large vocabularies.
    If 2d, squeeze the last axis (1, vocab_size) -> (vocab_size,).
    """
    if not _HAS_TORCH:
        raise ImportError(
            "PyTorch module is not installed. Please install it with 'pip install torch'."
        )

    ndim = logits.dim()
    if ndim == 2:
        logits = torch.squeeze(logits, dim=0)
    elif ndim != 1:
        raise ValueError("Logits must be a 1D or 2D tensor.")

    vocab_size = logits.size(0)

    if not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")

    top_k = min(top_k, vocab_size)

    if vocab_size == 0 or top_k == 0:
        return torch.tensor([], dtype=torch.long), torch.tensor([], dtype=logits.dtype)

    # Use torch.topk which is optimized and avoids sorting the entire array
    top_k_values, top_k_indices = torch.topk(logits, k=top_k, largest=True, sorted=True)

    return top_k_indices, top_k_values
