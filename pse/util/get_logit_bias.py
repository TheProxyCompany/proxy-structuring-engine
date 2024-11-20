import numpy as np
import logging
from typing import TypeVar, overload

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

T = TypeVar("T", np.ndarray, "torch.Tensor", "mx.array", "jnp.ndarray")


@overload
def get_logit_bias(logits: np.ndarray, accepted_token_ids: set[int]) -> np.ndarray: ...

@overload
def get_logit_bias(logits: "torch.Tensor", accepted_token_ids: set[int]) -> "torch.Tensor": ...

@overload
def get_logit_bias(logits: "mx.array", accepted_token_ids: set[int]) -> "mx.array": ...

@overload
def get_logit_bias(logits: "jnp.ndarray", accepted_token_ids: set[int]) -> "jnp.ndarray": ...


def get_logit_bias(logits: T, accepted_token_ids: set[int]) -> T:
    """
    Apply a -inf bias to logits of tokens that are not accepted.

    This function dispatches to the appropriate implementation based on the type of `logits`.

    Args:
        logits: The logits array of shape (vocab_size,), which can be an array from mlx, numpy, jax, or PyTorch.
        accepted_token_ids (set[int]): Set of accepted token IDs.

    Returns:
        The biased logits array, of the same type as `logits`.

    Raises:
        ValueError: If `logits` is not a 1-dimensional array.
        TypeError: If `logits` is not an instance of one of the supported array types.
        ImportError: If the required library for the type of `logits` is not installed.
    """
    if _has_mlx and isinstance(logits, mx.array):
        return bias_logits_mlx(logits, accepted_token_ids)
    elif isinstance(logits, np.ndarray):
        return bias_logits_numpy(logits, accepted_token_ids)
    elif _has_jax and isinstance(logits, jnp.ndarray):
        return bias_logits_jax(logits, accepted_token_ids)
    elif _has_torch and isinstance(logits, torch.Tensor):
        return bias_logits_pytorch(logits, accepted_token_ids)
    else:
        raise TypeError(f"Unsupported array type for logits: {type(logits)}")


def bias_logits_mlx(logits, accepted_token_ids: set[int]):
    """
    Implementation using mlx arrays that supports multi-dimensional logits arrays.
    """
    if not _has_mlx:
        raise ImportError(
            "mlx module is not installed. Please install it with 'pip install mlx'."
        )

    if logits.ndim < 1:
        raise ValueError("logits must be at least a 1-dimensional array.")

    vocab_size = logits.shape[-1]
    num_accepted_tokens = len(accepted_token_ids)

    if not accepted_token_ids:
        logger.debug("Accepted token IDs are empty. Returning logits without bias.")
        return logits

    if num_accepted_tokens == vocab_size:
        return logits

    bias_shape = logits.shape
    bias = mx.full(
        shape=bias_shape,
        vals=0.0 if num_accepted_tokens > vocab_size / 2 else float("-inf"),
    )

    indices = [slice(None)] * (logits.ndim - 1)
    if num_accepted_tokens > vocab_size / 2:
        # Bias non-accepted tokens
        indices.append(
            mx.array(list(set(range(vocab_size)) - accepted_token_ids), dtype=mx.int32)
        )
        bias[tuple(indices)] = float("-inf")
    else:
        # Bias accepted tokens
        indices.append(mx.array(list(accepted_token_ids), dtype=mx.int32))
        bias[tuple(indices)] = 0.0

    return bias


def bias_logits_numpy(logits, accepted_token_ids: set[int]):
    """
    Implementation using NumPy arrays that supports multi-dimensional logits arrays.
    """
    if logits.ndim < 1:
        raise ValueError("logits must be at least a 1-dimensional array.")

    vocab_size = logits.shape[-1]
    num_accepted_tokens = len(accepted_token_ids)

    if not accepted_token_ids:
        logger.warning("Accepted token IDs are empty. Returning logits without bias.")
        return logits

    if num_accepted_tokens == vocab_size:
        # All tokens are accepted; return logits as is
        return logits

    # Initialize bias array
    bias_shape = logits.shape
    bias = np.full(
        shape=bias_shape,
        fill_value=0.0 if num_accepted_tokens > vocab_size / 2 else float("-inf"),
        dtype=logits.dtype,
    )

    # Prepare indices for biasing
    indices = [slice(None)] * (logits.ndim - 1)
    if num_accepted_tokens > vocab_size / 2:
        # Bias non-accepted tokens
        non_accepted_ids = np.setdiff1d(
            np.arange(vocab_size), np.array(list(accepted_token_ids))
        )
        indices.append(non_accepted_ids)
        bias[tuple(indices)] = float("-inf")
    else:
        # Bias accepted tokens
        accepted_indices = np.array(list(accepted_token_ids), dtype=int)
        indices.append(accepted_indices)
        bias[tuple(indices)] = 0.0

    return bias


def bias_logits_jax(logits, accepted_token_ids: set[int]):
    """
    Implementation using JAX arrays that supports multi-dimensional logits arrays.
    """
    if not _has_jax:
        raise ImportError(
            "JAX module is not installed. Please install it with 'pip install jax jaxlib'."
        )

    if logits.ndim < 1:
        raise ValueError("logits must be at least a 1-dimensional array.")

    vocab_size = logits.shape[-1]
    num_tokens = vocab_size
    num_accepted_tokens = len(accepted_token_ids)

    if not accepted_token_ids:
        logger.warning("Accepted token IDs are empty. Returning logits without bias.")
        return logits

    if num_accepted_tokens == num_tokens:
        # All tokens are accepted; return logits as is
        return logits

    # Initialize bias array
    bias_shape = logits.shape
    bias = jnp.full(
        shape=bias_shape,
        fill_value=0.0 if num_accepted_tokens > vocab_size / 2 else float("-inf"),
        dtype=logits.dtype,
    )

    # Prepare indices for biasing
    if num_accepted_tokens > num_tokens / 2:
        # Bias non-accepted tokens
        accepted_indices = jnp.array(list(accepted_token_ids), dtype=int)
        mask = jnp.ones(vocab_size, dtype=bool)
        mask = mask.at[accepted_indices].set(False)
    else:
        # Bias accepted tokens
        accepted_indices = jnp.array(list(accepted_token_ids), dtype=int)
        mask = jnp.zeros(vocab_size, dtype=bool)
        mask = mask.at[accepted_indices].set(True)

    # Expand mask to match the shape of logits
    mask_shape = [1] * (logits.ndim - 1) + [vocab_size]
    mask = mask.reshape(mask_shape)

    # Apply bias using the mask
    bias_value = float("-inf") if num_accepted_tokens > num_tokens / 2 else 0.0
    bias = jnp.where(mask, bias_value, bias)

    return bias


def bias_logits_pytorch(logits, accepted_token_ids: set[int]):
    """
    Implementation using PyTorch tensors that supports multi-dimensional logits tensors.
    """
    if not _has_torch:
        raise ImportError(
            "PyTorch module is not installed. Please install it with 'pip install torch'."
        )

    if logits.dim() < 1:
        raise ValueError("logits must be at least a 1-dimensional tensor.")

    if not logits.is_floating_point():
        logits = logits.float()

    vocab_size = logits.size(-1)
    num_tokens = vocab_size
    num_accepted_tokens = len(accepted_token_ids)
    device = logits.device

    if not accepted_token_ids:
        logger.warning("Accepted token IDs are empty. Returning logits without bias.")
        return logits

    if num_accepted_tokens == num_tokens:
        # All tokens are accepted; return logits as is
        return logits

    # Initialize bias tensor
    bias_shape = logits.shape
    bias = torch.full(
        size=bias_shape,
        fill_value=0.0 if num_accepted_tokens > num_tokens / 2 else float("-inf"),
        device=device,
        dtype=logits.dtype,
    )

    # Prepare indices for biasing
    if num_accepted_tokens > num_tokens / 2:
        # Bias non-accepted tokens
        accepted_indices = torch.tensor(
            list(accepted_token_ids), dtype=torch.long, device=device
        )
        mask = torch.ones(vocab_size, dtype=torch.bool, device=device)
        mask[accepted_indices] = False
    else:
        # Bias accepted tokens
        accepted_indices = torch.tensor(
            list(accepted_token_ids), dtype=torch.long, device=device
        )
        mask = torch.zeros(vocab_size, dtype=torch.bool, device=device)
        mask[accepted_indices] = True

    # Expand mask to match the shape of logits
    mask_shape = [1] * (logits.dim() - 1) + [vocab_size]
    mask = mask.view(*mask_shape)

    # Apply bias using the mask
    bias_value = float("-inf") if num_accepted_tokens > num_tokens / 2 else 0.0
    bias = torch.where(mask, bias_value, bias)

    return bias
