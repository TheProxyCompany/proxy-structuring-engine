import logging

logger = logging.getLogger(__name__)

# Attempt to import optional dependencies
try:
    import mlx.core as mx
    _has_mlx = True
except ImportError:
    _has_mlx = False

def get_top_logits(logits, top_k: int = 64) -> dict[int, float]:
    """
    Returns the top_k logits and their corresponding token ids.

    This function dispatches to the appropriate implementation based on the type of `logits`.

    Args:
        logits: The logits array of shape (vocab_size,), which can be an array from MLX.
        top_k (int): The number of top tokens to return.

    Returns:
        A list of tuples (token_id, logit), both arrays of length top_k.

    Raises:
        ValueError: If `logits` is not a 1-dimensional array or if `top_k` is not a positive integer.
        TypeError: If `logits` is not an instance of one of the supported array types.
    """
    if _has_mlx and isinstance(logits, mx.array):
        indices, values = get_top_logits_mlx(logits, top_k)
    else:
        raise TypeError(f"Unsupported array type for logits: {type(logits)}")

    return {int(i): float(v) for i, v in zip(indices, values, strict=True)}


def get_top_logits_mlx(logits, top_k: int):
    """
    Implementation using MLX arrays optimized for large vocabularies.
    If 2d, squeeze the last axis (1, vocab_size) -> (vocab_size,).
    """
    assert _has_mlx
    assert isinstance(logits, mx.array)
    assert isinstance(top_k, int) and top_k > 0
    ndim = logits.ndim
    if ndim == 2:
        logits = logits.squeeze(axis=0)

    vocab_size = logits.shape[0]
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
