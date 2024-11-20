import pytest
import numpy as np
import logging
from pse.util.get_top_logits import get_top_logits

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

logger = logging.getLogger(__name__)


@pytest.fixture(
    params=[
        pytest.param(("numpy", np.array, np), id="numpy"),
        pytest.param(
            ("mlx", mx.array, mx),
            id="mlx",
            marks=pytest.mark.skipif(not _has_mlx, reason="mlx not installed"),
        ),
        pytest.param(
            ("jax", jnp.array, jnp),
            id="jax",
            marks=pytest.mark.skipif(not _has_jax, reason="jax not installed"),
        ),
        pytest.param(
            ("torch", torch.tensor, torch),
            id="torch",
            marks=pytest.mark.skipif(not _has_torch, reason="torch not installed"),
        ),
    ]
)
def array_type(request):
    """
    Fixture to provide array constructor and corresponding module for different libraries.
    """
    return request.param


def test_handle_logits_top_k(array_type):
    """
    Test handle_logits with a simple logits array and positive top_k.
    """
    _, array_constructor, _ = array_type
    logits = array_constructor([0.1, 0.4, 0.2, 0.5, 0.3])

    top_k = 3
    token_ids_and_scores = get_top_logits(logits, top_k=top_k)

    expected_indices = [3, 1, 4]
    expected_values = [0.5, 0.4, 0.3]
    expected_token_ids_and_scores = list(zip(expected_indices, expected_values))

    assert isinstance(token_ids_and_scores, list)
    assert len(token_ids_and_scores) == top_k
    assert all(
        isinstance(token_id, int) and isinstance(score, float)
        for token_id, score in token_ids_and_scores
    )
    for (token_id, score), (exp_id, exp_score) in zip(
        token_ids_and_scores, expected_token_ids_and_scores
    ):
        assert token_id == exp_id, f"Expected token ID {exp_id}, got {token_id}"
        assert score == pytest.approx(
            exp_score, abs=1e-6
        ), f"Expected score {exp_score}, got {score}"


def test_handle_logits_top_k_equals_vocab_size(array_type):
    """
    Test handle_logits when top_k equals the size of the vocabulary.
    """
    _, array_constructor, _ = array_type
    logits = array_constructor([0.5, 0.2, 0.4, 0.1, 0.3])

    top_k = len(logits)  # Equal to vocab size
    token_ids_and_scores = get_top_logits(logits, top_k=top_k)

    expected_indices = [0, 2, 4, 1, 3]
    expected_values = [0.5, 0.4, 0.3, 0.2, 0.1]
    expected_token_ids_and_scores = list(zip(expected_indices, expected_values))

    assert isinstance(token_ids_and_scores, list)
    assert len(token_ids_and_scores) == len(logits)
    assert all(
        isinstance(token_id, int) and isinstance(score, float)
        for token_id, score in token_ids_and_scores
    )
    for (token_id, score), (exp_id, exp_score) in zip(
        token_ids_and_scores, expected_token_ids_and_scores
    ):
        assert token_id == exp_id, f"Expected token ID {exp_id}, got {token_id}"
        assert score == pytest.approx(
            exp_score, abs=1e-6
        ), f"Expected score {exp_score}, got {score}"


def test_handle_logits_top_k_greater_than_vocab_size(array_type):
    """
    Test handle_logits when top_k is greater than the size of the vocabulary.
    Should cap top_k at vocab size.
    """
    _, array_constructor, _ = array_type
    logits = array_constructor([0.5, 0.2, 0.4])

    top_k = 5  # Greater than vocab size
    token_ids_and_scores = get_top_logits(logits, top_k=top_k)

    expected_indices = [0, 2, 1]
    expected_values = [0.5, 0.4, 0.2]
    expected_token_ids_and_scores = list(zip(expected_indices, expected_values))

    assert isinstance(token_ids_and_scores, list)
    assert len(token_ids_and_scores) == len(logits)
    assert all(
        isinstance(token_id, int) and isinstance(score, float)
        for token_id, score in token_ids_and_scores
    )
    for (token_id, score), (exp_id, exp_score) in zip(
        token_ids_and_scores, expected_token_ids_and_scores
    ):
        assert token_id == exp_id, f"Expected token ID {exp_id}, got {token_id}"
        assert score == pytest.approx(
            exp_score, abs=1e-6
        ), f"Expected score {exp_score}, got {score}"


def test_handle_logits_with_duplicate_values(array_type):
    """
    Test handle_logits when there are duplicate values in logits.
    """
    _, array_constructor, _ = array_type
    logits = array_constructor([0.3, 0.2, 0.3, 0.1, 0.2])

    top_k = 4
    token_ids_and_scores = get_top_logits(logits, top_k=top_k)

    expected_values = [0.3, 0.3, 0.2, 0.2]

    assert isinstance(token_ids_and_scores, list)
    assert len(token_ids_and_scores) == top_k
    assert all(
        isinstance(token_id, int) and isinstance(score, float)
        for token_id, score in token_ids_and_scores
    )

    # Since there are duplicate values, the order of indices with same values may vary
    returned_scores = [score for _, score in token_ids_and_scores]
    returned_scores_sorted = sorted(returned_scores, reverse=True)
    expected_scores_sorted = sorted(expected_values, reverse=True)
    assert returned_scores_sorted == pytest.approx(
        expected_scores_sorted, abs=1e-6
    ), "Top scores do not match expected values"

    returned_token_ids = [token_id for token_id, _ in token_ids_and_scores]
    assert len(set(returned_token_ids)) == len(
        returned_token_ids
    ), "Duplicate token IDs found in the results"


def test_handle_logits_empty_logits(array_type):
    """
    Test handle_logits when logits array is empty.
    Should return an empty list.
    """
    _, array_constructor, _ = array_type
    logits = array_constructor([])

    top_k = 5
    token_ids_and_scores = get_top_logits(logits, top_k=top_k)

    assert isinstance(token_ids_and_scores, list)
    assert len(token_ids_and_scores) == 0, "Expected empty list when logits is empty"


def test_handle_logits_invalid_top_k(array_type):
    """
    Test handle_logits when top_k is negative or not an integer.
    Should raise ValueError.
    """
    _, array_constructor, _ = array_type
    logits = array_constructor([0.3, 0.2, 0.1])

    with pytest.raises(ValueError):
        get_top_logits(logits, top_k=-1)


def test_handle_logits_invalid_logits(array_type):
    """
    Test handle_logits when logits is not 1-dimensional.
    Should raise ValueError.
    """
    _, array_constructor, _ = array_type
    logits = array_constructor([[0.1, 0.2], [0.3, 0.4]])

    with pytest.raises(ValueError):
        get_top_logits(logits, top_k=2)
