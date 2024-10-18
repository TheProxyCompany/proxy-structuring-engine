import pytest
import numpy as np
import logging

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

from pse.util.bias_logits import bias_logits

# Configure logging for debugging purposes
logger = logging.getLogger(__name__)

@pytest.fixture(params=[
    pytest.param(('numpy', np.array, np), id='numpy'),
    pytest.param(('mlx', mx.array, mx), id='mlx', marks=pytest.mark.skipif(not _has_mlx, reason="mlx not installed")),
    pytest.param(('jax', jnp.array, jnp), id='jax', marks=pytest.mark.skipif(not _has_jax, reason="jax not installed")),
    pytest.param(('torch', torch.tensor, torch), id='torch', marks=pytest.mark.skipif(not _has_torch, reason="torch not installed")),
])
def array_type(request):
    """
    Fixture to provide array constructor and corresponding module for different libraries.
    """
    return request.param


def test_bias_logits_empty_accepted_tokens(array_type):
    """
    Test bias_logits with an empty set of accepted tokens.
    Should return the original logits without modification.
    """
    lib_name, array_constructor, lib_module = array_type
    logits = array_constructor([0.1, 0.2, 0.3, 0.4, 0.5])

    accepted_token_ids = set()
    biased_logits = bias_logits(logits, accepted_token_ids)
    if lib_name == 'torch':
        assert isinstance(biased_logits, torch.Tensor)
        assert torch.allclose(biased_logits, logits), "Biased logits should be equal to original logits"
    elif lib_name == 'jax':
        assert isinstance(biased_logits, jnp.ndarray)
        assert jnp.array_equal(biased_logits, logits), "Biased logits should be equal to original logits"
    elif lib_name == 'mlx':
        assert isinstance(biased_logits, mx.array)
        assert mx.allclose(biased_logits, logits), "Biased logits should be equal to original logits"
    else:
        assert isinstance(biased_logits, np.ndarray)
        assert np.array_equal(biased_logits, logits), "Biased logits should be equal to original logits"


def test_bias_logits_all_accepted_tokens(array_type):
    """
    Test bias_logits with all tokens accepted.
    The biased logits should be equal to the original logits.
    """
    lib_name, array_constructor, lib_module = array_type
    logits = array_constructor([0.1, 0.2, 0.3, 0.4, 0.5])

    accepted_token_ids = set(range(len(logits)))
    biased_logits = bias_logits(logits, accepted_token_ids)

    if lib_name == 'torch':
        assert isinstance(biased_logits, torch.Tensor)
        assert torch.allclose(biased_logits, logits), "Biased logits should be equal to original logits"
    elif lib_name == 'jax':
        assert isinstance(biased_logits, jnp.ndarray)
        assert jnp.array_equal(biased_logits, logits), "Biased logits should be equal to original logits"
    elif lib_name == 'mlx':
        assert isinstance(biased_logits, mx.array)
        assert mx.allclose(biased_logits, logits), "Biased logits should be equal to original logits"
    else:
        assert isinstance(biased_logits, np.ndarray)
        assert np.array_equal(biased_logits, logits), "Biased logits should be equal to original logits"


def test_bias_logits_some_accepted_tokens(array_type):
    """
    Test bias_logits with some tokens accepted.
    Tokens not in accepted_token_ids should be set to -inf.
    """
    lib_name, array_constructor, lib_module = array_type
    logits = array_constructor([0.1, 0.2, 0.3, 0.4, 0.5])

    accepted_token_ids = {0, 2, 4}
    biased_logits = bias_logits(logits, accepted_token_ids)

    expected = [0.1, float('-inf'), 0.3, float('-inf'), 0.5]
    expected_biased_logits = array_constructor(expected)

    if lib_name == 'torch':
        assert isinstance(biased_logits, torch.Tensor)
        assert torch.allclose(biased_logits, expected_biased_logits, equal_nan=True), "Biased logits do not match expected values"
    elif lib_name == 'jax':
        assert isinstance(biased_logits, jnp.ndarray)
        assert jnp.allclose(biased_logits, expected_biased_logits, equal_nan=True), "Biased logits do not match expected values"
    elif lib_name == 'mlx':
        assert isinstance(biased_logits, mx.array)
        assert mx.allclose(biased_logits, expected_biased_logits), "Biased logits do not match expected values"
    else:
        assert isinstance(biased_logits, np.ndarray)
        assert np.allclose(biased_logits, expected_biased_logits, equal_nan=True), "Biased logits do not match expected values"

def test_bias_logits_empty_logits(array_type):
    """
    Test bias_logits with an empty logits array.
    Should return an empty biased logits array.
    """
    lib_name, array_constructor, lib_module = array_type
    logits = array_constructor([])

    accepted_token_ids = set()
    biased_logits = bias_logits(logits, accepted_token_ids)

    assert biased_logits.shape == logits.shape, "Biased logits should have the same shape as logits"
    assert len(biased_logits) == 0, "Biased logits should be empty"


def test_bias_logits_logits_with_inf(array_type):
    """
    Test bias_logits when logits contains inf values.
    The inf values should remain unaffected if they are accepted.
    """
    lib_name, array_constructor, lib_module = array_type
    logits = array_constructor([0.1, float('inf'), 0.3, float('-inf'), 0.5])

    accepted_token_ids = {0, 1, 4}
    biased_logits = bias_logits(logits, accepted_token_ids)

    expected = [0.1, float('inf'), float('-inf'), float('-inf'), 0.5]
    expected_biased_logits = array_constructor(expected)

    if lib_name == 'torch':
        assert isinstance(biased_logits, torch.Tensor)
        assert torch.equal(biased_logits, expected_biased_logits), "Biased logits do not match expected values"
    elif lib_name == 'jax':
        assert isinstance(biased_logits, jnp.ndarray)
        assert jnp.all(biased_logits == expected_biased_logits), "Biased logits do not match expected values"
    elif lib_name == 'mlx':
        assert isinstance(biased_logits, mx.array)
        assert mx.all(biased_logits == expected_biased_logits), "Biased logits do not match expected values"
    else:
        assert isinstance(biased_logits, np.ndarray)
        assert np.all(biased_logits == expected_biased_logits), "Biased logits do not match expected values"

def test_bias_logits_empty_logits_and_empty_accepted_token_ids(array_type):
    """
    Test bias_logits with empty logits and empty accepted_token_ids.
    Should return an empty array.
    """
    lib_name, array_constructor, lib_module = array_type
    logits = array_constructor([])

    accepted_token_ids = set()
    biased_logits = bias_logits(logits, accepted_token_ids)

    assert biased_logits.shape == (0,), "Biased logits should have shape (0,)"
    assert len(biased_logits) == 0, "Biased logits should be empty"


def test_bias_logits_logits_with_different_dtypes(array_type):
    """
    Test bias_logits with logits of different dtypes.
    """
    lib_name, array_constructor, lib_module = array_type
    dtypes = []
    if lib_name == 'torch':
        dtypes = [torch.float32, torch.bfloat16]
    elif lib_name == 'numpy':
        dtypes = [np.float32, np.float16]
    elif lib_name == 'jax':
        dtypes = [jnp.float32, jnp.bfloat16]
    elif lib_name == 'mlx':
        dtypes = [mx.float32, mx.bfloat16]
    else:
        pytest.skip(f"Unknown library: {lib_name}")

    for dtype in dtypes:
        logits = array_constructor([0.1, 0.2, 0.3], dtype=dtype)
        accepted_token_ids = {1}
        biased_logits = bias_logits(logits, accepted_token_ids)

        expected = [float('-inf'), 0.2, float('-inf')]
        expected_biased_logits = array_constructor(expected, dtype=dtype)

        if lib_name == 'torch':
            assert isinstance(biased_logits, torch.Tensor)
            assert torch.allclose(biased_logits, expected_biased_logits, atol=1e-8), f"Biased logits do not match expected values for dtype {dtype}"
        elif lib_name == 'jax':
            assert isinstance(biased_logits, jnp.ndarray)
            assert jnp.allclose(biased_logits, expected_biased_logits, atol=1e-8), f"Biased logits do not match expected values for dtype {dtype}"
        elif lib_name == 'mlx':
            assert isinstance(biased_logits, mx.array)
            assert mx.allclose(biased_logits, expected_biased_logits, atol=1e-8), f"Biased logits do not match expected values for dtype {dtype}"
        else:
            assert isinstance(biased_logits, np.ndarray)
            assert np.allclose(biased_logits, expected_biased_logits, atol=1e-8), f"Biased logits do not match expected values for dtype {dtype}"
