import pytest
from typing import Dict, List
from transformers import PreTrainedTokenizerFast, LlamaTokenizer
import numpy as np
from pse.util.driver import OutputType, StructuredOutputDriver
from pse.util.errors import (
    TokenRejected,
    UnknownSchemaTypeError,
)

import logging

logger = logging.getLogger(__name__)


class MockTokenizer:
    """A mock tokenizer for testing purposes."""

    def __init__(self, vocab: Dict[str, int]):
        self.vocab = vocab
        self.inverse_vocab = {v: k for k, v in vocab.items()}
        self.all_special_ids = set()

    def decode(self, token_ids, **kwargs):
        return "".join([self.inverse_vocab.get(token_id, "") for token_id in token_ids])

    def encode(self, text, add_special_tokens=True, **kwargs):
        return [self.vocab.get(char, 9999) for char in text]

    def get_vocab(self):
        return self.vocab

    def convert_tokens_to_ids(self, tokens):
        return [self.vocab.get(token, 9999) for token in tokens]

    @property
    def eos_token_id(self):
        return self.vocab.get("<|eom_id|>", None)

    @property
    def all_special_tokens(self):
        return self.all_special_ids

    def batch_decode(self, token_ids_list):
        return [self.decode([token_id]) for token_id in token_ids_list]

    def add_special_tokens(self, special_tokens_dict):
        for token in special_tokens_dict.get("additional_special_tokens", []):
            token_id = max(self.vocab.values()) + 1
            self.vocab[token] = token_id
            self.inverse_vocab[token_id] = token
            self.all_special_ids.add(token_id)


@pytest.fixture(scope="module")
def mock_tokenizer() -> MockTokenizer:
    """Module-scoped fixture for the mock tokenizer."""
    vocab = {
        '"': 0,
        "abc": 1,
        "def": 2,
        "ghi": 3,
        "\n": 4,
        " ": 5,
        "1": 6,
        "2": 7,
        "3": 8,
        "4": 9,
        "\\n": 10,
        "\t": 11,
        "\\r": 12,
        "\\": 13,
        "<|eom_id|>": 14,
    }
    return MockTokenizer(vocab)


@pytest.fixture(scope="module")
def tokenizer() -> PreTrainedTokenizerFast:
    """Module-scoped fixture for the tokenizer."""
    TEST_MODEL = "fxmarty/tiny-llama-fast-tokenizer"
    return LlamaTokenizer.from_pretrained(TEST_MODEL, legacy=False)


@pytest.fixture(scope="module")
def driver(tokenizer: PreTrainedTokenizerFast) -> StructuredOutputDriver:
    """Module-scoped fixture for the StructuredOutputDriver."""
    driver = StructuredOutputDriver(tokenizer=tokenizer)
    return driver


@pytest.fixture(autouse=True)
def reset_driver(driver: StructuredOutputDriver) -> None:
    """Function-scoped fixture to reset the driver before each test."""
    driver.reset()


def check_correct_token_mask(
    driver: StructuredOutputDriver,
    test_logits: np.ndarray,
    vocab: Dict[str, int],
    valid_tokens: List[str],
) -> np.ndarray:
    """Check that the correct tokens are masked."""
    logits = driver.mask_invalid_tokens(test_logits)
    valid_token_ids = [vocab[token] for token in valid_tokens]
    for token_id, score in enumerate(logits):
        if token_id not in valid_token_ids:
            assert (
                score == -np.inf
            ), f"Token {driver.tokenizer.decode([token_id])} was incorrectly masked, valid tokens: {valid_tokens}"
        else:
            assert (
                score != -np.inf
            ), f"Token {driver.tokenizer.decode([token_id])} is not supposed to be masked, valid tokens: {valid_tokens}"

    return logits


def test_create_default(driver: StructuredOutputDriver) -> None:
    """Test the default initialization of StructuredOutputDriver."""
    driver.create_acceptor(schema={"type": "string"})
    assert isinstance(driver, StructuredOutputDriver)
    assert driver._waiting_for_trigger()


def test_create_with_non_default_parameters(driver: StructuredOutputDriver) -> None:
    """Test initializing StructuredOutputDriver with non-default parameters."""
    driver.create_acceptor(schema={"type": "string"}, encapsulated=False)
    assert not driver._waiting_for_trigger()


def test_create_acceptor_with_invalid_schema(driver: StructuredOutputDriver) -> None:
    """Test creating an acceptor with an invalid schema."""
    with pytest.raises(UnknownSchemaTypeError):
        driver.create_acceptor(schema={"type": "invalid_type"})


def test_create_acceptor_with_unsupported_output_type(
    driver: StructuredOutputDriver,
) -> None:
    """Test that creating an acceptor with an unsupported output type raises ValueError."""
    with pytest.raises(ValueError):
        driver.create_acceptor(schema={"type": "string"}, output_type=OutputType.PYTHON)


def test_create_acceptor_with_custom_delimiters(driver: StructuredOutputDriver) -> None:
    """Test creating an acceptor with custom delimiters."""
    from pse.acceptors.collections.encapsulated_acceptor import EncapsulatedAcceptor

    custom_opening: str = "<<<START>>>"
    custom_closing: str = "<<<END>>>"
    driver.create_acceptor(
        schema={"type": "string"},
        delimiters=(custom_opening, custom_closing),
    )

    assert isinstance(driver.acceptor, EncapsulatedAcceptor)
    assert driver.acceptor.opening_delimiter == custom_opening
    assert driver.acceptor.closing_delimiter == custom_closing


def test_advance_token_with_encapsulated_acceptor(
    driver: StructuredOutputDriver, tokenizer: PreTrainedTokenizerFast
) -> None:
    """Test that eos_token_id is in valid tokens when in accepted state."""
    driver.create_acceptor({"type": "string"})
    code = '```json\n"test string"\n```'
    for token_id in tokenizer.encode(code, add_special_tokens=False):
        print(f"advancing {tokenizer.decode([token_id])} with walker {driver.walkers}")
        driver.advance_token(token_id)

    assert driver.has_reached_accept_state


def test_advance_token_with_encapsulated_acceptor_partial_trigger(
    driver: StructuredOutputDriver, tokenizer: PreTrainedTokenizerFast
) -> None:
    """Test that eos_token_id is in valid tokens when in accepted state."""
    driver.create_acceptor({"type": "string"})

    assert driver._waiting_for_trigger()
    for token_id in tokenizer.encode("```", add_special_tokens=False):
        print(f"advancing {tokenizer.decode([token_id])} with walker {driver.walkers}")
        driver.advance_token(token_id)

    assert not driver._waiting_for_trigger()
    for token_id in tokenizer.encode("python", add_special_tokens=False):
        print(f"advancing {tokenizer.decode([token_id])} with walker {driver.walkers}")
        driver.advance_token(token_id)

    assert driver._waiting_for_trigger()
    for token_id in tokenizer.encode("```", add_special_tokens=False):
        print(f"advancing {tokenizer.decode([token_id])} with walker {driver.walkers}")
        driver.advance_token(token_id)

    assert not driver._waiting_for_trigger()
    assert driver.in_structured_state


def test_advance_token_with_acceptor_invalid(
    driver: StructuredOutputDriver, tokenizer: PreTrainedTokenizerFast
) -> None:
    """Test that eos_token_id is in valid tokens when in accepted state."""
    driver.create_acceptor({"type": "string"}, encapsulated=False)

    assert not driver._waiting_for_trigger()
    for token_id in tokenizer.encode("```", add_special_tokens=False):
        print(f"advancing {tokenizer.decode([token_id])} with walker {driver.walkers}")
        with pytest.raises(TokenRejected):
            driver.advance_token(token_id)

    assert not driver._waiting_for_trigger()
    assert not driver.has_reached_accept_state
    assert driver.in_structured_state


def test_create_acceptor_with_complex_schema(
    driver: StructuredOutputDriver,
) -> None:
    """Test create_acceptor with a complex schema."""
    complex_schema = {
        "type": "object",
        "properties": {
            "user": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "name": {"type": ["string", "null"]},
                    "roles": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["id"],
            },
            "active": {"type": "boolean"},
            "scores": {"type": "array", "items": {"type": "number"}},
        },
        "required": ["user", "active"],
    }
    driver.create_acceptor(schema=complex_schema)
    assert driver.acceptor is not None
    assert driver.walkers is not None


def test_create_acceptor_with_pattern_schema(
    driver: StructuredOutputDriver, tokenizer: PreTrainedTokenizerFast
) -> None:
    """Test create_acceptor with a schema that includes a pattern."""
    pattern_schema = {
        "type": "string",
        "pattern": "^[a-z]+$",
        "minLength": 1,
        "maxLength": 10,
    }
    driver.create_acceptor(schema=pattern_schema, encapsulated=False)
    assert driver.acceptor is not None
    assert driver.walkers is not None

    # Test valid input that matches the pattern
    valid_input = '"test"'
    for token_id in tokenizer.encode(valid_input, add_special_tokens=False):
        print(f"advancing {tokenizer.decode([token_id])} with walker {driver.walkers}")
        driver.advance_token(token_id)
    assert (
        driver.has_reached_accept_state
    ), "Driver should be in accepted state after consuming valid input."

    # Reset driver for invalid input test
    invalid_input = "123"
    for token_id in tokenizer.encode(invalid_input, add_special_tokens=False):
        with pytest.raises(TokenRejected):
            driver.advance_token(token_id)

    assert driver.has_reached_accept_state


def test_in_accepted_state_with_no_walkers(driver: StructuredOutputDriver) -> None:
    """Test in_accepted_state when walkers are empty."""
    driver.create_acceptor(schema={"type": "string"})
    driver.walkers = []
    assert not driver.has_reached_accept_state


# def test_mask_invalid_tokens(mock_tokenizer: MockTokenizer) -> None:
#     """Test that invalid tokens are masked correctly."""
#     driver = StructuredOutputDriver(tokenizer=mock_tokenizer)  # type: ignore
#     driver.create_acceptor({"type": "string"}, encapsulated=False)

#     test_vocab = {
#         '"': 0,
#         "abc": 1,
#         "def": 2,
#         "ghi": 3,
#         "\n": 4,
#         " ": 5,
#         "1": 6,
#         "2": 7,
#         "3": 8,
#         "4": 9,
#         "\\n": 10,
#         "\t": 11,
#         "\\r": 12,
#         "\\": 13,
#         "<|eom_id|>": 14,
#     }
#     test_logits = np.random.rand(len(test_vocab))
#     test_logits[0] = 1000

#     assert not driver.within_json_value
#     assert driver.in_structured_state

#     driver.get_valid_token(test_logits)

#     assert driver.within_json_value
#     assert not driver.in_structured_state

#     logits = driver.mask_invalid_tokens(test_logits)
#     print(logits)
#     assert driver.within_json_value
#     assert not driver.in_structured_state


def test_invalid_tokens_object() -> None:
    """Test that invalid tokens are masked correctly."""
    complex_schema = {
        "type": "object",
        "properties": {
            "name": {"const": "test"},
            "values": {"type": "array", "items": {"type": "number"}},
        },
        "required": ["name", "values"],
    }
    test_vocab = {
        '"': 0,
        '{"invalid': 1,
        "[": 2,
        "}": 3,
        "]": 4,
        "test": 5,
        "1": 6,
        "2": 7,
        "3": 8,
        ", ": 9,
        ",": 10,
        " ": 11,
        "true": 12,
        "false": 13,
        "-": 14,
        "\n": 15,
        "name": 16,
        "values": 17,
        '"name":': 18,
        '"values":': 19,
        '":': 20,
        '{"': 21,
        "{": 22,
    }
    mock_tokenizer = MockTokenizer(test_vocab)
    driver = StructuredOutputDriver(tokenizer=mock_tokenizer)  # type: ignore
    driver.create_acceptor(complex_schema, encapsulated=False)
    test_logits = np.random.rand(len(mock_tokenizer.vocab))

    assert not driver.within_json_value
    assert driver.in_structured_state

    logits = driver.mask_invalid_tokens(test_logits)
    valid_token_id: int = driver.get_valid_token(logits)
    assert valid_token_id in {21, 22}
