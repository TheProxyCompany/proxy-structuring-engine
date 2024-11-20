import pytest
from typing import Dict, List
from transformers import PreTrainedTokenizerFast, LlamaTokenizer
import numpy as np
from pse.core.engine import StructuringEngine
from pse.util.errors import UnknownSchemaTypeError

import logging

logger = logging.getLogger(__name__)

@pytest.fixture(scope="module")
def tokenizer() -> PreTrainedTokenizerFast:
    """Module-scoped fixture for the tokenizer."""
    TEST_MODEL = "fxmarty/tiny-llama-fast-tokenizer"
    return LlamaTokenizer.from_pretrained(TEST_MODEL, legacy=False)


@pytest.fixture(scope="module")
def engine(tokenizer: PreTrainedTokenizerFast) -> StructuringEngine:
    """Module-scoped fixture for the StructuredOutputDriver."""
    engine = StructuringEngine(tokenizer=tokenizer)
    return engine


def check_correct_token_mask(
    engine: StructuringEngine,
    test_logits: np.ndarray,
    vocab: Dict[str, int],
    valid_tokens: List[str],
) -> np.ndarray:
    """Check that the correct tokens are masked."""
    logits = engine.mask_invalid_tokens(test_logits)
    valid_token_ids = [vocab[token] for token in valid_tokens]
    for token_id, score in enumerate(logits):
        if token_id not in valid_token_ids:
            assert (
                score == -np.inf
            ), f"Token {engine.tokenizer.decode([token_id])} was incorrectly masked, valid tokens: {valid_tokens}"
        else:
            assert (
                score != -np.inf
            ), f"Token {engine.tokenizer.decode([token_id])} is not supposed to be masked, valid tokens: {valid_tokens}"

    return logits


def test_create_default(engine: StructuringEngine) -> None:
    """Test the default initialization of StructuredOutputDriver."""
    engine.set_schema({"type": "string"})
    assert engine._waiting_for_trigger()


def test_create_with_non_default_parameters(engine: StructuringEngine) -> None:
    """Test initializing StructuredOutputDriver with non-default parameters."""
    engine.set_schema({"type": "string"}, use_delimiters=False)
    assert not engine._waiting_for_trigger()


def test_create_acceptor_with_invalid_schema(engine: StructuringEngine) -> None:
    """Test creating an acceptor with an invalid schema."""
    with pytest.raises(UnknownSchemaTypeError):
        engine.set_schema(schema={"type": "invalid_type"})


def test_create_acceptor_with_custom_delimiters(engine: StructuringEngine) -> None:
    """Test creating an acceptor with custom delimiters."""
    from pse.acceptors.collections.encapsulated_acceptor import EncapsulatedAcceptor

    custom_opening: str = "<<<START>>>"
    custom_closing: str = "<<<END>>>"
    engine.set_schema(
        schema={"type": "string"},
        delimiters=(custom_opening, custom_closing),
    )

    assert isinstance(engine.acceptor, EncapsulatedAcceptor)
    assert engine.acceptor.opening_delimiter == custom_opening
    assert engine.acceptor.closing_delimiter == custom_closing


def test_delimitered_acceptor(engine: StructuringEngine) -> None:
    """Test that eos_token_id is in valid tokens when in accepted state."""
    engine.set_schema({"type": "string"})
    engine.process_input('```json\n"test string"\n```')
    assert engine.has_reached_accept_state()

def test_partial_delimiters(engine: StructuringEngine) -> None:
    """Test that delimiters are handled correctly."""
    engine.set_schema({"type": "string"})
    assert engine._waiting_for_trigger()
    engine.process_input("```")
    assert engine._waiting_for_trigger()
    engine.process_input("python")
    assert engine._waiting_for_trigger()
    engine.process_input("json")
    assert engine._waiting_for_trigger()
    engine.process_input("\n")
    assert not engine._waiting_for_trigger()
    assert engine.in_structured_state

def test_invalid_input(engine: StructuringEngine) -> None:
    """Test that eos_token_id is in valid tokens when in accepted state."""
    engine.set_schema({"type": "string"}, use_delimiters=False)

    assert not engine._waiting_for_trigger()
    engine.process_input("```")

    assert not engine._waiting_for_trigger()
    assert not engine.has_reached_accept_state()
    assert engine.in_structured_state


def test_create_acceptor_with_complex_schema(
    engine: StructuringEngine,
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
    engine.set_schema(complex_schema)
    assert engine.acceptor is not None
    assert engine.walkers is not None


def test_pattern_schema_success(engine: StructuringEngine) -> None:
    """Test create_acceptor with a schema that includes a pattern."""
    pattern_schema = {
        "type": "string",
        "pattern": "^[a-z]+$",
        "minLength": 1,
        "maxLength": 10,
    }
    engine.set_schema(schema=pattern_schema, use_delimiters=False)
    assert engine.acceptor is not None
    assert engine.walkers is not None

    # Test valid input that matches the pattern
    engine.process_input('"test"')
    assert engine.has_reached_accept_state(), "Driver should be in accepted state"


def test_pattern_schema_failure(engine: StructuringEngine) -> None:
    """Test create_acceptor with a schema that includes a pattern."""
    pattern_schema = {
        "type": "string",
        "pattern": "^[a-z]+$",
        "minLength": 1,
        "maxLength": 10,
    }
    engine.set_schema(schema=pattern_schema, use_delimiters=False)
    assert engine.acceptor is not None
    assert engine.walkers is not None

    # Reset engine for invalid input test
    engine.process_input("123")
    assert not engine.has_reached_accept_state()


def test_in_accepted_state_with_no_walkers(engine: StructuringEngine) -> None:
    """Test in_accepted_state when walkers are empty."""
    engine.set_schema(schema={"type": "string"})
    engine.walkers = []
    assert not engine.has_reached_accept_state()


def test_invalid_tokens_object(engine: StructuringEngine) -> None:
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
    StructuringEngine.build_vocabulary(engine.tokenizer, test_vocab)
    engine.set_schema(complex_schema, use_delimiters=False)
    test_logits = np.random.rand(len(test_vocab))

    assert not engine.within_json_value
    assert engine.in_structured_state

    logits = engine.mask_invalid_tokens(test_logits)
    valid_token_id = engine.get_next_token(logits)
    assert valid_token_id in {21, 22}


def test_simple_json_structure(engine: StructuringEngine) -> None:
    schema = {
        "type": "object",
        "properties": {"value": {"type": "number"}},
        "required": ["value"],
        "additionalProperties": False,
    }
    engine.set_schema(schema)

    assert not engine.within_json_value
    assert not engine.in_structured_state

    # test_logits = np.random.rand(len(tokenizer.get_vocab()))

    # logits = engine.mask_invalid_tokens(test_logits)
    # valid_token_id: int = engine.get_valid_token(logits)

    # assert tokenizer.decode([valid_token_id]).startswith("{")


def test_complex_json_structure(
    engine: StructuringEngine, tokenizer: PreTrainedTokenizerFast
) -> None:
    """Test parsing a complex JSON structure."""
    schema = {
        "type": "object",
        "properties": {
            "name": {"const": "metacognition"},
            "arguments": {
                "type": "object",
                "properties": {
                    "chain_of_thought": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "feelings": {
                        "type": ["string"],
                        "nullable": True,
                        "default": None,
                    },
                },
                "required": ["chain_of_thought"],
            },
        },
        "required": ["name", "arguments"],
    }
    engine.set_schema(schema)
    # test_logits = np.random.rand(len(tokenizer.get_vocab()))

    # logits = engine.mask_invalid_tokens(test_logits)
    # valid_token_id: int = engine.get_valid_token(logits)

    # assert tokenizer.decode([valid_token_id]).startswith("{")


def test_better_than_openai(
    engine: StructuringEngine, tokenizer: PreTrainedTokenizerFast
) -> None:
    """Test that OpenAI sucks."""
    # openAI's structured output blog post said:
    #
    #   "The following is a sample recursive schema that is supported on
    #   the OpenAI API with Structured Outputs but would not be possible to express with a FSM."
    #
    # let's test that.
    schema = {
        "name": "ui",
        "description": "Dynamically generated UI",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "type": {
                    "type": "string",
                    "description": "The type of the UI component",
                    "enum": ["div", "button", "header", "section", "field", "form"],
                },
                "label": {
                    "type": "string",
                    "description": "The label of the UI component, used for buttons or form fields",
                },
                "children": {
                    "type": "array",
                    "description": "Nested UI components",
                    "items": {"$ref": "#"},
                },
                "attributes": {
                    "type": "array",
                    "description": "Arbitrary attributes for the UI component, suitable for any element",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "The name of the attribute, for example onClick or className",
                            },
                            "value": {
                                "type": "string",
                                "description": "The value of the attribute",
                            },
                        },
                    },
                },
            },
            "required": ["type", "label", "children", "attributes"],
            "additionalProperties": False,
        },
    }
    engine.set_schema(schema, use_delimiters=False)
    # test_logits = np.random.rand(len(tokenizer.get_vocab()))

    # assert not engine.within_json_value
    # assert engine.in_structured_state

    # logits = engine.mask_invalid_tokens(test_logits)
    # valid_token_id: int = engine.get_valid_token(logits)

    # assert tokenizer.decode([valid_token_id]).startswith("{")

    # not done these tests yet
