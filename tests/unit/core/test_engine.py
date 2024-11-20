import pytest
import numpy as np
from transformers import LlamaTokenizer
from pse.core.engine import StructuringEngine
from pse.util.errors import UnknownSchemaTypeError


@pytest.fixture(scope="module")
def engine() -> StructuringEngine:
    """Module-scoped fixture for the StructuredOutputDriver."""
    TEST_MODEL = "fxmarty/tiny-llama-fast-tokenizer"
    tokenizer = LlamaTokenizer.from_pretrained(TEST_MODEL, legacy=False)
    engine = StructuringEngine(tokenizer=tokenizer)
    return engine


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
    engine.consume_raw_input('```json\n"test string"\n```')
    assert engine.has_reached_accept_state()


def test_partial_delimiters(engine: StructuringEngine) -> None:
    """Test that delimiters are handled correctly."""
    engine.set_schema({"type": "string"})
    assert engine._waiting_for_trigger()
    engine.consume_raw_input("```")
    assert engine._waiting_for_trigger()
    engine.consume_raw_input("python")
    assert engine._waiting_for_trigger()
    engine.consume_raw_input("json")
    assert engine._waiting_for_trigger()
    engine.consume_raw_input("\n")
    assert not engine._waiting_for_trigger()
    assert engine.in_structured_state


def test_invalid_input(engine: StructuringEngine) -> None:
    """Test that eos_token_id is in valid tokens when in accepted state."""
    engine.set_schema({"type": "string"}, use_delimiters=False)

    assert not engine._waiting_for_trigger()
    engine.consume_raw_input("```")

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
    engine.consume_raw_input('"test"')
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
    engine.consume_raw_input("123")
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

    logits = engine.generate_logit_bias_mask(test_logits)
    valid_token_id = engine.get_next_token(logits)
    assert valid_token_id in {21, 22}
