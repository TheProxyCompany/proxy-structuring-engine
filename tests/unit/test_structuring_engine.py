from typing import Any

import pytest
from transformers import LlamaTokenizer

try:
    _has_mlx = True
except ImportError:
    _has_mlx = False

from pse.structuring_engine import StructuringEngine


@pytest.fixture(scope="module")
def engine() -> StructuringEngine:
    """Module-scoped fixture for the StructuredOutputDriver."""
    TEST_MODEL = "fxmarty/tiny-llama-fast-tokenizer"
    tokenizer = LlamaTokenizer.from_pretrained(TEST_MODEL, legacy=False)
    engine = StructuringEngine(tokenizer)
    return engine


def generate_mock_logits(engine: StructuringEngine, input: dict[str, float], dtype: Any) -> Any:
    import mlx.core as mx

    logits = mx.full(len(engine.vocabulary), float("-inf"), dtype=dtype)
    for token, score in input.items():
        token_id = engine.vocabulary.get(token)
        if token_id is not None:
            logits[token_id] = score

    return logits


def test_create_acceptor_with_custom_delimiters(engine: StructuringEngine) -> None:
    """Test creating an state_machine with custom delimiters."""
    from pse.state_machines.composite.encapsulated import (
        EncapsulatedStateMachine,
    )

    delimiters = "<<<START>>>", "<<<END>>>"

    engine.configure(
        schema={"type": "string"},
        wrap_with_delimiters=True,
        delimiters=delimiters,
    )

    assert isinstance(engine.state_machine, EncapsulatedStateMachine)
    assert engine.state_machine.delimiters == delimiters


def test_delimitered_acceptor(engine: StructuringEngine) -> None:
    """Test that eos_token_id is in valid tokens when in accepted state."""
    engine.configure({"type": "string"}, wrap_with_delimiters=True)
    engine.consume_raw_input('```json\n"test string"\n```')
    assert engine.has_reached_accept_state, f"Engine with schema {engine.schema} should be in accepted state, walkers: {engine.walkers}"


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
    engine.configure(complex_schema, wrap_with_delimiters=False)
    assert engine.state_machine is not None
    assert engine.walkers is not None


def test_pattern_schema_success(engine: StructuringEngine) -> None:
    """Test create_acceptor with a schema that includes a pattern."""
    pattern_schema = {
        "type": "string",
        "pattern": "^[a-z]+$",
        "minLength": 1,
        "maxLength": 10,
    }
    engine.configure(schema=pattern_schema, wrap_with_delimiters=False)
    assert engine.state_machine is not None
    assert engine.walkers is not None

    # Test valid input that matches the pattern
    engine.consume_raw_input('"test"')
    assert engine.has_reached_accept_state, "Driver should be in accepted state"


def test_pattern_schema_failure(engine: StructuringEngine) -> None:
    """Test create_acceptor with a schema that includes a pattern."""
    pattern_schema = {
        "type": "string",
        "pattern": "^[a-z]+$",
        "minLength": 1,
        "maxLength": 10,
    }
    engine.configure(schema=pattern_schema, wrap_with_delimiters=False)
    assert engine.state_machine is not None
    assert engine.walkers is not None

    # Reset engine for invalid input test
    engine.consume_raw_input("123")
    assert not engine.has_reached_accept_state


def test_in_accepted_state_with_no_walkers(engine: StructuringEngine) -> None:
    """Test in_accepted_state when walkers are empty."""
    engine.configure(schema={"type": "string"}, wrap_with_delimiters=False)
    engine.walkers = []
    assert not engine.has_reached_accept_state


def test_multiple_schemas(engine: StructuringEngine) -> None:
    """Test AnyOfAcceptor with complex nested schemas."""
    schema1 = {
        "type": "object",
        "properties": {
            "name": {"type": "const", "const": "send_message"},
            "arguments": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "The final message content to be sent to the recipient.\nThis should be a packaged, markdown-formatted summary of the agent's work.\nSupports all Unicode characters, including emojis.",
                    }
                },
                "required": ["message"],
            },
        },
        "required": ["name", "arguments"],
    }
    schema2 = {
        "type": "object",
        "properties": {
            "name": {"type": "const", "const": "metacognition"},
            "arguments": {
                "type": "object",
                "properties": {
                    "chain_of_thought": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "A sequence of high-level thoughts, reasoning, and internal dialogue.\nIncludes complex ideas, strategic considerations, and conceptual frameworks.\nSupports all Unicode characters, including emojis.",
                    },
                    "feelings": {
                        "type": ["string"],
                        "description": "A reflection of the agent's emotional state.\nSupports all Unicode characters, including emojis.",
                        "nullable": True,
                        "default": None,
                    },
                },
                "required": ["chain_of_thought"],
            },
        },
        "required": ["name", "arguments"],
    }
    schema = {"anyOf": [schema1, schema2]}
    engine.configure(schema, wrap_with_delimiters=True)
    engine.consume_raw_input('Here is the response: ```json\n{\n"name":"')
    assert len(engine.walkers) == 2


@pytest.mark.parametrize(
    "value, followup_value",
    [
        ("Hello", "!"),
    ],
)
def test_edge_case_1(
    engine: StructuringEngine, value: str, followup_value: str
) -> None:
    """
    Test NumberAcceptor with an integer.

    Args:
        state_machine (NumberAcceptor): The NumberAcceptor instance.
    """

    schema = {
        "anyOf": [
            {
                "type": "object",
                "properties": {
                    "name": {"type": "const", "const": "send_message"},
                    "arguments": {
                        "type": "object",
                        "properties": {
                            "message": {
                                "type": "string",
                                "description": "The final message content to be sent to the recipient.\nThis should be a packaged, markdown-formatted summary of the agent's work.\nSupports all Unicode characters, including emojis.",
                            }
                        },
                        "required": ["message"],
                    },
                },
                "required": ["name", "arguments"],
            }
        ]
    }
    engine.configure(schema, wrap_with_delimiters=True)
    raw_input = '```json\n{"name": "send_message",'
    engine.consume_raw_input(raw_input)

    raw_input_2 = ' "arguments": {"message": "'
    engine.consume_raw_input(raw_input_2)

    token_id = engine.tokenizer.encode(str(value), add_special_tokens=False)[0]
    advanced_token = engine.advance_token(token_id)
    assert advanced_token is not None
    assert advanced_token == token_id

    advanced_token = engine.advance_token(
        engine.tokenizer.encode(str(followup_value), add_special_tokens=False)[0]
    )
    assert advanced_token is not None
    assert (
        advanced_token
        == engine.tokenizer.encode(str(followup_value), add_special_tokens=False)[0]
    )

    for token in engine.tokenizer.encode('"}}\n```', add_special_tokens=False):
        advanced_token = engine.advance_token(token)
        assert advanced_token is not None
        assert advanced_token == token

    assert engine.has_reached_accept_state


def test_wait_for_acceptor(engine: StructuringEngine) -> None:
    """Test that the wait for acceptor is working correctly."""
    engine.configure(
        schema={"type": "string", "const": "Hello World!"},
        wrap_with_delimiters=False,
        wait_for_acceptor=True,
    )
    engine.consume_raw_input("Sure, here is the response: ")
    assert len(engine.walkers) == 1
    assert not engine.has_reached_accept_state
    engine.consume_raw_input('"*')
    assert len(engine.walkers) == 1
    assert not engine.has_reached_accept_state
    engine.consume_raw_input("Hello ")
    engine.consume_raw_input('World!"')
    assert engine.has_reached_accept_state


@pytest.mark.skipif(not _has_mlx, reason="mlx not installed")
def test_logits_processing(engine: StructuringEngine) -> None:
    """Test that the logits processing is working correctly across different dtypes."""
    import mlx.core as mx

    dtypes = [mx.float32, mx.bfloat16, mx.float16]

    for dtype in dtypes:
        engine.configure(schema={"type": "string"}, wrap_with_delimiters=False)
        # we expect only tokens that start with a " character
        scores = generate_mock_logits(
            engine,
            {
                "Hello": 10.0,
                '"Hello': 3.0,
                "Hi": 8.0,
                '"Hi': 4.0,
                "Hey": 2.0,
                '"Hey': 1.0,
                '"': 1.0,
            },
            dtype,
        )
        adjusted_logits = engine(scores)
        expected_score = generate_mock_logits(
            engine,
            {
                "Hello": float("-inf"),
                "Hi": float("-inf"),
                "Hey": float("-inf"),
                '"': 1.0,
            },
            dtype,
        )
        assert mx.allclose(adjusted_logits, expected_score)


@pytest.mark.skipif(not _has_mlx, reason="mlx not installed")
def test_real_world_logits_processing(engine: StructuringEngine) -> None:
    """Test that the logits processing is working correctly."""
    import mlx.core as mx

    scores = generate_mock_logits(
        engine,
        {
            "val": 10.0,
            "type": 3.0,
            "value": 8.0,
            "validate": 4.0,
            "required": 1.0,
            "values": 1.0,
            '"': 1.0,
        },
        mx.bfloat16,
    )

    schema = {
        "type": "object",
        "properties": {"value": {"type": "number"}},
        "required": ["value"],
        "additionalProperties": False,
    }
    engine.configure(schema, wrap_with_delimiters=False, wait_for_acceptor=True)
    engine.consume_raw_input('Sure, here is the response: {"')
    assert not engine.has_reached_accept_state
    adjusted_logits = engine(scores)
    expected_score = generate_mock_logits(
        engine,
        {
            "val": 10.0,
            "type": float("-inf"),
            "value": 8.0,
            "validate": 4.0,
            "values": 1.0,
            '"': 1.0,
        },
        mx.bfloat16,
    )
    expected_scores_match = mx.allclose(adjusted_logits, expected_score)
    assert expected_scores_match
