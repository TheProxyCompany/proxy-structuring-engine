import logging
import sys
from typing import Any

import pytest
from pydantic import BaseModel
from transformers import LlamaTokenizer

from pse.types.base.encapsulated import EncapsulatedStateMachine
from pse.types.grammar import PythonStateMachine

try:
    import mlx.core as mx

    _has_mlx = True
except ImportError:
    _has_mlx = False

from pse.structuring_engine import StructuringEngine

logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)


@pytest.fixture(scope="module")
def engine() -> StructuringEngine:
    """Module-scoped fixture for the StructuredOutputDriver."""
    TEST_MODEL = "hf-internal-testing/llama-tokenizer"
    tokenizer = LlamaTokenizer.from_pretrained(TEST_MODEL, legacy=False)
    engine = StructuringEngine(tokenizer)
    return engine


def generate_mock_logits(
    engine: StructuringEngine, input: dict[str, float], dtype: Any
) -> Any:
    import mlx.core as mx

    reverse_vocab: dict[int, str] = engine.reverse_vocabulary
    logits = mx.full(len(reverse_vocab), float("-inf"), dtype=dtype)
    for text, score in input.items():
        token_ids = engine.tokenizer.encode(text, add_special_tokens=False)
        logits[token_ids[0]] = score

    return logits


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
    engine.configure(complex_schema)
    assert engine.state_machine is not None
    assert engine.steppers is not None
    engine.reset(hard_reset=True)


def test_pattern_schema_success(engine: StructuringEngine) -> None:
    """Test create_acceptor with a schema that includes a pattern."""
    pattern_schema = {
        "type": "string",
        "pattern": "^[a-z]+$",
        "minLength": 1,
        "maxLength": 10,
    }
    engine.configure(pattern_schema)
    assert engine.state_machine is not None
    assert engine.steppers is not None

    # Test valid input that matches the pattern
    engine.consume_text('"test"')
    assert engine.has_reached_accept_state, "Driver should be in accepted state"
    engine.reset(hard_reset=True)


def test_pattern_schema_failure(engine: StructuringEngine) -> None:
    """Test create_acceptor with a schema that includes a pattern."""
    pattern_schema = {
        "type": "string",
        "pattern": "^[a-z]+$",
        "minLength": 1,
        "maxLength": 10,
    }
    engine.configure(structure=pattern_schema)
    assert engine.state_machine is not None
    assert engine.steppers is not None

    # Reset engine for invalid input test
    engine.consume_text("123")
    assert not engine.has_reached_accept_state
    engine.reset(hard_reset=True)


def test_multiple_schemas(engine: StructuringEngine) -> None:
    """Test AnyOfAcceptor with complex nested schemas."""
    schema1 = {
        "type": "object",
        "properties": {
            "name": {"const": "send_message"},
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
            "name": {"const": "metacognition"},
            "arguments": {
                "type": "object",
                "properties": {
                    "chain_of_thought": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "A sequence of high-level thoughts, reasoning, and internal dialogue.\nIncludes complex ideas, strategic considerations, and conceptual frameworks.\nSupports all Unicode characters, including emojis.",
                    },
                    "feelings": {
                        "type": "string",
                        "description": "A reflection of the agent's emotional state.\nSupports all Unicode characters, including emojis.",
                        "nullable": True,
                    },
                },
                "required": ["chain_of_thought"],
            },
        },
        "required": ["name", "arguments"],
    }
    engine.configure(
        {"anyOf": [schema1, schema2]},
        delimiters=("```json\n", "\n```"),
        buffer_length=0,
    )

    engine.consume_text(
        'Here is the response: ```json\n{\n"name":"', token_healing=False
    )
    assert len(engine.steppers) == 2
    engine.reset(hard_reset=True)

def test_wait_for_acceptor(engine: StructuringEngine) -> None:
    """Test that the wait for acceptor is working correctly."""
    engine.configure({"type": "string", "const": "Hello World!"}, buffer_length=0)
    buffer = "Sure, here is the response: "
    engine.consume_text(buffer)
    assert len(engine.steppers) == 1
    assert not engine.has_reached_accept_state
    engine.consume_text('"*', token_healing=True)
    # example of token healing
    assert len(engine.steppers) == 1
    assert not engine.has_reached_accept_state
    assert engine.steppers[0].get_current_value() == '"'
    engine.consume_text("Hello ")
    engine.consume_text('World!"')
    assert engine.has_reached_accept_state
    engine.reset(hard_reset=True)


def test_token_healing(engine: StructuringEngine) -> None:
    """Test that the token healing is working correctly."""
    SIMPLE_JSON_SCHEMA = {
        "type": "object",
        "properties": {"value": {"type": "number"}},
        "required": ["value"],
    }
    engine.configure(SIMPLE_JSON_SCHEMA)
    engine.consume_text('{"')
    assert len(engine.steppers) == 1
    assert not engine.has_reached_accept_state
    assert engine.steppers[0].get_raw_value() == '{"'


@pytest.mark.skipif(not _has_mlx, reason="mlx not installed")
def test_logits_processing(engine: StructuringEngine) -> None:
    """Test that the logits processing is working correctly across different dtypes."""
    dtypes = [mx.float32, mx.bfloat16, mx.float16]

    for dtype in dtypes:
        engine.configure(structure={"type": "string"})
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
        adjusted_logits = engine.process_logits(None, scores[None])
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
    engine.reset(hard_reset=True)


def test_python_interpreter(engine: StructuringEngine) -> None:
    """Test that the python interpreter is working correctly."""
    python_state_machine = EncapsulatedStateMachine(
        state_machine=PythonStateMachine,
        delimiters=PythonStateMachine.delimiters,
        buffer_length=0,
    )

    engine.configure(python_state_machine)
    engine.consume_text("```python\nprint('Hello, world!')\n```")
    assert engine.steppers
    assert engine.has_reached_accept_state
    output = engine.get_structured_output()
    assert output == "print('Hello, world!')"

def test_get_structured_output_with_type(engine: StructuringEngine) -> None:
    """Test get_structured_output with a type parameter."""
    class SimpleModel(BaseModel):
        value: int

    engine.configure({"type": "object", "properties": {"value": {"type": "integer"}}, "required": ["value"]})
    engine.consume_text('{"value": 42}')
    assert engine.has_reached_accept_state

    # Test with type casting
    output = engine.get_structured_output(output_type=SimpleModel)
    assert isinstance(output, SimpleModel)
    assert output.value == 42

    # Reset for next test
    engine.reset(hard_reset=True)

def test_get_structured_output_invalid_json(engine: StructuringEngine) -> None:
    """Test get_structured_output with invalid JSON content."""
    engine.configure({"type": "string"})
    engine.consume_text('"not valid json"')
    assert engine.has_reached_accept_state

    # Test with attempted casting to dict
    output = engine.get_structured_output(output_type=dict)
    # Should return original string since it's not valid JSON
    assert output == "not valid json"

    # Reset for next test
    engine.reset(hard_reset=True)

def test_get_stateful_structured_output(engine: StructuringEngine) -> None:
    """Test get_stateful_structured_output method."""
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"}
        },
        "required": ["name", "age"]
    }

    engine.configure(schema)
    engine.consume_text('{"name": "John", "age": 30}')
    assert engine.has_reached_accept_state

    # Get stateful output
    stateful_output = list(engine.get_stateful_structured_output())
    assert len(stateful_output) > 0

    # Check that we have state identifiers and values
    for state, value in stateful_output:
        assert isinstance(state, str)
        assert value is not None

    # Reset for next test
    engine.reset(hard_reset=True)

def test_get_live_structured_output(engine: StructuringEngine) -> None:
    """Test get_live_structured_output method."""
    schema = {"type": "string"}
    engine.configure(schema)

    # Consume partial input
    engine.consume_text('"test')

    # Complete the input
    engine.consume_text('"')
    assert engine.has_reached_accept_state

    # Reset for next test
    engine.reset(hard_reset=True)

def test_configure_with_direct_state_machine(engine: StructuringEngine) -> None:
    """Test configure with a direct StateMachine instance."""
    from pse_core.state_machine import StateMachine

    from pse.types.base.phrase import PhraseStateMachine

    # Create a simple state machine directly
    state_machine = StateMachine(
        {
            0: [(PhraseStateMachine("hello"), 1)],
            1: [(PhraseStateMachine(" world"), 2)]
        },
        start_state=0,
        end_states=[2]
    )

    # Configure with state machine directly
    engine.configure(state_machine)

    # Check that configuration worked
    assert engine.state_machine is state_machine
    assert engine.steppers is not None

    # Test that the state machine works
    engine.consume_text("hello world")
    assert engine.has_reached_accept_state

    # Reset for next test
    engine.reset(hard_reset=True)
