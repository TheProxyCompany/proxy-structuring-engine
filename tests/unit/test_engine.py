import pytest
from transformers import LlamaTokenizer

from pse.structuring_engine import StructuringEngine


@pytest.fixture(scope="module")
def engine() -> StructuringEngine:
    """Module-scoped fixture for the StructuredOutputDriver."""
    TEST_MODEL = "fxmarty/tiny-llama-fast-tokenizer"
    tokenizer = LlamaTokenizer.from_pretrained(TEST_MODEL, legacy=False)
    engine = StructuringEngine(tokenizer)
    return engine

def test_engine_init(engine: StructuringEngine) -> None:
    """Test that the engine is initialized correctly."""
    assert engine.vocabulary is not None

def test_create_acceptor_with_custom_delimiters(engine: StructuringEngine) -> None:
    """Test creating an state_machine with custom delimiters."""
    from pse.state_machines.collections.encapsulated_acceptor import (
        EncapsulatedAcceptor,
    )

    delimiters = "<<<START>>>", "<<<END>>>"

    engine.configure(
        schema={"type": "string"},
        wrap_with_delimiters=True,
        delimiters=delimiters,
    )

    assert isinstance(engine.state_machine, EncapsulatedAcceptor)
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
    engine.consume_raw_input("Here is the response: ```json\n{\n")
    assert len(engine.walkers) == 2
