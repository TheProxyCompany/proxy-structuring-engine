import pytest
from typing import Any, Dict
from pse.acceptors.schema.object_schema_acceptor import ObjectSchemaAcceptor
from pse.util.errors import InvalidSchemaError
from unittest.mock import MagicMock


@pytest.fixture
def base_context() -> Dict[str, Any]:
    """Fixture to initialize the common test variables before each test."""
    return {"defs": {}, "path": "root"}


def test_initialization_valid_schema(base_context: Dict[str, Any]) -> None:
    """
    Test initializing ObjectSchemaAcceptor with a valid schema.
    Ensures that the instance is created without errors.
    """
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "number"},
        },
        "required": ["name"],
    }
    acceptor = ObjectSchemaAcceptor(schema, base_context)
    assert acceptor.schema == schema, "Schema should be set correctly."
    assert acceptor.context == base_context, "Context should be set correctly."
    assert (
        acceptor.properties == schema["properties"]
    ), "Properties should be extracted correctly."
    assert acceptor.required_property_names == [
        "name"
    ], "Required properties should be set correctly."


def test_initialization_missing_required_property_in_properties(
    base_context: Dict[str, Any],
) -> None:
    """
    Test initializing ObjectSchemaAcceptor where a required property is not defined in properties.
    Ensures that InvalidSchemaError is raised.
    """
    schema = {
        "type": "object",
        "properties": {
            "age": {"type": "number"},
        },
        "required": ["name"],
    }
    with pytest.raises(InvalidSchemaError):
        ObjectSchemaAcceptor(schema, base_context)


def test_value_started_hook_not_string(base_context: Dict[str, Any]) -> None:
    """
    Test that the value_started_hook is not called prematurely.
    """
    schema = {
        "type": "object",
        "properties": {
            "id": {"type": "number"},
            "email": {"type": "string"},
        },
        "required": ["id"],
    }

    started_hook: MagicMock = MagicMock()
    acceptor = ObjectSchemaAcceptor(schema, base_context, start_hook=started_hook)
    walkers = list(acceptor.get_walkers())
    for char in '{"id':
        walkers = [walker for _, walker in acceptor.advance_all(walkers, char)]
    # The hook should not be called yet
    started_hook.assert_not_called()

    # Continue parsing
    for char in '": 123}':
        walkers = [walker for _, walker in acceptor.advance_all(walkers, char)]

    assert any(
        walker.has_reached_accept_state() for walker in walkers
    ), "Transition to end state should return True when required properties are present."
    started_hook.assert_not_called()


def test_value_started_hook(base_context: Dict[str, Any]) -> None:
    """
    Test that the value_started_hook is called with the correct property name when a property's value starts.
    """
    schema = {
        "type": "object",
        "properties": {"id": {"type": "number"}, "email": {"type": "string"}},
        "required": ["id"],
    }

    started_hook: MagicMock = MagicMock()
    acceptor = ObjectSchemaAcceptor(schema, base_context, start_hook=started_hook)
    walkers = list(acceptor.get_walkers())
    for char in '{"email": ':
        walkers = [walker for _, walker in acceptor.advance_all(walkers, char)]

    started_hook.assert_not_called()
    # Simulate starting a string value
    walkers = [walker for _, walker in acceptor.advance_all(walkers, '"')]
    # The hook should be called now
    started_hook.assert_called_once()


def test_value_ended_hook(base_context: Dict[str, Any]) -> None:
    """
    Test that the value_ended_hook is called with the correct property name and value when a property's value ends.
    """
    schema = {
        "type": "object",
        "properties": {
            "id": {"type": "string"},
        },
    }
    ended_hook: MagicMock = MagicMock()
    acceptor = ObjectSchemaAcceptor(schema, base_context, end_hook=ended_hook)
    walkers = list(acceptor.get_walkers())

    additional_chars = '{"id": "hi'
    for char in additional_chars:
        walkers = [walker for _, walker in acceptor.advance_all(walkers, char)]
    ended_hook.assert_not_called()

    # Finish the string and the object
    for char in '"}':
        walkers = [walker for _, walker in acceptor.advance_all(walkers, char)]

    assert any(
        walker.has_reached_accept_state() for walker in walkers
    ), "Transition to end state should return True when all properties are present."
    ended_hook.assert_called_once()


def test_complex_json_structure(base_context: Dict[str, Any]) -> None:
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
    acceptor = ObjectSchemaAcceptor(schema, base_context)

    # Test initialization
    assert acceptor.schema == schema, "Schema should be set correctly."
    assert acceptor.context == base_context, "Context should be set correctly."
    assert (
        acceptor.properties == schema["properties"]
    ), "Properties should be extracted correctly."
    assert acceptor.required_property_names == [
        "name",
        "arguments",
    ], "Required properties should be set correctly."

    # Test parsing valid JSON input
    walkers = list(acceptor.get_walkers())
    valid_json = '{"name": "metacognition", "arguments": {"chain_of_thought": ["Thought 1", "Thought 2"]}}'
    for char in valid_json:
        walkers = [walker for _, walker in acceptor.advance_all(walkers, char)]

    assert any(
        walker.has_reached_accept_state() for walker in walkers
    ), "Transition to end state should return True for valid input."

    for walker in walkers:
        assert walker.current_value == {
            "name": "metacognition",
            "arguments": {
                "chain_of_thought": [
                    "Thought 1",
                    "Thought 2",
                ],
            },
        }



def test_complex_structure_partial_advancement():
    schema = {
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
        },
        "required": ["type", "label"],
        "additionalProperties": False,
    }
    acceptor = ObjectSchemaAcceptor(schema, {})
    walkers = list(acceptor.get_walkers())
    valid_json = '{\n  "'
    walkers = [walker for _, walker in acceptor.advance_all(walkers, valid_json)]
    assert len(walkers) == 2
    walkers = [walker for _, walker in acceptor.advance_all(walkers, "type")]
    assert len(walkers) == 1
    rest = '": "div", "label": "hello!" \n}'
    walkers = [walker for _, walker in acceptor.advance_all(walkers, rest)]
    assert len(walkers) == 1
    assert walkers[0].has_reached_accept_state()
    assert walkers[0].current_value == {"type": "div", "label": "hello!"}
