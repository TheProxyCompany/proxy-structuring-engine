from typing import Any

import pytest

from pse.types.json.json_object import ObjectSchemaStateMachine


@pytest.fixture
def base_context() -> dict[str, Any]:
    """Fixture to initialize the common test variables before each test."""
    return {"defs": {}, "path": "root"}


def test_initialization_valid_schema(base_context: dict[str, Any]) -> None:
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
    state_machine = ObjectSchemaStateMachine(schema, base_context)
    assert state_machine.schema == schema, "Schema should be set correctly."
    assert state_machine.context == base_context, "Context should be set correctly."
    assert state_machine.properties == schema["properties"], (
        "Properties should be extracted correctly."
    )
    assert state_machine.required_property_names == ["name"], (
        "Required properties should be set correctly."
    )


def test_nullable_required_property(base_context: dict[str, Any]) -> None:
    """
    Test that a required property with nullable=True is not actually required.
    This tests the special handling in __init__ where properties with nullable=True are removed from required list.
    """
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "nullable": True},
            "age": {"type": "number"},
        },
        "required": ["name"],
    }
    state_machine = ObjectSchemaStateMachine(schema, base_context)
    assert "name" not in state_machine.required_property_names, (
        "Nullable required property should be removed from required_property_names"
    )


def test_property_with_default_not_required(base_context: dict[str, Any]) -> None:
    """
    Test that a required property with a default value is not actually required.
    This tests the special handling in __init__ where properties with defaults are removed from required list.
    """
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "default": "default name"},
            "age": {"type": "number"},
        },
        "required": ["name"],
    }
    state_machine = ObjectSchemaStateMachine(schema, base_context)
    assert "name" not in state_machine.required_property_names, (
        "Required property with default should be removed from required_property_names"
    )


def test_initialization_missing_required_property_in_properties(
    base_context: dict[str, Any],
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
    with pytest.raises(ValueError):
        ObjectSchemaStateMachine(schema, base_context)


def test_complex_json_structure(base_context: dict[str, Any]) -> None:
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
    state_machine = ObjectSchemaStateMachine(schema, base_context)

    # Test initialization
    assert state_machine.schema == schema, "Schema should be set correctly."
    assert state_machine.context == base_context, "Context should be set correctly."
    assert state_machine.properties == schema["properties"], (
        "Properties should be extracted correctly."
    )
    assert state_machine.required_property_names == [
        "name",
        "arguments",
    ], "Required properties should be set correctly."

    # Test parsing valid JSON input
    steppers = list(state_machine.get_steppers())
    valid_json = '{"name": "metacognition", "arguments": {"chain_of_thought": ["Thought 1", "Thought 2"]}}'
    for char in valid_json:
        steppers = state_machine.advance_all_basic(steppers, char)

    assert any(stepper.has_reached_accept_state() for stepper in steppers), (
        "Transition to end state should return True for valid input."
    )

    for stepper in steppers:
        assert stepper.get_current_value() == {
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
        "orderedProperties": False,
        "additionalProperties": False,
    }
    state_machine = ObjectSchemaStateMachine(schema, {})
    steppers = list(state_machine.get_steppers())
    valid_json = '{\n  "'
    steppers = state_machine.advance_all_basic(steppers, valid_json)
    assert len(steppers) == 2
    steppers = state_machine.advance_all_basic(steppers, "type")
    assert len(steppers) == 1
    steppers = state_machine.advance_all_basic(steppers, '": "div"')
    assert len(steppers) == 2
    steppers = state_machine.advance_all_basic(steppers, ', "label": "hello!" \n}')
    assert len(steppers) == 1
    assert steppers[0].has_reached_accept_state()
    assert steppers[0].get_current_value() == {"type": "div", "label": "hello!"}


@pytest.mark.parametrize(
    "value, followup_value",
    [
        (10, None),
        (10, 1),
    ],
)
def test_object_schema_acceptor_edge_case(
    value: int, followup_value: int | str | None
) -> None:
    """
    Test NumberAcceptor with an integer.

    Args:
        state_machine (NumberAcceptor): The NumberAcceptor instance.
    """

    schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query string. Text only. An artificial assistant will be created to perform the search.",
            },
            "max_results": {
                "type": "integer",
                "description": "The maximum number of search results to return. Defaults to 5.",
                "default": 5,
            },
        },
        "required": ["query"],
    }
    state_machine = ObjectSchemaStateMachine(schema, {})
    steppers = state_machine.get_steppers()
    raw_input = '{"query": "popular favorite Pokémon",  "max_results": '
    steppers = state_machine.advance_all_basic(steppers, raw_input)
    assert len(steppers) == 3
    steppers = state_machine.advance_all_basic(steppers, str(value))
    steppers = state_machine.advance_all_basic(
        steppers, str(followup_value or "") + "}"
    )
    assert len(steppers) == 1, "Should have one stepper."
    assert steppers[0].has_reached_accept_state()


@pytest.mark.parametrize(
    "value, followup_value",
    [
        ("Hello", "!"),
    ],
)
def test_object_schema_acceptor_edge_case_2(value: str, followup_value: str) -> None:
    """
    Test NumberAcceptor with an integer.

    Args:
        state_machine (NumberAcceptor): The NumberAcceptor instance.
    """

    schema = {
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
    state_machine = ObjectSchemaStateMachine(schema, {})
    steppers = state_machine.get_steppers()
    raw_input = '{"name": "send_message", "arguments": {"message": "'
    steppers = state_machine.advance_all_basic(steppers, raw_input)
    assert len(steppers) == 3

    steppers = state_machine.advance_all_basic(steppers, str(value))
    assert len(steppers) == 3

    steppers = state_machine.advance_all_basic(steppers, followup_value)
    steppers = state_machine.advance_all_basic(steppers, '"}}')
    assert len(steppers) == 1
    assert steppers[0].has_reached_accept_state()


def test_object_with_additional_properties_dict(base_context: dict[str, Any]) -> None:
    """
    Test object schema with additionalProperties as a schema dictionary.
    Tests the branch in get_property_state_machines where additionalProperties is a dict.
    """
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
        },
        "additionalProperties": {"type": "number"},
        "required": ["name"],
    }
    state_machine = ObjectSchemaStateMachine(schema, base_context)

    # Parse valid JSON with additional properties
    steppers = state_machine.get_steppers()
    steppers = state_machine.advance_all_basic(steppers, '{"name": "test"')
    json_input = ', "additional1": 42, "additional2": 123}'

    for char in json_input:
        steppers = state_machine.advance_all_basic(steppers, char)

    assert any(stepper.has_reached_accept_state() for stepper in steppers), (
        "Object with additional properties that match schema should be accepted"
    )

    for stepper in steppers:
        if stepper.has_reached_accept_state():
            value = stepper.get_current_value()
            assert value["name"] == "test"
            assert value["additional1"] == 42
            assert value["additional2"] == 123


def test_object_with_additional_properties_true(base_context: dict[str, Any]) -> None:
    """
    Test object schema with additionalProperties as boolean true.
    Tests the branch in get_property_state_machines where additionalProperties is True.
    """
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
        },
        "additionalProperties": True,  # Allow any additional properties
        "required": ["name"],
    }
    state_machine = ObjectSchemaStateMachine(schema, base_context)

    # Parse valid JSON with additional properties of various types
    steppers = state_machine.get_steppers()
    json_input = '{"name": "test", "additional1": 42, "additional2": "string", "additional3": true}'

    for char in json_input:
        steppers = state_machine.advance_all_basic(steppers, char)

    assert any(stepper.has_reached_accept_state() for stepper in steppers), (
        "Object with additional properties should be accepted when additionalProperties is true"
    )

    for stepper in steppers:
        if stepper.has_reached_accept_state():
            value = stepper.get_current_value()
            assert value["name"] == "test"
            assert value["additional1"] == 42
            assert value["additional2"] == "string"
            assert value["additional3"] is True

@pytest.mark.skip(reason="TODO: fix equality check")
def test_object_equality_check(base_context: dict[str, Any]) -> None:
    """
    Test the object schema equality operator.
    """
    schema1 = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
    }
    schema2 = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
    }
    schema3 = {
        "type": "object",
        "properties": {"age": {"type": "number"}},
    }

    state_machine1 = ObjectSchemaStateMachine(schema1, base_context)
    state_machine2 = ObjectSchemaStateMachine(schema2, base_context)
    state_machine3 = ObjectSchemaStateMachine(schema3, base_context)

    # Test equality between same schemas
    assert state_machine1 == state_machine2, "State machines with same schema should be equal"

    # Test inequality between different schemas
    assert state_machine1 != state_machine3, "State machines with different schema should not be equal"
