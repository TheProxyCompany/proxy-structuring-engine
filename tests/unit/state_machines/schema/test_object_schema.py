from typing import Any

import pytest

from pse.state_machines.schema.object_schema import ObjectSchemaStateMachine


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


# def test_value_started_hook_not_string(base_context: dict[str, Any]) -> None:
#     """
#     Test that the value_started_hook is not called prematurely.
#     """
#     schema = {
#         "type": "object",
#         "properties": {
#             "id": {"type": "number"},
#             "email": {"type": "string"},
#         },
#         "required": ["id"],
#     }

#     state_machine = ObjectSchemaStateMachine(schema, base_context)
#     steppers = list(state_machine.get_steppers())
#     for char in '{"id':
#         steppers = [stepper for _, stepper in state_machine.advance_all(steppers, char)]
#     # The hook should not be called yet

#     # Continue parsing
#     for char in '": 123}':
#         steppers = [stepper for _, stepper in state_machine.advance_all(steppers, char)]

#     assert any(
#         stepper.has_reached_accept_state() for stepper in steppers
#     ), "Transition to end state should return True when required properties are present."
#     started_hook.assert_not_called()


# def test_value_started_hook(base_context: dict[str, Any]) -> None:
#     """
#     Test that the value_started_hook is called with the correct property name when a property's value starts.
#     """
#     schema = {
#         "type": "object",
#         "properties": {"id": {"type": "number"}, "email": {"type": "string"}},
#         "required": ["id"],
#     }

#     started_hook: MagicMock = MagicMock()
#     state_machine = ObjectSchemaStateMachine(
#         schema, base_context, start_hook=started_hook
#     )
#     steppers = list(state_machine.get_steppers())
#     for char in '{"email": ':
#         steppers = [stepper for _, stepper in state_machine.advance_all(steppers, char)]

#     started_hook.assert_not_called()
#     # Simulate starting a string value
#     steppers = [stepper for _, stepper in state_machine.advance_all(steppers, '"')]
#     # The hook should be called now
#     started_hook.assert_called_once()


# def test_value_ended_hook(base_context: dict[str, Any]) -> None:
#     """
#     Test that the value_ended_hook is called with the correct property name and value when a property's value ends.
#     """
#     schema = {
#         "type": "object",
#         "properties": {
#             "id": {"type": "string"},
#         },
#     }
#     ended_hook: MagicMock = MagicMock()
#     state_machine = ObjectSchemaStateMachine(schema, base_context, end_hook=ended_hook)
#     steppers = list(state_machine.get_steppers())

#     additional_chars = '{"id": "hi'
#     for char in additional_chars:
#         steppers = [stepper for _, stepper in state_machine.advance_all(steppers, char)]
#     ended_hook.assert_not_called()

#     # Finish the string and the object
#     for char in '"}':
#         steppers = [stepper for _, stepper in state_machine.advance_all(steppers, char)]

#     assert any(
#         stepper.has_reached_accept_state() for stepper in steppers
#     ), "Transition to end state should return True when all properties are present."
#     ended_hook.assert_called_once()


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
