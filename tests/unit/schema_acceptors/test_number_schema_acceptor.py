import pytest
from pse.schema_acceptors.number_schema_acceptor import NumberSchemaAcceptor
from pse.acceptors.basic.text_acceptor import TextAcceptor
from pse.schema_acceptors.object_schema_acceptor import ObjectSchemaAcceptor
from pse.state_machine.state_machine import StateMachine


@pytest.mark.parametrize(
    "value, expected",
    [
        (10, True),
        (15, True),
        (5, False),
    ],
)
def test_validate_value_minimum(value, expected):
    """
    Test that 'minimum' constraint is enforced correctly.
    """
    schema = {"type": "number", "minimum": 10}
    acceptor = NumberSchemaAcceptor(schema)
    assert acceptor.validate_value(value) == expected, f"Value {value} should be {'valid' if expected else 'invalid'} with 'minimum' constraint."


@pytest.mark.parametrize(
    "value, expected",
    [
        (10, False),
        (15, True),
        (5, False),
    ],
)
def test_validate_value_exclusive_minimum(value, expected):
    """
    Test that 'exclusiveMinimum' constraint is enforced correctly.
    """
    schema = {"type": "number", "exclusiveMinimum": 10}
    acceptor = NumberSchemaAcceptor(schema)
    assert acceptor.validate_value(value) == expected, f"Value {value} should be {'valid' if expected else 'invalid'} with 'exclusiveMinimum' constraint."


@pytest.mark.parametrize(
    "value, expected",
    [
        (20, True),
        (15, True),
        (25, False),
    ],
)
def test_validate_value_maximum(value, expected):
    """
    Test that 'maximum' constraint is enforced correctly.
    """
    schema = {"type": "number", "maximum": 20}
    acceptor = NumberSchemaAcceptor(schema)
    assert acceptor.validate_value(value) == expected, f"Value {value} should be {'valid' if expected else 'invalid'} with 'maximum' constraint."


@pytest.mark.parametrize(
    "value, expected",
    [
        (20, False),
        (15, True),
        (25, False),
    ],
)
def test_validate_value_exclusive_maximum(value, expected):
    """
    Test that 'exclusiveMaximum' constraint is enforced correctly.
    """
    schema = {"type": "number", "exclusiveMaximum": 20}
    acceptor = NumberSchemaAcceptor(schema)
    assert acceptor.validate_value(value) == expected, f"Value {value} should be {'valid' if expected else 'invalid'} with 'exclusiveMaximum' constraint."


@pytest.mark.parametrize(
    "value, expected",
    [
        (15, True),
        (20, True),
        (12, False),
        (17.5, False),
    ],
)
def test_validate_value_multiple_of(value, expected):
    """
    Test that 'multipleOf' constraint is enforced correctly.
    """
    schema = {"type": "number", "multipleOf": 5}
    acceptor = NumberSchemaAcceptor(schema)
    assert acceptor.validate_value(value) == expected, f"Value {value} should be {'valid' if expected else 'invalid'} with 'multipleOf' constraint."


@pytest.mark.parametrize(
    "value_str, expected",
    [
        ("15", True),
        ("25", False),
        ("5", False),
    ],
)
def test_complete_transition_with_validation(value_str, expected):
    """
    Test complete_transition method with validation.
    Ensures that valid values are accepted and invalid values are rejected.
    """
    schema = {"type": "number", "minimum": 10, "maximum": 20}
    acceptor = NumberSchemaAcceptor(schema)
    cursor = acceptor.Cursor(acceptor)
    result = cursor.complete_transition(value_str, 5, True)
    assert result == expected, f"Value '{value_str}' should be {'accepted' if expected else 'rejected'} by complete_transition."


def test_start_transition_integer_constraints():
    """
    Test that transitions to state 4 are blocked when type is 'integer' and current state is 3.
    """
    schema = {"type": "integer"}
    acceptor = NumberSchemaAcceptor(schema)
    cursor = acceptor.Cursor(acceptor)
    cursor.current_state = 3
    result = cursor.start_transition(acceptor, 4)
    assert not result, "Transition to state 4 should be blocked for 'integer' type when current_state is 3."


def test_start_transition_non_integer():
    """
    Test that transitions to state 4 are allowed when type is 'number' and current state is 3.
    """
    schema = {"type": "number"}
    acceptor = NumberSchemaAcceptor(schema)
    cursor = acceptor.Cursor(acceptor)
    cursor.current_state = 3
    result = cursor.start_transition(acceptor, 4)
    assert result, "Transition to state 4 should be allowed for 'number' type when current_state is 3."


@pytest.mark.parametrize(
    "value, expected",
    [
        (10, True),
        (10.5, False),
        ("10", False),
    ],
)
def test_validate_value_integer_type(value, expected):
    """
    Test that only integer values are considered valid when type is 'integer'.
    """
    schema = {"type": "integer"}
    acceptor = NumberSchemaAcceptor(schema)
    assert acceptor.validate_value(value) == expected, f"Value {value} should be {'valid' if expected else 'invalid'} for 'integer' type."


@pytest.mark.parametrize(
    "json_input, expected",
    [
        ('{"value": 20}', True),
        ('{"value": 25}', False),
    ],
)
def test_number_with_object_acceptor(json_input, expected):
    """
    Test NumberSchemaAcceptor within an ObjectSchemaAcceptor.
    Ensures that numeric values in objects adhere to schema constraints.
    """
    schema = {
        "type": "object",
        "properties": {
            "value": {
                "type": "number",
                "minimum": 0,
                "maximum": 100,
                "multipleOf": 10,
            }
        },
        "required": ["value"],
    }
    acceptor = ObjectSchemaAcceptor(schema, context={})

    cursors = acceptor.get_cursors()
    for char in json_input:
        cursors = acceptor.advance_all(cursors, char)
        if not cursors:
            break

    is_accepted = any(cursor.in_accepted_state() for cursor in cursors)
    assert is_accepted == expected, f"JSON input '{json_input}' should be {'accepted' if expected else 'rejected'} by ObjectSchemaAcceptor."


@pytest.mark.parametrize(
    "input_string, expected",
    [
        ("The number is 30", True),
        ("The number is 5", False),
    ],
)
def test_number_with_text_acceptor(input_string, expected):
    """
    Test NumberSchemaAcceptor in combination with TextAcceptor.
    Ensures that numeric values within text are correctly validated.
    """
    schema = {"type": "number", "minimum": 10, "maximum": 50}
    number_acceptor = NumberSchemaAcceptor(schema)
    text_acceptor = TextAcceptor("The number is ")

    state_machine = StateMachine(
        graph={
            0: [(text_acceptor, 1)],
            1: [(number_acceptor, "$")],
        },
        initial_state=0,
        end_states=["$"],
    )

    cursors = state_machine.get_cursors()
    for char in input_string:
        cursors = state_machine.advance_all(cursors, char)
        if not cursors:
            break

    is_accepted = any(cursor.in_accepted_state() for cursor in cursors)
    assert is_accepted == expected, f"Input '{input_string}' should be {'accepted' if expected else 'rejected'} by state machine."


@pytest.mark.parametrize(
    "value_str, expected",
    [
        ("30", True),
        ("5", False),
        ("150", False),
        ("25", False),
    ],
)
def test_number_with_cursor_validation(value_str, expected):
    """
    Test NumberSchemaAcceptor's cursor validation in various scenarios.
    Ensures that cursor correctly validates numbers based on schema constraints.
    """
    schema = {"type": "number", "minimum": 10, "maximum": 100, "multipleOf": 10}
    acceptor = NumberSchemaAcceptor(schema)
    cursor = acceptor.Cursor(acceptor)
    result = cursor.complete_transition(value_str, 5, True)
    assert result == expected, f"Value '{value_str}' should be {'accepted' if expected else 'rejected'} by cursor validation."
