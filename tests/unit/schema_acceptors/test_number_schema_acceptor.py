import pytest
from typing import Any, Dict

from pse.schema_acceptors.number_schema_acceptor import NumberSchemaAcceptor, NumberSchemaWalker
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
def test_validate_value_minimum(value: float, expected: bool) -> None:
    """
    Test that 'minimum' constraint is enforced correctly.
    """
    schema: Dict[str, Any] = {"type": "number", "minimum": 10}
    acceptor = NumberSchemaAcceptor(schema)
    assert (
        acceptor.validate_value(value) == expected
    ), f"Value {value} should be {'valid' if expected else 'invalid'} with 'minimum' constraint."


@pytest.mark.parametrize(
    "value, expected",
    [
        (10, False),
        (15, True),
        (5, False),
    ],
)
def test_validate_value_exclusive_minimum(value: float, expected: bool) -> None:
    """
    Test that 'exclusiveMinimum' constraint is enforced correctly.
    """
    schema: Dict[str, Any] = {"type": "number", "exclusiveMinimum": 10}
    acceptor = NumberSchemaAcceptor(schema)
    assert (
        acceptor.validate_value(value) == expected
    ), f"Value {value} should be {'valid' if expected else 'invalid'} with 'exclusiveMinimum' constraint."


@pytest.mark.parametrize(
    "value, expected",
    [
        (20, True),
        (15, True),
        (25, False),
    ],
)
def test_validate_value_maximum(value: float, expected: bool) -> None:
    """
    Test that 'maximum' constraint is enforced correctly.
    """
    schema: Dict[str, Any] = {"type": "number", "maximum": 20}
    acceptor = NumberSchemaAcceptor(schema)
    assert (
        acceptor.validate_value(value) == expected
    ), f"Value {value} should be {'valid' if expected else 'invalid'} with 'maximum' constraint."


@pytest.mark.parametrize(
    "value, expected",
    [
        (20, False),
        (15, True),
        (25, False),
    ],
)
def test_validate_value_exclusive_maximum(value: float, expected: bool) -> None:
    """
    Test that 'exclusiveMaximum' constraint is enforced correctly.
    """
    schema: Dict[str, Any] = {"type": "number", "exclusiveMaximum": 20}
    acceptor = NumberSchemaAcceptor(schema)
    assert (
        acceptor.validate_value(value) == expected
    ), f"Value {value} should be {'valid' if expected else 'invalid'} with 'exclusiveMaximum' constraint."


@pytest.mark.parametrize(
    "value, expected",
    [
        (15, True),
        (20, True),
        (12, False),
        (17.5, False),
    ],
)
def test_validate_value_multiple_of(value: float, expected: bool) -> None:
    """
    Test that 'multipleOf' constraint is enforced correctly.
    """
    schema: Dict[str, Any] = {"type": "number", "multipleOf": 5}
    acceptor = NumberSchemaAcceptor(schema)
    assert (
        acceptor.validate_value(value) == expected
    ), f"Value {value} should be {'valid' if expected else 'invalid'} with 'multipleOf' constraint."


@pytest.mark.parametrize(
    "value_str, expected",
    [
        ("15", True),
        ("25", False),
        ("5", False),
    ],
)
def test_complete_transition_with_validation(value_str: str, expected: bool) -> None:
    """
    Test complete_transition method with validation.
    Ensures that valid values are accepted and invalid values are rejected.
    """
    schema: Dict[str, Any] = {"type": "number", "minimum": 10, "maximum": 20}
    acceptor = NumberSchemaAcceptor(schema)
    walker = NumberSchemaWalker(acceptor)
    result = walker.should_complete_transition()
    assert (
        result == expected
    ), f"Value '{value_str}' should be {'accepted' if expected else 'rejected'} by complete_transition."


@pytest.mark.parametrize(
    "value, expected",
    [
        (10, True),
        (10.5, False),
        ("10", False),
    ],
)
def test_validate_value_integer_type(value: Any, expected: bool) -> None:
    """
    Test that only integer values are considered valid when type is 'integer'.
    """
    schema: Dict[str, Any] = {"type": "integer"}
    acceptor = NumberSchemaAcceptor(schema)
    assert (
        acceptor.validate_value(value) == expected
    ), f"Value {value} should be {'valid' if expected else 'invalid'} for 'integer' type."


@pytest.mark.parametrize(
    "json_input, expected",
    [
        ('{"value": 20}', True),
        ('{"value": 25}', False),
    ],
)
def test_number_with_object_acceptor(json_input: str, expected: bool) -> None:
    """
    Test NumberSchemaAcceptor within an ObjectSchemaAcceptor.
    Ensures that numeric values in objects adhere to schema constraints.
    """
    schema: Dict[str, Any] = {
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

    walkers = acceptor.get_walkers()
    for char in json_input:
        walkers = [walker for _, walker in acceptor.advance_all_walkers(walkers, char)]
        if not walkers:
            break

    is_accepted: bool = any(walker.has_reached_accept_state() for walker in walkers)
    assert (
        is_accepted == expected
    ), f"JSON input '{json_input}' should be {'accepted' if expected else 'rejected'} by ObjectSchemaAcceptor."


@pytest.mark.parametrize(
    "input_string, expected",
    [
        ("The number is 30", True),
        ("The number is 5", False),
    ],
)
def test_number_with_text_acceptor(input_string: str, expected: bool) -> None:
    """
    Test NumberSchemaAcceptor in combination with TextAcceptor.
    Ensures that numeric values within text are correctly validated.
    """
    schema: Dict[str, Any] = {"type": "number", "minimum": 10, "maximum": 50}
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

    walkers = state_machine.get_walkers()
    for char in input_string:
        walkers = [walker for _, walker in state_machine.advance_all_walkers(walkers, char)]
        if not walkers:
            break

    is_accepted: bool = any(walker.has_reached_accept_state() for walker in walkers)
    assert (
        is_accepted == expected
    ), f"Input '{input_string}' should be {'accepted' if expected else 'rejected'} by state machine."


@pytest.mark.parametrize(
    "value_str, expected",
    [
        ("30", True),
        ("5", False),
        ("150", False),
        ("25", False),
    ],
)
def test_number_with_walker_validation(value_str: str, expected: bool) -> None:
    """
    Test NumberSchemaAcceptor's walker validation in various scenarios.
    Ensures that walker correctly validates numbers based on schema constraints.
    """
    schema: Dict[str, Any] = {"type": "number", "minimum": 10, "maximum": 100, "multipleOf": 10}
    acceptor = NumberSchemaAcceptor(schema)
    walker = NumberSchemaWalker(acceptor)
    result = walker.should_complete_transition()
    assert (
        result == expected
    ), f"Value '{value_str}' should be {'accepted' if expected else 'rejected'} by walker validation."
