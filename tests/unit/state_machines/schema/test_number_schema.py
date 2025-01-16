from typing import Any

import pytest

from pse.state_machines.schema.number_schema import NumberSchemaStateMachine


@pytest.fixture
def basic_schema() -> dict[str, Any]:
    """Basic number schema with no constraints."""
    return {"type": "number"}


@pytest.fixture
def integer_schema() -> dict[str, Any]:
    """Basic integer schema with no constraints."""
    return {"type": "integer"}


def test_integer_schema_decimal_transition() -> None:
    """
    Test that integer schema prevents transitions to decimal states.
    This tests the should_start_transition override in NumberSchemaWalker.
    """
    schema = {"type": "integer"}
    state_machine = NumberSchemaStateMachine(schema)

    # Test that integer part advances normally
    walkers = state_machine.get_walkers()
    for char in "123":
        walkers = [walker for _, walker in state_machine.advance_all(walkers, char)]
        assert len(walkers) > 0, f"Should accept digit {char}"

    # Test that decimal point is rejected
    walkers = [walker for _, walker in state_machine.advance_all(walkers, ".")]
    assert len(walkers) == 0, "Integer schema should reject decimal point transition"


def test_integer_schema_exponential_transition() -> None:
    """
    Test that integer schema allows exponential notation but validates final value.
    """
    schema = {"type": "integer"}
    state_machine = NumberSchemaStateMachine(schema)

    # Test integer with exponential notation that results in integer
    walkers = state_machine.get_walkers()
    for char in "1e2":  # 100
        walkers = [walker for _, walker in state_machine.advance_all(walkers, char)]
        assert len(walkers) > 0, f"Should accept char {char}"
    assert any(walker.has_reached_accept_state() for walker in walkers)

    # Test integer with exponential notation that results in decimal
    walkers = state_machine.get_walkers()
    for char in "1e-1":  # 0.1
        walkers = [walker for _, walker in state_machine.advance_all(walkers, char)]
    assert not any(walker.has_reached_accept_state() for walker in walkers)


def test_schema_validation_during_transitions() -> None:
    """
    Test that schema validation happens at the right time during transitions.
    This tests the should_complete_transition override in NumberSchemaWalker.
    """
    schema = {"type": "number", "minimum": 10, "maximum": 20}
    state_machine = NumberSchemaStateMachine(schema)

    # Test partial number (should not validate constraints yet)
    walkers = state_machine.get_walkers()
    for char in "1":  # Below minimum but incomplete
        walkers = [walker for _, walker in state_machine.advance_all(walkers, char)]
        assert len(walkers) > 0, "Should allow partial numbers during transitions"

    # Complete the number to valid value
    for char in "5":  # Now 15, within range
        walkers = [walker for _, walker in state_machine.advance_all(walkers, char)]
    assert any(walker.has_reached_accept_state() for walker in walkers)

    # Test complete number outside range
    walkers = state_machine.get_walkers()
    for char in "25":  # Above maximum
        walkers = [walker for _, walker in state_machine.advance_all(walkers, char)]
    assert not any(walker.has_reached_accept_state() for walker in walkers)


def test_multiple_constraints_during_transitions() -> None:
    """
    Test that multiple constraints are checked during transitions.
    """
    schema = {"type": "integer", "minimum": 10, "maximum": 100, "multipleOf": 5}
    state_machine = NumberSchemaStateMachine(schema)

    # Test building up to valid number
    walkers = state_machine.get_walkers()
    for char in "15":  # Valid: integer, >= 10, <= 100, multiple of 5
        walkers = [walker for _, walker in state_machine.advance_all(walkers, char)]
    assert any(walker.has_reached_accept_state() for walker in walkers)

    # Test building up to invalid number
    walkers = state_machine.get_walkers()
    for char in "23":  # Invalid: not multiple of 5
        walkers = [walker for _, walker in state_machine.advance_all(walkers, char)]
    assert not any(walker.has_reached_accept_state() for walker in walkers)


def test_decimal_validation_with_integer_schema() -> None:
    """
    Test validation of decimal numbers with integer schema.
    Tests both the transition prevention and final validation.
    """
    schema = {"type": "integer"}
    state_machine = NumberSchemaStateMachine(schema)

    # Test integer-like decimal (1.0)
    walkers = state_machine.get_walkers()
    for char in "1.0":
        walkers = [walker for _, walker in state_machine.advance_all(walkers, char)]
    assert not any(walker.has_reached_accept_state() for walker in walkers)

    # Test obvious decimal (1.5)
    walkers = state_machine.get_walkers()
    for char in "1.5":
        walkers = [walker for _, walker in state_machine.advance_all(walkers, char)]
    assert not any(walker.has_reached_accept_state() for walker in walkers)


def test_negative_number_transitions() -> None:
    """
    Test transitions with negative numbers and schema constraints.
    """
    schema = {"type": "number", "minimum": -10, "maximum": 10}
    state_machine = NumberSchemaStateMachine(schema)

    # Test valid negative number
    walkers = state_machine.get_walkers()
    for char in "-5":
        walkers = [walker for _, walker in state_machine.advance_all(walkers, char)]
    assert any(walker.has_reached_accept_state() for walker in walkers)

    # Test negative number below minimum
    walkers = state_machine.get_walkers()
    for char in "-15":
        walkers = [walker for _, walker in state_machine.advance_all(walkers, char)]
    assert not any(walker.has_reached_accept_state() for walker in walkers)


def test_exponential_notation_transitions() -> None:
    """
    Test transitions with exponential notation and schema constraints.
    """
    schema = {"type": "number", "minimum": 1, "maximum": 1000}
    state_machine = NumberSchemaStateMachine(schema)

    # Test valid exponential notation
    walkers = state_machine.get_walkers()
    for char in "1e2":  # 100
        walkers = [walker for _, walker in state_machine.advance_all(walkers, char)]
    assert any(walker.has_reached_accept_state() for walker in walkers)

    # Test exponential notation exceeding maximum
    walkers = state_machine.get_walkers()
    for char in "1e4":  # 10000
        walkers = [walker for _, walker in state_machine.advance_all(walkers, char)]
    assert not any(walker.has_reached_accept_state() for walker in walkers)


def test_validation_timing() -> None:
    """
    Test that validation happens at the correct time during parsing.
    """
    schema = {"type": "number", "minimum": 100}
    state_machine = NumberSchemaStateMachine(schema)

    walkers = state_machine.get_walkers()

    # Test that partial numbers don't trigger validation
    for char in "12":  # Below minimum but incomplete
        walkers = [walker for _, walker in state_machine.advance_all(walkers, char)]
        assert len(walkers) > 0, "Should allow partial numbers"

    # Test that completing the number triggers validation
    for char in "3":  # Now 123, above minimum
        walkers = [walker for _, walker in state_machine.advance_all(walkers, char)]
    assert any(walker.has_reached_accept_state() for walker in walkers)


def test_schema_walker_creation() -> None:
    """
    Test that NumberSchemaStateMachine creates the correct walker type.
    """
    schema = {"type": "number"}
    state_machine = NumberSchemaStateMachine(schema)
    walker = state_machine.get_new_walker()

    # Test walker type
    assert walker.__class__.__name__ == "NumberSchemaWalker"

    # Test walker behavior
    walkers = state_machine.get_walkers()
    for char in "12.34":
        walkers = [walker for _, walker in state_machine.advance_all(walkers, char)]
    assert any(walker.has_reached_accept_state() for walker in walkers)
