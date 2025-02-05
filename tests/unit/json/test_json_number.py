from typing import Any

import pytest

from pse.json.json_number import NumberSchemaStateMachine


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
    This tests the should_start_step override in NumberSchemaStepper.
    """
    schema = {"type": "integer"}
    state_machine = NumberSchemaStateMachine(schema)

    # Test that integer part advances normally
    steppers = state_machine.get_steppers()
    for char in "123":
        steppers = state_machine.advance_all_basic(steppers, char)
        assert len(steppers) > 0, f"Should accept digit {char}"

    # Test that decimal point is rejected
    steppers = state_machine.advance_all_basic(steppers, ".")
    assert len(steppers) == 0, "Integer schema should reject decimal point transition"


def test_integer_schema_exponential_transition() -> None:
    """
    Test that integer schema allows exponential notation but validates final value.
    """
    schema = {"type": "integer"}
    state_machine = NumberSchemaStateMachine(schema)

    # Test integer with exponential notation that results in integer
    steppers = state_machine.get_steppers()
    for char in "1e2":  # 100
        steppers = state_machine.advance_all_basic(steppers, char)
        assert len(steppers) > 0, f"Should accept char {char}"
    assert any(stepper.has_reached_accept_state() for stepper in steppers)

    # Test integer with exponential notation that results in decimal
    steppers = state_machine.get_steppers()
    for char in "1e-1":  # 0.1
        steppers = state_machine.advance_all_basic(steppers, char)
    assert not any(stepper.has_reached_accept_state() for stepper in steppers)


def test_schema_validation_during_transitions() -> None:
    """
    Test that schema validation happens at the right time during transitions.
    This tests the should_complete_step override in NumberSchemaStepper.
    """
    schema = {"type": "number", "minimum": 10, "maximum": 20}
    state_machine = NumberSchemaStateMachine(schema)

    # Test partial number (should not validate constraints yet)
    steppers = state_machine.get_steppers()
    for char in "1":  # Below minimum but incomplete
        steppers = state_machine.advance_all_basic(steppers, char)
        assert len(steppers) > 0, "Should allow partial numbers during transitions"

    # Complete the number to valid value
    for char in "5":  # Now 15, within range
        steppers = state_machine.advance_all_basic(steppers, char)
    assert any(stepper.has_reached_accept_state() for stepper in steppers)

    # Test complete number outside range
    steppers = state_machine.get_steppers()
    for char in "25":  # Above maximum
        steppers = state_machine.advance_all_basic(steppers, char)
    assert not any(stepper.has_reached_accept_state() for stepper in steppers)


def test_multiple_constraints_during_transitions() -> None:
    """
    Test that multiple constraints are checked during transitions.
    """
    schema = {"type": "integer", "minimum": 10, "maximum": 100, "multipleOf": 5}
    state_machine = NumberSchemaStateMachine(schema)

    # Test building up to valid number
    steppers = state_machine.get_steppers()
    for char in "15":  # Valid: integer, >= 10, <= 100, multiple of 5
        steppers = state_machine.advance_all_basic(steppers, char)
    assert any(stepper.has_reached_accept_state() for stepper in steppers)

    # Test building up to invalid number
    steppers = state_machine.get_steppers()
    for char in "23":  # Invalid: not multiple of 5
        steppers = state_machine.advance_all_basic(steppers, char)
    assert not any(stepper.has_reached_accept_state() for stepper in steppers)


def test_decimal_validation_with_integer_schema() -> None:
    """
    Test validation of decimal numbers with integer schema.
    Tests both the transition prevention and final validation.
    """
    schema = {"type": "integer"}
    state_machine = NumberSchemaStateMachine(schema)

    # Test integer-like decimal (1.0)
    steppers = state_machine.get_steppers()
    for char in "1.0":
        steppers = state_machine.advance_all_basic(steppers, char)
    assert not any(stepper.has_reached_accept_state() for stepper in steppers)

    # Test obvious decimal (1.5)
    steppers = state_machine.get_steppers()
    for char in "1.5":
        steppers = state_machine.advance_all_basic(steppers, char)
    assert not any(stepper.has_reached_accept_state() for stepper in steppers)


def test_negative_number_transitions() -> None:
    """
    Test transitions with negative numbers and schema constraints.
    """
    schema = {"type": "number", "minimum": -10, "maximum": 10}
    state_machine = NumberSchemaStateMachine(schema)

    # Test valid negative number
    steppers = state_machine.get_steppers()
    for char in "-5":
        steppers = state_machine.advance_all_basic(steppers, char)
    assert any(stepper.has_reached_accept_state() for stepper in steppers)

    # Test negative number below minimum
    steppers = state_machine.get_steppers()
    for char in "-15":
        steppers = state_machine.advance_all_basic(steppers, char)
    assert not any(stepper.has_reached_accept_state() for stepper in steppers)


def test_exponential_notation_transitions() -> None:
    """
    Test transitions with exponential notation and schema constraints.
    """
    schema = {"type": "number", "minimum": 1, "maximum": 1000}
    state_machine = NumberSchemaStateMachine(schema)

    # Test valid exponential notation
    steppers = state_machine.get_steppers()
    for char in "1e2":  # 100
        steppers = state_machine.advance_all_basic(steppers, char)
    assert any(stepper.has_reached_accept_state() for stepper in steppers)

    # Test exponential notation exceeding maximum
    steppers = state_machine.get_steppers()
    for char in "1e4":  # 10000
        steppers = state_machine.advance_all_basic(steppers, char)
    assert not any(stepper.has_reached_accept_state() for stepper in steppers)


def test_validation_timing() -> None:
    """
    Test that validation happens at the correct time during parsing.
    """
    schema = {"type": "number", "minimum": 100}
    state_machine = NumberSchemaStateMachine(schema)

    steppers = state_machine.get_steppers()

    # Test that partial numbers don't trigger validation
    for char in "12":  # Below minimum but incomplete
        steppers = state_machine.advance_all_basic(steppers, char)
        assert len(steppers) > 0, "Should allow partial numbers"

    # Test that completing the number triggers validation
    for char in "3":  # Now 123, above minimum
        steppers = state_machine.advance_all_basic(steppers, char)
    assert any(stepper.has_reached_accept_state() for stepper in steppers)


def test_schema_stepper_creation() -> None:
    """
    Test that NumberSchemaStateMachine creates the correct stepper type.
    """
    schema = {"type": "number"}
    state_machine = NumberSchemaStateMachine(schema)
    stepper = state_machine.get_new_stepper()

    # Test stepper type
    assert stepper.__class__.__name__ == "NumberSchemaStepper"

    # Test stepper behavior
    steppers = state_machine.get_steppers()
    for char in "12.34":
        steppers = state_machine.advance_all_basic(steppers, char)
    assert any(stepper.has_reached_accept_state() for stepper in steppers)
