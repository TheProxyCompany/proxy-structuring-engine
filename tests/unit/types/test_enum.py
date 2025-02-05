import pytest

from pse.types.enum import EnumStateMachine


def test_accept_valid_enum_value():
    """Test that the state machine correctly accepts a value present in the enum."""
    sm = EnumStateMachine(["value1", "value2", "value3"], require_quotes=False)
    steppers = sm.get_steppers()
    steppers = sm.advance_all_basic(steppers, "value1")

    assert any(stepper.has_reached_accept_state() for stepper in steppers)
    for stepper in steppers:
        if stepper.has_reached_accept_state():
            assert stepper.get_current_value() == "value1"


def test_reject_invalid_enum_value():
    """Test that the state machine correctly rejects a value not present in the enum."""
    sm = EnumStateMachine(["value1", "value2", "value3"], require_quotes=False)
    steppers = sm.get_steppers()
    steppers = sm.advance_all_basic(steppers, "invalid_value")

    assert not any(stepper.has_reached_accept_state() for stepper in steppers)


@pytest.mark.parametrize("value", ["value1", "value2", "value3"])
def test_accept_multiple_enum_values(value):
    """Test that the state machine correctly accepts multiple different valid enum values."""
    sm = EnumStateMachine(["value1", "value2", "value3"], require_quotes=False)
    steppers = sm.get_steppers()
    steppers = sm.advance_all_basic(steppers, value)

    assert any(stepper.has_reached_accept_state() for stepper in steppers)
    for stepper in steppers:
        if stepper.has_reached_accept_state():
            assert stepper.get_current_value() == value


def test_partial_enum_value_rejection():
    """Test that the state machine does not accept prefixes of valid enum values."""
    sm = EnumStateMachine(["value1", "value2", "value3"], require_quotes=False)
    steppers = sm.get_steppers()
    steppers = sm.advance_all_basic(steppers, "val")

    assert not any(stepper.has_reached_accept_state() for stepper in steppers)


def test_init_with_empty_enum():
    """Test initializing EnumStateMachine with empty enum values raises ValueError."""
    with pytest.raises(ValueError):
        EnumStateMachine(enum_values=[])


@pytest.mark.parametrize("special_value", ["val!@#", "val-123", "val_😊"])
def test_accept_enum_with_special_characters(special_value):
    """Test that the state machine correctly handles enum values with special characters."""
    sm = EnumStateMachine([special_value], require_quotes=False)
    steppers = sm.get_steppers()
    steppers = sm.advance_all_basic(steppers, special_value)

    assert any(stepper.has_reached_accept_state() for stepper in steppers)
    for stepper in steppers:
        if stepper.has_reached_accept_state():
            assert stepper.get_current_value() == special_value


def test_char_by_char_enum_parsing():
    """Test parsing enum values character by character."""
    sm = EnumStateMachine(["value1", "value2", "value3"], require_quotes=False)
    steppers = sm.get_steppers()

    for char in "value1":
        steppers = sm.advance_all_basic(steppers, char)
        if not steppers:
            break

    assert any(stepper.has_reached_accept_state() for stepper in steppers)
    for stepper in steppers:
        if stepper.has_reached_accept_state():
            assert stepper.get_current_value() == "value1"


@pytest.mark.parametrize(
    "value",
    [
        '"test"',
        "'test'",
    ],
)
def test_enum_with_quotes(value):
    """Test enum values with quotes requirement (default behavior)."""
    sm = EnumStateMachine(["test"])  # require_quotes defaults to True
    steppers = sm.get_steppers()
    steppers = sm.advance_all_basic(steppers, value)

    for stepper in steppers:
        assert stepper.has_reached_accept_state()
        assert stepper.get_current_value() == "test"


def test_enum_without_quotes():
    """Test enum values without quotes requirement."""
    sm = EnumStateMachine(["test"], require_quotes=False)
    steppers = sm.get_steppers()
    steppers = sm.advance_all_basic(steppers, "test")

    assert any(stepper.has_reached_accept_state() for stepper in steppers)
    for stepper in steppers:
        if stepper.has_reached_accept_state():
            assert stepper.get_current_value() == "test"


def test_enum_requires_quotes_by_default():
    """Test that enum values require quotes by default."""
    sm = EnumStateMachine(["test"])  # require_quotes defaults to True
    steppers = sm.get_steppers()
    steppers = sm.advance_all_basic(steppers, "test")  # no quotes

    assert not any(stepper.has_reached_accept_state() for stepper in steppers)
