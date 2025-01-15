import pytest

from pse.state_machines.types.enum import EnumStateMachine


def test_accept_valid_enum_value():
    """Test that the state machine correctly accepts a value present in the enum."""
    sm = EnumStateMachine(["value1", "value2", "value3"], require_quotes=False)
    walkers = sm.get_walkers()
    advanced = list(EnumStateMachine.advance_all(walkers, "value1"))
    walkers = [walker for _, walker in advanced]

    assert any(walker.has_reached_accept_state() for walker in walkers)
    for walker in walkers:
        if walker.has_reached_accept_state():
            assert walker.get_current_value() == "value1"


def test_reject_invalid_enum_value():
    """Test that the state machine correctly rejects a value not present in the enum."""
    sm = EnumStateMachine(["value1", "value2", "value3"], require_quotes=False)
    walkers = sm.get_walkers()
    advanced = list(EnumStateMachine.advance_all(walkers, "invalid_value"))
    walkers = [walker for _, walker in advanced]

    assert not any(walker.has_reached_accept_state() for walker in walkers)


@pytest.mark.parametrize("value", ["value1", "value2", "value3"])
def test_accept_multiple_enum_values(value):
    """Test that the state machine correctly accepts multiple different valid enum values."""
    sm = EnumStateMachine(["value1", "value2", "value3"], require_quotes=False)
    walkers = sm.get_walkers()
    advanced = list(EnumStateMachine.advance_all(walkers, value))
    walkers = [walker for _, walker in advanced]

    assert any(walker.has_reached_accept_state() for walker in walkers)
    for walker in walkers:
        if walker.has_reached_accept_state():
            assert walker.get_current_value() == value


def test_partial_enum_value_rejection():
    """Test that the state machine does not accept prefixes of valid enum values."""
    sm = EnumStateMachine(["value1", "value2", "value3"], require_quotes=False)
    walkers = sm.get_walkers()
    advanced = list(EnumStateMachine.advance_all(walkers, "val"))
    walkers = [walker for _, walker in advanced]

    assert not any(walker.has_reached_accept_state() for walker in walkers)


def test_init_with_empty_enum():
    """Test initializing EnumStateMachine with empty enum values raises ValueError."""
    with pytest.raises(ValueError):
        EnumStateMachine(enum_values=[])


@pytest.mark.parametrize("special_value", ["val!@#", "val-123", "val_😊"])
def test_accept_enum_with_special_characters(special_value):
    """Test that the state machine correctly handles enum values with special characters."""
    sm = EnumStateMachine([special_value], require_quotes=False)
    walkers = sm.get_walkers()
    advanced = list(EnumStateMachine.advance_all(walkers, special_value))
    walkers = [walker for _, walker in advanced]

    assert any(walker.has_reached_accept_state() for walker in walkers)
    for walker in walkers:
        if walker.has_reached_accept_state():
            assert walker.get_current_value() == special_value


def test_char_by_char_enum_parsing():
    """Test parsing enum values character by character."""
    sm = EnumStateMachine(["value1", "value2", "value3"], require_quotes=False)
    walkers = sm.get_walkers()

    for char in "value1":
        advanced = list(EnumStateMachine.advance_all(walkers, char))
        walkers = [walker for _, walker in advanced]
        if not walkers:
            break

    assert any(walker.has_reached_accept_state() for walker in walkers)
    for walker in walkers:
        if walker.has_reached_accept_state():
            assert walker.get_current_value() == "value1"


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
    walkers = sm.get_walkers()
    advanced = list(EnumStateMachine.advance_all(walkers, value))
    walkers = [walker for _, walker in advanced]

    for walker in walkers:
        assert walker.has_reached_accept_state()
        assert walker.get_current_value() == value


def test_enum_without_quotes():
    """Test enum values without quotes requirement."""
    sm = EnumStateMachine(["test"], require_quotes=False)
    walkers = sm.get_walkers()
    advanced = list(EnumStateMachine.advance_all(walkers, "test"))
    walkers = [walker for _, walker in advanced]

    assert any(walker.has_reached_accept_state() for walker in walkers)
    for walker in walkers:
        if walker.has_reached_accept_state():
            assert walker.get_current_value() == "test"


def test_enum_requires_quotes_by_default():
    """Test that enum values require quotes by default."""
    sm = EnumStateMachine(["test"])  # require_quotes defaults to True
    walkers = sm.get_walkers()
    advanced = list(EnumStateMachine.advance_all(walkers, "test"))  # no quotes
    walkers = [walker for _, walker in advanced]

    assert not any(walker.has_reached_accept_state() for walker in walkers)
