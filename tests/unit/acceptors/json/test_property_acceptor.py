import pytest
from pse.acceptors.json.property_acceptor import PropertyAcceptor, PropertyWalker
from pse.state_machine.state_machine import StateMachine


@pytest.fixture
def property_acceptor() -> PropertyAcceptor:
    return PropertyAcceptor()


def test_walker_initialization(property_acceptor: PropertyAcceptor):
    walker = PropertyWalker(property_acceptor)
    assert walker.prop_name is None
    assert walker.prop_value is None
    assert walker.can_accept_more_input() is True


@pytest.mark.parametrize(
    "input_string, expected_name, expected_value",
    [
        ('"key": "value"', "key", "value"),
        ('"complex_key": {"nested": "value"}', "complex_key", {"nested": "value"}),
        ('"": "empty_key"', "", "empty_key"),
        ('"unicode_key": "unicode_value🎉"', "unicode_key", "unicode_value🎉"),
        ('"spaced_key"  :  "spaced_value"', "spaced_key", "spaced_value"),
    ],
)
def test_property_parsing(
    property_acceptor: PropertyAcceptor, input_string, expected_name, expected_value
):
    walkers = list(property_acceptor.get_walkers())
    for char in input_string:
        walkers = list(StateMachine.advance_all(walkers, char))

    accepted_walkers = [
        walker for walker in walkers if walker.has_reached_accept_state()
    ]
    assert accepted_walkers, f"No walker reached an accepted state for: {input_string}"

    for walker in accepted_walkers:
        name, value = walker.get_current_value()
        assert name == expected_name
        assert value == expected_value


@pytest.mark.parametrize(
    "invalid_input",
    [
        'key: "value"',  # missing quotes around key
        '"key" "value"',  # missing colon
        '"key":',  # missing value
        '"key": value',  # unquoted value
        ':"value"',  # missing key
    ],
)
def test_invalid_property_formats(property_acceptor: PropertyAcceptor, invalid_input):
    walkers = list(property_acceptor.get_walkers())
    for char in invalid_input:
        walkers = list(StateMachine.advance_all(walkers, char))

    assert not any(
        walker.has_reached_accept_state() for walker in walkers
    ), f"Walker should not reach accepted state for invalid input: {invalid_input}"


def test_walker_state_transitions(property_acceptor: PropertyAcceptor):
    walker = PropertyWalker(property_acceptor)

    # Test property name transition
    assert walker.should_complete_transition("test_key", False)
    assert walker.prop_name == "test_key"

    # Test property value transition
    assert walker.should_complete_transition("test_value", True)
    assert walker.prop_value == "test_value"

    # Verify final value
    name, value = walker.get_current_value()
    assert name == "test_key"
    assert value == "test_value"


def test_walker_in_value_state(property_acceptor: PropertyAcceptor):
    walker = PropertyWalker(property_acceptor)

    # Not in value state initially
    assert not walker.is_within_value()

    # Set state to value parsing (state 4)
    walker.current_state = 4
    assert walker.is_within_value()
