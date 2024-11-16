import pytest
from pse.acceptors.json.property_acceptor import PropertyAcceptor
from pse.core.state_machine import StateMachine


@pytest.fixture
def property_acceptor() -> PropertyAcceptor:
    return PropertyAcceptor()


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
        walkers = [
            walker for _, walker in StateMachine.advance_all_walkers(walkers, char)
        ]

    accepted_walkers = [
        walker for walker in walkers if walker.has_reached_accept_state()
    ]
    assert accepted_walkers, f"No walker reached an accepted state for: {input_string}"

    for walker in accepted_walkers:
        name, value = walker.current_value
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
        walkers = [
            walker for _, walker in StateMachine.advance_all_walkers(walkers, char)
        ]

    assert not any(
        walker.has_reached_accept_state() for walker in walkers
    ), f"Walker should not reach accepted state for invalid input: {invalid_input}"
