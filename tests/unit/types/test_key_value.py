import pytest

from pse.types.base.phrase import PhraseStateMachine
from pse.types.key_value import KeyValueStateMachine


@pytest.mark.parametrize(
    "input_string, expected_name, expected_value",
    [
        ('"key": "value"', "key", "value"),
        ('"complex_key": {"nested": "value"}', "complex_key", {"nested": "value"}),
        ('"": "empty_key"', "", None),
        ('"unicode_key": "unicode_value🎉"', "unicode_key", "unicode_value🎉"),
        ('"spaced_key"  :  "spaced_value"', "spaced_key", "spaced_value"),
    ],
)
def test_property_parsing(input_string, expected_name, expected_value):
    sm = KeyValueStateMachine()
    steppers = sm.get_steppers()
    for char in input_string:
        steppers = sm.advance_all_basic(steppers, char)

    accepted_steppers = [
        stepper for stepper in steppers if stepper.has_reached_accept_state()
    ]
    assert accepted_steppers, (
        f"No stepper reached an accepted state for: {input_string}"
    )

    for stepper in accepted_steppers:
        name, value = stepper.get_current_value()
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
def test_invalid_property_formats(invalid_input):
    sm = KeyValueStateMachine()
    steppers = sm.get_steppers()
    for char in invalid_input:
        steppers = sm.advance_all_basic(steppers, char)

    assert not any(stepper.has_reached_accept_state() for stepper in steppers), (
        f"Stepper should not reach accepted state for invalid input: {invalid_input}"
    )


def test_empty_key_value():
    sm = KeyValueStateMachine()
    steppers = sm.get_steppers()
    assert len(steppers) == 1
    assert not steppers[0].should_complete_step()
    assert steppers[0].get_current_value() == ("", None)


def test_invalid_sub_stepper_json():
    sm = KeyValueStateMachine()
    steppers = sm.get_steppers()
    assert len(steppers) == 1
    stepper = steppers[0]
    stepper.sub_stepper = PhraseStateMachine('"invalid json').get_new_stepper()
    steppers = sm.advance_all_basic(steppers, '"invalid json')
    assert len(steppers) == 1
    assert not steppers[0].should_complete_step()
