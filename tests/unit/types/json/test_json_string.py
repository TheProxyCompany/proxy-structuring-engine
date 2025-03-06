import pytest

from pse.types.json.json_string import StringSchemaStateMachine


def test_init_default_schema():
    """
    Test initializing StringSchemaAcceptor with default schema.
    Ensures default min_length and max_length are set correctly.
    """
    state_machine = StringSchemaStateMachine(schema={})
    assert state_machine.min_length() == 0, "Default min_length should be 0."
    assert state_machine.max_length() == 10000, "Default max_length should be 10000."


def test_init_custom_min_length():
    """
    Test initializing StringSchemaAcceptor with a custom min_length.
    """
    schema = {"minLength": 5}
    state_machine = StringSchemaStateMachine(schema=schema)
    assert state_machine.min_length() == 5, "Custom min_length should be set to 5."
    assert state_machine.max_length() == 10000, (
        "Default max_length should remain 10000 when only min_length is set."
    )


def test_init_custom_max_length():
    """
    Test initializing StringSchemaAcceptor with a custom max_length.
    """
    schema = {"maxLength": 50}
    state_machine = StringSchemaStateMachine(schema=schema)
    assert state_machine.min_length() == 0, (
        "Default min_length should remain 0 when only max_length is set."
    )
    assert state_machine.max_length() == 50, "Custom max_length should be set to 50."


def test_validate_value_success():
    """
    Test validate_value with a string that meets min_length and max_length requirements.
    """
    schema = {"minLength": 3, "maxLength": 10}
    state_machine = StringSchemaStateMachine(schema=schema)
    stepper = state_machine.get_new_stepper()
    assert stepper.validate_value("test"), (
        "String 'test' should be valid as it meets min and max length requirements."
    )


def test_validate_value_too_long():
    """
    Test validate_value with a string longer than max_length.
    """
    schema = {"maxLength": 5}
    state_machine = StringSchemaStateMachine(schema=schema)
    stepper = state_machine.get_new_stepper()
    assert not stepper.validate_value("exceeds"), (
        "String 'exceeds' should be invalid as it exceeds max_length."
    )


def test_validate_value_pattern_match():
    """
    Test validate_value with a string that matches the pattern.
    """
    schema = {"pattern": r"^\d{3}-\d{2}-\d{4}$"}
    state_machine = StringSchemaStateMachine(schema=schema)
    stepper = state_machine.get_new_stepper()
    assert stepper.validate_value("123-45-6789"), (
        "String '123-45-6789' should match the pattern."
    )


def test_validate_value_pattern_no_match():
    """
    Test validate_value with a string that does not match the pattern.
    """
    schema = {"pattern": r"^\d{3}-\d{2}-\d{4}$"}
    state_machine = StringSchemaStateMachine(schema=schema)
    stepper = state_machine.get_new_stepper()
    assert not stepper.validate_value("123456789"), (
        "String '123456789' should not match the pattern."
    )


def test_validate_value_format_email_valid():
    """
    Test validate_value with a valid email format.
    """
    schema = {"format": "email"}
    state_machine = StringSchemaStateMachine(schema=schema)
    stepper = state_machine.get_new_stepper()
    assert stepper.validate_value("test@example.com"), (
        "Email 'test@example.com' should be valid."
    )


def test_validate_value_format_email_invalid():
    """
    Test validate_value with an invalid email format.
    """
    schema = {"format": "email"}
    state_machine = StringSchemaStateMachine(schema=schema)
    stepper = state_machine.get_new_stepper()
    assert not stepper.validate_value("test@com"), "Email 'test@com' should be invalid."


def test_validate_value_format_date_time_valid():
    """
    Test validate_value with a valid date-time format.
    """
    schema = {"format": "date-time"}
    state_machine = StringSchemaStateMachine(schema=schema)
    stepper = state_machine.get_new_stepper()
    assert stepper.validate_value("2023-10-20T12:34:56"), (
        "Date-time '2023-10-20T12:34:56' should be valid."
    )


def test_validate_value_format_date_time_invalid():
    """
    Test validate_value with an invalid date-time format.
    """
    schema = {"format": "date-time"}
    state_machine = StringSchemaStateMachine(schema=schema)
    stepper = state_machine.get_new_stepper()
    assert not stepper.validate_value("20-10-2023 12:34:56"), (
        "Date-time '20-10-2023 12:34:56' should be invalid."
    )


def test_validate_value_format_uri_valid():
    """
    Test validate_value with a valid URI format.
    """
    schema = {"format": "uri"}
    state_machine = StringSchemaStateMachine(schema=schema)
    stepper = state_machine.get_new_stepper()
    assert stepper.validate_value("https://www.example.com"), (
        "URI 'https://www.example.com' should be valid."
    )


def test_validate_value_format_uri_invalid():
    """
    Test validate_value with an invalid URI format.
    """
    schema = {"format": "uri"}
    state_machine = StringSchemaStateMachine(schema=schema)
    stepper = state_machine.get_new_stepper()
    assert not stepper.validate_value("www.example.com"), (
        "URI 'www.example.com' should be invalid."
    )


def test_stepper_rejects_on_pattern_mismatch_during_parsing():
    """
    Test that the stepper rejects input early if pattern does not match during parsing.
    """
    schema = {"pattern": r"abc.*"}
    state_machine = StringSchemaStateMachine(schema=schema)
    steppers = state_machine.get_steppers()
    steppers = state_machine.advance_all_basic(steppers, "abx")
    assert len(steppers) == 0, "Steppers should be empty after input 'abx'"
    accepted = any(stepper.has_reached_accept_state() for stepper in steppers)
    assert not accepted, "Stepper should have rejected input due to pattern mismatch"


def test_stepper_accepts_on_pattern_match_during_parsing() -> None:
    """
    Test that the stepper accepts input when pattern matches during parsing.

    Ensures that the state_machine correctly accepts input that matches the pattern.
    """

    schema: dict = {"pattern": r"abc.*"}
    state_machine = StringSchemaStateMachine(schema=schema)
    steppers = state_machine.get_steppers()
    input_string: str = '"abcd"'

    for idx, char in enumerate(input_string):
        steppers = state_machine.advance_all_basic(steppers, char)
        assert len(steppers) > 0, (
            f"Steppers should not be empty after input '{input_string[: idx + 1]}'"
        )

    accepted: bool = any(stepper.has_reached_accept_state() for stepper in steppers)
    assert accepted, "Stepper should accept input as it matches pattern during parsing."


def test_stepper_start_transition_min_length_met() -> None:
    """
    Test that the state_machine accepts a string when `min_length` is met.

    Verifies that the state_machine correctly accepts strings that meet the minimum length requirement.
    """

    schema: dict = {"minLength": 3}
    state_machine = StringSchemaStateMachine(schema=schema)
    input_string: str = '"abc"'  # length 3

    steppers = state_machine.get_steppers()
    advanced_steppers = state_machine.advance_all_basic(steppers, input_string)

    accepted: bool = any(
        stepper.has_reached_accept_state() for stepper in advanced_steppers
    )
    assert accepted, "Stepper should accept input when min_length is met."


def test_stepper_start_transition_min_length_not_met() -> None:
    """
    Test that the state_machine rejects a string when `min_length` is not met.

    Ensures that the state_machine correctly rejects strings shorter than the minimum length.
    """

    schema: dict = {"minLength": 4}
    state_machine = StringSchemaStateMachine(schema=schema)
    input_string: str = '"abc"'  # length 3, less than minLength 4

    steppers = state_machine.get_steppers()
    advanced_steppers = state_machine.advance_all_basic(steppers, input_string)

    accepted: bool = any(
        stepper.has_reached_accept_state() for stepper in advanced_steppers
    )
    assert not accepted, "Stepper should reject input when min_length is not met."


def test_stepper_start_transition_max_length_met() -> None:
    """
    Test that the state_machine accepts a string when `max_length` is met.

    Checks that the state_machine correctly accepts strings that meet the maximum length requirement.
    """

    schema: dict = {"maxLength": 5}
    state_machine = StringSchemaStateMachine(schema=schema)
    input_string: str = '"abcde"'  # length 5

    steppers = state_machine.get_steppers()
    advanced_steppers = state_machine.advance_all_basic(steppers, input_string)

    accepted: bool = any(
        stepper.has_reached_accept_state() for stepper in advanced_steppers
    )
    assert accepted, "Stepper should accept input when max_length is met."


def test_stepper_start_transition_max_length_exceeded():
    """
    Test that the state_machine rejects a string when max_length is exceeded.
    """
    schema = {"maxLength": 5}
    state_machine = StringSchemaStateMachine(schema=schema)
    input_string = '"abcdef"'  # length 6, exceeds maxLength

    steppers = state_machine.get_steppers()
    advanced_steppers = state_machine.advance_all_basic(steppers, input_string)

    accepted = any(stepper.has_reached_accept_state() for stepper in advanced_steppers)
    assert not accepted, "Stepper should reject input when max_length is exceeded."


def test_min_length_not_met_after_consuming_input() -> None:
    """
    Test that the state_machine rejects a string when `min_length` is not met after consuming input.
    """
    schema: dict = {"minLength": 1}
    state_machine = StringSchemaStateMachine(schema=schema)
    steppers = state_machine.get_steppers()
    steppers = state_machine.advance_all_basic(steppers, '"')
    steppers = state_machine.advance_all(steppers, '" // this comment is illegal')
    assert len(steppers) == 0, "Steppers should be empty"


def test_clean_value_empty_string():
    """Test clean_value with an empty string."""
    schema = {}
    state_machine = StringSchemaStateMachine(schema=schema)
    stepper = state_machine.get_new_stepper()

    # Empty JSON string
    assert stepper.clean_value('""') == "", 'JSON string "" should be cleaned to empty string'

    # Regular empty string
    assert stepper.clean_value("") == "", "Empty string should remain empty"


def test_clean_value_quoted_string():
    """Test clean_value with a quoted string."""
    schema = {}
    state_machine = StringSchemaStateMachine(schema=schema)
    stepper = state_machine.get_new_stepper()

    # Fully quoted string
    assert stepper.clean_value('"hello"') == "hello", '"hello" should be cleaned to hello'

    # Partially quoted string (only opening quote)
    assert stepper.clean_value('"hello') == "hello", '"hello should remove the opening quote'

    # String with internal quotes
    assert stepper.clean_value('"hello"world"') == 'hello', '"hello"world" should be cleaned to hello'


def test_string_stepper_consume_with_remaining_input():
    """
    Test consume method handling of remaining_input when token is partially consumed.
    This tests line 131 in json_string.py.
    """
    schema = {"pattern": "^hello"}
    state_machine = StringSchemaStateMachine(schema=schema)

    # Set up a stepper in the middle of string parsing
    steppers = state_machine.get_steppers()
    steppers = state_machine.advance_all_basic(steppers, '"h')
    steppers = state_machine.advance_all_basic(steppers, 'e')
    assert len(steppers) == 3, "Should have 1 stepper after parsing 'he'"

    # Now test consume with a token that has a valid prefix but an invalid suffix
    new_steppers = state_machine.advance_all_basic(steppers, 'llo world"')
    # Verify the stepper consumed 'llo' and set remaining_input to ' world'
    assert any(new_stepper.has_reached_accept_state() for new_stepper in new_steppers)


def test_validate_value_empty_string():
    """
    Test validate_value rejects empty strings.
    This specifically tests line 192 in json_string.py.
    """
    schema = {"minLength": 1}
    state_machine = StringSchemaStateMachine(schema=schema)
    stepper = state_machine.get_new_stepper()

    # Empty string should be rejected
    assert not stepper.validate_value(""), "Empty string should be rejected with minLength=1"

    # Non-empty string should be accepted
    assert stepper.validate_value("a"), "Non-empty string should be accepted with minLength=1"


def test_invalid_format_validator():
    """
    Test that calling validate_value with an invalid format raises an error.
    This tests line 207 in json_string.py.
    """

    # Trying to validate should raise a ValueError
    with pytest.raises(ValueError) as excinfo:
        schema = {"format": "custom-format"}
        state_machine = StringSchemaStateMachine(schema=schema)
        stepper = state_machine.get_new_stepper()
        stepper.validate_value("test")

    # Check the error message
    assert excinfo is not None, "Should raise error about missing validator"


def test_get_valid_prefix_no_pattern():
    """Test get_valid_prefix when there's no pattern in the schema."""
    schema = {}
    state_machine = StringSchemaStateMachine(schema=schema)
    stepper = state_machine.get_new_stepper()

    # Without pattern, should return the entire string
    assert stepper.get_valid_prefix("test") == "test", "Should return entire string when no pattern"


def test_unsupported_format():
    """Test that an unsupported format raises a ValueError."""
    schema = {"format": "unsupported-format"}

    try:
        StringSchemaStateMachine(schema=schema)
        raise ValueError("Should have raised ValueError for unsupported format")
    except ValueError as e:
        assert "not supported" in str(e), "Error message should mention that format is not supported"


def test_invalid_pattern():
    """Test that an invalid pattern raises a ValueError."""
    schema = {"pattern": "("}  # Invalid regex pattern

    try:
        StringSchemaStateMachine(schema=schema)
        raise ValueError("Should have raised ValueError for invalid pattern")
    except ValueError as e:
        assert "Invalid pattern" in str(e), "Error message should mention invalid pattern"
