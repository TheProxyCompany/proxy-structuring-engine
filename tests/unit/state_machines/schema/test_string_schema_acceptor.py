from pse.state_machines.schema.string_schema_acceptor import StringSchemaAcceptor


def test_init_default_schema():
    """
    Test initializing StringSchemaAcceptor with default schema.
    Ensures default min_length and max_length are set correctly.
    """
    state_machine = StringSchemaAcceptor(schema={})
    assert state_machine.min_length() == 0, "Default min_length should be 0."
    assert state_machine.max_length() == 10000, "Default max_length should be 10000."


def test_init_custom_min_length():
    """
    Test initializing StringSchemaAcceptor with a custom min_length.
    """
    schema = {"minLength": 5}
    state_machine = StringSchemaAcceptor(schema=schema)
    assert state_machine.min_length() == 5, "Custom min_length should be set to 5."
    assert (
        state_machine.max_length() == 10000
    ), "Default max_length should remain 10000 when only min_length is set."


def test_init_custom_max_length():
    """
    Test initializing StringSchemaAcceptor with a custom max_length.
    """
    schema = {"maxLength": 50}
    state_machine = StringSchemaAcceptor(schema=schema)
    assert (
        state_machine.min_length() == 0
    ), "Default min_length should remain 0 when only max_length is set."
    assert state_machine.max_length() == 50, "Custom max_length should be set to 50."


def test_validate_value_success():
    """
    Test validate_value with a string that meets min_length and max_length requirements.
    """
    schema = {"minLength": 3, "maxLength": 10}
    state_machine = StringSchemaAcceptor(schema=schema)
    assert state_machine.validate_value(
        "test"
    ), "String 'test' should be valid as it meets min and max length requirements."


def test_validate_value_too_short():
    """
    Test validate_value with a string shorter than min_length.
    """
    schema = {"minLength": 5}
    state_machine = StringSchemaAcceptor(schema=schema)
    assert not state_machine.validate_value(
        "hey"
    ), "String 'hey' should be invalid as it is shorter than min_length."


def test_validate_value_too_long():
    """
    Test validate_value with a string longer than max_length.
    """
    schema = {"maxLength": 5}
    state_machine = StringSchemaAcceptor(schema=schema)
    assert not state_machine.validate_value(
        "exceeds"
    ), "String 'exceeds' should be invalid as it exceeds max_length."


def test_validate_value_pattern_match():
    """
    Test validate_value with a string that matches the pattern.
    """
    schema = {"pattern": r"^\d{3}-\d{2}-\d{4}$"}
    state_machine = StringSchemaAcceptor(schema=schema)
    assert state_machine.validate_value(
        "123-45-6789"
    ), "String '123-45-6789' should match the pattern."


def test_validate_value_pattern_no_match():
    """
    Test validate_value with a string that does not match the pattern.
    """
    schema = {"pattern": r"^\d{3}-\d{2}-\d{4}$"}
    state_machine = StringSchemaAcceptor(schema=schema)
    assert not state_machine.validate_value(
        "123456789"
    ), "String '123456789' should not match the pattern."


def test_validate_value_format_email_valid():
    """
    Test validate_value with a valid email format.
    """
    schema = {"format": "email"}
    state_machine = StringSchemaAcceptor(schema=schema)
    assert state_machine.validate_value(
        "test@example.com"
    ), "Email 'test@example.com' should be valid."


def test_validate_value_format_email_invalid():
    """
    Test validate_value with an invalid email format.
    """
    schema = {"format": "email"}
    state_machine = StringSchemaAcceptor(schema=schema)
    assert not state_machine.validate_value(
        "test@com"
    ), "Email 'test@com' should be invalid."


def test_validate_value_format_date_time_valid():
    """
    Test validate_value with a valid date-time format.
    """
    schema = {"format": "date-time"}
    state_machine = StringSchemaAcceptor(schema=schema)
    assert state_machine.validate_value(
        "2023-10-20T12:34:56"
    ), "Date-time '2023-10-20T12:34:56' should be valid."


def test_validate_value_format_date_time_invalid():
    """
    Test validate_value with an invalid date-time format.
    """
    schema = {"format": "date-time"}
    state_machine = StringSchemaAcceptor(schema=schema)
    assert not state_machine.validate_value(
        "20-10-2023 12:34:56"
    ), "Date-time '20-10-2023 12:34:56' should be invalid."


def test_validate_value_format_uri_valid():
    """
    Test validate_value with a valid URI format.
    """
    schema = {"format": "uri"}
    state_machine = StringSchemaAcceptor(schema=schema)
    assert state_machine.validate_value(
        "https://www.example.com"
    ), "URI 'https://www.example.com' should be valid."


def test_validate_value_format_uri_invalid():
    """
    Test validate_value with an invalid URI format.
    """
    schema = {"format": "uri"}
    state_machine = StringSchemaAcceptor(schema=schema)
    assert not state_machine.validate_value(
        "www.example.com"
    ), "URI 'www.example.com' should be invalid."


def test_walker_rejects_on_pattern_mismatch_during_parsing():
    """
    Test that the walker rejects input early if pattern does not match during parsing.
    """
    schema = {"pattern": r"abc.*"}
    state_machine = StringSchemaAcceptor(schema=schema)
    walkers = state_machine.get_walkers()
    input_string = '"abx'

    for idx, char in enumerate(input_string):
        walkers = [walker for _, walker in state_machine.advance_all(walkers, char)]
        if idx == len(input_string) - 1:
            assert (
                len(walkers) == 0
            ), f"Walkers should be empty after input '{input_string[:idx+1]}' due to pattern mismatch"
        else:
            assert (
                len(walkers) > 0
            ), f"Walkers should not be empty after input '{input_string[:idx+1]}', idx: {idx}"

    accepted = any(walker.has_reached_accept_state() for walker in walkers)
    assert not accepted, "Walker should have rejected input due to pattern mismatch"


def test_walker_accepts_on_pattern_match_during_parsing() -> None:
    """
    Test that the walker accepts input when pattern matches during parsing.

    Ensures that the state_machine correctly accepts input that matches the pattern.
    """

    schema: dict = {"pattern": r"abc.*"}
    state_machine = StringSchemaAcceptor(schema=schema)
    walkers = state_machine.get_walkers()
    input_string: str = '"abcd"'

    for idx, char in enumerate(input_string):
        walkers = [walker for _, walker in state_machine.advance_all(walkers, char)]
        assert (
            len(walkers) > 0
        ), f"Walkers should not be empty after input '{input_string[:idx+1]}'"

    accepted: bool = any(walker.has_reached_accept_state() for walker in walkers)
    assert accepted, "Walker should accept input as it matches pattern during parsing."


def test_walker_start_transition_min_length_met() -> None:
    """
    Test that the state_machine accepts a string when `min_length` is met.

    Verifies that the state_machine correctly accepts strings that meet the minimum length requirement.
    """

    schema: dict = {"minLength": 3}
    state_machine = StringSchemaAcceptor(schema=schema)
    input_string: str = '"abc"'  # length 3

    walkers = state_machine.get_walkers()
    advanced_walkers = [
        walker for _, walker in state_machine.advance_all(walkers, input_string)
    ]

    accepted: bool = any(
        walker.has_reached_accept_state() for walker in advanced_walkers
    )
    assert accepted, "Walker should accept input when min_length is met."


def test_walker_start_transition_min_length_not_met() -> None:
    """
    Test that the state_machine rejects a string when `min_length` is not met.

    Ensures that the state_machine correctly rejects strings shorter than the minimum length.
    """

    schema: dict = {"minLength": 4}
    state_machine = StringSchemaAcceptor(schema=schema)
    input_string: str = '"abc"'  # length 3, less than minLength 4

    walkers = state_machine.get_walkers()
    advanced_walkers = [
        walker for _, walker in state_machine.advance_all(walkers, input_string)
    ]

    accepted: bool = any(
        walker.has_reached_accept_state() for walker in advanced_walkers
    )
    assert not accepted, "Walker should reject input when min_length is not met."


def test_walker_start_transition_max_length_met() -> None:
    """
    Test that the state_machine accepts a string when `max_length` is met.

    Checks that the state_machine correctly accepts strings that meet the maximum length requirement.
    """

    schema: dict = {"maxLength": 5}
    state_machine = StringSchemaAcceptor(schema=schema)
    input_string: str = '"abcde"'  # length 5

    walkers = state_machine.get_walkers()
    advanced_walkers = [
        walker for _, walker in state_machine.advance_all(walkers, input_string)
    ]

    accepted: bool = any(
        walker.has_reached_accept_state() for walker in advanced_walkers
    )
    assert accepted, "Walker should accept input when max_length is met."


def test_walker_start_transition_max_length_exceeded():
    """
    Test that the state_machine rejects a string when max_length is exceeded.
    """
    schema = {"maxLength": 5}
    state_machine = StringSchemaAcceptor(schema=schema)
    input_string = '"abcdef"'  # length 6, exceeds maxLength

    walkers = state_machine.get_walkers()
    advanced_walkers = [
        walker for _, walker in state_machine.advance_all(walkers, input_string)
    ]

    accepted = any(walker.has_reached_accept_state() for walker in advanced_walkers)
    assert not accepted, "Walker should reject input when max_length is exceeded."
