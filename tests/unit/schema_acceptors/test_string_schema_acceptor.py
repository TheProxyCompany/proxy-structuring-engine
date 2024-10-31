from pse.schema_acceptors.string_schema_acceptor import StringSchemaAcceptor


def test_init_default_schema():
    """
    Test initializing StringSchemaAcceptor with default schema.
    Ensures default min_length and max_length are set correctly.
    """
    acceptor = StringSchemaAcceptor(schema={})
    assert acceptor.min_length() == 0, "Default min_length should be 0."
    assert acceptor.max_length() == 10000, "Default max_length should be 10000."


def test_init_custom_min_length():
    """
    Test initializing StringSchemaAcceptor with a custom min_length.
    """
    schema = {"minLength": 5}
    acceptor = StringSchemaAcceptor(schema=schema)
    assert acceptor.min_length() == 5, "Custom min_length should be set to 5."
    assert (
        acceptor.max_length() == 10000
    ), "Default max_length should remain 10000 when only min_length is set."


def test_init_custom_max_length():
    """
    Test initializing StringSchemaAcceptor with a custom max_length.
    """
    schema = {"maxLength": 50}
    acceptor = StringSchemaAcceptor(schema=schema)
    assert (
        acceptor.min_length() == 0
    ), "Default min_length should remain 0 when only max_length is set."
    assert acceptor.max_length() == 50, "Custom max_length should be set to 50."


def test_validate_value_success():
    """
    Test validate_value with a string that meets min_length and max_length requirements.
    """
    schema = {"minLength": 3, "maxLength": 10}
    acceptor = StringSchemaAcceptor(schema=schema)
    assert acceptor.validate_value(
        "test"
    ), "String 'test' should be valid as it meets min and max length requirements."


def test_validate_value_too_short():
    """
    Test validate_value with a string shorter than min_length.
    """
    schema = {"minLength": 5}
    acceptor = StringSchemaAcceptor(schema=schema)
    assert not acceptor.validate_value(
        "hey"
    ), "String 'hey' should be invalid as it is shorter than min_length."


def test_validate_value_too_long():
    """
    Test validate_value with a string longer than max_length.
    """
    schema = {"maxLength": 5}
    acceptor = StringSchemaAcceptor(schema=schema)
    assert not acceptor.validate_value(
        "exceeds"
    ), "String 'exceeds' should be invalid as it exceeds max_length."


def test_validate_value_pattern_match():
    """
    Test validate_value with a string that matches the pattern.
    """
    schema = {"pattern": r"^\d{3}-\d{2}-\d{4}$"}
    acceptor = StringSchemaAcceptor(schema=schema)
    assert acceptor.validate_value(
        "123-45-6789"
    ), "String '123-45-6789' should match the pattern."


def test_validate_value_pattern_no_match():
    """
    Test validate_value with a string that does not match the pattern.
    """
    schema = {"pattern": r"^\d{3}-\d{2}-\d{4}$"}
    acceptor = StringSchemaAcceptor(schema=schema)
    assert not acceptor.validate_value(
        "123456789"
    ), "String '123456789' should not match the pattern."


def test_validate_value_format_email_valid():
    """
    Test validate_value with a valid email format.
    """
    schema = {"format": "email"}
    acceptor = StringSchemaAcceptor(schema=schema)
    assert acceptor.validate_value(
        "test@example.com"
    ), "Email 'test@example.com' should be valid."


def test_validate_value_format_email_invalid():
    """
    Test validate_value with an invalid email format.
    """
    schema = {"format": "email"}
    acceptor = StringSchemaAcceptor(schema=schema)
    assert not acceptor.validate_value(
        "test@com"
    ), "Email 'test@com' should be invalid."


def test_validate_value_format_date_time_valid():
    """
    Test validate_value with a valid date-time format.
    """
    schema = {"format": "date-time"}
    acceptor = StringSchemaAcceptor(schema=schema)
    assert acceptor.validate_value(
        "2023-10-20T12:34:56"
    ), "Date-time '2023-10-20T12:34:56' should be valid."


def test_validate_value_format_date_time_invalid():
    """
    Test validate_value with an invalid date-time format.
    """
    schema = {"format": "date-time"}
    acceptor = StringSchemaAcceptor(schema=schema)
    assert not acceptor.validate_value(
        "20-10-2023 12:34:56"
    ), "Date-time '20-10-2023 12:34:56' should be invalid."


def test_validate_value_format_uri_valid():
    """
    Test validate_value with a valid URI format.
    """
    schema = {"format": "uri"}
    acceptor = StringSchemaAcceptor(schema=schema)
    assert acceptor.validate_value(
        "https://www.example.com"
    ), "URI 'https://www.example.com' should be valid."


def test_validate_value_format_uri_invalid():
    """
    Test validate_value with an invalid URI format.
    """
    schema = {"format": "uri"}
    acceptor = StringSchemaAcceptor(schema=schema)
    assert not acceptor.validate_value(
        "www.example.com"
    ), "URI 'www.example.com' should be invalid."


def test_walker_rejects_on_pattern_mismatch_during_parsing():
    """
    Test that the walker rejects input early if pattern does not match during parsing.
    """
    schema = {"pattern": r"abc.*"}
    acceptor = StringSchemaAcceptor(schema=schema)
    walkers = acceptor.get_walkers()
    input_string = '"abx'

    for idx, char in enumerate(input_string):
        walkers = list(acceptor.advance_all(walkers, char))
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


def test_walker_accepts_on_pattern_match_during_parsing():
    """
    Test that the walker accepts input when pattern matches during parsing.
    """
    schema = {"pattern": r"abc.*"}
    acceptor = StringSchemaAcceptor(schema=schema)
    walkers = acceptor.get_walkers()
    input_string = '"abcd"'

    for idx, char in enumerate(input_string):
        walkers = list(acceptor.advance_all(walkers, char))
        assert (
            len(walkers) > 0
        ), f"Walkers should not be empty after input '{input_string[:idx+1]}'"

    accepted = any(walker.has_reached_accept_state() for walker in walkers)
    assert accepted, "Walker should accept input as it matches pattern during parsing"


def test_walker_start_transition_min_length_met():
    """
    Test that the acceptor accepts a string when min_length is met.
    """
    schema = {"minLength": 3}
    acceptor = StringSchemaAcceptor(schema=schema)
    input_string = '"abc"'  # length 3

    walkers = acceptor.get_walkers()
    walkers = list(acceptor.advance_all(walkers, input_string))

    accepted = any(walker.has_reached_accept_state() for walker in walkers)
    assert accepted, "Walker should accept input when min_length is met."


def test_walker_start_transition_min_length_not_met():
    """
    Test that the acceptor rejects a string when min_length is not met.
    """
    schema = {"minLength": 4}
    acceptor = StringSchemaAcceptor(schema=schema)
    input_string = '"abc"'  # length 3, less than minLength 4

    walkers = acceptor.get_walkers()
    walkers = list(acceptor.advance_all(walkers, input_string))

    accepted = any(walker.has_reached_accept_state() for walker in walkers)
    assert not accepted, "Walker should reject input when min_length is not met."


def test_walker_start_transition_max_length_met():
    """
    Test that the acceptor accepts a string when max_length is met.
    """
    schema = {"maxLength": 5}
    acceptor = StringSchemaAcceptor(schema=schema)
    input_string = '"abcde"'  # length 5

    walkers = acceptor.get_walkers()
    walkers = list(acceptor.advance_all(walkers, input_string))

    accepted = any(walker.has_reached_accept_state() for walker in walkers)
    assert accepted, "Walker should accept input when max_length is met."


def test_walker_start_transition_max_length_exceeded():
    """
    Test that the acceptor rejects a string when max_length is exceeded.
    """
    schema = {"maxLength": 5}
    acceptor = StringSchemaAcceptor(schema=schema)
    input_string = '"abcdef"'  # length 6, exceeds maxLength

    walkers = acceptor.get_walkers()
    walkers = list(acceptor.advance_all(walkers, input_string))

    accepted = any(walker.has_reached_accept_state() for walker in walkers)
    assert not accepted, "Walker should reject input when max_length is exceeded."
