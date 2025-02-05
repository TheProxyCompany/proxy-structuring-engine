import pytest

from pse.types.string import StringStateMachine


def test_basic() -> None:
    string_acceptor = StringStateMachine()
    steppers = list(string_acceptor.get_steppers())
    assert len(steppers) == 1
    steppers = string_acceptor.advance_all_basic(steppers, '"')
    assert len(steppers) == 3
    steppers = string_acceptor.advance_all_basic(steppers, "Hello")
    assert len(steppers) == 3
    assert any(
        stepper.sub_stepper and stepper.sub_stepper.has_reached_accept_state()
        for stepper in steppers
    )
    steppers = string_acceptor.advance_all_basic(steppers, '"')
    assert len(steppers) == 1
    assert steppers[0].has_reached_accept_state()
    assert steppers[0].get_current_value() == "Hello"


@pytest.mark.parametrize(
    "input_string, expected_value",
    [
        ('"hello"', "hello"),
        ('"He said, \\"hello\\""', 'He said, "hello"'),
        ('"unicode:\\u1234"', "unicode:\u1234"),
        ('"Backslash: \\\\"', "Backslash: \\"),
        ('"Escapes: \\\\ \\/ \\b \\f \\n \\r \\t"', "Escapes: \\ / \b \f \n \r \t"),
        ('"Smile: \\uD83D\\uDE00"', "Smile: \U0001f600"),
        ('""', ""),
        ('"Tab\\tcharacter"', "Tab\tcharacter"),
        ('"Escaped solidus: \\/"', "Escaped solidus: /"),
    ],
)
def test_valid_strings(input_string: str, expected_value: str) -> None:
    """
    Test StringAcceptor with various valid JSON strings.

    Args:
        string_acceptor (StringAcceptor): The StringAcceptor instance.
        input_string (str): The JSON string input.
        expected_value (str): The expected parsed string value.
    """
    string_acceptor = StringStateMachine()
    steppers = list(string_acceptor.get_steppers())
    for ch in input_string:
        steppers = string_acceptor.advance_all_basic(steppers, ch)

    accepted_steppers = [
        stepper for stepper in steppers if stepper.has_reached_accept_state()
    ]
    assert accepted_steppers, f"No stepper accepted the input: {input_string}"

    for stepper in accepted_steppers:
        assert stepper.get_current_value() == expected_value, (
            f"Expected '{expected_value}', got '{stepper.get_current_value()}'"
        )


@pytest.mark.parametrize(
    "input_string, error_message",
    [
        (
            '"Invalid escape: \\x"',
            "StringAcceptor did not accept string with invalid escape sequence",
        ),
        ('"incomplete string', "StringAcceptor incorrectly accepted incomplete string"),
        (
            '"Invalid unicode: \\u12G4"',
            "StringAcceptor incorrectly accepted string with invalid unicode escape",
        ),
        (
            '"Invalid \x0b string"',
            "StringAcceptor incorrectly accepted string with unescaped control characters",
        ),
        (
            '"Tab\tcharacter"',
            "StringAcceptor incorrectly accepted string with unescaped tab character",
        ),
        (
            'missing start quote"',
            "StringAcceptor incorrectly accepted string missing starting quote",
        ),
    ],
)
def test_invalid_strings(input_string: str, error_message: str) -> None:
    """
    Test StringAcceptor with various invalid JSON strings.

    Args:
        string_acceptor (StringAcceptor): The StringAcceptor instance.
        input_string (str): The invalid JSON string input.
        error_message (str): The assertion error message.
    """
    string_acceptor = StringStateMachine()
    steppers = list(string_acceptor.get_steppers())
    for ch in input_string:
        steppers = string_acceptor.advance_all_basic(steppers, ch)

    assert not any(stepper.has_reached_accept_state() for stepper in steppers), (
        error_message
    )


def test_empty_string() -> None:
    """
    Test StringAcceptor with an empty JSON string.

    Args:
        string_acceptor (StringAcceptor): The StringAcceptor instance.
    """
    string_acceptor = StringStateMachine()
    input_string: str = '""'
    expected_value: str = ""

    steppers = list(string_acceptor.get_steppers())
    for ch in input_string:
        steppers = string_acceptor.advance_all_basic(steppers, ch)

    accepted_steppers = [
        stepper for stepper in steppers if stepper.has_reached_accept_state()
    ]
    assert accepted_steppers, "StringAcceptor did not accept empty string"

    for stepper in accepted_steppers:
        assert stepper.get_current_value() == expected_value, (
            f"Expected empty string, got '{stepper.get_current_value()}'"
        )


def test_string_with_valid_escaped_tab() -> None:
    """
    Test StringAcceptor with a valid escaped tab character.

    Args:
        string_acceptor (StringAcceptor): The StringAcceptor instance.
    """
    string_acceptor = StringStateMachine()
    input_string: str = '"Tab\\tcharacter"'
    expected_value: str = "Tab\tcharacter"

    steppers = list(string_acceptor.get_steppers())
    for ch in input_string:
        steppers = string_acceptor.advance_all_basic(steppers, ch)

    accepted_steppers = [
        stepper for stepper in steppers if stepper.has_reached_accept_state()
    ]
    assert accepted_steppers, (
        "StringAcceptor did not accept string with escaped tab character"
    )

    for stepper in accepted_steppers:
        assert stepper.get_current_value() == expected_value, (
            f"Expected '{expected_value}', got '{stepper.get_current_value()}'"
        )


def test_string_with_escaped_solidus() -> None:
    """
    Test StringAcceptor with a string containing an escaped solidus.

    Args:
        string_acceptor (StringAcceptor): The StringAcceptor instance.
    """
    string_acceptor = StringStateMachine()
    input_string: str = '"Escaped solidus: \\/"'
    expected_value: str = "Escaped solidus: /"

    steppers = list(string_acceptor.get_steppers())
    for ch in input_string:
        steppers = string_acceptor.advance_all_basic(steppers, ch)

    accepted_steppers = [
        stepper for stepper in steppers if stepper.has_reached_accept_state()
    ]
    assert accepted_steppers, (
        "StringAcceptor did not accept string with escaped solidus"
    )

    for stepper in accepted_steppers:
        assert stepper.get_current_value() == expected_value, (
            f"Expected '{expected_value}', got '{stepper.get_current_value()}'"
        )


def test_string_with_unescaped_control_characters() -> None:
    """
    Test StringAcceptor with unescaped control characters (should fail).

    Args:
        string_acceptor (StringAcceptor): The StringAcceptor instance.
    """
    string_acceptor = StringStateMachine()
    input_string: str = '"Invalid \x0b string"'  # Vertical tab, should be escaped

    steppers = list(string_acceptor.get_steppers())
    for ch in input_string:
        steppers = string_acceptor.advance_all_basic(steppers, ch)

    assert not any(stepper.has_reached_accept_state() for stepper in steppers), (
        "StringAcceptor incorrectly accepted string with unescaped control characters"
    )


def test_string_with_invalid_unicode_escape() -> None:
    """
    Test StringAcceptor with invalid unicode escape sequence.

    Args:
        string_acceptor (StringAcceptor): The StringAcceptor instance.
    """
    string_acceptor = StringStateMachine()
    input_string: str = '"Invalid unicode: \\u12G4"'  # 'G' is not a hex digit

    steppers = list(string_acceptor.get_steppers())
    for ch in input_string:
        steppers = string_acceptor.advance_all_basic(steppers, ch)

    assert not any(stepper.has_reached_accept_state() for stepper in steppers), (
        "StringAcceptor incorrectly accepted string with invalid unicode escape"
    )


def test_string_with_incomplete_unicode_escape() -> None:
    """
    Test StringAcceptor with an incomplete unicode escape sequence.

    Args:
        string_acceptor (StringAcceptor): The StringAcceptor instance.
    """
    string_acceptor = StringStateMachine()
    input_string: str = '"Incomplete unicode: \\u123"'  # Missing one hex digit

    steppers = list(string_acceptor.get_steppers())
    for ch in input_string:
        steppers = string_acceptor.advance_all_basic(steppers, ch)

    assert not any(stepper.has_reached_accept_state() for stepper in steppers), (
        "StringAcceptor incorrectly accepted string with incomplete unicode escape"
    )


def test_string_missing_start_quote() -> None:
    """
    Test StringAcceptor with a string missing the starting quote.

    Args:
        string_acceptor (StringAcceptor): The StringAcceptor instance.
    """
    string_acceptor = StringStateMachine()
    input_string: str = 'missing start quote"'

    steppers = list(string_acceptor.get_steppers())
    for ch in input_string:
        steppers = string_acceptor.advance_all_basic(steppers, ch)

    assert not any(stepper.has_reached_accept_state() for stepper in steppers), (
        "StringAcceptor incorrectly accepted string missing starting quote"
    )


def test_incomplete_string() -> None:
    """
    Test StringAcceptor with an incomplete string (missing closing quote).

    Args:
        string_acceptor (StringAcceptor): The StringAcceptor instance.
    """
    string_acceptor = StringStateMachine()
    input_string: str = '"incomplete string'

    steppers = list(string_acceptor.get_steppers())
    for ch in input_string:
        steppers = string_acceptor.advance_all_basic(steppers, ch)

    assert not any(stepper.has_reached_accept_state() for stepper in steppers), (
        "StringAcceptor incorrectly accepted incomplete string"
    )


@pytest.mark.parametrize(
    "input_string, expected_value",
    [
        ('"hello"', "hello"),
        ('"He said, \\"hello\\""', 'He said, "hello"'),
        ('"unicode:\\u1234"', "unicode:\u1234"),
        ('"Backslash: \\\\"', "Backslash: \\"),
        ('"Escapes: \\\\ \\/ \\b \\f \\n \\r \\t"', "Escapes: \\ / \b \f \n \r \t"),
        ('"Smile: \\uD83D\\uDE00"', "Smile: \U0001f600"),
        ('""', ""),
        ('"Tab\\tcharacter"', "Tab\tcharacter"),
        ('"Escaped solidus: \\/"', "Escaped solidus: /"),
    ],
)
def test_valid_strings_char_by_char(input_string: str, expected_value: str) -> None:
    """
    Test StringAcceptor with various valid JSON strings by advancing character by character.

    Args:
        string_acceptor (StringAcceptor): The StringAcceptor instance.
        input_string (str): The JSON string input.
        expected_value (str): The expected parsed string value.
    """
    string_acceptor = StringStateMachine()
    steppers = list(string_acceptor.get_steppers())
    for ch in input_string:
        steppers = string_acceptor.advance_all_basic(steppers, ch)

    accepted_steppers = [
        stepper for stepper in steppers if stepper.has_reached_accept_state()
    ]
    assert accepted_steppers, f"No stepper accepted the input: {input_string}"

    for stepper in accepted_steppers:
        assert stepper.get_current_value() == expected_value, (
            f"Expected '{expected_value}', got '{stepper.get_current_value()}'"
        )


@pytest.mark.parametrize(
    "input_string, error_message",
    [
        (
            '"Invalid escape: \\x"',
            "StringAcceptor did not accept string with invalid escape sequence",
        ),
        ('"incomplete string', "StringAcceptor incorrectly accepted incomplete string"),
        (
            '"Invalid unicode: \\u12G4"',
            "StringAcceptor incorrectly accepted string with invalid unicode escape",
        ),
        (
            '"Invalid \x0b string"',
            "StringAcceptor incorrectly accepted string with unescaped control characters",
        ),
        (
            '"Tab\tcharacter"',
            "StringAcceptor incorrectly accepted string with unescaped tab character",
        ),
        (
            'missing start quote"',
            "StringAcceptor incorrectly accepted string missing starting quote",
        ),
    ],
)
def test_invalid_strings_char_by_char(input_string: str, error_message: str) -> None:
    """
    Test StringAcceptor with various invalid JSON strings by advancing character by character.

    Args:
        string_acceptor (StringAcceptor): The StringAcceptor instance.
        input_string (str): The invalid JSON string input.
        error_message (str): The assertion error message.
    """
    string_acceptor = StringStateMachine()
    steppers = list(string_acceptor.get_steppers())
    for ch in input_string:
        steppers = string_acceptor.advance_all_basic(steppers, ch)

    assert not any(stepper.has_reached_accept_state() for stepper in steppers), (
        error_message
    )


def test_empty_string_char_by_char() -> None:
    """
    Test StringAcceptor with an empty JSON string by advancing character by character.

    Args:
        string_acceptor (StringAcceptor): The StringAcceptor instance.
    """
    input_string: str = '""'
    expected_value: str = ""

    string_acceptor = StringStateMachine()
    steppers = list(string_acceptor.get_steppers())
    for ch in input_string:
        steppers = string_acceptor.advance_all_basic(steppers, ch)

    assert any(stepper.has_reached_accept_state() for stepper in steppers), (
        "StringAcceptor did not accept empty string"
    )
    for stepper in steppers:
        if stepper.has_reached_accept_state():
            assert stepper.get_current_value() == expected_value, (
                f"Expected empty string, got '{stepper.get_current_value()}'"
            )


def test_string_with_valid_escaped_tab_char_by_char() -> None:
    """
    Test StringAcceptor with a valid escaped tab character by advancing character by character.

    Args:
        string_acceptor (StringAcceptor): The StringAcceptor instance.
    """
    string_acceptor = StringStateMachine()
    input_string: str = '"Tab\\tcharacter"'
    expected_value: str = "Tab\tcharacter"

    steppers = string_acceptor.get_steppers()
    for ch in input_string:
        steppers = string_acceptor.advance_all_basic(steppers, ch)

    assert any(stepper.has_reached_accept_state() for stepper in steppers)
    for stepper in steppers:
        if stepper.has_reached_accept_state():
            assert stepper.get_current_value() == expected_value, (
                f"Expected '{expected_value}', got '{stepper.get_current_value()}'"
            )


def test_string_with_escaped_solidus_char_by_char() -> None:
    """
    Test StringAcceptor with a string containing an escaped solidus by advancing character by character.

    Args:
        string_acceptor (StringAcceptor): The StringAcceptor instance.
    """
    string_acceptor = StringStateMachine()
    input_string: str = '"Escaped solidus: \\/"'
    expected_value: str = "Escaped solidus: /"

    steppers = list(string_acceptor.get_steppers())
    for ch in input_string:
        steppers = string_acceptor.advance_all_basic(steppers, ch)

    assert any(stepper.has_reached_accept_state() for stepper in steppers), (
        "StringAcceptor did not accept string with escaped solidus"
    )
    for stepper in steppers:
        if stepper.has_reached_accept_state():
            assert stepper.get_current_value() == expected_value, (
                f"Expected '{expected_value}', got '{stepper.get_current_value()}'"
            )


def test_edge_case():
    string_acceptor = StringStateMachine()
    steppers = string_acceptor.get_steppers()
    assert len(steppers) == 1
    steppers = string_acceptor.advance_all_basic(steppers, '"')
    assert len(steppers) == 3
    steppers = string_acceptor.advance_all_basic(steppers, "Hello There!")
    assert len(steppers) == 3
    steppers = string_acceptor.advance_all_basic(steppers, '"')
    assert len(steppers) == 1
    assert steppers[0].has_reached_accept_state()
    assert steppers[0].get_current_value() == "Hello There!"
