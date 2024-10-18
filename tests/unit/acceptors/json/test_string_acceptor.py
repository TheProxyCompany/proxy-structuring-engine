import pytest
from pse.acceptors.json.string_acceptor import StringAcceptor

@pytest.fixture
def string_acceptor() -> StringAcceptor:
    """Fixture to create a new instance of StringAcceptor for each test."""
    return StringAcceptor()


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
def test_valid_strings(string_acceptor: StringAcceptor, input_string: str, expected_value: str) -> None:
    """
    Test StringAcceptor with various valid JSON strings.

    Args:
        string_acceptor (StringAcceptor): The StringAcceptor instance.
        input_string (str): The JSON string input.
        expected_value (str): The expected parsed string value.
    """
    cursors = list(string_acceptor.get_cursors())
    for ch in input_string:
        cursors = list(string_acceptor.advance_all(cursors, ch))

    accepted_cursors = [cursor for cursor in cursors if cursor.in_accepted_state()]
    assert accepted_cursors, f"No cursor accepted the input: {input_string}"

    for cursor in accepted_cursors:
        assert cursor.get_value() == expected_value, f"Expected '{expected_value}', got '{cursor.get_value()}'"


@pytest.mark.parametrize(
    "input_string, error_message",
    [
        ('"Invalid escape: \\x"', "StringAcceptor did not accept string with invalid escape sequence"),
        ('"incomplete string', "StringAcceptor incorrectly accepted incomplete string"),
        ('"Invalid unicode: \\u12G4"', "StringAcceptor incorrectly accepted string with invalid unicode escape"),
        ('"Invalid \x0b string"', "StringAcceptor incorrectly accepted string with unescaped control characters"),
        ('"Tab\tcharacter"', "StringAcceptor incorrectly accepted string with unescaped tab character"),
        ('missing start quote"', "StringAcceptor incorrectly accepted string missing starting quote"),
    ],
)
def test_invalid_strings(string_acceptor: StringAcceptor, input_string: str, error_message: str) -> None:
    """
    Test StringAcceptor with various invalid JSON strings.

    Args:
        string_acceptor (StringAcceptor): The StringAcceptor instance.
        input_string (str): The invalid JSON string input.
        error_message (str): The assertion error message.
    """
    cursors = list(string_acceptor.get_cursors())
    for ch in input_string:
        cursors = list(string_acceptor.advance_all(cursors, ch))

    assert not any(cursor.in_accepted_state() for cursor in cursors), error_message


def test_empty_string(string_acceptor: StringAcceptor) -> None:
    """
    Test StringAcceptor with an empty JSON string.

    Args:
        string_acceptor (StringAcceptor): The StringAcceptor instance.
    """
    input_string: str = '""'
    expected_value: str = ""

    cursors = list(string_acceptor.get_cursors())
    for ch in input_string:
        cursors = list(string_acceptor.advance_all(cursors, ch))

    accepted_cursors = [cursor for cursor in cursors if cursor.in_accepted_state()]
    assert accepted_cursors, "StringAcceptor did not accept empty string"

    for cursor in accepted_cursors:
        assert cursor.get_value() == expected_value, f"Expected empty string, got '{cursor.get_value()}'"


def test_string_with_valid_escaped_tab(string_acceptor: StringAcceptor) -> None:
    """
    Test StringAcceptor with a valid escaped tab character.

    Args:
        string_acceptor (StringAcceptor): The StringAcceptor instance.
    """
    input_string: str = '"Tab\\tcharacter"'
    expected_value: str = "Tab\tcharacter"

    cursors = list(string_acceptor.get_cursors())
    for ch in input_string:
        cursors = list(string_acceptor.advance_all(cursors, ch))

    accepted_cursors = [cursor for cursor in cursors if cursor.in_accepted_state()]
    assert accepted_cursors, "StringAcceptor did not accept string with escaped tab character"

    for cursor in accepted_cursors:
        assert cursor.get_value() == expected_value, f"Expected '{expected_value}', got '{cursor.get_value()}'"


def test_string_with_escaped_solidus(string_acceptor: StringAcceptor) -> None:
    """
    Test StringAcceptor with a string containing an escaped solidus.

    Args:
        string_acceptor (StringAcceptor): The StringAcceptor instance.
    """
    input_string: str = '"Escaped solidus: \\/"'
    expected_value: str = "Escaped solidus: /"

    cursors = list(string_acceptor.get_cursors())
    for ch in input_string:
        cursors = list(string_acceptor.advance_all(cursors, ch))

    accepted_cursors = [cursor for cursor in cursors if cursor.in_accepted_state()]
    assert accepted_cursors, "StringAcceptor did not accept string with escaped solidus"

    for cursor in accepted_cursors:
        assert cursor.get_value() == expected_value, f"Expected '{expected_value}', got '{cursor.get_value()}'"


def test_string_with_unescaped_control_characters(string_acceptor: StringAcceptor) -> None:
    """
    Test StringAcceptor with unescaped control characters (should fail).

    Args:
        string_acceptor (StringAcceptor): The StringAcceptor instance.
    """
    input_string: str = '"Invalid \x0b string"'  # Vertical tab, should be escaped

    cursors = list(string_acceptor.get_cursors())
    for ch in input_string:
        cursors = list(string_acceptor.advance_all(cursors, ch))

    assert not any(cursor.in_accepted_state() for cursor in cursors), \
        "StringAcceptor incorrectly accepted string with unescaped control characters"


def test_string_with_invalid_unicode_escape(string_acceptor: StringAcceptor) -> None:
    """
    Test StringAcceptor with invalid unicode escape sequence.

    Args:
        string_acceptor (StringAcceptor): The StringAcceptor instance.
    """
    input_string: str = '"Invalid unicode: \\u12G4"'  # 'G' is not a hex digit

    cursors = list(string_acceptor.get_cursors())
    for ch in input_string:
        cursors = list(string_acceptor.advance_all(cursors, ch))

    assert not any(cursor.in_accepted_state() for cursor in cursors), \
        "StringAcceptor incorrectly accepted string with invalid unicode escape"


def test_string_with_incomplete_unicode_escape(string_acceptor: StringAcceptor) -> None:
    """
    Test StringAcceptor with an incomplete unicode escape sequence.

    Args:
        string_acceptor (StringAcceptor): The StringAcceptor instance.
    """
    input_string: str = '"Incomplete unicode: \\u123"'  # Missing one hex digit

    cursors = list(string_acceptor.get_cursors())
    for ch in input_string:
        cursors = list(string_acceptor.advance_all(cursors, ch))

    assert not any(cursor.in_accepted_state() for cursor in cursors), \
        "StringAcceptor incorrectly accepted string with incomplete unicode escape"


def test_string_missing_start_quote(string_acceptor: StringAcceptor) -> None:
    """
    Test StringAcceptor with a string missing the starting quote.

    Args:
        string_acceptor (StringAcceptor): The StringAcceptor instance.
    """
    input_string: str = 'missing start quote"'

    cursors = list(string_acceptor.get_cursors())
    for ch in input_string:
        cursors = list(string_acceptor.advance_all(cursors, ch))

    assert not any(cursor.in_accepted_state() for cursor in cursors), \
        "StringAcceptor incorrectly accepted string missing starting quote"


def test_incomplete_string(string_acceptor: StringAcceptor) -> None:
    """
    Test StringAcceptor with an incomplete string (missing closing quote).

    Args:
        string_acceptor (StringAcceptor): The StringAcceptor instance.
    """
    input_string: str = '"incomplete string'

    cursors = list(string_acceptor.get_cursors())
    for ch in input_string:
        cursors = list(string_acceptor.advance_all(cursors, ch))

    assert not any(cursor.in_accepted_state() for cursor in cursors), \
        "StringAcceptor incorrectly accepted incomplete string"


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
def test_valid_strings_char_by_char(string_acceptor: StringAcceptor, input_string: str, expected_value: str) -> None:
    """
    Test StringAcceptor with various valid JSON strings by advancing character by character.

    Args:
        string_acceptor (StringAcceptor): The StringAcceptor instance.
        input_string (str): The JSON string input.
        expected_value (str): The expected parsed string value.
    """
    cursors = list(string_acceptor.get_cursors())
    for ch in input_string:
        new_cursors = []
        for cursor in cursors:
            new_cursors.extend(string_acceptor.advance_all([cursor], ch))
        cursors = new_cursors

    accepted_cursors = [cursor for cursor in cursors if cursor.in_accepted_state()]
    assert accepted_cursors, f"No cursor accepted the input: {input_string}"

    for cursor in accepted_cursors:
        assert cursor.get_value() == expected_value, f"Expected '{expected_value}', got '{cursor.get_value()}'"


@pytest.mark.parametrize(
    "input_string, error_message",
    [
        ('"Invalid escape: \\x"', "StringAcceptor did not accept string with invalid escape sequence"),
        ('"incomplete string', "StringAcceptor incorrectly accepted incomplete string"),
        ('"Invalid unicode: \\u12G4"', "StringAcceptor incorrectly accepted string with invalid unicode escape"),
        ('"Invalid \x0b string"', "StringAcceptor incorrectly accepted string with unescaped control characters"),
        ('"Tab\tcharacter"', "StringAcceptor incorrectly accepted string with unescaped tab character"),
        ('missing start quote"', "StringAcceptor incorrectly accepted string missing starting quote"),
    ],
)
def test_invalid_strings_char_by_char(string_acceptor: StringAcceptor, input_string: str, error_message: str) -> None:
    """
    Test StringAcceptor with various invalid JSON strings by advancing character by character.

    Args:
        string_acceptor (StringAcceptor): The StringAcceptor instance.
        input_string (str): The invalid JSON string input.
        error_message (str): The assertion error message.
    """
    cursors = list(string_acceptor.get_cursors())
    for ch in input_string:
        new_cursors = []
        for cursor in cursors:
            new_cursors.extend(string_acceptor.advance_all([cursor], ch))
        cursors = new_cursors

    assert not any(cursor.in_accepted_state() for cursor in cursors), error_message


def test_empty_string_char_by_char(string_acceptor: StringAcceptor) -> None:
    """
    Test StringAcceptor with an empty JSON string by advancing character by character.

    Args:
        string_acceptor (StringAcceptor): The StringAcceptor instance.
    """
    input_string: str = '""'
    expected_value: str = ""

    cursors = list(string_acceptor.get_cursors())
    for ch in input_string:
        new_cursors = []
        for cursor in cursors:
            new_cursors.extend(string_acceptor.advance_all([cursor], ch))
        cursors = new_cursors

    accepted_cursors = [cursor for cursor in cursors if cursor.in_accepted_state()]
    assert accepted_cursors, "StringAcceptor did not accept empty string"

    for cursor in accepted_cursors:
        assert cursor.get_value() == expected_value, f"Expected empty string, got '{cursor.get_value()}'"


def test_string_with_valid_escaped_tab_char_by_char(string_acceptor: StringAcceptor) -> None:
    """
    Test StringAcceptor with a valid escaped tab character by advancing character by character.

    Args:
        string_acceptor (StringAcceptor): The StringAcceptor instance.
    """
    input_string: str = '"Tab\\tcharacter"'
    expected_value: str = "Tab\tcharacter"

    cursors = list(string_acceptor.get_cursors())
    for ch in input_string:
        new_cursors = []
        for cursor in cursors:
            new_cursors.extend(string_acceptor.advance_all([cursor], ch))
        cursors = new_cursors

    accepted_cursors = [cursor for cursor in cursors if cursor.in_accepted_state()]
    assert accepted_cursors, "StringAcceptor did not accept string with escaped tab character"

    for cursor in accepted_cursors:
        assert cursor.get_value() == expected_value, f"Expected '{expected_value}', got '{cursor.get_value()}'"


def test_string_with_escaped_solidus_char_by_char(string_acceptor: StringAcceptor) -> None:
    """
    Test StringAcceptor with a string containing an escaped solidus by advancing character by character.

    Args:
        string_acceptor (StringAcceptor): The StringAcceptor instance.
    """
    input_string: str = '"Escaped solidus: \\/"'
    expected_value: str = "Escaped solidus: /"

    cursors = list(string_acceptor.get_cursors())
    for ch in input_string:
        new_cursors = []
        for cursor in cursors:
            new_cursors.extend(string_acceptor.advance_all([cursor], ch))
        cursors = new_cursors

    accepted_cursors = [cursor for cursor in cursors if cursor.in_accepted_state()]
    assert accepted_cursors, "StringAcceptor did not accept string with escaped solidus"

    for cursor in accepted_cursors:
        assert cursor.get_value() == expected_value, f"Expected '{expected_value}', got '{cursor.get_value()}'"


def test_complete_transition_invalid_string(string_acceptor: StringAcceptor) -> None:
    """
    Test StringAcceptor with invalid string.
    """
    cursor = string_acceptor.Cursor(acceptor=string_acceptor)
    cursor.complete_transition("\"hi", "$", True)
    assert cursor.value is None
    assert cursor.text == "\"hi"
    assert cursor.get_value() == "\"hi"
