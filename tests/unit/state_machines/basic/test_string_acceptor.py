import pytest

from pse.state_machines.basic.string_acceptor import StringAcceptor


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
def test_valid_strings(
    string_acceptor: StringAcceptor, input_string: str, expected_value: str
) -> None:
    """
    Test StringAcceptor with various valid JSON strings.

    Args:
        string_acceptor (StringAcceptor): The StringAcceptor instance.
        input_string (str): The JSON string input.
        expected_value (str): The expected parsed string value.
    """
    walkers = list(string_acceptor.get_walkers())
    for ch in input_string:
        walkers = [walker for _, walker in string_acceptor.advance_all(walkers, ch)]

    accepted_walkers = [
        walker for walker in walkers if walker.has_reached_accept_state()
    ]
    assert accepted_walkers, f"No walker accepted the input: {input_string}"

    for walker in accepted_walkers:
        assert (
            walker.get_current_value() == expected_value
        ), f"Expected '{expected_value}', got '{walker.get_current_value()}'"


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
def test_invalid_strings(
    string_acceptor: StringAcceptor, input_string: str, error_message: str
) -> None:
    """
    Test StringAcceptor with various invalid JSON strings.

    Args:
        string_acceptor (StringAcceptor): The StringAcceptor instance.
        input_string (str): The invalid JSON string input.
        error_message (str): The assertion error message.
    """
    walkers = list(string_acceptor.get_walkers())
    for ch in input_string:
        walkers = [walker for _, walker in string_acceptor.advance_all(walkers, ch)]

    assert not any(
        walker.has_reached_accept_state() for walker in walkers
    ), error_message


def test_empty_string(string_acceptor: StringAcceptor) -> None:
    """
    Test StringAcceptor with an empty JSON string.

    Args:
        string_acceptor (StringAcceptor): The StringAcceptor instance.
    """
    input_string: str = '""'
    expected_value: str = ""

    walkers = list(string_acceptor.get_walkers())
    for ch in input_string:
        walkers = [walker for _, walker in string_acceptor.advance_all(walkers, ch)]

    accepted_walkers = [
        walker for walker in walkers if walker.has_reached_accept_state()
    ]
    assert accepted_walkers, "StringAcceptor did not accept empty string"

    for walker in accepted_walkers:
        assert (
            walker.get_current_value() == expected_value
        ), f"Expected empty string, got '{walker.get_current_value()}'"


def test_string_with_valid_escaped_tab(string_acceptor: StringAcceptor) -> None:
    """
    Test StringAcceptor with a valid escaped tab character.

    Args:
        string_acceptor (StringAcceptor): The StringAcceptor instance.
    """
    input_string: str = '"Tab\\tcharacter"'
    expected_value: str = "Tab\tcharacter"

    walkers = list(string_acceptor.get_walkers())
    for ch in input_string:
        walkers = [walker for _, walker in string_acceptor.advance_all(walkers, ch)]

    accepted_walkers = [
        walker for walker in walkers if walker.has_reached_accept_state()
    ]
    assert (
        accepted_walkers
    ), "StringAcceptor did not accept string with escaped tab character"

    for walker in accepted_walkers:
        assert (
            walker.get_current_value() == expected_value
        ), f"Expected '{expected_value}', got '{walker.get_current_value()}'"


def test_string_with_escaped_solidus(string_acceptor: StringAcceptor) -> None:
    """
    Test StringAcceptor with a string containing an escaped solidus.

    Args:
        string_acceptor (StringAcceptor): The StringAcceptor instance.
    """
    input_string: str = '"Escaped solidus: \\/"'
    expected_value: str = "Escaped solidus: /"

    walkers = list(string_acceptor.get_walkers())
    for ch in input_string:
        walkers = [walker for _, walker in string_acceptor.advance_all(walkers, ch)]

    accepted_walkers = [
        walker for walker in walkers if walker.has_reached_accept_state()
    ]
    assert accepted_walkers, "StringAcceptor did not accept string with escaped solidus"

    for walker in accepted_walkers:
        assert (
            walker.get_current_value() == expected_value
        ), f"Expected '{expected_value}', got '{walker.get_current_value()}'"


def test_string_with_unescaped_control_characters(
    string_acceptor: StringAcceptor,
) -> None:
    """
    Test StringAcceptor with unescaped control characters (should fail).

    Args:
        string_acceptor (StringAcceptor): The StringAcceptor instance.
    """
    input_string: str = '"Invalid \x0b string"'  # Vertical tab, should be escaped

    walkers = list(string_acceptor.get_walkers())
    for ch in input_string:
        walkers = [walker for _, walker in string_acceptor.advance_all(walkers, ch)]

    assert not any(
        walker.has_reached_accept_state() for walker in walkers
    ), "StringAcceptor incorrectly accepted string with unescaped control characters"


def test_string_with_invalid_unicode_escape(string_acceptor: StringAcceptor) -> None:
    """
    Test StringAcceptor with invalid unicode escape sequence.

    Args:
        string_acceptor (StringAcceptor): The StringAcceptor instance.
    """
    input_string: str = '"Invalid unicode: \\u12G4"'  # 'G' is not a hex digit

    walkers = list(string_acceptor.get_walkers())
    for ch in input_string:
        walkers = [walker for _, walker in string_acceptor.advance_all(walkers, ch)]

    assert not any(
        walker.has_reached_accept_state() for walker in walkers
    ), "StringAcceptor incorrectly accepted string with invalid unicode escape"


def test_string_with_incomplete_unicode_escape(string_acceptor: StringAcceptor) -> None:
    """
    Test StringAcceptor with an incomplete unicode escape sequence.

    Args:
        string_acceptor (StringAcceptor): The StringAcceptor instance.
    """
    input_string: str = '"Incomplete unicode: \\u123"'  # Missing one hex digit

    walkers = list(string_acceptor.get_walkers())
    for ch in input_string:
        walkers = [walker for _, walker in string_acceptor.advance_all(walkers, ch)]

    assert not any(
        walker.has_reached_accept_state() for walker in walkers
    ), "StringAcceptor incorrectly accepted string with incomplete unicode escape"


def test_string_missing_start_quote(string_acceptor: StringAcceptor) -> None:
    """
    Test StringAcceptor with a string missing the starting quote.

    Args:
        string_acceptor (StringAcceptor): The StringAcceptor instance.
    """
    input_string: str = 'missing start quote"'

    walkers = list(string_acceptor.get_walkers())
    for ch in input_string:
        walkers = [walker for _, walker in string_acceptor.advance_all(walkers, ch)]

    assert not any(
        walker.has_reached_accept_state() for walker in walkers
    ), "StringAcceptor incorrectly accepted string missing starting quote"


def test_incomplete_string(string_acceptor: StringAcceptor) -> None:
    """
    Test StringAcceptor with an incomplete string (missing closing quote).

    Args:
        string_acceptor (StringAcceptor): The StringAcceptor instance.
    """
    input_string: str = '"incomplete string'

    walkers = list(string_acceptor.get_walkers())
    for ch in input_string:
        walkers = [walker for _, walker in string_acceptor.advance_all(walkers, ch)]

    assert not any(
        walker.has_reached_accept_state() for walker in walkers
    ), "StringAcceptor incorrectly accepted incomplete string"


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
def test_valid_strings_char_by_char(
    string_acceptor: StringAcceptor, input_string: str, expected_value: str
) -> None:
    """
    Test StringAcceptor with various valid JSON strings by advancing character by character.

    Args:
        string_acceptor (StringAcceptor): The StringAcceptor instance.
        input_string (str): The JSON string input.
        expected_value (str): The expected parsed string value.
    """
    walkers = list(string_acceptor.get_walkers())
    for ch in input_string:
        new_walkers = []
        for walker in walkers:
            advanced_walkers = string_acceptor.advance_all([walker], ch)
            new_walkers.extend([w for _, w in advanced_walkers])
        walkers = new_walkers

    accepted_walkers = [
        walker for walker in walkers if walker.has_reached_accept_state()
    ]
    assert accepted_walkers, f"No walker accepted the input: {input_string}"

    for walker in accepted_walkers:
        assert (
            walker.get_current_value() == expected_value
        ), f"Expected '{expected_value}', got '{walker.get_current_value()}'"


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
def test_invalid_strings_char_by_char(
    string_acceptor: StringAcceptor, input_string: str, error_message: str
) -> None:
    """
    Test StringAcceptor with various invalid JSON strings by advancing character by character.

    Args:
        string_acceptor (StringAcceptor): The StringAcceptor instance.
        input_string (str): The invalid JSON string input.
        error_message (str): The assertion error message.
    """
    walkers = list(string_acceptor.get_walkers())
    for ch in input_string:
        new_walkers = []
        for walker in walkers:
            advanced_walkers = string_acceptor.advance_all([walker], ch)
            new_walkers.extend([w for _, w in advanced_walkers])
        walkers = new_walkers

    assert not any(
        walker.has_reached_accept_state() for walker in walkers
    ), error_message


def test_empty_string_char_by_char(string_acceptor: StringAcceptor) -> None:
    """
    Test StringAcceptor with an empty JSON string by advancing character by character.

    Args:
        string_acceptor (StringAcceptor): The StringAcceptor instance.
    """
    input_string: str = '""'
    expected_value: str = ""

    walkers = list(string_acceptor.get_walkers())
    for ch in input_string:
        new_walkers = []
        for walker in walkers:
            advanced_walkers = string_acceptor.advance_all([walker], ch)
            new_walkers.extend([w for _, w in advanced_walkers])
        walkers = new_walkers

    accepted_walkers = [
        walker for walker in walkers if walker.has_reached_accept_state()
    ]
    assert accepted_walkers, "StringAcceptor did not accept empty string"

    for walker in accepted_walkers:
        assert (
            walker.get_current_value() == expected_value
        ), f"Expected empty string, got '{walker.get_current_value()}'"


def test_string_with_valid_escaped_tab_char_by_char(
    string_acceptor: StringAcceptor,
) -> None:
    """
    Test StringAcceptor with a valid escaped tab character by advancing character by character.

    Args:
        string_acceptor (StringAcceptor): The StringAcceptor instance.
    """
    input_string: str = '"Tab\\tcharacter"'
    expected_value: str = "Tab\tcharacter"

    walkers = list(string_acceptor.get_walkers())
    for ch in input_string:
        new_walkers = []
        for walker in walkers:
            advanced_walkers = string_acceptor.advance_all([walker], ch)
            new_walkers.extend([w for _, w in advanced_walkers])
        walkers = new_walkers

    accepted_walkers = [
        walker for walker in walkers if walker.has_reached_accept_state()
    ]
    assert (
        accepted_walkers
    ), "StringAcceptor did not accept string with escaped tab character"

    for walker in accepted_walkers:
        assert (
            walker.get_current_value() == expected_value
        ), f"Expected '{expected_value}', got '{walker.get_current_value()}'"


def test_string_with_escaped_solidus_char_by_char(string_acceptor: StringAcceptor) -> None:
    """
    Test StringAcceptor with a string containing an escaped solidus by advancing character by character.

    Args:
        string_acceptor (StringAcceptor): The StringAcceptor instance.
    """
    input_string: str = '"Escaped solidus: \\/"'
    expected_value: str = "Escaped solidus: /"

    walkers = list(string_acceptor.get_walkers())
    for ch in input_string:
        new_walkers = []
        for walker in walkers:
            advanced_walkers = string_acceptor.advance_all([walker], ch)
            new_walkers.extend([w for _, w in advanced_walkers])
        walkers = new_walkers

    accepted_walkers = [
        walker for walker in walkers if walker.has_reached_accept_state()
    ]
    assert accepted_walkers, "StringAcceptor did not accept string with escaped solidus"

    for walker in accepted_walkers:
        assert (
            walker.get_current_value() == expected_value
        ), f"Expected '{expected_value}', got '{walker.get_current_value()}'"


def test_edge_case():
    string_acceptor = StringAcceptor()
    walkers = string_acceptor.get_walkers()
    assert len(walkers) == 1
    walkers = [walker for _, walker in string_acceptor.advance_all(walkers, '"')]
    assert len(walkers) == 3
    walkers = [walker for _, walker in string_acceptor.advance_all(walkers, 'Hello There!')]
    assert len(walkers) == 3
    walkers = [walker for _, walker in string_acceptor.advance_all(walkers, '"')]
    assert len(walkers) == 1
    assert walkers[0].has_reached_accept_state()
    assert walkers[0].get_current_value() == 'Hello There!'
