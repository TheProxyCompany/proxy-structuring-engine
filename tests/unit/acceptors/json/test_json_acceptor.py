import pytest
from typing import Any
from pse.acceptors.json.json_acceptor import JsonAcceptor


@pytest.fixture
def json_acceptor():
    return JsonAcceptor()


@pytest.fixture
def parse_json(json_acceptor):
    def parser(json_string: str) -> Any:
        cursors = json_acceptor.get_cursors()
        for char in json_string:
            cursors = json_acceptor.advance_all(cursors, char)
            if not cursors:
                raise AssertionError("No cursors after parsing")
        accepted_values = [
            cursor.get_value() for cursor in cursors if cursor.in_accepted_state()
        ]
        if not accepted_values:
            raise AssertionError("No accepted cursors after parsing")
        return accepted_values[0]
    return parser


def test_json_acceptor_initialization(json_acceptor):
    assert isinstance(json_acceptor, JsonAcceptor), "JsonAcceptor instance was not created."


@pytest.mark.parametrize(
    "json_string, expected",
    [
        ('{"name": "John", "age": 30, "city": "New York"}', {"name": "John", "age": 30, "city": "New York"}),
        ('["apple", "banana", "cherry"]', ["apple", "banana", "cherry"]),
        ('{"message": "Hello, \\nWorld! \\t😊"}', {"message": "Hello, \nWorld! \t😊"}),
        ('{"greeting": "こんにちは世界", "emoji": "🚀🌟"}', {"greeting": "こんにちは世界", "emoji": "🚀🌟"}),
    ]
)
def test_parse_valid_json(json_string, expected, parse_json):
    parsed = parse_json(json_string)
    assert parsed == expected, "Parsed JSON does not match expected output."


@pytest.mark.parametrize(
    "json_string",
    [
        '{"name": "John", "age": 30,, "city": "New York"}',  # Invalid syntax
        "",  # Empty JSON string
    ]
)
def test_parse_invalid_json(json_string, parse_json):
    with pytest.raises(AssertionError):
        parse_json(json_string)
