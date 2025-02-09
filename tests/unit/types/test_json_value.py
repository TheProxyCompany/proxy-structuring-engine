from typing import Any

import pytest

from pse.types.json.json_value import JsonStateMachine


@pytest.fixture
def json_acceptor():
    return JsonStateMachine()


@pytest.fixture
def parse_json(json_acceptor: JsonStateMachine):
    def parser(json_string: str) -> Any:
        steppers = list(json_acceptor.get_steppers())
        for char in json_string:
            steppers = json_acceptor.advance_all_basic(steppers, char)
            if not steppers:
                raise AssertionError("No steppers after parsing")
        accepted_values = [
            stepper.get_current_value()
            for stepper in steppers
            if stepper.has_reached_accept_state()
        ]
        if not accepted_values:
            raise AssertionError("No accepted steppers after parsing")
        return accepted_values[0]

    return parser


def test_json_acceptor_initialization(json_acceptor):
    assert isinstance(json_acceptor, JsonStateMachine), (
        "JsonAcceptor instance was not created."
    )


@pytest.mark.parametrize(
    "json_string, expected",
    [
        (
            '{"name": "John", "age": 30, "city": "New York"}',
            {"name": "John", "age": 30, "city": "New York"},
        ),
        ('["apple", "banana", "cherry"]', ["apple", "banana", "cherry"]),
        ('{"message": "Hello, \\nWorld! \\t😊"}', {"message": "Hello, \nWorld! \t😊"}),
        (
            '{"greeting": "こんにちは世界", "emoji": "🚀🌟"}',
            {"greeting": "こんにちは世界", "emoji": "🚀🌟"},
        ),
    ],
)
def test_parse_valid_json(json_string, expected, parse_json):
    parsed = parse_json(json_string)
    assert parsed == expected


@pytest.mark.parametrize(
    "json_string",
    [
        '{"name": "John", "age": 30,, "city": "New York"}',  # Invalid syntax
        "",  # Empty JSON string
    ],
)
def test_parse_invalid_json(json_string, parse_json):
    with pytest.raises(AssertionError):
        parse_json(json_string)
