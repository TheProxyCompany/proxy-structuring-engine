import pytest
from pse_core.state_machine import StateMachine

from pse.state_machines.json.object_acceptor import ObjectAcceptor


@pytest.fixture
def object_acceptor() -> ObjectAcceptor:
    return ObjectAcceptor()


@pytest.mark.parametrize(
    "json_string, expected",
    [
        ("{}", {}),
        ('{"key1": "value1"}', {"key1": "value1"}),
        ('{"key1": "value1", "key2": "value2"}', {"key1": "value1", "key2": "value2"}),
        ('{"outer": {"inner": "value"}}', {"outer": {"inner": "value"}}),
        (
            '{"ke@y1": "val€ue1", "ke¥2": "valu😊e2"}',
            {"ke@y1": "val€ue1", "ke¥2": "valu😊e2"},
        ),
        ('{"key1": ""}', {"key1": ""}),
        (
            '{  "key1"  :  "value1"  }',
            {"key1": "value1"},
        ),
        (
            '{ "a" :  "b","x":   "y"}',
            {"a": "b", "x": "y"},
        ),
    ],
)
def test_valid_json_objects(object_acceptor: ObjectAcceptor, json_string, expected):
    walkers = list(object_acceptor.get_walkers())
    for char in json_string:
        walkers = [walker for _, walker in StateMachine.advance_all(walkers, char)]

    accepted_walkers = [
        walker for walker in walkers if walker.has_reached_accept_state()
    ]
    assert accepted_walkers, f"No walker reached an accepted state for: {json_string}"

    for walker in accepted_walkers:
        assert walker.get_current_value() == expected


@pytest.mark.parametrize(
    "json_string, expected",
    [
        ("{}", {}),
        ('{"key1": "value1"}', {"key1": "value1"}),
        ('{"key1": "value1", "key2": "value2"}', {"key1": "value1", "key2": "value2"}),
        ('{"outer": {"inner": "value"}}', {"outer": {"inner": "value"}}),
        (
            '{"ke@y1": "val€ue1", "ke¥2": "valu😊e2"}',
            {"ke@y1": "val€ue1", "ke¥2": "valu😊e2"},
        ),
        ('{"key1": ""}', {"key1": ""}),
        (
            '{   "key1"   :    "value1"   ,   "key2"   :    "value2"   }',
            {"key1": "value1", "key2": "value2"},
        ),
    ],
)
def test_valid_json_objects_all_at_once(
    object_acceptor: ObjectAcceptor, json_string, expected
):
    walkers = list(object_acceptor.get_walkers())
    walkers = [walker for _, walker in StateMachine.advance_all(walkers, json_string)]

    accepted_walkers = [
        walker for walker in walkers if walker.has_reached_accept_state()
    ]
    assert accepted_walkers, f"No walker reached an accepted state for: {json_string}"

    for walker in accepted_walkers:
        assert walker.get_current_value() == expected


@pytest.mark.parametrize(
    "json_string",
    [
        (
            "{key: 'value'}"
        ),  # Missing quotes around key and inconsistent quotes around value
        ('{"key1": undefined}'),
        ('{"a": "b",, "c": "d"}'),
        ('{"key1": "value1"'),
    ],
)
def test_invalid_json_objects(object_acceptor: ObjectAcceptor, json_string):
    walkers = list(object_acceptor.get_walkers())
    for char in json_string:
        walkers = [walker for _, walker in StateMachine.advance_all(walkers, char)]
    assert not any(
        walker.has_reached_accept_state() for walker in walkers
    ), f"Walkers should not be in an accepted state for invalid JSON: {json_string}"


@pytest.mark.parametrize(
    "json_string, expected",
    [
        # Deep nesting tests
        ('{"a":{"b":{"c":{"d":"e"}}}}', {"a": {"b": {"c": {"d": "e"}}}}),
        # Multiple nested objects
        ('{"a":{"x":"y"},"b":{"z":"w"}}', {"a": {"x": "y"}, "b": {"z": "w"}}),
        # Mixed whitespace
        (
            '{\n"key1"\t:\r"value1",\n"key2"  :  "value2"\n}',
            {"key1": "value1", "key2": "value2"},
        ),
        # Empty string values
        ('{"":"","x":""}', {"": "", "x": ""}),
        # Unicode in keys and values with escapes
        ('{"key\\u0020space": "value\\n\\t"}', {"key space": "value\n\t"}),
    ],
)
def test_complex_json_objects(object_acceptor: ObjectAcceptor, json_string, expected):
    walkers = list(object_acceptor.get_walkers())
    for char in json_string:
        walkers = [walker for _, walker in StateMachine.advance_all(walkers, char)]

    accepted_walkers = [
        walker for walker in walkers if walker.has_reached_accept_state()
    ]
    assert accepted_walkers, f"No walker reached an accepted state for: {json_string}"

    for walker in accepted_walkers:
        assert walker.get_current_value() == expected

def test_edge_case(object_acceptor: ObjectAcceptor):
    walkers = list(object_acceptor.get_walkers())
    walkers = [walker for _, walker in StateMachine.advance_all(walkers, '{"":""')]
    assert len(walkers) == 3

    walkers = [walker for _, walker in StateMachine.advance_all(walkers, ',')]
    assert len(walkers) == 1
    walkers = [walker for _, walker in StateMachine.advance_all(walkers, '"')]
    assert len(walkers) == 3


@pytest.mark.parametrize(
    "json_string",
    [
        '{"key": value}',  # Unquoted value
        '{"key": undefined}',  # JavaScript undefined
        "{\"key\": 'value'}",  # Single quotes
        '{key: "value"}',  # Unquoted key
        '{"key": "value",}',  # Trailing comma
        '{"key":: "value"}',  # Double colon
        '{"key": "value"',  # Unclosed object
        '{"a":{"b":"c"}',  # Unclosed nested object
    ],
)
def test_more_invalid_json_objects(object_acceptor: ObjectAcceptor, json_string):
    walkers = list(object_acceptor.get_walkers())
    for char in json_string:
        walkers = [walker for _, walker in StateMachine.advance_all(walkers, char)]
    assert not any(
        walker.has_reached_accept_state() for walker in walkers
    ), f"Walkers should not be in an accepted state for invalid JSON: {json_string}"


def test_no_spaces(object_acceptor: ObjectAcceptor):
    input_string = '{"a":"b","c":"d"}'
    walkers = list(object_acceptor.get_walkers())
    walkers = [walker for _, walker in StateMachine.advance_all(walkers, input_string)]
    assert len(walkers) == 1
    walker = walkers[0]
    assert walker.has_reached_accept_state()
    assert walker.get_current_value() == {"a": "b", "c": "d"}
