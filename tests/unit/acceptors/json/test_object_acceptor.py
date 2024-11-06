import pytest
from pse.acceptors.json.object_acceptor import ObjectAcceptor
from pse.acceptors.basic.text_acceptor import TextAcceptor
from pse.acceptors.basic.whitespace_acceptor import WhitespaceAcceptor
from pse.acceptors.json.string_acceptor import StringAcceptor
from pse.state_machine.state_machine import StateMachine
from pse.acceptors.json.property_acceptor import PropertyAcceptor


@pytest.fixture
def object_acceptor() -> ObjectAcceptor:
    return ObjectAcceptor()


def test_get_edges_initial_state(object_acceptor: ObjectAcceptor):
    edges = object_acceptor.get_edges(0)
    assert len(edges) == 1
    assert isinstance(edges[0][0], TextAcceptor)
    assert edges[0][1] == 1


def test_get_edges_property_parsing(object_acceptor: ObjectAcceptor):
    edges = object_acceptor.get_edges(2)
    assert len(edges) == 2
    assert isinstance(edges[1][0], PropertyAcceptor)
    assert edges[1][1] == 3


def test_get_edges_end_state(object_acceptor: ObjectAcceptor):
    edges = object_acceptor.get_edges(4)
    assert len(edges) == 2
    assert isinstance(edges[0][0], TextAcceptor)
    assert edges[1][1] == "$"


def test_walker_initialization(object_acceptor: ObjectAcceptor):
    walker = object_acceptor.walker_class(object_acceptor)
    assert walker.value == {}
    assert walker.acceptor is object_acceptor


def test_walker_complete_transition_valid(object_acceptor: ObjectAcceptor):
    walker = object_acceptor.walker_class(object_acceptor)
    walker.current_state = 2
    result = walker.should_complete_transition(("key", "value"), 3, False)
    assert result
    assert walker.value == {"key": "value"}


def test_walker_get_value_empty(object_acceptor: ObjectAcceptor):
    walker = object_acceptor.walker_class(object_acceptor)
    assert walker.current_value() == {}


def test_walker_get_value_with_data(object_acceptor: ObjectAcceptor):
    walker = object_acceptor.walker_class(object_acceptor)
    walker.value = {"key1": "value1", "key2": "value2"}
    assert walker.current_value() == {"key1": "value1", "key2": "value2"}


def test_property_acceptor_initialization():
    prop_acceptor = PropertyAcceptor()
    assert len(prop_acceptor.acceptors) == 5
    assert isinstance(prop_acceptor.acceptors[0], StringAcceptor)
    assert isinstance(prop_acceptor.acceptors[1], WhitespaceAcceptor)


def test_property_acceptor_walker_complete_transition():
    prop_acceptor = PropertyAcceptor()
    walker = prop_acceptor.walker_class(prop_acceptor)

    result_key = walker.should_complete_transition("key", 1, False)
    assert result_key
    assert walker.prop_name == "key"

    result_value = walker.should_complete_transition("value", 5, True)
    assert result_value
    assert walker.prop_value == "value"


def test_property_acceptor_walker_get_value_valid():
    prop_acceptor = PropertyAcceptor()
    walker = prop_acceptor.walker_class(prop_acceptor)
    walker.prop_name = "key"
    walker.prop_value = "value"

    assert walker.current_value() == ("key", "value")


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
        # (
        #     '{  "key1"  :  "value1"  }',
        #     {"key1": "value1"},
        # ),
        # (
        #     '{ "a" :  "b","x":   "y"}',
        #     {"a": "b", "x": "y"},
        # ),
    ],
)
def test_valid_json_objects(object_acceptor: ObjectAcceptor, json_string, expected):
    walkers = list(object_acceptor.get_walkers())
    for char in json_string:
        walkers = list(StateMachine.advance_all(walkers, char))

    accepted_walkers = [
        walker for walker in walkers if walker.has_reached_accept_state()
    ]
    assert accepted_walkers, f"No walker reached an accepted state for: {json_string}"

    for walker in accepted_walkers:
        assert walker.current_value() == expected


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
    walkers = StateMachine.advance_all(walkers, json_string)

    accepted_walkers = [
        walker for walker in walkers if walker.has_reached_accept_state()
    ]
    assert accepted_walkers, f"No walker reached an accepted state for: {json_string}"

    for walker in accepted_walkers:
        assert walker.current_value() == expected


@pytest.mark.parametrize(
    "json_string",
    [
        (
            "{key: 'value'}"
        ),  # Missing quotes around key and inconsistent quotes around value
        ('{"key1": undefined}'),
        ('{"key1": "value1",, "key2": "value2"}'),
        ('{"key1": "value1"'),
    ],
)
def test_invalid_json_objects(object_acceptor: ObjectAcceptor, json_string):
    walkers = list(object_acceptor.get_walkers())
    for char in json_string:
        walkers = StateMachine.advance_all(walkers, char)
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
        walkers = list(StateMachine.advance_all(walkers, char))

    accepted_walkers = [
        walker for walker in walkers if walker.has_reached_accept_state()
    ]
    assert accepted_walkers, f"No walker reached an accepted state for: {json_string}"

    for walker in accepted_walkers:
        assert walker.current_value() == expected


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
        walkers = list(StateMachine.advance_all(walkers, char))
    assert not any(
        walker.has_reached_accept_state() for walker in walkers
    ), f"Walkers should not be in an accepted state for invalid JSON: {json_string}"
