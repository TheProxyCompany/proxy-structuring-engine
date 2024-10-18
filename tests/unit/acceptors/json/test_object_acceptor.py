import pytest
from pse.acceptors.json.object_acceptor import ObjectAcceptor
from pse.acceptors.basic.text_acceptor import TextAcceptor
from pse.acceptors.basic.whitespace_acceptor import WhitespaceAcceptor
from pse.acceptors.json.string_acceptor import StringAcceptor
from pse.state_machine.state_machine import StateMachine

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
    assert isinstance(edges[1][0], ObjectAcceptor.PropertyAcceptor)
    assert edges[1][1] == 3

def test_get_edges_end_state(object_acceptor: ObjectAcceptor):
    edges = object_acceptor.get_edges(4)
    assert len(edges) == 2
    assert isinstance(edges[0][0], TextAcceptor)
    assert edges[1][1] == "$"


def test_cursor_initialization(object_acceptor: ObjectAcceptor):
    cursor = object_acceptor.Cursor(object_acceptor)
    assert cursor.value == {}
    assert cursor.acceptor is object_acceptor

def test_cursor_complete_transition_valid(object_acceptor: ObjectAcceptor):
    cursor = object_acceptor.Cursor(object_acceptor)
    cursor.current_state = 2
    result = cursor.complete_transition(("key", "value"), 3, False)
    assert result
    assert cursor.value == {"key": "value"}

def test_cursor_get_value_empty(object_acceptor: ObjectAcceptor):
    cursor = object_acceptor.Cursor(object_acceptor)
    assert cursor.get_value() == {}

def test_cursor_get_value_with_data(object_acceptor: ObjectAcceptor):
    cursor = object_acceptor.Cursor(object_acceptor)
    cursor.value = {"key1": "value1", "key2": "value2"}
    assert cursor.get_value() == {"key1": "value1", "key2": "value2"}

def test_property_acceptor_initialization():
    prop_acceptor = ObjectAcceptor.PropertyAcceptor()
    assert len(prop_acceptor.graph.values()) == 5
    assert isinstance(prop_acceptor.graph[0][0][0], StringAcceptor)
    assert isinstance(prop_acceptor.graph[1][0][0], WhitespaceAcceptor)


def test_property_acceptor_cursor_complete_transition():
    prop_acceptor = ObjectAcceptor.PropertyAcceptor()
    cursor = prop_acceptor.Cursor(prop_acceptor)

    result_key = cursor.complete_transition("key", 1, False)
    assert result_key
    assert cursor.prop_name == "key"

    result_value = cursor.complete_transition("value", 5, True)
    assert result_value
    assert cursor.prop_value == "value"

def test_property_acceptor_cursor_get_value_valid():
    prop_acceptor = ObjectAcceptor.PropertyAcceptor()
    cursor = prop_acceptor.Cursor(prop_acceptor)
    cursor.prop_name = "key"
    cursor.prop_value = "value"

    assert cursor.get_value() == ("key", "value")


@pytest.mark.parametrize(
    "json_string, expected",
    [
        ("{}", {}),
        ('{"key1": "value1"}', {"key1": "value1"}),
        ('{"key1": "value1", "key2": "value2"}', {"key1": "value1", "key2": "value2"}),
        ('{"outer": {"inner": "value"}}', {"outer": {"inner": "value"}}),
        ('{"ke@y1": "val€ue1", "ke¥2": "valu😊e2"}', {"ke@y1": "val€ue1", "ke¥2": "valu😊e2"}),
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
    cursors = list(object_acceptor.get_cursors())
    for char in json_string:
        cursors = list(StateMachine.advance_all(cursors, char))

    accepted_cursors = [cursor for cursor in cursors if cursor.in_accepted_state()]
    assert accepted_cursors, f"No cursor reached an accepted state for: {json_string}"

    for cursor in accepted_cursors:
        assert cursor.get_value() == expected

@pytest.mark.parametrize(
    "json_string, expected",
    [
        ("{}", {}),
        ('{"key1": "value1"}', {"key1": "value1"}),
        ('{"key1": "value1", "key2": "value2"}', {"key1": "value1", "key2": "value2"}),
        ('{"outer": {"inner": "value"}}', {"outer": {"inner": "value"}}),
        ('{"ke@y1": "val€ue1", "ke¥2": "valu😊e2"}', {"ke@y1": "val€ue1", "ke¥2": "valu😊e2"}),
        ('{"key1": ""}', {"key1": ""}),
        ('{   "key1"   :    "value1"   ,   "key2"   :    "value2"   }', {"key1": "value1", "key2": "value2"}),
    ],
)
def test_valid_json_objects_all_at_once(object_acceptor: ObjectAcceptor, json_string, expected):
    cursors = list(object_acceptor.get_cursors())
    cursors = StateMachine.advance_all(cursors, json_string)

    accepted_cursors = [cursor for cursor in cursors if cursor.in_accepted_state()]
    assert accepted_cursors, f"No cursor reached an accepted state for: {json_string}"

    for cursor in accepted_cursors:
        assert cursor.get_value() == expected

@pytest.mark.parametrize(
    "json_string",
    [
        ("{key: 'value'}"),  # Missing quotes around key and inconsistent quotes around value
        ('{"key1": undefined}'),
        ('{"key1": "value1",, "key2": "value2"}'),
        ('{"key1": "value1"'),
    ],
)
def test_invalid_json_objects(object_acceptor: ObjectAcceptor, json_string):
    cursors = list(object_acceptor.get_cursors())
    for char in json_string:
        cursors = StateMachine.advance_all(cursors, char)
    assert not any(cursor.in_accepted_state() for cursor in cursors), f"Cursors should not be in an accepted state for invalid JSON: {json_string}"
