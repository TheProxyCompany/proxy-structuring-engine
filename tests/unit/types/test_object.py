import pytest

from pse.types.object import ObjectStateMachine


@pytest.fixture
def object_acceptor() -> ObjectStateMachine:
    return ObjectStateMachine()


def test_basic():
    sm = ObjectStateMachine()
    steppers = sm.get_steppers()
    input = '{"value": '
    steppers = sm.advance_all_basic(steppers, input)
    steppers = sm.advance_all_basic(steppers, "1")
    steppers = sm.advance_all_basic(steppers, ".2")
    steppers = sm.advance_all_basic(steppers, "e2}")
    assert len(steppers) == 1
    assert steppers[0].has_reached_accept_state()


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
def test_valid_json_objects(json_string, expected):
    sm = ObjectStateMachine(is_optional=True)  # optional is needed for empty objects
    steppers = sm.get_steppers()
    for char in json_string:
        steppers = sm.advance_all_basic(steppers, char)

    accepted_steppers = [
        stepper for stepper in steppers if stepper.has_reached_accept_state()
    ]
    assert accepted_steppers, f"No stepper reached an accepted state for: {json_string}"

    for stepper in accepted_steppers:
        assert stepper.get_current_value() == expected


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
def test_valid_json_objects_all_at_once(json_string, expected):
    sm = ObjectStateMachine(is_optional=True)  # optional is needed for empty objects
    steppers = list(sm.get_steppers())
    steppers = sm.advance_all_basic(steppers, json_string)

    accepted_steppers = [
        stepper for stepper in steppers if stepper.has_reached_accept_state()
    ]
    assert accepted_steppers, f"No stepper reached an accepted state for: {json_string}"

    for stepper in accepted_steppers:
        assert stepper.get_current_value() == expected


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
def test_invalid_json_objects(object_acceptor: ObjectStateMachine, json_string):
    steppers = list(object_acceptor.get_steppers())
    for char in json_string:
        steppers = object_acceptor.advance_all_basic(steppers, char)
    assert not any(stepper.has_reached_accept_state() for stepper in steppers), (
        f"Steppers should not be in an accepted state for invalid JSON: {json_string}"
    )


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
def test_complex_json_objects(
    object_acceptor: ObjectStateMachine, json_string, expected
):
    steppers = list(object_acceptor.get_steppers())
    for char in json_string:
        steppers = object_acceptor.advance_all_basic(steppers, char)

    accepted_steppers = [
        stepper for stepper in steppers if stepper.has_reached_accept_state()
    ]
    assert accepted_steppers, f"No stepper reached an accepted state for: {json_string}"

    for stepper in accepted_steppers:
        assert stepper.get_current_value() == expected


def test_edge_case(object_acceptor: ObjectStateMachine):
    steppers = list(object_acceptor.get_steppers())
    steppers = object_acceptor.advance_all_basic(steppers, '{"":""')
    assert len(steppers) == 3
    steppers = object_acceptor.advance_all_basic(steppers, ",")
    assert len(steppers) == 2
    steppers = object_acceptor.advance_all_basic(steppers, '"')
    assert len(steppers) == 3


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
def test_more_invalid_json_objects(object_acceptor: ObjectStateMachine, json_string):
    steppers = list(object_acceptor.get_steppers())
    for char in json_string:
        steppers = object_acceptor.advance_all_basic(steppers, char)
    assert not any(stepper.has_reached_accept_state() for stepper in steppers), (
        f"Steppers should not be in an accepted state for invalid JSON: {json_string}"
    )


def test_no_spaces(object_acceptor: ObjectStateMachine):
    input_string = '{"a":"b","c":"d"}'
    steppers = list(object_acceptor.get_steppers())
    steppers = object_acceptor.advance_all_basic(steppers, input_string)
    assert len(steppers) == 1
    stepper = steppers[0]
    assert stepper.has_reached_accept_state()
    assert stepper.get_current_value() == {"a": "b", "c": "d"}


def test_basic_token_by_token():
    object_acceptor = ObjectStateMachine()
    steppers = object_acceptor.get_steppers()
    steppers = object_acceptor.advance_all_basic(steppers, "{")
    assert len(steppers) == 2
    assert all(not stepper.has_reached_accept_state() for stepper in steppers)
    steppers = object_acceptor.advance_all_basic(steppers, "\n\n")
    assert len(steppers) == 2
    assert all(not stepper.has_reached_accept_state() for stepper in steppers)


def test_whitespace_acceptor_integration_with_object_acceptor():
    """Test WhitespaceAcceptor in the context of ObjectAcceptor."""
    token = '{ "key": "value", "number": 42 }'
    state_machine = ObjectStateMachine()
    steppers = state_machine.get_steppers()
    advanced_steppers = state_machine.advance_all_basic(steppers, token)

    assert any(stepper.has_reached_accept_state() for stepper in advanced_steppers)
    for stepper in advanced_steppers:
        if stepper.has_reached_accept_state():
            obj = stepper.get_current_value()
            assert obj == {"key": "value", "number": 42}


def test_nested_object():
    sm = ObjectStateMachine()
    steppers = sm.get_steppers()
    steppers = sm.advance_all_basic(steppers, '{"a":{"b":"c')
    assert len(steppers) == 3
    steppers = sm.advance_all_basic(steppers, '"}}')
    assert any(stepper.has_reached_accept_state() for stepper in steppers)
