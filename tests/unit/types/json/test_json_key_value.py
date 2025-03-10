from pse.types.json.json_key_value import (
    KeyValueSchemaStateMachine,
    KeyValueSchemaStepper,
)


def test_property_parsing():
    state_machine = KeyValueSchemaStateMachine(
        prop_name="type",
        prop_schema={"type": "string"},
        context={"defs": {}},
    )
    steppers = list(state_machine.get_steppers())
    steppers = state_machine.advance_all_basic(steppers, '"')

    assert len(steppers) == 1
    steppers = state_machine.advance_all_basic(steppers, "type")
    assert len(steppers) == 1
    steppers = state_machine.advance_all_basic(steppers, '": "hi"')
    assert len(steppers) == 1
    assert steppers[0].has_reached_accept_state()
    assert steppers[0].get_current_value() == ("type", "hi")


def test_property_parsing_with_string_sm():
    """Test KeyValueSchemaStateMachine when prop_name is None, using StringStateMachine."""
    state_machine = KeyValueSchemaStateMachine(
        prop_name=None,  # This tests the branch that uses StringStateMachine
        prop_schema={"type": "string"},
        context={"defs": {}, "path": "/parent"},
    )

    steppers = list(state_machine.get_steppers())
    steppers = state_machine.advance_all_basic(steppers, '"dynamic_key"')
    assert len(steppers) > 0

    steppers = state_machine.advance_all_basic(steppers, ': "value"')
    assert len(steppers) > 0
    assert any(stepper.has_reached_accept_state() for stepper in steppers)

    for stepper in steppers:
        if stepper.has_reached_accept_state():
            assert stepper.get_current_value() == ("dynamic_key", "value")


def test_key_value_schema_stepper_initialization():
    """Test KeyValueSchemaStepper initialization and state machine access."""
    state_machine = KeyValueSchemaStateMachine(
        prop_name="test_prop",
        prop_schema={"type": "string"},
        context={"defs": {}},
    )

    stepper = state_machine.get_new_stepper()
    assert isinstance(stepper, KeyValueSchemaStepper), "Should return a KeyValueSchemaStepper instance"
    assert stepper.state_machine is state_machine, "Stepper should reference the correct state machine"

def test_key_value_schema_stepper_equality():
    """Test KeyValueSchemaStepper equality."""
    state_machine = KeyValueSchemaStateMachine(
        prop_name="test_prop",
        prop_schema={"type": "string"},
        context={"defs": {}},
    )

    steppers1 = state_machine.get_steppers()
    steppers2 = state_machine.get_steppers()
    assert steppers1[0] == steppers2[0]

    steppers1 = state_machine.advance_all_basic(steppers1, '"test_prop": "value"')
    assert len(steppers1) == 1
    steppers2 = state_machine.advance_all_basic(steppers2, '"test_prop": "value"')
    assert len(steppers2) == 1
    assert steppers1[0] == steppers2[0]

    stepper3 = state_machine.get_new_stepper()
    assert steppers1[0] != stepper3
