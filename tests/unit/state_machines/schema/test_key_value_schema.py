from pse_core.state_machine import StateMachine

from pse.state_machines.schema.key_value_schema import KeyValueSchemaStateMachine


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
