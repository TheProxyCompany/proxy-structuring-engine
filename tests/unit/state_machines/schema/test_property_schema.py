from pse_core.state_machine import StateMachine

from pse.state_machines.schema.property_schema import PropertySchemaStateMachine


def test_property_parsing():
    state_machine = PropertySchemaStateMachine(
        prop_name="type",
        prop_schema={"type": "string"},
        context={"defs": {}},
    )
    walkers = list(state_machine.get_walkers())
    walkers = [walker for _, walker in StateMachine.advance_all(walkers, '"')]

    assert len(walkers) == 1
    walkers = [walker for _, walker in StateMachine.advance_all(walkers, "type")]
    assert len(walkers) == 1
    walkers = [walker for _, walker in StateMachine.advance_all(walkers, '": "hi"')]
    assert len(walkers) == 1
    assert walkers[0].has_reached_accept_state()
    assert walkers[0].get_current_value() == ("type", "hi")
