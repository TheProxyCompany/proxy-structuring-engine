from pse.acceptors.schema.property_schema_acceptor import PropertySchemaAcceptor
from pse.state_machine import HierarchicalStateMachine


def test_property_parsing():
    acceptor = PropertySchemaAcceptor(
        prop_name="type",
        prop_schema={"type": "string"},
        context={"defs": {}},
    )
    walkers = list(acceptor.get_walkers())
    walkers = [
        walker for _, walker in HierarchicalStateMachine.advance_all(walkers, '"')
    ]

    assert len(walkers) == 1
    walkers = [
        walker for _, walker in HierarchicalStateMachine.advance_all(walkers, "type")
    ]
    assert len(walkers) == 1
    walkers = [
        walker for _, walker in HierarchicalStateMachine.advance_all(walkers, '": "hi"')
    ]
    assert len(walkers) == 1
    assert walkers[0].has_reached_accept_state()
    assert walkers[0].current_value == ("type", "hi")
