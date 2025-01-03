"""
Acceptors for JSON parsing or constraining LLM generation to JSON outputs.
"""

from pse_core import Edge, State
from pse_core.state_machine import StateMachine
from pse_core.walker import Walker


class JsonAcceptor(StateMachine):
    """
    Acceptor for parsing any JSON value, delegating to specific acceptors based on the value type.
    """

    def get_edges(self, state: State) -> list[Edge]:
        """
        Retrieve the graph edges for transitions out of the current state.

        This method delegates to the appropriate state_machine based on the initial character of the JSON value.

        Args:
            state (int): The current state in the state machine.

        Returns:
            List[Tuple[TokenAcceptor, StateMachineAcceptor.StateType]]: A list of possible transitions represented
            by tuples of TokenAcceptors and their corresponding target states.
        """
        if state == 0:
            from pse.state_machines.basic.boolean_acceptor import BooleanAcceptor
            from pse.state_machines.basic.number_acceptor import NumberAcceptor
            from pse.state_machines.basic.string_acceptor import StringAcceptor
            from pse.state_machines.basic.text_acceptor import (
                TextAcceptor as NullAcceptor,
            )
            from pse.state_machines.collections.array_acceptor import ArrayAcceptor
            from pse.state_machines.json.object_acceptor import ObjectAcceptor

            return [
                (ObjectAcceptor(), "$"),
                (ArrayAcceptor(), "$"),
                (StringAcceptor(), "$"),
                (NullAcceptor("null"), "$"),
                (BooleanAcceptor(), "$"),
                (NumberAcceptor(), "$"),
            ]
        return []

    def get_walkers(self, state: State | None = None) -> list[Walker]:
        walkers = []
        for edge, _ in self.get_edges(state or 0):
            walkers.extend(edge.get_walkers())
        return walkers
