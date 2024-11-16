"""
Acceptors for JSON parsing or constraining LLM generation to JSON outputs.
"""

from typing import (
    Iterable,
    Optional,
)
from pse.core.state_machine import StateMachine, State, Edge
from pse.core.walker import Walker

class JsonAcceptor(StateMachine):
    """
    Acceptor for parsing any JSON value, delegating to specific acceptors based on the value type.
    """

    def get_edges(self, state: State) -> Iterable[Edge]:
        """
        Retrieve the graph edges for transitions out of the current state.

        This method delegates to the appropriate acceptor based on the initial character of the JSON value.

        Args:
            state (int): The current state in the state machine.

        Returns:
            List[Tuple[TokenAcceptor, StateMachineAcceptor.StateType]]: A list of possible transitions represented
            by tuples of TokenAcceptors and their corresponding target states.
        """
        if state == 0:
            from pse.acceptors.basic.text_acceptor import TextAcceptor as NullAcceptor
            from pse.acceptors.basic.boolean_acceptors import BooleanAcceptor
            from pse.acceptors.basic.string_acceptor import StringAcceptor
            from pse.acceptors.basic.number_acceptor import NumberAcceptor
            from pse.acceptors.collections.array_acceptor import ArrayAcceptor
            from pse.acceptors.json.object_acceptor import ObjectAcceptor

            return [
                (ObjectAcceptor(), "$"),
                (ArrayAcceptor(), "$"),
                (StringAcceptor(), "$"),
                (NullAcceptor("null"), "$"),
                (BooleanAcceptor(), "$"),
                (NumberAcceptor(), "$"),
            ]
        return []

    def get_walkers(self, state: Optional[State] = None) -> Iterable[Walker]:
        for edge, _ in self.get_edges(state or 0):
            yield from edge.get_walkers()
