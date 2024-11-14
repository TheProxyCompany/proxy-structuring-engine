"""
Acceptors for JSON parsing or constraining LLM generation to JSON outputs.
Thread-safe implementations with caching, error handling, and performance optimizations.
"""

from typing import (
    Iterable,
    Optional,
)
from pse.state_machine.state_machine import (
    StateMachine,
    StateType,
    EdgeType
)
from pse.state_machine.walker import Walker


class JsonAcceptor(StateMachine):
    """
    Acceptor for parsing any JSON value, delegating to specific acceptors based on the value type.
    """

    def get_edges(self, state: StateType) -> Iterable[EdgeType]:
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
            from .object_acceptor import ObjectAcceptor
            from ..basic.primitive_acceptors import BooleanAcceptor, NullAcceptor
            from .string_acceptor import StringAcceptor
            from ..collections.array_acceptor import ArrayAcceptor
            from ..basic.number.number_acceptor import NumberAcceptor

            return [
                (ObjectAcceptor(), "$"),
                (ArrayAcceptor(), "$"),
                (StringAcceptor(), "$"),
                (NullAcceptor(), "$"),
                (BooleanAcceptor(), "$"),
                (NumberAcceptor(), "$"),
            ]
        return []

    def get_walkers(self, state: Optional[StateType] = None) -> Iterable[Walker]:
        for edge, _ in self.get_edges(state or 0):
            yield from edge.get_walkers()
