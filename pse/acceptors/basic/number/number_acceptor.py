from __future__ import annotations

from typing import Iterable, Optional, Type
from pse.state_machine.state_machine import StateMachine, StateMachineWalker
from pse.state_machine.types import EdgeType, StateMachineGraph, StateType
from pse.acceptors.basic.character_acceptor import CharacterAcceptor
from pse.acceptors.basic.number.integer_acceptor import IntegerAcceptor
from pse.acceptors.basic.text_acceptor import TextAcceptor
from pse.acceptors.collections.sequence_acceptor import SequenceAcceptor
import logging

from pse.state_machine.walker import Walker

logger = logging.getLogger(__name__)


class NumberAcceptor(StateMachine):
    """
    Accepts a well-formed JSON number.

    This acceptor defines the state transitions for parsing JSON numbers, handling integer,
    decimal, and exponential formats as specified by the JSON standard.
    """

    _cached_tries = {}

    def get_edges(self, state: StateType) -> Iterable[EdgeType]:
        """
        Get the edges for a given state.
        """
        if state == 0:
            yield from super().get_edges(1)
            yield from super().get_edges(state)
        elif state == 2:
            yield from super().get_edges(state)
            yield from super().get_edges(3)
        elif state == 4:
            yield from super().get_edges(5)
            yield from super().get_edges(state)
        else:
            yield from super().get_edges(state)

    def __init__(self, walker_type: Optional[Type[Walker]] = None):
        """
        Initialize the NumberAcceptor with its state transitions.
        """
        graph: StateMachineGraph = {
            0: [
                (TextAcceptor("-"), 1),
            ],
            1: [
                (IntegerAcceptor(), 2),
            ],
            2: [
                (
                    SequenceAcceptor([TextAcceptor("."), IntegerAcceptor(drop_leading_zeros=False)]),
                    3,
                ),
            ],
            3: [
                (CharacterAcceptor("eE"), 4),
            ],
            4: [
                (CharacterAcceptor("+-"), 5),
            ],
            5: [
                (IntegerAcceptor(), "$"),
            ],
        }
        super().__init__(
            graph,
            end_states=[2, 3, "$"],
            walker_type=walker_type or NumberWalker
        )


class NumberWalker(StateMachineWalker):
    """
    Walker for NumberAcceptor.

    Manages the current state and accumulated value during JSON number parsing.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._accepts_more_input = True

    def can_accept_more_input(self) -> bool:
        if self.transition_walker:
            if self.transition_walker.can_accept_more_input():
                return True

        return (
            bool(self.acceptor.graph.get(self.current_state))
            and self._accepts_more_input
            and not self.remaining_input
        )
