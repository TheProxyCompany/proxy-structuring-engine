from __future__ import annotations

from typing import Iterable, Optional, Union, Type
from pse.state_machine.state_machine import StateMachine, StateMachineWalker
from pse.state_machine.types import EdgeType, StateMachineGraph, StateType
from pse.acceptors.basic.character_acceptors import CharacterAcceptor
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
            end_states=[1, 2, 3, "$"],
            walker_type=walker_type or NumberWalker
        )


class NumberWalker(StateMachineWalker):
    """
    Walker for NumberAcceptor.

    Manages the current state and accumulated value during JSON number parsing.
    """

    def __init__(self, acceptor: NumberAcceptor, current_state: int = 0):
        """
        Initialize the walker.

        Args:
            acceptor (NumberAcceptor): The parent acceptor.
        """
        super().__init__(acceptor, current_state)
        self.acceptor = acceptor
        self.value: Optional[Union[int, float]] = None

    def should_start_transition(self, token: str) -> bool:
        if not self.transition_walker or self.transition_walker.should_start_transition(
            token
        ):
            return True

        reached_accept_state = self.transition_walker.has_reached_accept_state()

        if reached_accept_state:
            self.accepted_history.append(self.transition_walker)
            logger.debug(f"Appended walker to accept history: {self.accepted_history}")
        else:
            logger.debug(f"Walker cannot start transition with {repr(token)}")

        self.transition_walker = None
        self.target_state = None

        return reached_accept_state

    def should_complete_transition(self) -> bool:
        """
        Handle the completion of a transition.

        Args:
            transition_value (str): The value transitioned with.
            target_state (Any): The target state after transition.
            is_end_state (bool): Indicates if the transition leads to an end state.

        Returns:
            bool: Success of the transition.
        """
        self._accepts_remaining_input = True

        return super().should_complete_transition()
