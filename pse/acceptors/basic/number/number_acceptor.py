from __future__ import annotations

import json
from typing import Any, Iterable, Optional, Union
from pse.state_machine.state_machine import StateMachine, StateMachineWalker
from pse.state_machine.types import EdgeType, StateMachineGraph, StateType
from pse.acceptors.basic.character_acceptors import CharacterAcceptor, digit_acceptor
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
            yield from super().get_edges(state)
            yield from super().get_edges(1)
        elif state == 4:
            yield from super().get_edges(state)
            yield from super().get_edges(5)
        else:
            yield from super().get_edges(state)

    def __init__(self):
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
                    SequenceAcceptor([TextAcceptor("."), digit_acceptor]),
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
        super().__init__(graph, end_states=[1, 2, "$"])

    def get_walkers(self) -> Iterable[Walker]:
        initial_walker = NumberWalker(acceptor=self)
        yield from self._branch_walkers(initial_walker)


class NumberWalker(StateMachineWalker):
    """
    Walker for NumberAcceptor.

    Manages the current state and accumulated value during JSON number parsing.
    """

    def __init__(self, acceptor: NumberAcceptor):
        """
        Initialize the walker.

        Args:
            acceptor (NumberAcceptor): The parent acceptor.
        """
        super().__init__(acceptor)
        self.acceptor = acceptor
        self.text: str = ""
        self.value: Optional[Union[int, float]] = None

    def should_start_transition(self, token: str) -> bool:
        if self.transition_walker and not self.transition_walker.should_start_transition(token):
            transition_reached_accept_state = self.transition_walker.has_reached_accept_state()
            logger.debug(f"Walker cannot start transition with {repr(token)}")
            self.transition_walker = None
            self.target_state = None
            if transition_reached_accept_state:
                logger.debug(f"starting transition with {repr(token)} from\n{self}")
            return transition_reached_accept_state

        return True

    def should_complete_transition(
        self, transition_value: Any, is_end_state: bool
    ) -> bool:
        """
        Handle the completion of a transition.

        Args:
            transition_value (str): The value transitioned with.
            target_state (Any): The target state after transition.
            is_end_state (bool): Indicates if the transition leads to an end state.

        Returns:
            bool: Success of the transition.
        """
        logger.debug(f"{self} transitioning with {repr(transition_value)}")
        current_value = "".join(
            str(walker.accumulated_value()) for walker in self.accept_history
        )
        self.text = (
            current_value + str(transition_value) if transition_value else current_value
        )

        if not self.text:
            return True

        self._accepts_remaining_input = True

        try:
            self.value = int(self.text)
        except ValueError:
            try:
                self.value = float(self.text)
            except ValueError:
                try:
                    self.value = json.loads(self.text)
                except ValueError:
                    logger.error(
                        f"value error {self.text}, transition_value: {repr(transition_value)}"
                    )
                    if transition_value:
                        self._accepts_remaining_input = False

        return True

    def accumulated_value(self) -> Union[str, Union[int, float]]:
        """
        Get the current parsing value.

        Returns:
            Union[str, Union[int, float]]: The accumulated text or the parsed number.
        """
        return self.value if self.value is not None else self.text
