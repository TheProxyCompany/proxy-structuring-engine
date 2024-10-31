from __future__ import annotations

from collections.abc import Iterable
import json
from typing import Any, Optional, Union
from pse.state_machine.state_machine import StateMachine, StateMachineWalker
from pse.state_machine.types import StateMachineGraph
from pse.acceptors.basic.character_acceptors import CharacterAcceptor, digit_acceptor
from pse.acceptors.basic.number.integer_acceptor import IntegerAcceptor
from pse.acceptors.basic.text_acceptor import TextAcceptor
from pse.state_machine.empty_transition import EmptyTransition
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

    # State constants
    STATE_START = 0
    STATE_SIGN = 1
    STATE_INTEGER = 2
    STATE_FLOAT = 3
    STATE_DECIMAL = 4
    STATE_EXPONENT = 5
    STATE_EXPONENT_SIGN = 6
    STATE_EXPONENT_NUMBER = 7
    STATE_END = "$"

    def __init__(self):
        """
        Initialize the NumberAcceptor with its state transitions.
        """
        graph: StateMachineGraph = {
            self.STATE_START: [
                (TextAcceptor("-", is_optional=True), self.STATE_INTEGER),
            ],
            self.STATE_INTEGER: [
                (IntegerAcceptor(), self.STATE_FLOAT),
            ],
            self.STATE_FLOAT: [
                (TextAcceptor("."), self.STATE_DECIMAL),
                (EmptyTransition, self.STATE_EXPONENT),
            ],
            self.STATE_DECIMAL: [
                (digit_acceptor, self.STATE_EXPONENT),
            ],
            self.STATE_EXPONENT: [
                (CharacterAcceptor("eE"), self.STATE_EXPONENT_SIGN),
                (EmptyTransition, self.STATE_END),
            ],
            self.STATE_EXPONENT_SIGN: [
                (CharacterAcceptor("+-", is_optional=True), self.STATE_EXPONENT_NUMBER),
            ],
            self.STATE_EXPONENT_NUMBER: [
                (IntegerAcceptor(), self.STATE_END),
            ],
        }
        super().__init__(graph)

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
        self._accepts_remaining_input = True

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
