from __future__ import annotations

import json
from typing import Any, Optional, Union
from pse.state_machine.state_machine import StateMachine
from pse.state_machine.types import StateMachineGraph
from pse.acceptors.basic.character_acceptors import CharacterAcceptor
from pse.acceptors.basic.number.integer_acceptor import IntegerAcceptor
from pse.acceptors.basic.number.float_acceptor import FloatAcceptor
from pse.state_machine.empty_transition import EmptyTransition
from pse.state_machine.accepted_state import AcceptedState
import logging

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
    STATE_NUMBER = 2
    STATE_EXPONENT = 3
    STATE_EXPONENT_SIGN = 4
    STATE_EXPONENT_NUMBER = 5
    STATE_END = "$"

    def __init__(self):
        """
        Initialize the NumberAcceptor with its state transitions.
        """
        graph: StateMachineGraph = {
            self.STATE_START: [
                (CharacterAcceptor("-"), self.STATE_NUMBER),
                (EmptyTransition, self.STATE_NUMBER),
            ],
            self.STATE_NUMBER: [
                (FloatAcceptor(), self.STATE_EXPONENT),
                (IntegerAcceptor(), self.STATE_EXPONENT),
            ],
            self.STATE_EXPONENT: [
                (CharacterAcceptor("eE"), self.STATE_EXPONENT_SIGN),
                (EmptyTransition, self.STATE_END),
            ],
            self.STATE_EXPONENT_SIGN: [
                (CharacterAcceptor("+-"), self.STATE_EXPONENT_NUMBER),
                (EmptyTransition, self.STATE_EXPONENT_NUMBER),
            ],
            self.STATE_EXPONENT_NUMBER: [
                (IntegerAcceptor(), self.STATE_END),
            ],
        }
        super().__init__(graph)

    class Cursor(StateMachine.Cursor):
        """
        Cursor for NumberAcceptor.

        Manages the current state and accumulated value during JSON number parsing.
        """

        def __init__(self, acceptor: NumberAcceptor):
            """
            Initialize the cursor.

            Args:
                acceptor (NumberAcceptor): The parent acceptor.
            """
            super().__init__(acceptor)
            self.acceptor = acceptor
            self.text: str = ""
            self.value: Optional[Union[int, float]] = None
            self._accepts_remaining_input = True

        def complete_transition(
            self, transition_value: str | None, target_state: Any, is_end_state: bool
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
            logger.debug(
                f"{self} transiting to {target_state} with {repr(transition_value)}"
            )
            current_value = "".join(
                str(cursor.get_value()) for cursor in self.accept_history
            )
            self.text = (
                current_value + str(transition_value)
                if transition_value
                else current_value
            )
            self.current_state = target_state

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

        def get_value(self) -> Union[str, Union[int, float]]:
            """
            Get the current parsing value.

            Returns:
                Union[str, Union[int, float]]: The accumulated text or the parsed number.
            """
            return self.value if self.value is not None else self.text

        def __repr__(self) -> str:
            """
            Return the string representation of the cursor.

            Returns:
                str: The string representation.
            """
            if self.current_state == self.acceptor.STATE_NUMBER:
                if isinstance(self.transition_cursor, IntegerAcceptor.Cursor):
                    return f"{self.acceptor}.IntegerCursor({self.value if self.value is not None else self.text})"
                elif isinstance(self.transition_cursor, FloatAcceptor.Cursor):
                    return f"{self.acceptor}.FloatCursor({self.value if self.value is not None else self.text})"
            return f"NumberAcceptor.Cursor({self.text if self.text else ''} state={self.current_state}, value={self.value})"

        @property
        def can_handle_remaining_input(self) -> bool:
            """
            Determine if the cursor can handle more input.

            Returns:
                bool: True if the cursor or its transition cursor can handle more input.
            """
            if not self._accepts_remaining_input:
                return False

            if self.accept_history:
                while (
                    self.accept_history
                    and isinstance(self.accept_history[-1], AcceptedState)
                    and self.accept_history[-1].is_empty_transition()
                ):
                    self.accept_history.pop()

                if self.accept_history:
                    return self.accept_history[-1].can_handle_remaining_input

            return bool(self.acceptor.get_edges(self.current_state))
