from __future__ import annotations
from typing import Any

from pse.state_machine.state_machine import StateMachine
from pse.acceptors.basic.character_acceptors import digit_acceptor
from pse.acceptors.basic.number.integer_acceptor import IntegerAcceptor
from pse.acceptors.basic.text_acceptor import TextAcceptor

class FloatAcceptor(StateMachine):
    """
    Accepts a well-formed floating-point number as per JSON specification.
    """

    def __init__(self) -> None:
        """
        Initialize the PropertyAcceptor with a predefined sequence of token acceptors.

        Args:
            sequence (Optional[List[TokenAcceptor]], optional): Custom sequence of acceptors.
                If None, a default sequence is used to parse a JSON property.
                Defaults to None.
        """
        super().__init__({
            0: [(IntegerAcceptor(), 1)],
            1: [(TextAcceptor("."), 2)],
            2: [(digit_acceptor, "$")],
        })

    class Cursor(StateMachine.Cursor):
        """
        Cursor for navigating through the FloatAcceptor.
        Designed for inspectability and debugging purposes.
        """

        def get_value(self) -> Any:
            """
            Get the current parsing value.

            Returns:
                Any: The accumulated text or the parsed number.
            """
            if self.current_state in self.acceptor.end_states and self.accept_history:
                return float("".join([str(cursor.get_value()) for cursor in self.accept_history]))
            return "".join([str(cursor.get_value()) for cursor in self.accept_history])

        def __repr__(self) -> str:
            return f"FloatAcceptor.Cursor(state={self.current_state}, value={self.get_value()})"
