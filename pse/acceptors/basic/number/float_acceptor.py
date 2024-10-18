from __future__ import annotations
from typing import Any

from pse.acceptors.basic.character_acceptors import digit_acceptor
from pse.acceptors.basic.number.integer_acceptor import IntegerAcceptor
from pse.acceptors.basic.sequence_acceptor import SequenceAcceptor
from pse.acceptors.basic.text_acceptor import TextAcceptor

class FloatAcceptor(SequenceAcceptor):
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
        sequence = [IntegerAcceptor(), TextAcceptor("."), digit_acceptor]
        super().__init__(sequence)
        self._accepts_remaining_input = True

    def __repr__(self) -> str:
        return "FloatAcceptor()"

    class Cursor(SequenceAcceptor.Cursor):
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
