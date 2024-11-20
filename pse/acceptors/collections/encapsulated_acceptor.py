from __future__ import annotations

from typing import Type
from pse.acceptors.basic.text_acceptor import TextAcceptor
from pse.acceptors.basic.acceptor import Acceptor
from pse.acceptors.collections.wait_for_acceptor import WaitForAcceptor
from pse.core.state_machine import StateMachine, StateMachineWalker
from pse.core.walker import Walker


class EncapsulatedAcceptor(StateMachine):
    """
    Accepts JSON data within a larger text, delimited by specific markers.

    This class encapsulates an acceptor that recognizes JSON content framed by
    specified opening and closing delimiters.
    """

    def __init__(
        self,
        acceptor: Acceptor,
        open_delimiter: str,
        close_delimiter: str,
    ) -> None:
        """
        Initialize the EncapsulatedAcceptor with delimiters and the JSON acceptor.

        Args:
            acceptor: The acceptor responsible for validating the JSON content.
            open_delimiter: The string that denotes the start of the JSON content.
            close_delimiter: The string that denotes the end of the JSON content.
        """
        super().__init__(
            {
                0: [
                    (WaitForAcceptor(TextAcceptor(open_delimiter)), 1),
                ],
                1: [
                    (acceptor, 2),
                ],
                2: [(TextAcceptor(close_delimiter), "$")],
            }
        )
        self.opening_delimiter = open_delimiter
        self.closing_delimiter = close_delimiter
        self.wait_for_acceptor = acceptor

    @property
    def walker_class(self) -> Type[Walker]:
        return EncapsulatedWalker


class EncapsulatedWalker(StateMachineWalker):


    def is_within_value(self) -> bool:
        return self.current_state == 1
