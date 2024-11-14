from __future__ import annotations

from pse.acceptors.basic.text_acceptor import TextAcceptor
from pse.acceptors.token_acceptor import TokenAcceptor
from pse.acceptors.collections.wait_for_acceptor import WaitForAcceptor
from pse.state_machine.state_machine import (
    StateMachine,
    StateMachineGraph,
    StateMachineWalker
)


class EncapsulatedAcceptor(StateMachine):
    """
    Accepts JSON data within a larger text, delimited by specific markers.

    This class encapsulates an acceptor that recognizes JSON content framed by
    specified opening and closing delimiters.
    """

    def __init__(
        self,
        acceptor: TokenAcceptor,
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
        graph: StateMachineGraph = {
            0: [
                (WaitForAcceptor(TextAcceptor(open_delimiter)), 1),
            ],
            1: [
                (acceptor, 2),
            ],
            2: [(TextAcceptor(close_delimiter), "$")],
        }
        self.opening_delimiter: str = open_delimiter
        self.closing_delimiter: str = close_delimiter
        self.wait_for_acceptor: TokenAcceptor = acceptor
        super().__init__(graph, walker_type=EncapsulatedWalker)

class EncapsulatedWalker(StateMachineWalker):

    def is_within_value(self) -> bool:
        return self.current_state == 1
