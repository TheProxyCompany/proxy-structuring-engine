from __future__ import annotations
from typing import Iterable

from pse.acceptors.basic.text_acceptor import TextAcceptor
from pse.acceptors.token_acceptor import TokenAcceptor
from pse.acceptors.collections.wait_for_acceptor import WaitForAcceptor
from pse.state_machine.walker import Walker
from pse.state_machine.state_machine import (
    StateMachine,
    StateMachineGraph,
    StateMachineWalker,
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
        super().__init__(graph)

    def expects_more_input(self, walker: Walker) -> bool:
        return walker.current_state not in self.end_states

    def get_walkers(self) -> Iterable[EncapsulatedWalker]:
        yield EncapsulatedWalker(self)


class EncapsulatedWalker(StateMachineWalker):
    """
    Walker for the EncapsulatedAcceptor.
    """

    def __init__(self, acceptor: EncapsulatedAcceptor) -> None:
        """
        Initialize the Walker for EncapsulatedAcceptor.

        Args:
            acceptor: The EncapsulatedAcceptor instance this walker belongs to.
        """
        super().__init__(acceptor)
        self.acceptor: EncapsulatedAcceptor = acceptor

    def is_within_value(self) -> bool:
        """
        Determine if the walker is currently within a value.

        Returns:
            bool: True if in a value, False otherwise.
        """
        return self.current_state != self.acceptor.initial_state or (
            self.transition_walker is not None
            and self.transition_walker.is_within_value()
        )
