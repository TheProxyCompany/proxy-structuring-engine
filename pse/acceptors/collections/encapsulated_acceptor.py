from __future__ import annotations

from pse_core import State

from pse.acceptors.basic.text_acceptor import TextAcceptor
from pse.acceptors.collections.wait_for_acceptor import WaitForAcceptor
from pse.state_machine import HierarchicalStateMachine, StateMachineWalker


class EncapsulatedAcceptor(HierarchicalStateMachine):
    """
    Accepts JSON data within a larger text, delimited by specific markers.

    This class encapsulates an acceptor that recognizes JSON content framed by
    specified opening and closing delimiters.
    """

    def __init__(
        self,
        state_machine: HierarchicalStateMachine,
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
                    (state_machine, 2),
                ],
                2: [(TextAcceptor(close_delimiter), "$")],
            }
        )
        self.opening_delimiter = open_delimiter
        self.closing_delimiter = close_delimiter

    def get_new_walker(self, state: State | None = None) -> EncapsulatedWalker:
        return EncapsulatedWalker(self, state)


class EncapsulatedWalker(StateMachineWalker):
    def is_within_value(self) -> bool:
        return self.current_state == 1
