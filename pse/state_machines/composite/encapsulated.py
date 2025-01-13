from __future__ import annotations

from pse_core import State
from pse_core.state_machine import StateMachine
from pse_core.walker import Walker

from pse.state_machines.base.phrase import PhraseStateMachine
from pse.state_machines.composite.wait_for import WaitForStateMachine


class EncapsulatedStateMachine(StateMachine):
    """
    Accepts JSON data within a larger text, delimited by specific markers.

    This class encapsulates an state_machine that recognizes JSON content framed by
    specified opening and closing delimiters.
    """

    def __init__(
        self,
        state_machine: StateMachine,
        delimiters: tuple[str, str],
    ) -> None:
        """
        Initialize the EncapsulatedAcceptor with delimiters and the JSON state_machine.

        Args:
            state_machine: The state_machine responsible for validating the JSON content.
            open_delimiter: The string that denotes the start of the JSON content.
            close_delimiter: The string that denotes the end of the JSON content.
        """
        super().__init__(
            {
                0: [
                    (WaitForStateMachine(PhraseStateMachine(delimiters[0])), 1),
                ],
                1: [
                    (state_machine, 2),
                ],
                2: [(PhraseStateMachine(delimiters[1]), "$")],
            }
        )
        self.inner_state_machine = state_machine
        self.delimiters = delimiters

    def get_new_walker(self, state: State | None = None) -> EncapsulatedWalker:
        return EncapsulatedWalker(self, state)

    def __str__(self) -> str:
        components = []
        if self.delimiters[0]:
            components.append(f"{self.delimiters[0]!r}")
        if self.delimiters[1]:
            components.append(f"{self.delimiters[1]!r}")
        if self.inner_state_machine:
            components.append(str(self.inner_state_machine))
        return f"Encapsulated({', '.join(components)})"


class EncapsulatedWalker(Walker):
    def is_within_value(self) -> bool:
        return self.current_state != 0

    def accepts_any_token(self) -> bool:
        """
        Indicates that this state_machine matches all characters until a trigger is found.

        Returns:
            bool: Always True.
        """
        if not self.is_within_value():
            return True

        if self.transition_walker:
            return self.transition_walker.accepts_any_token()

        return False
