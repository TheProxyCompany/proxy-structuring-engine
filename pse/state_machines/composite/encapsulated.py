from __future__ import annotations

from typing import Any

from pse_core import State
from pse_core.state_machine import StateMachine
from pse_core.walker import Walker

from pse.state_machines.base.phrase import PhraseStateMachine
from pse.state_machines.composite.wait_for import WaitForStateMachine


class EncapsulatedStateMachine(StateMachine):
    """
    Accepts JSON data within a larger text, delimited by specific markers.

    This class encapsulates an state_machine that recognizes JSON content framed by
    specified opening and closing delimiters. Stores text before the delimiters as
    scratchpad.
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

        if self.inner_state_machine:
            components.append(str(self.inner_state_machine))

        if self.delimiters[1]:
            components.append(f"{self.delimiters[1]!r}")
        return f"Encapsulated({', '.join(components)})"


class EncapsulatedWalker(Walker):

    def __init__(self, state_machine: EncapsulatedStateMachine, state: State | None = None) -> None:
        super().__init__(state_machine, state)
        self.state_machine: EncapsulatedStateMachine = state_machine
        self.scratch_pad = ""
        self.inner_walker: Walker | None = None

    def is_within_value(self) -> bool:
        return (
            self.transition_walker is not None
            and self.transition_walker.state_machine == self.state_machine.inner_state_machine
        )

    def add_to_history(self, walker: Walker) -> None:
        if walker.state_machine == self.state_machine.inner_state_machine:
            self.inner_walker = walker
        elif walker.current_state == 1 and isinstance(walker.get_current_value(), tuple):
            self.scratch_pad += walker.get_current_value()[0]
        return super().add_to_history(walker)

    def get_current_value(self) -> tuple[str, Any]:
        if self.inner_walker:
            return self.scratch_pad, self.inner_walker.get_current_value()
        return self.scratch_pad, None
