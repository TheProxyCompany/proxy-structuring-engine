from __future__ import annotations

from typing import Any, Self

from pse_core import StateId
from pse_core.state_machine import StateMachine
from pse_core.stepper import Stepper

from pse.base.phrase import PhraseStateMachine
from pse.base.wait_for import WaitFor
from pse.lark.grammar import Grammar
from pse.lark.grammar.grammar import GrammarStateMachine


class WrappedGrammarStateMachine(StateMachine):


    def __init__(
        self,
        grammar: Grammar,
        delimiters: tuple[str, str],
        min_buffer_length: int = 0,
    ) -> None:
        # Ensure delimiters end with newline to separate from content
        super().__init__(
            {
                0: [
                    (
                        WaitFor(
                            PhraseStateMachine(delimiters[0]),
                            min_buffer_length=min_buffer_length,
                        ),
                        1,
                    ),
                ],
                1: [(GrammarStateMachine(grammar), 2)],
                2: [
                    (
                        WaitFor(
                            PhraseStateMachine(delimiters[1]),
                            min_buffer_length=0,
                        ),
                        "$",
                    )
                ],
            }
        )

    def get_new_stepper(self, state: StateId | None = None) -> WrappedGrammarStepper:
        return WrappedGrammarStepper(self, state)


class WrappedGrammarStepper(Stepper):
    def __init__(
        self, state_machine: WrappedGrammarStateMachine, state: StateId | None = None
    ) -> None:
        super().__init__(state_machine, state)
        self.state_machine: WrappedGrammarStateMachine = state_machine
        self.inner_stepper: Stepper | None = None

    def clone(self) -> Self:
        clone = super().clone()
        if self.inner_stepper:
            clone.inner_stepper = self.inner_stepper.clone()
        return clone

    def is_within_value(self) -> bool:
        if self.current_state == 0 and self.sub_stepper:
            return self.sub_stepper.is_within_value()
        return self.current_state != 0

    def add_to_history(self, stepper: Stepper) -> None:
        if self.current_state == 2:
            self.inner_stepper = stepper
        return super().add_to_history(stepper)

    def get_current_value(self) -> tuple[str, Any]:
        if self.inner_stepper:
            return self.inner_stepper.get_current_value()
        return super().get_current_value()
