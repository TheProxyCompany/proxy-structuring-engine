from __future__ import annotations

import logging

from pse_core import StateId
from pse_core.stepper import Stepper

from pse.base.any_character import AnyCharacterStateMachine, AnyCharacterStepper
from pse.lark.grammar import Grammar

logger = logging.getLogger(__name__)


class GrammarStateMachine(AnyCharacterStateMachine):
    def __init__(self, grammar: Grammar):
        super().__init__(char_min=1)
        self.parser = grammar.lark_grammar
        self.validator = grammar.validator_function

    def get_new_stepper(self, state: StateId | None) -> GrammarStepper:
        return GrammarStepper(self)


class GrammarStepper(AnyCharacterStepper):
    def __init__(self, state_machine: GrammarStateMachine):
        super().__init__(state_machine)
        self.state_machine: GrammarStateMachine = state_machine

    def should_start_step(self, token: str) -> bool:
        valid_prefix, _ = self.get_valid_prefix(token)
        return valid_prefix is not None

    def has_reached_accept_state(self) -> bool:
        return super().has_reached_accept_state() and self.validate_input(
            self.get_raw_value(), strict=True
        )

    def consume(self, token: str) -> list[Stepper]:
        valid_input, remaining_input = self.get_valid_prefix(token)
        if not valid_input:
            return []
        new_stepper = self.step(
            self.get_raw_value() + valid_input,
            remaining_input or None,
        )
        return [new_stepper]

    def get_valid_prefix(self, new_input: str) -> tuple[str | None, str]:
        """
        Identify the smallest valid prefix of the new_input.
        """
        cur_input = self.get_raw_value()
        # find first valid prefix
        for i in range(1, len(new_input) + 1):
            candidate = cur_input + new_input[:i]
            if self.validate_input(candidate, strict=False):
                return new_input[:i], new_input[i:]

        return None, ""

    def validate_input(
        self, input: str, strict: bool = False, start: str = "file_input"
    ) -> bool:
        return self.state_machine.validator(input, strict, start)
