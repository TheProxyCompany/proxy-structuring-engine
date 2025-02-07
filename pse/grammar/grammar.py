from __future__ import annotations

import logging

from pse_core import StateId
from pse_core.stepper import Stepper

from pse.base.character import CharacterStateMachine, CharacterStepper
from pse.grammar import Grammar

logger = logging.getLogger(__name__)


class GrammarStateMachine(CharacterStateMachine):
    def __init__(self, grammar: Grammar):
        super().__init__(char_min=1)
        self.grammar = grammar

    def get_new_stepper(self, state: StateId | None) -> GrammarStepper:
        """
        Get a new stepper for the grammar.
        """
        return GrammarStepper(self)

    def __str__(self) -> str:
        return f"Grammar({self.grammar.name})"

    def __repr__(self) -> str:
        return self.__str__()


class GrammarStepper(CharacterStepper):
    def __init__(self, state_machine: GrammarStateMachine):
        """
        Initialize the stepper.
        """
        super().__init__(state_machine)
        self.state_machine: GrammarStateMachine = state_machine

    def should_start_step(self, token: str) -> bool:
        """
        Should the stepper start a new step?
        """
        valid_prefix, _ = self.get_valid_prefix(token)
        return valid_prefix is not None

    def has_reached_accept_state(self) -> bool:
        """
        Has the stepper reached the accept state?
        """
        return (
            super().has_reached_accept_state()
            and self.state_machine.grammar.validate(self.get_raw_value(), strict=True)
        )

    def consume(self, token: str) -> list[Stepper]:
        """
        Consume the token.
        """
        valid_input, remaining_input = self.get_valid_prefix(token)
        if not valid_input:
            return []

        return [
            self.step(
                self.get_raw_value() + valid_input,
                remaining_input or None,
            )
        ]

    def get_valid_prefix(self, new_input: str) -> tuple[str | None, str]:
        """
        Get the valid prefix for the new input.
        """
        cur_input = self.get_raw_value()
        candidate = cur_input

        # Pre-check if the entire input is valid to avoid character-by-character validation
        if self.state_machine.grammar.validate(cur_input + new_input, False):
            return new_input, ""

        # Only do character-by-character if full validation fails
        for i, ch in enumerate(new_input, 1):
            candidate += ch
            if self.state_machine.grammar.validate(candidate, False):
                return new_input[:i], new_input[i:]

        return None, ""
