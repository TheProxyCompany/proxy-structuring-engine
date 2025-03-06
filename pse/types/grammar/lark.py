from __future__ import annotations

import logging

from pse_core import StateId
from pse_core.stepper import Stepper

from pse.types.base.character import CharacterStateMachine, CharacterStepper
from pse.types.grammar import LarkGrammar

logger = logging.getLogger(__name__)

class LarkGrammarStateMachine(CharacterStateMachine):
    def __init__(self, grammar: LarkGrammar):
        super().__init__(char_min=1)
        self.grammar = grammar

    @property
    def delimiters(self) -> tuple[str, str] | None:
        return self.grammar.delimiters

    def get_new_stepper(self, state: StateId | None) -> LarkGrammarStepper:
        """
        Get a new stepper for the grammar.
        """
        return LarkGrammarStepper(self)

    def __str__(self) -> str:
        return self.grammar.name


class LarkGrammarStepper(CharacterStepper):
    """
    A stepper for the grammar state machine.
    """

    def __init__(self, state_machine: LarkGrammarStateMachine):
        """
        Initialize the grammar stepper with a state machine.

        Args:
            state_machine: The grammar state machine that defines the valid transitions
        """
        super().__init__(state_machine)
        self.state_machine: LarkGrammarStateMachine = state_machine

    def get_identifier(self) -> str | None:
        return self.state_machine.grammar.name.lower()

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
        valid_input = self.get_raw_value()
        return self.state_machine.grammar.validate(valid_input, strict=True)

    def consume(self, token: str) -> list[Stepper]:
        """
        Consume the input token and return possible next states.

        Args:
            token: The input string to consume

        Returns:
            A list of new steppers after consuming the token.
            Returns empty list if no valid transitions are possible.
        """
        valid_input, remaining_input = self.get_valid_prefix(token)
        assert valid_input is not None
        return [
            self.step(
                self.get_raw_value() + valid_input,
                remaining_input or None,
            )
        ]

    def get_valid_prefix(self, new_input: str) -> tuple[str | None, str]:
        """
        Get the first prefix of the new input that maintains a valid grammar state.

        Args:
            new_input: The input string to validate

        Returns:
            A tuple of (valid_prefix, remaining_input) where valid_prefix is None if no
            valid prefix exists
        """
        candidate_base = self.get_raw_value()
        max_valid_index = None
        for i in range(1, len(new_input) + 1):
            candidate = candidate_base + new_input[:i]
            if self.state_machine.grammar.validate(candidate, False):
                max_valid_index = i
                break

        if max_valid_index is not None:
            return new_input[:max_valid_index], new_input[max_valid_index:]
        else:
            return None, ""
