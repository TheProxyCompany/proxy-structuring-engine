from __future__ import annotations

import logging

from pse_core import Edge, StateId
from pse_core.state_machine import StateMachine

from pse.state_machines.base.character import CharacterStateMachine
from pse.state_machines.base.phrase import PhraseStateMachine
from pse.state_machines.composite.chain import ChainStateMachine
from pse.state_machines.types.integer import IntegerStateMachine

logger = logging.getLogger(__name__)


class NumberStateMachine(StateMachine):
    """
    Accepts a well-formed JSON number.

    This state_machine defines the state transitions for parsing JSON numbers, handling integer,
    decimal, and exponential formats as specified by the JSON standard.
    """

    def __init__(self):
        super().__init__(
            {
                0: [
                    (PhraseStateMachine("-", is_optional=True), 1),
                ],
                1: [
                    (IntegerStateMachine(), 2),
                ],
                2: [
                    (
                        ChainStateMachine(
                            [
                                PhraseStateMachine("."),
                                IntegerStateMachine(drop_leading_zeros=False),
                            ],
                        ),
                        3,
                    ),
                ],
                3: [
                    (CharacterStateMachine("eE", char_limit=1), 4),
                ],
                4: [
                    (CharacterStateMachine("+-", char_limit=1), 5),
                ],
                5: [
                    (IntegerStateMachine(), "$"),
                ],
            },
            end_states=[2, 3, "$"],
        )

    def get_edges(self, state: StateId) -> list[Edge]:
        """
        Get the edges for a given state.
        """
        if state == 2:
            return [*super().get_edges(state), *super().get_edges(3)]
        elif state == 4:
            return [*super().get_edges(5), *super().get_edges(state)]
        else:
            return [*super().get_edges(state)]

    def __str__(self) -> str:
        return "Number"
