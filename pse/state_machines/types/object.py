from __future__ import annotations

import logging
from typing import Any

from pse_core import State
from pse_core.state_machine import StateMachine
from pse_core.walker import Walker

from pse.state_machines.base.phrase import PhraseStateMachine
from pse.state_machines.composite.chain import ChainStateMachine
from pse.state_machines.types.key_value import KeyValueStateMachine
from pse.state_machines.types.whitespace import WhitespaceStateMachine

logger = logging.getLogger()


class ObjectStateMachine(StateMachine):
    """
    Accepts a well-formed JSON object and manages state transitions during parsing.

    This state_machine handles the parsing of JSON objects by defining state transitions
    and maintaining the current object properties being parsed.
    """

    def __init__(self) -> None:
        """
        Initialize the ObjectAcceptor with a predefined state transition graph.

        Sets up the state transition graph for parsing JSON objects.
        """
        super().__init__(
            {
                0: [
                    (PhraseStateMachine("{"), 1),
                ],
                1: [
                    (WhitespaceStateMachine(), 2),
                    (PhraseStateMachine("}"), "$"),  # Empty object
                ],
                2: [
                    (KeyValueStateMachine(), 3),
                ],
                3: [
                    (WhitespaceStateMachine(), 4),
                ],
                4: [
                    (
                        ChainStateMachine(
                            [PhraseStateMachine(","), WhitespaceStateMachine()]
                        ),
                        2,
                    ),
                    (PhraseStateMachine("}"), "$"),  # End of object
                ],
            }
        )

    def get_new_walker(self, state: State | None = None) -> ObjectWalker:
        return ObjectWalker(self, state)

    def __str__(self) -> str:
        return "Object"


class ObjectWalker(Walker):
    """
    Walker for ObjectAcceptor that maintains the current state and accumulated key-value pairs.
    """

    def __init__(
        self, state_machine: ObjectStateMachine, current_state: State | None = None
    ) -> None:
        """
        Initialize the ObjectAcceptor.Walker with the parent state_machine and an empty dictionary.

        Args:
            state_machine (ObjectAcceptor): The parent ObjectAcceptor instance.
        """
        super().__init__(state_machine, current_state)
        self.value: dict[str, Any] = {}

    def clone(self) -> ObjectWalker:
        cloned_walker = super().clone()
        cloned_walker.value = self.value.copy()
        return cloned_walker

    def is_within_value(self) -> bool:
        return self.current_state == 3

    def add_to_history(self, walker: Walker) -> None:
        if self.is_within_value():
            prop_name, prop_value = walker.get_current_value()
            logger.debug(f"🟢 Adding {prop_name}: {prop_value} to {self.value}")
            self.value[prop_name] = prop_value
        super().add_to_history(walker)

    def get_current_value(self) -> dict[str, Any]:
        """
        Get the current parsed JSON object.

        Returns:
            dict[str, Any]: The accumulated key-value pairs representing the JSON object.
        """
        if not self.get_raw_value():
            return {}
        return self.value
