from __future__ import annotations

import json
import logging
from typing import Any

from pse_core import State
from pse_core.state_machine import StateMachine
from pse_core.walker import Walker

from pse.state_machines.base.phrase import PhraseStateMachine
from pse.state_machines.composite.chain import ChainStateMachine
from pse.state_machines.types.json import JsonStateMachine
from pse.state_machines.types.string import StringStateMachine
from pse.state_machines.types.whitespace import WhitespaceStateMachine

logger = logging.getLogger()


class KeyValueStateMachine(ChainStateMachine):
    """
    Acceptor for individual properties within a JSON object.

    This state_machine defines the sequence of token acceptors required to parse a property
    key-value pair in a JSON object.
    """

    def __init__(self, sequence: list[StateMachine] | None = None) -> None:
        """
        Initialize the PropertyAcceptor with a predefined sequence of token acceptors.
        """
        super().__init__(
            sequence or [
                StringStateMachine(),
                WhitespaceStateMachine(),
                PhraseStateMachine(":"),
                WhitespaceStateMachine(),
                JsonStateMachine(),
            ]
        )

    def get_new_walker(self, state: State | None = None) -> KeyValueWalker:
        return KeyValueWalker(self, state)

    def __str__(self) -> str:
        return "KeyValue"


class KeyValueWalker(Walker):
    """
    Walker for PropertyAcceptor that maintains the parsed property name and value.
    """

    def __init__(
        self,
        state_machine: KeyValueStateMachine,
        current_acceptor_index: State | None = None,
    ) -> None:
        """
        Initialize the PropertyAcceptor

        Args:
            state_machine (PropertyAcceptor): The parent PropertyAcceptor
        """
        super().__init__(state_machine, current_acceptor_index)
        self.prop_name = ""
        self.prop_value: Any | None = None

    def clone(self) -> KeyValueWalker:
        cloned_walker = super().clone()
        cloned_walker.prop_name = self.prop_name
        cloned_walker.prop_value = self.prop_value
        return cloned_walker

    def should_complete_transition(self) -> bool:
        """
        Handle the completion of a transition by setting the property name and value.

        Returns:
            bool: True if the transition was successful, False otherwise.
        """
        if not super().should_complete_transition() or not self.transition_walker:
            return False

        try:
            if self.target_state == 1:
                self.prop_name = json.loads(self.transition_walker.get_raw_value())
            elif self.target_state in self.state_machine.end_states:
                self.prop_value = json.loads(self.transition_walker.get_raw_value())
        except Exception:
            return False

        return True

    def get_current_value(self) -> tuple[str, Any]:
        """
        Get the parsed property as a key-value pair.

        Returns:
            Tuple[str, Any]: A tuple containing the property name and its corresponding value.
        """
        if self.prop_name is None:
            return ("", None)
        return (self.prop_name, self.prop_value)
