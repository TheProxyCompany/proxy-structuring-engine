from __future__ import annotations

import json
import logging
from typing import Any

from pse_core import State
from pse_core.state_machine import StateMachine

from pse.state_machines.basic.string_acceptor import StringAcceptor
from pse.state_machines.basic.text_acceptor import TextAcceptor
from pse.state_machines.basic.whitespace_acceptor import WhitespaceAcceptor
from pse.state_machines.collections.sequence_acceptor import (
    SequenceAcceptor,
    SequenceWalker,
)
from pse.state_machines.json.json_acceptor import JsonAcceptor

logger = logging.getLogger()


class PropertyAcceptor(SequenceAcceptor):
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
            sequence
            or [
                StringAcceptor(),
                WhitespaceAcceptor(),
                TextAcceptor(":"),
                WhitespaceAcceptor(),
                JsonAcceptor(),
            ]
        )

    def get_new_walker(self, state: State | None = None) -> PropertyWalker:
        return PropertyWalker(self, state)


class PropertyWalker(SequenceWalker):
    """
    Walker for PropertyAcceptor that maintains the parsed property name and value.
    """

    def __init__(
        self,
        state_machine: PropertyAcceptor,
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

    def should_complete_transition(self) -> bool:
        """
        Handle the completion of a transition by setting the property name and value.

        Returns:
            bool: True if the transition was successful, False otherwise.
        """
        if (
            not self.transition_walker
            or self.target_state is None
            or not self.transition_walker.get_raw_value()
        ):
            return False

        try:
            if (
                self.target_state == 1
                and (raw_value := self.transition_walker.get_raw_value())
            ):
                self.prop_name = json.loads(raw_value)
            elif (
                self.target_state in self.state_machine.end_states
                and (raw_value := self.transition_walker.get_raw_value())
            ):
                self.prop_value = json.loads(raw_value)
        except Exception:
            return False

        return True

    def is_within_value(self) -> bool:
        """
        Indicates whether the walker is currently parsing a property value.

        Returns:
            bool: True if parsing the property value, False otherwise.
        """
        if self.current_state == 4:
            return super().is_within_value()
        return False

    def get_current_value(self) -> tuple[str, Any]:
        """
        Get the parsed property as a key-value pair.

        Returns:
            Tuple[str, Any]: A tuple containing the property name and its corresponding value.
        """
        if self.prop_name is None:
            return ("", None)
        return (self.prop_name, self.prop_value)

    def can_accept_more_input(self) -> bool:
        if self.transition_walker and self.transition_walker.can_accept_more_input():
            return True

        return self.current_state not in self.state_machine.end_states
