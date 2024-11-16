from __future__ import annotations
from typing import Tuple, Optional, Any, Type
import logging
import json

from pse.acceptors.basic.acceptor import Acceptor

from pse.acceptors.collections.sequence_acceptor import SequenceAcceptor, SequenceWalker
from pse.acceptors.basic.text_acceptor import TextAcceptor
from pse.acceptors.basic.whitespace_acceptor import WhitespaceAcceptor
from pse.acceptors.basic.string_acceptor import StringAcceptor
from pse.acceptors.json.json_acceptor import JsonAcceptor
from pse.core.walker import Walker

logger = logging.getLogger()


class PropertyAcceptor(SequenceAcceptor):
    """
    Acceptor for individual properties within a JSON object.

    This acceptor defines the sequence of token acceptors required to parse a property
    key-value pair in a JSON object.
    """

    def __init__(self, sequence: Optional[list[Acceptor]] = None) -> None:
        """
        Initialize the PropertyAcceptor with a predefined sequence of token acceptors.
        """

        super().__init__(
            sequence or [
                StringAcceptor(),
                WhitespaceAcceptor(),
                TextAcceptor(":"),
                WhitespaceAcceptor(),
                JsonAcceptor(),
            ]
        )

    @property
    def walker_class(self) -> Type[Walker]:
        return PropertyWalker


class PropertyWalker(SequenceWalker):
    """
    Walker for PropertyAcceptor that maintains the parsed property name and value.
    """

    def __init__(
        self,
        acceptor: PropertyAcceptor,
        current_acceptor_index: int = 0,
    ) -> None:
        """
        Initialize the PropertyAcceptor

        Args:
            acceptor (PropertyAcceptor): The parent PropertyAcceptor
        """
        super().__init__(acceptor, current_acceptor_index)
        self.prop_name = ""
        self.prop_value: Optional[Any] = None

    def should_complete_transition(self) -> bool:
        """
        Handle the completion of a transition by setting the property name and value.

        Returns:
            bool: True if the transition was successful, False otherwise.
        """
        if (
            not self.transition_walker
            or self.target_state is None
            or not self.transition_walker.raw_value
        ):
            return False

        try:
            if self.target_state == 1:
                self.prop_name = json.loads(self.transition_walker.raw_value)
            elif self.target_state in self.acceptor.end_states:
                self.prop_value = json.loads(self.transition_walker.raw_value)
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

    @property
    def current_value(self) -> Tuple[str, Any]:
        """
        Get the parsed property as a key-value pair.

        Returns:
            Tuple[str, Any]: A tuple containing the property name and its corresponding value.
        """
        if self.prop_name is None:
            return ("", None)
        return (self.prop_name, self.prop_value)
