from __future__ import annotations
import json
from typing import Tuple, Optional, Any, List, Type
import logging

from pse.acceptors.token_acceptor import TokenAcceptor
from pse.acceptors.collections.sequence_acceptor import SequenceAcceptor, SequenceWalker
from pse.acceptors.basic.text_acceptor import TextAcceptor
from pse.acceptors.basic.whitespace_acceptor import WhitespaceAcceptor
from pse.acceptors.json.string_acceptor import StringAcceptor
from pse.acceptors.json.json_acceptor import JsonAcceptor
from pse.state_machine.walker import Walker

logger = logging.getLogger()


class PropertyAcceptor(SequenceAcceptor):
    """
    Acceptor for individual properties within a JSON object.

    This acceptor defines the sequence of token acceptors required to parse a property
    key-value pair in a JSON object.
    """

    def __init__(
        self,
        sequence: Optional[List[TokenAcceptor]] = None,
        walker_type: Optional[Type[Walker]] = None,
    ) -> None:
        """
        Initialize the PropertyAcceptor with a predefined sequence of token acceptors.

        Args:
            sequence (Optional[List[TokenAcceptor]], optional): Custom sequence of acceptors.
                If None, a default sequence is used to parse a JSON property.
                Defaults to None.
        """
        if sequence is None:
            sequence = [
                StringAcceptor(),
                WhitespaceAcceptor(),
                TextAcceptor(":"),
                WhitespaceAcceptor(),
                JsonAcceptor({}),
            ]
        super().__init__(
            sequence,
            walker_type=walker_type or PropertyWalker,
        )


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

    def get_current_value(self) -> Tuple[str, Any]:
        """
        Get the parsed property as a key-value pair.

        Returns:
            Tuple[str, Any]: A tuple containing the property name and its corresponding value.

        Raises:
            JSONParsingError: If the property name is missing.
        """
        if self.prop_name is None:
            return ("", None)
        return (self.prop_name, self.prop_value)

    @property
    def raw_value(self) -> Optional[str]:
        if not self.prop_name:
            return None

        if not self.prop_value:
            return f"{self.prop_name}:"
        return f"{self.prop_name}: {self.prop_value}"
