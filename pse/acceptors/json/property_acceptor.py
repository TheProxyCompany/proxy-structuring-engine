from __future__ import annotations
from typing import Tuple, Optional, Any, List, Type
import logging

from pse.acceptors.token_acceptor import TokenAcceptor
from pse.acceptors.collections.sequence_acceptor import SequenceAcceptor, SequenceWalker
from pse.acceptors.basic.text_acceptor import TextAcceptor
from pse.acceptors.basic.whitespace_acceptor import WhitespaceAcceptor
from pse.acceptors.json.string_acceptor import StringAcceptor
from pse.acceptors.json.json_acceptor import JsonAcceptor

logger = logging.getLogger()


class PropertyAcceptor(SequenceAcceptor):
    """
    Acceptor for individual properties within a JSON object.

    This acceptor defines the sequence of token acceptors required to parse a property
    key-value pair in a JSON object.
    """

    def __init__(self, sequence: Optional[List[TokenAcceptor]] = None) -> None:
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
        super().__init__(sequence)

    def __repr__(self) -> str:
        return f"PropertyAcceptor({self.acceptors})"

    @property
    def walker_class(self) -> Type[Propertywalker]:
        return Propertywalker


class Propertywalker(SequenceWalker):
    """
    Walker for PropertyAcceptor that maintains the parsed property name and value.
    """

    def __init__(self, acceptor: PropertyAcceptor) -> None:
        """
        Initialize the PropertyAcceptor

        Args:
            acceptor (PropertyAcceptor): The parent PropertyAcceptor
        """
        super().__init__(acceptor)
        self.prop_name: Optional[str] = None
        self.prop_value: Optional[Any] = None
        self._accepts_remaining_input = True

    @property
    def can_handle_remaining_input(self) -> bool:
        return self._accepts_remaining_input

    def should_complete_transition(
        self, transition_value: Any, target_state: Any, is_end_state: bool
    ) -> bool:
        """
        Handle the completion of a transition by setting the property name and value.

        Args:
            transition_value (Any): The value transitioned with.
            target_state (Any): The target state after transition.
            is_end_state (bool): Indicates if the transition leads to an end state.

        Returns:
            bool: True if the transition was successful, False otherwise.
        """
        if target_state == 1:
            self.prop_name = transition_value
        elif is_end_state:
            self.prop_value = transition_value
        return True

    def accumulated_value(self) -> Tuple[str, Any]:
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

    def is_within_value(self) -> bool:
        """
        Indicates whether the walker is currently parsing a property value.

        Returns:
            bool: True if parsing the property value, False otherwise.
        """
        if self.current_state == 4:
            return super().is_within_value()
        return False

    def __repr__(self) -> str:
        """
        Provide a string representation of the Walker.

        Returns:
            str: A string representation of the Walker.
        """
        value = self.transition_walker or "".join(
            [str(walker.accumulated_value()) for walker in self.accept_history]
        )

        return f"PropertyAcceptor.Walker({value})"
