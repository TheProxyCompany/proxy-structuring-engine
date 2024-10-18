from __future__ import annotations
from typing import Tuple, Optional, Any, List, Dict
import logging

from pse.acceptors.token_acceptor import TokenAcceptor
from pse.state_machine.state_machine import (
    StateMachine,
    StateMachineGraph,
    StateType,
)
from pse.acceptors.collections.sequence_acceptor import SequenceAcceptor
from pse.acceptors.basic.text_acceptor import TextAcceptor
from pse.acceptors.basic.whitespace_acceptor import WhitespaceAcceptor
from pse.acceptors.json.string_acceptor import StringAcceptor
from pse.state_machine.cursor import Cursor
from pse.acceptors.json.json_acceptor import JsonAcceptor

logger = logging.getLogger()
class ObjectAcceptor(StateMachine):
    """
    Accepts a well-formed JSON object and manages state transitions during parsing.

    This acceptor handles the parsing of JSON objects by defining state transitions
    and maintaining the current object properties being parsed.
    """

    def __init__(self) -> None:
        """
        Initialize the ObjectAcceptor with a predefined state transition graph.

        Sets up the state transition graph for parsing JSON objects.
        """
        graph: StateMachineGraph = {
            0: [
                (TextAcceptor("{"), 1),
            ],
            1: [
                (WhitespaceAcceptor(), 2),
            ],
            2: [
                (TextAcceptor("}"), "$"),  # Empty object
                (ObjectAcceptor.PropertyAcceptor(), 3),
            ],
            3: [
                (WhitespaceAcceptor(), 4),
            ],
            4: [
                (TextAcceptor(","), 1),     # Loop back to state 1 for more properties
                (TextAcceptor("}"), "$"),   # End of object
            ],
        }
        super().__init__(graph)

    def expects_more_input(self, cursor: Cursor) -> bool:
        return cursor.current_state not in self.end_states

    class Cursor(StateMachine.Cursor):
        """
        Cursor for ObjectAcceptor that maintains the current state and accumulated key-value pairs.
        """

        def __init__(self, acceptor: ObjectAcceptor) -> None:
            """
            Initialize the ObjectAcceptor.Cursor with the parent acceptor and an empty dictionary.

            Args:
                acceptor (ObjectAcceptor): The parent ObjectAcceptor instance.
            """
            super().__init__(acceptor)
            self.value: Dict[str, Any] = {}

        def complete_transition(
            self, transition_value: Any, target_state: StateType, is_end_state: bool
        ) -> bool:
            """
            Handle the completion of a transition by updating the accumulated key-value pairs.

            Args:
                transition_value (Any): The value transitioned with.
                target_state (StateMachineAcceptor.StateType): The target state after the transition.
                is_end_state (bool): Indicates if the transition leads to an end state.

            Returns:
                bool: True if the transition was successful, False otherwise.
            """
            if self.current_state == 2 and isinstance(transition_value, tuple):
                prop_name, prop_value = transition_value
                self.value[prop_name] = prop_value
            return True

        def get_value(self) -> Dict[str, Any]:
            """
            Get the current parsed JSON object.

            Returns:
                Dict[str, Any]: The accumulated key-value pairs representing the JSON object.
            """
            return self.value

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

        class Cursor(SequenceAcceptor.Cursor):
            """
            Cursor for PropertyAcceptor that maintains the parsed property name and value.
            """

            def __init__(self, acceptor: ObjectAcceptor.PropertyAcceptor) -> None:
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

            def complete_transition(
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

            def get_value(self) -> Tuple[str, Any]:
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

            def is_in_value(self) -> bool:
                """
                Indicates whether the cursor is currently parsing a property value.

                Returns:
                    bool: True if parsing the property value, False otherwise.
                """
                if self.current_state == 4:
                    return super().is_in_value()
                return False

            def __repr__(self) -> str:
                """
                Provide a string representation of the Cursor.

                Returns:
                    str: A string representation of the Cursor.
                """
                value = (
                    self.transition_cursor
                    or "".join(
                        [str(cursor.get_value()) for cursor in self.accept_history]
                    )
                )

                return f"PropertyAcceptor.Cursor({value})"
