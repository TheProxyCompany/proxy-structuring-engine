from __future__ import annotations
from typing import Any, Dict, Type
import logging

from pse.state_machine.state_machine import (
    StateMachine,
    StateMachineGraph,
    StateType,
    StateMachineWalker,
)
from pse.acceptors.basic.text_acceptor import TextAcceptor
from pse.acceptors.basic.whitespace_acceptor import WhitespaceAcceptor
from pse.acceptors.json.property_acceptor import PropertyAcceptor
from pse.state_machine.walker import Walker

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
                (PropertyAcceptor(), 3),
            ],
            3: [
                (WhitespaceAcceptor(), 4),
            ],
            4: [
                (TextAcceptor(","), 1),  # Loop back to state 1 for more properties
                (TextAcceptor("}"), "$"),  # End of object
            ],
        }
        super().__init__(graph)

    def expects_more_input(self, walker: Walker) -> bool:
        return walker.current_state not in self.end_states

    @property
    def walker_class(self) -> Type[Objectwalker]:
        return Objectwalker


class Objectwalker(StateMachineWalker):
    """
    Walker for ObjectAcceptor that maintains the current state and accumulated key-value pairs.
    """

    def __init__(self, acceptor: ObjectAcceptor) -> None:
        """
        Initialize the ObjectAcceptor.Walker with the parent acceptor and an empty dictionary.

        Args:
            acceptor (ObjectAcceptor): The parent ObjectAcceptor instance.
        """
        super().__init__(acceptor)
        self.value: Dict[str, Any] = {}

    def should_complete_transition(
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

    def accumulated_value(self) -> Dict[str, Any]:
        """
        Get the current parsed JSON object.

        Returns:
            Dict[str, Any]: The accumulated key-value pairs representing the JSON object.
        """
        return self.value
