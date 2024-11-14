from __future__ import annotations
from typing import Any, Dict, Optional, Type
import logging

from pse.acceptors.collections.sequence_acceptor import SequenceAcceptor
from pse.state_machine.state_machine import (
    StateMachine,
    StateMachineGraph,
    StateMachineWalker,
)
from pse.state_machine.walker import Walker
from pse.acceptors.basic.text_acceptor import TextAcceptor
from pse.acceptors.basic.whitespace_acceptor import WhitespaceAcceptor
from pse.acceptors.json.property_acceptor import PropertyAcceptor

logger = logging.getLogger()


class ObjectAcceptor(StateMachine):
    """
    Accepts a well-formed JSON object and manages state transitions during parsing.

    This acceptor handles the parsing of JSON objects by defining state transitions
    and maintaining the current object properties being parsed.
    """

    def __init__(
        self,
        walker_type: Optional[Type[Walker]] = None,
    ) -> None:
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
                (TextAcceptor("}"), "$"),  # Empty object
            ],
            2: [
                (PropertyAcceptor(), 3),
            ],
            3: [
                (WhitespaceAcceptor(), 4),
            ],
            4: [
                (SequenceAcceptor([TextAcceptor(","), WhitespaceAcceptor()]), 2),
                (TextAcceptor("}"), "$"),  # End of object
            ],
        }
        super().__init__(graph, walker_type=walker_type or ObjectWalker)


class ObjectWalker(StateMachineWalker):
    """
    Walker for ObjectAcceptor that maintains the current state and accumulated key-value pairs.
    """

    def __init__(self, acceptor: ObjectAcceptor, current_state: int = 0) -> None:
        """
        Initialize the ObjectAcceptor.Walker with the parent acceptor and an empty dictionary.

        Args:
            acceptor (ObjectAcceptor): The parent ObjectAcceptor instance.
        """
        super().__init__(acceptor, current_state)
        self.value: Dict[str, Any] = {}

    def should_complete_transition(self) -> bool:
        """
        Handle the completion of a transition by updating the accumulated key-value pairs.

        Returns:
            bool: True if the transition was successful, False otherwise.
        """
        if (
            self.target_state == 3
            and self.transition_walker
            and self.transition_walker.has_reached_accept_state()
        ):
            prop_name, prop_value = self.transition_walker.get_current_value()
            logger.debug(f"🟢 Adding {prop_name}: {prop_value} to {self.value}")
            self.value[prop_name] = prop_value

        return True

    def get_current_value(self) -> Dict[str, Any]:
        """
        Get the current parsed JSON object.

        Returns:
            Dict[str, Any]: The accumulated key-value pairs representing the JSON object.
        """
        if not self.raw_value:
            return {}
        return self.value
