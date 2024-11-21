from __future__ import annotations
from typing import Dict, Any, Callable, Type, Optional, Iterable, Tuple
from pse.acceptors.json.object_acceptor import ObjectAcceptor, ObjectWalker
from pse.core.walker import Walker
from pse.util.errors import InvalidSchemaError
from pse.util.state_machine.types import State
from pse.acceptors.schema.property_schema_acceptor import PropertySchemaAcceptor

class ObjectSchemaAcceptor(ObjectAcceptor):

    def __init__(
        self,
        schema: Dict[str, Any],
        context: Dict[str, Any],
        start_hook: Callable | None = None,
        end_hook: Callable | None = None,
    ):
        super().__init__()
        self.schema = schema
        self.context = context
        self.properties: Dict[str, Any] = schema.get("properties", {})
        self.start_hook = start_hook
        self.end_hook = end_hook

        # Determine if additional properties are allowed based on the schema
        self.allow_additional_properties = schema.get("additionalProperties", True) is not False

        # Validate required properties
        self.required_property_names = schema.get("required", [])
        undefined_required_properties = [
            prop
            for prop in self.required_property_names
            if prop not in self.properties
        ]
        if undefined_required_properties:
            raise InvalidSchemaError(
                f"Required properties not defined in schema: {', '.join(undefined_required_properties)}"
            )

    @property
    def walker_class(self) -> Type[Walker]:
        return ObjectSchemaWalker

    def get_edges(self, state, value: Dict[str, Any] = {}):
        if state == 2:
            return [
                (
                    PropertySchemaAcceptor(
                        prop_name,
                        prop_schema,
                        self.context,
                        self.start_hook,
                        self.end_hook,
                    ),
                    3,
                )
                for prop_name, prop_schema in self.properties.items()
                if prop_name not in value
            ]
        else:
            return super().get_edges(state)

    def get_transitions(
        self, walker: Walker, state: Optional[State] = None
    ) -> Iterable[Tuple[Walker, State, State]]:
        """Retrieve transition walkers from the current state.

        For each edge from the current state, yields walkers that can traverse that edge.
        Handles optional acceptors and pass-through transitions appropriately.

        Args:
            walker: The walker initiating the transition.
            state: Optional starting state. If None, uses the walker's current state.

        Returns:
            Iterable of tuples (transition_walker, source_state, target_state).
        """
        current_state = state or walker.current_state
        for acceptor, target_state in self.get_edges(current_state, walker.current_value):
            for transition in acceptor.get_walkers():
                yield transition, current_state, target_state

            if (
                acceptor.is_optional
                and target_state not in self.end_states
                and walker.can_accept_more_input()
            ):
                yield from self.get_transitions(walker, target_state)


class ObjectSchemaWalker(ObjectWalker):
    """
    Walker for ObjectAcceptor
    """

    def __init__(self, acceptor: ObjectSchemaAcceptor, current_state: int = 0):
        super().__init__(acceptor, current_state)
        self.acceptor = acceptor

    def should_start_transition(self, token: str) -> bool:
        if self.target_state == "$":
            return all(
                prop_name in self.value
                for prop_name in self.acceptor.required_property_names
            )
        if self.current_state == 2 and self.target_state == 3:
            # Check if the property name is already in the object
            return token not in self.value
        if self.current_state == 4 and self.target_state == 2:
            # Are all allowed properties already set?
            return len(self.value.keys()) < len(self.acceptor.properties)

        return super().should_start_transition(token)
