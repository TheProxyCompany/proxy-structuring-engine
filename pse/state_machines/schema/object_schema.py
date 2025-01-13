from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pse_core import Edge, State
from pse_core.walker import Walker

from pse.state_machines.base.phrase import PhraseStateMachine
from pse.state_machines.composite.chain import ChainStateMachine
from pse.state_machines.schema.property_schema import PropertySchemaStateMachine
from pse.state_machines.types.object import ObjectStateMachine, ObjectWalker
from pse.state_machines.types.whitespace import WhitespaceStateMachine


class ObjectSchemaStateMachine(ObjectStateMachine):
    def __init__(
        self,
        schema: dict[str, Any],
        context: dict[str, Any],
        start_hook: Callable | None = None,
        end_hook: Callable | None = None,
    ):
        super().__init__()
        self.schema = schema
        self.context = context
        self.properties: dict[str, Any] = schema.get("properties", {})
        self.start_hook = start_hook
        self.end_hook = end_hook

        # Determine if additional properties are allowed based on the schema
        self.allow_additional_properties = (
            schema.get("additionalProperties", True) is not False
        )

        # Validate required properties
        self.required_property_names = schema.get("required", [])
        undefined_required_properties = [
            prop for prop in self.required_property_names if prop not in self.properties
        ]
        if undefined_required_properties:
            raise ValueError(
                f"Required properties not defined in schema: {', '.join(undefined_required_properties)}"
            )

    def get_new_walker(self, state: State | None = None) -> ObjectSchemaWalker:
        return ObjectSchemaWalker(self, state)

    def get_edges(self, state: State, value: dict[str, Any]) -> list[Edge]:
        edges = []
        if state == 2:
            for prop_name, prop_schema in self.properties.items():
                if prop_name not in value:
                    edge = (
                        PropertySchemaStateMachine(
                            prop_name,
                            prop_schema,
                            self.context,
                            self.start_hook,
                            self.end_hook,
                        ),
                        3,
                    )
                    edges.append(edge)
        elif state == 4:
            has_all_required_properties = all(
                prop_name in value for prop_name in self.required_property_names
            )
            edges.append(
                (
                    ChainStateMachine(
                        [PhraseStateMachine(","), WhitespaceStateMachine()]
                    ),
                    2,
                )
            )
            if has_all_required_properties:
                edges.append((PhraseStateMachine("}"), "$"))
        else:
            edges.extend(self.state_graph.get(state, []))
        return edges

    def get_transitions(
        self, walker: Walker, state: State | None = None
    ) -> list[tuple[Walker, State, State]]:
        """Retrieve transition walkers from the current state.

        For each edge from the current state, returns walkers that can traverse that edge.
        Handles optional acceptors and pass-through transitions appropriately.

        Args:
            walker: The walker initiating the transition.
            state: Optional starting state. If None, uses the walker's current state.

        Returns:
            list[tuple[Walker, State, State]]: A list of tuples representing transitions.
        """
        current_state = state or walker.current_state
        transitions = []
        for state_machine, target_state in self.get_edges(
            current_state, walker.get_current_value()
        ):
            for transition in state_machine.get_walkers():
                transitions.append((transition, current_state, target_state))

            if (
                state_machine.is_optional
                and target_state not in self.end_states
                and walker.can_accept_more_input()
            ):
                transitions.extend(self.get_transitions(walker, target_state))
        return transitions

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ObjectSchemaStateMachine):
            return other.__eq__(self)

        return super().__eq__(other) and self.schema == other.schema

class ObjectSchemaWalker(ObjectWalker):
    """
    Walker for ObjectAcceptor
    """

    def __init__(
        self,
        state_machine: ObjectSchemaStateMachine,
        current_state: State | None = None,
    ):
        super().__init__(state_machine, current_state)
        self.state_machine: ObjectSchemaStateMachine = state_machine

    def should_start_transition(self, token: str) -> bool:
        if self.target_state == "$":
            has_all_required_properties = all(
                prop_name in self.value
                for prop_name in self.state_machine.required_property_names
            )
            return has_all_required_properties
        # if self.current_state == 2 and self.target_state == 3:
        #     # Check if the property name is already in the object
        #     return token not in self.value
        if self.current_state == 4 and self.target_state == 2:
            # Are all allowed properties already set?
            return len(self.value.keys()) < len(self.state_machine.properties)

        return super().should_start_transition(token)
