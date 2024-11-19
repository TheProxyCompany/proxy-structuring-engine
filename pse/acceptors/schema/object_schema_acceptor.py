from __future__ import annotations
from typing import Dict, Any, Callable, Type
from pse.acceptors.json.object_acceptor import ObjectAcceptor, ObjectWalker
from pse.core.walker import Walker
from pse.util.errors import InvalidSchemaError
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
        # Note that, according to the JSON schema specification, additional properties
        # should be allowed by default.
        if "additionalProperties" in schema:
            if schema["additionalProperties"] is False:
                self.allow_additional_properties = False
            else:
                # Implement handling for additionalProperties schema if necessary
                self.allow_additional_properties = True
        else:
            self.allow_additional_properties = True  # Default behavior per JSON Schema
        self.required_property_names = schema.get("required", [])
        for required_property_name in self.required_property_names:
            if required_property_name not in self.properties:
                raise InvalidSchemaError(
                    f"Required property '{required_property_name}' not defined"
                )

        assert self.properties is not None

    @property
    def walker_class(self) -> Type[Walker]:
        return ObjectSchemaWalker

    def get_edges(self, state):
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
            ]
        else:
            return super().get_edges(state)


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
