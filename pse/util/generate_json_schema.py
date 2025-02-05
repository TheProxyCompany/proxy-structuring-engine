from collections.abc import Callable
from typing import Any, TypeAlias

from pydantic import BaseModel

SchemaDefinition: TypeAlias = (
    type[BaseModel] | dict[str, Any] | Callable[..., Any] | str
)
SchemaType: TypeAlias = SchemaDefinition | list[SchemaDefinition]
"""
The different object types that can be used as a schema in the structuring engine.
"""


def generate_json_schema(source: SchemaType) -> dict[str, Any]:
    """
    Convert the given schema into an object that can be used by the engine.

    Args:
        schema: The schema to convert.
        Can be a Pydantic model, a callable, a dictionary, or a list of any of the above.

    Returns:
        The converted schema.
    """
    from pse.structure.from_function import callable_to_schema
    from pse.structure.from_pydantic import pydantic_to_schema

    if isinstance(source, type) and issubclass(source, BaseModel):
        return pydantic_to_schema(source)
    elif callable(source):
        return callable_to_schema(source)
    elif isinstance(source, dict):
        return source["schema"] if "schema" in source else source
    elif isinstance(source, list):
        schemas = []
        for s in source:
            if isinstance(s, type) and issubclass(s, BaseModel):
                schemas.append(pydantic_to_schema(s))
            elif isinstance(s, dict):
                schemas.append(s)
            else:
                raise ValueError(f"Invalid schema: {s}")
        return {"oneOf": schemas}
    else:
        try:
            import json

            return json.loads(source)
        except Exception as e:
            raise ValueError(f"Invalid schema: {source}") from e
