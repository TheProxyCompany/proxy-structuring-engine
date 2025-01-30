from collections.abc import Callable
from typing import Any, TypeAlias

from pydantic import BaseModel

SchemaType: TypeAlias = (
    type[BaseModel]
    | list[type[BaseModel]]
    | dict[str, Any]
    | list[dict[str, Any]]
    | Callable[..., Any]
    | str
)
"""
The different object types that can be used as a schema in the structuring engine.
"""


def get_schema_dict(schema: SchemaType) -> dict[str, Any]:
    """
    Convert the given schema into an object that can be used by the engine.
    """
    from pse.structure.from_function import callable_to_schema
    from pse.structure.from_pydantic import pydantic_to_schema

    if isinstance(schema, type) and issubclass(schema, BaseModel):
        return pydantic_to_schema(schema)
    elif callable(schema):
        return callable_to_schema(schema)
    elif isinstance(schema, dict):
        return schema["schema"] if "schema" in schema else schema
    elif isinstance(schema, list):
        schemas = []
        for s in schema:
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

            return json.loads(schema)
        except Exception as e:
            raise ValueError(f"Invalid schema: {schema}") from e
