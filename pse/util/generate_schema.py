import inspect
from collections.abc import Callable
from typing import Any, get_args, get_origin

from docstring_parser import parse
from pydantic import BaseModel


def pydantic_to_json_schema(model: type[BaseModel]) -> dict[str, Any]:
    """
    Convert a Pydantic model class to a standardized schema format.

    Returns:
        Dict[str, Any]: A dictionary representing the schema of the Pydantic model.
    """
    # Generate the JSON schema from the Pydantic model
    # Get base schema from Pydantic model
    schema = model.model_json_schema()

    # Extract docstring info
    docstring = parse(model.__doc__ or "")
    docstring_params = {
        param.arg_name: param.description
        for param in docstring.params
        if param.description
    }

    # Get description from schema or docstring
    description = (
        schema.get("description")
        or docstring.long_description
        or docstring.short_description
        or ""
    )

    # Extract parameters, excluding metadata fields
    parameters = {k: v for k, v in schema.items() if k not in {"title", "description"}}

    # Process properties and required fields
    properties = parameters.get("properties", {})
    required_fields = set(parameters.get("required", []))

    assert isinstance(properties, dict)

    # Update field schemas with descriptions and requirements
    for field_name, field in model.model_fields.items():
        if field_name not in properties:
            continue

        field_schema = properties[field_name]

        # Add field to required list if needed
        if field.is_required():
            required_fields.add(field_name)

        # Set description from field or docstring
        field_schema["description"] = field.description or docstring_params.get(
            field_name, ""
        )

        # Add any extra schema properties
        if field.json_schema_extra:
            field_schema.update(field.json_schema_extra)

    parameters["required"] = list(required_fields)

    return {
        "name": schema.get("title", model.__name__),
        "description": description,
        "parameters": parameters,
    }


def callable_to_json_schema(
    function: Callable, name: str | None = None, description: str | None = None
) -> dict[str, Any]:
    """
    Generate a JSON schema for the specified Python function.

    This function introspects the given function, parses its signature and docstring,
    and constructs a JSON schema representing the function's parameters. It supports
    both standard types and Pydantic models.

    Args:
        function (Callable): The Python function to generate a schema for.
        name (Optional[str]): Custom name for the schema. Defaults to the function's name if not provided.
        description (Optional[str]): Custom description for the schema. Defaults to the function's docstring
            if not provided.

    Returns:
        Dict[str, Any]: A dictionary representing the JSON schema of the function's parameters.
    """
    sig = inspect.signature(function)
    docstring = parse(function.__doc__ or "No docstring provided")

    schema: dict[str, Any] = {
        "name": name or function.__name__,
        "description": description
        or docstring.long_description
        or "No description provided.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    }

    for param in sig.parameters.values():
        if param.name == "self":
            continue  # Skip 'self' parameter in methods

        # Ensure the parameter has a type annotation
        if param.annotation == inspect.Parameter.empty:
            raise TypeError(
                f"Parameter '{param.name}' in '{function.__name__}' lacks type annotation."
            )

        # Retrieve the parameter's docstring description
        param_doc = next(
            (d for d in docstring.params if d.arg_name == param.name), None
        )
        if not param_doc or not param_doc.description:
            raise ValueError(
                f"Parameter '{param.name}' in '{function.__name__}' lacks docstring description."
            )

        # Determine the JSON schema for the parameter
        if inspect.isclass(param.annotation) and issubclass(param.annotation, BaseModel):
            # Use Pydantic model schema if the parameter is a BaseModel subclass
            param_schema = pydantic_to_json_schema(param.annotation)
        else:
            # Handle Optional types
            origin = get_origin(param.annotation)
            args = get_args(param.annotation)
            actual_types = [argument for argument in args if argument is not type(None)]
            if args:
                param_schema = {
                    "type": [_get_json_type(t) for t in actual_types],
                    "description": param_doc.description,
                }
            elif origin in [list, set, tuple]:
                param_schema = {
                    "type": "array",
                    "items": {"type": _get_json_type(args[0]) if args else "any"},
                    "description": param_doc.description,
                }
            elif origin is dict:
                param_schema = {
                    "type": "object",
                    "additionalProperties": {
                        "type": _get_json_type(args[1]) if len(args) > 1 else "any"
                    },
                    "description": param_doc.description,
                }
            else:
                param_schema = {
                    "type": _get_json_type(param.annotation),
                    "description": param_doc.description,
                }

        # Include default values in the schema
        if param.default != inspect.Parameter.empty:
            param_schema["default"] = param.default

        schema["parameters"]["properties"][param.name] = param_schema

        # Append to required fields if no default value is provided
        if param.default == inspect.Parameter.empty:
            schema["parameters"]["required"].append(param.name)

    return schema


def _get_json_type(py_type: Any) -> str:
    """Map a Python type to a JSON schema type."""

    type_map = {
        int: "integer",
        str: "string",
        bool: "boolean",
        float: "number",
        list: "array",
        dict: "object",
        tuple: "array",
        set: "set",
        type(None): "null",
    }

    type_name = get_origin(py_type) or py_type
    return type_map.get(type_name, "unknown")
