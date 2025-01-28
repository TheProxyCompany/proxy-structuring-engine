import enum
import inspect
import logging
from collections.abc import Callable
from typing import Any, get_args, get_origin

from docstring_parser import Docstring, DocstringParam, parse
from pydantic import BaseModel

from pse.structure.from_pydantic import pydantic_to_schema

logger = logging.getLogger(__name__)


def callable_to_schema(function: Callable) -> dict[str, Any]:
    """
    Generate a schema for the specified Python function.

    This takes a callable and parses its signature and docstring,
    and constructs a schema representing the function's parameters.

    Args:
        function (Callable): The Python function to generate a schema for.

    Returns:
        dict[str, Any]: A dictionary representing the JSON schema of the function's parameters.
    """
    sig = inspect.signature(function)
    docstring = parse(function.__doc__ or "No docstring provided")

    schema: dict[str, Any] = {
        "name": function.__name__,
        "description": docstring.description,
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    }

    param_index = 0
    for param in sig.parameters.values():
        if param.name == "self":
            continue  # skip 'self' parameter

        param_docstring = (
            docstring.params[param_index]
            if len(docstring.params) > param_index
            else None
        )
        param_schema = parameter_to_schema(param, param_docstring, docstring)
        schema["parameters"]["properties"][param.name] = param_schema

        if param.default is inspect.Parameter.empty and param_schema.get("nullable", False) is False:
            schema["parameters"]["required"].append(param.name)

        if not schema["parameters"]["required"]:
            del schema["parameters"]["required"]
            schema["parameters"]["nullable"] = True

        param_index += 1
    return schema


def parameter_to_schema(
        param: inspect.Parameter,
        param_docstring: DocstringParam | None,
        docstring: Docstring
) -> dict[str, Any]:
    """
    Generate a schema for a function parameter.

    Args:
        param (inspect.Parameter): The parameter to generate a schema for.
        docstring (Docstring): The docstring for the function.
    """

    parameter_schema: dict[str, Any] = {}
    if param_docstring:
        parameter_schema["description"] = param_docstring.description or docstring.short_description or ""

    if inspect.isclass(param.annotation) and issubclass(param.annotation, BaseModel):
        # Use Pydantic model if the parameter is a BaseModel subclass
        return pydantic_to_schema(param.annotation)
    elif param.annotation == inspect.Parameter.empty:
        logger.warning(f"Parameter '{param.name}' lacks type annotation.")
        parameter_schema["type"] = "any"
        return parameter_schema
    elif param.default is not inspect.Parameter.empty:
        parameter_schema["default"] = str(param.default)

    #######
    parameter_type_schemas = []
    parameter_arguments = get_args(param.annotation)
    for index, argument in enumerate(parameter_arguments or [param.annotation]):
        parameter_type_schema = {}
        parameter_type = get_type(argument)
        match parameter_type:
            case "null":
                parameter_schema["nullable"] = True
            case "array" | "set":
                if index > 0:
                    break
                parameter_type_schema["type"] = parameter_type
                item_type = get_args(argument)[0]
                parameter_type_schema["items"] = {"type": get_type(item_type)}
            case "enum":
                assert issubclass(argument, enum.Enum)
                parameter_type_schema["enum"] = [
                    enum_member.value for enum_member in argument
                ]
            case "dict":
                if index != 1:
                    continue
                parameter_type_schema["type"] = "object"
                parameter_type_schema["additionalProperties"] = {
                    "type": get_type(parameter_arguments[1])
                }
            case _:
                parameter_type_schema["type"] = parameter_type

        if parameter_type_schema:
            parameter_type_schemas.append(parameter_type_schema)

    if len(parameter_type_schemas) > 1:
        parameter_schema["type"] = parameter_type_schemas
    elif parameter_type_schemas:
        parameter_schema.update(**parameter_type_schemas[0])
    else:
        parameter_schema["type"] = "any"

    return parameter_schema

def get_type(python_type: Any) -> str:
    """Map a Python type to a JSON schema type."""
    if python_type is type(None):
        return "null"

    type_name = get_origin(python_type) or python_type
    type_map: dict[type | Any, str] = {
        int: "integer",
        str: "string",
        bool: "boolean",
        float: "number",
        list: "array",
        dict: "object",
        tuple: "array",
        set: "set",
        enum.EnumType: "enum",
        type(None): "null",
        BaseModel: "object",
        Any: "any",
    }
    if type_name not in type_map:
        if type(python_type) in type_map:
            return type_map[type(python_type)]

        logger.warning(f"Unknown type: {python_type}")
        breakpoint()
        return "any"

    return type_map[type_name]
