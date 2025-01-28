import logging
from typing import Any

from docstring_parser import parse
from pydantic import BaseModel

logger = logging.getLogger(__name__)


def pydantic_to_schema(model: type[BaseModel]) -> dict[str, Any]:
    """
    Convert a Pydantic model class to a standardized schema format.

    Returns:
        dict[str, Any]: A dictionary representing the schema.
    """
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
