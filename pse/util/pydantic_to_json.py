from typing import Any, Dict, Type, List
from docstring_parser import parse
from pydantic import BaseModel


def pydantic_to_json(model: Type[BaseModel]) -> Dict[str, Any]:
    """
    Convert a Pydantic model class to a standardized schema format.

    Args:
        model (Type[BaseModel]): The Pydantic model class to convert.

    Returns:
        Dict[str, Any]: A dictionary representing the schema of the Pydantic model.

    Raises:
        ValueError: If no description is found in the schema or docstring.
    """
    # Generate the JSON schema from the Pydantic model
    schema = model.model_json_schema()

    # Parse the model's docstring for additional descriptions
    docstring = parse(model.__doc__ or "")
    docstring_params: Dict[str, str] = {
        param.arg_name: param.description
        for param in docstring.params
        if param.description
    }

    # Extract parameters excluding 'title' and 'description'
    parameters = {k: v for k, v in schema.items() if k not in {"title", "description"}}

    properties = parameters.get("properties", {})
    required_fields: List[str] = parameters.get("required", [])

    for field_name, field in model.model_fields.items():
        if field_name in properties:
            field_schema = properties[field_name]
            # Assign description from field or docstring
            field_schema["description"] = field.description or docstring_params.get(
                field_name, ""
            )

            # Append to required fields if the field is mandatory
            if field.is_required():
                required_fields.append(field_name)

            # Update schema with any additional JSON schema properties
            if field.json_schema_extra:
                field_schema.update(field.json_schema_extra)

    # Remove duplicate entries in required fields
    parameters["required"] = list(set(required_fields))

    # Determine the schema description from various sources
    schema_description: str = (
        schema.get("description")
        or docstring.long_description
        or docstring.short_description
        or ""
    )
    if not schema_description:
        raise ValueError("No description found in schema or docstring.")

    return {
        "name": schema.get("title", model.__name__),
        "description": schema_description,
        "parameters": parameters,
    }
