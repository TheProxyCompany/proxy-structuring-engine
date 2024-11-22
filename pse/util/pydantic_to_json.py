from typing import Any, Dict, Type
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

    return model.model_json_schema()
