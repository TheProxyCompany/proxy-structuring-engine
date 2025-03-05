"""Tests for the from_pydantic module."""
from pydantic import BaseModel, Field

from pse.types.json.schema_sources.from_pydantic import pydantic_to_schema


class SimpleModel(BaseModel):
    """A simple model with basic fields.

    This model contains string and integer fields.
    """

    name: str = Field(description="The name field")
    age: int = Field(description="The age field")


class ModelWithDefaults(BaseModel):
    """A model with default values.

    Args:
        name: A name field with default
        age: An age field that's required
    """

    name: str = Field(default="John", description="The name field")
    age: int = Field(description="The age field")


class ComplexModel(BaseModel):
    """A complex model with nested types.

    Args:
        name: The user's name
        age: The user's age
        tags: A list of string tags
        address: An optional address
    """

    name: str
    age: int
    tags: list[str] = Field(default_factory=list)
    address: str | None = None


class NestedModel(BaseModel):
    """A model with nested models.

    Args:
        title: The title
        user: A user object
    """

    title: str
    user: SimpleModel


class ModelWithoutDocs(BaseModel):
    name: str
    age: int


class ModelWithJsonExtra(BaseModel):
    """A model with json_schema_extra in fields."""

    name: str = Field(
        description="The name field",
        json_schema_extra={"example": "John Doe", "pattern": "^[a-zA-Z ]+$"}
    )
    age: int = Field(
        description="The age field",
        json_schema_extra={"minimum": 0, "maximum": 120}
    )


def test_pydantic_to_schema_simple():
    """Test pydantic_to_schema with a simple model."""
    schema = pydantic_to_schema(SimpleModel)

    assert schema["name"] == "SimpleModel"
    assert "description" in schema
    assert "properties" in schema
    assert "name" in schema["properties"]
    assert "age" in schema["properties"]
    assert schema["properties"]["name"]["description"] == "The name field"
    assert schema["properties"]["age"]["description"] == "The age field"
    assert "required" in schema
    assert "name" in schema["required"]
    assert "age" in schema["required"]


def test_pydantic_to_schema_with_defaults():
    """Test pydantic_to_schema with a model having default values."""
    schema = pydantic_to_schema(ModelWithDefaults)

    assert "properties" in schema
    assert "name" in schema["properties"]
    assert "age" in schema["properties"]
    assert "default" in schema["properties"]["name"]
    assert schema["properties"]["name"]["default"] == "John"
    assert "required" in schema
    assert "age" in schema["required"]
    assert "name" not in schema["required"]


def test_pydantic_to_schema_complex():
    """Test pydantic_to_schema with a model having complex types."""
    schema = pydantic_to_schema(ComplexModel)

    assert "properties" in schema
    assert "name" in schema["properties"]
    assert "age" in schema["properties"]
    assert "tags" in schema["properties"]
    assert "address" in schema["properties"]

    # Check array type
    assert schema["properties"]["tags"]["type"] == "array"
    assert "items" in schema["properties"]["tags"]

    # Check nullable field - Pydantic v2 uses anyOf for nullable fields
    address_schema = schema["properties"]["address"]
    if "anyOf" in address_schema:
        # Check if one of the types is null
        has_null_type = any(
            item.get("type") == "null" for item in address_schema["anyOf"]
        )
        assert has_null_type
    else:
        assert address_schema.get("nullable", False) is True

    # Check required fields
    assert "required" in schema
    assert "name" in schema["required"]
    assert "age" in schema["required"]
    assert "address" not in schema["required"]
    assert "tags" not in schema["required"]


def test_pydantic_to_schema_nested():
    """Test pydantic_to_schema with nested models."""
    schema = pydantic_to_schema(NestedModel)

    assert "properties" in schema
    assert "title" in schema["properties"]
    assert "user" in schema["properties"]

    # Check nested model - Pydantic v2 uses $ref
    user_schema = schema["properties"]["user"]

    # For Pydantic v2, it might use $ref
    if "$ref" in user_schema:
        # If using $ref, then SimpleModel should be in $defs
        assert "$defs" in schema
        assert "SimpleModel" in schema["$defs"] or user_schema["$ref"].split("/")[-1] in schema["$defs"]
    else:
        # For direct embedding of model (older versions)
        assert "properties" in user_schema
        assert "name" in user_schema["properties"]
        assert "age" in user_schema["properties"]
        assert "required" in user_schema
        assert "name" in user_schema["required"]
        assert "age" in user_schema["required"]


def test_pydantic_to_schema_without_docs():
    """Test pydantic_to_schema with a model without docstrings."""
    schema = pydantic_to_schema(ModelWithoutDocs)

    assert schema["name"] == "ModelWithoutDocs"
    assert schema["description"] == ""
    assert "properties" in schema
    assert "name" in schema["properties"]
    assert "age" in schema["properties"]
    assert "description" in schema["properties"]["name"]
    assert schema["properties"]["name"]["description"] == ""


def test_pydantic_to_schema_with_json_extra():
    """Test pydantic_to_schema with json_schema_extra in fields."""
    schema = pydantic_to_schema(ModelWithJsonExtra)

    assert "properties" in schema
    assert "name" in schema["properties"]
    assert "age" in schema["properties"]

    # Check extra schema properties
    name_schema = schema["properties"]["name"]
    assert "example" in name_schema
    assert name_schema["example"] == "John Doe"
    assert "pattern" in name_schema
    assert name_schema["pattern"] == "^[a-zA-Z ]+$"

    age_schema = schema["properties"]["age"]
    assert "minimum" in age_schema
    assert age_schema["minimum"] == 0
    assert "maximum" in age_schema
    assert age_schema["maximum"] == 120
