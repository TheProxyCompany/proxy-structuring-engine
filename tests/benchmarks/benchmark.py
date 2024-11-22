import pytest
import json
from pydantic import BaseModel, Field
from typing import Tuple, List
from enum import Enum
from pse.core.engine import StructuringEngine

import logging

logger = logging.getLogger(__name__)

try:
    import mlx.nn as nn
    from mlx_lm.utils import load
    from tests.util import generate_response
except ImportError:
    pytest.skip(
        "mlx or mlx_lm is not installed. Skipping tests.", allow_module_level=True
    )

class UiType(str, Enum):
    DIV = "div"
    BUTTON = "button"
    HEADER = "header"
    SECTION = "section"
    FIELD = "field"
    FORM = "form"

class Attribute(BaseModel):
    name: str = Field(description="The name of the attribute, for example onClick or className")
    value: str = Field(description="The value of the attribute")

class DynamicUI(BaseModel):
    type: UiType = Field(description="The type of the UI component")
    label: str = Field(description="The label of the UI component, used for buttons or form fields")
    children: List["DynamicUI"] = Field(description="Nested UI components")
    attributes: List[Attribute] = Field(
        description="Arbitrary attributes for the UI component, suitable for any element"
    )

    class ConfigDict:
        extra = "forbid"

@pytest.fixture(scope="module")
def model_and_engine() -> Tuple[nn.Module, StructuringEngine]:
    """Module-scoped fixture for the StructuredOutputDriver."""
    model_path_hf = "meta-llama/Llama-3.2-3B-Instruct"
    model, tokenizer = load(model_path_hf)
    engine = StructuringEngine(tokenizer._tokenizer)
    return model, engine

def test_dynamic_UI_with_pydantic_schema(
    model_and_engine: Tuple[nn.Module, StructuringEngine],
) -> None:
    """
    Validates that the engine can generate a simple JSON object
    adhering to a specified schema using real LLM output.
    """
    model, engine = model_and_engine
    schema = DynamicUI.model_json_schema()
    raw_prompt = (
        f"Please return a div that has a child button that says 'Hello, World!'."
        f"Please format your response to follow the following schema: {schema}."
    )
    engine.set_schema(DynamicUI, use_delimiters=False)
    completed_generation = generate_response(raw_prompt, model, engine)
    # Validate the generated output
    try:
        output = json.loads(completed_generation.output)
    except json.JSONDecodeError:
        pytest.fail(f"Failed to parse JSON output for {completed_generation.output}.")

    assert output["type"] == "div"
    assert len(output["children"]) == 1
    assert output["children"][0]["type"] == "button"
    assert output["children"][0]["label"] == "Hello, World!"
