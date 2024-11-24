import pytest
import json
from typing import Tuple
from pse.core.engine import StructuringEngine

import logging

logger = logging.getLogger(__name__)

try:
    import mlx.nn as nn
    from mlx_lm.utils import load
    from pse.util.generate.mlx import generate_response
except ImportError:
    pytest.skip(
        "mlx or mlx_lm is not installed. Skipping tests.", allow_module_level=True
    )


@pytest.fixture(scope="module")
def model_and_engine() -> Tuple[nn.Module, StructuringEngine]:
    """Module-scoped fixture for the StructuredOutputDriver."""
    model_path_hf = "meta-llama/Llama-3.2-3B-Instruct"
    model, tokenizer = load(model_path_hf)
    engine = StructuringEngine(tokenizer._tokenizer)
    return model, engine

def test_simple_json_structure(
    model_and_engine: Tuple[nn.Module, StructuringEngine],
) -> None:
    """
    Validates that the engine can generate a simple JSON object
    adhering to a specified schema using real LLM output.
    """
    model, engine = model_and_engine
    # Define schema and prompt
    schema = {
        "type": "object",
        "properties": {"value": {"type": "number"}},
        "required": ["value"],
        "additionalProperties": False,
    }
    raw_prompt = (
        f"Generate a JSON object with the number 9.11. Follow this schema: {schema}"
    )
    engine.set_schema(schema, use_delimiters=False)
    completed_generation = generate_response(raw_prompt, model, engine)
    # Validate the generated output
    assert json.loads(completed_generation.output) == {"value": 9.11}


def test_complex_json_structure(model_and_engine: Tuple[nn.Module, StructuringEngine]) -> None:
    """Test parsing a complex JSON structure."""
    model, engine = model_and_engine
    schema = {
        "type": "object",
        "properties": {
            "name": {"const": "metacognition"},
            "arguments": {
                "type": "object",
                "properties": {
                    "chain_of_thought": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "feelings": {
                        "type": ["string"],
                        "nullable": True,
                    },
                },
                "required": ["chain_of_thought"],
            },
        },
        "required": ["name", "arguments"],
    }
    raw_prompt = (
        f"This is a test of your abilities."
        f"Please structure your response to follow the following schema: {schema}."
        f"You must wrap your response with ```json\n and \n```."
    )
    engine.set_schema(schema, use_delimiters=True)
    completed_generation = generate_response(raw_prompt, model, engine)
    raw_output = completed_generation.output
    try:
        only_json = raw_output.split("```json\n", 1)[1].split("\n```", 1)[0]
    except IndexError:
        pytest.fail(f"Failed to extract JSON content from delimited output: {raw_output}")

    try:
        output = json.loads(only_json)
    except json.JSONDecodeError:
        pytest.fail(f"Failed to parse JSON output for {completed_generation.output}.")

    assert output["name"] == "metacognition"
    assert "arguments" in output
    assert "chain_of_thought" in output["arguments"]
    assert isinstance(output["arguments"]["chain_of_thought"], list)

def test_better_than_openai(model_and_engine: Tuple[nn.Module, StructuringEngine]) -> None:
    """Test that OpenAI sucks."""
    # openAI's structured output blog post said:
    #
    #   "The following is a sample recursive schema that is supported on
    #   the OpenAI API with Structured Outputs but would not be possible to express with a FSM."
    #
    # let's test that.
    model, engine = model_and_engine
    schema = {
        "name": "ui",
        "description": "Dynamically generated UI",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "type": {
                    "type": "string",
                    "description": "The type of the UI component",
                    "enum": ["div", "button", "header", "section", "field", "form"],
                },
                "label": {
                    "type": "string",
                    "description": "The label of the UI component, used for buttons or form fields",
                },
                "children": {
                    "type": "array",
                    "description": "Nested UI components",
                    "items": {"$ref": "#"},
                },
                "attributes": {
                    "type": "array",
                    "description": "Arbitrary attributes for the UI component, suitable for any element",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "The name of the attribute, for example onClick or className",
                            },
                            "value": {
                                "type": "string",
                                "description": "The value of the attribute",
                            },
                        },
                    },
                },
            },
            "required": ["type", "label", "children", "attributes"],
            "additionalProperties": False,
        },
    }
    raw_prompt = (
        f"Please return a div that has one child - a button that says 'Hello, World!'."
        f"Please format your response to follow the following schema: {schema}."
    )
    engine.set_schema(schema, use_delimiters=False)
    completed_generation = generate_response(raw_prompt, model, engine)
    try:
        output = json.loads(completed_generation.output)
    except json.JSONDecodeError:
        pytest.fail(f"Failed to parse JSON output for {completed_generation.output}.")

    assert output["type"] == "div"
    assert len(output["children"]) == 1
    assert output["children"][0]["type"] == "button"
    assert output["children"][0]["label"] == "Hello, World!"
