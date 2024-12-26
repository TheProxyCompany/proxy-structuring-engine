import json
import logging

import pytest

from pse.structuring_engine import StructuringEngine

logger = logging.getLogger(__name__)

try:
    import mlx.nn as nn
    from mlx_lm.utils import load

    from pse.util.generate_mlx import generate, sample
except ImportError:
    pytest.skip(
        "mlx or mlx_lm is not installed. Skipping tests.", allow_module_level=True
    )


@pytest.fixture(scope="module")
def model_and_engine() -> tuple[nn.Module, StructuringEngine]:
    """Module-scoped fixture for the StructuredOutputDriver."""
    model_path_hf = "meta-llama/Llama-3.2-3B-Instruct"
    model, tokenizer = load(model_path_hf)
    engine = StructuringEngine(tokenizer._tokenizer)
    return model, engine

def get_dict_from_raw_output(raw_output: str) -> dict:
    try:
        if raw_output.startswith("```json\n"):
            only_json = raw_output.split("```json\n", 1)[1].split("\n```", 1)[0]
        else:
            only_json = raw_output
    except IndexError:
        pytest.fail(
            f"Failed to extract JSON content from delimited output: {raw_output}"
        )
    try:
        return json.loads(only_json)
    except json.JSONDecodeError:
        pytest.fail(f"Failed to parse JSON output for {raw_output}.")


def test_simple_json_structure(
    model_and_engine: tuple[nn.Module, StructuringEngine],
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
    engine.configure(schema, wrap_with_delimiters=False)
    completed_generation = generate(raw_prompt, model, engine)
    # Validate the generated output
    assert engine.has_reached_accept_state
    assert json.loads(completed_generation.output) == {"value": 9.11}


def test_token_by_token_generation(
    model_and_engine: tuple[nn.Module, StructuringEngine],
) -> None:
    """Test that the engine can generate tokens one at a time."""
    model, engine = model_and_engine
    schema = {"type": "string"}
    engine.configure(schema, wrap_with_delimiters=False)
    step_1 = sample("Respond with a string.", model, engine)
    assert engine.tokenizer.decode([step_1.token_id]).startswith('"')


def test_complex_json_structure(
    model_and_engine: tuple[nn.Module, StructuringEngine],
) -> None:
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
                        "minItems": 1,
                    },
                    "feelings": {
                        "type": "string",
                    },
                },
                "required": ["chain_of_thought", "feelings"],
            },
        },
        "required": ["name", "arguments"],
    }
    raw_prompt = (
        f"This is a test of your abilities."
        f"Please structure your response to follow the following schema: {schema}."
        f"You must wrap your response with ```json\n and \n```."
    )
    engine.configure(schema, wrap_with_delimiters=True)
    completed_generation = generate(raw_prompt, model, engine)
    raw_output = completed_generation.output
    output = get_dict_from_raw_output(raw_output)

    logger.info(f"Output: {output}")
    assert output["name"] == "metacognition"
    assert "arguments" in output
    assert "chain_of_thought" in output["arguments"]
    assert isinstance(output["arguments"]["chain_of_thought"], list)
    assert engine.has_reached_accept_state


def test_better_than_openai(
    model_and_engine: tuple[nn.Module, StructuringEngine],
) -> None:
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
        f"Please generate a div component that has one child - a button component that says 'Hello, World!'."
        f"Please follow the following schema: {schema}."
    )
    engine.configure(schema, wrap_with_delimiters=False)
    completed_generation = generate(raw_prompt, model, engine)
    output = get_dict_from_raw_output(completed_generation.output)

    assert output["type"] == "div"
    assert any(child["type"] == "button" for child in output["children"])
    assert any(child["label"] == "Hello, World!" for child in output["children"])


def test_multiple_schemas(
    model_and_engine: tuple[nn.Module, StructuringEngine],
) -> None:
    """Test that the engine can generate multiple schemas."""
    model, engine = model_and_engine
    schema = {
        "oneOf": [
            {
                "type": "object",
                "properties": {
                    "name": {"type": "const", "const": "send_message"},
                    "arguments": {
                        "type": "object",
                        "properties": {
                            "message": {
                                "type": "string",
                                "description": "The final message content to be sent to the recipient.\nThis should be a packaged, markdown-formatted summary of the agent's work.\nSupports all Unicode characters, including emojis.",
                            }
                        },
                        "required": ["message"],
                    },
                },
                "required": ["name", "arguments"],
            },
            {
                "type": "object",
                "properties": {
                    "name": {"type": "const", "const": "metacognition"},
                    "arguments": {
                        "type": "object",
                        "properties": {
                            "chain_of_thought": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "A sequence of high-level thoughts, reasoning, and internal dialogue.\nIncludes complex ideas, strategic considerations, and conceptual frameworks.\nSupports all Unicode characters, including emojis.",
                            },
                            "feelings": {
                                "type": ["string"],
                                "description": "A reflection of the agent's emotional state.\nSupports all Unicode characters, including emojis.",
                                "nullable": True,
                                "default": None,
                            },
                        },
                        "required": ["chain_of_thought"],
                    },
                },
                "required": ["name", "arguments"],
            },
        ]
    }
    raw_prompt = (
        f"This is a test of your abilities."
        f"Please structure your response to follow the following schemas: {schema}."
        f"You must wrap your response with ```json\n and \n```."
        "Please use the metacognition schema."
    )
    engine.configure(schema, wrap_with_delimiters=True)
    completed_generation = generate(raw_prompt, model, engine)
    assert engine.has_reached_accept_state
    output = get_dict_from_raw_output(completed_generation.output)
    assert output["name"] == "metacognition"


def test_schema_web_search(
    model_and_engine: tuple[nn.Module, StructuringEngine],
) -> None:
    """Test that the engine can generate a schema for web search."""
    model, engine = model_and_engine
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "const", "const": "web_search"},
            "arguments": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query string. Text only. An artificial assistant will be created to perform the search.",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "The maximum number of search results to return. Defaults to 5.",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        },
        "required": ["name", "arguments"],
    }
    engine.configure(schema, wrap_with_delimiters=False)
    prefill = '```json\n{"name": "web_search", "arguments": {"query": "popular favorite Pokémon",'
    engine.consume_raw_input(prefill)
    raw_prompt = (
        f"This is a test of your abilities."
        f" Please structure your response to follow the following schemas: {schema}."
        f" You must wrap your response with ```json\n and \n```."
        " Please use the web_search schema to find popular favoirte pokemon."
    )
    completed_generation = generate(raw_prompt, model, engine, prefill=prefill)
    assert engine.has_reached_accept_state
    output = get_dict_from_raw_output(completed_generation.output)
    assert output["name"] == "web_search"
    assert output["arguments"]["query"] == "popular favorite Pokémon"
    assert output["arguments"]["max_results"] is not None


def test_message(
    model_and_engine: tuple[nn.Module, StructuringEngine],
) -> None:
    """Test that the engine can generate a schema for web search."""
    model, engine = model_and_engine
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "const", "const": "send_message"},
            "arguments": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "The final message content to be sent to the recipient.\nThis should be a packaged, markdown-formatted summary of the agent's work.\nSupports all Unicode characters, including emojis.",
                    }
                },
                "required": ["message"],
            },
        },
        "required": ["name", "arguments"],
    }
    engine.configure(schema, wrap_with_delimiters=False)
    prefill = '```json\n{"name": "send_message", "arguments": {"message": ":D Ah, I see what you\'re getting at'
    engine.consume_raw_input(prefill)
    raw_prompt = (
        f"This is a test of your abilities."
        f"Please structure your response to follow the following schemas: {schema}."
        f"You must wrap your response with ```json\n and \n```."
    )
    completed_generation = generate(raw_prompt, model, engine, prefill=prefill)
    assert engine.has_reached_accept_state
    output = get_dict_from_raw_output(completed_generation.output)
    assert output["name"] == "send_message"
