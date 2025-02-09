import logging

import pytest

from pse.engine.structuring_engine import StructuringEngine

logger = logging.getLogger(__name__)

try:
    import mlx.nn as nn
    from mlx_proxy.utils import load

    from pse.util.generate_mlx import generate
except ImportError:
    pytest.skip(
        "mlx or mlx_lm is not installed. Skipping tests.", allow_module_level=True
    )


@pytest.fixture(scope="module")
def model_and_engine() -> tuple[nn.Module, StructuringEngine]:
    """Module-scoped fixture for the StructuredOutputDriver."""
    model_path_hf = "meta-llama/Llama-3.2-3B-Instruct"
    model, tokenizer = load(model_path_hf)
    engine = StructuringEngine(tokenizer)
    return model, engine


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
    engine.configure(schema)
    generate(
        raw_prompt,
        model,
        engine,
    )
    # Validate the generated output
    assert engine.has_reached_accept_state
    final_output = engine.parse_structured_output()
    assert final_output == {"value": 9.11}


def test_simple_json_structure_with_delimiters(
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
    delimiters = ("<tool_call>\n", "\n</tool_call>")
    raw_prompt = (
        "Generate a JSON object with the number 9.11."
        f"Wrap your output in {delimiters[0]} and {delimiters[1]}."
        f"Follow this schema: {schema}"
    )
    engine.configure(schema, json_delimiters=delimiters)
    generate(
        raw_prompt,
        model,
        engine,
    )
    # Validate the generated output
    assert engine.has_reached_accept_state
    final_output = engine.parse_structured_output()
    assert final_output == {"value": 9.11}


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
                        "maxItems": 3,
                    },
                    "feelings": {
                        "type": "string",
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
    engine.configure(schema, json_delimiters=("```json\n", "\n```"))
    generate(raw_prompt, model, engine)
    final_output = engine.parse_structured_output()
    assert final_output
    assert isinstance(final_output, dict)
    assert final_output["name"] == "metacognition"
    assert "arguments" in final_output
    assert "chain_of_thought" in final_output["arguments"]
    assert isinstance(final_output["arguments"]["chain_of_thought"], list)

    assert engine.has_reached_accept_state


def test_match_openai(model_and_engine: tuple[nn.Module, StructuringEngine]) -> None:
    # openAI's structured output blog post said:
    #
    #   "The following is a sample recursive schema that is supported on
    #   the OpenAI API with Structured Outputs
    #   but would not be possible to express with a FSM."
    #
    # let's test that.
    #
    # note: models above 3b params have an easier time with this.
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
                    "nullable": True,
                    "minItems": 1,
                    "maxItems": 1,
                },
            },
            "required": ["type"],
            "additionalProperties": False,
        },
    }
    raw_prompt = (
        f"Please generate a JSON object that represents a div component that only has one child button."
        f"Please follow the following schema: {schema}."
    )
    engine.configure(schema, min_buffer_length=-1)
    generate(raw_prompt, model, engine)
    final_output = engine.parse_structured_output()
    assert final_output["type"] == "div"
    assert len(final_output["children"]) == 1


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
                    "name": {"const": "send_message"},
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
                    "name": {"const": "metacognition"},
                    "arguments": {
                        "type": "object",
                        "properties": {
                            "chain_of_thought": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "A sequence of high-level thoughts, reasoning, and internal dialogue.\nIncludes complex ideas, strategic considerations, and conceptual frameworks.\nSupports all Unicode characters, including emojis.",
                            },
                            "feelings": {
                                "type": "string",
                                "description": "A reflection of the agent's emotional state.\nSupports all Unicode characters, including emojis.",
                                "nullable": True,
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
    engine.configure(
        schema, json_delimiters=("```json\n", "\n```"), min_buffer_length=0
    )
    generate(raw_prompt, model, engine)
    final_output = engine.parse_structured_output()
    assert final_output["name"] == "metacognition"


def test_schema_web_search(
    model_and_engine: tuple[nn.Module, StructuringEngine],
) -> None:
    """Test that the engine can generate a schema for web search."""
    model, engine = model_and_engine
    schema = {
        "type": "object",
        "properties": {
            "name": {"const": "web_search"},
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
    engine.reset(hard_reset=True)
    engine.configure(schema, json_delimiters=("<tool>", "</tool>"))
    prefill = '<tool>{"name": "web_search", "arguments": {"query": "popular favorite Pokémon",'
    engine.consume_text(prefill)
    raw_prompt = (
        f"This is a test of your abilities."
        f" Please structure your response to follow the following schemas: {schema}."
        f" You must wrap your response with <tool> and </tool>."
        " Please use the web_search schema to find popular favoirte pokemon."
    )
    generate(raw_prompt, model, engine, prefill)
    final_output = engine.parse_structured_output()
    assert final_output["name"] == "web_search"
    assert final_output["arguments"]["query"] == "popular favorite Pokémon"
    assert final_output["arguments"]["max_results"] is not None


def test_python_interpreter(
    model_and_engine: tuple[nn.Module, StructuringEngine],
) -> None:
    """Test that the engine can generate a python interpreter."""
    model, engine = model_and_engine
    engine.reset(hard_reset=True)
    engine.configure({}, include_python=True, min_buffer_length=0)
    raw_prompt = "Please generate the python code `print('Hello, world!')`."
    raw_prompt += "Wrap the code in ```python\n and \n```."
    generate(raw_prompt, model, engine)
    assert engine.has_reached_accept_state
    output = engine.parse_structured_output()
    assert output == "print('Hello, world!')"

def test_python_edge_case(model_and_engine: tuple[nn.Module, StructuringEngine]) -> None:
    """Test that the engine can generate a python strawberry."""
    model, engine = model_and_engine
    engine.reset(hard_reset=True)
    engine.configure({}, include_python=True, min_buffer_length=-1)
    requested_code = '"strawberry".count("r")'
    raw_prompt = f"Please return the following: ```python\n{requested_code}\n```"
    prefill = '```python\n"strawberry".count("r")\n``'
    engine.consume_text(prefill)
    generate(raw_prompt, model, engine, prefill)
    assert engine.has_reached_accept_state
    output = engine.parse_structured_output()
    assert output == '"strawberry".count("r")'

if __name__ == "__main__":
    pytest.main()
