import json
import logging
import timeit
from collections.abc import Callable
from enum import Enum
from typing import Any

import pytest
from openai import OpenAI
from outlines import generate, models
from pydantic import BaseModel, Field

from pse.structuring_engine import StructuringEngine

logger = logging.getLogger(__name__)

try:
    from mlx_lm.utils import load

    from pse.util.generate_mlx import generate as generate_mlx
except ImportError:
    pytest.skip("mlx or mlx_lm is not installed. Skipping tests.", allow_module_level=True)


class UiType(str, Enum):
    DIV = "div"
    BUTTON = "button"
    HEADER = "header"
    SECTION = "section"
    FIELD = "field"
    FORM = "form"


class Attribute(BaseModel):
    name: str = Field(
        description="The name of the attribute, for example onClick or className"
    )
    value: str = Field(description="The value of the attribute")


class DynamicUI(BaseModel):
    type: UiType = Field(description="The type of the UI component")
    label: str = Field(
        description="The label of the UI component, used for buttons or form fields"
    )
    children: list["DynamicUI"] = Field(description="Nested UI components")
    attributes: list[Attribute] = Field(
        description="Arbitrary attributes for the UI component, suitable for any element"
    )

    class ConfigDict:
        extra = "forbid"


def run_benchmark(
    name: str,
    generator_func: Callable[[str, Any], tuple[dict[str, Any], float]],
    prompt: str,
    schema: Any,
) -> tuple[dict[str, Any], float]:
    output, timing = generator_func(prompt, schema)
    # Perform assertions
    logger.info(f"{name} timing: {timing} seconds")
    return output, timing


def generate_local_pse(prompt: str, schema: Any) -> tuple[Any, float]:
    model_path_hf = "meta-llama/Llama-3.2-3B-Instruct"
    model, tokenizer = load(model_path_hf)
    engine = StructuringEngine(tokenizer._tokenizer)
    engine.configure(schema, buffer_length=-1)

    start_time = timeit.default_timer()
    generate_mlx(prompt, model, engine)
    for output in engine.read_output():
        end_time = timeit.default_timer()
        total_time = end_time - start_time
        return output.value, total_time
    return None, 0


def generate_outlines(prompt: str, schema: Any) -> tuple[dict[str, Any], float]:
    if isinstance(schema, dict):
        schema = json.dumps(schema)

    model_path_hf = "mlx-community/Llama-3.2-3B-Instruct"
    model = models.mlxlm(model_name=model_path_hf)  # type: ignore[reportPrivateImportUsage]
    generator = generate.json(model, schema)  # type: ignore[reportPrivateImportUsage]

    start_time = timeit.default_timer()
    output = generator(prompt)
    end_time = timeit.default_timer()
    total_time = end_time - start_time

    return output, total_time  # type: ignore[reportReturnType]


def generate_openai_instructor(
    prompt: str, schema: Any
) -> tuple[dict[str, Any], float]:
    import instructor

    # Patch the OpenAI client
    try:
        client = OpenAI()
    except Exception as e:
        pytest.skip(f"OpenAI client failed to initialize: {e}")
    client = instructor.from_openai(client)

    if isinstance(schema, dict):
        pytest.skip("OpenAI Instructor does not support Dict schema")

    # Extract structured data from natural language
    start_time = timeit.default_timer()
    result = client.chat.completions.create(
        model="gpt-4o-mini",
        response_model=schema,
        messages=[{"role": "user", "content": prompt}],
    )
    end_time = timeit.default_timer()
    total_time = end_time - start_time

    return result.model_dump(), total_time


def generate_openai_structured_output(
    prompt: str, schema: Any
) -> tuple[dict[str, Any], float]:
    # Initialize the OpenAI client
    try:
        client = OpenAI()
    except Exception as e:
        pytest.skip(f"{e}")

    start_time = timeit.default_timer()

    if isinstance(schema, dict):
        schema = {
            "type": "json_schema",
            "json_schema": {
                "name": "test",
                "schema": schema,
            },
        }

        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            response_format=schema,
        )

        result = completion.choices[0].message.content

    else:
        completion = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            response_format=schema,
        )
        result = completion.choices[0].message.parsed

    if not result:
        pytest.fail("Failed to parse dynamic UI.")

    end_time = timeit.default_timer()
    total_time = end_time - start_time

    return result.model_dump() if isinstance(result, BaseModel) else json.loads(
        result
    ), total_time


@pytest.mark.parametrize(
    "generator_name,generator_func",
    [
        ("Outlines", generate_outlines),
        ("OpenAI Instructor", generate_openai_instructor),
        ("OpenAI Structured", generate_openai_structured_output),
        ("PSE", generate_local_pse),
    ],
)
def test_simple_json_object(generator_name: str, generator_func: Callable) -> None:
    """
    Validates that the engine can generate a simple JSON object
    adhering to a specified schema using real LLM output.
    """
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {"value": {"type": "number"}},
        "required": ["value"],
    }
    prompt = (
        f"Generate a JSON object with the number 9.11. Follow this schema: {schema}"
    )

    try:
        output, timing = run_benchmark(generator_name, generator_func, prompt, schema)
        assert output["value"] == 9.11
        logger.info(f"✅ {generator_name}: {timing:.4f}s")
    except Exception as e:
        pytest.fail(f"❌ {generator_name} benchmark failed with error: {e}")

if __name__ == "__main__":
    pytest.main()
