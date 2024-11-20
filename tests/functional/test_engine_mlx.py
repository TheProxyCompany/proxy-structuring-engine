import json
from typing import Tuple, cast
import pytest
from pse.core.engine import StructuringEngine

import mlx.nn as nn
import mlx.core as mx
from mlx_lm.utils import load
from mlx_lm.sample_utils import categorical_sampling

pytest.importorskip("mlx_lm", reason="mlx_lm is not installed. Skipping tests.")

@pytest.fixture(scope="module")
def model_and_engine() -> Tuple[nn.Module, StructuringEngine]:
    """Module-scoped fixture for the StructuredOutputDriver."""
    TEST_MODEL = (
        "/Users/jckwind/Documents/ProxyBot/language_models/Llama-3.1-SuperNova-Lite"
    )
    model, tokenizer = load(TEST_MODEL)
    engine = StructuringEngine(tokenizer._tokenizer)
    return model, engine

def generate_step(
    prompt: mx.array,
    model: nn.Module,
    engine: StructuringEngine,
    temp: float = 1.0,
) -> Tuple[mx.array, mx.array]:
    """
    A generator producing token ids based on the given prompt from the model.

    Args:
        prompt (mx.array): The input prompt.
        model (nn.Module): The model to use for generation.
        engine (StructuringEngine): The engine to use for generation.
        temp (float): The temperature for sampling, if 0.0 the argmax is used.
          Default: ``1.0``.
    Yields:
        Tuple[mx.array, mx.array]: A tuple of one token and a vector of log probabilities.
    """
    logits = model(prompt[None])
    logits = logits[:, -1, :]
    logits += engine.generate_logit_bias_mask(logits)

    if engine.acceptor and not engine.has_reached_accept_state():
        valid_token_id = engine.get_next_token(logits[0, :], top_k=4)
        token = mx.array([valid_token_id])
    else:
        token: mx.array = categorical_sampling(logits, temp) if temp > 0.0 else mx.argmax(logits, axis=-1)

    logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    mx.async_eval(token, logprobs)
    return token, logprobs

def test_simple_json_structure(model_and_engine: Tuple[nn.Module, StructuringEngine]) -> None:
    model, engine = model_and_engine
    schema = {
        "type": "object",
        "properties": {"value": {"type": "number"}},
        "required": ["value"],
        "additionalProperties": False,
    }
    raw_prompt = f"Please generate a simple JSON object with the number 9.11. Follow this schema: {str(schema)}"
    engine.set_schema(schema, use_delimiters=False)

    assert not engine.within_json_value
    assert engine.in_structured_state

    messages = [{"role": "user", "content": raw_prompt}]
    encoded_prompt = engine.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    prompt = mx.array(encoded_prompt)
    tokens = []

    while not engine.has_reached_accept_state():
        token, _ = generate_step(prompt, model, engine)
        prompt = mx.concatenate([prompt, token])
        tokens.append(cast(int, token.item()))

    output = engine.tokenizer.decode(tokens)
    assert json.loads(output) == {"value": 9.11}


# def test_complex_json_structure(model_and_engine: Tuple[nn.Module, StructuringEngine]) -> None:
#     """Test parsing a complex JSON structure."""
#     model, engine = model_and_engine
#     schema = {
#         "type": "object",
#         "properties": {
#             "name": {"const": "metacognition"},
#             "arguments": {
#                 "type": "object",
#                 "properties": {
#                     "chain_of_thought": {
#                         "type": "array",
#                         "items": {"type": "string"},
#                     },
#                     "feelings": {
#                         "type": ["string"],
#                         "nullable": True,
#                         "default": None,
#                     },
#                 },
#                 "required": ["chain_of_thought"],
#             },
#         },
#         "required": ["name", "arguments"],
#     }
#     engine.set_schema(schema)


# def test_better_than_openai(model_and_engine: Tuple[nn.Module, StructuringEngine]) -> None:
#     """Test that OpenAI sucks."""
#     # openAI's structured output blog post said:
#     #
#     #   "The following is a sample recursive schema that is supported on
#     #   the OpenAI API with Structured Outputs but would not be possible to express with a FSM."
#     #
#     # let's test that.
#     model, engine = model_and_engine
#     schema = {
#         "name": "ui",
#         "description": "Dynamically generated UI",
#         "strict": True,
#         "schema": {
#             "type": "object",
#             "properties": {
#                 "type": {
#                     "type": "string",
#                     "description": "The type of the UI component",
#                     "enum": ["div", "button", "header", "section", "field", "form"],
#                 },
#                 "label": {
#                     "type": "string",
#                     "description": "The label of the UI component, used for buttons or form fields",
#                 },
#                 "children": {
#                     "type": "array",
#                     "description": "Nested UI components",
#                     "items": {"$ref": "#"},
#                 },
#                 "attributes": {
#                     "type": "array",
#                     "description": "Arbitrary attributes for the UI component, suitable for any element",
#                     "items": {
#                         "type": "object",
#                         "properties": {
#                             "name": {
#                                 "type": "string",
#                                 "description": "The name of the attribute, for example onClick or className",
#                             },
#                             "value": {
#                                 "type": "string",
#                                 "description": "The value of the attribute",
#                             },
#                         },
#                     },
#                 },
#             },
#             "required": ["type", "label", "children", "attributes"],
#             "additionalProperties": False,
#         },
#     }
#     engine.set_schema(schema, use_delimiters=False)
#     messages = [
#         {
#             "role": "user",
#             "content": f"Please generate a simple UI with a div that has a label that says 'Hello, world!'. Follow this schema: {str(schema)}",
#         }
#     ]
#     prompt = engine.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
#     assert isinstance(prompt, list)
#     prompt = mx.array(prompt)

#     tokens = []

#     for token_id, log_probs in generate_step(prompt=prompt, model=model, engine=engine):
#         token_id = cast(mx.array, token_id)
#         tokens.append(token_id)
