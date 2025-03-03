import json
import logging

import torch
from pydantic import BaseModel
from transformers import AutoTokenizer, LlamaForCausalLM

from pse.structuring_engine import StructuringEngine
from pse.util.torch_mixin import PSETorchMixin

# toggle this to logging.DEBUG to see the PSE debug logs!
logging.basicConfig(level=logging.DEBUG)


class PSE_Torch(PSETorchMixin, LlamaForCausalLM):
    pass


model_path = "meta-llama/Llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = PSE_Torch.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

model.config.pad_token_id = model.config.eos_token_id[0]
if model.generation_config:
    model.generation_config.top_p = None
    model.generation_config.top_k = 8
    model.generation_config.do_sample = True
    model.generation_config.temperature = 1.0
    model.generation_config.max_new_tokens = 1000
    model.generation_config.pad_token_id = model.config.eos_token_id[0]

# create structuring engine normally
model.engine = StructuringEngine(tokenizer, multi_token_sampling=True)
SIMPLE_JSON_SCHEMA = {
    "type": "object",
    "properties": {"value": {"type": "number"}},
    "required": ["value"],
}
model.engine.configure(SIMPLE_JSON_SCHEMA)
prompt = (
    "Please generate a json object with the value 9.11, with the following schema:\n"
)
prompt += json.dumps(SIMPLE_JSON_SCHEMA, indent=2)

messages = [{"role": "user", "content": prompt}]
input_ids = tokenizer.apply_chat_template(
    messages, return_tensors="pt", add_generation_prompt=True
)
assert isinstance(input_ids, torch.Tensor)
input_ids = input_ids.to(model.device)
assert isinstance(input_ids, torch.Tensor)
output = model.generate(
    input_ids,
    do_sample=True,
)
# you can print the prompt + output:
#   print(tokenizer.decode(output[0]))
# you can also access just the structured output:
#   engine.parse_structured_output()
structured_output = model.engine.structured_output(output_type=dict)
print(100 * "-")
print(json.dumps(structured_output, indent=2))

ADVANCED_JSON_SCHEMA = {
    "type": "object",
    "description": "High-level thoughts, reasoning and internal dialogue.\n Used for step by step reasoning.",
    "properties": {
        "chain_of_thought": {
            "type": "array",
            "items": {
                "type": "string",
                "minLength": 20,  # minimum length of a thought (optional)
            },
            "minItems": 1,  # floor the number of thoughts (optional)
            "maxItems": 3,  # limit the number of thoughts (optional)
        },
    },
    "required": ["chain_of_thought"],
}
model.engine.configure(ADVANCED_JSON_SCHEMA)
raw_prompt = (
    f"This is a test of your thought process.\n"
    f"I want to see your private internal monologue.\n"
    f"Please follow the following schema when generating your response:\n{json.dumps(ADVANCED_JSON_SCHEMA, indent=2)}\n"
)
messages = [{"role": "user", "content": raw_prompt}]
input_ids = tokenizer.apply_chat_template(
    messages, return_tensors="pt", add_generation_prompt=True
)
assert isinstance(input_ids, torch.Tensor)
input_ids = input_ids.to(model.device)
assert isinstance(input_ids, torch.Tensor)
greedy_output = model.generate(input_ids)
structured_output = model.engine.structured_output(output_type=dict)
if not structured_output:
    breakpoint()
print(100 * "-")
print(json.dumps(structured_output, indent=2))


class CursorPositionModel(BaseModel):
    """
    An object representing the position and click state of a cursor.

    Attributes:
        x_pos: The horizontal position of the cursor in pixels
        y_pos: The vertical position of the cursor in pixels
        left_click: Whether the left mouse button is currently pressed. Default is False.
    """

    x_pos: int
    y_pos: int
    left_click: bool = False


json_schema: dict = model.engine.configure(
    CursorPositionModel, json_delimiters=("<cursor>", "</cursor>")
)
prompt = (
    "Please use the following schema to generate a cursor position:\n"
    f"{json.dumps(json_schema, indent=2)}.\n"
    "Pretend to move the cursor to x = 100 and y = 100, with the left mouse button clicked.\n"
    "Wrap your response in <cursor>CursorPositionModel</cursor>."
)
messages = [{"role": "user", "content": prompt}]
input_ids = tokenizer.apply_chat_template(
    messages, return_tensors="pt", add_generation_prompt=True
)
assert isinstance(input_ids, torch.Tensor)
input_ids = input_ids.to(model.device)
assert isinstance(input_ids, torch.Tensor)
output = model.generate(
    input_ids,
    do_sample=True,
)
structured_output = model.engine.structured_output(output_type=CursorPositionModel)
print(100 * "-")
print(json.dumps(structured_output.model_dump(), indent=2))
