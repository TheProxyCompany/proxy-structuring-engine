import json
import logging

import torch
from transformers import AutoTokenizer, LlamaForCausalLM

from pse.engine.structuring_engine import StructuringEngine
from pse.util.torch_mixin import PSETorchMixin

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
    model.generation_config.pad_token_id = model.config.eos_token_id[0]
    model.generation_config.max_new_tokens = 1000

model.engine = StructuringEngine(tokenizer)
# SIMPLE_JSON_SCHEMA = {
#     "type": "object",
#     "properties": {"value": {"type": "number"}},
#     "required": ["value"],
# }
# model.engine.configure(SIMPLE_JSON_SCHEMA)
# prompt = "Please generate a json object with the value 9.11, with the following schema:\n"
# prompt += json.dumps(SIMPLE_JSON_SCHEMA, indent=2)

# messages = [{"role": "user", "content": prompt}]
# input_ids = tokenizer.apply_chat_template(
#     messages,
#     return_tensors="pt",
#     add_generation_prompt=True
# )
# assert isinstance(input_ids, torch.Tensor)
# input_ids = input_ids.to(model.device)
# assert isinstance(input_ids, torch.Tensor)
# output = model.generate(
#     input_ids,
#     do_sample=True,
#     temperature=1.0,
#     min_p=0.1,
# )
# print("Output:\n" + 100 * "-")
# print(tokenizer.decode(output[0]))

# @title Test advanced-json generation
ADVANCED_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"const": "metacognition"},
        "arguments": {
            "type": "object",
            "properties": {
                "chain_of_thought": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "description": "A thought or instance of awareness.",
                    },
                    "description": "A sequence of high-level thoughts, reasoning and internal dialogue.",
                    "minItems": 3,
                    "maxItems": 5,
                },
            },
            "required": ["chain_of_thought"],
        },
    },
    "required": ["name", "arguments"],
}
model.engine.configure(ADVANCED_JSON_SCHEMA)
prompt = "This is a test of your schema following abilities.\n"
prompt += "Structure your response into a json object that matches this schema: {schema}"
prompt = prompt.format(schema=json.dumps(ADVANCED_JSON_SCHEMA, indent=2))

messages = [{"role": "user", "content": prompt}]
input_ids = tokenizer.apply_chat_template(
    messages, return_tensors="pt", add_generation_prompt=True
)
assert isinstance(input_ids, torch.Tensor)
input_ids = input_ids.to(model.device)
assert isinstance(input_ids, torch.Tensor)
greedy_output = model.generate(
    input_ids,
    do_sample=True,
    temperature=1.0,
    min_p=0.08,  # you can use your favorite sampling options!
)
print("Output:\n" + 100 * "-")
print(tokenizer.decode(greedy_output[0]))
