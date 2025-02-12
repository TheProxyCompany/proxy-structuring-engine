import json

import torch
from transformers import AutoTokenizer, LlamaForCausalLM

from pse.engine.structuring_engine import StructuringEngine
from pse.util.torch_mixin import PSETorchMixin


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

model.engine = StructuringEngine(tokenizer)
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
    messages,
    return_tensors="pt",
    add_generation_prompt=True
)
assert isinstance(input_ids, torch.Tensor)
input_ids = input_ids.to(model.device)
assert isinstance(input_ids, torch.Tensor)
print(repr(model.engine))
output = model.generate(
    input_ids,
    do_sample=True,
    temperature=1.0,
    min_p=0.1,
)
print("Output:\n" + 100 * "-")
print(tokenizer.decode(output[0]))
