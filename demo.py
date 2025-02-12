import json

import torch
from transformers import (
    AutoTokenizer,
    LlamaForCausalLM,
    LogitsProcessorList,
)

from pse.engine.structuring_engine import StructuringEngine

model_path = "meta-llama/Llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
engine = StructuringEngine(tokenizer)
logits_processor = LogitsProcessorList([engine.process_logits])

model = LlamaForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto",
)
model.config.pad_token_id = model.config.eos_token_id[0]
if model.generation_config:
    model.generation_config.pad_token_id = (
        model.config.pad_token_id or model.config.eos_token_id[0]
    )
    model.generation_config.max_new_tokens = 10


SIMPLE_JSON_SCHEMA = {
    "type": "object",
    "properties": {"value": {"type": "number"}},
    "required": ["value"],
}
engine.configure(SIMPLE_JSON_SCHEMA)
prompt = f"Please generate the number 9.11 in a json object with the following schema:\n{json.dumps(SIMPLE_JSON_SCHEMA, indent=2)}"
messages = [{"role": "user", "content": prompt}]
input_ids = tokenizer.apply_chat_template(
    messages, return_tensors="pt", add_generation_prompt=True
)
assert isinstance(input_ids, torch.Tensor)
input_ids = input_ids.to(model.device)
assert isinstance(input_ids, torch.Tensor)
greedy_output = model.generate(
    input_ids,
    logits_processor=logits_processor,
    do_sample=True,
    temperature=1.0,
)
print("Output:\n" + 100 * "-")
print(tokenizer.decode(greedy_output[0]))
