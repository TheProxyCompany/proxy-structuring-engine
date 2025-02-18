import json

import torch
from pydantic import BaseModel
from transformers import AutoTokenizer, LlamaForCausalLM

from pse.engine.structuring_engine import StructuringEngine
from pse.util.torch_mixin import PSETorchMixin


# 1. Define your desired output structure using Pydantic
class Product(BaseModel):
    name: str
    price: float
    description: str | None = None

# 2. Load your model and tokenizer.  Apply the PSE mixin.
class PSE_Torch(PSETorchMixin, LlamaForCausalLM):
    pass

model_path = "meta-llama/Llama-3.2-1B-Instruct" # Or any model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = PSE_Torch.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto") # Load to GPU, if available

# Ensure padding token is set for generation
model.config.pad_token_id = model.config.eos_token_id[0]
if model.generation_config:
    model.generation_config.pad_token_id = model.config.eos_token_id[0]

# 3. Create a StructuringEngine and configure it with your schema
model.engine = StructuringEngine(tokenizer)
model.engine.configure(Product)

# 4.  Create your prompt. Include the schema for the LLM's context.
prompt = f"""
You are a product catalog assistant.  Create a product description in JSON format,
following this schema:

{Product.model_json_schema()}

Create a product description for a new type of noise-cancelling headphones.
"""

messages = [{"role": "user", "content": prompt}]
input_ids = tokenizer.apply_chat_template(
    messages, return_tensors="pt", add_generation_prompt=True
)
# 5. Generate!
assert isinstance(input_ids, torch.Tensor)
input_ids = input_ids.to(model.device)
assert isinstance(input_ids, torch.Tensor)
output = model.generate(
    input_ids,
    do_sample=True,
    max_new_tokens=100,
    top_k=10,
    top_p=None,
)
# 6. Parse the structured output
structured_output = model.engine.parse_structured_output(output_type=Product)
print(json.dumps(structured_output.model_dump(), indent=2))
