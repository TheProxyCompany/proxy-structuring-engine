import json

import torch
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from pse.engine.structuring_engine import StructuringEngine
from pse.util.torch_mixin import PSETorchMixin


# 1. Define your desired output structure using Pydantic
class Product(BaseModel):
    name: str
    price: float
    description: str | None = None

# 2. Load your model and tokenizer.  Apply the PSE mixin.
class PSE_Torch(PSETorchMixin, AutoModelForCausalLM):
    pass

model_path = "meta-llama/Llama-3.2-1B-Instruct" # Or any Hugging Face model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = PSE_Torch.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto") # Load to GPU, if available

# Ensure padding token is set for generation
if model.generation_config.pad_token_id is None:
    model.generation_config.pad_token_id = model.generation_config.eos_token_id

# 3. Create a StructuringEngine and configure it with your schema
engine = StructuringEngine(tokenizer)
engine.configure(Product)  # The engine automatically converts the Pydantic model to a JSON schema

# 4.  Create your prompt. Include the schema for the LLM's context.
prompt = f"""
You are a product catalog assistant.  Create a product description in JSON format,
following this schema:

{json.dumps(Product.model_json_schema(), indent=2)}

Create a product description for a new type of noise-cancelling headphones.
"""

# 5. Generate!
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
# disable truncation samplers like top_p
output = model.generate(input_ids, do_sample=True, max_new_tokens=100, top_k=10, top_p=0)
# 6. Parse the structured output
structured_output = engine.parse_structured_output(output_type=Product)
print(json.dumps(structured_output.model_dump(), indent=2)) # type: ignore[union-attr]
