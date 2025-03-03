import json
import logging
import sys

import mlx.core as mx
from mlx_lm.utils import generate_step, load

from pse.structuring_engine import StructuringEngine

logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)

ADVANCED_JSON_SCHEMA = {
    "type": "object",
    "description": "High-level thoughts, reasoning and internal dialogue.\n Used for step by step reasoning.",
    "properties": {
        "chain_of_thought": {
            "type": "array",
            "items": {
                "type": "string",
                "minLength": 200,
            },
            "minItems": 1,
            "maxItems": 3,
        },
    },
    "required": ["chain_of_thought"],
}

system_message = f"""
You are an AI that can think step by step.
You must follow this schema when generating your response:
{json.dumps(ADVANCED_JSON_SCHEMA, indent=2)}
"""

prompt = "This is a test - I want to see your private internal monologue."
messages = [
    {"role": "system", "content": system_message},
    {"role": "user", "content": prompt},
]
model_path_hf = "meta-llama/Llama-3.2-3B-Instruct"
model, tokenizer = load(model_path_hf)
engine = StructuringEngine(tokenizer._tokenizer)  # noqa: SLF001
engine.configure(ADVANCED_JSON_SCHEMA)

encoded_prompt = engine.tokenizer.apply_chat_template(
    conversation=messages,
    add_generation_prompt=True,
)

for tokens, _ in generate_step(
    prompt=mx.array(encoded_prompt),
    model=model,
    logits_processors=[engine.process_logits],
    sampler=lambda x: engine.sample(x, mx.argmax),  # type: ignore [arg-type]
    max_tokens=-1,
):
    encoded_prompt.append(tokens)  # type: ignore [attr-defined]
    if engine:
        break

output = engine.structured_output()
print(output)
