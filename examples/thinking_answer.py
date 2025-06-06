import logging

import torch  # type: ignore[reportMissingImports]
from pse_core.state_machine import StateMachine
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaForCausalLM

from pse.structuring_engine import StructuringEngine
from pse.types.base.loop import LoopStateMachine
from pse.types.misc.fenced_freeform import FencedFreeformStateMachine
from pse.util.torch_mixin import PSETorchMixin

# toggle this to logging.DEBUG to see the PSE debug logs!
logging.basicConfig(level=logging.DEBUG)


class PSE_Torch(PSETorchMixin, LlamaForCausalLM):
    pass


# you can change the model path to any other model on huggingface
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

# define custom state machines
thinking_delimiters = ("[thinking]", "[/thinking]")
answer_delimiters = ("[answer]", "[/answer]")

# encapsulated state machines are used to allow a language model
# to generate unstructured content before the structured output
# starts. This "scratchpad" is disabled by default (min_buffer_length=-1)
thinking_state_machine = FencedFreeformStateMachine("thinking", thinking_delimiters)
# the answer state machine is used to wrap the structured output
answer_state_machine = FencedFreeformStateMachine("answer", answer_delimiters)
# Configure the engine with a state machine that enforces the following flow:
#
# The model starts in the 'thinking' state where it can express its reasoning.
# From there, it can transition to providing its final answer.
#
#      ┌──────────────┐
#      │              │
#      ▼              │
# ┌──────────┐        │
# │          │        │
# │ thinking ├────────┘
# │          │
# └──────┬───┘
#        │
#        ▼
# ┌──────────┐
# │          │
# │  answer  │
# │          │
# └──────────┘
#
# This ensures the model follows a structured thought process before
# providing its final answer.

model.engine.configure(
    StateMachine(
        {
            "thinking": [
                (
                    LoopStateMachine(
                        thinking_state_machine,
                        min_loop_count=1,
                        max_loop_count=2,
                    ),
                    "answer",
                )
            ],
            "answer": [
                (
                    answer_state_machine,
                    "done",
                ),
            ],
        },
        start_state="thinking",
        end_states=["done"],
    )
)

system_prompt = (
    f"Reason step by step using delimiters to seperate your thought process.\n"
    "For example, when asked a question, you should think and then answer.\n"
    "Example:\n"
    f"{thinking_delimiters[0]}Thinking goes here{thinking_delimiters[1]}"
    f"{answer_delimiters[0]}Answer goes here{answer_delimiters[1]}\n"
    "you can think multiple times before providing your answer.\n\n"
)
prompt = "Please pick a favorite color. Think about it first."

input_ids = tokenizer.apply_chat_template(
    [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ],
    return_tensors="pt",
    add_generation_prompt=True,
)
assert isinstance(input_ids, torch.Tensor)
input_ids = input_ids.to(model.device)
assert isinstance(input_ids, torch.Tensor)
output = model.generate(input_ids)

for label, output in model.engine.get_labeled_output():
    print("-" * 100)
    print(f"[{label}]")
    print(output)
    print(f"[/{label}]")
