# flake8: noqa
import logging

import torch
from transformers import AutoTokenizer, LlamaForCausalLM

from pse.engine.structuring_engine import StructuringEngine
from pse.util.torch_mixin import PSETorchMixin

# toggle this to logging.DEBUG to see the PSE debug logs!
logging.basicConfig(level=logging.INFO)

class PSE_Torch(PSETorchMixin, LlamaForCausalLM):
    pass


# you can change the model path to any other model on huggingface
# we upgraded to the 3b Llama instruct model for this example
model_path = "meta-llama/Llama-3.2-3B-Instruct"
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
    model.generation_config.min_p = 0.02
    model.generation_config.max_new_tokens = 1000
    model.generation_config.generation_kwargs = {"logits_to_keep": 8}
    model.generation_config.pad_token_id = model.config.eos_token_id[0]

# create structuring engine normally
model.engine = StructuringEngine(tokenizer, multi_token_sampling=True)

# define custom state machines
from pse.types.base.character import CharacterStateMachine
from pse.types.base.encapsulated import EncapsulatedStateMachine

any_input = CharacterStateMachine(
    charset="", # empty charset means any character is valid
    blacklist_charset="[", # the character that starts the delimiter is blacklisted,
    char_min=None, # no minimum number of characters
    char_limit=None, # no maximum number of characters
)

thinking_delimiters = ("[thinking]", "[/thinking]")
answer_delimiters = ("[answer]", "[/answer]")

# encapsulated state machines are used to allow a language model
# to generate unstructured content before the structured output
# starts. This "scratchpad" is disabled by default (min_buffer_length=-1)
thinking_state_machine = EncapsulatedStateMachine(
    state_machine=any_input,
    delimiters=thinking_delimiters,
)
# the answer state machine is used to wrap the structured output
answer_state_machine = EncapsulatedStateMachine(
    state_machine=any_input,
    delimiters=answer_delimiters,
)

# create a state machine that combines the thinking and answer state machines
from pse_core.state_machine import StateMachine
model.engine.configure(
    StateMachine(
        start_state="thinking",
        end_states=["answer"],
        state_graph={
            "thinking": [
                (
                    thinking_state_machine,
                    "verify",
                )
            ],
            "verify": [
                (
                    thinking_state_machine,
                    "thinking",
                ),
                (
                    answer_state_machine,
                    "answer",
                ),
            ],
        },
    )
)


system_prompt = (
    f"Reason step by step using delimiters to seperate your thought process.\n"
    "For example, when asked a question, you should think and then answer.\n"
    "Example:\n"
    f"{thinking_delimiters[0]}your step by step thinking here{thinking_delimiters[1]}"
    f"{answer_delimiters[0]}your answer here{answer_delimiters[1]}\n"
    "you can think multiple times before providing your answer.\n\n"
)
prompt = "What is the capital of Virginia, and what major US city is Arlington closest to?"

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
output = model.generate(
    input_ids,
    do_sample=True,
)

structured_output = model.engine.parse_structured_output()
print(f"raw output:\n\x1b[33m{structured_output}\x1b[0m\n")
print(
    f"\x1b[3m{structured_output.split(thinking_delimiters[0])[1].split(thinking_delimiters[1])[0]}\x1b[0m"
)
print(100 * "-")
print(structured_output.split(answer_delimiters[0])[1].split(answer_delimiters[1])[0])
