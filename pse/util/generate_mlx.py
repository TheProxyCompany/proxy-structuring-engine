import logging
from collections.abc import Callable
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from mlx_proxy.generate_step import generate_step
from mlx_proxy.samplers import make_sampler

from pse.structure.engine import StructuringEngine

logger = logging.getLogger(__name__)

def generate(
    prompt: str,
    model: nn.Module,
    engine: StructuringEngine,
    prefill: str | None = None,
) -> str:
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = engine.tokenizer.apply_chat_template(
        conversation=messages,
        add_generation_prompt=True,
        tokenize=False
    )
    assert isinstance(formatted_prompt, str)
    formatted_prompt = formatted_prompt + (prefill or "")
    logger.info(formatted_prompt)

    encoded_prompt = engine.tokenizer.encode(formatted_prompt, add_special_tokens=False)
    output_tokens: list[int] = []
    for tokens, _ in generate_step(
        prompt=mx.array(encoded_prompt),
        model=model,
        logits_processors=[engine.process_logits],
        sampler=sampler(engine),
    ):
        assert isinstance(tokens, mx.array)
        assert tokens.ndim == 1
        if tokens.shape[0] == 1:
            token_value = tokens.item()
            assert isinstance(token_value, int)
            encoded_prompt.append(token_value)
            output_tokens.append(token_value)
        else:
            # multiple tokens
            token_list = tokens.tolist()
            assert isinstance(token_list, list)
            encoded_prompt.extend(token_list)
            output_tokens.extend(token_list)

        if engine.has_reached_accept_state:
            break

    output = engine.tokenizer.decode(output_tokens)
    if prefill:
        output = prefill + output

    return output

def sampler(engine: StructuringEngine, **kwargs: Any) -> Callable[..., Any]:
    """
    Return a sampler function.
    If structured is True, use the structured sampler.
    Otherwise, use the simple sampler.
    """
    temp = float(kwargs.get("temp", 0.5))
    min_p = float(kwargs.get("min_p", 0.0))
    min_tokens_to_keep = int(kwargs.get("min_tokens_to_keep", 1))
    sampler = make_sampler(
        temp=temp, min_p=min_p, min_tokens_to_keep=min_tokens_to_keep
    )
    return lambda x: engine.sample(x, sampler)
