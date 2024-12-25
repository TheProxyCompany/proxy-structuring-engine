import logging
from dataclasses import dataclass
from typing import cast

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.sample_utils import categorical_sampling

from pse.structuring_engine import StructuringEngine

logger = logging.getLogger(__name__)


@dataclass
class GenerateStepResult:
    token: mx.array
    logits: mx.array
    token_id: int
    time_to_generate_mask: float | None
    time_to_next_token: float | None
    total_time: float


@dataclass
class CompletedGeneration:
    output: str
    average_mask_latency: float
    average_time_to_get_next_token: float
    average_total_time: float


def sample(
    prompt: str | mx.array,
    model: nn.Module,
    engine: StructuringEngine,
    temp: float = 1.0,
) -> GenerateStepResult:
    """
    A generator producing token ids based on the given prompt from the model.

    Args:
        prompt (mx.array): The input prompt.
        model (nn.Module): The model to use for generation.
        engine (StructuringEngine): The engine to use for generation.
        temp (float): The temperature for sampling, if 0.0 the argmax is used.
          Default: ``1.0``.
    returns:
        Tuple[mx.array, mx.array]: A tuple of one token and a vector of log probabilities.
    """
    import timeit

    start_total = timeit.default_timer()

    if isinstance(prompt, str):
        messages = [{"role": "user", "content": prompt}]
        encoded_prompt = engine.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True
        )
        prompt = mx.array(encoded_prompt)

    # Generate logits
    logits = model(prompt[None])
    logits = logits[:, -1, :]
    mx.async_eval(logits)

    # Time the generation of the logit bias mask
    start_bias_mask = timeit.default_timer()
    logits = engine(prompt, logits[0, :])
    end_bias_mask = timeit.default_timer()
    logprobs = logits - mx.logsumexp(logits, keepdims=True)

    # Time the process of getting the next token
    def _sample(logprobs: mx.array) -> mx.array:
        return (
            categorical_sampling(logprobs, temp)
            if temp > 0.0
            else mx.argmax(logprobs, axis=-1)
        )

    end_next_token = None
    token = None
    start_next_token = timeit.default_timer()

    while not token:
        token_id = cast(int, _sample(logprobs).item())
        valid_token_id = engine.advance_token(token_id)
        if valid_token_id is not None:
            token = mx.array([valid_token_id])
            end_next_token = timeit.default_timer()
        else:
            logprobs[token_id] = float("-inf")

    end_token = timeit.default_timer()

    return GenerateStepResult(
        token,
        logits,
        cast(int, token.item()),
        end_bias_mask - start_bias_mask,
        (end_next_token - start_next_token) if end_next_token else None,
        end_token - start_total,
    )


def generate(
    prompt: str,
    model: nn.Module,
    engine: StructuringEngine,
    prefill: str | None = None,
) -> CompletedGeneration:
    messages = [{"role": "user", "content": prompt}]
    encoded_prompt = mx.array(
        engine.tokenizer.apply_chat_template(
            conversation=messages,
            add_generation_prompt=True,
        )
    )

    if prefill:
        encoded_prompt = mx.concatenate(
            [
                encoded_prompt,
                mx.array(engine.tokenizer.encode(prefill)),
            ]
        )

    logger.info(engine.tokenizer.decode(encoded_prompt.tolist()))
    encoded_prompt = mx.array(encoded_prompt)
    generation_results: list[GenerateStepResult] = []
    while not engine.has_reached_accept_state:
        try:
            result = sample(encoded_prompt, model, engine)
            encoded_prompt = mx.concatenate([encoded_prompt, result.token])
            generation_results.append(result)
        except Exception as e:
            logger.warning(f"Token rejected: {e}")
            break

    output = engine.tokenizer.decode([result.token_id for result in generation_results])
    if prefill:
        output = prefill + output

    mask_latencies = [
        mask_time
        for result in generation_results
        if (mask_time := result.time_to_generate_mask) is not None
    ]
    average_mask_latency = (
        sum(mask_latencies) / len(mask_latencies) if mask_latencies else 0.0
    )
    time_to_next_tokens = [
        time_to_next_token
        for result in generation_results
        if (time_to_next_token := result.time_to_next_token) is not None
    ]
    average_time_to_next_token = (
        sum(time_to_next_tokens) / len(time_to_next_tokens)
        if time_to_next_tokens
        else 0.0
    )
    total_times = [result.total_time for result in generation_results]
    average_total_time = sum(total_times) / len(total_times) if total_times else 0.0

    # Log performance metrics
    logger.info(
        f"Average time to generate logit bias mask: "
        f"{average_time_to_next_token:.6f} seconds"
    )
    logger.info(
        f"Average time to get next token: " f"{average_time_to_next_token:.6f} seconds"
    )
    logger.info(f"Average total time: {average_total_time:.6f} seconds")

    return CompletedGeneration(
        output,
        average_mask_latency,
        average_time_to_next_token,
        average_total_time,
    )
