from dataclasses import dataclass
from typing import cast
from pse.core.engine import StructuringEngine
from pse.util.get_top_logits import get_top_logits
import mlx.core as mx
import mlx.nn as nn
from mlx_lm.sample_utils import categorical_sampling

import logging

from pse.util.errors import TokenRejected

logger = logging.getLogger(__name__)


@dataclass
class GenerateStepResult:
    token: mx.array
    logprobs: mx.array
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


def generate_step(
    prompt: mx.array,
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
    Yields:
        Tuple[mx.array, mx.array]: A tuple of one token and a vector of log probabilities.
    """
    import timeit

    start_total = timeit.default_timer()

    # Generate logits
    logits = model(prompt[None])
    logits = logits[:, -1, :]
    mx.async_eval(logits)

    # Time the generation of the logit bias mask
    start_bias_mask = timeit.default_timer()
    logits += engine.generate_logit_bias_mask(logits)
    end_bias_mask = timeit.default_timer()

    logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    # Time the process of getting the next token
    choose_token_time = None
    if engine.acceptor and not engine.has_reached_accept_state():
        top_logits = get_top_logits(logprobs[0, :])
        start_next_token = timeit.default_timer()
        valid_token_id = engine.get_next_token(
            logprobs=logprobs[0, :], top_logprobs=top_logits
        )
        choose_token_time = timeit.default_timer() - start_next_token
        token = mx.array([valid_token_id])
    else:
        token: mx.array = (
            categorical_sampling(logits, temp)
            if temp > 0.0
            else mx.argmax(logits, axis=-1)
        )

    # Calculate log probabilities

    # Async evaluation
    mx.async_eval(token)
    end_token = timeit.default_timer()

    return GenerateStepResult(
        token,
        logprobs,
        cast(int, token.item()),
        end_bias_mask - start_bias_mask,
        choose_token_time,
        end_token - start_total,
    )


def generate_response(
    prompt: str,
    model: nn.Module,
    engine: StructuringEngine,
) -> CompletedGeneration:
    messages = [{"role": "user", "content": prompt}]
    encoded_prompt = engine.tokenizer.apply_chat_template(
        messages, add_generation_prompt=True
    )
    encoded_prompt = mx.array(encoded_prompt)
    generation_results: list[GenerateStepResult] = []
    while not engine.has_reached_accept_state():
        try:
            generation_result = generate_step(encoded_prompt, model, engine)
            encoded_prompt = mx.concatenate([encoded_prompt, generation_result.token])
            generation_results.append(generation_result)
        except TokenRejected:
            logger.warning("Token rejected.")
            break

    output = engine.tokenizer.decode([result.token_id for result in generation_results])
    mask_latencies = [
        mask_time
        for result in generation_results
        if (mask_time := result.time_to_generate_mask) is not None
    ]
    average_mask_latency = sum(mask_latencies) / len(mask_latencies)
    time_to_next_tokens = [
        time_to_next_token
        for result in generation_results
        if (time_to_next_token := result.time_to_next_token) is not None
    ]
    average_time_to_next_token = sum(time_to_next_tokens) / len(time_to_next_tokens)
    total_times = [result.total_time for result in generation_results]
    average_total_time = sum(total_times) / len(total_times)

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
