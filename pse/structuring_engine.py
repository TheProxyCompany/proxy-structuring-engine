from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Iterator
from typing import Any, TypeVar

from pse_core.engine import Engine
from pse_core.state_machine import StateMachine
from pse_core.stepper import Stepper
from pydantic import BaseModel
from transformers import PreTrainedTokenizerBase, PreTrainedTokenizerFast

from pse.types.json import JSONSchemaSource, json_schema_state_machine
from pse.util.get_top_logits import get_top_k

logger = logging.getLogger(__name__)

Array_Type = TypeVar("Array_Type", bound=Any)
OutputType = TypeVar("OutputType")


class StructuringEngine(Engine):
    """
    The types of objects that the engine can use as a schema.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerFast | PreTrainedTokenizerBase,
        whitelist_control_tokens: list[str] | None = None,
        multi_token_sampling: bool = False,
        max_resample_attempts: int = 5,
    ) -> None:
        """
        Initialize the StructuringEngine with a tokenizer and vocabulary.
        """
        self.tokenizer = tokenizer
        self.control_tokens = self.build_control_tokens(whitelist_control_tokens)
        super().__init__(
            tokenizer.get_vocab(),
            lambda x: tokenizer.encode(x, add_special_tokens=False),
            lambda x: tokenizer.decode(x),
            multi_token_sampling=multi_token_sampling,
            control_tokens=self.control_tokens,
            max_resamples=max_resample_attempts,
        )

    def configure(
        self,
        structure: JSONSchemaSource | StateMachine,
        **kwargs: Any,
    ) -> None:
        """
        Configure the structuring engine with a schema.
        """
        if isinstance(structure, StateMachine):
            self.state_machine = structure
        else:
            _, self.state_machine = json_schema_state_machine(structure, **kwargs)

        self.steppers = self.state_machine.get_steppers()

    def process_logits(self, _: Any, raw_logits: Array_Type) -> Array_Type:
        """
        Process the logits and return the processed logits.
        """
        tic = time.perf_counter()
        self.multi_token_mapping: dict[int, list[int]] = {}
        # move logits to cpu if they aren't already on cpu
        original_device = None
        if hasattr(raw_logits, "device") and raw_logits.device.type != "cpu":
            original_device = raw_logits.device.type
            raw_logits = raw_logits.cpu()

        # process logits
        self.print_top_logits(raw_logits, 5, "Before 🟡")
        adjusted_logits = self.mask_invalid_tokens(raw_logits)
        self.print_top_logits(adjusted_logits, 5, "After 🟢")
        # move logits back to original device if they didn't start on cpu
        if original_device:
            adjusted_logits = adjusted_logits.to(original_device)

        toc = time.perf_counter()
        logger.debug(f"Logit processing took {toc - tic:0.4f} seconds")
        return adjusted_logits

    def sample(
        self, logprobs: Array_Type, sampler: Callable[..., Array_Type]
    ) -> Array_Type:
        """
        Sample tokens from logprobs using the provided sampler function.

        Args:
            logprobs: 2D array of shape (batch_size, sequence_length) containing log probabilities
            sampler: Callable that implements the sampling strategy

        Returns:
            Array of sampled token indices with same type as input logprobs

        Note:
            Parent class expects single-batch input of shape (1, sequence_length)
        """
        tic = time.perf_counter()
        # move logits to cpu if they aren't already on cpu
        original_device = None
        if hasattr(logprobs, "device") and logprobs.device.type != "cpu":
            original_device = logprobs.device.type
            logprobs = logprobs.cpu()

        # Process each batch individually
        samples = [
            self.select_next_tokens(batch[None], sampler)
            for batch in logprobs
            if batch is not None and batch.ndim == 1
        ]
        # Unwrap single batch
        sampled_token_id = samples[0] if len(samples) == 1 else samples
        result = type(logprobs)(sampled_token_id)
        # move logits back to original device if they didn't start on cpu
        if original_device is not None:
            result = result.to(original_device)

        toc = time.perf_counter()
        logger.debug(f"Sampling completed in {toc - tic:.4f}s: \033[33m{result}\033[0m")
        return result

    def get_structured_output(
        self,
        output_type: type[OutputType] | None = None,
        raise_on_error: bool = False,
    ) -> OutputType | Any:
        """
        Parse and cast the output to the given type.
        """
        for stepper in self.steppers:
            if stepper.has_reached_accept_state():
                token_safe_output = stepper.get_token_safe_output(
                    lambda x: self.tokenizer.decode(x)
                )
                return self.cast_output(
                    token_safe_output,
                    output_type,
                    raise_on_error,
                )

    def get_stateful_structured_output(
        self,
        output_type: type[OutputType] | None = None,
        raise_on_error: bool = False,
    ) -> Iterator[tuple[str, OutputType | Any]]:
        """
        Get each part of the output labeled with the identifier of the step that produced it.
        """
        for stepper in self.steppers:
            if stepper.has_reached_accept_state():
                yield from self._iter_state_and_output(
                    stepper,
                    output_type,
                    raise_on_error,
                )

    def _iter_state_and_output(
        self,
        stepper: Stepper,
        output_type: type[OutputType] | None,
        raise_on_error: bool,
    ) -> Iterator[tuple[str, OutputType | Any]]:
        """
        Helper method to parse and yield structured output from a stepper.
        """
        final_states: list[Stepper] = []
        for step in stepper.history:
            final_states.extend(step.get_final_state())

        for final_stepper in final_states:
            identifier = final_stepper.get_identifier() or str(
                final_stepper.current_state
            )
            token_safe_output = final_stepper.get_token_safe_output(
                lambda x: self.tokenizer.decode(x)
            )
            output = self.cast_output(token_safe_output, output_type, raise_on_error)
            yield identifier.lower(), output

    def cast_output(
        self,
        input: str,
        output_type: type[OutputType] | None,
        raise_on_error: bool,
    ) -> OutputType | Any:
        """
        Cast the output to the given type.
        """
        output = input
        try:
            deserialized = json.loads(input)
            if output_type and issubclass(output_type, BaseModel):
                output = output_type.model_validate(deserialized)
            else:
                output = deserialized
        except json.JSONDecodeError as e:
            if output_type:
                logger.error(f"JSON decoding failed: {e.msg} at position {e.pos}")
                if raise_on_error:
                    raise
        except Exception as e:
            logger.error(f"Failed to convert output to {output_type}: {e}")
            if raise_on_error:
                raise

        return output

    def build_control_tokens(self, whitelist_control_tokens: list[str] | None = None) -> list[int]:
        control_tokens: dict[str, int] = self.tokenizer.get_added_vocab()  # type: ignore [reportCallIssue]
        # do not mask control tokens that might be used as part of the schema
        for whitelisted_token in whitelist_control_tokens or []:
            if (
                whitelisted_token
                not in self.tokenizer.all_special_tokens
                and whitelisted_token in control_tokens
            ):
                # only prevent masking if the token is not a special token
                # because special tokens like eos need to be masked by default
                del control_tokens[whitelisted_token]

        return list(control_tokens.values())

    def get_live_structured_output(self) -> tuple[str, str] | None:
        """
        Get the live structured output.
        """
        return self.get_live_token_safe_output(lambda x: self.tokenizer.decode(x))

    def print_top_logits(self, logits: Any, top_n: int = 10, flag: str = "🔵") -> str:
        """
        Format and return a string showing the top tokens and their scores.
        """
        if logger.getEffectiveLevel() > logging.DEBUG:
            # short circuit if not debugging
            return ""

        top_logits = get_top_k(logits, top_n)
        rows = []

        for token_id, score in top_logits.items():
            if score <= float("-inf") or score < -1e10:
                continue

            token = repr(self.tokenizer.decode(token_id))

            if token_id in self.multi_token_mapping:
                multi_tokens = self.multi_token_mapping[token_id]
                if len(multi_tokens) > 1:
                    multi_repr = repr(self.tokenizer.decode(multi_tokens))
                    token = f"{token} -📶-> {multi_repr}"

            rows.append(f"{token_id:<8} | {score:>10.4f} | {token}")

        if not rows:
            return f"{flag} No valid tokens found"

        header = f"{'Token ID':<8} | {'Score':>10} | Token"
        separator = "-" * 9 + "+" + "-" * 12 + "+" + "-" * 20

        value = f"{flag}\n" + "\n".join([header, separator, *rows])
        logger.debug(value)
        return value
