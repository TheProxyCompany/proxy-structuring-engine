from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from typing import Any, TypeVar

from pse_core.engine import Engine
from pse_core.state_machine import StateMachine
from pydantic import BaseModel
from transformers import PreTrainedTokenizerBase, PreTrainedTokenizerFast

from pse.engine import StructuringMachine
from pse.types.grammar.python import PythonGrammar
from pse.types.json import JSONSchemaSource
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
        multi_token_sampling: bool = False,
    ) -> None:
        """
        Initialize the StructuringEngine with a tokenizer and vocabulary.
        """
        self.tokenizer = tokenizer
        reverse_vocab: dict[int, str] = {}
        for token, token_id in self.tokenizer.get_vocab().items():
            if "▁" == token:
                token = " "
            else:
                token = self.tokenizer.decode(token_id)
            reverse_vocab[token_id] = token

        super().__init__(
            reverse_vocab,
            multi_token_sampling=multi_token_sampling,
            control_tokens=list(self.tokenizer.all_special_ids),
        )

    def configure(
        self, schema: JSONSchemaSource | StateMachine, **kwargs: Any
    ) -> dict[str, Any]:
        """
        Configure the structuring engine with a schema and optional delimiters.

        Args:
            schema: Schema to use when structuring output

        Returns:
            The JSON schema used by the structuring engine.
        """
        self.json_delimiters = None
        self.json_schema = {}
        if isinstance(schema, StateMachine):
            self.state_machine: StateMachine = schema
        else:
            self.state_machine = StructuringMachine(schema, **kwargs)
            if self.state_machine.json_delimiters:
                self.json_delimiters = self.state_machine.json_delimiters
            if self.state_machine.json_schema:
                self.json_schema = self.state_machine.json_schema

        self.steppers = self.state_machine.get_steppers()
        return self.json_schema

    def process_logits(self, _: Any, raw_logits: Array_Type) -> Array_Type:
        """
        Process the logits and return the processed logits.
        """
        if not self.state_machine:
            return raw_logits
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
        if not self.state_machine:
            return sampler(logprobs)

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

    def parse_structured_output(
        self,
        raw_output: str | None = None,
        output_type: type[OutputType] | None = None,
    ) -> OutputType | Any:
        """
        Parse and cast the output to the given type.

        Args:
            raw_output: The raw string output to parse. If None, attempts to get from steppers.
            output_type: The type to cast the output to. If None, returns parsed but uncast value.

        Returns:
            Parsed and optionally cast output value.

        Raises:
            ValueError: If parsing fails in an unexpected way
            TypeError: If output type conversion fails
        """
        # Get output from steppers if none provided
        if not raw_output and self.steppers:
            for stepper in self.steppers:
                raw_output = stepper.get_current_value()
                if stepper.has_reached_accept_state():
                    break

        if not raw_output:
            return None

        # remove delimiters from raw_output
        match self.current_state:
            case "json" if self.json_delimiters:
                delimiters = self.json_delimiters
            case "python":
                delimiters = PythonGrammar.delimiters
            case _:
                delimiters = None

        if delimiters and isinstance(raw_output, str):
            start, end = delimiters
            if start in raw_output:
                raw_output = raw_output.split(start, 1)[1]
            if end in raw_output:
                raw_output = raw_output.split(end, 1)[0]

        # Handle JSON parsing
        if self.current_state == "json" and isinstance(raw_output, str):
            try:
                raw_output = json.loads(raw_output)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON: {e}")
                pass

        # Handle type conversion if needed
        if (
            output_type
            and issubclass(output_type, BaseModel)
            and isinstance(raw_output, dict)
        ):
            try:
                return output_type.model_validate(raw_output)
            except Exception as e:
                logger.error(f"Failed to convert to {output_type}: {e}")
                pass

        return raw_output

    def print_top_logits(self, logits: Any, top_n: int = 10, flag: str = "🔵") -> str:
        """
        Format and return a string showing the top tokens and their scores.
        """
        if logger.getEffectiveLevel() < logging.DEBUG:
            return ""

        top_logits = get_top_k(logits, top_n)
        rows = []

        for token_id, score in top_logits.items():
            if score <= float("-inf") or score < -1e10:
                continue

            token = repr(self.tokenizer.decode(token_id))

            if token_id in self.multi_token_mapping:
                multi_tokens = self.multi_token_mapping[token_id]
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
